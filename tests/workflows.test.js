import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { inflateRawSync } from 'node:zlib';
import { prepareDatasetQueue } from '../server/datasetQueue.js';
import { streamInference, validateConversation, listInferenceModels } from '../server/inference.js';
import { exportConversation, validateExport } from '../server/conversationExport.js';
import { startServer } from '../server/index.js';
import { runs } from '../server/pipeline.js';
import {
  crawlWebsite, extractReadableResource, isPublicAddress, robotsPolicy,
  validateCrawlRequest, validatePublicUrl,
} from '../server/websiteDataset.js';

const chatRecord = (content = 'Answer') => ({
  messages: [{ role: 'user', content: 'Question' }, { role: 'assistant', content }],
});

async function fixture(callback) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'modeltrainer-test-'));
  try { await callback(directory); }
  finally { fs.rmSync(directory, { recursive: true, force: true }); }
}

function writeDataset(directory, name, records) {
  const filename = path.join(directory, name);
  fs.writeFileSync(filename, records.map((record) => JSON.stringify(record)).join('\r\n'));
  return filename;
}

test('text queue combines all files in order and normalizes prompt/completion; hash is stable', async () => {
  await fixture(async (directory) => {
    const chat = writeDataset(directory, 'chat.jsonl', [chatRecord('First')]);
    fs.writeFileSync(chat, `\uFEFF${fs.readFileSync(chat, 'utf8')}\n\n`);
    const pairs = writeDataset(directory, 'pairs.jsonl', [{ prompt: 'Second question', completion: 'Second answer' }]);
    const datasets = [{ path: chat, connector: 'chat-jsonl' }, { path: pairs, connector: 'prompt-completion-jsonl' }];
    const prepared = await prepareDatasetQueue(datasets, 'text', directory);
    const repeated = await prepareDatasetQueue(datasets, 'text', directory);
    assert.equal(prepared.path, repeated.path);
    assert.equal(prepared.records, 2);
    assert.equal(prepared.datasets, 2);
    const records = fs.readFileSync(prepared.path, 'utf8').trim().split('\n').map(JSON.parse);
    assert.equal(records[0].messages[1].content, 'First');
    assert.equal(records[1].messages[0].content, 'Second question');
    assert.equal(records[1].messages[1].content, 'Second answer');
    const reversed = await prepareDatasetQueue([...datasets].reverse(), 'text', directory);
    assert.notEqual(prepared.sha256, reversed.sha256);
  });
});

test('video queue resolves duplicate clip filenames against each source directory', async () => {
  await fixture(async (directory) => {
    const datasets = ['first', 'second'].map((folder) => {
      const source = path.join(directory, folder);
      fs.mkdirSync(source);
      fs.writeFileSync(path.join(source, 'clip.mp4'), 'TEST FIXTURE - NOT A REAL VIDEO');
      return {
        path: writeDataset(source, 'manifest.jsonl', [{ ...chatRecord(), video: 'clip.mp4' }]),
        connector: 'video-jsonl', videosDir: source,
      };
    });
    const prepared = await prepareDatasetQueue(datasets, 'video', directory);
    const records = fs.readFileSync(prepared.path, 'utf8').trim().split('\n').map(JSON.parse);
    assert.equal(records[0].video, path.join(directory, 'first', 'clip.mp4'));
    assert.equal(records[1].video, path.join(directory, 'second', 'clip.mp4'));
  });
});

test('queue rejects empty files, invalid schemas, bad connectors, wrong modalities, missing clips and malformed JSON', async () => {
  await fixture(async (directory) => {
    const empty = writeDataset(directory, 'empty.jsonl', []);
    const invalid = writeDataset(directory, 'invalid.jsonl', [{ messages: [{ role: 'user', content: 'No answer' }] }]);
    const wrongPair = writeDataset(directory, 'pair.jsonl', [{ prompt: 'Q', completion: 10 }]);
    const video = writeDataset(directory, 'video.jsonl', [{ ...chatRecord(), video: 'missing.mp4' }]);
    const malformed = path.join(directory, 'broken.jsonl');
    fs.writeFileSync(malformed, '{bad json');
    await assert.rejects(prepareDatasetQueue([], 'text', directory), /between 1 and 100/);
    await assert.rejects(prepareDatasetQueue([{ path: empty, connector: 'chat-jsonl' }], 'text', directory), /empty/);
    await assert.rejects(prepareDatasetQueue([{ path: invalid, connector: 'chat-jsonl' }], 'text', directory), /line 1/);
    await assert.rejects(prepareDatasetQueue([{ path: invalid, connector: 'unknown' }], 'text', directory), /supported connector/);
    await assert.rejects(prepareDatasetQueue([{ path: wrongPair, connector: 'prompt-completion-jsonl' }], 'text', directory), /string prompt/);
    await assert.rejects(prepareDatasetQueue([{ path: video, connector: 'video-jsonl' }], 'text', directory), /cannot be mixed/);
    await assert.rejects(prepareDatasetQueue([{ path: video, connector: 'chat-jsonl' }], 'text', directory), /Video records require/);
    await assert.rejects(prepareDatasetQueue([{ path: video, connector: 'video-jsonl' }], 'video', directory), /line 1/);
    await assert.rejects(prepareDatasetQueue([{ path: malformed, connector: 'chat-jsonl' }], 'text', directory), /line 1/);
    assert.deepEqual(fs.readdirSync(path.join(directory, 'uploads', 'prepared')), []);
  });
});

test('conversation validation retains the entire thread and prevents invalid roles/order', () => {
  const messages = [
    { role: 'user', content: 'Original prompt' }, { role: 'assistant', content: 'Previous response' },
    { role: 'user', content: 'Continue' },
  ];
  assert.deepEqual(validateConversation({ model: ' test-model ', messages }), { model: 'test-model', messages });
  for (const body of [
    { model: '', messages }, { model: 'test', messages: [] },
    { model: 'test', messages: [{ role: 'system', content: 'Bad' }] },
    { model: 'test', messages: messages.slice(0, 2) },
    { model: 'test', messages: [{ role: 'user', content: ' ' }] },
    { model: 'test', messages: Array(201).fill(messages[0]) },
  ]) assert.throws(() => validateConversation(body));
});

function streamedResponse(lines) {
  const bytes = new TextEncoder().encode(lines);
  return new Response(new ReadableStream({
    start(controller) {
      // Deliberately split UTF-8 and NDJSON boundaries.
      for (let index = 0; index < bytes.length; index += 7) controller.enqueue(bytes.slice(index, index + 7));
      controller.close();
    },
  }));
}

test('inference streams tokens, preserves Unicode and full conversation, and requires completion', async (context) => {
  const conversation = validateConversation({ model: 'fixture-model', messages: [{ role: 'user', content: 'Hello' }] });
  let request;
  const mock = context.mock.method(globalThis, 'fetch', async (_url, options) => {
    request = JSON.parse(options.body);
    return streamedResponse('{"message":{"content":"Caf\\u00e9 "},"done":false}\n{"message":{"content":"next"},"done":true}');
  });
  const events = [];
  await streamInference(conversation, new AbortController().signal, (event) => events.push(event));
  assert.deepEqual(request, { ...conversation, stream: true });
  assert.equal(events.filter((event) => event.type === 'token').map((event) => event.content).join(''), 'Caf\u00e9 next');
  assert.equal(events.at(-1).type, 'complete');
  mock.mock.mockImplementation(async () => streamedResponse('{"message":{"content":"partial"}}\n'));
  await assert.rejects(streamInference(conversation, new AbortController().signal, () => {}), /disconnected/);
  mock.mock.mockImplementation(async () => streamedResponse('{"error":"Not enough memory"}\n'));
  await assert.rejects(streamInference(conversation, new AbortController().signal, () => {}), /Not enough memory/);
  mock.mock.mockImplementation(async () => new Response('', { status: 404 }));
  await assert.rejects(streamInference(conversation, new AbortController().signal, () => {}), /HTTP 404/);
});

test('Ollama discovery surfaces unavailable and malformed responses', async (context) => {
  const mock = context.mock.method(globalThis, 'fetch', async () => Response.json({ models: [{ name: 'local-model', size: 123 }] }));
  assert.deepEqual(await listInferenceModels(), [{ name: 'local-model', size: 123 }]);
  mock.mock.mockImplementation(async () => Response.json({}));
  await assert.rejects(listInferenceModels(), /invalid model list/);
  mock.mock.mockImplementation(async () => new Response('', { status: 503 }));
  await assert.rejects(listInferenceModels(), /HTTP 503/);
});

test('inference cancellation propagates to the local runtime', async (context) => {
  context.mock.method(globalThis, 'fetch', async (_url, options) => {
    return new Promise((_resolve, reject) => options.signal.addEventListener('abort',
      () => reject(options.signal.reason), { once: true }));
  });
  const controller = new AbortController();
  const pending = streamInference(validateConversation({
    model: 'test-only', messages: [{ role: 'user', content: 'Cancel test' }],
  }), controller.signal, () => {});
  controller.abort(new Error('Test cancellation'));
  await assert.rejects(pending, /Test cancellation/);
});

test('site crawling requires permission and blocks private networks, credentials and insecure schemes', async () => {
  assert.throws(() => validateCrawlRequest({ url: 'https://example.com', maxPages: 5 }), /permission/);
  assert.throws(() => validateCrawlRequest({ url: 'http://example.com', maxPages: 5, permission: true }), /HTTPS/);
  assert.throws(() => validateCrawlRequest({ url: 'https://user:password@example.com', maxPages: 5, permission: true }), /credentials/);
  assert.throws(() => validateCrawlRequest({ url: 'https://example.com', maxPages: 101, permission: true }), /1 and 100/);
  assert.equal(validateCrawlRequest({ url: 'https://example.com/docs#part', maxPages: 5, permission: true }).url, 'https://example.com/docs');
  for (const address of ['127.0.0.1', '10.1.2.3', '169.254.169.254', '192.168.0.2', '172.17.0.1', '100.64.1.1',
    '0.0.0.0', '203.0.113.1', '::1', 'fc00::1', 'fe80::1', '::ffff:127.0.0.1', '2001:db8::1']) {
    assert.equal(isPublicAddress(address), false, `${address} must not be crawled`);
  }
  assert.equal(isPublicAddress('93.184.216.34'), true);
  assert.equal(isPublicAddress('2606:4700:4700::1111'), true);
  await assert.rejects(validatePublicUrl('https://example.com', async () => [{ address: '127.0.0.1', family: 4 }]), /not permitted/);
});

test('robots policy honors longest allow/disallow, wildcard rules and bot-specific groups', () => {
  const policy = robotsPolicy('User-agent: *\nDisallow: /private\nAllow: /private/public\nDisallow: /*.pdf$\nCrawl-delay: 2');
  assert.equal(policy.allows('https://example.com/private/document'), false);
  assert.equal(policy.allows('https://example.com/private/public/page'), true);
  assert.equal(policy.allows('https://example.com/file.pdf'), false);
  assert.equal(policy.allows('https://example.com/file.pdf.txt'), true);
  assert.equal(policy.waitSeconds, 2);
  const specific = robotsPolicy('User-agent: *\nDisallow: /\nUser-agent: ModelTrainerBot\nAllow: /docs');
  assert.equal(specific.allows('https://example.com/docs'), true);
});

test('readable extraction handles HTML, plain text, real Word and PDF documents', async () => {
  const html = await extractReadableResource({
    url: 'https://example.com/docs', type: 'text/html',
    buffer: Buffer.from('<title>Reference</title><body><nav>Navigation</nav><main><h1>Readable heading</h1><p>Readable paragraph.</p><script>secret script</script><p hidden>Hidden text</p><a href="/guide.pdf">PDF guide</a></main></body>'),
  });
  assert.match(html.text, /Readable heading/);
  assert.match(html.text, /Readable paragraph/);
  assert.doesNotMatch(html.text, /Navigation|secret script|Hidden text/);
  assert.deepEqual(html.links, ['/guide.pdf']);
  const word = await exportConversation({
    title: 'Readable Word title', model: 'test model', format: 'docx',
    messages: [{ role: 'assistant', content: 'Readable Word paragraph and more detail.' }],
  });
  const extracted = await extractReadableResource({
    url: 'https://example.com/guide.docx', type: 'application/octet-stream', buffer: word,
  });
  assert.match(extracted.text, /Readable Word title/);
  assert.match(extracted.text, /Readable Word paragraph/);
  const pdf = await exportConversation({
    title: 'Readable PDF title', model: 'test model', format: 'pdf',
    messages: [{ role: 'assistant', content: 'Readable PDF paragraph with useful information.' }],
  });
  const pdfText = await extractReadableResource({
    url: 'https://example.com/guide.pdf', type: 'application/pdf', buffer: pdf,
  });
  assert.match(pdfText.text, /Readable PDF paragraph/);
  const plain = await extractReadableResource({ url: 'https://example.com/readme.txt', type: 'text/plain', buffer: Buffer.from('Plain text document') });
  assert.equal(plain.text, 'Plain text document');
  await assert.rejects(extractReadableResource({ url: 'https://example.com/image.png', type: 'image/png', buffer: Buffer.from('fixture') }), /OCR/);
});

test('website crawl collects same-site pages and documents, records provenance and exposes partial failures', async () => {
  await fixture(async (directory) => {
    const visited = [];
    const fixtures = {
      'https://example.com/robots.txt': { status: 200, buffer: Buffer.from('User-agent: *\nDisallow: /private') },
      'https://example.com/': {
        status: 200, type: 'text/html',
        buffer: Buffer.from('<title>Home</title><main><p>A sufficiently long public home page with useful information.</p><a href="/guide.txt">Guide</a><a href="/missing">Missing</a><a href="/private">Private</a><a href="https://other.example/">Other site</a></main>'),
      },
      'https://example.com/guide.txt': { status: 200, type: 'text/plain', buffer: Buffer.from('Detailed plain text guide for this authorized test site.') },
      'https://example.com/missing': { status: 403, buffer: Buffer.alloc(0), type: '' },
    };
    const progress = [];
    const result = await crawlWebsite({ url: 'https://example.com/', maxPages: 10 },
      new AbortController().signal, (event) => progress.push(event), {
        root: directory, wait: async () => {},
        download: async (url) => {
          visited.push(url);
          assert.ok(fixtures[url], `Unexpected request ${url}`);
          return { ...fixtures[url], url };
        },
      });
    assert.equal(result.records, 2);
    assert.equal(result.failed, 2);
    assert.ok(!visited.includes('https://example.com/private'));
    assert.ok(!visited.includes('https://other.example/'));
    assert.equal(progress.at(-1).collected, 2);
    const records = fs.readFileSync(result.path, 'utf8').trim().split('\n').map(JSON.parse);
    assert.equal(records[0].source.url, 'https://example.com/');
    assert.ok(!Number.isNaN(Date.parse(records[0].source.collectedAt)));
    assert.match(records[0].source.sha256, /^[a-f0-9]{64}$/);
    const prepared = await prepareDatasetQueue([result], 'text', directory);
    assert.equal(prepared.records, 2);
  });
});

test('website crawling reports unreadable sources, robots failures, truncation, limits and cancellation', async () => {
  await fixture(async (directory) => {
    const controller = new AbortController();
    const options = {
      root: directory, wait: async () => {},
      download: async (url) => url.endsWith('robots.txt')
        ? { status: 404, buffer: Buffer.alloc(0), url }
        : { status: 200, type: 'text/html', buffer: Buffer.from('<main></main>'), url },
    };
    await assert.rejects(crawlWebsite({ url: 'https://example.com/', maxPages: 1 }, controller.signal, () => {}, options), /No readable/);
    await assert.rejects(crawlWebsite({ url: 'https://example.com/', maxPages: 1 }, controller.signal, () => {}, {
      ...options, download: async () => ({ status: 403 }),
    }), /robots.txt/);
    const limited = await crawlWebsite({ url: 'https://example.com/', maxPages: 1 }, controller.signal, () => {}, {
      ...options, extract: async () => ({ title: 'Long source', format: 'html', text: 'a'.repeat(100001), links: ['/more'] }),
    });
    assert.equal(limited.truncated, 1);
    assert.equal(limited.limitReached, true);
    const excerpts = fs.readFileSync(limited.path, 'utf8').trim().split('\n').map(JSON.parse);
    assert.equal(excerpts.map((record) => record.messages[1].content).join(''), 'a'.repeat(100000));
    assert.equal(excerpts[0].source.chunk, 1);
    assert.equal(excerpts.at(-1).source.chunks, excerpts.length);
    controller.abort(new Error('Cancelled fixture'));
    await assert.rejects(crawlWebsite({ url: 'https://example.com/', maxPages: 1 }, controller.signal, () => {}, options), /Cancelled fixture/);
  });
});

function zipText(buffer, filename) {
  for (let index = 0; index < buffer.length - 46; index += 1) {
    if (buffer.readUInt32LE(index) !== 0x02014b50) continue;
    const nameSize = buffer.readUInt16LE(index + 28);
    if (buffer.toString('utf8', index + 46, index + 46 + nameSize) !== filename) continue;
    const local = buffer.readUInt32LE(index + 42);
    const size = buffer.readUInt32LE(index + 20);
    const start = local + 30 + buffer.readUInt16LE(local + 26) + buffer.readUInt16LE(local + 28);
    const data = buffer.subarray(start, start + size);
    return (buffer.readUInt16LE(index + 10) === 8 ? inflateRawSync(data) : data).toString('utf8');
  }
  throw new Error(`ZIP entry not found: ${filename}`);
}

test('exports generate real PDF and DOCX with headings, model, complete thread and escaped text', async () => {
  const conversation = {
    title: 'Test conversation', model: 'test-model',
    messages: [{ role: 'user', content: 'Question <test>' }, { role: 'assistant', content: 'First line\nSecond line & details' }],
  };
  const word = await exportConversation(validateExport({ ...conversation, format: 'docx' }));
  assert.equal(word.subarray(0, 2).toString(), 'PK');
  const xml = zipText(word, 'word/document.xml');
  for (const text of ['Test conversation', 'test-model', 'Prompt', 'Response', 'Question &lt;test&gt;', 'First line', 'Second line &amp; details']) {
    assert.ok(xml.includes(text), `Word document missing ${text}`);
  }
  const pdf = await exportConversation(validateExport({ ...conversation, format: 'pdf' }));
  assert.equal(pdf.subarray(0, 5).toString(), '%PDF-');
  assert.ok(pdf.includes(Buffer.from('%%EOF')));
  assert.match(pdf.toString('latin1'), /\/Title \d+ 0 R/);
  assert.ok(pdf.includes(Buffer.from('(Test conversation)')));
  assert.throws(() => validateExport({ ...conversation, format: 'html' }));
  assert.throws(() => validateExport({ ...conversation, format: 'pdf', messages: [] }));
});

test('API chat remains usable during an active training run; errors, uploads, exports and queue contracts', async (context) => {
  const originalFetch = globalThis.fetch;
  let inferenceRequests = 0;
  context.mock.method(globalThis, 'fetch', async (url, options) => {
    if (String(url).startsWith('http://127.0.0.1:11434')) {
      inferenceRequests += 1;
      if (String(url).endsWith('/api/tags')) return Response.json({ models: [{ name: 'test-only-model', size: 1 }] });
      return streamedResponse('{"message":{"content":"Test-only response"},"done":true}\n');
    }
    return originalFetch(url, options);
  });
  // Isolated fixture represents a running pipeline; no hardware work or mock production path is introduced.
  runs.set('test-active-training', { done: false, stageResults: { train: 'running' } });
  const server = await startServer({ port: 0 });
  const base = `http://127.0.0.1:${server.address().port}`;
  const uploaded = [];
  try {
    const [models, chat] = await Promise.all([
      fetch(`${base}/api/inference/models`),
      fetch(`${base}/api/inference/chat`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: 'test-only-model', messages: [{ role: 'user', content: 'Test fixture prompt' }] }),
      }),
    ]);
    assert.equal((await models.json()).models[0].name, 'test-only-model');
    assert.match(await chat.text(), /Test-only response/);
    assert.equal(inferenceRequests, 2);
    assert.equal(runs.get('test-active-training').done, false);
    const badChat = await fetch(`${base}/api/inference/chat`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}',
    });
    assert.equal(badChat.status, 400);
    const unauthorizedCrawl = await fetch(`${base}/api/datasets/crawl`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: 'https://example.com/', maxPages: 1, permission: false }),
    });
    assert.equal(unauthorizedCrawl.status, 400);
    assert.match((await unauthorizedCrawl.json()).error, /permission/);
    const privateCrawl = await fetch(`${base}/api/datasets/crawl`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: 'https://127.0.0.1/', maxPages: 1, permission: true }),
    });
    assert.equal(privateCrawl.status, 202);
    const crawlId = (await privateCrawl.json()).id;
    let crawlStatus;
    for (let attempt = 0; attempt < 20; attempt += 1) {
      crawlStatus = await (await fetch(`${base}/api/datasets/crawl/${crawlId}`)).json();
      if (crawlStatus.status !== 'running') break;
      await new Promise((resolve) => setTimeout(resolve, 10));
    }
    assert.equal(crawlStatus.status, 'failed');
    assert.match(crawlStatus.error, /not permitted/);
    for (const text of ['First uploaded record', 'Second uploaded record']) {
      const form = new FormData();
      form.append('file', new Blob([JSON.stringify(chatRecord(text))]), 'same-name.jsonl');
      const response = await fetch(`${base}/api/upload-dataset`, { method: 'POST', body: form });
      const result = await response.json();
      assert.equal(result.ok, true);
      uploaded.push(result.path);
    }
    assert.notEqual(uploaded[0], uploaded[1]);
    const preparedResponse = await fetch(`${base}/api/datasets/prepare`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ kind: 'text', datasets: uploaded.map((file) => ({ path: file, connector: 'chat-jsonl' })) }),
    });
    const prepared = await preparedResponse.json();
    assert.equal(prepared.records, 2);
    uploaded.push(prepared.path);
    const exported = await fetch(`${base}/api/inference/export`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ format: 'pdf', model: 'fixture', messages: [{ role: 'assistant', content: 'API export fixture' }] }),
    });
    assert.equal(exported.headers.get('content-type'), 'application/pdf');
    assert.equal(Buffer.from(await exported.arrayBuffer()).subarray(0, 5).toString(), '%PDF-');
  } finally {
    runs.delete('test-active-training');
    await new Promise((resolve) => server.close(resolve));
    for (const file of uploaded) fs.rmSync(file, { force: true });
  }
});
