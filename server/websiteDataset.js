import fs from 'node:fs';
import path from 'node:path';
import https from 'node:https';
import dns from 'node:dns/promises';
import { BlockList, isIP } from 'node:net';
import crypto from 'node:crypto';
import { setTimeout as delay } from 'node:timers/promises';
import { load } from 'cheerio';
import { unzipSync } from 'fflate';
import { workspaceRoot } from './runtimePaths.js';

const USER_AGENT = 'ModelTrainerBot/1.0';
const MAX_BYTES = 8 * 1024 * 1024;
const privateNetworks = new BlockList();
for (const [network, prefix] of [
  ['0.0.0.0', 8], ['10.0.0.0', 8], ['100.64.0.0', 10], ['127.0.0.0', 8],
  ['169.254.0.0', 16], ['172.16.0.0', 12], ['192.0.0.0', 24], ['192.168.0.0', 16],
  ['192.0.2.0', 24], ['192.88.99.0', 24], ['198.18.0.0', 15], ['198.51.100.0', 24],
  ['203.0.113.0', 24], ['224.0.0.0', 4], ['240.0.0.0', 4],
]) privateNetworks.addSubnet(network, prefix, 'ipv4');
privateNetworks.addSubnet('2001::', 23, 'ipv6');
privateNetworks.addSubnet('2001:db8::', 32, 'ipv6');
privateNetworks.addSubnet('2002::', 16, 'ipv6');
const publicV6 = new BlockList();
publicV6.addSubnet('2000::', 3, 'ipv6');

export function isPublicAddress(address) {
  const family = isIP(address);
  return family === 4 ? !privateNetworks.check(address, 'ipv4')
    : family === 6 && publicV6.check(address, 'ipv6') && !privateNetworks.check(address, 'ipv6');
}

export async function validatePublicUrl(value, lookup = dns.lookup) {
  const url = new URL(value);
  if (url.protocol !== 'https:' || url.username || url.password || (url.port && url.port !== '443')) {
    throw new Error('Use a public HTTPS URL without credentials or a custom port.');
  }
  const hostname = url.hostname.replace(/^\[|\]$/g, '');
  const addresses = await lookup(hostname, { all: true });
  if (!addresses.length || addresses.some(({ address }) => !isPublicAddress(address))) {
    throw new Error('Private, loopback, link-local, and reserved network destinations are not permitted.');
  }
  url.hash = '';
  return { url, addresses };
}

export async function downloadPublicResource(value, signal, origin, allows = () => true) {
  let current = value;
  for (let redirects = 0; redirects <= 5; redirects += 1) {
    const { url, addresses } = await validatePublicUrl(current);
    if (origin && url.origin !== origin) throw new Error('Redirect leaves the selected site; skipped.');
    if (!allows(url.href)) throw new Error('Blocked by robots.txt.');
    const result = await new Promise((resolve, reject) => {
      // Pin the validated DNS answer so a second lookup cannot redirect the request to a private address.
      const request = https.get(url, {
        signal,
        headers: { 'User-Agent': USER_AGENT, 'Accept-Encoding': 'identity', Accept: 'text/html,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,text/plain;q=0.9,*/*;q=0.1' },
        lookup: (_hostname, options, callback) => {
          const selected = addresses.find((address) => !options.family || address.family === options.family) || addresses[0];
          if (options.all) callback(null, [selected]);
          else callback(null, selected.address, selected.family);
        },
      }, (response) => {
        const status = response.statusCode || 500;
        if (status >= 300 && status < 400) {
          response.resume();
          resolve({ status, location: response.headers.location });
          return;
        }
        if (status !== 200) {
          response.resume();
          resolve({ status, buffer: Buffer.alloc(0), type: '' });
          return;
        }
        let size = 0;
        const chunks = [];
        response.on('data', (chunk) => {
          size += chunk.length;
          if (size > MAX_BYTES) {
            response.destroy(new Error('Resource exceeds the 8 MB download limit.'));
            return;
          }
          chunks.push(chunk);
        });
        response.on('error', reject);
        response.on('end', () => resolve({
          status, buffer: Buffer.concat(chunks),
          type: String(response.headers['content-type'] || '').split(';')[0].toLowerCase(),
        }));
      });
      request.setTimeout(15000, () => request.destroy(new Error('Resource download timed out.')));
      request.on('error', reject);
    });
    if (result.location) {
      current = new URL(result.location, url).href;
      continue;
    }
    return { ...result, url: url.href };
  }
  throw new Error('Too many redirects.');
}

export function robotsPolicy(text) {
  const groups = [];
  let group = null;
  let hasRules = false;
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.split('#')[0].trim();
    const colon = line.indexOf(':');
    if (colon < 0) continue;
    const key = line.slice(0, colon).trim().toLowerCase();
    const value = line.slice(colon + 1).trim();
    if (key === 'user-agent') {
      if (!group || hasRules) {
        group = { agents: [], rules: [], wait: 0 };
        groups.push(group);
        hasRules = false;
      }
      group.agents.push(value.toLowerCase());
    } else if (group && ['allow', 'disallow', 'crawl-delay'].includes(key)) {
      hasRules = true;
      if (key === 'crawl-delay') group.wait = Number(value) || 0;
      else if (value) group.rules.push({ allow: key === 'allow', pattern: value });
    }
  }
  const specific = groups.filter((item) => item.agents.some((agent) => agent !== '*' && USER_AGENT.toLowerCase().startsWith(agent)));
  const applicable = specific.length ? specific : groups.filter((item) => item.agents.includes('*'));
  const waitSeconds = Math.max(0.5, ...applicable.map((item) => item.wait));
  return {
    waitSeconds,
    allows(url) {
      const target = `${new URL(url).pathname}${new URL(url).search}`;
      const matched = applicable.flatMap((item) => item.rules).filter((rule) => {
        const pattern = rule.pattern.replace(/[.+?^${}()|[\]\\]/g, '\\$&').replace(/\*/g, '.*').replace(/\\\$$/, '$');
        return new RegExp(`^${pattern}`).test(target);
      }).sort((a, b) => b.pattern.length - a.pattern.length || Number(b.allow) - Number(a.allow));
      return matched[0]?.allow ?? true;
    },
  };
}

export async function extractReadableResource({ buffer, type, url }) {
  const extension = path.extname(new URL(url).pathname).toLowerCase();
  if (type === 'application/pdf' || extension === '.pdf') {
    const { PDFParse } = await import('pdf-parse');
    const parser = new PDFParse({ data: new Uint8Array(buffer) });
    try {
      const parsed = await parser.getText();
      return { title: path.basename(new URL(url).pathname), text: parsed.text, links: [], format: 'pdf' };
    } finally {
      await parser.destroy();
    }
  }
  if (type === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' || extension === '.docx') {
    let expandedBytes = 0;
    const parts = unzipSync(buffer, { filter: (file) => {
      if (!/^word\/(document|header\d*|footer\d*|footnotes|endnotes|comments)\.xml$/.test(file.name)) return false;
      expandedBytes += file.originalSize;
      if (expandedBytes > 32 * 1024 * 1024) throw new Error('Word document exceeds the 32 MB extracted-text safety limit.');
      return true;
    } });
    if (!parts['word/document.xml']) throw new Error('Not a readable Word (.docx) document.');
    const text = Object.keys(parts).sort((a, b) =>
      Number(b === 'word/document.xml') - Number(a === 'word/document.xml') || a.localeCompare(b))
      .map((name) => {
        const $ = load(Buffer.from(parts[name]).toString('utf8'), { xml: true });
        $('w\\:instrText,w\\:del').remove();
        $('w\\:tab').replaceWith('\t');
        $('w\\:br').replaceWith('\n');
        $('w\\:p').append('\n');
        return $.root().text();
      }).join('\n\n');
    return { title: path.basename(new URL(url).pathname), text, links: [], format: 'docx' };
  }
  if (type === 'text/html' || type === 'application/xhtml+xml' || ['.html', '.htm'].includes(extension)) {
    const $ = load(buffer.toString('utf8'));
    const title = $('title').text().trim() || new URL(url).pathname;
    const links = $('a[href]').map((_index, anchor) => $(anchor).attr('href')).get();
    $('script,style,noscript,svg,canvas,iframe,nav,header,footer,form,[hidden],[aria-hidden="true"]').remove();
    $('br').replaceWith('\n');
    $('p,div,section,article,li,h1,h2,h3,h4,pre,tr').append('\n');
    const main = $('main');
    const text = (main.length ? main : $('body')).text().replace(/[ \t]+/g, ' ').replace(/\n\s*\n+/g, '\n\n').trim();
    return { title, text, links, format: 'html' };
  }
  if (type.startsWith('text/') || ['.txt', '.md', '.csv', '.json', '.jsonl'].includes(extension)) {
    return { title: path.basename(new URL(url).pathname), text: buffer.toString('utf8'), links: [], format: 'text' };
  }
  throw new Error(`Unsupported content type ${type || '(missing)'}. Supported: HTML, text, PDF and DOCX. Images/scans require OCR.`);
}

export function validateCrawlRequest(body) {
  if (body?.permission !== true) throw new Error('Confirm that you own or have permission to use this site and its documents for training.');
  if (typeof body.url !== 'string' || body.url.length > 2048) throw new Error('Enter a public HTTPS site URL.');
  const url = new URL(body.url);
  if (url.protocol !== 'https:' || url.username || url.password || (url.port && url.port !== '443')) {
    throw new Error('Enter a public HTTPS site URL without credentials or a custom port.');
  }
  const maxPages = Number(body.maxPages);
  if (!Number.isInteger(maxPages) || maxPages < 1 || maxPages > 100) throw new Error('Set a crawl limit between 1 and 100 pages/documents.');
  url.hash = '';
  return { url: url.href, maxPages };
}

export async function crawlWebsite(config, signal, onProgress, {
  download = downloadPublicResource, extract = extractReadableResource, wait = delay, root = workspaceRoot,
} = {}) {
  const origin = new URL(config.url).origin;
  const robots = await download(`${origin}/robots.txt`, signal, origin);
  if (![200, 404, 410].includes(robots.status)) throw new Error(`Cannot verify robots.txt (HTTP ${robots.status}). Crawl stopped.`);
  const policy = robotsPolicy(robots.status === 200 ? robots.buffer.toString('utf8') : '');
  if (policy.waitSeconds > 30) throw new Error('The site requests a crawl delay longer than 30 seconds. Use a direct authorized dataset export instead.');
  const queue = [config.url];
  const visited = new Set();
  const contentHashes = new Set();
  const records = [];
  const issues = [];
  let truncated = 0;
  let attempted = 0;
  let failed = 0;
  while (queue.length && attempted < config.maxPages) {
    signal.throwIfAborted();
    const url = queue.shift();
    if (visited.has(url)) continue;
    visited.add(url);
    attempted += 1;
    try {
      if (!policy.allows(url)) throw new Error('Blocked by robots.txt.');
      if (attempted > 1) await wait(policy.waitSeconds * 1000, undefined, { signal });
      const resource = await download(url, signal, origin, policy.allows);
      if (resource.status !== 200) throw new Error(`HTTP ${resource.status}`);
      if (!policy.allows(resource.url)) throw new Error('Redirect target is blocked by robots.txt.');
      const result = await extract(resource);
      const text = result.text.trim();
      if (text.length < 20) throw new Error('No readable text. This may be an image-only scan, an empty page, or a site requiring JavaScript/login. OCR and browser rendering are not available.');
      for (const link of result.links) {
        try {
          const candidate = new URL(link, resource.url);
          candidate.hash = '';
          if (candidate.origin === origin && !candidate.username && !candidate.password &&
              !visited.has(candidate.href) && queue.length < 1000 &&
              !/\.(png|jpe?g|gif|svg|webp|mp[34]|zip|exe|css|js|woff2?)$/i.test(candidate.pathname)) {
            queue.push(candidate.href);
          }
        } catch { /* Non-URL navigation links are not crawl targets. */ }
      }
      const hash = crypto.createHash('sha256').update(text).digest('hex');
      if (contentHashes.has(hash)) {
        issues.push({ url, error: 'Duplicate readable content skipped.' });
      } else {
        contentHashes.add(hash);
        const characters = Array.from(text);
        const wasTruncated = characters.length > 100000;
        if (wasTruncated) truncated += 1;
        const readable = characters.slice(0, 100000);
        const chunks = [];
        for (let index = 0; index < readable.length; index += 1500) {
          const chunk = readable.slice(index, index + 1500).join('');
          if (chunk.trim()) chunks.push(chunk);
        }
        const chunkCount = chunks.length;
        const collectedAt = new Date().toISOString();
        for (let index = 0; index < chunkCount; index += 1) {
          records.push({
            messages: [
              { role: 'user', content: `What information is provided in "${result.title.slice(0, 200)}", excerpt ${index + 1} of ${chunkCount}?` },
              { role: 'assistant', content: chunks[index] },
            ],
            source: { url: resource.url, title: result.title, format: result.format, collectedAt, sha256: hash, truncated: wasTruncated, chunk: index + 1, chunks: chunkCount },
          });
        }
      }
    } catch (error) {
      signal.throwIfAborted();
      failed += 1;
      issues.push({ url, error: error.message });
    }
    onProgress({ attempted, collected: contentHashes.size, pending: queue.length, failed, truncated });
  }
  if (!records.length) throw new Error(`No readable pages or documents collected. ${issues[0]?.error || 'No supported content found.'}`);
  signal.throwIfAborted();
  const directory = path.join(root, 'uploads', 'website-datasets');
  await fs.promises.mkdir(directory, { recursive: true });
  const filename = `site-${crypto.randomUUID()}.jsonl`;
  const datasetPath = path.join(directory, filename);
  await fs.promises.writeFile(datasetPath, records.map((record) => JSON.stringify(record)).join('\n') + '\n', { flag: 'wx' });
  if (signal.aborted) {
    await fs.promises.rm(datasetPath, { force: true });
    signal.throwIfAborted();
  }
  return {
    path: datasetPath, name: `${new URL(config.url).hostname} (${contentHashes.size} sources)`,
    connector: 'chat-jsonl', sizeBytes: fs.statSync(datasetPath).size,
    records: records.length, sources: contentHashes.size, attempted, failed, truncated, limitReached: queue.length > 0,
    issues, collectedAt: new Date().toISOString(),
  };
}
