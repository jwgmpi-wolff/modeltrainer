import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import readline from 'node:readline';
import { workspaceRoot } from './runtimePaths.js';

const connectors = new Set(['chat-jsonl', 'prompt-completion-jsonl', 'video-jsonl']);
const roles = new Set(['system', 'user', 'assistant']);

export async function prepareDatasetQueue(datasets, kind = 'text', root = workspaceRoot) {
  if (!['text', 'video'].includes(kind)) throw new Error('Choose text or video training.');
  if (!Array.isArray(datasets) || datasets.length === 0 || datasets.length > 100) {
    throw new Error('Queue between 1 and 100 datasets.');
  }
  const directory = path.join(root, 'uploads', 'prepared');
  fs.mkdirSync(directory, { recursive: true });
  const temporaryPath = path.join(directory, `${crypto.randomUUID()}.tmp`);
  const output = await fs.promises.open(temporaryPath, 'wx');
  const hash = crypto.createHash('sha256');
  let count = 0;
  try {
    for (const dataset of datasets) {
      if (!dataset || typeof dataset.path !== 'string' || !connectors.has(dataset.connector)) {
        throw new Error('Every queued dataset needs a local file path and a supported connector.');
      }
      if ((dataset.connector === 'video-jsonl') !== (kind === 'video')) {
        throw new Error('Text and video connectors cannot be mixed in a training run.');
      }
      if (!fs.statSync(dataset.path).isFile()) throw new Error(`Not a dataset file: ${dataset.path}`);
      const input = fs.createReadStream(dataset.path, { encoding: 'utf8' });
      const lines = readline.createInterface({ input, crlfDelay: Infinity });
      let lineNumber = 0;
      let fileRecords = 0;
      try {
        for await (const raw of lines) {
          lineNumber += 1;
          const line = raw.replace(/^\uFEFF/, '').trim();
          if (!line) continue;
          try {
            let record = JSON.parse(line);
            if (!record || typeof record !== 'object' || Array.isArray(record)) {
              throw new Error('Each record must be an object.');
            }
            if (kind === 'text' && record.video != null) {
              throw new Error('Video records require the video connector and a vision-language model.');
            }
            if (dataset.connector === 'prompt-completion-jsonl') {
              if (typeof record.prompt !== 'string' || typeof record.completion !== 'string') {
                throw new Error('The prompt/completion connector requires string prompt and completion fields.');
              }
              record = { messages: [
                { role: 'user', content: record.prompt },
                { role: 'assistant', content: record.completion },
              ] };
            }
            if (!Array.isArray(record.messages) || !record.messages.length ||
                record.messages.some((message) => !message || !roles.has(message.role) ||
                  typeof message.content !== 'string' || !message.content.trim()) ||
                !record.messages.some((message) => message.role === 'user') ||
                !record.messages.some((message) => message.role === 'assistant')) {
              throw new Error('Expected non-empty messages with system/user/assistant roles, including user and assistant.');
            }
            if (kind === 'video') {
              if (typeof record.video !== 'string' || !record.video.trim()) {
                throw new Error('The video connector requires a video filename.');
              }
              const video = path.resolve(dataset.videosDir || path.dirname(dataset.path), record.video);
              if (!fs.statSync(video).isFile()) throw new Error(`Video file not found: ${record.video}`);
              record = { ...record, video };
            }
            const normalized = `${JSON.stringify(record)}\n`;
            await output.write(normalized);
            hash.update(normalized);
            count += 1;
            fileRecords += 1;
          } catch (error) {
            throw new Error(`${path.basename(dataset.path)}, line ${lineNumber}: ${error.message}`);
          }
        }
      } finally {
        lines.close();
        input.destroy();
      }
      if (!fileRecords) throw new Error(`Dataset is empty: ${path.basename(dataset.path)}`);
    }
    await output.close();
    const digest = hash.digest('hex');
    const preparedPath = path.join(directory, `${kind}-${digest}.jsonl`);
    if (fs.existsSync(preparedPath)) await fs.promises.unlink(temporaryPath);
    else await fs.promises.rename(temporaryPath, preparedPath);
    return { path: preparedPath, videosDir: directory, records: count, datasets: datasets.length, sha256: digest };
  } catch (error) {
    await output.close();
    await fs.promises.rm(temporaryPath, { force: true });
    throw error;
  }
}
