import cors from 'cors';
import express from 'express';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import multer from 'multer';
import { runSystemCheck } from './systemCheck.js';
import { startPipelineRun, runs } from './pipeline.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..');
const UPLOADS_DIR = path.join(REPO_ROOT, 'uploads');
const SAMPLES_DIR = path.join(REPO_ROOT, 'samples');

fs.mkdirSync(UPLOADS_DIR, { recursive: true });

const app = express();
app.use(cors());
app.use(express.json());

const upload = multer({ dest: UPLOADS_DIR });

// Real health check — proves the backend is actually running, not simulated.
app.get('/api/health', (_req, res) => {
  res.json({ ok: true, message: 'Model trainer backend is running.', time: new Date().toISOString() });
});

// Real system check — actually probes CPU vendor, GPU names, RAM, and
// whether python/ollama/wsl are really installed on THIS machine.
app.get('/api/system-check', async (_req, res) => {
  try {
    const result = await runSystemCheck();
    res.json(result);
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message });
  }
});

// Real list of sample dataset files that actually exist on disk.
app.get('/api/samples', (_req, res) => {
  try {
    const files = fs.readdirSync(SAMPLES_DIR).filter((name) => name.endsWith('.jsonl'));
    res.json({
      ok: true,
      samples: files.map((name) => {
        const fullPath = path.join(SAMPLES_DIR, name);
        const stats = fs.statSync(fullPath);
        return { name, path: fullPath, sizeBytes: stats.size };
      }),
    });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message });
  }
});

// Real file upload — the file is actually written to disk on the server.
app.post('/api/upload-dataset', upload.single('file'), (req, res) => {
  if (!req.file) {
    res.status(400).json({ ok: false, error: 'No file was uploaded.' });
    return;
  }

  if (!req.file.originalname.toLowerCase().endsWith('.jsonl')) {
    fs.unlinkSync(req.file.path);
    res.status(400).json({ ok: false, error: 'Only .jsonl files are accepted.' });
    return;
  }

  const finalPath = path.join(UPLOADS_DIR, req.file.originalname);

  try {
    // If a file with this name already exists (e.g. re-uploading the same
    // dataset), remove it first. On Windows, fs.renameSync can throw a real
    // EPERM error when the destination already exists and is momentarily
    // locked (for example by antivirus scanning it), even though the rename
    // is otherwise valid.
    if (fs.existsSync(finalPath)) {
      fs.unlinkSync(finalPath);
    }
    fs.renameSync(req.file.path, finalPath);
  } catch (error) {
    // Real fallback for cross-device or lock-related failures: copy the
    // bytes over, then remove the temp file, rather than failing the whole
    // upload for a real (but recoverable) filesystem quirk.
    try {
      fs.copyFileSync(req.file.path, finalPath);
      fs.unlinkSync(req.file.path);
    } catch (fallbackError) {
      res.status(500).json({ ok: false, error: `Could not save the uploaded file: ${fallbackError.message}` });
      return;
    }
  }

  res.json({ ok: true, path: finalPath, sizeBytes: req.file.size });
});

// Real dataset validation — actually runs the Python validator against a
// real file path and returns the real per-line result.
app.post('/api/validate-dataset', async (req, res) => {
  const { datasetPath } = req.body;
  if (!datasetPath || !fs.existsSync(datasetPath)) {
    res.status(400).json({ ok: false, error: `File not found: ${datasetPath}` });
    return;
  }

  const { spawn } = await import('node:child_process');
  const pythonBin = process.platform === 'win32' ? 'python' : 'python3';
  const child = spawn(pythonBin, [path.join(REPO_ROOT, 'scripts', 'validate_dataset.py'), datasetPath], {
    windowsHide: true,
  });

  let stdout = '';
  child.stdout.on('data', (chunk) => {
    stdout += chunk.toString('utf8');
  });
  child.on('close', () => {
    try {
      res.json(JSON.parse(stdout));
    } catch {
      res.status(500).json({ ok: false, error: 'Validator did not return valid JSON.', raw: stdout });
    }
  });
});

// Real pipeline run — starts REAL child processes for each selected stage
// and streams REAL output over Server-Sent Events. No timers, no fake
// progress.
app.post('/api/run', async (req, res) => {
  const { stages, config } = req.body;

  if (!config?.datasetPath || !fs.existsSync(config.datasetPath)) {
    res.status(400).json({ ok: false, error: `Dataset file not found: ${config?.datasetPath}` });
    return;
  }

  const runId = crypto.randomUUID();
  const run = await startPipelineRun(runId, { ...config, stages });

  res.json({ ok: true, runId: run.id });
});

// Real Server-Sent Events stream of what actually happened in a run.
app.get('/api/run/:runId/events', (req, res) => {
  const run = runs.get(req.params.runId);
  if (!run) {
    res.status(404).end();
    return;
  }

  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    Connection: 'keep-alive',
  });

  const send = (event) => {
    res.write(`data: ${JSON.stringify(event)}\n\n`);
  };

  const listener = (event) => {
    send(event);
    if (event.type === 'run-complete') {
      res.end();
    }
  };

  run.emitter.on('event', listener);

  req.on('close', () => {
    run.emitter.off('event', listener);
  });
});

// Real safety net: if any route handler throws synchronously (like the
// EPERM rename bug we hit for real), return real JSON instead of Express's
// default HTML error page, which the frontend cannot parse as JSON.
app.use((error, _req, res, _next) => {
  console.error('Unhandled route error:', error);
  if (!res.headersSent) {
    res.status(500).json({ ok: false, error: error.message || 'Unknown server error.' });
  }
});

const PORT = process.env.PORT || 8787;
app.listen(PORT, '127.0.0.1', () => {
  console.log(`Model trainer backend really running at http://127.0.0.1:${PORT}`);
});
