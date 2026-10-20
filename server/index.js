import cors from 'cors';
import express from 'express';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import multer from 'multer';
import { runSystemCheck } from './systemCheck.js';
import { startPipelineRun, runs } from './pipeline.js';
import { samplesRoot, scriptsRoot, workspaceRoot } from './runtimePaths.js';
import { runAcceleratorSelfTest } from './acceleratorService.js';
import { prepareDatasetQueue } from './datasetQueue.js';
import { listInferenceModels, streamInference, validateConversation } from './inference.js';
import { exportConversation, validateExport } from './conversationExport.js';
import { findCompatibleTrainingPython } from './pythonRuntime.js';
import { crawlWebsite, validateCrawlRequest } from './websiteDataset.js';
import {
  activateAgentModel,
  crawlForModels,
  discoverAgentModels,
  getActiveAgentModel,
  listCheckpoints,
  moveDiscoveredModels,
  validateResumeCheckpoint,
} from './modelArtifacts.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const UPLOADS_DIR = path.join(workspaceRoot, 'uploads');

fs.mkdirSync(UPLOADS_DIR, { recursive: true });

const app = express();
app.use(cors());
app.use(express.json({ limit: '8mb' }));

app.get('/api/inference/models', async (_req, res) => {
  try {
    res.json({ ok: true, models: await listInferenceModels() });
  } catch (error) {
    res.status(503).json({ ok: false, error: `Local Ollama is unavailable. Start Ollama and install a chat model. ${error.message}` });
  }
});

app.post('/api/inference/chat', async (req, res) => {
  let conversation;
  try {
    conversation = validateConversation(req.body);
  } catch (error) {
    res.status(400).json({ ok: false, error: error.message });
    return;
  }
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15 * 60 * 1000);
  res.on('close', () => controller.abort());
  res.writeHead(200, { 'Content-Type': 'application/x-ndjson', 'Cache-Control': 'no-cache' });
  res.flushHeaders();
  try {
    await streamInference(conversation, controller.signal, (event) => res.write(`${JSON.stringify(event)}\n`));
  } catch (error) {
    if (!res.destroyed) res.write(`${JSON.stringify({ type: 'error', error: `Inference failed: ${error.message}` })}\n`);
  } finally {
    clearTimeout(timeout);
    res.end();
  }
});

app.post('/api/datasets/prepare', async (req, res) => {
  try {
    res.json({ ok: true, ...(await prepareDatasetQueue(req.body.datasets, req.body.kind)) });
  } catch (error) {
    res.status(400).json({ ok: false, error: error.message });
  }
});

app.post('/api/inference/export', async (req, res) => {
  try {
    const conversation = validateExport(req.body);
    const buffer = await exportConversation(conversation);
    res.setHeader('Content-Type', conversation.format === 'pdf'
      ? 'application/pdf'
      : 'application/vnd.openxmlformats-officedocument.wordprocessingml.document');
    res.setHeader('Content-Disposition', `attachment; filename="conversation.${conversation.format}"`);
    res.send(buffer);
  } catch (error) {
    res.status(400).json({ ok: false, error: `Could not export conversation: ${error.message}` });
  }
});

const upload = multer({ dest: UPLOADS_DIR });
const crawls = new Map();

app.post('/api/datasets/crawl', (req, res) => {
  try {
    const config = validateCrawlRequest(req.body);
    if ([...crawls.values()].some((job) => job.status === 'running')) {
      res.status(409).json({ ok: false, error: 'A site crawl is already running. Stop it or wait for completion.' });
      return;
    }
    const id = crypto.randomUUID();
    const controller = new AbortController();
    const job = { id, status: 'running', attempted: 0, collected: 0, failed: 0, pending: 1, controller };
    crawls.set(id, job);
    const timeout = setTimeout(() => controller.abort(new Error('Crawl exceeded the 30-minute limit.')), 30 * 60 * 1000);
    timeout.unref();
    crawlWebsite(config, controller.signal, (progress) => Object.assign(job, progress))
      .then((result) => Object.assign(job, { status: 'complete', result }))
      .catch((error) => Object.assign(job, { status: 'failed', error: error.message }))
      .finally(() => {
        clearTimeout(timeout);
        const retention = setTimeout(() => crawls.delete(id), 60 * 60 * 1000);
        retention.unref();
      });
    res.status(202).json({ ok: true, id });
  } catch (error) {
    res.status(400).json({ ok: false, error: error.message });
  }
});

app.get('/api/datasets/crawl/:id', (req, res) => {
  const job = crawls.get(req.params.id);
  if (!job) { res.status(404).json({ ok: false, error: 'Crawl not found. It may have expired or the service restarted.' }); return; }
  const { controller, ...status } = job;
  res.json({ ok: true, ...status });
});

app.post('/api/datasets/crawl/:id/stop', (req, res) => {
  const job = crawls.get(req.params.id);
  if (!job) { res.status(404).json({ ok: false, error: 'Crawl not found.' }); return; }
  job.controller.abort(new Error('Crawl stopped by the user. No dataset was queued.'));
  res.json({ ok: true });
});

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

app.post('/api/accelerator-self-test', async (_req, res) => {
  try {
    res.json({ ok: true, ...(await runAcceleratorSelfTest()) });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message });
  }
});

app.get('/api/checkpoints', (req, res) => {
  try {
    res.json({ ok: true, checkpoints: listCheckpoints(String(req.query.outputName || '')) });
  } catch (error) {
    res.status(400).json({ ok: false, error: error.message });
  }
});

app.get('/api/agent-models', (req, res) => {
  try {
    const discovery = discoverAgentModels(String(req.query.root || ''));
    res.json({ ok: true, ...discovery, activeModel: getActiveAgentModel() });
  } catch (error) {
    res.status(400).json({ ok: false, error: error.message });
  }
});

app.post('/api/agent-models/activate', (req, res) => {
  try {
    res.json({
      ok: true,
      activeModel: activateAgentModel(req.body?.adapterPath, req.body?.searchRoot),
    });
  } catch (error) {
    res.status(400).json({ ok: false, error: error.message });
  }
});

let selectDirectory = null;

app.post('/api/select-folder', async (req, res) => {
  if (!selectDirectory) {
    res.status(501).json({
      ok: false,
      error: 'Native folder selection is available in the Local Model Lab desktop app.',
    });
    return;
  }
  try {
    const selectedPath = await selectDirectory(String(req.body?.initialPath || ''));
    res.json({ ok: true, path: selectedPath });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message });
  }
});

app.post('/api/model-files/scan', (req, res) => {
  try {
    res.json({
      ok: true,
      ...crawlForModels(req.body?.sourceRoot, req.body?.destinationRoot),
    });
  } catch (error) {
    res.status(400).json({ ok: false, error: error.message });
  }
});

app.post('/api/model-files/move', (req, res) => {
  try {
    res.json({
      ok: true,
      ...moveDiscoveredModels(req.body?.sourceRoot, req.body?.destinationRoot),
    });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message, moved: error.moved || [] });
  }
});

// Real list of sample dataset files that actually exist on disk.
app.get('/api/samples', (_req, res) => {
  try {
    const files = fs.readdirSync(samplesRoot).filter((name) => name.endsWith('.jsonl'));
    res.json({
      ok: true,
      samples: files.map((name) => {
        const fullPath = path.join(samplesRoot, name);
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

  const finalPath = path.join(UPLOADS_DIR, `${crypto.randomUUID()}-${path.basename(req.file.originalname)}`);

  try {
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

// Real video dataset upload — writes the manifest .jsonl and every uploaded
// video file to a real per-upload folder on disk, then reports back where
// they actually landed (used by validate/train stages via --videos-dir).
const VIDEO_UPLOADS_DIR = path.join(UPLOADS_DIR, 'video-datasets');
fs.mkdirSync(VIDEO_UPLOADS_DIR, { recursive: true });

const videoUpload = multer({ dest: path.join(UPLOADS_DIR, 'tmp-video') });
fs.mkdirSync(path.join(UPLOADS_DIR, 'tmp-video'), { recursive: true });

const ALLOWED_VIDEO_EXTENSIONS = new Set(['.mp4', '.mov', '.mkv', '.webm', '.avi', '.m4v', '.mpeg', '.mpg']);

app.post(
  '/api/upload-video-dataset',
  videoUpload.fields([
    { name: 'manifest', maxCount: 1 },
    { name: 'videos', maxCount: 500 },
  ]),
  (req, res) => {
    const manifestFile = req.files?.manifest?.[0];
    const videoFiles = req.files?.videos || [];

    if (!manifestFile) {
      res.status(400).json({ ok: false, error: 'No manifest file was uploaded (expected field name "manifest").' });
      return;
    }
    if (!manifestFile.originalname.toLowerCase().endsWith('.jsonl')) {
      res.status(400).json({ ok: false, error: 'The manifest must be a .jsonl file.' });
      return;
    }
    if (videoFiles.length === 0) {
      res.status(400).json({ ok: false, error: 'No video files were uploaded (expected field name "videos").' });
      return;
    }

    const badVideo = videoFiles.find(
      (file) => !ALLOWED_VIDEO_EXTENSIONS.has(path.extname(file.originalname).toLowerCase())
    );
    if (badVideo) {
      res.status(400).json({
        ok: false,
        error: `'${badVideo.originalname}' has an unsupported video extension. Supported: ${[...ALLOWED_VIDEO_EXTENSIONS].join(', ')}.`,
      });
      return;
    }

    const uploadId = crypto.randomUUID();
    const destDir = path.join(VIDEO_UPLOADS_DIR, uploadId);
    const videosDir = path.join(destDir, 'videos');
    fs.mkdirSync(videosDir, { recursive: true });

    try {
      const manifestDestPath = path.join(destDir, 'manifest.jsonl');
      fs.renameSync(manifestFile.path, manifestDestPath);

      for (const file of videoFiles) {
        fs.renameSync(file.path, path.join(videosDir, file.originalname));
      }

      res.json({
        ok: true,
        manifestPath: manifestDestPath,
        videosDir,
        videoCount: videoFiles.length,
      });
    } catch (error) {
      res.status(500).json({ ok: false, error: `Could not save the uploaded video dataset: ${error.message}` });
    }
  }
);

// Real dataset validation — actually runs the Python validator against a
// real file path and returns the real per-line result. Dispatches to the
// video-manifest validator when kind === 'video', since the schema and
// per-file checks (video extension, existence, ffprobe) differ from the
// plain text/chat JSONL validator.
app.post('/api/validate-dataset', async (req, res) => {
  const { datasetPath, kind, videosDir } = req.body;
  if (kind && !['text', 'video'].includes(kind)) {
    res.status(400).json({ ok: false, error: 'Unsupported dataset kind.' });
    return;
  }
  if (!datasetPath || !fs.existsSync(datasetPath)) {
    res.status(400).json({ ok: false, error: `File not found: ${datasetPath}` });
    return;
  }

  const { spawn } = await import('node:child_process');
  const runtime = await findCompatibleTrainingPython();
  if (!runtime.compatible) {
    res.status(503).json({ ok: false, error: 'No compatible Python was found. Prepare the training environment first or configure MODEL_TRAINER_PYTHON.' });
    return;
  }
  const pythonBin = runtime.command;
  const scriptArgs =
    kind === 'video'
      ? [path.join(scriptsRoot, 'validate_video_dataset.py'), datasetPath, '--videos-dir', videosDir || path.dirname(datasetPath)]
      : [path.join(scriptsRoot, 'validate_dataset.py'), datasetPath];
  const child = spawn(pythonBin, scriptArgs, {
    windowsHide: true,
  });

  let stdout = '';
  let stderr = '';
  child.stdout.on('data', (chunk) => {
    stdout += chunk.toString('utf8');
  });
  child.stderr.on('data', (chunk) => { stderr += chunk.toString('utf8'); });
  child.on('error', (error) => {
    if (!res.headersSent) res.status(500).json({ ok: false, error: `Could not start the dataset validator: ${error.message}` });
  });
  child.on('close', (code) => {
    if (res.headersSent) return;
    try {
      const result = JSON.parse(stdout);
      res.json({ ...result, ok: code === 0 && result.ok === true });
    } catch {
      res.status(500).json({ ok: false, error: `Validator failed to return JSON (exit ${code}). ${stderr.slice(-1000)}` });
    }
  });
});

// Real pipeline run — starts REAL child processes for each selected stage
// and streams REAL output over Server-Sent Events. No timers, no fake
// progress.
app.post('/api/run', async (req, res) => {
  const { stages, config } = req.body;

  if (config?.datasets) {
    try {
      const prepared = await prepareDatasetQueue(config.datasets, config.datasetKind);
      config.datasetPath = prepared.path;
      config.videosDir = prepared.videosDir;
    } catch (error) {
      res.status(400).json({ ok: false, error: error.message });
      return;
    }
  }
  if (!config?.datasetPath || !fs.existsSync(config.datasetPath)) {
    res.status(400).json({ ok: false, error: `Dataset file not found: ${config?.datasetPath}` });
    return;
  }
  if (!Array.isArray(stages) || stages.length === 0) {
    res.status(400).json({ ok: false, error: 'Select at least one pipeline stage.' });
    return;
  }
  if (
    !Number.isInteger(config.epochs) ||
    config.epochs <= 0 ||
    !Number.isInteger(config.batchSize) ||
    config.batchSize <= 0 ||
    !Number.isFinite(config.learningRate) ||
    config.learningRate <= 0
  ) {
    res.status(400).json({
      ok: false,
      error: 'Epochs and batch size must be positive integers, and learning rate must be positive.',
    });
    return;
  }

  let resumeCheckpoint = null;
  try {
    resumeCheckpoint = validateResumeCheckpoint(config.name, config.resumeCheckpoint);
  } catch (error) {
    res.status(400).json({ ok: false, error: error.message });
    return;
  }

  const runId = crypto.randomUUID();
  const run = await startPipelineRun(runId, { ...config, resumeCheckpoint, stages });

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

  const history = run.events.slice();
  run.emitter.on('event', listener);
  history.forEach(send);
  if (run.done) {
    run.emitter.off('event', listener);
    res.end();
  }

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

export function startServer({
  port = process.env.PORT || 8787,
  host = '127.0.0.1',
  chooseDirectory = null,
} = {}) {
  selectDirectory = chooseDirectory;
  return new Promise((resolve, reject) => {
    const server = app.listen(port, host, () => {
      const address = server.address();
      console.log(`Model trainer backend running at http://${host}:${address.port}`);
      resolve(server);
    });
    server.once('error', reject);
  });
}

if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(__dirname, 'index.js')) {
  startServer().catch((error) => {
    console.error('Could not start the model trainer backend:', error);
    process.exitCode = 1;
  });
}
