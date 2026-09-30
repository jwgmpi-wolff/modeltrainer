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

app.get('/api/agent-models', (_req, res) => {
  try {
    res.json({ ok: true, models: discoverAgentModels(), activeModel: getActiveAgentModel() });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message });
  }
});

app.post('/api/agent-models/activate', (req, res) => {
  try {
    res.json({ ok: true, activeModel: activateAgentModel(req.body?.adapterPath) });
  } catch (error) {
    res.status(400).json({ ok: false, error: error.message });
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
  if (!datasetPath || !fs.existsSync(datasetPath)) {
    res.status(400).json({ ok: false, error: `File not found: ${datasetPath}` });
    return;
  }

  const { spawn } = await import('node:child_process');
  const pythonBin = process.platform === 'win32' ? 'python' : 'python3';
  const scriptArgs =
    kind === 'video'
      ? [path.join(scriptsRoot, 'validate_video_dataset.py'), datasetPath, '--videos-dir', videosDir]
      : [path.join(scriptsRoot, 'validate_dataset.py'), datasetPath];
  const child = spawn(pythonBin, scriptArgs, {
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

export function startServer({ port = process.env.PORT || 8787, host = '127.0.0.1' } = {}) {
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
