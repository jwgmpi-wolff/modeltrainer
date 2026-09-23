import cors from 'cors';
import express from 'express';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { spawn } from 'node:child_process';
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

function runCommand(command, args, options = {}) {
  return new Promise((resolve) => {
    const lines = [`$ ${command} ${args.join(' ')}`];
    const child = spawn(command, args, { cwd: REPO_ROOT, windowsHide: true, shell: process.platform === 'win32', ...options });

    child.stdout?.on('data', (chunk) => {
      lines.push(...chunk.toString('utf8').split(/\r?\n/).filter(Boolean));
    });

    child.stderr?.on('data', (chunk) => {
      lines.push(...chunk.toString('utf8').split(/\r?\n/).filter(Boolean).map((line) => `stderr: ${line}`));
    });

    child.on('error', (error) => {
      lines.push(`ERROR: ${error.message}`);
      resolve({ ok: false, exitCode: null, lines });
    });

    child.on('close', (exitCode) => {
      resolve({ ok: exitCode === 0, exitCode, lines });
    });
  });
}

function writeFileEnsuringDir(relativePath, content) {
  const fullPath = path.join(REPO_ROOT, relativePath);
  fs.mkdirSync(path.dirname(fullPath), { recursive: true });
  fs.writeFileSync(fullPath, content, 'utf8');
  return fullPath;
}

function githubPagesWorkflow() {
  return `name: Deploy Local Model Lab to GitHub Pages

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm
      - run: npm ci
      - run: npm run build
      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v3
        with:
          path: dist

  deploy:
    environment:
      name: github-pages
      url: \${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    needs: build
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
`;
}

function azureYaml() {
  return `name: local-model-lab
services:
  web:
    project: .
    language: js
    host: staticwebapp
`;
}

function azureBicep() {
  return `@description('Name prefix for Azure resources')
param resourcePrefix string = 'local-model-lab'

@description('Azure region')
param location string = resourceGroup().location

@allowed([
  'Free'
  'Standard'
])
@description('Azure Static Web Apps SKU')
param sku string = 'Free'

var uniqueSuffix = uniqueString(resourceGroup().id)

resource staticWebApp 'Microsoft.Web/staticSites@2022-09-01' = {
  name: '\${resourcePrefix}-swa-\${uniqueSuffix}'
  location: location
  sku: {
    name: sku
    tier: sku
  }
  properties: {
    buildProperties: {
      appLocation: '/'
      outputLocation: 'dist'
    }
  }
  tags: {
    'azd-service-name': 'web'
  }
}

output endpoint string = 'https://\${staticWebApp.properties.defaultHostname}'
output resourceId string = staticWebApp.id
`;
}

function staticWebAppConfig() {
  return `{
  "navigationFallback": {
    "rewrite": "/index.html",
    "exclude": [
      "/assets/*",
      "/*.{css,js,ico,png,jpg,jpeg,gif,svg,webp,woff,woff2}"
    ]
  },
  "globalHeaders": {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY"
  }
}
`;
}

function localDeployScript() {
  return `param(
  [int]$FrontendPort = 3000,
  [int]$BackendPort = 8787
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

npm run install:all
npm run build

Write-Host "Start backend with: $env:PORT=$BackendPort; npm run dev:server"
Write-Host "Start frontend preview with: npm run preview -- --host 0.0.0.0 --port $FrontendPort"
`;
}

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

// Real deployment action runner. This does not fake a deployment: it writes
// real deploy files, runs the real production build, and reports exactly what
// it did. Cloud deployment files are generated locally so users can review and
// commit them before a GitHub/Azure provider deploys them.
app.post('/api/deploy', async (req, res) => {
  const { target } = req.body;
  const lines = [];
  const writtenFiles = [];

  try {
    if (!['local', 'github-pages', 'azure-static-web-app'].includes(target)) {
      res.status(400).json({ ok: false, error: `Unknown deploy target: ${target}` });
      return;
    }

    lines.push(`DEPLOY START: ${target}`);

    if (target === 'local') {
      writtenFiles.push(writeFileEnsuringDir('scripts/deploy-local.ps1', localDeployScript()));
      lines.push('Wrote real local deployment helper script.');
    }

    if (target === 'github-pages') {
      writtenFiles.push(writeFileEnsuringDir('.github/workflows/deploy-pages.yml', githubPagesWorkflow()));
      lines.push('Wrote real GitHub Pages workflow at .github/workflows/deploy-pages.yml.');
    }

    if (target === 'azure-static-web-app') {
      writtenFiles.push(writeFileEnsuringDir('azure.yaml', azureYaml()));
      writtenFiles.push(writeFileEnsuringDir('infra/main.bicep', azureBicep()));
      writtenFiles.push(writeFileEnsuringDir('public/staticwebapp.config.json', staticWebAppConfig()));
      lines.push('Wrote real Azure Developer CLI Static Web Apps files: azure.yaml, infra/main.bicep, public/staticwebapp.config.json.');

      const azdCheck = await runCommand('azd', ['version']);
      lines.push(...azdCheck.lines);
      if (!azdCheck.ok) {
        lines.push('AZD CHECK: Azure Developer CLI was not found or failed. Install azd, then run "azd up" from this folder.');
      } else {
        lines.push('AZD CHECK: Azure Developer CLI is available. Run "azd up" from this folder to provision/deploy Azure Static Web Apps.');
      }
    }

    const build = await runCommand('npm', ['run', 'build']);
    lines.push(...build.lines);

    if (!build.ok) {
      res.status(500).json({
        ok: false,
        target,
        error: `Production build failed with exit code ${build.exitCode}.`,
        writtenFiles,
        lines,
      });
      return;
    }

    lines.push(`DEPLOY READY: ${target}`);
    res.json({ ok: true, target, writtenFiles, lines });
  } catch (error) {
    res.status(500).json({ ok: false, target, error: error.message, writtenFiles, lines });
  }
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
