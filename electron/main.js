import { app, BrowserWindow, dialog } from 'electron';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
let apiServer;

function configureRuntimePaths() {
  const appRoot = path.resolve(__dirname, '..');
  const resourcesRoot = app.isPackaged ? process.resourcesPath : appRoot;
  const workspaceRoot = app.isPackaged
    ? path.join(app.getPath('userData'), 'workspace')
    : appRoot;

  process.env.MODEL_TRAINER_WORKSPACE = workspaceRoot;
  process.env.MODEL_TRAINER_SCRIPTS_ROOT = path.join(resourcesRoot, 'scripts');
  process.env.MODEL_TRAINER_SAMPLES_ROOT = path.join(resourcesRoot, 'samples');
}

async function createWindow() {
  configureRuntimePaths();
  const { startServer } = await import('../server/index.js');
  apiServer = await startServer({ port: 0 });
  const address = apiServer.address();
  const apiBase = `http://127.0.0.1:${address.port}`;

  const window = new BrowserWindow({
    width: 1440,
    height: 960,
    minWidth: 1000,
    minHeight: 700,
    title: 'Local Model Lab',
    autoHideMenuBar: true,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });

  await window.loadFile(path.join(__dirname, '..', 'dist', 'index.html'), {
    query: { apiBase },
  });
}

const hasSingleInstanceLock = app.requestSingleInstanceLock();

if (!hasSingleInstanceLock) {
  app.quit();
} else {
  app.on('second-instance', () => {
    const window = BrowserWindow.getAllWindows()[0];
    if (window) {
      if (window.isMinimized()) window.restore();
      window.focus();
    }
  });

  app.whenReady().then(createWindow).catch((error) => {
    dialog.showErrorBox('Local Model Lab could not start', error.stack || error.message);
    app.quit();
  });

  app.on('window-all-closed', () => {
    app.quit();
  });

  app.on('before-quit', () => {
    apiServer?.close();
  });
}
