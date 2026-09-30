import { app, BrowserWindow, dialog } from 'electron';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
let apiServer;
let mainWindow;

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
  apiServer = await startServer({
    port: 0,
    chooseDirectory: async (initialPath) => {
      const result = await dialog.showOpenDialog(mainWindow, {
        title: 'Select a model folder',
        defaultPath: initialPath || app.getPath('documents'),
        properties: ['openDirectory', 'createDirectory'],
      });
      return result.canceled ? null : result.filePaths[0];
    },
  });
  const address = apiServer.address();
  const apiBase = `http://127.0.0.1:${address.port}`;

  mainWindow = new BrowserWindow({
    width: 1440,
    height: 960,
    minWidth: 1000,
    minHeight: 700,
    show: false,
    title: 'Local Model Lab',
    icon: path.join(path.resolve(__dirname, '..'), 'build', 'icon.ico'),
    autoHideMenuBar: true,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });

  mainWindow.once('ready-to-show', () => {
    mainWindow.center();
    mainWindow.show();
    mainWindow.restore();
    mainWindow.focus();
    mainWindow.moveTop();
  });
  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  await mainWindow.loadFile(path.join(__dirname, '..', 'dist', 'index.html'), {
    query: { apiBase },
  });
}

const hasSingleInstanceLock = app.requestSingleInstanceLock();

if (!hasSingleInstanceLock) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.show();
      mainWindow.focus();
      mainWindow.moveTop();
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
