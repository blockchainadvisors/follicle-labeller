import { app, BrowserWindow, ipcMain, dialog } from 'electron';
import path from 'path';
import fs from 'fs';

let mainWindow: BrowserWindow | null = null;

function createWindow(): void {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 700,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
    title: 'Follicle Labeler',
  });

  // Load content based on environment
  if (process.env.NODE_ENV === 'development' || process.env.VITE_DEV_SERVER_URL) {
    mainWindow.loadURL(process.env.VITE_DEV_SERVER_URL || 'http://localhost:5173');
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, '../dist/index.html'));
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// IPC Handlers

// Open image file dialog
ipcMain.handle('dialog:openImage', async () => {
  const window = BrowserWindow.getFocusedWindow();
  if (!window) return null;

  const result = await dialog.showOpenDialog(window, {
    properties: ['openFile'],
    filters: [
      { name: 'Images', extensions: ['png', 'jpg', 'jpeg', 'tiff', 'tif', 'bmp', 'webp'] }
    ]
  });

  if (result.canceled || result.filePaths.length === 0) return null;

  const filePath = result.filePaths[0];
  const data = fs.readFileSync(filePath);

  return {
    filePath,
    fileName: path.basename(filePath),
    data: data.buffer,
  };
});

// Save JSON file dialog
ipcMain.handle('dialog:saveJson', async (_, data: string) => {
  const window = BrowserWindow.getFocusedWindow();
  if (!window) return false;

  const result = await dialog.showSaveDialog(window, {
    defaultPath: 'follicle-labels.json',
    filters: [
      { name: 'JSON', extensions: ['json'] }
    ]
  });

  if (result.canceled || !result.filePath) return false;

  fs.writeFileSync(result.filePath, data, 'utf-8');
  return true;
});

// Open JSON file dialog (import)
ipcMain.handle('dialog:openJson', async () => {
  const window = BrowserWindow.getFocusedWindow();
  if (!window) return null;

  const result = await dialog.showOpenDialog(window, {
    properties: ['openFile'],
    filters: [
      { name: 'JSON', extensions: ['json'] }
    ]
  });

  if (result.canceled || result.filePaths.length === 0) return null;

  const filePath = result.filePaths[0];
  const data = fs.readFileSync(filePath, 'utf-8');

  return { filePath, data };
});

// App lifecycle
app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
