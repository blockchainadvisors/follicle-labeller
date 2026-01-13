import { app, BrowserWindow, ipcMain, dialog } from 'electron';
import path from 'path';
import fs from 'fs';
import JSZip from 'jszip';

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
    title: 'Follicle Labeller',
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

// Save project as .fol archive (contains image + JSON)
ipcMain.handle('dialog:saveProject', async (_, imageData: ArrayBuffer, imageFileName: string, jsonData: string) => {
  const window = BrowserWindow.getFocusedWindow();
  if (!window) return false;

  const defaultName = imageFileName ? imageFileName.replace(/\.[^.]+$/, '.fol') : 'project.fol';

  const result = await dialog.showSaveDialog(window, {
    defaultPath: defaultName,
    filters: [
      { name: 'Follicle Project', extensions: ['fol'] }
    ]
  });

  if (result.canceled || !result.filePath) return false;

  try {
    const zip = new JSZip();

    // Add image to archive
    zip.file(imageFileName, Buffer.from(imageData));

    // Add JSON data to archive
    zip.file('annotations.json', jsonData);

    // Generate and save the archive
    const content = await zip.generateAsync({ type: 'nodebuffer', compression: 'DEFLATE' });
    fs.writeFileSync(result.filePath, content);

    return true;
  } catch (error) {
    console.error('Failed to save project:', error);
    return false;
  }
});

// Load project from .fol archive
ipcMain.handle('dialog:loadProject', async () => {
  const window = BrowserWindow.getFocusedWindow();
  if (!window) return null;

  const result = await dialog.showOpenDialog(window, {
    properties: ['openFile'],
    filters: [
      { name: 'Follicle Project', extensions: ['fol'] }
    ]
  });

  if (result.canceled || result.filePaths.length === 0) return null;

  try {
    const filePath = result.filePaths[0];
    const data = fs.readFileSync(filePath);

    const zip = await JSZip.loadAsync(data);

    // Find the image file (any file that's not annotations.json)
    let imageFileName = '';
    let imageData: ArrayBuffer | null = null;
    let jsonData = '';

    for (const fileName of Object.keys(zip.files)) {
      if (fileName === 'annotations.json') {
        jsonData = await zip.files[fileName].async('string');
      } else {
        imageFileName = fileName;
        const imageBuffer = await zip.files[fileName].async('nodebuffer');
        imageData = imageBuffer.buffer.slice(
          imageBuffer.byteOffset,
          imageBuffer.byteOffset + imageBuffer.byteLength
        );
      }
    }

    if (!imageData || !jsonData) {
      throw new Error('Invalid .fol file: missing image or annotations');
    }

    return {
      imageFileName,
      imageData,
      jsonData,
    };
  } catch (error) {
    console.error('Failed to load project:', error);
    return null;
  }
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
