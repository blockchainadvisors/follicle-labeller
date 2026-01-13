import { app, BrowserWindow, ipcMain, dialog, Menu, MenuItemConstructorOptions } from 'electron';
import path from 'path';
import fs from 'fs';
import JSZip from 'jszip';

let mainWindow: BrowserWindow | null = null;

function createWindow(): void {
  // Determine icon path based on platform
  const iconName = process.platform === 'win32' ? 'icon.ico' : 'icon.png';
  const iconPath = app.isPackaged
    ? path.join(process.resourcesPath, iconName)
    : path.join(__dirname, '../public', iconName);

  // Check if icon exists (may not exist in dev if not generated)
  const iconExists = fs.existsSync(iconPath);

  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 700,
    ...(iconExists && { icon: iconPath }),
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

// Load project from .fol archive (legacy V1)
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

// Helper function to save project to a specific path
async function saveProjectToPath(
  filePath: string,
  images: Array<{ id: string; fileName: string; data: ArrayBuffer }>,
  manifestJson: string,
  annotationsJson: string
): Promise<boolean> {
  try {
    const zip = new JSZip();

    // Add manifest
    zip.file('manifest.json', manifestJson);

    // Add images in images/ folder
    const imagesFolder = zip.folder('images');
    if (imagesFolder) {
      for (const image of images) {
        const archiveFileName = `${image.id}-${image.fileName}`;
        imagesFolder.file(archiveFileName, Buffer.from(image.data));
      }
    }

    // Add annotations
    zip.file('annotations.json', annotationsJson);

    // Generate and save the archive
    const content = await zip.generateAsync({ type: 'nodebuffer', compression: 'DEFLATE' });
    fs.writeFileSync(filePath, content);

    return true;
  } catch (error) {
    console.error('Failed to save project:', error);
    return false;
  }
}

// V2 Save project with multiple images (Save As - shows dialog)
ipcMain.handle('dialog:saveProjectV2', async (
  _,
  images: Array<{ id: string; fileName: string; data: ArrayBuffer }>,
  manifestJson: string,
  annotationsJson: string,
  defaultPath?: string
) => {
  const window = BrowserWindow.getFocusedWindow();
  if (!window) return { success: false };

  const result = await dialog.showSaveDialog(window, {
    defaultPath: defaultPath || 'project.fol',
    filters: [
      { name: 'Follicle Project', extensions: ['fol'] }
    ]
  });

  if (result.canceled || !result.filePath) return { success: false };

  const success = await saveProjectToPath(result.filePath, images, manifestJson, annotationsJson);
  return { success, filePath: success ? result.filePath : undefined };
});

// V2 Save project to specific path (silent save - no dialog)
ipcMain.handle('file:saveProjectV2', async (
  _,
  filePath: string,
  images: Array<{ id: string; fileName: string; data: ArrayBuffer }>,
  manifestJson: string,
  annotationsJson: string
) => {
  const success = await saveProjectToPath(filePath, images, manifestJson, annotationsJson);
  return { success, filePath: success ? filePath : undefined };
});

// V2 Load project with support for both V1 and V2 formats
ipcMain.handle('dialog:loadProjectV2', async () => {
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

    // Check if this is V2 (has manifest.json)
    const hasManifest = 'manifest.json' in zip.files;

    if (hasManifest) {
      // V2 format
      const manifest = await zip.files['manifest.json'].async('string');
      const annotations = await zip.files['annotations.json'].async('string');

      // Load all images from images/ folder
      const images: Array<{ id: string; fileName: string; data: ArrayBuffer }> = [];
      const imagesFolder = zip.folder('images');

      if (imagesFolder) {
        for (const [relativePath, file] of Object.entries(imagesFolder.files)) {
          if (file.dir) continue;

          // Extract filename from path (remove 'images/' prefix)
          const archiveFileName = relativePath.replace('images/', '');
          if (!archiveFileName) continue;

          // Parse id and fileName from archiveFileName (format: {id}-{fileName})
          const dashIndex = archiveFileName.indexOf('-');
          if (dashIndex === -1) continue;

          const id = archiveFileName.substring(0, dashIndex);
          const fileName = archiveFileName.substring(dashIndex + 1);

          const imageBuffer = await file.async('nodebuffer');
          const imageData = imageBuffer.buffer.slice(
            imageBuffer.byteOffset,
            imageBuffer.byteOffset + imageBuffer.byteLength
          );

          images.push({ id, fileName, data: imageData });
        }
      }

      return {
        version: '2.0' as const,
        filePath,
        manifest,
        images,
        annotations,
      };
    } else {
      // V1 format - return in V1 structure for migration
      let imageFileName = '';
      let imageData: ArrayBuffer | null = null;
      let jsonData = '';

      for (const fileName of Object.keys(zip.files)) {
        if (fileName === 'annotations.json') {
          jsonData = await zip.files[fileName].async('string');
        } else if (!zip.files[fileName].dir) {
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
        version: '1.0' as const,
        filePath,
        imageFileName,
        imageData,
        jsonData,
      };
    }
  } catch (error) {
    console.error('Failed to load project:', error);
    return null;
  }
});

// Create application menu
function createMenu(): void {
  const isMac = process.platform === 'darwin';

  const template: MenuItemConstructorOptions[] = [
    // macOS app menu
    ...(isMac ? [{
      label: app.name,
      submenu: [
        { role: 'about' as const },
        { type: 'separator' as const },
        { role: 'quit' as const },
      ]
    }] : []),

    // File menu
    {
      label: 'File',
      submenu: [
        {
          label: 'Open Image...',
          accelerator: 'CmdOrCtrl+O',
          click: () => mainWindow?.webContents.send('menu:openImage'),
        },
        { type: 'separator' },
        {
          label: 'Load Project...',
          accelerator: 'CmdOrCtrl+Shift+O',
          click: () => mainWindow?.webContents.send('menu:loadProject'),
        },
        {
          label: 'Save Project',
          accelerator: 'CmdOrCtrl+S',
          click: () => mainWindow?.webContents.send('menu:saveProject'),
        },
        {
          label: 'Save Project As...',
          accelerator: 'CmdOrCtrl+Shift+S',
          click: () => mainWindow?.webContents.send('menu:saveProjectAs'),
        },
        { type: 'separator' },
        isMac ? { role: 'close' as const } : { role: 'quit' as const },
      ],
    },

    // Edit menu
    {
      label: 'Edit',
      submenu: [
        {
          label: 'Undo',
          accelerator: 'CmdOrCtrl+Z',
          click: () => mainWindow?.webContents.send('menu:undo'),
        },
        {
          label: 'Redo',
          accelerator: 'CmdOrCtrl+Shift+Z',
          click: () => mainWindow?.webContents.send('menu:redo'),
        },
        { type: 'separator' },
        {
          label: 'Clear All Annotations',
          click: () => mainWindow?.webContents.send('menu:clearAll'),
        },
      ],
    },

    // View menu
    {
      label: 'View',
      submenu: [
        {
          label: 'Toggle Shapes (O)',
          click: () => mainWindow?.webContents.send('menu:toggleShapes'),
        },
        {
          label: 'Toggle Labels (L)',
          click: () => mainWindow?.webContents.send('menu:toggleLabels'),
        },
        { type: 'separator' },
        {
          label: 'Zoom In',
          accelerator: 'CmdOrCtrl+=',
          click: () => mainWindow?.webContents.send('menu:zoomIn'),
        },
        {
          label: 'Zoom Out',
          accelerator: 'CmdOrCtrl+-',
          click: () => mainWindow?.webContents.send('menu:zoomOut'),
        },
        {
          label: 'Reset Zoom',
          accelerator: 'CmdOrCtrl+0',
          click: () => mainWindow?.webContents.send('menu:resetZoom'),
        },
        { type: 'separator' },
        { role: 'toggleDevTools' },
      ],
    },

    // Help menu
    {
      label: 'Help',
      submenu: [
        {
          label: 'User Guide',
          accelerator: 'F1',
          click: () => mainWindow?.webContents.send('menu:showHelp'),
        },
        { type: 'separator' },
        {
          label: 'About Follicle Labeller',
          click: () => {
            if (mainWindow) {
              dialog.showMessageBox(mainWindow, {
                type: 'info',
                title: 'About Follicle Labeller',
                message: 'Follicle Labeller v1.0.0',
                detail: 'Medical image annotation tool for follicle labeling.',
              });
            }
          },
        },
      ],
    },
  ];

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
}

// App lifecycle
app.whenReady().then(() => {
  createWindow();
  createMenu();

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
