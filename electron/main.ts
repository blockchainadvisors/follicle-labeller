import {
  app,
  BrowserWindow,
  ipcMain,
  dialog,
  Menu,
  MenuItemConstructorOptions,
  powerMonitor,
} from "electron";
import path from "path";
import fs from "fs";
import { spawn, ChildProcess } from "child_process";
import JSZip from "jszip";
import { initUpdater, checkForUpdates } from "./updater";
import {
  setupPythonEnvironment,
  getVenvPythonPath,
  isVenvValid,
  getSetupStatus,
  getGPUHardwareInfo,
  installGPUPackages,
  checkYOLODependencies,
  installYOLODependencies,
  upgradePyTorchToCUDA,
} from "./python-env";

// Set Windows AppUserModelId for proper notifications (must be early)
if (process.platform === "win32") {
  app.setAppUserModelId("Follicle Labeller");
}

let mainWindow: BrowserWindow | null = null;

// BLOB detection server process
let blobServerProcess: ChildProcess | null = null;
const BLOB_SERVER_PORT = 5555;

// Python environment setup status for UI feedback
let pythonSetupStatus = "";

// File to open when launched via file association
let fileToOpen: string | null = null;

// Track if we're force closing (bypass unsaved changes check)
let forceClose = false;

function createWindow(): void {
  // Determine icon path based on platform
  const iconName = process.platform === "win32" ? "icon.ico" : "icon.png";
  const iconPath = app.isPackaged
    ? path.join(process.resourcesPath, iconName)
    : path.join(__dirname, "../public", iconName);

  // Check if icon exists (may not exist in dev if not generated)
  const iconExists = fs.existsSync(iconPath);

  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 700,
    ...(iconExists && { icon: iconPath }),
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
    title: "Follicle Labeller",
  });

  // Load content based on environment
  if (
    process.env.NODE_ENV === "development" ||
    process.env.VITE_DEV_SERVER_URL
  ) {
    mainWindow.loadURL(
      process.env.VITE_DEV_SERVER_URL || "http://localhost:5173",
    );
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, "../dist/index.html"));
  }

  // Handle window close - check for unsaved changes
  mainWindow.on("close", (event) => {
    if (forceClose) {
      forceClose = false;
      return; // Allow close
    }

    // Prevent close and ask renderer about unsaved changes
    event.preventDefault();
    mainWindow?.webContents.send("app:checkUnsavedChanges");
  });

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

// IPC handler for renderer to confirm close after checking unsaved changes
ipcMain.on("app:confirmClose", (_, canClose: boolean) => {
  if (canClose && mainWindow) {
    forceClose = true;
    mainWindow.close();
  }
});

// IPC Handlers

// Open image file dialog
ipcMain.handle("dialog:openImage", async () => {
  const window = BrowserWindow.getFocusedWindow();
  if (!window) return null;

  const result = await dialog.showOpenDialog(window, {
    properties: ["openFile"],
    filters: [
      {
        name: "Images",
        extensions: ["png", "jpg", "jpeg", "tiff", "tif", "bmp", "webp"],
      },
    ],
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

// Open generic file dialog with custom filters
ipcMain.handle("dialog:openFile", async (_, options: {
  filters?: Array<{ name: string; extensions: string[] }>;
  title?: string;
}) => {
  const window = BrowserWindow.getFocusedWindow();
  if (!window) return null;

  const result = await dialog.showOpenDialog(window, {
    properties: ["openFile"],
    title: options.title || "Open File",
    filters: options.filters || [{ name: "All Files", extensions: ["*"] }],
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
ipcMain.handle(
  "dialog:saveProject",
  async (
    _,
    imageData: ArrayBuffer,
    imageFileName: string,
    jsonData: string,
  ) => {
    const window = BrowserWindow.getFocusedWindow();
    if (!window) return false;

    const defaultName = imageFileName
      ? imageFileName.replace(/\.[^.]+$/, ".fol")
      : "project.fol";

    const result = await dialog.showSaveDialog(window, {
      defaultPath: defaultName,
      filters: [{ name: "Follicle Project", extensions: ["fol"] }],
    });

    if (result.canceled || !result.filePath) return false;

    try {
      const zip = new JSZip();

      // Add image to archive
      zip.file(imageFileName, Buffer.from(imageData));

      // Add JSON data to archive
      zip.file("annotations.json", jsonData);

      // Generate and save the archive
      const content = await zip.generateAsync({
        type: "nodebuffer",
        compression: "DEFLATE",
      });
      fs.writeFileSync(result.filePath, content);

      return true;
    } catch (error) {
      console.error("Failed to save project:", error);
      return false;
    }
  },
);

// Load project from .fol archive (legacy V1)
ipcMain.handle("dialog:loadProject", async () => {
  const window = BrowserWindow.getFocusedWindow();
  if (!window) return null;

  const result = await dialog.showOpenDialog(window, {
    properties: ["openFile"],
    filters: [{ name: "Follicle Project", extensions: ["fol"] }],
  });

  if (result.canceled || result.filePaths.length === 0) return null;

  try {
    const filePath = result.filePaths[0];
    const data = fs.readFileSync(filePath);

    const zip = await JSZip.loadAsync(data);

    // Find the image file (any file that's not annotations.json)
    let imageFileName = "";
    let imageData: ArrayBuffer | null = null;
    let jsonData = "";

    for (const fileName of Object.keys(zip.files)) {
      if (fileName === "annotations.json") {
        jsonData = await zip.files[fileName].async("string");
      } else {
        imageFileName = fileName;
        const imageBuffer = await zip.files[fileName].async("nodebuffer");
        imageData = imageBuffer.buffer.slice(
          imageBuffer.byteOffset,
          imageBuffer.byteOffset + imageBuffer.byteLength,
        );
      }
    }

    if (!imageData || !jsonData) {
      throw new Error("Invalid .fol file: missing image or annotations");
    }

    return {
      imageFileName,
      imageData,
      jsonData,
    };
  } catch (error) {
    console.error("Failed to load project:", error);
    return null;
  }
});

// Helper function to save project to a specific path
async function saveProjectToPath(
  filePath: string,
  images: Array<{ id: string; fileName: string; data: ArrayBuffer }>,
  manifestJson: string,
  annotationsJson: string,
): Promise<boolean> {
  try {
    const zip = new JSZip();

    // Add manifest
    zip.file("manifest.json", manifestJson);

    // Add images in images/ folder
    const imagesFolder = zip.folder("images");
    if (imagesFolder) {
      for (const image of images) {
        const archiveFileName = `${image.id}-${image.fileName}`;
        imagesFolder.file(archiveFileName, Buffer.from(image.data));
      }
    }

    // Add annotations
    zip.file("annotations.json", annotationsJson);

    // Generate and save the archive
    const content = await zip.generateAsync({
      type: "nodebuffer",
      compression: "DEFLATE",
    });
    fs.writeFileSync(filePath, content);

    return true;
  } catch (error) {
    console.error("Failed to save project:", error);
    return false;
  }
}

// V2 Save project with multiple images (Save As - shows dialog)
ipcMain.handle(
  "dialog:saveProjectV2",
  async (
    _,
    images: Array<{ id: string; fileName: string; data: ArrayBuffer }>,
    manifestJson: string,
    annotationsJson: string,
    defaultPath?: string,
  ) => {
    const window = BrowserWindow.getFocusedWindow();
    if (!window) return { success: false };

    const result = await dialog.showSaveDialog(window, {
      defaultPath: defaultPath || "project.fol",
      filters: [{ name: "Follicle Project", extensions: ["fol"] }],
    });

    if (result.canceled || !result.filePath) return { success: false };

    const success = await saveProjectToPath(
      result.filePath,
      images,
      manifestJson,
      annotationsJson,
    );
    return { success, filePath: success ? result.filePath : undefined };
  },
);

// V2 Save project to specific path (silent save - no dialog)
ipcMain.handle(
  "file:saveProjectV2",
  async (
    _,
    filePath: string,
    images: Array<{ id: string; fileName: string; data: ArrayBuffer }>,
    manifestJson: string,
    annotationsJson: string,
  ) => {
    const success = await saveProjectToPath(
      filePath,
      images,
      manifestJson,
      annotationsJson,
    );
    return { success, filePath: success ? filePath : undefined };
  },
);

// V2 Load project with support for both V1 and V2 formats
ipcMain.handle("dialog:loadProjectV2", async () => {
  const window = BrowserWindow.getFocusedWindow();
  if (!window) return null;

  const result = await dialog.showOpenDialog(window, {
    properties: ["openFile"],
    filters: [{ name: "Follicle Project", extensions: ["fol"] }],
  });

  if (result.canceled || result.filePaths.length === 0) return null;

  try {
    const filePath = result.filePaths[0];
    const data = fs.readFileSync(filePath);

    const zip = await JSZip.loadAsync(data);

    // Check if this is V2 (has manifest.json)
    const hasManifest = "manifest.json" in zip.files;

    if (hasManifest) {
      // V2 format
      const manifest = await zip.files["manifest.json"].async("string");
      const annotations = await zip.files["annotations.json"].async("string");

      // Load all images from images/ folder
      const images: Array<{ id: string; fileName: string; data: ArrayBuffer }> =
        [];
      const imagesFolder = zip.folder("images");

      if (imagesFolder) {
        for (const [relativePath, file] of Object.entries(imagesFolder.files)) {
          if (file.dir) continue;

          // Extract filename from path (remove 'images/' prefix)
          const archiveFileName = relativePath.replace("images/", "");
          if (!archiveFileName) continue;

          // Parse id and fileName from archiveFileName (format: {id}-{fileName})
          const dashIndex = archiveFileName.indexOf("-");
          if (dashIndex === -1) continue;

          const id = archiveFileName.substring(0, dashIndex);
          const fileName = archiveFileName.substring(dashIndex + 1);

          const imageBuffer = await file.async("nodebuffer");
          const imageData = imageBuffer.buffer.slice(
            imageBuffer.byteOffset,
            imageBuffer.byteOffset + imageBuffer.byteLength,
          );

          images.push({ id, fileName, data: imageData });
        }
      }

      return {
        version: "2.0" as const,
        filePath,
        manifest,
        images,
        annotations,
      };
    } else {
      // V1 format - return in V1 structure for migration
      let imageFileName = "";
      let imageData: ArrayBuffer | null = null;
      let jsonData = "";

      for (const fileName of Object.keys(zip.files)) {
        if (fileName === "annotations.json") {
          jsonData = await zip.files[fileName].async("string");
        } else if (!zip.files[fileName].dir) {
          imageFileName = fileName;
          const imageBuffer = await zip.files[fileName].async("nodebuffer");
          imageData = imageBuffer.buffer.slice(
            imageBuffer.byteOffset,
            imageBuffer.byteOffset + imageBuffer.byteLength,
          );
        }
      }

      if (!imageData || !jsonData) {
        throw new Error("Invalid .fol file: missing image or annotations");
      }

      return {
        version: "1.0" as const,
        filePath,
        imageFileName,
        imageData,
        jsonData,
      };
    }
  } catch (error) {
    console.error("Failed to load project:", error);
    return null;
  }
});

// Load project from specific file path (for file association)
ipcMain.handle("file:loadProject", async (_, filePath: string) => {
  try {
    if (!fs.existsSync(filePath)) {
      return null;
    }

    const data = fs.readFileSync(filePath);
    const zip = await JSZip.loadAsync(data);

    // Check if this is V2 (has manifest.json)
    const hasManifest = "manifest.json" in zip.files;

    if (hasManifest) {
      // V2 format
      const manifest = await zip.files["manifest.json"].async("string");
      const annotations = await zip.files["annotations.json"].async("string");

      // Load all images from images/ folder
      const images: Array<{ id: string; fileName: string; data: ArrayBuffer }> =
        [];
      const imagesFolder = zip.folder("images");

      if (imagesFolder) {
        for (const [relativePath, file] of Object.entries(imagesFolder.files)) {
          if (file.dir) continue;

          const archiveFileName = relativePath.replace("images/", "");
          if (!archiveFileName) continue;

          const dashIndex = archiveFileName.indexOf("-");
          if (dashIndex === -1) continue;

          const id = archiveFileName.substring(0, dashIndex);
          const fileName = archiveFileName.substring(dashIndex + 1);

          const imageBuffer = await file.async("nodebuffer");
          const imageData = imageBuffer.buffer.slice(
            imageBuffer.byteOffset,
            imageBuffer.byteOffset + imageBuffer.byteLength,
          );

          images.push({ id, fileName, data: imageData });
        }
      }

      return {
        version: "2.0" as const,
        filePath,
        manifest,
        images,
        annotations,
      };
    } else {
      // V1 format
      let imageFileName = "";
      let imageData: ArrayBuffer | null = null;
      let jsonData = "";

      for (const fileName of Object.keys(zip.files)) {
        if (fileName === "annotations.json") {
          jsonData = await zip.files[fileName].async("string");
        } else if (!zip.files[fileName].dir) {
          imageFileName = fileName;
          const imageBuffer = await zip.files[fileName].async("nodebuffer");
          imageData = imageBuffer.buffer.slice(
            imageBuffer.byteOffset,
            imageBuffer.byteOffset + imageBuffer.byteLength,
          );
        }
      }

      if (!imageData || !jsonData) {
        throw new Error("Invalid .fol file: missing image or annotations");
      }

      return {
        version: "1.0" as const,
        filePath,
        imageFileName,
        imageData,
        jsonData,
      };
    }
  } catch (error) {
    console.error("Failed to load project from path:", error);
    return null;
  }
});

// Menu item references for dynamic enable/disable
let saveMenuItem: Electron.MenuItem | null = null;
let saveAsMenuItem: Electron.MenuItem | null = null;
let closeProjectMenuItem: Electron.MenuItem | null = null;

// Update menu items based on project state
ipcMain.on("menu:setProjectState", (_, hasProject: boolean) => {
  if (saveMenuItem) saveMenuItem.enabled = hasProject;
  if (saveAsMenuItem) saveAsMenuItem.enabled = hasProject;
  if (closeProjectMenuItem) closeProjectMenuItem.enabled = hasProject;
});

// Create application menu
function createMenu(): void {
  const isMac = process.platform === "darwin";

  const template: MenuItemConstructorOptions[] = [
    // macOS app menu
    ...(isMac
      ? [
          {
            label: app.name,
            submenu: [
              { role: "about" as const },
              { type: "separator" as const },
              { role: "quit" as const },
            ],
          },
        ]
      : []),

    // File menu
    {
      label: "File",
      submenu: [
        {
          label: "Open Image...",
          accelerator: "CmdOrCtrl+O",
          click: () => mainWindow?.webContents.send("menu:openImage"),
        },
        { type: "separator" },
        {
          label: "Load Project...",
          accelerator: "CmdOrCtrl+Shift+O",
          click: () => mainWindow?.webContents.send("menu:loadProject"),
        },
        {
          id: "save-project",
          label: "Save Project",
          accelerator: "CmdOrCtrl+S",
          enabled: false,
          click: () => mainWindow?.webContents.send("menu:saveProject"),
        },
        {
          id: "save-project-as",
          label: "Save Project As...",
          accelerator: "CmdOrCtrl+Shift+S",
          enabled: false,
          click: () => mainWindow?.webContents.send("menu:saveProjectAs"),
        },
        { type: "separator" },
        {
          id: "close-project",
          label: "Close Project",
          accelerator: "CmdOrCtrl+W",
          enabled: false,
          click: () => mainWindow?.webContents.send("menu:closeProject"),
        },
        { type: "separator" },
        isMac ? { role: "close" as const } : { role: "quit" as const },
      ],
    },

    // Edit menu
    {
      label: "Edit",
      submenu: [
        {
          label: "Undo",
          accelerator: "CmdOrCtrl+Z",
          click: () => mainWindow?.webContents.send("menu:undo"),
        },
        {
          label: "Redo",
          accelerator: "CmdOrCtrl+Shift+Z",
          click: () => mainWindow?.webContents.send("menu:redo"),
        },
        { type: "separator" },
        {
          label: "Clear All Annotations",
          click: () => mainWindow?.webContents.send("menu:clearAll"),
        },
      ],
    },

    // View menu
    {
      label: "View",
      submenu: [
        {
          label: "Toggle Shapes (O)",
          click: () => mainWindow?.webContents.send("menu:toggleShapes"),
        },
        {
          label: "Toggle Labels (L)",
          click: () => mainWindow?.webContents.send("menu:toggleLabels"),
        },
        { type: "separator" },
        {
          label: "Zoom In",
          accelerator: "CmdOrCtrl+=",
          click: () => mainWindow?.webContents.send("menu:zoomIn"),
        },
        {
          label: "Zoom Out",
          accelerator: "CmdOrCtrl+-",
          click: () => mainWindow?.webContents.send("menu:zoomOut"),
        },
        {
          label: "Reset Zoom",
          accelerator: "CmdOrCtrl+0",
          click: () => mainWindow?.webContents.send("menu:resetZoom"),
        },
        { type: "separator" },
        { role: "toggleDevTools" },
      ],
    },

    // Help menu
    {
      label: "Help",
      submenu: [
        {
          label: "User Guide",
          accelerator: "F1",
          click: () => mainWindow?.webContents.send("menu:showHelp"),
        },
        { type: "separator" },
        {
          label: "Check for Updates...",
          click: () => checkForUpdates(),
        },
        { type: "separator" },
        {
          label: "About Follicle Labeller",
          click: () => {
            if (mainWindow) {
              dialog.showMessageBox(mainWindow, {
                type: "info",
                title: "About Follicle Labeller",
                message: `Follicle Labeller v${app.getVersion()}`,
                detail: "Medical image annotation tool for follicle labeling.",
              });
            }
          },
        },
      ],
    },
  ];

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);

  // Store references to menu items for dynamic enable/disable
  saveMenuItem = menu.getMenuItemById("save-project");
  saveAsMenuItem = menu.getMenuItemById("save-project-as");
  closeProjectMenuItem = menu.getMenuItemById("close-project");
}

// Show download options dialog when there's an active selection
// Returns: 'all' | 'currentImage' | 'selected' | 'cancel'
ipcMain.handle(
  "dialog:downloadOptions",
  async (
    _,
    selectedCount: number,
    currentImageCount: number,
    totalCount: number,
  ) => {
    const window = BrowserWindow.getFocusedWindow();
    if (!window) return "cancel";

    const result = await dialog.showMessageBox(window, {
      type: "question",
      title: "Download Follicle Images",
      message: "What would you like to download?",
      buttons: [
        `Selected Only (${selectedCount})`,
        `Current Image (${currentImageCount})`,
        `All Follicles (${totalCount})`,
        "Cancel",
      ],
      defaultId: 0,
      cancelId: 3,
    });

    switch (result.response) {
      case 0:
        return "selected";
      case 1:
        return "currentImage";
      case 2:
        return "all";
      default:
        return "cancel";
    }
  },
);

// Show unsaved changes dialog
// Returns: 'save' | 'discard' | 'cancel'
ipcMain.handle("dialog:unsavedChanges", async () => {
  const window = BrowserWindow.getFocusedWindow();
  if (!window) return "discard";

  const result = await dialog.showMessageBox(window, {
    type: "warning",
    title: "Unsaved Changes",
    message: "You have unsaved changes. Do you want to save before closing?",
    buttons: ["Save", "Don't Save", "Cancel"],
    defaultId: 0,
    cancelId: 2,
  });

  switch (result.response) {
    case 0:
      return "save";
    case 1:
      return "discard";
    default:
      return "cancel";
  }
});

// ============================================
// BLOB Detection Server Management
// ============================================

/**
 * Get the path to the BLOB server Python script.
 */
function getBlobServerPath(): string {
  if (app.isPackaged) {
    return path.join(process.resourcesPath, "python", "blob_server.py");
  }
  // In dev mode, __dirname is dist-electron/, so go up one level to find electron/python/
  return path.join(__dirname, "..", "electron", "python", "blob_server.py");
}

/**
 * Check if Python is available.
 */
async function checkPythonAvailable(): Promise<{
  available: boolean;
  version?: string;
  error?: string;
}> {
  return new Promise((resolve) => {
    // Use 'python' on all platforms - conda environments use 'python' not 'python3'
    const pythonCmd = "python";
    const proc = spawn(pythonCmd, ["--version"]);

    let stdout = "";
    let stderr = "";

    proc.stdout?.on("data", (data) => {
      stdout += data.toString();
    });
    proc.stderr?.on("data", (data) => {
      stderr += data.toString();
    });

    proc.on("close", (code) => {
      if (code === 0) {
        const version = stdout.trim() || stderr.trim();
        resolve({ available: true, version });
      } else {
        resolve({ available: false, error: "Python not found" });
      }
    });

    proc.on("error", () => {
      resolve({ available: false, error: "Python not found" });
    });

    // Timeout after 5 seconds
    setTimeout(() => {
      proc.kill();
      resolve({ available: false, error: "Python check timed out" });
    }, 5000);
  });
}

/**
 * Start the BLOB detection server process.
 * Automatically sets up Python venv and installs dependencies if needed.
 */
async function startBlobServer(): Promise<{
  success: boolean;
  error?: string;
}> {
  // Check if already running
  if (blobServerProcess && !blobServerProcess.killed) {
    return { success: true };
  }

  console.log("[BLOB Server] Initializing Python environment...");
  pythonSetupStatus = "Initializing Python environment...";

  // Get requirements.txt path
  const requirementsPath = app.isPackaged
    ? path.join(process.resourcesPath, "python", "requirements.txt")
    : path.join(__dirname, "..", "electron", "python", "requirements.txt");

  // Setup Python environment (creates venv and installs deps if needed)
  const setupResult = await setupPythonEnvironment(
    requirementsPath,
    (status) => {
      pythonSetupStatus = status;
      // Notify renderer of progress
      mainWindow?.webContents.send("blob:setupProgress", status);
    }
  );

  if (!setupResult.success) {
    pythonSetupStatus = `Setup failed: ${setupResult.error}`;
    return { success: false, error: setupResult.error };
  }

  // Check if server script exists
  const serverPath = getBlobServerPath();
  if (!fs.existsSync(serverPath)) {
    return { success: false, error: "BLOB server script not found" };
  }

  // Get Python path from venv
  const pythonPath = getVenvPythonPath();

  return new Promise((resolve) => {
    console.log("[BLOB Server] Starting server...");
    console.log("[BLOB Server] Python path:", pythonPath);
    console.log("[BLOB Server] Script path:", serverPath);
    console.log("[BLOB Server] CWD:", path.dirname(serverPath));

    pythonSetupStatus = "Starting detection server...";
    mainWindow?.webContents.send("blob:setupProgress", pythonSetupStatus);

    // Use venv Python directly - no shell needed for more reliable execution
    blobServerProcess = spawn(
      pythonPath,
      [serverPath, "--port", BLOB_SERVER_PORT.toString()],
      {
        cwd: path.dirname(serverPath),
        stdio: ["ignore", "pipe", "pipe"],
        // No shell: true - direct execution is more reliable
      },
    );

    let resolved = false;
    let errorOutput = "";

    blobServerProcess.stdout?.on("data", (data) => {
      const output = data.toString();
      console.log("[BLOB Server]", output);

      // Check for successful startup (case-insensitive to support both Flask and uvicorn)
      // Flask: "* Running on http://..."
      // Uvicorn: "Uvicorn running on http://..."
      if (!resolved && output.toLowerCase().includes("running on")) {
        resolved = true;
        pythonSetupStatus = "Detection server ready";
        mainWindow?.webContents.send("blob:setupProgress", pythonSetupStatus);
        resolve({ success: true });
      }
    });

    blobServerProcess.stderr?.on("data", (data) => {
      const output = data.toString();
      // Flask/werkzeug and uvicorn output startup info to stderr
      console.log("[BLOB Server stderr]", output);

      // Check for successful startup in stderr (case-insensitive)
      if (!resolved && output.toLowerCase().includes("running on")) {
        resolved = true;
        pythonSetupStatus = "Detection server ready";
        mainWindow?.webContents.send("blob:setupProgress", pythonSetupStatus);
        resolve({ success: true });
      }

      // Accumulate all stderr for debugging
      errorOutput += output;
    });

    blobServerProcess.on("close", (code) => {
      console.log(`BLOB server exited with code ${code}`);
      blobServerProcess = null;
      if (!resolved) {
        resolved = true;
        pythonSetupStatus = `Server exited with code ${code}`;
        resolve({
          success: false,
          error: errorOutput || `Process exited with code ${code}`,
        });
      }
    });

    blobServerProcess.on("error", (err) => {
      console.error("Failed to start BLOB server:", err);
      if (!resolved) {
        resolved = true;
        pythonSetupStatus = `Failed to start: ${err.message}`;
        resolve({ success: false, error: err.message });
      }
    });

    // Timeout after 60 seconds (increased for first-time dependency installation)
    setTimeout(() => {
      if (!resolved) {
        resolved = true;
        pythonSetupStatus = "Server startup timed out";
        resolve({ success: false, error: "Server startup timed out" });
      }
    }, 60000);
  });
}

/**
 * Stop the BLOB detection server process.
 */
async function stopBlobServer(): Promise<void> {
  if (blobServerProcess && !blobServerProcess.killed) {
    console.log("Stopping BLOB server...");

    return new Promise((resolve) => {
      // Try graceful shutdown first via HTTP
      const http = require("http");
      const req = http.request(
        {
          hostname: "127.0.0.1",
          port: BLOB_SERVER_PORT,
          path: "/shutdown",
          method: "POST",
          timeout: 2000,
        },
        () => {
          // Response received, server is shutting down
        },
      );

      req.on("error", () => {
        // If HTTP fails, kill the process
        if (blobServerProcess && !blobServerProcess.killed) {
          blobServerProcess.kill();
        }
      });

      req.end();

      // Force kill after timeout if still running
      setTimeout(() => {
        if (blobServerProcess && !blobServerProcess.killed) {
          blobServerProcess.kill("SIGKILL");
        }
        blobServerProcess = null;
        resolve();
      }, 3000);
    });
  }
}

/**
 * Check if BLOB server is running.
 */
async function isBlobServerRunning(): Promise<boolean> {
  return new Promise((resolve) => {
    const http = require("http");
    const req = http.request(
      {
        hostname: "127.0.0.1",
        port: BLOB_SERVER_PORT,
        path: "/health",
        method: "GET",
        timeout: 2000,
      },
      (res: any) => {
        resolve(res.statusCode === 200);
      },
    );

    req.on("error", () => {
      resolve(false);
    });

    req.on("timeout", () => {
      req.destroy();
      resolve(false);
    });

    req.end();
  });
}

// BLOB Detection IPC Handlers
ipcMain.handle("blob:startServer", async () => {
  return startBlobServer();
});

ipcMain.handle("blob:stopServer", async () => {
  stopBlobServer();
  return { success: true };
});

ipcMain.handle("blob:isAvailable", async () => {
  return isBlobServerRunning();
});

ipcMain.handle("blob:checkPython", async () => {
  return checkPythonAvailable();
});

ipcMain.handle("blob:getServerInfo", async () => {
  return {
    port: BLOB_SERVER_PORT,
    running: await isBlobServerRunning(),
    scriptPath: getBlobServerPath(),
  };
});

ipcMain.handle("blob:getSetupStatus", () => {
  return pythonSetupStatus;
});

ipcMain.handle("blob:getGPUInfo", async () => {
  // Make HTTP request to the blob server's /gpu-info endpoint
  return new Promise((resolve) => {
    const http = require("http");
    const req = http.request(
      {
        hostname: "127.0.0.1",
        port: BLOB_SERVER_PORT,
        path: "/gpu-info",
        method: "GET",
        timeout: 5000,
      },
      (res: any) => {
        let data = "";
        res.on("data", (chunk: string) => {
          data += chunk;
        });
        res.on("end", () => {
          try {
            const gpuInfo = JSON.parse(data);
            resolve({
              activeBackend: gpuInfo.active_backend || "cpu",
              deviceName: gpuInfo.device_name || "CPU (OpenCV)",
              memoryGB: gpuInfo.details?.cuda?.memory_gb,
              available: {
                cuda: gpuInfo.backends?.cuda || false,
                mps: gpuInfo.backends?.mps || false,
              },
            });
          } catch {
            resolve({
              activeBackend: "cpu",
              deviceName: "CPU (OpenCV)",
              available: { cuda: false, mps: false },
            });
          }
        });
      }
    );

    req.on("error", () => {
      resolve({
        activeBackend: "cpu",
        deviceName: "CPU (OpenCV)",
        available: { cuda: false, mps: false },
      });
    });

    req.on("timeout", () => {
      req.destroy();
      resolve({
        activeBackend: "cpu",
        deviceName: "CPU (OpenCV)",
        available: { cuda: false, mps: false },
      });
    });

    req.end();
  });
});

// ============================================
// GPU Hardware Detection & Installation
// ============================================

// Get GPU hardware info (works before packages installed)
ipcMain.handle("gpu:getHardwareInfo", async () => {
  return getGPUHardwareInfo();
});

// Install GPU packages
ipcMain.handle("gpu:installPackages", async () => {
  return installGPUPackages((message, percent) => {
    mainWindow?.webContents.send("gpu:installProgress", { message, percent });
  });
});

// Restart blob server after installation
ipcMain.handle("blob:restartServer", async () => {
  await stopBlobServer();
  return startBlobServer();
});

// Check YOLO training dependencies
ipcMain.handle("yolo:checkDependencies", async () => {
  return checkYOLODependencies();
});

// Install YOLO training dependencies
ipcMain.handle("yolo:installDependencies", async () => {
  return installYOLODependencies((message, percent) => {
    mainWindow?.webContents.send("yolo:installProgress", { message, percent });
  });
});

// Upgrade PyTorch to CUDA version for GPU training
ipcMain.handle("yolo:upgradeToCUDA", async () => {
  return upgradePyTorchToCUDA((message, percent) => {
    mainWindow?.webContents.send("yolo:installProgress", { message, percent });
  });
});

// ============================================
// YOLO Keypoint Training API
// ============================================

// Helper for making HTTP requests to the blob server
function makeBlobServerRequest(
  path: string,
  method: string,
  body?: any,
  timeoutMs: number = 30000
): Promise<any> {
  return new Promise((resolve, reject) => {
    const http = require("http");
    const postData = body ? JSON.stringify(body) : "";

    const options = {
      hostname: "127.0.0.1",
      port: BLOB_SERVER_PORT,
      path,
      method,
      timeout: timeoutMs,
      headers: {
        "Content-Type": "application/json",
        "Content-Length": Buffer.byteLength(postData),
      },
    };

    const req = http.request(options, (res: any) => {
      let data = "";
      res.on("data", (chunk: string) => {
        data += chunk;
      });
      res.on("end", () => {
        try {
          const parsed = JSON.parse(data);
          if (res.statusCode >= 400) {
            reject(new Error(parsed.detail || `HTTP ${res.statusCode}`));
          } else {
            resolve(parsed);
          }
        } catch {
          if (res.statusCode >= 400) {
            reject(new Error(`HTTP ${res.statusCode}: ${data}`));
          } else {
            resolve(data);
          }
        }
      });
    });

    req.on("error", reject);
    req.on("timeout", () => {
      req.destroy();
      reject(new Error("Request timed out"));
    });

    if (postData) {
      req.write(postData);
    }
    req.end();
  });
}

// Get YOLO keypoint service status
ipcMain.handle("yolo-keypoint:getStatus", async () => {
  try {
    return await makeBlobServerRequest("/yolo-keypoint/status", "GET");
  } catch (error) {
    return {
      available: false,
      sseAvailable: false,
      activeTrainingJobs: 0,
    };
  }
});

// Get system info for training (CPU/GPU, memory)
ipcMain.handle("yolo-keypoint:getSystemInfo", async () => {
  try {
    return await makeBlobServerRequest("/yolo-keypoint/system-info", "GET");
  } catch (error) {
    return {
      device: "unknown",
      device_name: "Unknown",
      error: error instanceof Error ? error.message : "Failed to get system info",
    };
  }
});

// Validate dataset
ipcMain.handle("yolo-keypoint:validateDataset", async (_, datasetPath: string) => {
  return makeBlobServerRequest("/yolo-keypoint/validate-dataset", "POST", {
    datasetPath,
  });
});

// Start training
ipcMain.handle(
  "yolo-keypoint:startTraining",
  async (_, datasetPath: string, config: any, modelName?: string) => {
    return makeBlobServerRequest("/yolo-keypoint/train/start", "POST", {
      datasetPath,
      config,
      modelName,
    });
  }
);

// Stop training
ipcMain.handle("yolo-keypoint:stopTraining", async (_, jobId: string) => {
  return makeBlobServerRequest(`/yolo-keypoint/train/stop/${jobId}`, "POST");
});

// Active SSE connections for training progress
const activeSseConnections: Map<string, { request: any; closed: boolean }> =
  new Map();

// Subscribe to training progress via SSE (proxied through main process)
ipcMain.handle(
  "yolo-keypoint:subscribeProgress",
  async (event, jobId: string) => {
    const http = require("http");

    // Close any existing connection for this job
    const existing = activeSseConnections.get(jobId);
    if (existing && !existing.closed) {
      existing.request.destroy();
      existing.closed = true;
    }

    return new Promise<void>((resolve) => {
      const req = http.request(
        {
          hostname: "127.0.0.1",
          port: BLOB_SERVER_PORT,
          path: `/yolo-keypoint/train/progress/${jobId}`,
          method: "GET",
          headers: {
            Accept: "text/event-stream",
            "Cache-Control": "no-cache",
            Connection: "keep-alive",
          },
        },
        (res: any) => {
          if (res.statusCode !== 200) {
            console.error(`SSE connection failed with status ${res.statusCode}`);
            event.sender.send("yolo-keypoint:progress-error", jobId, `HTTP ${res.statusCode}`);
            resolve();
            return;
          }

          res.setEncoding("utf8");
          let buffer = "";

          res.on("data", (chunk: string) => {
            buffer += chunk;

            // Normalize line endings (SSE can use \r\n or \n)
            buffer = buffer.replace(/\r\n/g, "\n");

            // Parse SSE events (format: "event: name\ndata: json\n\n")
            const events = buffer.split("\n\n");
            buffer = events.pop() || ""; // Keep incomplete event in buffer

            for (const eventBlock of events) {
              if (!eventBlock.trim()) continue;

              const lines = eventBlock.split("\n");
              let eventType = "message";
              let eventData = "";

              for (const line of lines) {
                if (line.startsWith("event:")) {
                  eventType = line.slice(6).trim();
                } else if (line.startsWith("data:")) {
                  eventData = line.slice(5).trim();
                }
              }

              if (eventType === "progress" && eventData) {
                try {
                  const progress = JSON.parse(eventData);
                  event.sender.send("yolo-keypoint:progress", jobId, progress);

                  // Check if training is done
                  if (
                    ["completed", "failed", "stopped"].includes(progress.status)
                  ) {
                    event.sender.send("yolo-keypoint:progress-complete", jobId);
                    req.destroy();
                    activeSseConnections.delete(jobId);
                  }
                } catch (e) {
                  console.error("Failed to parse SSE progress:", e);
                }
              }
            }
          });

          res.on("end", () => {
            activeSseConnections.delete(jobId);
            event.sender.send("yolo-keypoint:progress-complete", jobId);
          });

          res.on("error", (err: Error) => {
            console.error("SSE response error:", err);
            event.sender.send("yolo-keypoint:progress-error", jobId, err.message);
            activeSseConnections.delete(jobId);
          });

          resolve();
        }
      );

      req.on("error", (err: Error) => {
        console.error("SSE request error:", err);
        event.sender.send("yolo-keypoint:progress-error", jobId, err.message);
        activeSseConnections.delete(jobId);
        resolve();
      });

      activeSseConnections.set(jobId, { request: req, closed: false });
      req.end();
    });
  }
);

// Unsubscribe from training progress
ipcMain.handle("yolo-keypoint:unsubscribeProgress", async (_, jobId: string) => {
  const connection = activeSseConnections.get(jobId);
  if (connection && !connection.closed) {
    connection.request.destroy();
    connection.closed = true;
  }
  activeSseConnections.delete(jobId);
});

// List models
ipcMain.handle("yolo-keypoint:listModels", async () => {
  return makeBlobServerRequest("/yolo-keypoint/models", "GET");
});

// Load model
ipcMain.handle("yolo-keypoint:loadModel", async (_, modelPath: string) => {
  return makeBlobServerRequest("/yolo-keypoint/load-model", "POST", {
    modelPath,
  });
});

// Predict
ipcMain.handle("yolo-keypoint:predict", async (_, imageData: string) => {
  return makeBlobServerRequest("/yolo-keypoint/predict", "POST", {
    imageData,
  });
});

// Show save dialog for ONNX export
ipcMain.handle(
  "yolo-keypoint:showExportDialog",
  async (_, defaultFileName: string) => {
    const window = BrowserWindow.getFocusedWindow();
    if (!window) return { canceled: true };

    const result = await dialog.showSaveDialog(window, {
      defaultPath: defaultFileName,
      filters: [{ name: "ONNX Model", extensions: ["onnx"] }],
    });

    return {
      canceled: result.canceled,
      filePath: result.filePath,
    };
  }
);

// Export ONNX - use longer timeout (5 min) since it may install dependencies
ipcMain.handle(
  "yolo-keypoint:exportONNX",
  async (_, modelPath: string, outputPath: string) => {
    return makeBlobServerRequest(
      "/yolo-keypoint/export-onnx",
      "POST",
      {
        modelPath,
        outputPath,
      },
      300000 // 5 minute timeout for ONNX export
    );
  }
);

// Delete model
ipcMain.handle("yolo-keypoint:deleteModel", async (_, modelId: string) => {
  const http = require("http");
  return new Promise((resolve, reject) => {
    const req = http.request(
      {
        hostname: "127.0.0.1",
        port: BLOB_SERVER_PORT,
        path: `/yolo-keypoint/models/${modelId}`,
        method: "DELETE",
        timeout: 10000,
      },
      (res: any) => {
        let data = "";
        res.on("data", (chunk: string) => {
          data += chunk;
        });
        res.on("end", () => {
          try {
            resolve(JSON.parse(data));
          } catch {
            resolve({ success: res.statusCode < 400 });
          }
        });
      }
    );
    req.on("error", reject);
    req.on("timeout", () => {
      req.destroy();
      reject(new Error("Request timed out"));
    });
    req.end();
  });
});

// Write dataset files to temp directory (for training from current project)
ipcMain.handle(
  "yolo-keypoint:writeDatasetToTemp",
  async (
    _,
    files: Array<{ path: string; content: ArrayBuffer | string }>
  ) => {
    try {
      const os = require("os");
      const crypto = require("crypto");
      const fsPromises = require("fs").promises;

      // Create unique temp directory
      const tempId = crypto.randomBytes(8).toString("hex");
      const datasetPath = path.join(
        os.tmpdir(),
        "follicle-labeller-datasets",
        `dataset_${tempId}`
      );

      // Create base directory
      await fsPromises.mkdir(datasetPath, { recursive: true });

      // Write all files
      for (const file of files) {
        const filePath = path.join(datasetPath, file.path);
        const fileDir = path.dirname(filePath);

        // Ensure directory exists
        await fsPromises.mkdir(fileDir, { recursive: true });

        // Write file content - fix data.yaml to use absolute path
        if (typeof file.content === "string") {
          let content = file.content;
          // Fix data.yaml to use absolute path instead of relative "."
          if (file.path === "data.yaml") {
            // Replace "path: ." with absolute path (use forward slashes for YOLO compatibility)
            const absolutePath = datasetPath.replace(/\\/g, "/");
            content = content.replace(/^path:\s*\.?\s*$/m, `path: ${absolutePath}`);
          }
          await fsPromises.writeFile(filePath, content, "utf8");
        } else {
          // ArrayBuffer - convert to Buffer
          await fsPromises.writeFile(filePath, Buffer.from(file.content));
        }
      }

      console.log(`[YOLO Dataset] Written ${files.length} files to ${datasetPath}`);

      return { success: true, datasetPath };
    } catch (error) {
      console.error("[YOLO Dataset] Error writing dataset:", error);
      return {
        success: false,
        error: error instanceof Error ? error.message : "Failed to write dataset",
      };
    }
  }
);

// ============================================
// YOLO Detection API
// ============================================

// Active SSE connections for detection training progress
const activeDetectionSseConnections: Map<string, { request: any; closed: boolean }> =
  new Map();

// Get YOLO detection service status
ipcMain.handle("yolo-detection:getStatus", async () => {
  try {
    return await makeBlobServerRequest("/yolo-detect/status", "GET");
  } catch (error) {
    return {
      available: false,
      sseAvailable: false,
      activeTrainingJobs: 0,
      loadedModel: null,
    };
  }
});

// Validate detection dataset
ipcMain.handle("yolo-detection:validateDataset", async (_, datasetPath: string) => {
  return makeBlobServerRequest("/yolo-detect/validate-dataset", "POST", {
    datasetPath,
  });
});

// Start detection training
ipcMain.handle(
  "yolo-detection:startTraining",
  async (_, datasetPath: string, config: any, modelName?: string) => {
    return makeBlobServerRequest("/yolo-detect/train/start", "POST", {
      datasetPath,
      config,
      modelName,
    });
  }
);

// Stop detection training
ipcMain.handle("yolo-detection:stopTraining", async (_, jobId: string) => {
  return makeBlobServerRequest(`/yolo-detect/train/stop/${jobId}`, "POST");
});

// Subscribe to detection training progress via SSE
ipcMain.handle(
  "yolo-detection:subscribeProgress",
  async (event, jobId: string) => {
    const http = require("http");

    // Close any existing connection for this job
    const existing = activeDetectionSseConnections.get(jobId);
    if (existing && !existing.closed) {
      existing.request.destroy();
      existing.closed = true;
    }

    return new Promise<void>((resolve) => {
      const req = http.request(
        {
          hostname: "127.0.0.1",
          port: BLOB_SERVER_PORT,
          path: `/yolo-detect/train/progress/${jobId}`,
          method: "GET",
          headers: {
            Accept: "text/event-stream",
            "Cache-Control": "no-cache",
            Connection: "keep-alive",
          },
        },
        (res: any) => {
          if (res.statusCode !== 200) {
            console.error(`Detection SSE connection failed with status ${res.statusCode}`);
            event.sender.send("yolo-detection:progress-error", jobId, `HTTP ${res.statusCode}`);
            resolve();
            return;
          }

          res.setEncoding("utf8");
          let buffer = "";

          res.on("data", (chunk: string) => {
            buffer += chunk;
            buffer = buffer.replace(/\r\n/g, "\n");

            const events = buffer.split("\n\n");
            buffer = events.pop() || "";

            for (const eventBlock of events) {
              if (!eventBlock.trim()) continue;

              const lines = eventBlock.split("\n");
              let eventType = "message";
              let eventData = "";

              for (const line of lines) {
                if (line.startsWith("event:")) {
                  eventType = line.slice(6).trim();
                } else if (line.startsWith("data:")) {
                  eventData = line.slice(5).trim();
                }
              }

              if (eventType === "progress" && eventData) {
                try {
                  const progress = JSON.parse(eventData);
                  event.sender.send("yolo-detection:progress", jobId, progress);

                  if (
                    ["completed", "failed", "stopped"].includes(progress.status)
                  ) {
                    event.sender.send("yolo-detection:progress-complete", jobId);
                    req.destroy();
                    activeDetectionSseConnections.delete(jobId);
                  }
                } catch (e) {
                  console.error("Failed to parse detection SSE progress:", e);
                }
              }
            }
          });

          res.on("end", () => {
            activeDetectionSseConnections.delete(jobId);
            event.sender.send("yolo-detection:progress-complete", jobId);
          });

          res.on("error", (err: Error) => {
            console.error("Detection SSE response error:", err);
            event.sender.send("yolo-detection:progress-error", jobId, err.message);
            activeDetectionSseConnections.delete(jobId);
          });

          resolve();
        }
      );

      req.on("error", (err: Error) => {
        console.error("Detection SSE request error:", err);
        event.sender.send("yolo-detection:progress-error", jobId, err.message);
        activeDetectionSseConnections.delete(jobId);
        resolve();
      });

      activeDetectionSseConnections.set(jobId, { request: req, closed: false });
      req.end();
    });
  }
);

// Unsubscribe from detection training progress
ipcMain.handle("yolo-detection:unsubscribeProgress", async (_, jobId: string) => {
  const connection = activeDetectionSseConnections.get(jobId);
  if (connection && !connection.closed) {
    connection.request.destroy();
    connection.closed = true;
  }
  activeDetectionSseConnections.delete(jobId);
});

// List detection models
ipcMain.handle("yolo-detection:listModels", async () => {
  return makeBlobServerRequest("/yolo-detect/models", "GET");
});

// Get resumable models (incomplete training with last.pt)
ipcMain.handle("yolo-detection:getResumableModels", async () => {
  return makeBlobServerRequest("/yolo-detect/resumable-models", "GET");
});

// Load detection model
ipcMain.handle("yolo-detection:loadModel", async (_, modelPath: string) => {
  return makeBlobServerRequest("/yolo-detect/load-model", "POST", {
    modelPath,
  });
});

// Run detection prediction on full image
ipcMain.handle(
  "yolo-detection:predict",
  async (_, imageData: string, confidenceThreshold: number = 0.5) => {
    return makeBlobServerRequest("/yolo-detect/predict", "POST", {
      imageData,
      confidenceThreshold,
    });
  }
);

// Run tiled detection prediction (for large images with small objects)
ipcMain.handle(
  "yolo-detection:predictTiled",
  async (
    _,
    imageData: string,
    confidenceThreshold: number = 0.5,
    tileSize: number = 1024,
    overlap: number = 128,
    nmsThreshold: number = 0.5,
    scaleFactor: number = 1.0
  ) => {
    return makeBlobServerRequest("/yolo-detect/predict-tiled", "POST", {
      imageData,
      confidenceThreshold,
      tileSize,
      overlap,
      nmsThreshold,
      scaleFactor,
    });
  }
);

// Show save dialog for ONNX export (detection)
ipcMain.handle(
  "yolo-detection:showExportDialog",
  async (_, defaultFileName: string) => {
    const window = BrowserWindow.getFocusedWindow();
    if (!window) return { canceled: true };

    const result = await dialog.showSaveDialog(window, {
      defaultPath: defaultFileName,
      filters: [{ name: "ONNX Model", extensions: ["onnx"] }],
    });

    return {
      canceled: result.canceled,
      filePath: result.filePath,
    };
  }
);

// Export detection model to ONNX
ipcMain.handle(
  "yolo-detection:exportONNX",
  async (_, modelPath: string, outputPath: string) => {
    return makeBlobServerRequest(
      "/yolo-detect/export-onnx",
      "POST",
      {
        modelPath,
        outputPath,
      },
      300000 // 5 minute timeout for ONNX export
    );
  }
);

// Delete detection model
ipcMain.handle("yolo-detection:deleteModel", async (_, modelId: string) => {
  const http = require("http");
  return new Promise((resolve, reject) => {
    const req = http.request(
      {
        hostname: "127.0.0.1",
        port: BLOB_SERVER_PORT,
        path: `/yolo-detect/models/${modelId}`,
        method: "DELETE",
        timeout: 10000,
      },
      (res: any) => {
        let data = "";
        res.on("data", (chunk: string) => {
          data += chunk;
        });
        res.on("end", () => {
          try {
            resolve(JSON.parse(data));
          } catch {
            resolve({ success: res.statusCode < 400 });
          }
        });
      }
    );
    req.on("error", reject);
    req.on("timeout", () => {
      req.destroy();
      reject(new Error("Request timed out"));
    });
    req.end();
  });
});

// Write detection dataset files to temp directory
ipcMain.handle(
  "yolo-detection:writeDatasetToTemp",
  async (
    _,
    files: Array<{ path: string; content: ArrayBuffer | string }>
  ) => {
    try {
      const os = require("os");
      const crypto = require("crypto");
      const fsPromises = require("fs").promises;

      // Create unique temp directory
      const tempId = crypto.randomBytes(8).toString("hex");
      const datasetPath = path.join(
        os.tmpdir(),
        "follicle-labeller-datasets",
        `detection_dataset_${tempId}`
      );

      // Create base directory
      await fsPromises.mkdir(datasetPath, { recursive: true });

      // Write all files
      for (const file of files) {
        const filePath = path.join(datasetPath, file.path);
        const fileDir = path.dirname(filePath);

        // Ensure directory exists
        await fsPromises.mkdir(fileDir, { recursive: true });

        // Write file content - fix data.yaml to use absolute path
        if (typeof file.content === "string") {
          let content = file.content;
          if (file.path === "data.yaml") {
            const absolutePath = datasetPath.replace(/\\/g, "/");
            content = content.replace(/^path:\s*\.?\s*$/m, `path: ${absolutePath}`);
          }
          await fsPromises.writeFile(filePath, content, "utf8");
        } else {
          await fsPromises.writeFile(filePath, Buffer.from(file.content));
        }
      }

      console.log(`[YOLO Detection Dataset] Written ${files.length} files to ${datasetPath}`);

      return { success: true, datasetPath };
    } catch (error) {
      console.error("[YOLO Detection Dataset] Error writing dataset:", error);
      return {
        success: false,
        error: error instanceof Error ? error.message : "Failed to write dataset",
      };
    }
  }
);

// ============================================
// File Handling
// ============================================

// Handle file open from command line (Windows/Linux)
function getFileFromArgs(args: string[]): string | null {
  // Skip the first arg (executable path) and look for .fol files
  for (let i = 1; i < args.length; i++) {
    const arg = args[i];
    if (arg && arg.endsWith(".fol") && fs.existsSync(arg)) {
      return arg;
    }
  }
  return null;
}

// macOS: Handle file open before app is ready
app.on("open-file", (event, filePath) => {
  event.preventDefault();
  if (filePath.endsWith(".fol")) {
    if (mainWindow) {
      // App is already running, send to renderer
      mainWindow.webContents.send("file:open", filePath);
    } else {
      // App is starting, store for later
      fileToOpen = filePath;
    }
  }
});

// IPC handler for renderer to request file to open on startup
ipcMain.handle("app:getFileToOpen", () => {
  const file = fileToOpen;
  fileToOpen = null; // Clear after returning
  return file;
});

// App lifecycle
app.whenReady().then(() => {
  // Check for file in command line args (Windows/Linux)
  if (process.platform !== "darwin") {
    fileToOpen = getFileFromArgs(process.argv);
  }

  createWindow();
  createMenu();

  // Initialize auto-updater (only in production)
  if (app.isPackaged && mainWindow) {
    initUpdater(mainWindow);
  }

  // Listen for system suspend (sleep/hibernate) to auto-save before sleep
  powerMonitor.on("suspend", () => {
    console.log("System suspending - triggering auto-save");
    mainWindow?.webContents.send("system:suspend");
  });

  // Log when system resumes (for debugging)
  powerMonitor.on("resume", () => {
    console.log("System resumed from sleep");
  });

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

// Allow multiple instances - each double-clicked file opens a new window

app.on("window-all-closed", () => {
  // Stop BLOB server before quitting
  stopBlobServer();

  if (process.platform !== "darwin") {
    app.quit();
  }
});

// Ensure BLOB server is stopped on app quit
app.on("will-quit", () => {
  stopBlobServer();
});
