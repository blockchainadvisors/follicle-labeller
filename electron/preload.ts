import { contextBridge, ipcRenderer, IpcRendererEvent } from "electron";

// Menu action callback type
type MenuCallback = () => void;

// Define the API exposed to the renderer
const electronAPI = {
  // Open image file dialog - returns file path and data as ArrayBuffer
  openImageDialog: (): Promise<{
    filePath: string;
    fileName: string;
    data: ArrayBuffer;
  } | null> => ipcRenderer.invoke("dialog:openImage"),

  // Legacy V1: Save project as .fol archive (image + annotations)
  saveProject: (
    imageData: ArrayBuffer,
    imageFileName: string,
    jsonData: string,
  ): Promise<boolean> =>
    ipcRenderer.invoke(
      "dialog:saveProject",
      imageData,
      imageFileName,
      jsonData,
    ),

  // Legacy V1: Load project from .fol archive
  loadProject: (): Promise<{
    imageFileName: string;
    imageData: ArrayBuffer;
    jsonData: string;
  } | null> => ipcRenderer.invoke("dialog:loadProject"),

  // V2: Save project with multiple images (Save As - shows dialog)
  saveProjectV2: (
    images: Array<{ id: string; fileName: string; data: ArrayBuffer }>,
    manifest: string,
    annotations: string,
    defaultPath?: string,
  ): Promise<{ success: boolean; filePath?: string }> =>
    ipcRenderer.invoke(
      "dialog:saveProjectV2",
      images,
      manifest,
      annotations,
      defaultPath,
    ),

  // V2: Save project to specific path (silent save - no dialog)
  saveProjectV2ToPath: (
    filePath: string,
    images: Array<{ id: string; fileName: string; data: ArrayBuffer }>,
    manifest: string,
    annotations: string,
  ): Promise<{ success: boolean; filePath?: string }> =>
    ipcRenderer.invoke(
      "file:saveProjectV2",
      filePath,
      images,
      manifest,
      annotations,
    ),

  // Update menu state based on project
  setProjectState: (hasProject: boolean): void => {
    ipcRenderer.send("menu:setProjectState", hasProject);
  },

  // V2: Load project with support for V1 and V2 formats
  loadProjectV2: (): Promise<{
    version: "1.0" | "2.0";
    filePath: string;
    imageFileName?: string;
    imageData?: ArrayBuffer;
    jsonData?: string;
    manifest?: string;
    images?: Array<{ id: string; fileName: string; data: ArrayBuffer }>;
    annotations?: string;
  } | null> => ipcRenderer.invoke("dialog:loadProjectV2"),

  // Get file to open on startup (from file association)
  getFileToOpen: (): Promise<string | null> =>
    ipcRenderer.invoke("app:getFileToOpen"),

  // Load project from specific file path (for file association)
  loadProjectFromPath: (
    filePath: string,
  ): Promise<{
    version: "1.0" | "2.0";
    filePath: string;
    imageFileName?: string;
    imageData?: ArrayBuffer;
    jsonData?: string;
    manifest?: string;
    images?: Array<{ id: string; fileName: string; data: ArrayBuffer }>;
    annotations?: string;
  } | null> => ipcRenderer.invoke("file:loadProject", filePath),

  // Listen for file open events (when app is already running)
  onFileOpen: (callback: (filePath: string) => void) => {
    const handler = (_event: IpcRendererEvent, filePath: string) =>
      callback(filePath);
    ipcRenderer.on("file:open", handler);
    return () => ipcRenderer.removeListener("file:open", handler);
  },

  // Unsaved changes dialog - returns 'save' | 'discard' | 'cancel'
  showUnsavedChangesDialog: (): Promise<"save" | "discard" | "cancel"> =>
    ipcRenderer.invoke("dialog:unsavedChanges"),

  // Download options dialog - returns 'all' | 'currentImage' | 'selected' | 'cancel'
  showDownloadOptionsDialog: (
    selectedCount: number,
    currentImageCount: number,
    totalCount: number,
  ): Promise<"all" | "currentImage" | "selected" | "cancel"> =>
    ipcRenderer.invoke(
      "dialog:downloadOptions",
      selectedCount,
      currentImageCount,
      totalCount,
    ),

  // Listen for unsaved changes check request from main
  onCheckUnsavedChanges: (callback: () => void) => {
    const handler = (_event: IpcRendererEvent) => callback();
    ipcRenderer.on("app:checkUnsavedChanges", handler);
    return () => ipcRenderer.removeListener("app:checkUnsavedChanges", handler);
  },

  // Confirm close to main process
  confirmClose: (canClose: boolean): void => {
    ipcRenderer.send("app:confirmClose", canClose);
  },

  // Update download progress listener (for optional custom UI)
  onUpdateDownloadProgress: (
    callback: (progress: {
      percent: number;
      transferred: number;
      total: number;
      bytesPerSecond: number;
    }) => void,
  ) => {
    const handler = (
      _event: IpcRendererEvent,
      progress: {
        percent: number;
        transferred: number;
        total: number;
        bytesPerSecond: number;
      },
    ) => callback(progress);
    ipcRenderer.on("update:downloadProgress", handler);
    return () => ipcRenderer.removeListener("update:downloadProgress", handler);
  },

  // Menu event listeners (return cleanup function)
  onMenuOpenImage: (callback: MenuCallback) => {
    const handler = (_event: IpcRendererEvent) => callback();
    ipcRenderer.on("menu:openImage", handler);
    return () => ipcRenderer.removeListener("menu:openImage", handler);
  },
  onMenuLoadProject: (callback: MenuCallback) => {
    const handler = (_event: IpcRendererEvent) => callback();
    ipcRenderer.on("menu:loadProject", handler);
    return () => ipcRenderer.removeListener("menu:loadProject", handler);
  },
  onMenuSaveProject: (callback: MenuCallback) => {
    const handler = (_event: IpcRendererEvent) => callback();
    ipcRenderer.on("menu:saveProject", handler);
    return () => ipcRenderer.removeListener("menu:saveProject", handler);
  },
  onMenuSaveProjectAs: (callback: MenuCallback) => {
    const handler = (_event: IpcRendererEvent) => callback();
    ipcRenderer.on("menu:saveProjectAs", handler);
    return () => ipcRenderer.removeListener("menu:saveProjectAs", handler);
  },
  onMenuCloseProject: (callback: MenuCallback) => {
    const handler = (_event: IpcRendererEvent) => callback();
    ipcRenderer.on("menu:closeProject", handler);
    return () => ipcRenderer.removeListener("menu:closeProject", handler);
  },
  onMenuUndo: (callback: MenuCallback) => {
    const handler = (_event: IpcRendererEvent) => callback();
    ipcRenderer.on("menu:undo", handler);
    return () => ipcRenderer.removeListener("menu:undo", handler);
  },
  onMenuRedo: (callback: MenuCallback) => {
    const handler = (_event: IpcRendererEvent) => callback();
    ipcRenderer.on("menu:redo", handler);
    return () => ipcRenderer.removeListener("menu:redo", handler);
  },
  onMenuClearAll: (callback: MenuCallback) => {
    const handler = (_event: IpcRendererEvent) => callback();
    ipcRenderer.on("menu:clearAll", handler);
    return () => ipcRenderer.removeListener("menu:clearAll", handler);
  },
  onMenuToggleShapes: (callback: MenuCallback) => {
    const handler = (_event: IpcRendererEvent) => callback();
    ipcRenderer.on("menu:toggleShapes", handler);
    return () => ipcRenderer.removeListener("menu:toggleShapes", handler);
  },
  onMenuToggleLabels: (callback: MenuCallback) => {
    const handler = (_event: IpcRendererEvent) => callback();
    ipcRenderer.on("menu:toggleLabels", handler);
    return () => ipcRenderer.removeListener("menu:toggleLabels", handler);
  },
  onMenuZoomIn: (callback: MenuCallback) => {
    const handler = (_event: IpcRendererEvent) => callback();
    ipcRenderer.on("menu:zoomIn", handler);
    return () => ipcRenderer.removeListener("menu:zoomIn", handler);
  },
  onMenuZoomOut: (callback: MenuCallback) => {
    const handler = (_event: IpcRendererEvent) => callback();
    ipcRenderer.on("menu:zoomOut", handler);
    return () => ipcRenderer.removeListener("menu:zoomOut", handler);
  },
  onMenuResetZoom: (callback: MenuCallback) => {
    const handler = (_event: IpcRendererEvent) => callback();
    ipcRenderer.on("menu:resetZoom", handler);
    return () => ipcRenderer.removeListener("menu:resetZoom", handler);
  },
  onMenuShowHelp: (callback: MenuCallback) => {
    const handler = (_event: IpcRendererEvent) => callback();
    ipcRenderer.on("menu:showHelp", handler);
    return () => ipcRenderer.removeListener("menu:showHelp", handler);
  },

  // System power events - triggered before sleep/hibernate
  onSystemSuspend: (callback: MenuCallback) => {
    const handler = (_event: IpcRendererEvent) => callback();
    ipcRenderer.on("system:suspend", handler);
    return () => ipcRenderer.removeListener("system:suspend", handler);
  },

  // BLOB Detection Server API
  blob: {
    // Start the BLOB detection server
    startServer: (): Promise<{ success: boolean; error?: string }> =>
      ipcRenderer.invoke("blob:startServer"),

    // Stop the BLOB detection server
    stopServer: (): Promise<{ success: boolean }> =>
      ipcRenderer.invoke("blob:stopServer"),

    // Check if BLOB server is available
    isAvailable: (): Promise<boolean> => ipcRenderer.invoke("blob:isAvailable"),

    // Check if Python is installed
    checkPython: (): Promise<{
      available: boolean;
      version?: string;
      error?: string;
    }> => ipcRenderer.invoke("blob:checkPython"),

    // Get BLOB server info
    getServerInfo: (): Promise<{
      port: number;
      running: boolean;
      scriptPath: string;
    }> => ipcRenderer.invoke("blob:getServerInfo"),
  },
};

// Expose the API to the renderer process
contextBridge.exposeInMainWorld("electronAPI", electronAPI);

// Export type for TypeScript
export type ElectronAPI = typeof electronAPI;
