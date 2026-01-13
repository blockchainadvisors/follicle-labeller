import { contextBridge, ipcRenderer, IpcRendererEvent } from 'electron';

// Menu action callback type
type MenuCallback = () => void;

// Define the API exposed to the renderer
const electronAPI = {
  // Open image file dialog - returns file path and data as ArrayBuffer
  openImageDialog: (): Promise<{ filePath: string; fileName: string; data: ArrayBuffer } | null> =>
    ipcRenderer.invoke('dialog:openImage'),

  // Legacy V1: Save project as .fol archive (image + annotations)
  saveProject: (imageData: ArrayBuffer, imageFileName: string, jsonData: string): Promise<boolean> =>
    ipcRenderer.invoke('dialog:saveProject', imageData, imageFileName, jsonData),

  // Legacy V1: Load project from .fol archive
  loadProject: (): Promise<{ imageFileName: string; imageData: ArrayBuffer; jsonData: string } | null> =>
    ipcRenderer.invoke('dialog:loadProject'),

  // V2: Save project with multiple images
  saveProjectV2: (
    images: Array<{ id: string; fileName: string; data: ArrayBuffer }>,
    manifest: string,
    annotations: string
  ): Promise<boolean> =>
    ipcRenderer.invoke('dialog:saveProjectV2', images, manifest, annotations),

  // V2: Load project with support for V1 and V2 formats
  loadProjectV2: (): Promise<{
    version: '1.0' | '2.0';
    imageFileName?: string;
    imageData?: ArrayBuffer;
    jsonData?: string;
    manifest?: string;
    images?: Array<{ id: string; fileName: string; data: ArrayBuffer }>;
    annotations?: string;
  } | null> =>
    ipcRenderer.invoke('dialog:loadProjectV2'),

  // Save screenshot to file
  saveScreenshot: (imageData: ArrayBuffer, suggestedName: string): Promise<boolean> =>
    ipcRenderer.invoke('dialog:saveScreenshot', imageData, suggestedName),

  // Menu event listeners (return cleanup function)
  onMenuOpenImage: (callback: MenuCallback) => {
    const handler = (_event: IpcRendererEvent) => callback();
    ipcRenderer.on('menu:openImage', handler);
    return () => ipcRenderer.removeListener('menu:openImage', handler);
  },
  onMenuLoadProject: (callback: MenuCallback) => {
    const handler = (_event: IpcRendererEvent) => callback();
    ipcRenderer.on('menu:loadProject', handler);
    return () => ipcRenderer.removeListener('menu:loadProject', handler);
  },
  onMenuSaveProject: (callback: MenuCallback) => {
    const handler = (_event: IpcRendererEvent) => callback();
    ipcRenderer.on('menu:saveProject', handler);
    return () => ipcRenderer.removeListener('menu:saveProject', handler);
  },
  onMenuUndo: (callback: MenuCallback) => {
    const handler = (_event: IpcRendererEvent) => callback();
    ipcRenderer.on('menu:undo', handler);
    return () => ipcRenderer.removeListener('menu:undo', handler);
  },
  onMenuRedo: (callback: MenuCallback) => {
    const handler = (_event: IpcRendererEvent) => callback();
    ipcRenderer.on('menu:redo', handler);
    return () => ipcRenderer.removeListener('menu:redo', handler);
  },
  onMenuClearAll: (callback: MenuCallback) => {
    const handler = (_event: IpcRendererEvent) => callback();
    ipcRenderer.on('menu:clearAll', handler);
    return () => ipcRenderer.removeListener('menu:clearAll', handler);
  },
  onMenuToggleShapes: (callback: MenuCallback) => {
    const handler = (_event: IpcRendererEvent) => callback();
    ipcRenderer.on('menu:toggleShapes', handler);
    return () => ipcRenderer.removeListener('menu:toggleShapes', handler);
  },
  onMenuToggleLabels: (callback: MenuCallback) => {
    const handler = (_event: IpcRendererEvent) => callback();
    ipcRenderer.on('menu:toggleLabels', handler);
    return () => ipcRenderer.removeListener('menu:toggleLabels', handler);
  },
  onMenuZoomIn: (callback: MenuCallback) => {
    const handler = (_event: IpcRendererEvent) => callback();
    ipcRenderer.on('menu:zoomIn', handler);
    return () => ipcRenderer.removeListener('menu:zoomIn', handler);
  },
  onMenuZoomOut: (callback: MenuCallback) => {
    const handler = (_event: IpcRendererEvent) => callback();
    ipcRenderer.on('menu:zoomOut', handler);
    return () => ipcRenderer.removeListener('menu:zoomOut', handler);
  },
  onMenuResetZoom: (callback: MenuCallback) => {
    const handler = (_event: IpcRendererEvent) => callback();
    ipcRenderer.on('menu:resetZoom', handler);
    return () => ipcRenderer.removeListener('menu:resetZoom', handler);
  },
  onMenuShowHelp: (callback: MenuCallback) => {
    const handler = (_event: IpcRendererEvent) => callback();
    ipcRenderer.on('menu:showHelp', handler);
    return () => ipcRenderer.removeListener('menu:showHelp', handler);
  },
};

// Expose the API to the renderer process
contextBridge.exposeInMainWorld('electronAPI', electronAPI);

// Export type for TypeScript
export type ElectronAPI = typeof electronAPI;
