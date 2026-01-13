import { contextBridge, ipcRenderer } from 'electron';

// Define the API exposed to the renderer
const electronAPI = {
  // Open image file dialog - returns file path and data as ArrayBuffer
  openImageDialog: (): Promise<{ filePath: string; fileName: string; data: ArrayBuffer } | null> =>
    ipcRenderer.invoke('dialog:openImage'),

  // Save project as .fol archive (image + annotations)
  saveProject: (imageData: ArrayBuffer, imageFileName: string, jsonData: string): Promise<boolean> =>
    ipcRenderer.invoke('dialog:saveProject', imageData, imageFileName, jsonData),

  // Load project from .fol archive
  loadProject: (): Promise<{ imageFileName: string; imageData: ArrayBuffer; jsonData: string } | null> =>
    ipcRenderer.invoke('dialog:loadProject'),
};

// Expose the API to the renderer process
contextBridge.exposeInMainWorld('electronAPI', electronAPI);

// Export type for TypeScript
export type ElectronAPI = typeof electronAPI;
