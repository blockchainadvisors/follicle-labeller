import { contextBridge, ipcRenderer } from 'electron';

// Define the API exposed to the renderer
const electronAPI = {
  // Open image file dialog - returns file path and data as ArrayBuffer
  openImageDialog: (): Promise<{ filePath: string; fileName: string; data: ArrayBuffer } | null> =>
    ipcRenderer.invoke('dialog:openImage'),

  // Save JSON to file - returns true if saved successfully
  saveJsonDialog: (data: string): Promise<boolean> =>
    ipcRenderer.invoke('dialog:saveJson', data),

  // Open JSON file dialog - returns file path and content
  openJsonDialog: (): Promise<{ filePath: string; data: string } | null> =>
    ipcRenderer.invoke('dialog:openJson'),
};

// Expose the API to the renderer process
contextBridge.exposeInMainWorld('electronAPI', electronAPI);

// Export type for TypeScript
export type ElectronAPI = typeof electronAPI;
