/**
 * Electron Menu Adapter
 * Wraps window.electronAPI menu listener operations for Electron platform
 */

import type { MenuAdapter } from '../types';

export class ElectronMenuAdapter implements MenuAdapter {
  onMenuOpenImage(callback: () => void): () => void {
    return window.electronAPI.onMenuOpenImage(callback);
  }

  onMenuLoadProject(callback: () => void): () => void {
    return window.electronAPI.onMenuLoadProject(callback);
  }

  onMenuSaveProject(callback: () => void): () => void {
    return window.electronAPI.onMenuSaveProject(callback);
  }

  onMenuSaveProjectAs(callback: () => void): () => void {
    return window.electronAPI.onMenuSaveProjectAs(callback);
  }

  onMenuCloseProject(callback: () => void): () => void {
    return window.electronAPI.onMenuCloseProject(callback);
  }

  onMenuUndo(callback: () => void): () => void {
    return window.electronAPI.onMenuUndo(callback);
  }

  onMenuRedo(callback: () => void): () => void {
    return window.electronAPI.onMenuRedo(callback);
  }

  onMenuClearAll(callback: () => void): () => void {
    return window.electronAPI.onMenuClearAll(callback);
  }

  onMenuToggleShapes(callback: () => void): () => void {
    return window.electronAPI.onMenuToggleShapes(callback);
  }

  onMenuToggleLabels(callback: () => void): () => void {
    return window.electronAPI.onMenuToggleLabels(callback);
  }

  onMenuZoomIn(callback: () => void): () => void {
    return window.electronAPI.onMenuZoomIn(callback);
  }

  onMenuZoomOut(callback: () => void): () => void {
    return window.electronAPI.onMenuZoomOut(callback);
  }

  onMenuResetZoom(callback: () => void): () => void {
    return window.electronAPI.onMenuResetZoom(callback);
  }

  onMenuShowHelp(callback: () => void): () => void {
    return window.electronAPI.onMenuShowHelp(callback);
  }

  onCheckUnsavedChanges(callback: () => void): () => void {
    return window.electronAPI.onCheckUnsavedChanges(callback);
  }

  onFileOpen(callback: (filePath: string) => void): () => void {
    return window.electronAPI.onFileOpen(callback);
  }

  onSystemSuspend(callback: () => void): () => void {
    return window.electronAPI.onSystemSuspend(callback);
  }

  confirmClose(canClose: boolean): void {
    window.electronAPI.confirmClose(canClose);
  }
}
