/**
 * Web Menu Adapter
 * No-op implementation for Web platform (no native menus)
 */

import type { MenuAdapter } from '../types';

/**
 * Web menu adapter - all methods are no-ops
 * Web platform uses in-page UI instead of native menus
 */
export class WebMenuAdapter implements MenuAdapter {
  setProjectState(_hasProject: boolean): void {
    // No-op for web - no native menu to update
  }

  async getFileToOpen(): Promise<string | null> {
    // Web platform doesn't support file associations
    return null;
  }

  onMenuOpenImage(_callback: () => void): () => void {
    return () => {};
  }

  onMenuLoadProject(_callback: () => void): () => void {
    return () => {};
  }

  onMenuSaveProject(_callback: () => void): () => void {
    return () => {};
  }

  onMenuSaveProjectAs(_callback: () => void): () => void {
    return () => {};
  }

  onMenuCloseProject(_callback: () => void): () => void {
    return () => {};
  }

  onMenuUndo(_callback: () => void): () => void {
    return () => {};
  }

  onMenuRedo(_callback: () => void): () => void {
    return () => {};
  }

  onMenuClearAll(_callback: () => void): () => void {
    return () => {};
  }

  onMenuToggleShapes(_callback: () => void): () => void {
    return () => {};
  }

  onMenuToggleLabels(_callback: () => void): () => void {
    return () => {};
  }

  onMenuZoomIn(_callback: () => void): () => void {
    return () => {};
  }

  onMenuZoomOut(_callback: () => void): () => void {
    return () => {};
  }

  onMenuResetZoom(_callback: () => void): () => void {
    return () => {};
  }

  onMenuShowHelp(_callback: () => void): () => void {
    return () => {};
  }

  onCheckUnsavedChanges(_callback: () => void): () => void {
    return () => {};
  }

  onFileOpen(_callback: (filePath: string) => void): () => void {
    return () => {};
  }

  onSystemSuspend(_callback: () => void): () => void {
    return () => {};
  }

  confirmClose(_canClose: boolean): void {
    // No-op for web
  }
}
