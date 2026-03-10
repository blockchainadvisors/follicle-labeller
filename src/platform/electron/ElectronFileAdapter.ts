/**
 * Electron File Adapter
 * Wraps window.electronAPI file operations for Electron platform
 */

import type {
  FileAdapter,
  OpenImageResult,
  OpenFileOptions,
  SaveProjectResult,
  LoadProjectResult,
  ProjectImageData,
} from '../types';

export class ElectronFileAdapter implements FileAdapter {
  async openImageDialog(): Promise<OpenImageResult | null> {
    return window.electronAPI.openImageDialog();
  }

  async openFileDialog(options: OpenFileOptions): Promise<OpenImageResult | null> {
    return window.electronAPI.openFileDialog(options);
  }

  async saveProjectV2(
    images: ProjectImageData[],
    manifest: string,
    annotations: string,
    defaultPath?: string
  ): Promise<SaveProjectResult> {
    return window.electronAPI.saveProjectV2(images, manifest, annotations, defaultPath);
  }

  async saveProjectV2ToPath(
    filePath: string,
    images: ProjectImageData[],
    manifest: string,
    annotations: string
  ): Promise<SaveProjectResult> {
    return window.electronAPI.saveProjectV2ToPath(filePath, images, manifest, annotations);
  }

  async loadProjectV2(): Promise<LoadProjectResult | null> {
    return window.electronAPI.loadProjectV2();
  }

  async loadProjectFromPath(filePath: string): Promise<LoadProjectResult | null> {
    return window.electronAPI.loadProjectFromPath(filePath);
  }

  async fileExists(filePath: string): Promise<boolean> {
    return window.electronAPI.fileExists(filePath);
  }

  async showUnsavedChangesDialog(): Promise<'save' | 'discard' | 'cancel'> {
    return window.electronAPI.showUnsavedChangesDialog();
  }

  async showDownloadOptionsDialog(
    selectedCount: number,
    currentImageCount: number,
    totalCount: number
  ): Promise<'all' | 'currentImage' | 'selected' | 'cancel'> {
    return window.electronAPI.showDownloadOptionsDialog(selectedCount, currentImageCount, totalCount);
  }

  async getFileToOpen(): Promise<string | null> {
    return window.electronAPI.getFileToOpen();
  }

  setProjectState(hasProject: boolean): void {
    window.electronAPI.setProjectState(hasProject);
  }
}
