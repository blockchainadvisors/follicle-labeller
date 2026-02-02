/**
 * Web File Adapter
 * Uses HTML5 File API and server storage for Web platform
 */

import type {
  FileAdapter,
  OpenImageResult,
  OpenFileOptions,
  SaveProjectResult,
  LoadProjectResult,
  ProjectImageData,
  ServerProjectList,
} from '../types';
import { config } from '../config';

/**
 * Helper to create file input and trigger click
 */
function createFileInput(accept: string, multiple = false): HTMLInputElement {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = accept;
  input.multiple = multiple;
  input.style.display = 'none';
  document.body.appendChild(input);
  return input;
}

/**
 * Helper to read file as ArrayBuffer
 */
function readFileAsArrayBuffer(file: File): Promise<ArrayBuffer> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as ArrayBuffer);
    reader.onerror = () => reject(reader.error);
    reader.readAsArrayBuffer(file);
  });
}

export class WebFileAdapter implements FileAdapter {
  async openImageDialog(): Promise<OpenImageResult | null> {
    return new Promise((resolve) => {
      const input = createFileInput('image/*');

      input.onchange = async () => {
        const file = input.files?.[0];
        document.body.removeChild(input);

        if (!file) {
          resolve(null);
          return;
        }

        try {
          const data = await readFileAsArrayBuffer(file);
          resolve({
            filePath: file.name, // Web doesn't have real file paths
            fileName: file.name,
            data,
          });
        } catch {
          resolve(null);
        }
      };

      input.oncancel = () => {
        document.body.removeChild(input);
        resolve(null);
      };

      input.click();
    });
  }

  async openFileDialog(options: OpenFileOptions): Promise<OpenImageResult | null> {
    return new Promise((resolve) => {
      // Convert Electron-style filters to accept string
      let accept = '*/*';
      if (options.filters && options.filters.length > 0) {
        const extensions = options.filters.flatMap((f) => f.extensions);
        accept = extensions.map((ext) => `.${ext}`).join(',');
      }

      const input = createFileInput(accept);

      input.onchange = async () => {
        const file = input.files?.[0];
        document.body.removeChild(input);

        if (!file) {
          resolve(null);
          return;
        }

        try {
          const data = await readFileAsArrayBuffer(file);
          resolve({
            filePath: file.name,
            fileName: file.name,
            data,
          });
        } catch {
          resolve(null);
        }
      };

      input.oncancel = () => {
        document.body.removeChild(input);
        resolve(null);
      };

      input.click();
    });
  }

  async saveProjectV2(
    images: ProjectImageData[],
    manifest: string,
    annotations: string,
    _defaultPath?: string
  ): Promise<SaveProjectResult> {
    try {
      // Create FormData with all project data
      const formData = new FormData();
      formData.append('manifest', manifest);
      formData.append('annotations', annotations);

      // Add each image as a file
      for (const img of images) {
        const blob = new Blob([img.data], { type: 'application/octet-stream' });
        formData.append('images', blob, `${img.id}-${img.fileName}`);
      }

      const response = await fetch(`${config.backendUrl}/projects/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      const result = await response.json();
      return {
        success: true,
        filePath: result.projectId, // Server returns project ID as "path"
      };
    } catch (error) {
      console.error('Failed to save project:', error);
      return { success: false };
    }
  }

  async saveProjectV2ToPath(
    filePath: string,
    images: ProjectImageData[],
    manifest: string,
    annotations: string
  ): Promise<SaveProjectResult> {
    try {
      // Update existing project on server
      const formData = new FormData();
      formData.append('manifest', manifest);
      formData.append('annotations', annotations);

      for (const img of images) {
        const blob = new Blob([img.data], { type: 'application/octet-stream' });
        formData.append('images', blob, `${img.id}-${img.fileName}`);
      }

      const response = await fetch(`${config.backendUrl}/projects/${filePath}`, {
        method: 'PUT',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Update failed: ${response.statusText}`);
      }

      return { success: true, filePath };
    } catch (error) {
      console.error('Failed to update project:', error);
      return { success: false };
    }
  }

  async loadProjectV2(): Promise<LoadProjectResult | null> {
    try {
      // Show project picker dialog (custom implementation)
      const projectId = await this.showProjectPicker();
      if (!projectId) return null;

      return this.loadProjectFromPath(projectId);
    } catch (error) {
      console.error('Failed to load project:', error);
      return null;
    }
  }

  async loadProjectFromPath(filePath: string): Promise<LoadProjectResult | null> {
    try {
      const response = await fetch(`${config.backendUrl}/projects/${filePath}/download`);

      if (!response.ok) {
        throw new Error(`Download failed: ${response.statusText}`);
      }

      const data = await response.json();

      return {
        version: '2.0',
        filePath,
        manifest: data.manifest,
        images: data.images,
        annotations: data.annotations,
      };
    } catch (error) {
      console.error('Failed to load project from path:', error);
      return null;
    }
  }

  async fileExists(filePath: string): Promise<boolean> {
    try {
      // For web, check if file exists on server
      const response = await fetch(`${config.backendUrl}/files/exists?path=${encodeURIComponent(filePath)}`);
      if (!response.ok) return false;
      const data = await response.json();
      return data.exists;
    } catch {
      return false;
    }
  }

  async showUnsavedChangesDialog(): Promise<'save' | 'discard' | 'cancel'> {
    // Use browser's confirm dialog for web
    const result = window.confirm(
      'You have unsaved changes. Do you want to save before continuing?\n\n' +
        'Click OK to save, Cancel to discard changes.'
    );
    return result ? 'save' : 'discard';
  }

  async showDownloadOptionsDialog(
    selectedCount: number,
    currentImageCount: number,
    _totalCount: number
  ): Promise<'all' | 'currentImage' | 'selected' | 'cancel'> {
    // Simple prompt for web
    const options = ['all', 'currentImage'];
    if (selectedCount > 0) options.push('selected');

    const choice = window.prompt(
      `Download options:\n` +
        `1. all - Download all annotations\n` +
        `2. currentImage - Download current image annotations (${currentImageCount})\n` +
        (selectedCount > 0 ? `3. selected - Download selected annotations (${selectedCount})\n` : '') +
        `\nEnter your choice (all/currentImage${selectedCount > 0 ? '/selected' : ''}):`
    );

    if (!choice) return 'cancel';
    const normalized = choice.toLowerCase().trim();
    if (normalized === 'all' || normalized === '1') return 'all';
    if (normalized === 'currentimage' || normalized === '2') return 'currentImage';
    if (normalized === 'selected' || normalized === '3') return 'selected';
    return 'cancel';
  }

  async getFileToOpen(): Promise<string | null> {
    // Web doesn't support command line file opening
    return null;
  }

  setProjectState(_hasProject: boolean): void {
    // No menu state in web - this is a no-op
  }

  /**
   * Show project picker dialog for web
   * Lists available projects from server
   */
  private async showProjectPicker(): Promise<string | null> {
    try {
      const response = await fetch(`${config.backendUrl}/projects/list`);
      if (!response.ok) {
        throw new Error('Failed to fetch projects');
      }

      const data: ServerProjectList = await response.json();

      if (data.projects.length === 0) {
        alert('No projects found on server.');
        return null;
      }

      // Create project list string
      const projectList = data.projects
        .map((p, i) => `${i + 1}. ${p.name} (${p.imageCount} images, ${p.annotationCount} annotations)`)
        .join('\n');

      const choice = window.prompt(
        `Select a project to load:\n\n${projectList}\n\nEnter project number:`
      );

      if (!choice) return null;

      const index = parseInt(choice, 10) - 1;
      if (index >= 0 && index < data.projects.length) {
        return data.projects[index].id;
      }

      return null;
    } catch (error) {
      console.error('Failed to show project picker:', error);
      return null;
    }
  }
}
