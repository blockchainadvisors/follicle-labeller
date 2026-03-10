/**
 * Web Model Adapter
 * Uses HTTP to communicate with backend for Web platform
 */

import type {
  ModelAdapter,
  ExportPackageResult,
  PreviewPackageResult,
  ImportPackageResult,
} from '../types';
import { config } from '../config';

export class WebModelAdapter implements ModelAdapter {
  async exportPackage(
    modelId: string,
    modelPath: string,
    modelConfig: Record<string, unknown>,
    suggestedFileName?: string
  ): Promise<ExportPackageResult> {
    try {
      const response = await fetch(`${config.backendUrl}/model/export-package`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_id: modelId,
          model_path: modelPath,
          config: modelConfig,
          suggested_filename: suggestedFileName,
        }),
      });

      if (!response.ok) {
        return { success: false, error: 'Export failed' };
      }

      // Server returns the package as a blob - trigger download
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = suggestedFileName || `model-${modelId}.fmp`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      return { success: true, filePath: suggestedFileName };
    } catch (error) {
      return { success: false, error: String(error) };
    }
  }

  async previewPackage(
    expectedModelType?: 'detection' | 'keypoint'
  ): Promise<PreviewPackageResult> {
    return new Promise((resolve) => {
      // Create file input for selecting .fmp file
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = '.fmp';
      input.style.display = 'none';
      document.body.appendChild(input);

      input.onchange = async () => {
        const file = input.files?.[0];
        document.body.removeChild(input);

        if (!file) {
          resolve({ valid: false, canceled: true });
          return;
        }

        try {
          const formData = new FormData();
          formData.append('file', file);
          if (expectedModelType) {
            formData.append('expected_type', expectedModelType);
          }

          const response = await fetch(`${config.backendUrl}/model/preview-package`, {
            method: 'POST',
            body: formData,
          });

          if (!response.ok) {
            resolve({ valid: false, error: 'Preview failed' });
            return;
          }

          const data = await response.json();
          resolve({
            valid: data.valid,
            filePath: file.name,
            config: data.config,
            modelType: data.model_type,
            hasEngine: data.has_engine,
          });
        } catch (error) {
          resolve({ valid: false, error: String(error) });
        }
      };

      input.oncancel = () => {
        document.body.removeChild(input);
        resolve({ valid: false, canceled: true });
      };

      input.click();
    });
  }

  async importPackage(
    filePath: string,
    newModelName?: string
  ): Promise<ImportPackageResult> {
    try {
      // For web, we need to upload the file first
      // This assumes the file was already uploaded during preview
      const response = await fetch(`${config.backendUrl}/model/import-package`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          file_path: filePath,
          new_model_name: newModelName,
        }),
      });

      if (!response.ok) {
        return { success: false, error: 'Import failed' };
      }

      const data = await response.json();
      return {
        success: true,
        modelId: data.model_id,
        modelPath: data.model_path,
        modelName: data.model_name,
        modelType: data.model_type,
      };
    } catch (error) {
      return { success: false, error: String(error) };
    }
  }
}
