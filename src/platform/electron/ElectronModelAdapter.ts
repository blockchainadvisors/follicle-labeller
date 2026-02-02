/**
 * Electron Model Adapter
 * Wraps window.electronAPI.model operations for Electron platform
 */

import type {
  ModelAdapter,
  ExportPackageResult,
  PreviewPackageResult,
  ImportPackageResult,
} from '../types';

export class ElectronModelAdapter implements ModelAdapter {
  async exportPackage(
    modelId: string,
    modelPath: string,
    config: Record<string, unknown>,
    suggestedFileName?: string
  ): Promise<ExportPackageResult> {
    return window.electronAPI.model.exportPackage(modelId, modelPath, config, suggestedFileName);
  }

  async previewPackage(
    expectedModelType?: 'detection' | 'keypoint'
  ): Promise<PreviewPackageResult> {
    return window.electronAPI.model.previewPackage(expectedModelType);
  }

  async importPackage(
    filePath: string,
    newModelName?: string
  ): Promise<ImportPackageResult> {
    return window.electronAPI.model.importPackage(filePath, newModelName);
  }
}
