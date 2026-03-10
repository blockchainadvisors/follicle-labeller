/**
 * Electron YOLO Keypoint Adapter
 * Wraps window.electronAPI.yoloKeypoint operations for Electron platform
 */

import type {
  YoloKeypointAdapter,
  SystemInfo,
  WriteDatasetResult,
  StartTrainingResult,
  ExportDialogResult,
  ExportONNXResult,
  ExportTensorRTResult,
  PredictKeypointResult,
} from '../types';
import type {
  YoloKeypointStatus,
  YoloDependenciesInfo,
  DatasetValidation,
  TrainingConfig,
  TrainingProgress,
  ModelInfo,
  TensorRTStatus,
} from '../../types';

export class ElectronYoloKeypointAdapter implements YoloKeypointAdapter {
  async getStatus(): Promise<YoloKeypointStatus> {
    return window.electronAPI.yoloKeypoint.getStatus();
  }

  async getSystemInfo(): Promise<SystemInfo> {
    return window.electronAPI.yoloKeypoint.getSystemInfo();
  }

  async checkDependencies(): Promise<YoloDependenciesInfo> {
    return window.electronAPI.yoloKeypoint.checkDependencies();
  }

  async installDependencies(): Promise<{ success: boolean; error?: string }> {
    return window.electronAPI.yoloKeypoint.installDependencies();
  }

  async upgradeToCUDA(): Promise<{ success: boolean; error?: string }> {
    return window.electronAPI.yoloKeypoint.upgradeToCUDA();
  }

  onInstallProgress(callback: (data: { message: string; percent?: number }) => void): () => void {
    return window.electronAPI.yoloKeypoint.onInstallProgress(callback);
  }

  async validateDataset(datasetPath: string): Promise<DatasetValidation> {
    return window.electronAPI.yoloKeypoint.validateDataset(datasetPath);
  }

  async writeDatasetToTemp(
    files: Array<{ path: string; content: ArrayBuffer | string }>
  ): Promise<WriteDatasetResult> {
    return window.electronAPI.yoloKeypoint.writeDatasetToTemp(files);
  }

  async startTraining(
    datasetPath: string,
    config: TrainingConfig,
    modelName?: string
  ): Promise<StartTrainingResult> {
    return window.electronAPI.yoloKeypoint.startTraining(datasetPath, config, modelName);
  }

  async stopTraining(jobId: string): Promise<{ success: boolean }> {
    return window.electronAPI.yoloKeypoint.stopTraining(jobId);
  }

  subscribeProgress(
    jobId: string,
    onProgress: (progress: TrainingProgress) => void,
    onError: (error: string) => void,
    onComplete: () => void
  ): () => void {
    return window.electronAPI.yoloKeypoint.subscribeProgress(jobId, onProgress, onError, onComplete);
  }

  async listModels(): Promise<{ models: ModelInfo[] }> {
    return window.electronAPI.yoloKeypoint.listModels();
  }

  async loadModel(modelPath: string): Promise<{ success: boolean }> {
    return window.electronAPI.yoloKeypoint.loadModel(modelPath);
  }

  async predict(imageData: string): Promise<PredictKeypointResult> {
    return window.electronAPI.yoloKeypoint.predict(imageData);
  }

  async showExportDialog(defaultFileName: string): Promise<ExportDialogResult> {
    return window.electronAPI.yoloKeypoint.showExportDialog(defaultFileName);
  }

  async exportONNX(modelPath: string, outputPath: string): Promise<ExportONNXResult> {
    return window.electronAPI.yoloKeypoint.exportONNX(modelPath, outputPath);
  }

  async deleteModel(modelId: string): Promise<{ success: boolean }> {
    return window.electronAPI.yoloKeypoint.deleteModel(modelId);
  }

  async checkTensorRTAvailable(): Promise<TensorRTStatus> {
    return window.electronAPI.yoloKeypoint.checkTensorRTAvailable();
  }

  async exportToTensorRT(
    modelPath: string,
    outputPath?: string,
    half?: boolean,
    imgsz?: number
  ): Promise<ExportTensorRTResult> {
    return window.electronAPI.yoloKeypoint.exportToTensorRT(modelPath, outputPath, half, imgsz);
  }
}
