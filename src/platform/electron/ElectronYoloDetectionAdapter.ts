/**
 * Electron YOLO Detection Adapter
 * Wraps window.electronAPI.yoloDetection operations for Electron platform
 */

import type {
  YoloDetectionAdapter,
  DetectionTrainingConfig,
  DetectionTrainingProgress,
  DetectionModelInfo,
  ResumableModelInfo,
  DetectionValidation,
  WriteDatasetResult,
  StartTrainingResult,
  ExportDialogResult,
  ExportONNXResult,
  ExportTensorRTResult,
  PredictDetectionResult,
  PredictTiledResult,
} from '../types';
import type { YoloDetectionStatus, TensorRTStatus } from '../../types';

export class ElectronYoloDetectionAdapter implements YoloDetectionAdapter {
  async getStatus(): Promise<YoloDetectionStatus> {
    return window.electronAPI.yoloDetection.getStatus();
  }

  async validateDataset(datasetPath: string): Promise<DetectionValidation> {
    return window.electronAPI.yoloDetection.validateDataset(datasetPath);
  }

  async startTraining(
    datasetPath: string,
    config: DetectionTrainingConfig,
    modelName?: string
  ): Promise<StartTrainingResult> {
    return window.electronAPI.yoloDetection.startTraining(datasetPath, config, modelName);
  }

  async stopTraining(jobId: string): Promise<{ success: boolean }> {
    return window.electronAPI.yoloDetection.stopTraining(jobId);
  }

  subscribeProgress(
    jobId: string,
    onProgress: (progress: DetectionTrainingProgress) => void,
    onError: (error: string) => void,
    onComplete: () => void
  ): () => void {
    return window.electronAPI.yoloDetection.subscribeProgress(jobId, onProgress, onError, onComplete);
  }

  async listModels(): Promise<{ models: DetectionModelInfo[] }> {
    return window.electronAPI.yoloDetection.listModels();
  }

  async getResumableModels(): Promise<{ models: ResumableModelInfo[] }> {
    return window.electronAPI.yoloDetection.getResumableModels();
  }

  async loadModel(modelPath: string): Promise<{ success: boolean }> {
    return window.electronAPI.yoloDetection.loadModel(modelPath);
  }

  async predict(imageData: string, confidenceThreshold?: number): Promise<PredictDetectionResult> {
    return window.electronAPI.yoloDetection.predict(imageData, confidenceThreshold);
  }

  async predictTiled(
    imageData: string,
    confidenceThreshold?: number,
    tileSize?: number,
    overlap?: number,
    nmsThreshold?: number,
    scaleFactor?: number
  ): Promise<PredictTiledResult> {
    return window.electronAPI.yoloDetection.predictTiled(
      imageData,
      confidenceThreshold,
      tileSize,
      overlap,
      nmsThreshold,
      scaleFactor
    );
  }

  async showExportDialog(defaultFileName: string): Promise<ExportDialogResult> {
    return window.electronAPI.yoloDetection.showExportDialog(defaultFileName);
  }

  async exportONNX(modelPath: string, outputPath: string): Promise<ExportONNXResult> {
    return window.electronAPI.yoloDetection.exportONNX(modelPath, outputPath);
  }

  async deleteModel(modelId: string): Promise<{ success: boolean }> {
    return window.electronAPI.yoloDetection.deleteModel(modelId);
  }

  async writeDatasetToTemp(
    files: Array<{ path: string; content: ArrayBuffer | string }>
  ): Promise<WriteDatasetResult> {
    return window.electronAPI.yoloDetection.writeDatasetToTemp(files);
  }

  async checkTensorRTAvailable(): Promise<TensorRTStatus> {
    return window.electronAPI.yoloDetection.checkTensorRTAvailable();
  }

  async exportToTensorRT(
    modelPath: string,
    outputPath?: string,
    half?: boolean,
    imgsz?: number
  ): Promise<ExportTensorRTResult> {
    return window.electronAPI.yoloDetection.exportToTensorRT(modelPath, outputPath, half, imgsz);
  }
}
