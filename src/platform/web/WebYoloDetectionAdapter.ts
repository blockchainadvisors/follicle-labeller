/**
 * Web YOLO Detection Adapter
 * Uses HTTP to communicate with backend for Web platform
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
import { config } from '../config';

export class WebYoloDetectionAdapter implements YoloDetectionAdapter {
  async getStatus(): Promise<YoloDetectionStatus> {
    try {
      const response = await fetch(`${config.backendUrl}/yolo-detect/status`);
      if (!response.ok) {
        return { available: false, sseAvailable: false, activeTrainingJobs: 0, loadedModel: null };
      }
      return response.json();
    } catch {
      return { available: false, sseAvailable: false, activeTrainingJobs: 0, loadedModel: null };
    }
  }

  async validateDataset(datasetPath: string): Promise<DetectionValidation> {
    const response = await fetch(`${config.backendUrl}/yolo-detect/validate-dataset`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset_path: datasetPath }),
    });

    if (!response.ok) {
      throw new Error('Failed to validate dataset');
    }

    return response.json();
  }

  async startTraining(
    datasetPath: string,
    trainingConfig: DetectionTrainingConfig,
    modelName?: string
  ): Promise<StartTrainingResult> {
    const response = await fetch(`${config.backendUrl}/yolo-detect/train/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset_path: datasetPath,
        model_size: trainingConfig.modelSize,
        epochs: trainingConfig.epochs,
        img_size: trainingConfig.imgSize,
        batch_size: trainingConfig.batchSize,
        patience: trainingConfig.patience,
        device: trainingConfig.device,
        resume_from: trainingConfig.resumeFrom,
        model_name: modelName,
      }),
    });

    if (!response.ok) {
      throw new Error('Failed to start training');
    }

    const data = await response.json();
    return {
      jobId: data.job_id,
      status: data.status,
    };
  }

  async stopTraining(jobId: string): Promise<{ success: boolean }> {
    const response = await fetch(`${config.backendUrl}/yolo-detect/train/stop/${jobId}`, {
      method: 'POST',
    });

    return { success: response.ok };
  }

  subscribeProgress(
    jobId: string,
    onProgress: (progress: DetectionTrainingProgress) => void,
    onError: (error: string) => void,
    onComplete: () => void
  ): () => void {
    const eventSource = new EventSource(
      `${config.backendUrl}/yolo-detect/train/progress/${jobId}`
    );

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.status === 'completed') {
          onComplete();
          eventSource.close();
        } else if (data.status === 'failed') {
          onError(data.message || 'Training failed');
          eventSource.close();
        } else {
          onProgress({
            status: data.status,
            epoch: data.epoch || 0,
            totalEpochs: data.total_epochs || 0,
            loss: data.loss || 0,
            boxLoss: data.box_loss || 0,
            clsLoss: data.cls_loss || 0,
            dflLoss: data.dfl_loss || 0,
            metrics: data.metrics || {},
            eta: data.eta || '',
            message: data.message || '',
          });
        }
      } catch (e) {
        console.error('Failed to parse SSE message:', e);
      }
    };

    eventSource.onerror = () => {
      onError('Connection lost');
      eventSource.close();
    };

    return () => {
      eventSource.close();
    };
  }

  async listModels(): Promise<{ models: DetectionModelInfo[] }> {
    const response = await fetch(`${config.backendUrl}/yolo-detect/models`);
    if (!response.ok) {
      return { models: [] };
    }
    return response.json();
  }

  async getResumableModels(): Promise<{ models: ResumableModelInfo[] }> {
    const response = await fetch(`${config.backendUrl}/yolo-detect/resumable-models`);
    if (!response.ok) {
      return { models: [] };
    }
    return response.json();
  }

  async loadModel(modelPath: string): Promise<{ success: boolean }> {
    const response = await fetch(`${config.backendUrl}/yolo-detect/load-model`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_path: modelPath }),
    });

    return { success: response.ok };
  }

  async predict(imageData: string, confidenceThreshold?: number): Promise<PredictDetectionResult> {
    const response = await fetch(`${config.backendUrl}/yolo-detect/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_data: imageData,
        confidence_threshold: confidenceThreshold,
      }),
    });

    if (!response.ok) {
      return { success: false, detections: [], count: 0 };
    }

    return response.json();
  }

  async predictTiled(
    imageData: string,
    confidenceThreshold?: number,
    tileSize?: number,
    overlap?: number,
    nmsThreshold?: number,
    scaleFactor?: number
  ): Promise<PredictTiledResult> {
    const response = await fetch(`${config.backendUrl}/yolo-detect/predict-tiled`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_data: imageData,
        confidence_threshold: confidenceThreshold,
        tile_size: tileSize,
        overlap,
        nms_threshold: nmsThreshold,
        scale_factor: scaleFactor,
      }),
    });

    if (!response.ok) {
      return {
        success: false,
        detections: [],
        count: 0,
        method: 'tiled',
        tileSize: tileSize || 640,
        overlap: overlap || 0.25,
        scaleFactor: scaleFactor || 1.0,
      };
    }

    return response.json();
  }

  async showExportDialog(_defaultFileName: string): Promise<ExportDialogResult> {
    // Web doesn't have native save dialogs - return a generated path
    const fileName = window.prompt('Enter export filename:', _defaultFileName);
    if (!fileName) {
      return { canceled: true };
    }
    return { canceled: false, filePath: fileName };
  }

  async exportONNX(modelPath: string, outputPath: string): Promise<ExportONNXResult> {
    const response = await fetch(`${config.backendUrl}/yolo-detect/export-onnx`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_path: modelPath, output_path: outputPath }),
    });

    if (!response.ok) {
      return { success: false };
    }

    const data = await response.json();
    return { success: true, outputPath: data.output_path };
  }

  async deleteModel(modelId: string): Promise<{ success: boolean }> {
    const response = await fetch(`${config.backendUrl}/yolo-detect/models/${modelId}`, {
      method: 'DELETE',
    });

    return { success: response.ok };
  }

  async writeDatasetToTemp(
    files: Array<{ path: string; content: ArrayBuffer | string }>
  ): Promise<WriteDatasetResult> {
    try {
      const formData = new FormData();

      for (const file of files) {
        const content =
          typeof file.content === 'string'
            ? new Blob([file.content], { type: 'text/plain' })
            : new Blob([file.content], { type: 'application/octet-stream' });
        formData.append('files', content, file.path);
      }

      const response = await fetch(`${config.backendUrl}/yolo-detect/write-dataset`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to write dataset');
      }

      const data = await response.json();
      return {
        success: true,
        datasetPath: data.dataset_path,
      };
    } catch (error) {
      return {
        success: false,
        error: String(error),
      };
    }
  }

  async checkTensorRTAvailable(): Promise<TensorRTStatus> {
    const response = await fetch(`${config.backendUrl}/yolo-detect/check-tensorrt`);
    if (!response.ok) {
      return { available: false, version: null };
    }
    return response.json();
  }

  async exportToTensorRT(
    modelPath: string,
    outputPath?: string,
    half?: boolean,
    imgsz?: number
  ): Promise<ExportTensorRTResult> {
    const response = await fetch(`${config.backendUrl}/yolo-detect/export-tensorrt`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model_path: modelPath,
        output_path: outputPath,
        half,
        imgsz,
      }),
    });

    if (!response.ok) {
      return { success: false, error: 'Export failed' };
    }

    return response.json();
  }
}
