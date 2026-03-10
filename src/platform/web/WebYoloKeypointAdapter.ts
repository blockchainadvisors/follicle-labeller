/**
 * Web YOLO Keypoint Adapter
 * Uses HTTP to communicate with backend for Web platform
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
import { config } from '../config';

export class WebYoloKeypointAdapter implements YoloKeypointAdapter {
  async getStatus(): Promise<YoloKeypointStatus> {
    try {
      const response = await fetch(`${config.backendUrl}/yolo-keypoint/status`);
      if (!response.ok) {
        return { available: false, sseAvailable: false, activeTrainingJobs: 0 };
      }
      return response.json();
    } catch {
      return { available: false, sseAvailable: false, activeTrainingJobs: 0 };
    }
  }

  async getSystemInfo(): Promise<SystemInfo> {
    const response = await fetch(`${config.backendUrl}/yolo-keypoint/system-info`);
    if (!response.ok) {
      throw new Error('Failed to get system info');
    }
    return response.json();
  }

  async checkDependencies(): Promise<YoloDependenciesInfo> {
    // On web/server mode, dependencies are pre-installed
    return {
      installed: true,
      missing: [],
      estimatedSize: '0 MB',
    };
  }

  async installDependencies(): Promise<{ success: boolean; error?: string }> {
    // Dependencies are pre-installed on server
    return { success: true };
  }

  async upgradeToCUDA(): Promise<{ success: boolean; error?: string }> {
    // Server already has CUDA configured
    return { success: true };
  }

  onInstallProgress(_callback: (data: { message: string; percent?: number }) => void): () => void {
    // No installation progress for web mode
    return () => {};
  }

  async validateDataset(datasetPath: string): Promise<DatasetValidation> {
    const response = await fetch(`${config.backendUrl}/yolo-keypoint/validate-dataset`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset_path: datasetPath }),
    });

    if (!response.ok) {
      throw new Error('Failed to validate dataset');
    }

    const data = await response.json();
    return {
      valid: data.valid,
      trainImages: data.train_images,
      valImages: data.val_images,
      trainLabels: data.train_labels,
      valLabels: data.val_labels,
      errors: data.errors,
      warnings: data.warnings,
    };
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

      const response = await fetch(`${config.backendUrl}/yolo-keypoint/write-dataset`, {
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

  async startTraining(
    datasetPath: string,
    trainingConfig: TrainingConfig,
    modelName?: string
  ): Promise<StartTrainingResult> {
    const response = await fetch(`${config.backendUrl}/yolo-keypoint/train/start`, {
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
    const response = await fetch(`${config.backendUrl}/yolo-keypoint/train/stop/${jobId}`, {
      method: 'POST',
    });

    return { success: response.ok };
  }

  subscribeProgress(
    jobId: string,
    onProgress: (progress: TrainingProgress) => void,
    onError: (error: string) => void,
    onComplete: () => void
  ): () => void {
    const eventSource = new EventSource(
      `${config.backendUrl}/yolo-keypoint/train/progress/${jobId}`
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
            poseLoss: data.pose_loss || 0,
            kobjLoss: data.kobj_loss || 0,
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

  async listModels(): Promise<{ models: ModelInfo[] }> {
    const response = await fetch(`${config.backendUrl}/yolo-keypoint/models`);
    if (!response.ok) {
      return { models: [] };
    }
    return response.json();
  }

  async loadModel(modelPath: string): Promise<{ success: boolean }> {
    const response = await fetch(`${config.backendUrl}/yolo-keypoint/load-model`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_path: modelPath }),
    });

    return { success: response.ok };
  }

  async predict(imageData: string): Promise<PredictKeypointResult> {
    const response = await fetch(`${config.backendUrl}/yolo-keypoint/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_data: imageData }),
    });

    if (!response.ok) {
      return { success: false, message: 'Prediction failed' };
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
    const response = await fetch(`${config.backendUrl}/yolo-keypoint/export-onnx`, {
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
    const response = await fetch(`${config.backendUrl}/yolo-keypoint/models/${modelId}`, {
      method: 'DELETE',
    });

    return { success: response.ok };
  }

  async checkTensorRTAvailable(): Promise<TensorRTStatus> {
    const response = await fetch(`${config.backendUrl}/yolo-keypoint/check-tensorrt`);
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
    const response = await fetch(`${config.backendUrl}/yolo-keypoint/export-tensorrt`, {
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
