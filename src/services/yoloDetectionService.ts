/**
 * YOLO Detection Training Service
 *
 * Provides a TypeScript interface for YOLO11 detection model training
 * and inference for follicle bounding box detection.
 */

import {
  DetectionTrainingConfig,
  DetectionTrainingProgress,
  DetectionPrediction,
  DetectionModelInfo,
  DetectionDatasetValidation,
  YoloDetectionStatus,
  DEFAULT_DETECTION_TRAINING_CONFIG,
} from '../types';

/**
 * Sleep for specified milliseconds.
 */
const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Retry a function with exponential backoff.
 */
async function withRetry<T>(
  fn: () => Promise<T>,
  options: {
    maxRetries?: number;
    initialDelayMs?: number;
    maxDelayMs?: number;
    shouldRetry?: (error: unknown) => boolean;
  } = {}
): Promise<T> {
  const {
    maxRetries = 5,
    initialDelayMs = 500,
    maxDelayMs = 5000,
    shouldRetry = () => true,
  } = options;

  let lastError: unknown;
  let delay = initialDelayMs;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;

      // Check if we should retry
      if (attempt >= maxRetries || !shouldRetry(error)) {
        throw error;
      }

      // Wait before retry
      console.log(`Retry attempt ${attempt + 1}/${maxRetries} after ${delay}ms...`);
      await sleep(delay);

      // Exponential backoff with cap
      delay = Math.min(delay * 2, maxDelayMs);
    }
  }

  throw lastError;
}

/**
 * Check if an error is a connection error (server not ready).
 */
function isConnectionError(error: unknown): boolean {
  if (error instanceof Error) {
    const msg = error.message.toLowerCase();
    return (
      msg.includes('econnrefused') ||
      msg.includes('connection refused') ||
      msg.includes('network error') ||
      msg.includes('fetch failed')
    );
  }
  return false;
}

/**
 * Service for managing YOLO detection model training and inference.
 *
 * Singleton pattern - use YOLODetectionService.getInstance() to get the instance.
 */
export class YOLODetectionService {
  private static instance: YOLODetectionService | null = null;
  private activeJobId: string | null = null;
  private cleanupFunction: (() => void) | null = null;

  private constructor() {}

  /**
   * Get the singleton instance of the service.
   */
  public static getInstance(): YOLODetectionService {
    if (!YOLODetectionService.instance) {
      YOLODetectionService.instance = new YOLODetectionService();
    }
    return YOLODetectionService.instance;
  }

  /**
   * Get the YOLO detection service status.
   * Uses retry logic to handle server startup race condition.
   */
  async getStatus(): Promise<YoloDetectionStatus> {
    try {
      return await withRetry(
        () => window.electronAPI.yoloDetection.getStatus(),
        {
          maxRetries: 8,
          initialDelayMs: 500,
          maxDelayMs: 3000,
          shouldRetry: isConnectionError,
        }
      );
    } catch (error) {
      console.error('Failed to get YOLO detection status:', error);
      return {
        available: false,
        sseAvailable: false,
        activeTrainingJobs: 0,
        loadedModel: null,
      };
    }
  }

  /**
   * Validate a YOLO detection dataset.
   *
   * @param datasetPath Path to the dataset directory
   * @returns Validation result with errors and warnings
   */
  async validateDataset(datasetPath: string): Promise<DetectionDatasetValidation> {
    try {
      const result = await window.electronAPI.yoloDetection.validateDataset(datasetPath);
      return {
        valid: result.valid,
        trainImages: result.train_images,
        valImages: result.val_images,
        trainLabels: result.train_labels,
        valLabels: result.val_labels,
        errors: result.errors,
        warnings: result.warnings,
      };
    } catch (error) {
      console.error('Failed to validate dataset:', error);
      return {
        valid: false,
        trainImages: 0,
        valImages: 0,
        trainLabels: 0,
        valLabels: 0,
        errors: [error instanceof Error ? error.message : 'Validation failed'],
        warnings: [],
      };
    }
  }

  /**
   * Start model training.
   *
   * @param datasetPath Path to the dataset directory
   * @param config Training configuration
   * @param onProgress Callback for progress updates
   * @param modelName Optional custom model name
   * @returns Job ID for tracking
   */
  async startTraining(
    datasetPath: string,
    config: DetectionTrainingConfig = DEFAULT_DETECTION_TRAINING_CONFIG,
    onProgress: (progress: DetectionTrainingProgress) => void,
    modelName?: string
  ): Promise<string> {
    // Stop any existing training
    await this.stopTraining();

    try {
      // Start training
      const result = await window.electronAPI.yoloDetection.startTraining(
        datasetPath,
        config,
        modelName
      );

      if (!result.jobId) {
        throw new Error('Failed to start training - no job ID returned');
      }

      this.activeJobId = result.jobId;

      // Subscribe to progress updates
      this.cleanupFunction = window.electronAPI.yoloDetection.subscribeProgress(
        result.jobId,
        (rawProgress) => {
          // Map to DetectionTrainingProgress type
          const progress: DetectionTrainingProgress = {
            status: rawProgress.status as DetectionTrainingProgress['status'],
            epoch: rawProgress.epoch,
            totalEpochs: rawProgress.totalEpochs,
            loss: rawProgress.loss,
            boxLoss: rawProgress.boxLoss,
            clsLoss: rawProgress.clsLoss,
            dflLoss: rawProgress.dflLoss,
            metrics: rawProgress.metrics,
            eta: rawProgress.eta,
            message: rawProgress.message,
          };
          onProgress(progress);
        },
        (error) => {
          console.error('Detection training progress error:', error);
          onProgress({
            status: 'failed',
            epoch: 0,
            totalEpochs: 0,
            loss: 0,
            boxLoss: 0,
            clsLoss: 0,
            dflLoss: 0,
            metrics: {},
            eta: '',
            message: error,
          });
        },
        () => {
          // Training complete
          this.activeJobId = null;
          this.cleanupFunction = null;
        }
      );

      return result.jobId;
    } catch (error) {
      console.error('Failed to start detection training:', error);
      throw error;
    }
  }

  /**
   * Stop the current training job.
   */
  async stopTraining(): Promise<void> {
    // Cleanup progress subscription
    if (this.cleanupFunction) {
      this.cleanupFunction();
      this.cleanupFunction = null;
    }

    // Stop the training job
    if (this.activeJobId) {
      try {
        await window.electronAPI.yoloDetection.stopTraining(this.activeJobId);
      } catch (error) {
        console.error('Failed to stop training:', error);
      }
      this.activeJobId = null;
    }
  }

  /**
   * Check if training is currently running.
   */
  isTraining(): boolean {
    return this.activeJobId !== null;
  }

  /**
   * Get the active job ID.
   */
  getActiveJobId(): string | null {
    return this.activeJobId;
  }

  /**
   * List all trained models.
   * Uses retry logic to handle server startup race condition.
   */
  async listModels(): Promise<DetectionModelInfo[]> {
    try {
      const result = await withRetry(
        () => window.electronAPI.yoloDetection.listModels(),
        {
          maxRetries: 8,
          initialDelayMs: 500,
          maxDelayMs: 3000,
          shouldRetry: isConnectionError,
        }
      );
      return result.models.map((m) => ({
        id: m.id,
        name: m.name,
        path: m.path,
        createdAt: m.createdAt,
        epochsTrained: m.epochsTrained,
        imgSize: m.imgSize,
        metrics: m.metrics,
      }));
    } catch (error) {
      console.error('Failed to list detection models:', error);
      return [];
    }
  }

  /**
   * Load a model for inference.
   * Uses retry logic to handle server startup race condition.
   *
   * @param modelPath Path to the model file
   * @returns True if loaded successfully
   */
  async loadModel(modelPath: string): Promise<boolean> {
    try {
      const result = await withRetry(
        () => window.electronAPI.yoloDetection.loadModel(modelPath),
        {
          maxRetries: 5,
          initialDelayMs: 500,
          maxDelayMs: 3000,
          shouldRetry: isConnectionError,
        }
      );
      return result.success;
    } catch (error) {
      console.error('Failed to load detection model:', error);
      return false;
    }
  }

  /**
   * Run detection prediction on a full image.
   * Uses retry logic to handle server connection issues.
   *
   * @param imageBase64 Base64-encoded image data
   * @param confidenceThreshold Minimum confidence threshold (default: 0.5)
   * @returns Array of detection predictions
   */
  async predict(
    imageBase64: string,
    confidenceThreshold: number = 0.5
  ): Promise<DetectionPrediction[]> {
    try {
      const result = await withRetry(
        () => window.electronAPI.yoloDetection.predict(imageBase64, confidenceThreshold),
        {
          maxRetries: 3,
          initialDelayMs: 500,
          maxDelayMs: 2000,
          shouldRetry: isConnectionError,
        }
      );

      if (!result.success) {
        return [];
      }

      return result.detections.map((d) => ({
        x: d.x,
        y: d.y,
        width: d.width,
        height: d.height,
        confidence: d.confidence,
        classId: d.classId,
        className: d.className,
      }));
    } catch (error) {
      console.error('Detection prediction failed:', error);
      return [];
    }
  }

  /**
   * Show a save dialog for ONNX export.
   *
   * @param defaultFileName Default file name for the export
   * @returns Selected file path or null if canceled
   */
  async showExportDialog(defaultFileName: string): Promise<string | null> {
    try {
      const result = await window.electronAPI.yoloDetection.showExportDialog(defaultFileName);
      return result.canceled ? null : result.filePath || null;
    } catch (error) {
      console.error('Failed to show export dialog:', error);
      return null;
    }
  }

  /**
   * Export a model to ONNX format.
   *
   * @param modelPath Path to the source model
   * @param outputPath Path for the output ONNX file
   * @returns Path to exported file or null if failed
   */
  async exportONNX(modelPath: string, outputPath: string): Promise<string | null> {
    try {
      const result = await window.electronAPI.yoloDetection.exportONNX(
        modelPath,
        outputPath
      );
      return result.success ? result.outputPath || outputPath : null;
    } catch (error) {
      console.error('ONNX export failed:', error);
      return null;
    }
  }

  /**
   * Delete a trained model.
   *
   * @param modelId Model ID to delete
   * @returns True if deleted successfully
   */
  async deleteModel(modelId: string): Promise<boolean> {
    try {
      const result = await window.electronAPI.yoloDetection.deleteModel(modelId);
      return result.success;
    } catch (error) {
      console.error('Failed to delete detection model:', error);
      return false;
    }
  }

  /**
   * Get models that can be resumed (have last.pt and incomplete training).
   * Uses retry logic to handle server startup race condition.
   *
   * @returns List of resumable models with progress info
   */
  async getResumableModels(): Promise<DetectionModelInfo[]> {
    try {
      const result = await withRetry(
        () => window.electronAPI.yoloDetection.getResumableModels(),
        {
          maxRetries: 5,
          initialDelayMs: 500,
          maxDelayMs: 3000,
          shouldRetry: isConnectionError,
        }
      );
      return result.models.map((m) => ({
        id: m.id,
        name: m.name,
        path: m.path,
        createdAt: m.createdAt,
        epochsTrained: m.epochsTrained,
        imgSize: m.imgSize,
        metrics: m.metrics,
        epochsCompleted: m.epochsCompleted,
        totalEpochs: m.totalEpochs,
        canResume: m.canResume,
      }));
    } catch (error) {
      console.error('Failed to get resumable models:', error);
      return [];
    }
  }

  /**
   * Write dataset files to a temporary directory.
   *
   * @param files Array of files to write
   * @returns Dataset path or error
   */
  async writeDatasetToTemp(
    files: Array<{ path: string; content: ArrayBuffer | string }>
  ): Promise<{ success: boolean; datasetPath?: string; error?: string }> {
    try {
      return await window.electronAPI.yoloDetection.writeDatasetToTemp(files);
    } catch (error) {
      console.error('Failed to write dataset to temp:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to write dataset',
      };
    }
  }
}

// Export singleton instance
export const yoloDetectionService = YOLODetectionService.getInstance();
