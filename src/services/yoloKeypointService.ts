/**
 * YOLO Keypoint Training Service
 *
 * Provides a TypeScript interface for YOLO11-pose model training
 * and inference for follicle origin detection.
 *
 * This service uses the platform adapter to work in both Electron and Web modes.
 */

import {
  TrainingConfig,
  TrainingProgress,
  KeypointPrediction,
  ModelInfo,
  DatasetValidation,
  YoloKeypointStatus,
  YoloDependenciesInfo,
  DEFAULT_TRAINING_CONFIG,
} from '../types';
import { getPlatform } from '../platform';

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
 * Service for managing YOLO keypoint model training and inference.
 *
 * Singleton pattern - use YOLOKeypointService.getInstance() to get the instance.
 */
export class YOLOKeypointService {
  private static instance: YOLOKeypointService | null = null;
  private activeJobId: string | null = null;
  private cleanupFunction: (() => void) | null = null;

  // Cache for dependencies check - only check once per application session
  private cachedDependencies: YoloDependenciesInfo | null = null;
  private dependenciesCheckInProgress: Promise<YoloDependenciesInfo> | null = null;

  private constructor() {}

  /**
   * Get the singleton instance of the service.
   */
  public static getInstance(): YOLOKeypointService {
    if (!YOLOKeypointService.instance) {
      YOLOKeypointService.instance = new YOLOKeypointService();
    }
    return YOLOKeypointService.instance;
  }

  /**
   * Get the YOLO keypoint service status.
   * Uses retry logic to handle server startup race condition.
   */
  async getStatus(): Promise<YoloKeypointStatus> {
    try {
      return await withRetry(
        () => getPlatform().yoloKeypoint.getStatus(),
        {
          maxRetries: 8,
          initialDelayMs: 500,
          maxDelayMs: 3000,
          shouldRetry: isConnectionError,
        }
      );
    } catch (error) {
      console.error('Failed to get YOLO keypoint status:', error);
      return {
        available: false,
        sseAvailable: false,
        activeTrainingJobs: 0,
      };
    }
  }

  /**
   * Check if YOLO training dependencies are installed.
   * These are large packages (ultralytics, onnx, etc.) that are installed on-demand.
   * Uses retry logic to handle server startup race condition.
   *
   * Results are cached for the application session - dependencies only need to be
   * checked once since they don't change during runtime (except via installDependencies).
   */
  async checkDependencies(): Promise<YoloDependenciesInfo> {
    // Return cached result if available
    if (this.cachedDependencies !== null) {
      return this.cachedDependencies;
    }

    // If a check is already in progress, wait for it
    if (this.dependenciesCheckInProgress !== null) {
      return this.dependenciesCheckInProgress;
    }

    // Start the check and cache the promise to prevent duplicate requests
    this.dependenciesCheckInProgress = this.fetchDependencies();

    try {
      const result = await this.dependenciesCheckInProgress;
      this.cachedDependencies = result;
      return result;
    } finally {
      this.dependenciesCheckInProgress = null;
    }
  }

  /**
   * Internal method to actually fetch dependencies from the server.
   */
  private async fetchDependencies(): Promise<YoloDependenciesInfo> {
    try {
      return await withRetry(
        () => getPlatform().yoloKeypoint.checkDependencies(),
        {
          maxRetries: 8,
          initialDelayMs: 500,
          maxDelayMs: 3000,
          shouldRetry: isConnectionError,
        }
      );
    } catch (error) {
      console.error('Failed to check YOLO dependencies:', error);
      return {
        installed: false,
        missing: ['ultralytics', 'sse-starlette', 'onnx', 'onnxruntime'],
        estimatedSize: '~2GB',
      };
    }
  }

  /**
   * Invalidate the cached dependencies, forcing a re-check on next call.
   * Called after installing dependencies.
   */
  invalidateDependenciesCache(): void {
    this.cachedDependencies = null;
  }

  /**
   * Install YOLO training dependencies.
   *
   * @param onProgress Optional callback for progress updates
   * @returns Success status and optional error message
   */
  async installDependencies(
    onProgress?: (message: string, percent?: number) => void
  ): Promise<{ success: boolean; error?: string }> {
    const platform = getPlatform();

    // Set up progress listener if callback provided
    let cleanup: (() => void) | undefined;
    if (onProgress) {
      cleanup = platform.yoloKeypoint.onInstallProgress((data) => {
        onProgress(data.message, data.percent);
      });
    }

    try {
      const result = await platform.yoloKeypoint.installDependencies();
      // Invalidate cache after successful installation so next check will see updated state
      if (result.success) {
        this.invalidateDependenciesCache();
      }
      return result;
    } catch (error) {
      console.error('Failed to install YOLO dependencies:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Installation failed',
      };
    } finally {
      cleanup?.();
    }
  }

  /**
   * Upgrade PyTorch to CUDA version for GPU training.
   * This is for users who already have YOLO dependencies installed but with CPU-only PyTorch.
   *
   * @param onProgress Optional callback for progress updates
   * @returns Success status and optional error message
   */
  async upgradeToCUDA(
    onProgress?: (message: string, percent?: number) => void
  ): Promise<{ success: boolean; error?: string }> {
    const platform = getPlatform();

    // Set up progress listener if callback provided
    let cleanup: (() => void) | undefined;
    if (onProgress) {
      cleanup = platform.yoloKeypoint.onInstallProgress((data) => {
        onProgress(data.message, data.percent);
      });
    }

    try {
      const result = await platform.yoloKeypoint.upgradeToCUDA();
      // Invalidate cache after successful upgrade so next check will see updated state
      if (result.success) {
        this.invalidateDependenciesCache();
      }
      return result;
    } catch (error) {
      console.error('Failed to upgrade PyTorch to CUDA:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Upgrade failed',
      };
    } finally {
      cleanup?.();
    }
  }

  /**
   * Validate a YOLO keypoint dataset.
   *
   * @param datasetPath Path to the dataset directory
   * @returns Validation result with errors and warnings
   */
  async validateDataset(datasetPath: string): Promise<DatasetValidation> {
    try {
      const result = await getPlatform().yoloKeypoint.validateDataset(datasetPath);
      return {
        valid: result.valid,
        trainImages: result.trainImages,
        valImages: result.valImages,
        trainLabels: result.trainLabels,
        valLabels: result.valLabels,
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
    config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
    onProgress: (progress: TrainingProgress) => void,
    modelName?: string
  ): Promise<string> {
    const platform = getPlatform();

    // Stop any existing training
    await this.stopTraining();

    try {
      // Start training
      const result = await platform.yoloKeypoint.startTraining(
        datasetPath,
        config,
        modelName
      );

      if (!result.jobId) {
        throw new Error('Failed to start training - no job ID returned');
      }

      this.activeJobId = result.jobId;

      // Subscribe to progress updates
      this.cleanupFunction = platform.yoloKeypoint.subscribeProgress(
        result.jobId,
        (rawProgress) => {
          // Map to TrainingProgress type
          const progress: TrainingProgress = {
            status: rawProgress.status as TrainingProgress['status'],
            epoch: rawProgress.epoch,
            totalEpochs: rawProgress.totalEpochs,
            loss: rawProgress.loss,
            boxLoss: rawProgress.boxLoss,
            poseLoss: rawProgress.poseLoss,
            kobjLoss: rawProgress.kobjLoss,
            metrics: rawProgress.metrics,
            eta: rawProgress.eta,
            message: rawProgress.message,
          };
          onProgress(progress);
        },
        (error) => {
          console.error('Training progress error:', error);
          onProgress({
            status: 'failed',
            epoch: 0,
            totalEpochs: 0,
            loss: 0,
            boxLoss: 0,
            poseLoss: 0,
            kobjLoss: 0,
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
      console.error('Failed to start training:', error);
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
        await getPlatform().yoloKeypoint.stopTraining(this.activeJobId);
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
  async listModels(): Promise<ModelInfo[]> {
    try {
      const result = await withRetry(
        () => getPlatform().yoloKeypoint.listModels(),
        {
          maxRetries: 8,
          initialDelayMs: 500,
          maxDelayMs: 3000,
          shouldRetry: isConnectionError,
        }
      );

      // Log debug info for diagnosing path issues in production
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const debugInfo = (result as any)._debug;
      if (debugInfo) {
        console.log('[KeypointService] Models directory info:', debugInfo);
      }

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
      console.error('Failed to list models:', error);
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
        () => getPlatform().yoloKeypoint.loadModel(modelPath),
        {
          maxRetries: 5,
          initialDelayMs: 500,
          maxDelayMs: 3000,
          shouldRetry: isConnectionError,
        }
      );
      return result.success;
    } catch (error) {
      console.error('Failed to load model:', error);
      return false;
    }
  }

  /**
   * Run keypoint prediction on a cropped follicle image.
   * Uses retry logic to handle server connection issues.
   *
   * @param imageBase64 Base64-encoded image data
   * @returns Keypoint prediction or null if failed
   */
  async predict(imageBase64: string): Promise<KeypointPrediction | null> {
    try {
      const result = await withRetry(
        () => getPlatform().yoloKeypoint.predict(imageBase64),
        {
          maxRetries: 3,
          initialDelayMs: 500,
          maxDelayMs: 2000,
          shouldRetry: isConnectionError,
        }
      );

      if (!result.success || !result.prediction) {
        return null;
      }

      return {
        origin: result.prediction.origin,
        directionEndpoint: result.prediction.directionEndpoint,
        confidence: result.prediction.confidence,
      };
    } catch (error) {
      console.error('Prediction failed:', error);
      return null;
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
      const result = await getPlatform().yoloKeypoint.showExportDialog(defaultFileName);
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
      const result = await getPlatform().yoloKeypoint.exportONNX(
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
      const result = await getPlatform().yoloKeypoint.deleteModel(modelId);
      return result.success;
    } catch (error) {
      console.error('Failed to delete model:', error);
      return false;
    }
  }

  /**
   * Check if TensorRT is available on this system for keypoint inference.
   *
   * TensorRT provides GPU-optimized inference for faster keypoint prediction
   * on NVIDIA GPUs.
   *
   * @returns TensorRT availability status
   */
  async checkTensorRTAvailable(): Promise<{ available: boolean; version: string | null }> {
    try {
      return await getPlatform().yoloKeypoint.checkTensorRTAvailable();
    } catch (error) {
      console.error('Failed to check TensorRT availability:', error);
      return { available: false, version: null };
    }
  }

  /**
   * Export a PyTorch model to TensorRT engine format.
   *
   * TensorRT provides GPU-optimized inference for faster keypoint prediction.
   * The exported .engine file is GPU-architecture specific and not
   * portable between different GPU types.
   *
   * @param modelPath Path to the source .pt model
   * @param outputPath Optional path for output .engine file
   * @param half Use FP16 precision (default: true, recommended for consumer GPUs)
   * @param imgsz Input image size for the engine (default: 640)
   * @returns Export result with engine path or error
   */
  async exportToTensorRT(
    modelPath: string,
    outputPath?: string,
    half: boolean = true,
    imgsz: number = 640
  ): Promise<{ success: boolean; engine_path?: string; error?: string }> {
    try {
      return await getPlatform().yoloKeypoint.exportToTensorRT(
        modelPath,
        outputPath,
        half,
        imgsz
      );
    } catch (error) {
      console.error('TensorRT export failed:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'TensorRT export failed',
      };
    }
  }
}

// Export singleton instance
export const yoloKeypointService = YOLOKeypointService.getInstance();
