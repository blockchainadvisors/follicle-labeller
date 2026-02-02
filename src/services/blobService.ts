/**
 * BLOB Detection Service Client
 *
 * This service communicates with the Python BLOB detection backend server to provide
 * OpenCV-based follicle detection capabilities. It supports:
 * - Session management for image contexts
 * - Annotation sync for size learning
 * - Auto-detection using SimpleBlobDetector + contour fallback
 *
 * Requires minimum 3 user annotations before auto-detection can run.
 */

import type { DetectedBlob, GPUInfo, FollicleOrigin, KeypointPrediction } from "../types";
import { yoloKeypointService } from "./yoloKeypointService";

export interface BlobServerConfig {
  host: string;
  port: number;
}

export interface BlobDetection {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  method: "blob" | "contour";
}

export interface AnnotationBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

const DEFAULT_CONFIG: BlobServerConfig = {
  host: "127.0.0.1",
  port: 5555,
};

/**
 * BLOB Detection Service for OpenCV-based follicle detection.
 */
export class BlobService {
  private config: BlobServerConfig;
  private currentSessionId: string | null = null;

  constructor(config: Partial<BlobServerConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Get the base URL for the BLOB server.
   */
  private get baseUrl(): string {
    return `http://${this.config.host}:${this.config.port}`;
  }

  /**
   * Check if the BLOB server is available.
   * Uses IPC to avoid browser console errors during startup.
   */
  async isAvailable(): Promise<boolean> {
    try {
      // Use IPC to check availability - this avoids ERR_CONNECTION_REFUSED in browser console
      return await window.electronAPI.blob.isAvailable();
    } catch {
      return false;
    }
  }

  /**
   * Check if the BLOB server is running.
   * Uses IPC to avoid browser console errors during startup.
   */
  async isServerRunning(): Promise<boolean> {
    try {
      // Use IPC to check server status - this avoids ERR_CONNECTION_REFUSED in browser console
      return await window.electronAPI.blob.isAvailable();
    } catch {
      return false;
    }
  }

  /**
   * Get the current session ID.
   */
  getSessionId(): string | null {
    return this.currentSessionId;
  }

  /**
   * Set an image for a new session.
   * This uploads the image to the server and returns a session ID.
   *
   * @param imageData - Image data as ArrayBuffer, Blob, or base64 string
   * @returns Session ID and image dimensions
   */
  async setImage(
    imageData: ArrayBuffer | Blob | string,
  ): Promise<{ sessionId: string; width: number; height: number }> {
    // Convert to base64 if needed
    let base64: string;

    if (typeof imageData === "string") {
      base64 = imageData;
    } else if (imageData instanceof Blob) {
      base64 = await this.blobToBase64(imageData);
    } else {
      const blob = new Blob([imageData]);
      base64 = await this.blobToBase64(blob);
    }

    const response = await fetch(`${this.baseUrl}/set-image`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: base64 }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || "Failed to set image");
    }

    const data = await response.json();
    this.currentSessionId = data.sessionId;

    return {
      sessionId: data.sessionId,
      width: data.width,
      height: data.height,
    };
  }

  /**
   * Add a single annotation for size learning.
   *
   * @param sessionId - Session ID from setImage
   * @param annotation - Annotation bounding box
   * @returns Annotation count and whether detection is available
   */
  async addAnnotation(
    sessionId: string,
    annotation: AnnotationBox,
  ): Promise<{
    annotationCount: number;
    canDetect: boolean;
    minRequired: number;
  }> {
    const response = await fetch(`${this.baseUrl}/add-annotation`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sessionId,
        x: annotation.x,
        y: annotation.y,
        width: annotation.width,
        height: annotation.height,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || "Failed to add annotation");
    }

    return response.json();
  }

  /**
   * Sync all annotations from frontend to backend.
   * This replaces all annotations in the session.
   *
   * @param sessionId - Session ID from setImage
   * @param annotations - Array of annotation bounding boxes
   * @returns Annotation count and whether detection is available
   */
  async syncAnnotations(
    sessionId: string,
    annotations: AnnotationBox[],
  ): Promise<{
    annotationCount: number;
    canDetect: boolean;
    minRequired: number;
  }> {
    const response = await fetch(`${this.baseUrl}/sync-annotations`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sessionId,
        annotations: annotations.map((a) => ({
          x: a.x,
          y: a.y,
          width: a.width,
          height: a.height,
        })),
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || "Failed to sync annotations");
    }

    return response.json();
  }

  /**
   * Get the current annotation count for a session.
   *
   * @param sessionId - Session ID from setImage
   * @returns Annotation count and whether detection is available
   */
  async getAnnotationCount(sessionId: string): Promise<{
    annotationCount: number;
    canDetect: boolean;
    minRequired: number;
  }> {
    const response = await fetch(`${this.baseUrl}/get-annotation-count`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sessionId }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || "Failed to get annotation count");
    }

    return response.json();
  }

  /**
   * Get learned statistics from annotations for the Learn from Selection dialog.
   *
   * @param sessionId - Session ID from setImage
   * @returns Learned stats and whether detection is available
   */
  async getLearnedStats(sessionId: string): Promise<{
    stats: {
      examplesAnalyzed: number;
      minWidth: number;
      maxWidth: number;
      minHeight: number;
      maxHeight: number;
      minAspectRatio: number;
      maxAspectRatio: number;
      meanIntensity: number;
    };
    canDetect: boolean;
    minRequired: number;
  }> {
    const response = await fetch(`${this.baseUrl}/get-learned-stats`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sessionId }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || "Failed to get learned stats");
    }

    return response.json();
  }

  /**
   * Run BLOB detection on the session image.
   * Works with either annotations (for size learning) or manual settings.
   *
   * @param sessionId - Session ID from setImage
   * @param settings - Optional detection settings
   * @returns Array of detected follicles
   */
  async blobDetect(
    sessionId: string,
    settings?: {
      // Manual size settings
      minWidth?: number;
      maxWidth?: number;
      minHeight?: number;
      maxHeight?: number;
      // Learned mode settings
      useLearnedStats?: boolean;
      tolerance?: number;
      // Common settings
      darkBlobs?: boolean;
      useCLAHE?: boolean;
      claheClipLimit?: number;
      claheTileSize?: number;
      // Backend selection
      forceCPU?: boolean;
    },
  ): Promise<{
    detections: BlobDetection[];
    count: number;
    learnedSize: number;
  }> {
    const response = await fetch(`${this.baseUrl}/blob-detect`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sessionId, settings }),
    });

    if (!response.ok) {
      const error = await response.json();
      // Handle both string and object error details
      const detail = error.detail;
      let errorMessage = "Detection failed";
      if (typeof detail === "string") {
        errorMessage = detail;
      } else if (detail && typeof detail === "object") {
        errorMessage = detail.error || "Detection failed";
        // Log additional debug info
        console.error("Detection error details:", detail);
        if (detail.gpu_available === false) {
          errorMessage += " (GPU not available)";
        }
      } else if (error.error) {
        errorMessage = error.error;
      }
      throw new Error(errorMessage);
    }

    return response.json();
  }

  /**
   * Convert BLOB detections to the app's DetectedBlob format.
   *
   * @param detections - Array of BLOB detections from server
   * @returns Array of DetectedBlob objects
   */
  convertToDetectedBlobs(detections: BlobDetection[]): DetectedBlob[] {
    return detections.map((d) => ({
      x: d.x,
      y: d.y,
      width: d.width,
      height: d.height,
      area: d.width * d.height,
      aspectRatio: d.width / d.height,
      confidence: d.confidence,
    }));
  }

  /**
   * Clear a session and free server resources.
   *
   * @param sessionId - Session ID to clear
   */
  async clearSession(sessionId: string): Promise<void> {
    try {
      await fetch(`${this.baseUrl}/clear-session`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessionId }),
      });

      if (this.currentSessionId === sessionId) {
        this.currentSessionId = null;
      }
    } catch {
      // Ignore errors on cleanup
    }
  }

  /**
   * Stop the BLOB server.
   */
  async stopServer(): Promise<void> {
    try {
      await fetch(`${this.baseUrl}/shutdown`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
    } catch {
      // Server is shutting down, connection may fail
    }
    this.currentSessionId = null;
  }

  /**
   * Get GPU backend information.
   *
   * @returns GPU info including active backend and available backends
   */
  async getGPUInfo(): Promise<GPUInfo> {
    try {
      const response = await fetch(`${this.baseUrl}/gpu-info`, {
        method: "GET",
        headers: { "Content-Type": "application/json" },
      });

      if (!response.ok) {
        return {
          activeBackend: "cpu",
          deviceName: "CPU (OpenCV)",
          available: { cuda: false, mps: false },
        };
      }

      const data = await response.json();
      return {
        activeBackend: data.active_backend || "cpu",
        deviceName: data.device_name || "CPU (OpenCV)",
        memoryGB: data.details?.cuda?.memory_gb,
        available: {
          cuda: data.backends?.cuda || false,
          mps: data.backends?.mps || false,
        },
      };
    } catch {
      return {
        activeBackend: "cpu",
        deviceName: "CPU (OpenCV)",
        available: { cuda: false, mps: false },
      };
    }
  }

  /**
   * Convert a Blob to base64 string.
   */
  private blobToBase64(blob: Blob): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const result = reader.result as string;
        // Remove data URL prefix if present
        const base64 = result.includes(",") ? result.split(",")[1] : result;
        resolve(base64);
      };
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }

  /**
   * Run combined blob detection with keypoint prediction.
   * First detects blobs using OpenCV, then runs YOLO keypoint inference on each crop.
   *
   * @param sessionId - Session ID from setImage
   * @param settings - Detection settings
   * @param keypointModel - Optional path to YOLO keypoint model (uses loaded model if not specified)
   * @param useTensorRT - Whether to use TensorRT engine instead of PyTorch model
   * @param confidenceThreshold - Minimum confidence threshold for keypoint predictions (default 0.3)
   * @returns Detections and predicted follicle origins
   */
  async detectWithKeypoints(
    sessionId: string,
    settings?: {
      minWidth?: number;
      maxWidth?: number;
      minHeight?: number;
      maxHeight?: number;
      useLearnedStats?: boolean;
      tolerance?: number;
      darkBlobs?: boolean;
      useCLAHE?: boolean;
      claheClipLimit?: number;
      claheTileSize?: number;
      forceCPU?: boolean;
    },
    keypointModel?: string,
    useTensorRT?: boolean,
    confidenceThreshold: number = 0.3,
  ): Promise<{
    detections: BlobDetection[];
    origins: Map<string, FollicleOrigin>;
    count: number;
  }> {
    // First, run blob detection
    const blobResult = await this.blobDetect(sessionId, settings);
    const origins = new Map<string, FollicleOrigin>();

    // Check if YOLO keypoint service is available
    const keypointStatus = await yoloKeypointService.getStatus();
    if (!keypointStatus.available) {
      // Return detections without keypoint predictions
      return {
        detections: blobResult.detections,
        origins,
        count: blobResult.count,
      };
    }

    // Load model if specified, or auto-load first available model
    if (keypointModel) {
      // If TensorRT requested, try to use .engine file
      let modelToLoad = keypointModel;
      if (useTensorRT && !keypointModel.endsWith('.engine')) {
        const enginePath = keypointModel.replace(/\.pt$/i, '.engine');
        // Check if engine file exists via electron API
        try {
          const engineExists = await window.electronAPI.fileExists(enginePath);
          if (engineExists) {
            modelToLoad = enginePath;
            console.log('Using TensorRT engine for keypoint model:', enginePath);
          } else {
            console.warn('TensorRT requested but engine file not found, falling back to PyTorch:', keypointModel);
          }
        } catch {
          console.warn('Could not check for engine file, using PyTorch model');
        }
      }
      await yoloKeypointService.loadModel(modelToLoad);
    } else {
      // Try to auto-load the first available model if none specified
      const models = await yoloKeypointService.listModels();
      console.log('[Keypoint] Found models:', models.length, models.map(m => ({ id: m.id, name: m.name, path: m.path })));

      if (models.length > 0) {
        let modelPath = models[0].path;
        console.log('[Keypoint] Auto-loading first model:', modelPath);

        // If TensorRT requested, try to use .engine file
        if (useTensorRT && !modelPath.endsWith('.engine')) {
          const enginePath = modelPath.replace(/\.pt$/i, '.engine');
          try {
            const engineExists = await window.electronAPI.fileExists(enginePath);
            if (engineExists) {
              modelPath = enginePath;
              console.log('[Keypoint] Using TensorRT engine:', enginePath);
            } else {
              console.warn('[Keypoint] TensorRT requested but engine file not found, falling back to PyTorch:', models[0].path);
            }
          } catch {
            console.warn('[Keypoint] Could not check for engine file, using PyTorch model');
          }
        }

        console.log('[Keypoint] Loading model from path:', modelPath);
        const loaded = await yoloKeypointService.loadModel(modelPath);
        console.log('[Keypoint] Model load result:', loaded);

        if (!loaded) {
          console.error('[Keypoint] Failed to load model:', modelPath);
          return {
            detections: blobResult.detections,
            origins,
            count: blobResult.count,
          };
        }
      } else {
        console.warn('[Keypoint] No keypoint models available for inference');
        return {
          detections: blobResult.detections,
          origins,
          count: blobResult.count,
        };
      }
    }

    // For each detection, crop and predict keypoints
    for (let i = 0; i < blobResult.detections.length; i++) {
      const detection = blobResult.detections[i];
      const detectionId = `det_${i}_${detection.x}_${detection.y}`;

      try {
        // Crop the detection region and get base64
        const cropBase64 = await this.getCropBase64(sessionId, detection);
        if (!cropBase64) continue;

        // Run keypoint prediction
        const prediction = await yoloKeypointService.predict(cropBase64);
        if (!prediction || prediction.confidence < confidenceThreshold) continue;

        // Transform keypoints from normalized crop coords to image coords
        const origin = this.transformKeypointPrediction(prediction, detection);
        origins.set(detectionId, origin);
      } catch (error) {
        console.warn(`Keypoint prediction failed for detection ${i}:`, error);
      }
    }

    return {
      detections: blobResult.detections,
      origins,
      count: blobResult.count,
    };
  }

  /**
   * Get a cropped region as base64 for keypoint prediction.
   *
   * @param sessionId - Session ID
   * @param detection - Detection bounding box
   * @returns Base64 encoded crop image
   */
  async getCropBase64(
    sessionId: string,
    detection: BlobDetection | { x: number; y: number; width: number; height: number },
  ): Promise<string | null> {
    try {
      const response = await fetch(`${this.baseUrl}/crop-region`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sessionId,
          x: detection.x,
          y: detection.y,
          width: detection.width,
          height: detection.height,
        }),
      });

      if (!response.ok) return null;

      const data = await response.json();
      return data.image;
    } catch {
      return null;
    }
  }

  /**
   * Transform keypoint prediction from normalized crop coordinates to image coordinates.
   *
   * @param prediction - Keypoint prediction in normalized crop space
   * @param detection - Detection bounding box in image space
   * @returns FollicleOrigin in image coordinates
   */
  private transformKeypointPrediction(
    prediction: KeypointPrediction,
    detection: BlobDetection,
  ): FollicleOrigin {
    // Transform origin point from normalized (0-1) to image coordinates
    const originX = detection.x + prediction.origin.x * detection.width;
    const originY = detection.y + prediction.origin.y * detection.height;

    // Transform direction endpoint
    const dirEndX = detection.x + prediction.directionEndpoint.x * detection.width;
    const dirEndY = detection.y + prediction.directionEndpoint.y * detection.height;

    // Calculate direction angle and length
    const dx = dirEndX - originX;
    const dy = dirEndY - originY;
    const directionAngle = Math.atan2(dy, dx);
    const directionLength = Math.sqrt(dx * dx + dy * dy);

    return {
      originPoint: { x: originX, y: originY },
      directionAngle,
      directionLength,
    };
  }

  /**
   * Predict origins for specific rectangle annotations.
   * Only predicts for rectangles that don't already have origins.
   *
   * Uses optimized batch prediction for PyTorch (4.9x faster) and
   * parallel requests for TensorRT (since TensorRT batch doesn't work well).
   *
   * @param sessionId - Session ID from setImage
   * @param rectangles - Array of rectangle annotations to predict origins for
   * @param useTensorRT - Whether to use TensorRT engine instead of PyTorch model
   * @param onProgress - Optional callback for progress updates (current, total)
   * @returns Map of annotation ID to predicted FollicleOrigin
   */
  async predictOriginsForRectangles(
    sessionId: string,
    rectangles: Array<{ id: string; x: number; y: number; width: number; height: number }>,
    useTensorRT?: boolean,
    onProgress?: (current: number, total: number) => void,
    confidenceThreshold: number = 0.3,
  ): Promise<Map<string, FollicleOrigin>> {
    const origins = new Map<string, FollicleOrigin>();

    if (rectangles.length === 0) return origins;

    // Check if YOLO keypoint service is available
    const keypointStatus = await yoloKeypointService.getStatus();
    if (!keypointStatus.available) {
      console.warn('YOLO keypoint service not available');
      return origins;
    }

    // Try to auto-load the first available model if needed
    const models = await yoloKeypointService.listModels();
    if (models.length === 0) {
      console.warn('No keypoint models available for inference');
      return origins;
    }

    // Select model path based on backend preference
    let modelPath = models[0].path;
    let actuallyUsingTensorRT = false;
    if (useTensorRT && !modelPath.endsWith('.engine')) {
      const enginePath = modelPath.replace(/\.pt$/i, '.engine');
      try {
        const engineExists = await window.electronAPI.fileExists(enginePath);
        if (engineExists) {
          modelPath = enginePath;
          actuallyUsingTensorRT = true;
          console.log('Using TensorRT engine for keypoint prediction:', enginePath);
        } else {
          console.warn('TensorRT requested but engine file not found, using PyTorch:', modelPath);
        }
      } catch {
        console.warn('Could not check for engine file, using PyTorch model');
      }
    } else if (modelPath.endsWith('.engine')) {
      actuallyUsingTensorRT = true;
    }

    const loaded = await yoloKeypointService.loadModel(modelPath);
    if (!loaded) {
      console.warn('Failed to load keypoint model');
      return origins;
    }

    // First, get all crops in parallel batches to reduce HTTP overhead
    const crops: Array<{ id: string; rect: typeof rectangles[0]; base64: string | null }> = [];
    const cropConcurrency = 16; // Process 16 crop requests concurrently

    for (let i = 0; i < rectangles.length; i += cropConcurrency) {
      const batch = rectangles.slice(i, i + cropConcurrency);
      const batchResults = await Promise.all(
        batch.map(async (rect) => {
          try {
            const cropBase64 = await this.getCropBase64(sessionId, {
              x: rect.x,
              y: rect.y,
              width: rect.width,
              height: rect.height,
            });
            return { id: rect.id, rect, base64: cropBase64 };
          } catch {
            return { id: rect.id, rect, base64: null };
          }
        })
      );
      crops.push(...batchResults);
      // Report crop progress (first half)
      onProgress?.(Math.floor((i + batch.length) / 2), rectangles.length);
    }

    const validCrops = crops.filter(c => c.base64 !== null);

    // Use batch prediction for both backends
    // PyTorch: batch 64 is optimal (limited by GPU memory efficiency)
    // TensorRT: batch 8 max (limited by engine export with dynamic batching)
    // Send batches in parallel waves to reduce HTTP overhead for both backends
    if (validCrops.length > 1) {
      const batchSize = actuallyUsingTensorRT ? 8 : 64;
      // Process multiple batches concurrently to reduce HTTP overhead
      // TensorRT: 8 concurrent (8x8=64 images per wave)
      // PyTorch: 4 concurrent (4x64=256 images per wave)
      const concurrentBatches = actuallyUsingTensorRT ? 8 : 4;
      let processedCount = 0;

      // Create all batches first
      const allBatches: Array<{ batch: typeof validCrops; startIndex: number }> = [];
      for (let i = 0; i < validCrops.length; i += batchSize) {
        allBatches.push({
          batch: validCrops.slice(i, i + batchSize),
          startIndex: i,
        });
      }

      // Process batches in waves (concurrently within each wave)
      for (let waveStart = 0; waveStart < allBatches.length; waveStart += concurrentBatches) {
        const wave = allBatches.slice(waveStart, waveStart + concurrentBatches);

        // Send all batches in this wave concurrently
        const waveResults = await Promise.all(
          wave.map(async ({ batch }) => {
            const images = batch.map(c => c.base64!);
            try {
              const response = await fetch(`${this.baseUrl}/yolo-keypoint/predict-batch`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ images }),
              });
              if (response.ok) {
                const result = await response.json();
                return { success: true, predictions: result.predictions || [], batch };
              }
              return { success: false, predictions: [], batch };
            } catch {
              return { success: false, predictions: [], batch };
            }
          })
        );

        // Process results from this wave
        for (const { success, predictions, batch } of waveResults) {
          if (success) {
            for (let j = 0; j < batch.length && j < predictions.length; j++) {
              const pred = predictions[j];
              if (pred && pred.confidence >= confidenceThreshold) {
                const crop = batch[j];
                const origin = this.transformKeypointPrediction(
                  {
                    origin: pred.origin,
                    directionEndpoint: pred.directionEndpoint,
                    confidence: pred.confidence,
                  },
                  { x: crop.rect.x, y: crop.rect.y, width: crop.rect.width, height: crop.rect.height, confidence: 1, method: 'blob' }
                );
                origins.set(crop.id, origin);
              }
            }
          } else {
            // Fall back to sequential for failed batch
            for (const crop of batch) {
              try {
                const prediction = await yoloKeypointService.predict(crop.base64!);
                if (prediction && prediction.confidence >= confidenceThreshold) {
                  const origin = this.transformKeypointPrediction(prediction, {
                    x: crop.rect.x, y: crop.rect.y, width: crop.rect.width, height: crop.rect.height,
                    confidence: 1, method: 'blob',
                  });
                  origins.set(crop.id, origin);
                }
              } catch {
                // Skip failed prediction
              }
            }
          }
          processedCount += batch.length;
        }
        onProgress?.(rectangles.length / 2 + processedCount / 2, rectangles.length);
      }
    } else {
      // Single image: Use sequential prediction
      for (let i = 0; i < validCrops.length; i++) {
        const crop = validCrops[i];
        try {
          const prediction = await yoloKeypointService.predict(crop.base64!);
          if (prediction && prediction.confidence >= confidenceThreshold) {
            const origin = this.transformKeypointPrediction(prediction, {
              x: crop.rect.x, y: crop.rect.y, width: crop.rect.width, height: crop.rect.height,
              confidence: 1, method: 'blob',
            });
            origins.set(crop.id, origin);
          }
        } catch (error) {
          console.warn(`Keypoint prediction failed for rectangle ${crop.id}:`, error);
        }
        onProgress?.(rectangles.length / 2 + (i + 1) / 2, rectangles.length);
      }
    }

    console.log(`[BENCHMARK] Keypoint prediction completed: ${origins.size}/${rectangles.length} origins predicted`);

    // Unload the model and free all GPU memory after predictions
    // This completely frees the ~1.5GB TensorRT engine from VRAM
    // The model will auto-reload on next prediction
    try {
      const unloadResult = await this.unloadKeypointModel();
      if (unloadResult.memory_freed_mb && unloadResult.memory_freed_mb > 0) {
        console.log(`[GPU] Unloaded model, freed ${unloadResult.memory_freed_mb}MB after keypoint prediction`);
      }
    } catch (err) {
      console.warn('Failed to unload keypoint model after prediction:', err);
    }

    return origins;
  }

  /**
   * Clear GPU memory used by keypoint prediction.
   * Runs garbage collection and empties CUDA cache.
   */
  async clearKeypointGpuMemory(): Promise<{ success: boolean; memory_freed_mb?: number }> {
    try {
      const response = await fetch(`${this.baseUrl}/yolo-keypoint/clear-gpu-memory`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      if (response.ok) {
        const result = await response.json();
        if (result.memory_freed_mb > 0) {
          console.log(`[GPU] Freed ${result.memory_freed_mb}MB GPU memory after keypoint prediction`);
        }
        return result;
      }
      return { success: false };
    } catch (error) {
      console.warn('GPU memory cleanup request failed:', error);
      return { success: false };
    }
  }

  /**
   * Unload the keypoint model and free all GPU memory.
   * Unlike clearKeypointGpuMemory which only clears cached tensors,
   * this completely removes the model from GPU memory.
   * The model will be automatically reloaded on the next prediction.
   */
  async unloadKeypointModel(): Promise<{ success: boolean; memory_freed_mb?: number; model_was_loaded?: boolean }> {
    try {
      const response = await fetch(`${this.baseUrl}/yolo-keypoint/unload-model`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      if (response.ok) {
        const result = await response.json();
        if (result.memory_freed_mb > 0) {
          console.log(`[GPU] Unloaded keypoint model, freed ${result.memory_freed_mb}MB GPU memory`);
        } else if (result.model_was_loaded === false) {
          console.log('[GPU] No keypoint model was loaded');
        }
        return result;
      }
      return { success: false };
    } catch (error) {
      console.warn('Keypoint model unload request failed:', error);
      return { success: false };
    }
  }

  /**
   * Clear GPU memory used by YOLO detection.
   * Runs garbage collection and empties CUDA cache.
   */
  async clearDetectionGpuMemory(): Promise<{ success: boolean; memory_freed_mb?: number }> {
    try {
      const response = await fetch(`${this.baseUrl}/yolo-detect/clear-gpu-memory`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      if (response.ok) {
        const result = await response.json();
        if (result.memory_freed_mb > 0) {
          console.log(`[GPU] Freed ${result.memory_freed_mb}MB GPU memory after detection`);
        }
        return result;
      }
      return { success: false };
    } catch (error) {
      console.warn('Detection GPU memory cleanup request failed:', error);
      return { success: false };
    }
  }

  /**
   * Clear GPU memory used by blob detection preprocessing (CuPy).
   * Frees all blocks in the CuPy memory pool.
   */
  async clearBlobGpuMemory(): Promise<{ success: boolean; memory_freed_mb?: number }> {
    try {
      const response = await fetch(`${this.baseUrl}/clear-gpu-memory`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      if (response.ok) {
        const result = await response.json();
        if (result.memory_freed_mb > 0) {
          console.log(`[GPU] Freed ${result.memory_freed_mb}MB CuPy GPU memory after blob detection`);
        }
        return result;
      }
      return { success: false };
    } catch (error) {
      console.warn('Blob GPU memory cleanup request failed:', error);
      return { success: false };
    }
  }

  /**
   * Get the singleton instance of BlobService.
   */
  private static instance: BlobService | null = null;

  static getInstance(config?: Partial<BlobServerConfig>): BlobService {
    if (!BlobService.instance) {
      BlobService.instance = new BlobService(config);
    }
    return BlobService.instance;
  }
}

// Export singleton instance
export const blobService = BlobService.getInstance();
