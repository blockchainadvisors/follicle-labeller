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
      await yoloKeypointService.loadModel(keypointModel);
    } else {
      // Try to auto-load the first available model if none specified
      const models = await yoloKeypointService.listModels();
      if (models.length > 0) {
        const loaded = await yoloKeypointService.loadModel(models[0].path);
        if (!loaded) {
          console.warn('Failed to auto-load keypoint model');
          return {
            detections: blobResult.detections,
            origins,
            count: blobResult.count,
          };
        }
      } else {
        console.warn('No keypoint models available for inference');
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
        if (!prediction || prediction.confidence < 0.3) continue;

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
  private async getCropBase64(
    sessionId: string,
    detection: BlobDetection,
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
   * @param sessionId - Session ID from setImage
   * @param rectangles - Array of rectangle annotations to predict origins for
   * @returns Map of annotation ID to predicted FollicleOrigin
   */
  async predictOriginsForRectangles(
    sessionId: string,
    rectangles: Array<{ id: string; x: number; y: number; width: number; height: number }>,
  ): Promise<Map<string, FollicleOrigin>> {
    const origins = new Map<string, FollicleOrigin>();

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

    const loaded = await yoloKeypointService.loadModel(models[0].path);
    if (!loaded) {
      console.warn('Failed to load keypoint model');
      return origins;
    }

    // For each rectangle, crop and predict keypoints
    for (const rect of rectangles) {
      try {
        // Crop the region and get base64
        const cropBase64 = await this.getCropBase64(sessionId, {
          x: rect.x,
          y: rect.y,
          width: rect.width,
          height: rect.height,
          confidence: 1,
          method: 'blob',
        });
        if (!cropBase64) continue;

        // Run keypoint prediction
        const prediction = await yoloKeypointService.predict(cropBase64);
        if (!prediction || prediction.confidence < 0.3) continue;

        // Transform keypoints from normalized crop coords to image coords
        const origin = this.transformKeypointPrediction(prediction, {
          x: rect.x,
          y: rect.y,
          width: rect.width,
          height: rect.height,
          confidence: 1,
          method: 'blob',
        });
        origins.set(rect.id, origin);
      } catch (error) {
        console.warn(`Keypoint prediction failed for rectangle ${rect.id}:`, error);
      }
    }

    return origins;
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
