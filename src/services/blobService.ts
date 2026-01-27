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

import type { DetectedBlob, GPUInfo } from "../types";

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
   */
  async isAvailable(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/health`, {
        method: "GET",
        headers: { "Content-Type": "application/json" },
      });

      if (!response.ok) return false;

      const data = await response.json();
      return data.status === "ok";
    } catch {
      return false;
    }
  }

  /**
   * Check if the BLOB server is running.
   */
  async isServerRunning(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/health`, {
        method: "GET",
        headers: { "Content-Type": "application/json" },
      });
      return response.ok;
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
      throw new Error(error.error || "Detection failed");
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
