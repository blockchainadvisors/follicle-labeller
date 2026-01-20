/**
 * SAM 2 (Segment Anything Model 2) Service Client
 *
 * This service communicates with the Python SAM 2 backend server to provide
 * AI-powered segmentation capabilities. It supports:
 * - Point-based segmentation (click to segment)
 * - Auto-detection with grid-based prompting
 * - Session management for image contexts
 */

import type { Point, DetectedBlob } from '../types';

export interface SAMServerConfig {
  host: string;
  port: number;
}

export interface SAMSegmentResult {
  masks: string[];       // Base64-encoded mask images
  scores: number[];      // Confidence scores
  bboxes: Array<{
    x: number;
    y: number;
    width: number;
    height: number;
  }>;
}

export interface SAMDetection {
  x: number;
  y: number;
  width: number;
  height: number;
  area: number;
  confidence: number;
  maskBase64: string;
}

export interface SAMAutoDetectOptions {
  gridSize?: number;    // Grid spacing in pixels (default: 32)
  minArea?: number;     // Minimum mask area (default: 100)
  maxOverlap?: number;  // Maximum IoU for NMS (default: 0.5)
}

const DEFAULT_CONFIG: SAMServerConfig = {
  host: '127.0.0.1',
  port: 5555,
};

/**
 * SAM 2 Service for AI-powered segmentation.
 */
export class SAMService {
  private config: SAMServerConfig;
  private currentSessionId: string | null = null;

  constructor(config: Partial<SAMServerConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Get the base URL for the SAM server.
   */
  private get baseUrl(): string {
    return `http://${this.config.host}:${this.config.port}`;
  }

  /**
   * Check if the SAM server is available and model is loaded.
   */
  async isAvailable(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/health`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });

      if (!response.ok) return false;

      const data = await response.json();
      return data.status === 'ok' && data.model_loaded === true;
    } catch {
      return false;
    }
  }

  /**
   * Check if the SAM server is running (may not have model loaded).
   */
  async isServerRunning(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/health`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  /**
   * Set an image for a new session.
   * This uploads the image to the server and returns a session ID.
   *
   * @param imageData - Image data as ArrayBuffer, Blob, or base64 string
   * @returns Session ID and image dimensions
   */
  async setImage(
    imageData: ArrayBuffer | Blob | string
  ): Promise<{ sessionId: string; width: number; height: number }> {
    // Convert to base64 if needed
    let base64: string;

    if (typeof imageData === 'string') {
      base64 = imageData;
    } else if (imageData instanceof Blob) {
      base64 = await this.blobToBase64(imageData);
    } else {
      const blob = new Blob([imageData]);
      base64 = await this.blobToBase64(blob);
    }

    const response = await fetch(`${this.baseUrl}/set-image`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: base64 }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to set image');
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
   * Segment from point prompts.
   * Returns multiple mask options with confidence scores.
   *
   * @param sessionId - Session ID from setImage
   * @param points - Array of point coordinates
   * @param labels - Array of labels (1 = foreground, 0 = background)
   * @returns Segmentation results with masks, scores, and bounding boxes
   */
  async segmentFromPoints(
    sessionId: string,
    points: Point[],
    labels: number[]
  ): Promise<SAMSegmentResult> {
    const response = await fetch(`${this.baseUrl}/segment`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sessionId,
        points: points.map(p => ({ x: p.x, y: p.y })),
        labels,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Segmentation failed');
    }

    return response.json();
  }

  /**
   * Auto-detect objects in the image using grid-based prompting.
   * This provides SAM-based automatic detection similar to the Python version.
   *
   * @param sessionId - Session ID from setImage
   * @param options - Auto-detection options
   * @returns Array of detected objects
   */
  async autoDetect(
    sessionId: string,
    options: SAMAutoDetectOptions = {}
  ): Promise<SAMDetection[]> {
    const response = await fetch(`${this.baseUrl}/auto-detect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sessionId,
        gridSize: options.gridSize ?? 32,
        minArea: options.minArea ?? 100,
        maxOverlap: options.maxOverlap ?? 0.5,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Auto-detection failed');
    }

    const data = await response.json();
    return data.detections;
  }

  /**
   * Convert SAM detections to DetectedBlob format.
   */
  detectionsToBlobs(detections: SAMDetection[]): DetectedBlob[] {
    return detections.map(det => ({
      x: det.x,
      y: det.y,
      width: det.width,
      height: det.height,
      area: det.area,
      aspectRatio: det.width / det.height,
      confidence: det.confidence,
    }));
  }

  /**
   * Close a session and free server resources.
   */
  async closeSession(sessionId: string): Promise<void> {
    try {
      await fetch(`${this.baseUrl}/close-session`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sessionId }),
      });

      if (this.currentSessionId === sessionId) {
        this.currentSessionId = null;
      }
    } catch {
      // Ignore errors on close
    }
  }

  /**
   * Shutdown the SAM server.
   */
  async shutdown(): Promise<void> {
    try {
      await fetch(`${this.baseUrl}/shutdown`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
    } catch {
      // Ignore errors (server may close before responding)
    }
  }

  /**
   * Get the current session ID.
   */
  getCurrentSessionId(): string | null {
    return this.currentSessionId;
  }

  /**
   * Convert a Blob to base64 string.
   */
  private blobToBase64(blob: Blob): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        const result = reader.result as string;
        // Remove data URL prefix if present
        const base64 = result.includes(',')
          ? result.split(',')[1]
          : result;
        resolve(base64);
      };
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }
}

// Singleton instance
let samServiceInstance: SAMService | null = null;

/**
 * Get the SAM service singleton.
 */
export function getSAMService(config?: Partial<SAMServerConfig>): SAMService {
  if (!samServiceInstance) {
    samServiceInstance = new SAMService(config);
  }
  return samServiceInstance;
}

/**
 * Reset the SAM service singleton (useful for testing or reconfiguration).
 */
export function resetSAMService(): void {
  samServiceInstance = null;
}
