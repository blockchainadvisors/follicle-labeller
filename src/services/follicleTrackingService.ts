/**
 * Follicle Tracking Service
 *
 * Provides cross-image follicle tracking using homography-based
 * feature matching with model.track() fallback.
 *
 * Uses the platform adapter to work in both Electron and Web modes.
 */

import { TrackAcrossImagesResult, TrackPrepareResult, TrackMatchSingleResult } from '../types';
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
    maxRetries = 3,
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

      if (attempt >= maxRetries || !shouldRetry(error)) {
        throw error;
      }

      console.log(`Retry attempt ${attempt + 1}/${maxRetries} after ${delay}ms...`);
      await sleep(delay);
      delay = Math.min(delay * 2, maxDelayMs);
    }
  }

  throw lastError;
}

/**
 * Check if an error is a connection error.
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
 * Service for tracking follicles across multiple images.
 *
 * Singleton pattern - use FollicleTrackingService.getInstance().
 */
export class FollicleTrackingService {
  private static instance: FollicleTrackingService | null = null;

  private constructor() {}

  public static getInstance(): FollicleTrackingService {
    if (!FollicleTrackingService.instance) {
      FollicleTrackingService.instance = new FollicleTrackingService();
    }
    return FollicleTrackingService.instance;
  }

  /**
   * Track follicles across two images from different angles.
   *
   * @param sourceImageBase64 Base64-encoded source image
   * @param targetImageBase64 Base64-encoded target image
   * @param confidenceThreshold Detection confidence threshold (default: 0.5)
   * @param matchDistanceThreshold Max pixel distance for matching after homography (default: 50)
   * @param method Matching method: 'auto', 'homography', or 'track' (default: 'auto')
   * @returns Tracking result with detections and matches
   */
  async trackAcrossImages(
    sourceImageBase64: string,
    targetImageBase64: string,
    confidenceThreshold: number = 0.5,
    matchDistanceThreshold: number = 50.0,
    method: 'auto' | 'homography' | 'track' = 'auto'
  ): Promise<TrackAcrossImagesResult> {
    try {
      return await withRetry(
        () =>
          getPlatform().yoloDetection.trackAcrossImages(
            sourceImageBase64,
            targetImageBase64,
            confidenceThreshold,
            matchDistanceThreshold,
            method
          ),
        {
          maxRetries: 3,
          initialDelayMs: 500,
          maxDelayMs: 2000,
          shouldRetry: isConnectionError,
        }
      );
    } catch (error) {
      console.error('Cross-image tracking failed:', error);
      return {
        success: false,
        sourceDetections: [],
        targetDetections: [],
        matches: [],
        method,
        error: error instanceof Error ? error.message : 'Tracking failed',
      };
    }
  }

  async trackPrepare(
    sourceImageBase64: string,
    targetImageBase64: string,
    confidenceThreshold: number = 0.5,
    matchDistanceThreshold: number = 50.0
  ): Promise<TrackPrepareResult> {
    try {
      return await withRetry(
        () =>
          getPlatform().yoloDetection.trackPrepare(
            sourceImageBase64, targetImageBase64,
            confidenceThreshold, matchDistanceThreshold
          ),
        { maxRetries: 3, initialDelayMs: 500, maxDelayMs: 2000, shouldRetry: isConnectionError }
      );
    } catch (error) {
      console.error('Track prepare failed:', error);
      return { success: false, sessionId: '', error: error instanceof Error ? error.message : 'Prepare failed' };
    }
  }

  async trackMatchSingle(
    sessionId: string,
    sourceBbox: { x: number; y: number; width: number; height: number }
  ): Promise<TrackMatchSingleResult> {
    try {
      return await withRetry(
        () => getPlatform().yoloDetection.trackMatchSingle(sessionId, sourceBbox),
        { maxRetries: 3, initialDelayMs: 500, maxDelayMs: 2000, shouldRetry: isConnectionError }
      );
    } catch (error) {
      console.error('Track match single failed:', error);
      return { success: false, match: null, error: error instanceof Error ? error.message : 'Match failed' };
    }
  }

  async templatePrepare(
    targetFilePath: string,
  ): Promise<TrackPrepareResult> {
    try {
      return await withRetry(
        () => getPlatform().yoloDetection.templatePrepare(targetFilePath),
        { maxRetries: 3, initialDelayMs: 500, maxDelayMs: 2000, shouldRetry: isConnectionError }
      );
    } catch (error) {
      console.error('Template prepare failed:', error);
      return { success: false, sessionId: '', error: error instanceof Error ? error.message : 'Prepare failed' };
    }
  }

  async templateMatchSingle(
    sessionId: string,
    sourcePatchData: string,
    follicleOffsetX: number,
    follicleOffsetY: number,
    follicleWidth: number,
    follicleHeight: number,
  ): Promise<TrackMatchSingleResult> {
    try {
      return await withRetry(
        () => getPlatform().yoloDetection.templateMatchSingle(
          sessionId, sourcePatchData, follicleOffsetX, follicleOffsetY, follicleWidth, follicleHeight
        ),
        { maxRetries: 3, initialDelayMs: 500, maxDelayMs: 2000, shouldRetry: isConnectionError }
      );
    } catch (error) {
      console.error('Template match single failed:', error);
      return { success: false, match: null, error: error instanceof Error ? error.message : 'Match failed' };
    }
  }
}

// Export singleton instance
export const follicleTrackingService = FollicleTrackingService.getInstance();
