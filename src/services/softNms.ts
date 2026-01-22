/**
 * Soft-NMS (Non-Maximum Suppression) implementation for blob detection.
 *
 * Unlike traditional hard NMS which completely removes overlapping detections,
 * Soft-NMS gradually decreases the confidence score of overlapping detections
 * based on their IoU (Intersection over Union) with higher-scoring detections.
 *
 * Reference: "Soft-NMS -- Improving Object Detection With One Line of Code"
 * https://arxiv.org/abs/1704.04503
 */

import type { DetectedBlob } from '../types';

/**
 * Soft-NMS configuration options.
 */
export interface SoftNMSOptions {
  /** Gaussian decay sigma (default: 0.5). Higher = more suppression */
  sigma?: number;
  /** Minimum score threshold to keep detection (default: 0.1) */
  scoreThreshold?: number;
  /** IoU threshold for considering overlap (default: 0.3) */
  iouThreshold?: number;
  /** NMS method: 'gaussian' (soft) or 'linear' (default: 'gaussian') */
  method?: 'gaussian' | 'linear';
}

/**
 * Default Soft-NMS options.
 */
export const DEFAULT_SOFT_NMS_OPTIONS: Required<SoftNMSOptions> = {
  sigma: 0.5,
  scoreThreshold: 0.1,
  iouThreshold: 0.3,
  method: 'gaussian',
};

/**
 * Calculate Intersection over Union (IoU) between two bounding boxes.
 */
function calculateIoU(a: DetectedBlob, b: DetectedBlob): number {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.width, b.x + b.width);
  const y2 = Math.min(a.y + a.height, b.y + b.height);

  const intersectionWidth = Math.max(0, x2 - x1);
  const intersectionHeight = Math.max(0, y2 - y1);
  const intersectionArea = intersectionWidth * intersectionHeight;

  if (intersectionArea === 0) {
    return 0;
  }

  const areaA = a.width * a.height;
  const areaB = b.width * b.height;
  const unionArea = areaA + areaB - intersectionArea;

  return intersectionArea / unionArea;
}

/**
 * Apply Soft-NMS to a list of detected blobs.
 *
 * The algorithm:
 * 1. Sort blobs by confidence score (highest first)
 * 2. For each blob M (starting from highest score):
 *    - Keep M in the result
 *    - For all other blobs with lower scores:
 *      - Calculate IoU with M
 *      - If IoU > threshold, decay the confidence using Gaussian or linear function
 * 3. Remove blobs with confidence below scoreThreshold
 *
 * @param blobs - Array of detected blobs with confidence scores
 * @param options - Soft-NMS configuration options
 * @returns Filtered array of blobs with adjusted confidence scores
 */
export function softNMS(
  blobs: DetectedBlob[],
  options: SoftNMSOptions = {}
): DetectedBlob[] {
  const opts = { ...DEFAULT_SOFT_NMS_OPTIONS, ...options };

  if (blobs.length === 0) {
    return [];
  }

  // Create working copy with mutable confidence
  const workingBlobs = blobs.map(b => ({ ...b }));

  // Sort by confidence (highest first)
  workingBlobs.sort((a, b) => b.confidence - a.confidence);

  const result: DetectedBlob[] = [];

  while (workingBlobs.length > 0) {
    // Get highest scoring blob
    const maxBlob = workingBlobs.shift()!;

    // Skip if below threshold
    if (maxBlob.confidence < opts.scoreThreshold) {
      continue;
    }

    // Add to result
    result.push(maxBlob);

    // Decay confidence of remaining overlapping blobs
    for (const blob of workingBlobs) {
      const iou = calculateIoU(maxBlob, blob);

      if (iou > opts.iouThreshold) {
        if (opts.method === 'gaussian') {
          // Gaussian penalty: score *= exp(-iou^2 / sigma)
          blob.confidence *= Math.exp(-(iou * iou) / opts.sigma);
        } else {
          // Linear penalty: score *= (1 - iou)
          blob.confidence *= 1 - iou;
        }
      }
    }

    // Re-sort after confidence updates
    workingBlobs.sort((a, b) => b.confidence - a.confidence);
  }

  // Sort final result by confidence
  result.sort((a, b) => b.confidence - a.confidence);

  return result;
}

/**
 * Calculate initial confidence score for a detected blob.
 *
 * Confidence is computed based on multiple factors:
 * 1. Size fit: How well the blob size fits within the expected range
 * 2. Aspect ratio fit: How close the aspect ratio is to ideal (1.0 for circles)
 * 3. Area density: Ratio of blob area to bounding box area
 *
 * @param blob - Detected blob without confidence
 * @param options - Size constraints for scoring
 * @returns Confidence score between 0 and 1
 */
export function calculateBlobConfidence(
  blob: Omit<DetectedBlob, 'confidence'>,
  options: {
    minWidth: number;
    maxWidth: number;
    minHeight: number;
    maxHeight: number;
    idealAspectRatio?: number;  // Default: 1.0 (circular)
  }
): number {
  const { minWidth, maxWidth, minHeight, maxHeight, idealAspectRatio = 1.0 } = options;

  // Size fit score (0-1): How well the blob fits in the size range
  // Optimal is in the middle of the range, penalty for being at edges
  const widthRange = maxWidth - minWidth;
  const heightRange = maxHeight - minHeight;

  const widthMid = minWidth + widthRange / 2;
  const heightMid = minHeight + heightRange / 2;

  // Distance from center of range, normalized to 0-1
  const widthDeviation = Math.abs(blob.width - widthMid) / (widthRange / 2);
  const heightDeviation = Math.abs(blob.height - heightMid) / (heightRange / 2);

  // Clamp deviations to [0, 1] (might be slightly outside range after merging)
  const widthScore = Math.max(0, 1 - Math.min(1, widthDeviation));
  const heightScore = Math.max(0, 1 - Math.min(1, heightDeviation));
  const sizeScore = (widthScore + heightScore) / 2;

  // Aspect ratio score (0-1): How close to ideal aspect ratio
  const aspectRatioDeviation = Math.abs(blob.aspectRatio - idealAspectRatio);
  const aspectScore = Math.max(0, 1 - aspectRatioDeviation);

  // Area density score (0-1): Ratio of actual area to bounding box area
  // Higher density = more blob-like (vs. irregular shape)
  const boundingArea = blob.width * blob.height;
  const densityScore = blob.area / boundingArea;

  // Weighted combination
  // Size is most important (50%), then density (30%), then aspect ratio (20%)
  const confidence = 0.5 * sizeScore + 0.3 * densityScore + 0.2 * aspectScore;

  // Ensure result is in [0, 1]
  return Math.max(0, Math.min(1, confidence));
}

/**
 * Merge overlapping blobs and compute merged confidence.
 * Used when combining blobs from overlapping tiles.
 *
 * @param blobs - Array of overlapping blobs to merge
 * @returns Single merged blob with averaged confidence
 */
export function mergeOverlappingBlobs(blobs: DetectedBlob[]): DetectedBlob {
  if (blobs.length === 0) {
    throw new Error('Cannot merge empty blob array');
  }

  if (blobs.length === 1) {
    return { ...blobs[0] };
  }

  // Compute merged bounding box
  const minX = Math.min(...blobs.map(b => b.x));
  const minY = Math.min(...blobs.map(b => b.y));
  const maxX = Math.max(...blobs.map(b => b.x + b.width));
  const maxY = Math.max(...blobs.map(b => b.y + b.height));

  const width = maxX - minX;
  const height = maxY - minY;

  // Sum areas (approximate total area)
  const totalArea = blobs.reduce((sum, b) => sum + b.area, 0);

  // Weighted average confidence based on area
  const totalWeight = blobs.reduce((sum, b) => sum + b.area, 0);
  const weightedConfidence = blobs.reduce(
    (sum, b) => sum + b.confidence * b.area,
    0
  ) / totalWeight;

  return {
    x: minX,
    y: minY,
    width,
    height,
    area: totalArea,
    aspectRatio: width / height,
    confidence: weightedConfidence,
  };
}
