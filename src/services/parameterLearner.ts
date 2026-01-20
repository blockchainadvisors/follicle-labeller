/**
 * Parameter learning service for BLOB detection.
 * Extracts detection parameters from selected example annotations.
 */

import type {
  Follicle,
  LearnedDetectionParams,
  EffectiveSizeRange,
} from '../types';
import { isCircle, isRectangle, isLinear } from '../types';

/**
 * Bounding box representation for an annotation.
 */
export interface AnnotationBounds {
  width: number;
  height: number;
  centerX: number;
  centerY: number;
  aspectRatio: number;
}

/**
 * Get bounding box dimensions for any annotation type.
 *
 * @param annotation - The annotation to measure
 * @returns Bounding box dimensions
 */
export function getAnnotationBounds(annotation: Follicle): AnnotationBounds {
  if (isCircle(annotation)) {
    const diameter = annotation.radius * 2;
    return {
      width: diameter,
      height: diameter,
      centerX: annotation.center.x,
      centerY: annotation.center.y,
      aspectRatio: 1,
    };
  }

  if (isRectangle(annotation)) {
    return {
      width: annotation.width,
      height: annotation.height,
      centerX: annotation.x + annotation.width / 2,
      centerY: annotation.y + annotation.height / 2,
      aspectRatio: annotation.width / annotation.height,
    };
  }

  if (isLinear(annotation)) {
    // Linear annotation is a rotated rectangle defined by centerline + half-width
    const { startPoint, endPoint, halfWidth } = annotation;

    // Calculate the length of the centerline
    const dx = endPoint.x - startPoint.x;
    const dy = endPoint.y - startPoint.y;
    const length = Math.sqrt(dx * dx + dy * dy);

    // The "length" dimension is along the centerline, "width" is perpendicular
    // We need to compute the axis-aligned bounding box of the rotated rectangle
    const angle = Math.atan2(dy, dx);

    // Half dimensions along the rotated axes
    const halfLength = length / 2;
    const halfW = halfWidth;

    // Compute corners of the rotated rectangle relative to center
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);

    // Four corners relative to center
    const corners = [
      { x: halfLength * cos - halfW * sin, y: halfLength * sin + halfW * cos },
      { x: halfLength * cos + halfW * sin, y: halfLength * sin - halfW * cos },
      { x: -halfLength * cos - halfW * sin, y: -halfLength * sin + halfW * cos },
      { x: -halfLength * cos + halfW * sin, y: -halfLength * sin - halfW * cos },
    ];

    // Find bounding box
    const xs = corners.map(c => c.x);
    const ys = corners.map(c => c.y);
    const width = Math.max(...xs) - Math.min(...xs);
    const height = Math.max(...ys) - Math.min(...ys);

    const centerX = (startPoint.x + endPoint.x) / 2;
    const centerY = (startPoint.y + endPoint.y) / 2;

    return {
      width,
      height,
      centerX,
      centerY,
      aspectRatio: width / height,
    };
  }

  // Fallback (should never reach here)
  return {
    width: 0,
    height: 0,
    centerX: 0,
    centerY: 0,
    aspectRatio: 1,
  };
}

/**
 * Calculate standard deviation of an array
 */
function standardDeviation(values: number[]): number {
  if (values.length < 2) return 0;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const squaredDiffs = values.map(v => Math.pow(v - mean, 2));
  return Math.sqrt(squaredDiffs.reduce((a, b) => a + b, 0) / values.length);
}

/**
 * Learn detection parameters from selected example annotations.
 *
 * The algorithm adds automatic padding to ensure the learned range
 * captures natural variation in follicle sizes:
 * - Uses standard deviation to expand the range
 * - Ensures minimum spread of 30% of mean size
 * - Adds base padding for robustness
 *
 * @param annotations - Array of example annotations
 * @param imageData - Optional image data for intensity analysis
 * @returns Learned detection parameters
 */
export function learnFromExamples(
  annotations: Follicle[],
  imageData?: ImageData
): LearnedDetectionParams {
  if (annotations.length === 0) {
    // Return reasonable defaults if no examples provided
    return {
      minWidth: 10,
      maxWidth: 200,
      minHeight: 10,
      maxHeight: 200,
      minAspectRatio: 0.5,
      maxAspectRatio: 2.0,
      exampleCount: 0,
    };
  }

  // Extract bounds for all annotations
  const bounds = annotations.map(getAnnotationBounds);

  // Calculate statistics
  const widths = bounds.map(b => b.width);
  const heights = bounds.map(b => b.height);
  const aspectRatios = bounds.map(b => b.aspectRatio);

  // Calculate means and standard deviations
  const meanWidth = widths.reduce((a, b) => a + b, 0) / widths.length;
  const meanHeight = heights.reduce((a, b) => a + b, 0) / heights.length;
  const stdWidth = standardDeviation(widths);
  const stdHeight = standardDeviation(heights);

  // Base min/max from examples
  let minWidth = Math.min(...widths);
  let maxWidth = Math.max(...widths);
  let minHeight = Math.min(...heights);
  let maxHeight = Math.max(...heights);

  // Expand range by 2 standard deviations (covers ~95% of normal distribution)
  // This helps when examples are very similar in size
  const stdPadding = 2;
  minWidth = Math.min(minWidth, meanWidth - stdPadding * stdWidth);
  maxWidth = Math.max(maxWidth, meanWidth + stdPadding * stdWidth);
  minHeight = Math.min(minHeight, meanHeight - stdPadding * stdHeight);
  maxHeight = Math.max(maxHeight, meanHeight + stdPadding * stdHeight);

  // Ensure minimum spread of 30% of mean size
  // This prevents overly narrow ranges when all examples are nearly identical
  const minSpreadFactor = 0.3;
  const minWidthSpread = meanWidth * minSpreadFactor;
  const minHeightSpread = meanHeight * minSpreadFactor;

  if (maxWidth - minWidth < minWidthSpread) {
    const center = (minWidth + maxWidth) / 2;
    minWidth = center - minWidthSpread / 2;
    maxWidth = center + minWidthSpread / 2;
  }

  if (maxHeight - minHeight < minHeightSpread) {
    const center = (minHeight + maxHeight) / 2;
    minHeight = center - minHeightSpread / 2;
    maxHeight = center + minHeightSpread / 2;
  }

  // Add 10% base padding for robustness
  const basePadding = 0.1;
  minWidth *= (1 - basePadding);
  maxWidth *= (1 + basePadding);
  minHeight *= (1 - basePadding);
  maxHeight *= (1 + basePadding);

  // Ensure minimums don't go below reasonable values
  minWidth = Math.max(5, minWidth);
  minHeight = Math.max(5, minHeight);

  // Aspect ratio with some padding
  const minAspectRatio = Math.min(...aspectRatios) * 0.8;
  const maxAspectRatio = Math.max(...aspectRatios) * 1.2;

  // Calculate mean intensity if image data is provided
  let meanIntensity: number | undefined;
  if (imageData) {
    const intensities: number[] = [];

    for (const bound of bounds) {
      // Sample pixels around the center of each annotation
      const sampleRadius = Math.min(bound.width, bound.height) / 4;
      const samples = sampleIntensity(
        imageData,
        bound.centerX,
        bound.centerY,
        sampleRadius
      );
      intensities.push(...samples);
    }

    if (intensities.length > 0) {
      meanIntensity = intensities.reduce((a, b) => a + b, 0) / intensities.length;
    }
  }

  return {
    minWidth: Math.round(minWidth),
    maxWidth: Math.round(maxWidth),
    minHeight: Math.round(minHeight),
    maxHeight: Math.round(maxHeight),
    minAspectRatio: Math.round(minAspectRatio * 100) / 100,
    maxAspectRatio: Math.round(maxAspectRatio * 100) / 100,
    meanIntensity,
    exampleCount: annotations.length,
  };
}

/**
 * Sample pixel intensities around a point in image data.
 */
function sampleIntensity(
  imageData: ImageData,
  centerX: number,
  centerY: number,
  radius: number
): number[] {
  const intensities: number[] = [];
  const { data, width, height } = imageData;

  const x0 = Math.max(0, Math.floor(centerX - radius));
  const x1 = Math.min(width - 1, Math.ceil(centerX + radius));
  const y0 = Math.max(0, Math.floor(centerY - radius));
  const y1 = Math.min(height - 1, Math.ceil(centerY + radius));

  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const dx = x - centerX;
      const dy = y - centerY;
      if (dx * dx + dy * dy <= radius * radius) {
        const idx = (y * width + x) * 4;
        // Convert to grayscale intensity
        const gray =
          0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
        intensities.push(gray);
      }
    }
  }

  return intensities;
}

/**
 * Apply tolerance padding to learned parameters.
 *
 * @param params - Learned detection parameters
 * @param tolerance - Tolerance factor (0-1), e.g., 0.2 for 20%
 * @returns Effective size range with tolerance applied
 */
export function applyTolerance(
  params: LearnedDetectionParams,
  tolerance: number
): EffectiveSizeRange {
  // Clamp tolerance to valid range
  const t = Math.max(0, Math.min(1, tolerance));

  return {
    minWidth: Math.max(1, Math.round(params.minWidth * (1 - t))),
    maxWidth: Math.round(params.maxWidth * (1 + t)),
    minHeight: Math.max(1, Math.round(params.minHeight * (1 - t))),
    maxHeight: Math.round(params.maxHeight * (1 + t)),
  };
}

/**
 * Format size range for display.
 */
export function formatSizeRange(params: LearnedDetectionParams): string {
  const { minWidth, maxWidth, minHeight, maxHeight } = params;
  return `${minWidth}-${maxWidth}px (W) × ${minHeight}-${maxHeight}px (H)`;
}

/**
 * Format aspect ratio range for display.
 */
export function formatAspectRatio(params: LearnedDetectionParams): string {
  const { minAspectRatio, maxAspectRatio } = params;
  if (Math.abs(minAspectRatio - maxAspectRatio) < 0.1) {
    return `~${minAspectRatio.toFixed(2)}`;
  }
  return `${minAspectRatio.toFixed(2)} - ${maxAspectRatio.toFixed(2)}`;
}
