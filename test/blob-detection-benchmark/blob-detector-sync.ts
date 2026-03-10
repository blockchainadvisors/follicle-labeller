/**
 * Standalone synchronous blob detector for Node.js environment.
 * Extracted from blobDetector.ts to avoid web worker dependencies.
 */

import type { BlobDetectionOptions, DetectedBlob } from '../../src/types';

/**
 * Default detection options.
 */
export const DEFAULT_DETECTION_OPTIONS: BlobDetectionOptions = {
  minWidth: 10,
  maxWidth: 200,
  minHeight: 10,
  maxHeight: 200,
  threshold: undefined,
  darkBlobs: true,
  useGPU: false,
  workerCount: undefined,
  tileSize: 0,
  tileOverlap: 0.2,
  useCLAHE: true,
  claheClipLimit: 3.0,
  claheTileSize: 8,
  useSoftNMS: true,
  softNMSSigma: 0.5,
  softNMSThreshold: 0.1,
  useGaussianBlur: true,
  gaussianKernelSize: 5,
  useMorphOpen: true,
  morphKernelSize: 3,
  minCircularity: 0.2,
};

/**
 * Soft-NMS options.
 */
interface SoftNMSOptions {
  sigma: number;
  scoreThreshold: number;
  iouThreshold: number;
  method: 'linear' | 'gaussian';
}

/**
 * Compute Otsu's threshold from grayscale data.
 */
function computeOtsuThreshold(grayscale: Uint8Array): number {
  const histogram = new Uint32Array(256);
  for (let i = 0; i < grayscale.length; i++) {
    histogram[grayscale[i]]++;
  }

  const total = grayscale.length;
  let sumTotal = 0;
  for (let i = 0; i < 256; i++) {
    sumTotal += i * histogram[i];
  }

  let sumBackground = 0;
  let weightBackground = 0;
  let maxVariance = 0;
  let threshold = 0;

  for (let t = 0; t < 256; t++) {
    weightBackground += histogram[t];
    if (weightBackground === 0) continue;

    const weightForeground = total - weightBackground;
    if (weightForeground === 0) break;

    sumBackground += t * histogram[t];

    const meanBackground = sumBackground / weightBackground;
    const meanForeground = (sumTotal - sumBackground) / weightForeground;

    const variance = weightBackground * weightForeground *
                     Math.pow(meanBackground - meanForeground, 2);

    if (variance > maxVariance) {
      maxVariance = variance;
      threshold = t;
    }
  }

  return threshold;
}

/**
 * Apply Gaussian blur using separable convolution.
 */
function applyGaussianBlurSync(
  grayscale: Uint8Array,
  width: number,
  height: number,
  kernelSize: number = 5
): Uint8Array {
  const sigma = 0.3 * ((kernelSize - 1) * 0.5 - 1) + 0.8;
  const kernel = new Float32Array(kernelSize);
  const half = Math.floor(kernelSize / 2);
  let sum = 0;

  for (let i = 0; i < kernelSize; i++) {
    const x = i - half;
    kernel[i] = Math.exp(-(x * x) / (2 * sigma * sigma));
    sum += kernel[i];
  }
  for (let i = 0; i < kernelSize; i++) {
    kernel[i] /= sum;
  }

  const temp = new Uint8Array(grayscale.length);
  const result = new Uint8Array(grayscale.length);

  // Horizontal pass
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let val = 0;
      for (let k = -half; k <= half; k++) {
        const px = Math.min(Math.max(x + k, 0), width - 1);
        val += grayscale[y * width + px] * kernel[k + half];
      }
      temp[y * width + x] = Math.round(val);
    }
  }

  // Vertical pass
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let val = 0;
      for (let k = -half; k <= half; k++) {
        const py = Math.min(Math.max(y + k, 0), height - 1);
        val += temp[py * width + x] * kernel[k + half];
      }
      result[y * width + x] = Math.round(val);
    }
  }

  return result;
}

/**
 * Apply morphological opening (erosion + dilation).
 */
function morphologicalOpenSync(
  binary: Uint8Array,
  width: number,
  height: number,
  kernelSize: number = 3
): Uint8Array {
  const half = Math.floor(kernelSize / 2);
  const eroded = new Uint8Array(binary.length);
  const opened = new Uint8Array(binary.length);

  // Erosion
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let allOnes = true;
      for (let ky = -half; ky <= half && allOnes; ky++) {
        for (let kx = -half; kx <= half && allOnes; kx++) {
          const py = y + ky;
          const px = x + kx;
          if (py < 0 || py >= height || px < 0 || px >= width || binary[py * width + px] === 0) {
            allOnes = false;
          }
        }
      }
      eroded[y * width + x] = allOnes ? 1 : 0;
    }
  }

  // Dilation
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let anyOne = false;
      for (let ky = -half; ky <= half && !anyOne; ky++) {
        for (let kx = -half; kx <= half && !anyOne; kx++) {
          const py = y + ky;
          const px = x + kx;
          if (py >= 0 && py < height && px >= 0 && px < width && eroded[py * width + px] === 1) {
            anyOne = true;
          }
        }
      }
      opened[y * width + x] = anyOne ? 1 : 0;
    }
  }

  return opened;
}

/**
 * Component stats including perimeter for circularity calculation.
 */
interface SyncComponentStats {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  area: number;
  perimeterCount: number;
}

/**
 * Compute IoU (Intersection over Union) between two boxes.
 */
function computeIoU(a: DetectedBlob, b: DetectedBlob): number {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.width, b.x + b.width);
  const y2 = Math.min(a.y + a.height, b.y + b.height);

  if (x2 <= x1 || y2 <= y1) return 0;

  const intersection = (x2 - x1) * (y2 - y1);
  const areaA = a.width * a.height;
  const areaB = b.width * b.height;
  const union = areaA + areaB - intersection;

  return intersection / union;
}

/**
 * Apply Soft-NMS to suppress overlapping detections.
 */
function softNMS(blobs: DetectedBlob[], options: SoftNMSOptions): DetectedBlob[] {
  if (blobs.length === 0) return [];

  // Create working copy with indices
  const boxes = blobs.map((blob, index) => ({
    blob: { ...blob },
    index,
    score: blob.confidence,
  }));

  // Sort by score (descending)
  boxes.sort((a, b) => b.score - a.score);

  const result: DetectedBlob[] = [];

  while (boxes.length > 0) {
    // Pick the box with highest score
    const current = boxes.shift()!;

    if (current.score < options.scoreThreshold) {
      break;
    }

    // Update confidence and add to result
    current.blob.confidence = current.score;
    result.push(current.blob);

    // Update scores of remaining boxes
    for (const box of boxes) {
      const iou = computeIoU(current.blob, box.blob);

      if (iou > options.iouThreshold) {
        if (options.method === 'gaussian') {
          box.score *= Math.exp(-(iou * iou) / options.sigma);
        } else {
          box.score *= 1 - iou;
        }
      }
    }

    // Re-sort by score
    boxes.sort((a, b) => b.score - a.score);
  }

  return result;
}

/**
 * Calculate confidence score.
 */
function calculateSyncConfidence(
  width: number,
  height: number,
  area: number,
  aspectRatio: number,
  options: BlobDetectionOptions
): number {
  const { minWidth, maxWidth, minHeight, maxHeight } = options;

  const widthRange = maxWidth - minWidth;
  const heightRange = maxHeight - minHeight;
  const widthMid = minWidth + widthRange / 2;
  const heightMid = minHeight + heightRange / 2;

  const widthDeviation = Math.abs(width - widthMid) / (widthRange / 2);
  const heightDeviation = Math.abs(height - heightMid) / (heightRange / 2);

  const widthScore = Math.max(0, 1 - Math.min(1, widthDeviation));
  const heightScore = Math.max(0, 1 - Math.min(1, heightDeviation));
  const sizeScore = (widthScore + heightScore) / 2;

  const idealAspectRatio = 1.0;
  const aspectRatioDeviation = Math.abs(aspectRatio - idealAspectRatio);
  const aspectScore = Math.max(0, 1 - aspectRatioDeviation);

  const boundingArea = width * height;
  const densityScore = area / boundingArea;

  const confidence = 0.5 * sizeScore + 0.3 * densityScore + 0.2 * aspectScore;

  return Math.max(0, Math.min(1, confidence));
}

/**
 * Quick synchronous detection for small images (no workers).
 * Pipeline: grayscale -> blur -> threshold -> morphology -> labeling
 */
export function detectBlobsSync(
  imageData: ImageData,
  options: Partial<BlobDetectionOptions> = {}
): DetectedBlob[] {
  const opts: BlobDetectionOptions = {
    ...DEFAULT_DETECTION_OPTIONS,
    ...options,
  };

  const width = imageData.width;
  const height = imageData.height;

  // Step 1: Convert to grayscale
  let grayscale = new Uint8Array(width * height);
  const data = imageData.data;
  for (let i = 0; i < grayscale.length; i++) {
    const idx = i * 4;
    grayscale[i] = Math.round(
      0.299 * data[idx] +
      0.587 * data[idx + 1] +
      0.114 * data[idx + 2]
    );
  }

  // Step 2: Apply Gaussian blur
  if (opts.useGaussianBlur !== false) {
    const kernelSize = opts.gaussianKernelSize ?? 5;
    grayscale = applyGaussianBlurSync(grayscale, width, height, kernelSize);
  }

  // Step 3: Compute threshold
  const threshold = opts.threshold ?? computeOtsuThreshold(grayscale);

  // Step 4: Apply threshold
  let binary = new Uint8Array(grayscale.length);
  for (let i = 0; i < grayscale.length; i++) {
    if (opts.darkBlobs) {
      binary[i] = grayscale[i] < threshold ? 1 : 0;
    } else {
      binary[i] = grayscale[i] >= threshold ? 1 : 0;
    }
  }

  // Step 5: Apply morphological opening
  if (opts.useMorphOpen !== false) {
    const kernelSize = opts.morphKernelSize ?? 3;
    binary = morphologicalOpenSync(binary, width, height, kernelSize);
  }

  // Step 6: Connected component labeling
  const labels = new Int32Array(binary.length);
  const parent: number[] = [];
  let nextLabel = 1;

  function find(x: number): number {
    if (parent[x] !== x) {
      parent[x] = find(parent[x]);
    }
    return parent[x];
  }

  function union(x: number, y: number): void {
    const rootX = find(x);
    const rootY = find(y);
    if (rootX !== rootY) {
      parent[rootX] = rootY;
    }
  }

  // First pass
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      if (binary[idx] === 0) continue;

      const neighbors: number[] = [];

      if (x > 0 && binary[idx - 1] === 1) neighbors.push(labels[idx - 1]);
      if (y > 0 && binary[idx - width] === 1) neighbors.push(labels[idx - width]);
      if (x > 0 && y > 0 && binary[idx - width - 1] === 1) neighbors.push(labels[idx - width - 1]);
      if (x < width - 1 && y > 0 && binary[idx - width + 1] === 1) neighbors.push(labels[idx - width + 1]);

      if (neighbors.length === 0) {
        labels[idx] = nextLabel;
        parent[nextLabel] = nextLabel;
        nextLabel++;
      } else {
        const minLabel = Math.min(...neighbors);
        labels[idx] = minLabel;
        for (const n of neighbors) {
          if (n !== minLabel) union(minLabel, n);
        }
      }
    }
  }

  // Step 7: Resolve labels and collect stats with perimeter
  const stats = new Map<number, SyncComponentStats>();
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      if (labels[idx] === 0) continue;

      const root = find(labels[idx]);
      labels[idx] = root;

      let s = stats.get(root);
      if (!s) {
        s = { minX: x, minY: y, maxX: x, maxY: y, area: 0, perimeterCount: 0 };
        stats.set(root, s);
      }

      s.minX = Math.min(s.minX, x);
      s.minY = Math.min(s.minY, y);
      s.maxX = Math.max(s.maxX, x);
      s.maxY = Math.max(s.maxY, y);
      s.area++;

      // Check if boundary pixel (4-connectivity)
      const isBoundary =
        x === 0 || binary[idx - 1] === 0 ||
        x === width - 1 || binary[idx + 1] === 0 ||
        y === 0 || binary[idx - width] === 0 ||
        y === height - 1 || binary[idx + width] === 0;

      if (isBoundary) {
        s.perimeterCount++;
      }
    }
  }

  // Step 8: Convert to blobs with circularity filter
  const blobs: DetectedBlob[] = [];
  const minCircularity = opts.minCircularity ?? 0.2;

  for (const s of stats.values()) {
    const blobWidth = s.maxX - s.minX + 1;
    const blobHeight = s.maxY - s.minY + 1;
    const aspectRatio = blobWidth / blobHeight;

    // Size filter
    if (
      blobWidth < opts.minWidth ||
      blobWidth > opts.maxWidth ||
      blobHeight < opts.minHeight ||
      blobHeight > opts.maxHeight
    ) {
      continue;
    }

    // Circularity filter
    if (s.perimeterCount > 0 && minCircularity > 0) {
      const circularity = (4 * Math.PI * s.area) / (s.perimeterCount * s.perimeterCount);
      if (circularity < minCircularity) {
        continue;
      }
    }

    // Calculate confidence score
    const confidence = calculateSyncConfidence(
      blobWidth,
      blobHeight,
      s.area,
      aspectRatio,
      opts
    );

    blobs.push({
      x: s.minX,
      y: s.minY,
      width: blobWidth,
      height: blobHeight,
      area: s.area,
      aspectRatio,
      confidence,
    });
  }

  // Apply Soft-NMS if enabled
  if (opts.useSoftNMS !== false) {
    const softNMSOptions: SoftNMSOptions = {
      sigma: opts.softNMSSigma ?? 0.5,
      scoreThreshold: opts.softNMSThreshold ?? 0.1,
      iouThreshold: 0.3,
      method: 'gaussian',
    };
    return softNMS(blobs, softNMSOptions);
  }

  // Sort by area if not using Soft-NMS
  blobs.sort((a, b) => b.area - a.area);

  return blobs;
}
