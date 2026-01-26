/**
 * Web Worker for parallel BLOB detection processing.
 * Processes image tiles independently using connected component labeling.
 */

import type { BlobDetectionOptions, DetectedBlob, BlobWorkerMessage, BlobWorkerResult } from '../types';

// Component statistics during labeling
interface ComponentStats {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  pixelCount: number;
  perimeterCount: number; // Number of boundary pixels for circularity calculation
}

/**
 * Convert ImageData to grayscale using luminance formula.
 * Uses ITU-R BT.601 coefficients: 0.299*R + 0.587*G + 0.114*B
 */
function toGrayscale(imageData: ImageData): Uint8Array {
  const gray = new Uint8Array(imageData.width * imageData.height);
  const data = imageData.data;

  for (let i = 0; i < gray.length; i++) {
    const idx = i * 4;
    gray[i] = Math.round(
      0.299 * data[idx] +
      0.587 * data[idx + 1] +
      0.114 * data[idx + 2]
    );
  }

  return gray;
}

/**
 * Apply Gaussian blur using separable convolution.
 * Matches OpenCV's GaussianBlur with sigma=0 (auto-computed from kernel size).
 */
function applyGaussianBlur(
  grayscale: Uint8Array,
  width: number,
  height: number,
  kernelSize: number = 5
): Uint8Array {
  // Generate 1D Gaussian kernel (sigma = 0 means auto-compute from kernel size)
  const sigma = 0.3 * ((kernelSize - 1) * 0.5 - 1) + 0.8;
  const kernel = new Float32Array(kernelSize);
  const half = Math.floor(kernelSize / 2);
  let sum = 0;

  for (let i = 0; i < kernelSize; i++) {
    const x = i - half;
    kernel[i] = Math.exp(-(x * x) / (2 * sigma * sigma));
    sum += kernel[i];
  }
  // Normalize kernel
  for (let i = 0; i < kernelSize; i++) {
    kernel[i] /= sum;
  }

  // Separable convolution: horizontal pass then vertical pass
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
 * Compute Otsu's threshold for automatic binarization.
 * Finds the threshold that maximizes between-class variance.
 */
function computeOtsuThreshold(grayscale: Uint8Array): number {
  // Compute histogram
  const histogram = new Uint32Array(256);
  for (let i = 0; i < grayscale.length; i++) {
    histogram[grayscale[i]]++;
  }

  const total = grayscale.length;

  // Compute total mean
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

    // Between-class variance
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
 * Apply threshold to grayscale image.
 * Returns binary image where 1 = foreground (blob), 0 = background.
 */
function applyThreshold(
  grayscale: Uint8Array,
  threshold: number,
  darkBlobs: boolean
): Uint8Array {
  const binary = new Uint8Array(grayscale.length);

  for (let i = 0; i < grayscale.length; i++) {
    if (darkBlobs) {
      // Dark blobs: pixels below threshold are foreground
      binary[i] = grayscale[i] < threshold ? 1 : 0;
    } else {
      // Light blobs: pixels above threshold are foreground
      binary[i] = grayscale[i] >= threshold ? 1 : 0;
    }
  }

  return binary;
}

/**
 * Apply morphological opening (erosion followed by dilation).
 * Separates touching objects and removes small noise.
 * Matches OpenCV's morphologyEx with MORPH_OPEN.
 */
function morphologicalOpen(
  binary: Uint8Array,
  width: number,
  height: number,
  kernelSize: number = 3
): Uint8Array {
  const half = Math.floor(kernelSize / 2);
  const eroded = new Uint8Array(binary.length);
  const opened = new Uint8Array(binary.length);

  // Erosion: pixel is 1 only if ALL neighbors within kernel are 1
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

  // Dilation: pixel is 1 if ANY neighbor within kernel is 1
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
 * Union-Find data structure for efficient connected component labeling.
 */
class UnionFind {
  private parent: Int32Array;
  private rank: Uint8Array;

  constructor(size: number) {
    this.parent = new Int32Array(size);
    this.rank = new Uint8Array(size);
    for (let i = 0; i < size; i++) {
      this.parent[i] = i;
    }
  }

  find(x: number): number {
    if (this.parent[x] !== x) {
      this.parent[x] = this.find(this.parent[x]); // Path compression
    }
    return this.parent[x];
  }

  union(x: number, y: number): void {
    const rootX = this.find(x);
    const rootY = this.find(y);

    if (rootX === rootY) return;

    // Union by rank
    if (this.rank[rootX] < this.rank[rootY]) {
      this.parent[rootX] = rootY;
    } else if (this.rank[rootX] > this.rank[rootY]) {
      this.parent[rootY] = rootX;
    } else {
      this.parent[rootY] = rootX;
      this.rank[rootX]++;
    }
  }
}

/**
 * Connected component labeling using two-pass algorithm with Union-Find.
 * Returns array of component labels for each pixel.
 */
function labelConnectedComponents(
  binary: Uint8Array,
  width: number,
  height: number
): { labels: Int32Array; maxLabel: number } {
  const labels = new Int32Array(binary.length);
  const uf = new UnionFind(binary.length);
  let nextLabel = 1;

  // First pass: assign provisional labels and record equivalences
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;

      if (binary[idx] === 0) continue; // Background

      const neighbors: number[] = [];

      // Check left neighbor
      if (x > 0 && binary[idx - 1] === 1) {
        neighbors.push(labels[idx - 1]);
      }

      // Check top neighbor
      if (y > 0 && binary[idx - width] === 1) {
        neighbors.push(labels[idx - width]);
      }

      // Check top-left neighbor (8-connectivity)
      if (x > 0 && y > 0 && binary[idx - width - 1] === 1) {
        neighbors.push(labels[idx - width - 1]);
      }

      // Check top-right neighbor (8-connectivity)
      if (x < width - 1 && y > 0 && binary[idx - width + 1] === 1) {
        neighbors.push(labels[idx - width + 1]);
      }

      if (neighbors.length === 0) {
        // New component
        labels[idx] = nextLabel++;
      } else {
        // Use minimum label from neighbors
        const minLabel = Math.min(...neighbors);
        labels[idx] = minLabel;

        // Union all neighbor labels
        for (const neighbor of neighbors) {
          if (neighbor !== minLabel) {
            uf.union(minLabel, neighbor);
          }
        }
      }
    }
  }

  // Second pass: resolve equivalences and collect component stats
  for (let i = 0; i < labels.length; i++) {
    if (labels[i] > 0) {
      labels[i] = uf.find(labels[i]);
    }
  }

  return { labels, maxLabel: nextLabel - 1 };
}

/**
 * Extract component statistics (bounding boxes and perimeter) from labeled image.
 * Perimeter is counted as pixels that border background (4-connectivity).
 */
function extractComponentStats(
  labels: Int32Array,
  binary: Uint8Array,
  width: number,
  height: number,
  _maxLabel: number
): Map<number, ComponentStats> {
  const stats = new Map<number, ComponentStats>();

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      const label = labels[idx];
      if (label === 0) continue;

      let s = stats.get(label);
      if (!s) {
        s = {
          minX: x,
          minY: y,
          maxX: x,
          maxY: y,
          pixelCount: 0,
          perimeterCount: 0,
        };
        stats.set(label, s);
      }

      s.minX = Math.min(s.minX, x);
      s.minY = Math.min(s.minY, y);
      s.maxX = Math.max(s.maxX, x);
      s.maxY = Math.max(s.maxY, y);
      s.pixelCount++;

      // Check if this pixel is on the boundary (4-connectivity)
      // A pixel is on boundary if any of its 4 neighbors is background or out of bounds
      const isBoundary =
        x === 0 || binary[idx - 1] === 0 ||           // left
        x === width - 1 || binary[idx + 1] === 0 ||   // right
        y === 0 || binary[idx - width] === 0 ||       // top
        y === height - 1 || binary[idx + width] === 0; // bottom

      if (isBoundary) {
        s.perimeterCount++;
      }
    }
  }

  return stats;
}

/**
 * Calculate confidence score for a blob based on size fit, aspect ratio, and density.
 * Higher scores indicate more likely follicle detections.
 */
function calculateConfidence(
  width: number,
  height: number,
  area: number,
  aspectRatio: number,
  options: BlobDetectionOptions
): number {
  const { minWidth, maxWidth, minHeight, maxHeight } = options;

  // Size fit score (0-1): How well the blob fits in the size range
  const widthRange = maxWidth - minWidth;
  const heightRange = maxHeight - minHeight;
  const widthMid = minWidth + widthRange / 2;
  const heightMid = minHeight + heightRange / 2;

  // Distance from center of range, normalized to 0-1
  const widthDeviation = Math.abs(width - widthMid) / (widthRange / 2);
  const heightDeviation = Math.abs(height - heightMid) / (heightRange / 2);

  const widthScore = Math.max(0, 1 - Math.min(1, widthDeviation));
  const heightScore = Math.max(0, 1 - Math.min(1, heightDeviation));
  const sizeScore = (widthScore + heightScore) / 2;

  // Aspect ratio score (0-1): How close to ideal aspect ratio (1.0 for circular)
  const idealAspectRatio = 1.0;
  const aspectRatioDeviation = Math.abs(aspectRatio - idealAspectRatio);
  const aspectScore = Math.max(0, 1 - aspectRatioDeviation);

  // Area density score (0-1): Ratio of actual area to bounding box area
  const boundingArea = width * height;
  const densityScore = area / boundingArea;

  // Weighted combination: size 50%, density 30%, aspect ratio 20%
  const confidence = 0.5 * sizeScore + 0.3 * densityScore + 0.2 * aspectScore;

  return Math.max(0, Math.min(1, confidence));
}

/**
 * Convert component stats to DetectedBlob objects.
 * Filters by size constraints and circularity.
 */
function statsToBlobs(
  stats: Map<number, ComponentStats>,
  options: BlobDetectionOptions,
  offsetX: number,
  offsetY: number
): DetectedBlob[] {
  const blobs: DetectedBlob[] = [];
  const minCircularity = options.minCircularity ?? 0.2;

  for (const [, s] of stats) {
    const width = s.maxX - s.minX + 1;
    const height = s.maxY - s.minY + 1;
    const aspectRatio = width / height;

    // Filter by size constraints
    if (
      width < options.minWidth ||
      width > options.maxWidth ||
      height < options.minHeight ||
      height > options.maxHeight
    ) {
      continue;
    }

    // Calculate circularity: 4 * PI * area / perimeter^2
    // Perfect circle = 1.0, lower values = more elongated/irregular
    if (s.perimeterCount > 0 && minCircularity > 0) {
      const circularity = (4 * Math.PI * s.pixelCount) / (s.perimeterCount * s.perimeterCount);
      if (circularity < minCircularity) {
        continue; // Skip non-circular shapes
      }
    }

    const confidence = calculateConfidence(
      width,
      height,
      s.pixelCount,
      aspectRatio,
      options
    );

    blobs.push({
      x: s.minX + offsetX,
      y: s.minY + offsetY,
      width,
      height,
      area: s.pixelCount,
      aspectRatio,
      confidence,
    });
  }

  return blobs;
}

/**
 * Process a single image tile and return detected blobs.
 * Pipeline matches OpenCV blob detection: grayscale -> blur -> threshold -> morphology -> labeling
 */
function processImageTile(
  imageData: ImageData,
  options: BlobDetectionOptions,
  offsetX: number,
  offsetY: number
): DetectedBlob[] {
  const width = imageData.width;
  const height = imageData.height;

  // Step 1: Convert to grayscale
  let grayscale = toGrayscale(imageData);

  // Step 2: Apply Gaussian blur (reduces noise, matches OpenCV pipeline)
  if (options.useGaussianBlur !== false) {
    const kernelSize = options.gaussianKernelSize ?? 5;
    grayscale = applyGaussianBlur(grayscale, width, height, kernelSize);
  }

  // Step 3: Compute or use threshold
  const threshold = options.threshold ?? computeOtsuThreshold(grayscale);

  // Step 4: Apply threshold
  let binary = applyThreshold(grayscale, threshold, options.darkBlobs);

  // Step 5: Apply morphological opening (separates touching objects)
  if (options.useMorphOpen !== false) {
    const kernelSize = options.morphKernelSize ?? 3;
    binary = morphologicalOpen(binary, width, height, kernelSize);
  }

  // Step 6: Connected component labeling
  const { labels, maxLabel } = labelConnectedComponents(binary, width, height);

  // Step 7: Extract component stats (pass binary for perimeter calculation)
  const stats = extractComponentStats(labels, binary, width, height, maxLabel);

  // Step 8: Convert to blobs with offset adjustment and circularity filter
  return statsToBlobs(stats, options, offsetX, offsetY);
}

// Web Worker message handler
self.onmessage = (e: MessageEvent<BlobWorkerMessage>) => {
  const { imageData, tileX, tileY, options } = e.data;

  try {
    const blobs = processImageTile(imageData, options, tileX, tileY);

    const result: BlobWorkerResult = {
      blobs,
      tileX,
      tileY,
    };

    self.postMessage(result);
  } catch (error) {
    console.error('Worker error:', error);
    self.postMessage({ blobs: [], tileX, tileY } as BlobWorkerResult);
  }
};
