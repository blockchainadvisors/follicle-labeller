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
 * Extract component statistics (bounding boxes) from labeled image.
 */
function extractComponentStats(
  labels: Int32Array,
  width: number,
  height: number,
  _maxLabel: number
): Map<number, ComponentStats> {
  const stats = new Map<number, ComponentStats>();

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const label = labels[y * width + x];
      if (label === 0) continue;

      let s = stats.get(label);
      if (!s) {
        s = {
          minX: x,
          minY: y,
          maxX: x,
          maxY: y,
          pixelCount: 0,
        };
        stats.set(label, s);
      }

      s.minX = Math.min(s.minX, x);
      s.minY = Math.min(s.minY, y);
      s.maxX = Math.max(s.maxX, x);
      s.maxY = Math.max(s.maxY, y);
      s.pixelCount++;
    }
  }

  return stats;
}

/**
 * Convert component stats to DetectedBlob objects.
 */
function statsToBlobs(
  stats: Map<number, ComponentStats>,
  options: BlobDetectionOptions,
  offsetX: number,
  offsetY: number
): DetectedBlob[] {
  const blobs: DetectedBlob[] = [];

  for (const [, s] of stats) {
    const width = s.maxX - s.minX + 1;
    const height = s.maxY - s.minY + 1;
    const aspectRatio = width / height;

    // Filter by size constraints
    if (
      width >= options.minWidth &&
      width <= options.maxWidth &&
      height >= options.minHeight &&
      height <= options.maxHeight
    ) {
      blobs.push({
        x: s.minX + offsetX,
        y: s.minY + offsetY,
        width,
        height,
        area: s.pixelCount,
        aspectRatio,
      });
    }
  }

  return blobs;
}

/**
 * Process a single image tile and return detected blobs.
 */
function processImageTile(
  imageData: ImageData,
  options: BlobDetectionOptions,
  offsetX: number,
  offsetY: number
): DetectedBlob[] {
  // Step 1: Convert to grayscale
  const grayscale = toGrayscale(imageData);

  // Step 2: Compute or use threshold
  const threshold = options.threshold ?? computeOtsuThreshold(grayscale);

  // Step 3: Apply threshold
  const binary = applyThreshold(grayscale, threshold, options.darkBlobs);

  // Step 4: Connected component labeling
  const { labels, maxLabel } = labelConnectedComponents(
    binary,
    imageData.width,
    imageData.height
  );

  // Step 5: Extract component stats
  const stats = extractComponentStats(
    labels,
    imageData.width,
    imageData.height,
    maxLabel
  );

  // Step 6: Convert to blobs with offset adjustment
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
