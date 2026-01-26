/**
 * Main BLOB detection service.
 * Orchestrates parallel processing using Web Workers and optional GPU acceleration.
 */

import type { BlobDetectionOptions, DetectedBlob, BlobWorkerMessage, BlobWorkerResult } from '../types';
import { getGPUProcessor } from './gpuProcessor';
import { softNMS, type SoftNMSOptions } from './softNms';
import { applyCLAHEToImageData } from './claheProcessor';

// Import worker as URL for Vite
import BlobDetectorWorker from './blobDetector.worker.ts?worker';

/**
 * Default detection options.
 * Values tuned to match the Python/OpenCV pipeline for consistent results.
 */
export const DEFAULT_DETECTION_OPTIONS: BlobDetectionOptions = {
  minWidth: 10,
  maxWidth: 200,
  minHeight: 10,
  maxHeight: 200,
  threshold: undefined,   // Auto (Otsu's method)
  darkBlobs: true,        // Follicles are typically dark
  useGPU: true,           // Use WebGL if available
  workerCount: undefined, // Auto (navigator.hardwareConcurrency)
  // SAHI-style tiling (disabled by default for backward compatibility)
  tileSize: 0,            // 0 = auto based on workerCount (legacy behavior)
  tileOverlap: 0.2,       // 20% overlap when using fixed tile size
  // CLAHE preprocessing (matches Python: clipLimit=3.0, always enabled)
  useCLAHE: true,         // Enabled by default (matches Python)
  claheClipLimit: 3.0,    // Match Python value (was 2.0)
  claheTileSize: 8,
  // Soft-NMS
  useSoftNMS: true,       // Use Soft-NMS for better overlap handling
  softNMSSigma: 0.5,
  softNMSThreshold: 0.1,
  // Gaussian blur preprocessing (matches OpenCV pipeline)
  useGaussianBlur: true,  // Enabled by default
  gaussianKernelSize: 5,  // 5x5 kernel (matches Python)
  // Morphological opening (separates touching objects)
  useMorphOpen: true,     // Enabled by default
  morphKernelSize: 3,     // 3x3 kernel (matches Python)
  // Circularity filter (rejects elongated shapes)
  minCircularity: 0.2,    // Match OpenCV SimpleBlobDetector
};

/**
 * Component statistics for merging blobs across tile boundaries.
 */
interface ComponentStats {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  area: number;
}

/**
 * Compute Otsu's threshold from grayscale data.
 * This is used when GPU processing is enabled to compute the threshold before
 * sending to workers, ensuring consistent thresholding across all tiles.
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
 * Get ImageData from an ImageBitmap using an OffscreenCanvas.
 */
function getImageData(imageBitmap: ImageBitmap): ImageData {
  const canvas = new OffscreenCanvas(imageBitmap.width, imageBitmap.height);
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(imageBitmap, 0, 0);
  return ctx.getImageData(0, 0, imageBitmap.width, imageBitmap.height);
}

/**
 * Get ImageData for a specific tile region.
 */
function getTileImageData(
  imageData: ImageData,
  tileX: number,
  tileY: number,
  tileWidth: number,
  tileHeight: number
): ImageData {
  // Create a temporary ImageData from the source region
  const tileData = new ImageData(tileWidth, tileHeight);

  for (let y = 0; y < tileHeight; y++) {
    for (let x = 0; x < tileWidth; x++) {
      const srcX = tileX + x;
      const srcY = tileY + y;
      const srcIdx = (srcY * imageData.width + srcX) * 4;
      const dstIdx = (y * tileWidth + x) * 4;

      tileData.data[dstIdx] = imageData.data[srcIdx];
      tileData.data[dstIdx + 1] = imageData.data[srcIdx + 1];
      tileData.data[dstIdx + 2] = imageData.data[srcIdx + 2];
      tileData.data[dstIdx + 3] = imageData.data[srcIdx + 3];
    }
  }

  return tileData;
}

/**
 * Check if two blobs overlap or touch (within tolerance).
 */
function blobsOverlap(a: DetectedBlob, b: DetectedBlob, tolerance: number = 2): boolean {
  return !(
    a.x + a.width + tolerance < b.x ||
    b.x + b.width + tolerance < a.x ||
    a.y + a.height + tolerance < b.y ||
    b.y + b.height + tolerance < a.y
  );
}

/**
 * Merge overlapping blobs into a single blob.
 * Confidence is computed as weighted average based on area.
 */
function mergeBlobs(blobs: DetectedBlob[]): DetectedBlob {
  const minX = Math.min(...blobs.map(b => b.x));
  const minY = Math.min(...blobs.map(b => b.y));
  const maxX = Math.max(...blobs.map(b => b.x + b.width));
  const maxY = Math.max(...blobs.map(b => b.y + b.height));

  const width = maxX - minX;
  const height = maxY - minY;
  const totalArea = blobs.reduce((sum, b) => sum + b.area, 0);

  // Weighted average confidence based on area
  const weightedConfidence = blobs.reduce(
    (sum, b) => sum + b.confidence * b.area,
    0
  ) / totalArea;

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

/**
 * Merge blobs from multiple tiles using Union-Find for touching blobs.
 * Used as first pass before Soft-NMS to merge blobs that are clearly the same object.
 */
function mergeTouchingBlobs(
  allBlobs: DetectedBlob[],
  options: BlobDetectionOptions
): DetectedBlob[] {
  if (allBlobs.length === 0) return [];

  // Build overlap groups using Union-Find approach
  const parent = new Array(allBlobs.length).fill(0).map((_, i) => i);

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

  // Check all pairs for overlap (touching within 2px tolerance)
  for (let i = 0; i < allBlobs.length; i++) {
    for (let j = i + 1; j < allBlobs.length; j++) {
      if (blobsOverlap(allBlobs[i], allBlobs[j], 2)) {
        union(i, j);
      }
    }
  }

  // Group blobs by their root
  const groups = new Map<number, DetectedBlob[]>();
  for (let i = 0; i < allBlobs.length; i++) {
    const root = find(i);
    if (!groups.has(root)) {
      groups.set(root, []);
    }
    groups.get(root)!.push(allBlobs[i]);
  }

  // Merge each group into a single blob
  const mergedBlobs: DetectedBlob[] = [];
  for (const group of groups.values()) {
    const merged = mergeBlobs(group);

    // Re-filter by size constraints after merging
    if (
      merged.width >= options.minWidth &&
      merged.width <= options.maxWidth &&
      merged.height >= options.minHeight &&
      merged.height <= options.maxHeight
    ) {
      mergedBlobs.push(merged);
    }
  }

  return mergedBlobs;
}

/**
 * Merge blobs from multiple tiles, handling blobs that span tile boundaries.
 * Uses Union-Find for touching blobs, then optionally Soft-NMS for overlapping detections.
 */
function mergeTileResults(
  allBlobs: DetectedBlob[],
  options: BlobDetectionOptions
): DetectedBlob[] {
  if (allBlobs.length === 0) return [];

  // First pass: merge touching blobs using Union-Find
  let mergedBlobs = mergeTouchingBlobs(allBlobs, options);

  // Second pass: apply Soft-NMS to handle overlapping detections
  if (options.useSoftNMS !== false) {
    const softNMSOptions: SoftNMSOptions = {
      sigma: options.softNMSSigma ?? 0.5,
      scoreThreshold: options.softNMSThreshold ?? 0.1,
      iouThreshold: 0.3,
      method: 'gaussian',
    };
    mergedBlobs = softNMS(mergedBlobs, softNMSOptions);
  } else {
    // Sort by area (largest first) if not using Soft-NMS
    mergedBlobs.sort((a, b) => b.area - a.area);
  }

  return mergedBlobs;
}

/**
 * Detect blobs in an image using parallel processing.
 *
 * @param imageBitmap - The image to process
 * @param options - Detection options (merged with defaults)
 * @returns Promise resolving to array of detected blobs
 */
export async function detectBlobs(
  imageBitmap: ImageBitmap,
  options: Partial<BlobDetectionOptions> = {}
): Promise<DetectedBlob[]> {
  // Merge with defaults
  const opts: BlobDetectionOptions = {
    ...DEFAULT_DETECTION_OPTIONS,
    ...options,
  };

  const width = imageBitmap.width;
  const height = imageBitmap.height;

  // Get full image data
  let imageData = getImageData(imageBitmap);

  // Apply CLAHE preprocessing if enabled
  if (opts.useCLAHE) {
    imageData = applyCLAHEToImageData(imageData, {
      clipLimit: opts.claheClipLimit ?? 2.0,
      tileGridSize: opts.claheTileSize ?? 8,
    });
  }

  // Compute global threshold if not provided
  let globalThreshold = opts.threshold;
  if (globalThreshold === undefined) {
    // Try GPU for grayscale conversion
    const gpu = opts.useGPU ? getGPUProcessor() : null;
    let grayscale: Uint8Array;

    if (gpu) {
      grayscale = gpu.toGrayscale(imageData);
    } else {
      // CPU fallback for grayscale
      grayscale = new Uint8Array(width * height);
      const data = imageData.data;
      for (let i = 0; i < grayscale.length; i++) {
        const idx = i * 4;
        grayscale[i] = Math.round(
          0.299 * data[idx] +
          0.587 * data[idx + 1] +
          0.114 * data[idx + 2]
        );
      }
    }

    globalThreshold = computeOtsuThreshold(grayscale);
  }

  // Create options with computed threshold for workers
  const workerOpts: BlobDetectionOptions = {
    ...opts,
    threshold: globalThreshold,
  };

  // Determine worker count
  const workerCount = opts.workerCount ?? (navigator.hardwareConcurrency || 4);

  // Generate tile coordinates
  const tiles: Array<{ x: number; y: number; w: number; h: number }> = [];

  // SAHI-style tiling with fixed tile size and overlap
  if (opts.tileSize && opts.tileSize > 0) {
    const tileSize = opts.tileSize;
    const overlap = opts.tileOverlap ?? 0.2;
    const overlapPx = Math.floor(tileSize * overlap);
    const stride = tileSize - overlapPx;

    // Generate tiles with overlap
    for (let y = 0; y < height; y += stride) {
      for (let x = 0; x < width; x += stride) {
        const tw = Math.min(tileSize, width - x);
        const th = Math.min(tileSize, height - y);

        if (tw > 0 && th > 0) {
          tiles.push({ x, y, w: tw, h: th });
        }
      }
    }
  } else {
    // Legacy: worker-count-based tiling (no overlap)
    const tilesPerRow = Math.ceil(Math.sqrt(workerCount));
    const tilesPerCol = Math.ceil(workerCount / tilesPerRow);

    const tileWidth = Math.ceil(width / tilesPerRow);
    const tileHeight = Math.ceil(height / tilesPerCol);

    for (let row = 0; row < tilesPerCol; row++) {
      for (let col = 0; col < tilesPerRow; col++) {
        const tileX = col * tileWidth;
        const tileY = row * tileHeight;
        const tw = Math.min(tileWidth, width - tileX);
        const th = Math.min(tileHeight, height - tileY);

        if (tw > 0 && th > 0) {
          tiles.push({ x: tileX, y: tileY, w: tw, h: th });
        }
      }
    }
  }

  // Create worker pool and process tiles
  const workers: Worker[] = [];
  const promises: Promise<BlobWorkerResult>[] = [];

  // Process tiles with worker pool
  const actualWorkerCount = Math.min(workerCount, tiles.length);

  for (let i = 0; i < actualWorkerCount; i++) {
    const worker = new BlobDetectorWorker();
    workers.push(worker);
  }

  // Distribute tiles across workers
  for (let i = 0; i < tiles.length; i++) {
    const tile = tiles[i];
    const worker = workers[i % workers.length];

    const tileImageData = getTileImageData(
      imageData,
      tile.x,
      tile.y,
      tile.w,
      tile.h
    );

    const message: BlobWorkerMessage = {
      imageData: tileImageData,
      tileX: tile.x,
      tileY: tile.y,
      tileWidth: tile.w,
      tileHeight: tile.h,
      options: workerOpts,
    };

    const promise = new Promise<BlobWorkerResult>((resolve) => {
      const handler = (e: MessageEvent<BlobWorkerResult>) => {
        if (e.data.tileX === tile.x && e.data.tileY === tile.y) {
          worker.removeEventListener('message', handler);
          resolve(e.data);
        }
      };
      worker.addEventListener('message', handler);
      worker.postMessage(message);
    });

    promises.push(promise);
  }

  // Wait for all tiles to complete
  const results = await Promise.all(promises);

  // Terminate workers
  for (const worker of workers) {
    worker.terminate();
  }

  // Collect all blobs from all tiles
  const allBlobs: DetectedBlob[] = [];
  for (const result of results) {
    allBlobs.push(...result.blobs);
  }

  // Merge blobs that span tile boundaries
  const mergedBlobs = mergeTileResults(allBlobs, workerOpts);

  return mergedBlobs;
}

/**
 * Apply Gaussian blur using separable convolution (sync version).
 */
function applyGaussianBlurSync(
  grayscale: Uint8Array<ArrayBuffer>,
  width: number,
  height: number,
  kernelSize: number = 5
): Uint8Array<ArrayBuffer> {
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
 * Apply morphological opening (erosion + dilation) (sync version).
 */
function morphologicalOpenSync(
  binary: Uint8Array<ArrayBuffer>,
  width: number,
  height: number,
  kernelSize: number = 3
): Uint8Array<ArrayBuffer> {
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
 * Extended component stats including perimeter for circularity calculation.
 */
interface SyncComponentStats extends ComponentStats {
  perimeterCount: number;
}

/**
 * Quick synchronous detection for small images (no workers).
 * Useful for previews or when parallel overhead isn't worthwhile.
 * Pipeline matches the worker: grayscale -> blur -> threshold -> morphology -> labeling
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

/**
 * Calculate confidence score for sync detection.
 */
function calculateSyncConfidence(
  width: number,
  height: number,
  area: number,
  aspectRatio: number,
  options: BlobDetectionOptions
): number {
  const { minWidth, maxWidth, minHeight, maxHeight } = options;

  // Size fit score (0-1)
  const widthRange = maxWidth - minWidth;
  const heightRange = maxHeight - minHeight;
  const widthMid = minWidth + widthRange / 2;
  const heightMid = minHeight + heightRange / 2;

  const widthDeviation = Math.abs(width - widthMid) / (widthRange / 2);
  const heightDeviation = Math.abs(height - heightMid) / (heightRange / 2);

  const widthScore = Math.max(0, 1 - Math.min(1, widthDeviation));
  const heightScore = Math.max(0, 1 - Math.min(1, heightDeviation));
  const sizeScore = (widthScore + heightScore) / 2;

  // Aspect ratio score (0-1)
  const idealAspectRatio = 1.0;
  const aspectRatioDeviation = Math.abs(aspectRatio - idealAspectRatio);
  const aspectScore = Math.max(0, 1 - aspectRatioDeviation);

  // Area density score (0-1)
  const boundingArea = width * height;
  const densityScore = area / boundingArea;

  // Weighted combination
  const confidence = 0.5 * sizeScore + 0.3 * densityScore + 0.2 * aspectScore;

  return Math.max(0, Math.min(1, confidence));
}
