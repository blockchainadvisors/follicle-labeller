/**
 * Gaussian Heatmap Generator
 *
 * Generates density heatmaps from detection centers using Gaussian kernels.
 * Supports multiple colormaps and WebGL acceleration for real-time rendering.
 */

import type { DetectedBlob, Follicle, Point } from '../types';
import { isCircle, isRectangle, isLinear } from '../types';

export type ColormapType = 'jet' | 'viridis' | 'plasma' | 'hot' | 'grayscale';

export interface HeatmapOptions {
  /** Gaussian sigma (spread) in pixels (default: 30) */
  sigma: number;
  /** Color map to use (default: 'jet') */
  colormap: ColormapType;
  /** Opacity/alpha value (0-1, default: 0.5) */
  alpha: number;
  /** Maximum value for normalization (0 = auto, default: 0) */
  maxValue: number;
  /** Scale factor for intensity (default: 1.0) */
  intensityScale: number;
}

export const DEFAULT_HEATMAP_OPTIONS: HeatmapOptions = {
  sigma: 30,
  colormap: 'jet',
  alpha: 0.5,
  maxValue: 0,
  intensityScale: 1.0,
};

/**
 * Color lookup tables for different colormaps.
 * Each entry is [r, g, b] normalized to 0-1.
 */
const COLORMAPS: Record<ColormapType, [number, number, number][]> = {
  jet: [
    [0, 0, 0.5],
    [0, 0, 1],
    [0, 0.5, 1],
    [0, 1, 1],
    [0.5, 1, 0.5],
    [1, 1, 0],
    [1, 0.5, 0],
    [1, 0, 0],
    [0.5, 0, 0],
  ],
  viridis: [
    [0.267, 0.004, 0.329],
    [0.282, 0.140, 0.457],
    [0.254, 0.265, 0.530],
    [0.206, 0.371, 0.553],
    [0.163, 0.471, 0.558],
    [0.127, 0.566, 0.550],
    [0.135, 0.658, 0.517],
    [0.267, 0.749, 0.440],
    [0.477, 0.821, 0.318],
    [0.741, 0.873, 0.150],
    [0.993, 0.906, 0.144],
  ],
  plasma: [
    [0.050, 0.030, 0.527],
    [0.294, 0.012, 0.615],
    [0.492, 0.012, 0.658],
    [0.658, 0.138, 0.618],
    [0.797, 0.275, 0.473],
    [0.899, 0.392, 0.322],
    [0.967, 0.520, 0.169],
    [0.988, 0.652, 0.039],
    [0.940, 0.975, 0.131],
  ],
  hot: [
    [0, 0, 0],
    [0.33, 0, 0],
    [0.67, 0, 0],
    [1, 0, 0],
    [1, 0.33, 0],
    [1, 0.67, 0],
    [1, 1, 0],
    [1, 1, 0.5],
    [1, 1, 1],
  ],
  grayscale: [
    [0, 0, 0],
    [1, 1, 1],
  ],
};

/**
 * Interpolate color from colormap based on normalized value (0-1).
 */
function interpolateColor(
  value: number,
  colormap: ColormapType
): [number, number, number] {
  const colors = COLORMAPS[colormap];
  const clampedValue = Math.max(0, Math.min(1, value));

  if (clampedValue === 0) return colors[0];
  if (clampedValue === 1) return colors[colors.length - 1];

  const scaledValue = clampedValue * (colors.length - 1);
  const index = Math.floor(scaledValue);
  const t = scaledValue - index;

  const c1 = colors[index];
  const c2 = colors[Math.min(index + 1, colors.length - 1)];

  return [
    c1[0] + t * (c2[0] - c1[0]),
    c1[1] + t * (c2[1] - c1[1]),
    c1[2] + t * (c2[2] - c1[2]),
  ];
}

/**
 * Extract center points from follicle annotations.
 */
export function getFollicleCenters(follicles: Follicle[]): Point[] {
  return follicles.map(f => {
    if (isCircle(f)) {
      return f.center;
    } else if (isRectangle(f)) {
      return {
        x: f.x + f.width / 2,
        y: f.y + f.height / 2,
      };
    } else if (isLinear(f)) {
      return {
        x: (f.startPoint.x + f.endPoint.x) / 2,
        y: (f.startPoint.y + f.endPoint.y) / 2,
      };
    }
    return { x: 0, y: 0 };
  });
}

/**
 * Extract center points from detected blobs.
 */
export function getBlobCenters(blobs: DetectedBlob[]): Point[] {
  return blobs.map(b => ({
    x: b.x + b.width / 2,
    y: b.y + b.height / 2,
  }));
}

/**
 * Generate a Gaussian heatmap from center points.
 * Returns a Float32Array of intensity values.
 *
 * @param centers - Array of center points
 * @param width - Image width
 * @param height - Image height
 * @param options - Heatmap options
 * @returns Float32Array of intensity values (length = width * height)
 */
export function generateHeatmapData(
  centers: Point[],
  width: number,
  height: number,
  options: Partial<HeatmapOptions> = {}
): Float32Array {
  const opts = { ...DEFAULT_HEATMAP_OPTIONS, ...options };
  const { sigma, intensityScale } = opts;

  const heatmap = new Float32Array(width * height);

  if (centers.length === 0) {
    return heatmap;
  }

  // Precompute Gaussian coefficient
  const twoSigmaSq = 2 * sigma * sigma;
  const radius = Math.ceil(sigma * 3); // 3-sigma coverage

  // Add Gaussian contribution from each center
  for (const center of centers) {
    const cx = Math.round(center.x);
    const cy = Math.round(center.y);

    // Bounded region for this Gaussian
    const x0 = Math.max(0, cx - radius);
    const x1 = Math.min(width - 1, cx + radius);
    const y0 = Math.max(0, cy - radius);
    const y1 = Math.min(height - 1, cy + radius);

    for (let y = y0; y <= y1; y++) {
      const dy = y - cy;
      const dySquared = dy * dy;

      for (let x = x0; x <= x1; x++) {
        const dx = x - cx;
        const distSquared = dx * dx + dySquared;

        // Gaussian value
        const value = Math.exp(-distSquared / twoSigmaSq) * intensityScale;
        heatmap[y * width + x] += value;
      }
    }
  }

  return heatmap;
}

/**
 * Render heatmap data to ImageData with colormap.
 *
 * @param heatmap - Float32Array of intensity values
 * @param width - Image width
 * @param height - Image height
 * @param options - Heatmap options
 * @returns ImageData for canvas rendering
 */
export function renderHeatmapToImageData(
  heatmap: Float32Array,
  width: number,
  height: number,
  options: Partial<HeatmapOptions> = {}
): ImageData {
  const opts = { ...DEFAULT_HEATMAP_OPTIONS, ...options };
  const { colormap, alpha, maxValue } = opts;

  const imageData = new ImageData(width, height);

  // Find max value for normalization
  let max = maxValue;
  if (max === 0) {
    for (let i = 0; i < heatmap.length; i++) {
      if (heatmap[i] > max) max = heatmap[i];
    }
  }

  // Avoid division by zero
  if (max === 0) max = 1;

  // Render to ImageData
  for (let i = 0; i < heatmap.length; i++) {
    const normalizedValue = heatmap[i] / max;
    const [r, g, b] = interpolateColor(normalizedValue, colormap);

    const idx = i * 4;
    imageData.data[idx] = Math.round(r * 255);
    imageData.data[idx + 1] = Math.round(g * 255);
    imageData.data[idx + 2] = Math.round(b * 255);
    // Alpha is proportional to intensity for better blending
    imageData.data[idx + 3] = Math.round(normalizedValue * alpha * 255);
  }

  return imageData;
}

/**
 * Generate and render a complete heatmap from center points.
 *
 * @param centers - Array of center points
 * @param width - Image width
 * @param height - Image height
 * @param options - Heatmap options
 * @returns ImageData for canvas rendering
 */
export function generateHeatmap(
  centers: Point[],
  width: number,
  height: number,
  options: Partial<HeatmapOptions> = {}
): ImageData {
  const heatmapData = generateHeatmapData(centers, width, height, options);
  return renderHeatmapToImageData(heatmapData, width, height, options);
}

/**
 * HeatmapRenderer class for efficient incremental updates.
 * Caches the heatmap data and only re-renders the colormap when options change.
 */
export class HeatmapRenderer {
  private heatmapData: Float32Array | null = null;
  private width = 0;
  private height = 0;
  private canvas: OffscreenCanvas | null = null;
  private ctx: OffscreenCanvasRenderingContext2D | null = null;

  /**
   * Update the heatmap data from new centers.
   */
  updateData(
    centers: Point[],
    width: number,
    height: number,
    options: Partial<HeatmapOptions> = {}
  ): void {
    this.width = width;
    this.height = height;
    this.heatmapData = generateHeatmapData(centers, width, height, options);

    // Recreate canvas if size changed
    if (!this.canvas || this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas = new OffscreenCanvas(width, height);
      this.ctx = this.canvas.getContext('2d');
    }
  }

  /**
   * Render the cached heatmap data with given options.
   * Useful for changing colormap/alpha without recalculating Gaussians.
   */
  render(options: Partial<HeatmapOptions> = {}): ImageBitmap | null {
    if (!this.heatmapData || !this.ctx || !this.canvas) {
      return null;
    }

    const imageData = renderHeatmapToImageData(
      this.heatmapData,
      this.width,
      this.height,
      options
    );

    this.ctx.putImageData(imageData, 0, 0);

    // Return as ImageBitmap for efficient canvas drawing
    return this.canvas.transferToImageBitmap();
  }

  /**
   * Get the raw heatmap intensity data.
   */
  getData(): Float32Array | null {
    return this.heatmapData;
  }

  /**
   * Clear cached data.
   */
  clear(): void {
    this.heatmapData = null;
    this.canvas = null;
    this.ctx = null;
    this.width = 0;
    this.height = 0;
  }
}

// Singleton renderer
let rendererInstance: HeatmapRenderer | null = null;

/**
 * Get the shared heatmap renderer instance.
 */
export function getHeatmapRenderer(): HeatmapRenderer {
  if (!rendererInstance) {
    rendererInstance = new HeatmapRenderer();
  }
  return rendererInstance;
}
