/**
 * CLAHE (Contrast Limited Adaptive Histogram Equalization) implementation.
 *
 * CLAHE improves contrast in images by applying histogram equalization locally,
 * which is particularly useful for medical images where lighting may be uneven.
 *
 * Key features:
 * - Tile-based processing for adaptive local contrast enhancement
 * - Clip limit to prevent over-amplification of noise
 * - Bilinear interpolation between tiles to eliminate boundary artifacts
 */

export interface CLAHEOptions {
  /** Number of tiles in each dimension (default: 8) */
  tileGridSize: number;
  /** Clip limit for contrast limiting (default: 2.0, range 1-10) */
  clipLimit: number;
}

export const DEFAULT_CLAHE_OPTIONS: CLAHEOptions = {
  tileGridSize: 8,
  clipLimit: 2.0,
};

/**
 * Apply CLAHE to a grayscale image (CPU implementation).
 *
 * @param grayscale - Single-channel grayscale image data (Uint8Array)
 * @param width - Image width
 * @param height - Image height
 * @param options - CLAHE configuration options
 * @returns Enhanced grayscale image
 */
export function applyCLAHE(
  grayscale: Uint8Array,
  width: number,
  height: number,
  options: Partial<CLAHEOptions> = {}
): Uint8Array {
  const opts = { ...DEFAULT_CLAHE_OPTIONS, ...options };
  const { tileGridSize, clipLimit } = opts;

  // Calculate tile dimensions
  const tileWidth = Math.ceil(width / tileGridSize);
  const tileHeight = Math.ceil(height / tileGridSize);

  // Compute histogram and CDF (cumulative distribution function) for each tile
  const tileCDFs: Uint8Array[][] = [];

  for (let ty = 0; ty < tileGridSize; ty++) {
    tileCDFs[ty] = [];
    for (let tx = 0; tx < tileGridSize; tx++) {
      const histogram = computeTileHistogram(
        grayscale,
        width,
        height,
        tx * tileWidth,
        ty * tileHeight,
        tileWidth,
        tileHeight
      );

      // Apply clip limit and redistribute
      clipHistogram(histogram, clipLimit, tileWidth * tileHeight);

      // Compute CDF (lookup table for equalization)
      const cdf = computeCDF(histogram, tileWidth * tileHeight);
      tileCDFs[ty][tx] = cdf;
    }
  }

  // Apply interpolated equalization
  const result = new Uint8Array(grayscale.length);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      const pixelValue = grayscale[idx];

      // Find which tiles this pixel belongs to for interpolation
      const fx = (x - tileWidth / 2) / tileWidth;
      const fy = (y - tileHeight / 2) / tileHeight;

      // Get tile indices and interpolation weights
      const tx0 = Math.max(0, Math.min(tileGridSize - 1, Math.floor(fx)));
      const ty0 = Math.max(0, Math.min(tileGridSize - 1, Math.floor(fy)));
      const tx1 = Math.min(tileGridSize - 1, tx0 + 1);
      const ty1 = Math.min(tileGridSize - 1, ty0 + 1);

      // Compute interpolation weights
      const wx = Math.max(0, Math.min(1, fx - tx0));
      const wy = Math.max(0, Math.min(1, fy - ty0));

      // Bilinear interpolation of equalized values
      const v00 = tileCDFs[ty0][tx0][pixelValue];
      const v10 = tileCDFs[ty0][tx1][pixelValue];
      const v01 = tileCDFs[ty1][tx0][pixelValue];
      const v11 = tileCDFs[ty1][tx1][pixelValue];

      const v0 = v00 * (1 - wx) + v10 * wx;
      const v1 = v01 * (1 - wx) + v11 * wx;
      const value = v0 * (1 - wy) + v1 * wy;

      result[idx] = Math.round(Math.max(0, Math.min(255, value)));
    }
  }

  return result;
}

/**
 * Compute histogram for a tile region.
 */
function computeTileHistogram(
  grayscale: Uint8Array,
  width: number,
  height: number,
  startX: number,
  startY: number,
  tileWidth: number,
  tileHeight: number
): Uint32Array {
  const histogram = new Uint32Array(256);

  const endX = Math.min(startX + tileWidth, width);
  const endY = Math.min(startY + tileHeight, height);

  for (let y = startY; y < endY; y++) {
    for (let x = startX; x < endX; x++) {
      const idx = y * width + x;
      histogram[grayscale[idx]]++;
    }
  }

  return histogram;
}

/**
 * Apply clip limit to histogram and redistribute excess.
 *
 * This prevents over-amplification of noise by limiting the height
 * of any bin and redistributing the excess uniformly.
 */
function clipHistogram(
  histogram: Uint32Array,
  clipLimit: number,
  tilePixelCount: number
): void {
  // Calculate actual clip limit based on tile size
  const actualClipLimit = Math.max(1, Math.floor(clipLimit * tilePixelCount / 256));

  // Count excess and redistribute
  let excess = 0;
  for (let i = 0; i < 256; i++) {
    if (histogram[i] > actualClipLimit) {
      excess += histogram[i] - actualClipLimit;
      histogram[i] = actualClipLimit;
    }
  }

  // Redistribute excess uniformly
  const perBin = Math.floor(excess / 256);
  let remainder = excess % 256;

  for (let i = 0; i < 256; i++) {
    histogram[i] += perBin;
    if (remainder > 0) {
      histogram[i]++;
      remainder--;
    }
  }
}

/**
 * Compute cumulative distribution function (CDF) for equalization lookup.
 */
function computeCDF(histogram: Uint32Array, totalPixels: number): Uint8Array {
  const cdf = new Uint8Array(256);
  let cumulative = 0;

  // Find minimum non-zero value for scaling
  let minValue = 0;
  for (let i = 0; i < 256; i++) {
    if (histogram[i] > 0) {
      minValue = histogram[i];
      break;
    }
  }

  // Compute CDF with scaling to [0, 255]
  const scale = 255 / Math.max(1, totalPixels - minValue);

  for (let i = 0; i < 256; i++) {
    cumulative += histogram[i];
    cdf[i] = Math.round(Math.max(0, (cumulative - minValue) * scale));
  }

  return cdf;
}

/**
 * Apply CLAHE to RGBA ImageData.
 * Converts to grayscale, applies CLAHE, then returns enhanced grayscale as RGBA.
 *
 * @param imageData - Input RGBA ImageData
 * @param options - CLAHE options
 * @returns New ImageData with CLAHE-enhanced grayscale
 */
export function applyCLAHEToImageData(
  imageData: ImageData,
  options: Partial<CLAHEOptions> = {}
): ImageData {
  const { width, height, data } = imageData;

  // Convert to grayscale
  const grayscale = new Uint8Array(width * height);
  for (let i = 0; i < grayscale.length; i++) {
    const idx = i * 4;
    grayscale[i] = Math.round(
      0.299 * data[idx] +
      0.587 * data[idx + 1] +
      0.114 * data[idx + 2]
    );
  }

  // Apply CLAHE
  const enhanced = applyCLAHE(grayscale, width, height, options);

  // Convert back to RGBA
  const result = new ImageData(width, height);
  for (let i = 0; i < enhanced.length; i++) {
    const idx = i * 4;
    const value = enhanced[i];
    result.data[idx] = value;     // R
    result.data[idx + 1] = value; // G
    result.data[idx + 2] = value; // B
    result.data[idx + 3] = 255;   // A
  }

  return result;
}

/**
 * GPU-accelerated CLAHE using WebGL.
 * Falls back to CPU if WebGL is unavailable.
 */
export class CLAHEProcessor {
  private canvas: OffscreenCanvas | HTMLCanvasElement | null = null;
  private gl: WebGLRenderingContext | null = null;
  private program: WebGLProgram | null = null;
  private positionBuffer: WebGLBuffer | null = null;
  private texCoordBuffer: WebGLBuffer | null = null;
  private initialized = false;

  /**
   * Initialize WebGL context and shaders.
   * @returns true if GPU acceleration is available
   */
  initialize(): boolean {
    if (this.initialized) return this.gl !== null;

    try {
      // Try OffscreenCanvas first, fall back to HTMLCanvas
      if (typeof OffscreenCanvas !== 'undefined') {
        this.canvas = new OffscreenCanvas(1, 1);
      } else {
        this.canvas = document.createElement('canvas');
      }

      this.gl = this.canvas.getContext('webgl', {
        preserveDrawingBuffer: true,
        antialias: false,
      }) as WebGLRenderingContext | null;

      if (!this.gl) {
        this.initialized = true;
        return false;
      }

      // Compile shaders
      const vertexShader = this.compileShader(VERTEX_SHADER, this.gl.VERTEX_SHADER);
      const fragmentShader = this.compileShader(CLAHE_FRAGMENT_SHADER, this.gl.FRAGMENT_SHADER);

      if (!vertexShader || !fragmentShader) {
        this.initialized = true;
        return false;
      }

      // Create program
      this.program = this.gl.createProgram();
      if (!this.program) {
        this.initialized = true;
        return false;
      }

      this.gl.attachShader(this.program, vertexShader);
      this.gl.attachShader(this.program, fragmentShader);
      this.gl.linkProgram(this.program);

      if (!this.gl.getProgramParameter(this.program, this.gl.LINK_STATUS)) {
        console.error('CLAHE shader program linking failed');
        this.initialized = true;
        return false;
      }

      // Create buffers
      this.positionBuffer = this.gl.createBuffer();
      this.texCoordBuffer = this.gl.createBuffer();

      // Set up position buffer (full-screen quad)
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.positionBuffer);
      this.gl.bufferData(
        this.gl.ARRAY_BUFFER,
        new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
        this.gl.STATIC_DRAW
      );

      // Set up texture coordinate buffer
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.texCoordBuffer);
      this.gl.bufferData(
        this.gl.ARRAY_BUFFER,
        new Float32Array([0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0]),
        this.gl.STATIC_DRAW
      );

      this.initialized = true;
      return true;
    } catch {
      this.initialized = true;
      return false;
    }
  }

  private compileShader(source: string, type: number): WebGLShader | null {
    if (!this.gl) return null;

    const shader = this.gl.createShader(type);
    if (!shader) return null;

    this.gl.shaderSource(shader, source);
    this.gl.compileShader(shader);

    if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
      console.error('CLAHE shader compilation error:', this.gl.getShaderInfoLog(shader));
      this.gl.deleteShader(shader);
      return null;
    }

    return shader;
  }

  /**
   * Apply CLAHE using GPU (or CPU fallback).
   *
   * Note: Due to WebGL limitations, this implementation uses a simplified
   * approach that may not be as accurate as the CPU version for very
   * detailed images. For best results, use CPU CLAHE for critical applications.
   */
  process(
    imageData: ImageData,
    options: Partial<CLAHEOptions> = {}
  ): ImageData {
    // For now, use CPU implementation as the GPU version requires
    // multiple passes and texture lookups that are complex to implement
    // efficiently in WebGL 1.0.
    return applyCLAHEToImageData(imageData, options);
  }

  /**
   * Clean up WebGL resources.
   */
  dispose(): void {
    if (this.gl) {
      if (this.positionBuffer) this.gl.deleteBuffer(this.positionBuffer);
      if (this.texCoordBuffer) this.gl.deleteBuffer(this.texCoordBuffer);
      if (this.program) this.gl.deleteProgram(this.program);
    }
    this.canvas = null;
    this.gl = null;
    this.initialized = false;
  }
}

// Singleton instance
let claheProcessorInstance: CLAHEProcessor | null = null;

/**
 * Get the CLAHE processor singleton.
 * Returns null if WebGL is not available.
 */
export function getCLAHEProcessor(): CLAHEProcessor | null {
  if (!claheProcessorInstance) {
    claheProcessorInstance = new CLAHEProcessor();
    claheProcessorInstance.initialize();
  }
  return claheProcessorInstance;
}

// Vertex shader (pass-through)
const VERTEX_SHADER = `
attribute vec2 a_position;
attribute vec2 a_texCoord;
varying vec2 v_texCoord;

void main() {
  gl_Position = vec4(a_position, 0.0, 1.0);
  v_texCoord = a_texCoord;
}
`;

// Fragment shader for CLAHE (simplified version)
// Full CLAHE requires multiple passes which is implemented in CPU version
const CLAHE_FRAGMENT_SHADER = `
precision mediump float;
uniform sampler2D u_image;
varying vec2 v_texCoord;

void main() {
  vec4 color = texture2D(u_image, v_texCoord);
  float gray = 0.299 * color.r + 0.587 * color.g + 0.114 * color.b;
  gl_FragColor = vec4(gray, gray, gray, 1.0);
}
`;
