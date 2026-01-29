/**
 * Standalone CLAHE processor for Node.js environment.
 * Extracted from claheProcessor.ts to avoid web dependencies.
 */

export interface CLAHEOptions {
  tileGridSize: number;
  clipLimit: number;
}

const DEFAULT_CLAHE_OPTIONS: CLAHEOptions = {
  tileGridSize: 8,
  clipLimit: 2.0,
};

/**
 * Apply CLAHE to a grayscale image.
 */
function applyCLAHE(
  grayscale: Uint8Array,
  width: number,
  height: number,
  options: Partial<CLAHEOptions> = {}
): Uint8Array {
  const opts = { ...DEFAULT_CLAHE_OPTIONS, ...options };
  const { tileGridSize, clipLimit } = opts;

  const tileWidth = Math.ceil(width / tileGridSize);
  const tileHeight = Math.ceil(height / tileGridSize);

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

      clipHistogram(histogram, clipLimit, tileWidth * tileHeight);
      const cdf = computeCDF(histogram, tileWidth * tileHeight);
      tileCDFs[ty][tx] = cdf;
    }
  }

  const result = new Uint8Array(grayscale.length);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      const pixelValue = grayscale[idx];

      const fx = (x - tileWidth / 2) / tileWidth;
      const fy = (y - tileHeight / 2) / tileHeight;

      const tx0 = Math.max(0, Math.min(tileGridSize - 1, Math.floor(fx)));
      const ty0 = Math.max(0, Math.min(tileGridSize - 1, Math.floor(fy)));
      const tx1 = Math.min(tileGridSize - 1, tx0 + 1);
      const ty1 = Math.min(tileGridSize - 1, ty0 + 1);

      const wx = Math.max(0, Math.min(1, fx - tx0));
      const wy = Math.max(0, Math.min(1, fy - ty0));

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
 */
function clipHistogram(
  histogram: Uint32Array,
  clipLimit: number,
  tilePixelCount: number
): void {
  const actualClipLimit = Math.max(1, Math.floor(clipLimit * tilePixelCount / 256));

  let excess = 0;
  for (let i = 0; i < 256; i++) {
    if (histogram[i] > actualClipLimit) {
      excess += histogram[i] - actualClipLimit;
      histogram[i] = actualClipLimit;
    }
  }

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
 * Compute cumulative distribution function for equalization lookup.
 */
function computeCDF(histogram: Uint32Array, totalPixels: number): Uint8Array {
  const cdf = new Uint8Array(256);
  let cumulative = 0;

  let minValue = 0;
  for (let i = 0; i < 256; i++) {
    if (histogram[i] > 0) {
      minValue = histogram[i];
      break;
    }
  }

  const scale = 255 / Math.max(1, totalPixels - minValue);

  for (let i = 0; i < 256; i++) {
    cumulative += histogram[i];
    cdf[i] = Math.round(Math.max(0, (cumulative - minValue) * scale));
  }

  return cdf;
}

/**
 * Apply CLAHE to RGBA ImageData.
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
  const resultData = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < enhanced.length; i++) {
    const idx = i * 4;
    const value = enhanced[i];
    resultData[idx] = value;
    resultData[idx + 1] = value;
    resultData[idx + 2] = value;
    resultData[idx + 3] = 255;
  }

  return {
    data: resultData,
    width,
    height,
    colorSpace: 'srgb' as PredefinedColorSpace,
  };
}
