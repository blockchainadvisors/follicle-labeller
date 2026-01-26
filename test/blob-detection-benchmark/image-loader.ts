/**
 * Image loader using sharp for Node.js environment.
 */

import sharp from 'sharp';
import * as fs from 'fs';
import * as path from 'path';

export interface LoadedImage {
  imageData: ImageData;
  width: number;
  height: number;
  filePath: string;
  originalWidth: number;
  originalHeight: number;
  wasResized: boolean;
}

export interface LoadImageOptions {
  /** Maximum dimension (width or height). Image will be downscaled proportionally. Default: 2000 */
  maxDimension?: number;
}

/**
 * Load a test image and convert it to ImageData.
 * Uses sharp which has pre-built binaries for Windows.
 * Optionally downscales large images for faster processing.
 */
export async function loadTestImage(
  imagePath: string,
  options: LoadImageOptions = {}
): Promise<LoadedImage> {
  const absolutePath = path.resolve(imagePath);

  if (!fs.existsSync(absolutePath)) {
    throw new Error(`Image file not found: ${absolutePath}`);
  }

  const maxDimension = options.maxDimension ?? 2000;

  // Load image and get metadata
  let image = sharp(absolutePath);
  const metadata = await image.metadata();

  if (!metadata.width || !metadata.height) {
    throw new Error(`Could not determine image dimensions for: ${absolutePath}`);
  }

  const originalWidth = metadata.width;
  const originalHeight = metadata.height;
  let width = originalWidth;
  let height = originalHeight;
  let wasResized = false;

  // Downscale if necessary
  if (width > maxDimension || height > maxDimension) {
    const scale = maxDimension / Math.max(width, height);
    width = Math.round(width * scale);
    height = Math.round(height * scale);
    wasResized = true;

    image = image.resize(width, height, {
      fit: 'inside',
      withoutEnlargement: true,
    });
  }

  // Get raw RGBA pixel data
  const rawBuffer = await image
    .ensureAlpha() // Ensure 4 channels (RGBA)
    .raw()
    .toBuffer();

  // Create ImageData-like object
  const imageData = createImageData(rawBuffer, width, height);

  return {
    imageData,
    width,
    height,
    filePath: absolutePath,
    originalWidth,
    originalHeight,
    wasResized,
  };
}

/**
 * Create an ImageData-compatible object from raw buffer.
 * In Node.js environment, we need to polyfill ImageData.
 */
function createImageData(buffer: Buffer, width: number, height: number): ImageData {
  // Create a Uint8ClampedArray from the buffer
  const data = new Uint8ClampedArray(buffer.buffer, buffer.byteOffset, buffer.byteLength);

  // Return an object that matches ImageData interface
  return {
    data,
    width,
    height,
    colorSpace: 'srgb' as PredefinedColorSpace,
  };
}

/**
 * Scan for test images in a directory.
 */
export function findTestImages(directory: string): string[] {
  const absoluteDir = path.resolve(directory);

  if (!fs.existsSync(absoluteDir)) {
    return [];
  }

  const files = fs.readdirSync(absoluteDir);
  const imageExtensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif'];

  return files
    .filter(file => imageExtensions.includes(path.extname(file).toLowerCase()))
    .map(file => path.join(absoluteDir, file));
}
