import JSZip from 'jszip';
import { Follicle, ProjectImage, isCircle, isRectangle, isLinear } from '../types';

// Bounding box interface
interface BoundingBox {
  x: number;      // Top-left x (in image coordinates)
  y: number;      // Top-left y (in image coordinates)
  width: number;
  height: number;
}

/**
 * Calculate the bounding box for any follicle annotation.
 * For non-rectangular shapes (circle, linear), approximates with axis-aligned bounding box.
 */
export function getFollicleBoundingBox(follicle: Follicle): BoundingBox {
  if (isCircle(follicle)) {
    // Circle: bounding box is center +/- radius
    return {
      x: follicle.center.x - follicle.radius,
      y: follicle.center.y - follicle.radius,
      width: follicle.radius * 2,
      height: follicle.radius * 2,
    };
  } else if (isRectangle(follicle)) {
    // Rectangle: already has bounding box properties
    return {
      x: follicle.x,
      y: follicle.y,
      width: follicle.width,
      height: follicle.height,
    };
  } else if (isLinear(follicle)) {
    // Linear: rotated rectangle defined by centerline + half-width
    // Calculate the four corner points and find the axis-aligned bounding box
    const { startPoint, endPoint, halfWidth } = follicle;

    // Calculate direction vector and perpendicular
    const dx = endPoint.x - startPoint.x;
    const dy = endPoint.y - startPoint.y;
    const length = Math.sqrt(dx * dx + dy * dy);

    if (length === 0) {
      // Degenerate case: start and end are the same
      return {
        x: startPoint.x - halfWidth,
        y: startPoint.y - halfWidth,
        width: halfWidth * 2,
        height: halfWidth * 2,
      };
    }

    // Perpendicular unit vector (rotated 90 degrees)
    const perpX = -dy / length * halfWidth;
    const perpY = dx / length * halfWidth;

    // Four corners of the rotated rectangle
    const corners = [
      { x: startPoint.x + perpX, y: startPoint.y + perpY },
      { x: startPoint.x - perpX, y: startPoint.y - perpY },
      { x: endPoint.x + perpX, y: endPoint.y + perpY },
      { x: endPoint.x - perpX, y: endPoint.y - perpY },
    ];

    // Find bounding box
    const minX = Math.min(...corners.map(c => c.x));
    const maxX = Math.max(...corners.map(c => c.x));
    const minY = Math.min(...corners.map(c => c.y));
    const maxY = Math.max(...corners.map(c => c.y));

    return {
      x: minX,
      y: minY,
      width: maxX - minX,
      height: maxY - minY,
    };
  }

  // Fallback (should never reach here)
  throw new Error(`Unknown follicle shape: ${(follicle as Follicle).shape}`);
}

/**
 * Extract a region from an image based on a bounding box.
 * Returns a Blob containing the extracted image as PNG.
 */
export async function extractImageRegion(
  image: ProjectImage,
  boundingBox: BoundingBox
): Promise<Blob> {
  // Clamp bounding box to image bounds
  const x = Math.max(0, Math.floor(boundingBox.x));
  const y = Math.max(0, Math.floor(boundingBox.y));
  const width = Math.min(image.width - x, Math.ceil(boundingBox.width));
  const height = Math.min(image.height - y, Math.ceil(boundingBox.height));

  // Create a canvas for extraction
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;

  const ctx = canvas.getContext('2d');
  if (!ctx) {
    throw new Error('Failed to get canvas 2D context');
  }

  // Draw the region from the source image
  ctx.drawImage(
    image.imageBitmap,
    x, y, width, height,  // Source rectangle
    0, 0, width, height   // Destination rectangle
  );

  // Convert to blob
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (blob) {
          resolve(blob);
        } else {
          reject(new Error('Failed to create image blob'));
        }
      },
      'image/png'
    );
  });
}

/**
 * Extract all follicle regions from an image and package them into a ZIP file.
 * Returns a Blob containing the ZIP archive.
 */
export async function extractFolliclesToZip(
  image: ProjectImage,
  follicles: Follicle[]
): Promise<Blob> {
  const zip = new JSZip();

  // Filter follicles for this image
  const imageFollicles = follicles.filter(f => f.imageId === image.id);

  if (imageFollicles.length === 0) {
    throw new Error('No follicles to extract');
  }

  // Extract each follicle region
  for (let i = 0; i < imageFollicles.length; i++) {
    const follicle = imageFollicles[i];
    const boundingBox = getFollicleBoundingBox(follicle);

    try {
      const blob = await extractImageRegion(image, boundingBox);

      // Create a sanitized filename from the label
      const sanitizedLabel = follicle.label
        .replace(/[^a-zA-Z0-9_-]/g, '_')
        .substring(0, 50);

      const filename = `${String(i + 1).padStart(3, '0')}_${sanitizedLabel}.png`;

      zip.file(filename, blob);
    } catch (error) {
      console.error(`Failed to extract follicle ${follicle.id}:`, error);
      // Continue with other follicles
    }
  }

  // Generate the ZIP file
  return zip.generateAsync({ type: 'blob' });
}

/**
 * Extract follicles from all images and package them into a ZIP file.
 * Each image's follicles are placed in a subfolder named after the image.
 */
export async function extractAllFolliclesToZip(
  images: Map<string, ProjectImage>,
  follicles: Follicle[]
): Promise<Blob> {
  const zip = new JSZip();
  let totalExtracted = 0;

  for (const [imageId, image] of images) {
    const imageFollicles = follicles.filter(f => f.imageId === imageId);

    if (imageFollicles.length === 0) continue;

    // Create a folder for this image (use sanitized filename)
    const folderName = image.fileName
      .replace(/\.[^/.]+$/, '')  // Remove extension
      .replace(/[^a-zA-Z0-9_-]/g, '_')
      .substring(0, 50);

    const folder = zip.folder(folderName);
    if (!folder) continue;

    // Extract each follicle
    for (let i = 0; i < imageFollicles.length; i++) {
      const follicle = imageFollicles[i];
      const boundingBox = getFollicleBoundingBox(follicle);

      try {
        const blob = await extractImageRegion(image, boundingBox);

        const sanitizedLabel = follicle.label
          .replace(/[^a-zA-Z0-9_-]/g, '_')
          .substring(0, 50);

        const filename = `${String(i + 1).padStart(3, '0')}_${sanitizedLabel}.png`;

        folder.file(filename, blob);
        totalExtracted++;
      } catch (error) {
        console.error(`Failed to extract follicle ${follicle.id}:`, error);
      }
    }
  }

  if (totalExtracted === 0) {
    throw new Error('No follicles could be extracted');
  }

  return zip.generateAsync({ type: 'blob' });
}

/**
 * Trigger a download of a blob as a file.
 */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
