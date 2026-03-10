/**
 * YOLO Detection Export Utilities
 *
 * Exports rectangle annotations in YOLO detection format for training
 * YOLO11 object detection models.
 *
 * YOLO Detection Format:
 * <class> <x_center> <y_center> <width> <height>
 *
 * Where:
 * - Class is always 0 (follicle)
 * - All coordinates are normalized to image dimensions (0-1)
 * - x_center, y_center = center of bounding box
 * - width, height = size of bounding box
 */

import {
  RectangleAnnotation,
  ProjectImage,
  Follicle,
  isRectangle,
} from '../types';

/**
 * Configuration for detection dataset export
 */
export interface DetectionExportConfig {
  /** Train/validation split ratio (0-1, default 0.8 = 80% train) */
  trainSplit: number;
  /** Whether to shuffle before splitting (default true) */
  shuffle: boolean;
  /** Tile size for large images (default 1024). Set to 0 to disable tiling. */
  tileSize: number;
  /** Minimum overlap between tiles in pixels (default 64) */
  tileOverlap: number;
  /** Minimum fraction of annotation that must be in tile to include it (default 0.5) */
  minAnnotationOverlap: number;
}

export const DEFAULT_DETECTION_EXPORT_CONFIG: DetectionExportConfig = {
  trainSplit: 0.8,
  shuffle: true,
  tileSize: 1024,
  tileOverlap: 64,
  minAnnotationOverlap: 0.5,
};

/**
 * Generate YOLO detection label line for an annotation relative to a tile.
 *
 * @param annotation Rectangle annotation (in original image coordinates)
 * @param tileX Tile X offset in original image
 * @param tileY Tile Y offset in original image
 * @param tileWidth Tile width
 * @param tileHeight Tile height
 * @returns YOLO format label string or null if annotation doesn't fit
 */
function generateTileLabel(
  annotation: RectangleAnnotation,
  tileX: number,
  tileY: number,
  tileWidth: number,
  tileHeight: number,
  minOverlap: number
): string | null {
  // Calculate annotation bounds
  const annLeft = annotation.x;
  const annTop = annotation.y;
  const annRight = annotation.x + annotation.width;
  const annBottom = annotation.y + annotation.height;

  // Calculate tile bounds
  const tileRight = tileX + tileWidth;
  const tileBottom = tileY + tileHeight;

  // Check if annotation intersects tile
  if (annRight <= tileX || annLeft >= tileRight || annBottom <= tileY || annTop >= tileBottom) {
    return null; // No intersection
  }

  // Calculate intersection
  const intersectLeft = Math.max(annLeft, tileX);
  const intersectTop = Math.max(annTop, tileY);
  const intersectRight = Math.min(annRight, tileRight);
  const intersectBottom = Math.min(annBottom, tileBottom);

  const intersectArea = (intersectRight - intersectLeft) * (intersectBottom - intersectTop);
  const annotationArea = annotation.width * annotation.height;

  // Check if enough of annotation is in tile
  if (intersectArea / annotationArea < minOverlap) {
    return null;
  }

  // Calculate clipped annotation in tile coordinates
  const clippedX = intersectLeft - tileX;
  const clippedY = intersectTop - tileY;
  const clippedWidth = intersectRight - intersectLeft;
  const clippedHeight = intersectBottom - intersectTop;

  // Convert to YOLO format (normalized center + size)
  const classId = 0;
  const xCenter = (clippedX + clippedWidth / 2) / tileWidth;
  const yCenter = (clippedY + clippedHeight / 2) / tileHeight;
  const normWidth = clippedWidth / tileWidth;
  const normHeight = clippedHeight / tileHeight;

  // Clamp to valid range
  const clampedXCenter = Math.max(0, Math.min(1, xCenter));
  const clampedYCenter = Math.max(0, Math.min(1, yCenter));
  const clampedWidth = Math.max(0, Math.min(1, normWidth));
  const clampedHeight = Math.max(0, Math.min(1, normHeight));

  return [
    classId,
    clampedXCenter.toFixed(6),
    clampedYCenter.toFixed(6),
    clampedWidth.toFixed(6),
    clampedHeight.toFixed(6),
  ].join(' ');
}

/**
 * Generate YOLO detection label line for an annotation.
 *
 * Format: <class> <x_center> <y_center> <width> <height>
 *
 * @param annotation Rectangle annotation
 * @param imageWidth Source image width
 * @param imageHeight Source image height
 * @returns YOLO format label string
 */
export function generateDetectionLabel(
  annotation: RectangleAnnotation,
  imageWidth: number,
  imageHeight: number
): string {
  const classId = 0; // Single class: follicle

  // Calculate normalized center and size
  const xCenter = (annotation.x + annotation.width / 2) / imageWidth;
  const yCenter = (annotation.y + annotation.height / 2) / imageHeight;
  const width = annotation.width / imageWidth;
  const height = annotation.height / imageHeight;

  // Clamp to valid range (0-1)
  const clampedXCenter = Math.max(0, Math.min(1, xCenter));
  const clampedYCenter = Math.max(0, Math.min(1, yCenter));
  const clampedWidth = Math.max(0, Math.min(1, width));
  const clampedHeight = Math.max(0, Math.min(1, height));

  return [
    classId,
    clampedXCenter.toFixed(6),
    clampedYCenter.toFixed(6),
    clampedWidth.toFixed(6),
    clampedHeight.toFixed(6),
  ].join(' ');
}

/**
 * Generate data.yaml config for YOLO detection training.
 */
export function generateDetectionDataYaml(): string {
  return `# YOLO Detection Dataset Configuration
# Generated by Follicle Labeller
# For YOLO11 detection training

path: .
train: images/train
val: images/val

# Classes
nc: 1
names:
  0: follicle
`;
}

/**
 * Shuffle array in place using Fisher-Yates algorithm.
 */
function shuffleArray<T>(array: T[]): T[] {
  const result = [...array];
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

/**
 * Export result containing all files for the YOLO detection dataset.
 */
export interface DetectionDatasetFile {
  /** Path within the dataset (e.g., "images/train/img_001.jpg") */
  path: string;
  /** File content (blob for images, string for text files) */
  content: Blob | string;
}

/**
 * Statistics about the exported dataset.
 */
export interface DetectionDatasetStats {
  totalAnnotations: number;
  totalImages: number;
  trainImages: number;
  valImages: number;
  trainAnnotations: number;
  valAnnotations: number;
  tilesGenerated?: number;
}

/**
 * Represents a tile extracted from an image.
 */
interface Tile {
  x: number;
  y: number;
  width: number;
  height: number;
  imageId: string;
  annotations: RectangleAnnotation[];
  labels: string[];
}

/**
 * Group annotations by image ID.
 */
function groupAnnotationsByImage(
  annotations: RectangleAnnotation[]
): Map<string, RectangleAnnotation[]> {
  const grouped = new Map<string, RectangleAnnotation[]>();
  for (const annotation of annotations) {
    const existing = grouped.get(annotation.imageId) || [];
    existing.push(annotation);
    grouped.set(annotation.imageId, existing);
  }
  return grouped;
}

/**
 * Extract a tile from an ImageBitmap as a JPEG Blob.
 */
async function extractTileAsBlob(
  imageBitmap: ImageBitmap,
  x: number,
  y: number,
  width: number,
  height: number
): Promise<Blob> {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');

  if (!ctx) {
    throw new Error('Failed to get canvas context');
  }

  // Draw the tile portion of the image
  ctx.drawImage(imageBitmap, x, y, width, height, 0, 0, width, height);

  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (blob) {
          resolve(blob);
        } else {
          reject(new Error('Failed to create blob from canvas'));
        }
      },
      'image/jpeg',
      0.95
    );
  });
}

/**
 * Generate tiles for an image with overlap.
 */
function generateTileGrid(
  imageWidth: number,
  imageHeight: number,
  tileSize: number,
  overlap: number
): Array<{ x: number; y: number; width: number; height: number }> {
  const tiles: Array<{ x: number; y: number; width: number; height: number }> = [];

  const step = tileSize - overlap;

  for (let y = 0; y < imageHeight; y += step) {
    for (let x = 0; x < imageWidth; x += step) {
      // Calculate actual tile dimensions (may be smaller at edges)
      const tileWidth = Math.min(tileSize, imageWidth - x);
      const tileHeight = Math.min(tileSize, imageHeight - y);

      // Skip very small edge tiles (less than half tile size)
      if (tileWidth < tileSize / 2 || tileHeight < tileSize / 2) {
        continue;
      }

      tiles.push({ x, y, width: tileWidth, height: tileHeight });
    }
  }

  return tiles;
}

/**
 * Export all rectangle annotations as a YOLO detection dataset.
 *
 * For large single images, uses tiling to create multiple training samples.
 *
 * Creates a dataset structure:
 * - data.yaml (config file)
 * - images/train/*.jpg (training images/tiles)
 * - images/val/*.jpg (validation images/tiles)
 * - labels/train/*.txt (training labels)
 * - labels/val/*.txt (validation labels)
 *
 * @param images Map of project images
 * @param follicles All annotations
 * @param config Export configuration
 * @returns Object with files array and statistics
 */
export async function exportYOLODetectionDataset(
  images: Map<string, ProjectImage>,
  follicles: Follicle[],
  config: DetectionExportConfig = DEFAULT_DETECTION_EXPORT_CONFIG
): Promise<{ files: DetectionDatasetFile[]; stats: DetectionDatasetStats }> {
  const files: DetectionDatasetFile[] = [];

  // Filter to rectangle annotations only
  const rectangles = follicles.filter((f): f is RectangleAnnotation =>
    isRectangle(f)
  );

  // Group annotations by image
  const annotationsByImage = groupAnnotationsByImage(rectangles);

  // Get list of image IDs that have annotations
  let imageIds = Array.from(annotationsByImage.keys());

  const stats: DetectionDatasetStats = {
    totalAnnotations: rectangles.length,
    totalImages: imageIds.length,
    trainImages: 0,
    valImages: 0,
    trainAnnotations: 0,
    valAnnotations: 0,
    tilesGenerated: 0,
  };

  if (imageIds.length === 0) {
    // Add empty data.yaml
    files.push({
      path: 'data.yaml',
      content: generateDetectionDataYaml(),
    });
    return { files, stats };
  }

  // Add data.yaml
  files.push({
    path: 'data.yaml',
    content: generateDetectionDataYaml(),
  });

  // Collect all tiles from all images
  const allTiles: Tile[] = [];

  for (const imageId of imageIds) {
    const image = images.get(imageId);
    if (!image || !image.imageBitmap) continue;

    const imageAnnotations = annotationsByImage.get(imageId) || [];
    if (imageAnnotations.length === 0) continue;

    const imgWidth = image.imageBitmap.width;
    const imgHeight = image.imageBitmap.height;

    // Decide whether to use tiling
    const useTiling = config.tileSize > 0 &&
      (imgWidth > config.tileSize * 1.5 || imgHeight > config.tileSize * 1.5);

    if (useTiling) {
      // Generate tile grid
      const tileGrid = generateTileGrid(imgWidth, imgHeight, config.tileSize, config.tileOverlap);

      for (const tileInfo of tileGrid) {
        // Find annotations that belong to this tile
        const tileLabels: string[] = [];
        const tileAnnotations: RectangleAnnotation[] = [];

        for (const annotation of imageAnnotations) {
          const label = generateTileLabel(
            annotation,
            tileInfo.x,
            tileInfo.y,
            tileInfo.width,
            tileInfo.height,
            config.minAnnotationOverlap
          );

          if (label) {
            tileLabels.push(label);
            tileAnnotations.push(annotation);
          }
        }

        // Only include tiles that have at least one annotation
        if (tileLabels.length > 0) {
          allTiles.push({
            x: tileInfo.x,
            y: tileInfo.y,
            width: tileInfo.width,
            height: tileInfo.height,
            imageId,
            annotations: tileAnnotations,
            labels: tileLabels,
          });
        }
      }
    } else {
      // No tiling - use whole image as single "tile"
      const labels = imageAnnotations.map((annotation) =>
        generateDetectionLabel(annotation, imgWidth, imgHeight)
      );

      allTiles.push({
        x: 0,
        y: 0,
        width: imgWidth,
        height: imgHeight,
        imageId,
        annotations: imageAnnotations,
        labels,
      });
    }
  }

  stats.tilesGenerated = allTiles.length;

  if (allTiles.length === 0) {
    return { files, stats };
  }

  // Shuffle tiles if requested
  let tilesToProcess = config.shuffle ? shuffleArray(allTiles) : allTiles;

  // Split tiles into train/val
  const splitIndex = Math.max(1, Math.floor(tilesToProcess.length * config.trainSplit));
  const trainTiles = tilesToProcess.slice(0, splitIndex);
  const valTiles = tilesToProcess.slice(splitIndex);

  // Ensure we have at least one validation tile if we have multiple tiles
  if (valTiles.length === 0 && trainTiles.length > 1) {
    valTiles.push(trainTiles.pop()!);
  }

  stats.trainImages = trainTiles.length;
  stats.valImages = valTiles.length;

  // Process train tiles
  let index = 0;
  for (const tile of trainTiles) {
    const image = images.get(tile.imageId);
    if (!image || !image.imageBitmap) continue;

    const baseName = `tile_${String(index).padStart(5, '0')}`;

    // Extract tile as blob
    const tileBlob = await extractTileAsBlob(
      image.imageBitmap,
      tile.x,
      tile.y,
      tile.width,
      tile.height
    );

    files.push({
      path: `images/train/${baseName}.jpg`,
      content: tileBlob,
    });

    files.push({
      path: `labels/train/${baseName}.txt`,
      content: tile.labels.join('\n'),
    });

    stats.trainAnnotations += tile.labels.length;
    index++;
  }

  // Process val tiles
  for (const tile of valTiles) {
    const image = images.get(tile.imageId);
    if (!image || !image.imageBitmap) continue;

    const baseName = `tile_${String(index).padStart(5, '0')}`;

    // Extract tile as blob
    const tileBlob = await extractTileAsBlob(
      image.imageBitmap,
      tile.x,
      tile.y,
      tile.width,
      tile.height
    );

    files.push({
      path: `images/val/${baseName}.jpg`,
      content: tileBlob,
    });

    files.push({
      path: `labels/val/${baseName}.txt`,
      content: tile.labels.join('\n'),
    });

    stats.valAnnotations += tile.labels.length;
    index++;
  }

  return { files, stats };
}

/**
 * Create a ZIP file from the detection dataset files.
 *
 * @param files Dataset files from exportYOLODetectionDataset
 * @returns Promise resolving to ZIP blob
 */
export async function createDetectionDatasetZip(
  files: DetectionDatasetFile[]
): Promise<Blob> {
  const JSZip = (await import('jszip')).default;
  const zip = new JSZip();

  for (const file of files) {
    if (typeof file.content === 'string') {
      zip.file(file.path, file.content);
    } else {
      zip.file(file.path, file.content);
    }
  }

  return zip.generateAsync({
    type: 'blob',
    compression: 'DEFLATE',
    compressionOptions: { level: 6 },
  });
}
