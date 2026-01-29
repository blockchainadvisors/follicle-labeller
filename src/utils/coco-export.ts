/**
 * COCO JSON Export Utilities
 *
 * Exports annotations in COCO (Common Objects in Context) format.
 * COCO format is widely supported and enables interoperability with
 * various ML frameworks and annotation tools.
 *
 * Format:
 * - info: Dataset description
 * - images: List of images with dimensions
 * - annotations: Bounding boxes with optional keypoints
 * - categories: Class definitions
 */

import {
  Follicle,
  ProjectImage,
  COCODataset,
  COCOImage,
  COCOAnnotation,
  COCOCategory,
  COCOExportOptions,
  isCircle,
  isRectangle,
  isLinear,
  RectangleAnnotation,
} from '../types';

/**
 * Get bounding box for any annotation type in COCO format [x, y, width, height]
 */
function getAnnotationBBox(f: Follicle): [number, number, number, number] {
  if (isCircle(f)) {
    const diameter = f.radius * 2;
    return [
      Math.round((f.center.x - f.radius) * 100) / 100,
      Math.round((f.center.y - f.radius) * 100) / 100,
      Math.round(diameter * 100) / 100,
      Math.round(diameter * 100) / 100,
    ];
  } else if (isRectangle(f)) {
    return [
      Math.round(f.x * 100) / 100,
      Math.round(f.y * 100) / 100,
      Math.round(f.width * 100) / 100,
      Math.round(f.height * 100) / 100,
    ];
  } else if (isLinear(f)) {
    // Compute axis-aligned bounding box for rotated rectangle
    const dx = f.endPoint.x - f.startPoint.x;
    const dy = f.endPoint.y - f.startPoint.y;
    const length = Math.sqrt(dx * dx + dy * dy);

    // Normal vector perpendicular to the line
    const nx = -dy / length;
    const ny = dx / length;

    // Four corners of the rotated rectangle
    const corners = [
      { x: f.startPoint.x + nx * f.halfWidth, y: f.startPoint.y + ny * f.halfWidth },
      { x: f.startPoint.x - nx * f.halfWidth, y: f.startPoint.y - ny * f.halfWidth },
      { x: f.endPoint.x + nx * f.halfWidth, y: f.endPoint.y + ny * f.halfWidth },
      { x: f.endPoint.x - nx * f.halfWidth, y: f.endPoint.y - ny * f.halfWidth },
    ];

    const xs = corners.map(c => c.x);
    const ys = corners.map(c => c.y);

    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    return [
      Math.round(minX * 100) / 100,
      Math.round(minY * 100) / 100,
      Math.round((maxX - minX) * 100) / 100,
      Math.round((maxY - minY) * 100) / 100,
    ];
  }

  return [0, 0, 0, 0];
}

/**
 * Get keypoints for a rectangle annotation with origin data
 * Returns [origin_x, origin_y, visibility, direction_x, direction_y, visibility]
 * Visibility: 0 = not labeled, 1 = labeled but occluded, 2 = labeled and visible
 */
function getAnnotationKeypoints(f: Follicle): number[] | undefined {
  if (!isRectangle(f) || !f.origin) {
    return undefined;
  }

  const origin = f.origin;
  const originX = Math.round(origin.originPoint.x * 100) / 100;
  const originY = Math.round(origin.originPoint.y * 100) / 100;

  // Calculate direction endpoint from angle and length
  const directionX = Math.round(
    (origin.originPoint.x + Math.cos(origin.directionAngle) * origin.directionLength) * 100
  ) / 100;
  const directionY = Math.round(
    (origin.originPoint.y + Math.sin(origin.directionAngle) * origin.directionLength) * 100
  ) / 100;

  // Visibility 2 = labeled and visible for both keypoints
  return [originX, originY, 2, directionX, directionY, 2];
}

/**
 * Default export options
 */
export const DEFAULT_COCO_EXPORT_OPTIONS: COCOExportOptions = {
  includeKeypoints: true,
  exportImages: false,
  categoryName: 'follicle',
};

/**
 * Export annotations in COCO JSON format
 *
 * @param images Map of project images
 * @param follicles All annotations
 * @param options Export options
 * @returns COCO format JSON string
 */
export function exportToCOCO(
  images: Map<string, ProjectImage> | ProjectImage[],
  follicles: Follicle[],
  options: COCOExportOptions = DEFAULT_COCO_EXPORT_OPTIONS
): string {
  const imageArray = Array.isArray(images) ? images : Array.from(images.values());
  const categoryName = options.categoryName || 'follicle';

  // Count annotations with keypoints
  const hasAnyKeypoints = options.includeKeypoints && follicles.some(
    f => isRectangle(f) && (f as RectangleAnnotation).origin
  );

  // Build COCO dataset
  const dataset: COCODataset = {
    info: {
      description: 'Exported from Follicle Labeller',
      version: '1.0',
      year: new Date().getFullYear(),
      date_created: new Date().toISOString(),
    },
    images: [],
    annotations: [],
    categories: [],
  };

  // Create image ID mapping (imageId -> numeric id for COCO)
  const imageIdMap = new Map<string, number>();
  let imageNumericId = 1;

  for (const image of imageArray) {
    const cocoImage: COCOImage = {
      id: imageNumericId,
      file_name: image.fileName,
      width: image.width,
      height: image.height,
    };
    dataset.images.push(cocoImage);
    imageIdMap.set(image.id, imageNumericId);
    imageNumericId++;
  }

  // Create category
  const category: COCOCategory = {
    id: 1,
    name: categoryName,
    supercategory: 'object',
  };

  // Add keypoint information if any annotations have keypoints
  if (hasAnyKeypoints) {
    category.keypoints = ['origin', 'direction'];
    category.skeleton = [[1, 2]]; // Connect origin to direction
  }

  dataset.categories.push(category);

  // Create annotations
  let annotationId = 1;

  for (const f of follicles) {
    const imageNumId = imageIdMap.get(f.imageId);
    if (imageNumId === undefined) continue;

    const bbox = getAnnotationBBox(f);
    const area = bbox[2] * bbox[3]; // width * height

    const annotation: COCOAnnotation = {
      id: annotationId,
      image_id: imageNumId,
      category_id: 1,
      bbox,
      area: Math.round(area * 100) / 100,
      iscrowd: 0,
    };

    // Add keypoints if enabled and available
    if (options.includeKeypoints) {
      const keypoints = getAnnotationKeypoints(f);
      if (keypoints) {
        annotation.keypoints = keypoints;
        annotation.num_keypoints = 2;
      }
    }

    dataset.annotations.push(annotation);
    annotationId++;
  }

  return JSON.stringify(dataset, null, 2);
}

/**
 * Export COCO JSON with images as ZIP
 *
 * @param images Map of project images
 * @param follicles All annotations
 * @param options Export options
 * @returns Promise resolving to ZIP blob
 */
export async function exportToCOCOWithImages(
  images: Map<string, ProjectImage> | ProjectImage[],
  follicles: Follicle[],
  options: COCOExportOptions = DEFAULT_COCO_EXPORT_OPTIONS
): Promise<Blob> {
  const imageArray = Array.isArray(images) ? images : Array.from(images.values());

  // Generate COCO JSON
  const cocoJson = exportToCOCO(images, follicles, options);

  // Create ZIP
  const JSZip = (await import('jszip')).default;
  const zip = new JSZip();

  // Add annotations.json
  zip.file('annotations.json', cocoJson);

  // Add images in images/ folder
  const imagesFolder = zip.folder('images');
  if (imagesFolder) {
    for (const image of imageArray) {
      imagesFolder.file(image.fileName, image.imageData);
    }
  }

  // Generate ZIP blob
  return zip.generateAsync({ type: 'blob' });
}

/**
 * Statistics from COCO export
 */
export interface COCOExportStats {
  imageCount: number;
  annotationCount: number;
  annotationsWithKeypoints: number;
  annotationsWithoutKeypoints: number;
}

/**
 * Get export statistics
 */
export function getCOCOExportStats(
  images: Map<string, ProjectImage> | ProjectImage[],
  follicles: Follicle[]
): COCOExportStats {
  const imageArray = Array.isArray(images) ? images : Array.from(images.values());

  let withKeypoints = 0;
  let withoutKeypoints = 0;

  for (const f of follicles) {
    if (isRectangle(f) && (f as RectangleAnnotation).origin) {
      withKeypoints++;
    } else {
      withoutKeypoints++;
    }
  }

  return {
    imageCount: imageArray.length,
    annotationCount: follicles.length,
    annotationsWithKeypoints: withKeypoints,
    annotationsWithoutKeypoints: withoutKeypoints,
  };
}
