/**
 * COCO JSON Import Utilities
 *
 * Imports annotations from COCO (Common Objects in Context) format.
 * Supports both bounding box detection annotations and keypoint annotations.
 */

import {
  Follicle,
  RectangleAnnotation,
  COCODataset,
  COCOAnnotation,
  FollicleOrigin,
} from '../types';
import { generateId } from './id-generator';

/**
 * Result of COCO import validation
 */
export interface COCOValidationResult {
  valid: boolean;
  error?: string;
  dataset?: COCODataset;
  stats?: {
    imageCount: number;
    annotationCount: number;
    annotationsWithKeypoints: number;
    categoryNames: string[];
  };
}

/**
 * Options for COCO import
 */
export interface COCOImportOptions {
  /** Target image ID to assign to imported annotations */
  targetImageId: string;
  /** Whether to import keypoints as origin/direction */
  importKeypoints?: boolean;
  /** Filter by specific image filename (optional) */
  filterByImage?: string;
}

/**
 * Validate COCO JSON data
 */
export function validateCOCOData(json: string): COCOValidationResult {
  try {
    const data = JSON.parse(json);

    // Check required fields
    if (!data.images || !Array.isArray(data.images)) {
      return { valid: false, error: 'Invalid COCO format: missing "images" array' };
    }
    if (!data.annotations || !Array.isArray(data.annotations)) {
      return { valid: false, error: 'Invalid COCO format: missing "annotations" array' };
    }
    if (!data.categories || !Array.isArray(data.categories)) {
      return { valid: false, error: 'Invalid COCO format: missing "categories" array' };
    }

    // Validate images have required fields
    for (const img of data.images) {
      if (typeof img.id !== 'number' || typeof img.file_name !== 'string') {
        return { valid: false, error: 'Invalid COCO format: images must have id and file_name' };
      }
    }

    // Validate annotations have required fields
    for (const ann of data.annotations) {
      if (typeof ann.id !== 'number' || typeof ann.image_id !== 'number') {
        return { valid: false, error: 'Invalid COCO format: annotations must have id and image_id' };
      }
      if (!ann.bbox || !Array.isArray(ann.bbox) || ann.bbox.length !== 4) {
        return { valid: false, error: 'Invalid COCO format: annotations must have bbox [x, y, w, h]' };
      }
    }

    // Count annotations with keypoints
    const annotationsWithKeypoints = data.annotations.filter(
      (ann: COCOAnnotation) => ann.keypoints && ann.keypoints.length >= 6
    ).length;

    // Get category names
    const categoryNames = data.categories.map((cat: { name: string }) => cat.name);

    return {
      valid: true,
      dataset: data as COCODataset,
      stats: {
        imageCount: data.images.length,
        annotationCount: data.annotations.length,
        annotationsWithKeypoints,
        categoryNames,
      },
    };
  } catch (e) {
    return {
      valid: false,
      error: `Failed to parse JSON: ${e instanceof Error ? e.message : 'Unknown error'}`,
    };
  }
}

/**
 * Convert COCO keypoints to FollicleOrigin
 * COCO keypoints format: [x1, y1, v1, x2, y2, v2, ...]
 * where v = 0: not labeled, 1: occluded, 2: visible
 */
function keypointsToOrigin(keypoints: number[]): FollicleOrigin | undefined {
  if (!keypoints || keypoints.length < 6) {
    return undefined;
  }

  const [originX, originY, v1, directionX, directionY, v2] = keypoints;

  // Only use keypoints that are labeled (v > 0)
  if (v1 === 0 || v2 === 0) {
    return undefined;
  }

  // Calculate direction angle and length
  const dx = directionX - originX;
  const dy = directionY - originY;
  const directionLength = Math.sqrt(dx * dx + dy * dy);
  const directionAngle = Math.atan2(dy, dx);

  return {
    originPoint: { x: originX, y: originY },
    directionAngle,
    directionLength,
  };
}

/**
 * Import annotations from COCO JSON
 *
 * @param json COCO JSON string
 * @param options Import options
 * @returns Array of imported follicles
 */
export function importFromCOCO(
  json: string,
  options: COCOImportOptions
): Follicle[] {
  const validation = validateCOCOData(json);
  if (!validation.valid || !validation.dataset) {
    throw new Error(validation.error || 'Invalid COCO data');
  }

  const dataset = validation.dataset;
  const { targetImageId, importKeypoints = true, filterByImage } = options;

  // Build image ID mapping if filtering by image name
  let imageIdFilter: Set<number> | null = null;
  if (filterByImage) {
    imageIdFilter = new Set();
    for (const img of dataset.images) {
      if (img.file_name === filterByImage) {
        imageIdFilter.add(img.id);
      }
    }
  }

  // Colors for imported annotations (cycles through)
  const ANNOTATION_COLORS = [
    '#FF6B6B',
    '#4ECDC4',
    '#45B7D1',
    '#96CEB4',
    '#FFEAA7',
    '#DDA0DD',
    '#98D8C8',
    '#F7DC6F',
    '#74B9FF',
    '#A29BFE',
  ];

  const now = Date.now();
  const follicles: Follicle[] = [];

  for (let i = 0; i < dataset.annotations.length; i++) {
    const ann = dataset.annotations[i];

    // Filter by image if specified
    if (imageIdFilter && !imageIdFilter.has(ann.image_id)) {
      continue;
    }

    const [x, y, width, height] = ann.bbox;

    // Convert keypoints to origin if available
    let origin: FollicleOrigin | undefined;
    if (importKeypoints && ann.keypoints) {
      origin = keypointsToOrigin(ann.keypoints);
    }

    const follicle: RectangleAnnotation = {
      id: generateId(),
      imageId: targetImageId,
      shape: 'rectangle',
      x,
      y,
      width,
      height,
      label: `COCO ${i + 1}`,
      notes: `Imported from COCO (id: ${ann.id})`,
      color: ANNOTATION_COLORS[i % ANNOTATION_COLORS.length],
      createdAt: now,
      updatedAt: now,
      origin,
    };

    follicles.push(follicle);
  }

  return follicles;
}

/**
 * Check if JSON is in COCO format (quick check)
 */
export function isCOCOFormat(json: string): boolean {
  try {
    const data = JSON.parse(json);
    return (
      Array.isArray(data.images) &&
      Array.isArray(data.annotations) &&
      Array.isArray(data.categories)
    );
  } catch {
    return false;
  }
}

/**
 * Import result with duplicate detection
 */
export interface COCOImportResult {
  /** New annotations to import */
  newAnnotations: Follicle[];
  /** Total annotations in the file */
  totalImported: number;
  /** Statistics */
  stats: {
    withKeypoints: number;
    withoutKeypoints: number;
  };
}

/**
 * Import COCO annotations with statistics
 */
export function importCOCOWithStats(
  json: string,
  options: COCOImportOptions
): COCOImportResult {
  const follicles = importFromCOCO(json, options);

  let withKeypoints = 0;
  let withoutKeypoints = 0;

  for (const f of follicles) {
    if (f.shape === 'rectangle' && (f as RectangleAnnotation).origin) {
      withKeypoints++;
    } else {
      withoutKeypoints++;
    }
  }

  return {
    newAnnotations: follicles,
    totalImported: follicles.length,
    stats: {
      withKeypoints,
      withoutKeypoints,
    },
  };
}
