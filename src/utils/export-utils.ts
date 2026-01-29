import {
  Follicle,
  FollicleExportV1,
  CircleAnnotation,
  RectangleAnnotation,
  LinearAnnotation,
  ProjectImage,
  ProjectManifestV2,
  AnnotationsFileV2,
  AnnotationExportV2,
  DetectionSettingsExport,
  isCircle,
  isRectangle,
  isLinear
} from '../types';
import { generateImageId } from '../store/projectStore';
import { generateId } from './id-generator';
import {
  exportYOLOKeypointDataset,
  createKeypointDatasetZip,
  KeypointExportConfig,
  DEFAULT_KEYPOINT_EXPORT_CONFIG,
  KeypointDatasetStats,
} from './yolo-keypoint-export';

/**
 * Enhanced annotation export with additional fields for ML training.
 */
export interface EnhancedAnnotationExport {
  id: string;
  imageId: string;
  shape: 'circle' | 'rectangle' | 'linear';
  label: string;
  notes: string;
  color: string;
  // Bounding box (all shapes)
  x: number;
  y: number;
  width: number;
  height: number;
  // Center point (normalized 0-1)
  x_center: number;
  y_center: number;
  // Additional metrics
  area: number;
  aspectRatio: number;
  confidence: number;  // 1.0 for manual annotations
  // Original shape-specific data preserved
  originalData: {
    centerX?: number;
    centerY?: number;
    radius?: number;
    startX?: number;
    startY?: number;
    endX?: number;
    endY?: number;
    halfWidth?: number;
  };
}

/**
 * Generate export JSON from follicles (annotations) - V1 format (legacy)
 */
export function generateExport(
  follicles: Follicle[],
  fileName: string,
  imageWidth: number,
  imageHeight: number
): FollicleExportV1 {
  return {
    version: '1.0',
    image: {
      fileName: fileName || 'unknown',
      width: imageWidth,
      height: imageHeight,
    },
    metadata: {
      exportedAt: new Date().toISOString(),
      applicationVersion: '1.0.0',
      annotationCount: follicles.length,
    },
    annotations: follicles.map(f => {
      if (isCircle(f)) {
        return {
          id: f.id,
          shape: 'circle' as const,
          label: f.label,
          notes: f.notes,
          color: f.color,
          centerX: Math.round(f.center.x * 100) / 100,
          centerY: Math.round(f.center.y * 100) / 100,
          radius: Math.round(f.radius * 100) / 100,
        };
      } else if (isRectangle(f)) {
        return {
          id: f.id,
          shape: 'rectangle' as const,
          label: f.label,
          notes: f.notes,
          color: f.color,
          x: Math.round(f.x * 100) / 100,
          y: Math.round(f.y * 100) / 100,
          width: Math.round(f.width * 100) / 100,
          height: Math.round(f.height * 100) / 100,
          // Include origin data if set
          ...(f.origin && {
            originX: Math.round(f.origin.originPoint.x * 100) / 100,
            originY: Math.round(f.origin.originPoint.y * 100) / 100,
            directionAngle: Math.round(f.origin.directionAngle * 10000) / 10000,
            directionLength: Math.round(f.origin.directionLength * 100) / 100,
          }),
        };
      } else {
        // Linear shape
        return {
          id: f.id,
          shape: 'linear' as const,
          label: f.label,
          notes: f.notes,
          color: f.color,
          startX: Math.round(f.startPoint.x * 100) / 100,
          startY: Math.round(f.startPoint.y * 100) / 100,
          endX: Math.round(f.endPoint.x * 100) / 100,
          endY: Math.round(f.endPoint.y * 100) / 100,
          halfWidth: Math.round(f.halfWidth * 100) / 100,
        };
      }
    }),
  };
}

/**
 * Generate export data for V2 multi-image format
 */
export function generateExportV2(
  images: ProjectImage[],
  follicles: Follicle[],
  globalSettings?: DetectionSettingsExport,
  imageSettingsOverrides?: Map<string, Partial<DetectionSettingsExport>>
): {
  manifest: ProjectManifestV2;
  annotations: AnnotationsFileV2;
  imageList: Array<{ id: string; fileName: string; data: ArrayBuffer }>;
} {
  const manifest: ProjectManifestV2 = {
    version: '2.0',
    metadata: {
      exportedAt: new Date().toISOString(),
      applicationVersion: '2.0.0',
      imageCount: images.length,
      annotationCount: follicles.length,
    },
    images: images.map(img => {
      const entry: ProjectManifestV2['images'][0] = {
        id: img.id,
        fileName: img.fileName,
        archiveFileName: `${img.id}-${img.fileName}`,
        width: img.width,
        height: img.height,
        sortOrder: img.sortOrder,
        viewport: img.viewport,
      };
      // Add per-image settings if present
      const override = imageSettingsOverrides?.get(img.id);
      if (override && Object.keys(override).length > 0) {
        entry.detectionSettings = override;
      }
      return entry;
    }),
    // Add global settings if present
    ...(globalSettings && {
      settings: { detection: globalSettings }
    }),
  };

  const annotations: AnnotationsFileV2 = {
    annotations: follicles.map(f => {
      const base: Partial<AnnotationExportV2> = {
        id: f.id,
        imageId: f.imageId,
        label: f.label,
        notes: f.notes,
        color: f.color,
      };

      if (isCircle(f)) {
        return {
          ...base,
          shape: 'circle' as const,
          centerX: Math.round(f.center.x * 100) / 100,
          centerY: Math.round(f.center.y * 100) / 100,
          radius: Math.round(f.radius * 100) / 100,
        } as AnnotationExportV2;
      } else if (isRectangle(f)) {
        return {
          ...base,
          shape: 'rectangle' as const,
          x: Math.round(f.x * 100) / 100,
          y: Math.round(f.y * 100) / 100,
          width: Math.round(f.width * 100) / 100,
          height: Math.round(f.height * 100) / 100,
          // Include origin data if set
          ...(f.origin && {
            originX: Math.round(f.origin.originPoint.x * 100) / 100,
            originY: Math.round(f.origin.originPoint.y * 100) / 100,
            directionAngle: Math.round(f.origin.directionAngle * 10000) / 10000,
            directionLength: Math.round(f.origin.directionLength * 100) / 100,
          }),
        } as AnnotationExportV2;
      } else {
        return {
          ...base,
          shape: 'linear' as const,
          startX: Math.round(f.startPoint.x * 100) / 100,
          startY: Math.round(f.startPoint.y * 100) / 100,
          endX: Math.round(f.endPoint.x * 100) / 100,
          endY: Math.round(f.endPoint.y * 100) / 100,
          halfWidth: Math.round(f.halfWidth * 100) / 100,
        } as AnnotationExportV2;
      }
    }),
  };

  const imageList = images.map(img => ({
    id: img.id,
    fileName: img.fileName,
    data: img.imageData,
  }));

  return { manifest, annotations, imageList };
}

/**
 * Parse imported JSON to follicles (annotations) - V1 format
 */
export function parseImport(json: string, imageId?: string): Follicle[] {
  const data = JSON.parse(json) as FollicleExportV1;

  if (data.version !== '1.0') {
    throw new Error(`Unsupported export version: ${data.version}`);
  }

  return data.annotations.map(f => {
    const base = {
      id: f.id,
      imageId: imageId || '',  // Empty string for V1 compatibility
      label: f.label,
      notes: f.notes,
      color: f.color,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };

    if (f.shape === 'rectangle' && f.x !== undefined && f.y !== undefined && f.width !== undefined && f.height !== undefined) {
      // Parse origin data if present
      const origin = (f.originX !== undefined && f.originY !== undefined)
        ? {
            originPoint: { x: f.originX, y: f.originY },
            directionAngle: f.directionAngle ?? 0,
            directionLength: f.directionLength ?? 30,
          }
        : undefined;

      return {
        ...base,
        shape: 'rectangle' as const,
        x: f.x,
        y: f.y,
        width: f.width,
        height: f.height,
        origin,
      } as RectangleAnnotation;
    } else if (f.shape === 'linear' && f.startX !== undefined && f.startY !== undefined && f.endX !== undefined && f.endY !== undefined && f.halfWidth !== undefined) {
      return {
        ...base,
        shape: 'linear' as const,
        startPoint: { x: f.startX, y: f.startY },
        endPoint: { x: f.endX, y: f.endY },
        halfWidth: f.halfWidth,
      } as LinearAnnotation;
    } else {
      // Default to circle for backwards compatibility
      return {
        ...base,
        shape: 'circle' as const,
        center: { x: f.centerX ?? f.x ?? 0, y: f.centerY ?? f.y ?? 0 },
        radius: f.radius ?? 50,
      } as CircleAnnotation;
    }
  });
}

/**
 * Parse V2 multi-image format, with V1 migration support
 */
export async function parseImportV2(result: {
  version: '1.0' | '2.0';
  imageFileName?: string;
  imageData?: ArrayBuffer;
  jsonData?: string;
  manifest?: string;
  images?: Array<{ id: string; fileName: string; data: ArrayBuffer }>;
  annotations?: string;
}): Promise<{
  loadedImages: ProjectImage[];
  loadedFollicles: Follicle[];
  globalSettings?: DetectionSettingsExport;
  imageSettingsMap?: Map<string, Partial<DetectionSettingsExport>>;
}> {
  if (result.version === '1.0') {
    // V1 migration: create a single ProjectImage and assign imageId to all annotations
    if (!result.imageData || !result.jsonData || !result.imageFileName) {
      throw new Error('Invalid V1 project file');
    }

    const imageId = generateImageId();
    const blob = new Blob([result.imageData]);
    const imageSrc = URL.createObjectURL(blob);
    const imageBitmap = await createImageBitmap(blob, { imageOrientation: 'from-image' });

    const loadedImage: ProjectImage = {
      id: imageId,
      fileName: result.imageFileName,
      width: imageBitmap.width,
      height: imageBitmap.height,
      imageData: result.imageData,
      imageBitmap,
      imageSrc,
      viewport: { offsetX: 0, offsetY: 0, scale: 1 },
      createdAt: Date.now(),
      sortOrder: 0,
    };

    // Parse annotations with the new imageId
    const loadedFollicles = parseImport(result.jsonData, imageId);

    // V1 files don't have settings - return undefined (will use defaults)
    return { loadedImages: [loadedImage], loadedFollicles, globalSettings: undefined, imageSettingsMap: undefined };
  } else {
    // V2 format
    if (!result.manifest || !result.annotations || !result.images) {
      throw new Error('Invalid V2 project file');
    }

    const manifest = JSON.parse(result.manifest) as ProjectManifestV2;
    const annotationsFile = JSON.parse(result.annotations) as AnnotationsFileV2;

    // Create ProjectImage objects for each image
    const loadedImages: ProjectImage[] = [];
    for (const imageData of result.images) {
      const manifestEntry = manifest.images.find(m => m.id === imageData.id);
      if (!manifestEntry) continue;

      const blob = new Blob([imageData.data]);
      const imageSrc = URL.createObjectURL(blob);
      const imageBitmap = await createImageBitmap(blob, { imageOrientation: 'from-image' });

      loadedImages.push({
        id: imageData.id,
        fileName: imageData.fileName,
        width: manifestEntry.width,
        height: manifestEntry.height,
        imageData: imageData.data,
        imageBitmap,
        imageSrc,
        viewport: manifestEntry.viewport,
        createdAt: Date.now(),
        sortOrder: manifestEntry.sortOrder,
      });
    }

    // Sort images by sortOrder
    loadedImages.sort((a, b) => a.sortOrder - b.sortOrder);

    // Parse annotations
    const loadedFollicles: Follicle[] = annotationsFile.annotations.map(f => {
      const base = {
        id: f.id,
        imageId: f.imageId,
        label: f.label,
        notes: f.notes,
        color: f.color,
        createdAt: Date.now(),
        updatedAt: Date.now(),
      };

      if (f.shape === 'rectangle' && f.x !== undefined && f.y !== undefined && f.width !== undefined && f.height !== undefined) {
        // Parse origin data if present
        const origin = (f.originX !== undefined && f.originY !== undefined)
          ? {
              originPoint: { x: f.originX, y: f.originY },
              directionAngle: f.directionAngle ?? 0,
              directionLength: f.directionLength ?? 30,
            }
          : undefined;

        return {
          ...base,
          shape: 'rectangle' as const,
          x: f.x,
          y: f.y,
          width: f.width,
          height: f.height,
          origin,
        } as RectangleAnnotation;
      } else if (f.shape === 'linear' && f.startX !== undefined && f.startY !== undefined && f.endX !== undefined && f.endY !== undefined && f.halfWidth !== undefined) {
        return {
          ...base,
          shape: 'linear' as const,
          startPoint: { x: f.startX, y: f.startY },
          endPoint: { x: f.endX, y: f.endY },
          halfWidth: f.halfWidth,
        } as LinearAnnotation;
      } else {
        return {
          ...base,
          shape: 'circle' as const,
          center: { x: f.centerX ?? 0, y: f.centerY ?? 0 },
          radius: f.radius ?? 50,
        } as CircleAnnotation;
      }
    });

    // Extract settings (V2 only)
    const globalSettings = manifest.settings?.detection;
    const imageSettingsMap = new Map<string, Partial<DetectionSettingsExport>>();
    for (const entry of manifest.images) {
      if (entry.detectionSettings) {
        imageSettingsMap.set(entry.id, entry.detectionSettings);
      }
    }

    return { loadedImages, loadedFollicles, globalSettings, imageSettingsMap };
  }
}

// ============================================
// Enhanced Export Functions for ML Training
// ============================================

/**
 * Get bounding box for any annotation type.
 */
function getAnnotationBounds(f: Follicle): { x: number; y: number; width: number; height: number } {
  if (isCircle(f)) {
    const diameter = f.radius * 2;
    return {
      x: f.center.x - f.radius,
      y: f.center.y - f.radius,
      width: diameter,
      height: diameter,
    };
  } else if (isRectangle(f)) {
    return {
      x: f.x,
      y: f.y,
      width: f.width,
      height: f.height,
    };
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

    return {
      x: minX,
      y: minY,
      width: maxX - minX,
      height: maxY - minY,
    };
  }

  return { x: 0, y: 0, width: 0, height: 0 };
}

/**
 * Generate enhanced JSON export with additional ML training fields.
 */
export function generateEnhancedExport(
  follicles: Follicle[],
  imageWidth: number,
  imageHeight: number
): EnhancedAnnotationExport[] {
  return follicles.map(f => {
    const bounds = getAnnotationBounds(f);

    // Normalized center (0-1)
    const centerX = bounds.x + bounds.width / 2;
    const centerY = bounds.y + bounds.height / 2;

    // Approximate area (bounding box area for simplicity)
    const area = bounds.width * bounds.height;

    // Build original shape data
    let originalData: EnhancedAnnotationExport['originalData'] = {};

    if (isCircle(f)) {
      originalData = {
        centerX: f.center.x,
        centerY: f.center.y,
        radius: f.radius,
      };
    } else if (isRectangle(f)) {
      // Rectangle data is already in bounds
    } else if (isLinear(f)) {
      originalData = {
        startX: f.startPoint.x,
        startY: f.startPoint.y,
        endX: f.endPoint.x,
        endY: f.endPoint.y,
        halfWidth: f.halfWidth,
      };
    }

    return {
      id: f.id,
      imageId: f.imageId,
      shape: f.shape,
      label: f.label,
      notes: f.notes,
      color: f.color,
      x: Math.round(bounds.x * 100) / 100,
      y: Math.round(bounds.y * 100) / 100,
      width: Math.round(bounds.width * 100) / 100,
      height: Math.round(bounds.height * 100) / 100,
      x_center: Math.round((centerX / imageWidth) * 100000) / 100000,
      y_center: Math.round((centerY / imageHeight) * 100000) / 100000,
      area: Math.round(area),
      aspectRatio: Math.round((bounds.width / bounds.height) * 100) / 100,
      confidence: 1.0,  // Manual annotations have full confidence
      originalData,
    };
  });
}

/**
 * Export annotations in YOLO format.
 *
 * YOLO format: <class_id> <x_center> <y_center> <width> <height>
 * All values normalized to 0-1 relative to image dimensions.
 *
 * @param follicles - Array of annotations
 * @param imageWidth - Image width in pixels
 * @param imageHeight - Image height in pixels
 * @param classId - Class ID for all annotations (default: 0 for single-class)
 * @returns String in YOLO format (one line per annotation)
 */
export function exportToYOLO(
  follicles: Follicle[],
  imageWidth: number,
  imageHeight: number,
  classId: number = 0
): string {
  const lines: string[] = [];

  for (const f of follicles) {
    const bounds = getAnnotationBounds(f);

    // Normalized center
    const x_center = (bounds.x + bounds.width / 2) / imageWidth;
    const y_center = (bounds.y + bounds.height / 2) / imageHeight;

    // Normalized dimensions
    const width = bounds.width / imageWidth;
    const height = bounds.height / imageHeight;

    // Format: class x_center y_center width height (6 decimal places)
    lines.push(
      `${classId} ${x_center.toFixed(6)} ${y_center.toFixed(6)} ${width.toFixed(6)} ${height.toFixed(6)}`
    );
  }

  return lines.join('\n');
}

/**
 * Export all images and annotations in YOLO format for training.
 * Returns a structure ready for creating a YOLO dataset.
 *
 * @param images - Array of project images
 * @param follicles - Array of all annotations
 * @returns Object with image files and corresponding label files
 */
export function exportYOLODataset(
  images: ProjectImage[],
  follicles: Follicle[]
): {
  files: Array<{
    imageName: string;
    labelName: string;
    labelContent: string;
    imageData: ArrayBuffer;
  }>;
  dataYaml: string;
} {
  const files: Array<{
    imageName: string;
    labelName: string;
    labelContent: string;
    imageData: ArrayBuffer;
  }> = [];

  for (const image of images) {
    const imageFollicles = follicles.filter(f => f.imageId === image.id);

    // Generate base name (without extension)
    const baseName = image.fileName.replace(/\.[^.]+$/, '');
    const ext = image.fileName.split('.').pop() || 'jpg';

    const labelContent = exportToYOLO(imageFollicles, image.width, image.height);

    files.push({
      imageName: `${baseName}.${ext}`,
      labelName: `${baseName}.txt`,
      labelContent,
      imageData: image.imageData,
    });
  }

  // Generate data.yaml for YOLO training
  const dataYaml = `# YOLO Dataset Configuration
# Generated by Follicle Labeller

path: .
train: images
val: images

# Classes
nc: 1
names:
  0: follicle
`;

  return { files, dataYaml };
}

/**
 * Export annotations as CSV for analysis.
 */
export function exportToCSV(
  follicles: Follicle[],
  imageWidth: number,
  imageHeight: number
): string {
  const headers = [
    'id',
    'imageId',
    'shape',
    'label',
    'x',
    'y',
    'width',
    'height',
    'x_center_norm',
    'y_center_norm',
    'area',
    'aspect_ratio',
  ];

  const rows: string[] = [headers.join(',')];

  for (const f of follicles) {
    const bounds = getAnnotationBounds(f);
    const centerX = bounds.x + bounds.width / 2;
    const centerY = bounds.y + bounds.height / 2;
    const area = bounds.width * bounds.height;

    rows.push([
      f.id,
      f.imageId,
      f.shape,
      `"${f.label.replace(/"/g, '""')}"`,  // Escape quotes
      bounds.x.toFixed(2),
      bounds.y.toFixed(2),
      bounds.width.toFixed(2),
      bounds.height.toFixed(2),
      (centerX / imageWidth).toFixed(6),
      (centerY / imageHeight).toFixed(6),
      Math.round(area).toString(),
      (bounds.width / bounds.height).toFixed(2),
    ].join(','));
  }

  return rows.join('\n');
}

// ============================================
// YOLO Keypoint Export Functions
// ============================================

/**
 * Export annotations with follicle origin data as YOLO keypoint dataset ZIP.
 *
 * This creates a dataset for training YOLO11-pose models to detect
 * follicle origin points and growth directions.
 *
 * @param images Map of project images
 * @param follicles All annotations
 * @param config Export configuration (optional)
 * @returns Promise resolving to ZIP blob and statistics
 */
export async function exportYOLOKeypointDatasetZip(
  images: Map<string, ProjectImage>,
  follicles: Follicle[],
  config: KeypointExportConfig = DEFAULT_KEYPOINT_EXPORT_CONFIG
): Promise<{ blob: Blob; stats: KeypointDatasetStats }> {
  const { files, stats } = await exportYOLOKeypointDataset(images, follicles, config);
  const blob = await createKeypointDatasetZip(files);
  return { blob, stats };
}

// Re-export types for convenience
export type { KeypointExportConfig, KeypointDatasetStats };
export { DEFAULT_KEYPOINT_EXPORT_CONFIG };

// ============================================
// Selected Annotations Export/Import
// ============================================

/**
 * Interface for selected annotations export format
 */
export interface SelectedAnnotationsExport {
  version: '1.0';
  type: 'selected-annotations';
  exportedAt: string;
  imageDimensions: { width: number; height: number };
  annotations: Array<{
    id: string;
    shape: 'circle' | 'rectangle' | 'linear';
    label: string;
    notes: string;
    color: string;
    // Shape-specific fields
    centerX?: number;
    centerY?: number;
    radius?: number;
    x?: number;
    y?: number;
    width?: number;
    height?: number;
    originX?: number;
    originY?: number;
    directionAngle?: number;
    directionLength?: number;
    startX?: number;
    startY?: number;
    endX?: number;
    endY?: number;
    halfWidth?: number;
  }>;
}

/**
 * Export selected annotations as JSON string
 *
 * @param follicles All follicles (will be filtered by selectedIds)
 * @param selectedIds Set of selected annotation IDs
 * @param imageDimensions The dimensions of the source image
 * @returns JSON string with selected annotations
 */
export function exportSelectedAnnotationsJSON(
  follicles: Follicle[],
  selectedIds: Set<string>,
  imageDimensions: { width: number; height: number }
): string {
  const selectedFollicles = follicles.filter(f => selectedIds.has(f.id));

  const exportData: SelectedAnnotationsExport = {
    version: '1.0',
    type: 'selected-annotations',
    exportedAt: new Date().toISOString(),
    imageDimensions,
    annotations: selectedFollicles.map(f => {
      const base = {
        id: f.id,
        shape: f.shape,
        label: f.label,
        notes: f.notes,
        color: f.color,
      };

      if (isCircle(f)) {
        return {
          ...base,
          centerX: Math.round(f.center.x * 100) / 100,
          centerY: Math.round(f.center.y * 100) / 100,
          radius: Math.round(f.radius * 100) / 100,
        };
      } else if (isRectangle(f)) {
        return {
          ...base,
          x: Math.round(f.x * 100) / 100,
          y: Math.round(f.y * 100) / 100,
          width: Math.round(f.width * 100) / 100,
          height: Math.round(f.height * 100) / 100,
          ...(f.origin && {
            originX: Math.round(f.origin.originPoint.x * 100) / 100,
            originY: Math.round(f.origin.originPoint.y * 100) / 100,
            directionAngle: Math.round(f.origin.directionAngle * 10000) / 10000,
            directionLength: Math.round(f.origin.directionLength * 100) / 100,
          }),
        };
      } else {
        // Linear
        return {
          ...base,
          startX: Math.round(f.startPoint.x * 100) / 100,
          startY: Math.round(f.startPoint.y * 100) / 100,
          endX: Math.round(f.endPoint.x * 100) / 100,
          endY: Math.round(f.endPoint.y * 100) / 100,
          halfWidth: Math.round(f.halfWidth * 100) / 100,
        };
      }
    }),
  };

  return JSON.stringify(exportData, null, 2);
}

/**
 * Check if two annotations are duplicates based on shape and position
 * Uses a tolerance for floating point comparison
 */
function areAnnotationsDuplicate(a: Follicle, b: Follicle, tolerance = 2): boolean {
  if (a.shape !== b.shape) return false;

  if (isCircle(a) && isCircle(b)) {
    return (
      Math.abs(a.center.x - b.center.x) < tolerance &&
      Math.abs(a.center.y - b.center.y) < tolerance &&
      Math.abs(a.radius - b.radius) < tolerance
    );
  }

  if (isRectangle(a) && isRectangle(b)) {
    return (
      Math.abs(a.x - b.x) < tolerance &&
      Math.abs(a.y - b.y) < tolerance &&
      Math.abs(a.width - b.width) < tolerance &&
      Math.abs(a.height - b.height) < tolerance
    );
  }

  if (isLinear(a) && isLinear(b)) {
    return (
      Math.abs(a.startPoint.x - b.startPoint.x) < tolerance &&
      Math.abs(a.startPoint.y - b.startPoint.y) < tolerance &&
      Math.abs(a.endPoint.x - b.endPoint.x) < tolerance &&
      Math.abs(a.endPoint.y - b.endPoint.y) < tolerance &&
      Math.abs(a.halfWidth - b.halfWidth) < tolerance
    );
  }

  return false;
}

/**
 * Augmentable annotation - existing annotation can be updated with origin from imported
 */
export interface AugmentableAnnotation {
  /** The imported annotation with origin data */
  imported: RectangleAnnotation;
  /** The existing annotation ID to update */
  existingId: string;
}

/**
 * Result of duplicate detection during import
 */
export interface ImportDuplicateResult {
  /** Annotations that are new (no duplicates found) */
  newAnnotations: Follicle[];
  /** Annotations that are exact duplicates (same position, same origin state) */
  duplicates: Follicle[];
  /** Annotations that match existing but have origin data the existing lacks */
  augmentable: AugmentableAnnotation[];
  /** Annotations that match existing but existing already has origin (imported is redundant) */
  alreadyAugmented: Follicle[];
  /** Total annotations in the import file */
  totalImported: number;
}

/**
 * Import annotations from JSON and detect duplicates
 *
 * @param json JSON string containing annotations
 * @param targetImageId ID of the image to import annotations into
 * @param existingFollicles Existing follicles to check for duplicates
 * @returns Object containing new annotations, duplicates, augmentable, alreadyAugmented, and counts
 */
export function importAnnotationsFromJSONWithDuplicateCheck(
  json: string,
  targetImageId: string,
  existingFollicles: Follicle[]
): ImportDuplicateResult {
  const allImported = importAnnotationsFromJSON(json, targetImageId);
  const existingForImage = existingFollicles.filter(f => f.imageId === targetImageId);

  const newAnnotations: Follicle[] = [];
  const duplicates: Follicle[] = [];
  const augmentable: AugmentableAnnotation[] = [];
  const alreadyAugmented: Follicle[] = [];

  for (const imported of allImported) {
    const matchingExisting = existingForImage.find(existing =>
      areAnnotationsDuplicate(imported, existing)
    );

    if (!matchingExisting) {
      // No match - it's a new annotation
      newAnnotations.push(imported);
    } else if (
      isRectangle(imported) &&
      isRectangle(matchingExisting) &&
      imported.origin &&
      !matchingExisting.origin
    ) {
      // Imported has origin, existing doesn't - can augment existing
      augmentable.push({
        imported: imported as RectangleAnnotation,
        existingId: matchingExisting.id,
      });
    } else if (
      isRectangle(imported) &&
      isRectangle(matchingExisting) &&
      !imported.origin &&
      matchingExisting.origin
    ) {
      // Existing has origin, imported doesn't - existing is already better
      alreadyAugmented.push(imported);
    } else {
      // Exact duplicate (both have origin, or both don't, or non-rectangle shapes)
      duplicates.push(imported);
    }
  }

  return {
    newAnnotations,
    duplicates,
    augmentable,
    alreadyAugmented,
    totalImported: allImported.length,
  };
}

/**
 * Import annotations from JSON into current image
 *
 * @param json JSON string containing annotations
 * @param targetImageId ID of the image to import annotations into
 * @returns Array of new follicles with fresh IDs
 */
export function importAnnotationsFromJSON(
  json: string,
  targetImageId: string
): Follicle[] {
  const data = JSON.parse(json) as SelectedAnnotationsExport;

  // Validate format
  if (data.type !== 'selected-annotations' || data.version !== '1.0') {
    throw new Error('Invalid annotation file format. Expected selected-annotations v1.0');
  }

  if (!data.annotations || !Array.isArray(data.annotations)) {
    throw new Error('No annotations found in file');
  }

  const now = Date.now();

  return data.annotations.map(ann => {
    const base = {
      id: generateId(), // Generate new unique ID to avoid conflicts
      imageId: targetImageId,
      label: ann.label || '',
      notes: ann.notes || '',
      color: ann.color || '#4ECDC4',
      createdAt: now,
      updatedAt: now,
    };

    if (ann.shape === 'circle' && ann.centerX !== undefined && ann.centerY !== undefined && ann.radius !== undefined) {
      return {
        ...base,
        shape: 'circle' as const,
        center: { x: ann.centerX, y: ann.centerY },
        radius: ann.radius,
      } as CircleAnnotation;
    } else if (ann.shape === 'rectangle' && ann.x !== undefined && ann.y !== undefined && ann.width !== undefined && ann.height !== undefined) {
      const origin = (ann.originX !== undefined && ann.originY !== undefined)
        ? {
            originPoint: { x: ann.originX, y: ann.originY },
            directionAngle: ann.directionAngle ?? 0,
            directionLength: ann.directionLength ?? 30,
          }
        : undefined;

      return {
        ...base,
        shape: 'rectangle' as const,
        x: ann.x,
        y: ann.y,
        width: ann.width,
        height: ann.height,
        origin,
      } as RectangleAnnotation;
    } else if (ann.shape === 'linear' && ann.startX !== undefined && ann.startY !== undefined && ann.endX !== undefined && ann.endY !== undefined && ann.halfWidth !== undefined) {
      return {
        ...base,
        shape: 'linear' as const,
        startPoint: { x: ann.startX, y: ann.startY },
        endPoint: { x: ann.endX, y: ann.endY },
        halfWidth: ann.halfWidth,
      } as LinearAnnotation;
    } else {
      // Default to circle if shape is unknown
      return {
        ...base,
        shape: 'circle' as const,
        center: { x: ann.centerX ?? 100, y: ann.centerY ?? 100 },
        radius: ann.radius ?? 50,
      } as CircleAnnotation;
    }
  });
}
