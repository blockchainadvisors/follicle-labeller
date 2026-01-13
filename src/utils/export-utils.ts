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
  isCircle,
  isRectangle
} from '../types';
import { generateImageId } from '../store/projectStore';

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
  follicles: Follicle[]
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
    images: images.map(img => ({
      id: img.id,
      fileName: img.fileName,
      archiveFileName: `${img.id}-${img.fileName}`,
      width: img.width,
      height: img.height,
      sortOrder: img.sortOrder,
      viewport: img.viewport,
    })),
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
      return {
        ...base,
        shape: 'rectangle' as const,
        x: f.x,
        y: f.y,
        width: f.width,
        height: f.height,
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
}> {
  if (result.version === '1.0') {
    // V1 migration: create a single ProjectImage and assign imageId to all annotations
    if (!result.imageData || !result.jsonData || !result.imageFileName) {
      throw new Error('Invalid V1 project file');
    }

    const imageId = generateImageId();
    const blob = new Blob([result.imageData]);
    const imageSrc = URL.createObjectURL(blob);
    const imageBitmap = await createImageBitmap(blob);

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

    return { loadedImages: [loadedImage], loadedFollicles };
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
      const imageBitmap = await createImageBitmap(blob);

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
        return {
          ...base,
          shape: 'rectangle' as const,
          x: f.x,
          y: f.y,
          width: f.width,
          height: f.height,
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

    return { loadedImages, loadedFollicles };
  }
}
