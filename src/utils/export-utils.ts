import { Follicle, FollicleExportV1, CircleAnnotation, RectangleAnnotation, LinearAnnotation, isCircle, isRectangle } from '../types';

/**
 * Generate export JSON from follicles (annotations)
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
 * Parse imported JSON to follicles (annotations)
 */
export function parseImport(json: string): Follicle[] {
  const data = JSON.parse(json) as FollicleExportV1;

  if (data.version !== '1.0') {
    throw new Error(`Unsupported export version: ${data.version}`);
  }

  return data.annotations.map(f => {
    const base = {
      id: f.id,
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
