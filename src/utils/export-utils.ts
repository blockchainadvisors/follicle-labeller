import { Follicle, FollicleExportV1 } from '../types';

/**
 * Generate export JSON from follicles
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
      follicleCount: follicles.length,
    },
    follicles: follicles.map(f => ({
      id: f.id,
      x: Math.round(f.center.x * 100) / 100,
      y: Math.round(f.center.y * 100) / 100,
      radius: Math.round(f.radius * 100) / 100,
      label: f.label,
      notes: f.notes,
      color: f.color,
    })),
  };
}

/**
 * Parse imported JSON to follicles
 */
export function parseImport(json: string): Follicle[] {
  const data = JSON.parse(json) as FollicleExportV1;

  if (data.version !== '1.0') {
    throw new Error(`Unsupported export version: ${data.version}`);
  }

  return data.follicles.map(f => ({
    id: f.id,
    center: { x: f.x, y: f.y },
    radius: f.radius,
    label: f.label,
    notes: f.notes,
    color: f.color,
    createdAt: Date.now(),
    updatedAt: Date.now(),
  }));
}
