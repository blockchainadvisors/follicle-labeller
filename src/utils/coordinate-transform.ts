import { Point, Viewport } from '../types';

/**
 * Convert screen coordinates to image coordinates
 */
export function screenToImage(
  screenX: number,
  screenY: number,
  viewport: Viewport,
  canvasRect: DOMRect
): Point {
  return {
    x: (screenX - canvasRect.left - viewport.offsetX) / viewport.scale,
    y: (screenY - canvasRect.top - viewport.offsetY) / viewport.scale,
  };
}

/**
 * Convert image coordinates to screen coordinates
 */
export function imageToScreen(
  imagePoint: Point,
  viewport: Viewport
): Point {
  return {
    x: imagePoint.x * viewport.scale + viewport.offsetX,
    y: imagePoint.y * viewport.scale + viewport.offsetY,
  };
}

/**
 * Calculate distance between two points
 */
export function distance(p1: Point, p2: Point): number {
  return Math.sqrt(Math.pow(p2.x - p1.x, 2) + Math.pow(p2.y - p1.y, 2));
}

/**
 * Check if a point is inside a circle
 */
export function isPointInCircle(
  point: Point,
  center: Point,
  radius: number
): boolean {
  return distance(point, center) <= radius;
}

/**
 * Check if a point is on the edge of a circle (for resize handle)
 */
export function isPointOnCircleEdge(
  point: Point,
  center: Point,
  radius: number,
  tolerance: number = 10
): boolean {
  const dist = distance(point, center);
  return Math.abs(dist - radius) <= tolerance;
}
