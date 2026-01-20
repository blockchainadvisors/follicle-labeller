import { Point, Follicle, SelectionBounds, isCircle, isRectangle, isLinear } from '../types';

/**
 * Check if a point is inside a polygon using ray casting algorithm
 */
export function isPointInPolygon(point: Point, polygon: Point[]): boolean {
  if (polygon.length < 3) return false;

  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i].x;
    const yi = polygon[i].y;
    const xj = polygon[j].x;
    const yj = polygon[j].y;

    if (((yi > point.y) !== (yj > point.y)) &&
        (point.x < (xj - xi) * (point.y - yi) / (yj - yi) + xi)) {
      inside = !inside;
    }
  }

  return inside;
}

/**
 * Get the bounding box of a follicle/annotation
 */
export function getFollicleBounds(follicle: Follicle): SelectionBounds {
  if (isCircle(follicle)) {
    return {
      minX: follicle.center.x - follicle.radius,
      minY: follicle.center.y - follicle.radius,
      maxX: follicle.center.x + follicle.radius,
      maxY: follicle.center.y + follicle.radius,
    };
  } else if (isRectangle(follicle)) {
    return {
      minX: follicle.x,
      minY: follicle.y,
      maxX: follicle.x + follicle.width,
      maxY: follicle.y + follicle.height,
    };
  } else if (isLinear(follicle)) {
    // Get corners of the rotated rectangle
    const corners = getLinearCorners(follicle.startPoint, follicle.endPoint, follicle.halfWidth);
    const xs = corners.map(c => c.x);
    const ys = corners.map(c => c.y);
    return {
      minX: Math.min(...xs),
      minY: Math.min(...ys),
      maxX: Math.max(...xs),
      maxY: Math.max(...ys),
    };
  }

  // Fallback (should never reach here)
  return { minX: 0, minY: 0, maxX: 0, maxY: 0 };
}

/**
 * Get the center point of a follicle
 */
export function getFollicleCenter(follicle: Follicle): Point {
  if (isCircle(follicle)) {
    return follicle.center;
  } else if (isRectangle(follicle)) {
    return {
      x: follicle.x + follicle.width / 2,
      y: follicle.y + follicle.height / 2,
    };
  } else if (isLinear(follicle)) {
    return {
      x: (follicle.startPoint.x + follicle.endPoint.x) / 2,
      y: (follicle.startPoint.y + follicle.endPoint.y) / 2,
    };
  }
  return { x: 0, y: 0 };
}

/**
 * Helper to get the 4 corners of a linear (rotated rectangle)
 */
function getLinearCorners(startPoint: Point, endPoint: Point, halfWidth: number): Point[] {
  const dx = endPoint.x - startPoint.x;
  const dy = endPoint.y - startPoint.y;
  const length = Math.sqrt(dx * dx + dy * dy);

  if (length === 0) return [startPoint, startPoint, startPoint, startPoint];

  // Perpendicular unit vector
  const perpX = -dy / length;
  const perpY = dx / length;

  // Calculate 4 corners
  return [
    { x: startPoint.x + perpX * halfWidth, y: startPoint.y + perpY * halfWidth },
    { x: startPoint.x - perpX * halfWidth, y: startPoint.y - perpY * halfWidth },
    { x: endPoint.x - perpX * halfWidth, y: endPoint.y - perpY * halfWidth },
    { x: endPoint.x + perpX * halfWidth, y: endPoint.y + perpY * halfWidth },
  ];
}

/**
 * Check if a follicle is inside a rectangular selection bounds
 * Uses center point containment for determining selection
 */
export function isFollicleInBounds(follicle: Follicle, bounds: SelectionBounds): boolean {
  const center = getFollicleCenter(follicle);
  return center.x >= bounds.minX &&
         center.x <= bounds.maxX &&
         center.y >= bounds.minY &&
         center.y <= bounds.maxY;
}

/**
 * Check if a follicle is inside a lasso polygon
 * Uses center point containment for determining selection
 */
export function isFollicleInPolygon(follicle: Follicle, polygon: Point[]): boolean {
  const center = getFollicleCenter(follicle);
  return isPointInPolygon(center, polygon);
}

/**
 * Simplify a path using Ramer-Douglas-Peucker algorithm
 * Reduces the number of points while preserving the shape
 */
export function simplifyPath(points: Point[], tolerance: number = 2): Point[] {
  if (points.length <= 2) return points;

  // Find the point with maximum distance from the line between first and last
  let maxDistance = 0;
  let maxIndex = 0;

  const first = points[0];
  const last = points[points.length - 1];

  for (let i = 1; i < points.length - 1; i++) {
    const distance = perpendicularDistance(points[i], first, last);
    if (distance > maxDistance) {
      maxDistance = distance;
      maxIndex = i;
    }
  }

  // If max distance is greater than tolerance, recursively simplify
  if (maxDistance > tolerance) {
    const left = simplifyPath(points.slice(0, maxIndex + 1), tolerance);
    const right = simplifyPath(points.slice(maxIndex), tolerance);

    // Combine results (remove duplicate point at maxIndex)
    return [...left.slice(0, -1), ...right];
  } else {
    // Return just the endpoints
    return [first, last];
  }
}

/**
 * Calculate perpendicular distance from a point to a line segment
 */
function perpendicularDistance(point: Point, lineStart: Point, lineEnd: Point): number {
  const dx = lineEnd.x - lineStart.x;
  const dy = lineEnd.y - lineStart.y;

  // Handle zero-length line
  if (dx === 0 && dy === 0) {
    return Math.sqrt(
      (point.x - lineStart.x) ** 2 + (point.y - lineStart.y) ** 2
    );
  }

  const numerator = Math.abs(
    dy * point.x - dx * point.y + lineEnd.x * lineStart.y - lineEnd.y * lineStart.x
  );
  const denominator = Math.sqrt(dx * dx + dy * dy);

  return numerator / denominator;
}

/**
 * Get all follicles that are inside the given rectangular bounds
 */
export function getFolliclesInBounds(follicles: Follicle[], bounds: SelectionBounds): Follicle[] {
  return follicles.filter(f => isFollicleInBounds(f, bounds));
}

/**
 * Get all follicles that are inside the given lasso polygon
 */
export function getFolliclesInPolygon(follicles: Follicle[], polygon: Point[]): Follicle[] {
  if (polygon.length < 3) return [];
  return follicles.filter(f => isFollicleInPolygon(f, polygon));
}

/**
 * Create selection bounds from two points (start and end of marquee drag)
 */
export function createSelectionBounds(start: Point, end: Point): SelectionBounds {
  return {
    minX: Math.min(start.x, end.x),
    minY: Math.min(start.y, end.y),
    maxX: Math.max(start.x, end.x),
    maxY: Math.max(start.y, end.y),
  };
}
