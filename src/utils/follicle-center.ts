import { Follicle, Point, isCircle, isRectangle, isLinear } from '../types';

export function getFollicleCenter(follicle: Follicle): Point {
  if (isCircle(follicle)) {
    return { x: follicle.center.x, y: follicle.center.y };
  }
  if (isRectangle(follicle)) {
    return {
      x: follicle.x + follicle.width / 2,
      y: follicle.y + follicle.height / 2,
    };
  }
  if (isLinear(follicle)) {
    return {
      x: (follicle.startPoint.x + follicle.endPoint.x) / 2,
      y: (follicle.startPoint.y + follicle.endPoint.y) / 2,
    };
  }
  throw new Error(`Unknown follicle shape: ${(follicle as { shape: string }).shape}`);
}
