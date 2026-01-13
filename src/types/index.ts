// Core geometric types
export interface Point {
  x: number;
  y: number;
}

// Shape types
export type ShapeType = 'circle' | 'rectangle';

// Base annotation properties shared by all shapes
interface BaseAnnotation {
  id: string;
  label: string;
  notes: string;
  color: string;
  createdAt: number;
  updatedAt: number;
}

// Circle annotation
export interface CircleAnnotation extends BaseAnnotation {
  shape: 'circle';
  center: Point;
  radius: number;
}

// Rectangle annotation
export interface RectangleAnnotation extends BaseAnnotation {
  shape: 'rectangle';
  x: number;           // Top-left x
  y: number;           // Top-left y
  width: number;
  height: number;
}

// Union type for all annotations (keeping Follicle name for compatibility)
export type Follicle = CircleAnnotation | RectangleAnnotation;

// Type guards
export function isCircle(f: Follicle): f is CircleAnnotation {
  return f.shape === 'circle';
}

export function isRectangle(f: Follicle): f is RectangleAnnotation {
  return f.shape === 'rectangle';
}

// Viewport state for canvas
export interface Viewport {
  offsetX: number;      // Pan offset in screen pixels
  offsetY: number;
  scale: number;        // Zoom level (1.0 = 100%)
}

// Interaction modes
export type InteractionMode = 'select' | 'create' | 'pan';

// Drag state during interactions
export interface DragState {
  isDragging: boolean;
  startPoint: Point | null;
  currentPoint: Point | null;
  dragType: 'create' | 'move' | 'resize' | 'pan' | null;
  targetId: string | null;
  resizeHandle?: string;  // For rectangles: 'nw', 'ne', 'sw', 'se'
}

// JSON export schema
export interface AnnotationExport {
  id: string;
  shape: ShapeType;
  label: string;
  notes: string;
  color: string;
  // Circle properties
  centerX?: number;
  centerY?: number;
  radius?: number;
  // Rectangle properties
  x?: number;
  y?: number;
  width?: number;
  height?: number;
}

export interface FollicleExportV1 {
  version: '1.0';
  image: {
    fileName: string;
    width: number;
    height: number;
  };
  metadata: {
    exportedAt: string;
    applicationVersion: string;
    annotationCount: number;
  };
  annotations: AnnotationExport[];
}

// Electron API type declaration
declare global {
  interface Window {
    electronAPI: {
      openImageDialog: () => Promise<{ filePath: string; fileName: string; data: ArrayBuffer } | null>;
      saveProject: (imageData: ArrayBuffer, imageFileName: string, jsonData: string) => Promise<boolean>;
      loadProject: () => Promise<{ imageFileName: string; imageData: ArrayBuffer; jsonData: string } | null>;
    };
  }
}
