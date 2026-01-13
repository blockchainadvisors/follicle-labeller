// Core geometric types
export interface Point {
  x: number;
  y: number;
}

// Follicle annotation type
export interface Follicle {
  id: string;
  center: Point;        // Image coordinates (not screen)
  radius: number;       // In image pixels
  label: string;
  notes: string;
  color: string;        // Hex color for visualization
  createdAt: number;    // Timestamp
  updatedAt: number;
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
}

// JSON export schema
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
    follicleCount: number;
  };
  follicles: Array<{
    id: string;
    x: number;
    y: number;
    radius: number;
    label: string;
    notes: string;
    color: string;
  }>;
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
