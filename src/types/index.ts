// Core geometric types
export interface Point {
  x: number;
  y: number;
}

// Image identifier type
export type ImageId = string;

// Shape types
export type ShapeType = 'circle' | 'rectangle' | 'linear';

// Base annotation properties shared by all shapes
interface BaseAnnotation {
  id: string;
  imageId: ImageId;  // Links annotation to its parent image
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

// Linear annotation (rotated rectangle defined by centerline + half-width)
export interface LinearAnnotation extends BaseAnnotation {
  shape: 'linear';
  startPoint: Point;   // Start of the centerline
  endPoint: Point;     // End of the centerline
  halfWidth: number;   // Half-width perpendicular to centerline
}

// Union type for all annotations (keeping Follicle name for compatibility)
export type Follicle = CircleAnnotation | RectangleAnnotation | LinearAnnotation;

// Type guards
export function isCircle(f: Follicle): f is CircleAnnotation {
  return f.shape === 'circle';
}

export function isRectangle(f: Follicle): f is RectangleAnnotation {
  return f.shape === 'rectangle';
}

export function isLinear(f: Follicle): f is LinearAnnotation {
  return f.shape === 'linear';
}

// Viewport state for canvas
export interface Viewport {
  offsetX: number;      // Pan offset in screen pixels
  offsetY: number;
  scale: number;        // Zoom level (1.0 = 100%)
}

// Project image - stores image data and per-image viewport
export interface ProjectImage {
  id: ImageId;
  fileName: string;
  width: number;
  height: number;
  imageData: ArrayBuffer;     // Raw image bytes for saving
  imageBitmap: ImageBitmap;   // Pre-decoded for rendering
  imageSrc: string;           // Object URL for display
  viewport: Viewport;         // Per-image zoom/pan state
  createdAt: number;
  sortOrder: number;          // For ordering in explorer
}

// Interaction modes
export type InteractionMode = 'select' | 'create' | 'pan';

// Selection tool types for multi-selection
export type SelectionToolType = 'click' | 'marquee' | 'lasso';

// Selection bounds for marquee selection
export interface SelectionBounds {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

// Drag state during interactions
export interface DragState {
  isDragging: boolean;
  startPoint: Point | null;
  currentPoint: Point | null;
  dragType: 'create' | 'move' | 'resize' | 'pan' | 'marquee' | 'lasso' | null;
  targetId: string | null;
  resizeHandle?: string;  // For rectangles: 'nw', 'ne', 'sw', 'se'; for linear: 'start', 'end', 'width'
  // Multi-phase creation for linear shapes
  createPhase?: 'line' | 'width';  // 'line' = defining centerline, 'width' = defining half-width
  lineEndPoint?: Point;  // Stored end point after first phase of linear creation
  // Multi-selection lasso path tracking
  lassoPoints?: Point[];
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
  // Linear properties
  startX?: number;
  startY?: number;
  endX?: number;
  endY?: number;
  halfWidth?: number;
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

// V2 Multi-image export types
export interface AnnotationExportV2 extends AnnotationExport {
  imageId: ImageId;
}

export interface ImageManifestEntry {
  id: ImageId;
  fileName: string;
  archiveFileName: string;  // {id}-{fileName} in archive
  width: number;
  height: number;
  sortOrder: number;
  viewport: Viewport;
}

export interface ProjectManifestV2 {
  version: '2.0';
  metadata: {
    exportedAt: string;
    applicationVersion: string;
    imageCount: number;
    annotationCount: number;
  };
  images: ImageManifestEntry[];
}

export interface AnnotationsFileV2 {
  annotations: AnnotationExportV2[];
}

// Electron API type declaration
declare global {
  interface Window {
    electronAPI: {
      // Single image dialog (for adding images)
      openImageDialog: () => Promise<{ filePath: string; fileName: string; data: ArrayBuffer } | null>;
      // Legacy V1 operations (still supported for backward compatibility)
      saveProject: (imageData: ArrayBuffer, imageFileName: string, jsonData: string) => Promise<boolean>;
      loadProject: () => Promise<{ imageFileName: string; imageData: ArrayBuffer; jsonData: string } | null>;
      // V2 Multi-image operations
      saveProjectV2: (
        images: Array<{ id: string; fileName: string; data: ArrayBuffer }>,
        manifest: string,
        annotations: string,
        defaultPath?: string
      ) => Promise<{ success: boolean; filePath?: string }>;
      saveProjectV2ToPath: (
        filePath: string,
        images: Array<{ id: string; fileName: string; data: ArrayBuffer }>,
        manifest: string,
        annotations: string
      ) => Promise<{ success: boolean; filePath?: string }>;
      loadProjectV2: () => Promise<{
        version: '1.0' | '2.0';
        filePath: string;
        // V1 format (single image)
        imageFileName?: string;
        imageData?: ArrayBuffer;
        jsonData?: string;
        // V2 format (multiple images)
        manifest?: string;
        images?: Array<{ id: string; fileName: string; data: ArrayBuffer }>;
        annotations?: string;
      } | null>;
      // Update menu state
      setProjectState: (hasProject: boolean) => void;
      // Menu event listeners (return cleanup function)
      onMenuOpenImage: (callback: () => void) => () => void;
      onMenuLoadProject: (callback: () => void) => () => void;
      onMenuSaveProject: (callback: () => void) => () => void;
      onMenuSaveProjectAs: (callback: () => void) => () => void;
      onMenuCloseProject: (callback: () => void) => () => void;
      onMenuUndo: (callback: () => void) => () => void;
      onMenuRedo: (callback: () => void) => () => void;
      onMenuClearAll: (callback: () => void) => () => void;
      onMenuToggleShapes: (callback: () => void) => () => void;
      onMenuToggleLabels: (callback: () => void) => () => void;
      onMenuZoomIn: (callback: () => void) => () => void;
      onMenuZoomOut: (callback: () => void) => () => void;
      onMenuResetZoom: (callback: () => void) => () => void;
      onMenuShowHelp: (callback: () => void) => () => void;
      // Unsaved changes handling
      showUnsavedChangesDialog: () => Promise<'save' | 'discard' | 'cancel'>;
      onCheckUnsavedChanges: (callback: () => void) => () => void;
      confirmClose: (canClose: boolean) => void;
      // File association handlers
      getFileToOpen: () => Promise<string | null>;
      loadProjectFromPath: (filePath: string) => Promise<{
        version: '1.0' | '2.0';
        filePath: string;
        imageFileName?: string;
        imageData?: ArrayBuffer;
        jsonData?: string;
        manifest?: string;
        images?: Array<{ id: string; fileName: string; data: ArrayBuffer }>;
        annotations?: string;
      } | null>;
      onFileOpen: (callback: (filePath: string) => void) => () => void;
      // Update progress listener (for optional custom UI)
      onUpdateDownloadProgress: (callback: (progress: { percent: number; transferred: number; total: number; bytesPerSecond: number }) => void) => () => void;
      // System power events - triggered before sleep/hibernate
      onSystemSuspend: (callback: () => void) => () => void;
    };
  }
}
