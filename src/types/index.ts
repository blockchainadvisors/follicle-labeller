// Core geometric types
export interface Point {
  x: number;
  y: number;
}

// BLOB Detection types
export interface BlobDetectionOptions {
  minWidth: number;       // Minimum blob width (default: 10)
  maxWidth: number;       // Maximum blob width (default: 200)
  minHeight: number;      // Minimum blob height (default: 10)
  maxHeight: number;      // Maximum blob height (default: 200)
  threshold?: number;     // Manual threshold (auto if not set - uses Otsu's method)
  darkBlobs: boolean;     // true = detect dark blobs on light bg
  useGPU: boolean;        // Enable WebGL acceleration
  workerCount?: number;   // Number of Web Workers (default: navigator.hardwareConcurrency)
  // SAHI-style tiling options
  tileSize?: number;      // Fixed tile size in pixels (default: 512, 0 = auto based on workerCount)
  tileOverlap?: number;   // Overlap ratio between tiles (default: 0.2 = 20%)
  // CLAHE preprocessing options
  useCLAHE?: boolean;     // Enable CLAHE preprocessing (default: false)
  claheClipLimit?: number; // CLAHE clip limit (default: 2.0)
  claheTileSize?: number; // CLAHE tile grid size (default: 8)
  // Soft-NMS options for merging overlapping detections
  useSoftNMS?: boolean;   // Use Soft-NMS instead of Union-Find (default: true)
  softNMSSigma?: number;  // Soft-NMS Gaussian decay sigma (default: 0.5)
  softNMSThreshold?: number; // Soft-NMS minimum score threshold (default: 0.1)
  // Gaussian blur preprocessing (matches OpenCV pipeline)
  useGaussianBlur?: boolean;   // Enable Gaussian blur (default: true)
  gaussianKernelSize?: number; // Kernel size, must be odd (default: 5)
  // Morphological opening to separate touching objects
  useMorphOpen?: boolean;      // Enable morphological opening (default: true)
  morphKernelSize?: number;    // Structuring element size (default: 3)
  // Circularity filter (DISABLE for elongated shapes like hair follicles)
  filterByCircularity?: boolean; // Enable circularity filtering (default: false for hair follicles)
  minCircularity?: number;       // Minimum circularity 0-1, 1=perfect circle (default: 0.2)
  // Inertia filter (for elongated shapes like hair follicles)
  filterByInertia?: boolean;     // Enable inertia filtering (default: true for hair follicles)
  minInertiaRatio?: number;      // Min inertia ratio 0-1, low=elongated OK (default: 0.01)
  maxInertiaRatio?: number;      // Max inertia ratio 0-1, 1=circular (default: 1.0)
  // Convexity filter
  filterByConvexity?: boolean;   // Enable convexity filtering (default: false)
  minConvexity?: number;         // Minimum convexity 0-1, 1=perfectly convex (default: 0.5)
}

export interface DetectedBlob {
  x: number;              // Top-left X (image coords)
  y: number;              // Top-left Y (image coords)
  width: number;          // Bounding box width
  height: number;         // Bounding box height
  area: number;           // Blob area in pixels
  aspectRatio: number;    // width/height ratio
  confidence: number;     // Detection confidence score (0-1)
}

// Message types for Web Worker communication
export interface BlobWorkerMessage {
  imageData: ImageData;
  tileX: number;
  tileY: number;
  tileWidth: number;
  tileHeight: number;
  options: BlobDetectionOptions;
}

export interface BlobWorkerResult {
  blobs: DetectedBlob[];
  tileX: number;
  tileY: number;
}

// Learned detection parameters from example annotations
export interface LearnedDetectionParams {
  minWidth: number;
  maxWidth: number;
  minHeight: number;
  maxHeight: number;
  minAspectRatio: number;
  maxAspectRatio: number;
  meanIntensity?: number;
  exampleCount: number;
}

// Effective size range after applying tolerance
export interface EffectiveSizeRange {
  minWidth: number;
  maxWidth: number;
  minHeight: number;
  maxHeight: number;
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

// Follicle origin data (where follicle enters skin and growth direction)
export interface FollicleOrigin {
  originPoint: Point;      // Where follicle enters skin (image coordinates)
  directionAngle: number;  // Direction in radians
  directionLength: number; // Arrow length in pixels
}

// Rectangle annotation
export interface RectangleAnnotation extends BaseAnnotation {
  shape: 'rectangle';
  x: number;           // Top-left x
  y: number;           // Top-left y
  width: number;
  height: number;
  origin?: FollicleOrigin;  // When set, rectangle is "locked" (no resize)
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
  // Additive selection mode (Ctrl held when starting selection)
  additive?: boolean;
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
  // Rectangle origin properties (follicle entry point and direction)
  originX?: number;
  originY?: number;
  directionAngle?: number;
  directionLength?: number;
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
      // Generic file dialog with filters
      openFileDialog: (options: {
        filters?: Array<{ name: string; extensions: string[] }>;
        title?: string;
      }) => Promise<{ filePath: string; fileName: string; data: ArrayBuffer } | null>;
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
      // Download options dialog
      showDownloadOptionsDialog: (
        selectedCount: number,
        currentImageCount: number,
        totalCount: number
      ) => Promise<'all' | 'currentImage' | 'selected' | 'cancel'>;
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
      // SAM 2 Server API
      sam: {
        startServer: (modelSize?: string) => Promise<{ success: boolean; error?: string }>;
        stopServer: () => Promise<{ success: boolean }>;
        isAvailable: () => Promise<boolean>;
        checkPython: () => Promise<{ available: boolean; version?: string; error?: string }>;
        getServerInfo: () => Promise<{ port: number; running: boolean; scriptPath: string }>;
      };
      // BLOB Detection Server API
      blob: {
        startServer: () => Promise<{ success: boolean; error?: string }>;
        stopServer: () => Promise<{ success: boolean }>;
        isAvailable: () => Promise<boolean>;
        checkPython: () => Promise<{ available: boolean; version?: string; error?: string }>;
        getServerInfo: () => Promise<{ port: number; running: boolean; scriptPath: string }>;
        getSetupStatus: () => Promise<string>;
        onSetupProgress: (callback: (status: string) => void) => () => void;
        getGPUInfo: () => Promise<GPUInfo>;
        restartServer: () => Promise<{ success: boolean; error?: string }>;
      };
      // GPU Hardware Detection & Package Installation API
      gpu: {
        getHardwareInfo: () => Promise<GPUHardwareInfo>;
        installPackages: () => Promise<{ success: boolean; error?: string }>;
        onInstallProgress: (callback: (data: { message: string; percent?: number }) => void) => () => void;
      };
      // YOLO Keypoint Training API
      yoloKeypoint: {
        getStatus: () => Promise<YoloKeypointStatus>;
        getSystemInfo: () => Promise<{
          python_version: string;
          platform: string;
          cpu_count: number;
          cpu_percent: number;
          memory_total_gb: number;
          memory_used_gb: number;
          memory_percent: number;
          device: string;
          device_name: string;
          cuda_available: boolean;
          mps_available: boolean;
          torch_version: string | null;
          gpu_memory_total_gb: number | null;
          gpu_memory_used_gb: number | null;
        }>;
        checkDependencies: () => Promise<YoloDependenciesInfo>;
        installDependencies: () => Promise<{ success: boolean; error?: string }>;
        upgradeToCUDA: () => Promise<{ success: boolean; error?: string }>;
        onInstallProgress: (callback: (data: { message: string; percent?: number }) => void) => () => void;
        validateDataset: (datasetPath: string) => Promise<DatasetValidation>;
        writeDatasetToTemp: (
          files: Array<{ path: string; content: ArrayBuffer | string }>
        ) => Promise<{ success: boolean; datasetPath?: string; error?: string }>;
        startTraining: (
          datasetPath: string,
          config: TrainingConfig,
          modelName?: string
        ) => Promise<{ jobId: string; status: string }>;
        stopTraining: (jobId: string) => Promise<{ success: boolean }>;
        subscribeProgress: (
          jobId: string,
          onProgress: (progress: TrainingProgress) => void,
          onError: (error: string) => void,
          onComplete: () => void
        ) => () => void;
        listModels: () => Promise<{ models: ModelInfo[] }>;
        loadModel: (modelPath: string) => Promise<{ success: boolean }>;
        predict: (imageData: string) => Promise<{
          success: boolean;
          prediction?: KeypointPrediction;
          message?: string;
        }>;
        showExportDialog: (
          defaultFileName: string
        ) => Promise<{ canceled: boolean; filePath?: string }>;
        exportONNX: (
          modelPath: string,
          outputPath: string
        ) => Promise<{ success: boolean; outputPath?: string }>;
        deleteModel: (modelId: string) => Promise<{ success: boolean }>;
      };
    };
  }
}

// GPU Backend Info
export interface GPUInfo {
  activeBackend: 'cuda' | 'mps' | 'cpu';
  deviceName: string;
  memoryGB?: number;
  available: {
    cuda: boolean;
    mps: boolean;
  };
}

// GPU Hardware Info (for detection before packages installed)
export interface GPUHardwareInfo {
  hardware: {
    nvidia: { found: boolean; name?: string; driver_version?: string };
    apple_silicon: { found: boolean; chip?: string };
  };
  packages: { cupy: boolean; torch: boolean };
  canEnableGpu: boolean;
  gpuEnabled: boolean;
}

// ============================================
// YOLO Keypoint Training Types
// ============================================

/**
 * Configuration for YOLO keypoint model training.
 */
export interface TrainingConfig {
  /** Model size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large) */
  modelSize: 'n' | 's' | 'm' | 'l';
  /** Number of training epochs */
  epochs: number;
  /** Input image size */
  imgSize: number;
  /** Batch size */
  batchSize: number;
  /** Early stopping patience */
  patience: number;
  /** Training device: 'auto', 'cuda', 'mps', 'cpu' */
  device: 'auto' | 'cuda' | 'mps' | 'cpu';
}

/**
 * Default training configuration.
 */
export const DEFAULT_TRAINING_CONFIG: TrainingConfig = {
  modelSize: 'n',
  epochs: 100,
  imgSize: 640,
  batchSize: 16,
  patience: 50,
  device: 'auto',
};

/**
 * Training progress update.
 */
export interface TrainingProgress {
  /** Current status */
  status: 'preparing' | 'training' | 'completed' | 'failed' | 'stopped';
  /** Current epoch */
  epoch: number;
  /** Total epochs */
  totalEpochs: number;
  /** Total loss */
  loss: number;
  /** Box loss */
  boxLoss: number;
  /** Pose/keypoint loss */
  poseLoss: number;
  /** Keypoint objectness loss */
  kobjLoss: number;
  /** Training metrics */
  metrics: {
    [key: string]: number;
  };
  /** Estimated time remaining */
  eta: string;
  /** Status message */
  message: string;
}

/**
 * Keypoint prediction result.
 */
export interface KeypointPrediction {
  /** Origin point (where follicle enters skin) */
  origin: Point;
  /** Direction endpoint (indicates growth direction) */
  directionEndpoint: Point;
  /** Prediction confidence */
  confidence: number;
}

/**
 * Information about a trained model.
 */
export interface ModelInfo {
  /** Model ID (unique identifier) */
  id: string;
  /** Model name */
  name: string;
  /** Path to model file */
  path: string;
  /** Creation timestamp (ISO string) */
  createdAt: string;
  /** Number of epochs trained */
  epochsTrained: number;
  /** Training image size */
  imgSize: number;
  /** Training metrics */
  metrics: {
    [key: string]: number;
  };
}

/**
 * Dataset validation result.
 */
export interface DatasetValidation {
  /** Whether the dataset is valid */
  valid: boolean;
  /** Number of training images */
  trainImages: number;
  /** Number of validation images */
  valImages: number;
  /** Number of training labels */
  trainLabels: number;
  /** Number of validation labels */
  valLabels: number;
  /** Validation errors */
  errors: string[];
  /** Validation warnings */
  warnings: string[];
}

/**
 * YOLO dependencies installation status.
 */
export interface YoloDependenciesInfo {
  /** Whether all dependencies are installed */
  installed: boolean;
  /** List of missing packages */
  missing: string[];
  /** Estimated download size */
  estimatedSize: string;
}

/**
 * YOLO Keypoint service status.
 */
export interface YoloKeypointStatus {
  /** Whether the service is available */
  available: boolean;
  /** Whether SSE streaming is available */
  sseAvailable: boolean;
  /** Number of active training jobs */
  activeTrainingJobs: number;
}
