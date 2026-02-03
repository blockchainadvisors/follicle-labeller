/**
 * Platform adapter interfaces
 * Defines the contract for platform-specific operations
 */

import type {
  GPUInfo,
  GPUHardwareInfo,
  TensorRTStatus,
  YoloKeypointStatus,
  YoloDetectionStatus,
  YoloDependenciesInfo,
  DatasetValidation,
  TrainingConfig,
  TrainingProgress,
  ModelInfo,
  KeypointPrediction,
  DetectionPrediction,
} from '../types';

// ============================================
// File Adapter Interface
// ============================================

export interface OpenImageResult {
  filePath: string;
  fileName: string;
  data: ArrayBuffer;
}

export interface OpenFileOptions {
  filters?: Array<{ name: string; extensions: string[] }>;
  title?: string;
}

export interface SaveProjectResult {
  success: boolean;
  filePath?: string;
}

export interface LoadProjectResult {
  version: '1.0' | '2.0';
  filePath: string;
  // V1 format
  imageFileName?: string;
  imageData?: ArrayBuffer;
  jsonData?: string;
  // V2 format
  manifest?: string;
  images?: Array<{ id: string; fileName: string; data: ArrayBuffer }>;
  annotations?: string;
}

export interface ProjectImageData {
  id: string;
  fileName: string;
  data: ArrayBuffer;
}

/**
 * File operations adapter
 * Handles file dialogs, project save/load, and file system operations
 */
export interface FileAdapter {
  /** Open single image dialog (for adding images) */
  openImageDialog(): Promise<OpenImageResult | null>;

  /** Open file dialog with filters */
  openFileDialog(options: OpenFileOptions): Promise<OpenImageResult | null>;

  /** Save V2 project (multi-image) */
  saveProjectV2(
    images: ProjectImageData[],
    manifest: string,
    annotations: string,
    defaultPath?: string
  ): Promise<SaveProjectResult>;

  /** Save V2 project to specific path */
  saveProjectV2ToPath(
    filePath: string,
    images: ProjectImageData[],
    manifest: string,
    annotations: string
  ): Promise<SaveProjectResult>;

  /** Load V2 project */
  loadProjectV2(): Promise<LoadProjectResult | null>;

  /** Load project from specific path */
  loadProjectFromPath(filePath: string): Promise<LoadProjectResult | null>;

  /** Check if file exists */
  fileExists(filePath: string): Promise<boolean>;

  /** Show unsaved changes dialog (returns 'save' | 'discard' | 'cancel') */
  showUnsavedChangesDialog(): Promise<'save' | 'discard' | 'cancel'>;

  /** Show download options dialog */
  showDownloadOptionsDialog(
    selectedCount: number,
    currentImageCount: number,
    totalCount: number
  ): Promise<'all' | 'currentImage' | 'selected' | 'cancel'>;

  /** Get file to open (from command line args) */
  getFileToOpen(): Promise<string | null>;

  /** Update menu state */
  setProjectState(hasProject: boolean): void;
}

// ============================================
// Blob Server Adapter Interface
// ============================================

/**
 * BLOB server adapter
 * Handles server lifecycle and availability checks
 */
export interface BlobAdapter {
  /** Start BLOB server */
  startServer(): Promise<{ success: boolean; error?: string; errorDetails?: string }>;

  /** Stop BLOB server */
  stopServer(): Promise<{ success: boolean }>;

  /** Check if server is available */
  isAvailable(): Promise<boolean>;

  /** Check Python availability */
  checkPython(): Promise<{ available: boolean; version?: string; error?: string }>;

  /** Get server info */
  getServerInfo(): Promise<{ port: number; running: boolean; scriptPath: string }>;

  /** Get setup status */
  getSetupStatus(): Promise<string>;

  /** Subscribe to setup progress (includes download percent when downloading Python) */
  onSetupProgress(callback: (status: string, percent?: number) => void): () => void;

  /** Get GPU info from running server */
  getGPUInfo(): Promise<GPUInfo>;

  /** Restart BLOB server */
  restartServer(): Promise<{ success: boolean; error?: string; errorDetails?: string }>;
}

// ============================================
// YOLO Keypoint Adapter Interface
// ============================================

export interface WriteDatasetResult {
  success: boolean;
  datasetPath?: string;
  error?: string;
}

export interface StartTrainingResult {
  jobId: string;
  status: string;
}

export interface ExportDialogResult {
  canceled: boolean;
  filePath?: string;
}

export interface ExportONNXResult {
  success: boolean;
  outputPath?: string;
}

export interface ExportTensorRTResult {
  success: boolean;
  engine_path?: string;
  error?: string;
}

export interface SystemInfo {
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
}

export interface PredictKeypointResult {
  success: boolean;
  prediction?: KeypointPrediction;
  message?: string;
}

/**
 * YOLO Keypoint adapter
 * Handles origin/direction prediction model training and inference
 */
export interface YoloKeypointAdapter {
  /** Get service status */
  getStatus(): Promise<YoloKeypointStatus>;

  /** Get system info (CPU, GPU, memory) */
  getSystemInfo(): Promise<SystemInfo>;

  /** Check dependencies */
  checkDependencies(): Promise<YoloDependenciesInfo>;

  /** Install dependencies */
  installDependencies(): Promise<{ success: boolean; error?: string }>;

  /** Upgrade to CUDA */
  upgradeToCUDA(): Promise<{ success: boolean; error?: string }>;

  /** Subscribe to install progress */
  onInstallProgress(callback: (data: { message: string; percent?: number }) => void): () => void;

  /** Validate dataset */
  validateDataset(datasetPath: string): Promise<DatasetValidation>;

  /** Write dataset to temp directory */
  writeDatasetToTemp(
    files: Array<{ path: string; content: ArrayBuffer | string }>
  ): Promise<WriteDatasetResult>;

  /** Start training */
  startTraining(
    datasetPath: string,
    config: TrainingConfig,
    modelName?: string
  ): Promise<StartTrainingResult>;

  /** Stop training */
  stopTraining(jobId: string): Promise<{ success: boolean }>;

  /** Subscribe to training progress */
  subscribeProgress(
    jobId: string,
    onProgress: (progress: TrainingProgress) => void,
    onError: (error: string) => void,
    onComplete: () => void
  ): () => void;

  /** List models */
  listModels(): Promise<{ models: ModelInfo[] }>;

  /** Load model for inference */
  loadModel(modelPath: string): Promise<{ success: boolean }>;

  /** Run prediction */
  predict(imageData: string): Promise<PredictKeypointResult>;

  /** Show export dialog (Electron only - Web returns mock) */
  showExportDialog(defaultFileName: string): Promise<ExportDialogResult>;

  /** Export to ONNX */
  exportONNX(modelPath: string, outputPath: string): Promise<ExportONNXResult>;

  /** Delete model */
  deleteModel(modelId: string): Promise<{ success: boolean }>;

  /** Check TensorRT availability */
  checkTensorRTAvailable(): Promise<TensorRTStatus>;

  /** Export to TensorRT */
  exportToTensorRT(
    modelPath: string,
    outputPath?: string,
    half?: boolean,
    imgsz?: number
  ): Promise<ExportTensorRTResult>;
}

// ============================================
// YOLO Detection Adapter Interface
// ============================================

export interface DetectionTrainingConfig {
  modelSize?: string;
  epochs?: number;
  imgSize?: number;
  batchSize?: number;
  patience?: number;
  device?: string;
  resumeFrom?: string;
}

export interface DetectionTrainingProgress {
  status: string;
  epoch: number;
  totalEpochs: number;
  loss: number;
  boxLoss: number;
  clsLoss: number;
  dflLoss: number;
  metrics: Record<string, number>;
  eta: string;
  message: string;
}

export interface DetectionModelInfo {
  id: string;
  name: string;
  path: string;
  createdAt: string;
  epochsTrained: number;
  imgSize: number;
  metrics: Record<string, number>;
}

export interface ResumableModelInfo extends DetectionModelInfo {
  epochsCompleted: number;
  totalEpochs: number;
  canResume: boolean;
}

export interface DetectionValidation {
  valid: boolean;
  train_images: number;
  val_images: number;
  train_labels: number;
  val_labels: number;
  errors: string[];
  warnings: string[];
}

export interface PredictDetectionResult {
  success: boolean;
  detections: DetectionPrediction[];
  count: number;
  backend?: string;
}

export interface PredictTiledResult extends PredictDetectionResult {
  method: string;
  tileSize: number;
  overlap: number;
  scaleFactor: number;
}

/**
 * YOLO Detection adapter
 * Handles bounding box detection model training and inference
 */
export interface YoloDetectionAdapter {
  /** Get service status */
  getStatus(): Promise<YoloDetectionStatus>;

  /** Validate dataset */
  validateDataset(datasetPath: string): Promise<DetectionValidation>;

  /** Start training */
  startTraining(
    datasetPath: string,
    config: DetectionTrainingConfig,
    modelName?: string
  ): Promise<StartTrainingResult>;

  /** Stop training */
  stopTraining(jobId: string): Promise<{ success: boolean }>;

  /** Subscribe to training progress */
  subscribeProgress(
    jobId: string,
    onProgress: (progress: DetectionTrainingProgress) => void,
    onError: (error: string) => void,
    onComplete: () => void
  ): () => void;

  /** List models */
  listModels(): Promise<{ models: DetectionModelInfo[] }>;

  /** Get resumable models */
  getResumableModels(): Promise<{ models: ResumableModelInfo[] }>;

  /** Load model for inference */
  loadModel(modelPath: string): Promise<{ success: boolean }>;

  /** Run prediction */
  predict(imageData: string, confidenceThreshold?: number): Promise<PredictDetectionResult>;

  /** Run tiled prediction (SAHI-style) */
  predictTiled(
    imageData: string,
    confidenceThreshold?: number,
    tileSize?: number,
    overlap?: number,
    nmsThreshold?: number,
    scaleFactor?: number
  ): Promise<PredictTiledResult>;

  /** Show export dialog (Electron only - Web returns mock) */
  showExportDialog(defaultFileName: string): Promise<ExportDialogResult>;

  /** Export to ONNX */
  exportONNX(modelPath: string, outputPath: string): Promise<ExportONNXResult>;

  /** Delete model */
  deleteModel(modelId: string): Promise<{ success: boolean }>;

  /** Write dataset to temp directory */
  writeDatasetToTemp(
    files: Array<{ path: string; content: ArrayBuffer | string }>
  ): Promise<WriteDatasetResult>;

  /** Check TensorRT availability */
  checkTensorRTAvailable(): Promise<TensorRTStatus>;

  /** Export to TensorRT */
  exportToTensorRT(
    modelPath: string,
    outputPath?: string,
    half?: boolean,
    imgsz?: number
  ): Promise<ExportTensorRTResult>;
}

// ============================================
// GPU Adapter Interface
// ============================================

/**
 * GPU hardware detection and package installation
 */
export interface GpuAdapter {
  /** Get hardware info */
  getHardwareInfo(): Promise<GPUHardwareInfo>;

  /** Install GPU packages (cupy, torch-cuda, etc.) */
  installPackages(): Promise<{ success: boolean; error?: string }>;

  /** Subscribe to install progress */
  onInstallProgress(callback: (data: { message: string; percent?: number }) => void): () => void;
}

// ============================================
// TensorRT Adapter Interface
// ============================================

/**
 * TensorRT installation and management
 */
export interface TensorRTAdapter {
  /** Check TensorRT availability */
  check(): Promise<{ available: boolean; version: string | null; canInstall: boolean }>;

  /** Install TensorRT */
  install(): Promise<{ success: boolean; error?: string }>;

  /** Subscribe to install progress */
  onInstallProgress(callback: (data: { message: string; percent?: number }) => void): () => void;
}

// ============================================
// Model Export/Import Adapter Interface
// ============================================

export interface ExportPackageResult {
  success: boolean;
  filePath?: string;
  canceled?: boolean;
  error?: string;
}

export interface PreviewPackageResult {
  valid: boolean;
  filePath?: string;
  config?: Record<string, unknown>;
  modelType?: 'detection' | 'keypoint';
  hasEngine?: boolean;
  canceled?: boolean;
  error?: string;
}

export interface ImportPackageResult {
  success: boolean;
  modelId?: string;
  modelPath?: string;
  modelName?: string;
  modelType?: 'detection' | 'keypoint';
  error?: string;
}

/**
 * Model package export/import
 */
export interface ModelAdapter {
  /** Export model package */
  exportPackage(
    modelId: string,
    modelPath: string,
    config: Record<string, unknown>,
    suggestedFileName?: string
  ): Promise<ExportPackageResult>;

  /** Preview model package before import */
  previewPackage(
    expectedModelType?: 'detection' | 'keypoint'
  ): Promise<PreviewPackageResult>;

  /** Import model package */
  importPackage(
    filePath: string,
    newModelName?: string
  ): Promise<ImportPackageResult>;
}

// ============================================
// Menu Listener Adapter Interface (Electron only)
// ============================================

/**
 * Menu event listeners (Electron only)
 * Web platform returns no-op functions
 */
export interface MenuAdapter {
  /** Set project state (enables/disables menu items) */
  setProjectState(hasProject: boolean): void;
  /** Get file to open on startup (from file association) */
  getFileToOpen(): Promise<string | null>;
  onMenuOpenImage(callback: () => void): () => void;
  onMenuLoadProject(callback: () => void): () => void;
  onMenuSaveProject(callback: () => void): () => void;
  onMenuSaveProjectAs(callback: () => void): () => void;
  onMenuCloseProject(callback: () => void): () => void;
  onMenuUndo(callback: () => void): () => void;
  onMenuRedo(callback: () => void): () => void;
  onMenuClearAll(callback: () => void): () => void;
  onMenuToggleShapes(callback: () => void): () => void;
  onMenuToggleLabels(callback: () => void): () => void;
  onMenuZoomIn(callback: () => void): () => void;
  onMenuZoomOut(callback: () => void): () => void;
  onMenuResetZoom(callback: () => void): () => void;
  onMenuShowHelp(callback: () => void): () => void;
  onCheckUnsavedChanges(callback: () => void): () => void;
  onFileOpen(callback: (filePath: string) => void): () => void;
  onSystemSuspend(callback: () => void): () => void;
  confirmClose(canClose: boolean): void;
}

// ============================================
// Platform Interface
// ============================================

/**
 * Combined platform interface
 * Provides access to all platform-specific adapters
 */
export interface Platform {
  /** Platform name */
  readonly name: 'electron' | 'web';

  /** File operations */
  readonly file: FileAdapter;

  /** BLOB server operations */
  readonly blob: BlobAdapter;

  /** YOLO keypoint operations */
  readonly yoloKeypoint: YoloKeypointAdapter;

  /** YOLO detection operations */
  readonly yoloDetection: YoloDetectionAdapter;

  /** GPU hardware operations */
  readonly gpu: GpuAdapter;

  /** TensorRT operations */
  readonly tensorrt: TensorRTAdapter;

  /** Model export/import */
  readonly model: ModelAdapter;

  /** Menu listeners (Electron only - Web returns no-ops) */
  readonly menu: MenuAdapter;
}

// ============================================
// Server Project Storage Types (Web only)
// ============================================

export interface ServerProject {
  id: string;
  name: string;
  createdAt: string;
  updatedAt: string;
  imageCount: number;
  annotationCount: number;
}

export interface ServerProjectList {
  projects: ServerProject[];
}
