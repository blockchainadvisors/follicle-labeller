import { useState, useEffect, useRef, useCallback } from 'react';
import { X, Cpu, Zap, Download, Loader2, AlertCircle, Crosshair, Box, Brain, ChevronDown } from 'lucide-react';
import type { BlobDetectionOptions, GPUInfo, GPUHardwareInfo, ModelInfo, DetectionModelInfo, DetectionMethod, TensorRTStatus, YoloInferenceBackend } from '../../types';
import { blobService } from '../../services/blobService';
import { yoloKeypointService } from '../../services/yoloKeypointService';
import { yoloDetectionService } from '../../services/yoloDetectionService';
import './DetectionSettingsDialog.css';

export interface DetectionSettings {
  // Detection method: 'blob' (SimpleBlobDetector) or 'yolo' (YOLO AI)
  detectionMethod: DetectionMethod;

  // YOLO Detection settings
  yoloModelId: string | null;  // Selected model ID (null = pre-trained/default)
  yoloModelName: string | null;  // Human-readable model name for display
  yoloModelSource: 'pretrained' | 'custom';  // Whether this is a pre-trained or custom model

  // Keypoint model settings (for origin prediction)
  keypointModelId: string | null;  // Selected keypoint model ID
  keypointModelName: string | null;  // Human-readable keypoint model name
  keypointModelSource: 'pretrained' | 'custom';  // Keypoint model type
  yoloConfidenceThreshold: number;  // Confidence threshold for rectangle detection (0-1)
  keypointConfidenceThreshold: number;  // Confidence threshold for origin prediction (0-1)
  yoloUseTiledInference: boolean;  // Use tiled inference for large images
  yoloTileSize: number;  // Tile size for tiled inference (should match training tile size)
  yoloTileOverlap: number;  // Overlap between tiles in pixels
  yoloNmsThreshold: number;  // IoU threshold for NMS when merging tile results
  yoloScaleFactor: number;  // Upscale factor for images with smaller objects than training data
  yoloAutoScaleMode: 'auto' | 'none';  // 'auto' = infer from annotations or image size, 'none' = manual
  yoloTrainingImageSize: number;  // Reference training image size (pixels, for imageSize mode)
  yoloTrainingAnnotationSize: number;  // Reference annotation size from training (pixels, for annotations mode)

  // Basic size parameters (for blob detection)
  minWidth: number;
  maxWidth: number;
  minHeight: number;
  maxHeight: number;
  darkBlobs: boolean;

  // Backend selection (CPU vs GPU)
  forceCPU: boolean;

  // Gaussian blur preprocessing
  useGaussianBlur: boolean;
  gaussianKernelSize: number;

  // Morphological opening
  useMorphOpen: boolean;
  morphKernelSize: number;

  // Circularity filter
  useCircularityFilter: boolean;
  minCircularity: number;

  // Inertia filter (for elongated shapes like hair follicles)
  useInertiaFilter: boolean;
  minInertiaRatio: number;
  maxInertiaRatio: number;

  // Convexity filter
  useConvexityFilter: boolean;
  minConvexity: number;

  // CLAHE preprocessing
  useCLAHE: boolean;
  claheClipLimit: number;
  claheTileSize: number;

  // SAHI-style tiling
  useSAHI: boolean;
  tileSize: number;
  tileOverlap: number;

  // Soft-NMS
  useSoftNMS: boolean;
  softNMSSigma: number;
  softNMSThreshold: number;

  // YOLO Keypoint prediction (auto-predict origins for detected follicles)
  useKeypointPrediction: boolean;

  // YOLO inference backend selection (PyTorch or TensorRT)
  yoloInferenceBackend: YoloInferenceBackend;

  // Keypoint inference backend selection (PyTorch or TensorRT)
  keypointInferenceBackend: YoloInferenceBackend;
}

export const DEFAULT_DETECTION_SETTINGS: DetectionSettings = {
  // Detection method - default to blob detection
  detectionMethod: 'blob',
  // YOLO Detection settings
  yoloModelId: null,
  yoloModelName: null,
  yoloModelSource: 'pretrained',
  // Keypoint model settings
  keypointModelId: null,
  keypointModelName: null,
  keypointModelSource: 'pretrained',
  yoloConfidenceThreshold: 0.5,
  keypointConfidenceThreshold: 0.3,  // Default threshold for origin prediction
  yoloUseTiledInference: true,  // Enable by default for better results on large images
  yoloTileSize: 1024,  // Match typical training tile size
  yoloTileOverlap: 128,  // 128px overlap between tiles
  yoloNmsThreshold: 0.5,  // Standard IoU threshold for NMS
  yoloScaleFactor: 1.0,  // No upscaling by default (use 1.5-2.0 for lower res images)
  yoloAutoScaleMode: 'auto',  // Default: auto-select based on available annotations
  yoloTrainingImageSize: 12240,  // Reference training image size (12240x12240)
  yoloTrainingAnnotationSize: 57,  // Average annotation size from training (~57px)
  // Blob detection size parameters
  minWidth: 10,
  maxWidth: 200,
  minHeight: 10,
  maxHeight: 200,
  darkBlobs: true,
  // Backend selection - use GPU by default when available
  forceCPU: false,
  // Gaussian blur - matches OpenCV pipeline
  useGaussianBlur: true,
  gaussianKernelSize: 5,
  // Morphological opening - separates touching objects
  useMorphOpen: true,
  morphKernelSize: 3,
  // Circularity filter - DISABLED for hair follicles (they are elongated, not circular)
  useCircularityFilter: false,
  minCircularity: 0.2,
  // Inertia filter - ENABLED for hair follicles (allows elongated shapes)
  useInertiaFilter: true,
  minInertiaRatio: 0.01,  // Very low = allows elongated shapes
  maxInertiaRatio: 1.0,   // Up to perfectly circular
  // Convexity filter - disabled by default
  useConvexityFilter: false,
  minConvexity: 0.5,
  // CLAHE - matches Python clipLimit=3.0
  useCLAHE: true,
  claheClipLimit: 3.0,
  claheTileSize: 8,
  // SAHI tiling
  useSAHI: false,
  tileSize: 512,
  tileOverlap: 0.2,
  // Soft-NMS
  useSoftNMS: true,
  softNMSSigma: 0.5,
  softNMSThreshold: 0.1,
  // YOLO Keypoint prediction (disabled by default - requires loaded model)
  useKeypointPrediction: false,
  // YOLO inference backend - default to PyTorch
  yoloInferenceBackend: 'pytorch',
  // Keypoint inference backend - default to PyTorch
  keypointInferenceBackend: 'pytorch',
};

// Install state that can be managed by parent for persistence
export interface GPUInstallState {
  isInstalling: boolean;
  progress: string;
  error: string | null;
}

// TensorRT export state that can be managed by parent for persistence
export interface TensorRTExportState {
  isExporting: boolean;
  progress: string;
  error: string | null;
  enginePath: string | null;
  completed: boolean;
  startTime?: number;  // Timestamp when export started (for elapsed time display)
  estimatedSeconds?: number;  // Rough estimate of total export time
}

interface DetectionSettingsDialogProps {
  settings: DetectionSettings;
  onClose: () => void;
  blobServerConnected?: boolean;
  onServerRestarted?: () => void;
  // Optional install state from parent for persistence across dialog close/reopen
  installState?: GPUInstallState;
  onInstallStateChange?: (state: GPUInstallState) => void;
  // Optional TensorRT export state for detection models (persists across dialog close/reopen)
  detectionExportState?: TensorRTExportState;
  onDetectionExportStateChange?: (state: TensorRTExportState) => void;
  // Optional TensorRT export state for keypoint models (persists across dialog close/reopen)
  keypointExportState?: TensorRTExportState;
  onKeypointExportStateChange?: (state: TensorRTExportState) => void;
  // Live settings update callbacks (changes apply immediately)
  onSettingsChange: (settings: DetectionSettings) => void;
  // Per-image settings support
  activeImageId?: string | null;
  activeImageName?: string;
  hasImageOverride?: boolean;
  globalSettings?: DetectionSettings;  // Global settings for comparison/reset
  onImageSettingsChange?: (imageId: string, settings: Partial<DetectionSettings>) => void;
  onClearImageOverride?: (imageId: string) => void;
  // Navigation to Model Library
  onOpenModelLibrary?: (tab: 'detection' | 'origin') => void;
}

export function DetectionSettingsDialog({
  settings,
  onClose,
  blobServerConnected = false,
  onServerRestarted,
  installState,
  onInstallStateChange,
  detectionExportState,
  onDetectionExportStateChange,
  keypointExportState,
  onKeypointExportStateChange,
  onSettingsChange,
  activeImageId,
  activeImageName,
  hasImageOverride = false,
  globalSettings,
  onImageSettingsChange,
  onClearImageOverride,
  onOpenModelLibrary,
}: DetectionSettingsDialogProps) {
  // Track whether we're editing per-image or global settings
  const [applyToImageOnly, setApplyToImageOnly] = useState(hasImageOverride);
  const [gpuInfo, setGpuInfo] = useState<GPUInfo | null>(null);
  const [gpuHardware, setGpuHardware] = useState<GPUHardwareInfo | null>(null);
  const [loadedKeypointModel, setLoadedKeypointModel] = useState<ModelInfo | null>(null);
  const [keypointModelsAvailable, setKeypointModelsAvailable] = useState(false);

  // YOLO Detection models state
  const [detectionModels, setDetectionModels] = useState<DetectionModelInfo[]>([]);
  const [yoloDetectionAvailable, setYoloDetectionAvailable] = useState(false);

  // TensorRT state
  const [tensorrtStatus, setTensorrtStatus] = useState<TensorRTStatus | null>(null);
  const [tensorrtInstalling, setTensorrtInstalling] = useState(false);
  const [tensorrtInstallProgress, setTensorrtInstallProgress] = useState<string>('');
  const [tensorrtInstallError, setTensorrtInstallError] = useState<string | null>(null);
  const [tensorrtCanInstall, setTensorrtCanInstall] = useState(false);

  // TensorRT export state (for exporting .pt to .engine) - Detection
  // Local state used when parent doesn't provide controlled state
  const [localDetectionExportState, setLocalDetectionExportState] = useState<TensorRTExportState>({
    isExporting: false,
    progress: '',
    error: null,
    enginePath: null,
    completed: false,
  });
  const [engineAvailable, setEngineAvailable] = useState<boolean | null>(null);

  // Use parent-controlled state if provided, otherwise use local state
  const detectionExport = detectionExportState ?? localDetectionExportState;
  // Use a ref to always have the latest state value (avoids stale closure)
  const detectionExportRef = useRef(detectionExport);
  detectionExportRef.current = detectionExport;

  const updateDetectionExportState = useCallback((updates: Partial<TensorRTExportState>) => {
    const newState = { ...detectionExportRef.current, ...updates };
    // Update ref immediately so consecutive calls see the latest state
    detectionExportRef.current = newState;
    if (onDetectionExportStateChange) {
      onDetectionExportStateChange(newState);
    } else {
      setLocalDetectionExportState(newState);
    }
  }, [onDetectionExportStateChange]);

  // Keypoint TensorRT state
  const [keypointTensorrtStatus, setKeypointTensorrtStatus] = useState<TensorRTStatus | null>(null);
  const [localKeypointExportState, setLocalKeypointExportState] = useState<TensorRTExportState>({
    isExporting: false,
    progress: '',
    error: null,
    enginePath: null,
    completed: false,
  });
  const [keypointEngineAvailable, setKeypointEngineAvailable] = useState<boolean | null>(null);

  // Use parent-controlled state if provided, otherwise use local state
  const keypointExport = keypointExportState ?? localKeypointExportState;
  // Use a ref to always have the latest state value (avoids stale closure)
  const keypointExportRef = useRef(keypointExport);
  keypointExportRef.current = keypointExport;

  const updateKeypointExportState = useCallback((updates: Partial<TensorRTExportState>) => {
    const newState = { ...keypointExportRef.current, ...updates };
    // Update ref immediately so consecutive calls see the latest state
    keypointExportRef.current = newState;
    if (onKeypointExportStateChange) {
      onKeypointExportStateChange(newState);
    } else {
      setLocalKeypointExportState(newState);
    }
  }, [onKeypointExportStateChange]);

  // Collapsible section state
  const [detectionCollapsed, setDetectionCollapsed] = useState(false);
  const [keypointCollapsed, setKeypointCollapsed] = useState(false);

  // Elapsed time for export progress display
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  // Update elapsed time every second during export
  useEffect(() => {
    if (!keypointExport.isExporting && !detectionExport.isExporting) {
      setElapsedSeconds(0);
      return;
    }

    const startTime = keypointExport.isExporting ? keypointExport.startTime : detectionExport.startTime;
    if (!startTime) return;

    // Calculate initial elapsed time (in case dialog was reopened)
    setElapsedSeconds(Math.floor((Date.now() - startTime) / 1000));

    const interval = setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - startTime) / 1000));
    }, 1000);

    return () => clearInterval(interval);
  }, [keypointExport.isExporting, keypointExport.startTime, detectionExport.isExporting, detectionExport.startTime]);

  // Draggable dialog state
  const [position, setPosition] = useState<{ x: number; y: number } | null>(null);
  const dragRef = useRef<{ startX: number; startY: number; startPosX: number; startPosY: number } | null>(null);
  const dialogRef = useRef<HTMLDivElement>(null);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    // Don't start drag if clicking on a button
    if ((e.target as HTMLElement).closest('button')) return;

    const dialog = dialogRef.current;
    if (!dialog) return;

    const rect = dialog.getBoundingClientRect();
    const currentX = position?.x ?? rect.left + rect.width / 2 - window.innerWidth / 2;
    const currentY = position?.y ?? rect.top + rect.height / 2 - window.innerHeight / 2;

    dragRef.current = {
      startX: e.clientX,
      startY: e.clientY,
      startPosX: currentX,
      startPosY: currentY,
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  }, [position]);

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!dragRef.current) return;

    const deltaX = e.clientX - dragRef.current.startX;
    const deltaY = e.clientY - dragRef.current.startY;

    setPosition({
      x: dragRef.current.startPosX + deltaX,
      y: dragRef.current.startPosY + deltaY,
    });
  }, []);

  const handleMouseUp = useCallback(() => {
    dragRef.current = null;
    document.removeEventListener('mousemove', handleMouseMove);
    document.removeEventListener('mouseup', handleMouseUp);
  }, [handleMouseMove]);

  // Cleanup event listeners on unmount
  useEffect(() => {
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [handleMouseMove, handleMouseUp]);

  // Use parent state if provided, otherwise use local state
  const [localIsInstalling, setLocalIsInstalling] = useState(false);
  const [localInstallProgress, setLocalInstallProgress] = useState<string>('');
  const [localInstallError, setLocalInstallError] = useState<string | null>(null);

  // Determine which state to use (parent-controlled or local)
  const isInstalling = installState?.isInstalling ?? localIsInstalling;
  const installProgress = installState?.progress ?? localInstallProgress;
  const installError = installState?.error ?? localInstallError;

  // Helper to update install state (updates parent if controlled, local otherwise)
  const updateInstallState = (updates: Partial<GPUInstallState>) => {
    if (onInstallStateChange && installState) {
      onInstallStateChange({ ...installState, ...updates });
    } else {
      if (updates.isInstalling !== undefined) setLocalIsInstalling(updates.isInstalling);
      if (updates.progress !== undefined) setLocalInstallProgress(updates.progress);
      if (updates.error !== undefined) setLocalInstallError(updates.error);
    }
  };

  // Fetch GPU info from server when connected
  useEffect(() => {
    if (!blobServerConnected) return;

    const fetchGpuInfo = async () => {
      try {
        const info = await blobService.getGPUInfo();
        setGpuInfo(info);
      } catch (error) {
        console.error('Failed to fetch GPU info:', error);
      }
    };
    fetchGpuInfo();
  }, [blobServerConnected]);

  // Fetch hardware info and listen for install progress on mount
  useEffect(() => {
    const fetchHardwareInfo = async () => {
      try {
        const info = await window.electronAPI.gpu.getHardwareInfo();
        setGpuHardware(info);
      } catch (error) {
        console.error('Failed to fetch GPU hardware info:', error);
      }
    };
    fetchHardwareInfo();

    // Listen for install progress
    const cleanup = window.electronAPI.gpu.onInstallProgress(({ message }) => {
      updateInstallState({ progress: message });
    });

    return cleanup;
  }, [installState, onInstallStateChange]);

  // Check for available YOLO keypoint models
  useEffect(() => {
    const checkKeypointModels = async () => {
      try {
        const status = await yoloKeypointService.getStatus();
        if (status.available) {
          const models = await yoloKeypointService.listModels();
          setKeypointModelsAvailable(models.length > 0);
          // Check if any model is currently loaded by finding one marked as loaded
          // For now, we just show if models are available
          if (models.length > 0) {
            setLoadedKeypointModel(models[0]); // Show first model as available
          }
        }
      } catch (error) {
        console.error('Failed to check keypoint models:', error);
      }
    };
    checkKeypointModels();
  }, []);

  // Check keypoint TensorRT availability
  useEffect(() => {
    const checkKeypointTensorRT = async () => {
      try {
        const trtStatus = await yoloKeypointService.checkTensorRTAvailable();
        setKeypointTensorrtStatus(trtStatus);
      } catch (error) {
        console.error('Failed to check keypoint TensorRT:', error);
      }
    };
    checkKeypointTensorRT();
  }, []);

  // Check for available YOLO detection models and TensorRT status
  useEffect(() => {
    const checkDetectionModels = async () => {
      try {
        const status = await yoloDetectionService.getStatus();
        setYoloDetectionAvailable(status.available);
        if (status.available) {
          const models = await yoloDetectionService.listModels();
          setDetectionModels(models);
          // Check TensorRT availability via Python service
          const trtStatus = await yoloDetectionService.checkTensorRTAvailable();
          setTensorrtStatus(trtStatus);
        }
        // Also check via native API for install capability
        const nativeTrtStatus = await window.electronAPI.tensorrt.check();
        setTensorrtCanInstall(nativeTrtStatus.canInstall && !nativeTrtStatus.available);
      } catch (error) {
        console.error('Failed to check detection models:', error);
      }
    };
    checkDetectionModels();

    // Listen for TensorRT install progress
    const cleanup = window.electronAPI.tensorrt.onInstallProgress(({ message }) => {
      setTensorrtInstallProgress(message);
    });

    return cleanup;
  }, []);

  // Check if TensorRT engine file exists for the current model
  useEffect(() => {
    const checkEngineAvailability = async () => {
      // Only check if TensorRT is available and we have a custom model selected
      if (!tensorrtStatus?.available) {
        setEngineAvailable(null);
        updateDetectionExportState({ enginePath: null });
        return;
      }

      // Get the model path
      let modelPath: string | null = null;
      if (settings.yoloModelId && settings.yoloModelSource === 'custom') {
        const selectedModel = detectionModels.find(m => m.id === settings.yoloModelId);
        if (selectedModel) {
          modelPath = selectedModel.path;
        }
      }

      if (!modelPath) {
        // Pre-trained model - engine would need to be created differently
        setEngineAvailable(null);
        updateDetectionExportState({ enginePath: null });
        return;
      }

      // Check if .engine file exists (replace .pt with .engine)
      const expectedEnginePath = modelPath.replace(/\.pt$/i, '.engine');
      try {
        const exists = await window.electronAPI.fileExists(expectedEnginePath);
        setEngineAvailable(exists);
        updateDetectionExportState({ enginePath: exists ? expectedEnginePath : modelPath });
      } catch (error) {
        console.error('Failed to check engine file:', error);
        setEngineAvailable(false);
        updateDetectionExportState({ enginePath: modelPath });
      }
    };

    checkEngineAvailability();
  }, [settings.yoloModelId, settings.yoloModelSource, tensorrtStatus?.available, detectionModels]);

  // Handle TensorRT engine export for detection
  const handleExportEngine = async () => {
    if (!detectionExport.enginePath || detectionExport.isExporting) return;

    const modelPath = detectionExport.enginePath.endsWith('.engine')
      ? detectionExport.enginePath.replace(/\.engine$/i, '.pt')
      : detectionExport.enginePath;

    // Use the model's actual image size for TensorRT export
    // This is critical - mismatched image size causes incorrect predictions
    const selectedModel = detectionModels.find(m => m.id === settings.yoloModelId);
    const imgSize = selectedModel?.imgSize || 640;

    // Estimate export time based on model variant/parameters and image size
    // YOLO variants: nano(n) < small(s) < medium(m) < large(l) < xlarge(x)
    let variantMultiplier = 1.0;

    // Use actual model metadata if available
    if (selectedModel?.modelVariant) {
      // Use real model variant from metadata
      const variant = selectedModel.modelVariant;
      variantMultiplier = variant === 'n' ? 0.5 : variant === 's' ? 0.7 : variant === 'm' ? 1.0 : variant === 'l' ? 1.3 : 1.8;
    } else if (selectedModel?.parameters) {
      // Estimate from parameter count
      const params = selectedModel.parameters;
      if (params < 5_000_000) variantMultiplier = 0.5;
      else if (params < 15_000_000) variantMultiplier = 0.7;
      else if (params < 25_000_000) variantMultiplier = 1.0;
      else if (params < 40_000_000) variantMultiplier = 1.3;
      else variantMultiplier = 1.8;
    } else {
      // Fallback: try to detect from model name
      const modelName = (selectedModel?.name || '').toLowerCase();
      if (modelName.includes('nano') || modelName.includes('-n-') || modelName.startsWith('n-')) {
        variantMultiplier = 0.5;
      } else if (modelName.includes('small') || modelName.includes('-s-') || modelName.startsWith('s-')) {
        variantMultiplier = 0.7;
      } else if (modelName.includes('medium') || modelName.includes('-m-') || modelName.startsWith('m-')) {
        variantMultiplier = 1.0;
      } else if (modelName.includes('large') || modelName.includes('-l-') || modelName.startsWith('l-')) {
        variantMultiplier = 1.3;
      } else if (modelName.includes('xlarge') || modelName.includes('-x-') || modelName.startsWith('x-')) {
        variantMultiplier = 1.8;
      }
    }

    // Base time by image size, then multiply by variant
    // Calibrated from real-world data: large model at 320px took ~6 minutes
    const baseSeconds = imgSize <= 320 ? 280 : imgSize <= 640 ? 480 : 720;
    const estimatedSeconds = Math.round(baseSeconds * variantMultiplier);

    updateDetectionExportState({
      isExporting: true,
      error: null,
      progress: 'Starting TensorRT export...',
      completed: false,
      startTime: Date.now(),
      estimatedSeconds,
    });

    try {
      // Update progress to show we're building
      updateDetectionExportState({ progress: 'Building TensorRT engine (FP16)...' });
      console.log(`Exporting detection model to TensorRT with imgsz=${imgSize}`);
      const result = await yoloDetectionService.exportToTensorRT(modelPath, undefined, true, imgSize);

      if (result.success && result.engine_path) {
        setEngineAvailable(true);
        updateDetectionExportState({
          isExporting: false,
          enginePath: result.engine_path,
          progress: '',
          completed: true,
        });
        console.log('TensorRT engine exported:', result.engine_path);
      } else {
        updateDetectionExportState({
          isExporting: false,
          error: result.error || 'Export failed',
          progress: '',
        });
      }
    } catch (error) {
      updateDetectionExportState({
        isExporting: false,
        error: error instanceof Error ? error.message : 'Export failed',
        progress: '',
      });
    }
  };

  // Check if TensorRT engine file exists for the current keypoint model
  useEffect(() => {
    const checkKeypointEngineAvailability = async () => {
      // Only check if TensorRT is available and we have a keypoint model
      if (!keypointTensorrtStatus?.available || !loadedKeypointModel) {
        setKeypointEngineAvailable(null);
        updateKeypointExportState({ enginePath: null });
        return;
      }

      const modelPath = loadedKeypointModel.path;

      // Check if .engine file exists (replace .pt with .engine)
      const expectedEnginePath = modelPath.replace(/\.pt$/i, '.engine');
      try {
        const exists = await window.electronAPI.fileExists(expectedEnginePath);
        setKeypointEngineAvailable(exists);
        updateKeypointExportState({ enginePath: exists ? expectedEnginePath : modelPath });
      } catch (error) {
        console.error('Failed to check keypoint engine file:', error);
        setKeypointEngineAvailable(false);
        updateKeypointExportState({ enginePath: modelPath });
      }
    };

    checkKeypointEngineAvailability();
  }, [loadedKeypointModel, keypointTensorrtStatus?.available]);

  // Handle TensorRT engine export for keypoint
  const handleExportKeypointEngine = async () => {
    if (!keypointExport.enginePath || keypointExport.isExporting) return;

    const modelPath = keypointExport.enginePath.endsWith('.engine')
      ? keypointExport.enginePath.replace(/\.engine$/i, '.pt')
      : keypointExport.enginePath;

    // Use the model's actual image size for TensorRT export
    // This is critical - mismatched image size causes incorrect predictions
    const imgSize = loadedKeypointModel?.imgSize || 640;

    // Estimate export time based on model variant/parameters and image size
    // YOLO variants: nano(n) < small(s) < medium(m) < large(l) < xlarge(x)
    let variantMultiplier = 1.0;

    // Use actual model metadata if available
    if (loadedKeypointModel?.modelVariant) {
      // Use real model variant from metadata
      const variant = loadedKeypointModel.modelVariant;
      variantMultiplier = variant === 'n' ? 0.5 : variant === 's' ? 0.7 : variant === 'm' ? 1.0 : variant === 'l' ? 1.3 : 1.8;
    } else if (loadedKeypointModel?.parameters) {
      // Estimate from parameter count
      const params = loadedKeypointModel.parameters;
      if (params < 5_000_000) variantMultiplier = 0.5;
      else if (params < 15_000_000) variantMultiplier = 0.7;
      else if (params < 25_000_000) variantMultiplier = 1.0;
      else if (params < 40_000_000) variantMultiplier = 1.3;
      else variantMultiplier = 1.8;
    } else {
      // Fallback: try to detect from model name
      const modelName = (loadedKeypointModel?.name || '').toLowerCase();
      if (modelName.includes('nano') || modelName.includes('-n-') || modelName.startsWith('n-')) {
        variantMultiplier = 0.5;
      } else if (modelName.includes('small') || modelName.includes('-s-') || modelName.startsWith('s-')) {
        variantMultiplier = 0.7;
      } else if (modelName.includes('medium') || modelName.includes('-m-') || modelName.startsWith('m-')) {
        variantMultiplier = 1.0;
      } else if (modelName.includes('large') || modelName.includes('-l-') || modelName.startsWith('l-')) {
        variantMultiplier = 1.3;
      } else if (modelName.includes('xlarge') || modelName.includes('-x-') || modelName.startsWith('x-')) {
        variantMultiplier = 1.8;
      }
    }

    // Base time by image size, then multiply by variant
    // Calibrated from real-world data: large model at 320px took ~6 minutes
    const baseSeconds = imgSize <= 320 ? 280 : imgSize <= 640 ? 480 : 720;
    const estimatedSeconds = Math.round(baseSeconds * variantMultiplier);

    updateKeypointExportState({
      isExporting: true,
      error: null,
      progress: 'Starting TensorRT export...',
      completed: false,
      startTime: Date.now(),
      estimatedSeconds,
    });

    try {
      // Update progress to show we're building
      updateKeypointExportState({ progress: 'Building TensorRT engine (FP16)...' });
      console.log(`Exporting keypoint model to TensorRT with imgsz=${imgSize}`);
      const result = await yoloKeypointService.exportToTensorRT(modelPath, undefined, true, imgSize);

      if (result.success && result.engine_path) {
        setKeypointEngineAvailable(true);
        updateKeypointExportState({
          isExporting: false,
          enginePath: result.engine_path,
          progress: '',
          completed: true,
        });
        console.log('Keypoint TensorRT engine exported:', result.engine_path);
      } else {
        updateKeypointExportState({
          isExporting: false,
          error: result.error || 'Export failed',
          progress: '',
        });
      }
    } catch (error) {
      updateKeypointExportState({
        isExporting: false,
        error: error instanceof Error ? error.message : 'Export failed',
        progress: '',
      });
    }
  };

  // Handle GPU package installation
  const handleInstallGPU = async () => {
    updateInstallState({ isInstalling: true, error: null, progress: 'Starting installation...' });

    try {
      const result = await window.electronAPI.gpu.installPackages();

      if (result.success) {
        updateInstallState({ progress: 'Restarting detection server...' });
        // Restart the blob server to pick up new packages
        await window.electronAPI.blob.restartServer();

        // Notify parent to recreate session (server restart invalidates old sessions)
        onServerRestarted?.();

        // Refresh GPU info
        const info = await window.electronAPI.gpu.getHardwareInfo();
        setGpuHardware(info);

        // Also refresh gpuInfo for active backend display
        const gpuStatus = await blobService.getGPUInfo();
        setGpuInfo(gpuStatus);

        updateInstallState({ isInstalling: false, progress: '' });
      } else {
        updateInstallState({ isInstalling: false, error: result.error || 'Installation failed' });
      }
    } catch (error) {
      updateInstallState({ isInstalling: false, error: error instanceof Error ? error.message : 'Installation failed' });
    }
  };

  // Handle TensorRT installation
  const handleInstallTensorRT = async () => {
    setTensorrtInstalling(true);
    setTensorrtInstallError(null);
    setTensorrtInstallProgress('Starting TensorRT installation...');

    try {
      const result = await window.electronAPI.tensorrt.install();

      if (result.success) {
        setTensorrtInstallProgress('Restarting detection server...');
        // Restart the blob server to pick up new packages
        await window.electronAPI.blob.restartServer();

        // Notify parent to recreate session
        onServerRestarted?.();

        // Refresh TensorRT status
        const trtStatus = await yoloDetectionService.checkTensorRTAvailable();
        setTensorrtStatus(trtStatus);
        setTensorrtCanInstall(false);

        setTensorrtInstalling(false);
        setTensorrtInstallProgress('');
      } else {
        setTensorrtInstalling(false);
        setTensorrtInstallError(result.error || 'TensorRT installation failed');
      }
    } catch (error) {
      setTensorrtInstalling(false);
      setTensorrtInstallError(error instanceof Error ? error.message : 'TensorRT installation failed');
    }
  };

  // Live update handler - immediately saves changes
  const handleChange = <K extends keyof DetectionSettings>(
    key: K,
    value: DetectionSettings[K]
  ) => {
    const newSettings = { ...settings, [key]: value };
    if (applyToImageOnly && activeImageId && onImageSettingsChange) {
      // Update per-image settings
      onImageSettingsChange(activeImageId, newSettings);
    } else {
      // Update global settings
      onSettingsChange(newSettings);
    }
  };

  const handleReset = () => {
    if (applyToImageOnly && globalSettings) {
      // Reset to global settings when in per-image mode
      if (activeImageId && onImageSettingsChange) {
        onImageSettingsChange(activeImageId, globalSettings);
      }
    } else {
      // Reset to defaults when in global mode
      onSettingsChange({ ...DEFAULT_DETECTION_SETTINGS });
    }
  };

  const handleClearImageOverride = () => {
    if (activeImageId && onClearImageOverride) {
      onClearImageOverride(activeImageId);
      setApplyToImageOnly(false);
    }
  };

  // Format seconds into "Xm Ys" format
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    if (mins > 0) {
      return `${mins}m ${secs}s`;
    }
    return `${secs}s`;
  };

  // Generate export progress message with elapsed time and estimate
  const getExportProgressMessage = (): string => {
    const exportState = keypointExport.isExporting ? keypointExport : detectionExport;
    const baseMessage = exportState.progress || 'Exporting to TensorRT...';
    const estimate = exportState.estimatedSeconds;

    if (elapsedSeconds > 0) {
      const elapsed = formatTime(elapsedSeconds);
      if (estimate) {
        const remaining = Math.max(0, estimate - elapsedSeconds);
        if (remaining > 0) {
          return `${baseMessage} (${elapsed} elapsed, ~${formatTime(remaining)} remaining)`;
        } else {
          return `${baseMessage} (${elapsed} elapsed, finishing up...)`;
        }
      }
      return `${baseMessage} (${elapsed} elapsed)`;
    }
    return baseMessage;
  };

  return (
    <div className="detection-settings-overlay" onClick={onClose}>
      <div
        ref={dialogRef}
        className="detection-settings-dialog"
        onClick={e => e.stopPropagation()}
        style={position ? { transform: `translate(${position.x}px, ${position.y}px)` } : undefined}
      >
        <div className="dialog-header" onMouseDown={handleMouseDown}>
          <h2>Inference Settings</h2>
          <button className="close-button" onClick={onClose}>
            <X size={18} />
          </button>
        </div>

        <div className="dialog-content">
          {/* Export progress indicator (visible at top when exporting) */}
          {(keypointExport.isExporting || detectionExport.isExporting) && (
            <div className="export-progress-banner">
              <Loader2 size={16} className="spin" />
              <span>{getExportProgressMessage()}</span>
            </div>
          )}

          {/* Per-Image Settings Toggle */}
          {activeImageId && onImageSettingsChange && (
            <section className="settings-section per-image-section">
              <div className="per-image-header">
                <label className="per-image-toggle">
                  <input
                    type="checkbox"
                    checked={applyToImageOnly}
                    onChange={e => setApplyToImageOnly(e.target.checked)}
                  />
                  <span>Apply to this image only</span>
                </label>
                {hasImageOverride && (
                  <button
                    className="clear-override-btn"
                    onClick={handleClearImageOverride}
                    title="Remove custom settings for this image"
                  >
                    Reset to Global
                  </button>
                )}
              </div>
              <p className="per-image-hint">
                {applyToImageOnly
                  ? `Settings will only apply to "${activeImageName || 'current image'}"`
                  : 'Settings will apply to all images in the project'}
              </p>
              {hasImageOverride && !applyToImageOnly && (
                <p className="per-image-warning">
                  ⚠ This image has custom settings. Saving globally won't change per-image settings.
                </p>
              )}
            </section>
          )}

          {/* Hardware Acceleration - Shared by both detection and keypoint */}
          <section className="settings-section shared-infrastructure">
            <h3>Hardware Acceleration</h3>

            {/* GPU Status Indicator */}
            {/* State 1: GPU Active - packages installed and working */}
            {gpuInfo && gpuInfo.activeBackend !== 'cpu' && (
              <div className={`gpu-status gpu-status-${gpuInfo.activeBackend}`}>
                <Zap size={16} />
                <div className="gpu-status-content">
                  <span className="gpu-status-label">
                    {gpuInfo.activeBackend === 'cuda' && 'CUDA Acceleration'}
                    {gpuInfo.activeBackend === 'mps' && 'Metal Acceleration'}
                  </span>
                  <span className="gpu-status-device">{gpuInfo.deviceName}</span>
                </div>
                {gpuInfo.memoryGB && (
                  <span className="gpu-status-memory">
                    {gpuInfo.memoryGB.toFixed(1)} GB
                  </span>
                )}
              </div>
            )}

            {/* State 2: GPU Available - hardware detected, packages not installed, show install button */}
            {gpuHardware?.canEnableGpu && !gpuHardware.gpuEnabled && !isInstalling && gpuInfo?.activeBackend === 'cpu' && (
              <div className="gpu-status gpu-status-available">
                <Zap size={16} />
                <div className="gpu-status-content">
                  <span className="gpu-status-label">GPU Available</span>
                  <span className="gpu-status-device">
                    {gpuHardware.hardware.nvidia.found
                      ? gpuHardware.hardware.nvidia.name
                      : gpuHardware.hardware.apple_silicon.chip}
                  </span>
                </div>
                <button className="gpu-install-btn" onClick={handleInstallGPU}>
                  <Download size={14} />
                  Install ({typeof window !== 'undefined' && navigator.platform.includes('Mac') ? '~2GB' : '~1.5GB'})
                </button>
              </div>
            )}

            {/* State 3: Installing - show progress */}
            {isInstalling && (
              <div className="gpu-status gpu-status-installing">
                <Loader2 size={16} className="spin" />
                <div className="gpu-status-content">
                  <div className="gpu-install-progress-container">
                    <span className="gpu-status-label">Installing GPU Packages</span>
                    <span className="gpu-status-progress">{installProgress || 'Starting...'}</span>
                    <div className="gpu-install-progress-bar">
                      <div className="gpu-install-progress-bar-indeterminate" />
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* State 4: Installation Error */}
            {installError && !isInstalling && (
              <div className="gpu-status gpu-status-error">
                <AlertCircle size={16} />
                <div className="gpu-status-content">
                  <span className="gpu-status-label">Installation Failed</span>
                  <span className="gpu-status-error-msg">{installError}</span>
                </div>
                <button className="gpu-retry-btn" onClick={handleInstallGPU}>
                  Retry
                </button>
              </div>
            )}

            {/* State 5: No GPU - CPU only mode */}
            {gpuInfo && gpuInfo.activeBackend === 'cpu' && !gpuHardware?.canEnableGpu && !isInstalling && !installError && (
              <div className="gpu-status gpu-status-cpu">
                <Cpu size={16} />
                <div className="gpu-status-content">
                  <span className="gpu-status-label">CPU Mode</span>
                  <span className="gpu-status-device">No GPU Detected</span>
                </div>
              </div>
            )}

            {/* Backend Selection - Show when GPU is available (installed or active) */}
            {gpuHardware?.gpuEnabled && (
              <div className="backend-selection-inline">
                <div className="backend-toggle">
                  <button
                    className={`backend-btn ${!settings.forceCPU ? 'active' : ''}`}
                    onClick={() => handleChange('forceCPU', false)}
                  >
                    <Zap size={14} />
                    GPU
                  </button>
                  <button
                    className={`backend-btn ${settings.forceCPU ? 'active' : ''}`}
                    onClick={() => handleChange('forceCPU', true)}
                  >
                    <Cpu size={14} />
                    CPU
                  </button>
                </div>
                <p className="backend-hint">
                  {settings.forceCPU
                    ? 'Using CPU for processing (slower but more compatible)'
                    : 'Using GPU for processing (faster)'}
                </p>
              </div>
            )}
          </section>

          {/* Section 1: Follicle Area Detection */}
          <section className="settings-section detection-section">
            <h3
              className="section-header clickable"
              onClick={() => setDetectionCollapsed(!detectionCollapsed)}
            >
              <ChevronDown size={16} className={`chevron ${detectionCollapsed ? 'rotated' : ''}`} />
              <Box size={16} />
              Follicle Area Detection
            </h3>

            {!detectionCollapsed && (
              <div className="section-content">
                {/* Detection Method Toggle */}
                <div className="detection-method-toggle">
                  <button
                    className={`method-btn ${settings.detectionMethod === 'blob' ? 'active' : ''}`}
                    onClick={() => handleChange('detectionMethod', 'blob')}
                  >
                    <Box size={16} />
                    SimpleBlobDetector
                  </button>
                  <button
                    className={`method-btn ${settings.detectionMethod === 'yolo' ? 'active' : ''}`}
                    onClick={() => handleChange('detectionMethod', 'yolo')}
                  >
                    <Brain size={16} />
                    YOLO (AI)
                  </button>
                </div>
                <p className="method-description">
                  {settings.detectionMethod === 'blob'
                    ? 'Uses OpenCV SimpleBlobDetector for classical computer vision based detection. Fast and reliable for uniform images.'
                    : 'Uses YOLO neural network for AI-powered detection. Better for complex images and varied lighting.'}
                </p>

                {/* YOLO Detection Settings (shown when YOLO method selected) */}
                {settings.detectionMethod === 'yolo' && (
                  <div className="method-settings yolo-settings">

              {/* Active Model Display */}
              <div className="current-model-display">
                <div className="model-label">Active Model</div>
                <div className="model-info-box">
                  {settings.yoloModelSource === 'pretrained' || !settings.yoloModelId ? (
                    <div className="model-details">
                      <span className="model-name">Pre-trained (yolo11n.pt)</span>
                      <span className="model-meta">Default YOLO model</span>
                    </div>
                  ) : (
                    <div className="model-details">
                      <span className="model-name">{settings.yoloModelName || 'Custom Model'}</span>
                      <span className="model-meta">Custom trained model</span>
                    </div>
                  )}
                  <button
                    className="switch-model-btn"
                    onClick={() => onOpenModelLibrary?.('detection')}
                  >
                    Switch Model
                  </button>
                </div>
              </div>
              {/* Show warning if saved model is not available */}
              {settings.yoloModelId && settings.yoloModelSource === 'custom' &&
               !detectionModels.find(m => m.id === settings.yoloModelId) && (
                <div className="yolo-warning">
                  <AlertCircle size={14} />
                  <span>
                    Saved model "{settings.yoloModelName || settings.yoloModelId}" not found.
                    Select another model from the Model Library.
                  </span>
                </div>
              )}

              {/* Inference Backend Selection */}
              <div className="settings-row">
                <label>Inference Backend</label>
                <select
                  value={settings.yoloInferenceBackend}
                  onChange={e => handleChange('yoloInferenceBackend', e.target.value as YoloInferenceBackend)}
                  disabled={!tensorrtStatus?.available && settings.yoloInferenceBackend !== 'tensorrt'}
                >
                  <option value="pytorch">PyTorch (Default)</option>
                  <option value="tensorrt" disabled={!tensorrtStatus?.available}>
                    TensorRT {tensorrtStatus?.available ? `(v${tensorrtStatus.version})` : '(Not installed)'}
                  </option>
                </select>
              </div>
              {/* TensorRT Install Section */}
              {!tensorrtStatus?.available && tensorrtCanInstall && !tensorrtInstalling && !tensorrtInstallError && (
                <div className="tensorrt-install-section">
                  <div className="tensorrt-install-info">
                    <Zap size={14} />
                    <span>TensorRT can speed up inference on NVIDIA GPUs</span>
                  </div>
                  <button className="tensorrt-install-btn" onClick={handleInstallTensorRT}>
                    <Download size={14} />
                    Install TensorRT (~500MB)
                  </button>
                </div>
              )}
              {tensorrtInstalling && (
                <div className="tensorrt-install-section installing">
                  <Loader2 size={14} className="spin" />
                  <div className="tensorrt-install-progress">
                    <span>{tensorrtInstallProgress || 'Installing TensorRT...'}</span>
                    <div className="tensorrt-install-progress-bar">
                      <div className="tensorrt-install-progress-bar-indeterminate" />
                    </div>
                  </div>
                </div>
              )}
              {tensorrtInstallError && !tensorrtInstalling && (
                <div className="tensorrt-install-section error">
                  <AlertCircle size={14} />
                  <span>{tensorrtInstallError}</span>
                  <button className="tensorrt-retry-btn" onClick={handleInstallTensorRT}>
                    Retry
                  </button>
                </div>
              )}
              {!tensorrtStatus?.available && !tensorrtCanInstall && gpuHardware?.hardware.nvidia.found && (
                <p className="setting-hint tensorrt-hint">
                  <Zap size={12} />
                  TensorRT requires CUDA. Install GPU packages first to enable TensorRT support.
                </p>
              )}
              {/* TensorRT engine status for selected model */}
              {settings.yoloInferenceBackend === 'tensorrt' && tensorrtStatus?.available && settings.yoloModelSource === 'custom' && (
                <>
                  {engineAvailable === false && !detectionExport.isExporting && !detectionExport.error && !detectionExport.completed && (
                    <div className="tensorrt-export-section">
                      <div className="tensorrt-export-warning">
                        <AlertCircle size={14} />
                        <span>No TensorRT engine found for "{settings.yoloModelName || settings.yoloModelId}"</span>
                      </div>
                      <button className="tensorrt-export-btn" onClick={handleExportEngine}>
                        <Zap size={14} />
                        Export to TensorRT
                      </button>
                      <p className="setting-hint">
                        Export creates a GPU-optimized .engine file for faster inference.
                      </p>
                    </div>
                  )}
                  {detectionExport.isExporting && (
                    <div className="tensorrt-export-section exporting">
                      <div className="export-progress-container">
                        <div className="export-progress-header">
                          <Loader2 size={14} className="spin" />
                          <span>{detectionExport.progress || 'Exporting to TensorRT...'}</span>
                        </div>
                        <div className="export-progress-bar">
                          <div className="export-progress-bar-indeterminate" />
                        </div>
                      </div>
                    </div>
                  )}
                  {detectionExport.error && !detectionExport.isExporting && (
                    <div className="tensorrt-export-section error">
                      <AlertCircle size={14} />
                      <span>{detectionExport.error}</span>
                      <button className="tensorrt-retry-btn" onClick={handleExportEngine}>
                        Retry
                      </button>
                    </div>
                  )}
                  {(engineAvailable === true || detectionExport.completed) && (
                    <p className="setting-hint tensorrt-ready">
                      <Zap size={12} />
                      TensorRT engine ready. Using GPU-optimized inference.
                    </p>
                  )}
                </>
              )}
              {settings.yoloInferenceBackend === 'tensorrt' && tensorrtStatus?.available && settings.yoloModelSource === 'pretrained' && (
                <p className="setting-hint">
                  TensorRT requires a custom trained model. Train a model first, then export it to TensorRT.
                </p>
              )}

              {/* Confidence Threshold */}
              <div className="settings-row">
                <label>Confidence Threshold</label>
                <input
                  type="range"
                  min="0.1"
                  max="0.9"
                  step="0.05"
                  value={settings.yoloConfidenceThreshold}
                  onChange={e => handleChange('yoloConfidenceThreshold', parseFloat(e.target.value))}
                />
                <span className="value-display">{(settings.yoloConfidenceThreshold * 100).toFixed(0)}%</span>
              </div>
              <p className="setting-hint">
                Higher threshold = fewer but more confident detections. Lower = more detections but possible false positives.
              </p>

              {/* Tiled Inference Toggle */}
              <div className="settings-row checkbox">
                <label>
                  <input
                    type="checkbox"
                    checked={settings.yoloUseTiledInference}
                    onChange={e => handleChange('yoloUseTiledInference', e.target.checked)}
                  />
                  Use Tiled Inference (for large images)
                </label>
              </div>
              <p className="setting-hint">
                Splits large images into overlapping tiles matching training size. Essential when model was trained on tiles.
              </p>

              {/* Tiled Inference Settings */}
              {settings.yoloUseTiledInference && (
                <>
                  <div className="settings-row">
                    <label>Tile Size (px)</label>
                    <input
                      type="number"
                      value={settings.yoloTileSize}
                      onChange={e => handleChange('yoloTileSize', parseInt(e.target.value) || 1024)}
                      min={256}
                      max={2048}
                      step={64}
                    />
                    <span className="hint">Match training tile size</span>
                  </div>
                  <div className="settings-row">
                    <label>Tile Overlap (px)</label>
                    <input
                      type="number"
                      value={settings.yoloTileOverlap}
                      onChange={e => handleChange('yoloTileOverlap', parseInt(e.target.value) || 128)}
                      min={0}
                      max={512}
                      step={16}
                    />
                    <span className="hint">Overlap between tiles</span>
                  </div>
                  <div className="settings-row">
                    <label>NMS Threshold</label>
                    <input
                      type="range"
                      min="0.1"
                      max="0.9"
                      step="0.05"
                      value={settings.yoloNmsThreshold}
                      onChange={e => handleChange('yoloNmsThreshold', parseFloat(e.target.value))}
                    />
                    <span className="value-display">{(settings.yoloNmsThreshold * 100).toFixed(0)}%</span>
                  </div>
                  <p className="setting-hint">
                    IoU threshold for merging detections at tile boundaries. Lower = more aggressive merging.
                  </p>

                  {/* Auto Scale Toggle */}
                  <div className="settings-row checkbox">
                    <label>
                      <input
                        type="checkbox"
                        checked={settings.yoloAutoScaleMode === 'auto'}
                        onChange={e => handleChange('yoloAutoScaleMode', e.target.checked ? 'auto' : 'none')}
                      />
                      Auto-scale (Recommended)
                    </label>
                  </div>
                  <p className="setting-hint">
                    {settings.yoloAutoScaleMode === 'auto'
                      ? 'Automatically scales based on your annotations (if 3+ exist) or image size. Uses existing follicle sizes as reference.'
                      : 'Manually set the scale factor below.'}
                  </p>

                  {/* Manual Scale Factor (only shown when auto is off) */}
                  {settings.yoloAutoScaleMode === 'none' && (
                    <div className="settings-row">
                      <label>Scale Factor</label>
                      <input
                        type="range"
                        min="1.0"
                        max="3.0"
                        step="0.1"
                        value={settings.yoloScaleFactor}
                        onChange={e => handleChange('yoloScaleFactor', parseFloat(e.target.value))}
                      />
                      <span className="value-display">{settings.yoloScaleFactor.toFixed(1)}x</span>
                    </div>
                  )}

                  {/* Reference sizes (shown when auto is on) */}
                  {settings.yoloAutoScaleMode === 'auto' && (
                    <>
                      <div className="settings-row">
                        <label>Training Annotation Size</label>
                        <input
                          type="number"
                          value={settings.yoloTrainingAnnotationSize}
                          onChange={e => handleChange('yoloTrainingAnnotationSize', parseInt(e.target.value) || 57)}
                          min={10}
                          max={500}
                          step={1}
                        />
                        <span className="hint">pixels (for annotation mode)</span>
                      </div>
                      <div className="settings-row">
                        <label>Training Image Size</label>
                        <input
                          type="number"
                          value={settings.yoloTrainingImageSize}
                          onChange={e => handleChange('yoloTrainingImageSize', parseInt(e.target.value) || 12240)}
                          min={1000}
                          max={20000}
                          step={100}
                        />
                        <span className="hint">pixels (fallback if no annotations)</span>
                      </div>
                    </>
                  )}
                </>
              )}

                    {/* Service status - hide during export since server is busy */}
                    {!yoloDetectionAvailable && !detectionExport.isExporting && !keypointExport.isExporting && (
                      <div className="yolo-warning">
                        <AlertCircle size={14} />
                        <span>YOLO service not available. Make sure YOLO dependencies are installed.</span>
                      </div>
                    )}

                    {detectionModels.length === 0 && yoloDetectionAvailable && (
                      <div className="yolo-info">
                        <span>No custom models trained yet. Using pre-trained model for detection.</span>
                      </div>
                    )}
                  </div>
                )}

                {/* Blob Detection Settings (shown when Blob method selected) */}
                {settings.detectionMethod === 'blob' && (
                  <div className="method-settings blob-settings">
                    {/* Basic Size Parameters */}
                    <div className="blob-subsection">
                      <h4>Size Range</h4>
                      <div className="settings-row">
                        <label>Min Width</label>
                        <input
                          type="number"
                          value={settings.minWidth}
                          onChange={e => handleChange('minWidth', parseInt(e.target.value) || 0)}
                          min={1}
                          max={1000}
                        />
                      </div>
                      <div className="settings-row">
                        <label>Max Width</label>
                        <input
                          type="number"
                          value={settings.maxWidth}
                          onChange={e => handleChange('maxWidth', parseInt(e.target.value) || 0)}
                          min={1}
                          max={5000}
                        />
                      </div>
                      <div className="settings-row">
                        <label>Min Height</label>
                        <input
                          type="number"
                          value={settings.minHeight}
                          onChange={e => handleChange('minHeight', parseInt(e.target.value) || 0)}
                          min={1}
                          max={1000}
                        />
                      </div>
                      <div className="settings-row">
                        <label>Max Height</label>
                        <input
                          type="number"
                          value={settings.maxHeight}
                          onChange={e => handleChange('maxHeight', parseInt(e.target.value) || 0)}
                          min={1}
                          max={5000}
                        />
                      </div>
                      <div className="settings-row checkbox">
                        <label>
                          <input
                            type="checkbox"
                            checked={settings.darkBlobs}
                            onChange={e => handleChange('darkBlobs', e.target.checked)}
                          />
                          Dark Blobs (detect dark regions on light background)
                        </label>
                      </div>
                    </div>

                    {/* Gaussian Blur */}
                    <div className="blob-subsection">
                      <h4>
                        <label className="section-toggle">
                          <input
                            type="checkbox"
                            checked={settings.useGaussianBlur}
                            onChange={e => handleChange('useGaussianBlur', e.target.checked)}
                          />
                          Gaussian Blur
                        </label>
                      </h4>
                      <p className="section-description">
                        Smooths the image to reduce noise before detection. Helps avoid false positives from image artifacts.
                      </p>
                      {settings.useGaussianBlur && (
                        <div className="settings-row">
                          <label>Kernel Size</label>
                          <select
                            value={settings.gaussianKernelSize}
                            onChange={e => handleChange('gaussianKernelSize', parseInt(e.target.value))}
                          >
                            <option value={3}>3x3 (light blur)</option>
                            <option value={5}>5x5 (recommended)</option>
                            <option value={7}>7x7 (strong blur)</option>
                          </select>
                        </div>
                      )}
                    </div>

                    {/* Morphological Opening */}
                    <div className="blob-subsection">
                      <h4>
                        <label className="section-toggle">
                          <input
                            type="checkbox"
                            checked={settings.useMorphOpen}
                            onChange={e => handleChange('useMorphOpen', e.target.checked)}
                          />
                          Morphological Opening
                        </label>
                      </h4>
                      <p className="section-description">
                        Separates touching objects by eroding then dilating. Essential for splitting merged detections.
                      </p>
                      {settings.useMorphOpen && (
                        <div className="settings-row">
                          <label>Kernel Size</label>
                          <select
                            value={settings.morphKernelSize}
                            onChange={e => handleChange('morphKernelSize', parseInt(e.target.value))}
                          >
                            <option value={3}>3x3 (recommended)</option>
                            <option value={5}>5x5 (stronger separation)</option>
                            <option value={7}>7x7 (aggressive)</option>
                          </select>
                        </div>
                      )}
                    </div>

                    {/* Circularity Filter */}
                    <div className="blob-subsection">
                      <h4>
                        <label className="section-toggle">
                          <input
                            type="checkbox"
                            checked={settings.useCircularityFilter}
                            onChange={e => handleChange('useCircularityFilter', e.target.checked)}
                          />
                          Circularity Filter
                        </label>
                      </h4>
                      <p className="section-description">
                        Rejects elongated or irregular shapes. Only keeps detections that are roughly circular.
                        <strong> Disable for hair follicles</strong> (they are elongated).
                      </p>
                      {settings.useCircularityFilter && (
                        <div className="settings-row">
                          <label>Min Circularity</label>
                          <input
                            type="number"
                            value={settings.minCircularity}
                            onChange={e => handleChange('minCircularity', parseFloat(e.target.value) || 0)}
                            min={0}
                            max={1}
                            step={0.05}
                          />
                          <span className="hint">0-1, 1.0 = perfect circle</span>
                        </div>
                      )}
                    </div>

                    {/* Inertia Filter */}
                    <div className="blob-subsection">
                      <h4>
                        <label className="section-toggle">
                          <input
                            type="checkbox"
                            checked={settings.useInertiaFilter}
                            onChange={e => handleChange('useInertiaFilter', e.target.checked)}
                          />
                          Inertia Filter
                        </label>
                      </h4>
                      <p className="section-description">
                        Controls allowed elongation of detected shapes. Low min ratio allows elongated shapes like
                        <strong> hair follicles</strong>. High ratio requires more circular shapes.
                      </p>
                      {settings.useInertiaFilter && (
                        <>
                          <div className="settings-row">
                            <label>Min Inertia Ratio</label>
                            <input
                              type="number"
                              value={settings.minInertiaRatio}
                              onChange={e => handleChange('minInertiaRatio', parseFloat(e.target.value) || 0)}
                              min={0}
                              max={1}
                              step={0.01}
                            />
                            <span className="hint">0-1, 0.01 = very elongated OK</span>
                          </div>
                          <div className="settings-row">
                            <label>Max Inertia Ratio</label>
                            <input
                              type="number"
                              value={settings.maxInertiaRatio}
                              onChange={e => handleChange('maxInertiaRatio', parseFloat(e.target.value) || 1)}
                              min={0}
                              max={1}
                              step={0.01}
                            />
                            <span className="hint">0-1, 1.0 = perfect circle</span>
                          </div>
                        </>
                      )}
                    </div>

                    {/* Convexity Filter */}
                    <div className="blob-subsection">
                      <h4>
                        <label className="section-toggle">
                          <input
                            type="checkbox"
                            checked={settings.useConvexityFilter}
                            onChange={e => handleChange('useConvexityFilter', e.target.checked)}
                          />
                          Convexity Filter
                        </label>
                      </h4>
                      <p className="section-description">
                        Filters by how convex (non-concave) the detected shape is. High values reject shapes with indentations.
                      </p>
                      {settings.useConvexityFilter && (
                        <div className="settings-row">
                          <label>Min Convexity</label>
                          <input
                            type="number"
                            value={settings.minConvexity}
                            onChange={e => handleChange('minConvexity', parseFloat(e.target.value) || 0)}
                            min={0}
                            max={1}
                            step={0.05}
                          />
                          <span className="hint">0-1, 1.0 = perfectly convex</span>
                        </div>
                      )}
                    </div>

                    {/* CLAHE Preprocessing */}
                    <div className="blob-subsection">
                      <h4>
                        <label className="section-toggle">
                          <input
                            type="checkbox"
                            checked={settings.useCLAHE}
                            onChange={e => handleChange('useCLAHE', e.target.checked)}
                          />
                          CLAHE Preprocessing
                        </label>
                      </h4>
                      <p className="section-description">
                        Contrast Limited Adaptive Histogram Equalization improves detection in
                        images with uneven lighting or low contrast.
                      </p>
                      {settings.useCLAHE && (
                        <>
                          <div className="settings-row">
                            <label>Clip Limit</label>
                            <input
                              type="number"
                              value={settings.claheClipLimit}
                              onChange={e => handleChange('claheClipLimit', parseFloat(e.target.value) || 1)}
                              min={1}
                              max={10}
                              step={0.5}
                            />
                            <span className="hint">1-10, higher = more contrast</span>
                          </div>
                          <div className="settings-row">
                            <label>Tile Size</label>
                            <input
                              type="number"
                              value={settings.claheTileSize}
                              onChange={e => handleChange('claheTileSize', parseInt(e.target.value) || 4)}
                              min={2}
                              max={16}
                            />
                            <span className="hint">2-16, tiles per dimension</span>
                          </div>
                        </>
                      )}
                    </div>

                    {/* SAHI-style Tiling */}
                    <div className="blob-subsection">
                      <h4>
                        <label className="section-toggle">
                          <input
                            type="checkbox"
                            checked={settings.useSAHI}
                            onChange={e => handleChange('useSAHI', e.target.checked)}
                          />
                          SAHI Tiling
                        </label>
                      </h4>
                      <p className="section-description">
                        Sliced Aided Hyper Inference processes large images in overlapping tiles
                        for better detection of small objects.
                      </p>
                      {settings.useSAHI && (
                        <>
                          <div className="settings-row">
                            <label>Tile Size (px)</label>
                            <input
                              type="number"
                              value={settings.tileSize}
                              onChange={e => handleChange('tileSize', parseInt(e.target.value) || 256)}
                              min={128}
                              max={2048}
                              step={64}
                            />
                            <span className="hint">128-2048px</span>
                          </div>
                          <div className="settings-row">
                            <label>Overlap</label>
                            <input
                              type="number"
                              value={settings.tileOverlap}
                              onChange={e => handleChange('tileOverlap', parseFloat(e.target.value) || 0.1)}
                              min={0}
                              max={0.5}
                              step={0.05}
                            />
                            <span className="hint">0-0.5, fraction of tile size</span>
                          </div>
                        </>
                      )}
                    </div>

                    {/* Soft-NMS */}
                    <div className="blob-subsection">
                      <h4>
                        <label className="section-toggle">
                          <input
                            type="checkbox"
                            checked={settings.useSoftNMS}
                            onChange={e => handleChange('useSoftNMS', e.target.checked)}
                          />
                          Soft-NMS (Non-Maximum Suppression)
                        </label>
                      </h4>
                      <p className="section-description">
                        Reduces overlapping detections using soft suppression with Gaussian decay.
                      </p>
                      {settings.useSoftNMS && (
                        <>
                          <div className="settings-row">
                            <label>Sigma</label>
                            <input
                              type="number"
                              value={settings.softNMSSigma}
                              onChange={e => handleChange('softNMSSigma', parseFloat(e.target.value) || 0.3)}
                              min={0.1}
                              max={2}
                              step={0.1}
                            />
                            <span className="hint">0.1-2, Gaussian decay rate</span>
                          </div>
                          <div className="settings-row">
                            <label>Threshold</label>
                            <input
                              type="number"
                              value={settings.softNMSThreshold}
                              onChange={e => handleChange('softNMSThreshold', parseFloat(e.target.value) || 0.05)}
                              min={0.01}
                              max={0.5}
                              step={0.01}
                            />
                            <span className="hint">Min confidence to keep</span>
                          </div>
                        </>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}
          </section>

          {/* Section 2: Keypoint Prediction */}
          <section className="settings-section keypoint-section">
            <h3
              className="section-header clickable"
              onClick={() => setKeypointCollapsed(!keypointCollapsed)}
            >
              <ChevronDown size={16} className={`chevron ${keypointCollapsed ? 'rotated' : ''}`} />
              <Crosshair size={16} />
              Origin & Direction Prediction
            </h3>

            {!keypointCollapsed && (
              <div className="section-content">
                {/* Show export progress if exporting */}
                {keypointExport.isExporting && (
                  <div className="export-progress-section">
                    <Loader2 size={16} className="spin" />
                    <span>{getExportProgressMessage()}</span>
                  </div>
                )}

                {/* Show warning only if not exporting and no models available */}
                {!keypointModelsAvailable && !keypointExport.isExporting ? (
                  <div className="yolo-warning">
                    <AlertCircle size={14} />
                    <span>No trained models available. Train a model first using the YOLO Training dialog.</span>
                  </div>
                ) : !keypointExport.isExporting && (
                  <>
                    {/* Active Keypoint Model Display */}
                    <div className="current-model-display">
                      <div className="model-label">Active Keypoint Model</div>
                      <div className="model-info-box">
                        {!settings.keypointModelId ? (
                          <div className="model-details">
                            <span className="model-name">{loadedKeypointModel?.name || 'No model selected'}</span>
                            <span className="model-meta">
                              {loadedKeypointModel
                                ? `${loadedKeypointModel.epochsTrained} epochs, ${loadedKeypointModel.imgSize}px`
                                : 'Select a trained model from the library'}
                            </span>
                          </div>
                        ) : (
                          <div className="model-details">
                            <span className="model-name">{settings.keypointModelName || 'Custom Model'}</span>
                            <span className="model-meta">Custom trained model</span>
                          </div>
                        )}
                        <button
                          className="switch-model-btn"
                          onClick={() => onOpenModelLibrary?.('origin')}
                        >
                          Switch Model
                        </button>
                      </div>
                    </div>

                    {/* Inference Backend */}
                    <div className="settings-row">
                      <label>Inference Backend</label>
                      <select
                        value={settings.keypointInferenceBackend}
                        onChange={e => handleChange('keypointInferenceBackend', e.target.value as YoloInferenceBackend)}
                        disabled={!keypointTensorrtStatus?.available && settings.keypointInferenceBackend !== 'tensorrt'}
                      >
                        <option value="pytorch">PyTorch (Default)</option>
                        <option value="tensorrt" disabled={!keypointTensorrtStatus?.available}>
                          TensorRT {keypointTensorrtStatus?.available ? `(v${keypointTensorrtStatus.version})` : '(Not installed)'}
                        </option>
                      </select>
                    </div>

                    {/* TensorRT engine status */}
                    {settings.keypointInferenceBackend === 'tensorrt' && keypointTensorrtStatus?.available && loadedKeypointModel && (
                      <>
                        {keypointEngineAvailable === false && !keypointExport.isExporting && !keypointExport.error && !keypointExport.completed && (
                          <div className="tensorrt-export-section">
                            <div className="tensorrt-export-warning">
                              <AlertCircle size={14} />
                              <span>No TensorRT engine found for "{loadedKeypointModel.name}"</span>
                            </div>
                            <button className="tensorrt-export-btn" onClick={handleExportKeypointEngine}>
                              <Zap size={14} />
                              Export to TensorRT
                            </button>
                            <p className="setting-hint">
                              Export creates a GPU-optimized .engine file for faster prediction.
                            </p>
                          </div>
                        )}
                        {keypointExport.isExporting && (
                          <div className="tensorrt-export-section exporting">
                            <div className="export-progress-container">
                              <div className="export-progress-header">
                                <Loader2 size={14} className="spin" />
                                <span>{keypointExport.progress || 'Exporting to TensorRT...'}</span>
                              </div>
                              <div className="export-progress-bar">
                                <div className="export-progress-bar-indeterminate" />
                              </div>
                            </div>
                          </div>
                        )}
                        {keypointExport.error && !keypointExport.isExporting && (
                          <div className="tensorrt-export-section error">
                            <AlertCircle size={14} />
                            <span>{keypointExport.error}</span>
                            <button className="tensorrt-retry-btn" onClick={handleExportKeypointEngine}>
                              Retry
                            </button>
                          </div>
                        )}
                        {(keypointEngineAvailable === true || keypointExport.completed) && (
                          <p className="setting-hint tensorrt-ready">
                            <Zap size={12} />
                            TensorRT engine ready. Using GPU-optimized inference.
                          </p>
                        )}
                      </>
                    )}

                    {/* Keypoint Confidence Threshold */}
                    <div className="settings-row" style={{ marginTop: '12px' }}>
                      <label>Confidence Threshold</label>
                      <input
                        type="range"
                        min="0.1"
                        max="0.9"
                        step="0.05"
                        value={settings.keypointConfidenceThreshold}
                        onChange={e => handleChange('keypointConfidenceThreshold', parseFloat(e.target.value))}
                      />
                      <span className="value-display">{(settings.keypointConfidenceThreshold * 100).toFixed(0)}%</span>
                    </div>
                    <p className="setting-hint">
                      Minimum confidence for origin predictions. Higher = fewer but more accurate predictions.
                    </p>

                    {/* Auto-Predict Toggle */}
                    <div className="settings-row checkbox" style={{ marginTop: '12px' }}>
                      <label>
                        <input
                          type="checkbox"
                          checked={settings.useKeypointPrediction}
                          onChange={e => handleChange('useKeypointPrediction', e.target.checked)}
                        />
                        Auto-predict origins during detection
                      </label>
                    </div>
                    <p className="setting-hint">
                      When enabled, origin points will be automatically predicted for all detected follicles.
                    </p>
                  </>
                )}
              </div>
            )}
          </section>
        </div>

        <div className="dialog-footer">
          <button className="button-secondary" onClick={handleReset}>
            Reset to Defaults
          </button>
          <span className="auto-save-hint">Changes are saved automatically</span>
        </div>
      </div>
    </div>
  );
}

/**
 * Convert DetectionSettings to BlobDetectionOptions
 */
export function settingsToOptions(settings: DetectionSettings): Partial<BlobDetectionOptions> {
  return {
    minWidth: settings.minWidth,
    maxWidth: settings.maxWidth,
    minHeight: settings.minHeight,
    maxHeight: settings.maxHeight,
    darkBlobs: settings.darkBlobs,
    // Gaussian blur
    useGaussianBlur: settings.useGaussianBlur,
    gaussianKernelSize: settings.gaussianKernelSize,
    // Morphological opening
    useMorphOpen: settings.useMorphOpen,
    morphKernelSize: settings.morphKernelSize,
    // Circularity filter
    filterByCircularity: settings.useCircularityFilter,
    minCircularity: settings.minCircularity,
    // Inertia filter (for elongated shapes like hair follicles)
    filterByInertia: settings.useInertiaFilter,
    minInertiaRatio: settings.minInertiaRatio,
    maxInertiaRatio: settings.maxInertiaRatio,
    // Convexity filter
    filterByConvexity: settings.useConvexityFilter,
    minConvexity: settings.minConvexity,
    // CLAHE
    useCLAHE: settings.useCLAHE,
    claheClipLimit: settings.claheClipLimit,
    claheTileSize: settings.claheTileSize,
    // SAHI tiling
    tileSize: settings.useSAHI ? settings.tileSize : 0,
    tileOverlap: settings.tileOverlap,
    // Soft-NMS
    useSoftNMS: settings.useSoftNMS,
    softNMSSigma: settings.softNMSSigma,
    softNMSThreshold: settings.softNMSThreshold,
    useGPU: true,
    workerCount: navigator.hardwareConcurrency || 4,
  };
}
