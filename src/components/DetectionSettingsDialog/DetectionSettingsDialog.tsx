import { useState, useEffect, useRef, useCallback } from 'react';
import { X, Cpu, Zap, Download, Loader2, AlertCircle, Crosshair, Box, Brain } from 'lucide-react';
import type { BlobDetectionOptions, GPUInfo, GPUHardwareInfo, ModelInfo, DetectionModelInfo, DetectionMethod } from '../../types';
import { blobService } from '../../services/blobService';
import { yoloKeypointService } from '../../services/yoloKeypointService';
import { yoloDetectionService } from '../../services/yoloDetectionService';
import './DetectionSettingsDialog.css';

export interface DetectionSettings {
  // Detection method: 'blob' (SimpleBlobDetector) or 'yolo' (YOLO AI)
  detectionMethod: DetectionMethod;

  // YOLO Detection settings
  yoloModelId: string | null;  // Selected model ID (null = pre-trained/default)
  yoloConfidenceThreshold: number;  // Confidence threshold for YOLO (0-1)
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
}

export const DEFAULT_DETECTION_SETTINGS: DetectionSettings = {
  // Detection method - default to blob detection
  detectionMethod: 'blob',
  // YOLO Detection settings
  yoloModelId: null,
  yoloConfidenceThreshold: 0.5,
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
};

// Install state that can be managed by parent for persistence
export interface GPUInstallState {
  isInstalling: boolean;
  progress: string;
  error: string | null;
}

interface DetectionSettingsDialogProps {
  settings: DetectionSettings;
  onSave: (settings: DetectionSettings) => void;
  onCancel: () => void;
  blobServerConnected?: boolean;
  onServerRestarted?: () => void;
  // Optional install state from parent for persistence across dialog close/reopen
  installState?: GPUInstallState;
  onInstallStateChange?: (state: GPUInstallState) => void;
}

export function DetectionSettingsDialog({
  settings,
  onSave,
  onCancel,
  blobServerConnected = false,
  onServerRestarted,
  installState,
  onInstallStateChange,
}: DetectionSettingsDialogProps) {
  const [localSettings, setLocalSettings] = useState<DetectionSettings>({ ...settings });
  const [gpuInfo, setGpuInfo] = useState<GPUInfo | null>(null);
  const [gpuHardware, setGpuHardware] = useState<GPUHardwareInfo | null>(null);
  const [loadedKeypointModel, setLoadedKeypointModel] = useState<ModelInfo | null>(null);
  const [keypointModelsAvailable, setKeypointModelsAvailable] = useState(false);

  // YOLO Detection models state
  const [detectionModels, setDetectionModels] = useState<DetectionModelInfo[]>([]);
  const [yoloDetectionAvailable, setYoloDetectionAvailable] = useState(false);
  const [loadingDetectionModels, setLoadingDetectionModels] = useState(false);

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

  // Check for available YOLO detection models
  useEffect(() => {
    const checkDetectionModels = async () => {
      setLoadingDetectionModels(true);
      try {
        const status = await yoloDetectionService.getStatus();
        setYoloDetectionAvailable(status.available);
        if (status.available) {
          const models = await yoloDetectionService.listModels();
          setDetectionModels(models);
        }
      } catch (error) {
        console.error('Failed to check detection models:', error);
      } finally {
        setLoadingDetectionModels(false);
      }
    };
    checkDetectionModels();
  }, []);

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

  const handleChange = <K extends keyof DetectionSettings>(
    key: K,
    value: DetectionSettings[K]
  ) => {
    setLocalSettings(prev => ({ ...prev, [key]: value }));
  };

  const handleReset = () => {
    setLocalSettings({ ...DEFAULT_DETECTION_SETTINGS });
  };

  const handleSave = () => {
    onSave(localSettings);
  };

  return (
    <div className="detection-settings-overlay" onClick={onCancel}>
      <div
        ref={dialogRef}
        className="detection-settings-dialog"
        onClick={e => e.stopPropagation()}
        style={position ? { transform: `translate(${position.x}px, ${position.y}px)` } : undefined}
      >
        <div className="dialog-header" onMouseDown={handleMouseDown}>
          <h2>Detection Settings</h2>
          <button className="close-button" onClick={onCancel}>
            <X size={18} />
          </button>
        </div>

        <div className="dialog-content">
          {/* Detection Method Toggle */}
          <section className="settings-section detection-method-section">
            <h3>Detection Method</h3>
            <div className="detection-method-toggle">
              <button
                className={`method-btn ${localSettings.detectionMethod === 'blob' ? 'active' : ''}`}
                onClick={() => handleChange('detectionMethod', 'blob')}
              >
                <Box size={16} />
                SimpleBlobDetector
              </button>
              <button
                className={`method-btn ${localSettings.detectionMethod === 'yolo' ? 'active' : ''}`}
                onClick={() => handleChange('detectionMethod', 'yolo')}
              >
                <Brain size={16} />
                YOLO (AI)
              </button>
            </div>
            <p className="method-description">
              {localSettings.detectionMethod === 'blob'
                ? 'Uses OpenCV SimpleBlobDetector for classical computer vision based detection. Fast and reliable for uniform images.'
                : 'Uses YOLO neural network for AI-powered detection. Better for complex images and varied lighting.'}
            </p>
          </section>

          {/* YOLO Detection Settings (shown when YOLO method selected) */}
          {localSettings.detectionMethod === 'yolo' && (
            <section className="settings-section yolo-detection-section">
              <h3>YOLO Detection Settings</h3>

              {/* Model Selection */}
              <div className="settings-row">
                <label>Model</label>
                <select
                  value={localSettings.yoloModelId || 'pretrained'}
                  onChange={e => handleChange('yoloModelId', e.target.value === 'pretrained' ? null : e.target.value)}
                  disabled={loadingDetectionModels}
                >
                  <option value="pretrained">Pre-trained (yolo11n.pt)</option>
                  {detectionModels.map(model => (
                    <option key={model.id} value={model.id}>
                      {model.name} ({model.epochsTrained} epochs)
                    </option>
                  ))}
                </select>
              </div>

              {/* Confidence Threshold */}
              <div className="settings-row">
                <label>Confidence Threshold</label>
                <input
                  type="range"
                  min="0.1"
                  max="0.9"
                  step="0.05"
                  value={localSettings.yoloConfidenceThreshold}
                  onChange={e => handleChange('yoloConfidenceThreshold', parseFloat(e.target.value))}
                />
                <span className="value-display">{(localSettings.yoloConfidenceThreshold * 100).toFixed(0)}%</span>
              </div>
              <p className="setting-hint">
                Higher threshold = fewer but more confident detections. Lower = more detections but possible false positives.
              </p>

              {/* Tiled Inference Toggle */}
              <div className="settings-row checkbox">
                <label>
                  <input
                    type="checkbox"
                    checked={localSettings.yoloUseTiledInference}
                    onChange={e => handleChange('yoloUseTiledInference', e.target.checked)}
                  />
                  Use Tiled Inference (for large images)
                </label>
              </div>
              <p className="setting-hint">
                Splits large images into overlapping tiles matching training size. Essential when model was trained on tiles.
              </p>

              {/* Tiled Inference Settings */}
              {localSettings.yoloUseTiledInference && (
                <>
                  <div className="settings-row">
                    <label>Tile Size (px)</label>
                    <input
                      type="number"
                      value={localSettings.yoloTileSize}
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
                      value={localSettings.yoloTileOverlap}
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
                      value={localSettings.yoloNmsThreshold}
                      onChange={e => handleChange('yoloNmsThreshold', parseFloat(e.target.value))}
                    />
                    <span className="value-display">{(localSettings.yoloNmsThreshold * 100).toFixed(0)}%</span>
                  </div>
                  <p className="setting-hint">
                    IoU threshold for merging detections at tile boundaries. Lower = more aggressive merging.
                  </p>

                  {/* Auto Scale Toggle */}
                  <div className="settings-row checkbox">
                    <label>
                      <input
                        type="checkbox"
                        checked={localSettings.yoloAutoScaleMode === 'auto'}
                        onChange={e => handleChange('yoloAutoScaleMode', e.target.checked ? 'auto' : 'none')}
                      />
                      Auto-scale (Recommended)
                    </label>
                  </div>
                  <p className="setting-hint">
                    {localSettings.yoloAutoScaleMode === 'auto'
                      ? 'Automatically scales based on your annotations (if 3+ exist) or image size. Uses existing follicle sizes as reference.'
                      : 'Manually set the scale factor below.'}
                  </p>

                  {/* Manual Scale Factor (only shown when auto is off) */}
                  {localSettings.yoloAutoScaleMode === 'none' && (
                    <div className="settings-row">
                      <label>Scale Factor</label>
                      <input
                        type="range"
                        min="1.0"
                        max="3.0"
                        step="0.1"
                        value={localSettings.yoloScaleFactor}
                        onChange={e => handleChange('yoloScaleFactor', parseFloat(e.target.value))}
                      />
                      <span className="value-display">{localSettings.yoloScaleFactor.toFixed(1)}x</span>
                    </div>
                  )}

                  {/* Reference sizes (shown when auto is on) */}
                  {localSettings.yoloAutoScaleMode === 'auto' && (
                    <>
                      <div className="settings-row">
                        <label>Training Annotation Size</label>
                        <input
                          type="number"
                          value={localSettings.yoloTrainingAnnotationSize}
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
                          value={localSettings.yoloTrainingImageSize}
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

              {/* Service status */}
              {!yoloDetectionAvailable && (
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
            </section>
          )}

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
                <span className="gpu-status-label">Installing GPU Packages</span>
                <span className="gpu-status-progress">{installProgress || 'Starting...'}</span>
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
            <section className="settings-section backend-selection">
              <h3>Processing Backend</h3>
              <div className="backend-toggle">
                <button
                  className={`backend-btn ${!localSettings.forceCPU ? 'active' : ''}`}
                  onClick={() => handleChange('forceCPU', false)}
                >
                  <Zap size={14} />
                  GPU
                </button>
                <button
                  className={`backend-btn ${localSettings.forceCPU ? 'active' : ''}`}
                  onClick={() => handleChange('forceCPU', true)}
                >
                  <Cpu size={14} />
                  CPU
                </button>
              </div>
              <p className="backend-hint">
                {localSettings.forceCPU
                  ? 'Using CPU for detection (slower but more compatible)'
                  : 'Using GPU for detection (faster processing)'}
              </p>
            </section>
          )}

          {/* Blob Detection Settings (shown when Blob method selected) */}
          {localSettings.detectionMethod === 'blob' && (
            <>
          {/* Basic Size Parameters */}
          <section className="settings-section">
            <h3>Size Range</h3>
            <div className="settings-row">
              <label>Min Width</label>
              <input
                type="number"
                value={localSettings.minWidth}
                onChange={e => handleChange('minWidth', parseInt(e.target.value) || 0)}
                min={1}
                max={1000}
              />
            </div>
            <div className="settings-row">
              <label>Max Width</label>
              <input
                type="number"
                value={localSettings.maxWidth}
                onChange={e => handleChange('maxWidth', parseInt(e.target.value) || 0)}
                min={1}
                max={5000}
              />
            </div>
            <div className="settings-row">
              <label>Min Height</label>
              <input
                type="number"
                value={localSettings.minHeight}
                onChange={e => handleChange('minHeight', parseInt(e.target.value) || 0)}
                min={1}
                max={1000}
              />
            </div>
            <div className="settings-row">
              <label>Max Height</label>
              <input
                type="number"
                value={localSettings.maxHeight}
                onChange={e => handleChange('maxHeight', parseInt(e.target.value) || 0)}
                min={1}
                max={5000}
              />
            </div>
            <div className="settings-row checkbox">
              <label>
                <input
                  type="checkbox"
                  checked={localSettings.darkBlobs}
                  onChange={e => handleChange('darkBlobs', e.target.checked)}
                />
                Dark Blobs (detect dark regions on light background)
              </label>
            </div>
          </section>

          {/* Gaussian Blur */}
          <section className="settings-section">
            <h3>
              <label className="section-toggle">
                <input
                  type="checkbox"
                  checked={localSettings.useGaussianBlur}
                  onChange={e => handleChange('useGaussianBlur', e.target.checked)}
                />
                Gaussian Blur
              </label>
            </h3>
            <p className="section-description">
              Smooths the image to reduce noise before detection. Helps avoid false positives from image artifacts.
            </p>
            {localSettings.useGaussianBlur && (
              <div className="settings-row">
                <label>Kernel Size</label>
                <select
                  value={localSettings.gaussianKernelSize}
                  onChange={e => handleChange('gaussianKernelSize', parseInt(e.target.value))}
                >
                  <option value={3}>3x3 (light blur)</option>
                  <option value={5}>5x5 (recommended)</option>
                  <option value={7}>7x7 (strong blur)</option>
                </select>
              </div>
            )}
          </section>

          {/* Morphological Opening */}
          <section className="settings-section">
            <h3>
              <label className="section-toggle">
                <input
                  type="checkbox"
                  checked={localSettings.useMorphOpen}
                  onChange={e => handleChange('useMorphOpen', e.target.checked)}
                />
                Morphological Opening
              </label>
            </h3>
            <p className="section-description">
              Separates touching objects by eroding then dilating. Essential for splitting merged detections.
            </p>
            {localSettings.useMorphOpen && (
              <div className="settings-row">
                <label>Kernel Size</label>
                <select
                  value={localSettings.morphKernelSize}
                  onChange={e => handleChange('morphKernelSize', parseInt(e.target.value))}
                >
                  <option value={3}>3x3 (recommended)</option>
                  <option value={5}>5x5 (stronger separation)</option>
                  <option value={7}>7x7 (aggressive)</option>
                </select>
              </div>
            )}
          </section>

          {/* Circularity Filter */}
          <section className="settings-section">
            <h3>
              <label className="section-toggle">
                <input
                  type="checkbox"
                  checked={localSettings.useCircularityFilter}
                  onChange={e => handleChange('useCircularityFilter', e.target.checked)}
                />
                Circularity Filter
              </label>
            </h3>
            <p className="section-description">
              Rejects elongated or irregular shapes. Only keeps detections that are roughly circular.
              <strong> Disable for hair follicles</strong> (they are elongated).
            </p>
            {localSettings.useCircularityFilter && (
              <div className="settings-row">
                <label>Min Circularity</label>
                <input
                  type="number"
                  value={localSettings.minCircularity}
                  onChange={e => handleChange('minCircularity', parseFloat(e.target.value) || 0)}
                  min={0}
                  max={1}
                  step={0.05}
                />
                <span className="hint">0-1, 1.0 = perfect circle</span>
              </div>
            )}
          </section>

          {/* Inertia Filter */}
          <section className="settings-section">
            <h3>
              <label className="section-toggle">
                <input
                  type="checkbox"
                  checked={localSettings.useInertiaFilter}
                  onChange={e => handleChange('useInertiaFilter', e.target.checked)}
                />
                Inertia Filter
              </label>
            </h3>
            <p className="section-description">
              Controls allowed elongation of detected shapes. Low min ratio allows elongated shapes like
              <strong> hair follicles</strong>. High ratio requires more circular shapes.
            </p>
            {localSettings.useInertiaFilter && (
              <>
                <div className="settings-row">
                  <label>Min Inertia Ratio</label>
                  <input
                    type="number"
                    value={localSettings.minInertiaRatio}
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
                    value={localSettings.maxInertiaRatio}
                    onChange={e => handleChange('maxInertiaRatio', parseFloat(e.target.value) || 1)}
                    min={0}
                    max={1}
                    step={0.01}
                  />
                  <span className="hint">0-1, 1.0 = perfect circle</span>
                </div>
              </>
            )}
          </section>

          {/* Convexity Filter */}
          <section className="settings-section">
            <h3>
              <label className="section-toggle">
                <input
                  type="checkbox"
                  checked={localSettings.useConvexityFilter}
                  onChange={e => handleChange('useConvexityFilter', e.target.checked)}
                />
                Convexity Filter
              </label>
            </h3>
            <p className="section-description">
              Filters by how convex (non-concave) the detected shape is. High values reject shapes with indentations.
            </p>
            {localSettings.useConvexityFilter && (
              <div className="settings-row">
                <label>Min Convexity</label>
                <input
                  type="number"
                  value={localSettings.minConvexity}
                  onChange={e => handleChange('minConvexity', parseFloat(e.target.value) || 0)}
                  min={0}
                  max={1}
                  step={0.05}
                />
                <span className="hint">0-1, 1.0 = perfectly convex</span>
              </div>
            )}
          </section>

          {/* CLAHE Preprocessing */}
          <section className="settings-section">
            <h3>
              <label className="section-toggle">
                <input
                  type="checkbox"
                  checked={localSettings.useCLAHE}
                  onChange={e => handleChange('useCLAHE', e.target.checked)}
                />
                CLAHE Preprocessing
              </label>
            </h3>
            <p className="section-description">
              Contrast Limited Adaptive Histogram Equalization improves detection in
              images with uneven lighting or low contrast.
            </p>
            {localSettings.useCLAHE && (
              <>
                <div className="settings-row">
                  <label>Clip Limit</label>
                  <input
                    type="number"
                    value={localSettings.claheClipLimit}
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
                    value={localSettings.claheTileSize}
                    onChange={e => handleChange('claheTileSize', parseInt(e.target.value) || 4)}
                    min={2}
                    max={16}
                  />
                  <span className="hint">2-16, tiles per dimension</span>
                </div>
              </>
            )}
          </section>

          {/* SAHI-style Tiling */}
          <section className="settings-section">
            <h3>
              <label className="section-toggle">
                <input
                  type="checkbox"
                  checked={localSettings.useSAHI}
                  onChange={e => handleChange('useSAHI', e.target.checked)}
                />
                SAHI Tiling
              </label>
            </h3>
            <p className="section-description">
              Sliced Aided Hyper Inference processes large images in overlapping tiles
              for better detection of small objects.
            </p>
            {localSettings.useSAHI && (
              <>
                <div className="settings-row">
                  <label>Tile Size (px)</label>
                  <input
                    type="number"
                    value={localSettings.tileSize}
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
                    value={localSettings.tileOverlap}
                    onChange={e => handleChange('tileOverlap', parseFloat(e.target.value) || 0.1)}
                    min={0}
                    max={0.5}
                    step={0.05}
                  />
                  <span className="hint">0-0.5, fraction of tile size</span>
                </div>
              </>
            )}
          </section>

          {/* Soft-NMS */}
          <section className="settings-section">
            <h3>
              <label className="section-toggle">
                <input
                  type="checkbox"
                  checked={localSettings.useSoftNMS}
                  onChange={e => handleChange('useSoftNMS', e.target.checked)}
                />
                Soft-NMS (Non-Maximum Suppression)
              </label>
            </h3>
            <p className="section-description">
              Reduces overlapping detections using soft suppression with Gaussian decay.
            </p>
            {localSettings.useSoftNMS && (
              <>
                <div className="settings-row">
                  <label>Sigma</label>
                  <input
                    type="number"
                    value={localSettings.softNMSSigma}
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
                    value={localSettings.softNMSThreshold}
                    onChange={e => handleChange('softNMSThreshold', parseFloat(e.target.value) || 0.05)}
                    min={0.01}
                    max={0.5}
                    step={0.01}
                  />
                  <span className="hint">Min confidence to keep</span>
                </div>
              </>
            )}
          </section>
            </>
          )}

          {/* YOLO Keypoint Prediction */}
          <section className="settings-section">
            <h3>
              <label className="section-toggle">
                <input
                  type="checkbox"
                  checked={localSettings.useKeypointPrediction}
                  onChange={e => handleChange('useKeypointPrediction', e.target.checked)}
                  disabled={!keypointModelsAvailable}
                />
                <Crosshair size={16} style={{ marginRight: '6px' }} />
                Auto-Predict Origins
              </label>
            </h3>
            <p className="section-description">
              Automatically predict follicle origin points and growth direction using YOLO keypoint model.
              {!keypointModelsAvailable && (
                <span className="warning-text"> No trained models available. Train a model first using the YOLO Training dialog.</span>
              )}
            </p>
            {localSettings.useKeypointPrediction && keypointModelsAvailable && loadedKeypointModel && (
              <div className="keypoint-model-info">
                <span className="model-name">Using: {loadedKeypointModel.name}</span>
                <span className="model-meta">
                  {loadedKeypointModel.epochsTrained} epochs, {loadedKeypointModel.imgSize}px
                </span>
              </div>
            )}
          </section>
        </div>

        <div className="dialog-footer">
          <button className="button-secondary" onClick={handleReset}>
            Reset to Defaults
          </button>
          <div className="footer-actions">
            <button className="button-secondary" onClick={onCancel}>
              Cancel
            </button>
            <button className="button-primary" onClick={handleSave}>
              Save Settings
            </button>
          </div>
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
