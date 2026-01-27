import { useState, useEffect } from 'react';
import { X, Cpu, Zap, Download, Loader2, AlertCircle } from 'lucide-react';
import type { BlobDetectionOptions, GPUInfo, GPUHardwareInfo } from '../../types';
import { blobService } from '../../services/blobService';
import './DetectionSettingsDialog.css';

export interface DetectionSettings {
  // Basic size parameters
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
}

export const DEFAULT_DETECTION_SETTINGS: DetectionSettings = {
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
  // Circularity filter - rejects elongated shapes
  useCircularityFilter: true,
  minCircularity: 0.2,
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
      <div className="detection-settings-dialog" onClick={e => e.stopPropagation()}>
        <div className="dialog-header">
          <h2>Detection Settings</h2>
          <button className="close-button" onClick={onCancel}>
            <X size={18} />
          </button>
        </div>

        <div className="dialog-content">
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
    // Circularity filter (0 disables it)
    minCircularity: settings.useCircularityFilter ? settings.minCircularity : 0,
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
