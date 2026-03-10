import { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import { Play, Square, Loader2, AlertCircle, CheckCircle, FolderOpen, Download, FileImage, Cpu, Monitor, Zap } from 'lucide-react';
import { yoloKeypointService } from '../../services/yoloKeypointService';
import { useProjectStore } from '../../store/projectStore';
import { useFollicleStore } from '../../store/follicleStore';
import { exportYOLOKeypointDataset } from '../../utils/yolo-keypoint-export';
import {
  TrainingConfig,
  TrainingProgress,
  DatasetValidation,
  YoloDependenciesInfo,
  DEFAULT_TRAINING_CONFIG,
  RectangleAnnotation,
  GPUHardwareInfo,
} from '../../types';

interface SystemInfo {
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

type TrainingStatus = 'idle' | 'validating' | 'preparing' | 'training' | 'completed' | 'failed';
type DatasetSource = 'project' | 'external';

export function OriginTab() {
  // Project data
  const images = useProjectStore((state) => state.images);
  const follicles = useFollicleStore((state) => state.follicles);

  // Dependencies state
  const [dependencies, setDependencies] = useState<YoloDependenciesInfo | null>(null);
  const [installingDeps, setInstallingDeps] = useState(false);
  const [installProgress, setInstallProgress] = useState<string>('');
  const [installError, setInstallError] = useState<string | null>(null);

  // System info state
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const systemInfoIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // GPU hardware state (for detection and installation)
  const [gpuHardware, setGpuHardware] = useState<GPUHardwareInfo | null>(null);
  const [isInstallingGpu, setIsInstallingGpu] = useState(false);
  const [gpuInstallProgress, setGpuInstallProgress] = useState<string>('');
  const [gpuInstallError, setGpuInstallError] = useState<string | null>(null);

  // Dataset source state
  const [datasetSource, setDatasetSource] = useState<DatasetSource>('project');
  const [datasetPath, setDatasetPath] = useState('');
  const [validation, setValidation] = useState<DatasetValidation | null>(null);
  const [validationStatus, setValidationStatus] = useState<'idle' | 'validating' | 'valid' | 'invalid'>('idle');

  // Training config
  const [config, setConfig] = useState<TrainingConfig>(DEFAULT_TRAINING_CONFIG);
  const [modelName, setModelName] = useState('');

  // Training state
  const [status, setStatus] = useState<TrainingStatus>('idle');
  const [progress, setProgress] = useState<TrainingProgress | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [preparingMessage, setPreparingMessage] = useState<string>('');

  // Service availability
  const [serviceAvailable, setServiceAvailable] = useState<boolean | null>(null);

  // Calculate project stats - follicles with origin defined
  const projectStats = useMemo(() => {
    const rectangleFollicles = follicles.filter(
      (f): f is RectangleAnnotation => f.shape === 'rectangle'
    );

    const withOrigin = rectangleFollicles.filter((f) => f.origin !== undefined);

    // Group by image
    const imageStats = new Map<string, { total: number; withOrigin: number }>();
    rectangleFollicles.forEach((f) => {
      const stats = imageStats.get(f.imageId) || { total: 0, withOrigin: 0 };
      stats.total++;
      if (f.origin) stats.withOrigin++;
      imageStats.set(f.imageId, stats);
    });

    const imagesWithOrigins = Array.from(imageStats.values()).filter((s) => s.withOrigin > 0).length;

    return {
      totalFollicles: rectangleFollicles.length,
      withOrigin: withOrigin.length,
      totalImages: images.size,
      imagesWithOrigins,
    };
  }, [follicles, images]);

  // Check if project has enough data for training
  const projectHasEnoughData = projectStats.withOrigin >= 10;

  // Check dependencies on mount
  useEffect(() => {
    yoloKeypointService.checkDependencies().then((deps) => {
      setDependencies(deps);
      if (deps.installed) {
        yoloKeypointService.getStatus().then((status) => {
          setServiceAvailable(status.available);
        });
        // Also fetch system info
        window.electronAPI.yoloKeypoint.getSystemInfo().then((info) => {
          setSystemInfo(info);
        });
      }
    });

    // Fetch GPU hardware info
    window.electronAPI.gpu.getHardwareInfo().then((info) => {
      setGpuHardware(info);
      // Auto-select GPU if available and packages installed
      if (info.gpuEnabled && config.device === 'auto') {
        // Keep auto - it will use GPU automatically
      }
    }).catch((err) => {
      console.warn('Failed to fetch GPU hardware info:', err);
    });

    // Listen for GPU install progress
    const cleanupProgress = window.electronAPI.gpu.onInstallProgress(({ message }) => {
      setGpuInstallProgress(message);
    });

    return () => {
      cleanupProgress();
    };
  }, [config.device]);

  // Poll system info during training for live updates
  useEffect(() => {
    if (status === 'training' || status === 'preparing') {
      // Start polling
      const fetchSystemInfo = () => {
        window.electronAPI.yoloKeypoint.getSystemInfo().then((info) => {
          setSystemInfo(info);
        }).catch(() => {
          // Ignore errors during polling
        });
      };

      // Initial fetch
      fetchSystemInfo();

      // Poll every 2 seconds
      systemInfoIntervalRef.current = setInterval(fetchSystemInfo, 2000);

      return () => {
        if (systemInfoIntervalRef.current) {
          clearInterval(systemInfoIntervalRef.current);
          systemInfoIntervalRef.current = null;
        }
      };
    } else {
      // Stop polling when not training
      if (systemInfoIntervalRef.current) {
        clearInterval(systemInfoIntervalRef.current);
        systemInfoIntervalRef.current = null;
      }
    }
  }, [status]);

  // Install dependencies
  const handleInstallDependencies = useCallback(async () => {
    setInstallingDeps(true);
    setInstallError(null);
    setInstallProgress('Starting installation...');

    const result = await yoloKeypointService.installDependencies((message) => {
      setInstallProgress(message);
    });

    setInstallingDeps(false);

    if (result.success) {
      const deps = await yoloKeypointService.checkDependencies();
      setDependencies(deps);
      if (deps.installed) {
        try {
          await window.electronAPI.blob.restartServer();
        } catch (e) {
          console.warn('Failed to restart server:', e);
        }
        const status = await yoloKeypointService.getStatus();
        setServiceAvailable(status.available);
        // Refresh GPU hardware info after YOLO deps installed
        const gpuInfo = await window.electronAPI.gpu.getHardwareInfo();
        setGpuHardware(gpuInfo);
      }
    } else {
      setInstallError(result.error || 'Installation failed');
    }
  }, []);

  // Install GPU packages (CUDA/Metal support)
  const handleInstallGPU = useCallback(async () => {
    setIsInstallingGpu(true);
    setGpuInstallError(null);
    setGpuInstallProgress('Starting GPU package installation...');

    try {
      const result = await window.electronAPI.gpu.installPackages();

      if (result.success) {
        setGpuInstallProgress('Restarting server...');
        // Restart the blob server to pick up new packages
        try {
          await window.electronAPI.blob.restartServer();
        } catch (e) {
          console.warn('Failed to restart server:', e);
        }

        // Refresh GPU hardware info
        const gpuInfo = await window.electronAPI.gpu.getHardwareInfo();
        setGpuHardware(gpuInfo);

        // Refresh system info to show GPU device
        const sysInfo = await window.electronAPI.yoloKeypoint.getSystemInfo();
        setSystemInfo(sysInfo);

        setIsInstallingGpu(false);
        setGpuInstallProgress('');
      } else {
        setGpuInstallError(result.error || 'GPU package installation failed');
        setIsInstallingGpu(false);
      }
    } catch (error) {
      setGpuInstallError(error instanceof Error ? error.message : 'GPU installation failed');
      setIsInstallingGpu(false);
    }
  }, []);

  // Upgrade PyTorch to CUDA version for GPU training
  const handleUpgradeToGPU = useCallback(async () => {
    setIsInstallingGpu(true);
    setGpuInstallError(null);
    setGpuInstallProgress('Upgrading PyTorch to CUDA version...');

    try {
      const result = await yoloKeypointService.upgradeToCUDA((message) => {
        setGpuInstallProgress(message);
      });

      if (result.success) {
        setGpuInstallProgress('Restarting server...');
        // Restart the blob server to pick up new packages
        try {
          await window.electronAPI.blob.restartServer();
        } catch (e) {
          console.warn('Failed to restart server:', e);
        }

        // Refresh GPU hardware info
        const gpuInfo = await window.electronAPI.gpu.getHardwareInfo();
        setGpuHardware(gpuInfo);

        // Refresh system info to show GPU device
        const sysInfo = await window.electronAPI.yoloKeypoint.getSystemInfo();
        setSystemInfo(sysInfo);

        setIsInstallingGpu(false);
        setGpuInstallProgress('');
      } else {
        setGpuInstallError(result.error || 'PyTorch CUDA upgrade failed');
        setIsInstallingGpu(false);
      }
    } catch (error) {
      setGpuInstallError(error instanceof Error ? error.message : 'PyTorch upgrade failed');
      setIsInstallingGpu(false);
    }
  }, []);

  // Validate external dataset
  const handleValidateDataset = useCallback(async () => {
    if (!datasetPath.trim()) return;

    setValidationStatus('validating');
    setValidation(null);
    setErrorMessage(null);

    try {
      const result = await yoloKeypointService.validateDataset(datasetPath);
      setValidation(result);
      setValidationStatus(result.valid ? 'valid' : 'invalid');
    } catch (error) {
      setValidationStatus('invalid');
      setErrorMessage(error instanceof Error ? error.message : 'Validation failed');
    }
  }, [datasetPath]);

  // Start training
  const handleStartTraining = useCallback(async () => {
    setStatus('preparing');
    setProgress(null);
    setErrorMessage(null);

    let trainingDatasetPath = datasetPath;

    try {
      // If using project data, export to temp directory first
      if (datasetSource === 'project') {
        setPreparingMessage('Preparing dataset from project (cropping images)...');

        // Use the existing export function to generate files in the renderer
        // This is necessary because cropping uses canvas which is only available here
        const { files, stats } = await exportYOLOKeypointDataset(images, follicles);

        if (stats.annotationsWithOrigin === 0) {
          throw new Error('No annotations with origin data found');
        }

        setPreparingMessage(`Processed ${stats.trainCount + stats.valCount} annotations. Writing to disk...`);

        // Convert blobs to ArrayBuffers for IPC transfer
        const filesForIPC: Array<{ path: string; content: ArrayBuffer | string }> = [];
        for (const file of files) {
          if (file.content instanceof Blob) {
            const buffer = await file.content.arrayBuffer();
            filesForIPC.push({ path: file.path, content: buffer });
          } else {
            filesForIPC.push({ path: file.path, content: file.content });
          }
        }

        // Send files to main process to write to temp directory
        const result = await window.electronAPI.yoloKeypoint.writeDatasetToTemp(filesForIPC);

        if (!result.success || !result.datasetPath) {
          throw new Error(result.error || 'Failed to write dataset to disk');
        }

        trainingDatasetPath = result.datasetPath;
        setPreparingMessage('Dataset prepared. Starting training...');
      }

      setStatus('training');

      await yoloKeypointService.startTraining(
        trainingDatasetPath,
        config,
        (progressUpdate) => {
          setProgress(progressUpdate);

          if (progressUpdate.status === 'completed') {
            setStatus('completed');
          } else if (progressUpdate.status === 'failed') {
            setStatus('failed');
            setErrorMessage(progressUpdate.message);
          } else if (progressUpdate.status === 'stopped') {
            setStatus('idle');
          }
        },
        modelName || undefined
      );
    } catch (error) {
      setStatus('failed');
      setErrorMessage(error instanceof Error ? error.message : 'Training failed');
    }
  }, [datasetSource, datasetPath, images, follicles, config, modelName]);

  // Stop training
  const handleStopTraining = useCallback(async () => {
    await yoloKeypointService.stopTraining();
    setStatus('idle');
  }, []);

  // Update config
  const updateConfig = useCallback(<K extends keyof TrainingConfig>(
    key: K,
    value: TrainingConfig[K]
  ) => {
    setConfig((prev) => ({ ...prev, [key]: value }));
  }, []);

  const isTraining = status === 'training' || status === 'preparing';
  const canStartTraining =
    ((datasetSource === 'project' && projectHasEnoughData) ||
     (datasetSource === 'external' && validation?.valid)) &&
    !isTraining &&
    serviceAvailable &&
    dependencies?.installed;

  const progressPercent = progress?.totalEpochs
    ? Math.round((progress.epoch / progress.totalEpochs) * 100)
    : 0;

  return (
    <div className="training-tab-content">
      {/* Dependencies Installation Section */}
      {dependencies && !dependencies.installed && (
        <div className="dependencies-section">
          <div className="dependencies-warning">
            <AlertCircle size={20} />
            <div className="dependencies-info">
              <h4>YOLO Training Dependencies Required</h4>
              <p>
                The following packages need to be installed to enable YOLO training:
              </p>
              <ul className="missing-packages-list">
                {dependencies.missing.map((pkg) => (
                  <li key={pkg}>{pkg}</li>
                ))}
              </ul>
              <p className="size-warning">
                Estimated download size: <strong>{dependencies.estimatedSize}</strong>
              </p>
            </div>
          </div>

          {installError && (
            <div className="install-error">
              <AlertCircle size={16} />
              <span>{installError}</span>
            </div>
          )}

          {installingDeps && (
            <div className="install-progress">
              <Loader2 size={16} className="spin" />
              <span>{installProgress}</span>
            </div>
          )}

          <button
            className="install-btn"
            onClick={handleInstallDependencies}
            disabled={installingDeps}
          >
            {installingDeps ? (
              <>
                <Loader2 size={16} className="spin" />
                Installing...
              </>
            ) : (
              <>
                <Download size={16} />
                Install Dependencies
              </>
            )}
          </button>
        </div>
      )}

      {/* Show loading state while checking dependencies */}
      {dependencies === null && (
        <div className="loading-state">
          <Loader2 size={24} className="spin" />
          <span>Checking dependencies...</span>
        </div>
      )}

      {/* Main content - only show when dependencies are installed */}
      {dependencies?.installed && (
        <>
          {serviceAvailable === false && (
            <div className="service-warning">
              <AlertCircle size={16} />
              <span>YOLO training service not available. Please restart the application.</span>
            </div>
          )}

          {/* GPU Status Section */}
          {/* State 1: GPU Active - packages installed and working */}
          {gpuHardware?.gpuEnabled && systemInfo && (systemInfo.cuda_available || systemInfo.mps_available) && (
            <div className={`gpu-status gpu-status-${systemInfo.cuda_available ? 'cuda' : 'mps'}`}>
              <Zap size={16} />
              <div className="gpu-status-content">
                <span className="gpu-status-label">
                  {systemInfo.cuda_available && 'CUDA Acceleration Available'}
                  {systemInfo.mps_available && !systemInfo.cuda_available && 'Metal Acceleration Available'}
                </span>
                <span className="gpu-status-device">{systemInfo.device_name}</span>
              </div>
              {systemInfo.gpu_memory_total_gb && (
                <span className="gpu-status-memory">
                  {systemInfo.gpu_memory_total_gb.toFixed(1)} GB
                </span>
              )}
            </div>
          )}

          {/* State 2: GPU Available - hardware detected, packages not installed, show install button */}
          {gpuHardware?.canEnableGpu && !gpuHardware.gpuEnabled && !isInstallingGpu && !gpuInstallError && (
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
              <button className="gpu-install-btn" onClick={handleInstallGPU} disabled={isTraining}>
                <Download size={14} />
                Install ({gpuHardware.hardware.apple_silicon.found ? '~2GB' : '~1.5GB'})
              </button>
            </div>
          )}

          {/* State 2b: GPU hardware available, deps installed, but PyTorch has no CUDA - show upgrade button */}
          {gpuHardware?.canEnableGpu && systemInfo && !systemInfo.cuda_available && !systemInfo.mps_available && !isInstallingGpu && !gpuInstallError && (
            <div className="gpu-status gpu-status-upgrade">
              <AlertCircle size={16} />
              <div className="gpu-status-content">
                <span className="gpu-status-label">GPU Not Enabled</span>
                <span className="gpu-status-device">
                  PyTorch installed without {gpuHardware.hardware.nvidia.found ? 'CUDA' : 'Metal'} support
                </span>
              </div>
              <button className="gpu-install-btn" onClick={handleUpgradeToGPU} disabled={isTraining}>
                <Download size={14} />
                Upgrade (~2.5GB)
              </button>
            </div>
          )}

          {/* State 3: Installing GPU packages - show progress */}
          {isInstallingGpu && (
            <div className="gpu-status gpu-status-installing">
              <Loader2 size={16} className="spin" />
              <div className="gpu-status-content">
                <span className="gpu-status-label">Installing GPU Packages</span>
                <span className="gpu-status-progress">{gpuInstallProgress || 'Starting...'}</span>
              </div>
            </div>
          )}

          {/* State 4: GPU Installation Error */}
          {gpuInstallError && !isInstallingGpu && (
            <div className="gpu-status gpu-status-error">
              <AlertCircle size={16} />
              <div className="gpu-status-content">
                <span className="gpu-status-label">GPU Installation Failed</span>
                <span className="gpu-status-error-msg">{gpuInstallError}</span>
              </div>
              <button className="gpu-retry-btn" onClick={handleInstallGPU} disabled={isTraining}>
                Retry
              </button>
            </div>
          )}

          {/* State 5: No GPU - CPU only mode */}
          {!gpuHardware?.canEnableGpu && !isInstallingGpu && !gpuInstallError && (
            <div className="gpu-status gpu-status-cpu">
              <Cpu size={16} />
              <div className="gpu-status-content">
                <span className="gpu-status-label">CPU Mode</span>
                <span className="gpu-status-device">No GPU Detected</span>
              </div>
            </div>
          )}

          {/* Backend Selection - Show when GPU packages are installed */}
          {gpuHardware?.gpuEnabled && (
            <section className="section backend-selection">
              <h3>Processing Backend</h3>
              <div className="backend-toggle">
                <button
                  className={`backend-btn ${config.device !== 'cpu' ? 'active' : ''}`}
                  onClick={() => updateConfig('device', 'auto')}
                  disabled={isTraining}
                >
                  <Zap size={14} />
                  GPU
                </button>
                <button
                  className={`backend-btn ${config.device === 'cpu' ? 'active' : ''}`}
                  onClick={() => updateConfig('device', 'cpu')}
                  disabled={isTraining}
                >
                  <Cpu size={14} />
                  CPU
                </button>
              </div>
              <p className="backend-hint">
                {config.device === 'cpu'
                  ? 'Using CPU for training (slower but more compatible)'
                  : 'Using GPU for training (faster processing)'}
              </p>
            </section>
          )}

          {/* Dataset Source Selection */}
          <section className="section config-section">
            <h3>Dataset Source</h3>
            <div className="source-tabs">
              <button
                className={`source-tab ${datasetSource === 'project' ? 'active' : ''}`}
                onClick={() => setDatasetSource('project')}
                disabled={isTraining}
              >
                <FileImage size={16} />
                Current Project
              </button>
              <button
                className={`source-tab ${datasetSource === 'external' ? 'active' : ''}`}
                onClick={() => setDatasetSource('external')}
                disabled={isTraining}
              >
                <FolderOpen size={16} />
                External Dataset
              </button>
            </div>

            {/* Current Project Stats */}
            {datasetSource === 'project' && (
              <div className={`project-stats ${projectHasEnoughData ? 'valid' : 'invalid'}`}>
                <div className="stats-header">
                  {projectHasEnoughData ? (
                    <>
                      <CheckCircle size={16} />
                      <span>Project data ready for training</span>
                    </>
                  ) : (
                    <>
                      <AlertCircle size={16} />
                      <span>Not enough training data</span>
                    </>
                  )}
                </div>
                <div className="stats-details">
                  <div className="stat-row">
                    <span className="stat-label">Follicles with origin/direction:</span>
                    <span className={`stat-value ${projectStats.withOrigin >= 10 ? 'good' : 'warning'}`}>
                      {projectStats.withOrigin}
                    </span>
                  </div>
                  <div className="stat-row">
                    <span className="stat-label">Images with annotations:</span>
                    <span className="stat-value">{projectStats.imagesWithOrigins} / {projectStats.totalImages}</span>
                  </div>
                  <div className="stat-row">
                    <span className="stat-label">Total follicles:</span>
                    <span className="stat-value">{projectStats.totalFollicles}</span>
                  </div>
                </div>
                {!projectHasEnoughData && (
                  <p className="stats-hint">
                    Add origin/direction to at least 10 follicles using the Follicle Origin dialog.
                  </p>
                )}
              </div>
            )}

            {/* External Dataset Input */}
            {datasetSource === 'external' && (
              <>
                <div className="dataset-input">
                  <input
                    type="text"
                    value={datasetPath}
                    onChange={(e) => setDatasetPath(e.target.value)}
                    placeholder="Path to dataset directory"
                    disabled={isTraining}
                  />
                  <button
                    className="validate-button"
                    onClick={handleValidateDataset}
                    disabled={!datasetPath.trim() || validationStatus === 'validating' || isTraining}
                  >
                    {validationStatus === 'validating' ? (
                      <Loader2 size={16} className="spin" />
                    ) : (
                      <FolderOpen size={16} />
                    )}
                    Validate
                  </button>
                </div>

                {validation && (
                  <div className={`validation-result ${validation.valid ? 'valid' : 'invalid'}`}>
                    <div className="validation-header">
                      {validation.valid ? (
                        <>
                          <CheckCircle size={16} />
                          <span>Dataset valid</span>
                        </>
                      ) : (
                        <>
                          <AlertCircle size={16} />
                          <span>Dataset invalid</span>
                        </>
                      )}
                    </div>
                    <div className="validation-stats">
                      <span>Train: {validation.trainImages} images, {validation.trainLabels} labels</span>
                      <span>Val: {validation.valImages} images, {validation.valLabels} labels</span>
                    </div>
                    {validation.errors.length > 0 && (
                      <ul className="validation-errors">
                        {validation.errors.map((err, i) => (
                          <li key={i}>{err}</li>
                        ))}
                      </ul>
                    )}
                    {validation.warnings.length > 0 && (
                      <ul className="validation-warnings">
                        {validation.warnings.map((warn, i) => (
                          <li key={i}>{warn}</li>
                        ))}
                      </ul>
                    )}
                  </div>
                )}
              </>
            )}
          </section>

          {/* Configuration Section */}
          <section className="section config-section">
            <h3>Configuration</h3>
            <div className="config-grid">
              <label>
                <span>Model Size</span>
                <select
                  value={config.modelSize}
                  onChange={(e) => updateConfig('modelSize', e.target.value as TrainingConfig['modelSize'])}
                  disabled={isTraining}
                >
                  <option value="n">Nano (fastest)</option>
                  <option value="s">Small</option>
                  <option value="m">Medium</option>
                  <option value="l">Large (most accurate)</option>
                </select>
              </label>

              <label>
                <span>Epochs</span>
                <input
                  type="number"
                  value={config.epochs}
                  onChange={(e) => updateConfig('epochs', parseInt(e.target.value) || 100)}
                  min={1}
                  max={1000}
                  disabled={isTraining}
                />
              </label>

              <label>
                <span>Image Size</span>
                <select
                  value={config.imgSize}
                  onChange={(e) => updateConfig('imgSize', parseInt(e.target.value))}
                  disabled={isTraining}
                >
                  <option value={320}>320</option>
                  <option value={480}>480</option>
                  <option value={640}>640 (recommended)</option>
                  <option value={800}>800</option>
                </select>
              </label>

              <label>
                <span>Batch Size</span>
                <input
                  type="number"
                  value={config.batchSize}
                  onChange={(e) => updateConfig('batchSize', parseInt(e.target.value) || 16)}
                  min={1}
                  max={128}
                  disabled={isTraining}
                />
              </label>

              <label>
                <span>Patience</span>
                <input
                  type="number"
                  value={config.patience}
                  onChange={(e) => updateConfig('patience', parseInt(e.target.value) || 50)}
                  min={1}
                  max={200}
                  disabled={isTraining}
                />
              </label>
            </div>

            <label className="model-name-label">
              <span>Model Name (optional)</span>
              <input
                type="text"
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                placeholder="Auto-generated if empty"
                disabled={isTraining}
              />
            </label>
          </section>

          {/* Preparing Section */}
          {status === 'preparing' && (
            <div className="preparing-section">
              <Loader2 size={20} className="spin" />
              <span>{preparingMessage || 'Preparing...'}</span>
            </div>
          )}

          {/* System Info Section - shown during training or when available */}
          {systemInfo && (status === 'training' || status === 'preparing' || status === 'idle') && dependencies?.installed && (
            <section className="section system-info-section">
              <h3>
                {systemInfo.device === 'cuda' || systemInfo.device === 'mps' ? (
                  <><Monitor size={14} /> GPU Training</>
                ) : (
                  <><Cpu size={14} /> CPU Training</>
                )}
              </h3>
              <div className="system-info-grid">
                <div className="system-info-item">
                  <span className="info-label">Device</span>
                  <span className="info-value" title={systemInfo.device_name}>
                    {systemInfo.device.toUpperCase()}
                    {systemInfo.device === 'cuda' && ' (NVIDIA)'}
                    {systemInfo.device === 'mps' && ' (Apple)'}
                  </span>
                </div>
                <div className="system-info-item">
                  <span className="info-label">Device Name</span>
                  <span className="info-value truncate" title={systemInfo.device_name}>
                    {systemInfo.device_name.length > 25
                      ? systemInfo.device_name.substring(0, 25) + '...'
                      : systemInfo.device_name}
                  </span>
                </div>
                <div className="system-info-item">
                  <span className="info-label">PyTorch</span>
                  <span className="info-value">{systemInfo.torch_version || 'N/A'}</span>
                </div>
                <div className="system-info-item">
                  <span className="info-label">CPU Usage</span>
                  <span className="info-value">
                    {systemInfo.cpu_percent.toFixed(1)}%
                    <span className="info-sub">({systemInfo.cpu_count} cores)</span>
                  </span>
                </div>
                <div className="system-info-item">
                  <span className="info-label">RAM Usage</span>
                  <span className="info-value">
                    {systemInfo.memory_used_gb.toFixed(1)} / {systemInfo.memory_total_gb.toFixed(1)} GB
                    <span className="info-sub">({systemInfo.memory_percent.toFixed(0)}%)</span>
                  </span>
                </div>
                {systemInfo.gpu_memory_total_gb && (
                  <div className="system-info-item">
                    <span className="info-label">GPU Memory</span>
                    <span className="info-value">
                      {(systemInfo.gpu_memory_used_gb || 0).toFixed(1)} / {systemInfo.gpu_memory_total_gb.toFixed(1)} GB
                    </span>
                  </div>
                )}
              </div>
            </section>
          )}

          {/* Progress Section */}
          {(status === 'training' || progress) && (
            <section className="section progress-section">
              <h3>Training Progress</h3>
              <div className="progress-bar">
                <div
                  className="progress-fill"
                  style={{ width: `${progressPercent}%` }}
                />
              </div>
              <div className="progress-info">
                <span>
                  {progress?.epoch || 0} / {progress?.totalEpochs || config.epochs} epochs
                </span>
                {progress?.eta && <span>ETA: {progress.eta}</span>}
              </div>
              {progress && (
                <div className="metrics-grid">
                  <div className="metric">
                    <span className="metric-label">Total Loss</span>
                    <span className="metric-value">{progress.loss.toFixed(4)}</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Box Loss</span>
                    <span className="metric-value">{progress.boxLoss.toFixed(4)}</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Pose Loss</span>
                    <span className="metric-value">{progress.poseLoss.toFixed(4)}</span>
                  </div>
                </div>
              )}
              {progress?.message && (
                <div className="progress-message">{progress.message}</div>
              )}
            </section>
          )}

          {/* Status Messages */}
          {status === 'completed' && (
            <div className="status-message success">
              <CheckCircle size={16} />
              <span>Training completed successfully!</span>
            </div>
          )}

          {(status === 'failed' || errorMessage) && (
            <div className="status-message error">
              <AlertCircle size={16} />
              <span>{errorMessage || 'Training failed'}</span>
            </div>
          )}
        </>
      )}

      {/* Footer Actions */}
      <div className="tab-footer">
        {isTraining ? (
          <button className="stop-btn" onClick={handleStopTraining}>
            <Square size={16} />
            Stop Training
          </button>
        ) : (
          <button
            className="start-btn"
            onClick={handleStartTraining}
            disabled={!canStartTraining}
          >
            <Play size={16} />
            Start Training
          </button>
        )}
      </div>
    </div>
  );
}

export default OriginTab;
