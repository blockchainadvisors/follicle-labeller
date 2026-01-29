import { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import { X, Play, Square, Loader2, AlertCircle, CheckCircle, FolderOpen, Download, FileImage, Cpu, Monitor, Zap, RotateCcw } from 'lucide-react';
import { yoloDetectionService } from '../../services/yoloDetectionService';
import { yoloKeypointService } from '../../services/yoloKeypointService';
import { useProjectStore } from '../../store/projectStore';
import { useFollicleStore } from '../../store/follicleStore';
import { exportYOLODetectionDataset } from '../../utils/yolo-detection-export';
import {
  DetectionTrainingConfig,
  DetectionTrainingProgress,
  DetectionDatasetValidation,
  DetectionModelInfo,
  YoloDependenciesInfo,
  DEFAULT_DETECTION_TRAINING_CONFIG,
  RectangleAnnotation,
  GPUHardwareInfo,
} from '../../types';
import './YOLODetectionTrainingDialog.css';

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

interface YOLODetectionTrainingDialogProps {
  onClose: () => void;
}

type TrainingStatus = 'idle' | 'validating' | 'preparing' | 'training' | 'completed' | 'failed';
type DatasetSource = 'project' | 'external';

export function YOLODetectionTrainingDialog({ onClose }: YOLODetectionTrainingDialogProps) {
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

  // GPU hardware state
  const [gpuHardware, setGpuHardware] = useState<GPUHardwareInfo | null>(null);
  const [isInstallingGpu, setIsInstallingGpu] = useState(false);
  const [gpuInstallProgress, setGpuInstallProgress] = useState<string>('');
  const [gpuInstallError, setGpuInstallError] = useState<string | null>(null);

  // Dataset source state
  const [datasetSource, setDatasetSource] = useState<DatasetSource>('project');
  const [datasetPath, setDatasetPath] = useState('');
  const [validation, setValidation] = useState<DetectionDatasetValidation | null>(null);
  const [validationStatus, setValidationStatus] = useState<'idle' | 'validating' | 'valid' | 'invalid'>('idle');

  // Training config
  const [config, setConfig] = useState<DetectionTrainingConfig>(DEFAULT_DETECTION_TRAINING_CONFIG);
  const [modelName, setModelName] = useState('');

  // Training state
  const [status, setStatus] = useState<TrainingStatus>('idle');
  const [progress, setProgress] = useState<DetectionTrainingProgress | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [preparingMessage, setPreparingMessage] = useState<string>('');

  // Service availability
  const [serviceAvailable, setServiceAvailable] = useState<boolean | null>(null);

  // Resume training state
  const [resumableModels, setResumableModels] = useState<DetectionModelInfo[]>([]);
  const [selectedResumeModel, setSelectedResumeModel] = useState<string | null>(null);
  const [isResumeMode, setIsResumeMode] = useState(false);

  // Calculate project stats - rectangle annotations (bounding boxes)
  const projectStats = useMemo(() => {
    const rectangleFollicles = follicles.filter(
      (f): f is RectangleAnnotation => f.shape === 'rectangle'
    );

    // Group by image
    const imageStats = new Map<string, number>();
    rectangleFollicles.forEach((f) => {
      const count = imageStats.get(f.imageId) || 0;
      imageStats.set(f.imageId, count + 1);
    });

    const imagesWithAnnotations = imageStats.size;

    return {
      totalAnnotations: rectangleFollicles.length,
      totalImages: images.size,
      imagesWithAnnotations,
    };
  }, [follicles, images]);

  // Check if project has enough data for training
  // Need at least 20 annotations on at least 1 image
  const projectHasEnoughData = projectStats.totalAnnotations >= 20 && projectStats.imagesWithAnnotations >= 1;

  // Check dependencies on mount (using keypoint service as they share dependencies)
  useEffect(() => {
    yoloKeypointService.checkDependencies().then((deps) => {
      setDependencies(deps);
      if (deps.installed) {
        yoloDetectionService.getStatus().then((status) => {
          setServiceAvailable(status.available);
        });
        // Also fetch system info
        window.electronAPI.yoloKeypoint.getSystemInfo().then((info) => {
          setSystemInfo(info);
        });
        // Fetch resumable models
        yoloDetectionService.getResumableModels().then((models) => {
          setResumableModels(models);
        });
      }
    });

    // Fetch GPU hardware info
    window.electronAPI.gpu.getHardwareInfo().then((info) => {
      setGpuHardware(info);
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
  }, []);

  // Poll system info during training
  useEffect(() => {
    if (status === 'training' || status === 'preparing') {
      const fetchSystemInfo = () => {
        window.electronAPI.yoloKeypoint.getSystemInfo().then((info) => {
          setSystemInfo(info);
        }).catch(() => {});
      };

      fetchSystemInfo();
      systemInfoIntervalRef.current = setInterval(fetchSystemInfo, 2000);

      return () => {
        if (systemInfoIntervalRef.current) {
          clearInterval(systemInfoIntervalRef.current);
          systemInfoIntervalRef.current = null;
        }
      };
    } else {
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
        const status = await yoloDetectionService.getStatus();
        setServiceAvailable(status.available);
        const info = await window.electronAPI.yoloKeypoint.getSystemInfo();
        setSystemInfo(info);
      }
    } else {
      setInstallError(result.error || 'Installation failed');
    }
  }, []);

  // Install GPU packages
  const handleInstallGpu = useCallback(async () => {
    setIsInstallingGpu(true);
    setGpuInstallError(null);
    setGpuInstallProgress('Starting GPU package installation...');

    try {
      const result = await window.electronAPI.gpu.installPackages();
      if (result.success) {
        await window.electronAPI.blob.restartServer();
        const info = await window.electronAPI.gpu.getHardwareInfo();
        setGpuHardware(info);
        const sysInfo = await window.electronAPI.yoloKeypoint.getSystemInfo();
        setSystemInfo(sysInfo);
      } else {
        setGpuInstallError(result.error || 'GPU installation failed');
      }
    } catch (err) {
      setGpuInstallError(err instanceof Error ? err.message : 'Installation failed');
    } finally {
      setIsInstallingGpu(false);
      setGpuInstallProgress('');
    }
  }, []);

  // Browse for external dataset
  const handleBrowseDataset = useCallback(async () => {
    const result = await window.electronAPI.openFileDialog({
      title: 'Select YOLO Detection Dataset Folder',
      filters: [{ name: 'YAML Config', extensions: ['yaml', 'yml'] }],
    });

    if (result) {
      // Get the directory from the yaml file path
      const dir = result.filePath.replace(/[/\\][^/\\]*$/, '');
      setDatasetPath(dir);
      setValidationStatus('idle');
      setValidation(null);
    }
  }, []);

  // Validate dataset
  const handleValidateDataset = useCallback(async () => {
    if (datasetSource === 'external' && !datasetPath) {
      return;
    }

    setValidationStatus('validating');
    setValidation(null);

    try {
      if (datasetSource === 'external') {
        const result = await yoloDetectionService.validateDataset(datasetPath);
        setValidation(result);
        setValidationStatus(result.valid ? 'valid' : 'invalid');
      } else {
        // For project source, validate that we have enough data
        if (projectHasEnoughData) {
          // Check if images are large enough to benefit from tiling
          const firstImageId = Array.from(images.keys())[0];
          const firstImage = firstImageId ? images.get(firstImageId) : null;
          const imgWidth = firstImage?.imageBitmap?.width || 0;
          const imgHeight = firstImage?.imageBitmap?.height || 0;
          const willUseTiling = imgWidth > 1536 || imgHeight > 1536; // 1024 * 1.5

          // Get rectangle annotations for this image
          const imageAnnotations = follicles.filter(
            (f): f is RectangleAnnotation => f.shape === 'rectangle' && f.imageId === firstImageId
          );

          // Estimate tile count for large images
          const tileSize = 1024;
          const tileOverlap = 64;
          const tileStep = tileSize - tileOverlap;

          let totalTiles = 1;
          let tilesWithAnnotations = 1;

          if (willUseTiling) {
            // Calculate grid
            const tilesX = Math.ceil(imgWidth / tileStep);
            const tilesY = Math.ceil(imgHeight / tileStep);
            totalTiles = tilesX * tilesY;

            // Count tiles that have at least one annotation
            tilesWithAnnotations = 0;
            for (let ty = 0; ty < tilesY; ty++) {
              for (let tx = 0; tx < tilesX; tx++) {
                const tileX = tx * tileStep;
                const tileY = ty * tileStep;
                const tileRight = Math.min(tileX + tileSize, imgWidth);
                const tileBottom = Math.min(tileY + tileSize, imgHeight);

                // Check if any annotation intersects this tile (with 50% overlap requirement)
                const hasAnnotation = imageAnnotations.some((ann) => {
                  const annRight = ann.x + ann.width;
                  const annBottom = ann.y + ann.height;

                  // Check intersection
                  if (annRight <= tileX || ann.x >= tileRight || annBottom <= tileY || ann.y >= tileBottom) {
                    return false;
                  }

                  // Calculate intersection area
                  const intersectLeft = Math.max(ann.x, tileX);
                  const intersectTop = Math.max(ann.y, tileY);
                  const intersectRight = Math.min(annRight, tileRight);
                  const intersectBottom = Math.min(annBottom, tileBottom);
                  const intersectArea = (intersectRight - intersectLeft) * (intersectBottom - intersectTop);
                  const annotationArea = ann.width * ann.height;

                  return intersectArea / annotationArea >= 0.5;
                });

                if (hasAnnotation) {
                  tilesWithAnnotations++;
                }
              }
            }
          }

          const trainTiles = Math.max(1, Math.floor(tilesWithAnnotations * 0.8));
          const valTiles = tilesWithAnnotations - trainTiles;

          const warnings: string[] = [];
          if (projectStats.totalAnnotations < 100) {
            warnings.push('Consider adding more annotations for better results');
          }
          if (willUseTiling) {
            const emptyTiles = totalTiles - tilesWithAnnotations;
            warnings.push(`Image split into ${totalTiles} tiles → ${tilesWithAnnotations} have annotations (${emptyTiles} empty tiles excluded)`);
          }

          setValidation({
            valid: true,
            trainImages: trainTiles,
            valImages: valTiles,
            trainLabels: Math.floor(projectStats.totalAnnotations * 0.8),
            valLabels: Math.ceil(projectStats.totalAnnotations * 0.2),
            errors: [],
            warnings,
          });
          setValidationStatus('valid');
        } else {
          setValidation({
            valid: false,
            trainImages: projectStats.imagesWithAnnotations,
            valImages: 0,
            trainLabels: projectStats.totalAnnotations,
            valLabels: 0,
            errors: [
              projectStats.totalAnnotations < 20
                ? `Need at least 20 rectangle annotations (have ${projectStats.totalAnnotations})`
                : '',
              projectStats.imagesWithAnnotations < 1
                ? `Need annotations on at least 1 image`
                : '',
            ].filter(Boolean),
            warnings: [],
          });
          setValidationStatus('invalid');
        }
      }
    } catch (err) {
      setValidation({
        valid: false,
        trainImages: 0,
        valImages: 0,
        trainLabels: 0,
        valLabels: 0,
        errors: [err instanceof Error ? err.message : 'Validation failed'],
        warnings: [],
      });
      setValidationStatus('invalid');
    }
  }, [datasetSource, datasetPath, projectHasEnoughData, projectStats]);

  // Start training
  const handleStartTraining = useCallback(async () => {
    // For resume mode, we don't need validation - just need the selected model
    if (!isResumeMode && validationStatus !== 'valid') {
      return;
    }

    if (isResumeMode && !selectedResumeModel) {
      return;
    }

    setStatus('preparing');
    setErrorMessage(null);
    setProgress(null);

    try {
      let trainingDatasetPath = datasetPath;
      let trainingConfig = { ...config };

      if (isResumeMode && selectedResumeModel) {
        // Resume mode - use the selected model's checkpoint
        setPreparingMessage(`Resuming training from ${selectedResumeModel}...`);
        trainingConfig.resumeFrom = selectedResumeModel;
        // Dataset path doesn't matter for resume - YOLO reads from checkpoint
        trainingDatasetPath = '';
      } else if (datasetSource === 'project') {
        // If using project source, export dataset first
        setPreparingMessage('Exporting detection dataset from project...');

        // Get all rectangle annotations with their images
        const rectangleFollicles = follicles.filter(
          (f): f is RectangleAnnotation => f.shape === 'rectangle'
        );

        // Export dataset with tiling for large images
        const { files: datasetFiles, stats: exportStats } = await exportYOLODetectionDataset(
          images,
          rectangleFollicles,
          {
            trainSplit: 0.8,
            shuffle: true,
            tileSize: 1024,      // Split large images into 1024x1024 tiles
            tileOverlap: 64,     // 64px overlap between tiles
            minAnnotationOverlap: 0.5, // Include annotation if 50%+ is in tile
          }
        );

        console.log(`Dataset export: ${exportStats.tilesGenerated} tiles, ${exportStats.trainAnnotations} train labels, ${exportStats.valAnnotations} val labels`);

        setPreparingMessage('Writing dataset to disk...');

        // Convert Blobs to ArrayBuffers for IPC transfer
        const filesForIpc = await Promise.all(
          datasetFiles.map(async (file) => ({
            path: file.path,
            content: file.content instanceof Blob
              ? await file.content.arrayBuffer()
              : file.content,
          }))
        );

        // Write to temp directory via IPC
        const writeResult = await window.electronAPI.yoloDetection.writeDatasetToTemp(filesForIpc);

        if (!writeResult.success || !writeResult.datasetPath) {
          throw new Error(writeResult.error || 'Failed to write dataset');
        }

        trainingDatasetPath = writeResult.datasetPath;
      }

      setPreparingMessage(isResumeMode ? 'Resuming training...' : 'Starting training...');
      setStatus('training');

      // Start training with progress callback
      await yoloDetectionService.startTraining(
        trainingDatasetPath,
        trainingConfig,
        (progressUpdate) => {
          setProgress(progressUpdate);

          if (progressUpdate.status === 'completed') {
            setStatus('completed');
          } else if (progressUpdate.status === 'failed') {
            setStatus('failed');
            setErrorMessage(progressUpdate.message || 'Training failed');
          } else if (progressUpdate.status === 'stopped') {
            setStatus('idle');
          }
        },
        isResumeMode ? (selectedResumeModel || undefined) : (modelName || undefined)
      );
    } catch (err) {
      setStatus('failed');
      setErrorMessage(err instanceof Error ? err.message : 'Training failed');
    }
  }, [validationStatus, datasetSource, datasetPath, images, follicles, config, modelName, isResumeMode, selectedResumeModel]);

  // Stop training
  const handleStopTraining = useCallback(async () => {
    await yoloDetectionService.stopTraining();
    setStatus('idle');
  }, []);

  // Update config
  const updateConfig = useCallback(<K extends keyof DetectionTrainingConfig>(
    key: K,
    value: DetectionTrainingConfig[K]
  ) => {
    setConfig((prev) => ({ ...prev, [key]: value }));
  }, []);

  // Determine GPU status
  const gpuStatus = useMemo(() => {
    if (!systemInfo) return 'unknown';
    if (systemInfo.cuda_available) return 'cuda';
    if (systemInfo.mps_available) return 'mps';
    return 'cpu';
  }, [systemInfo]);

  // Can train: either resume mode with selected model, or new training with valid dataset
  const canResume = isResumeMode && selectedResumeModel !== null;
  const canNewTrain = !isResumeMode && validationStatus === 'valid';
  const canTrain = dependencies?.installed && serviceAvailable && (canResume || canNewTrain) && status === 'idle';
  const isTraining = status === 'training' || status === 'preparing';

  return (
    <div className="yolo-detection-training-overlay" onClick={onClose}>
      <div className="yolo-detection-training-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="dialog-header">
          <h2>YOLO Detection Training</h2>
          <button className="close-button" onClick={onClose} disabled={isTraining}>
            <X size={18} />
          </button>
        </div>

        <div className="dialog-content">
          {/* Dependencies Section */}
          {dependencies && !dependencies.installed && (
            <section className="section dependencies-section">
              <h3>
                <AlertCircle size={16} className="section-icon warning" />
                Dependencies Required
              </h3>
              <p className="section-description">
                YOLO training requires additional packages ({dependencies.estimatedSize}).
                These will be installed into the application's Python environment.
              </p>
              <div className="missing-packages">
                {dependencies.missing.map((pkg) => (
                  <span key={pkg} className="package-chip">{pkg}</span>
                ))}
              </div>
              {installingDeps ? (
                <div className="install-progress">
                  <Loader2 size={16} className="spin" />
                  <span>{installProgress}</span>
                </div>
              ) : installError ? (
                <div className="install-error">
                  <AlertCircle size={16} />
                  <span>{installError}</span>
                  <button className="retry-btn" onClick={handleInstallDependencies}>
                    Retry
                  </button>
                </div>
              ) : (
                <button className="install-btn" onClick={handleInstallDependencies}>
                  <Download size={16} />
                  Install Dependencies
                </button>
              )}
            </section>
          )}

          {/* GPU Status Section */}
          {dependencies?.installed && systemInfo && (
            <section className="section gpu-section">
              <h3>
                {gpuStatus === 'cuda' && <Zap size={16} className="section-icon success" />}
                {gpuStatus === 'mps' && <Monitor size={16} className="section-icon success" />}
                {gpuStatus === 'cpu' && <Cpu size={16} className="section-icon" />}
                Processing Device
              </h3>
              <div className="gpu-info">
                <div className="gpu-device">
                  <span className="device-name">{systemInfo.device_name}</span>
                  <span className={`device-type ${gpuStatus}`}>
                    {gpuStatus === 'cuda' && 'CUDA'}
                    {gpuStatus === 'mps' && 'Metal'}
                    {gpuStatus === 'cpu' && 'CPU'}
                  </span>
                </div>
                {systemInfo.gpu_memory_total_gb && (
                  <div className="gpu-memory">
                    <span>GPU Memory:</span>
                    <span>{systemInfo.gpu_memory_used_gb?.toFixed(1) || '0'} / {systemInfo.gpu_memory_total_gb?.toFixed(1) || '0'} GB</span>
                  </div>
                )}
                <div className="system-memory">
                  <span>System Memory:</span>
                  <span>{systemInfo.memory_used_gb?.toFixed(1) || '0'} / {systemInfo.memory_total_gb?.toFixed(1) || '0'} GB</span>
                </div>
              </div>

              {/* GPU Available but not installed */}
              {gpuHardware?.canEnableGpu && !gpuHardware.gpuEnabled && gpuStatus === 'cpu' && (
                <div className="gpu-available-notice">
                  <Zap size={16} />
                  <span>
                    {gpuHardware.hardware.nvidia.found
                      ? `GPU Available: ${gpuHardware.hardware.nvidia.name}`
                      : `GPU Available: ${gpuHardware.hardware.apple_silicon.chip}`}
                  </span>
                  {isInstallingGpu ? (
                    <div className="gpu-install-status">
                      <Loader2 size={14} className="spin" />
                      <span>{gpuInstallProgress}</span>
                    </div>
                  ) : gpuInstallError ? (
                    <div className="gpu-install-error">
                      <span>{gpuInstallError}</span>
                      <button onClick={handleInstallGpu}>Retry</button>
                    </div>
                  ) : (
                    <button className="enable-gpu-btn" onClick={handleInstallGpu}>
                      Enable GPU
                    </button>
                  )}
                </div>
              )}
            </section>
          )}

          {/* Resume Training Section */}
          {dependencies?.installed && resumableModels.length > 0 && (
            <section className="section resume-section">
              <h3>
                <RotateCcw size={16} className="section-icon" />
                Resume Incomplete Training
              </h3>

              <div className="resume-toggle">
                <label className="toggle-label">
                  <input
                    type="checkbox"
                    checked={isResumeMode}
                    onChange={(e) => {
                      setIsResumeMode(e.target.checked);
                      if (!e.target.checked) {
                        setSelectedResumeModel(null);
                      }
                    }}
                    disabled={isTraining}
                  />
                  <span>Resume from previous training</span>
                </label>
              </div>

              {isResumeMode && (
                <div className="resumable-models">
                  <select
                    value={selectedResumeModel || ''}
                    onChange={(e) => setSelectedResumeModel(e.target.value || null)}
                    disabled={isTraining}
                    className="resume-model-select"
                  >
                    <option value="">Select a model to resume...</option>
                    {resumableModels.map((model) => (
                      <option key={model.id} value={model.id}>
                        {model.name} - Epoch {model.epochsCompleted}/{model.totalEpochs}
                        {model.metrics?.mAP50 !== undefined && ` (mAP50: ${(model.metrics.mAP50 * 100).toFixed(1)}%)`}
                      </option>
                    ))}
                  </select>

                  {selectedResumeModel && (
                    <div className="resume-model-info">
                      {(() => {
                        const model = resumableModels.find((m) => m.id === selectedResumeModel);
                        if (!model) return null;
                        return (
                          <>
                            <div className="resume-stat">
                              <span className="stat-label">Progress</span>
                              <span className="stat-value">
                                {model.epochsCompleted} / {model.totalEpochs} epochs
                                ({((model.epochsCompleted || 0) / (model.totalEpochs || 100) * 100).toFixed(0)}%)
                              </span>
                            </div>
                            {model.metrics && Object.keys(model.metrics).length > 0 && (
                              <div className="resume-stat">
                                <span className="stat-label">Best Metrics</span>
                                <span className="stat-value">
                                  {model.metrics.mAP50 !== undefined && `mAP50: ${(model.metrics.mAP50 * 100).toFixed(1)}%`}
                                  {model.metrics.precision !== undefined && ` | Precision: ${(model.metrics.precision * 100).toFixed(1)}%`}
                                </span>
                              </div>
                            )}
                            <div className="resume-hint">
                              Training will continue from epoch {(model.epochsCompleted || 0) + 1}
                            </div>
                          </>
                        );
                      })()}
                    </div>
                  )}
                </div>
              )}
            </section>
          )}

          {/* Dataset Source Section */}
          {dependencies?.installed && !isResumeMode && (
            <section className="section dataset-section">
              <h3>
                <FileImage size={16} className="section-icon" />
                Training Dataset
              </h3>

              <div className="dataset-source-toggle">
                <button
                  className={`source-btn ${datasetSource === 'project' ? 'active' : ''}`}
                  onClick={() => {
                    setDatasetSource('project');
                    setValidationStatus('idle');
                    setValidation(null);
                  }}
                >
                  Current Project
                </button>
                <button
                  className={`source-btn ${datasetSource === 'external' ? 'active' : ''}`}
                  onClick={() => {
                    setDatasetSource('external');
                    setValidationStatus('idle');
                    setValidation(null);
                  }}
                >
                  External Dataset
                </button>
              </div>

              {datasetSource === 'project' ? (
                <div className="project-stats">
                  <div className="stat">
                    <span className="stat-label">Rectangle Annotations</span>
                    <span className={`stat-value ${projectStats.totalAnnotations >= 20 ? 'good' : 'bad'}`}>
                      {projectStats.totalAnnotations}
                    </span>
                  </div>
                  <div className="stat">
                    <span className="stat-label">Images with Annotations</span>
                    <span className={`stat-value ${projectStats.imagesWithAnnotations >= 1 ? 'good' : 'bad'}`}>
                      {projectStats.imagesWithAnnotations} / {projectStats.totalImages}
                    </span>
                  </div>
                  <p className="dataset-hint">
                    {projectHasEnoughData
                      ? 'Your project has enough annotations to train a detection model.'
                      : 'Add more rectangle annotations to train a detection model (minimum 20 annotations).'}
                  </p>
                </div>
              ) : (
                <div className="external-dataset">
                  <div className="dataset-path-row">
                    <input
                      type="text"
                      value={datasetPath}
                      onChange={(e) => setDatasetPath(e.target.value)}
                      placeholder="Select dataset folder..."
                      className="dataset-path-input"
                    />
                    <button className="browse-btn" onClick={handleBrowseDataset}>
                      <FolderOpen size={16} />
                      Browse
                    </button>
                  </div>
                  <p className="dataset-hint">
                    Select a YOLO detection dataset folder with data.yaml, images/, and labels/ directories.
                  </p>
                </div>
              )}

              {/* Validate Button */}
              {(datasetSource === 'project' || datasetPath) && validationStatus === 'idle' && (
                <button className="validate-btn" onClick={handleValidateDataset}>
                  Validate Dataset
                </button>
              )}

              {/* Validation Status */}
              {validationStatus === 'validating' && (
                <div className="validation-status validating">
                  <Loader2 size={16} className="spin" />
                  <span>Validating dataset...</span>
                </div>
              )}

              {validation && validationStatus === 'valid' && (
                <div className="validation-status valid">
                  <CheckCircle size={16} />
                  <div className="validation-details">
                    <span>Dataset Valid</span>
                    <span className="validation-counts">
                      Train: {validation.trainImages} images, {validation.trainLabels} labels |
                      Val: {validation.valImages} images, {validation.valLabels} labels
                    </span>
                    {validation.warnings.length > 0 && (
                      <div className="validation-warnings">
                        {validation.warnings.map((w, i) => (
                          <span key={i} className="warning">{w}</span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              )}

              {validation && validationStatus === 'invalid' && (
                <div className="validation-status invalid">
                  <AlertCircle size={16} />
                  <div className="validation-details">
                    <span>Dataset Invalid</span>
                    {validation.errors.map((e, i) => (
                      <span key={i} className="error">{e}</span>
                    ))}
                  </div>
                </div>
              )}
            </section>
          )}

          {/* Training Config Section - hide in resume mode since config comes from checkpoint */}
          {dependencies?.installed && validationStatus === 'valid' && !isResumeMode && (
            <section className="section config-section">
              <h3>Training Configuration</h3>

              <div className="config-row">
                <label>Model Size</label>
                <select
                  value={config.modelSize}
                  onChange={(e) => updateConfig('modelSize', e.target.value as 'n' | 's' | 'm' | 'l')}
                  disabled={isTraining}
                >
                  <option value="n">Nano (fastest, ~6MB)</option>
                  <option value="s">Small (~22MB)</option>
                  <option value="m">Medium (~52MB)</option>
                  <option value="l">Large (most accurate, ~87MB)</option>
                </select>
              </div>

              <div className="config-row">
                <label>Epochs</label>
                <input
                  type="number"
                  value={config.epochs}
                  onChange={(e) => updateConfig('epochs', parseInt(e.target.value) || 100)}
                  min={10}
                  max={1000}
                  disabled={isTraining}
                />
              </div>

              <div className="config-row">
                <label>Batch Size</label>
                <input
                  type="number"
                  value={config.batchSize}
                  onChange={(e) => updateConfig('batchSize', parseInt(e.target.value) || 16)}
                  min={1}
                  max={128}
                  disabled={isTraining}
                />
              </div>

              <div className="config-row">
                <label>Image Size</label>
                <select
                  value={config.imgSize}
                  onChange={(e) => updateConfig('imgSize', parseInt(e.target.value))}
                  disabled={isTraining}
                >
                  <option value={320}>320px</option>
                  <option value={416}>416px</option>
                  <option value={640}>640px (recommended)</option>
                  <option value={800}>800px</option>
                  <option value={1024}>1024px</option>
                </select>
              </div>

              <div className="config-row">
                <label>Early Stopping Patience</label>
                <input
                  type="number"
                  value={config.patience}
                  onChange={(e) => updateConfig('patience', parseInt(e.target.value) || 50)}
                  min={5}
                  max={200}
                  disabled={isTraining}
                />
              </div>

              <div className="config-row">
                <label>Model Name (optional)</label>
                <input
                  type="text"
                  value={modelName}
                  onChange={(e) => setModelName(e.target.value)}
                  placeholder="Auto-generated if empty"
                  disabled={isTraining}
                />
              </div>
            </section>
          )}

          {/* Training Progress Section */}
          {(status === 'preparing' || status === 'training') && (
            <section className="section progress-section">
              <h3>
                <Loader2 size={16} className="spin section-icon" />
                Training Progress
              </h3>

              {status === 'preparing' && (
                <div className="preparing-status">
                  <span>{preparingMessage}</span>
                </div>
              )}

              {progress && status === 'training' && (
                <div className="training-progress">
                  <div className="progress-header">
                    <span className="epoch">Epoch {progress.epoch} / {progress.totalEpochs}</span>
                    {progress.eta && <span className="eta">ETA: {progress.eta}</span>}
                  </div>

                  <div className="progress-bar">
                    <div
                      className="progress-fill"
                      style={{ width: `${(progress.epoch / progress.totalEpochs) * 100}%` }}
                    />
                  </div>

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
                      <span className="metric-label">Class Loss</span>
                      <span className="metric-value">{progress.clsLoss.toFixed(4)}</span>
                    </div>
                    <div className="metric">
                      <span className="metric-label">DFL Loss</span>
                      <span className="metric-value">{progress.dflLoss.toFixed(4)}</span>
                    </div>
                  </div>

                  {progress.message && (
                    <div className="progress-message">{progress.message}</div>
                  )}
                </div>
              )}
            </section>
          )}

          {/* Completed Section */}
          {status === 'completed' && (
            <section className="section completed-section">
              <h3>
                <CheckCircle size={16} className="section-icon success" />
                Training Completed
              </h3>
              <p>Your detection model has been trained successfully and saved.</p>
              {progress?.metrics && Object.keys(progress.metrics).length > 0 && (
                <div className="final-metrics">
                  <h4>Final Metrics</h4>
                  <div className="metrics-grid">
                    {Object.entries(progress.metrics).slice(0, 6).map(([key, value]) => (
                      <div key={key} className="metric">
                        <span className="metric-label">{key}</span>
                        <span className="metric-value">{typeof value === 'number' ? value.toFixed(4) : value}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              <button className="close-completed-btn" onClick={onClose}>
                Close
              </button>
            </section>
          )}

          {/* Error Section */}
          {status === 'failed' && errorMessage && (
            <section className="section error-section">
              <h3>
                <AlertCircle size={16} className="section-icon error" />
                Training Failed
              </h3>
              <p className="error-message">{errorMessage}</p>
              <button className="retry-training-btn" onClick={() => setStatus('idle')}>
                Try Again
              </button>
            </section>
          )}
        </div>

        <div className="dialog-footer">
          {isTraining ? (
            <button className="stop-btn" onClick={handleStopTraining}>
              <Square size={16} />
              Stop Training
            </button>
          ) : (
            <>
              <button className="cancel-btn" onClick={onClose}>
                Cancel
              </button>
              <button
                className="start-btn"
                onClick={handleStartTraining}
                disabled={!canTrain}
              >
                {isResumeMode ? <RotateCcw size={16} /> : <Play size={16} />}
                {isResumeMode ? 'Resume Training' : 'Start Training'}
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default YOLODetectionTrainingDialog;
