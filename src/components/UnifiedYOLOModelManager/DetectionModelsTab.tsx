import { useState, useEffect, useCallback } from 'react';
import { Loader2, Trash2, Download, CheckCircle, Cpu, RefreshCw, Zap } from 'lucide-react';
import { yoloDetectionService } from '../../services/yoloDetectionService';
import { DetectionModelInfo, TensorRTStatus } from '../../types';

interface DetectionModelsTabProps {
  onModelLoaded?: (modelPath: string) => void;
}

export function DetectionModelsTab({ onModelLoaded }: DetectionModelsTabProps) {
  const [models, setModels] = useState<DetectionModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingModel, setLoadingModel] = useState<string | null>(null);
  const [exportingModel, setExportingModel] = useState<string | null>(null);
  const [exportingTensorRT, setExportingTensorRT] = useState<string | null>(null);
  const [deletingModel, setDeletingModel] = useState<string | null>(null);
  const [loadedModelPath, setLoadedModelPath] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tensorrtStatus, setTensorrtStatus] = useState<TensorRTStatus | null>(null);

  // Load models on mount
  const loadModels = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const modelList = await yoloDetectionService.listModels();
      setModels(modelList);

      // Check which model is currently loaded
      const status = await yoloDetectionService.getStatus();
      if (status.loadedModel) {
        setLoadedModelPath(status.loadedModel);
      }

      // Check TensorRT availability
      const trtStatus = await yoloDetectionService.checkTensorRTAvailable();
      setTensorrtStatus(trtStatus);
    } catch (err) {
      setError('Failed to load models');
      console.error('Failed to load models:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadModels();
  }, [loadModels]);

  // Load model for inference
  const handleLoadModel = useCallback(async (model: DetectionModelInfo) => {
    setLoadingModel(model.id);
    setError(null);
    try {
      const success = await yoloDetectionService.loadModel(model.path);
      if (success) {
        setLoadedModelPath(model.path);
        onModelLoaded?.(model.path);
      } else {
        setError('Failed to load model');
      }
    } catch (err) {
      setError('Failed to load model');
      console.error('Failed to load model:', err);
    } finally {
      setLoadingModel(null);
    }
  }, [onModelLoaded]);

  // Export model to ONNX
  const handleExportONNX = useCallback(async (model: DetectionModelInfo) => {
    setError(null);
    try {
      // Show save dialog to let user choose export location
      const defaultFileName = `${model.name.replace(/[^a-zA-Z0-9_-]/g, '_')}.onnx`;
      const outputPath = await yoloDetectionService.showExportDialog(defaultFileName);

      if (!outputPath) {
        // User canceled
        return;
      }

      setExportingModel(model.id);
      const result = await yoloDetectionService.exportONNX(model.path, outputPath);
      if (result) {
        alert(`Model exported to: ${result}`);
      } else {
        setError('ONNX export failed');
      }
    } catch (err) {
      setError('ONNX export failed');
      console.error('ONNX export failed:', err);
    } finally {
      setExportingModel(null);
    }
  }, []);

  // Export model to TensorRT
  const handleExportTensorRT = useCallback(async (model: DetectionModelInfo) => {
    setError(null);
    try {
      // Confirm export since it can take a while
      if (!confirm(`Export "${model.name}" to TensorRT?\n\nThis may take several minutes. The .engine file will be saved alongside the model.`)) {
        return;
      }

      setExportingTensorRT(model.id);

      // Generate output path next to the .pt file
      const outputPath = model.path.replace(/\.pt$/, '.engine');

      const result = await yoloDetectionService.exportToTensorRT(
        model.path,
        outputPath,
        true, // half precision
        model.imgSize // use same image size as training
      );

      if (result.success && result.engine_path) {
        alert(`TensorRT engine exported to:\n${result.engine_path}`);
        // Refresh model list to show the engine file
        loadModels();
      } else {
        setError(result.error || 'TensorRT export failed');
      }
    } catch (err) {
      setError('TensorRT export failed');
      console.error('TensorRT export failed:', err);
    } finally {
      setExportingTensorRT(null);
    }
  }, [loadModels]);

  // Delete model
  const handleDeleteModel = useCallback(async (model: DetectionModelInfo) => {
    if (!confirm(`Delete model "${model.name}"? This cannot be undone.`)) {
      return;
    }

    setDeletingModel(model.id);
    setError(null);
    try {
      const success = await yoloDetectionService.deleteModel(model.id);
      if (success) {
        setModels((prev) => prev.filter((m) => m.id !== model.id));
        if (loadedModelPath === model.path) {
          setLoadedModelPath(null);
        }
      } else {
        setError('Failed to delete model');
      }
    } catch (err) {
      setError('Failed to delete model');
      console.error('Failed to delete model:', err);
    } finally {
      setDeletingModel(null);
    }
  }, [loadedModelPath]);

  // Format date
  const formatDate = (isoDate: string) => {
    try {
      return new Date(isoDate).toLocaleString();
    } catch {
      return isoDate;
    }
  };

  // Format metrics
  const formatMetrics = (metrics: Record<string, number>) => {
    const keys = Object.keys(metrics).slice(0, 3);
    return keys.map((key) => {
      const value = metrics[key];
      return `${key}: ${typeof value === 'number' ? value.toFixed(3) : value}`;
    }).join(', ');
  };

  return (
    <div className="model-manager-tab-content">
      <div className="tab-header">
        <button className="refresh-button" onClick={loadModels} disabled={loading}>
          <RefreshCw size={16} className={loading ? 'spin' : ''} />
          Refresh
        </button>
      </div>

      {error && (
        <div className="error-message">{error}</div>
      )}

      {loading ? (
        <div className="loading-state">
          <Loader2 size={32} className="spin" />
          <span>Loading models...</span>
        </div>
      ) : models.length === 0 ? (
        <div className="empty-state">
          <span>No trained detection models found.</span>
          <span className="hint">Train a model using the YOLO Training dialog (Detection tab).</span>
        </div>
      ) : (
        <div className="models-list">
          {models.map((model) => (
            <div
              key={model.id}
              className={`model-card ${loadedModelPath === model.path ? 'loaded' : ''}`}
            >
              <div className="model-info">
                <div className="model-name">
                  {model.name}
                  {loadedModelPath === model.path && (
                    <span className="loaded-badge">
                      <CheckCircle size={12} />
                      Loaded
                    </span>
                  )}
                </div>
                <div className="model-meta">
                  <span>Created: {formatDate(model.createdAt)}</span>
                  <span>Epochs: {model.epochsTrained}</span>
                  <span>Size: {model.imgSize}px</span>
                </div>
                {Object.keys(model.metrics).length > 0 && (
                  <div className="model-metrics">
                    {formatMetrics(model.metrics)}
                  </div>
                )}
              </div>

              <div className="model-actions">
                <button
                  className="action-button load"
                  onClick={() => handleLoadModel(model)}
                  disabled={loadingModel === model.id || loadedModelPath === model.path}
                  title="Load for inference"
                >
                  {loadingModel === model.id ? (
                    <Loader2 size={16} className="spin" />
                  ) : (
                    <Cpu size={16} />
                  )}
                </button>

                {/* TensorRT Export Button - only show if TensorRT is available */}
                {tensorrtStatus?.available && (
                  <button
                    className="action-button export-tensorrt"
                    onClick={() => handleExportTensorRT(model)}
                    disabled={exportingTensorRT === model.id}
                    title="Export to TensorRT (faster GPU inference)"
                  >
                    {exportingTensorRT === model.id ? (
                      <Loader2 size={16} className="spin" />
                    ) : (
                      <Zap size={16} />
                    )}
                  </button>
                )}

                <button
                  className="action-button export"
                  onClick={() => handleExportONNX(model)}
                  disabled={exportingModel === model.id}
                  title="Export to ONNX"
                >
                  {exportingModel === model.id ? (
                    <Loader2 size={16} className="spin" />
                  ) : (
                    <Download size={16} />
                  )}
                </button>

                <button
                  className="action-button delete"
                  onClick={() => handleDeleteModel(model)}
                  disabled={deletingModel === model.id}
                  title="Delete model"
                >
                  {deletingModel === model.id ? (
                    <Loader2 size={16} className="spin" />
                  ) : (
                    <Trash2 size={16} />
                  )}
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="tab-footer">
        <span className="model-count">{models.length} model{models.length !== 1 ? 's' : ''}</span>
      </div>
    </div>
  );
}

export default DetectionModelsTab;
