import { useState, useEffect, useCallback } from 'react';
import { X, Loader2, Trash2, Download, CheckCircle, Cpu, RefreshCw } from 'lucide-react';
import { yoloKeypointService } from '../../services/yoloKeypointService';
import { ModelInfo } from '../../types';
import './YOLOModelManager.css';

interface YOLOModelManagerProps {
  onClose: () => void;
  onModelLoaded?: (modelPath: string) => void;
}

export function YOLOModelManager({ onClose, onModelLoaded }: YOLOModelManagerProps) {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingModel, setLoadingModel] = useState<string | null>(null);
  const [exportingModel, setExportingModel] = useState<string | null>(null);
  const [deletingModel, setDeletingModel] = useState<string | null>(null);
  const [loadedModelPath, setLoadedModelPath] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Load models on mount
  const loadModels = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const modelList = await yoloKeypointService.listModels();
      setModels(modelList);
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
  const handleLoadModel = useCallback(async (model: ModelInfo) => {
    setLoadingModel(model.id);
    setError(null);
    try {
      const success = await yoloKeypointService.loadModel(model.path);
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
  const handleExportONNX = useCallback(async (model: ModelInfo) => {
    setError(null);
    try {
      // Show save dialog to let user choose export location
      const defaultFileName = `${model.name.replace(/[^a-zA-Z0-9_-]/g, '_')}.onnx`;
      const outputPath = await yoloKeypointService.showExportDialog(defaultFileName);

      if (!outputPath) {
        // User canceled
        return;
      }

      setExportingModel(model.id);
      const result = await yoloKeypointService.exportONNX(model.path, outputPath);
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

  // Delete model
  const handleDeleteModel = useCallback(async (model: ModelInfo) => {
    if (!confirm(`Delete model "${model.name}"? This cannot be undone.`)) {
      return;
    }

    setDeletingModel(model.id);
    setError(null);
    try {
      const success = await yoloKeypointService.deleteModel(model.id);
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
    return keys.map((key) => `${key}: ${metrics[key].toFixed(3)}`).join(', ');
  };

  return (
    <div className="yolo-model-manager-overlay" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="yolo-model-manager">
        <div className="dialog-header">
          <h2>YOLO Keypoint Models</h2>
          <div className="header-actions">
            <button className="refresh-button" onClick={loadModels} disabled={loading}>
              <RefreshCw size={16} className={loading ? 'animate-spin' : ''} />
            </button>
            <button className="close-button" onClick={onClose}>
              <X size={20} />
            </button>
          </div>
        </div>

        <div className="dialog-content">
          {error && (
            <div className="error-message">{error}</div>
          )}

          {loading ? (
            <div className="loading-state">
              <Loader2 size={32} className="animate-spin" />
              <span>Loading models...</span>
            </div>
          ) : models.length === 0 ? (
            <div className="empty-state">
              <span>No trained models found.</span>
              <span className="hint">Train a model using the Training dialog.</span>
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
                        <Loader2 size={16} className="animate-spin" />
                      ) : (
                        <Cpu size={16} />
                      )}
                    </button>

                    <button
                      className="action-button export"
                      onClick={() => handleExportONNX(model)}
                      disabled={exportingModel === model.id}
                      title="Export to ONNX"
                    >
                      {exportingModel === model.id ? (
                        <Loader2 size={16} className="animate-spin" />
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
                        <Loader2 size={16} className="animate-spin" />
                      ) : (
                        <Trash2 size={16} />
                      )}
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="dialog-footer">
          <span className="model-count">{models.length} model{models.length !== 1 ? 's' : ''}</span>
          <button className="close-dialog-button" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
