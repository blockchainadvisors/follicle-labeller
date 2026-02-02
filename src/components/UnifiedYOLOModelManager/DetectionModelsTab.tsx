import { useState, useEffect, useCallback } from 'react';
import { X, Loader2, Trash2, Download, CheckCircle, RefreshCw, Zap, Upload, Package } from 'lucide-react';
import { yoloDetectionService } from '../../services/yoloDetectionService';
import { DetectionModelInfo, TensorRTStatus } from '../../types';
import { createModelPackageConfig, formatMetrics as formatPackageMetrics, generateExportFileName, ModelPackageConfig } from '../../utils/model-export';

interface DetectionModelsTabProps {
  selectedModelId?: string | null;  // Currently selected/active model ID
  onModelSelected?: (modelId: string | null, modelName: string | null) => void;  // Selection callback
}

export function DetectionModelsTab({ selectedModelId, onModelSelected }: DetectionModelsTabProps) {
  const [models, setModels] = useState<DetectionModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [exportingModel, setExportingModel] = useState<string | null>(null);
  const [exportingTensorRT, setExportingTensorRT] = useState<string | null>(null);
  const [deletingModel, setDeletingModel] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tensorrtStatus, setTensorrtStatus] = useState<TensorRTStatus | null>(null);
  const [exportingPackage, setExportingPackage] = useState<string | null>(null);
  const [importingPackage, setImportingPackage] = useState(false);
  const [importPreview, setImportPreview] = useState<{
    filePath: string;
    config: ModelPackageConfig;
    hasEngine: boolean;
  } | null>(null);

  // Load models on mount
  const loadModels = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const modelList = await yoloDetectionService.listModels();
      setModels(modelList);

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
      } else {
        setError('Failed to delete model');
      }
    } catch (err) {
      setError('Failed to delete model');
      console.error('Failed to delete model:', err);
    } finally {
      setDeletingModel(null);
    }
  }, []);

  // Export model as portable package (ZIP)
  const handleExportPackage = useCallback(async (model: DetectionModelInfo) => {
    setError(null);
    setExportingPackage(model.id);
    try {
      // Create config for package
      const config = createModelPackageConfig(model);

      // Generate descriptive filename
      const suggestedFileName = generateExportFileName(model, 'detection');

      const result = await window.electronAPI.model.exportPackage(
        model.id,
        model.path,
        config as unknown as Record<string, unknown>,
        suggestedFileName
      );

      if (result.canceled) {
        // User canceled - do nothing
        return;
      }

      if (result.success && result.filePath) {
        alert(`Model exported to:\n${result.filePath}`);
      } else {
        setError(result.error || 'Export failed');
      }
    } catch (err) {
      setError('Failed to export model package');
      console.error('Failed to export model package:', err);
    } finally {
      setExportingPackage(null);
    }
  }, []);

  // Preview model package before import
  const handleImportPackagePreview = useCallback(async () => {
    setError(null);
    setImportingPackage(true);
    try {
      // Pass 'detection' to validate that we're importing a detection model
      const result = await window.electronAPI.model.previewPackage('detection');

      if (result.canceled) {
        // User canceled - do nothing
        return;
      }

      if (result.valid && result.filePath && result.config) {
        setImportPreview({
          filePath: result.filePath,
          config: result.config as unknown as ModelPackageConfig,
          hasEngine: result.hasEngine || false,
        });
      } else {
        setError(result.error || 'Invalid model package');
      }
    } catch (err) {
      setError('Failed to read model package');
      console.error('Failed to read model package:', err);
    } finally {
      setImportingPackage(false);
    }
  }, []);

  // Confirm and import model package
  const handleConfirmImport = useCallback(async () => {
    if (!importPreview) return;

    setError(null);
    setImportingPackage(true);
    try {
      const result = await window.electronAPI.model.importPackage(
        importPreview.filePath,
        importPreview.config.modelName
      );

      if (result.success) {
        alert(`Model "${result.modelName}" imported successfully!`);
        setImportPreview(null);
        // Refresh model list
        loadModels();
      } else {
        setError(result.error || 'Import failed');
      }
    } catch (err) {
      setError('Failed to import model package');
      console.error('Failed to import model package:', err);
    } finally {
      setImportingPackage(false);
    }
  }, [importPreview, loadModels]);

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
        <button
          className="import-model-button"
          onClick={handleImportPackagePreview}
          disabled={loading || importingPackage}
          title="Import model package"
        >
          {importingPackage ? (
            <Loader2 size={16} className="spin" />
          ) : (
            <Upload size={16} />
          )}
          Import Model
        </button>
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
          {models.map((model) => {
            const isActive = selectedModelId === model.id;
            return (
            <div
              key={model.id}
              className={`model-card ${isActive ? 'active' : ''} ${onModelSelected ? 'selectable' : ''}`}
              onClick={onModelSelected ? () => onModelSelected(model.id, model.name) : undefined}
            >
              <div className="model-info">
                <div className="model-name">
                  {model.name}
                  {isActive && (
                    <span className="active-badge">
                      <CheckCircle size={12} />
                      Active
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

              <div className="model-actions" onClick={(e) => e.stopPropagation()}>
                {/* Activate Button - only show if selection is supported and not already active */}
                {onModelSelected && !isActive && (
                  <button
                    className="action-button activate"
                    onClick={() => onModelSelected(model.id, model.name)}
                    title="Activate this model for inference"
                  >
                    <CheckCircle size={16} />
                  </button>
                )}

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
                  className="action-button export-package"
                  onClick={() => handleExportPackage(model)}
                  disabled={exportingPackage === model.id}
                  title="Export as portable package (ZIP)"
                >
                  {exportingPackage === model.id ? (
                    <Loader2 size={16} className="spin" />
                  ) : (
                    <Package size={16} />
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
          );
          })}
        </div>
      )}

      {/* Import Preview Dialog */}
      {importPreview && (
        <div className="import-preview-overlay" onClick={() => setImportPreview(null)}>
          <div className="import-preview-dialog" onClick={(e) => e.stopPropagation()}>
            <div className="import-preview-header">
              <h3>Import Model Package</h3>
              <button className="close-button" onClick={() => setImportPreview(null)}>
                <X size={18} />
              </button>
            </div>

            <div className="import-preview-content">
              <div className="preview-item">
                <span className="preview-label">Model Name:</span>
                <span className="preview-value">{importPreview.config.modelName}</span>
              </div>
              <div className="preview-item">
                <span className="preview-label">Training:</span>
                <span className="preview-value">
                  {importPreview.config.training.epochs} epochs, {importPreview.config.training.imgSize}px
                </span>
              </div>
              <div className="preview-item">
                <span className="preview-label">Metrics:</span>
                <span className="preview-value">
                  {formatPackageMetrics(importPreview.config.metrics)}
                </span>
              </div>
              <div className="preview-item">
                <span className="preview-label">Trained:</span>
                <span className="preview-value">
                  {formatDate(importPreview.config.trainingDate)}
                </span>
              </div>
              {importPreview.hasEngine && (
                <div className="preview-warning">
                  <Zap size={14} />
                  <span>
                    Package includes TensorRT engine (may not work on different GPU)
                  </span>
                </div>
              )}
            </div>

            <div className="import-preview-footer">
              <button
                className="cancel-button"
                onClick={() => setImportPreview(null)}
              >
                Cancel
              </button>
              <button
                className="import-button"
                onClick={handleConfirmImport}
                disabled={importingPackage}
              >
                {importingPackage ? (
                  <>
                    <Loader2 size={16} className="spin" />
                    Importing...
                  </>
                ) : (
                  <>
                    <Upload size={16} />
                    Import Model
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="tab-footer">
        <span className="model-count">{models.length} model{models.length !== 1 ? 's' : ''}</span>
      </div>
    </div>
  );
}

export default DetectionModelsTab;
