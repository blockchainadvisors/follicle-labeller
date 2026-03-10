/**
 * Model Export/Import Utilities
 *
 * Provides functionality to export trained YOLO models as portable packages
 * that can be imported on other machines.
 *
 * Package format (ZIP):
 * - model.pt              - Trained YOLO model weights
 * - config.json           - Training configuration and metadata
 * - model.engine          - Optional TensorRT engine (not portable across GPUs)
 *
 * Export filename format: {modelType}-{modelVariant}-{imgSize}px-{epochs}ep-{date}-{shortId}.zip
 * Example: detection-l-640px-100ep-20260115-a8f3c2.zip
 */

import { DetectionModelInfo, ModelInfo } from '../types';

/** Model type for export/import */
export type ModelType = 'detection' | 'keypoint';

/**
 * Model package configuration stored in config.json
 */
export interface ModelPackageConfig {
  /** Package format version */
  version: '1.0' | '1.1';
  /** Model type: 'detection' for bounding box, 'keypoint' for origin prediction */
  modelType: ModelType;
  /** Original model name */
  modelName: string;
  /** Unique model identifier */
  modelId: string;
  /** Training configuration */
  training: {
    /** Number of epochs trained */
    epochs: number;
    /** Input image size used for training */
    imgSize: number;
    /** Tile size if tiled training was used */
    tileSize?: number;
    /** Model size variant (n, s, m, l, x) */
    modelSize?: string;
  };
  /** Training metrics */
  metrics: {
    mAP50?: number;
    mAP50_95?: number;
    precision?: number;
    recall?: number;
    [key: string]: number | undefined;
  };
  /** Class names the model was trained on */
  classNames: string[];
  /** When the model was trained */
  trainingDate: string;
  /** When the package was exported */
  exportedAt: string;
  /** Application version that created this package */
  applicationVersion: string;
  /** Files included in the package */
  files: {
    model: string;
    engine?: string;
  };
  /** Number of model parameters (optional) */
  parameters?: number;
}

/**
 * Result of parsing a model package for import preview
 */
export interface ModelPackagePreview {
  /** Whether the package is valid */
  valid: boolean;
  /** Error message if invalid */
  error?: string;
  /** Package configuration if valid */
  config?: ModelPackageConfig;
  /** Whether the package includes a TensorRT engine */
  hasEngine?: boolean;
}

/**
 * Create a model package configuration from detection model info
 */
export function createModelPackageConfig(
  model: DetectionModelInfo,
  options?: {
    tileSize?: number;
    modelSize?: string;
    classNames?: string[];
  }
): ModelPackageConfig {
  return {
    version: '1.1',
    modelType: 'detection',
    modelName: model.name,
    modelId: model.id,
    training: {
      epochs: model.epochsTrained,
      imgSize: model.imgSize,
      tileSize: options?.tileSize,
      modelSize: model.modelVariant || options?.modelSize || 'n',
    },
    metrics: {
      ...model.metrics,
    },
    classNames: options?.classNames || ['follicle'],
    trainingDate: model.createdAt,
    exportedAt: new Date().toISOString(),
    applicationVersion: '1.0.0',
    files: {
      model: 'model.pt',
    },
    parameters: model.parameters,
  };
}

/**
 * Create a model package configuration from keypoint (origin) model info
 */
export function createKeypointModelPackageConfig(
  model: ModelInfo,
  options?: {
    modelSize?: string;
  }
): ModelPackageConfig {
  return {
    version: '1.1',
    modelType: 'keypoint',
    modelName: model.name,
    modelId: model.id,
    training: {
      epochs: model.epochsTrained,
      imgSize: model.imgSize,
      modelSize: model.modelVariant || options?.modelSize || 'n',
    },
    metrics: {
      ...model.metrics,
    },
    classNames: ['follicle'],
    trainingDate: model.createdAt,
    exportedAt: new Date().toISOString(),
    applicationVersion: '1.0.0',
    files: {
      model: 'model.pt',
    },
    parameters: model.parameters,
  };
}

/**
 * Generate a descriptive export filename for a model package.
 * Format: {modelType}-{modelVariant}-{imgSize}px-{epochs}ep-{date}-{shortId}.zip
 * Example: detection-l-640px-100ep-20260115-a8f3c2.zip
 */
export function generateExportFileName(
  model: DetectionModelInfo | ModelInfo,
  modelType: ModelType
): string {
  const variant = model.modelVariant || 'n';
  const imgSize = model.imgSize || 640;
  const epochs = model.epochsTrained || 0;
  const date = new Date(model.createdAt).toISOString().slice(0, 10).replace(/-/g, '');
  const shortId = model.id.slice(0, 6);

  return `${modelType}-${variant}-${imgSize}px-${epochs}ep-${date}-${shortId}.zip`;
}

/**
 * Validate a model package configuration
 * Supports both v1.0 (legacy) and v1.1 (with modelType) formats
 */
export function validateModelPackageConfig(config: unknown): config is ModelPackageConfig {
  if (!config || typeof config !== 'object') return false;

  const c = config as Record<string, unknown>;

  // Check version - support both 1.0 and 1.1
  if (c.version !== '1.0' && c.version !== '1.1') return false;

  // v1.1 requires modelType, v1.0 doesn't (defaults to 'detection')
  if (c.version === '1.1' && c.modelType !== 'detection' && c.modelType !== 'keypoint') {
    return false;
  }

  if (typeof c.modelName !== 'string') return false;
  if (typeof c.modelId !== 'string') return false;
  if (!c.training || typeof c.training !== 'object') return false;
  if (!c.metrics || typeof c.metrics !== 'object') return false;
  if (!Array.isArray(c.classNames)) return false;
  if (typeof c.trainingDate !== 'string') return false;
  if (typeof c.exportedAt !== 'string') return false;
  if (!c.files || typeof c.files !== 'object') return false;

  const training = c.training as Record<string, unknown>;
  if (typeof training.epochs !== 'number') return false;
  if (typeof training.imgSize !== 'number') return false;

  const files = c.files as Record<string, unknown>;
  if (typeof files.model !== 'string') return false;

  return true;
}

/**
 * Get the model type from a config, handling legacy v1.0 packages
 */
export function getModelTypeFromConfig(config: ModelPackageConfig): ModelType {
  // v1.1 has explicit modelType
  if (config.version === '1.1' && config.modelType) {
    return config.modelType;
  }
  // v1.0 packages are all detection models (keypoint didn't exist)
  return 'detection';
}

/**
 * Format file size for display
 */
export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

/**
 * Format metrics for display
 */
export function formatMetrics(metrics: Record<string, number | undefined>): string {
  const parts: string[] = [];

  if (metrics.mAP50 !== undefined) {
    parts.push(`mAP50: ${(metrics.mAP50 * 100).toFixed(1)}%`);
  }
  if (metrics.precision !== undefined) {
    parts.push(`Precision: ${(metrics.precision * 100).toFixed(1)}%`);
  }
  if (metrics.recall !== undefined) {
    parts.push(`Recall: ${(metrics.recall * 100).toFixed(1)}%`);
  }

  return parts.join(', ') || 'No metrics available';
}
