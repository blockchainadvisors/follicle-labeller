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
 */

import { DetectionModelInfo } from '../types';

/**
 * Model package configuration stored in config.json
 */
export interface ModelPackageConfig {
  /** Package format version */
  version: '1.0';
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
    /** Model size variant (n, s, m, l) */
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
 * Create a model package configuration from model info
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
    version: '1.0',
    modelName: model.name,
    modelId: model.id,
    training: {
      epochs: model.epochsTrained,
      imgSize: model.imgSize,
      tileSize: options?.tileSize,
      modelSize: options?.modelSize,
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
  };
}

/**
 * Validate a model package configuration
 */
export function validateModelPackageConfig(config: unknown): config is ModelPackageConfig {
  if (!config || typeof config !== 'object') return false;

  const c = config as Record<string, unknown>;

  // Check required fields
  if (c.version !== '1.0') return false;
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
