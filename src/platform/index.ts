/**
 * Platform abstraction layer
 * Provides unified API for Electron and Web platforms
 */

import type { Platform } from './types';
import { isElectron, isWeb, getPlatformName } from './detect';
import { config } from './config';

// Lazy-loaded platform instance
let platformInstance: Platform | null = null;

/**
 * Get the platform adapter for the current environment
 * Automatically detects Electron vs Web and returns appropriate implementation
 */
export function getPlatform(): Platform {
  if (platformInstance) {
    return platformInstance;
  }

  if (isElectron()) {
    // Dynamic import to avoid loading Electron code in web builds
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { ElectronPlatform } = require('./electron');
    platformInstance = new ElectronPlatform();
  } else {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { WebPlatform } = require('./web');
    platformInstance = new WebPlatform();
  }

  return platformInstance!;
}

/**
 * Reset platform instance (for testing)
 */
export function resetPlatform(): void {
  platformInstance = null;
}

// Re-export detection utilities and config
export { isElectron, isWeb, getPlatformName, config };

// Re-export types
export type {
  Platform,
  FileAdapter,
  BlobAdapter,
  YoloKeypointAdapter,
  YoloDetectionAdapter,
  GpuAdapter,
  TensorRTAdapter,
  ModelAdapter,
  MenuAdapter,
  OpenImageResult,
  OpenFileOptions,
  SaveProjectResult,
  LoadProjectResult,
  ProjectImageData,
  ServerProject,
  ServerProjectList,
} from './types';
