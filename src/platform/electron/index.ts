/**
 * Electron Platform
 * Combines all Electron adapters into a single Platform implementation
 */

import type { Platform } from '../types';
import { ElectronFileAdapter } from './ElectronFileAdapter';
import { ElectronBlobAdapter } from './ElectronBlobAdapter';
import { ElectronYoloKeypointAdapter } from './ElectronYoloKeypointAdapter';
import { ElectronYoloDetectionAdapter } from './ElectronYoloDetectionAdapter';
import { ElectronGpuAdapter } from './ElectronGpuAdapter';
import { ElectronTensorRTAdapter } from './ElectronTensorRTAdapter';
import { ElectronModelAdapter } from './ElectronModelAdapter';
import { ElectronMenuAdapter } from './ElectronMenuAdapter';

/**
 * Electron platform implementation
 * Uses window.electronAPI IPC bridge for all operations
 */
export class ElectronPlatform implements Platform {
  readonly name = 'electron' as const;
  readonly file = new ElectronFileAdapter();
  readonly blob = new ElectronBlobAdapter();
  readonly yoloKeypoint = new ElectronYoloKeypointAdapter();
  readonly yoloDetection = new ElectronYoloDetectionAdapter();
  readonly gpu = new ElectronGpuAdapter();
  readonly tensorrt = new ElectronTensorRTAdapter();
  readonly model = new ElectronModelAdapter();
  readonly menu = new ElectronMenuAdapter();
}

// Export individual adapters for testing
export {
  ElectronFileAdapter,
  ElectronBlobAdapter,
  ElectronYoloKeypointAdapter,
  ElectronYoloDetectionAdapter,
  ElectronGpuAdapter,
  ElectronTensorRTAdapter,
  ElectronModelAdapter,
  ElectronMenuAdapter,
};
