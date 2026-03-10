/**
 * Web Platform
 * Combines all Web adapters into a single Platform implementation
 */

import type { Platform } from '../types';
import { WebFileAdapter } from './WebFileAdapter';
import { WebBlobAdapter } from './WebBlobAdapter';
import { WebYoloKeypointAdapter } from './WebYoloKeypointAdapter';
import { WebYoloDetectionAdapter } from './WebYoloDetectionAdapter';
import { WebGpuAdapter } from './WebGpuAdapter';
import { WebTensorRTAdapter } from './WebTensorRTAdapter';
import { WebModelAdapter } from './WebModelAdapter';
import { WebMenuAdapter } from './WebMenuAdapter';

/**
 * Web platform implementation
 * Uses HTTP to communicate directly with FastAPI backend
 */
export class WebPlatform implements Platform {
  readonly name = 'web' as const;
  readonly file = new WebFileAdapter();
  readonly blob = new WebBlobAdapter();
  readonly yoloKeypoint = new WebYoloKeypointAdapter();
  readonly yoloDetection = new WebYoloDetectionAdapter();
  readonly gpu = new WebGpuAdapter();
  readonly tensorrt = new WebTensorRTAdapter();
  readonly model = new WebModelAdapter();
  readonly menu = new WebMenuAdapter();
}

// Export individual adapters for testing
export {
  WebFileAdapter,
  WebBlobAdapter,
  WebYoloKeypointAdapter,
  WebYoloDetectionAdapter,
  WebGpuAdapter,
  WebTensorRTAdapter,
  WebModelAdapter,
  WebMenuAdapter,
};
