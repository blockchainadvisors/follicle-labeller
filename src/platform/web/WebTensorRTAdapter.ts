/**
 * Web TensorRT Adapter
 * Uses HTTP to communicate with backend for Web platform
 */

import type { TensorRTAdapter } from '../types';
import { config } from '../config';

export class WebTensorRTAdapter implements TensorRTAdapter {
  async check(): Promise<{ available: boolean; version: string | null; canInstall: boolean }> {
    try {
      const response = await fetch(`${config.backendUrl}/yolo-keypoint/check-tensorrt`);
      if (!response.ok) {
        return { available: false, version: null, canInstall: false };
      }

      const data = await response.json();
      return {
        available: data.available,
        version: data.version,
        canInstall: false, // Web can't install - server manages this
      };
    } catch {
      return { available: false, version: null, canInstall: false };
    }
  }

  async install(): Promise<{ success: boolean; error?: string }> {
    // TensorRT is pre-installed on server
    return { success: false, error: 'TensorRT must be installed on the server' };
  }

  onInstallProgress(_callback: (data: { message: string; percent?: number }) => void): () => void {
    // No installation progress for web mode
    return () => {};
  }
}
