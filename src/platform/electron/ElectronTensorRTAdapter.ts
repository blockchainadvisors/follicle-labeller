/**
 * Electron TensorRT Adapter
 * Wraps window.electronAPI.tensorrt operations for Electron platform
 */

import type { TensorRTAdapter } from '../types';

export class ElectronTensorRTAdapter implements TensorRTAdapter {
  async check(): Promise<{ available: boolean; version: string | null; canInstall: boolean }> {
    return window.electronAPI.tensorrt.check();
  }

  async install(): Promise<{ success: boolean; error?: string }> {
    return window.electronAPI.tensorrt.install();
  }

  onInstallProgress(callback: (data: { message: string; percent?: number }) => void): () => void {
    return window.electronAPI.tensorrt.onInstallProgress(callback);
  }
}
