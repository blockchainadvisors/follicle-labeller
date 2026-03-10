/**
 * Electron GPU Adapter
 * Wraps window.electronAPI.gpu operations for Electron platform
 */

import type { GpuAdapter } from '../types';
import type { GPUHardwareInfo } from '../../types';

export class ElectronGpuAdapter implements GpuAdapter {
  async getHardwareInfo(): Promise<GPUHardwareInfo> {
    return window.electronAPI.gpu.getHardwareInfo();
  }

  async installPackages(): Promise<{ success: boolean; error?: string }> {
    return window.electronAPI.gpu.installPackages();
  }

  onInstallProgress(callback: (data: { message: string; percent?: number }) => void): () => void {
    return window.electronAPI.gpu.onInstallProgress(callback);
  }
}
