/**
 * Web GPU Adapter
 * Uses HTTP to communicate with backend for Web platform
 */

import type { GpuAdapter } from '../types';
import type { GPUHardwareInfo } from '../../types';
import { config } from '../config';

export class WebGpuAdapter implements GpuAdapter {
  async getHardwareInfo(): Promise<GPUHardwareInfo> {
    try {
      const response = await fetch(`${config.backendUrl}/gpu-info`);
      if (!response.ok) {
        throw new Error('Failed to get GPU info');
      }

      const gpuInfo = await response.json();

      // Convert GPUInfo to GPUHardwareInfo format
      return {
        hardware: {
          nvidia: {
            found: gpuInfo.available?.cuda || false,
            name: gpuInfo.deviceName,
          },
          apple_silicon: {
            found: gpuInfo.available?.mps || false,
          },
        },
        packages: {
          cupy: gpuInfo.activeBackend === 'cuda',
          torch: true, // Server has torch installed
        },
        canEnableGpu: gpuInfo.available?.cuda || gpuInfo.available?.mps || false,
        gpuEnabled: gpuInfo.activeBackend !== 'cpu',
      };
    } catch {
      return {
        hardware: {
          nvidia: { found: false },
          apple_silicon: { found: false },
        },
        packages: { cupy: false, torch: false },
        canEnableGpu: false,
        gpuEnabled: false,
      };
    }
  }

  async installPackages(): Promise<{ success: boolean; error?: string }> {
    // Packages are pre-installed on server
    return { success: true };
  }

  onInstallProgress(_callback: (data: { message: string; percent?: number }) => void): () => void {
    // No installation progress for web mode
    return () => {};
  }
}
