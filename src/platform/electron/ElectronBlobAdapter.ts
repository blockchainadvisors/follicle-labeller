/**
 * Electron Blob Adapter
 * Wraps window.electronAPI.blob operations for Electron platform
 */

import type { BlobAdapter } from '../types';
import type { GPUInfo } from '../../types';

export class ElectronBlobAdapter implements BlobAdapter {
  async startServer(): Promise<{ success: boolean; error?: string; errorDetails?: string }> {
    return window.electronAPI.blob.startServer();
  }

  async stopServer(): Promise<{ success: boolean }> {
    return window.electronAPI.blob.stopServer();
  }

  async isAvailable(): Promise<boolean> {
    return window.electronAPI.blob.isAvailable();
  }

  async checkPython(): Promise<{ available: boolean; version?: string; error?: string }> {
    return window.electronAPI.blob.checkPython();
  }

  async getServerInfo(): Promise<{ port: number; running: boolean; scriptPath: string }> {
    return window.electronAPI.blob.getServerInfo();
  }

  async getSetupStatus(): Promise<string> {
    return window.electronAPI.blob.getSetupStatus();
  }

  onSetupProgress(callback: (status: string, percent?: number) => void): () => void {
    return window.electronAPI.blob.onSetupProgress(callback);
  }

  async getGPUInfo(): Promise<GPUInfo> {
    return window.electronAPI.blob.getGPUInfo();
  }

  async restartServer(): Promise<{ success: boolean; error?: string; errorDetails?: string }> {
    return window.electronAPI.blob.restartServer();
  }
}
