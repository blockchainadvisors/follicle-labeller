/**
 * Web Blob Adapter
 * Uses HTTP to communicate with BLOB server for Web platform
 */

import type { BlobAdapter } from '../types';
import type { GPUInfo } from '../../types';
import { config } from '../config';

export class WebBlobAdapter implements BlobAdapter {
  async startServer(): Promise<{ success: boolean; error?: string }> {
    // In web mode, server is already running on GPU server
    // Just check if it's available
    try {
      const available = await this.isAvailable();
      if (available) {
        return { success: true };
      }
      return { success: false, error: 'Server not available' };
    } catch (error) {
      return { success: false, error: String(error) };
    }
  }

  async stopServer(): Promise<{ success: boolean }> {
    // Web mode cannot stop the remote server
    console.warn('Web mode cannot stop the remote server');
    return { success: false };
  }

  async isAvailable(): Promise<boolean> {
    try {
      const response = await fetch(`${config.backendUrl}/health`, {
        method: 'GET',
        signal: AbortSignal.timeout(5000),
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  async checkPython(): Promise<{ available: boolean; version?: string; error?: string }> {
    // In web mode, Python is on the server - just check health
    try {
      const response = await fetch(`${config.backendUrl}/health`);
      if (response.ok) {
        return { available: true, version: 'remote' };
      }
      return { available: false, error: 'Server not responding' };
    } catch (error) {
      return { available: false, error: String(error) };
    }
  }

  async getServerInfo(): Promise<{ port: number; running: boolean; scriptPath: string }> {
    // Extract port from config URL
    const url = new URL(config.backendUrl);
    const port = parseInt(url.port) || 5555;

    const running = await this.isAvailable();

    return {
      port,
      running,
      scriptPath: 'remote', // Web mode uses remote server
    };
  }

  async getSetupStatus(): Promise<string> {
    const available = await this.isAvailable();
    return available ? 'ready' : 'unavailable';
  }

  onSetupProgress(_callback: (status: string) => void): () => void {
    // Web mode doesn't have setup progress - server is already set up
    return () => {};
  }

  async getGPUInfo(): Promise<GPUInfo> {
    try {
      const response = await fetch(`${config.backendUrl}/gpu-info`);
      if (!response.ok) {
        throw new Error('Failed to get GPU info');
      }
      return response.json();
    } catch {
      return {
        activeBackend: 'cpu',
        deviceName: 'Unknown (remote)',
        available: { cuda: false, mps: false },
      };
    }
  }

  async restartServer(): Promise<{ success: boolean; error?: string }> {
    // Web mode cannot restart the remote server
    console.warn('Web mode cannot restart the remote server');
    return { success: false, error: 'Cannot restart remote server from web client' };
  }
}
