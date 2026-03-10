/**
 * Platform configuration
 * Environment-specific settings for Electron vs Web deployment
 */

import { isElectron } from './detect';

export type StorageMode = 'local' | 'server';

export interface PlatformConfig {
  /** Backend API URL for HTTP communication */
  backendUrl: string;
  /** Where to store projects - local filesystem or server */
  storageMode: StorageMode;
  /** Whether file dialogs are available (Electron only) */
  hasNativeDialogs: boolean;
  /** Whether menu listeners are available (Electron only) */
  hasMenuListeners: boolean;
}

/**
 * Get the backend URL from environment or default
 */
const getBackendUrl = (): string => {
  // Check for Vite environment variable
  if (typeof import.meta !== 'undefined' && import.meta.env?.VITE_BACKEND_URL) {
    return import.meta.env.VITE_BACKEND_URL;
  }
  // Default to localhost for development
  return 'http://127.0.0.1:5555';
};

/**
 * Platform configuration object
 * Automatically configured based on runtime environment
 */
export const config: PlatformConfig = {
  backendUrl: getBackendUrl(),
  storageMode: isElectron() ? 'local' : 'server',
  hasNativeDialogs: isElectron(),
  hasMenuListeners: isElectron(),
};

/**
 * Update backend URL at runtime (useful for testing)
 */
export const setBackendUrl = (url: string): void => {
  (config as { backendUrl: string }).backendUrl = url;
};
