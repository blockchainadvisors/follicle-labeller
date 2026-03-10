/**
 * Platform detection utilities
 * Determines whether the app is running in Electron or Web mode
 */

/**
 * Check if running in Electron environment
 * This checks for the presence of the electronAPI bridge
 */
export const isElectron = (): boolean => {
  return typeof window !== 'undefined' &&
         window.electronAPI !== undefined;
};

/**
 * Check if running in Web (browser) environment
 * This is the inverse of isElectron
 */
export const isWeb = (): boolean => {
  return !isElectron();
};

/**
 * Get the current platform name
 */
export const getPlatformName = (): 'electron' | 'web' => {
  return isElectron() ? 'electron' : 'web';
};
