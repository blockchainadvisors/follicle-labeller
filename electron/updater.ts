import { autoUpdater, UpdateInfo, ProgressInfo } from 'electron-updater';
import { BrowserWindow, dialog, app } from 'electron';

// Disable default logging, use console instead
autoUpdater.logger = null;

// Configure updater - don't auto download, let user decide
autoUpdater.autoDownload = false;
autoUpdater.autoInstallOnAppQuit = false;

let mainWindow: BrowserWindow | null = null;
let isCheckingForUpdate = false;
let updateAvailable = false;

export function initUpdater(window: BrowserWindow): void {
  mainWindow = window;

  // Update available - prompt user to download
  autoUpdater.on('update-available', (info: UpdateInfo) => {
    isCheckingForUpdate = false;
    updateAvailable = true;

    if (!mainWindow) return;

    dialog.showMessageBox(mainWindow, {
      type: 'info',
      title: 'Update Available',
      message: `A new version (${info.version}) is available.`,
      detail: `You are currently on version ${app.getVersion()}. Would you like to download the update now?`,
      buttons: ['Download', 'Later'],
      defaultId: 0,
      cancelId: 1,
    }).then((result) => {
      if (result.response === 0) {
        // User chose to download
        autoUpdater.downloadUpdate();
      }
    });
  });

  // No update available
  autoUpdater.on('update-not-available', () => {
    isCheckingForUpdate = false;
    updateAvailable = false;

    if (!mainWindow) return;

    dialog.showMessageBox(mainWindow, {
      type: 'info',
      title: 'No Updates Available',
      message: 'You are running the latest version.',
      detail: `Current version: ${app.getVersion()}`,
      buttons: ['OK'],
    });
  });

  // Download progress
  autoUpdater.on('download-progress', (progress: ProgressInfo) => {
    if (mainWindow) {
      // Update window progress bar
      mainWindow.setProgressBar(progress.percent / 100);

      // Send progress to renderer if needed
      mainWindow.webContents.send('update:downloadProgress', {
        percent: Math.round(progress.percent),
        transferred: progress.transferred,
        total: progress.total,
        bytesPerSecond: progress.bytesPerSecond,
      });
    }
  });

  // Update downloaded - prompt to install
  autoUpdater.on('update-downloaded', (info: UpdateInfo) => {
    if (mainWindow) {
      // Clear progress bar
      mainWindow.setProgressBar(-1);
    }

    if (!mainWindow) return;

    dialog.showMessageBox(mainWindow, {
      type: 'info',
      title: 'Update Ready',
      message: `Version ${info.version} has been downloaded.`,
      detail: 'The update will be installed when you restart the application. Restart now?',
      buttons: ['Restart Now', 'Later'],
      defaultId: 0,
      cancelId: 1,
    }).then((result) => {
      if (result.response === 0) {
        // User chose to restart
        autoUpdater.quitAndInstall(false, true);
      }
    });
  });

  // Error handling
  autoUpdater.on('error', (error: Error) => {
    isCheckingForUpdate = false;

    if (mainWindow) {
      mainWindow.setProgressBar(-1);
    }

    console.error('Update error:', error);

    if (!mainWindow) return;

    dialog.showMessageBox(mainWindow, {
      type: 'error',
      title: 'Update Error',
      message: 'Failed to check for updates.',
      detail: error.message || 'An unknown error occurred. Please try again later.',
      buttons: ['OK'],
    });
  });

  // Checking for update started
  autoUpdater.on('checking-for-update', () => {
    isCheckingForUpdate = true;
  });
}

export function checkForUpdates(): void {
  if (isCheckingForUpdate) {
    // Already checking
    return;
  }

  autoUpdater.checkForUpdates().catch((error) => {
    console.error('Failed to check for updates:', error);
  });
}

export function downloadUpdate(): void {
  if (updateAvailable) {
    autoUpdater.downloadUpdate();
  }
}

export function installUpdate(): void {
  autoUpdater.quitAndInstall(false, true);
}

export function isUpdateAvailable(): boolean {
  return updateAvailable;
}
