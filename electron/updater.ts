import { autoUpdater, UpdateInfo, ProgressInfo } from 'electron-updater';
import { BrowserWindow, dialog, app, Notification } from 'electron';

// Disable default logging, use console instead
autoUpdater.logger = null;

// Configure updater for smooth experience:
// - Auto-download updates in background (no prompt)
// - Auto-install when user quits the app
autoUpdater.autoDownload = true;
autoUpdater.autoInstallOnAppQuit = true;

let mainWindow: BrowserWindow | null = null;
let isCheckingForUpdate = false;
let updateAvailable = false;
let updateDownloaded = false;
let downloadedVersion: string | null = null;
let isSilentCheck = false;  // For startup check - don't show "no updates" message
let isManualCheck = false;  // For manual "Check for Updates" menu item

export function initUpdater(window: BrowserWindow): void {
  mainWindow = window;

  // Update available - download silently in background
  autoUpdater.on('update-available', (info: UpdateInfo) => {
    isCheckingForUpdate = false;
    updateAvailable = true;
    console.log(`Update available: ${info.version} (downloading in background...)`);

    // For manual checks, show a brief notification that download started
    if (isManualCheck && Notification.isSupported()) {
      new Notification({
        title: 'Update Available',
        body: `Version ${info.version} is downloading in the background.`,
        silent: true,
      }).show();
    }
    isManualCheck = false;
  });

  // No update available
  autoUpdater.on('update-not-available', () => {
    isCheckingForUpdate = false;
    updateAvailable = false;
    console.log('No updates available');

    // Only show dialog for manual "Check for Updates" menu action
    if (isManualCheck && mainWindow) {
      dialog.showMessageBox(mainWindow, {
        type: 'info',
        title: 'No Updates Available',
        message: 'You are running the latest version.',
        detail: `Current version: ${app.getVersion()}`,
        buttons: ['OK'],
      });
    }
    isManualCheck = false;
    isSilentCheck = false;
  });

  // Download progress - just log, no UI interruption
  autoUpdater.on('download-progress', (progress: ProgressInfo) => {
    console.log(`Update download progress: ${Math.round(progress.percent)}%`);
    // Optionally show subtle progress in dock/taskbar
    if (mainWindow && progress.percent < 100) {
      mainWindow.setProgressBar(progress.percent / 100);
    }
  });

  // Update downloaded - show non-intrusive notification
  // Update will auto-install on quit (no action required from user)
  autoUpdater.on('update-downloaded', (info: UpdateInfo) => {
    updateDownloaded = true;
    downloadedVersion = info.version;
    console.log(`Update ${info.version} downloaded and ready to install on quit`);

    if (mainWindow) {
      // Clear progress bar
      mainWindow.setProgressBar(-1);
    }

    // Show a subtle notification (not a blocking dialog)
    if (Notification.isSupported()) {
      const notification = new Notification({
        title: 'Update Ready',
        body: `Version ${info.version} will be installed when you quit the app.`,
        silent: true,
      });
      notification.show();
    }

    // Notify renderer (for optional UI indicator)
    if (mainWindow) {
      mainWindow.webContents.send('update:ready', { version: info.version });
    }
  });

  // Error handling - only show dialog for manual checks
  autoUpdater.on('error', (error: Error) => {
    isCheckingForUpdate = false;
    console.error('Update error:', error);

    if (mainWindow) {
      mainWindow.setProgressBar(-1);
    }

    // Only show error dialog for manual "Check for Updates" action
    if (isManualCheck && mainWindow) {
      dialog.showMessageBox(mainWindow, {
        type: 'error',
        title: 'Update Error',
        message: 'Failed to check for updates.',
        detail: error.message || 'An unknown error occurred. Please try again later.',
        buttons: ['OK'],
      });
    }
    isManualCheck = false;
    isSilentCheck = false;
  });

  // Checking for update started
  autoUpdater.on('checking-for-update', () => {
    isCheckingForUpdate = true;
  });

  // Check for updates on startup (silent - only notify if update available)
  setTimeout(() => {
    checkForUpdatesOnStartup();
  }, 3000);  // Wait 3 seconds for app to fully load
}

// Manual check from menu - will show dialogs for results
export function checkForUpdates(): void {
  if (isCheckingForUpdate) {
    return;
  }

  console.log('Manual update check triggered');
  isManualCheck = true;
  isSilentCheck = false;
  autoUpdater.checkForUpdates().catch((error) => {
    console.error('Failed to check for updates:', error);
    isManualCheck = false;
  });
}

// Silent check on startup - no dialogs, just downloads if available
function checkForUpdatesOnStartup(): void {
  if (isCheckingForUpdate) {
    return;
  }

  console.log('Checking for updates on startup (silent)...');
  isSilentCheck = true;
  isManualCheck = false;
  autoUpdater.checkForUpdates().catch((error) => {
    console.error('Failed to check for updates on startup:', error);
    isSilentCheck = false;
  });
}

// Force install now (if user wants to restart immediately)
export function installUpdateNow(): void {
  if (updateDownloaded) {
    autoUpdater.quitAndInstall(false, true);
  }
}

export function isUpdateAvailable(): boolean {
  return updateAvailable;
}

export function isUpdateDownloaded(): boolean {
  return updateDownloaded;
}

export function getDownloadedVersion(): string | null {
  return downloadedVersion;
}
