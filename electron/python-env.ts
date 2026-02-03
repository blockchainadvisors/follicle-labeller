/**
 * Manages Python virtual environment and dependencies
 * - Downloads portable Python on first run (python-build-standalone)
 * - Creates venv in app data directory
 * - Installs requirements.txt automatically
 * - Provides paths to venv Python executable
 */

import { app } from "electron";
import path from "path";
import fs from "fs";
import https from "https";
import { spawn, execSync, execFile } from "child_process";
import * as tar from "tar";

// Track setup status for UI feedback
let currentSetupStatus = "";

// Progress callback for Python download
let downloadProgressCallback: ((status: string, percent?: number) => void) | null = null;

/**
 * Set the download progress callback (called from main.ts)
 */
export function setDownloadProgressCallback(callback: ((status: string, percent?: number) => void) | null): void {
  downloadProgressCallback = callback;
}

// ============================================
// Python Runtime Download (python-build-standalone)
// ============================================

// Python version and build info
// Using Python 3.11 for broad compatibility (some packages don't support 3.12+ yet)
const PYTHON_VERSION = "3.11.9";
const PYTHON_BUILD_VERSION = "20240726";

// Download URLs for different platforms
// From: https://github.com/indygreg/python-build-standalone/releases
const PYTHON_DOWNLOAD_URLS: Record<string, string> = {
  "win32-x64": `https://github.com/indygreg/python-build-standalone/releases/download/${PYTHON_BUILD_VERSION}/cpython-${PYTHON_VERSION}+${PYTHON_BUILD_VERSION}-x86_64-pc-windows-msvc-install_only.tar.gz`,
  "darwin-x64": `https://github.com/indygreg/python-build-standalone/releases/download/${PYTHON_BUILD_VERSION}/cpython-${PYTHON_VERSION}+${PYTHON_BUILD_VERSION}-x86_64-apple-darwin-install_only.tar.gz`,
  "darwin-arm64": `https://github.com/indygreg/python-build-standalone/releases/download/${PYTHON_BUILD_VERSION}/cpython-${PYTHON_VERSION}+${PYTHON_BUILD_VERSION}-aarch64-apple-darwin-install_only.tar.gz`,
  "linux-x64": `https://github.com/indygreg/python-build-standalone/releases/download/${PYTHON_BUILD_VERSION}/cpython-${PYTHON_VERSION}+${PYTHON_BUILD_VERSION}-x86_64-unknown-linux-gnu-install_only.tar.gz`,
};

/**
 * Get the path to the downloaded Python runtime directory.
 * Location: {userData}/python-runtime/
 */
export function getPythonRuntimeDir(): string {
  return path.join(app.getPath("userData"), "python-runtime");
}

/**
 * Get the path to the downloaded Python executable.
 * Returns null if Python hasn't been downloaded yet.
 */
export function getDownloadedPythonPath(): string | null {
  const runtimeDir = getPythonRuntimeDir();

  // python-build-standalone extracts to a 'python' subdirectory
  const pythonExe = process.platform === "win32"
    ? path.join(runtimeDir, "python", "python.exe")
    : path.join(runtimeDir, "python", "bin", "python3");

  if (fs.existsSync(pythonExe)) {
    return pythonExe;
  }
  return null;
}

/**
 * Check if we have a downloaded Python runtime available.
 */
export function hasDownloadedPython(): boolean {
  return getDownloadedPythonPath() !== null;
}

/**
 * Download a file with progress reporting.
 * Handles HTTP redirects (GitHub releases use redirects).
 */
async function downloadFile(
  url: string,
  destPath: string,
  onProgress: (percent: number) => void,
  redirectCount = 0
): Promise<void> {
  // Prevent infinite redirects
  if (redirectCount > 5) {
    throw new Error("Too many redirects");
  }

  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(destPath);

    const request = https.get(url, (response) => {
      // Handle redirects (301, 302, 307, 308)
      if (response.statusCode && response.statusCode >= 300 && response.statusCode < 400 && response.headers.location) {
        file.close();
        fs.unlinkSync(destPath); // Remove the empty file
        downloadFile(response.headers.location, destPath, onProgress, redirectCount + 1)
          .then(resolve)
          .catch(reject);
        return;
      }

      if (response.statusCode !== 200) {
        file.close();
        fs.unlinkSync(destPath);
        reject(new Error(`HTTP ${response.statusCode}: ${response.statusMessage}`));
        return;
      }

      const totalSize = parseInt(response.headers["content-length"] || "0", 10);
      let downloadedSize = 0;
      let lastReportedPercent = -1;

      response.on("data", (chunk: Buffer) => {
        downloadedSize += chunk.length;
        if (totalSize > 0) {
          const percent = Math.round((downloadedSize / totalSize) * 100);
          // Only report when percent changes to avoid too many updates
          if (percent !== lastReportedPercent) {
            lastReportedPercent = percent;
            onProgress(percent);
          }
        }
      });

      response.pipe(file);

      file.on("finish", () => {
        file.close();
        resolve();
      });

      file.on("error", (err) => {
        file.close();
        fs.unlinkSync(destPath);
        reject(err);
      });
    });

    request.on("error", (err) => {
      file.close();
      if (fs.existsSync(destPath)) {
        fs.unlinkSync(destPath);
      }
      reject(err);
    });

    // Set a timeout for the request
    request.setTimeout(300000, () => { // 5 minute timeout
      request.destroy();
      reject(new Error("Download timed out"));
    });
  });
}

/**
 * Extract a .tar.gz file to a directory.
 */
async function extractTarGz(tarPath: string, destDir: string): Promise<void> {
  await tar.extract({
    file: tarPath,
    cwd: destDir,
  });
}

/**
 * Download the Python runtime for the current platform.
 * Downloads from python-build-standalone and extracts to userData.
 */
export async function downloadPythonRuntime(
  onProgress?: (status: string, percent?: number) => void
): Promise<string> {
  const platform = `${process.platform}-${process.arch}`;
  const url = PYTHON_DOWNLOAD_URLS[platform];

  if (!url) {
    throw new Error(`No Python build available for platform: ${platform}. Supported: ${Object.keys(PYTHON_DOWNLOAD_URLS).join(", ")}`);
  }

  const runtimeDir = getPythonRuntimeDir();
  const tarPath = path.join(runtimeDir, "python.tar.gz");

  // Create runtime directory
  fs.mkdirSync(runtimeDir, { recursive: true });

  // Check if already downloaded
  const existingPython = getDownloadedPythonPath();
  if (existingPython) {
    console.log("[Python Download] Already have Python runtime at:", existingPython);
    return existingPython;
  }

  console.log("[Python Download] Downloading Python runtime...");
  console.log("[Python Download] URL:", url);
  console.log("[Python Download] Destination:", runtimeDir);

  try {
    // Download with progress
    onProgress?.("Downloading Python runtime...", 0);
    setSetupStatus("Downloading Python runtime...");

    await downloadFile(url, tarPath, (percent) => {
      const status = `Downloading Python runtime... ${percent}%`;
      setSetupStatus(status);
      onProgress?.(status, percent);
      downloadProgressCallback?.(status, percent);
    });

    console.log("[Python Download] Download complete, extracting...");
    onProgress?.("Extracting Python runtime...", 100);
    setSetupStatus("Extracting Python runtime...");
    downloadProgressCallback?.("Extracting Python runtime...", 100);

    // Extract tar.gz
    await extractTarGz(tarPath, runtimeDir);

    // Cleanup tar file
    fs.unlinkSync(tarPath);

    // Verify extraction
    const pythonPath = getDownloadedPythonPath();
    if (!pythonPath) {
      throw new Error("Python extraction failed - executable not found");
    }

    // On Unix, ensure python executable is... executable
    if (process.platform !== "win32") {
      fs.chmodSync(pythonPath, 0o755);
      // Also make pip executable
      const pipPath = path.join(path.dirname(pythonPath), "pip3");
      if (fs.existsSync(pipPath)) {
        fs.chmodSync(pipPath, 0o755);
      }
    }

    console.log("[Python Download] Python runtime ready at:", pythonPath);
    onProgress?.("Python runtime ready", 100);
    setSetupStatus("Python runtime ready");
    downloadProgressCallback?.("Python runtime ready", 100);

    return pythonPath;
  } catch (error) {
    // Cleanup on failure
    if (fs.existsSync(tarPath)) {
      fs.unlinkSync(tarPath);
    }
    throw error;
  }
}

/**
 * Get the path to the Python virtual environment.
 * Location:
 *   Windows: %APPDATA%/follicle-labeller/python-venv
 *   macOS: ~/Library/Application Support/follicle-labeller/python-venv
 *   Linux: ~/.config/follicle-labeller/python-venv
 */
export function getVenvPath(): string {
  return path.join(app.getPath("userData"), "python-venv");
}

/**
 * Get the path to the Python executable within the venv.
 */
export function getVenvPythonPath(): string {
  const venvPath = getVenvPath();
  if (process.platform === "win32") {
    return path.join(venvPath, "Scripts", "python.exe");
  }
  return path.join(venvPath, "bin", "python");
}

/**
 * Get the path to pip within the venv.
 */
export function getVenvPipPath(): string {
  const venvPath = getVenvPath();
  if (process.platform === "win32") {
    return path.join(venvPath, "Scripts", "pip.exe");
  }
  return path.join(venvPath, "bin", "pip");
}

/**
 * Get the current setup status message.
 */
export function getSetupStatus(): string {
  return currentSetupStatus;
}

/**
 * Set the current setup status (for internal use).
 */
function setSetupStatus(status: string): void {
  currentSetupStatus = status;
  console.log(`[Python Env] ${status}`);
}

/**
 * Find Python executable.
 * Priority: 1) Downloaded Python, 2) System Python in PATH
 */
async function findPython(): Promise<{
  found: boolean;
  pythonPath?: string;
  version?: string;
  source?: "downloaded" | "system";
  error?: string;
}> {
  // First, check for downloaded Python runtime
  const downloadedPython = getDownloadedPythonPath();
  if (downloadedPython) {
    try {
      const result = execSync(`"${downloadedPython}" --version`, {
        timeout: 10000,
        encoding: "utf-8",
        stdio: ["pipe", "pipe", "pipe"],
      });
      const version = result.trim();
      console.log("[Python Env] Using downloaded Python:", downloadedPython, version);
      return { found: true, pythonPath: downloadedPython, version, source: "downloaded" };
    } catch (err) {
      console.warn("[Python Env] Downloaded Python found but failed to run:", err);
      // Fall through to system Python
    }
  }

  // Try system Python commands
  const pythonCmds = process.platform === "win32"
    ? ["python", "python3", "py"]
    : ["python3", "python"];

  for (const cmd of pythonCmds) {
    try {
      const result = execSync(`${cmd} --version`, {
        timeout: 10000,
        encoding: "utf-8",
        stdio: ["pipe", "pipe", "pipe"],
      });

      const version = result.trim();
      // Verify it's Python 3.8+
      const match = version.match(/Python (\d+)\.(\d+)/);
      if (match) {
        const major = parseInt(match[1]);
        const minor = parseInt(match[2]);
        if (major >= 3 && minor >= 8) {
          console.log("[Python Env] Using system Python:", cmd, version);
          return { found: true, pythonPath: cmd, version, source: "system" };
        }
      }
    } catch {
      // Try next command
    }
  }

  return {
    found: false,
    error: "Python 3.8+ not found"
  };
}

/**
 * Find Python executable in system PATH only (legacy function for compatibility).
 */
async function findSystemPython(): Promise<{
  found: boolean;
  pythonPath?: string;
  version?: string;
  error?: string;
}> {
  const result = await findPython();
  return {
    found: result.found,
    pythonPath: result.pythonPath,
    version: result.version,
    error: result.error,
  };
}

/**
 * Check if the virtual environment exists and is valid.
 */
export function isVenvValid(): boolean {
  const pythonPath = getVenvPythonPath();
  return fs.existsSync(pythonPath);
}

/**
 * Ensure the virtual environment exists.
 * Creates it if it doesn't exist.
 * Will download Python automatically if not available.
 */
export async function ensureVenv(
  onProgress?: (status: string, percent?: number) => void
): Promise<{
  success: boolean;
  error?: string;
}> {
  const venvPath = getVenvPath();
  const venvPythonPath = getVenvPythonPath();

  // Check if venv already exists and is valid
  if (fs.existsSync(venvPythonPath)) {
    setSetupStatus("Virtual environment ready");
    return { success: true };
  }

  setSetupStatus("Finding Python...");
  onProgress?.("Finding Python...");

  // Try to find Python (downloaded or system)
  let pythonCheck = await findPython();

  // If no Python found, download it
  if (!pythonCheck.found) {
    setSetupStatus("Python not found, downloading...");
    onProgress?.("Python not found, downloading...", 0);

    try {
      const downloadedPath = await downloadPythonRuntime(onProgress);
      pythonCheck = {
        found: true,
        pythonPath: downloadedPath,
        version: `Python ${PYTHON_VERSION}`,
        source: "downloaded",
      };
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error);
      return {
        success: false,
        error: `Failed to download Python: ${errorMsg}`,
      };
    }
  }

  if (!pythonCheck.pythonPath) {
    return { success: false, error: "No Python available" };
  }

  setSetupStatus(`Creating virtual environment (${pythonCheck.version})...`);
  onProgress?.(`Creating virtual environment (${pythonCheck.version})...`);

  // Ensure parent directory exists
  const parentDir = path.dirname(venvPath);
  if (!fs.existsSync(parentDir)) {
    fs.mkdirSync(parentDir, { recursive: true });
  }

  // Create venv using the found Python
  return new Promise((resolve) => {
    const args = ["-m", "venv", venvPath];

    // For downloaded Python, use full path; for system Python, use command name
    const pythonCmd = pythonCheck.source === "downloaded"
      ? pythonCheck.pythonPath!
      : pythonCheck.pythonPath!;

    console.log("[Python Env] Creating venv with:", pythonCmd, args.join(" "));

    const proc = spawn(pythonCmd, args, {
      stdio: ["ignore", "pipe", "pipe"],
      timeout: 120000, // 2 minute timeout
    });

    let stderr = "";

    proc.stderr?.on("data", (data) => {
      stderr += data.toString();
    });

    proc.on("close", (code) => {
      if (code === 0 && fs.existsSync(venvPythonPath)) {
        setSetupStatus("Virtual environment created");
        onProgress?.("Virtual environment created");
        resolve({ success: true });
      } else {
        resolve({
          success: false,
          error: `Failed to create venv (exit code ${code}): ${stderr}`,
        });
      }
    });

    proc.on("error", (err) => {
      resolve({
        success: false,
        error: `Failed to create venv: ${err.message}`,
      });
    });
  });
}

/**
 * Install dependencies from requirements.txt into the venv.
 */
export async function installDependencies(
  requirementsPath: string,
  onProgress?: (message: string) => void
): Promise<{
  success: boolean;
  error?: string;
}> {
  const pipPath = getVenvPipPath();
  const pythonPath = getVenvPythonPath();

  if (!fs.existsSync(pipPath)) {
    return { success: false, error: "Virtual environment pip not found" };
  }

  if (!fs.existsSync(requirementsPath)) {
    return { success: false, error: `Requirements file not found: ${requirementsPath}` };
  }

  setSetupStatus("Installing dependencies...");
  onProgress?.("Installing dependencies...");

  return new Promise((resolve) => {
    // Use python -m pip for reliability
    const args = ["-m", "pip", "install", "-r", requirementsPath, "--quiet"];

    const proc = spawn(pythonPath, args, {
      stdio: ["ignore", "pipe", "pipe"],
      timeout: 600000, // 10 minute timeout for large packages
    });

    let stderr = "";
    let lastLine = "";

    proc.stdout?.on("data", (data) => {
      const text = data.toString();
      const lines = text.split("\n").filter((l: string) => l.trim());
      if (lines.length > 0) {
        lastLine = lines[lines.length - 1];
        // Extract package name from pip output
        const match = lastLine.match(/Installing.*?(\S+)/i) ||
                     lastLine.match(/Collecting\s+(\S+)/i);
        if (match) {
          const status = `Installing ${match[1]}...`;
          setSetupStatus(status);
          onProgress?.(status);
        }
      }
    });

    proc.stderr?.on("data", (data) => {
      stderr += data.toString();
    });

    proc.on("close", (code) => {
      if (code === 0) {
        setSetupStatus("Dependencies installed");
        onProgress?.("Dependencies installed");
        resolve({ success: true });
      } else {
        resolve({
          success: false,
          error: `pip install failed (exit code ${code}): ${stderr}`,
        });
      }
    });

    proc.on("error", (err) => {
      resolve({
        success: false,
        error: `pip install failed: ${err.message}`,
      });
    });
  });
}

/**
 * Check if required dependencies are installed in the venv.
 */
export async function checkDependencies(): Promise<{
  installed: boolean;
  missing: string[];
}> {
  const pythonPath = getVenvPythonPath();

  if (!fs.existsSync(pythonPath)) {
    return {
      installed: false,
      missing: ["opencv-python", "numpy", "pillow", "fastapi", "uvicorn"]
    };
  }

  // Note: sse-starlette, ultralytics, onnx, onnxruntime are installed on-demand
  // when the user wants to use YOLO training features (see installYOLODependencies)
  const checkCode = `
import sys
missing = []
try:
    import cv2
except ImportError:
    missing.append('opencv-python')
try:
    import numpy
except ImportError:
    missing.append('numpy')
try:
    from PIL import Image
except ImportError:
    missing.append('pillow')
try:
    import fastapi
except ImportError:
    missing.append('fastapi')
try:
    import uvicorn
except ImportError:
    missing.append('uvicorn')
print(','.join(missing) if missing else 'OK')
`;

  return new Promise((resolve) => {
    const proc = spawn(pythonPath, ["-c", checkCode], {
      stdio: ["ignore", "pipe", "pipe"],
      timeout: 30000,
    });

    let stdout = "";

    proc.stdout?.on("data", (data) => {
      stdout += data.toString();
    });

    proc.on("close", (code) => {
      if (code === 0) {
        const result = stdout.trim();
        if (result === "OK") {
          resolve({ installed: true, missing: [] });
        } else {
          resolve({
            installed: false,
            missing: result.split(",").filter(s => s)
          });
        }
      } else {
        resolve({
          installed: false,
          missing: ["opencv-python", "numpy", "pillow", "fastapi", "uvicorn"]
        });
      }
    });

    proc.on("error", () => {
      resolve({
        installed: false,
        missing: ["opencv-python", "numpy", "pillow", "fastapi", "uvicorn"]
      });
    });
  });
}

/**
 * Full environment setup: ensures venv exists and dependencies are installed.
 * This is the main entry point for setting up the Python environment.
 * Will download Python automatically if not available on the system.
 */
export async function setupPythonEnvironment(
  requirementsPath: string,
  onProgress?: (message: string, percent?: number) => void
): Promise<{
  success: boolean;
  pythonPath?: string;
  error?: string;
}> {
  // Step 1: Ensure venv exists (will download Python if needed)
  const venvResult = await ensureVenv(onProgress);
  if (!venvResult.success) {
    return { success: false, error: venvResult.error };
  }

  // Step 2: Check dependencies
  const depsCheck = await checkDependencies();
  if (!depsCheck.installed) {
    setSetupStatus(`Installing missing packages: ${depsCheck.missing.join(", ")}`);
    onProgress?.(`Installing missing packages: ${depsCheck.missing.join(", ")}`);

    // Step 3: Install dependencies if needed
    const installResult = await installDependencies(requirementsPath, onProgress);
    if (!installResult.success) {
      return { success: false, error: installResult.error };
    }
  }

  const pythonPath = getVenvPythonPath();
  setSetupStatus("Python environment ready");
  onProgress?.("Python environment ready");

  return { success: true, pythonPath };
}

/**
 * Get info about the current Python environment.
 */
export async function getPythonInfo(): Promise<{
  venvPath: string;
  pythonPath: string;
  venvExists: boolean;
  pythonVersion?: string;
}> {
  const venvPath = getVenvPath();
  const pythonPath = getVenvPythonPath();
  const venvExists = fs.existsSync(pythonPath);

  let pythonVersion: string | undefined;

  if (venvExists) {
    try {
      pythonVersion = execSync(`"${pythonPath}" --version`, {
        timeout: 5000,
        encoding: "utf-8",
      }).trim();
    } catch {
      // Ignore version check errors
    }
  }

  return {
    venvPath,
    pythonPath,
    venvExists,
    pythonVersion,
  };
}

/**
 * GPU Hardware Info type
 */
export interface GPUHardwareInfo {
  hardware: {
    nvidia: { found: boolean; name?: string; driver_version?: string };
    apple_silicon: { found: boolean; chip?: string };
  };
  packages: { cupy: boolean; torch: boolean };
  canEnableGpu: boolean;
  gpuEnabled: boolean;
}

/**
 * Get GPU hardware info by running detection script.
 * Works before GPU packages are installed.
 */
export async function getGPUHardwareInfo(): Promise<GPUHardwareInfo> {
  const pythonPath = getVenvPythonPath();

  // Get script path - different in packaged vs dev mode
  const scriptPath = app.isPackaged
    ? path.join(process.resourcesPath, "python", "gpu_hardware_detect.py")
    : path.join(__dirname, "..", "electron", "python", "gpu_hardware_detect.py");

  // Check if venv exists
  if (!fs.existsSync(pythonPath)) {
    return {
      hardware: { nvidia: { found: false }, apple_silicon: { found: false } },
      packages: { cupy: false, torch: false },
      canEnableGpu: false,
      gpuEnabled: false,
    };
  }

  return new Promise((resolve) => {
    execFile(pythonPath, [scriptPath], { timeout: 10000 }, (error, stdout) => {
      if (error || !stdout) {
        resolve({
          hardware: { nvidia: { found: false }, apple_silicon: { found: false } },
          packages: { cupy: false, torch: false },
          canEnableGpu: false,
          gpuEnabled: false,
        });
        return;
      }
      try {
        const result = JSON.parse(stdout);
        resolve({
          hardware: result.hardware,
          packages: result.packages,
          canEnableGpu: result.can_enable_gpu,
          gpuEnabled: result.gpu_enabled,
        });
      } catch {
        resolve({
          hardware: { nvidia: { found: false }, apple_silicon: { found: false } },
          packages: { cupy: false, torch: false },
          canEnableGpu: false,
          gpuEnabled: false,
        });
      }
    });
  });
}

/**
 * Install GPU packages based on platform.
 */
export async function installGPUPackages(
  onProgress?: (message: string, percent?: number) => void
): Promise<{ success: boolean; error?: string }> {
  const pythonPath = getVenvPythonPath();

  if (!fs.existsSync(pythonPath)) {
    return { success: false, error: "Python virtual environment not found" };
  }

  // Determine packages based on platform
  let packages: string[];
  if (process.platform === "darwin") {
    packages = ["torch", "kornia"];
    onProgress?.("Installing PyTorch for Metal acceleration (~2GB)...", 0);
  } else {
    packages = [
      "cupy-cuda12x",
      "nvidia-cuda-runtime-cu12",
      "nvidia-cublas-cu12",
      "nvidia-cuda-nvrtc-cu12",
    ];
    onProgress?.("Installing CuPy for CUDA acceleration (~1.5GB)...", 0);
  }

  return new Promise((resolve) => {
    const proc = spawn(pythonPath, ["-m", "pip", "install", ...packages], {
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stderr = "";

    proc.stdout?.on("data", (data) => {
      const line = data.toString();
      // Parse pip output for progress
      if (line.includes("Downloading") || line.includes("Installing") || line.includes("Collecting")) {
        onProgress?.(line.trim().substring(0, 100), undefined);
      }
    });

    proc.stderr?.on("data", (data) => {
      stderr += data.toString();
      const line = data.toString();
      // pip often writes progress to stderr
      if (line.includes("Downloading") || line.includes("Installing") || line.includes("Collecting")) {
        onProgress?.(line.trim().substring(0, 100), undefined);
      }
    });

    proc.on("close", (code) => {
      if (code === 0) {
        onProgress?.("Installation complete!", 100);
        resolve({ success: true });
      } else {
        resolve({ success: false, error: `Installation failed with code ${code}: ${stderr.substring(0, 500)}` });
      }
    });

    proc.on("error", (err) => {
      resolve({ success: false, error: `Failed to start pip: ${err.message}` });
    });
  });
}

/**
 * TensorRT Info type
 */
export interface TensorRTInfo {
  available: boolean;
  version: string | null;
  canInstall: boolean; // True if CUDA is available (required for TensorRT)
}

/**
 * Check if TensorRT is available.
 */
export async function checkTensorRT(): Promise<TensorRTInfo> {
  const pythonPath = getVenvPythonPath();

  if (!fs.existsSync(pythonPath)) {
    return { available: false, version: null, canInstall: false };
  }

  const checkCode = `
import sys
import json
result = {"available": False, "version": None, "cuda_available": False}
try:
    import torch
    result["cuda_available"] = torch.cuda.is_available()
except:
    pass
try:
    import tensorrt
    result["available"] = True
    result["version"] = tensorrt.__version__
except ImportError:
    pass
print(json.dumps(result))
`;

  return new Promise((resolve) => {
    const proc = spawn(pythonPath, ["-c", checkCode], {
      stdio: ["ignore", "pipe", "pipe"],
      timeout: 30000,
    });

    let stdout = "";

    proc.stdout?.on("data", (data) => {
      stdout += data.toString();
    });

    proc.on("close", (code) => {
      if (code === 0) {
        try {
          const result = JSON.parse(stdout.trim());
          resolve({
            available: result.available,
            version: result.version,
            canInstall: result.cuda_available,
          });
        } catch {
          resolve({ available: false, version: null, canInstall: false });
        }
      } else {
        resolve({ available: false, version: null, canInstall: false });
      }
    });

    proc.on("error", () => {
      resolve({ available: false, version: null, canInstall: false });
    });
  });
}

/**
 * Install TensorRT packages for GPU-optimized YOLO inference.
 * Requires CUDA to be available.
 */
export async function installTensorRT(
  onProgress?: (message: string, percent?: number) => void
): Promise<{ success: boolean; error?: string }> {
  const pythonPath = getVenvPythonPath();

  if (!fs.existsSync(pythonPath)) {
    return { success: false, error: "Python virtual environment not found" };
  }

  // Check if CUDA is available first
  const trtInfo = await checkTensorRT();
  if (!trtInfo.canInstall) {
    return {
      success: false,
      error: "CUDA is not available. TensorRT requires an NVIDIA GPU with CUDA support.",
    };
  }

  // TensorRT packages - tensorrt is the main package, tensorrt-cu12 provides CUDA 12 bindings
  // Note: tensorrt-cu12-bindings and tensorrt-cu12-libs are typically pulled as dependencies
  const packages = ["tensorrt>=10.0.0"];
  onProgress?.("Installing TensorRT for GPU-optimized inference (~500MB)...", 0);

  return new Promise((resolve) => {
    const proc = spawn(pythonPath, ["-m", "pip", "install", ...packages], {
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stderr = "";

    proc.stdout?.on("data", (data) => {
      const line = data.toString();
      if (line.includes("Downloading") || line.includes("Installing") || line.includes("Collecting")) {
        onProgress?.(line.trim().substring(0, 100), undefined);
      }
    });

    proc.stderr?.on("data", (data) => {
      stderr += data.toString();
      const line = data.toString();
      if (line.includes("Downloading") || line.includes("Installing") || line.includes("Collecting")) {
        onProgress?.(line.trim().substring(0, 100), undefined);
      }
    });

    proc.on("close", (code) => {
      if (code === 0) {
        onProgress?.("TensorRT installation complete!", 100);
        resolve({ success: true });
      } else {
        resolve({
          success: false,
          error: `Installation failed with code ${code}: ${stderr.substring(0, 500)}`,
        });
      }
    });

    proc.on("error", (err) => {
      resolve({ success: false, error: `Failed to start pip: ${err.message}` });
    });
  });
}

/**
 * YOLO Dependencies Info type
 */
export interface YOLODependenciesInfo {
  installed: boolean;
  missing: string[];
  estimatedSize: string;
}

/**
 * Check if YOLO training dependencies are installed.
 * These are large packages installed on-demand when the user wants to use training.
 */
export async function checkYOLODependencies(): Promise<YOLODependenciesInfo> {
  const pythonPath = getVenvPythonPath();

  if (!fs.existsSync(pythonPath)) {
    return {
      installed: false,
      missing: ["ultralytics", "sse-starlette", "onnx", "onnxruntime"],
      estimatedSize: "~2GB"
    };
  }

  const checkCode = `
import sys
missing = []
try:
    import ultralytics
except ImportError:
    missing.append('ultralytics')
try:
    from sse_starlette.sse import EventSourceResponse
except ImportError:
    missing.append('sse-starlette')
try:
    import onnx
except ImportError:
    missing.append('onnx')
try:
    import onnxruntime
except ImportError:
    missing.append('onnxruntime')
print(','.join(missing) if missing else 'OK')
`;

  return new Promise((resolve) => {
    const proc = spawn(pythonPath, ["-c", checkCode], {
      stdio: ["ignore", "pipe", "pipe"],
      timeout: 30000,
    });

    let stdout = "";

    proc.stdout?.on("data", (data) => {
      stdout += data.toString();
    });

    proc.on("close", (code) => {
      if (code === 0) {
        const result = stdout.trim();
        if (result === "OK") {
          resolve({ installed: true, missing: [], estimatedSize: "0" });
        } else {
          const missing = result.split(",").filter(s => s);
          // Estimate size based on what's missing
          const hasUltralytics = missing.includes("ultralytics");
          const estimatedSize = hasUltralytics ? "~2GB" : "~50MB";
          resolve({ installed: false, missing, estimatedSize });
        }
      } else {
        resolve({
          installed: false,
          missing: ["ultralytics", "sse-starlette", "onnx", "onnxruntime"],
          estimatedSize: "~2GB"
        });
      }
    });

    proc.on("error", () => {
      resolve({
        installed: false,
        missing: ["ultralytics", "sse-starlette", "onnx", "onnxruntime"],
        estimatedSize: "~2GB"
      });
    });
  });
}

/**
 * Install YOLO training dependencies.
 * This installs ultralytics (YOLO), sse-starlette (progress streaming), onnx and onnxruntime.
 * If GPU hardware is detected, it installs PyTorch with CUDA/MPS support first.
 */
export async function installYOLODependencies(
  onProgress?: (message: string, percent?: number) => void
): Promise<{ success: boolean; error?: string }> {
  const pythonPath = getVenvPythonPath();

  if (!fs.existsSync(pythonPath)) {
    return { success: false, error: "Python virtual environment not found" };
  }

  // First, check for GPU hardware and install appropriate PyTorch version
  const gpuInfo = await getGPUHardwareInfo();

  if (gpuInfo.hardware.nvidia.found) {
    // Install PyTorch with CUDA support first (before ultralytics)
    // Python 3.13+ requires CUDA 12.4 wheels, earlier versions use CUDA 12.1
    onProgress?.("Installing PyTorch with CUDA support (~2.5GB)...", 0);

    // Detect Python minor version to choose correct CUDA version
    let pythonMinor = 12; // Default to 3.12
    try {
      const versionOutput = execSync(`"${pythonPath}" -c "import sys; print(sys.version_info.minor)"`, {
        encoding: "utf-8",
        timeout: 5000,
      }).trim();
      pythonMinor = parseInt(versionOutput) || 12;
    } catch {
      // Default to 3.12 if detection fails
    }

    // Python 3.13+ needs cu124 (CUDA 12.4), earlier versions use cu121 (CUDA 12.1)
    const cudaVersion = pythonMinor >= 13 ? "cu124" : "cu121";
    const torchVersion = pythonMinor >= 13 ? "2.6.0" : "2.5.1";
    const torchvisionVersion = pythonMinor >= 13 ? "0.21.0" : "0.20.1";

    const torchResult = await new Promise<{ success: boolean; error?: string }>((resolve) => {
      // Must specify exact version with +cuXXX suffix to get CUDA wheels
      const proc = spawn(pythonPath, [
        "-m", "pip", "install",
        "--force-reinstall",
        "--no-cache-dir",
        `torch==${torchVersion}+${cudaVersion}`,
        `torchvision==${torchvisionVersion}+${cudaVersion}`,
        "--index-url", `https://download.pytorch.org/whl/${cudaVersion}`,
        "--extra-index-url", "https://pypi.org/simple"
      ], {
        stdio: ["ignore", "pipe", "pipe"],
        timeout: 1200000,
      });

      let stderr = "";

      proc.stdout?.on("data", (data) => {
        const line = data.toString();
        if (line.includes("Downloading") || line.includes("Installing") || line.includes("Collecting")) {
          onProgress?.(line.trim().substring(0, 100), undefined);
        }
      });

      proc.stderr?.on("data", (data) => {
        stderr += data.toString();
        const line = data.toString();
        if (line.includes("Downloading") || line.includes("Installing") || line.includes("Collecting")) {
          onProgress?.(line.trim().substring(0, 100), undefined);
        }
      });

      proc.on("close", (code) => {
        if (code === 0) {
          resolve({ success: true });
        } else {
          resolve({ success: false, error: `PyTorch CUDA installation failed: ${stderr.substring(0, 500)}` });
        }
      });

      proc.on("error", (err) => {
        resolve({ success: false, error: `Failed to install PyTorch: ${err.message}` });
      });
    });

    if (!torchResult.success) {
      return torchResult;
    }
  } else if (gpuInfo.hardware.apple_silicon.found) {
    // For Apple Silicon, install PyTorch which automatically supports MPS
    onProgress?.("Installing PyTorch with Metal support (~1GB)...", 0);

    const torchResult = await new Promise<{ success: boolean; error?: string }>((resolve) => {
      const proc = spawn(pythonPath, ["-m", "pip", "install", "torch", "torchvision"], {
        stdio: ["ignore", "pipe", "pipe"],
        timeout: 1200000,
      });

      let stderr = "";

      proc.stdout?.on("data", (data) => {
        const line = data.toString();
        if (line.includes("Downloading") || line.includes("Installing") || line.includes("Collecting")) {
          onProgress?.(line.trim().substring(0, 100), undefined);
        }
      });

      proc.stderr?.on("data", (data) => {
        stderr += data.toString();
        const line = data.toString();
        if (line.includes("Downloading") || line.includes("Installing") || line.includes("Collecting")) {
          onProgress?.(line.trim().substring(0, 100), undefined);
        }
      });

      proc.on("close", (code) => {
        if (code === 0) {
          resolve({ success: true });
        } else {
          resolve({ success: false, error: `PyTorch installation failed: ${stderr.substring(0, 500)}` });
        }
      });

      proc.on("error", (err) => {
        resolve({ success: false, error: `Failed to install PyTorch: ${err.message}` });
      });
    });

    if (!torchResult.success) {
      return torchResult;
    }
  }

  // Now install ultralytics and other dependencies
  // ultralytics will use the already-installed GPU-enabled PyTorch
  const packages = ["ultralytics>=8.3.0", "sse-starlette>=1.6.0", "onnx>=1.15.0", "onnxruntime>=1.16.0"];
  onProgress?.("Installing YOLO training dependencies...", 50);

  return new Promise((resolve) => {
    const proc = spawn(pythonPath, ["-m", "pip", "install", ...packages], {
      stdio: ["ignore", "pipe", "pipe"],
      timeout: 1200000, // 20 minute timeout for large packages
    });

    let stderr = "";

    proc.stdout?.on("data", (data) => {
      const line = data.toString();
      // Parse pip output for progress
      if (line.includes("Downloading") || line.includes("Installing") || line.includes("Collecting")) {
        onProgress?.(line.trim().substring(0, 100), undefined);
      }
    });

    proc.stderr?.on("data", (data) => {
      stderr += data.toString();
      const line = data.toString();
      // pip often writes progress to stderr
      if (line.includes("Downloading") || line.includes("Installing") || line.includes("Collecting")) {
        onProgress?.(line.trim().substring(0, 100), undefined);
      }
    });

    proc.on("close", (code) => {
      if (code === 0) {
        onProgress?.("YOLO dependencies installed!", 100);
        resolve({ success: true });
      } else {
        resolve({ success: false, error: `Installation failed with code ${code}: ${stderr.substring(0, 500)}` });
      }
    });

    proc.on("error", (err) => {
      resolve({ success: false, error: `Failed to start pip: ${err.message}` });
    });
  });
}

/**
 * Upgrade PyTorch to CUDA version for GPU training.
 * This is for users who already have YOLO dependencies installed with CPU-only PyTorch.
 */
export async function upgradePyTorchToCUDA(
  onProgress?: (message: string, percent?: number) => void
): Promise<{ success: boolean; error?: string }> {
  const pythonPath = getVenvPythonPath();

  if (!fs.existsSync(pythonPath)) {
    return { success: false, error: "Python virtual environment not found" };
  }

  const gpuInfo = await getGPUHardwareInfo();

  if (!gpuInfo.hardware.nvidia.found && !gpuInfo.hardware.apple_silicon.found) {
    return { success: false, error: "No GPU hardware detected" };
  }

  if (gpuInfo.hardware.nvidia.found) {
    onProgress?.("Upgrading PyTorch to CUDA version (~2.5GB)...", 0);

    // Detect Python minor version to choose correct CUDA version
    let pythonMinor = 12; // Default to 3.12
    try {
      const versionOutput = execSync(`"${pythonPath}" -c "import sys; print(sys.version_info.minor)"`, {
        encoding: "utf-8",
        timeout: 5000,
      }).trim();
      pythonMinor = parseInt(versionOutput) || 12;
    } catch {
      // Default to 3.12 if detection fails
    }

    // Python 3.13+ needs cu124 (CUDA 12.4), earlier versions use cu121 (CUDA 12.1)
    const cudaVersion = pythonMinor >= 13 ? "cu124" : "cu121";
    const torchVersion = pythonMinor >= 13 ? "2.6.0" : "2.5.1";
    const torchvisionVersion = pythonMinor >= 13 ? "0.21.0" : "0.20.1";

    return new Promise((resolve) => {
      // Must specify exact version with +cuXXX suffix to get CUDA wheels
      const proc = spawn(pythonPath, [
        "-m", "pip", "install",
        "--force-reinstall",
        "--no-cache-dir",
        `torch==${torchVersion}+${cudaVersion}`,
        `torchvision==${torchvisionVersion}+${cudaVersion}`,
        "--index-url", `https://download.pytorch.org/whl/${cudaVersion}`,
        "--extra-index-url", "https://pypi.org/simple"
      ], {
        stdio: ["ignore", "pipe", "pipe"],
        timeout: 1200000,
      });

      let stderr = "";

      proc.stdout?.on("data", (data) => {
        const line = data.toString();
        if (line.includes("Downloading") || line.includes("Installing") || line.includes("Collecting")) {
          onProgress?.(line.trim().substring(0, 100), undefined);
        }
      });

      proc.stderr?.on("data", (data) => {
        stderr += data.toString();
        const line = data.toString();
        if (line.includes("Downloading") || line.includes("Installing") || line.includes("Collecting")) {
          onProgress?.(line.trim().substring(0, 100), undefined);
        }
      });

      proc.on("close", (code) => {
        if (code === 0) {
          onProgress?.("PyTorch CUDA upgrade complete!", 100);
          resolve({ success: true });
        } else {
          resolve({ success: false, error: `PyTorch CUDA upgrade failed: ${stderr.substring(0, 500)}` });
        }
      });

      proc.on("error", (err) => {
        resolve({ success: false, error: `Failed to upgrade PyTorch: ${err.message}` });
      });
    });
  } else if (gpuInfo.hardware.apple_silicon.found) {
    // For Apple Silicon, reinstall PyTorch (should automatically support MPS)
    onProgress?.("Reinstalling PyTorch for Metal support...", 0);

    return new Promise((resolve) => {
      const proc = spawn(pythonPath, [
        "-m", "pip", "install",
        "--force-reinstall",
        "torch", "torchvision"
      ], {
        stdio: ["ignore", "pipe", "pipe"],
        timeout: 1200000,
      });

      let stderr = "";

      proc.stdout?.on("data", (data) => {
        const line = data.toString();
        if (line.includes("Downloading") || line.includes("Installing") || line.includes("Collecting")) {
          onProgress?.(line.trim().substring(0, 100), undefined);
        }
      });

      proc.stderr?.on("data", (data) => {
        stderr += data.toString();
        const line = data.toString();
        if (line.includes("Downloading") || line.includes("Installing") || line.includes("Collecting")) {
          onProgress?.(line.trim().substring(0, 100), undefined);
        }
      });

      proc.on("close", (code) => {
        if (code === 0) {
          onProgress?.("PyTorch upgrade complete!", 100);
          resolve({ success: true });
        } else {
          resolve({ success: false, error: `PyTorch upgrade failed: ${stderr.substring(0, 500)}` });
        }
      });

      proc.on("error", (err) => {
        resolve({ success: false, error: `Failed to upgrade PyTorch: ${err.message}` });
      });
    });
  }

  return { success: false, error: "No supported GPU found" };
}
