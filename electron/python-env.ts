/**
 * Manages Python virtual environment and dependencies
 * - Creates venv in app data directory
 * - Installs requirements.txt automatically
 * - Provides paths to venv Python executable
 */

import { app } from "electron";
import path from "path";
import fs from "fs";
import { spawn, execSync, execFile, SpawnOptions } from "child_process";

// Track setup status for UI feedback
let currentSetupStatus = "";

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
 * Find Python executable in system PATH.
 * Tries 'python' first (works with conda/venv), then 'python3'.
 */
async function findSystemPython(): Promise<{
  found: boolean;
  pythonPath?: string;
  version?: string;
  error?: string;
}> {
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
          return { found: true, pythonPath: cmd, version };
        }
      }
    } catch {
      // Try next command
    }
  }

  return {
    found: false,
    error: "Python 3.8+ not found. Please install Python from python.org"
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
 */
export async function ensureVenv(): Promise<{
  success: boolean;
  error?: string;
}> {
  const venvPath = getVenvPath();
  const pythonPath = getVenvPythonPath();

  // Check if venv already exists and is valid
  if (fs.existsSync(pythonPath)) {
    setSetupStatus("Virtual environment ready");
    return { success: true };
  }

  setSetupStatus("Finding system Python...");

  // Find system Python
  const pythonCheck = await findSystemPython();
  if (!pythonCheck.found || !pythonCheck.pythonPath) {
    return { success: false, error: pythonCheck.error };
  }

  setSetupStatus(`Creating virtual environment (${pythonCheck.version})...`);

  // Ensure parent directory exists
  const parentDir = path.dirname(venvPath);
  if (!fs.existsSync(parentDir)) {
    fs.mkdirSync(parentDir, { recursive: true });
  }

  // Create venv
  return new Promise((resolve) => {
    const args = ["-m", "venv", venvPath];

    const proc = spawn(pythonCheck.pythonPath!, args, {
      stdio: ["ignore", "pipe", "pipe"],
      timeout: 120000, // 2 minute timeout
    });

    let stderr = "";

    proc.stderr?.on("data", (data) => {
      stderr += data.toString();
    });

    proc.on("close", (code) => {
      if (code === 0 && fs.existsSync(pythonPath)) {
        setSetupStatus("Virtual environment created");
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
      missing: ["opencv-python", "numpy", "pillow", "flask"]
    };
  }

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
    import flask
except ImportError:
    missing.append('flask')
try:
    from flask_cors import CORS
except ImportError:
    missing.append('flask-cors')
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
          missing: ["opencv-python", "numpy", "pillow", "flask", "flask-cors"]
        });
      }
    });

    proc.on("error", () => {
      resolve({
        installed: false,
        missing: ["opencv-python", "numpy", "pillow", "flask", "flask-cors"]
      });
    });
  });
}

/**
 * Full environment setup: ensures venv exists and dependencies are installed.
 * This is the main entry point for setting up the Python environment.
 */
export async function setupPythonEnvironment(
  requirementsPath: string,
  onProgress?: (message: string) => void
): Promise<{
  success: boolean;
  pythonPath?: string;
  error?: string;
}> {
  // Step 1: Ensure venv exists
  const venvResult = await ensureVenv();
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
