"""
Detect GPU hardware without importing GPU packages.
Works on fresh systems before cupy/torch are installed.
"""
import subprocess
import platform
import json
import sys


def detect_nvidia_gpu():
    """Detect NVIDIA GPU using nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(', ')
            return {
                'found': True,
                'name': parts[0] if parts else 'NVIDIA GPU',
                'driver_version': parts[1] if len(parts) > 1 else 'Unknown'
            }
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return {'found': False}


def detect_apple_silicon():
    """Detect Apple Silicon on macOS."""
    if platform.system() != 'Darwin':
        return {'found': False}

    try:
        # Apple Silicon uses ARM architecture
        machine = platform.machine()
        if machine == 'arm64':
            # Get chip name
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                capture_output=True, text=True, timeout=5
            )
            chip = result.stdout.strip() if result.returncode == 0 else 'Apple Silicon'
            return {'found': True, 'chip': chip}
    except Exception:
        pass
    return {'found': False}


def check_packages_installed():
    """Check if GPU packages are installed (without importing them)."""
    import importlib.util
    return {
        'cupy': importlib.util.find_spec('cupy') is not None,
        'torch': importlib.util.find_spec('torch') is not None,
    }


def main():
    nvidia = detect_nvidia_gpu()
    apple = detect_apple_silicon()
    packages = check_packages_installed()

    result = {
        'hardware': {
            'nvidia': nvidia,
            'apple_silicon': apple,
        },
        'packages': packages,
        'can_enable_gpu': nvidia['found'] or apple['found'],
        'gpu_enabled': packages['cupy'] or packages['torch'],
    }
    print(json.dumps(result))


if __name__ == '__main__':
    main()
