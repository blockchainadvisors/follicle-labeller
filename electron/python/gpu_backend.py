"""
GPU Backend Manager for BLOB Detection

Detects available GPU backends and provides a unified interface for
GPU-accelerated image processing operations.

Supported backends:
- CUDA via CuPy (Windows/Linux with NVIDIA GPUs)
- Metal/MPS via PyTorch (macOS with Apple Silicon)
- CPU via OpenCV (fallback for all platforms)
"""

import logging
from typing import Dict, Optional, Any
from abc import ABC, abstractmethod
import numpy as np

logger = logging.getLogger(__name__)


class BaseGPUBackend(ABC):
    """Abstract base class for GPU backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name (e.g., 'cuda', 'mps', 'cpu')."""
        pass

    @property
    @abstractmethod
    def device_name(self) -> str:
        """Device name for display."""
        pass

    @abstractmethod
    def grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert BGR image to grayscale."""
        pass

    @abstractmethod
    def clahe(self, gray: np.ndarray, clip_limit: float = 3.0, tile_size: int = 8) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        pass

    @abstractmethod
    def gaussian_blur(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Apply Gaussian blur."""
        pass

    @abstractmethod
    def threshold_otsu(self, gray: np.ndarray, invert: bool = True) -> np.ndarray:
        """Apply Otsu's thresholding."""
        pass

    @abstractmethod
    def morphological_open(self, binary: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Apply morphological opening."""
        pass

    @abstractmethod
    def connected_components(self, binary: np.ndarray) -> tuple:
        """
        Find connected components.
        Returns (num_labels, labels, stats, centroids).
        """
        pass

    def get_info(self) -> Dict[str, Any]:
        """Get backend info for display."""
        return {
            'name': self.name,
            'device': self.device_name,
        }


def detect_cuda_backend() -> Optional[Dict[str, Any]]:
    """Check if CUDA is available via CuPy."""
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() > 0:
            device = cp.cuda.Device(0)
            mem_info = device.mem_info
            return {
                'device': cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8'),
                'memory_gb': mem_info[1] / (1024**3),
                'compute_capability': f"{device.compute_capability[0]}.{device.compute_capability[1]}",
            }
    except ImportError:
        logger.debug("CuPy not installed - CUDA backend unavailable")
    except Exception as e:
        logger.debug(f"CUDA detection failed: {e}")
    return None


def detect_mps_backend() -> Optional[Dict[str, Any]]:
    """Check if Metal Performance Shaders is available via PyTorch."""
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # MPS is available on Apple Silicon
            return {
                'device': 'Apple Silicon GPU',
                'built_in': torch.backends.mps.is_built(),
            }
    except ImportError:
        logger.debug("PyTorch not installed - MPS backend unavailable")
    except Exception as e:
        logger.debug(f"MPS detection failed: {e}")
    return None


class GPUBackendManager:
    """
    Manages GPU backend detection and selection.

    Usage:
        manager = GPUBackendManager()
        backend = manager.get_backend()
        gray = backend.grayscale(image)
    """

    def __init__(self):
        self._backends: Dict[str, Dict[str, Any]] = {}
        self._active_backend: Optional[BaseGPUBackend] = None
        self._detect_backends()

    def _detect_backends(self) -> None:
        """Detect all available GPU backends."""
        logger.info("Detecting GPU backends...")

        # Check CUDA
        cuda_info = detect_cuda_backend()
        if cuda_info:
            self._backends['cuda'] = cuda_info
            logger.info(f"CUDA backend available: {cuda_info['device']} ({cuda_info['memory_gb']:.1f} GB)")

        # Check MPS (Metal)
        mps_info = detect_mps_backend()
        if mps_info:
            self._backends['mps'] = mps_info
            logger.info(f"MPS backend available: {mps_info['device']}")

        # CPU is always available
        self._backends['cpu'] = {'device': 'CPU (OpenCV)'}

        if not cuda_info and not mps_info:
            logger.info("No GPU acceleration available, using CPU backend")

    def get_backend(self) -> BaseGPUBackend:
        """
        Get the best available backend.
        Priority: CUDA > MPS > CPU
        """
        if self._active_backend:
            return self._active_backend

        # Try CUDA first
        if 'cuda' in self._backends:
            try:
                from cupy_backend import CuPyBackend
                self._active_backend = CuPyBackend()
                logger.info(f"Using CUDA backend: {self._active_backend.device_name}")
                return self._active_backend
            except Exception as e:
                logger.warning(f"Failed to initialize CUDA backend: {e}")

        # Try MPS (Metal)
        if 'mps' in self._backends:
            try:
                from torch_backend import TorchMPSBackend
                self._active_backend = TorchMPSBackend()
                logger.info(f"Using MPS backend: {self._active_backend.device_name}")
                return self._active_backend
            except Exception as e:
                logger.warning(f"Failed to initialize MPS backend: {e}")

        # Fall back to CPU
        from cpu_backend import CPUBackend
        self._active_backend = CPUBackend()
        logger.info("Using CPU backend")
        return self._active_backend

    def get_status(self) -> Dict[str, Any]:
        """Get status information for the frontend."""
        backend = self.get_backend()
        return {
            'active_backend': backend.name,
            'device_name': backend.device_name,
            'available': {
                'cuda': 'cuda' in self._backends,
                'mps': 'mps' in self._backends,
                'cpu': True,
            },
            'backends_info': self._backends,
        }

    def get_available_backends(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available backends."""
        return self._backends.copy()

    def clear_gpu_memory(self) -> Dict[str, Any]:
        """
        Clear GPU memory for the active backend.

        Returns:
            Dict with memory stats and cleanup result
        """
        backend = self.get_backend()

        # Check if backend has clear_memory method
        if hasattr(backend, 'clear_memory'):
            result = backend.clear_memory()
            if result.get('memory_freed_mb', 0) > 0:
                logger.info(f"GPU memory cleanup ({backend.name}): freed {result['memory_freed_mb']:.1f}MB")
            return result

        return {
            "success": True,
            "message": f"Backend '{backend.name}' does not require memory cleanup",
            "memory_freed_mb": 0
        }


# Singleton instance
_manager: Optional[GPUBackendManager] = None


def get_gpu_manager() -> GPUBackendManager:
    """Get the global GPU backend manager instance."""
    global _manager
    if _manager is None:
        _manager = GPUBackendManager()
    return _manager
