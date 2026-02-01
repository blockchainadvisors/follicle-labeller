"""
PyTorch MPS Backend for BLOB Detection

Uses PyTorch with Metal Performance Shaders (MPS) for GPU-accelerated
image processing on macOS with Apple Silicon.

Also includes kornia for advanced image processing operations.
"""

import numpy as np
from gpu_backend import BaseGPUBackend

try:
    import torch
    TORCH_AVAILABLE = True
    MPS_AVAILABLE = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    MPS_AVAILABLE = False
    torch = None

try:
    import kornia
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False
    kornia = None


class TorchMPSBackend(BaseGPUBackend):
    """
    Metal Performance Shaders backend using PyTorch for macOS Apple Silicon.

    Requires:
        pip install torch kornia
    """

    def __init__(self):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed. Install with: pip install torch")

        if not MPS_AVAILABLE:
            raise RuntimeError("MPS (Metal) is not available on this system")

        self._device = torch.device('mps')

    @property
    def name(self) -> str:
        return 'mps'

    @property
    def device_name(self) -> str:
        return 'Apple Silicon GPU (MPS)'

    def _to_tensor(self, array: np.ndarray, dtype=torch.float32) -> 'torch.Tensor':
        """Convert numpy array to PyTorch tensor on MPS device."""
        tensor = torch.from_numpy(array.copy()).to(dtype)
        return tensor.to(self._device)

    def _to_numpy(self, tensor: 'torch.Tensor') -> np.ndarray:
        """Convert PyTorch tensor back to numpy array."""
        return tensor.cpu().numpy()

    def _ensure_batch_channel(self, tensor: 'torch.Tensor') -> 'torch.Tensor':
        """Ensure tensor has batch and channel dimensions (B, C, H, W)."""
        if tensor.dim() == 2:
            # (H, W) -> (1, 1, H, W)
            return tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.dim() == 3:
            # (H, W, C) -> (1, C, H, W)
            return tensor.permute(2, 0, 1).unsqueeze(0)
        return tensor

    def _remove_batch_channel(self, tensor: 'torch.Tensor', original_dim: int) -> 'torch.Tensor':
        """Remove batch and channel dimensions if they were added."""
        if original_dim == 2:
            return tensor.squeeze(0).squeeze(0)
        elif original_dim == 3:
            return tensor.squeeze(0).permute(1, 2, 0)
        return tensor

    def grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert BGR image to grayscale on GPU.
        """
        if len(image.shape) == 2:
            return image  # Already grayscale

        # Convert to tensor
        tensor = self._to_tensor(image, dtype=torch.float32)

        if KORNIA_AVAILABLE:
            # Use kornia's rgb_to_grayscale (expects RGB)
            # OpenCV uses BGR, so we need to convert: BGR -> RGB -> Gray
            tensor = self._ensure_batch_channel(tensor)
            # Reverse channel order (BGR to RGB)
            tensor = torch.flip(tensor, dims=[1])
            gray = kornia.color.rgb_to_grayscale(tensor)
            gray = gray.squeeze(0).squeeze(0)
        else:
            # Manual conversion: Gray = 0.114*B + 0.587*G + 0.299*R
            weights = torch.tensor([0.114, 0.587, 0.299], device=self._device, dtype=torch.float32)
            gray = torch.matmul(tensor, weights)

        # Convert back to uint8 numpy
        gray = gray.clamp(0, 255).to(torch.uint8)
        return self._to_numpy(gray)

    def clahe(self, gray: np.ndarray, clip_limit: float = 3.0, tile_size: int = 8) -> np.ndarray:
        """
        Apply CLAHE on GPU using kornia.
        """
        # Convert to tensor
        tensor = self._to_tensor(gray, dtype=torch.float32) / 255.0
        tensor = self._ensure_batch_channel(tensor)

        if KORNIA_AVAILABLE:
            # kornia's equalize_clahe expects values in [0, 1]
            enhanced = kornia.enhance.equalize_clahe(
                tensor,
                clip_limit=clip_limit,
                grid_size=(tile_size, tile_size)
            )
        else:
            # Fallback: simple histogram equalization without CLAHE
            # This is a simplified version - for best results use kornia
            flat = tensor.flatten()
            hist = torch.histc(flat, bins=256, min=0, max=1)
            cdf = torch.cumsum(hist, dim=0)
            cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min() + 1e-8)

            # Map values
            indices = (tensor * 255).long().clamp(0, 255)
            enhanced = cdf[indices.flatten()].reshape(tensor.shape)

        # Convert back
        enhanced = (enhanced.squeeze(0).squeeze(0) * 255).clamp(0, 255).to(torch.uint8)
        return self._to_numpy(enhanced)

    def gaussian_blur(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Apply Gaussian blur on GPU using kornia.
        """
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Convert to tensor
        tensor = self._to_tensor(image, dtype=torch.float32)
        original_dim = tensor.dim()
        tensor = self._ensure_batch_channel(tensor)

        # Calculate sigma from kernel size
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8

        if KORNIA_AVAILABLE:
            blurred = kornia.filters.gaussian_blur2d(
                tensor,
                kernel_size=(kernel_size, kernel_size),
                sigma=(sigma, sigma)
            )
        else:
            # Manual Gaussian blur using convolution
            # Create Gaussian kernel
            x = torch.arange(kernel_size, device=self._device, dtype=torch.float32) - kernel_size // 2
            gaussian_1d = torch.exp(-x**2 / (2 * sigma**2))
            gaussian_1d = gaussian_1d / gaussian_1d.sum()
            kernel = gaussian_1d.outer(gaussian_1d)
            kernel = kernel.view(1, 1, kernel_size, kernel_size)

            # Apply convolution
            padding = kernel_size // 2
            blurred = torch.nn.functional.conv2d(tensor, kernel, padding=padding)

        # Convert back
        result = self._remove_batch_channel(blurred, original_dim)
        result = result.clamp(0, 255).to(torch.uint8)
        return self._to_numpy(result)

    def threshold_otsu(self, gray: np.ndarray, invert: bool = True) -> np.ndarray:
        """
        Apply Otsu's thresholding on GPU.
        """
        # Convert to tensor
        tensor = self._to_tensor(gray, dtype=torch.float32)

        # Compute histogram
        hist = torch.histc(tensor.flatten(), bins=256, min=0, max=255)
        hist = hist / hist.sum()

        # Compute cumulative sums
        omega = torch.cumsum(hist, dim=0)
        mu = torch.cumsum(hist * torch.arange(256, device=self._device, dtype=torch.float32), dim=0)

        # Compute between-class variance
        mu_t = mu[-1]
        sigma_b = torch.zeros(256, device=self._device, dtype=torch.float32)

        for t in range(256):
            if omega[t] > 0 and omega[t] < 1:
                mu_0 = mu[t] / omega[t]
                mu_1 = (mu_t - mu[t]) / (1 - omega[t])
                sigma_b[t] = omega[t] * (1 - omega[t]) * (mu_0 - mu_1) ** 2

        # Find optimal threshold
        threshold = torch.argmax(sigma_b).item()

        # Apply threshold
        if invert:
            binary = torch.where(tensor <= threshold, 255.0, 0.0)
        else:
            binary = torch.where(tensor > threshold, 255.0, 0.0)

        return self._to_numpy(binary.to(torch.uint8))

    def morphological_open(self, binary: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Apply morphological opening on GPU using kornia.
        """
        # Convert to tensor (kornia expects float [0, 1])
        tensor = self._to_tensor(binary, dtype=torch.float32) / 255.0
        tensor = self._ensure_batch_channel(tensor)

        if KORNIA_AVAILABLE:
            # Create structuring element
            kernel = torch.ones(kernel_size, kernel_size, device=self._device)

            # Erosion followed by dilation
            eroded = kornia.morphology.erosion(tensor, kernel)
            opened = kornia.morphology.dilation(eroded, kernel)
        else:
            # Manual morphological operations using max/min pooling
            padding = kernel_size // 2

            # Erosion = min pooling
            eroded = -torch.nn.functional.max_pool2d(
                -tensor, kernel_size=kernel_size, stride=1, padding=padding
            )

            # Dilation = max pooling
            opened = torch.nn.functional.max_pool2d(
                eroded, kernel_size=kernel_size, stride=1, padding=padding
            )

        # Convert back
        result = (opened.squeeze(0).squeeze(0) * 255).clamp(0, 255).to(torch.uint8)
        return self._to_numpy(result)

    def connected_components(self, binary: np.ndarray) -> tuple:
        """
        Find connected components.

        Note: Connected component labeling is complex on GPU.
        We fall back to CPU (OpenCV) for this operation, which is typically
        fast enough after GPU-accelerated preprocessing.
        """
        import cv2

        # Use OpenCV on CPU for connected components
        # This is a common pattern - GPU for preprocessing, CPU for labeling
        return cv2.connectedComponentsWithStats(binary, connectivity=8)

    def clear_memory(self) -> dict:
        """
        Clear MPS GPU memory cache.

        Returns:
            Dict with cleanup result
        """
        import gc

        result = {
            "success": True,
            "memory_before_mb": None,
            "memory_after_mb": None,
            "memory_freed_mb": 0
        }

        try:
            # Run Python garbage collection
            gc.collect()

            # Empty MPS cache (available in PyTorch 2.0+)
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()

            # Synchronize MPS operations
            if hasattr(torch.mps, 'synchronize'):
                torch.mps.synchronize()

            result["message"] = "MPS memory cache cleared"

        except Exception as e:
            result["success"] = False
            result["error"] = str(e)

        return result
