"""
CuPy CUDA Backend for BLOB Detection

Uses CuPy for GPU-accelerated image processing on NVIDIA GPUs.
Provides significant speedup for preprocessing operations.
"""

import numpy as np
from gpu_backend import BaseGPUBackend

try:
    import cupy as cp
    from cupyx.scipy import ndimage as cp_ndimage
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    cp_ndimage = None


class CuPyBackend(BaseGPUBackend):
    """
    CUDA backend using CuPy for GPU-accelerated processing.

    Requires:
        pip install cupy-cuda12x  # For CUDA 12.x
        # or
        pip install cupy-cuda11x  # For CUDA 11.x
    """

    def __init__(self):
        if not CUPY_AVAILABLE:
            raise ImportError("CuPy is not installed. Install with: pip install cupy-cuda12x")

        self._device = cp.cuda.Device(0)
        self._device_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8')

    @property
    def name(self) -> str:
        return 'cuda'

    @property
    def device_name(self) -> str:
        return self._device_name

    def _to_gpu(self, array: np.ndarray) -> 'cp.ndarray':
        """Transfer numpy array to GPU."""
        return cp.asarray(array)

    def _to_cpu(self, array: 'cp.ndarray') -> np.ndarray:
        """Transfer CuPy array back to CPU."""
        return cp.asnumpy(array)

    def grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert BGR image to grayscale on GPU.

        Uses weighted sum: Y = 0.299*R + 0.587*G + 0.114*B
        """
        if len(image.shape) == 2:
            return image  # Already grayscale

        # Transfer to GPU
        gpu_image = self._to_gpu(image)

        # BGR to Gray using standard weights
        # OpenCV uses BGR order, so: Gray = 0.114*B + 0.587*G + 0.299*R
        weights = cp.array([0.114, 0.587, 0.299], dtype=cp.float32)
        gray = cp.dot(gpu_image.astype(cp.float32), weights)

        # Convert back to uint8
        gray = cp.clip(gray, 0, 255).astype(cp.uint8)

        return self._to_cpu(gray)

    def clahe(self, gray: np.ndarray, clip_limit: float = 3.0, tile_size: int = 8) -> np.ndarray:
        """
        Apply CLAHE on GPU.

        Note: CLAHE is complex to implement on GPU efficiently.
        We use a simplified tile-based histogram equalization approach.
        For best results, consider using OpenCV's CLAHE for this step.
        """
        # Transfer to GPU
        gpu_gray = self._to_gpu(gray)
        h, w = gray.shape

        # Calculate tile dimensions
        tile_h = h // tile_size
        tile_w = w // tile_size

        result = cp.zeros_like(gpu_gray, dtype=cp.float32)

        for i in range(tile_size):
            for j in range(tile_size):
                # Extract tile
                y1, y2 = i * tile_h, (i + 1) * tile_h if i < tile_size - 1 else h
                x1, x2 = j * tile_w, (j + 1) * tile_w if j < tile_size - 1 else w
                tile = gpu_gray[y1:y2, x1:x2]

                # Compute histogram
                hist, _ = cp.histogram(tile.flatten(), bins=256, range=(0, 256))
                hist = hist.astype(cp.float32)

                # Clip histogram
                excess = cp.sum(cp.maximum(hist - clip_limit * tile.size / 256, 0))
                hist = cp.minimum(hist, clip_limit * tile.size / 256)
                hist += excess / 256

                # Cumulative distribution
                cdf = cp.cumsum(hist)
                cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min() + 1e-8)
                cdf = cdf.astype(cp.uint8)

                # Apply mapping
                result[y1:y2, x1:x2] = cdf[tile.astype(cp.int32)]

        return self._to_cpu(result.astype(cp.uint8))

    def gaussian_blur(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Apply Gaussian blur on GPU using CuPy's ndimage.
        """
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Transfer to GPU
        gpu_image = self._to_gpu(image.astype(np.float32))

        # Calculate sigma from kernel size (OpenCV convention)
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8

        # Apply Gaussian filter
        blurred = cp_ndimage.gaussian_filter(gpu_image, sigma=sigma)

        return self._to_cpu(blurred.astype(cp.uint8))

    def threshold_otsu(self, gray: np.ndarray, invert: bool = True) -> np.ndarray:
        """
        Apply Otsu's thresholding on GPU.

        Implements the classic Otsu algorithm for automatic threshold selection.
        """
        # Transfer to GPU
        gpu_gray = self._to_gpu(gray)

        # Compute histogram
        hist, _ = cp.histogram(gpu_gray.flatten(), bins=256, range=(0, 256))
        hist = hist.astype(cp.float32)
        hist /= hist.sum()

        # Compute cumulative sums
        omega = cp.cumsum(hist)
        mu = cp.cumsum(hist * cp.arange(256))

        # Compute between-class variance for all thresholds
        mu_t = mu[-1]
        sigma_b = cp.zeros(256, dtype=cp.float32)

        for t in range(256):
            if omega[t] > 0 and omega[t] < 1:
                mu_0 = mu[t] / omega[t]
                mu_1 = (mu_t - mu[t]) / (1 - omega[t])
                sigma_b[t] = omega[t] * (1 - omega[t]) * (mu_0 - mu_1) ** 2

        # Find optimal threshold
        threshold = int(cp.argmax(sigma_b))

        # Apply threshold
        if invert:
            binary = cp.where(gpu_gray <= threshold, 255, 0).astype(cp.uint8)
        else:
            binary = cp.where(gpu_gray > threshold, 255, 0).astype(cp.uint8)

        return self._to_cpu(binary)

    def morphological_open(self, binary: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Apply morphological opening on GPU.

        Opening = Erosion followed by Dilation
        """
        # Transfer to GPU
        gpu_binary = self._to_gpu(binary.astype(np.bool_))

        # Create structuring element
        struct = cp.ones((kernel_size, kernel_size), dtype=cp.bool_)

        # Erosion
        eroded = cp_ndimage.binary_erosion(gpu_binary, structure=struct)

        # Dilation
        opened = cp_ndimage.binary_dilation(eroded, structure=struct)

        # Convert back to uint8
        result = (opened * 255).astype(cp.uint8)

        return self._to_cpu(result)

    def connected_components(self, binary: np.ndarray) -> tuple:
        """
        Find connected components on GPU.

        Uses CuPy's ndimage.label for labeling.
        Note: Statistics calculation is done on CPU for compatibility.
        """
        # Transfer to GPU
        gpu_binary = self._to_gpu(binary > 0)

        # Label connected components
        labels, num_labels = cp_ndimage.label(gpu_binary)

        # Transfer back to CPU for stats calculation
        labels_cpu = self._to_cpu(labels)

        # Calculate statistics on CPU (complex to do on GPU)
        # For each label, compute: x, y, width, height, area
        stats = []
        centroids = []

        for i in range(1, num_labels + 1):
            mask = labels_cpu == i
            ys, xs = np.where(mask)

            if len(xs) > 0:
                x = int(xs.min())
                y = int(ys.min())
                w = int(xs.max() - x + 1)
                h = int(ys.max() - y + 1)
                area = int(len(xs))

                stats.append([x, y, w, h, area])
                centroids.append([xs.mean(), ys.mean()])
            else:
                stats.append([0, 0, 0, 0, 0])
                centroids.append([0, 0])

        # Format as OpenCV-compatible output
        # Add background label stats
        stats.insert(0, [0, 0, binary.shape[1], binary.shape[0], 0])
        centroids.insert(0, [0, 0])

        stats_array = np.array(stats, dtype=np.int32)
        centroids_array = np.array(centroids, dtype=np.float64)

        return (num_labels + 1, labels_cpu.astype(np.int32), stats_array, centroids_array)
