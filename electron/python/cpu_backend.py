"""
CPU Backend for BLOB Detection

Uses OpenCV for all image processing operations.
This is the fallback backend when no GPU is available.
"""

import cv2
import numpy as np
from gpu_backend import BaseGPUBackend


class CPUBackend(BaseGPUBackend):
    """
    CPU backend using OpenCV.
    Provides the same interface as GPU backends for seamless fallback.
    """

    @property
    def name(self) -> str:
        return 'cpu'

    @property
    def device_name(self) -> str:
        return 'CPU (OpenCV)'

    def grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert BGR image to grayscale."""
        if len(image.shape) == 2:
            return image  # Already grayscale
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def clahe(self, gray: np.ndarray, clip_limit: float = 3.0, tile_size: int = 8) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Args:
            gray: Grayscale image
            clip_limit: Threshold for contrast limiting
            tile_size: Size of grid for histogram equalization

        Returns:
            CLAHE-enhanced grayscale image
        """
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_size, tile_size)
        )
        return clahe.apply(gray)

    def gaussian_blur(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Apply Gaussian blur.

        Args:
            image: Input image
            kernel_size: Size of Gaussian kernel (must be odd)

        Returns:
            Blurred image
        """
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def threshold_otsu(self, gray: np.ndarray, invert: bool = True) -> np.ndarray:
        """
        Apply Otsu's thresholding.

        Args:
            gray: Grayscale image
            invert: If True, use THRESH_BINARY_INV (dark objects on light background)

        Returns:
            Binary image
        """
        thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        _, binary = cv2.threshold(gray, 0, 255, thresh_type + cv2.THRESH_OTSU)
        return binary

    def morphological_open(self, binary: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Apply morphological opening (erosion followed by dilation).

        This removes small noise and separates touching objects.

        Args:
            binary: Binary image
            kernel_size: Size of structuring element

        Returns:
            Opened binary image
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    def connected_components(self, binary: np.ndarray) -> tuple:
        """
        Find connected components with statistics.

        Args:
            binary: Binary image

        Returns:
            Tuple of (num_labels, labels, stats, centroids)
            - num_labels: Number of connected components (including background)
            - labels: Label image where each pixel is assigned a component ID
            - stats: Statistics for each component (x, y, width, height, area)
            - centroids: Centroid coordinates for each component
        """
        return cv2.connectedComponentsWithStats(binary, connectivity=8)

    def find_contours(self, binary: np.ndarray):
        """
        Find contours in binary image.

        Args:
            binary: Binary image

        Returns:
            List of contours
        """
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        return contours

    def get_bounding_rect(self, contour) -> tuple:
        """
        Get bounding rectangle for a contour.

        Args:
            contour: Contour from findContours

        Returns:
            Tuple of (x, y, width, height)
        """
        return cv2.boundingRect(contour)

    def contour_area(self, contour) -> float:
        """
        Calculate contour area.

        Args:
            contour: Contour from findContours

        Returns:
            Area in pixels
        """
        return cv2.contourArea(contour)
