#!/usr/bin/env python3
"""
BLOB Detection Backend Server (FastAPI)

This FastAPI server provides an API for the Follicle Labeller application to use
OpenCV-based BLOB detection for automated follicle detection. It supports:
- Setting an image for a session
- Adding user annotations for size learning
- Auto-detecting follicles using SimpleBlobDetector + contour fallback

The server learns follicle size from user annotations (minimum 3 required)
and uses that information to calibrate detection parameters.

FastAPI provides async request handling which avoids thread pool conflicts
with ProcessPoolExecutor used for parallel blob detection.

Usage:
    python blob_server.py [--port PORT] [--host HOST]
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import signal
import sys
import uuid
from typing import Any, Dict, List, Optional, Tuple

# Store early initialization messages (before logger is configured)
_early_init_messages = []

# Auto-detect and set CUDA_PATH if not already set (Windows only)
_cuda_path_set = False
_cuda_toolkit_installed = False
if sys.platform == 'win32' and not os.environ.get('CUDA_PATH'):
    # Common CUDA installation paths on Windows
    cuda_base = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'
    if os.path.exists(cuda_base):
        # Find the latest CUDA version
        try:
            versions = [d for d in os.listdir(cuda_base) if d.startswith('v')]
            if versions:
                # Sort by version number (e.g., v12.4, v12.3, v11.8)
                versions.sort(key=lambda x: [int(n) for n in x[1:].split('.')], reverse=True)
                cuda_path = os.path.join(cuda_base, versions[0])
                if os.path.exists(cuda_path):
                    os.environ['CUDA_PATH'] = cuda_path
                    _cuda_path_set = True
                    _cuda_toolkit_installed = True
                    _early_init_messages.append(f"CUDA_PATH auto-detected: {cuda_path}")
        except Exception as e:
            _early_init_messages.append(f"CUDA_PATH auto-detection failed: {e}")
    else:
        _early_init_messages.append(
            "CUDA Toolkit not installed. GPU acceleration via CuPy requires CUDA Toolkit. "
            "Detection will use CPU. Install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads for GPU support."
        )
elif os.environ.get('CUDA_PATH'):
    _cuda_path_set = True
    _cuda_toolkit_installed = True
    _early_init_messages.append(f"CUDA_PATH already set: {os.environ.get('CUDA_PATH')}")

# Setup NVIDIA CUDA DLL paths before importing GPU libraries (Windows only)
_cuda_dll_setup_success = False
if sys.platform == 'win32':
    try:
        # Find the venv site-packages nvidia directory
        import site
        nvidia_dll_paths = []
        site_packages = site.getsitepackages()
        _early_init_messages.append(f"CUDA DLL setup: Checking site-packages: {site_packages}")

        for site_path in site_packages:
            nvidia_base = os.path.join(site_path, 'nvidia')
            if os.path.exists(nvidia_base):
                _early_init_messages.append(f"CUDA DLL setup: Found nvidia directory at {nvidia_base}")
                dll_subdirs = [
                    'cuda_nvrtc/bin', 'cublas/bin', 'cuda_runtime/bin',
                    'cufft/bin', 'curand/bin', 'cusolver/bin',
                    'cusparse/bin', 'nvjitlink/bin'
                ]
                for subdir in dll_subdirs:
                    dll_path = os.path.join(nvidia_base, subdir)
                    if os.path.exists(dll_path):
                        nvidia_dll_paths.append(dll_path)
                        # Use os.add_dll_directory for Python 3.8+
                        os.add_dll_directory(dll_path)
                break

        if not nvidia_dll_paths:
            _early_init_messages.append("CUDA DLL setup: No NVIDIA DLL paths found in site-packages")

        # Also prepend to PATH for broader compatibility
        if nvidia_dll_paths:
            current_path = os.environ.get('PATH', '')
            new_paths = os.pathsep.join(nvidia_dll_paths)
            os.environ['PATH'] = new_paths + os.pathsep + current_path
            _early_init_messages.append(f"CUDA DLL setup: Added {len(nvidia_dll_paths)} paths to DLL search")
            _cuda_dll_setup_success = True
    except Exception as e:
        _early_init_messages.append(f"CUDA DLL setup ERROR: {type(e).__name__}: {e}")

import cv2
import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import GPU backend manager
_gpu_import_error = None
try:
    from gpu_backend import get_gpu_manager, GPUBackendManager
    GPU_AVAILABLE = True
    _early_init_messages.append("GPU backend: Successfully imported gpu_backend module")
except ImportError as e:
    GPU_AVAILABLE = False
    get_gpu_manager = None
    GPUBackendManager = None
    _gpu_import_error = str(e)
    _early_init_messages.append(f"GPU backend: Import failed - {e}")
except Exception as e:
    GPU_AVAILABLE = False
    get_gpu_manager = None
    GPUBackendManager = None
    _gpu_import_error = str(e)
    _early_init_messages.append(f"GPU backend: Unexpected error - {type(e).__name__}: {e}")

# Import parallel blob detection
_parallel_import_error = None
try:
    from parallel_blob_detect import parallel_blob_detect
    PARALLEL_BLOB_AVAILABLE = True
    _early_init_messages.append("Parallel detection: Successfully imported")
except ImportError as e:
    PARALLEL_BLOB_AVAILABLE = False
    parallel_blob_detect = None
    _parallel_import_error = str(e)
    _early_init_messages.append(f"Parallel detection: Import failed - {e}")
except Exception as e:
    PARALLEL_BLOB_AVAILABLE = False
    parallel_blob_detect = None
    _parallel_import_error = str(e)
    _early_init_messages.append(f"Parallel detection: Unexpected error - {type(e).__name__}: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log early initialization messages now that logger is available
for msg in _early_init_messages:
    if 'ERROR' in msg or 'failed' in msg.lower():
        logger.warning(msg)
    else:
        logger.info(msg)

# Log summary of initialization status
logger.info(f"Initialization summary: CUDA_DLL_SETUP={_cuda_dll_setup_success}, GPU_AVAILABLE={GPU_AVAILABLE}, PARALLEL_AVAILABLE={PARALLEL_BLOB_AVAILABLE}")
if _gpu_import_error:
    logger.error(f"GPU backend unavailable: {_gpu_import_error}")
if _parallel_import_error:
    logger.error(f"Parallel detection unavailable: {_parallel_import_error}")

# Create FastAPI app
app = FastAPI(
    title="BLOB Detection Server",
    description="OpenCV-based blob detection API for follicle labelling",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
sessions: Dict[str, dict] = {}

# GPU Backend Manager (initialized lazily)
gpu_manager: Optional['GPUBackendManager'] = None

# Server reference for shutdown
server_instance = None


# ============================================
# Pydantic Models for Request Validation
# ============================================

class SetImageRequest(BaseModel):
    image: str  # base64

class AnnotationBox(BaseModel):
    x: float
    y: float
    width: float
    height: float

class AddAnnotationRequest(BaseModel):
    sessionId: str
    x: float
    y: float
    width: float
    height: float

class SyncAnnotationsRequest(BaseModel):
    sessionId: str
    annotations: List[AnnotationBox]

class BlobDetectRequest(BaseModel):
    sessionId: str
    settings: Optional[Dict[str, Any]] = None

class CropRegionRequest(BaseModel):
    sessionId: str
    x: float
    y: float
    width: float
    height: float

class SessionRequest(BaseModel):
    sessionId: str


# ============================================
# Helper Functions
# ============================================

def get_gpu_backend():
    """Get the GPU backend manager, initializing if needed."""
    global gpu_manager
    if gpu_manager is None and GPU_AVAILABLE and get_gpu_manager:
        try:
            gpu_manager = get_gpu_manager()
            logger.info(f"GPU Backend: {gpu_manager.get_status()['active_backend']}")
        except Exception as e:
            logger.warning(f"Failed to initialize GPU backend: {e}")
    return gpu_manager

# Minimum annotations required for auto-detection
MIN_ANNOTATIONS_FOR_DETECTION = 3

# Parallel blob detection threshold (use parallel for images larger than this)
# 10 million pixels = ~3162x3162 or 2500x4000
PARALLEL_DETECTION_PIXEL_THRESHOLD = 10_000_000


def decode_image(image_data: str) -> Optional[np.ndarray]:
    """Decode base64 image data to numpy array (BGR format for OpenCV)."""
    try:
        # Handle data URL format
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # Apply EXIF rotation to match frontend display
        # This ensures coordinates from frontend match the actual pixel positions
        image = ImageOps.exif_transpose(image)

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert PIL to numpy array (RGB)
        rgb_array = np.array(image)

        # Convert RGB to BGR for OpenCV
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

        return bgr_array
    except Exception as e:
        logger.error(f"Failed to decode image: {e}")
        return None


def estimate_follicle_size(annotations: List[dict]) -> int:
    """
    Estimate average follicle size from user annotations.

    Args:
        annotations: List of annotation dicts with x, y, width, height

    Returns:
        Estimated follicle size in pixels (average of width and height)
    """
    if not annotations:
        return 20  # Default fallback

    sizes = []
    for ann in annotations:
        w = ann.get('width', 0)
        h = ann.get('height', 0)
        if w > 0 and h > 0:
            sizes.append((w + h) / 2)

    if sizes:
        return int(np.mean(sizes))
    return 20


def calculate_learned_stats(annotations: List[dict], image: Optional[np.ndarray] = None) -> dict:
    """
    Calculate detailed statistics from annotations for the Learn from Selection dialog.

    Args:
        annotations: List of annotation dicts with x, y, width, height
        image: Optional BGR image for mean intensity calculation

    Returns:
        Dict with examplesAnalyzed, minWidth, maxWidth, minHeight, maxHeight,
        minAspectRatio, maxAspectRatio, meanIntensity
    """
    if not annotations:
        return {
            'examplesAnalyzed': 0,
            'minWidth': 10,
            'maxWidth': 100,
            'minHeight': 10,
            'maxHeight': 100,
            'minAspectRatio': 1.0,
            'maxAspectRatio': 1.0,
            'meanIntensity': 128
        }

    widths = []
    heights = []
    aspect_ratios = []
    intensities = []

    for ann in annotations:
        w = ann.get('width', 0)
        h = ann.get('height', 0)
        x = ann.get('x', 0)
        y = ann.get('y', 0)

        if w > 0 and h > 0:
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h)

            # Calculate mean intensity if image is provided
            if image is not None:
                x1 = max(0, int(x))
                y1 = max(0, int(y))
                x2 = min(image.shape[1], int(x + w))
                y2 = min(image.shape[0], int(y + h))
                if x2 > x1 and y2 > y1:
                    roi = image[y1:y2, x1:x2]
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
                    intensities.append(np.mean(gray_roi))

    if not widths:
        return {
            'examplesAnalyzed': 0,
            'minWidth': 10,
            'maxWidth': 100,
            'minHeight': 10,
            'maxHeight': 100,
            'minAspectRatio': 1.0,
            'maxAspectRatio': 1.0,
            'meanIntensity': 128
        }

    return {
        'examplesAnalyzed': len(widths),
        'minWidth': int(min(widths)),
        'maxWidth': int(max(widths)),
        'minHeight': int(min(heights)),
        'maxHeight': int(max(heights)),
        'minAspectRatio': round(min(aspect_ratios), 2),
        'maxAspectRatio': round(max(aspect_ratios), 2),
        'meanIntensity': int(np.mean(intensities)) if intensities else 128
    }


def calculate_iou(box1: Tuple[int, int, int, int],
                   box2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union between two boxes.

    Args:
        box1, box2: Tuples of (x1, y1, x2, y2) - top-left and bottom-right corners

    Returns:
        IoU value between 0 and 1
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def overlaps_existing(box: Tuple[int, int, int, int],
                      existing_boxes: List[Tuple[int, int, int, int]],
                      iou_threshold: float = 0.3) -> bool:
    """
    Check if box overlaps significantly with any existing detection.

    Args:
        box: Tuple of (x1, y1, x2, y2)
        existing_boxes: List of existing box tuples
        iou_threshold: Maximum allowed IoU before considering overlap

    Returns:
        True if box overlaps with existing, False otherwise
    """
    for existing in existing_boxes:
        if calculate_iou(box, existing) > iou_threshold:
            return True
    return False


class SpatialGrid:
    """
    Fast spatial grid for O(1) average-case overlap checking.

    Instead of O(n) scans for each new detection, this uses a grid
    to only check nearby boxes. Reduces overlap filtering from O(n²) to O(n).
    """

    def __init__(self, cell_size: float = 100.0):
        """
        Initialize spatial grid.

        Args:
            cell_size: Size of each grid cell in pixels. Should be roughly
                      the maximum expected box size for best performance.
        """
        self.cell_size = cell_size
        self.grid: Dict[Tuple[int, int], List[Tuple[int, int, int, int]]] = {}

    def _get_cells(self, box: Tuple[int, int, int, int]) -> List[Tuple[int, int]]:
        """Get all grid cells that a box touches."""
        x1, y1, x2, y2 = box
        cell_x1 = int(x1 / self.cell_size)
        cell_y1 = int(y1 / self.cell_size)
        cell_x2 = int(x2 / self.cell_size)
        cell_y2 = int(y2 / self.cell_size)

        cells = []
        for cx in range(cell_x1, cell_x2 + 1):
            for cy in range(cell_y1, cell_y2 + 1):
                cells.append((cx, cy))
        return cells

    def add(self, box: Tuple[int, int, int, int]):
        """Add a box to the grid."""
        for cell in self._get_cells(box):
            if cell not in self.grid:
                self.grid[cell] = []
            self.grid[cell].append(box)

    def overlaps(self, box: Tuple[int, int, int, int], iou_threshold: float = 0.3) -> bool:
        """Check if box overlaps with any existing box in the grid."""
        # Check all cells the box touches plus neighbors
        x1, y1, x2, y2 = box
        cell_x1 = int(x1 / self.cell_size) - 1
        cell_y1 = int(y1 / self.cell_size) - 1
        cell_x2 = int(x2 / self.cell_size) + 1
        cell_y2 = int(y2 / self.cell_size) + 1

        checked = set()  # Avoid checking same box twice
        for cx in range(cell_x1, cell_x2 + 1):
            for cy in range(cell_y1, cell_y2 + 1):
                for existing in self.grid.get((cx, cy), []):
                    box_id = id(existing)
                    if box_id in checked:
                        continue
                    checked.add(box_id)
                    if calculate_iou(box, existing) > iou_threshold:
                        return True
        return False


def filter_overlapping_detections(
    keypoints: List,
    image_shape: Tuple[int, int],
    follicle_size: int,
    existing_annotations: List[Tuple[int, int, int, int]],
    iou_threshold: float = 0.3,
    method_name: str = 'blob'
) -> List[dict]:
    """
    Filter overlapping detections using spatial grid for O(n) performance.

    Args:
        keypoints: List of cv2.KeyPoint objects
        image_shape: (height, width) of the image
        follicle_size: Estimated follicle size for bounding box calculation
        existing_annotations: Pre-existing annotation boxes to avoid
        iou_threshold: IoU threshold for overlap detection
        method_name: Name of detection method for result metadata

    Returns:
        List of detection dicts with x, y, width, height, confidence, method
    """
    h, w = image_shape

    # Use cell size based on expected box size
    cell_size = max(follicle_size * 2, 50)
    grid = SpatialGrid(cell_size=cell_size)

    # Add existing annotations to grid
    for box in existing_annotations:
        grid.add(box)

    results = []
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        r = max(int(kp.size / 2), follicle_size // 2)
        box = (x - r, y - r, x + r, y + r)

        # Check bounds
        if box[0] < 0 or box[1] < 0 or box[2] > w or box[3] > h:
            continue

        # Check overlap using spatial grid (O(1) average case)
        if not grid.overlaps(box, iou_threshold):
            grid.add(box)
            results.append({
                'x': box[0],
                'y': box[1],
                'width': box[2] - box[0],
                'height': box[3] - box[1],
                'confidence': 0.8,
                'method': method_name
            })

    return results


def detect_blobs(image: np.ndarray,
                 annotations: List[dict],
                 settings: Optional[dict] = None) -> List[dict]:
    """
    Detect follicles using OpenCV SimpleBlobDetector with contour fallback.

    This implements the proven algorithm from hair_follicle_detection:
    1. Apply CLAHE for contrast enhancement (if enabled)
    2. Run SimpleBlobDetector with multi-threshold scanning
    3. Fall back to adaptive thresholding + contours if few detections
    4. Filter by size (learned from annotations OR manual settings)

    Args:
        image: BGR image as numpy array
        annotations: List of user annotations for size learning
        settings: Optional dict with manual settings:
            - minWidth, maxWidth, minHeight, maxHeight: Size range
            - useCLAHE: Whether to apply CLAHE preprocessing
            - claheClipLimit: CLAHE clip limit (default 3.0)
            - claheTileSize: CLAHE tile grid size (default 8)
            - darkBlobs: Detect dark blobs (True) or light blobs (False)
            - useSoftNMS: Whether to use soft-NMS (not yet implemented)

    Returns:
        List of detected follicle dicts with x, y, width, height, confidence
    """
    settings = settings or {}

    # Determine size range based on mode:
    # 1. learnedWithTolerance: Use learned stats with tolerance adjustment
    # 2. Manual settings: Use exact minWidth/maxWidth values
    # 3. Auto-learn: Use 3x tolerance from mean size

    if settings.get('useLearnedStats') and annotations:
        # Mode 1: Use learned stats from annotations with tolerance adjustment
        stats = calculate_learned_stats(annotations)
        tolerance = settings.get('tolerance', 20) / 100.0  # Default 20%

        width_range = stats['maxWidth'] - stats['minWidth']
        height_range = stats['maxHeight'] - stats['minHeight']

        min_width = max(1, int(stats['minWidth'] - width_range * tolerance))
        max_width = int(stats['maxWidth'] + width_range * tolerance)
        min_height = max(1, int(stats['minHeight'] - height_range * tolerance))
        max_height = int(stats['maxHeight'] + height_range * tolerance)

        min_area = min_width * min_height
        max_area = max_width * max_height
        follicle_size = int((min_width + max_width + min_height + max_height) / 4)
        logger.info(f"Using learned stats with {int(tolerance*100)}% tolerance: {min_width}-{max_width} x {min_height}-{max_height}")

    elif settings.get('minWidth') and settings.get('maxWidth'):
        # Mode 2: Use manual settings (exact values)
        min_width = settings.get('minWidth', 10)
        max_width = settings.get('maxWidth', 200)
        min_height = settings.get('minHeight', 10)
        max_height = settings.get('maxHeight', 200)
        min_area = min_width * min_height
        max_area = max_width * max_height
        follicle_size = int((min_width + max_width + min_height + max_height) / 4)
        logger.info(f"Using manual size settings: {min_width}-{max_width} x {min_height}-{max_height}")

    elif annotations:
        # Mode 3: Auto-learn with 3x tolerance (legacy behavior)
        follicle_size = estimate_follicle_size(annotations)
        min_area = max(10, (follicle_size // 3) ** 2)
        max_area = (follicle_size * 3) ** 2
        logger.info(f"Follicle size estimate from annotations: {follicle_size}px, area range: {min_area}-{max_area}")

    else:
        # Default fallback
        follicle_size = 30
        min_area = 100  # ~10x10
        max_area = 10000  # ~100x100
        logger.info(f"Using default size range: area {min_area}-{max_area}")

    # Get GPU backend if available (unless forceCPU is set)
    force_cpu = settings.get('forceCPU', False)
    manager = get_gpu_backend() if not force_cpu else None
    backend = manager.get_backend() if manager else None
    use_gpu = backend is not None and backend.name != 'cpu'

    if force_cpu:
        logger.info("Using CPU backend (forced by user preference)")
    elif use_gpu:
        logger.info(f"Using GPU backend: {backend.name} ({backend.device_name})")

    # Convert to grayscale (with GPU fallback to CPU on error)
    if use_gpu:
        try:
            gray = backend.grayscale(image)
        except Exception as e:
            logger.warning(f"GPU grayscale failed, falling back to CPU: {e}")
            use_gpu = False
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE for better contrast (if enabled)
    use_clahe = settings.get('useCLAHE', True)
    if use_clahe:
        clip_limit = settings.get('claheClipLimit', 3.0)
        tile_size = settings.get('claheTileSize', 8)
        if use_gpu:
            try:
                gray = backend.clahe(gray, clip_limit=clip_limit, tile_size=tile_size)
            except Exception as e:
                logger.warning(f"GPU CLAHE failed, falling back to CPU: {e}")
                use_gpu = False
                # Need to convert back from GPU array if grayscale succeeded on GPU
                if hasattr(gray, 'get'):
                    gray = gray.get()
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
                gray = clahe.apply(gray)
        else:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
            gray = clahe.apply(gray)
        logger.info(f"Applied CLAHE with clipLimit={clip_limit}, tileSize={tile_size} (GPU={use_gpu})")

    detected_boxes: List[Tuple[int, int, int, int]] = []
    detected_results: List[dict] = []

    # Convert existing annotations to boxes for overlap checking
    existing_boxes = []
    for ann in annotations:
        x, y = ann.get('x', 0), ann.get('y', 0)
        w, h = ann.get('width', 0), ann.get('height', 0)
        if w > 0 and h > 0:
            existing_boxes.append((x, y, x + w, y + h))

    # Method 1: SimpleBlobDetector with multi-threshold
    params = cv2.SimpleBlobDetector_Params()

    # Threshold parameters - scan from 10 to 220 in steps of 10 (22 levels)
    params.minThreshold = settings.get('minThreshold', 10)
    params.maxThreshold = settings.get('maxThreshold', 220)
    params.thresholdStep = settings.get('thresholdStep', 10)

    # Area filter
    params.filterByArea = True
    params.minArea = min_area
    params.maxArea = max_area

    # Circularity filter
    # Default: DISABLED - hair follicles are elongated (shaved ~1mm), not circular
    # Set filterByCircularity=True and minCircularity=0.2 to require circular shapes
    filter_by_circularity = settings.get('filterByCircularity', False)
    params.filterByCircularity = filter_by_circularity
    params.minCircularity = settings.get('minCircularity', 0.1)

    # Convexity filter
    # Default: DISABLED - elongated follicles may not be convex
    filter_by_convexity = settings.get('filterByConvexity', False)
    params.filterByConvexity = filter_by_convexity
    params.minConvexity = settings.get('minConvexity', 0.5)

    # Inertia filter (controls elongation: 0 = line, 1 = circle)
    # Default: ENABLED with low threshold to allow elongated shapes
    # minInertiaRatio=0.01 allows very elongated, maxInertiaRatio=1.0 allows any shape
    filter_by_inertia = settings.get('filterByInertia', True)
    params.filterByInertia = filter_by_inertia
    params.minInertiaRatio = settings.get('minInertiaRatio', 0.01)
    params.maxInertiaRatio = settings.get('maxInertiaRatio', 1.0)

    # Color filter - detect dark or light blobs based on settings
    # Default: dark blobs (hair follicles appear dark)
    dark_blobs = settings.get('darkBlobs', True)
    filter_by_color = settings.get('filterByColor', True)
    params.filterByColor = filter_by_color
    params.blobColor = 0 if dark_blobs else 255  # 0 = dark blobs, 255 = light blobs

    # Log all detection parameters
    logger.info("=" * 60)
    logger.info("BLOB DETECTION PARAMETERS:")
    logger.info(f"  Image size: {image.shape[1]}x{image.shape[0]}")
    logger.info(f"  Annotations count: {len(annotations)}")
    logger.info(f"  Settings received: {settings}")
    logger.info("-" * 40)
    logger.info("  SIZE SETTINGS:")
    logger.info(f"    Follicle size estimate: {follicle_size}px")
    logger.info(f"    Min area: {min_area}px²")
    logger.info(f"    Max area: {max_area}px²")
    logger.info("-" * 40)
    logger.info("  CLAHE SETTINGS:")
    logger.info(f"    Enabled: {use_clahe}")
    if use_clahe:
        logger.info(f"    Clip limit: {settings.get('claheClipLimit', 3.0)}")
        logger.info(f"    Tile size: {settings.get('claheTileSize', 8)}")
    logger.info("-" * 40)
    logger.info("  BLOB DETECTOR SETTINGS:")
    logger.info(f"    Dark blobs: {dark_blobs}")
    logger.info(f"    Blob color: {params.blobColor} (0=dark, 255=light)")
    logger.info(f"    Threshold range: {params.minThreshold}-{params.maxThreshold} (step: {params.thresholdStep})")
    logger.info(f"    Filter by area: {params.filterByArea}")
    logger.info(f"    Filter by color: {params.filterByColor}")
    logger.info(f"    Filter by circularity: {params.filterByCircularity} (min: {params.minCircularity})")
    logger.info(f"    Filter by convexity: {params.filterByConvexity} (min: {params.minConvexity})")
    logger.info(f"    Filter by inertia: {params.filterByInertia} (min: {params.minInertiaRatio}, max: {params.maxInertiaRatio})")
    logger.info("=" * 60)

    # Determine if we should use parallel detection for large images
    pixel_count = gray.shape[0] * gray.shape[1]
    use_parallel = (
        PARALLEL_BLOB_AVAILABLE
        and pixel_count >= PARALLEL_DETECTION_PIXEL_THRESHOLD
        and not settings.get('disableParallel', False)
    )

    if use_parallel:
        # Use parallel tiled blob detection for large images (10-15x faster)
        logger.info(f"Using PARALLEL blob detection ({pixel_count:,} pixels >= {PARALLEL_DETECTION_PIXEL_THRESHOLD:,} threshold)")

        keypoints = parallel_blob_detect(
            gray,
            min_threshold=params.minThreshold,
            max_threshold=params.maxThreshold,
            threshold_step=params.thresholdStep,
            min_area=min_area,
            max_area=max_area,
            filter_by_circularity=params.filterByCircularity,
            min_circularity=params.minCircularity,
            filter_by_convexity=params.filterByConvexity,
            min_convexity=params.minConvexity,
            filter_by_inertia=params.filterByInertia,
            min_inertia_ratio=params.minInertiaRatio,
            max_inertia_ratio=params.maxInertiaRatio,
            filter_by_color=params.filterByColor,
            blob_color=params.blobColor,
            n_tiles=16,
            overlap=300,
            dedup_cell_size=8.0
        )
    else:
        # Use standard sequential blob detection
        if pixel_count >= PARALLEL_DETECTION_PIXEL_THRESHOLD:
            logger.info(f"Using SEQUENTIAL blob detection (parallel not available)")
        else:
            logger.info(f"Using SEQUENTIAL blob detection ({pixel_count:,} pixels < {PARALLEL_DETECTION_PIXEL_THRESHOLD:,} threshold)")

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray)

    # Use optimized spatial grid filter (O(n) instead of O(n²))
    method_name = 'blob_parallel' if use_parallel else 'blob'
    detected_results = filter_overlapping_detections(
        keypoints=keypoints,
        image_shape=(image.shape[0], image.shape[1]),
        follicle_size=follicle_size,
        existing_annotations=existing_boxes,
        iou_threshold=0.3,
        method_name=method_name
    )

    # Track boxes for contour fallback
    detected_boxes = [
        (r['x'], r['y'], r['x'] + r['width'], r['y'] + r['height'])
        for r in detected_results
    ]

    logger.info(f"Blob detection found {len(detected_results)} follicles (parallel={use_parallel})")

    # Method 2: Contour-based fallback if blob detection found few results
    if len(detected_results) < 50:
        logger.info("Trying contour-based detection...")

        # Ensure gray is a numpy array (may still be GPU array if grayscale succeeded but CLAHE failed)
        if hasattr(gray, 'get'):
            gray = gray.get()

        # Use Gaussian blur + Otsu thresholding (better than adaptive for this use case)
        if use_gpu:
            try:
                blurred = backend.gaussian_blur(gray, kernel_size=5)
                thresh = backend.threshold_otsu(blurred, invert=True)
                # Morphological opening to remove noise and separate touching objects
                thresh = backend.morphological_open(thresh, kernel_size=3)
            except Exception as e:
                logger.warning(f"GPU contour ops failed, falling back to CPU: {e}")
                use_gpu = False
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                kernel = np.ones((3, 3), np.uint8)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        else:
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # Morphological opening to remove noise and separate touching objects
            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Ensure thresh is a numpy array before findContours
        if hasattr(thresh, 'get'):
            thresh = thresh.get()

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)

                # Check aspect ratio - reject if too elongated
                aspect = max(w, h) / max(min(w, h), 1)
                if aspect < 3:
                    box = (x, y, x + w, y + h)
                    if not overlaps_existing(box, existing_boxes + detected_boxes):
                        detected_boxes.append(box)
                        detected_results.append({
                            'x': x,
                            'y': y,
                            'width': w,
                            'height': h,
                            'confidence': 0.6,  # Lower confidence for contour method
                            'method': 'contour'
                        })
                        contour_count += 1

        logger.info(f"Contour detection added {contour_count} more follicles")

    logger.info(f"Total detected: {len(detected_results)} potential follicles")

    return detected_results


# ============================================
# FastAPI Routes
# ============================================

@app.get('/health')
async def health():
    """Health check endpoint."""
    return {
        'status': 'ok',
        'server': 'blob-detector',
        'version': '2.0.0',
        'framework': 'fastapi',
        'min_annotations': MIN_ANNOTATIONS_FOR_DETECTION,
        'parallel_detection': PARALLEL_BLOB_AVAILABLE,
        'parallel_threshold_pixels': PARALLEL_DETECTION_PIXEL_THRESHOLD,
        'gpu_available': GPU_AVAILABLE,
        'cuda_toolkit_installed': _cuda_toolkit_installed,
        'cuda_dll_setup': _cuda_dll_setup_success,
        'gpu_import_error': _gpu_import_error,
        'parallel_import_error': _parallel_import_error,
    }


@app.get('/gpu-info')
async def gpu_info():
    """Get GPU backend information."""
    manager = get_gpu_backend()

    if manager:
        status = manager.get_status()
        return {
            'available': True,
            'active_backend': status['active_backend'],
            'device_name': status['device_name'],
            'backends': {
                'cuda': status['available']['cuda'],
                'mps': status['available']['mps'],
                'cpu': True,
            },
            'details': status.get('backends_info', {}),
            'parallel_detection': PARALLEL_BLOB_AVAILABLE,
            'parallel_threshold_pixels': PARALLEL_DETECTION_PIXEL_THRESHOLD,
        }
    else:
        return {
            'available': False,
            'active_backend': 'cpu',
            'device_name': 'CPU (OpenCV)',
            'backends': {
                'cuda': False,
                'mps': False,
                'cpu': True,
            },
            'details': {},
            'parallel_detection': PARALLEL_BLOB_AVAILABLE,
            'parallel_threshold_pixels': PARALLEL_DETECTION_PIXEL_THRESHOLD,
        }


@app.post('/set-image')
async def set_image(req: SetImageRequest):
    """
    Set an image for a new session.

    Returns:
        - sessionId: Unique session identifier
        - width: Image width
        - height: Image height
    """
    # Decode image
    image = decode_image(req.image)
    if image is None:
        raise HTTPException(status_code=400, detail='Failed to decode image')

    # Create session
    session_id = str(uuid.uuid4())

    sessions[session_id] = {
        'image': image,
        'width': image.shape[1],
        'height': image.shape[0],
        'annotations': []  # User-drawn annotations for learning
    }

    logger.info(f"Created session {session_id} for image {image.shape[1]}x{image.shape[0]}")

    return {
        'sessionId': session_id,
        'width': image.shape[1],
        'height': image.shape[0]
    }


@app.post('/add-annotation')
async def add_annotation(req: AddAnnotationRequest):
    """
    Add a user annotation for size learning.

    Returns:
        - annotationCount: Total annotations in session
        - canDetect: Whether minimum annotations reached
    """
    if req.sessionId not in sessions:
        raise HTTPException(status_code=400, detail='Invalid session')

    # Add annotation
    session = sessions[req.sessionId]
    session['annotations'].append({
        'x': req.x,
        'y': req.y,
        'width': req.width,
        'height': req.height
    })

    annotation_count = len(session['annotations'])
    can_detect = annotation_count >= MIN_ANNOTATIONS_FOR_DETECTION

    logger.info(f"Session {req.sessionId}: {annotation_count} annotations, can_detect={can_detect}")

    return {
        'annotationCount': annotation_count,
        'canDetect': can_detect,
        'minRequired': MIN_ANNOTATIONS_FOR_DETECTION
    }


@app.post('/sync-annotations')
async def sync_annotations(req: SyncAnnotationsRequest):
    """
    Sync all annotations from frontend (replaces session annotations).

    Returns:
        - annotationCount: Total annotations in session
        - canDetect: Whether minimum annotations reached
    """
    if req.sessionId not in sessions:
        raise HTTPException(status_code=400, detail='Invalid session')

    # Replace session annotations
    session = sessions[req.sessionId]
    session['annotations'] = []

    for ann in req.annotations:
        session['annotations'].append({
            'x': ann.x,
            'y': ann.y,
            'width': ann.width,
            'height': ann.height
        })

    annotation_count = len(session['annotations'])
    can_detect = annotation_count >= MIN_ANNOTATIONS_FOR_DETECTION

    logger.info(f"Session {req.sessionId}: synced {annotation_count} annotations, can_detect={can_detect}")

    return {
        'annotationCount': annotation_count,
        'canDetect': can_detect,
        'minRequired': MIN_ANNOTATIONS_FOR_DETECTION
    }


@app.post('/get-learned-stats')
async def get_learned_stats(req: SessionRequest):
    """
    Get learned statistics from annotations for the Learn from Selection dialog.

    Returns:
        - stats: Object with examplesAnalyzed, minWidth, maxWidth, minHeight, maxHeight,
                 minAspectRatio, maxAspectRatio, meanIntensity
        - canDetect: Whether minimum annotations reached
    """
    if req.sessionId not in sessions:
        raise HTTPException(status_code=400, detail='Invalid session')

    session = sessions[req.sessionId]
    annotations = session['annotations']
    image = session.get('image')

    stats = calculate_learned_stats(annotations, image)
    can_detect = len(annotations) >= MIN_ANNOTATIONS_FOR_DETECTION

    logger.info(f"Session {req.sessionId}: learned stats = {stats}")

    return {
        'stats': stats,
        'canDetect': can_detect,
        'minRequired': MIN_ANNOTATIONS_FOR_DETECTION
    }


@app.post('/get-annotation-count')
async def get_annotation_count(req: SessionRequest):
    """
    Get the current annotation count for a session.

    Returns:
        - annotationCount: Total annotations in session
        - canDetect: Whether minimum annotations reached
    """
    if req.sessionId not in sessions:
        raise HTTPException(status_code=400, detail='Invalid session')

    session = sessions[req.sessionId]
    annotation_count = len(session['annotations'])

    return {
        'annotationCount': annotation_count,
        'canDetect': annotation_count >= MIN_ANNOTATIONS_FOR_DETECTION,
        'minRequired': MIN_ANNOTATIONS_FOR_DETECTION
    }


@app.post('/blob-detect')
async def blob_detect(req: BlobDetectRequest):
    """
    Run BLOB detection on the session image.

    Can work in two modes:
    1. With annotations (learns size from them)
    2. With manual settings (uses provided size range)

    Returns:
        - detections: Array of {x, y, width, height, confidence, method}
        - count: Number of detections
    """
    if req.sessionId not in sessions:
        raise HTTPException(status_code=400, detail='Invalid session')

    session = sessions[req.sessionId]
    annotations = session['annotations']
    settings = req.settings or {}

    # Check if we have either enough annotations OR manual size settings
    has_manual_settings = settings.get('minWidth') and settings.get('maxWidth')
    has_enough_annotations = len(annotations) >= MIN_ANNOTATIONS_FOR_DETECTION

    if not has_manual_settings and not has_enough_annotations:
        raise HTTPException(
            status_code=400,
            detail={
                'error': f'Either provide manual size settings or at least {MIN_ANNOTATIONS_FOR_DETECTION} annotations',
                'annotationCount': len(annotations),
                'minRequired': MIN_ANNOTATIONS_FOR_DETECTION
            }
        )

    try:
        # Run detection with settings
        detections = detect_blobs(session['image'], annotations, settings)

        return {
            'detections': detections,
            'count': len(detections),
            'learnedSize': estimate_follicle_size(annotations)
        }

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Detection failed: {e}\n{error_traceback}")
        # Include more context in the error for debugging
        error_detail = {
            'error': str(e),
            'type': type(e).__name__,
            'gpu_available': GPU_AVAILABLE,
            'parallel_available': PARALLEL_BLOB_AVAILABLE,
        }
        raise HTTPException(status_code=500, detail=error_detail)


@app.post('/crop-region')
async def crop_region(req: CropRegionRequest):
    """
    Extract a cropped region from the session image as base64.
    Used for YOLO keypoint inference on detected blobs.

    Returns:
        - image: Base64 encoded crop image
        - width: Crop width
        - height: Crop height
    """
    if req.sessionId not in sessions:
        raise HTTPException(status_code=400, detail='Invalid session')

    session = sessions[req.sessionId]
    image = session['image']
    img_height, img_width = image.shape[:2]

    # Ensure coordinates are within image bounds
    x = max(0, int(req.x))
    y = max(0, int(req.y))
    width = min(int(req.width), img_width - x)
    height = min(int(req.height), img_height - y)

    if width <= 0 or height <= 0:
        raise HTTPException(status_code=400, detail='Invalid crop region')

    # Extract crop
    crop = image[y:y+height, x:x+width]

    # Encode as base64 PNG
    _, buffer = cv2.imencode('.png', crop)
    crop_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        'image': crop_base64,
        'width': width,
        'height': height
    }


@app.post('/clear-session')
async def clear_session(req: SessionRequest):
    """
    Clear a session and free memory.

    Returns:
        - success: True if session was cleared
    """
    if req.sessionId and req.sessionId in sessions:
        del sessions[req.sessionId]
        logger.info(f"Cleared session {req.sessionId}")

    return {'success': True}


@app.post('/shutdown')
async def shutdown():
    """Shutdown the server gracefully."""
    global server_instance
    logger.info("Shutting down server...")

    # Clear all sessions
    sessions.clear()

    # Schedule shutdown
    if server_instance:
        server_instance.should_exit = True

    return {'status': 'shutting down'}


# ============================================
# YOLO Keypoint Training Endpoints
# ============================================

# Import YOLO keypoint service
try:
    from yolo_keypoint_service import (
        get_yolo_keypoint_service,
        TrainingConfig,
        TrainingProgress,
        YOLO_AVAILABLE as YOLO_KEYPOINT_AVAILABLE
    )
except ImportError:
    YOLO_KEYPOINT_AVAILABLE = False
    get_yolo_keypoint_service = None
    TrainingConfig = None
    TrainingProgress = None
    logger.warning("YOLO keypoint service not available")

# Import YOLO detection service
try:
    from yolo_detection_service import (
        get_yolo_detection_service,
        DetectionTrainingConfig,
        DetectionTrainingProgress,
        YOLO_AVAILABLE as YOLO_DETECTION_AVAILABLE
    )
except ImportError:
    YOLO_DETECTION_AVAILABLE = False
    get_yolo_detection_service = None
    DetectionTrainingConfig = None
    DetectionTrainingProgress = None
    logger.warning("YOLO detection service not available")

# SSE support for training progress
try:
    from sse_starlette.sse import EventSourceResponse
    SSE_AVAILABLE = True
except ImportError:
    SSE_AVAILABLE = False
    EventSourceResponse = None
    logger.warning("sse-starlette not installed. Training progress streaming will not be available.")

# Training progress queues (for SSE) - shared between keypoint and detection
training_progress_queues: Dict[str, asyncio.Queue] = {}

# Detection training progress queues (for SSE)
detection_training_queues: Dict[str, asyncio.Queue] = {}


class ValidateDatasetRequest(BaseModel):
    datasetPath: str


class StartTrainingRequest(BaseModel):
    datasetPath: str
    config: Optional[Dict[str, Any]] = None
    modelName: Optional[str] = None


class LoadModelRequest(BaseModel):
    modelPath: str


class PredictRequest(BaseModel):
    imageData: str  # base64 encoded


class ExportONNXRequest(BaseModel):
    modelPath: str
    outputPath: str


@app.post('/yolo-keypoint/validate-dataset')
async def yolo_validate_dataset(req: ValidateDatasetRequest):
    """
    Validate a YOLO keypoint dataset structure.

    Returns validation result with image/label counts and any errors.
    """
    if not YOLO_KEYPOINT_AVAILABLE or not get_yolo_keypoint_service:
        raise HTTPException(status_code=503, detail='YOLO keypoint service not available')

    service = get_yolo_keypoint_service()
    result = service.validate_dataset(req.datasetPath)

    return result.to_dict()


@app.post('/yolo-keypoint/train/start')
async def yolo_start_training(req: StartTrainingRequest):
    """
    Start YOLO keypoint model training.

    Returns job_id for progress tracking.
    """
    if not YOLO_KEYPOINT_AVAILABLE or not get_yolo_keypoint_service:
        raise HTTPException(status_code=503, detail='YOLO keypoint service not available')

    service = get_yolo_keypoint_service()

    # Parse config
    config_dict = req.config or {}
    config = TrainingConfig(
        model_size=config_dict.get('modelSize', 'n'),
        epochs=config_dict.get('epochs', 100),
        img_size=config_dict.get('imgSize', 640),
        batch_size=config_dict.get('batchSize', 16),
        patience=config_dict.get('patience', 50),
        device=config_dict.get('device', 'auto'),
    )

    # Create progress queue for this job
    progress_queue: asyncio.Queue = asyncio.Queue()

    # Capture the event loop BEFORE starting the training thread
    # This allows the callback to safely post updates from the training thread
    main_loop = asyncio.get_running_loop()

    def progress_callback(progress: TrainingProgress):
        # Put progress in queue (non-blocking) using the captured main loop
        try:
            logger.info(f"Progress callback: epoch {progress.epoch}/{progress.total_epochs}, status={progress.status}")
            asyncio.run_coroutine_threadsafe(
                progress_queue.put(progress.to_dict()),
                main_loop
            )
            logger.info(f"Progress queued successfully, queue size now: {progress_queue.qsize()}")
        except Exception as e:
            logger.warning(f"Failed to queue progress update: {e}")

    # Start training
    job_id, _ = await service.train(
        dataset_path=req.datasetPath,
        config=config,
        progress_callback=progress_callback,
        model_name=req.modelName
    )

    if job_id:
        training_progress_queues[job_id] = progress_queue

    return {
        'jobId': job_id,
        'status': 'started',
        'config': config_dict
    }


@app.get('/yolo-keypoint/train/progress/{job_id}')
async def yolo_training_progress(job_id: str):
    """
    Stream training progress via Server-Sent Events.
    """
    if not SSE_AVAILABLE:
        raise HTTPException(status_code=503, detail='SSE not available. Install sse-starlette.')

    if job_id not in training_progress_queues:
        raise HTTPException(status_code=404, detail='Training job not found')

    queue = training_progress_queues[job_id]

    async def event_generator():
        try:
            while True:
                try:
                    # Wait for progress update with timeout
                    progress = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield {
                        'event': 'progress',
                        'data': json.dumps(progress)
                    }

                    # Check if training is done
                    status = progress.get('status', '')
                    if status in ('completed', 'failed', 'stopped'):
                        break

                except asyncio.TimeoutError:
                    # Send heartbeat to keep connection alive
                    yield {
                        'event': 'heartbeat',
                        'data': json.dumps({'status': 'alive'})
                    }

        finally:
            # Cleanup queue
            if job_id in training_progress_queues:
                del training_progress_queues[job_id]

    return EventSourceResponse(event_generator())


@app.post('/yolo-keypoint/train/stop/{job_id}')
async def yolo_stop_training(job_id: str):
    """
    Stop a running training job.
    """
    if not YOLO_KEYPOINT_AVAILABLE or not get_yolo_keypoint_service:
        raise HTTPException(status_code=503, detail='YOLO keypoint service not available')

    service = get_yolo_keypoint_service()
    success = service.stop_training(job_id)

    return {
        'success': success,
        'jobId': job_id
    }


@app.get('/yolo-keypoint/models')
async def yolo_list_models():
    """
    List all trained YOLO keypoint models.
    """
    if not YOLO_KEYPOINT_AVAILABLE or not get_yolo_keypoint_service:
        raise HTTPException(status_code=503, detail='YOLO keypoint service not available')

    service = get_yolo_keypoint_service()
    models = service.list_models()

    return {
        'models': [m.to_dict() for m in models]
    }


@app.post('/yolo-keypoint/load-model')
async def yolo_load_model(req: LoadModelRequest):
    """
    Load a trained model for inference.
    """
    if not YOLO_KEYPOINT_AVAILABLE or not get_yolo_keypoint_service:
        raise HTTPException(status_code=503, detail='YOLO keypoint service not available')

    service = get_yolo_keypoint_service()
    success = service.load_model(req.modelPath)

    return {
        'success': success,
        'modelPath': req.modelPath
    }


@app.post('/yolo-keypoint/predict')
async def yolo_predict(req: PredictRequest):
    """
    Run keypoint prediction on a cropped follicle image.
    """
    if not YOLO_KEYPOINT_AVAILABLE or not get_yolo_keypoint_service:
        raise HTTPException(status_code=503, detail='YOLO keypoint service not available')

    service = get_yolo_keypoint_service()

    # Decode base64 image
    try:
        image_data = req.imageData
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Failed to decode image: {e}')

    # Run prediction
    result = service.predict(image_bytes)

    if result is None:
        return {
            'success': False,
            'prediction': None,
            'message': 'No keypoints detected'
        }

    return {
        'success': True,
        'prediction': result.to_dict()
    }


class BatchPredictRequest(BaseModel):
    images: List[str]  # List of base64-encoded images


@app.post('/yolo-keypoint/predict-batch')
async def yolo_predict_batch(req: BatchPredictRequest):
    """
    Run keypoint prediction on multiple cropped follicle images in batch.

    Batch inference is significantly faster on GPU as it processes
    multiple images in parallel.
    """
    if not YOLO_KEYPOINT_AVAILABLE or not get_yolo_keypoint_service:
        raise HTTPException(status_code=503, detail='YOLO keypoint service not available')

    service = get_yolo_keypoint_service()

    # Decode all base64 images
    images_bytes = []
    for image_data in req.images:
        try:
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            images_bytes.append(base64.b64decode(image_data))
        except Exception as e:
            logger.warning(f'Failed to decode image in batch: {e}')
            images_bytes.append(None)

    # Filter out None values but keep track of indices
    valid_indices = [i for i, img in enumerate(images_bytes) if img is not None]
    valid_images = [images_bytes[i] for i in valid_indices]

    if len(valid_images) == 0:
        return {
            'success': False,
            'predictions': [None] * len(req.images),
            'message': 'No valid images to process'
        }

    # Run batch prediction
    results = service.predict_batch(valid_images)

    # Map results back to original indices
    predictions = [None] * len(req.images)
    for result_idx, orig_idx in enumerate(valid_indices):
        if result_idx < len(results) and results[result_idx] is not None:
            predictions[orig_idx] = results[result_idx].to_dict()

    success_count = sum(1 for p in predictions if p is not None)

    return {
        'success': True,
        'predictions': predictions,
        'total': len(req.images),
        'successful': success_count
    }


@app.post('/yolo-keypoint/export-onnx')
async def yolo_export_onnx(req: ExportONNXRequest):
    """
    Export a trained model to ONNX format.
    """
    if not YOLO_KEYPOINT_AVAILABLE or not get_yolo_keypoint_service:
        raise HTTPException(status_code=503, detail='YOLO keypoint service not available')

    service = get_yolo_keypoint_service()
    output_path = service.export_onnx(req.modelPath, req.outputPath)

    return {
        'success': output_path is not None,
        'outputPath': output_path
    }


@app.delete('/yolo-keypoint/models/{model_id}')
async def yolo_delete_model(model_id: str):
    """
    Delete a trained model.
    """
    if not YOLO_KEYPOINT_AVAILABLE or not get_yolo_keypoint_service:
        raise HTTPException(status_code=503, detail='YOLO keypoint service not available')

    service = get_yolo_keypoint_service()
    success = service.delete_model(model_id)

    return {
        'success': success,
        'modelId': model_id
    }


@app.get('/yolo-keypoint/status')
async def yolo_keypoint_status():
    """
    Get YOLO keypoint service status.
    """
    return {
        'available': YOLO_KEYPOINT_AVAILABLE,
        'sseAvailable': SSE_AVAILABLE,
        'activeTrainingJobs': len(training_progress_queues),
        'jobIds': list(training_progress_queues.keys()),
        'queueSizes': {k: v.qsize() for k, v in training_progress_queues.items()}
    }


@app.post('/yolo-keypoint/clear-gpu-memory')
async def yolo_keypoint_clear_gpu_memory():
    """
    Clear GPU memory after prediction tasks complete.
    Runs garbage collection and empties CUDA cache.
    """
    if not YOLO_KEYPOINT_AVAILABLE or not get_yolo_keypoint_service:
        return {"success": False, "error": "YOLO keypoint service not available"}

    service = get_yolo_keypoint_service()
    result = service.clear_gpu_memory()
    return result


@app.get('/yolo-keypoint/system-info')
async def yolo_system_info():
    """
    Get system information for training (CPU/GPU, memory, etc.)
    """
    import platform
    import psutil

    info = {
        'python_version': platform.python_version(),
        'platform': platform.system(),
        'cpu_count': psutil.cpu_count(),
        'cpu_percent': psutil.cpu_percent(interval=0.1),
        'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'memory_used_gb': round(psutil.virtual_memory().used / (1024**3), 2),
        'memory_percent': psutil.virtual_memory().percent,
        'device': 'cpu',
        'device_name': platform.processor() or 'Unknown CPU',
        'cuda_available': False,
        'mps_available': False,
        'torch_version': None,
        'gpu_memory_total_gb': None,
        'gpu_memory_used_gb': None,
    }

    try:
        import torch
        info['torch_version'] = torch.__version__

        if torch.cuda.is_available():
            info['cuda_available'] = True
            info['device'] = 'cuda'
            info['device_name'] = torch.cuda.get_device_name(0)
            # Get GPU memory
            gpu_props = torch.cuda.get_device_properties(0)
            info['gpu_memory_total_gb'] = round(gpu_props.total_memory / (1024**3), 2)
            # Current memory usage
            info['gpu_memory_used_gb'] = round(torch.cuda.memory_allocated(0) / (1024**3), 2)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            info['mps_available'] = True
            info['device'] = 'mps'
            info['device_name'] = 'Apple Silicon GPU'
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Error getting torch info: {e}")

    return info


@app.get('/yolo-keypoint/check-tensorrt')
async def yolo_keypoint_check_tensorrt():
    """
    Check if TensorRT is available on this system for keypoint inference.

    Returns:
        Dict with 'available' bool and 'version' string (or None)
    """
    if not YOLO_KEYPOINT_AVAILABLE or not get_yolo_keypoint_service:
        raise HTTPException(status_code=503, detail='YOLO keypoint service not available')

    service = get_yolo_keypoint_service()
    return service.check_tensorrt_available()


class KeypointExportTensorRTRequest(BaseModel):
    modelPath: str
    outputPath: Optional[str] = None
    half: Optional[bool] = True
    imgsz: Optional[int] = 640


@app.post('/yolo-keypoint/export-tensorrt')
async def yolo_keypoint_export_tensorrt(req: KeypointExportTensorRTRequest):
    """
    Export a PyTorch keypoint model to TensorRT engine format.

    TensorRT provides GPU-optimized inference for faster keypoint prediction
    on NVIDIA GPUs. The exported .engine file is GPU-architecture
    specific and not portable between different GPU types.

    Body:
        modelPath: str - Path to source .pt model
        outputPath: str (optional) - Path for output .engine file
        half: bool (default True) - Use FP16 precision
        imgsz: int (default 640) - Input image size for the engine

    Returns:
        Dict with 'success' bool, 'engine_path' string, or 'error' string
    """
    if not YOLO_KEYPOINT_AVAILABLE or not get_yolo_keypoint_service:
        raise HTTPException(status_code=503, detail='YOLO keypoint service not available')

    service = get_yolo_keypoint_service()
    result = service.export_tensorrt(
        model_path=req.modelPath,
        output_path=req.outputPath,
        half=req.half if req.half is not None else True,
        imgsz=req.imgsz if req.imgsz is not None else 640
    )

    if not result.get('success'):
        raise HTTPException(status_code=400, detail=result.get('error', 'Export failed'))

    return result


# ============================================
# YOLO Detection Training Endpoints
# ============================================

class DetectionPredictRequest(BaseModel):
    imageData: str  # base64 encoded
    confidenceThreshold: Optional[float] = 0.5


class DetectionPredictTiledRequest(BaseModel):
    imageData: str  # base64 encoded
    confidenceThreshold: Optional[float] = 0.5
    tileSize: Optional[int] = 1024  # Match training tile size
    overlap: Optional[int] = 128  # Overlap between tiles
    nmsThreshold: Optional[float] = 0.5  # IoU threshold for NMS
    scaleFactor: Optional[float] = 1.0  # Upscale factor for smaller objects


class DetectionValidateDatasetRequest(BaseModel):
    datasetPath: str


class DetectionStartTrainingRequest(BaseModel):
    datasetPath: str
    config: Optional[Dict[str, Any]] = None
    modelName: Optional[str] = None


class DetectionLoadModelRequest(BaseModel):
    modelPath: str


class DetectionExportONNXRequest(BaseModel):
    modelPath: str
    outputPath: str


@app.post('/yolo-detect/validate-dataset')
async def yolo_detect_validate_dataset(req: DetectionValidateDatasetRequest):
    """
    Validate a YOLO detection dataset structure.

    Returns validation result with image/label counts and any errors.
    """
    if not YOLO_DETECTION_AVAILABLE or not get_yolo_detection_service:
        raise HTTPException(status_code=503, detail='YOLO detection service not available')

    service = get_yolo_detection_service()
    result = service.validate_dataset(req.datasetPath)

    return result.to_dict()


@app.post('/yolo-detect/train/start')
async def yolo_detect_start_training(req: DetectionStartTrainingRequest):
    """
    Start YOLO detection model training.

    Returns job_id for progress tracking.
    """
    if not YOLO_DETECTION_AVAILABLE or not get_yolo_detection_service:
        raise HTTPException(status_code=503, detail='YOLO detection service not available')

    service = get_yolo_detection_service()

    # Parse config
    config_dict = req.config or {}
    config = DetectionTrainingConfig(
        model_size=config_dict.get('modelSize', 'n'),
        epochs=config_dict.get('epochs', 100),
        img_size=config_dict.get('imgSize', 640),
        batch_size=config_dict.get('batchSize', 16),
        patience=config_dict.get('patience', 50),
        device=config_dict.get('device', 'auto'),
        resume_from=config_dict.get('resumeFrom'),  # Model ID to resume from
    )

    # Create progress queue for this job
    progress_queue: asyncio.Queue = asyncio.Queue()

    # Capture the event loop BEFORE starting the training thread
    main_loop = asyncio.get_running_loop()

    def progress_callback(progress: DetectionTrainingProgress):
        try:
            logger.info(f"Detection training progress: epoch {progress.epoch}/{progress.total_epochs}, status={progress.status}")
            asyncio.run_coroutine_threadsafe(
                progress_queue.put(progress.to_dict()),
                main_loop
            )
        except Exception as e:
            logger.warning(f"Failed to queue detection progress update: {e}")

    # Start training
    job_id, _ = await service.train(
        dataset_path=req.datasetPath,
        config=config,
        progress_callback=progress_callback,
        model_name=req.modelName
    )

    if job_id:
        detection_training_queues[job_id] = progress_queue

    return {
        'jobId': job_id,
        'status': 'started',
        'config': config_dict
    }


@app.get('/yolo-detect/train/progress/{job_id}')
async def yolo_detect_training_progress(job_id: str):
    """
    Stream detection training progress via Server-Sent Events.
    """
    if not SSE_AVAILABLE:
        raise HTTPException(status_code=503, detail='SSE not available. Install sse-starlette.')

    if job_id not in detection_training_queues:
        raise HTTPException(status_code=404, detail='Training job not found')

    queue = detection_training_queues[job_id]

    async def event_generator():
        try:
            while True:
                try:
                    # Wait for progress update with timeout
                    progress = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield {
                        'event': 'progress',
                        'data': json.dumps(progress)
                    }

                    # Check if training is done
                    status = progress.get('status', '')
                    if status in ('completed', 'failed', 'stopped'):
                        break

                except asyncio.TimeoutError:
                    # Send heartbeat to keep connection alive
                    yield {
                        'event': 'heartbeat',
                        'data': json.dumps({'status': 'alive'})
                    }

        finally:
            # Cleanup queue
            if job_id in detection_training_queues:
                del detection_training_queues[job_id]

    return EventSourceResponse(event_generator())


@app.post('/yolo-detect/train/stop/{job_id}')
async def yolo_detect_stop_training(job_id: str):
    """
    Stop a running detection training job.
    """
    if not YOLO_DETECTION_AVAILABLE or not get_yolo_detection_service:
        raise HTTPException(status_code=503, detail='YOLO detection service not available')

    service = get_yolo_detection_service()
    success = service.stop_training(job_id)

    return {
        'success': success,
        'jobId': job_id
    }


@app.get('/yolo-detect/models')
async def yolo_detect_list_models():
    """
    List all trained YOLO detection models.
    """
    if not YOLO_DETECTION_AVAILABLE or not get_yolo_detection_service:
        raise HTTPException(status_code=503, detail='YOLO detection service not available')

    service = get_yolo_detection_service()
    models = service.list_models()

    return {
        'models': [m.to_dict() for m in models]
    }


@app.post('/yolo-detect/load-model')
async def yolo_detect_load_model(req: DetectionLoadModelRequest):
    """
    Load a trained detection model for inference.
    """
    if not YOLO_DETECTION_AVAILABLE or not get_yolo_detection_service:
        raise HTTPException(status_code=503, detail='YOLO detection service not available')

    service = get_yolo_detection_service()
    success = service.load_model(req.modelPath)

    return {
        'success': success,
        'modelPath': req.modelPath
    }


@app.post('/yolo-detect/predict')
async def yolo_detect_predict(req: DetectionPredictRequest):
    """
    Run detection prediction on a full image.
    Returns list of bounding boxes with confidence.
    """
    if not YOLO_DETECTION_AVAILABLE or not get_yolo_detection_service:
        raise HTTPException(status_code=503, detail='YOLO detection service not available')

    service = get_yolo_detection_service()

    # Run prediction
    results = service.predict_base64(
        req.imageData,
        confidence_threshold=req.confidenceThreshold or 0.5
    )

    return {
        'success': True,
        'detections': [r.to_dict() for r in results],
        'count': len(results),
        'backend': service.get_loaded_model_backend()
    }


@app.post('/yolo-detect/predict-tiled')
async def yolo_detect_predict_tiled(req: DetectionPredictTiledRequest):
    """
    Run tiled detection prediction on a large image.

    Splits the image into overlapping tiles matching the training tile size,
    runs inference on each tile, and merges results with NMS.

    This is essential for detecting small objects in large images when the model
    was trained on tiles rather than full images.
    """
    if not YOLO_DETECTION_AVAILABLE or not get_yolo_detection_service:
        raise HTTPException(status_code=503, detail='YOLO detection service not available')

    service = get_yolo_detection_service()

    # Run tiled prediction
    results = service.predict_tiled_base64(
        req.imageData,
        confidence_threshold=req.confidenceThreshold or 0.5,
        tile_size=req.tileSize or 1024,
        overlap=req.overlap or 128,
        nms_threshold=req.nmsThreshold or 0.5,
        scale_factor=req.scaleFactor or 1.0
    )

    return {
        'success': True,
        'detections': [r.to_dict() for r in results],
        'count': len(results),
        'method': 'tiled',
        'tileSize': req.tileSize or 1024,
        'overlap': req.overlap or 128,
        'scaleFactor': req.scaleFactor or 1.0,
        'backend': service.get_loaded_model_backend()
    }


@app.post('/yolo-detect/export-onnx')
async def yolo_detect_export_onnx(req: DetectionExportONNXRequest):
    """
    Export a trained detection model to ONNX format.
    """
    if not YOLO_DETECTION_AVAILABLE or not get_yolo_detection_service:
        raise HTTPException(status_code=503, detail='YOLO detection service not available')

    service = get_yolo_detection_service()
    output_path = service.export_onnx(req.modelPath, req.outputPath)

    return {
        'success': output_path is not None,
        'outputPath': output_path
    }


@app.delete('/yolo-detect/models/{model_id}')
async def yolo_detect_delete_model(model_id: str):
    """
    Delete a trained detection model.
    """
    if not YOLO_DETECTION_AVAILABLE or not get_yolo_detection_service:
        raise HTTPException(status_code=503, detail='YOLO detection service not available')

    service = get_yolo_detection_service()
    success = service.delete_model(model_id)

    return {
        'success': success,
        'modelId': model_id
    }


@app.get('/yolo-detect/status')
async def yolo_detect_status():
    """
    Get YOLO detection service status.
    """
    loaded_model = None
    if YOLO_DETECTION_AVAILABLE and get_yolo_detection_service:
        service = get_yolo_detection_service()
        loaded_model = service.get_loaded_model_path()

    return {
        'available': YOLO_DETECTION_AVAILABLE,
        'sseAvailable': SSE_AVAILABLE,
        'activeTrainingJobs': len(detection_training_queues),
        'jobIds': list(detection_training_queues.keys()),
        'loadedModel': loaded_model
    }


@app.get('/yolo-detect/resumable-models')
async def yolo_detect_resumable_models():
    """
    Get list of models that can be resumed (have last.pt and incomplete training).
    """
    if not YOLO_DETECTION_AVAILABLE or not get_yolo_detection_service:
        raise HTTPException(status_code=503, detail='YOLO detection service not available')

    service = get_yolo_detection_service()
    models = service.get_resumable_models()

    return {
        'models': models
    }


@app.get('/yolo-detect/check-tensorrt')
async def yolo_detect_check_tensorrt():
    """
    Check if TensorRT is available on this system.

    Returns:
        Dict with 'available' bool and 'version' string (or None)
    """
    if not YOLO_DETECTION_AVAILABLE or not get_yolo_detection_service:
        raise HTTPException(status_code=503, detail='YOLO detection service not available')

    service = get_yolo_detection_service()
    return service.check_tensorrt_available()


@app.post('/yolo-detect/export-tensorrt')
async def yolo_detect_export_tensorrt(request: Request):
    """
    Export a PyTorch model to TensorRT engine format.

    TensorRT provides GPU-optimized inference for faster detection
    on NVIDIA GPUs. The exported .engine file is GPU-architecture
    specific and not portable between different GPU types.

    Body:
        modelPath: str - Path to source .pt model
        outputPath: str (optional) - Path for output .engine file
        half: bool (default True) - Use FP16 precision
        imgsz: int (default 640) - Input image size for the engine

    Returns:
        Dict with 'success' bool and 'engine_path' string
    """
    if not YOLO_DETECTION_AVAILABLE or not get_yolo_detection_service:
        raise HTTPException(status_code=503, detail='YOLO detection service not available')

    body = await request.json()
    model_path = body.get('modelPath')
    output_path = body.get('outputPath')
    half = body.get('half', True)
    imgsz = body.get('imgsz', 640)

    if not model_path:
        raise HTTPException(status_code=400, detail='modelPath is required')

    service = get_yolo_detection_service()
    result = service.export_tensorrt(model_path, output_path, half, imgsz)

    if not result.get('success'):
        raise HTTPException(status_code=500, detail=result.get('error', 'TensorRT export failed'))

    return result


@app.post('/yolo-detect/clear-gpu-memory')
async def yolo_detect_clear_gpu_memory():
    """
    Clear GPU memory after detection tasks complete.
    Runs garbage collection and empties CUDA cache.
    """
    if not YOLO_DETECTION_AVAILABLE or not get_yolo_detection_service:
        return {"success": False, "error": "YOLO detection service not available"}

    service = get_yolo_detection_service()
    result = service.clear_gpu_memory()
    return result


# ============================================
# Main Entry Point
# ============================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BLOB Detection Server (FastAPI)')
    parser.add_argument('--port', type=int, default=5555, help='Port to run on')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    logger.info(f"Starting BLOB Detection Server (FastAPI) on {args.host}:{args.port}")
    logger.info(f"Minimum annotations required: {MIN_ANNOTATIONS_FOR_DETECTION}")
    logger.info(f"Parallel blob detection available: {PARALLEL_BLOB_AVAILABLE}")

    # Configure uvicorn
    config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        log_level="debug" if args.debug else "info",
        access_log=True
    )
    server_instance = uvicorn.Server(config)

    # Run the server
    server_instance.run()
