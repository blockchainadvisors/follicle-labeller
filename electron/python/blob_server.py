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
import base64
import io
import logging
import os
import signal
import sys
import uuid
from typing import Any, Dict, List, Optional, Tuple

# Setup NVIDIA CUDA DLL paths before importing GPU libraries (Windows only)
if sys.platform == 'win32':
    try:
        # Find the venv site-packages nvidia directory
        import site
        nvidia_dll_paths = []
        for site_path in site.getsitepackages():
            nvidia_base = os.path.join(site_path, 'nvidia')
            if os.path.exists(nvidia_base):
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

        # Also prepend to PATH for broader compatibility
        if nvidia_dll_paths:
            current_path = os.environ.get('PATH', '')
            new_paths = os.pathsep.join(nvidia_dll_paths)
            os.environ['PATH'] = new_paths + os.pathsep + current_path
    except Exception:
        pass  # Ignore errors, GPU will just not be available

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import GPU backend manager
try:
    from gpu_backend import get_gpu_manager, GPUBackendManager
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    get_gpu_manager = None
    GPUBackendManager = None

# Import parallel blob detection
try:
    from parallel_blob_detect import parallel_blob_detect
    PARALLEL_BLOB_AVAILABLE = True
except ImportError:
    PARALLEL_BLOB_AVAILABLE = False
    parallel_blob_detect = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

    # Convert to grayscale
    if use_gpu:
        gray = backend.grayscale(image)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE for better contrast (if enabled)
    use_clahe = settings.get('useCLAHE', True)
    if use_clahe:
        clip_limit = settings.get('claheClipLimit', 3.0)
        tile_size = settings.get('claheTileSize', 8)
        if use_gpu:
            gray = backend.clahe(gray, clip_limit=clip_limit, tile_size=tile_size)
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

        # Use Gaussian blur + Otsu thresholding (better than adaptive for this use case)
        if use_gpu:
            blurred = backend.gaussian_blur(gray, kernel_size=5)
            thresh = backend.threshold_otsu(blurred, invert=True)
            # Morphological opening to remove noise and separate touching objects
            thresh = backend.morphological_open(thresh, kernel_size=3)
        else:
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # Morphological opening to remove noise and separate touching objects
            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

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
        'parallel_threshold_pixels': PARALLEL_DETECTION_PIXEL_THRESHOLD
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
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
