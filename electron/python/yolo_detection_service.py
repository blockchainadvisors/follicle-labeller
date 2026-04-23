#!/usr/bin/env python3
"""
YOLO Detection Training Service

Provides training and inference capabilities for YOLO11 detection models
to detect follicles as bounding boxes.

This service handles:
- Dataset validation
- Model training with progress streaming
- Model loading and inference
- ONNX export for deployment
- Trained models management
"""

import asyncio
import base64
import gc
import io
import json
import logging
import os
import shutil
import sys
import threading
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import ultralytics for YOLO training
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None
    logger.warning("Ultralytics not installed. YOLO training will not be available.")


# ============================================================================
# Dual-point video tracking: rigid consistency check thresholds
# ----------------------------------------------------------------------------
# Video tracking runs TWO independent NCC template matches per frame — one for
# the graft origin and one for the tip. The two matches must stay consistent
# with the initial origin-to-tip distance (hair grafts are rigid bodies). When
# the observed origin-to-tip distance deviates from the expected distance by
# more than `max(RIGID_ABS_TOLERANCE_PX, RIGID_REL_TOLERANCE * expected_dist)`,
# the lower-confidence match is rejected and its point is extrapolated from
# the trusted one using the source-image dx/dy relationship.
#
# Tune these if tracking is too aggressive or too forgiving.
# ============================================================================
RIGID_REL_TOLERANCE = 0.20    # 20% of expected distance
RIGID_ABS_TOLERANCE_PX = 5.0  # absolute floor, pixels


# ============================================================================
# Lucas-Kanade optical flow video tracking parameters
# ----------------------------------------------------------------------------
# The optical-flow tracker (new button alongside Template Match) runs pyramidal
# Lucas-Kanade on exactly two points per frame — the graft origin and tip —
# with a forward-backward error check for outlier rejection and a 2-point
# similarity transform for scale-jump validation.
#
# When LK fails on a point, the tracker falls back to a full NCC pyramid
# search on that point's cached source patch (same matcher used by the
# Template Match button), and if NCC also fails it extrapolates the failed
# point from the trusted one using the last known similarity transform.
# ============================================================================
LK_WIN_SIZE = (21, 21)           # OpenCV tutorial default; balances precision vs large-motion handling
LK_MAX_LEVEL = 3                 # 4-level pyramid; absorbs ~2^3 = 8x larger motions than single-level
LK_ITER_COUNT = 30               # max iterations per pyramid level
LK_ITER_EPS = 0.01               # convergence epsilon (pixel displacement)
LK_FB_ERROR_PX = 2.0             # forward-backward round-trip tolerance (pixels)
LK_MAX_SCALE_JUMP = 0.15         # reject frames where similarity-transform scale jumps >15% vs previous
LK_MIN_CONFIDENCE = 0.3          # below this per-point FB-derived confidence, consider the point failed

# ============================================================================
# Seek / Track / Cooldown state machine
# ----------------------------------------------------------------------------
# Wraps the LK tracker above with explicit state transitions. Origin is the
# paramount tracked point — if it's lost, the tracker enters a cooldown and
# then retries NCC seeding (scalp vs current frame). Tip-only loss keeps
# tracking and extrapolates tip from the previous similarity transform.
# ============================================================================
TRACK_COOLDOWN_DEFAULT_SEC = 5.0
TRACK_COOLDOWN_MIN_SEC = 0.5
TRACK_COOLDOWN_MAX_SEC = 60.0
LK_CONFIDENCE_THRESHOLD = 0.3    # per-point LK FB-derived confidence floor (origin must clear)


def _clamp_cooldown_sec(value: Optional[float]) -> float:
    """Clamp user-supplied cooldown to the safe range, defaulting on None/NaN."""
    if value is None:
        return TRACK_COOLDOWN_DEFAULT_SEC
    try:
        v = float(value)
    except (TypeError, ValueError):
        return TRACK_COOLDOWN_DEFAULT_SEC
    if not (v == v):  # NaN check
        return TRACK_COOLDOWN_DEFAULT_SEC
    return max(TRACK_COOLDOWN_MIN_SEC, min(TRACK_COOLDOWN_MAX_SEC, v))


@dataclass
class DetectionTrainingConfig:
    """Configuration for YOLO detection training."""
    model_size: str = 'n'  # 'n', 's', 'm', 'l', 'x'
    epochs: int = 100
    img_size: int = 640
    batch_size: int = 16
    patience: int = 50
    device: str = 'auto'  # 'auto', 'cuda', 'mps', 'cpu', or device number
    workers: int = 8
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: int = 3
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    close_mosaic: int = 10
    augment: bool = True
    cache: bool = False  # Cache images in RAM (faster but uses more memory)
    resume_from: Optional[str] = None  # Model ID to resume training from


@dataclass
class DetectionTrainingProgress:
    """Progress update during training."""
    status: str  # 'preparing', 'training', 'completed', 'failed', 'stopped'
    epoch: int = 0
    total_epochs: int = 0
    loss: float = 0.0
    box_loss: float = 0.0
    cls_loss: float = 0.0
    dfl_loss: float = 0.0
    metrics: Dict[str, float] = None
    eta: str = ''
    message: str = ''

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DetectionPrediction:
    """Single detection prediction result (bounding box)."""
    x: float  # Top-left x (absolute pixels)
    y: float  # Top-left y (absolute pixels)
    width: float  # Box width (pixels)
    height: float  # Box height (pixels)
    confidence: float
    class_id: int = 0
    class_name: str = 'follicle'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height,
            'confidence': self.confidence,
            'classId': self.class_id,
            'className': self.class_name
        }


@dataclass
class DetectionModelInfo:
    """Information about a trained detection model."""
    id: str
    name: str
    path: str
    created_at: str
    epochs_trained: int
    img_size: int
    metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict with camelCase keys for JavaScript compatibility."""
        return {
            'id': self.id,
            'name': self.name,
            'path': self.path,
            'createdAt': self.created_at,
            'epochsTrained': self.epochs_trained,
            'imgSize': self.img_size,
            'metrics': self.metrics,
        }

    def to_storage_dict(self) -> Dict[str, Any]:
        """Convert to dict with snake_case keys for file storage."""
        return asdict(self)


@dataclass
class DetectionDatasetValidation:
    """Result of dataset validation."""
    valid: bool
    train_images: int
    val_images: int
    train_labels: int
    val_labels: int
    errors: List[str]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class YOLODetectionService:
    """
    Service for YOLO detection model training and inference.

    Manages:
    - Dataset validation
    - Training jobs with progress callbacks
    - Model loading/unloading
    - Inference on full images
    - ONNX export
    - TensorRT export and inference
    - Model storage
    """

    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialize the YOLO detection service.

        Args:
            models_dir: Directory to store trained models. If None, checks
                       MODELS_BASE_DIR env var, then falls back to a
                       'models/detection' subdirectory next to this script.
        """
        if models_dir:
            self.models_dir = Path(models_dir)
        elif os.environ.get('MODELS_BASE_DIR'):
            # Use environment variable (set by Electron to persist models across updates)
            self.models_dir = Path(os.environ['MODELS_BASE_DIR']) / 'detection'
        else:
            # Default to models/detection next to script (dev mode)
            self.models_dir = Path(__file__).parent / 'models' / 'detection'

        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Currently loaded model for inference
        self._loaded_model: Optional['YOLO'] = None
        self._loaded_model_path: Optional[str] = None
        self._loaded_model_backend: str = 'pytorch'  # 'pytorch' or 'tensorrt'
        self._loaded_model_imgsz: int = 640  # Training image size for inference

        # Active training jobs
        self._training_jobs: Dict[str, dict] = {}

        # Cached tracking sessions for interactive single-follicle matching
        self._tracking_sessions: Dict[str, dict] = {}

        logger.info(f"YOLODetectionService initialized. Models dir: {self.models_dir}")

    def clear_gpu_memory(self) -> Dict[str, Any]:
        """
        Clear GPU memory by running garbage collection and emptying CUDA cache.

        This should be called after batch detection tasks complete to free
        up GPU memory for other operations.

        Returns:
            Dict with memory stats before and after cleanup
        """
        result = {
            "success": True,
            "memory_before": None,
            "memory_after": None,
            "memory_freed_mb": 0
        }

        try:
            import torch
            if torch.cuda.is_available():
                # Get memory stats before cleanup
                memory_before = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                memory_reserved_before = torch.cuda.memory_reserved() / (1024 * 1024)  # MB
                result["memory_before"] = {
                    "allocated_mb": round(memory_before, 2),
                    "reserved_mb": round(memory_reserved_before, 2)
                }

                # Multiple GC passes to clean up circular references
                for _ in range(3):
                    gc.collect()

                # Empty CUDA cache
                torch.cuda.empty_cache()

                # For TensorRT, also try to reset the memory allocator caching
                # This helps release fragmented memory blocks
                if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                    torch.cuda.reset_peak_memory_stats()

                # Synchronize to ensure cleanup is complete
                torch.cuda.synchronize()

                # Second round of cleanup after synchronization
                gc.collect()
                torch.cuda.empty_cache()

                # Get memory stats after cleanup
                memory_after = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                memory_reserved_after = torch.cuda.memory_reserved() / (1024 * 1024)  # MB
                result["memory_after"] = {
                    "allocated_mb": round(memory_after, 2),
                    "reserved_mb": round(memory_reserved_after, 2)
                }

                result["memory_freed_mb"] = round(memory_reserved_before - memory_reserved_after, 2)

                logger.info(f"GPU memory cleanup: freed {result['memory_freed_mb']:.1f}MB "
                           f"(reserved: {memory_reserved_before:.1f}MB -> {memory_reserved_after:.1f}MB, "
                           f"allocated: {memory_before:.1f}MB -> {memory_after:.1f}MB)")

            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # MPS (Apple Silicon) cleanup
                # MPS doesn't have detailed memory stats like CUDA
                gc.collect()

                # Empty MPS cache (available in PyTorch 2.0+)
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()

                # Synchronize MPS operations
                if hasattr(torch.mps, 'synchronize'):
                    torch.mps.synchronize()

                result["success"] = True
                result["message"] = "MPS memory cache cleared"
                logger.info("MPS GPU memory cleanup completed")

            else:
                result["success"] = True
                result["message"] = "No GPU available, no memory to clear"

        except Exception as e:
            logger.error(f"GPU memory cleanup failed: {e}")
            result["success"] = False
            result["error"] = str(e)

        return result

    def validate_dataset(self, dataset_path: str) -> DetectionDatasetValidation:
        """
        Validate a YOLO detection dataset structure.

        Expected structure:
        - data.yaml (with path, train, val, nc, names)
        - images/train/*.jpg
        - images/val/*.jpg
        - labels/train/*.txt (class x_center y_center width height format)
        - labels/val/*.txt

        Args:
            dataset_path: Path to dataset root directory

        Returns:
            DetectionDatasetValidation with validation results
        """
        errors = []
        warnings = []
        train_images = 0
        val_images = 0
        train_labels = 0
        val_labels = 0

        dataset_path = Path(dataset_path)

        # Check if path exists
        if not dataset_path.exists():
            return DetectionDatasetValidation(
                valid=False,
                train_images=0,
                val_images=0,
                train_labels=0,
                val_labels=0,
                errors=[f"Dataset path does not exist: {dataset_path}"],
                warnings=[]
            )

        # Check for data.yaml
        data_yaml = dataset_path / 'data.yaml'
        if not data_yaml.exists():
            errors.append("Missing data.yaml configuration file")
        else:
            try:
                import yaml
                with open(data_yaml) as f:
                    config = yaml.safe_load(f)

                required_keys = ['train', 'val', 'nc', 'names']
                for key in required_keys:
                    if key not in config:
                        errors.append(f"data.yaml missing required key: {key}")

                # Validate nc (number of classes)
                if 'nc' in config:
                    nc = config['nc']
                    if not isinstance(nc, int) or nc < 1:
                        errors.append("nc must be a positive integer")
                    elif nc != 1:
                        warnings.append(f"Expected nc=1 for follicle detection, got nc={nc}")

            except Exception as e:
                errors.append(f"Failed to parse data.yaml: {e}")

        # Check image directories
        train_img_dir = dataset_path / 'images' / 'train'
        val_img_dir = dataset_path / 'images' / 'val'

        if train_img_dir.exists():
            train_images = len(list(train_img_dir.glob('*.jpg')) + list(train_img_dir.glob('*.png')))
        else:
            errors.append("Missing images/train directory")

        if val_img_dir.exists():
            val_images = len(list(val_img_dir.glob('*.jpg')) + list(val_img_dir.glob('*.png')))
        else:
            errors.append("Missing images/val directory")

        # Check label directories
        train_label_dir = dataset_path / 'labels' / 'train'
        val_label_dir = dataset_path / 'labels' / 'val'

        if train_label_dir.exists():
            train_labels = len(list(train_label_dir.glob('*.txt')))
        else:
            errors.append("Missing labels/train directory")

        if val_label_dir.exists():
            val_labels = len(list(val_label_dir.glob('*.txt')))
        else:
            errors.append("Missing labels/val directory")

        # Check for mismatches
        if train_images > 0 and train_labels > 0:
            if train_images != train_labels:
                warnings.append(f"Train images ({train_images}) != labels ({train_labels})")

        if val_images > 0 and val_labels > 0:
            if val_images != val_labels:
                warnings.append(f"Val images ({val_images}) != labels ({val_labels})")

        # Check minimum counts
        if train_images < 10:
            warnings.append(f"Training set has only {train_images} images. Recommend at least 100+")
        if val_images < 5:
            warnings.append(f"Validation set has only {val_images} images. Recommend at least 20+")

        valid = len(errors) == 0 and train_images > 0 and val_images > 0

        return DetectionDatasetValidation(
            valid=valid,
            train_images=train_images,
            val_images=val_images,
            train_labels=train_labels,
            val_labels=val_labels,
            errors=errors,
            warnings=warnings
        )

    async def train(
        self,
        dataset_path: str,
        config: DetectionTrainingConfig,
        progress_callback: Callable[[DetectionTrainingProgress], None],
        model_name: Optional[str] = None
    ) -> Tuple[str, Optional[DetectionModelInfo]]:
        """
        Train a YOLO detection model.

        Args:
            dataset_path: Path to dataset root (must have data.yaml)
            config: Training configuration
            progress_callback: Called with progress updates
            model_name: Optional custom name for the model

        Returns:
            Tuple of (job_id, DetectionModelInfo if successful)
        """
        if not YOLO_AVAILABLE:
            progress_callback(DetectionTrainingProgress(
                status='failed',
                message='Ultralytics not installed. Please install with: pip install ultralytics'
            ))
            return '', None

        # Generate job ID
        job_id = str(uuid.uuid4())[:8]

        # Generate model name if not provided
        if not model_name:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f"detection_{config.model_size}_{timestamp}"

        # Create model directory
        model_dir = self.models_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Store job info
        self._training_jobs[job_id] = {
            'status': 'preparing',
            'model_name': model_name,
            'model_dir': str(model_dir),
            'stop_requested': False,
            'thread': None
        }

        # Run training in a separate thread to not block the async event loop
        def training_thread():
            try:
                self._run_training(
                    job_id, dataset_path, config, model_name, model_dir, progress_callback
                )
            except Exception as e:
                logger.exception(f"Training failed for job {job_id}")
                progress_callback(DetectionTrainingProgress(
                    status='failed',
                    message=str(e)
                ))

        thread = threading.Thread(target=training_thread, daemon=True)
        self._training_jobs[job_id]['thread'] = thread
        thread.start()

        return job_id, None  # DetectionModelInfo returned via progress callback when complete

    def _run_training(
        self,
        job_id: str,
        dataset_path: str,
        config: DetectionTrainingConfig,
        model_name: str,
        model_dir: Path,
        progress_callback: Callable[[DetectionTrainingProgress], None]
    ):
        """
        Internal training loop (runs in a separate thread).
        """
        try:
            # Check if we're resuming from a previous training
            is_resuming = config.resume_from is not None
            resume_model_path = None

            # On Windows, set environment variables to avoid CUDA multiprocessing deadlocks
            import sys
            if sys.platform == 'win32' and is_resuming:
                import os
                os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
                os.environ['OMP_NUM_THREADS'] = '1'
                os.environ['MKL_NUM_THREADS'] = '1'
                logger.info("Set CUDA_LAUNCH_BLOCKING=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1 for Windows resume")

            if is_resuming:
                progress_callback(DetectionTrainingProgress(
                    status='preparing',
                    message=f'Resuming training from {config.resume_from}...'
                ))

                # Find the last.pt checkpoint
                resume_model_dir = self.models_dir / config.resume_from
                resume_model_path = resume_model_dir / 'weights' / 'last.pt'

                if not resume_model_path.exists():
                    raise ValueError(f"Cannot resume: {resume_model_path} not found")

                logger.info(f"Resuming training from: {resume_model_path}")
            else:
                progress_callback(DetectionTrainingProgress(
                    status='preparing',
                    message='Loading YOLO model...'
                ))

            # Determine device
            device = config.device
            if device == 'auto':
                import torch
                if torch.cuda.is_available():
                    device = 0  # First CUDA device
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    device = 'mps'
                else:
                    device = 'cpu'

            # Load model - either resume checkpoint or fresh pretrained
            if is_resuming and resume_model_path:
                logger.info(f"Loading checkpoint for resume: {resume_model_path}")
                model = YOLO(str(resume_model_path))
            else:
                model_variant = f'yolo11{config.model_size}.pt'
                logger.info(f"Loading pretrained model: {model_variant}")
                model = YOLO(model_variant)

            progress_callback(DetectionTrainingProgress(
                status='preparing',
                message=f'Starting training on {device}...'
            ))

            # Custom callback for progress updates
            class ProgressCallback:
                def __init__(self, job_id, jobs, callback):
                    self.job_id = job_id
                    self.jobs = jobs
                    self.callback = callback
                    self.start_time = time.time()

                def on_train_epoch_end(self, trainer):
                    if self.jobs.get(self.job_id, {}).get('stop_requested'):
                        trainer.stop = True
                        return

                    epoch = trainer.epoch + 1
                    total = trainer.epochs

                    # Calculate ETA
                    elapsed = time.time() - self.start_time
                    if epoch > 0:
                        eta_seconds = (elapsed / epoch) * (total - epoch)
                        eta = self._format_time(eta_seconds)
                    else:
                        eta = 'calculating...'

                    # Get losses
                    loss = float(trainer.loss) if hasattr(trainer, 'loss') else 0.0

                    # Get individual losses from the loss items
                    box_loss = 0.0
                    cls_loss = 0.0
                    dfl_loss = 0.0
                    if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
                        items = trainer.loss_items
                        if len(items) >= 3:
                            box_loss = float(items[0])  # Box loss
                            cls_loss = float(items[1])  # Classification loss
                            dfl_loss = float(items[2])  # Distribution focal loss

                    # Get metrics
                    metrics = {}
                    if hasattr(trainer, 'metrics') and trainer.metrics:
                        for key, value in trainer.metrics.items():
                            if isinstance(value, (int, float)):
                                metrics[key] = float(value)

                    self.callback(DetectionTrainingProgress(
                        status='training',
                        epoch=epoch,
                        total_epochs=total,
                        loss=loss,
                        box_loss=box_loss,
                        cls_loss=cls_loss,
                        dfl_loss=dfl_loss,
                        metrics=metrics,
                        eta=eta,
                        message=f'Epoch {epoch}/{total}'
                    ))

                def _format_time(self, seconds):
                    hours = int(seconds // 3600)
                    minutes = int((seconds % 3600) // 60)
                    if hours > 0:
                        return f'{hours}h {minutes}m'
                    return f'{minutes}m'

            # Create callback instance
            callbacks = ProgressCallback(job_id, self._training_jobs, progress_callback)

            # Register callback with the model
            model.add_callback('on_train_epoch_end', callbacks.on_train_epoch_end)

            # Configure training arguments
            data_yaml = Path(dataset_path) / 'data.yaml'

            # Start training (with resume if applicable)
            if is_resuming:
                logger.info(f"Resuming training from checkpoint, device={device}")
                # When resuming, ultralytics needs resume=True to load optimizer state etc.
                # Use fewer workers on Windows to avoid multiprocessing issues with CUDA
                results = model.train(
                    resume=True,
                    device=device,
                    workers=0,  # Disable multiprocessing to avoid CUDA deadlocks on Windows
                    verbose=True
                )
            else:
                logger.info(f"Starting training: {config.epochs} epochs, device={device}")
                results = model.train(
                    data=str(data_yaml),
                    epochs=config.epochs,
                    imgsz=config.img_size,
                    batch=config.batch_size,
                    patience=config.patience,
                    device=device,
                    workers=config.workers,
                    lr0=config.lr0,
                    lrf=config.lrf,
                    momentum=config.momentum,
                    weight_decay=config.weight_decay,
                    warmup_epochs=config.warmup_epochs,
                    warmup_momentum=config.warmup_momentum,
                    warmup_bias_lr=config.warmup_bias_lr,
                    close_mosaic=config.close_mosaic,
                    augment=config.augment,
                    cache=config.cache,
                    project=str(model_dir.parent),
                    name=model_dir.name,
                    exist_ok=True,
                    verbose=True
                )

            # Check if training was stopped
            if self._training_jobs.get(job_id, {}).get('stop_requested'):
                progress_callback(DetectionTrainingProgress(
                    status='stopped',
                    message='Training stopped by user'
                ))
                return

            # Copy best model to model directory
            runs_dir = model_dir.parent / model_dir.name
            best_model = runs_dir / 'weights' / 'best.pt'
            last_model = runs_dir / 'weights' / 'last.pt'

            # Move model files to our model directory structure
            weights_dir = model_dir / 'weights'
            weights_dir.mkdir(exist_ok=True)

            # Only copy if source and destination are different
            dest_best = weights_dir / 'best.pt'
            dest_last = weights_dir / 'last.pt'

            if best_model.exists() and best_model.resolve() != dest_best.resolve():
                shutil.copy(best_model, dest_best)
            if last_model.exists() and last_model.resolve() != dest_last.resolve():
                shutil.copy(last_model, dest_last)

            # Save config and metrics
            config_file = model_dir / 'config.json'
            with open(config_file, 'w') as f:
                json.dump(asdict(config), f, indent=2)

            # Extract final metrics
            final_metrics = {}
            if results and hasattr(results, 'results_dict'):
                for key, value in results.results_dict.items():
                    if isinstance(value, (int, float)):
                        final_metrics[key] = float(value)

            metrics_file = model_dir / 'metrics.json'
            with open(metrics_file, 'w') as f:
                json.dump(final_metrics, f, indent=2)

            # Create model info
            model_info = DetectionModelInfo(
                id=model_name,
                name=model_name,
                path=str(weights_dir / 'best.pt'),
                created_at=datetime.now().isoformat(),
                epochs_trained=config.epochs,
                img_size=config.img_size,
                metrics=final_metrics
            )

            # Save model info (snake_case for storage)
            info_file = model_dir / 'model_info.json'
            with open(info_file, 'w') as f:
                json.dump(model_info.to_storage_dict(), f, indent=2)

            progress_callback(DetectionTrainingProgress(
                status='completed',
                epoch=config.epochs,
                total_epochs=config.epochs,
                metrics=final_metrics,
                message=f'Training completed. Model saved to {model_dir}'
            ))

            logger.info(f"Training completed for job {job_id}. Model: {model_name}")

        except Exception as e:
            logger.exception(f"Training error for job {job_id}")
            progress_callback(DetectionTrainingProgress(
                status='failed',
                message=str(e)
            ))
        finally:
            # Cleanup job entry
            if job_id in self._training_jobs:
                del self._training_jobs[job_id]
            # Clean up GPU memory after training
            cleanup_result = self.clear_gpu_memory()
            if cleanup_result.get('memory_freed_mb', 0) > 0:
                logger.info(f"Post-training GPU cleanup: freed {cleanup_result['memory_freed_mb']:.1f}MB")

    def stop_training(self, job_id: str) -> bool:
        """
        Request to stop a training job.

        Args:
            job_id: Training job ID

        Returns:
            True if stop was requested, False if job not found
        """
        if job_id in self._training_jobs:
            self._training_jobs[job_id]['stop_requested'] = True
            logger.info(f"Stop requested for training job {job_id}")
            return True
        return False

    def _get_inference_device(self):
        """Get the best available device for inference."""
        try:
            import torch
            if torch.cuda.is_available():
                return 0  # CUDA device
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'  # Apple Silicon
        except ImportError:
            pass
        return 'cpu'

    def load_model(self, model_path: str) -> bool:
        """
        Load a trained model for inference.

        Supports both PyTorch (.pt) and TensorRT (.engine) formats.
        Ultralytics handles TensorRT engines directly.
        Handles cross-device loading (e.g. CUDA-trained models on MPS/CPU).

        Args:
            model_path: Path to model file (.pt or .engine)

        Returns:
            True if loaded successfully
        """
        if not YOLO_AVAILABLE:
            logger.error("Cannot load model: Ultralytics not installed")
            return False

        try:
            # Unload current model if any
            if self._loaded_model is not None:
                del self._loaded_model
                self._loaded_model = None
                self._loaded_model_path = None
                self._loaded_model_backend = 'pytorch'
                self._loaded_model_imgsz = 640

            # Detect backend from file extension
            model_ext = Path(model_path).suffix.lower()
            if model_ext == '.engine':
                backend = 'tensorrt'
            else:
                backend = 'pytorch'

            # For PyTorch models, ensure CUDA-trained models can load on CPU/MPS
            # by pre-loading with map_location='cpu' if CUDA is not available
            if backend == 'pytorch':
                try:
                    import torch
                    if not torch.cuda.is_available():
                        # Force-remap CUDA tensors to CPU before YOLO loads them
                        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                        if isinstance(checkpoint, dict) and 'model' in checkpoint:
                            # Ensure model tensors are on CPU
                            if hasattr(checkpoint['model'], 'float'):
                                checkpoint['model'] = checkpoint['model'].float()
                        # Save remapped checkpoint to a temp file and load via YOLO
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
                            torch.save(checkpoint, tmp.name)
                            self._loaded_model = YOLO(tmp.name)
                        # Clean up temp file
                        try:
                            os.unlink(tmp.name)
                        except Exception:
                            pass
                        logger.info(f"Loaded CUDA-trained model on {self._get_inference_device()} via CPU remapping")
                    else:
                        self._loaded_model = YOLO(model_path)
                except Exception as e:
                    logger.warning(f"Cross-device load failed, trying direct load: {e}")
                    self._loaded_model = YOLO(model_path)
            else:
                # TensorRT engines - load directly (GPU-architecture specific)
                self._loaded_model = YOLO(model_path)

            self._loaded_model_path = model_path
            self._loaded_model_backend = backend

            # Store the model's training imgsz to ensure consistent inference
            self._loaded_model_imgsz = self._loaded_model.overrides.get('imgsz', 640)

            logger.info(f"Loaded model: {model_path} (backend: {backend}, imgsz: {self._loaded_model_imgsz})")
            return True

        except Exception as e:
            logger.exception(f"Failed to load model: {model_path}")
            return False

    def _load_default_model(self) -> bool:
        """
        Load the default pretrained YOLO detection model.

        This downloads yolo11n.pt from Ultralytics hub on first use (~6MB).

        Returns:
            True if loaded successfully
        """
        if not YOLO_AVAILABLE:
            logger.error("Cannot load default model: Ultralytics not installed")
            return False

        try:
            # Use the nano model for best balance of speed and accuracy
            default_model = 'yolo11n.pt'
            logger.info(f"Loading default pretrained model: {default_model}")

            self._loaded_model = YOLO(default_model)
            self._loaded_model_path = f"pretrained:{default_model}"
            self._loaded_model_imgsz = self._loaded_model.overrides.get('imgsz', 640)

            logger.info(f"Successfully loaded default model: {default_model} (imgsz: {self._loaded_model_imgsz})")
            return True

        except Exception as e:
            logger.exception(f"Failed to load default model")
            return False

    def predict(self, image_data: bytes, confidence_threshold: float = 0.5) -> List[DetectionPrediction]:
        """
        Run detection prediction on a full image.

        Args:
            image_data: Image bytes (JPEG or PNG)
            confidence_threshold: Minimum confidence to include detection

        Returns:
            List of DetectionPrediction objects
        """
        if self._loaded_model is None:
            # Auto-load default pretrained model on first use
            logger.info("No model loaded, auto-loading pretrained yolo11n.pt...")
            if not self._load_default_model():
                logger.error("Failed to auto-load default model")
                return []

        try:
            # Decode image
            image = Image.open(io.BytesIO(image_data))

            # CRITICAL: Apply EXIF orientation to match what frontend sees
            try:
                image = ImageOps.exif_transpose(image)
            except Exception as e:
                logger.warning(f"Failed to apply EXIF transpose: {e}")

            if image.mode != 'RGB':
                image = image.convert('RGB')

            img_array = np.array(image)
            img_height, img_width = img_array.shape[:2]

            # Run inference with explicit device and imgsz to ensure consistent results
            device = self._get_inference_device()
            results = self._loaded_model.predict(
                img_array,
                conf=confidence_threshold,
                verbose=False,
                imgsz=self._loaded_model_imgsz,
                device=device
            )

            if not results or len(results) == 0:
                return []

            result = results[0]

            # Extract bounding boxes
            predictions = []
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2] format
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)

                for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                    x1, y1, x2, y2 = box

                    # Clamp to image bounds
                    x1 = max(0, min(x1, img_width))
                    y1 = max(0, min(y1, img_height))
                    x2 = max(0, min(x2, img_width))
                    y2 = max(0, min(y2, img_height))

                    det_width = x2 - x1
                    det_height = y2 - y1

                    # Skip invalid boxes
                    if det_width <= 0 or det_height <= 0:
                        continue

                    predictions.append(DetectionPrediction(
                        x=float(x1),
                        y=float(y1),
                        width=float(det_width),
                        height=float(det_height),
                        confidence=float(conf),
                        class_id=int(cls_id),
                        class_name='follicle'
                    ))

            return predictions

        except Exception as e:
            logger.exception("Prediction failed")
            return []

    def predict_base64(self, image_base64: str, confidence_threshold: float = 0.5) -> List[DetectionPrediction]:
        """
        Run detection prediction on a base64-encoded image.

        Args:
            image_base64: Base64-encoded image string
            confidence_threshold: Minimum confidence to include detection

        Returns:
            List of DetectionPrediction objects
        """
        try:
            # Remove data URL prefix if present
            if ',' in image_base64:
                image_base64 = image_base64.split(',', 1)[1]

            image_data = base64.b64decode(image_base64)
            return self.predict(image_data, confidence_threshold)
        except Exception as e:
            logger.exception("Failed to decode base64 image")
            return []

    def predict_tiled(
        self,
        image_data: bytes,
        confidence_threshold: float = 0.5,
        tile_size: int = 1024,
        overlap: int = 128,
        nms_threshold: float = 0.5,
        scale_factor: float = 1.0
    ) -> List[DetectionPrediction]:
        """
        Run tiled detection prediction on a large image.

        Splits the image into overlapping tiles, runs inference on each,
        and merges results with Non-Maximum Suppression.

        Args:
            image_data: Image bytes (JPEG or PNG)
            confidence_threshold: Minimum confidence to include detection
            tile_size: Size of each tile (default 1024 to match training)
            overlap: Pixel overlap between tiles (default 128)
            nms_threshold: IoU threshold for NMS merging (default 0.5)
            scale_factor: Factor to upscale image before inference (default 1.0)
                         Use >1.0 for images with smaller objects than training data.
                         Coordinates are scaled back to original image space.

        Returns:
            List of DetectionPrediction objects
        """
        if self._loaded_model is None:
            logger.info("No model loaded, auto-loading pretrained yolo11n.pt...")
            if not self._load_default_model():
                logger.error("Failed to auto-load default model")
                return []

        try:
            # Decode image
            image = Image.open(io.BytesIO(image_data))

            # CRITICAL: Apply EXIF orientation to match what frontend sees
            # Without this, rotated images will have misaligned coordinates
            try:
                image = ImageOps.exif_transpose(image)
            except Exception as e:
                logger.warning(f"Failed to apply EXIF transpose: {e}")

            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Store original dimensions for coordinate scaling
            original_width, original_height = image.size

            # Apply upscaling if scale_factor > 1
            if scale_factor > 1.0:
                new_width = int(original_width * scale_factor)
                new_height = int(original_height * scale_factor)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.info(f"Upscaled image from {original_width}x{original_height} to {new_width}x{new_height} (scale={scale_factor})")

            img_array = np.array(image)
            img_height, img_width = img_array.shape[:2]

            logger.info(f"Tiled inference: image {img_width}x{img_height}, tile_size={tile_size}, overlap={overlap}, scale_factor={scale_factor}")

            # If image is smaller than tile size, use regular prediction
            if img_width <= tile_size and img_height <= tile_size:
                logger.info("Image smaller than tile size, using regular prediction")
                return self.predict(image_data, confidence_threshold)

            # Determine inference device
            device = self._get_inference_device()

            # Calculate tile positions
            step = tile_size - overlap
            all_predictions = []
            tile_count = 0

            for y in range(0, img_height, step):
                for x in range(0, img_width, step):
                    # Calculate tile boundaries
                    x1 = x
                    y1 = y
                    x2 = min(x + tile_size, img_width)
                    y2 = min(y + tile_size, img_height)

                    # Extract tile
                    tile = img_array[y1:y2, x1:x2]

                    # Skip very small tiles at edges
                    if tile.shape[0] < tile_size // 4 or tile.shape[1] < tile_size // 4:
                        continue

                    tile_count += 1

                    # Run inference on tile with explicit device and imgsz
                    results = self._loaded_model.predict(
                        tile,
                        conf=confidence_threshold,
                        verbose=False,
                        imgsz=self._loaded_model_imgsz,
                        device=device
                    )

                    if results and len(results) > 0:
                        result = results[0]
                        if result.boxes is not None and len(result.boxes) > 0:
                            boxes = result.boxes.xyxy.cpu().numpy()
                            confidences = result.boxes.conf.cpu().numpy()
                            class_ids = result.boxes.cls.cpu().numpy().astype(int)

                            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                                # Adjust coordinates to full image space
                                bx1, by1, bx2, by2 = box

                                # Convert to full image coordinates
                                full_x1 = bx1 + x1
                                full_y1 = by1 + y1
                                full_x2 = bx2 + x1
                                full_y2 = by2 + y1

                                # Clamp to image bounds to prevent out-of-bounds detections
                                full_x1 = max(0, min(full_x1, img_width))
                                full_y1 = max(0, min(full_y1, img_height))
                                full_x2 = max(0, min(full_x2, img_width))
                                full_y2 = max(0, min(full_y2, img_height))

                                # Calculate width/height after clamping
                                det_width = full_x2 - full_x1
                                det_height = full_y2 - full_y1

                                # Skip if box became invalid after clamping
                                if det_width <= 0 or det_height <= 0:
                                    continue

                                all_predictions.append(DetectionPrediction(
                                    x=float(full_x1),
                                    y=float(full_y1),
                                    width=float(det_width),
                                    height=float(det_height),
                                    confidence=float(conf),
                                    class_id=int(cls_id),
                                    class_name='follicle'
                                ))

            logger.info(f"Processed {tile_count} tiles, found {len(all_predictions)} raw detections")

            if not all_predictions:
                return []

            # Apply Non-Maximum Suppression to merge overlapping detections
            merged = self._apply_nms(all_predictions, nms_threshold)
            logger.info(f"After NMS: {len(merged)} detections")

            # Scale coordinates back to original image space if we upscaled
            if scale_factor > 1.0:
                scaled_predictions = []
                for pred in merged:
                    # Scale coordinates back down
                    scaled_x = pred.x / scale_factor
                    scaled_y = pred.y / scale_factor
                    scaled_width = pred.width / scale_factor
                    scaled_height = pred.height / scale_factor

                    # Clamp to original image bounds
                    scaled_x = max(0, min(scaled_x, original_width))
                    scaled_y = max(0, min(scaled_y, original_height))
                    scaled_width = min(scaled_width, original_width - scaled_x)
                    scaled_height = min(scaled_height, original_height - scaled_y)

                    if scaled_width > 0 and scaled_height > 0:
                        scaled_predictions.append(DetectionPrediction(
                            x=float(scaled_x),
                            y=float(scaled_y),
                            width=float(scaled_width),
                            height=float(scaled_height),
                            confidence=pred.confidence,
                            class_id=pred.class_id,
                            class_name=pred.class_name
                        ))

                logger.info(f"Scaled {len(scaled_predictions)} detections back to original image space")
                return scaled_predictions

            return merged

        except Exception as e:
            logger.exception("Tiled prediction failed")
            return []

    def _apply_nms(
        self,
        predictions: List[DetectionPrediction],
        iou_threshold: float = 0.5
    ) -> List[DetectionPrediction]:
        """
        Apply Non-Maximum Suppression to merge overlapping detections.

        Args:
            predictions: List of predictions to merge
            iou_threshold: IoU threshold for considering boxes as duplicates

        Returns:
            Filtered list of predictions
        """
        if not predictions:
            return []

        # Convert to numpy arrays for efficient computation
        boxes = np.array([[p.x, p.y, p.x + p.width, p.y + p.height] for p in predictions])
        scores = np.array([p.confidence for p in predictions])

        # Sort by confidence (descending)
        indices = np.argsort(scores)[::-1]

        keep = []
        while len(indices) > 0:
            # Keep the highest confidence detection
            current = indices[0]
            keep.append(current)

            if len(indices) == 1:
                break

            # Calculate IoU with remaining boxes
            current_box = boxes[current]
            remaining_boxes = boxes[indices[1:]]

            # Calculate intersection
            x1 = np.maximum(current_box[0], remaining_boxes[:, 0])
            y1 = np.maximum(current_box[1], remaining_boxes[:, 1])
            x2 = np.minimum(current_box[2], remaining_boxes[:, 2])
            y2 = np.minimum(current_box[3], remaining_boxes[:, 3])

            intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

            # Calculate union
            current_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
            remaining_areas = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * (remaining_boxes[:, 3] - remaining_boxes[:, 1])
            union = current_area + remaining_areas - intersection

            # Calculate IoU
            iou = intersection / (union + 1e-6)

            # Keep boxes with IoU below threshold
            mask = iou < iou_threshold
            indices = indices[1:][mask]

        return [predictions[i] for i in keep]

    def track_across_images(
        self,
        source_image_data: bytes,
        target_image_data: bytes,
        confidence_threshold: float = 0.5,
        match_distance_threshold: float = 50.0,
        method: str = 'auto'
    ) -> Dict[str, Any]:
        """
        Track follicles across two images of the same scalp from different angles.

        Uses homography-based feature matching to find corresponding follicles
        between two images. Falls back to model.track() for near-identical views.

        Args:
            source_image_data: Source image bytes (JPEG or PNG)
            target_image_data: Target image bytes (JPEG or PNG)
            confidence_threshold: Minimum confidence to include detection
            match_distance_threshold: Maximum pixel distance for matching (after homography transform)
            method: Matching method - 'auto', 'homography', or 'track'

        Returns:
            Dict with sourceDetections, targetDetections, matches, homographyMatrix, method
        """
        if self._loaded_model is None:
            logger.info("No model loaded, auto-loading pretrained yolo11n.pt...")
            if not self._load_default_model():
                logger.error("Failed to auto-load default model")
                return {
                    'success': False,
                    'error': 'No model loaded and failed to auto-load default model',
                    'sourceDetections': [],
                    'targetDetections': [],
                    'matches': [],
                    'method': method,
                }

        try:
            # Decode and preprocess both images
            source_img = Image.open(io.BytesIO(source_image_data))
            target_img = Image.open(io.BytesIO(target_image_data))

            # Apply EXIF orientation
            try:
                source_img = ImageOps.exif_transpose(source_img)
            except Exception:
                pass
            try:
                target_img = ImageOps.exif_transpose(target_img)
            except Exception:
                pass

            if source_img.mode != 'RGB':
                source_img = source_img.convert('RGB')
            if target_img.mode != 'RGB':
                target_img = target_img.convert('RGB')

            source_array = np.array(source_img)
            target_array = np.array(target_img)

            # Run detection on both images independently.
            # Use tiled inference for large images to match the main detection pipeline.
            source_h, source_w = source_array.shape[:2]
            target_h, target_w = target_array.shape[:2]
            tile_size = 1024
            use_tiled = max(source_w, source_h, target_w, target_h) > tile_size

            if use_tiled:
                logger.info(f"Using tiled inference for tracking (source: {source_w}x{source_h}, target: {target_w}x{target_h})")
                source_detections = self.predict_tiled(
                    source_image_data, confidence_threshold,
                    tile_size=tile_size, overlap=128, nms_threshold=0.5
                )
                target_detections = self.predict_tiled(
                    target_image_data, confidence_threshold,
                    tile_size=tile_size, overlap=128, nms_threshold=0.5
                )
            else:
                device = self._get_inference_device()
                source_results = self._loaded_model.predict(
                    source_array, conf=confidence_threshold, verbose=False,
                    imgsz=self._loaded_model_imgsz, device=device
                )
                target_results = self._loaded_model.predict(
                    target_array, conf=confidence_threshold, verbose=False,
                    imgsz=self._loaded_model_imgsz, device=device
                )
                source_detections = self._extract_detections(source_results, source_array.shape)
                target_detections = self._extract_detections(target_results, target_array.shape)

            logger.info(f"Cross-image tracking: {len(source_detections)} source, {len(target_detections)} target detections")

            if not source_detections or not target_detections:
                return {
                    'success': True,
                    'sourceDetections': [d.to_dict() for d in source_detections],
                    'targetDetections': [d.to_dict() for d in target_detections],
                    'matches': [],
                    'homographyMatrix': None,
                    'method': method if method != 'auto' else 'homography',
                }

            # Try homography-based matching
            used_method = method
            matches = []
            homography_matrix = None

            if method in ('auto', 'homography'):
                matches, homography_matrix = self._match_via_homography(
                    source_array, target_array,
                    source_detections, target_detections,
                    match_distance_threshold
                )
                used_method = 'homography'

                # Fall back to track if homography failed and method is auto
                if not matches and method == 'auto':
                    logger.info("Homography matching failed, falling back to model.track()")
                    track_result = self._match_via_track(
                        source_array, target_array,
                        confidence_threshold
                    )
                    matches = track_result['matches']
                    source_detections = track_result['source_detections']
                    target_detections = track_result['target_detections']
                    used_method = 'track'
                    homography_matrix = None

            elif method == 'track':
                # BoT-SORT: model.track() runs its own detection, so use its
                # detections directly instead of the tiled inference ones.
                track_result = self._match_via_track(
                    source_array, target_array,
                    confidence_threshold
                )
                matches = track_result['matches']
                source_detections = track_result['source_detections']
                target_detections = track_result['target_detections']
                used_method = 'track'

            logger.info(f"Cross-image tracking complete: {len(matches)} matches via {used_method}")

            return {
                'success': True,
                'sourceDetections': [d.to_dict() for d in source_detections],
                'targetDetections': [d.to_dict() for d in target_detections],
                'matches': matches,
                'homographyMatrix': homography_matrix,
                'method': used_method,
            }

        except Exception as e:
            logger.exception("Cross-image tracking failed")
            return {
                'success': False,
                'error': str(e),
                'sourceDetections': [],
                'targetDetections': [],
                'matches': [],
                'method': method,
            }

    def _extract_detections(
        self,
        results: list,
        img_shape: tuple
    ) -> List[DetectionPrediction]:
        """Extract DetectionPrediction list from YOLO results."""
        detections = []
        if not results or len(results) == 0:
            return detections

        result = results[0]
        img_height, img_width = img_shape[:2]

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = box
                x1 = max(0, min(float(x1), img_width))
                y1 = max(0, min(float(y1), img_height))
                x2 = max(0, min(float(x2), img_width))
                y2 = max(0, min(float(y2), img_height))

                w = x2 - x1
                h = y2 - y1
                if w <= 0 or h <= 0:
                    continue

                detections.append(DetectionPrediction(
                    x=x1, y=y1, width=w, height=h,
                    confidence=float(conf),
                    class_id=int(cls_id),
                    class_name='follicle'
                ))

        return detections

    def _match_via_homography(
        self,
        source_array: np.ndarray,
        target_array: np.ndarray,
        source_detections: List[DetectionPrediction],
        target_detections: List[DetectionPrediction],
        match_distance_threshold: float
    ) -> Tuple[List[Dict], Optional[List[List[float]]]]:
        """
        Match follicles across images using ORB feature matching + homography.
        Downscales large images for better feature detection.

        Returns:
            Tuple of (matches list, homography matrix as nested list or None)
        """
        try:
            # Convert to grayscale for feature detection
            source_gray = cv2.cvtColor(source_array, cv2.COLOR_RGB2GRAY)
            target_gray = cv2.cvtColor(target_array, cv2.COLOR_RGB2GRAY)

            # Downscale large images for better ORB feature detection.
            # ORB at full resolution on huge images (12000+ px) gives noisy features.
            max_dim = 2000
            source_h, source_w = source_gray.shape[:2]
            target_h, target_w = target_gray.shape[:2]

            source_scale = 1.0
            if max(source_w, source_h) > max_dim:
                source_scale = max_dim / max(source_w, source_h)
                source_gray_small = cv2.resize(source_gray, None, fx=source_scale, fy=source_scale, interpolation=cv2.INTER_AREA)
            else:
                source_gray_small = source_gray

            target_scale = 1.0
            if max(target_w, target_h) > max_dim:
                target_scale = max_dim / max(target_w, target_h)
                target_gray_small = cv2.resize(target_gray, None, fx=target_scale, fy=target_scale, interpolation=cv2.INTER_AREA)
            else:
                target_gray_small = target_gray

            logger.info(f"Homography feature detection: source scale={source_scale:.3f}, target scale={target_scale:.3f}")

            # Detect ORB features on downscaled images
            orb = cv2.ORB_create(nfeatures=10000)
            kp1, des1 = orb.detectAndCompute(source_gray_small, None)
            kp2, des2 = orb.detectAndCompute(target_gray_small, None)

            if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
                logger.warning("Insufficient features for homography matching")
                return [], None

            # Use KNN matching with ratio test instead of crossCheck
            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
            raw_matches = bf.knnMatch(des1, des2, k=2)

            # Apply ratio test to filter ambiguous matches
            good_matches = []
            for m_pair in raw_matches:
                if len(m_pair) == 2:
                    m, n = m_pair
                    if m.distance < 0.8 * n.distance:
                        good_matches.append(m)

            logger.info(f"ORB matching: {len(kp1)} source, {len(kp2)} target keypoints, {len(good_matches)} good matches after ratio test")

            if len(good_matches) < 4:
                logger.warning(f"Only {len(good_matches)} good matches, need at least 4 for homography")
                return [], None

            # Extract matched keypoints and scale back to original image coordinates
            src_pts = np.float32([
                [kp1[m.queryIdx].pt[0] / source_scale, kp1[m.queryIdx].pt[1] / source_scale]
                for m in good_matches
            ]).reshape(-1, 1, 2)
            dst_pts = np.float32([
                [kp2[m.trainIdx].pt[0] / target_scale, kp2[m.trainIdx].pt[1] / target_scale]
                for m in good_matches
            ]).reshape(-1, 1, 2)

            # Compute homography with RANSAC — use higher threshold for large images
            ransac_threshold = max(5.0, min(source_w, source_h) * 0.001)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold, maxIters=5000)

            if H is None:
                logger.warning("Homography computation failed")
                return [], None

            inlier_count = int(mask.sum()) if mask is not None else 0
            logger.info(f"Homography computed: {inlier_count}/{len(good_matches)} inlier feature matches")

            if inlier_count < 4:
                logger.warning("Too few inlier matches for reliable homography")
                return [], None

            # Scale match distance threshold proportional to image size.
            # The user-provided threshold (default 50px) is a base for ~1000px images.
            # For a 12000px image, scale it up proportionally.
            img_diagonal = np.sqrt(source_w ** 2 + source_h ** 2)
            scaled_threshold = match_distance_threshold * max(1.0, img_diagonal / 1414.0)  # 1414 = sqrt(1000^2+1000^2)
            logger.info(f"Match distance threshold: {match_distance_threshold} -> {scaled_threshold:.1f} (image diagonal: {img_diagonal:.0f})")

            # Transform source detection centers through the homography
            source_centers = np.float32([
                [d.x + d.width / 2, d.y + d.height / 2] for d in source_detections
            ]).reshape(-1, 1, 2)

            transformed_centers = cv2.perspectiveTransform(source_centers, H).reshape(-1, 2)

            # Build target centers array
            target_centers = np.array([
                [d.x + d.width / 2, d.y + d.height / 2] for d in target_detections
            ])

            # Vectorized distance matrix
            diff = transformed_centers[:, np.newaxis, :] - target_centers[np.newaxis, :, :]
            dist_matrix = np.sqrt((diff ** 2).sum(axis=2))  # shape: (n_source, n_target)

            # NCC patch verification setup
            PATCH_SIZE = 32
            PATCH_EXPAND = 1.5
            NCC_FLOOR = 0.4

            source_gray = cv2.cvtColor(source_array, cv2.COLOR_RGB2GRAY)
            target_gray = cv2.cvtColor(target_array, cv2.COLOR_RGB2GRAY)

            def _extract_patch(gray_img, det):
                img_h, img_w = gray_img.shape[:2]
                cx = det.x + det.width / 2.0
                cy = det.y + det.height / 2.0
                half_w = det.width * PATCH_EXPAND / 2.0
                half_h = det.height * PATCH_EXPAND / 2.0
                x1 = max(0, int(cx - half_w))
                y1 = max(0, int(cy - half_h))
                x2 = min(img_w, int(cx + half_w))
                y2 = min(img_h, int(cy + half_h))
                if x2 <= x1 or y2 <= y1:
                    return None
                crop = gray_img[y1:y2, x1:x2]
                interp = cv2.INTER_AREA if crop.shape[0] > PATCH_SIZE else cv2.INTER_LINEAR
                return cv2.resize(crop, (PATCH_SIZE, PATCH_SIZE), interpolation=interp)

            # Pre-extract all patches
            source_patches = [_extract_patch(source_gray, d) for d in source_detections]
            target_patches = [_extract_patch(target_gray, d) for d in target_detections]

            # Pre-compute source areas and target areas
            source_areas = np.array([d.width * d.height for d in source_detections])
            target_areas = np.array([d.width * d.height for d in target_detections])

            # For each source, find best candidate using distance + NCC + size
            candidates = []  # (combined_score, src_idx, tgt_idx)

            best_dists = dist_matrix.min(axis=1)
            source_order = np.argsort(best_dists)

            for src_idx in source_order:
                src_patch = source_patches[int(src_idx)]
                if src_patch is None:
                    continue

                row = dist_matrix[int(src_idx)]
                src_area = float(source_areas[int(src_idx)])

                # Find all targets within threshold
                candidate_indices = np.where(row <= scaled_threshold)[0]
                if len(candidate_indices) == 0:
                    continue

                best_score = -1.0
                best_tgt = -1

                for tgt_idx in candidate_indices:
                    tgt_idx = int(tgt_idx)
                    tgt_patch = target_patches[tgt_idx]
                    if tgt_patch is None:
                        continue

                    # NCC comparison
                    ncc = float(cv2.matchTemplate(
                        src_patch, tgt_patch, cv2.TM_CCOEFF_NORMED
                    )[0][0])

                    if ncc < NCC_FLOOR:
                        continue

                    dist = float(row[tgt_idx])
                    distance_score = max(0.0, 1.0 - dist / scaled_threshold)
                    tgt_area = float(target_areas[tgt_idx])
                    size_score = min(src_area, tgt_area) / max(src_area, tgt_area + 1e-6)
                    combined = distance_score * max(ncc, 0.0) * (0.5 + 0.5 * size_score)

                    if combined > best_score:
                        best_score = combined
                        best_tgt = tgt_idx

                if best_tgt >= 0:
                    candidates.append((best_score, int(src_idx), best_tgt))

            # Greedy one-to-one assignment sorted by combined score (best first)
            candidates.sort(key=lambda x: x[0], reverse=True)
            used_targets = set()
            matches = []

            for (score, src_idx, tgt_idx) in candidates:
                if tgt_idx in used_targets:
                    continue
                tx, ty = transformed_centers[src_idx]
                matches.append({
                    'sourceDetectionIndex': int(src_idx),
                    'targetDetectionIndex': tgt_idx,
                    'confidence': round(score, 4),
                    'transformedX': round(float(tx), 2),
                    'transformedY': round(float(ty), 2),
                })
                used_targets.add(tgt_idx)

            logger.info(f"Homography+NCC matching: {len(matches)} matches ({len(candidates)} candidates)")

            # Convert homography to serializable format
            homography_list = H.tolist()

            return matches, homography_list

        except Exception as e:
            logger.exception("Homography matching failed")
            return [], None

    def _match_via_track(
        self,
        source_array: np.ndarray,
        target_array: np.ndarray,
        confidence_threshold: float
    ) -> Dict[str, Any]:
        """
        Match follicles by running model.track() on a 2-frame sequence.
        BoT-SORT assigns track IDs — shared IDs across frames = same object.

        Returns:
            Dict with 'matches', 'source_detections', 'target_detections'
        """
        empty_result = {
            'matches': [],
            'source_detections': [],
            'target_detections': [],
        }

        try:
            device = self._get_inference_device()

            # Reset tracker state
            self._loaded_model.predictor = None

            # Run tracking on frame 1 (source)
            results1 = self._loaded_model.track(
                source_array, conf=confidence_threshold, verbose=False,
                imgsz=self._loaded_model_imgsz, device=device,
                persist=True, tracker='botsort.yaml'
            )

            # Run tracking on frame 2 (target) - tracker state persists
            results2 = self._loaded_model.track(
                target_array, conf=confidence_threshold, verbose=False,
                imgsz=self._loaded_model_imgsz, device=device,
                persist=True, tracker='botsort.yaml'
            )

            # Extract detections and track IDs from both frames
            source_detections = self._extract_detections(results1, source_array.shape)
            target_detections = self._extract_detections(results2, target_array.shape)

            frame1_tracks = {}  # track_id -> detection_index
            frame2_tracks = {}  # track_id -> detection_index

            if results1 and len(results1) > 0 and results1[0].boxes is not None:
                boxes = results1[0].boxes
                if boxes.id is not None:
                    track_ids = boxes.id.cpu().numpy().astype(int)
                    for idx, tid in enumerate(track_ids):
                        frame1_tracks[int(tid)] = idx

            if results2 and len(results2) > 0 and results2[0].boxes is not None:
                boxes = results2[0].boxes
                if boxes.id is not None:
                    track_ids = boxes.id.cpu().numpy().astype(int)
                    for idx, tid in enumerate(track_ids):
                        frame2_tracks[int(tid)] = idx

            # Match by shared track IDs
            matches = []
            common_ids = set(frame1_tracks.keys()) & set(frame2_tracks.keys())

            for tid in common_ids:
                src_idx = frame1_tracks[tid]
                tgt_idx = frame2_tracks[tid]
                matches.append({
                    'sourceDetectionIndex': src_idx,
                    'targetDetectionIndex': tgt_idx,
                    'confidence': 1.0,  # Track-based match — ID confirmed
                    'transformedX': 0.0,
                    'transformedY': 0.0,
                })

            logger.info(f"BoT-SORT tracking: {len(common_ids)} shared track IDs from {len(frame1_tracks)} source, {len(frame2_tracks)} target tracks")

            # Reset tracker state after use
            self._loaded_model.predictor = None

            return {
                'matches': matches,
                'source_detections': source_detections,
                'target_detections': target_detections,
            }

        except Exception as e:
            logger.exception("BoT-SORT tracking failed")
            try:
                self._loaded_model.predictor = None
            except Exception:
                pass
            return empty_result

    def track_across_images_base64(
        self,
        source_image_base64: str,
        target_image_base64: str,
        confidence_threshold: float = 0.5,
        match_distance_threshold: float = 50.0,
        method: str = 'auto'
    ) -> Dict[str, Any]:
        """
        Track follicles across two base64-encoded images.

        Args:
            source_image_base64: Base64-encoded source image
            target_image_base64: Base64-encoded target image
            confidence_threshold: Minimum detection confidence
            match_distance_threshold: Maximum pixel distance for matching
            method: 'auto', 'homography', or 'track'

        Returns:
            Tracking result dict
        """
        try:
            # Remove data URL prefix if present
            if ',' in source_image_base64:
                source_image_base64 = source_image_base64.split(',', 1)[1]
            if ',' in target_image_base64:
                target_image_base64 = target_image_base64.split(',', 1)[1]

            source_data = base64.b64decode(source_image_base64)
            target_data = base64.b64decode(target_image_base64)

            return self.track_across_images(
                source_data, target_data,
                confidence_threshold, match_distance_threshold, method
            )
        except Exception as e:
            logger.exception("Failed to decode base64 images for tracking")
            return {
                'success': False,
                'error': str(e),
                'sourceDetections': [],
                'targetDetections': [],
                'matches': [],
                'method': method,
            }

    def prepare_tracking_session(
        self,
        source_image_data: bytes,
        target_image_data: bytes,
        confidence_threshold: float = 0.5,
        match_distance_threshold: float = 50.0
    ) -> Dict[str, Any]:
        """
        Prepare a tracking session: compute homography between two images
        and cache everything for fast per-follicle matching later.

        Returns:
            Dict with success, sessionId, homographyMatrix
        """
        try:
            # Decode and preprocess both images
            source_img = Image.open(io.BytesIO(source_image_data))
            target_img = Image.open(io.BytesIO(target_image_data))

            try:
                source_img = ImageOps.exif_transpose(source_img)
            except Exception:
                pass
            try:
                target_img = ImageOps.exif_transpose(target_img)
            except Exception:
                pass

            if source_img.mode != 'RGB':
                source_img = source_img.convert('RGB')
            if target_img.mode != 'RGB':
                target_img = target_img.convert('RGB')

            source_array = np.array(source_img)
            target_array = np.array(target_img)

            # Compute homography via ORB features
            source_gray = cv2.cvtColor(source_array, cv2.COLOR_RGB2GRAY)
            target_gray = cv2.cvtColor(target_array, cv2.COLOR_RGB2GRAY)

            max_dim = 2000
            source_h, source_w = source_gray.shape[:2]
            target_h, target_w = target_gray.shape[:2]

            source_scale = 1.0
            if max(source_w, source_h) > max_dim:
                source_scale = max_dim / max(source_w, source_h)
                source_gray_small = cv2.resize(source_gray, None, fx=source_scale, fy=source_scale, interpolation=cv2.INTER_AREA)
            else:
                source_gray_small = source_gray

            target_scale = 1.0
            if max(target_w, target_h) > max_dim:
                target_scale = max_dim / max(target_w, target_h)
                target_gray_small = cv2.resize(target_gray, None, fx=target_scale, fy=target_scale, interpolation=cv2.INTER_AREA)
            else:
                target_gray_small = target_gray

            orb = cv2.ORB_create(nfeatures=10000)
            kp1, des1 = orb.detectAndCompute(source_gray_small, None)
            kp2, des2 = orb.detectAndCompute(target_gray_small, None)

            if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
                return {'success': False, 'error': 'Insufficient features for homography', 'sessionId': '', 'homographyMatrix': None}

            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
            raw_matches = bf.knnMatch(des1, des2, k=2)

            good_matches = []
            for m_pair in raw_matches:
                if len(m_pair) == 2:
                    m, n = m_pair
                    if m.distance < 0.8 * n.distance:
                        good_matches.append(m)

            if len(good_matches) < 4:
                return {'success': False, 'error': f'Only {len(good_matches)} feature matches, need 4+', 'sessionId': '', 'homographyMatrix': None}

            src_pts = np.float32([
                [kp1[m.queryIdx].pt[0] / source_scale, kp1[m.queryIdx].pt[1] / source_scale]
                for m in good_matches
            ]).reshape(-1, 1, 2)
            dst_pts = np.float32([
                [kp2[m.trainIdx].pt[0] / target_scale, kp2[m.trainIdx].pt[1] / target_scale]
                for m in good_matches
            ]).reshape(-1, 1, 2)

            ransac_threshold = max(5.0, min(source_w, source_h) * 0.001)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold, maxIters=5000)

            if H is None:
                return {'success': False, 'error': 'Homography computation failed', 'sessionId': '', 'homographyMatrix': None}

            inlier_count = int(mask.sum()) if mask is not None else 0
            if inlier_count < 4:
                return {'success': False, 'error': 'Too few inlier matches', 'sessionId': '', 'homographyMatrix': None}

            # Compute scaled threshold
            img_diagonal = np.sqrt(source_w ** 2 + source_h ** 2)
            scaled_threshold = match_distance_threshold * max(1.0, img_diagonal / 1414.0)

            # Cache session
            session_id = str(uuid.uuid4())
            self._tracking_sessions[session_id] = {
                'source_array': source_array,
                'target_array': target_array,
                'source_gray': source_gray,
                'target_gray': target_gray,
                'H': H,
                'scaled_threshold': scaled_threshold,
                'confidence_threshold': confidence_threshold,
            }

            logger.info(f"Tracking session {session_id} prepared: {inlier_count} inliers, threshold={scaled_threshold:.0f}")

            return {
                'success': True,
                'sessionId': session_id,
                'homographyMatrix': H.tolist(),
            }

        except Exception as e:
            logger.exception("Failed to prepare tracking session")
            return {'success': False, 'error': str(e), 'sessionId': '', 'homographyMatrix': None}

    def prepare_tracking_session_base64(
        self,
        source_image_base64: str,
        target_image_base64: str,
        confidence_threshold: float = 0.5,
        match_distance_threshold: float = 50.0
    ) -> Dict[str, Any]:
        """Base64 wrapper for prepare_tracking_session."""
        try:
            if ',' in source_image_base64:
                source_image_base64 = source_image_base64.split(',', 1)[1]
            if ',' in target_image_base64:
                target_image_base64 = target_image_base64.split(',', 1)[1]

            source_data = base64.b64decode(source_image_base64)
            target_data = base64.b64decode(target_image_base64)

            return self.prepare_tracking_session(
                source_data, target_data,
                confidence_threshold, match_distance_threshold
            )
        except Exception as e:
            logger.exception("Failed to decode base64 for tracking prepare")
            return {'success': False, 'error': str(e), 'sessionId': '', 'homographyMatrix': None}

    def match_single_follicle(
        self,
        session_id: str,
        source_bbox: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Match a single source follicle against the target image using
        a cached tracking session. Projects the source center through
        the homography, crops a local area in the target, runs YOLO
        detection on the crop, and NCC-verifies to find the best match.

        Args:
            session_id: ID of a prepared tracking session
            source_bbox: {x, y, width, height} of the source follicle

        Returns:
            Dict with success, match (or null)
        """
        session = self._tracking_sessions.get(session_id)
        if session is None:
            return {'success': False, 'error': f'Session {session_id} not found', 'match': None}

        if self._loaded_model is None:
            return {'success': False, 'error': 'No model loaded', 'match': None}

        try:
            H = session['H']
            target_array = session['target_array']
            source_gray = session['source_gray']
            target_gray = session['target_gray']
            scaled_threshold = session['scaled_threshold']
            confidence_threshold = session['confidence_threshold']

            PATCH_SIZE = 32
            PATCH_EXPAND = 1.5
            NCC_FLOOR = 0.4

            # Create source detection
            src_x = source_bbox['x']
            src_y = source_bbox['y']
            src_w = source_bbox['width']
            src_h = source_bbox['height']
            src_cx = src_x + src_w / 2.0
            src_cy = src_y + src_h / 2.0
            src_area = src_w * src_h

            # Extract source patch
            def _extract_patch(gray_img, x, y, w, h):
                img_h, img_w = gray_img.shape[:2]
                cx, cy = x + w / 2.0, y + h / 2.0
                half_w = w * PATCH_EXPAND / 2.0
                half_h = h * PATCH_EXPAND / 2.0
                x1 = max(0, int(cx - half_w))
                y1 = max(0, int(cy - half_h))
                x2 = min(img_w, int(cx + half_w))
                y2 = min(img_h, int(cy + half_h))
                if x2 <= x1 or y2 <= y1:
                    return None
                crop = gray_img[y1:y2, x1:x2]
                interp = cv2.INTER_AREA if crop.shape[0] > PATCH_SIZE else cv2.INTER_LINEAR
                return cv2.resize(crop, (PATCH_SIZE, PATCH_SIZE), interpolation=interp)

            src_patch = _extract_patch(source_gray, src_x, src_y, src_w, src_h)
            if src_patch is None:
                return {'success': True, 'match': None}

            # Project source center through homography
            src_center = np.float32([[src_cx, src_cy]]).reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(src_center, H).reshape(-1, 2)
            proj_x, proj_y = float(transformed[0][0]), float(transformed[0][1])

            # Crop a region around the projected point in target for YOLO detection
            target_h_px, target_w_px = target_array.shape[:2]
            crop_radius = max(int(scaled_threshold), int(max(src_w, src_h) * 5))
            crop_x1 = max(0, int(proj_x - crop_radius))
            crop_y1 = max(0, int(proj_y - crop_radius))
            crop_x2 = min(target_w_px, int(proj_x + crop_radius))
            crop_y2 = min(target_h_px, int(proj_y + crop_radius))

            if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                logger.info(f"Projected point ({proj_x:.0f},{proj_y:.0f}) outside target image")
                return {'success': True, 'match': None}

            target_crop = target_array[crop_y1:crop_y2, crop_x1:crop_x2]

            # Run YOLO on the crop
            device = self._get_inference_device()
            results = self._loaded_model.predict(
                target_crop, conf=confidence_threshold, verbose=False,
                imgsz=self._loaded_model_imgsz, device=device
            )

            # Extract detections and adjust coordinates to full image space
            crop_detections = self._extract_detections(results, target_crop.shape)
            target_detections = []
            for d in crop_detections:
                target_detections.append(DetectionPrediction(
                    x=d.x + crop_x1,
                    y=d.y + crop_y1,
                    width=d.width,
                    height=d.height,
                    confidence=d.confidence,
                    class_id=d.class_id,
                    class_name=d.class_name
                ))

            if not target_detections:
                logger.info(f"No detections in crop around ({proj_x:.0f},{proj_y:.0f})")
                return {'success': True, 'match': None}

            # NCC-verify each target detection against source patch
            best_score = -1.0
            best_det = None

            for det in target_detections:
                tgt_cx = det.x + det.width / 2.0
                tgt_cy = det.y + det.height / 2.0
                dist = np.sqrt((proj_x - tgt_cx) ** 2 + (proj_y - tgt_cy) ** 2)

                if dist > scaled_threshold:
                    continue

                tgt_patch = _extract_patch(target_gray, det.x, det.y, det.width, det.height)
                if tgt_patch is None:
                    continue

                ncc = float(cv2.matchTemplate(src_patch, tgt_patch, cv2.TM_CCOEFF_NORMED)[0][0])
                if ncc < NCC_FLOOR:
                    continue

                distance_score = max(0.0, 1.0 - dist / scaled_threshold)
                tgt_area = det.width * det.height
                size_score = min(src_area, tgt_area) / max(src_area, tgt_area + 1e-6)
                combined = distance_score * max(ncc, 0.0) * (0.5 + 0.5 * size_score)

                if combined > best_score:
                    best_score = combined
                    best_det = det

            if best_det is None:
                logger.info(f"No NCC-verified match near ({proj_x:.0f},{proj_y:.0f})")
                return {'success': True, 'match': None}

            logger.info(f"Single follicle matched: confidence={best_score:.3f}")
            return {
                'success': True,
                'match': {
                    'targetDetection': best_det.to_dict(),
                    'confidence': round(best_score, 4),
                    'transformedX': round(proj_x, 2),
                    'transformedY': round(proj_y, 2),
                }
            }

        except Exception as e:
            logger.exception("Failed to match single follicle")
            return {'success': False, 'error': str(e), 'match': None}

    # ------------------------------------------------------------------
    # Template-based single follicle tracking (no homography, no YOLO)
    # ------------------------------------------------------------------

    def prepare_template_session(
        self,
        target_file_path: str,
    ) -> Dict[str, Any]:
        """
        Prepare a template-matching session: read target image from disk
        and build a Gaussian pyramid.  Only the target is needed at prepare
        time — the source patch is sent per-click.

        Returns:
            Dict with success, sessionId
        """
        try:
            # cv2.imread handles EXIF rotation and decodes directly to
            # the format we need — ~2x faster than PIL → numpy → cvtColor
            # for 200MP images.
            target_gray = cv2.imread(target_file_path, cv2.IMREAD_GRAYSCALE)
            if target_gray is None:
                return {'success': False, 'error': f'Could not read image: {target_file_path}', 'sessionId': '', 'homographyMatrix': None}

            # Build 5-level Gaussian pyramid (deeper = faster coarse search
            # on very large images; L3 at 1/8 resolution is the sweet spot
            # for 200MP images)
            level_1 = cv2.pyrDown(target_gray)
            level_2 = cv2.pyrDown(level_1)
            level_3 = cv2.pyrDown(level_2)
            level_4 = cv2.pyrDown(level_3)
            target_pyramid = [target_gray, level_1, level_2, level_3, level_4]

            session_id = str(uuid.uuid4())
            self._tracking_sessions[session_id] = {
                'type': 'template',
                'target_pyramid': target_pyramid,
                'target_h': target_gray.shape[0],
                'target_w': target_gray.shape[1],
            }

            logger.info(
                f"Template session {session_id} prepared: "
                f"target={target_gray.shape[1]}x{target_gray.shape[0]}"
            )

            return {
                'success': True,
                'sessionId': session_id,
                'homographyMatrix': None,
            }

        except Exception as e:
            logger.exception("Failed to prepare template session")
            return {'success': False, 'error': str(e), 'sessionId': '', 'homographyMatrix': None}

    @staticmethod
    def _match_at_scale(template, search_img, scale):
        """Run matchTemplate at a given scale factor."""
        if abs(scale - 1.0) > 0.01:
            new_w = max(1, int(template.shape[1] * scale))
            new_h = max(1, int(template.shape[0] * scale))
            tmpl = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        else:
            tmpl = template

        if tmpl.shape[0] > search_img.shape[0] or tmpl.shape[1] > search_img.shape[1]:
            return None, -1.0

        result = cv2.matchTemplate(search_img, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        return max_loc, float(max_val)

    def _ncc_match_in_pyramid(
        self,
        context_patch: np.ndarray,
        target_pyramid: list,
        fc_ox: float,
        fc_oy: float,
        src_w: float,
        src_h: float,
        expected_scale: float = 1.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Core 3-pass NCC matching: coarse → medium → fine.
        Shared by template_match_single and match_video_frame's per-patch
        helper. Returns a single-point match dict or None.
        """
        NCC_FLOOR = 0.3
        MIN_TEMPLATE_DIM = 12
        base = expected_scale if expected_scale and expected_scale > 0 else 1.0
        min_patch_dim = min(context_patch.shape[0], context_patch.shape[1])

        # --- Coarse pass — dynamically choose pyramid level ---
        COARSE_LEVEL = 3
        for lvl in range(len(target_pyramid) - 1, -1, -1):
            down = 2 ** lvl
            worst_dim = int((min_patch_dim / down) * base * 0.7)
            if worst_dim >= MIN_TEMPLATE_DIM:
                COARSE_LEVEL = lvl
                break

        COARSE_DOWN = 2 ** COARSE_LEVEL
        coarse_template = cv2.resize(
            context_patch,
            (max(1, context_patch.shape[1] // COARSE_DOWN), max(1, context_patch.shape[0] // COARSE_DOWN)),
            interpolation=cv2.INTER_AREA
        )
        target_coarse = target_pyramid[COARSE_LEVEL]

        best_coarse_loc = None
        best_coarse_score = -1.0
        best_coarse_scale = 1.0

        for s in [base * 0.7, base * 0.85, base, base * 1.15, base * 1.3]:
            loc, score = self._match_at_scale(coarse_template, target_coarse, s)
            if loc is not None and score > best_coarse_score:
                best_coarse_score = score
                best_coarse_loc = loc
                best_coarse_scale = s

        if best_coarse_loc is None:
            return None

        ct_w = int(coarse_template.shape[1] * best_coarse_scale)
        ct_h = int(coarse_template.shape[0] * best_coarse_scale)
        coarse_center_x = (best_coarse_loc[0] + ct_w / 2.0) * COARSE_DOWN
        coarse_center_y = (best_coarse_loc[1] + ct_h / 2.0) * COARSE_DOWN

        # --- Medium pass (pyramid level 1 = ½ resolution) ---
        medium_template = cv2.resize(
            context_patch,
            (max(1, context_patch.shape[1] // 2), max(1, context_patch.shape[0] // 2)),
            interpolation=cv2.INTER_AREA
        )
        target_medium = target_pyramid[1]

        win_radius = max(medium_template.shape[0], medium_template.shape[1]) * 2
        med_cx = coarse_center_x / 2.0
        med_cy = coarse_center_y / 2.0
        med_x1 = max(0, int(med_cx - win_radius))
        med_y1 = max(0, int(med_cy - win_radius))
        med_x2 = min(target_medium.shape[1], int(med_cx + win_radius))
        med_y2 = min(target_medium.shape[0], int(med_cy + win_radius))
        search_medium = target_medium[med_y1:med_y2, med_x1:med_x2]

        best_med_loc = None
        best_med_score = -1.0
        best_med_scale = best_coarse_scale

        for s in [best_coarse_scale * 0.9, best_coarse_scale, best_coarse_scale * 1.1]:
            loc, score = self._match_at_scale(medium_template, search_medium, s)
            if loc is not None and score > best_med_score:
                best_med_score = score
                best_med_loc = loc
                best_med_scale = s

        if best_med_loc is None:
            return None

        mt_w = int(medium_template.shape[1] * best_med_scale)
        mt_h = int(medium_template.shape[0] * best_med_scale)
        medium_center_x = (best_med_loc[0] + med_x1 + mt_w / 2.0) * 2.0
        medium_center_y = (best_med_loc[1] + med_y1 + mt_h / 2.0) * 2.0

        # --- Fine pass (pyramid level 0 = full resolution) ---
        target_full = target_pyramid[0]

        win_radius_fine = int(max(context_patch.shape[0], context_patch.shape[1]) * 1.5)
        fine_x1 = max(0, int(medium_center_x - win_radius_fine))
        fine_y1 = max(0, int(medium_center_y - win_radius_fine))
        fine_x2 = min(target_full.shape[1], int(medium_center_x + win_radius_fine))
        fine_y2 = min(target_full.shape[0], int(medium_center_y + win_radius_fine))
        search_fine = target_full[fine_y1:fine_y2, fine_x1:fine_x2]

        best_fine_loc = None
        best_fine_score = -1.0
        best_fine_scale = best_med_scale

        for s in [best_med_scale - 0.05, best_med_scale, best_med_scale + 0.05]:
            if s <= 0:
                continue
            loc, score = self._match_at_scale(context_patch, search_fine, s)
            if loc is not None and score > best_fine_score:
                best_fine_score = score
                best_fine_loc = loc
                best_fine_scale = s

        # Optional rotation if NCC is weak
        if best_fine_score < 0.5:
            for angle in [-5, 5]:
                rot_center = (context_patch.shape[1] // 2, context_patch.shape[0] // 2)
                M = cv2.getRotationMatrix2D(rot_center, angle, 1.0)
                rotated = cv2.warpAffine(
                    context_patch, M,
                    (context_patch.shape[1], context_patch.shape[0]),
                    borderMode=cv2.BORDER_REPLICATE
                )
                loc, score = self._match_at_scale(rotated, search_fine, best_fine_scale)
                if loc is not None and score > best_fine_score:
                    best_fine_score = score
                    best_fine_loc = loc

        if best_fine_loc is None or best_fine_score < NCC_FLOOR:
            return None

        scaled_fc_ox = fc_ox * best_fine_scale
        scaled_fc_oy = fc_oy * best_fine_scale
        target_cx = fine_x1 + best_fine_loc[0] + scaled_fc_ox
        target_cy = fine_y1 + best_fine_loc[1] + scaled_fc_oy
        target_w = src_w * best_fine_scale
        target_h = src_h * best_fine_scale

        return {
            'targetDetection': {
                'x': round(target_cx - target_w / 2.0, 2),
                'y': round(target_cy - target_h / 2.0, 2),
                'width': round(target_w, 2),
                'height': round(target_h, 2),
                'confidence': round(best_fine_score, 4),
                'classId': 0,
                'className': 'follicle',
            },
            'confidence': round(best_fine_score, 4),
            'transformedX': round(target_cx, 2),
            'transformedY': round(target_cy, 2),
            'matchScale': float(best_fine_scale),
        }

    def template_match_single(
        self,
        session_id: str,
        source_patch_data: bytes,
        follicle_offset_x: float,
        follicle_offset_y: float,
        follicle_width: float,
        follicle_height: float,
        expected_scale: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Match a single source follicle in the target image using
        multi-scale pyramid NCC template matching.
        """
        session = self._tracking_sessions.get(session_id)
        if session is None:
            return {'success': False, 'error': f'Session {session_id} not found', 'match': None}

        if session.get('type') != 'template':
            return {'success': False, 'error': 'Session is not a template session', 'match': None}

        try:
            target_pyramid = session['target_pyramid']

            patch_array = np.frombuffer(source_patch_data, dtype=np.uint8)
            patch_img = cv2.imdecode(patch_array, cv2.IMREAD_GRAYSCALE)
            if patch_img is None or patch_img.size == 0:
                return {'success': True, 'match': None}

            match = self._ncc_match_in_pyramid(
                patch_img, target_pyramid,
                follicle_offset_x, follicle_offset_y,
                follicle_width, follicle_height,
                expected_scale
            )

            if match is None:
                return {'success': True, 'match': None}

            logger.info(
                f"Template match: ncc={match['confidence']:.3f}, "
                f"target=({match['transformedX']:.0f},{match['transformedY']:.0f})"
            )
            return {'success': True, 'match': match}

        except Exception as e:
            logger.exception("Failed template match for single follicle")
            return {'success': False, 'error': str(e), 'match': None}

    # ------------------------------------------------------------------
    # Video frame-by-frame tracking
    # ------------------------------------------------------------------

    def _auto_calibrate_scale(
        self,
        patch_gray: np.ndarray,
        target_gray: np.ndarray,
    ) -> float:
        """
        Scan a wide range of scales to find the true scale relationship
        between a source patch and a target image. Returns the best scale.
        Uses a coarse pyramid level for speed.
        """
        # Use a pyramid level for fast scanning
        target_small = cv2.pyrDown(cv2.pyrDown(target_gray))  # 1/4
        down = 4

        best_score = -1.0
        best_scale = 1.0

        # Coarse scan: 10% to 95% in 5% steps
        for s_pct in range(10, 100, 5):
            s = s_pct / 100.0
            nw = max(3, int(patch_gray.shape[1] / down * s))
            nh = max(3, int(patch_gray.shape[0] / down * s))
            tmpl = cv2.resize(patch_gray, (nw, nh), interpolation=cv2.INTER_AREA)
            if tmpl.shape[0] >= target_small.shape[0] or tmpl.shape[1] >= target_small.shape[1]:
                continue
            result = cv2.matchTemplate(target_small, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val > best_score:
                best_score = max_val
                best_scale = s

        # Fine refinement: ±10% around best in 2% steps
        fine_best_score = best_score
        fine_best_scale = best_scale
        for s_pct in range(int(best_scale * 100) - 10, int(best_scale * 100) + 11, 2):
            s = s_pct / 100.0
            if s <= 0:
                continue
            nw = max(3, int(patch_gray.shape[1] / down * s))
            nh = max(3, int(patch_gray.shape[0] / down * s))
            tmpl = cv2.resize(patch_gray, (nw, nh), interpolation=cv2.INTER_AREA)
            if tmpl.shape[0] >= target_small.shape[0] or tmpl.shape[1] >= target_small.shape[1]:
                continue
            result = cv2.matchTemplate(target_small, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val > fine_best_score:
                fine_best_score = max_val
                fine_best_scale = s

        logger.info(f"Auto-calibrated scale: {fine_best_scale:.2f} (ncc={fine_best_score:.3f})")
        return fine_best_scale

    # ------------------------------------------------------------------
    # Dual-point video tracking helpers
    # ------------------------------------------------------------------

    def _match_one_patch_in_frame(
        self,
        patch: np.ndarray,
        frame_gray: np.ndarray,
        point_in_patch_x: float,
        point_in_patch_y: float,
        prev_cx: Optional[float],
        prev_cy: Optional[float],
        prev_scale: float,
        expected_scale: float,
        follicle_width: float,
        follicle_height: float,
        get_pyramid: Callable[[], list],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[float], Optional[float], Optional[float]]:
        """
        Match one patch against one video frame. Uses fast temporal
        prediction (_local_match) when prev_cx/cy are available, falls back
        to full pyramid NCC on failure or at frame 0.

        Returns (match_dict or None, new_cx, new_cy, new_scale).
        On failure, returns (None, None, None, None).

        ``get_pyramid`` is a zero-arg callable that lazily builds and caches
        the target pyramid so both patch matches in a frame share one
        pyramid without building it unless a full search is actually needed.
        """
        # Fast path: temporal prediction
        if prev_cx is not None and prev_cy is not None:
            match, new_cx, new_cy, new_scale = self._local_match(
                patch, frame_gray,
                point_in_patch_x, point_in_patch_y,
                prev_cx, prev_cy, prev_scale,
                follicle_width, follicle_height,
            )
            if match is not None:
                return match, new_cx, new_cy, new_scale
            # Local match failed — fall through to full pyramid

        # Slow path: full 3-pass pyramid NCC
        target_pyramid = get_pyramid()
        match = self._ncc_match_in_pyramid(
            patch, target_pyramid,
            point_in_patch_x, point_in_patch_y,
            follicle_width, follicle_height,
            expected_scale,
        )
        if match is None:
            return None, None, None, None

        return (
            match,
            float(match['transformedX']),
            float(match['transformedY']),
            float(match.get('matchScale', expected_scale)),
        )

    @staticmethod
    def _rigid_check(
        origin_match: Optional[Dict[str, Any]],
        tip_match: Optional[Dict[str, Any]],
        initial_dist: float,
        expected_scale: float,
    ) -> str:
        """
        Verify that the two independent matches are geometrically consistent
        with the initial origin-to-tip distance. Returns one of:
          'both'         — both matches are trustworthy
          'origin_only'  — tip match is untrustworthy, keep origin
          'tip_only'     — origin match is untrustworthy, keep tip
          'neither'      — both failed
        """
        if origin_match is None and tip_match is None:
            return 'neither'
        if origin_match is None:
            return 'tip_only'
        if tip_match is None:
            return 'origin_only'

        ox = float(origin_match['transformedX'])
        oy = float(origin_match['transformedY'])
        tx = float(tip_match['transformedX'])
        ty = float(tip_match['transformedY'])
        observed_dist = float(np.hypot(tx - ox, ty - oy))
        expected_dist = initial_dist * expected_scale
        tolerance = max(
            RIGID_ABS_TOLERANCE_PX,
            RIGID_REL_TOLERANCE * expected_dist,
        )

        if abs(observed_dist - expected_dist) <= tolerance:
            return 'both'

        # Rigid check failed — reject the lower-confidence match
        origin_conf = float(origin_match['confidence'])
        tip_conf = float(tip_match['confidence'])
        if origin_conf >= tip_conf:
            return 'origin_only'
        else:
            return 'tip_only'

    @staticmethod
    def _extrapolate_from_trusted(
        trusted_cx: float,
        trusted_cy: float,
        trusted_scale: float,
        initial_dx: float,
        initial_dy: float,
        extrapolate_tip: bool,
    ) -> Tuple[float, float]:
        """
        Reconstruct a failed point's position from the trusted point using
        the initial source-image origin→tip relationship, scaled by the
        trusted point's current match scale.

        ``initial_dx/dy`` is ``(tipX - originX, tipY - originY)`` measured
        in source-image pixels when the session was prepared.

        If ``extrapolate_tip`` is True, we assume the tip failed and the
        origin is trusted:   tip = origin + initial_delta * scale.
        Otherwise we assume the origin failed and the tip is trusted:
                              origin = tip   - initial_delta * scale.

        This extrapolation uses the initial direction vector and therefore
        will be wrong under camera rotation — but the rescue strategy is
        that the failed patch's prev_* state is cleared so the next frame
        will attempt a full pyramid search and re-acquire at its true
        position once it can be found again.
        """
        if extrapolate_tip:
            return (
                trusted_cx + initial_dx * trusted_scale,
                trusted_cy + initial_dy * trusted_scale,
            )
        else:
            return (
                trusted_cx - initial_dx * trusted_scale,
                trusted_cy - initial_dy * trusted_scale,
            )

    # ------------------------------------------------------------------
    # Lucas-Kanade optical flow tracking helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_gray_pyramid(gray: np.ndarray) -> list:
        """
        Build the 5-level target pyramid [full, 1/2, 1/4, 1/8, 1/16] used by
        _ncc_match_in_pyramid. Factored out so both the NCC and LK matchers
        can reuse the exact same pyramid construction.
        """
        l1 = cv2.pyrDown(gray)
        l2 = cv2.pyrDown(l1)
        l3 = cv2.pyrDown(l2)
        l4 = cv2.pyrDown(l3)
        return [gray, l1, l2, l3, l4]

    @staticmethod
    def _fb_error_to_confidence(fb_err: float) -> float:
        """
        Map a forward-backward round-trip error (pixels) onto a 0..1
        confidence value comparable to NCC's [0,1] score. fb_err = 0 maps
        to 1.0, fb_err >= LK_FB_ERROR_PX maps to 0.0.
        """
        if fb_err <= 0:
            return 1.0
        if fb_err >= LK_FB_ERROR_PX:
            return 0.0
        return 1.0 - (fb_err / LK_FB_ERROR_PX)

    def _lk_track_two_points(
        self,
        prev_gray: np.ndarray,
        cur_gray: np.ndarray,
        prev_points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run pyramidal Lucas-Kanade on exactly 2 points between prev_gray
        and cur_gray, then perform a forward-backward round-trip check
        (track cur -> prev and compare against the original prev_points).

        Args:
            prev_gray: previous frame, uint8 single-channel
            cur_gray:  current frame, uint8 single-channel
            prev_points: shape (2, 1, 2) float32 array of (x, y) points to track

        Returns:
            new_points: shape (2, 1, 2) float32, forward-tracked positions
            fb_errors:  shape (2,)       float, per-point FB error in pixels
            lk_status:  shape (2,)       bool,  per-point LK success (forward AND backward)
        """
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            LK_ITER_COUNT,
            LK_ITER_EPS,
        )

        # Forward pass: prev -> cur
        new_points, fwd_status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, cur_gray, prev_points, None,
            winSize=LK_WIN_SIZE,
            maxLevel=LK_MAX_LEVEL,
            criteria=criteria,
        )

        # Backward pass: cur -> prev  (same points, reversed frames)
        back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
            cur_gray, prev_gray, new_points, None,
            winSize=LK_WIN_SIZE,
            maxLevel=LK_MAX_LEVEL,
            criteria=criteria,
        )

        # Round-trip error in pixels, per point
        fb_diff = (back_points - prev_points).reshape(-1, 2)
        fb_errors = np.sqrt(np.sum(fb_diff * fb_diff, axis=1))

        # Trusted iff BOTH forward and backward LK reported success
        fwd_ok = fwd_status.reshape(-1).astype(bool)
        back_ok = back_status.reshape(-1).astype(bool)
        lk_status = fwd_ok & back_ok

        return new_points, fb_errors, lk_status

    @staticmethod
    def _similarity_transform_from_2_points(
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
    ) -> Dict[str, float]:
        """
        Closed-form 4-DOF 2D similarity transform (translation + rotation +
        uniform scale) from two point correspondences. Two correspondences
        uniquely determine a 4-DOF similarity, so this is algebraically exact
        (no least squares).

        Args:
            src_pts: shape (2, 2) — source points  [[x0, y0], [x1, y1]]
            dst_pts: shape (2, 2) — target points

        Returns a dict with 'scale', 'rotation_rad', 'tx', 'ty'. When the
        source points coincide (degenerate), returns identity-ish defaults.
        """
        src_dx = float(src_pts[1][0] - src_pts[0][0])
        src_dy = float(src_pts[1][1] - src_pts[0][1])
        dst_dx = float(dst_pts[1][0] - dst_pts[0][0])
        dst_dy = float(dst_pts[1][1] - dst_pts[0][1])

        src_dist = float(np.hypot(src_dx, src_dy))
        dst_dist = float(np.hypot(dst_dx, dst_dy))

        # Degenerate: source points coincide. Return an identity scale/rotation
        # and a pure translation aligning src_pts[0] with dst_pts[0].
        if src_dist < 1e-6:
            return {
                'scale': 1.0,
                'rotation_rad': 0.0,
                'tx': float(dst_pts[0][0] - src_pts[0][0]),
                'ty': float(dst_pts[0][1] - src_pts[0][1]),
            }

        scale = dst_dist / src_dist
        rotation = float(np.arctan2(dst_dy, dst_dx) - np.arctan2(src_dy, src_dx))

        # Align src_pts[0] -> dst_pts[0] under the rotation+scale
        cos_r = float(np.cos(rotation))
        sin_r = float(np.sin(rotation))
        sx0 = float(src_pts[0][0])
        sy0 = float(src_pts[0][1])
        tx = float(dst_pts[0][0]) - (cos_r * sx0 - sin_r * sy0) * scale
        ty = float(dst_pts[0][1]) - (sin_r * sx0 + cos_r * sy0) * scale

        return {'scale': scale, 'rotation_rad': rotation, 'tx': tx, 'ty': ty}

    @staticmethod
    def _apply_similarity_transform(
        point: Tuple[float, float],
        transform: Dict[str, float],
    ) -> Tuple[float, float]:
        """
        Apply a similarity transform (dict from _similarity_transform_from_2_points)
        to a single (x, y) point. Used to extrapolate a failed LK point from
        the trusted one while carrying the current rotation + scale.
        """
        s = transform['scale']
        r = transform['rotation_rad']
        cos_r = float(np.cos(r))
        sin_r = float(np.sin(r))
        px = float(point[0])
        py = float(point[1])
        out_x = (cos_r * px - sin_r * py) * s + transform['tx']
        out_y = (sin_r * px + cos_r * py) * s + transform['ty']
        return (out_x, out_y)

    def _build_lk_match_dict(
        self,
        origin_xy: Tuple[float, float],
        tip_xy: Tuple[float, float],
        origin_conf: float,
        tip_conf: float,
        rigid_valid: bool,
        lost_point: Optional[str],
        follicle_width: float,
        follicle_height: float,
        match_scale: float,
    ) -> Dict[str, Any]:
        """
        Assemble a match dict with the same shape as the NCC matcher's
        output so VideoTrackingView renders LK results identically.
        The displayed bbox is derived from the origin position and the
        current similarity-transform scale (not from an NCC targetDetection).
        """
        target_w = float(follicle_width) * float(match_scale)
        target_h = float(follicle_height) * float(match_scale)
        ox = float(origin_xy[0])
        oy = float(origin_xy[1])
        tx = float(tip_xy[0])
        ty = float(tip_xy[1])

        # Overall confidence mirrors the NCC flow's "min of the two" when both
        # are trusted, otherwise whichever point is still reporting a score.
        if origin_conf > 0 and tip_conf > 0:
            overall_conf = min(origin_conf, tip_conf)
        else:
            overall_conf = max(origin_conf, tip_conf)

        return {
            'targetDetection': {
                'x': round(ox - target_w / 2.0, 2),
                'y': round(oy - target_h / 2.0, 2),
                'width': round(target_w, 2),
                'height': round(target_h, 2),
                'confidence': round(overall_conf, 4),
                'classId': 0,
                'className': 'follicle',
            },
            'confidence': round(overall_conf, 4),
            'transformedX': round(ox, 2),
            'transformedY': round(oy, 2),
            'tipX': round(tx, 2),
            'tipY': round(ty, 2),
            'originConfidence': round(float(origin_conf), 4),
            'tipConfidence': round(float(tip_conf), 4),
            'rigidValid': bool(rigid_valid),
            'lostPoint': lost_point,
        }

    def prepare_video_session(
        self,
        video_file_path: str,
        origin_patch_data: bytes,
        tip_patch_data: bytes,
        origin_in_origin_patch_x: float,
        origin_in_origin_patch_y: float,
        tip_in_tip_patch_x: float,
        tip_in_tip_patch_y: float,
        initial_dx: float,
        initial_dy: float,
        follicle_width: float,
        follicle_height: float,
        expected_scale: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Open a video file and cache TWO source patches for dual-point
        frame-by-frame matching: one centered on the graft origin, one
        centered on the graft tip. Each patch is matched independently per
        frame; a rigid-consistency check validates that the two matches
        stay geometrically aligned with the source-image relationship.

        ``initial_dx`` / ``initial_dy`` = ``(tipX - originX, tipY - originY)``
        in source-image pixels, used for the rigid check and as the
        extrapolation delta when one point is lost.

        Auto-calibrates the true scale by matching the ORIGIN patch against
        the first video frame. One calibrated scale is used for both
        patches — hair graft texture is locally uniform, so a single scale
        suffices in practice.
        """
        try:
            cap = cv2.VideoCapture(video_file_path)
            if not cap.isOpened():
                return {'success': False, 'error': f'Could not open video: {video_file_path}'}

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Decode both patches
            origin_arr = np.frombuffer(origin_patch_data, dtype=np.uint8)
            origin_patch = cv2.imdecode(origin_arr, cv2.IMREAD_GRAYSCALE)
            if origin_patch is None or origin_patch.size == 0:
                cap.release()
                return {'success': False, 'error': 'Invalid origin patch data'}

            tip_arr = np.frombuffer(tip_patch_data, dtype=np.uint8)
            tip_patch = cv2.imdecode(tip_arr, cv2.IMREAD_GRAYSCALE)
            if tip_patch is None or tip_patch.size == 0:
                cap.release()
                return {'success': False, 'error': 'Invalid tip patch data'}

            # Read first frame for auto-calibration
            ret, first_frame = cap.read()
            if not ret:
                cap.release()
                return {'success': False, 'error': 'Could not read first video frame'}

            first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

            # Auto-calibrate scale using the ORIGIN patch only. This handles
            # different fields of view / crops / zoom levels between source
            # and video. One scale for both patches.
            calibrated_scale = self._auto_calibrate_scale(origin_patch, first_gray)

            # Reset video to start (frame 0 was consumed for calibration)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            initial_dist = float(np.hypot(initial_dx, initial_dy))

            session_id = str(uuid.uuid4())
            self._tracking_sessions[session_id] = {
                'type': 'video',
                'cap': cap,
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,

                # Two patches, one per tracked point
                'origin_patch_gray': origin_patch,
                'tip_patch_gray': tip_patch,

                # Where each point sits inside its own patch (≈ patch
                # center unless edge-clipped at the source image boundary)
                'origin_in_origin_patch_x': float(origin_in_origin_patch_x),
                'origin_in_origin_patch_y': float(origin_in_origin_patch_y),
                'tip_in_tip_patch_x': float(tip_in_tip_patch_x),
                'tip_in_tip_patch_y': float(tip_in_tip_patch_y),

                'follicle_width': follicle_width,
                'follicle_height': follicle_height,

                # Initial rigid relationship in source-image pixels
                'initial_dx': float(initial_dx),
                'initial_dy': float(initial_dy),
                'initial_dist': initial_dist,

                'expected_scale': calibrated_scale,
                'current_frame': 0,

                # Temporal prediction state — one per patch
                'prev_origin_cx': None,
                'prev_origin_cy': None,
                'prev_origin_scale': calibrated_scale,
                'prev_tip_cx': None,
                'prev_tip_cy': None,
                'prev_tip_scale': calibrated_scale,

                # Pipelined frame reading
                'next_frame': None,
                '_executor': None,
            }

            logger.info(
                f"Video session {session_id} prepared (dual-point): "
                f"{width}x{height} @ {fps:.1f}fps, {frame_count} frames, "
                f"calibrated_scale={calibrated_scale:.2f}, "
                f"initial_dist={initial_dist:.1f}px"
            )

            return {
                'success': True,
                'sessionId': session_id,
                'fps': fps,
                'frameCount': frame_count,
                'width': width,
                'height': height,
            }

        except Exception as e:
            logger.exception("Failed to prepare video session")
            return {'success': False, 'error': str(e)}

    def prepare_video_session_lk(
        self,
        video_file_path: str,
        origin_patch_data: bytes,
        tip_patch_data: bytes,
        origin_in_origin_patch_x: float,
        origin_in_origin_patch_y: float,
        tip_in_tip_patch_x: float,
        tip_in_tip_patch_y: float,
        initial_dx: float,
        initial_dy: float,
        follicle_width: float,
        follicle_height: float,
        expected_scale: float = 1.0,
        cooldown_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Open a video file and prepare a session for the Seek/Track/Cooldown
        state machine driving Lucas-Kanade optical flow tracking of the
        graft origin and tip.

        Unlike the earlier revision, this prepare call **does not** run an
        initial NCC seed on the first frame. The session opens in
        ``phase='seeking'`` and the first NCC attempt runs inside
        ``_seek_step`` on the first ``match_video_frame`` call. If NCC
        fails, the state machine enters cooldown and retries on a later
        frame — the session never fails the way the old code did when the
        scalp patch couldn't be located in frame 0.

        Patch decoding and scale calibration still happen here so the
        session has everything it needs for NCC in the seeking phase.
        """
        try:
            cap = cv2.VideoCapture(video_file_path)
            if not cap.isOpened():
                return {'success': False, 'error': f'Could not open video: {video_file_path}'}

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Decode both patches
            origin_arr = np.frombuffer(origin_patch_data, dtype=np.uint8)
            origin_patch = cv2.imdecode(origin_arr, cv2.IMREAD_GRAYSCALE)
            if origin_patch is None or origin_patch.size == 0:
                cap.release()
                return {'success': False, 'error': 'Invalid origin patch data'}

            tip_arr = np.frombuffer(tip_patch_data, dtype=np.uint8)
            tip_patch = cv2.imdecode(tip_arr, cv2.IMREAD_GRAYSCALE)
            if tip_patch is None or tip_patch.size == 0:
                cap.release()
                return {'success': False, 'error': 'Invalid tip patch data'}

            # Read first frame for scale auto-calibration. The seeked
            # position is returned to frame 0 so the first
            # match_video_frame call will re-read it for NCC.
            ret, first_frame = cap.read()
            if not ret:
                cap.release()
                return {'success': False, 'error': 'Could not read first video frame'}
            first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
            calibrated_scale = self._auto_calibrate_scale(origin_patch, first_gray)

            # Rewind so frame 0 is read fresh by the state machine.
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            initial_dist = float(np.hypot(initial_dx, initial_dy))
            cooldown = _clamp_cooldown_sec(cooldown_sec)

            session_id = str(uuid.uuid4())
            self._tracking_sessions[session_id] = {
                'type': 'video_lk',
                'cap': cap,
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,

                # Source patches kept for NCC seeding in the state machine
                'origin_patch_gray': origin_patch,
                'tip_patch_gray': tip_patch,
                'origin_in_origin_patch_x': float(origin_in_origin_patch_x),
                'origin_in_origin_patch_y': float(origin_in_origin_patch_y),
                'tip_in_tip_patch_x': float(tip_in_tip_patch_x),
                'tip_in_tip_patch_y': float(tip_in_tip_patch_y),

                'follicle_width': follicle_width,
                'follicle_height': follicle_height,

                # Initial rigid relationship in source-image pixels
                'initial_dx': float(initial_dx),
                'initial_dy': float(initial_dy),
                'initial_dist': initial_dist,

                'expected_scale': calibrated_scale,
                'current_frame': 0,

                # Seek/Track/Cooldown state
                'phase': 'seeking',
                'cooldown_sec': cooldown,
                'cooldown_until': None,
                'cooldown_reason': None,
                'last_trusted_origin': None,
                'last_trusted_tip': None,
                'last_trusted_scale': None,
                'last_trusted_bbox': None,

                # LK state — populated by _seek_step on successful NCC seed
                'prev_gray': None,
                'prev_points': None,
                'initial_points': None,
                'prev_transform': {
                    'scale': 1.0,
                    'rotation_rad': 0.0,
                    'tx': 0.0,
                    'ty': 0.0,
                },
                'has_emitted_frame_zero': False,

                # Pipelined frame reading (shared with NCC session)
                'next_frame': None,
                '_executor': None,
            }

            logger.info(
                f"Video session {session_id} prepared (seek/track/cooldown): "
                f"{width}x{height} @ {fps:.1f}fps, {frame_count} frames, "
                f"calibrated_scale={calibrated_scale:.2f}, "
                f"cooldown_sec={cooldown:.1f}, phase=seeking"
            )

            return {
                'success': True,
                'sessionId': session_id,
                'fps': fps,
                'frameCount': frame_count,
                'width': width,
                'height': height,
            }

        except Exception as e:
            logger.exception("Failed to prepare LK video session")
            return {'success': False, 'error': str(e)}

    def prepare_camera_session_lk(
        self,
        origin_patch_data: bytes,
        tip_patch_data: bytes,
        origin_in_origin_patch_x: float,
        origin_in_origin_patch_y: float,
        tip_in_tip_patch_x: float,
        tip_in_tip_patch_y: float,
        initial_dx: float,
        initial_dy: float,
        follicle_width: float,
        follicle_height: float,
        first_frame_data: bytes,
        expected_scale: float = 1.0,
        cooldown_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Prepare a live-camera tracking session for the Seek/Track/Cooldown
        state machine. Same payload as ``prepare_video_session_lk`` except
        the video file path is replaced by ``first_frame_data`` — a raw
        JPEG frame captured from the frontend's getUserMedia stream.

        The first frame is used ONLY for scale auto-calibration; it is not
        used to seed origin/tip anchors (the old behavior that could fail
        when the camera wasn't pointed at the follicle). Seeding happens
        inside ``_seek_step`` on each subsequent frame until NCC succeeds.
        """
        try:
            # Decode both source patches (identical to file-backed prep)
            origin_arr = np.frombuffer(origin_patch_data, dtype=np.uint8)
            origin_patch = cv2.imdecode(origin_arr, cv2.IMREAD_GRAYSCALE)
            if origin_patch is None or origin_patch.size == 0:
                return {'success': False, 'error': 'Invalid origin patch data'}

            tip_arr = np.frombuffer(tip_patch_data, dtype=np.uint8)
            tip_patch = cv2.imdecode(tip_arr, cv2.IMREAD_GRAYSCALE)
            if tip_patch is None or tip_patch.size == 0:
                return {'success': False, 'error': 'Invalid tip patch data'}

            # Decode the frontend-supplied first frame (scale calibration only)
            first_arr = np.frombuffer(first_frame_data, dtype=np.uint8)
            first_frame = cv2.imdecode(first_arr, cv2.IMREAD_COLOR)
            if first_frame is None or first_frame.size == 0:
                return {'success': False, 'error': 'Invalid first frame data'}

            height, width = first_frame.shape[:2]
            first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
            calibrated_scale = self._auto_calibrate_scale(origin_patch, first_gray)

            initial_dist = float(np.hypot(initial_dx, initial_dy))
            cooldown = _clamp_cooldown_sec(cooldown_sec)

            session_id = str(uuid.uuid4())
            self._tracking_sessions[session_id] = {
                'type': 'camera_lk',
                # Camera mode owns no cap, executor, or pipeline frame —
                # stop_video_session handles a missing 'cap' gracefully.
                'fps': 0.0,
                'frame_count': -1,
                'width': width,
                'height': height,

                # Source patches kept for NCC seeding in the state machine
                'origin_patch_gray': origin_patch,
                'tip_patch_gray': tip_patch,
                'origin_in_origin_patch_x': float(origin_in_origin_patch_x),
                'origin_in_origin_patch_y': float(origin_in_origin_patch_y),
                'tip_in_tip_patch_x': float(tip_in_tip_patch_x),
                'tip_in_tip_patch_y': float(tip_in_tip_patch_y),

                'follicle_width': follicle_width,
                'follicle_height': follicle_height,

                'initial_dx': float(initial_dx),
                'initial_dy': float(initial_dy),
                'initial_dist': initial_dist,

                'expected_scale': calibrated_scale,
                'current_frame': 0,

                # Seek/Track/Cooldown state
                'phase': 'seeking',
                'cooldown_sec': cooldown,
                'cooldown_until': None,
                'cooldown_reason': None,
                'last_trusted_origin': None,
                'last_trusted_tip': None,
                'last_trusted_scale': None,
                'last_trusted_bbox': None,

                # LK state — populated by _seek_step on successful NCC seed
                'prev_gray': None,
                'prev_points': None,
                'initial_points': None,
                'prev_transform': {
                    'scale': 1.0,
                    'rotation_rad': 0.0,
                    'tx': 0.0,
                    'ty': 0.0,
                },
                'has_emitted_frame_zero': False,
            }

            logger.info(
                f"Camera session {session_id} prepared (seek/track/cooldown): "
                f"{width}x{height}, calibrated_scale={calibrated_scale:.2f}, "
                f"cooldown_sec={cooldown:.1f}, phase=seeking"
            )

            return {
                'success': True,
                'sessionId': session_id,
                'fps': 0.0,
                'frameCount': -1,
                'width': width,
                'height': height,
            }

        except Exception as e:
            logger.exception("Failed to prepare LK camera session")
            return {'success': False, 'error': str(e)}

    @staticmethod
    def _read_frame_bg(cap):
        """Read a frame in a background thread for pipelining."""
        ret, frame = cap.read()
        return (ret, frame)

    def _local_match(self, patch, frame_gray,
                     point_in_patch_x: float, point_in_patch_y: float,
                     prev_cx, prev_cy, prev_scale,
                     follicle_width, follicle_height, search_radius=300):
        """
        Fast local-window NCC match around the previous position.
        Used for temporal prediction after frame 0.

        ``point_in_patch_x/y`` is the coordinate of the tracked point within
        the source patch (typically the patch center, or shifted when the
        source crop was clipped by the image edge). The returned match dict's
        ``transformedX/Y`` is the video-frame position of that point — i.e.
        the matched patch top-left plus ``point_in_patch * best_scale`` —
        NOT the patch center. This keeps the semantics of transformedX/Y
        consistent with ``_ncc_match_in_pyramid``.

        Returns (match_dict_or_None, new_cx, new_cy, new_scale).
        """
        fh, fw = frame_gray.shape
        lx1 = max(0, int(prev_cx - search_radius))
        ly1 = max(0, int(prev_cy - search_radius))
        lx2 = min(fw, int(prev_cx + search_radius))
        ly2 = min(fh, int(prev_cy + search_radius))
        local = frame_gray[ly1:ly2, lx1:lx2]

        if local.size == 0:
            return None, prev_cx, prev_cy, prev_scale

        best_loc, best_score, best_s = None, -1.0, prev_scale
        for s in [prev_scale - 0.02, prev_scale, prev_scale + 0.02]:
            if s <= 0:
                continue
            loc, score = self._match_at_scale(patch, local, s)
            if loc is not None and score > best_score:
                best_score = score
                best_loc = loc
                best_s = s

        NCC_FLOOR = 0.3
        if best_loc is None or best_score < NCC_FLOOR:
            return None, prev_cx, prev_cy, prev_scale

        # Matched patch top-left in full-resolution video coordinates.
        patch_tl_x = lx1 + best_loc[0]
        patch_tl_y = ly1 + best_loc[1]

        # Tracked point position in video coordinates. Uses the explicit
        # point-in-patch offset so edge-clipped crops still yield the correct
        # point location (rather than the patch center).
        new_cx = patch_tl_x + point_in_patch_x * best_s
        new_cy = patch_tl_y + point_in_patch_y * best_s

        target_w = follicle_width * best_s
        target_h = follicle_height * best_s

        match_result: Dict[str, Any] = {
            'targetDetection': {
                'x': round(new_cx - target_w / 2.0, 2),
                'y': round(new_cy - target_h / 2.0, 2),
                'width': round(target_w, 2),
                'height': round(target_h, 2),
                'confidence': round(best_score, 4),
                'classId': 0,
                'className': 'follicle',
            },
            'confidence': round(best_score, 4),
            'transformedX': round(new_cx, 2),
            'transformedY': round(new_cy, 2),
            'matchScale': float(best_s),
        }
        return match_result, new_cx, new_cy, best_s

    def match_video_frame(self, session_id: str) -> Dict[str, Any]:
        """
        Read the next frame and run the appropriate per-frame matcher
        based on the session type. NCC sessions ('video') go to the
        two-NCC + rigid-check matcher; LK sessions ('video_lk') go to
        the Lucas-Kanade optical flow matcher with NCC rescue. Camera
        sessions ('camera_lk') have no on-disk source and must be
        driven via ``match_camera_frame`` with caller-supplied frames.
        """
        session = self._tracking_sessions.get(session_id)
        if session is None:
            return {'success': False, 'error': f'Session {session_id} not found', 'done': True}
        session_type = session.get('type')
        if session_type == 'video':
            return self._match_video_frame_ncc(session)
        elif session_type == 'video_lk':
            return self._match_video_frame_lk(session)
        elif session_type == 'camera_lk':
            return {
                'success': False,
                'error': 'Camera sessions must be driven via match_camera_frame',
                'done': True,
            }
        else:
            return {
                'success': False,
                'error': f'Unknown video session type: {session_type}',
                'done': True,
            }

    def match_camera_frame(self, session_id: str, frame_data) -> Dict[str, Any]:
        """
        Per-frame match for a live camera session. ``frame_data`` is
        raw JPEG bytes (or a base64-encoded string of the same) captured
        by the frontend from a ``getUserMedia`` stream. Unlike
        ``match_video_frame`` the backend does not own the frame source —
        each call is standalone, and session lifetime is controlled by
        the caller via ``stop_video_session``.
        """
        session = self._tracking_sessions.get(session_id)
        if session is None:
            return {'success': False, 'error': f'Session {session_id} not found', 'done': True}
        if session.get('type') != 'camera_lk':
            return {
                'success': False,
                'error': f'Session {session_id} is not a camera session',
                'done': True,
            }
        return self._match_camera_frame_lk(session, frame_data)

    def _match_video_frame_ncc(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """
        Match both the origin and tip patches independently via NCC, then
        run a rigid-consistency check to reject the weaker match when the
        two disagree geometrically. When exactly one point fails, its
        position is extrapolated from the trusted one using the initial
        source-image delta, and its temporal prediction state is cleared
        so the next frame will re-acquire via a full pyramid search.
        Pipelined frame reading overlaps I/O with matching.
        """
        try:
            from concurrent.futures import ThreadPoolExecutor

            cap = session['cap']

            # Use pipelined frame: if we pre-read a frame last iteration, use it
            if session.get('next_frame') is not None:
                ret, frame = session['next_frame']
                session['next_frame'] = None
            else:
                ret, frame = cap.read()

            if not ret:
                return {
                    'success': True,
                    'frameIndex': session['current_frame'],
                    'match': None,
                    'done': True,
                }

            frame_index = session['current_frame']
            session['current_frame'] += 1

            # Start reading next frame in background (pipeline)
            if session.get('_executor') is None:
                session['_executor'] = ThreadPoolExecutor(max_workers=1)
            future = session['_executor'].submit(self._read_frame_bg, cap)

            # Encode frame as JPEG for frontend display
            display_frame = frame
            fh, fw = frame.shape[:2]
            MAX_DISPLAY = 1280
            if max(fw, fh) > MAX_DISPLAY:
                scale_down = MAX_DISPLAY / max(fw, fh)
                display_frame = cv2.resize(frame, None, fx=scale_down, fy=scale_down, interpolation=cv2.INTER_AREA)
            _, jpeg_buf = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_jpeg_b64 = base64.b64encode(jpeg_buf).decode('ascii')

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            expected_scale = session['expected_scale']

            # Lazily build the target pyramid only when a full search is
            # actually needed. Both patches share the one pyramid.
            cached_pyramid: list = []

            def get_pyramid():
                if not cached_pyramid:
                    cached_pyramid.extend(self._build_gray_pyramid(gray))
                return cached_pyramid

            # Match origin patch
            origin_match, o_cx, o_cy, o_scale = self._match_one_patch_in_frame(
                session['origin_patch_gray'], gray,
                session['origin_in_origin_patch_x'],
                session['origin_in_origin_patch_y'],
                session.get('prev_origin_cx'),
                session.get('prev_origin_cy'),
                session.get('prev_origin_scale', expected_scale),
                expected_scale,
                session['follicle_width'], session['follicle_height'],
                get_pyramid,
            )

            # Match tip patch
            tip_match, t_cx, t_cy, t_scale = self._match_one_patch_in_frame(
                session['tip_patch_gray'], gray,
                session['tip_in_tip_patch_x'],
                session['tip_in_tip_patch_y'],
                session.get('prev_tip_cx'),
                session.get('prev_tip_cy'),
                session.get('prev_tip_scale', expected_scale),
                expected_scale,
                session['follicle_width'], session['follicle_height'],
                get_pyramid,
            )

            # Rigid consistency check
            status = self._rigid_check(
                origin_match, tip_match,
                session['initial_dist'], expected_scale,
            )

            # Combine results based on status
            final_origin_xy = None
            final_tip_xy = None
            lost_point: Optional[str] = None

            if status == 'both':
                final_origin_xy = (o_cx, o_cy)
                final_tip_xy = (t_cx, t_cy)
                session['prev_origin_cx'] = o_cx
                session['prev_origin_cy'] = o_cy
                session['prev_origin_scale'] = o_scale
                session['prev_tip_cx'] = t_cx
                session['prev_tip_cy'] = t_cy
                session['prev_tip_scale'] = t_scale

            elif status == 'origin_only':
                # Origin is trusted; extrapolate tip from origin using initial delta
                final_origin_xy = (o_cx, o_cy)
                final_tip_xy = self._extrapolate_from_trusted(
                    o_cx, o_cy, o_scale,
                    session['initial_dx'], session['initial_dy'],
                    extrapolate_tip=True,
                )
                session['prev_origin_cx'] = o_cx
                session['prev_origin_cy'] = o_cy
                session['prev_origin_scale'] = o_scale
                # Clear tip prev state so next frame attempts a full pyramid search
                session['prev_tip_cx'] = None
                session['prev_tip_cy'] = None
                session['prev_tip_scale'] = expected_scale
                lost_point = 'tip'

            elif status == 'tip_only':
                # Tip is trusted; extrapolate origin from tip using initial delta
                final_tip_xy = (t_cx, t_cy)
                final_origin_xy = self._extrapolate_from_trusted(
                    t_cx, t_cy, t_scale,
                    session['initial_dx'], session['initial_dy'],
                    extrapolate_tip=False,
                )
                session['prev_tip_cx'] = t_cx
                session['prev_tip_cy'] = t_cy
                session['prev_tip_scale'] = t_scale
                session['prev_origin_cx'] = None
                session['prev_origin_cy'] = None
                session['prev_origin_scale'] = expected_scale
                lost_point = 'origin'

            else:  # 'neither'
                # Wait for pipelined frame read before bailing out
                try:
                    session['next_frame'] = future.result(timeout=5)
                except Exception:
                    session['next_frame'] = None
                return {
                    'success': True,
                    'frameIndex': frame_index,
                    'match': None,
                    'frameData': frame_jpeg_b64,
                    'done': False,
                }

            # Assemble the combined match dict. The bbox comes from whichever
            # of the two matches is available (prefer origin when both are).
            bbox_source = origin_match if origin_match is not None else tip_match
            origin_conf = float(origin_match['confidence']) if origin_match is not None else 0.0
            tip_conf = float(tip_match['confidence']) if tip_match is not None else 0.0

            final_match = {
                'targetDetection': bbox_source['targetDetection'],
                'confidence': round(min(origin_conf, tip_conf) if (origin_match and tip_match) else max(origin_conf, tip_conf), 4),
                'transformedX': round(float(final_origin_xy[0]), 2),
                'transformedY': round(float(final_origin_xy[1]), 2),
                'tipX': round(float(final_tip_xy[0]), 2),
                'tipY': round(float(final_tip_xy[1]), 2),
                'originConfidence': round(origin_conf, 4),
                'tipConfidence': round(tip_conf, 4),
                'rigidValid': status == 'both',
                'lostPoint': lost_point,
            }

            # Wait for pipelined frame read
            try:
                session['next_frame'] = future.result(timeout=5)
            except Exception:
                session['next_frame'] = None

            return {
                'success': True,
                'frameIndex': frame_index,
                'match': final_match,
                'frameData': frame_jpeg_b64,
                'done': False,
            }

        except Exception as e:
            logger.exception(f"Failed to match video frame {session.get('current_frame', '?')}")
            return {'success': False, 'error': str(e), 'done': True}

    def _run_track_step(self, session: Dict[str, Any], gray: 'np.ndarray'):
        """
        Seek/Track/Cooldown state-machine dispatcher.

        Pure on (session, gray): no frame I/O, no JPEG encoding. Mutates
        the session's phase + LK state. Returns either a match dict (same
        shape as ``_build_lk_match_dict``) to emit, ``None`` when no
        markers should be drawn, or a full response-shaped dict when the
        state needs to surface ``cooldownRemaining`` / ``cooldownReason``
        (the caller unwraps this case — see ``_match_video_frame_lk``).
        """
        now = time.monotonic()

        # Auto-transition out of cooldown if the timer elapsed.
        if session.get('phase') == 'cooldown' and session.get('cooldown_until') is not None:
            if now >= session['cooldown_until']:
                logger.info(
                    f"Tracking: cooldown → seeking "
                    f"(reason={session.get('cooldown_reason')})"
                )
                session['phase'] = 'seeking'
                session['cooldown_until'] = None

        phase = session.get('phase', 'seeking')

        if phase == 'cooldown':
            return self._build_cooldown_match_dict(session, now)

        if phase == 'seeking':
            return self._seek_step(session, gray, now)

        # phase == 'tracking'
        return self._track_step(session, gray, now)

    def _seek_step(
        self,
        session: Dict[str, Any],
        gray: 'np.ndarray',
        now: float,
    ) -> Dict[str, Any]:
        """
        Seeking phase: run NCC on (origin, tip) patches against the
        current frame. On success, transition to tracking and emit a
        match dict. On failure, transition to cooldown.
        """
        expected_scale = session['expected_scale']
        target_pyramid = self._build_gray_pyramid(gray)

        origin_match = self._ncc_match_in_pyramid(
            session['origin_patch_gray'], target_pyramid,
            session['origin_in_origin_patch_x'],
            session['origin_in_origin_patch_y'],
            session['follicle_width'], session['follicle_height'],
            expected_scale,
        )
        if origin_match is None:
            self._enter_cooldown(session, now, 'seeking')
            return self._build_cooldown_match_dict(session, now)

        tip_match = self._ncc_match_in_pyramid(
            session['tip_patch_gray'], target_pyramid,
            session['tip_in_tip_patch_x'],
            session['tip_in_tip_patch_y'],
            session['follicle_width'], session['follicle_height'],
            expected_scale,
        )
        if tip_match is None:
            # Origin was found but tip failed NCC — we can't seed LK with
            # just one point (no rigid transform, no extrapolation basis).
            # Cool down and retry on a later frame.
            self._enter_cooldown(session, now, 'seeking')
            return self._build_cooldown_match_dict(session, now)

        origin_xy = (float(origin_match['transformedX']), float(origin_match['transformedY']))
        tip_xy = (float(tip_match['transformedX']), float(tip_match['transformedY']))

        # Seed LK state: prev_gray gets this frame, prev_points/initial_points
        # both hold the NCC-found positions, prev_transform resets to identity
        # at the session's expected_scale (the LK step will recompute each frame).
        initial_points = np.array(
            [[[origin_xy[0], origin_xy[1]]],
             [[tip_xy[0], tip_xy[1]]]],
            dtype=np.float32,
        )
        session['prev_gray'] = gray.copy()
        session['prev_points'] = initial_points
        session['initial_points'] = initial_points.copy()
        session['prev_transform'] = {
            'scale': float(expected_scale),
            'rotation_rad': 0.0,
            'tx': 0.0,
            'ty': 0.0,
        }

        # Update the last-trusted snapshot for future cooldown freeze-frames.
        self._update_last_trusted(session, origin_xy, tip_xy, float(expected_scale))

        # Transition to tracking. The next _track_step call will fall through
        # the frame-0 branch and emit the seed points as-is before LK begins
        # on the subsequent frame.
        session['phase'] = 'tracking'
        session['cooldown_reason'] = None
        session['has_emitted_frame_zero'] = True  # we emit the seed below

        logger.info(
            f"Tracking: seeking → tracking "
            f"(origin={origin_xy[0]:.0f},{origin_xy[1]:.0f}; "
            f"tip={tip_xy[0]:.0f},{tip_xy[1]:.0f}; "
            f"scale={expected_scale:.2f})"
        )

        return self._build_lk_match_dict(
            origin_xy=origin_xy,
            tip_xy=tip_xy,
            origin_conf=float(origin_match.get('confidence', 1.0)),
            tip_conf=float(tip_match.get('confidence', 1.0)),
            rigid_valid=True,
            lost_point=None,
            follicle_width=session['follicle_width'],
            follicle_height=session['follicle_height'],
            match_scale=float(expected_scale),
        )

    def _track_step(
        self,
        session: Dict[str, Any],
        gray: 'np.ndarray',
        now: float,
    ):
        """
        Tracking phase: run LK every frame. Origin is paramount — if
        origin is lost, cool down (no NCC rescue here). Tip-only loss
        stays in tracking with tip extrapolated. The rigid check's
        demotion rule only ever demotes tip, never origin.
        """
        prev_gray = session['prev_gray']
        prev_points = session['prev_points']

        new_points, fb_errors, lk_status = self._lk_track_two_points(
            prev_gray, gray, prev_points,
        )

        origin_trusted = (
            bool(lk_status[0])
            and float(fb_errors[0]) <= LK_FB_ERROR_PX
            and self._fb_error_to_confidence(float(fb_errors[0])) >= LK_CONFIDENCE_THRESHOLD
        )
        origin_conf = (
            self._fb_error_to_confidence(float(fb_errors[0])) if origin_trusted else 0.0
        )

        # Origin-paramount short-circuit: origin dead → cooldown.
        if not origin_trusted:
            self._enter_cooldown(session, now, 'origin_lost')
            return self._build_cooldown_match_dict(session, now)

        tip_trusted = (
            bool(lk_status[1])
            and float(fb_errors[1]) <= LK_FB_ERROR_PX
            and self._fb_error_to_confidence(float(fb_errors[1])) >= LK_CONFIDENCE_THRESHOLD
        )
        tip_conf = (
            self._fb_error_to_confidence(float(fb_errors[1])) if tip_trusted else 0.0
        )

        # Rigid-check demotion — tip only. Never demote origin.
        initial = session['initial_points'].reshape(2, 2)
        current = new_points.reshape(2, 2)
        if tip_trusted:
            transform = self._similarity_transform_from_2_points(initial, current)
            prev_scale = session['prev_transform']['scale']
            scale_jump = (
                abs(transform['scale'] - prev_scale) / max(prev_scale, 1e-6)
            )
            if scale_jump > LK_MAX_SCALE_JUMP:
                tip_trusted = False
                tip_conf = 0.0

        # Resolve final positions. Origin is always trusted at this point.
        if tip_trusted:
            final_origin = (float(current[0][0]), float(current[0][1]))
            final_tip = (float(current[1][0]), float(current[1][1]))
            lost_point: Optional[str] = None
            rigid_valid = True
            session['prev_transform'] = self._similarity_transform_from_2_points(
                initial, current,
            )
        else:
            final_origin = (float(current[0][0]), float(current[0][1]))
            final_tip = self._apply_similarity_transform(
                (float(initial[1][0]), float(initial[1][1])),
                session['prev_transform'],
            )
            lost_point = 'tip'
            rigid_valid = False

        # Update LK state for next frame.
        session['prev_gray'] = gray.copy()
        session['prev_points'] = np.array(
            [[[final_origin[0], final_origin[1]]],
             [[final_tip[0], final_tip[1]]]],
            dtype=np.float32,
        )

        match_scale = session['prev_transform']['scale']
        self._update_last_trusted(session, final_origin, final_tip, float(match_scale))

        return self._build_lk_match_dict(
            origin_xy=final_origin,
            tip_xy=final_tip,
            origin_conf=origin_conf,
            tip_conf=tip_conf,
            rigid_valid=rigid_valid,
            lost_point=lost_point,
            follicle_width=session['follicle_width'],
            follicle_height=session['follicle_height'],
            match_scale=match_scale,
        )

    def _enter_cooldown(
        self,
        session: Dict[str, Any],
        now: float,
        reason: str,
    ) -> None:
        """Transition to cooldown and clear LK state so the next seed is fresh."""
        cooldown = float(session.get('cooldown_sec', TRACK_COOLDOWN_DEFAULT_SEC))
        session['phase'] = 'cooldown'
        session['cooldown_until'] = now + cooldown
        session['cooldown_reason'] = reason
        # Clear LK state — next re-acquisition seeds from scratch via NCC.
        session['prev_gray'] = None
        session['prev_points'] = None
        session['initial_points'] = None
        session['has_emitted_frame_zero'] = False
        logger.info(
            f"Tracking: {session.get('phase_prev', '?')} → cooldown "
            f"(reason={reason}, duration={cooldown:.1f}s)"
        )

    @staticmethod
    def _update_last_trusted(
        session: Dict[str, Any],
        origin_xy: Tuple[float, float],
        tip_xy: Tuple[float, float],
        match_scale: float,
    ) -> None:
        """Snapshot the current trusted positions for cooldown UI freeze."""
        session['last_trusted_origin'] = (float(origin_xy[0]), float(origin_xy[1]))
        session['last_trusted_tip'] = (float(tip_xy[0]), float(tip_xy[1]))
        session['last_trusted_scale'] = float(match_scale)
        fw = float(session['follicle_width']) * float(match_scale)
        fh = float(session['follicle_height']) * float(match_scale)
        session['last_trusted_bbox'] = {
            'x': round(float(origin_xy[0]) - fw / 2.0, 2),
            'y': round(float(origin_xy[1]) - fh / 2.0, 2),
            'width': round(fw, 2),
            'height': round(fh, 2),
            'confidence': 0.0,
            'classId': 0,
            'className': 'follicle',
        }

    def _build_cooldown_match_dict(
        self,
        session: Dict[str, Any],
        now: float,
    ) -> Dict[str, Any]:
        """
        Wrap the outer response shape for a frame consumed during cooldown.
        Returned as a FULL response dict (including cooldownRemaining /
        cooldownReason) rather than just the inner match — the caller
        (_match_video_frame_lk / _match_camera_frame_lk) will detect the
        extra fields and forward them directly.
        """
        deadline = session.get('cooldown_until') or 0.0
        remaining = max(0.0, deadline - now)
        reason = session.get('cooldown_reason')
        last_origin = session.get('last_trusted_origin')
        last_tip = session.get('last_trusted_tip')
        match_scale = session.get('last_trusted_scale') or session.get('expected_scale', 1.0)
        bbox = session.get('last_trusted_bbox')

        if last_origin is None or last_tip is None or bbox is None:
            # No trusted snapshot yet — hide markers entirely, show only the label.
            match: Optional[Dict[str, Any]] = None
        else:
            match = {
                'targetDetection': bbox,
                'confidence': 0.0,
                'transformedX': round(float(last_origin[0]), 2),
                'transformedY': round(float(last_origin[1]), 2),
                'tipX': round(float(last_tip[0]), 2),
                'tipY': round(float(last_tip[1]), 2),
                'originConfidence': 0.0,
                'tipConfidence': 0.0,
                'rigidValid': False,
                'lostPoint': 'both',
                'matchScale': float(match_scale),
            }

        return {
            '__cooldown__': True,  # sentinel for the outer caller to unwrap
            'match': match,
            'cooldownRemaining': round(float(remaining), 2),
            'cooldownReason': reason,
        }

    def _match_video_frame_lk(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """
        File-backed Lucas-Kanade matcher: reads the next frame from
        ``session['cap']`` with background pipelining, JPEG-encodes it
        for the frontend, then delegates to ``_run_lk_step`` for the
        algorithm.
        """
        try:
            from concurrent.futures import ThreadPoolExecutor

            cap = session['cap']

            # Pipelined frame read (same pattern as NCC path)
            if session.get('next_frame') is not None:
                ret, frame = session['next_frame']
                session['next_frame'] = None
            else:
                ret, frame = cap.read()

            if not ret:
                return {
                    'success': True,
                    'frameIndex': session['current_frame'],
                    'match': None,
                    'done': True,
                }

            frame_index = session['current_frame']
            session['current_frame'] += 1

            # Kick off next frame read in background
            if session.get('_executor') is None:
                session['_executor'] = ThreadPoolExecutor(max_workers=1)
            future = session['_executor'].submit(self._read_frame_bg, cap)

            # Encode frame as JPEG for frontend display (same as NCC path)
            display_frame = frame
            fh, fw = frame.shape[:2]
            MAX_DISPLAY = 1280
            if max(fw, fh) > MAX_DISPLAY:
                scale_down = MAX_DISPLAY / max(fw, fh)
                display_frame = cv2.resize(
                    frame, None, fx=scale_down, fy=scale_down,
                    interpolation=cv2.INTER_AREA,
                )
            _, jpeg_buf = cv2.imencode(
                '.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
            )
            frame_jpeg_b64 = base64.b64encode(jpeg_buf).decode('ascii')

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result = self._run_track_step(session, gray)

            try:
                session['next_frame'] = future.result(timeout=5)
            except Exception:
                session['next_frame'] = None

            # State machine may return either a plain match dict (tracking)
            # or a cooldown-response wrapper carrying extra fields.
            if isinstance(result, dict) and result.get('__cooldown__'):
                return {
                    'success': True,
                    'frameIndex': frame_index,
                    'match': result.get('match'),
                    'cooldownRemaining': result.get('cooldownRemaining'),
                    'cooldownReason': result.get('cooldownReason'),
                    'frameData': frame_jpeg_b64,
                    'done': False,
                }

            return {
                'success': True,
                'frameIndex': frame_index,
                'match': result,
                'frameData': frame_jpeg_b64,
                'done': False,
            }

        except Exception as e:
            logger.exception(
                f"Failed to LK-match video frame {session.get('current_frame', '?')}"
            )
            return {'success': False, 'error': str(e), 'done': True}

    def _match_camera_frame_lk(
        self,
        session: Dict[str, Any],
        frame_data,
    ) -> Dict[str, Any]:
        """
        Live-camera Lucas-Kanade matcher. ``frame_data`` is raw JPEG
        bytes supplied by the frontend (captured from a ``getUserMedia``
        MediaStream). No cap, no pipelining, no JPEG-back: the frontend
        already holds the source bitmap for rendering.
        """
        try:
            if isinstance(frame_data, str):
                frame_bytes = base64.b64decode(frame_data)
            else:
                frame_bytes = bytes(frame_data)

            arr = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None or frame.size == 0:
                return {
                    'success': False,
                    'error': 'Could not decode camera frame',
                    'done': False,
                }

            frame_index = session['current_frame']
            session['current_frame'] += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result = self._run_track_step(session, gray)

            if isinstance(result, dict) and result.get('__cooldown__'):
                return {
                    'success': True,
                    'frameIndex': frame_index,
                    'match': result.get('match'),
                    'cooldownRemaining': result.get('cooldownRemaining'),
                    'cooldownReason': result.get('cooldownReason'),
                    'done': False,
                }

            return {
                'success': True,
                'frameIndex': frame_index,
                'match': result,
                'done': False,
            }
        except Exception as e:
            logger.exception(
                f"Failed to LK-match camera frame {session.get('current_frame', '?')}"
            )
            return {'success': False, 'error': str(e), 'done': False}

    def stop_video_session(self, session_id: str) -> Dict[str, Any]:
        """Release VideoCapture and clean up a video session."""
        session = self._tracking_sessions.pop(session_id, None)
        if session is None:
            return {'success': True}

        cap = session.get('cap')
        if cap:
            cap.release()

        executor = session.get('_executor')
        if executor:
            executor.shutdown(wait=False)

        logger.info(f"Video session {session_id} stopped")
        return {'success': True}

    def predict_tiled_base64(
        self,
        image_base64: str,
        confidence_threshold: float = 0.5,
        tile_size: int = 1024,
        overlap: int = 128,
        nms_threshold: float = 0.5,
        scale_factor: float = 1.0
    ) -> List[DetectionPrediction]:
        """
        Run tiled detection prediction on a base64-encoded image.

        Args:
            image_base64: Base64-encoded image string
            confidence_threshold: Minimum confidence to include detection
            tile_size: Size of each tile (default 1024 to match training)
            overlap: Pixel overlap between tiles (default 128)
            nms_threshold: IoU threshold for NMS merging (default 0.5)
            scale_factor: Factor to upscale image before inference (default 1.0)

        Returns:
            List of DetectionPrediction objects
        """
        try:
            # Remove data URL prefix if present
            if ',' in image_base64:
                image_base64 = image_base64.split(',', 1)[1]

            image_data = base64.b64decode(image_base64)
            return self.predict_tiled(image_data, confidence_threshold, tile_size, overlap, nms_threshold, scale_factor)
        except Exception as e:
            logger.exception("Failed to decode base64 image for tiled prediction")
            return []

    def export_onnx(self, model_path: str, output_path: str) -> Optional[str]:
        """
        Export a model to ONNX format.

        Args:
            model_path: Path to source .pt model
            output_path: Path for output .onnx file

        Returns:
            Path to exported ONNX file, or None if export failed
        """
        if not YOLO_AVAILABLE:
            logger.error("Cannot export: Ultralytics not installed")
            return None

        try:
            model = YOLO(model_path)

            # Export to ONNX
            export_path = model.export(format='onnx', dynamic=True)

            # Move to requested output path
            if export_path and Path(export_path).exists():
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(export_path, output_path)

                logger.info(f"Exported ONNX model to: {output_path}")
                return str(output_path)

            return None

        except Exception as e:
            logger.exception(f"ONNX export failed: {model_path}")
            return None

    def list_models(self) -> List[DetectionModelInfo]:
        """
        List all trained detection models.

        Returns:
            List of DetectionModelInfo objects
        """
        models = []

        for model_dir in self.models_dir.iterdir():
            if not model_dir.is_dir():
                continue

            info_file = model_dir / 'model_info.json'
            if info_file.exists():
                try:
                    with open(info_file) as f:
                        info_dict = json.load(f)

                    models.append(DetectionModelInfo(
                        id=info_dict.get('id', model_dir.name),
                        name=info_dict.get('name', model_dir.name),
                        path=info_dict.get('path', str(model_dir / 'weights' / 'best.pt')),
                        created_at=info_dict.get('created_at', ''),
                        epochs_trained=info_dict.get('epochs_trained', 0),
                        img_size=info_dict.get('img_size', 640),
                        metrics=info_dict.get('metrics', {})
                    ))
                except Exception as e:
                    logger.warning(f"Failed to load model info from {info_file}: {e}")

        # Sort by creation date (newest first)
        models.sort(key=lambda m: m.created_at, reverse=True)

        return models

    def delete_model(self, model_id: str) -> bool:
        """
        Delete a trained model.

        Args:
            model_id: Model ID to delete

        Returns:
            True if deleted successfully
        """
        model_dir = self.models_dir / model_id
        if model_dir.exists() and model_dir.is_dir():
            try:
                shutil.rmtree(model_dir)
                logger.info(f"Deleted model: {model_id}")
                return True
            except Exception as e:
                logger.exception(f"Failed to delete model: {model_id}")
                return False
        return False

    def get_resumable_models(self) -> List[Dict[str, Any]]:
        """
        Get models that have last.pt and incomplete training (can be resumed).

        Returns:
            List of model info dicts with resume-specific fields
        """
        import yaml

        resumable = []

        for model_dir in self.models_dir.iterdir():
            if not model_dir.is_dir():
                continue

            last_pt = model_dir / 'weights' / 'last.pt'
            results_csv = model_dir / 'results.csv'

            if not last_pt.exists():
                continue

            # Check if training was incomplete by reading results.csv
            epochs_completed = 0
            if results_csv.exists():
                try:
                    with open(results_csv) as f:
                        lines = f.readlines()
                        epochs_completed = len(lines) - 1  # Minus header
                except Exception as e:
                    logger.warning(f"Failed to read results.csv for {model_dir.name}: {e}")
                    continue

            # Load config to get total epochs
            total_epochs = 100  # Default
            args_file = model_dir / 'args.yaml'
            if args_file.exists():
                try:
                    with open(args_file) as f:
                        args = yaml.safe_load(f)
                        total_epochs = args.get('epochs', 100)
                except Exception as e:
                    logger.warning(f"Failed to read args.yaml for {model_dir.name}: {e}")

            # Only include if training was incomplete
            if epochs_completed < total_epochs:
                # Load model info
                info_file = model_dir / 'model_info.json'
                if info_file.exists():
                    try:
                        with open(info_file) as f:
                            info_dict = json.load(f)

                        # Get latest metrics from results.csv
                        metrics = info_dict.get('metrics', {})
                        if results_csv.exists():
                            try:
                                import csv
                                with open(results_csv) as f:
                                    reader = csv.DictReader(f)
                                    rows = list(reader)
                                    if rows:
                                        last_row = rows[-1]
                                        # Extract key metrics
                                        for key in ['metrics/mAP50(B)', 'metrics/precision(B)', 'metrics/recall(B)']:
                                            if key in last_row:
                                                try:
                                                    # Clean up key name for display
                                                    clean_key = key.replace('metrics/', '').replace('(B)', '')
                                                    metrics[clean_key] = float(last_row[key].strip())
                                                except (ValueError, AttributeError):
                                                    pass
                            except Exception as e:
                                logger.warning(f"Failed to parse metrics from results.csv: {e}")

                        resumable.append({
                            'id': info_dict.get('id', model_dir.name),
                            'name': info_dict.get('name', model_dir.name),
                            'path': str(last_pt),
                            'createdAt': info_dict.get('created_at', ''),
                            'epochsTrained': epochs_completed,
                            'imgSize': info_dict.get('img_size', 640),
                            'metrics': metrics,
                            'epochsCompleted': epochs_completed,
                            'totalEpochs': total_epochs,
                            'canResume': True,
                        })
                    except Exception as e:
                        logger.warning(f"Failed to load model info from {info_file}: {e}")
                else:
                    # Create minimal info from directory name
                    resumable.append({
                        'id': model_dir.name,
                        'name': model_dir.name,
                        'path': str(last_pt),
                        'createdAt': '',
                        'epochsTrained': epochs_completed,
                        'imgSize': 640,
                        'metrics': {},
                        'epochsCompleted': epochs_completed,
                        'totalEpochs': total_epochs,
                        'canResume': True,
                    })

        # Sort by creation date (newest first)
        resumable.sort(key=lambda m: m.get('createdAt', ''), reverse=True)

        logger.info(f"Found {len(resumable)} resumable models")
        return resumable

    def get_loaded_model_path(self) -> Optional[str]:
        """Get the path of the currently loaded model."""
        return self._loaded_model_path

    def get_loaded_model_backend(self) -> str:
        """Get the backend of the currently loaded model ('pytorch' or 'tensorrt')."""
        return self._loaded_model_backend

    def check_tensorrt_available(self) -> Dict[str, Any]:
        """
        Check if TensorRT is available on this system.

        Returns:
            Dict with 'available' bool and 'version' string (or None)
        """
        try:
            import tensorrt
            return {"available": True, "version": tensorrt.__version__}
        except ImportError:
            return {"available": False, "version": None}

    def export_tensorrt(
        self,
        model_path: str,
        output_path: Optional[str] = None,
        half: bool = True,
        imgsz: int = 640
    ) -> Dict[str, Any]:
        """
        Export a PyTorch model to TensorRT engine format.

        TensorRT provides GPU-optimized inference for faster detection
        on NVIDIA GPUs. The exported .engine file is GPU-architecture
        specific and not portable between different GPU types.

        Args:
            model_path: Path to source .pt model
            output_path: Optional path for output .engine file.
                        If None, saves alongside the .pt file.
            half: Use FP16 precision (recommended for consumer GPUs)
            imgsz: Input image size for the engine

        Returns:
            Dict with 'success' bool and 'engine_path' string
        """
        if not YOLO_AVAILABLE:
            return {"success": False, "error": "Ultralytics not installed"}

        try:
            # Check TensorRT availability first
            trt_status = self.check_tensorrt_available()
            if not trt_status["available"]:
                return {
                    "success": False,
                    "error": "TensorRT is not installed. Install it with: pip install tensorrt"
                }

            logger.info(f"Exporting {model_path} to TensorRT (half={half}, imgsz={imgsz})")

            # Load the PyTorch model
            model = YOLO(model_path)

            # Export to TensorRT format
            # Ultralytics handles the conversion internally
            engine_path = model.export(format='engine', half=half, imgsz=imgsz)

            if engine_path and Path(engine_path).exists():
                # If custom output path requested, move the file
                if output_path:
                    output_path = Path(output_path)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    import shutil
                    shutil.move(engine_path, output_path)
                    engine_path = str(output_path)

                logger.info(f"TensorRT export successful: {engine_path}")
                return {"success": True, "engine_path": str(engine_path)}
            else:
                return {"success": False, "error": "Export completed but engine file not found"}

        except Exception as e:
            logger.exception(f"TensorRT export failed: {model_path}")
            return {"success": False, "error": str(e)}


# Singleton instance
_detection_service_instance: Optional[YOLODetectionService] = None


def get_yolo_detection_service(models_dir: Optional[str] = None) -> YOLODetectionService:
    """
    Get the singleton YOLO detection service instance.

    Args:
        models_dir: Optional custom models directory (only used on first call)

    Returns:
        YOLODetectionService instance
    """
    global _detection_service_instance
    if _detection_service_instance is None:
        _detection_service_instance = YOLODetectionService(models_dir)
    return _detection_service_instance
