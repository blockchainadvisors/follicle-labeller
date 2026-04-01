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

        The source context patch (cropped in the frontend around the
        follicle) is received as image bytes — only a small region,
        not the full source image.

        Args:
            session_id: Prepared template session ID
            source_patch_data: PNG/JPEG bytes of the context patch
            follicle_offset_x: X offset of follicle center within the patch
            follicle_offset_y: Y offset of follicle center within the patch
            follicle_width: Original follicle bbox width
            follicle_height: Original follicle bbox height
        """
        session = self._tracking_sessions.get(session_id)
        if session is None:
            return {'success': False, 'error': f'Session {session_id} not found', 'match': None}

        if session.get('type') != 'template':
            return {'success': False, 'error': 'Session is not a template session', 'match': None}

        try:
            target_pyramid = session['target_pyramid']
            NCC_FLOOR = 0.3

            src_w = follicle_width
            src_h = follicle_height
            fc_ox = follicle_offset_x
            fc_oy = follicle_offset_y

            # Decode the source context patch sent from the frontend
            patch_array = np.frombuffer(source_patch_data, dtype=np.uint8)
            patch_img = cv2.imdecode(patch_array, cv2.IMREAD_GRAYSCALE)
            if patch_img is None or patch_img.size == 0:
                return {'success': True, 'match': None}

            context_patch = patch_img

            # --- Helper: run matchTemplate at one scale ---
            def _match_at_scale(template, search_img, scale):
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
                # max_loc is top-left of the match
                return max_loc, float(max_val)

            # --- Coarse pass — dynamically choose pyramid level ---
            # The template must stay >= MIN_TEMPLATE_DIM pixels after
            # downsampling + scale adjustment, otherwise NCC is unreliable.
            # For half-resolution targets (expected_scale=0.5) this means
            # using L2 instead of L3.
            MIN_TEMPLATE_DIM = 12
            base = expected_scale if expected_scale and expected_scale > 0 else 1.0
            min_patch_dim = min(context_patch.shape[0], context_patch.shape[1])

            COARSE_LEVEL = 3  # default for same-resolution
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
                loc, score = _match_at_scale(coarse_template, target_coarse, s)
                if loc is not None and score > best_coarse_score:
                    best_coarse_score = score
                    best_coarse_loc = loc
                    best_coarse_scale = s

            if best_coarse_loc is None:
                return {'success': True, 'match': None}

            # Scale coarse match center back to full resolution
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

            # Windowed search around coarse match (in half-res coords)
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
                loc, score = _match_at_scale(medium_template, search_medium, s)
                if loc is not None and score > best_med_score:
                    best_med_score = score
                    best_med_loc = loc
                    best_med_scale = s

            if best_med_loc is None:
                return {'success': True, 'match': None}

            # Scale medium match center back to full resolution
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
                loc, score = _match_at_scale(context_patch, search_fine, s)
                if loc is not None and score > best_fine_score:
                    best_fine_score = score
                    best_fine_loc = loc
                    best_fine_scale = s

            # --- Optional rotation if NCC is weak ---
            if best_fine_score < 0.5:
                for angle in [-5, 5]:
                    rot_center = (context_patch.shape[1] // 2, context_patch.shape[0] // 2)
                    M = cv2.getRotationMatrix2D(rot_center, angle, 1.0)
                    rotated = cv2.warpAffine(
                        context_patch, M,
                        (context_patch.shape[1], context_patch.shape[0]),
                        borderMode=cv2.BORDER_REPLICATE
                    )
                    loc, score = _match_at_scale(rotated, search_fine, best_fine_scale)
                    if loc is not None and score > best_fine_score:
                        best_fine_score = score
                        best_fine_loc = loc

            if best_fine_loc is None or best_fine_score < NCC_FLOOR:
                logger.info(f"Template match below threshold: ncc={best_fine_score:.3f}")
                return {'success': True, 'match': None}

            # --- Convert match location to target-image follicle center ---
            # best_fine_loc is top-left of the template match in the search window
            scaled_fc_ox = fc_ox * best_fine_scale
            scaled_fc_oy = fc_oy * best_fine_scale
            target_cx = fine_x1 + best_fine_loc[0] + scaled_fc_ox
            target_cy = fine_y1 + best_fine_loc[1] + scaled_fc_oy

            target_w = src_w * best_fine_scale
            target_h = src_h * best_fine_scale

            logger.info(
                f"Template match: ncc={best_fine_score:.3f}, "
                f"target=({target_cx:.0f},{target_cy:.0f}), scale={best_fine_scale:.2f}"
            )

            return {
                'success': True,
                'match': {
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
                }
            }

        except Exception as e:
            logger.exception("Failed template match for single follicle")
            return {'success': False, 'error': str(e), 'match': None}

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
