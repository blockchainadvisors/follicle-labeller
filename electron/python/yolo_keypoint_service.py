#!/usr/bin/env python3
"""
YOLO Keypoint Training Service

Provides training and inference capabilities for YOLO11-pose models
to detect follicle origin points and growth directions.

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
from PIL import Image

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
class TrainingConfig:
    """Configuration for YOLO keypoint training."""
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


@dataclass
class TrainingProgress:
    """Progress update during training."""
    status: str  # 'preparing', 'training', 'completed', 'failed', 'stopped'
    epoch: int = 0
    total_epochs: int = 0
    loss: float = 0.0
    box_loss: float = 0.0
    pose_loss: float = 0.0
    kobj_loss: float = 0.0
    metrics: Dict[str, float] = None
    eta: str = ''
    message: str = ''

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class KeypointPrediction:
    """Keypoint prediction result."""
    origin_x: float
    origin_y: float
    direction_end_x: float
    direction_end_y: float
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'origin': {'x': self.origin_x, 'y': self.origin_y},
            'directionEndpoint': {'x': self.direction_end_x, 'y': self.direction_end_y},
            'confidence': self.confidence
        }


@dataclass
class ModelInfo:
    """Information about a trained model."""
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
class DatasetValidation:
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


class YOLOKeypointService:
    """
    Service for YOLO keypoint model training and inference.

    Manages:
    - Dataset validation
    - Training jobs with progress callbacks
    - Model loading/unloading
    - Inference on cropped images
    - ONNX export
    - Model storage
    """

    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialize the YOLO keypoint service.

        Args:
            models_dir: Directory to store trained models. If None, uses
                       a 'models/keypoint' subdirectory next to this script.
        """
        if models_dir:
            self.models_dir = Path(models_dir)
        else:
            # Default to models/keypoint next to script
            self.models_dir = Path(__file__).parent / 'models' / 'keypoint'

        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Currently loaded model for inference
        self._loaded_model: Optional['YOLO'] = None
        self._loaded_model_path: Optional[str] = None
        self._loaded_model_backend: str = 'pytorch'  # 'pytorch' or 'tensorrt'

        # Active training jobs
        self._training_jobs: Dict[str, dict] = {}

        logger.info(f"YOLOKeypointService initialized. Models dir: {self.models_dir}")

    def clear_gpu_memory(self) -> Dict[str, Any]:
        """
        Clear GPU memory by running garbage collection and emptying CUDA cache.

        This should be called after batch prediction tasks complete to free
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

                # Run Python garbage collection first
                gc.collect()

                # Empty CUDA cache
                torch.cuda.empty_cache()

                # Synchronize to ensure cleanup is complete
                torch.cuda.synchronize()

                # Get memory stats after cleanup
                memory_after = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                memory_reserved_after = torch.cuda.memory_reserved() / (1024 * 1024)  # MB
                result["memory_after"] = {
                    "allocated_mb": round(memory_after, 2),
                    "reserved_mb": round(memory_reserved_after, 2)
                }

                result["memory_freed_mb"] = round(memory_reserved_before - memory_reserved_after, 2)

                logger.info(f"GPU memory cleanup: freed {result['memory_freed_mb']:.1f}MB "
                           f"(reserved: {memory_reserved_before:.1f}MB -> {memory_reserved_after:.1f}MB)")
            else:
                result["success"] = True
                result["message"] = "CUDA not available, no GPU memory to clear"

        except Exception as e:
            logger.error(f"GPU memory cleanup failed: {e}")
            result["success"] = False
            result["error"] = str(e)

        return result

    def validate_dataset(self, dataset_path: str) -> DatasetValidation:
        """
        Validate a YOLO keypoint dataset structure.

        Expected structure:
        - data.yaml (with path, train, val, names, kpt_shape)
        - images/train/*.jpg
        - images/val/*.jpg
        - labels/train/*.txt
        - labels/val/*.txt

        Args:
            dataset_path: Path to dataset root directory

        Returns:
            DatasetValidation with validation results
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
            return DatasetValidation(
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

                required_keys = ['train', 'val', 'names', 'kpt_shape']
                for key in required_keys:
                    if key not in config:
                        errors.append(f"data.yaml missing required key: {key}")

                # Validate kpt_shape
                if 'kpt_shape' in config:
                    kpt_shape = config['kpt_shape']
                    if not isinstance(kpt_shape, list) or len(kpt_shape) != 2:
                        errors.append("kpt_shape must be [num_keypoints, values_per_keypoint]")
                    elif kpt_shape[0] != 2 or kpt_shape[1] != 3:
                        warnings.append(f"Expected kpt_shape [2, 3] for origin+direction, got {kpt_shape}")

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

        return DatasetValidation(
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
        config: TrainingConfig,
        progress_callback: Callable[[TrainingProgress], None],
        model_name: Optional[str] = None
    ) -> Tuple[str, Optional[ModelInfo]]:
        """
        Train a YOLO keypoint model.

        Args:
            dataset_path: Path to dataset root (must have data.yaml)
            config: Training configuration
            progress_callback: Called with progress updates
            model_name: Optional custom name for the model

        Returns:
            Tuple of (job_id, ModelInfo if successful)
        """
        if not YOLO_AVAILABLE:
            progress_callback(TrainingProgress(
                status='failed',
                message='Ultralytics not installed. Please install with: pip install ultralytics'
            ))
            return '', None

        # Generate job ID
        job_id = str(uuid.uuid4())[:8]

        # Generate model name if not provided
        if not model_name:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f"keypoint_{config.model_size}_{timestamp}"

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
                progress_callback(TrainingProgress(
                    status='failed',
                    message=str(e)
                ))

        thread = threading.Thread(target=training_thread, daemon=True)
        self._training_jobs[job_id]['thread'] = thread
        thread.start()

        return job_id, None  # ModelInfo returned via progress callback when complete

    def _run_training(
        self,
        job_id: str,
        dataset_path: str,
        config: TrainingConfig,
        model_name: str,
        model_dir: Path,
        progress_callback: Callable[[TrainingProgress], None]
    ):
        """
        Internal training loop (runs in a separate thread).
        """
        try:
            progress_callback(TrainingProgress(
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

            # Load pretrained YOLO pose model
            model_variant = f'yolo11{config.model_size}-pose.pt'
            logger.info(f"Loading pretrained model: {model_variant}")
            model = YOLO(model_variant)

            progress_callback(TrainingProgress(
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
                    pose_loss = 0.0
                    kobj_loss = 0.0
                    if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
                        items = trainer.loss_items
                        if len(items) >= 4:
                            box_loss = float(items[0])
                            pose_loss = float(items[2])  # pose loss is typically index 2
                            kobj_loss = float(items[3])  # keypoint objectness

                    # Get metrics
                    metrics = {}
                    if hasattr(trainer, 'metrics') and trainer.metrics:
                        for key, value in trainer.metrics.items():
                            if isinstance(value, (int, float)):
                                metrics[key] = float(value)

                    self.callback(TrainingProgress(
                        status='training',
                        epoch=epoch,
                        total_epochs=total,
                        loss=loss,
                        box_loss=box_loss,
                        pose_loss=pose_loss,
                        kobj_loss=kobj_loss,
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

            # Start training
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
                progress_callback(TrainingProgress(
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
            model_info = ModelInfo(
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

            progress_callback(TrainingProgress(
                status='completed',
                epoch=config.epochs,
                total_epochs=config.epochs,
                metrics=final_metrics,
                message=f'Training completed. Model saved to {model_dir}'
            ))

            logger.info(f"Training completed for job {job_id}. Model: {model_name}")

        except Exception as e:
            logger.exception(f"Training error for job {job_id}")
            progress_callback(TrainingProgress(
                status='failed',
                message=str(e)
            ))
        finally:
            # Cleanup job entry
            if job_id in self._training_jobs:
                del self._training_jobs[job_id]

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

    def load_model(self, model_path: str) -> bool:
        """
        Load a trained model for inference.

        Args:
            model_path: Path to model .pt or .engine file

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

            # Detect backend from file extension
            model_ext = Path(model_path).suffix.lower()
            if model_ext == '.engine':
                backend = 'tensorrt'
            else:
                backend = 'pytorch'

            # Load new model (Ultralytics handles both .pt and .engine)
            # For TensorRT engines, we must specify task='pose' explicitly
            # so that keypoint outputs are properly parsed
            if backend == 'tensorrt':
                self._loaded_model = YOLO(model_path, task='pose')
            else:
                self._loaded_model = YOLO(model_path)
            self._loaded_model_path = model_path
            self._loaded_model_backend = backend

            logger.info(f"Loaded model: {model_path} (backend: {backend})")
            return True

        except Exception as e:
            logger.exception(f"Failed to load model: {model_path}")
            return False

    def predict(self, image_data: bytes) -> Optional[KeypointPrediction]:
        """
        Run keypoint prediction on a cropped follicle image.

        Args:
            image_data: Image bytes (JPEG or PNG)

        Returns:
            KeypointPrediction or None if prediction failed
        """
        if self._loaded_model is None:
            logger.error("No model loaded for inference")
            return None

        try:
            t0 = time.time()

            # Decode image
            image = Image.open(io.BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')

            img_array = np.array(image)
            img_width, img_height = image.size

            t1 = time.time()

            # Run inference
            results = self._loaded_model.predict(img_array, verbose=False)

            t2 = time.time()

            # Log timing every 100th prediction for debugging
            if hasattr(self, '_predict_count'):
                self._predict_count += 1
            else:
                self._predict_count = 1

            if self._predict_count <= 5 or self._predict_count % 100 == 0:
                decode_ms = (t1 - t0) * 1000
                infer_ms = (t2 - t1) * 1000
                logger.info(f"Predict timing #{self._predict_count} [{self._loaded_model_backend}]: "
                           f"decode={decode_ms:.1f}ms, inference={infer_ms:.1f}ms")

            if not results or len(results) == 0:
                return None

            result = results[0]

            # Extract keypoints
            if result.keypoints is None or len(result.keypoints.xy) == 0:
                return None

            # Use box confidence for filtering (measures detection quality)
            # Keypoint confidence measures visibility, not detection quality
            if result.boxes is None or len(result.boxes) == 0:
                return None
            box_confidence = float(result.boxes.conf[0])

            keypoints = result.keypoints.xy[0].cpu().numpy()  # First detection

            if len(keypoints) < 2:
                return None

            # Normalize coordinates to 0-1
            origin_x = float(keypoints[0][0]) / img_width
            origin_y = float(keypoints[0][1]) / img_height
            direction_x = float(keypoints[1][0]) / img_width
            direction_y = float(keypoints[1][1]) / img_height

            # Use box confidence instead of keypoint confidence
            confidence = box_confidence

            return KeypointPrediction(
                origin_x=origin_x,
                origin_y=origin_y,
                direction_end_x=direction_x,
                direction_end_y=direction_y,
                confidence=confidence
            )

        except Exception as e:
            logger.exception("Prediction failed")
            return None

    def predict_batch(self, images_data: List[bytes]) -> List[Optional[KeypointPrediction]]:
        """
        Run keypoint prediction on multiple cropped follicle images in batch.

        Batch inference is significantly faster on GPU as it processes
        multiple images in parallel.

        Args:
            images_data: List of image bytes (JPEG or PNG)

        Returns:
            List of KeypointPrediction or None for each image
        """
        if self._loaded_model is None:
            logger.error("No model loaded for inference")
            return [None] * len(images_data)

        if len(images_data) == 0:
            return []

        try:
            t0 = time.time()

            # Decode all images and track their sizes
            images = []
            sizes = []  # (width, height) for each image

            for img_data in images_data:
                try:
                    image = Image.open(io.BytesIO(img_data))
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    images.append(np.array(image))
                    sizes.append(image.size)  # (width, height)
                except Exception as e:
                    logger.warning(f"Failed to decode image in batch: {e}")
                    images.append(None)
                    sizes.append((0, 0))

            t1 = time.time()

            # Filter out failed images for batch inference
            valid_indices = [i for i, img in enumerate(images) if img is not None]
            valid_images = [images[i] for i in valid_indices]

            if len(valid_images) == 0:
                return [None] * len(images_data)

            # Run batch inference
            # YOLO model.predict() accepts a list of images for batch processing
            results = self._loaded_model.predict(valid_images, verbose=False)

            t2 = time.time()

            # Log timing every Nth batch for debugging
            if not hasattr(self, '_batch_count'):
                self._batch_count = 0
            self._batch_count += 1
            if self._batch_count <= 5 or self._batch_count % 50 == 0:
                decode_ms = (t1 - t0) * 1000
                infer_ms = (t2 - t1) * 1000
                logger.info(f"Batch predict timing #{self._batch_count} [{self._loaded_model_backend}] "
                           f"(n={len(valid_images)}): decode={decode_ms:.1f}ms, inference={infer_ms:.1f}ms")

            # Process results
            predictions = [None] * len(images_data)

            for result_idx, orig_idx in enumerate(valid_indices):
                if result_idx >= len(results):
                    continue

                result = results[result_idx]
                img_width, img_height = sizes[orig_idx]

                # Extract keypoints
                if result.keypoints is None or len(result.keypoints.xy) == 0:
                    continue

                # Use box confidence for filtering (measures detection quality)
                if result.boxes is None or len(result.boxes) == 0:
                    continue
                box_confidence = float(result.boxes.conf[0])

                keypoints = result.keypoints.xy[0].cpu().numpy()

                if len(keypoints) < 2:
                    continue

                # Normalize coordinates to 0-1
                origin_x = float(keypoints[0][0]) / img_width
                origin_y = float(keypoints[0][1]) / img_height
                direction_x = float(keypoints[1][0]) / img_width
                direction_y = float(keypoints[1][1]) / img_height

                # Use box confidence instead of keypoint confidence
                confidence = box_confidence

                predictions[orig_idx] = KeypointPrediction(
                    origin_x=origin_x,
                    origin_y=origin_y,
                    direction_end_x=direction_x,
                    direction_end_y=direction_y,
                    confidence=confidence
                )

            return predictions

        except Exception as e:
            logger.exception("Batch prediction failed")
            return [None] * len(images_data)

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

    def list_models(self) -> List[ModelInfo]:
        """
        List all trained models.

        Returns:
            List of ModelInfo objects
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

                    models.append(ModelInfo(
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

        TensorRT provides GPU-optimized inference for faster keypoint prediction
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
_service_instance: Optional[YOLOKeypointService] = None


def get_yolo_keypoint_service(models_dir: Optional[str] = None) -> YOLOKeypointService:
    """
    Get the singleton YOLO keypoint service instance.

    Args:
        models_dir: Optional custom models directory (only used on first call)

    Returns:
        YOLOKeypointService instance
    """
    global _service_instance
    if _service_instance is None:
        _service_instance = YOLOKeypointService(models_dir)
    return _service_instance
