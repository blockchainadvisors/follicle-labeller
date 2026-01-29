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
    - Model storage
    """

    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialize the YOLO detection service.

        Args:
            models_dir: Directory to store trained models. If None, uses
                       a 'models/detection' subdirectory next to this script.
        """
        if models_dir:
            self.models_dir = Path(models_dir)
        else:
            # Default to models/detection next to script
            self.models_dir = Path(__file__).parent / 'models' / 'detection'

        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Currently loaded model for inference
        self._loaded_model: Optional['YOLO'] = None
        self._loaded_model_path: Optional[str] = None

        # Active training jobs
        self._training_jobs: Dict[str, dict] = {}

        logger.info(f"YOLODetectionService initialized. Models dir: {self.models_dir}")

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
            model_path: Path to model .pt file

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

            # Load new model
            self._loaded_model = YOLO(model_path)
            self._loaded_model_path = model_path

            logger.info(f"Loaded model: {model_path}")
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

            logger.info(f"Successfully loaded default model: {default_model}")
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
            if image.mode != 'RGB':
                image = image.convert('RGB')

            img_array = np.array(image)

            # Run inference
            results = self._loaded_model.predict(
                img_array,
                conf=confidence_threshold,
                verbose=False
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
                    predictions.append(DetectionPrediction(
                        x=float(x1),
                        y=float(y1),
                        width=float(x2 - x1),
                        height=float(y2 - y1),
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
