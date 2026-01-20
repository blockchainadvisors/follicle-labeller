#!/usr/bin/env python3
"""
SAM 2 (Segment Anything Model 2) Backend Server

This Flask server provides an API for the Follicle Labeller application to use
SAM 2 for AI-powered segmentation. It supports:
- Setting an image for a session
- Segmenting from point prompts (click-based)
- Auto-detection using grid-based prompting

Usage:
    python sam_server.py [--port PORT] [--host HOST] [--model MODEL_SIZE]

Model sizes: tiny, small, base, large (default: small)
"""

import argparse
import base64
import io
import logging
import os
import sys
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from flask import Flask, jsonify, request
from flask_cors import CORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global state
sam_model = None
sam_predictor = None
sessions: Dict[str, dict] = {}

# Model paths (will be set based on model size)
MODEL_CONFIGS = {
    'tiny': {
        'checkpoint': 'sam2_hiera_tiny.pt',
        'config': 'sam2_hiera_t.yaml'
    },
    'small': {
        'checkpoint': 'sam2_hiera_small.pt',
        'config': 'sam2_hiera_s.yaml'
    },
    'base': {
        'checkpoint': 'sam2_hiera_base_plus.pt',
        'config': 'sam2_hiera_b+.yaml'
    },
    'large': {
        'checkpoint': 'sam2_hiera_large.pt',
        'config': 'sam2_hiera_l.yaml'
    }
}


def load_sam_model(model_size: str = 'small', device: str = 'auto') -> bool:
    """
    Load the SAM 2 model.

    Args:
        model_size: One of 'tiny', 'small', 'base', 'large'
        device: 'cuda', 'cpu', or 'auto' (auto-detect)

    Returns:
        True if model loaded successfully
    """
    global sam_model, sam_predictor

    try:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        # Auto-detect device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        logger.info(f"Loading SAM 2 model ({model_size}) on {device}...")

        # Get model config
        if model_size not in MODEL_CONFIGS:
            logger.error(f"Unknown model size: {model_size}")
            return False

        config = MODEL_CONFIGS[model_size]

        # Check for model checkpoint
        checkpoint_path = os.path.join(
            os.path.dirname(__file__),
            'checkpoints',
            config['checkpoint']
        )

        if not os.path.exists(checkpoint_path):
            # Try default SAM 2 location
            checkpoint_path = os.path.expanduser(
                f"~/.cache/sam2/checkpoints/{config['checkpoint']}"
            )

        if not os.path.exists(checkpoint_path):
            logger.error(f"Model checkpoint not found: {checkpoint_path}")
            logger.info("Please download SAM 2 checkpoints from:")
            logger.info("https://github.com/facebookresearch/segment-anything-2#model-checkpoints")
            return False

        # Build model
        sam_model = build_sam2(config['config'], checkpoint_path, device=device)
        sam_predictor = SAM2ImagePredictor(sam_model)

        logger.info("SAM 2 model loaded successfully")
        return True

    except ImportError as e:
        logger.error(f"Failed to import SAM 2: {e}")
        logger.info("Please install SAM 2:")
        logger.info("pip install git+https://github.com/facebookresearch/segment-anything-2.git")
        return False
    except Exception as e:
        logger.error(f"Failed to load SAM 2 model: {e}")
        return False


def decode_image(image_data: str) -> Optional[np.ndarray]:
    """Decode base64 image data to numpy array."""
    try:
        # Handle data URL format
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        return np.array(image)
    except Exception as e:
        logger.error(f"Failed to decode image: {e}")
        return None


def encode_mask(mask: np.ndarray) -> str:
    """Encode mask as base64 PNG."""
    # Convert boolean mask to uint8
    mask_uint8 = (mask * 255).astype(np.uint8)
    image = Image.fromarray(mask_uint8, mode='L')

    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def mask_to_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Convert mask to bounding box (x, y, width, height)."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        return (0, 0, 0, 0)

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return (int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1))


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'model_loaded': sam_predictor is not None,
        'active_sessions': len(sessions)
    })


@app.route('/set-image', methods=['POST'])
def set_image():
    """
    Set an image for a new session.

    Request body:
        - image: Base64-encoded image data (with or without data URL prefix)

    Returns:
        - sessionId: Unique session identifier
        - width: Image width
        - height: Image height
    """
    if sam_predictor is None:
        return jsonify({'error': 'Model not loaded'}), 503

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'Missing image data'}), 400

    # Decode image
    image = decode_image(data['image'])
    if image is None:
        return jsonify({'error': 'Failed to decode image'}), 400

    # Create session
    session_id = str(uuid.uuid4())

    try:
        # Set image in predictor
        sam_predictor.set_image(image)

        # Store session
        sessions[session_id] = {
            'image': image,
            'width': image.shape[1],
            'height': image.shape[0]
        }

        logger.info(f"Created session {session_id} for image {image.shape[1]}x{image.shape[0]}")

        return jsonify({
            'sessionId': session_id,
            'width': image.shape[1],
            'height': image.shape[0]
        })

    except Exception as e:
        logger.error(f"Failed to set image: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/segment', methods=['POST'])
def segment():
    """
    Segment from point prompts.

    Request body:
        - sessionId: Session identifier from set-image
        - points: Array of {x, y} coordinates
        - labels: Array of labels (1 = foreground, 0 = background)

    Returns:
        - masks: Array of base64-encoded mask images
        - scores: Array of confidence scores
        - bboxes: Array of bounding boxes {x, y, width, height}
    """
    if sam_predictor is None:
        return jsonify({'error': 'Model not loaded'}), 503

    data = request.get_json()
    if not data:
        return jsonify({'error': 'Missing request data'}), 400

    session_id = data.get('sessionId')
    points = data.get('points', [])
    labels = data.get('labels', [])

    if not session_id or session_id not in sessions:
        return jsonify({'error': 'Invalid session'}), 400

    if len(points) == 0:
        return jsonify({'error': 'No points provided'}), 400

    if len(points) != len(labels):
        return jsonify({'error': 'Points and labels must have same length'}), 400

    try:
        # Convert points and labels to numpy arrays
        point_coords = np.array([[p['x'], p['y']] for p in points])
        point_labels = np.array(labels)

        # Restore session image
        session = sessions[session_id]
        sam_predictor.set_image(session['image'])

        # Predict masks
        masks, scores, _ = sam_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )

        # Encode results
        result_masks = []
        result_scores = []
        result_bboxes = []

        for i, (mask, score) in enumerate(zip(masks, scores)):
            result_masks.append(encode_mask(mask))
            result_scores.append(float(score))
            result_bboxes.append(mask_to_bbox(mask))

        return jsonify({
            'masks': result_masks,
            'scores': result_scores,
            'bboxes': [{'x': b[0], 'y': b[1], 'width': b[2], 'height': b[3]} for b in result_bboxes]
        })

    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/auto-detect', methods=['POST'])
def auto_detect():
    """
    Auto-detect objects using grid-based prompting.

    Request body:
        - sessionId: Session identifier from set-image
        - gridSize: Grid spacing in pixels (default: 32)
        - minArea: Minimum mask area to include (default: 100)
        - maxOverlap: Maximum IoU for NMS (default: 0.5)

    Returns:
        - detections: Array of {x, y, width, height, confidence, maskBase64}
    """
    if sam_predictor is None:
        return jsonify({'error': 'Model not loaded'}), 503

    data = request.get_json()
    if not data:
        return jsonify({'error': 'Missing request data'}), 400

    session_id = data.get('sessionId')
    grid_size = data.get('gridSize', 32)
    min_area = data.get('minArea', 100)
    max_overlap = data.get('maxOverlap', 0.5)

    if not session_id or session_id not in sessions:
        return jsonify({'error': 'Invalid session'}), 400

    try:
        session = sessions[session_id]
        image = session['image']
        height, width = image.shape[:2]

        # Restore session image
        sam_predictor.set_image(image)

        # Generate grid points
        points = []
        for y in range(grid_size // 2, height, grid_size):
            for x in range(grid_size // 2, width, grid_size):
                points.append((x, y))

        logger.info(f"Auto-detecting with {len(points)} grid points...")

        # Collect all detections
        all_detections = []

        for px, py in points:
            try:
                masks, scores, _ = sam_predictor.predict(
                    point_coords=np.array([[px, py]]),
                    point_labels=np.array([1]),
                    multimask_output=True
                )

                # Use best mask
                best_idx = np.argmax(scores)
                mask = masks[best_idx]
                score = float(scores[best_idx])

                # Check minimum area
                area = int(np.sum(mask))
                if area < min_area:
                    continue

                bbox = mask_to_bbox(mask)
                if bbox[2] == 0 or bbox[3] == 0:
                    continue

                all_detections.append({
                    'x': bbox[0],
                    'y': bbox[1],
                    'width': bbox[2],
                    'height': bbox[3],
                    'area': area,
                    'confidence': score,
                    'mask': mask
                })

            except Exception as e:
                logger.warning(f"Failed to process point ({px}, {py}): {e}")
                continue

        logger.info(f"Found {len(all_detections)} raw detections")

        # Apply NMS
        detections = apply_nms(all_detections, max_overlap)

        logger.info(f"After NMS: {len(detections)} detections")

        # Encode masks and return
        result = []
        for det in detections:
            result.append({
                'x': det['x'],
                'y': det['y'],
                'width': det['width'],
                'height': det['height'],
                'area': det['area'],
                'confidence': det['confidence'],
                'maskBase64': encode_mask(det['mask'])
            })

        return jsonify({'detections': result})

    except Exception as e:
        logger.error(f"Auto-detection failed: {e}")
        return jsonify({'error': str(e)}), 500


def calculate_iou(det1: dict, det2: dict) -> float:
    """Calculate IoU between two detections."""
    x1 = max(det1['x'], det2['x'])
    y1 = max(det1['y'], det2['y'])
    x2 = min(det1['x'] + det1['width'], det2['x'] + det2['width'])
    y2 = min(det1['y'] + det1['height'], det2['y'] + det2['height'])

    if x1 >= x2 or y1 >= y2:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = det1['width'] * det1['height']
    area2 = det2['width'] * det2['height']
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def apply_nms(detections: List[dict], max_overlap: float) -> List[dict]:
    """Apply non-maximum suppression to detections."""
    if len(detections) == 0:
        return []

    # Sort by confidence (descending)
    sorted_dets = sorted(detections, key=lambda x: x['confidence'], reverse=True)

    result = []
    suppressed = set()

    for i, det in enumerate(sorted_dets):
        if i in suppressed:
            continue

        result.append(det)

        # Suppress overlapping detections
        for j in range(i + 1, len(sorted_dets)):
            if j in suppressed:
                continue

            iou = calculate_iou(det, sorted_dets[j])
            if iou > max_overlap:
                suppressed.add(j)

    return result


@app.route('/close-session', methods=['POST'])
def close_session():
    """Close a session and free resources."""
    data = request.get_json()
    if not data or 'sessionId' not in data:
        return jsonify({'error': 'Missing sessionId'}), 400

    session_id = data['sessionId']
    if session_id in sessions:
        del sessions[session_id]
        logger.info(f"Closed session {session_id}")

    return jsonify({'success': True})


@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Shutdown the server."""
    logger.info("Shutting down server...")
    func = request.environ.get('werkzeug.server.shutdown')
    if func:
        func()
    return jsonify({'success': True})


def main():
    parser = argparse.ArgumentParser(description='SAM 2 Backend Server')
    parser.add_argument('--port', type=int, default=5555, help='Server port')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Server host')
    parser.add_argument('--model', type=str, default='small',
                       choices=['tiny', 'small', 'base', 'large'],
                       help='Model size')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--no-model', action='store_true',
                       help='Start server without loading model (for testing)')

    args = parser.parse_args()

    # Load model
    if not args.no_model:
        if not load_sam_model(args.model, args.device):
            logger.warning("Starting server without model (API will return 503)")
    else:
        logger.info("Starting server without model (--no-model flag)")

    # Start server
    logger.info(f"Starting SAM 2 server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()
