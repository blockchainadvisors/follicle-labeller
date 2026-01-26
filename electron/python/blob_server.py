#!/usr/bin/env python3
"""
BLOB Detection Backend Server

This Flask server provides an API for the Follicle Labeller application to use
OpenCV-based BLOB detection for automated follicle detection. It supports:
- Setting an image for a session
- Adding user annotations for size learning
- Auto-detecting follicles using SimpleBlobDetector + contour fallback

The server learns follicle size from user annotations (minimum 3 required)
and uses that information to calibrate detection parameters.

Usage:
    python blob_server.py [--port PORT] [--host HOST]
"""

import argparse
import base64
import io
import logging
import os
import sys
import uuid
from typing import Dict, List, Optional, Tuple

import cv2
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
sessions: Dict[str, dict] = {}

# Minimum annotations required for auto-detection
MIN_ANNOTATIONS_FOR_DETECTION = 3


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


def detect_blobs(image: np.ndarray,
                 annotations: List[dict]) -> List[dict]:
    """
    Detect follicles using OpenCV SimpleBlobDetector with contour fallback.

    This implements the proven algorithm from hair_follicle_detection:
    1. Apply CLAHE for contrast enhancement
    2. Run SimpleBlobDetector with multi-threshold scanning
    3. Fall back to adaptive thresholding + contours if few detections
    4. Filter by size learned from annotations

    Args:
        image: BGR image as numpy array
        annotations: List of user annotations for size learning

    Returns:
        List of detected follicle dicts with x, y, width, height, confidence
    """
    # Estimate follicle size from annotations
    follicle_size = estimate_follicle_size(annotations)
    min_area = max(10, (follicle_size // 3) ** 2)
    max_area = (follicle_size * 3) ** 2

    logger.info(f"Follicle size estimate: {follicle_size}px, area range: {min_area}-{max_area}")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE for better contrast (clipLimit=3.0 as per working implementation)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

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
    params.minThreshold = 10
    params.maxThreshold = 220
    params.thresholdStep = 10

    # Area filter
    params.filterByArea = True
    params.minArea = min_area
    params.maxArea = max_area

    # Circularity filter - lenient for hair follicles
    params.filterByCircularity = True
    params.minCircularity = 0.2

    # Disable other filters
    params.filterByConvexity = False
    params.filterByInertia = False

    # Color filter - detect dark blobs
    params.filterByColor = True
    params.blobColor = 0  # 0 = dark blobs, 255 = light blobs

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray)

    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        r = max(int(kp.size / 2), follicle_size // 2)
        box = (x - r, y - r, x + r, y + r)

        # Check bounds
        if box[0] < 0 or box[1] < 0 or box[2] > image.shape[1] or box[3] > image.shape[0]:
            continue

        if not overlaps_existing(box, existing_boxes + detected_boxes):
            detected_boxes.append(box)
            detected_results.append({
                'x': box[0],
                'y': box[1],
                'width': box[2] - box[0],
                'height': box[3] - box[1],
                'confidence': 0.8,  # SimpleBlobDetector confidence
                'method': 'blob'
            })

    logger.info(f"Blob detection found {len(detected_results)} follicles")

    # Method 2: Contour-based fallback if blob detection found few results
    if len(detected_results) < 50:
        logger.info("Trying contour-based detection...")

        # Use Gaussian blur + Otsu thresholding (better than adaptive for this use case)
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
# Flask Routes
# ============================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'server': 'blob-detector',
        'version': '1.0.0',
        'min_annotations': MIN_ANNOTATIONS_FOR_DETECTION
    })


@app.route('/set-image', methods=['POST'])
def set_image():
    """
    Set an image for a new session.

    Request body:
        - image: Base64-encoded image data

    Returns:
        - sessionId: Unique session identifier
        - width: Image width
        - height: Image height
    """
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'Missing image data'}), 400

    # Decode image
    image = decode_image(data['image'])
    if image is None:
        return jsonify({'error': 'Failed to decode image'}), 400

    # Create session
    session_id = str(uuid.uuid4())

    sessions[session_id] = {
        'image': image,
        'width': image.shape[1],
        'height': image.shape[0],
        'annotations': []  # User-drawn annotations for learning
    }

    logger.info(f"Created session {session_id} for image {image.shape[1]}x{image.shape[0]}")

    return jsonify({
        'sessionId': session_id,
        'width': image.shape[1],
        'height': image.shape[0]
    })


@app.route('/add-annotation', methods=['POST'])
def add_annotation():
    """
    Add a user annotation for size learning.

    Request body:
        - sessionId: Session identifier
        - x: Top-left X coordinate
        - y: Top-left Y coordinate
        - width: Annotation width
        - height: Annotation height

    Returns:
        - annotationCount: Total annotations in session
        - canDetect: Whether minimum annotations reached
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Missing request data'}), 400

    session_id = data.get('sessionId')
    if not session_id or session_id not in sessions:
        return jsonify({'error': 'Invalid session'}), 400

    # Validate annotation data
    x = data.get('x')
    y = data.get('y')
    width = data.get('width')
    height = data.get('height')

    if None in [x, y, width, height]:
        return jsonify({'error': 'Missing annotation coordinates'}), 400

    # Add annotation
    session = sessions[session_id]
    session['annotations'].append({
        'x': x,
        'y': y,
        'width': width,
        'height': height
    })

    annotation_count = len(session['annotations'])
    can_detect = annotation_count >= MIN_ANNOTATIONS_FOR_DETECTION

    logger.info(f"Session {session_id}: {annotation_count} annotations, can_detect={can_detect}")

    return jsonify({
        'annotationCount': annotation_count,
        'canDetect': can_detect,
        'minRequired': MIN_ANNOTATIONS_FOR_DETECTION
    })


@app.route('/sync-annotations', methods=['POST'])
def sync_annotations():
    """
    Sync all annotations from frontend (replaces session annotations).

    Request body:
        - sessionId: Session identifier
        - annotations: Array of {x, y, width, height} objects

    Returns:
        - annotationCount: Total annotations in session
        - canDetect: Whether minimum annotations reached
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Missing request data'}), 400

    session_id = data.get('sessionId')
    if not session_id or session_id not in sessions:
        return jsonify({'error': 'Invalid session'}), 400

    annotations = data.get('annotations', [])

    # Replace session annotations
    session = sessions[session_id]
    session['annotations'] = []

    for ann in annotations:
        x = ann.get('x')
        y = ann.get('y')
        width = ann.get('width')
        height = ann.get('height')

        if None not in [x, y, width, height]:
            session['annotations'].append({
                'x': x,
                'y': y,
                'width': width,
                'height': height
            })

    annotation_count = len(session['annotations'])
    can_detect = annotation_count >= MIN_ANNOTATIONS_FOR_DETECTION

    logger.info(f"Session {session_id}: synced {annotation_count} annotations, can_detect={can_detect}")

    return jsonify({
        'annotationCount': annotation_count,
        'canDetect': can_detect,
        'minRequired': MIN_ANNOTATIONS_FOR_DETECTION
    })


@app.route('/get-annotation-count', methods=['POST'])
def get_annotation_count():
    """
    Get the current annotation count for a session.

    Request body:
        - sessionId: Session identifier

    Returns:
        - annotationCount: Total annotations in session
        - canDetect: Whether minimum annotations reached
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Missing request data'}), 400

    session_id = data.get('sessionId')
    if not session_id or session_id not in sessions:
        return jsonify({'error': 'Invalid session'}), 400

    session = sessions[session_id]
    annotation_count = len(session['annotations'])

    return jsonify({
        'annotationCount': annotation_count,
        'canDetect': annotation_count >= MIN_ANNOTATIONS_FOR_DETECTION,
        'minRequired': MIN_ANNOTATIONS_FOR_DETECTION
    })


@app.route('/blob-detect', methods=['POST'])
def blob_detect():
    """
    Run BLOB detection on the session image.

    Requires at least MIN_ANNOTATIONS_FOR_DETECTION annotations to have been
    added via /add-annotation for size learning.

    Request body:
        - sessionId: Session identifier

    Returns:
        - detections: Array of {x, y, width, height, confidence, method}
        - count: Number of detections
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Missing request data'}), 400

    session_id = data.get('sessionId')
    if not session_id or session_id not in sessions:
        return jsonify({'error': 'Invalid session'}), 400

    session = sessions[session_id]
    annotations = session['annotations']

    # Check minimum annotations
    if len(annotations) < MIN_ANNOTATIONS_FOR_DETECTION:
        return jsonify({
            'error': f'Minimum {MIN_ANNOTATIONS_FOR_DETECTION} annotations required',
            'annotationCount': len(annotations),
            'minRequired': MIN_ANNOTATIONS_FOR_DETECTION
        }), 400

    try:
        # Run detection
        detections = detect_blobs(session['image'], annotations)

        return jsonify({
            'detections': detections,
            'count': len(detections),
            'learnedSize': estimate_follicle_size(annotations)
        })

    except Exception as e:
        logger.error(f"Detection failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/clear-session', methods=['POST'])
def clear_session():
    """
    Clear a session and free memory.

    Request body:
        - sessionId: Session identifier

    Returns:
        - success: True if session was cleared
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Missing request data'}), 400

    session_id = data.get('sessionId')

    if session_id and session_id in sessions:
        del sessions[session_id]
        logger.info(f"Cleared session {session_id}")

    return jsonify({'success': True})


@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Shutdown the server gracefully."""
    logger.info("Shutting down server...")

    # Clear all sessions
    sessions.clear()

    # Shutdown Flask
    func = request.environ.get('werkzeug.server.shutdown')
    if func:
        func()
    else:
        # Fallback for newer versions of Werkzeug
        import signal
        os.kill(os.getpid(), signal.SIGTERM)

    return jsonify({'status': 'shutting down'})


# ============================================
# Main Entry Point
# ============================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BLOB Detection Server')
    parser.add_argument('--port', type=int, default=5555, help='Port to run on')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    logger.info(f"Starting BLOB Detection Server on {args.host}:{args.port}")
    logger.info(f"Minimum annotations required: {MIN_ANNOTATIONS_FOR_DETECTION}")

    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )
