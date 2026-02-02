#!/usr/bin/env python3
"""
Test script to compare PyTorch vs TensorRT keypoint prediction backends.

Run this directly:
    cd electron/python
    python test_keypoint_backends.py

This will load both backends and compare their predictions on sample crops.
"""

import base64
import io
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Import YOLO
try:
    from ultralytics import YOLO
    print("Ultralytics imported successfully")
except ImportError:
    print("ERROR: Ultralytics not installed")
    exit(1)


def load_sample_crop(blob_service_url: str = "http://127.0.0.1:5001"):
    """Try to get a sample crop from the blob service."""
    import requests

    # First, we need to know if there's an active session with an image
    # For testing, we'll just create a simple test image
    print("Creating test image (100x100 gray square)...")

    # Create a simple test image
    img = Image.new('RGB', (100, 100), color=(128, 128, 128))
    # Add some variation to simulate a follicle
    pixels = img.load()
    for i in range(40, 60):
        for j in range(40, 60):
            pixels[i, j] = (80, 80, 80)  # Darker center

    # Convert to bytes
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()


def test_model(model_path: str, image_bytes: bytes, label: str):
    """Test a model and return prediction results."""
    print(f"\n{'='*60}")
    print(f"Testing: {label}")
    print(f"Model: {model_path}")
    print(f"{'='*60}")

    if not Path(model_path).exists():
        print(f"ERROR: Model file not found: {model_path}")
        return None

    try:
        # Load model
        print("Loading model...")
        load_start = time.time()
        model = YOLO(model_path)
        load_time = time.time() - load_start
        print(f"Model loaded in {load_time:.2f}s")

        # Decode image
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        img_array = np.array(image)
        img_width, img_height = image.size
        print(f"Image size: {img_width}x{img_height}")

        # Run inference multiple times
        print("Running inference (5 iterations)...")
        times = []
        results_list = []

        for i in range(5):
            start = time.time()
            results = model.predict(img_array, verbose=False)
            elapsed = time.time() - start
            times.append(elapsed)
            results_list.append(results)

        avg_time = np.mean(times) * 1000
        print(f"Average inference time: {avg_time:.1f}ms")

        # Analyze results
        result = results_list[-1][0] if results_list[-1] else None

        if result is None:
            print("RESULT: No detections")
            return {"success": False, "error": "No detections"}

        print(f"\nResult analysis:")
        print(f"  - Boxes: {len(result.boxes) if result.boxes is not None else 0}")
        print(f"  - Keypoints: {result.keypoints is not None}")

        if result.keypoints is not None:
            print(f"  - Keypoints shape: {result.keypoints.xy.shape if hasattr(result.keypoints, 'xy') else 'N/A'}")

            if len(result.keypoints.xy) > 0:
                keypoints = result.keypoints.xy[0].cpu().numpy()
                confidences = result.keypoints.conf[0].cpu().numpy() if result.keypoints.conf is not None else [1.0, 1.0]

                print(f"  - Keypoint 0 (origin): ({keypoints[0][0]:.1f}, {keypoints[0][1]:.1f}) conf={confidences[0]:.3f}")
                if len(keypoints) > 1:
                    print(f"  - Keypoint 1 (direction): ({keypoints[1][0]:.1f}, {keypoints[1][1]:.1f}) conf={confidences[1]:.3f}")

                # Normalize
                origin_x = float(keypoints[0][0]) / img_width
                origin_y = float(keypoints[0][1]) / img_height
                direction_x = float(keypoints[1][0]) / img_width if len(keypoints) > 1 else origin_x
                direction_y = float(keypoints[1][1]) / img_height if len(keypoints) > 1 else origin_y
                avg_conf = float(np.mean(confidences[:2]))

                print(f"\n  Normalized prediction:")
                print(f"    origin: ({origin_x:.3f}, {origin_y:.3f})")
                print(f"    direction: ({direction_x:.3f}, {direction_y:.3f})")
                print(f"    confidence: {avg_conf:.3f}")

                return {
                    "success": True,
                    "origin": (origin_x, origin_y),
                    "direction": (direction_x, direction_y),
                    "confidence": avg_conf,
                    "inference_time_ms": avg_time
                }
            else:
                print("  - No keypoints detected in result")
                return {"success": False, "error": "No keypoints in result"}
        else:
            print("  - Keypoints is None")
            return {"success": False, "error": "Keypoints is None"}

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def main():
    print("="*60)
    print("Keypoint Backend Comparison Test")
    print("="*60)

    # Find models
    models_dir = Path(__file__).parent / 'models' / 'keypoint'

    if not models_dir.exists():
        print(f"ERROR: Models directory not found: {models_dir}")
        return

    # List all model directories
    model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]

    if not model_dirs:
        print("ERROR: No keypoint models found")
        return

    # Use the first model (most recent)
    model_dir = sorted(model_dirs, reverse=True)[0]
    weights_dir = model_dir / 'weights'

    pt_model = weights_dir / 'best.pt'
    engine_model = weights_dir / 'best.engine'

    print(f"\nModel directory: {model_dir.name}")
    print(f"PyTorch model exists: {pt_model.exists()}")
    print(f"TensorRT engine exists: {engine_model.exists()}")

    # Create test image
    test_image = load_sample_crop()
    print(f"Test image size: {len(test_image)} bytes")

    # Test PyTorch
    pytorch_result = None
    if pt_model.exists():
        pytorch_result = test_model(str(pt_model), test_image, "PyTorch Backend")
    else:
        print("\nSkipping PyTorch test - model not found")

    # Test TensorRT
    tensorrt_result = None
    if engine_model.exists():
        tensorrt_result = test_model(str(engine_model), test_image, "TensorRT Backend")
    else:
        print("\nSkipping TensorRT test - engine not found")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if pytorch_result:
        print(f"\nPyTorch:")
        if pytorch_result.get("success"):
            print(f"  - Success: Yes")
            print(f"  - Confidence: {pytorch_result['confidence']:.3f}")
            print(f"  - Time: {pytorch_result['inference_time_ms']:.1f}ms")
        else:
            print(f"  - Success: No")
            print(f"  - Error: {pytorch_result.get('error')}")

    if tensorrt_result:
        print(f"\nTensorRT:")
        if tensorrt_result.get("success"):
            print(f"  - Success: Yes")
            print(f"  - Confidence: {tensorrt_result['confidence']:.3f}")
            print(f"  - Time: {tensorrt_result['inference_time_ms']:.1f}ms")
        else:
            print(f"  - Success: No")
            print(f"  - Error: {tensorrt_result.get('error')}")

    # Comparison
    if pytorch_result and tensorrt_result:
        if pytorch_result.get("success") and tensorrt_result.get("success"):
            print(f"\nComparison:")
            print(f"  - Speed improvement: {pytorch_result['inference_time_ms'] / tensorrt_result['inference_time_ms']:.1f}x")

            # Check if predictions match
            origin_diff = np.sqrt(
                (pytorch_result['origin'][0] - tensorrt_result['origin'][0])**2 +
                (pytorch_result['origin'][1] - tensorrt_result['origin'][1])**2
            )
            print(f"  - Origin difference: {origin_diff:.4f}")
        elif pytorch_result.get("success") and not tensorrt_result.get("success"):
            print(f"\n*** TensorRT FAILED while PyTorch succeeded ***")
            print(f"    This suggests the TensorRT export may be broken.")
            print(f"    Consider re-exporting the model to TensorRT.")


if __name__ == "__main__":
    main()
