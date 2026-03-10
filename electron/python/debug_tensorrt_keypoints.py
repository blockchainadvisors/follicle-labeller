#!/usr/bin/env python3
"""
Debug TensorRT keypoint prediction to see what's actually returned.
"""

import base64
import io
import json
from pathlib import Path
from PIL import Image
import numpy as np

# Test with the YOLO model directly
from ultralytics import YOLO


def create_test_image(size=100):
    """Create a test image with an elliptical pattern."""
    img = Image.new('RGB', (size, size), color=(180, 160, 140))
    pixels = img.load()

    cx, cy = size // 2, size // 2
    rx, ry = size // 4, size // 3

    for x in range(size):
        for y in range(size):
            dx = (x - cx) / rx
            dy = (y - cy) / ry
            dist = dx*dx + dy*dy
            if dist < 1.0:
                darkness = int(80 * (1 - dist))
                base = 60
                pixels[x, y] = (base + darkness//2, base, base)

    return np.array(img)


def test_model(model_path: str, model_name: str):
    """Test a model and print detailed results."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"Path: {model_path}")
    print("="*60)

    # Determine if TensorRT
    is_tensorrt = model_path.endswith('.engine')

    # Load model with task='pose' for TensorRT
    if is_tensorrt:
        print("Loading as TensorRT with task='pose'...")
        model = YOLO(model_path, task='pose')
    else:
        print("Loading as PyTorch...")
        model = YOLO(model_path)

    # Print model info
    print(f"Model loaded: {type(model)}")
    print(f"Model task: {model.task}")

    # Create test image
    img = create_test_image(100)
    print(f"\nTest image shape: {img.shape}")

    # Run prediction
    print("\nRunning prediction...")
    results = model.predict(img, verbose=True)

    print(f"\nResults count: {len(results)}")

    if results:
        result = results[0]
        print(f"\nResult type: {type(result)}")
        print(f"Result keys: {result.keys() if hasattr(result, 'keys') else 'N/A'}")

        # Check boxes
        if hasattr(result, 'boxes'):
            print(f"\nBoxes: {result.boxes}")
            if result.boxes is not None and len(result.boxes) > 0:
                print(f"  - Box count: {len(result.boxes)}")
                print(f"  - First box xyxy: {result.boxes.xyxy[0].cpu().numpy() if len(result.boxes) > 0 else 'N/A'}")
                print(f"  - First box conf: {result.boxes.conf[0].cpu().numpy() if len(result.boxes) > 0 else 'N/A'}")

        # Check keypoints
        if hasattr(result, 'keypoints'):
            print(f"\nKeypoints: {result.keypoints}")
            if result.keypoints is not None:
                print(f"  - Keypoints shape xy: {result.keypoints.xy.shape if hasattr(result.keypoints, 'xy') else 'N/A'}")
                print(f"  - Keypoints conf: {result.keypoints.conf if hasattr(result.keypoints, 'conf') else 'N/A'}")

                if result.keypoints.xy is not None and len(result.keypoints.xy) > 0:
                    kpts = result.keypoints.xy[0].cpu().numpy()
                    print(f"  - Keypoints values: {kpts}")
                else:
                    print("  - No keypoints found!")
        else:
            print("\nNo keypoints attribute on result!")

        # Check if names match pose
        if hasattr(result, 'names'):
            print(f"\nClass names: {result.names}")

    # Try with a real crop from disk if available
    crops_dir = Path(__file__).parent / "test_crops"
    if crops_dir.exists():
        crop_files = list(crops_dir.glob("*.png")) + list(crops_dir.glob("*.jpg"))
        if crop_files:
            print(f"\n\nTesting with real crop: {crop_files[0]}")
            real_img = np.array(Image.open(crop_files[0]).convert('RGB'))
            print(f"Real crop shape: {real_img.shape}")

            real_results = model.predict(real_img, verbose=True)
            if real_results:
                real_result = real_results[0]
                if hasattr(real_result, 'keypoints') and real_result.keypoints is not None:
                    if real_result.keypoints.xy is not None and len(real_result.keypoints.xy) > 0:
                        kpts = real_result.keypoints.xy[0].cpu().numpy()
                        print(f"Real crop keypoints: {kpts}")
                    else:
                        print("Real crop: No keypoints found!")
                else:
                    print("Real crop: No keypoints attribute!")


def main():
    print("TensorRT Keypoint Debug Script")
    print("="*60)

    models_dir = Path(__file__).parent / "models" / "keypoint"

    # Find the keypoint model
    for model_dir in models_dir.iterdir():
        if not model_dir.is_dir():
            continue

        pt_model = model_dir / "weights" / "best.pt"
        engine_model = model_dir / "weights" / "best.engine"

        if pt_model.exists():
            test_model(str(pt_model), "PyTorch")

        if engine_model.exists():
            test_model(str(engine_model), "TensorRT")

        break  # Just test first model


if __name__ == "__main__":
    main()
