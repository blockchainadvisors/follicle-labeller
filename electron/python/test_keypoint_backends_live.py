#!/usr/bin/env python3
"""
Test keypoint prediction with both PyTorch and TensorRT backends
using the running blob server.

Run this while the app is running:
    python test_keypoint_backends_live.py

This script will:
1. Create sample crop images
2. Test prediction with PyTorch model
3. Test prediction with TensorRT model
4. Compare results
"""

import base64
import io
import json
import requests
import time
import sys
from pathlib import Path
from PIL import Image
import numpy as np

# Blob server URL
SERVER_URL = "http://127.0.0.1:5555"


def create_test_crops(count: int = 20):
    """Create realistic test crop images that look like follicles."""
    crops = []

    for i in range(count):
        # Create a 80x80 image with a dark blob (simulating follicle)
        size = 80 + (i % 3) * 10  # Vary size slightly
        img = Image.new('RGB', (size, size), color=(180, 160, 140))  # Skin-like background
        pixels = img.load()

        # Add a darker elliptical region (follicle)
        cx, cy = size // 2, size // 2
        rx, ry = size // 4, size // 3  # Ellipse radii

        for x in range(size):
            for y in range(size):
                # Ellipse equation
                dx = (x - cx) / rx
                dy = (y - cy) / ry
                dist = dx*dx + dy*dy

                if dist < 1.0:
                    # Inside ellipse - darker
                    darkness = int(80 * (1 - dist))
                    base = 60 + (i * 5) % 40
                    pixels[x, y] = (base + darkness//2, base, base)
                elif dist < 1.5:
                    # Edge gradient
                    blend = (1.5 - dist) * 2
                    r = int(180 * (1-blend) + 80 * blend)
                    g = int(160 * (1-blend) + 60 * blend)
                    b = int(140 * (1-blend) + 60 * blend)
                    pixels[x, y] = (r, g, b)

        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        crops.append({
            "index": i,
            "size": size,
            "image": f"data:image/png;base64,{img_base64}"
        })

    return crops


def list_keypoint_models():
    """List available keypoint models."""
    try:
        resp = requests.get(f"{SERVER_URL}/yolo-keypoint/models")
        if resp.status_code == 200:
            return resp.json().get("models", [])
        return []
    except Exception as e:
        print(f"Error listing models: {e}")
        return []


def load_model(model_path: str):
    """Load a keypoint model."""
    try:
        resp = requests.post(
            f"{SERVER_URL}/yolo-keypoint/load-model",
            json={"modelPath": model_path},
            timeout=60
        )
        if resp.status_code == 200:
            return resp.json().get("success", False)
        print(f"Load model failed: HTTP {resp.status_code} - {resp.text}")
        return False
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


def predict(image_base64: str):
    """Run keypoint prediction."""
    try:
        resp = requests.post(
            f"{SERVER_URL}/yolo-keypoint/predict",
            json={"imageData": image_base64},
            timeout=30
        )
        if resp.status_code == 200:
            return resp.json()
        return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_loaded_model_info():
    """Get info about currently loaded model."""
    try:
        resp = requests.get(f"{SERVER_URL}/yolo-keypoint/status")
        if resp.status_code == 200:
            return resp.json()
        return None
    except:
        return None


def benchmark_model(model_path: str, model_name: str, crops: list):
    """Benchmark a model with the given crops."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"Model: {model_path}")
    print(f"{'='*60}")

    if not Path(model_path).exists():
        print(f"ERROR: Model file not found!")
        return None

    # Load model
    print("Loading model...")
    load_start = time.time()
    success = load_model(model_path)
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.2f}s: {'Success' if success else 'FAILED'}")

    if not success:
        return None

    # Run predictions
    results = []
    total_time = 0
    success_count = 0
    low_confidence_count = 0
    failed_count = 0

    print(f"\nRunning predictions on {len(crops)} crops...")

    for crop in crops:
        start = time.time()
        result = predict(crop["image"])
        elapsed = time.time() - start
        total_time += elapsed

        if result.get("success") and result.get("prediction"):
            pred = result["prediction"]
            conf = pred.get("confidence", 0)
            results.append({
                "index": crop["index"],
                "success": True,
                "confidence": conf,
                "origin": pred.get("origin"),
                "direction": pred.get("directionEndpoint"),
                "time_ms": elapsed * 1000
            })
            if conf >= 0.3:
                success_count += 1
                print(f"  Crop {crop['index']}: OK (conf={conf:.3f}, {elapsed*1000:.0f}ms)")
            else:
                low_confidence_count += 1
                print(f"  Crop {crop['index']}: Low confidence {conf:.3f}")
        else:
            failed_count += 1
            error = result.get("error", "Unknown error")
            results.append({
                "index": crop["index"],
                "success": False,
                "error": error,
                "time_ms": elapsed * 1000
            })
            print(f"  Crop {crop['index']}: FAILED - {error[:80]}")

    avg_time = (total_time / len(crops)) * 1000 if crops else 0

    print(f"\nResults Summary:")
    print(f"  - Total predictions: {len(crops)}")
    print(f"  - Successful (conf >= 0.3): {success_count}")
    print(f"  - Low confidence (conf < 0.3): {low_confidence_count}")
    print(f"  - Failed: {failed_count}")
    print(f"  - Average time: {avg_time:.1f}ms per prediction")
    print(f"  - Total time: {total_time:.1f}s")

    return {
        "model": model_name,
        "model_path": model_path,
        "total": len(crops),
        "success": success_count,
        "low_confidence": low_confidence_count,
        "failed": failed_count,
        "avg_time_ms": avg_time,
        "total_time_s": total_time,
        "results": results
    }


def main():
    print("="*60)
    print("Keypoint Backend Comparison Test")
    print("="*60)

    # Check server health
    print("\nChecking blob server...")
    try:
        resp = requests.get(f"{SERVER_URL}/health", timeout=5)
        if resp.status_code != 200:
            print(f"ERROR: Server returned {resp.status_code}")
            return
        print("Server is running")
    except Exception as e:
        print(f"ERROR: Cannot connect to server: {e}")
        return

    # List keypoint models
    print("\nListing keypoint models...")
    models = list_keypoint_models()
    if not models:
        print("ERROR: No keypoint models found")
        return

    print(f"Found {len(models)} models:")
    for m in models:
        print(f"  - {m['name']}")

    model = models[0]  # Use the first (most recent) model
    print(f"\nUsing model: {model['name']}")

    weights_dir = Path(model['path']).parent
    pt_model = weights_dir / "best.pt"
    engine_model = weights_dir / "best.engine"

    print(f"PyTorch model: {pt_model}")
    print(f"  Exists: {pt_model.exists()}")
    print(f"TensorRT engine: {engine_model}")
    print(f"  Exists: {engine_model.exists()}")

    # Create test crops
    print("\nCreating test crop images...")
    crops = create_test_crops(20)
    print(f"Created {len(crops)} test crops")

    # Test PyTorch
    pytorch_result = None
    if pt_model.exists():
        pytorch_result = benchmark_model(str(pt_model), "PyTorch", crops)
    else:
        print("\nSkipping PyTorch - model not found")

    # Test TensorRT
    tensorrt_result = None
    if engine_model.exists():
        tensorrt_result = benchmark_model(str(engine_model), "TensorRT", crops)
    else:
        print("\nSkipping TensorRT - engine not found")

    # Summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)

    if pytorch_result:
        pct = 100 * pytorch_result['success'] / pytorch_result['total'] if pytorch_result['total'] > 0 else 0
        print(f"\nPyTorch:")
        print(f"  Success rate: {pytorch_result['success']}/{pytorch_result['total']} ({pct:.1f}%)")
        print(f"  Failed: {pytorch_result['failed']}")
        print(f"  Avg time: {pytorch_result['avg_time_ms']:.1f}ms")

    if tensorrt_result:
        pct = 100 * tensorrt_result['success'] / tensorrt_result['total'] if tensorrt_result['total'] > 0 else 0
        print(f"\nTensorRT:")
        print(f"  Success rate: {tensorrt_result['success']}/{tensorrt_result['total']} ({pct:.1f}%)")
        print(f"  Failed: {tensorrt_result['failed']}")
        print(f"  Avg time: {tensorrt_result['avg_time_ms']:.1f}ms")

    if pytorch_result and tensorrt_result:
        if pytorch_result['success'] > 0 and tensorrt_result['failed'] == tensorrt_result['total']:
            print("\n" + "*"*60)
            print("*** TensorRT FAILED on ALL predictions! ***")
            print("The TensorRT export appears to be broken.")
            print("Recommendation: Re-export the model to TensorRT.")
            print("*"*60)
        elif pytorch_result['success'] > 0 and tensorrt_result['success'] > 0:
            if tensorrt_result['avg_time_ms'] > 0:
                speedup = pytorch_result['avg_time_ms'] / tensorrt_result['avg_time_ms']
                print(f"\nSpeed comparison: TensorRT is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
        elif pytorch_result['failed'] == pytorch_result['total'] and tensorrt_result['failed'] == tensorrt_result['total']:
            print("\n*** Both backends failed on all predictions ***")
            print("The test images may not look enough like real follicles.")


if __name__ == "__main__":
    main()
