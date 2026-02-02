#!/usr/bin/env python3
"""
Diagnose TensorRT keypoint prediction performance issues.

This script investigates why TensorRT might be slower than PyTorch.

Run while the app is running:
    python diagnose_tensorrt.py
"""

import base64
import io
import time
import requests
from pathlib import Path
from PIL import Image
import numpy as np

SERVER_URL = "http://127.0.0.1:5555"


def create_test_image():
    """Create a single test image."""
    img = Image.new('RGB', (100, 100), color=(180, 160, 140))
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"


def load_model(model_path: str):
    """Load a keypoint model."""
    print(f"Loading model: {model_path}")
    start = time.time()
    resp = requests.post(
        f"{SERVER_URL}/yolo-keypoint/load-model",
        json={"modelPath": model_path},
        timeout=120
    )
    elapsed = time.time() - start
    print(f"  Load time: {elapsed:.2f}s")
    return resp.status_code == 200


def warmup_model(test_image: str, n: int = 5):
    """Warmup the model with a few predictions."""
    print(f"Warming up with {n} predictions...")
    for i in range(n):
        resp = requests.post(
            f"{SERVER_URL}/yolo-keypoint/predict",
            json={"imageData": test_image},
            timeout=30
        )


def benchmark_single_predictions(test_image: str, n: int = 20):
    """Benchmark individual predictions."""
    print(f"\nBenchmarking {n} single predictions...")

    times = []
    successes = 0

    for i in range(n):
        start = time.time()
        resp = requests.post(
            f"{SERVER_URL}/yolo-keypoint/predict",
            json={"imageData": test_image},
            timeout=30
        )
        elapsed = (time.time() - start) * 1000  # ms
        times.append(elapsed)

        if resp.status_code == 200:
            result = resp.json()
            if result.get("success") and result.get("prediction"):
                successes += 1

    times.sort()
    avg = sum(times) / len(times)
    median = times[len(times) // 2]
    min_t = times[0]
    max_t = times[-1]
    p95 = times[int(len(times) * 0.95)]

    print(f"  Success: {successes}/{n}")
    print(f"  Avg: {avg:.1f}ms, Median: {median:.1f}ms")
    print(f"  Min: {min_t:.1f}ms, Max: {max_t:.1f}ms, P95: {p95:.1f}ms")

    return avg, successes


def benchmark_batch_predictions(test_image: str, batch_size: int = 16):
    """Benchmark batch predictions."""
    print(f"\nBenchmarking batch prediction (batch_size={batch_size})...")

    images = [test_image] * batch_size

    start = time.time()
    resp = requests.post(
        f"{SERVER_URL}/yolo-keypoint/predict-batch",
        json={"images": images},
        timeout=60
    )
    elapsed = (time.time() - start) * 1000  # ms

    if resp.status_code == 200:
        result = resp.json()
        predictions = result.get("predictions", [])
        successes = sum(1 for p in predictions if p is not None)
        print(f"  Success: {successes}/{batch_size}")
        print(f"  Total time: {elapsed:.1f}ms")
        print(f"  Per image: {elapsed/batch_size:.1f}ms")
        return elapsed / batch_size, successes
    else:
        print(f"  Failed: {resp.status_code}")
        return 0, 0


def check_gpu_usage():
    """Check if GPU is being used."""
    print("\n" + "=" * 60)
    print("Checking GPU Status")
    print("=" * 60)

    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
            print(f"GPU memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.1f} MB")
    except ImportError:
        print("PyTorch not available for GPU check")

    try:
        import tensorrt
        print(f"TensorRT version: {tensorrt.__version__}")
    except ImportError:
        print("TensorRT not installed")


def main():
    print("=" * 60)
    print("TensorRT Keypoint Prediction Diagnostic")
    print("=" * 60)

    # Check server
    try:
        resp = requests.get(f"{SERVER_URL}/health", timeout=5)
        if resp.status_code != 200:
            print("ERROR: Server not responding")
            return
    except:
        print("ERROR: Cannot connect to server")
        return

    # Check GPU
    check_gpu_usage()

    # Get models
    resp = requests.get(f"{SERVER_URL}/yolo-keypoint/models")
    models = resp.json().get("models", [])
    if not models:
        print("ERROR: No keypoint models")
        return

    model = models[0]
    weights_dir = Path(model['path']).parent

    pt_model = weights_dir / "best.pt"
    engine_model = weights_dir / "best.engine"

    if not pt_model.exists():
        print(f"ERROR: PyTorch model not found: {pt_model}")
        return

    if not engine_model.exists():
        print(f"WARNING: TensorRT engine not found: {engine_model}")
        print("Only testing PyTorch model")
        backends = [("PyTorch", str(pt_model))]
    else:
        backends = [
            ("PyTorch", str(pt_model)),
            ("TensorRT", str(engine_model))
        ]

    # Create test image
    test_image = create_test_image()

    results = {}

    for backend_name, model_path in backends:
        print("\n" + "=" * 60)
        print(f"Testing {backend_name}")
        print("=" * 60)

        # Load model
        if not load_model(model_path):
            print(f"Failed to load {backend_name} model")
            continue

        # Warmup
        warmup_model(test_image, n=10)

        # Benchmark single predictions
        single_avg, single_success = benchmark_single_predictions(test_image, n=30)

        # Benchmark batch predictions
        batch_avg, batch_success = benchmark_batch_predictions(test_image, batch_size=16)

        results[backend_name] = {
            "single_avg_ms": single_avg,
            "single_success": single_success,
            "batch_avg_ms": batch_avg,
            "batch_success": batch_success
        }

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, r in results.items():
        print(f"\n{name}:")
        print(f"  Single prediction: {r['single_avg_ms']:.1f}ms (success: {r['single_success']}/30)")
        print(f"  Batch prediction:  {r['batch_avg_ms']:.1f}ms/image (success: {r['batch_success']}/16)")

    if "PyTorch" in results and "TensorRT" in results:
        pt = results["PyTorch"]
        trt = results["TensorRT"]

        print("\n" + "-" * 40)

        if pt["single_avg_ms"] > 0 and trt["single_avg_ms"] > 0:
            if trt["single_avg_ms"] < pt["single_avg_ms"]:
                speedup = pt["single_avg_ms"] / trt["single_avg_ms"]
                print(f"TensorRT is {speedup:.1f}x FASTER than PyTorch (single)")
            else:
                slowdown = trt["single_avg_ms"] / pt["single_avg_ms"]
                print(f"TensorRT is {slowdown:.1f}x SLOWER than PyTorch (single)")
                print("\nPossible causes:")
                print("1. Engine was compiled for different GPU")
                print("2. Data transfer overhead between CPU and GPU")
                print("3. Engine needs recompilation")
                print("4. FP16 precision issues")

        if trt["batch_success"] == 0:
            print("\nBatch prediction FAILED with TensorRT")
            print("This is a known limitation - TensorRT engines don't support dynamic batching")


if __name__ == "__main__":
    main()
