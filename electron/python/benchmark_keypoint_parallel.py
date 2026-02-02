#!/usr/bin/env python3
"""
Benchmark different keypoint prediction parallelization strategies.

Tests:
1. Sequential (current approach) - one prediction at a time
2. Batch inference - send multiple images to model at once
3. Parallel HTTP - concurrent HTTP requests (asyncio)

Run while the app is running with an image loaded:
    python benchmark_keypoint_parallel.py
"""

import base64
import io
import json
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image
import numpy as np

SERVER_URL = "http://127.0.0.1:5555"

# Number of samples to test
NUM_SAMPLES = 500


def create_test_crops(count: int = 100):
    """Create test crop images."""
    crops = []
    for i in range(count):
        size = 80 + (i % 3) * 10
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
                    base = 60 + (i * 5) % 40
                    pixels[x, y] = (base + darkness//2, base, base)

        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        crops.append(f"data:image/png;base64,{img_b64}")

    return crops


def load_model(model_path: str):
    """Load a keypoint model."""
    resp = requests.post(
        f"{SERVER_URL}/yolo-keypoint/load-model",
        json={"modelPath": model_path},
        timeout=60
    )
    return resp.status_code == 200


def benchmark_sequential(crops: list) -> dict:
    """Benchmark sequential prediction (current approach)."""
    print(f"\n1. Sequential prediction ({len(crops)} images)...")

    start = time.time()
    success_count = 0

    for i, crop in enumerate(crops):
        resp = requests.post(
            f"{SERVER_URL}/yolo-keypoint/predict",
            json={"imageData": crop},
            timeout=30
        )
        if resp.status_code == 200:
            result = resp.json()
            if result.get("success") and result.get("prediction"):
                success_count += 1

        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{len(crops)}")

    elapsed = time.time() - start

    return {
        "method": "Sequential",
        "total": len(crops),
        "success": success_count,
        "time_s": elapsed,
        "avg_ms": (elapsed / len(crops)) * 1000
    }


def benchmark_batch(crops: list, batch_size: int = 16) -> dict:
    """Benchmark batch prediction."""
    print(f"\n2. Batch prediction ({len(crops)} images, batch_size={batch_size})...")

    start = time.time()
    success_count = 0

    # Process in batches
    for i in range(0, len(crops), batch_size):
        batch = crops[i:i + batch_size]
        resp = requests.post(
            f"{SERVER_URL}/yolo-keypoint/predict-batch",
            json={"images": batch},
            timeout=60
        )
        if resp.status_code == 200:
            result = resp.json()
            predictions = result.get("predictions", [])
            success_count += sum(1 for p in predictions if p is not None)

        if (i + batch_size) % 50 == 0 or (i + batch_size) >= len(crops):
            print(f"  Progress: {min(i + batch_size, len(crops))}/{len(crops)}")

    elapsed = time.time() - start

    return {
        "method": f"Batch (size={batch_size})",
        "total": len(crops),
        "success": success_count,
        "time_s": elapsed,
        "avg_ms": (elapsed / len(crops)) * 1000
    }


def predict_single(crop: str) -> bool:
    """Make single prediction request."""
    try:
        resp = requests.post(
            f"{SERVER_URL}/yolo-keypoint/predict",
            json={"imageData": crop},
            timeout=30
        )
        if resp.status_code == 200:
            result = resp.json()
            return result.get("success") and result.get("prediction") is not None
    except:
        pass
    return False


def benchmark_parallel_threads(crops: list, num_workers: int = 10) -> dict:
    """Benchmark parallel HTTP requests using ThreadPoolExecutor."""
    print(f"\n3. Parallel threads ({len(crops)} images, workers={num_workers})...")

    start = time.time()
    success_count = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(predict_single, crop) for crop in crops]

        completed = 0
        for future in as_completed(futures):
            if future.result():
                success_count += 1
            completed += 1
            if completed % 20 == 0:
                print(f"  Progress: {completed}/{len(crops)}")

    elapsed = time.time() - start

    return {
        "method": f"Parallel threads (workers={num_workers})",
        "total": len(crops),
        "success": success_count,
        "time_s": elapsed,
        "avg_ms": (elapsed / len(crops)) * 1000
    }


def main():
    print("="*60)
    print("Keypoint Prediction Parallelization Benchmark")
    print("="*60)

    # Check server
    try:
        resp = requests.get(f"{SERVER_URL}/health", timeout=5)
        if resp.status_code != 200:
            print("ERROR: Server not responding")
            return
    except:
        print("ERROR: Cannot connect to server")
        return

    # Get models
    resp = requests.get(f"{SERVER_URL}/yolo-keypoint/models")
    models = resp.json().get("models", [])
    if not models:
        print("ERROR: No keypoint models")
        return

    model = models[0]
    weights_dir = Path(model['path']).parent

    # Test both PyTorch and TensorRT
    backends = []

    pt_model = weights_dir / "best.pt"
    if pt_model.exists():
        backends.append(("PyTorch", str(pt_model)))

    engine_model = weights_dir / "best.engine"
    if engine_model.exists():
        backends.append(("TensorRT", str(engine_model)))

    # Create test crops
    print(f"\nCreating {NUM_SAMPLES} test crops...")
    crops = create_test_crops(NUM_SAMPLES)

    all_results = []

    for backend_name, model_path in backends:
        print(f"\n{'='*60}")
        print(f"Testing with {backend_name}")
        print(f"{'='*60}")

        # Load model
        print(f"Loading model: {model_path}")
        if not load_model(model_path):
            print("Failed to load model")
            continue

        # Run benchmarks
        results = []

        # 1. Sequential
        results.append(benchmark_sequential(crops))

        # 2. Batch with different sizes
        for batch_size in [16, 32, 64, 128]:
            results.append(benchmark_batch(crops, batch_size))

        # 3. Parallel threads (only test if batch fails or for comparison)
        # Skip for PyTorch since batch is faster, but test for TensorRT
        if backend_name == "TensorRT":
            for num_workers in [4, 8]:
                result = benchmark_parallel_threads(crops, num_workers)
                results.append(result)

        # Print results for this backend
        print(f"\n{backend_name} Results:")
        print("-" * 60)
        print(f"{'Method':<35} {'Time (s)':<10} {'Avg (ms)':<10} {'Success'}")
        print("-" * 60)

        for r in results:
            print(f"{r['method']:<35} {r['time_s']:<10.2f} {r['avg_ms']:<10.1f} {r['success']}/{r['total']}")

        # Find fastest
        fastest = min(results, key=lambda x: x['time_s'])
        print(f"\nFastest: {fastest['method']} ({fastest['time_s']:.2f}s)")

        for r in results:
            r['backend'] = backend_name
        all_results.extend(results)

    # Overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print("="*60)

    if all_results:
        fastest_overall = min(all_results, key=lambda x: x['time_s'])
        slowest_overall = max(all_results, key=lambda x: x['time_s'])

        speedup = slowest_overall['time_s'] / fastest_overall['time_s']

        print(f"\nFastest overall: {fastest_overall['backend']} - {fastest_overall['method']}")
        print(f"  Time: {fastest_overall['time_s']:.2f}s ({fastest_overall['avg_ms']:.1f}ms per image)")
        print(f"\nSlowest: {slowest_overall['backend']} - {slowest_overall['method']}")
        print(f"  Time: {slowest_overall['time_s']:.2f}s ({slowest_overall['avg_ms']:.1f}ms per image)")
        print(f"\nMax speedup: {speedup:.1f}x")


if __name__ == "__main__":
    main()
