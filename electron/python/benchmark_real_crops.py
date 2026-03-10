#!/usr/bin/env python3
"""
Benchmark keypoint prediction using REAL crops from the loaded image session.

Run while the app is running with an image and annotations loaded:
    python benchmark_real_crops.py
"""

import json
import time
import requests

SERVER_URL = "http://127.0.0.1:5555"

def get_session_info():
    """Get current session info."""
    resp = requests.get(f"{SERVER_URL}/health", timeout=5)
    if resp.status_code != 200:
        return None
    return resp.json()


def get_real_crops(session_id: str, count: int = 500):
    """Get real crops from annotations in the session."""
    # Get annotations from session
    resp = requests.post(
        f"{SERVER_URL}/get-annotation-count",
        json={"sessionId": session_id},
        timeout=10
    )
    if resp.status_code != 200:
        print(f"Failed to get annotation count: {resp.text}")
        return []

    info = resp.json()
    annotation_count = info.get("annotationCount", 0)
    print(f"Session has {annotation_count} annotations")

    if annotation_count == 0:
        print("No annotations in session - cannot get real crops")
        return []

    # We need to get crops by requesting them individually
    # Use the blob detection to get sample regions
    resp = requests.post(
        f"{SERVER_URL}/blob-detect",
        json={
            "sessionId": session_id,
            "settings": {
                "useLearnedStats": True,
                "tolerance": 0.5
            }
        },
        timeout=120
    )

    if resp.status_code != 200:
        print(f"Failed to run detection: {resp.text}")
        return []

    result = resp.json()
    detections = result.get("detections", [])
    print(f"Got {len(detections)} detections to use as crop regions")

    # Limit to requested count
    detections = detections[:count]

    # Get crops for each detection
    crops = []
    for i, det in enumerate(detections):
        resp = requests.post(
            f"{SERVER_URL}/crop-region",
            json={
                "sessionId": session_id,
                "x": det["x"],
                "y": det["y"],
                "width": det["width"],
                "height": det["height"]
            },
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("image"):
                crops.append(data["image"])

        if (i + 1) % 100 == 0:
            print(f"  Cropping progress: {i + 1}/{len(detections)}")

    print(f"Got {len(crops)} real crops")
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
    """Benchmark sequential prediction."""
    print(f"\nSequential prediction ({len(crops)} real crops)...")

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

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{len(crops)}")

    elapsed = time.time() - start

    return {
        "method": "Sequential",
        "total": len(crops),
        "success": success_count,
        "time_s": elapsed,
        "avg_ms": (elapsed / len(crops)) * 1000 if crops else 0
    }


def benchmark_batch(crops: list, batch_size: int) -> dict:
    """Benchmark batch prediction."""
    print(f"\nBatch prediction ({len(crops)} real crops, batch_size={batch_size})...")

    start = time.time()
    success_count = 0

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

        if (i + batch_size) % 100 == 0 or (i + batch_size) >= len(crops):
            print(f"  Progress: {min(i + batch_size, len(crops))}/{len(crops)}")

    elapsed = time.time() - start

    return {
        "method": f"Batch (size={batch_size})",
        "total": len(crops),
        "success": success_count,
        "time_s": elapsed,
        "avg_ms": (elapsed / len(crops)) * 1000 if crops else 0
    }


def main():
    from pathlib import Path

    print("=" * 60)
    print("Keypoint Prediction Benchmark - REAL CROPS")
    print("=" * 60)

    # Check server and get session
    try:
        resp = requests.get(f"{SERVER_URL}/health", timeout=5)
        if resp.status_code != 200:
            print("ERROR: Server not responding")
            return
    except:
        print("ERROR: Cannot connect to server")
        return

    # Get current session
    # We need to know the session ID - try to get it from a test endpoint
    # For now, use a known session pattern
    print("\nNote: You need to have an image with annotations loaded in the app")

    # Try to get session from a detection request
    # The session should be set when an image is loaded
    session_id = input("Enter session ID (or press Enter to try auto-detect): ").strip()

    if not session_id:
        print("No session ID provided. Please load an image first and check the console for session ID.")
        return

    # Get models
    resp = requests.get(f"{SERVER_URL}/yolo-keypoint/models")
    models = resp.json().get("models", [])
    if not models:
        print("ERROR: No keypoint models")
        return

    model = models[0]
    weights_dir = Path(model['path']).parent

    # Get real crops
    print(f"\nGetting real crops from session {session_id}...")
    crops = get_real_crops(session_id, count=500)

    if len(crops) < 10:
        print(f"Not enough crops ({len(crops)}). Need at least 10 for meaningful benchmark.")
        return

    # Test both backends
    backends = []

    pt_model = weights_dir / "best.pt"
    if pt_model.exists():
        backends.append(("PyTorch", str(pt_model)))

    engine_model = weights_dir / "best.engine"
    if engine_model.exists():
        backends.append(("TensorRT", str(engine_model)))

    all_results = []

    for backend_name, model_path in backends:
        print(f"\n{'=' * 60}")
        print(f"Testing with {backend_name}")
        print(f"{'=' * 60}")

        print(f"Loading model: {model_path}")
        if not load_model(model_path):
            print("Failed to load model")
            continue

        results = []

        # Sequential (just first 100 for speed)
        if len(crops) > 100:
            seq_crops = crops[:100]
            seq_result = benchmark_sequential(seq_crops)
            seq_result["method"] = "Sequential (100 samples)"
            results.append(seq_result)
        else:
            results.append(benchmark_sequential(crops))

        # Batch with different sizes
        for batch_size in [32, 64, 128]:
            results.append(benchmark_batch(crops, batch_size))

        # Print results
        print(f"\n{backend_name} Results:")
        print("-" * 70)
        print(f"{'Method':<30} {'Time (s)':<10} {'Avg (ms)':<12} {'Success Rate'}")
        print("-" * 70)

        for r in results:
            success_rate = f"{r['success']}/{r['total']} ({100*r['success']/r['total']:.1f}%)" if r['total'] > 0 else "N/A"
            print(f"{r['method']:<30} {r['time_s']:<10.2f} {r['avg_ms']:<12.1f} {success_rate}")

        for r in results:
            r['backend'] = backend_name
        all_results.extend(results)

    # Summary
    if all_results:
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print("=" * 60)

        # Find best batch for each backend
        for backend in ["PyTorch", "TensorRT"]:
            backend_results = [r for r in all_results if r.get('backend') == backend and 'Batch' in r['method']]
            if backend_results:
                # Filter to those with good success rate
                good_results = [r for r in backend_results if r['success'] > r['total'] * 0.5]
                if good_results:
                    best = min(good_results, key=lambda x: x['avg_ms'])
                    print(f"\n{backend} best: {best['method']}")
                    print(f"  {best['avg_ms']:.1f}ms/image, {best['success']}/{best['total']} success")
                else:
                    print(f"\n{backend}: Batch prediction not working (low success rate)")


if __name__ == "__main__":
    main()
