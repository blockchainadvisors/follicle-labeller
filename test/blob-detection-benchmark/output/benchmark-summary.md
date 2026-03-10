# Blob Detection Benchmark Report

**Generated:** 2026-01-26T21:11:00.780Z
**Test Image:** 5.jpg (7500x10000)

## Summary

| Metric | Value |
|--------|-------|
| Total Tests | 242 |
| Duration | 9811.6s |
| Best Blob Count | 5305 |

## Best Configuration

| Parameter | Value |
|-----------|-------|
| Gaussian Blur | OFF |
| Morphological Opening | OFF |
| Circularity Filter | 0.00 |
| CLAHE | ON (1.1) |

## Parameter Impact Analysis

### Gaussian Blur

| Setting | Avg Blobs | Samples | Recommendation |
|---------|-----------|---------|----------------|
| OFF | 4263 | 78 | RECOMMENDED |
| 3x3 | 4067 | 66 |  |
| 5x5 | 3718 | 50 |  |
| 7x7 | 3621 | 48 |  |

### Morphological Opening

| Setting | Avg Blobs | Samples | Recommendation |
|---------|-----------|---------|----------------|
| OFF | 4451 | 91 | RECOMMENDED |
| 3x3 | 3995 | 54 |  |
| 5x5 | 3647 | 49 |  |
| 7x7 | 3358 | 48 |  |

### Circularity

| Setting | Avg Blobs | Samples | Recommendation |
|---------|-----------|---------|----------------|
| 0.03 | 5233 | 3 | RECOMMENDED |
| 0.02 | 5233 | 3 |  |
| 0.01 | 5233 | 3 |  |
| 0.05 | 4972 | 3 |  |
| 0.15 | 4948 | 2 |  |
| 0.10 | 4946 | 3 |  |
| 0.25 | 4774 | 2 |  |
| 0.30 | 4666 | 2 |  |
| 0.00 | 4113 | 84 |  |
| 0.20 | 3909 | 73 |  |
| 0.40 | 3503 | 64 |  |

### CLAHE

| Setting | Avg Blobs | Samples | Recommendation |
|---------|-----------|---------|----------------|
| 1.1 | 5238 | 2 | RECOMMENDED |
| 1.2 | 5198 | 2 |  |
| 1.3 | 5172 | 8 |  |
| 1.0 | 5171 | 11 |  |
| 1.4 | 5151 | 1 |  |
| 1.8 | 4710 | 5 |  |
| 2.0 | 4587 | 5 |  |
| 1.5 | 4474 | 64 |  |
| 3.0 | 3798 | 48 |  |
| 4.5 | 3677 | 48 |  |
| OFF | 3016 | 48 |  |

## Top 10 Configurations

| Rank | Blobs | Time (ms) | Configuration |
|------|-------|-----------|---------------|
| 1 | 5305 | 3243.5 | blur=OFF+morph=OFF+circ=0.00+clahe=1.1 |
| 2 | 5272 | 3155.7 | blur=OFF+morph=OFF+circ=0.03+clahe=1.0 |
| 3 | 5272 | 3202.1 | blur=OFF+morph=OFF+circ=0.02+clahe=1.0 |
| 4 | 5272 | 3244.8 | blur=OFF+morph=OFF+circ=0.00+clahe=1.0 |
| 5 | 5272 | 3344.2 | blur=OFF+morph=OFF+circ=0.01+clahe=1.0 |
| 6 | 5268 | 3192.9 | blur=OFF+morph=OFF+circ=0.01+clahe=1.3 |
| 7 | 5268 | 3229.2 | blur=OFF+morph=OFF+circ=0.02+clahe=1.3 |
| 8 | 5268 | 3244.7 | blur=OFF+morph=OFF+circ=0.03+clahe=1.3 |
| 9 | 5268 | 3269.0 | blur=OFF+morph=OFF+circ=0.00+clahe=1.2 |
| 10 | 5268 | 4367.5 | blur=OFF+morph=OFF+circ=0.00+clahe=1.3 |

## Phase Progress

| Phase | Tests | Duration | Best Blobs |
|-------|-------|----------|------------|
| Phase 1: Coarse Grid Search | 192 | 9419.3s | 5093 |
| Phase 2: Refinement | 35 | 296.7s | 5272 |
| Phase 3: Fine-Tuning | 15 | 95.6s | 5305 |
