# Chapter 7: Testing and Quality Assurance

This chapter documents the testing approaches, benchmark results, and quality assurance measures implemented during the project.

## 7.1 Testing Strategy

### 7.1.1 Testing Approach Overview

Due to the rapid development pace and visual nature of the application, the project employed a combination of testing strategies:

| Type | Coverage | Tools |
|------|----------|-------|
| Manual Testing | High | Direct user interaction |
| Benchmark Testing | Medium | Custom benchmark suite |
| Integration Testing | Medium | End-to-end workflows |
| Visual Testing | High | Screenshot comparison |

### 7.1.2 Testing Priorities

1. **Core Functionality**
   - Annotation creation (all shapes)
   - Save/load projects
   - Detection algorithms
   - Multi-image support

2. **Cross-Platform**
   - Windows 10/11
   - macOS (Intel and Apple Silicon)
   - Linux (Ubuntu)

3. **Edge Cases**
   - Very large images (10000+ pixels)
   - Empty projects
   - Corrupted files
   - Network failures (for updates)

---

## 7.2 Benchmark System

### 7.2.1 BLOB Detection Benchmark

**Location:** `test/blob-detection-benchmark/`

**Purpose:**
Compare detection performance across different backends and configurations.

**Metrics Captured:**
- Processing time (ms)
- Number of detections
- Memory usage
- GPU utilization

### 7.2.2 CPU vs GPU Performance Results

**Test Image:** 7500 x 10000 pixels (75 megapixels)

| Operation | CPU (ms) | GPU/CuPy (ms) | Ratio |
|-----------|----------|---------------|-------|
| Grayscale | 15 | 480 | 32x slower |
| CLAHE | 8 | 145 | 18x slower |
| Gaussian Blur | 12 | 95 | 8x slower |
| Morphology | 5 | 85 | 17x slower |
| **Total Preprocessing** | 40 | 805 | 20x slower |

**Conclusion:** For single-image preprocessing, CPU is faster due to GPU memory transfer overhead.

### 7.2.3 YOLO Training Performance

**Test Configuration:**
- Dataset: 200 images
- Epochs: 100
- Image Size: 640px
- Batch Size: 16

| Device | Time/Epoch | Total Training |
|--------|------------|----------------|
| CPU (i7-10700) | ~180s | ~5 hours |
| NVIDIA RTX 3080 | ~15s | ~25 minutes |
| Apple M2 Pro | ~25s | ~42 minutes |

**Conclusion:** GPU significantly benefits batch operations like model training.

---

## 7.3 Manual Testing Protocols

### 7.3.1 Annotation Testing Checklist

```
[ ] Create circle annotation
    [ ] Click to set center
    [ ] Drag to set radius
    [ ] Verify visual display
    [ ] Check data storage

[ ] Create rectangle annotation
    [ ] Click-drag from corner
    [ ] Verify dimensions
    [ ] Check aspect ratio handling

[ ] Create linear annotation
    [ ] Two-phase creation (line, then width)
    [ ] Verify rotation
    [ ] Check centerline display

[ ] Selection operations
    [ ] Single click select
    [ ] Shift+click multi-select
    [ ] Marquee selection
    [ ] Lasso selection

[ ] Editing operations
    [ ] Move annotation (drag)
    [ ] Resize annotation (handles)
    [ ] Delete annotation (Del key)
    [ ] Undo/Redo (Ctrl+Z/Ctrl+Shift+Z)
```

### 7.3.2 Detection Testing Checklist

```
[ ] BLOB Detection
    [ ] Open detection settings
    [ ] Adjust parameters
    [ ] Run detection
    [ ] Verify results match expected

[ ] Learned Detection
    [ ] Select 3+ examples
    [ ] Click "Learn from Selection"
    [ ] Verify learned parameters
    [ ] Run detection
    [ ] Compare with manual parameters

[ ] YOLO Detection
    [ ] Train model (if not exists)
    [ ] Load trained model
    [ ] Run inference
    [ ] Verify bounding boxes
```

### 7.3.3 File Operations Checklist

```
[ ] New Project
    [ ] Create empty project
    [ ] Add images
    [ ] Verify image loading

[ ] Save Project
    [ ] Save new project
    [ ] Verify .fol file created
    [ ] Check file contents (ZIP structure)

[ ] Load Project
    [ ] Open existing .fol file
    [ ] Verify images restored
    [ ] Verify annotations restored
    [ ] Check viewport state

[ ] Export
    [ ] Export JSON
    [ ] Export CSV
    [ ] Export YOLO dataset
    [ ] Verify file contents
```

---

## 7.4 Cross-Platform Testing

### 7.4.1 Windows Testing

**Versions Tested:**
- Windows 10 (21H2)
- Windows 11 (23H2)

**Known Issues Found:**
1. Resume training deadlock (fixed with workers=0)
2. File association registry issues (resolved)

**Status:** All features working

### 7.4.2 macOS Testing

**Versions Tested:**
- macOS 13 Ventura (Intel)
- macOS 14 Sonoma (Apple Silicon)

**Known Issues Found:**
1. Code signing required for distribution
2. Notarization required for Gatekeeper
3. Universal binary needed for both architectures

**Status:** All features working after signing

### 7.4.3 Linux Testing

**Distributions Tested:**
- Ubuntu 22.04 LTS
- Ubuntu 24.04 LTS

**Known Issues Found:**
1. AppImage permissions on some systems
2. GPU detection varies by driver version

**Status:** Core features working

---

## 7.5 Error Handling Testing

### 7.5.1 Error Scenarios Tested

| Scenario | Expected Behavior | Result |
|----------|-------------------|--------|
| Open corrupted .fol file | Error message, don't crash | Pass |
| Network failure during update | Graceful fallback | Pass |
| GPU not available | Fall back to CPU | Pass |
| Invalid image format | Error message | Pass |
| Disk full during save | Error message | Pass |
| Python service not started | Health check fails, retry | Pass |

### 7.5.2 Recovery Testing

| Scenario | Recovery Behavior |
|----------|-------------------|
| App crash | No data loss (auto-save) |
| System sleep | Auto-save triggered |
| Forced quit | Restore from last save |
| Network disconnect | Continue offline |

---

## 7.6 Performance Testing

### 7.6.1 Large Image Handling

**Test Images:**
- Small: 1024 x 768 (0.8 MP)
- Medium: 4096 x 3072 (12.6 MP)
- Large: 7500 x 10000 (75 MP)
- Very Large: 15000 x 20000 (300 MP)

**Results:**

| Size | Load Time | Zoom Response | Annotation Speed |
|------|-----------|---------------|------------------|
| Small | <100ms | Instant | Instant |
| Medium | ~500ms | Instant | Instant |
| Large | ~2s | Smooth | Smooth |
| Very Large | ~8s | Slight lag | Acceptable |

### 7.6.2 Many Annotations Performance

| Annotation Count | Render Time | Selection Time |
|------------------|-------------|----------------|
| 100 | <5ms | Instant |
| 1,000 | ~15ms | <50ms |
| 5,000 | ~50ms | ~100ms |
| 10,000 | ~120ms | ~250ms |

**Conclusion:** Performance acceptable up to ~5,000 annotations per image.

---

## 7.7 Quality Assurance Measures

### 7.7.1 Code Quality

**TypeScript Strict Mode:**
```json
{
  "compilerOptions": {
    "strict": true,
    "noImplicitAny": true,
    "strictNullChecks": true
  }
}
```

**ESLint Configuration:**
- React hooks rules
- TypeScript recommended rules
- Import order rules

### 7.7.2 Version Control Practices

- Feature branches for new features
- Pull requests for code review
- Descriptive commit messages
- Tags for releases

### 7.7.3 Documentation

| Document | Purpose |
|----------|---------|
| README.md | User guide |
| DESIGN_GUIDELINES.md | UI/UX standards |
| FOL-FILE-FORMAT.md | File format spec |
| Placement Report | Development history |

---

## 7.8 Known Issues and Limitations

### 7.8.1 Current Known Issues

| Issue | Severity | Workaround |
|-------|----------|------------|
| GPU slower for small ops | Low | Use CPU (auto) |
| Large images slow on old HW | Medium | Reduce image size |
| No batch export | Low | Export one at a time |

### 7.8.2 Accepted Limitations

| Limitation | Reason |
|------------|--------|
| Electron bundle size (~150MB) | Trade-off for cross-platform |
| Python required for detection | Best ML ecosystem |
| No mobile support | Desktop focus |

---

## 7.9 Future Testing Improvements

### 7.9.1 Recommended Additions

1. **Unit Tests**
   - Jest for React components
   - pytest for Python services
   - Coverage targets: 70%+

2. **End-to-End Tests**
   - Playwright for Electron
   - Automated UI testing
   - CI integration

3. **Performance Regression Tests**
   - Automated benchmarks
   - Alert on degradation
   - Historical tracking

4. **Accessibility Testing**
   - Screen reader compatibility
   - Keyboard navigation
   - ARIA compliance

### 7.9.2 CI/CD Improvements

1. Run tests before merge
2. Automated benchmark comparison
3. Screenshot diff testing
4. Multi-platform test matrix
