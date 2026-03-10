# Chapter 5: Challenges and Solutions

This chapter documents the technical challenges encountered during development, the approaches that failed, and the solutions that worked.

## 5.1 Detection Algorithm Challenges

### 5.1.1 TypeScript vs Python BLOB Detection

**Problem:**
Initial BLOB detection was implemented in TypeScript using a JavaScript port of OpenCV. The results were significantly inferior to the Python implementation.

**Session:** `a28e6916` - "Blob detection TypeScript vs Python comparison"

**Investigation:**
- Compared detection results between branches
- TypeScript implementation produced fewer accurate detections
- JavaScript OpenCV lacked advanced preprocessing options

**Failed Approaches:**
1. Tuning TypeScript parameters extensively
2. Adding custom preprocessing in JavaScript

**Solution:**
Migrated to Python-based BLOB detection:
- Full OpenCV functionality available
- Added CLAHE, morphological operations
- FastAPI server for HTTP communication
- IPC bridge from Electron

**Commit:** `bd2360c` - "Migrate to FastAPI and add shape filter controls"

### 5.1.2 GPU Performance Overhead

**Problem:**
GPU-accelerated image processing was slower than CPU for small operations.

**Session:** `c848cfc6` - "GPU perf optimization & blob detection parallelization"

**Analysis:**
```
Image size: 7500x10000
- Grayscale: 15ms CPU vs 480ms GPU (32x slower)
- CLAHE: 8ms CPU vs 145ms GPU (18x slower)
```

**Root Cause:**
- GPU memory transfer overhead
- Small operation kernel launch overhead
- CuPy not optimized for single-image processing

**Failed Approaches:**
1. Batching operations on GPU
2. Persistent GPU memory allocation
3. Custom CUDA kernels

**Solution:**
- Kept GPU for YOLO training (large batches benefit)
- Used CPU for preprocessing (faster for single images)
- Added backend selection UI for user choice
- Documented performance characteristics

### 5.1.3 BLOB Detection Parameter Tuning

**Problem:**
Fixed detection parameters worked poorly across different image types.

**Session:** `d88dda42` - "BLOB Detection with Example-Based Learning"

**Failed Approaches:**
1. Universal "best" parameters
2. Image histogram-based auto-tuning

**Solution:**
Implemented "Learn from Selection" feature:
- User selects 3+ example follicles
- System calculates parameter ranges automatically
- Tolerance slider for fine-tuning
- Statistics display shows learned values

---

## 5.2 Platform-Specific Issues

### 5.2.1 Windows Resume Training Deadlock

**Problem:**
YOLO training would freeze/deadlock when resuming from a checkpoint on Windows.

**Session:** Current session (Jan 29, 2026)

**Error Symptoms:**
- Training started but no progress
- Console showed no output
- Application became unresponsive

**Investigation:**
- Issue specific to Windows multiprocessing
- DataLoader workers causing deadlock
- Resume path triggered different code path

**Solution:**
Set `workers=0` for Windows when resuming:
```python
if sys.platform == 'win32' and resume_path:
    workers = 0  # Prevent deadlock on Windows
```

**Commit:** `9eddece` - "Fix resume training deadlock on Windows"

### 5.2.2 macOS Code Signing & Notarization

**Problem:**
macOS users received "app is damaged" or Gatekeeper warnings.

**Session:** `24e4253d` - "macOS signing, notarization, smooth auto-updates"

**Challenges:**
1. Developer ID certificate setup
2. Notarization requirements
3. Universal binary (Intel + Apple Silicon)
4. Auto-update signature verification

**Failed Approaches:**
1. Skipping notarization (app blocked by Gatekeeper)
2. Using ad-hoc signing (no distribution)

**Solution:**
Complete signing pipeline:
```yaml
# electron-builder config
mac:
  hardenedRuntime: true
  gatekeeperAssess: false
  entitlements: build/entitlements.mac.plist
  notarize:
    teamId: TEAM_ID
```

**Commits:**
```
1b4a602 Add macOS code signing, notarization, and smooth auto-update
80eee59 Parallelize macOS notarization for arm64 and x64
```

### 5.2.3 macOS Icon Generation in CI

**Problem:**
macOS builds failed due to icon format issues.

**Sessions:** Multiple sessions on Jan 20, 2026

**Failed Approaches:**
1. Pre-committed .icns file (corrupted in git)
2. Using iconutil without proper setup
3. Electron-builder icon auto-generation

**Solution:**
Generate icns during CI build:
```yaml
- name: Generate macOS icons
  if: matrix.os == 'macos-latest'
  run: |
    mkdir -p assets/icons.iconset
    sips -z 1024 1024 assets/icon.png --out assets/icons.iconset/icon_512x512@2x.png
    iconutil -c icns assets/icons.iconset -o assets/icon.icns
```

### 5.2.4 Silent CUDA Initialization Failures

**Problem:**
GPU detection showed "CUDA available" but operations silently failed.

**Session:** `dc405a15` - "CUDA bugfix + YOLO detector implementation"

**Root Cause:**
- CUDA toolkit not fully installed
- nvidia-smi detected GPU but CUDA libraries missing
- No error feedback to user

**Solution:**
1. Added CUDA toolkit detection:
```python
def find_cuda_path():
    # Check CUDA_PATH environment variable
    # Search standard installation paths
    # Verify nvcc.exe exists
```

2. Improved error messaging:
```python
if not cuda_toolkit_found:
    return {
        "available": False,
        "reason": "CUDA Toolkit not found. Please install CUDA Toolkit from nvidia.com"
    }
```

**Commits:**
```
8fec929 Add CUDA toolkit detection and improve GPU fallback messaging
2e7612c Fix silent CUDA/GPU initialization failures in production mode
```

---

## 5.3 UI/UX Challenges

### 5.3.1 CSS Inconsistency Across Components

**Problem:**
Different components used different color values and variable names.

**Session:** `a6a1d85d` - "Fix Frontend CSS Styling Inconsistencies"

**Examples:**
```css
/* Component A */
background: var(--panel-bg);

/* Component B */
background: var(--bg-secondary);

/* Component C */
background: #2a2a2a;  /* Hard-coded */
```

**Solution:**
Created centralized design system:
1. Unified CSS variable naming
2. Semantic color variables
3. Typography system with 8 sizes
4. Documented in DESIGN_GUIDELINES.md

**Commits:**
```
112091e Refactor component CSS to use typography variables
f4171bf Add comprehensive UI design guidelines document
```

### 5.3.2 Theme Not Applied to Canvas Background

**Problem:**
When switching themes, the canvas background didn't update.

**Session:** `5711d641` - "Sleep fix, theme colors, canvas background sync"

**Root Cause:**
Canvas uses direct drawing, not CSS-based styling.

**Solution:**
Subscribe to theme store in canvas renderer:
```typescript
useEffect(() => {
  const unsubscribe = themeStore.subscribe((state) => {
    canvasBackground = state.canvasBackground;
    redraw();
  });
  return unsubscribe;
}, []);
```

### 5.3.3 Dialog Layout Issues

**Problem:**
Detection settings dialog had broken CSS layout.

**Session:** `c4519d52` - "Fix Blob Detection CSS and Implement CPU/GPU Tests"

**Issues:**
- Input fields overflowing
- Scrollbar gaps
- Inconsistent spacing

**Solution:**
Complete CSS rewrite with:
- Flexbox-based layout
- Consistent padding/margins
- Proper overflow handling
- Max-height constraints

---

## 5.4 State Management Challenges

### 5.4.1 False Positive Dirty State

**Problem:**
Opening a project showed "unsaved changes" warning immediately.

**Commit:** `a666510` - "Fix false positive isDirty state when loading projects"

**Root Cause:**
Loading annotations triggered state change listeners that set `isDirty = true`.

**Solution:**
Add loading flag to skip dirty tracking during load:
```typescript
let isLoading = false;

const loadProject = async () => {
  isLoading = true;
  // Load annotations...
  isLoading = false;
  setIsDirty(false);
};
```

### 5.4.2 Import Duplicate Detection

**Problem:**
Importing annotations created duplicates if same follicles existed.

**Session:** `0e98e1ae` - "Origin Filter Fix & Import Enhancement"

**Challenges:**
1. Defining "duplicate" (exact match vs. overlap)
2. Handling slight position differences
3. User decision for each duplicate

**Solution:**
Comprehensive duplicate detection:
```typescript
function findDuplicates(existing: Follicle[], imported: Follicle[]) {
  return imported.filter(imp =>
    existing.some(ex =>
      Math.abs(imp.cx - ex.cx) < threshold &&
      Math.abs(imp.cy - ex.cy) < threshold
    )
  );
}
```

Created DuplicateDetectionDialog for user review:
- Shows duplicate count
- Preview of duplicates
- Options: Keep All, Skip Duplicates, Replace

---

## 5.5 IPC and Process Management

### 5.5.1 Browser Console ERR_CONNECTION_REFUSED

**Problem:**
Health check requests to Python servers showed errors in browser console.

**Commit:** `e8265bb` - "Route health checks through IPC to suppress browser console errors"

**Root Cause:**
Direct fetch() calls from renderer showed errors before server started.

**Solution:**
Route health checks through Electron IPC:
```typescript
// Renderer
const isServerReady = await window.electron.ipcRenderer.invoke('blob:health');

// Main process
ipcMain.handle('blob:health', async () => {
  try {
    const response = await fetch('http://localhost:5001/health');
    return response.ok;
  } catch {
    return false;
  }
});
```

### 5.5.2 Python Package Installation in Production

**Problem:**
Packaged app couldn't find Python or install packages.

**Commit:** `956109b` - "Fix packaged app missing Python files"

**Solution:**
1. Bundle Python files with app
2. Create virtual environment on first run
3. Install requirements.txt automatically
4. Handle both dev and production paths

---

## 5.6 Machine Learning Challenges

### 5.6.1 YOLO Training with Large Images

**Problem:**
Training images larger than YOLO input size caused issues.

**Session:** `9e520aa2` - "YOLO Detection Training UI, Tiling, Resume Feature"

**Solution:**
Implemented tiled inference:
```python
def detect_with_tiling(image, model, tile_size=640, overlap=0.2):
    tiles = []
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            tile = image[y:y+tile_size, x:x+tile_size]
            results = model.predict(tile)
            # Offset results by tile position
            tiles.extend(offset_results(results, x, y))
    return nms(tiles)  # Remove duplicates at tile boundaries
```

### 5.6.2 Training Progress Not Displayed

**Problem:**
YOLO training runs silently with no progress feedback.

**Solution:**
Implemented Server-Sent Events (SSE) for real-time progress:
```python
@app.get("/train/progress")
async def training_progress():
    async def event_generator():
        while training:
            yield f"data: {json.dumps(current_metrics)}\n\n"
            await asyncio.sleep(1)
    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

---

## 5.7 Lessons Learned

### Technical Lessons

1. **CPU vs GPU tradeoffs** - GPU isn't always faster; overhead matters for small operations
2. **Cross-platform testing essential** - Windows-specific issues like DataLoader deadlocks
3. **Error feedback is critical** - Silent failures waste debugging time
4. **CSS architecture matters** - Design system prevents inconsistency

### Process Lessons

1. **Incremental commits** - Small, focused commits make debugging easier
2. **Branch isolation** - Feature branches prevent breaking main
3. **Documentation alongside code** - DESIGN_GUIDELINES.md prevents style drift
4. **User feedback loop** - "Learn from Selection" solved parameter tuning

### Architecture Lessons

1. **IPC abstracts complexity** - Clean separation between renderer and main
2. **Python for ML, JS for UI** - Play to each language's strengths
3. **State management patterns** - Zustand + Zundo simpler than Redux
4. **FastAPI over Flask** - Async support and better typing
