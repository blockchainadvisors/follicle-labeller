# Chapter 6: Technologies and Tools

This chapter provides detailed documentation of all technologies, frameworks, libraries, and tools used in the project.

## 6.1 Frontend Technologies

### 6.1.1 React 18

**Role:** UI framework for building the user interface

**Key Features Used:**
- Functional components with hooks
- useState, useEffect, useCallback, useMemo
- useRef for canvas element access
- Context API (minimal usage, prefer Zustand)

**Why React:**
- Component-based architecture
- Large ecosystem
- Excellent TypeScript support
- Team familiarity

### 6.1.2 TypeScript 5.3

**Role:** Static type safety for JavaScript

**Configuration:**
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "strict": true,
    "jsx": "react-jsx",
    "moduleResolution": "bundler"
  }
}
```

**Benefits:**
- Catch errors at compile time
- Better IDE support (autocomplete, refactoring)
- Self-documenting interfaces
- Safer refactoring

### 6.1.3 Zustand 4.4

**Role:** Lightweight state management

**Stores Implemented:**

| Store | Purpose | Features |
|-------|---------|----------|
| projectStore | Project state | Images, viewports, paths |
| follicleStore | Annotations | Undo/redo via Zundo |
| canvasStore | UI state | Mode, display options |
| themeStore | Theming | Colors, backgrounds |

**Example:**
```typescript
import { create } from 'zustand';
import { temporal } from 'zundo';

const useFollicleStore = create(
  temporal(
    (set) => ({
      annotations: new Map(),
      addFollicle: (imageId, follicle) => set((state) => ({
        annotations: new Map(state.annotations).set(imageId, [
          ...(state.annotations.get(imageId) || []),
          follicle
        ])
      }))
    }),
    { limit: 50 }  // 50-level undo history
  )
);
```

### 6.1.4 Zundo 2.0

**Role:** Undo/redo middleware for Zustand

**Features:**
- Automatic history tracking
- Configurable history depth
- undo(), redo(), clear() methods
- Selective state tracking

### 6.1.5 Lucide React 0.469

**Role:** Icon library

**Icons Used:**
- MousePointer2, Hand, Circle, Square, Minus (tools)
- FolderOpen, Save, Download, Upload (file operations)
- ZoomIn, ZoomOut, Maximize (view controls)
- Sparkles, Brain (AI/detection)
- Plus, Trash2, ChevronDown (actions)

### 6.1.6 JSZip 3.10

**Role:** ZIP archive creation/extraction

**Usage:**
- Creating .fol project archives
- Reading saved projects
- Packaging images with annotations

### 6.1.7 HTML5 Canvas 2D API

**Role:** Image rendering and annotation visualization

**Key Techniques:**
- ImageBitmap for pre-decoded smooth rendering
- Transform matrix for pan/zoom
- Path2D for shape drawing
- requestAnimationFrame for smooth updates

---

## 6.2 Desktop Framework

### 6.2.1 Electron 28

**Role:** Cross-platform desktop application framework

**Architecture:**
```
┌─────────────────────────────────────────┐
│           Main Process                  │
│  - Window management                    │
│  - Native menus                         │
│  - File dialogs                         │
│  - Python process spawning              │
│  - IPC handling                         │
└─────────────────────────────────────────┘
                    │
              IPC Bridge
                    │
┌─────────────────────────────────────────┐
│         Renderer Process                │
│  - React application                    │
│  - Canvas rendering                     │
│  - State management                     │
└─────────────────────────────────────────┘
```

**IPC Communication:**
```typescript
// preload.ts - Bridge between main and renderer
contextBridge.exposeInMainWorld('electron', {
  ipcRenderer: {
    invoke: (channel: string, ...args: any[]) =>
      ipcRenderer.invoke(channel, ...args),
    on: (channel: string, listener: Function) =>
      ipcRenderer.on(channel, (_, ...args) => listener(...args))
  }
});
```

### 6.2.2 Electron Builder 24.9

**Role:** Application packaging and distribution

**Configuration:**
```json
{
  "appId": "com.blockchainadvisors.follicle-labeller",
  "productName": "Follicle Labeller",
  "directories": {
    "buildResources": "assets",
    "output": "dist"
  },
  "files": [
    "dist/**/*",
    "electron/**/*",
    "package.json"
  ],
  "win": {
    "target": "nsis",
    "icon": "assets/icon.ico"
  },
  "mac": {
    "target": "dmg",
    "icon": "assets/icon.icns",
    "hardenedRuntime": true,
    "gatekeeperAssess": false
  },
  "linux": {
    "target": "AppImage",
    "icon": "assets/icon.png"
  }
}
```

### 6.2.3 Vite 5

**Role:** Build tool and development server

**Benefits:**
- Fast HMR (Hot Module Replacement)
- ES modules for development
- Rollup for production builds
- TypeScript support out of box

**Configuration:**
```typescript
// vite.config.ts
export default defineConfig({
  plugins: [react()],
  base: './',
  build: {
    outDir: 'dist',
    emptyOutDir: true
  }
});
```

---

## 6.3 Python Backend

### 6.3.1 FastAPI 0.109

**Role:** Async web framework for ML services

**Why FastAPI:**
- Async support (important for streaming)
- Automatic OpenAPI documentation
- Type hints with Pydantic
- Better than Flask for modern Python

**Example Endpoint:**
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class DetectionRequest(BaseModel):
    image_path: str
    min_width: int = 10
    max_width: int = 200

@app.post("/detect")
async def detect(request: DetectionRequest):
    results = blob_detector.detect(request.image_path, request.dict())
    return {"follicles": results}
```

### 6.3.2 Uvicorn 0.27

**Role:** ASGI server for FastAPI

**Usage:**
```python
import uvicorn

if __name__ == "__main__":
    uvicorn.run("blob_server:app", host="127.0.0.1", port=5001)
```

### 6.3.3 OpenCV (opencv-python 4.8)

**Role:** Computer vision operations

**Functions Used:**
- `cv2.SimpleBlobDetector` - Primary blob detection
- `cv2.cvtColor` - Color space conversion
- `cv2.createCLAHE` - Contrast enhancement
- `cv2.GaussianBlur` - Noise reduction
- `cv2.morphologyEx` - Morphological operations
- `cv2.findContours` - Contour detection
- `cv2.fitEllipse` - Ellipse fitting

### 6.3.4 Ultralytics 8.3

**Role:** YOLO model training and inference

**Features Used:**
- YOLO11 detection model
- YOLO11 keypoint model
- Training API with callbacks
- ONNX export

**Training Example:**
```python
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
results = model.train(
    data='dataset/data.yaml',
    epochs=100,
    imgsz=640,
    device='cuda'
)
```

### 6.3.5 NumPy & Pillow

**Role:** Array manipulation and image I/O

**NumPy Usage:**
- Image as numpy array
- Coordinate calculations
- Statistical operations

**Pillow Usage:**
- Image loading/saving
- Format conversion
- Metadata handling

### 6.3.6 ONNX 1.15

**Role:** Model export format

**Benefits:**
- Framework-independent models
- Optimized inference
- Cross-platform deployment

---

## 6.4 GPU Acceleration Libraries

### 6.4.1 CuPy (NVIDIA CUDA)

**Role:** GPU array operations for NVIDIA GPUs

**Installation:**
```bash
pip install cupy-cuda12x  # For CUDA 12.x
```

**Usage:**
```python
import cupy as cp

# Transfer to GPU
gpu_image = cp.asarray(cpu_image)

# GPU operations
gpu_gray = cp.mean(gpu_image, axis=2)

# Transfer back
result = cp.asnumpy(gpu_gray)
```

### 6.4.2 PyTorch (Apple Metal/MPS)

**Role:** GPU operations on Apple Silicon

**MPS Support:**
```python
import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
    tensor = torch.tensor(image).to(device)
```

### 6.4.3 Kornia

**Role:** Computer vision operations on GPU via PyTorch

**Usage:**
```python
import kornia

# CLAHE on GPU
enhanced = kornia.enhance.equalize_clahe(tensor)
```

---

## 6.5 Development Tools

### 6.5.1 Git & GitHub

**Branching Strategy:**
```
main (production)
├── developer (integration)
│   ├── feature/follicle-origin-annotation
│   ├── feature/yolo-model-training
│   ├── fix/styling
│   └── bugfix/cuda-detection-production
└── lorenzo-branch (development)
```

**GitHub Features Used:**
- Pull requests for code review
- GitHub Actions for CI/CD
- Release management with tags
- Issue tracking

### 6.5.2 GitHub Actions

**CI/CD Pipeline:**
```yaml
name: Build and Release

on:
  push:
    tags: ['v*']

jobs:
  build:
    strategy:
      matrix:
        os: [windows-latest, macos-latest, ubuntu-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
      - run: npm ci
      - run: npm run build
      - run: npm run electron:build
      - uses: softprops/action-gh-release@v1
```

### 6.5.3 Claude Code CLI

**Role:** AI-assisted development

**Usage:**
- Code generation
- Bug fixing
- Architecture planning
- Documentation

**Sessions Tracked:** 46 development sessions

### 6.5.4 VS Code

**Extensions Used:**
- ESLint
- Prettier
- TypeScript
- Python
- GitLens

---

## 6.6 Testing Tools

### 6.6.1 Benchmark System

**Location:** `test/blob-detection-benchmark/`

**Purpose:**
- Compare CPU vs GPU performance
- Measure detection accuracy
- Track regression

**Usage:**
```bash
npm run test:benchmark
```

---

## 6.7 Dependencies Summary

### 6.7.1 package.json Dependencies

```json
{
  "dependencies": {
    "electron-updater": "^6.1.7",
    "jszip": "^3.10.1",
    "lucide-react": "^0.469.0",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "zustand": "^4.4.7",
    "zundo": "^2.0.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.45",
    "@vitejs/plugin-react": "^4.2.1",
    "electron": "^28.0.0",
    "electron-builder": "^24.9.1",
    "typescript": "^5.3.3",
    "vite": "^5.0.10"
  }
}
```

### 6.7.2 Python requirements.txt

```
fastapi>=0.109.0
uvicorn>=0.27.0
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0
ultralytics>=8.3.0
onnx>=1.15.0
# Optional GPU
cupy-cuda12x  # NVIDIA
torch  # Apple Metal
kornia  # GPU image processing
```

---

## 6.8 Technology Evaluation

### 6.8.1 Technologies That Worked Well

| Technology | Rating | Notes |
|------------|--------|-------|
| React + TypeScript | Excellent | Strong typing, good DX |
| Zustand | Excellent | Simple, powerful, undo support |
| Electron | Good | Cross-platform, some overhead |
| FastAPI | Excellent | Modern async Python |
| OpenCV | Excellent | Comprehensive CV library |
| YOLO (Ultralytics) | Good | Easy training, good accuracy |

### 6.8.2 Technologies with Challenges

| Technology | Issue | Workaround |
|------------|-------|------------|
| CuPy | High overhead for small ops | Use CPU for preprocessing |
| Electron | Large bundle size | Accept tradeoff for features |
| opencv-python | No TypeScript types | Use Python backend |

### 6.8.3 Technologies Considered but Not Used

| Technology | Reason Not Used |
|------------|-----------------|
| Redux | Zustand simpler for our needs |
| Flask | FastAPI better async support |
| TensorFlow | YOLO/Ultralytics easier |
| Tauri | Electron more mature |
| SAM | BLOB detection sufficient |
