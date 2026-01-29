# Chapter 2: Technical Architecture

## 2.1 System Overview

The Follicle Labeller follows a hybrid desktop application architecture, combining a web-based frontend with native desktop capabilities through Electron, and Python-based backend services for computer vision and machine learning tasks.

```
┌─────────────────────────────────────────────────────────────────┐
│                     Electron Main Process                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    main.ts                               │    │
│  │  - Window management                                     │    │
│  │  - Native menus                                          │    │
│  │  - File dialogs                                          │    │
│  │  - Python process management                             │    │
│  │  - Auto-updates                                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                         IPC Bridge                               │
│                         (preload.ts)                             │
│                              │                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                 Renderer Process                         │    │
│  │                 (React Application)                      │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │    │
│  │  │   Canvas    │  │   Toolbar   │  │   Panels    │      │    │
│  │  │  Renderer   │  │             │  │  (Property, │      │    │
│  │  │             │  │             │  │   Explorer) │      │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘      │    │
│  │                                                          │    │
│  │  ┌───────────────────────────────────────────────────┐  │    │
│  │  │               Zustand State Stores                │  │    │
│  │  │  - projectStore (images, viewports)               │  │    │
│  │  │  - follicleStore (annotations, undo/redo)         │  │    │
│  │  │  - canvasStore (mode, display options)            │  │    │
│  │  │  - themeStore (colors, background)                │  │    │
│  │  └───────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                         HTTP/REST
                              │
┌─────────────────────────────────────────────────────────────────┐
│                    Python Backend Services                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  blob_server.py │  │  yolo_keypoint  │  │  yolo_detection │  │
│  │    (FastAPI)    │  │   _service.py   │  │   _service.py   │  │
│  │                 │  │                 │  │                 │  │
│  │  - BLOB detect  │  │  - Keypoint     │  │  - Detection    │  │
│  │  - GPU backend  │  │    training     │  │    training     │  │
│  │  - Learned      │  │  - Inference    │  │  - Tiled        │  │
│  │    detection    │  │                 │  │    inference    │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    GPU Backends                           │   │
│  │  cpu_backend.py │ cupy_backend.py │ torch_backend.py     │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## 2.2 Frontend Architecture

### 2.2.1 Component Hierarchy

```
App
├── Toolbar
│   ├── File buttons (New, Open, Save)
│   ├── Mode selection (Create, Select, Pan)
│   ├── Shape selection (Circle, Rectangle, Linear)
│   ├── Detection tools (Auto Detect, Learn, YOLO)
│   └── View controls (Zoom, Fit, Theme)
├── ImageExplorer
│   └── Image list with thumbnails
├── ImageCanvas
│   ├── CanvasRenderer (shape rendering)
│   └── HeatmapOverlay (density visualization)
├── PropertyPanel
│   ├── Annotation properties editor
│   ├── Selection filter
│   └── Statistics display
└── Dialogs
    ├── DetectionSettingsDialog
    ├── LearnedDetectionDialog
    ├── FollicleOriginDialog
    ├── YOLOTrainingDialog
    ├── YOLODetectionTrainingDialog
    ├── ImportAnnotationsDialog
    └── DuplicateDetectionDialog
```

### 2.2.2 State Management

The application uses Zustand for state management with Zundo for undo/redo capability.

```typescript
// Store Structure
interface ProjectStore {
  images: ImageFile[];
  activeImageId: string | null;
  viewports: Map<string, Viewport>;
  projectPath: string | null;
  isDirty: boolean;
}

interface FollicleStore {
  annotations: Map<string, Follicle[]>;
  selectedIds: Set<string>;
  // Zundo provides: undo(), redo(), clear()
}

interface CanvasStore {
  mode: 'create' | 'select' | 'pan';
  shapeType: 'circle' | 'rectangle' | 'linear';
  showLabels: boolean;
  showShapes: boolean;
  showHeatmap: boolean;
}
```

### 2.2.3 Canvas Rendering Pipeline

```
Image Loading
    │
    ▼
ImageBitmap Creation (pre-decoded for smooth rendering)
    │
    ▼
Canvas Context Setup (2D context with transforms)
    │
    ▼
Render Loop
├── Clear canvas
├── Apply viewport transform (pan, zoom)
├── Draw image
├── Draw annotations
│   ├── Circles (arc + fill)
│   ├── Rectangles (rect + fill)
│   └── Linear shapes (transformed rect)
├── Draw selection highlights
├── Draw creation preview
└── Draw heatmap overlay (if enabled)
```

## 2.3 Backend Architecture

### 2.3.1 Python Services Overview

| Service | Port | Framework | Purpose |
|---------|------|-----------|---------|
| blob_server | 5001 | FastAPI | BLOB detection, GPU acceleration |
| yolo_keypoint_service | 5002 | FastAPI | Keypoint model training/inference |
| yolo_detection_service | 5003 | FastAPI | Detection model training/inference |

### 2.3.2 BLOB Detection Pipeline

```
Input Image
    │
    ▼
Preprocessing
├── Grayscale conversion
├── CLAHE (Contrast Limited Adaptive Histogram Equalization)
├── Gaussian blur
└── Morphological opening (optional)
    │
    ▼
Detection Methods
├── SimpleBlobDetector (primary)
│   ├── Thresholding (minThreshold → maxThreshold)
│   ├── Area filtering (minArea, maxArea)
│   ├── Circularity filtering
│   └── Convexity filtering
│
└── Contour Fallback (if no blobs found)
    ├── Otsu thresholding
    ├── Contour detection
    └── Ellipse fitting
    │
    ▼
Post-processing
├── Size filtering (minWidth, maxWidth)
├── Soft-NMS (overlap removal)
└── Rectangle conversion
    │
    ▼
Output: List of detected follicles
```

### 2.3.3 GPU Backend Selection

```
System Initialization
    │
    ▼
Hardware Detection (gpu_hardware_detect.py)
├── Check NVIDIA GPU → nvidia-smi
└── Check Apple Silicon → platform.processor()
    │
    ▼
Backend Selection
├── NVIDIA detected? → CuPy backend
│   ├── Check CUDA toolkit
│   ├── Install cupy-cuda12x
│   └── Enable CUDA acceleration
│
├── Apple Silicon? → PyTorch Metal backend
│   ├── Install torch (MPS support)
│   └── Enable Metal acceleration
│
└── Neither? → CPU backend (OpenCV only)
```

## 2.4 Data Flow

### 2.4.1 Annotation Creation Flow

```
User Input (mouse/keyboard)
    │
    ▼
ImageCanvas (event handling)
    │
    ▼
FollicleStore.addFollicle()
    │
    ▼
Zundo (temporal history)
    │
    ▼
State Update
    │
    ▼
React Re-render
    │
    ▼
Canvas Redraw
```

### 2.4.2 Detection Flow

```
User clicks "Auto Detect"
    │
    ▼
DetectionSettingsDialog (parameters)
    │
    ▼
blobService.detect() / yoloDetectionService.detect()
    │
    ▼
IPC → Electron Main Process
    │
    ▼
HTTP POST → Python Service
    │
    ▼
Image Processing (CPU/GPU)
    │
    ▼
Detection Results (JSON)
    │
    ▼
Duplicate Detection Check
    │
    ▼
DuplicateDetectionDialog (if duplicates)
    │
    ▼
FollicleStore.addFollicles()
    │
    ▼
Canvas Update
```

## 2.5 File Format

### 2.5.1 FOL Archive Structure (V2.0)

```
project.fol (ZIP archive)
├── manifest.json
│   {
│     "version": 2,
│     "name": "Project Name",
│     "images": [
│       { "id": "uuid", "filename": "image1.jpg" }
│     ],
│     "created": "ISO timestamp",
│     "modified": "ISO timestamp"
│   }
│
├── annotations.json
│   {
│     "version": 2,
│     "annotations": {
│       "image-uuid": [
│         { "id": "uuid", "type": "circle", "cx": 100, "cy": 100, "r": 25 },
│         { "id": "uuid", "type": "rectangle", "x": 50, "y": 50, "width": 30, "height": 40 },
│         { "id": "uuid", "type": "linear", "cx": 100, "cy": 100, "angle": 45, "length": 50, "width": 10 }
│       ]
│     }
│   }
│
└── images/
    ├── {id}-image1.jpg
    └── {id}-image2.png
```

## 2.6 IPC Communication

### 2.6.1 Electron IPC Channels

| Channel | Direction | Purpose |
|---------|-----------|---------|
| `file:open` | Renderer → Main | Open file dialog |
| `file:save` | Renderer → Main | Save file dialog |
| `blob:start` | Renderer → Main | Start BLOB detection server |
| `blob:stop` | Renderer → Main | Stop BLOB detection server |
| `gpu:detect-hardware` | Renderer → Main | Check GPU availability |
| `gpu:install-packages` | Renderer → Main | Install GPU packages |
| `yolo:train` | Renderer → Main | Start YOLO training (SSE) |
| `yolo:detect` | Renderer → Main | Run YOLO inference |
| `menu:event` | Main → Renderer | Menu item clicked |
| `update:available` | Main → Renderer | Update notification |

## 2.7 Technology Decisions

### 2.7.1 Why Electron?

- Cross-platform desktop support
- Native file system access
- Process management for Python services
- Native menus and dialogs
- Auto-update capabilities

### 2.7.2 Why Python for Backend?

- Rich computer vision ecosystem (OpenCV, Ultralytics)
- GPU acceleration libraries (CuPy, PyTorch)
- Easy integration with ML frameworks
- FastAPI for async HTTP services

### 2.7.3 Why Zustand over Redux?

- Simpler API, less boilerplate
- Better TypeScript support
- Easy undo/redo with Zundo middleware
- Smaller bundle size
