# Chapter 4: Features Implemented

This chapter provides detailed documentation of all features implemented during the placement period.

## 4.1 Core Annotation Features

### 4.1.1 Annotation Shapes

The application supports three distinct annotation shape types, each optimized for different follicle morphologies.

#### Circle
- **Use Case:** Round, well-defined follicles
- **Parameters:** Center point (cx, cy), Radius (r)
- **Creation:** Click to set center, drag to set radius
- **Display:** Filled circle with optional label

```typescript
interface CircleFollicle {
  id: string;
  type: 'circle';
  cx: number;
  cy: number;
  r: number;
  color?: string;
  origin?: { x: number; y: number; angle: number };
}
```

#### Rectangle
- **Use Case:** Bounding box annotations, non-circular follicles
- **Parameters:** Top-left corner (x, y), Width, Height
- **Creation:** Click-drag from corner to corner
- **Display:** Filled rectangle with optional label

```typescript
interface RectangleFollicle {
  id: string;
  type: 'rectangle';
  x: number;
  y: number;
  width: number;
  height: number;
  color?: string;
  origin?: { x: number; y: number; angle: number };
}
```

#### Linear (Oriented Box)
- **Use Case:** Hair follicles with clear orientation
- **Parameters:** Center (cx, cy), Length, Width, Angle
- **Creation:** Click to set start, drag to set end, then drag to set width
- **Display:** Rotated rectangle with centerline indicator

```typescript
interface LinearFollicle {
  id: string;
  type: 'linear';
  cx: number;
  cy: number;
  length: number;
  width: number;
  angle: number;  // radians
  color?: string;
  origin?: { x: number; y: number; angle: number };
}
```

### 4.1.2 Follicle Origin Annotation

A unique feature that marks the entry point and growth direction of each follicle.

**Purpose:**
- Track where follicles emerge from the skin
- Predict growth direction for analysis
- Enable keypoint-based ML detection

**Data Structure:**
```typescript
interface FollicleOrigin {
  x: number;      // Origin point x-coordinate
  y: number;      // Origin point y-coordinate
  angle: number;  // Growth direction in radians
}
```

**UI Flow:**
1. Select follicle(s) to annotate
2. Open Follicle Origin dialog
3. Click on follicle edge to mark origin
4. System automatically calculates direction vector

### 4.1.3 Selection Tools

#### Single Selection
- Click on annotation to select
- Shift+click for additive selection

#### Marquee Selection
- Click and drag to create rectangular selection area
- All annotations within the box are selected

#### Lasso Selection
- Freeform polygon selection
- Click to add points, close to complete

#### Selection Filtering
- Filter by shape type (Circle, Rectangle, Linear)
- Filter by origin status (With Origin, Without Origin)
- Filter by selection state

### 4.1.4 Undo/Redo System

Implemented using Zundo middleware for Zustand:

- **History Depth:** 50 levels
- **Tracked Actions:** Add, delete, modify annotations
- **Keyboard Shortcuts:** Ctrl+Z (Undo), Ctrl+Shift+Z (Redo)

---

## 4.2 Detection Algorithms

### 4.2.1 BLOB Detection (OpenCV-based)

A computer vision approach using OpenCV's SimpleBlobDetector.

**Pipeline:**
```
Image → Grayscale → CLAHE → Gaussian Blur → Morphological Opening → BLOB Detection → NMS → Results
```

**Configurable Parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| minThreshold | Start of threshold range | 10 |
| maxThreshold | End of threshold range | 200 |
| thresholdStep | Increment between thresholds | 10 |
| minWidth | Minimum blob width (px) | 10 |
| maxWidth | Maximum blob width (px) | 200 |
| minAspectRatio | Minimum width/height ratio | 0.3 |
| maxAspectRatio | Maximum width/height ratio | 3.0 |
| filterByCircularity | Enable circularity filter | true |
| minCircularity | Minimum circularity (0-1) | 0.3 |
| filterByConvexity | Enable convexity filter | true |
| minConvexity | Minimum convexity (0-1) | 0.7 |

**Preprocessing Options:**

| Option | Description |
|--------|-------------|
| CLAHE | Contrast Limited Adaptive Histogram Equalization |
| clipLimit | CLAHE contrast limit (1-4) |
| tileGridSize | CLAHE tile grid size |
| Gaussian Blur | Noise reduction |
| kernelSize | Blur kernel size (odd numbers) |
| Morphological Opening | Remove small noise |
| morphKernelSize | Morphology kernel size |
| morphIterations | Number of iterations |

### 4.2.2 Learned Detection

Learns detection parameters from user-selected examples.

**How It Works:**
1. User selects 3+ manually annotated follicles
2. System calculates statistics:
   - Mean/min/max width and height
   - Mean intensity (for dark/light detection)
   - Aspect ratio range
3. Parameters automatically adjusted with tolerance
4. Detection runs with learned parameters

**Benefits:**
- Adapts to specific image characteristics
- No manual parameter tuning required
- Handles varying follicle sizes

### 4.2.3 YOLO11 Detection

Deep learning-based object detection using Ultralytics YOLO11.

**Training Features:**
- Custom dataset generation from annotations
- Configurable epochs, batch size, image size
- GPU acceleration (CUDA/MPS)
- Resume training from checkpoint
- Real-time training progress (loss, mAP metrics)

**Training Configuration:**
```yaml
epochs: 100          # Training iterations
batch: 16            # Batch size
imgsz: 640           # Image size
device: cuda/mps/cpu # Compute device
patience: 50         # Early stopping patience
```

**Inference Features:**
- Tiled inference for large images (configurable overlap)
- Confidence threshold adjustment
- NMS for overlapping detections
- Batch processing support

### 4.2.4 YOLO Keypoint Detection

Predicts follicle entry point and growth direction.

**Output:**
- Follicle bounding box
- Origin keypoint (x, y)
- Direction vector (angle)

**Use Cases:**
- Automated origin annotation
- Growth pattern analysis
- Direction-based filtering

---

## 4.3 GPU Acceleration

### 4.3.1 NVIDIA CUDA Support

**Requirements:**
- NVIDIA GPU with CUDA support
- CUDA Toolkit (auto-detected)
- cupy-cuda12x package

**Supported Operations:**
- Image preprocessing (grayscale, CLAHE)
- Gaussian blur
- Morphological operations
- YOLO training and inference

**Installation:**
```
Detection Settings → GPU Acceleration → Install GPU Packages
```

### 4.3.2 Apple Metal Support

**Requirements:**
- Apple Silicon (M1/M2/M3) or AMD GPU
- macOS 12.0+
- PyTorch with MPS support

**Supported Operations:**
- Image preprocessing via Kornia
- YOLO training with MPS device

### 4.3.3 CPU Fallback

Automatic fallback to CPU when:
- No compatible GPU detected
- GPU packages not installed
- GPU initialization fails

---

## 4.4 Multi-Image Project Support

### 4.4.1 Project Structure

- Multiple images per project
- Per-image viewport (pan, zoom)
- Per-image annotation storage
- Single `.fol` archive file

### 4.4.2 Image Explorer

- Thumbnail list of all images
- Click to switch active image
- Annotation count per image
- Delete image with confirmation

### 4.4.3 Project Persistence

- Auto-save on system sleep/suspend
- Unsaved changes warning on close
- Recent projects tracking (via Electron)

---

## 4.5 Import/Export Capabilities

### 4.5.1 FOL Project Format

**Version 2.0 Features:**
- ZIP-based archive
- Embedded images
- JSON manifest and annotations
- Cross-platform compatibility

### 4.5.2 JSON Export/Import

**Export:**
- All annotations for current image
- Selected annotations only
- Full coordinate data

**Import:**
- From JSON file
- Preview before import
- Duplicate detection
- Position offset option

### 4.5.3 YOLO Dataset Export

**Detection Format:**
```
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

**Keypoint Format:**
- Same structure with keypoint annotations
- Origin point and direction included

### 4.5.4 CSV Export

Tabular format for spreadsheet analysis:
```csv
id,type,x,y,width,height,angle,origin_x,origin_y,origin_angle
```

### 4.5.5 ONNX Model Export

Export trained YOLO models for external use.

---

## 4.6 UI/UX Features

### 4.6.1 Theme System

**Backgrounds (8 options):**
- Default Dark, Charcoal, Midnight, Deep Purple
- Light Gray, Paper, Cream, Cool Gray

**Accent Colors (23 options):**
- Blues, Greens, Purples, Oranges, Pinks
- Each with semantic CSS variables

### 4.6.2 Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| C | Create mode |
| V | Select mode |
| H | Pan mode |
| 1-3 | Switch shape type |
| L | Toggle labels |
| O | Toggle shapes |
| Delete | Delete selected |
| Ctrl+Z | Undo |
| Ctrl+Shift+Z | Redo |
| Ctrl+A | Select all |
| ? | Toggle help panel |

### 4.6.3 Canvas Controls

- Mouse wheel zoom (centered on cursor)
- Middle mouse button pan (any mode)
- Right-click context menu
- Fit to view button
- Zoom percentage display

### 4.6.4 Statistics Panel

Real-time display of:
- Total annotation count
- Count by shape type
- Selection count
- Origin annotation percentage

### 4.6.5 Heatmap Overlay

Density visualization showing:
- Annotation concentration areas
- Color gradient (cool to hot)
- Toggle on/off

---

## 4.7 Platform Features

### 4.7.1 Auto-Updates

- Check for updates on startup
- Manual check via menu
- Download progress display
- Restart prompt after download
- Platform-specific installers

### 4.7.2 File Association

- `.fol` files open in Follicle Labeller
- Icon displayed in file explorer
- Double-click to open projects

### 4.7.3 Native Menus

Full native menu bar with:
- File (New, Open, Save, Export)
- Edit (Undo, Redo, Select All)
- View (Zoom, Fit, Theme)
- Help (About, Check Updates)

### 4.7.4 Cross-Platform Builds

| Platform | Format | Notes |
|----------|--------|-------|
| Windows | NSIS Installer | Signed (optional) |
| macOS | DMG | Universal (Intel + Apple Silicon), Notarized |
| Linux | AppImage | Portable |
