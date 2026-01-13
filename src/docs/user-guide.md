# Follicle Labeller User Guide

Welcome to Follicle Labeller, a medical image annotation tool for labeling follicles and other structures in medical images.

---

## Getting Started

### Opening an Image

1. Click the **Open Image** button in the toolbar
2. Select an image file (PNG, JPG, TIFF, BMP, or WebP)
3. The image will appear in the canvas area and automatically fit to view

**Tip:** You can add multiple images to a project. Each image maintains its own annotations and zoom/pan state.

---

## Creating Annotations

First, make sure you're in **Create** mode by clicking the Create button or pressing `C`.

### Circle Annotations

Best for round structures like follicles.

1. Select **Circle** shape (or press `1`)
2. Click on the center of the follicle
3. Move the mouse outward to preview the radius
4. Click again to create the annotation

### Rectangle Annotations

Best for rectangular regions or bounding boxes.

1. Select **Rectangle** shape (or press `2`)
2. Click on one corner of the area
3. Move the mouse to the opposite corner
4. Click again to create the annotation

### Linear Annotations

Best for elongated structures. Creates a rotated rectangle defined by its centerline.

1. Select **Linear** shape (or press `3`)
2. Click to set the start of the centerline
3. Move the mouse and click to set the end of the centerline
4. Move the mouse perpendicular to set the width
5. Click to finalize the annotation

**Tip:** Press `Escape` at any time to cancel shape creation.

---

## Editing Annotations

### Selecting an Annotation

1. Switch to **Select** mode (click Select or press `V`)
2. Click on any annotation to select it
3. The selected annotation shows resize handles

### Moving Annotations

- In Select mode, click and drag a selected annotation to move it
- The annotation will follow your mouse

### Resizing Annotations

Each shape type has different resize handles:

- **Circle:** Drag the edge handle to change the radius
- **Rectangle:** Drag any corner handle to resize
- **Linear:** Drag start/end points to change length, or drag the width handle

### Editing Properties

With an annotation selected, use the **Property Panel** on the right to:

- **Label:** Give the annotation a descriptive name
- **Notes:** Add additional information
- **Color:** Change the annotation color

### Deleting Annotations

- Select an annotation and press `Delete` or `Backspace`
- Or click the **Delete** button in the Property Panel

---

## Navigation

### Zooming

- **Mouse wheel:** Scroll to zoom in/out at the cursor position
- **Toolbar:** Use the `-` and `+` buttons
- **Reset:** Click Reset to fit the image to the view

Zoom range: 10% to 1000%

### Panning

- **Pan mode:** Press `H` or click Pan, then drag to move around
- **Middle mouse button:** Drag with the middle button in any mode

---

## Saving and Loading Projects

### Saving a Project

1. Click **Save** in the toolbar
2. Choose a location and filename
3. Projects are saved as `.fol` files

The `.fol` format preserves:
- All images in the project
- All annotations with their properties
- Zoom and pan state for each image

### Loading a Project

1. Click **Load** in the toolbar
2. Select a `.fol` file
3. All images and annotations will be restored

---

## Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Create mode | `C` |
| Select mode | `V` |
| Pan mode | `H` |
| Circle shape | `1` |
| Rectangle shape | `2` |
| Linear shape | `3` |
| Toggle shapes | `O` |
| Toggle labels | `L` |
| Undo | `Ctrl+Z` |
| Redo | `Ctrl+Shift+Z` |
| Delete annotation | `Delete` / `Backspace` |
| Cancel / Deselect | `Escape` |
| Toggle help | `?` |
| Next image | `Ctrl+Tab` |
| Previous image | `Ctrl+Shift+Tab` |

---

## Tips and Best Practices

- **Use keyboard shortcuts** for faster workflow - `C` for create, `V` for select, `1-3` for shapes
- **Middle-click and drag** to pan while in any mode
- **Zoom to details** with the mouse wheel before creating precise annotations
- **Add notes** to annotations for additional context
- **Save frequently** to avoid losing work
- **Use the Screenshot button** to capture the current canvas view

---

## Toolbar Overview

From left to right:

1. **File Operations:** Open Image, Load, Save
2. **Modes:** Create, Select, Pan
3. **Shapes:** Circle, Rectangle, Linear
4. **Zoom:** -, percentage, +, Reset
5. **History:** Undo, Redo
6. **View:** Shapes toggle, Labels toggle
7. **Clear All:** Remove all annotations (caution!)
8. **Screenshot:** Save current canvas view
9. **Help:** Open this guide

---

## Need More Help?

For additional support or to report issues, please visit the project repository.
