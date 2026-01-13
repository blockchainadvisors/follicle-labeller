# FOL File Format Specification (V2.0)

The `.fol` file is a ZIP archive containing medical image annotation data. Rename to `.zip` to inspect contents.

## Archive Structure

```
project.fol (ZIP archive)
├── manifest.json          # Project metadata and image list
├── annotations.json       # All shape annotations
└── images/                # Folder containing image files
    ├── {id1}-{filename1}  # Image file prefixed with its ID
    └── {id2}-{filename2}  # Additional images...
```

---

## manifest.json

Project metadata and image inventory.

```json
{
  "version": "2.0",
  "createdAt": 1704067200000,
  "images": [
    {
      "id": "img_1704067200000_abc123def",
      "fileName": "scan_001.png",
      "width": 1920,
      "height": 1080,
      "sortOrder": 0
    },
    {
      "id": "img_1704067300000_xyz789ghi",
      "fileName": "scan_002.jpg",
      "width": 2048,
      "height": 1536,
      "sortOrder": 1
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| version | string | Format version ("2.0") |
| createdAt | number | Unix timestamp (milliseconds) |
| images | array | List of images in project |
| images[].id | string | Unique image identifier |
| images[].fileName | string | Original filename |
| images[].width | number | Image width in pixels |
| images[].height | number | Image height in pixels |
| images[].sortOrder | number | Display order (0-indexed) |

---

## annotations.json

All shape annotations across all images.

```json
{
  "version": "2.0",
  "exportedAt": 1704067500000,
  "annotations": [
    {
      "id": "uuid-string",
      "imageId": "img_1704067200000_abc123def",
      "shape": "circle",
      "center": { "x": 500, "y": 300 },
      "radius": 45.5,
      "label": "Follicle 1",
      "notes": "Healthy follicle",
      "color": "#4ECDC4",
      "createdAt": 1704067400000,
      "updatedAt": 1704067400000
    },
    {
      "id": "uuid-string-2",
      "imageId": "img_1704067200000_abc123def",
      "shape": "rectangle",
      "x": 100,
      "y": 200,
      "width": 150,
      "height": 80,
      "label": "Region A",
      "notes": "",
      "color": "#FF6B6B",
      "createdAt": 1704067450000,
      "updatedAt": 1704067450000
    },
    {
      "id": "uuid-string-3",
      "imageId": "img_1704067300000_xyz789ghi",
      "shape": "linear",
      "startPoint": { "x": 200, "y": 150 },
      "endPoint": { "x": 400, "y": 250 },
      "halfWidth": 25,
      "label": "Linear 1",
      "notes": "Hair shaft",
      "color": "#FFEAA7",
      "createdAt": 1704067480000,
      "updatedAt": 1704067480000
    }
  ]
}
```

---

## Shape Types

### Circle

Circular annotation defined by center point and radius.

| Field | Type | Description |
|-------|------|-------------|
| shape | string | "circle" |
| center | object | { x, y } center coordinates (pixels) |
| radius | number | Radius in pixels |

### Rectangle

Axis-aligned rectangle defined by top-left corner and dimensions.

| Field | Type | Description |
|-------|------|-------------|
| shape | string | "rectangle" |
| x | number | Left edge X coordinate (pixels) |
| y | number | Top edge Y coordinate (pixels) |
| width | number | Width in pixels |
| height | number | Height in pixels |

### Linear

Rotated rectangle (oriented box) defined by centerline and half-width. Useful for elongated structures.

| Field | Type | Description |
|-------|------|-------------|
| shape | string | "linear" |
| startPoint | object | { x, y } line start (pixels) |
| endPoint | object | { x, y } line end (pixels) |
| halfWidth | number | Half-width perpendicular to line (pixels) |

**To calculate linear shape bounds:**
- Length = distance from startPoint to endPoint
- Total width = halfWidth × 2
- Angle = atan2(endPoint.y - startPoint.y, endPoint.x - startPoint.x)
- Center = midpoint of startPoint and endPoint

---

## Common Annotation Fields

| Field | Type | Description |
|-------|------|-------------|
| id | string | Unique annotation UUID |
| imageId | string | References manifest.images[].id |
| label | string | User-assigned label |
| notes | string | Optional notes/comments |
| color | string | Hex color code (e.g., "#4ECDC4") |
| createdAt | number | Creation timestamp (ms) |
| updatedAt | number | Last modified timestamp (ms) |

---

## Images Folder

Images are stored in the `images/` folder with filenames formatted as:

```
{imageId}-{originalFileName}
```

Example: `img_1704067200000_abc123def-scan_001.png`

This ensures unique filenames even if original names are duplicated.

---

## Coordinate System

- **Origin:** Top-left corner of image (0, 0)
- **X-axis:** Increases rightward
- **Y-axis:** Increases downward
- **Units:** Pixels (can be fractional)

---

## Quick Reference for LLMs

**To find annotations for a specific image:**
1. Get the image ID from `manifest.json` → `images[].id`
2. Filter `annotations.json` → `annotations` where `imageId` matches

**To extract actual image file:**
1. Find image in `manifest.json`
2. Locate file in `images/` folder: `{id}-{fileName}`

**To count annotations:**
- Total: `annotations.length`
- Per image: Filter by `imageId` and count
- By type: Filter by `shape` field (circle, rectangle, linear)

---

This format allows multi-image projects with annotations linked to specific images via the `imageId` field.
