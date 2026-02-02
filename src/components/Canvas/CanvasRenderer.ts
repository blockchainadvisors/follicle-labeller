import { Follicle, Viewport, DragState, ShapeType, isCircle, isRectangle, isLinear, CircleAnnotation, RectangleAnnotation, LinearAnnotation, Point, FollicleOrigin } from '../../types';

// Helper function to calculate distance from point to line segment
function pointToLineDistance(point: Point, lineStart: Point, lineEnd: Point): number {
  const dx = lineEnd.x - lineStart.x;
  const dy = lineEnd.y - lineStart.y;
  const lineLengthSquared = dx * dx + dy * dy;

  if (lineLengthSquared === 0) {
    return Math.sqrt((point.x - lineStart.x) ** 2 + (point.y - lineStart.y) ** 2);
  }

  // Project point onto line (unclamped)
  const t = ((point.x - lineStart.x) * dx + (point.y - lineStart.y) * dy) / lineLengthSquared;

  // Find closest point on line (not segment)
  const closestX = lineStart.x + t * dx;
  const closestY = lineStart.y + t * dy;

  return Math.sqrt((point.x - closestX) ** 2 + (point.y - closestY) ** 2);
}

// Helper to get the 4 corners of a linear (rotated rectangle)
function getLinearCorners(startPoint: Point, endPoint: Point, halfWidth: number): Point[] {
  const dx = endPoint.x - startPoint.x;
  const dy = endPoint.y - startPoint.y;
  const length = Math.sqrt(dx * dx + dy * dy);

  if (length === 0) return [startPoint, startPoint, startPoint, startPoint];

  // Perpendicular unit vector
  const perpX = -dy / length;
  const perpY = dx / length;

  // Calculate 4 corners
  return [
    { x: startPoint.x + perpX * halfWidth, y: startPoint.y + perpY * halfWidth },
    { x: startPoint.x - perpX * halfWidth, y: startPoint.y - perpY * halfWidth },
    { x: endPoint.x - perpX * halfWidth, y: endPoint.y - perpY * halfWidth },
    { x: endPoint.x + perpX * halfWidth, y: endPoint.y + perpY * halfWidth },
  ];
}

export class CanvasRenderer {
  private ctx: CanvasRenderingContext2D;
  private image: ImageBitmap | HTMLImageElement | null = null;

  constructor(ctx: CanvasRenderingContext2D) {
    this.ctx = ctx;
  }

  setImage(image: ImageBitmap | HTMLImageElement | null): void {
    this.image = image;
  }

  clear(width: number, height: number): void {
    // Read canvas background color from CSS variable (set by theme)
    const bgCanvas = getComputedStyle(document.documentElement).getPropertyValue('--bg-canvas').trim() || '#1a1a2e';
    this.ctx.fillStyle = bgCanvas;
    this.ctx.fillRect(0, 0, width, height);
  }

  render(
    canvasWidth: number,
    canvasHeight: number,
    viewport: Viewport,
    follicles: Follicle[],
    selectedIds: Set<string>,
    dragState: DragState,
    showLabels: boolean = true,
    showShapes: boolean = true,
    currentShapeType: ShapeType = 'circle',
    originPunchDiameter: number = 30
  ): void {
    this.clear(canvasWidth, canvasHeight);

    this.ctx.save();
    this.ctx.translate(viewport.offsetX, viewport.offsetY);
    this.ctx.scale(viewport.scale, viewport.scale);

    // Draw image
    if (this.image) {
      this.ctx.drawImage(this.image, 0, 0);
    }

    // Calculate visible bounds in image coordinates for culling
    const visibleBounds = {
      left: -viewport.offsetX / viewport.scale,
      top: -viewport.offsetY / viewport.scale,
      right: (canvasWidth - viewport.offsetX) / viewport.scale,
      bottom: (canvasHeight - viewport.offsetY) / viewport.scale,
    };

    // Determine if multi-selection (more than one selected)
    const isMultiSelect = selectedIds.size > 1;

    // Draw all annotations
    for (const follicle of follicles) {
      const isSelected = selectedIds.has(follicle.id);
      if (isCircle(follicle)) {
        this.drawCircle(follicle, isSelected, isMultiSelect, viewport.scale, showLabels, showShapes);
      } else if (isRectangle(follicle)) {
        this.drawRectangle(follicle, isSelected, isMultiSelect, viewport.scale, showLabels, showShapes, visibleBounds, originPunchDiameter);
      } else if (isLinear(follicle)) {
        this.drawLinear(follicle, isSelected, isMultiSelect, viewport.scale, showLabels, showShapes);
      }
    }

    // Draw preview for new shape during click-to-click creation
    // dragType is 'create' when actively creating a shape (regardless of isDragging)
    const isCreating = dragState.dragType === 'create' && dragState.startPoint && dragState.currentPoint;
    if (isCreating) {
      this.drawDragPreview(dragState, viewport.scale, currentShapeType);
    }

    // Draw selection preview (marquee or lasso)
    if (dragState.dragType === 'marquee' && dragState.startPoint && dragState.currentPoint) {
      this.drawMarqueePreview(dragState.startPoint, dragState.currentPoint, viewport.scale);
    }
    if (dragState.dragType === 'lasso' && dragState.lassoPoints && dragState.lassoPoints.length > 1) {
      this.drawLassoPreview(dragState.lassoPoints, viewport.scale);
    }

    this.ctx.restore();
  }

  private drawCircle(
    circle: CircleAnnotation,
    isSelected: boolean,
    isMultiSelect: boolean,
    scale: number,
    showLabels: boolean,
    showShapes: boolean
  ): void {
    const { center, radius, color, label } = circle;

    if (showShapes) {
      // Fill with semi-transparent color (slightly more opaque when selected)
      this.ctx.beginPath();
      this.ctx.arc(center.x, center.y, radius, 0, Math.PI * 2);
      this.ctx.fillStyle = this.hexToRgba(color, isSelected ? 0.35 : 0.25);
      this.ctx.fill();

      // Stroke - different style based on selection state
      if (isSelected && isMultiSelect) {
        // Multi-selected: dashed blue border, no resize handles
        this.ctx.strokeStyle = '#4A90D9';
        this.ctx.lineWidth = 3 / scale;
        this.ctx.setLineDash([6 / scale, 4 / scale]);
        this.ctx.stroke();
        this.ctx.setLineDash([]);
      } else {
        // Single selected or not selected: solid border
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = isSelected ? 3 / scale : 2 / scale;
        this.ctx.stroke();
      }

      // Selection indicator (only for single selection)
      if (isSelected && !isMultiSelect) {
        // Draw resize handle at right edge
        const handleRadius = 6 / scale;
        this.ctx.beginPath();
        this.ctx.arc(center.x + radius, center.y, handleRadius, 0, Math.PI * 2);
        this.ctx.fillStyle = '#ffffff';
        this.ctx.fill();
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 2 / scale;
        this.ctx.stroke();

        // Draw center point
        this.ctx.beginPath();
        this.ctx.arc(center.x, center.y, 4 / scale, 0, Math.PI * 2);
        this.ctx.fillStyle = color;
        this.ctx.fill();
      }
    }

    // Draw label
    if (showShapes && showLabels) {
      this.drawLabel(label, center.x, center.y - radius - 8 / scale, color, scale);
    }
  }

  private drawRectangle(
    rect: RectangleAnnotation,
    isSelected: boolean,
    isMultiSelect: boolean,
    scale: number,
    showLabels: boolean,
    showShapes: boolean,
    visibleBounds?: { left: number; top: number; right: number; bottom: number },
    originPunchDiameter: number = 30
  ): void {
    const { x, y, width, height, color, label, origin } = rect;
    const isLocked = !!origin;

    if (showShapes) {
      // Fill with semi-transparent color (slightly more opaque when selected)
      this.ctx.fillStyle = this.hexToRgba(color, isSelected ? 0.35 : 0.25);
      this.ctx.fillRect(x, y, width, height);

      // Stroke - different style based on selection and lock state
      if (isSelected && isMultiSelect) {
        // Multi-selected: dashed blue border, no resize handles
        this.ctx.strokeStyle = '#4A90D9';
        this.ctx.lineWidth = 3 / scale;
        this.ctx.setLineDash([6 / scale, 4 / scale]);
        this.ctx.strokeRect(x, y, width, height);
        this.ctx.setLineDash([]);
      } else if (isLocked) {
        // Locked: teal border to indicate locked state
        this.ctx.strokeStyle = '#4ECDC4';
        this.ctx.lineWidth = isSelected ? 3 / scale : 2 / scale;
        this.ctx.strokeRect(x, y, width, height);
      } else {
        // Normal: solid border with annotation color
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = isSelected ? 3 / scale : 2 / scale;
        this.ctx.strokeRect(x, y, width, height);
      }

      // Selection indicator - corner handles (only for single selection, not locked)
      if (isSelected && !isMultiSelect && !isLocked) {
        const handleRadius = 6 / scale;
        const corners = [
          { x: x, y: y },                    // top-left
          { x: x + width, y: y },            // top-right
          { x: x, y: y + height },           // bottom-left
          { x: x + width, y: y + height },   // bottom-right
        ];

        for (const corner of corners) {
          this.ctx.beginPath();
          this.ctx.arc(corner.x, corner.y, handleRadius, 0, Math.PI * 2);
          this.ctx.fillStyle = '#ffffff';
          this.ctx.fill();
          this.ctx.strokeStyle = color;
          this.ctx.lineWidth = 2 / scale;
          this.ctx.stroke();
        }

        // Draw center point
        this.ctx.beginPath();
        this.ctx.arc(x + width / 2, y + height / 2, 4 / scale, 0, Math.PI * 2);
        this.ctx.fillStyle = color;
        this.ctx.fill();
      }

      // Draw origin arrow for locked rectangles
      // Skip if zoomed out too far (origins would be too small to see)
      // or if origin is outside visible bounds
      if (origin && scale > 0.05) {
        const { originPoint } = origin;
        // Viewport culling - skip if origin is outside visible area (with margin)
        const margin = 50;
        if (!visibleBounds ||
            (originPoint.x >= visibleBounds.left - margin &&
             originPoint.x <= visibleBounds.right + margin &&
             originPoint.y >= visibleBounds.top - margin &&
             originPoint.y <= visibleBounds.bottom + margin)) {
          this.drawOriginArrow(origin, scale, originPunchDiameter);
        }
      }
    }

    // Draw label above the rectangle
    if (showShapes && showLabels) {
      this.drawLabel(label, x + width / 2, y - 8 / scale, color, scale);
    }
  }

  private drawOriginArrow(
    origin: FollicleOrigin,
    scale: number,
    punchDiameter: number = 30
  ): void {
    const { originPoint, directionAngle, directionLength } = origin;

    // Punch circle radius in image pixels (fixed size, scales with zoom)
    const punchRadius = punchDiameter / 2;

    // Line width scales with image but clamped to reasonable screen size
    const baseLineWidth = 2;
    const lineWidth = Math.min(baseLineWidth, 4 / scale);
    const headLen = Math.min(6, 10 / scale);

    // Skip drawing if punch circle would be smaller than 2 screen pixels
    if (punchRadius * scale < 2) return;

    // Arrow line and direction
    const len = Math.min(directionLength, 40);
    const endX = originPoint.x + Math.cos(directionAngle) * len;
    const endY = originPoint.y + Math.sin(directionAngle) * len;

    // Draw arrow line first (under the circle)
    this.ctx.strokeStyle = '#4ECDC4';
    this.ctx.lineWidth = lineWidth;
    this.ctx.beginPath();
    this.ctx.moveTo(originPoint.x, originPoint.y);
    this.ctx.lineTo(endX, endY);
    this.ctx.stroke();

    // Arrow head
    this.ctx.beginPath();
    this.ctx.moveTo(endX, endY);
    this.ctx.lineTo(
      endX - headLen * Math.cos(directionAngle - Math.PI / 6),
      endY - headLen * Math.sin(directionAngle - Math.PI / 6)
    );
    this.ctx.lineTo(
      endX - headLen * Math.cos(directionAngle + Math.PI / 6),
      endY - headLen * Math.sin(directionAngle + Math.PI / 6)
    );
    this.ctx.closePath();
    this.ctx.fillStyle = '#4ECDC4';
    this.ctx.fill();

    // Punch gripper circle (outline, not filled)
    // This represents the actual extraction tool size
    this.ctx.beginPath();
    this.ctx.arc(originPoint.x, originPoint.y, punchRadius, 0, Math.PI * 2);
    this.ctx.strokeStyle = '#FF6B6B';
    this.ctx.lineWidth = Math.max(1.5, 2 / scale); // Minimum 1.5px line width on screen
    this.ctx.stroke();

    // Small center dot for precise positioning
    const centerDotRadius = Math.min(3, 4 / scale);
    if (centerDotRadius * scale >= 1) {
      this.ctx.beginPath();
      this.ctx.arc(originPoint.x, originPoint.y, centerDotRadius, 0, Math.PI * 2);
      this.ctx.fillStyle = '#FF6B6B';
      this.ctx.fill();
    }
  }

  private drawLinear(
    linear: LinearAnnotation,
    isSelected: boolean,
    isMultiSelect: boolean,
    scale: number,
    showLabels: boolean,
    showShapes: boolean
  ): void {
    const { startPoint, endPoint, halfWidth, color, label } = linear;
    const corners = getLinearCorners(startPoint, endPoint, halfWidth);

    if (showShapes) {
      // Draw filled rotated rectangle
      this.ctx.beginPath();
      this.ctx.moveTo(corners[0].x, corners[0].y);
      this.ctx.lineTo(corners[1].x, corners[1].y);
      this.ctx.lineTo(corners[2].x, corners[2].y);
      this.ctx.lineTo(corners[3].x, corners[3].y);
      this.ctx.closePath();

      // Fill with semi-transparent color (slightly more opaque when selected)
      this.ctx.fillStyle = this.hexToRgba(color, isSelected ? 0.35 : 0.25);
      this.ctx.fill();

      // Stroke - different style based on selection state
      if (isSelected && isMultiSelect) {
        // Multi-selected: dashed blue border, no resize handles
        this.ctx.strokeStyle = '#4A90D9';
        this.ctx.lineWidth = 3 / scale;
        this.ctx.setLineDash([6 / scale, 4 / scale]);
        this.ctx.stroke();
        this.ctx.setLineDash([]);
      } else {
        // Single selected or not selected: solid border
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = isSelected ? 3 / scale : 2 / scale;
        this.ctx.stroke();
      }

      // Draw centerline (dashed) and handles when single-selected only
      if (isSelected && !isMultiSelect) {
        this.ctx.beginPath();
        this.ctx.setLineDash([4 / scale, 4 / scale]);
        this.ctx.moveTo(startPoint.x, startPoint.y);
        this.ctx.lineTo(endPoint.x, endPoint.y);
        this.ctx.strokeStyle = this.hexToRgba(color, 0.6);
        this.ctx.lineWidth = 1 / scale;
        this.ctx.stroke();
        this.ctx.setLineDash([]);

        // Draw handles at start and end points
        const handleRadius = 6 / scale;

        // Start point handle
        this.ctx.beginPath();
        this.ctx.arc(startPoint.x, startPoint.y, handleRadius, 0, Math.PI * 2);
        this.ctx.fillStyle = '#ffffff';
        this.ctx.fill();
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 2 / scale;
        this.ctx.stroke();

        // End point handle
        this.ctx.beginPath();
        this.ctx.arc(endPoint.x, endPoint.y, handleRadius, 0, Math.PI * 2);
        this.ctx.fillStyle = '#ffffff';
        this.ctx.fill();
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 2 / scale;
        this.ctx.stroke();

        // Draw width handles (perpendicular to centerline at midpoint)
        const midX = (startPoint.x + endPoint.x) / 2;
        const midY = (startPoint.y + endPoint.y) / 2;
        const dx = endPoint.x - startPoint.x;
        const dy = endPoint.y - startPoint.y;
        const length = Math.sqrt(dx * dx + dy * dy);

        if (length > 0) {
          const perpX = -dy / length;
          const perpY = dx / length;

          // Width handle on one side
          const widthHandle1 = { x: midX + perpX * halfWidth, y: midY + perpY * halfWidth };
          this.ctx.beginPath();
          this.ctx.arc(widthHandle1.x, widthHandle1.y, handleRadius * 0.8, 0, Math.PI * 2);
          this.ctx.fillStyle = '#ffffff';
          this.ctx.fill();
          this.ctx.strokeStyle = color;
          this.ctx.lineWidth = 2 / scale;
          this.ctx.stroke();

          // Width handle on other side
          const widthHandle2 = { x: midX - perpX * halfWidth, y: midY - perpY * halfWidth };
          this.ctx.beginPath();
          this.ctx.arc(widthHandle2.x, widthHandle2.y, handleRadius * 0.8, 0, Math.PI * 2);
          this.ctx.fillStyle = '#ffffff';
          this.ctx.fill();
          this.ctx.strokeStyle = color;
          this.ctx.lineWidth = 2 / scale;
          this.ctx.stroke();
        }

        // Draw center point
        this.ctx.beginPath();
        this.ctx.arc(midX, midY, 4 / scale, 0, Math.PI * 2);
        this.ctx.fillStyle = color;
        this.ctx.fill();
      }
    }

    // Draw label above the shape
    if (showShapes && showLabels) {
      const midX = (startPoint.x + endPoint.x) / 2;
      // Find the topmost point for label placement
      const minY = Math.min(corners[0].y, corners[1].y, corners[2].y, corners[3].y);
      this.drawLabel(label, midX, minY - 8 / scale, color, scale);
    }
  }

  private drawLabel(label: string, x: number, y: number, color: string, scale: number): void {
    const fontSize = Math.max(12, 14 / scale);
    this.ctx.font = `${fontSize}px system-ui, -apple-system, sans-serif`;
    this.ctx.textAlign = 'center';
    this.ctx.textBaseline = 'bottom';

    // Draw label background
    const textMetrics = this.ctx.measureText(label);
    const textHeight = fontSize;
    const padding = 4 / scale;

    this.ctx.fillStyle = this.hexToRgba(color, 0.8);
    this.ctx.fillRect(
      x - textMetrics.width / 2 - padding,
      y - textHeight - padding,
      textMetrics.width + padding * 2,
      textHeight + padding * 2
    );

    // Draw label text
    this.ctx.fillStyle = '#ffffff';
    this.ctx.fillText(label, x, y);
  }

  private drawDragPreview(dragState: DragState, scale: number, shapeType: ShapeType): void {
    if (!dragState.startPoint || !dragState.currentPoint) return;

    this.ctx.strokeStyle = '#ffffff';
    this.ctx.lineWidth = 2 / scale;
    this.ctx.setLineDash([5 / scale, 5 / scale]);

    if (shapeType === 'circle') {
      const radius = Math.sqrt(
        Math.pow(dragState.currentPoint.x - dragState.startPoint.x, 2) +
        Math.pow(dragState.currentPoint.y - dragState.startPoint.y, 2)
      );

      // Draw dashed circle
      this.ctx.beginPath();
      this.ctx.arc(dragState.startPoint.x, dragState.startPoint.y, radius, 0, Math.PI * 2);
      this.ctx.stroke();
      this.ctx.setLineDash([]);

      // Draw center point
      this.ctx.beginPath();
      this.ctx.arc(dragState.startPoint.x, dragState.startPoint.y, 4 / scale, 0, Math.PI * 2);
      this.ctx.fillStyle = '#ffffff';
      this.ctx.fill();

      // Draw radius line
      this.ctx.beginPath();
      this.ctx.moveTo(dragState.startPoint.x, dragState.startPoint.y);
      this.ctx.lineTo(dragState.currentPoint.x, dragState.currentPoint.y);
      this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
      this.ctx.lineWidth = 1 / scale;
      this.ctx.stroke();

      // Draw radius text
      const fontSize = 12 / scale;
      this.ctx.font = `${fontSize}px system-ui, sans-serif`;
      this.ctx.fillStyle = '#ffffff';
      this.ctx.textAlign = 'center';
      const midX = (dragState.startPoint.x + dragState.currentPoint.x) / 2;
      const midY = (dragState.startPoint.y + dragState.currentPoint.y) / 2;
      this.ctx.fillText(`r: ${Math.round(radius)}px`, midX, midY - 10 / scale);
    } else if (shapeType === 'rectangle') {
      // Rectangle preview
      const x = Math.min(dragState.startPoint.x, dragState.currentPoint.x);
      const y = Math.min(dragState.startPoint.y, dragState.currentPoint.y);
      const width = Math.abs(dragState.currentPoint.x - dragState.startPoint.x);
      const height = Math.abs(dragState.currentPoint.y - dragState.startPoint.y);

      // Draw dashed rectangle
      this.ctx.strokeRect(x, y, width, height);
      this.ctx.setLineDash([]);

      // Draw dimension text
      const fontSize = 12 / scale;
      this.ctx.font = `${fontSize}px system-ui, sans-serif`;
      this.ctx.fillStyle = '#ffffff';
      this.ctx.textAlign = 'center';
      this.ctx.fillText(
        `${Math.round(width)} x ${Math.round(height)}`,
        x + width / 2,
        y + height / 2
      );
    } else if (shapeType === 'linear') {
      // Linear shape preview
      if (dragState.createPhase === 'width' && dragState.lineEndPoint) {
        // Phase 2: Show rotated rectangle with current width
        const halfWidth = pointToLineDistance(
          dragState.currentPoint,
          dragState.startPoint,
          dragState.lineEndPoint
        );
        const corners = getLinearCorners(dragState.startPoint, dragState.lineEndPoint, halfWidth);

        // Draw the rotated rectangle
        this.ctx.beginPath();
        this.ctx.moveTo(corners[0].x, corners[0].y);
        this.ctx.lineTo(corners[1].x, corners[1].y);
        this.ctx.lineTo(corners[2].x, corners[2].y);
        this.ctx.lineTo(corners[3].x, corners[3].y);
        this.ctx.closePath();
        this.ctx.stroke();
        this.ctx.setLineDash([]);

        // Draw centerline
        this.ctx.beginPath();
        this.ctx.moveTo(dragState.startPoint.x, dragState.startPoint.y);
        this.ctx.lineTo(dragState.lineEndPoint.x, dragState.lineEndPoint.y);
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
        this.ctx.lineWidth = 1 / scale;
        this.ctx.stroke();

        // Draw endpoint markers
        this.ctx.beginPath();
        this.ctx.arc(dragState.startPoint.x, dragState.startPoint.y, 4 / scale, 0, Math.PI * 2);
        this.ctx.fillStyle = '#ffffff';
        this.ctx.fill();

        this.ctx.beginPath();
        this.ctx.arc(dragState.lineEndPoint.x, dragState.lineEndPoint.y, 4 / scale, 0, Math.PI * 2);
        this.ctx.fillStyle = '#ffffff';
        this.ctx.fill();

        // Draw dimensions
        const length = Math.sqrt(
          (dragState.lineEndPoint.x - dragState.startPoint.x) ** 2 +
          (dragState.lineEndPoint.y - dragState.startPoint.y) ** 2
        );
        const midX = (dragState.startPoint.x + dragState.lineEndPoint.x) / 2;
        const midY = (dragState.startPoint.y + dragState.lineEndPoint.y) / 2;
        const fontSize = 12 / scale;
        this.ctx.font = `${fontSize}px system-ui, sans-serif`;
        this.ctx.fillStyle = '#ffffff';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(
          `L: ${Math.round(length)} W: ${Math.round(halfWidth * 2)}`,
          midX,
          midY - 10 / scale
        );
      } else {
        // Phase 1: Drawing the centerline
        this.ctx.beginPath();
        this.ctx.moveTo(dragState.startPoint.x, dragState.startPoint.y);
        this.ctx.lineTo(dragState.currentPoint.x, dragState.currentPoint.y);
        this.ctx.stroke();
        this.ctx.setLineDash([]);

        // Draw start point
        this.ctx.beginPath();
        this.ctx.arc(dragState.startPoint.x, dragState.startPoint.y, 4 / scale, 0, Math.PI * 2);
        this.ctx.fillStyle = '#ffffff';
        this.ctx.fill();

        // Draw current point
        this.ctx.beginPath();
        this.ctx.arc(dragState.currentPoint.x, dragState.currentPoint.y, 4 / scale, 0, Math.PI * 2);
        this.ctx.fillStyle = '#ffffff';
        this.ctx.fill();

        // Draw length text
        const length = Math.sqrt(
          (dragState.currentPoint.x - dragState.startPoint.x) ** 2 +
          (dragState.currentPoint.y - dragState.startPoint.y) ** 2
        );
        const midX = (dragState.startPoint.x + dragState.currentPoint.x) / 2;
        const midY = (dragState.startPoint.y + dragState.currentPoint.y) / 2;
        const fontSize = 12 / scale;
        this.ctx.font = `${fontSize}px system-ui, sans-serif`;
        this.ctx.fillStyle = '#ffffff';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(`L: ${Math.round(length)}px`, midX, midY - 10 / scale);
      }
    }
  }

  private hexToRgba(hex: string, alpha: number): string {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }

  private drawMarqueePreview(start: Point, current: Point, scale: number): void {
    const x = Math.min(start.x, current.x);
    const y = Math.min(start.y, current.y);
    const width = Math.abs(current.x - start.x);
    const height = Math.abs(current.y - start.y);

    // Fill with light blue semi-transparent
    this.ctx.fillStyle = 'rgba(74, 144, 217, 0.15)';
    this.ctx.fillRect(x, y, width, height);

    // Stroke with dashed blue border
    this.ctx.strokeStyle = '#4A90D9';
    this.ctx.lineWidth = 1.5 / scale;
    this.ctx.setLineDash([6 / scale, 4 / scale]);
    this.ctx.strokeRect(x, y, width, height);
    this.ctx.setLineDash([]);
  }

  private drawLassoPreview(points: Point[], scale: number): void {
    if (points.length < 2) return;

    // Draw the lasso path
    this.ctx.beginPath();
    this.ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
      this.ctx.lineTo(points[i].x, points[i].y);
    }
    // Close the path back to the start
    this.ctx.lineTo(points[0].x, points[0].y);
    this.ctx.closePath();

    // Fill with light blue semi-transparent
    this.ctx.fillStyle = 'rgba(74, 144, 217, 0.15)';
    this.ctx.fill();

    // Stroke with dashed blue border
    this.ctx.strokeStyle = '#4A90D9';
    this.ctx.lineWidth = 1.5 / scale;
    this.ctx.setLineDash([6 / scale, 4 / scale]);
    this.ctx.stroke();
    this.ctx.setLineDash([]);
  }
}
