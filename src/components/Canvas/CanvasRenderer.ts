import { Follicle, Viewport, DragState, ShapeType, isCircle, isRectangle, CircleAnnotation, RectangleAnnotation } from '../../types';

export class CanvasRenderer {
  private ctx: CanvasRenderingContext2D;
  private image: HTMLImageElement | null = null;

  constructor(ctx: CanvasRenderingContext2D) {
    this.ctx = ctx;
  }

  setImage(image: HTMLImageElement): void {
    this.image = image;
  }

  clear(width: number, height: number): void {
    this.ctx.fillStyle = '#1a1a2e';
    this.ctx.fillRect(0, 0, width, height);
  }

  render(
    canvasWidth: number,
    canvasHeight: number,
    viewport: Viewport,
    follicles: Follicle[],
    selectedId: string | null,
    dragState: DragState,
    showLabels: boolean = true,
    showShapes: boolean = true,
    currentShapeType: ShapeType = 'circle'
  ): void {
    this.clear(canvasWidth, canvasHeight);

    this.ctx.save();
    this.ctx.translate(viewport.offsetX, viewport.offsetY);
    this.ctx.scale(viewport.scale, viewport.scale);

    // Draw image
    if (this.image) {
      this.ctx.drawImage(this.image, 0, 0);
    }

    // Draw all annotations
    for (const follicle of follicles) {
      if (isCircle(follicle)) {
        this.drawCircle(follicle, follicle.id === selectedId, viewport.scale, showLabels, showShapes);
      } else if (isRectangle(follicle)) {
        this.drawRectangle(follicle, follicle.id === selectedId, viewport.scale, showLabels, showShapes);
      }
    }

    // Draw drag preview for new shape
    if (dragState.isDragging && dragState.dragType === 'create' && dragState.startPoint && dragState.currentPoint) {
      this.drawDragPreview(dragState, viewport.scale, currentShapeType);
    }

    this.ctx.restore();
  }

  private drawCircle(
    circle: CircleAnnotation,
    isSelected: boolean,
    scale: number,
    showLabels: boolean,
    showShapes: boolean
  ): void {
    const { center, radius, color, label } = circle;

    if (showShapes) {
      // Fill with semi-transparent color
      this.ctx.beginPath();
      this.ctx.arc(center.x, center.y, radius, 0, Math.PI * 2);
      this.ctx.fillStyle = this.hexToRgba(color, 0.25);
      this.ctx.fill();

      // Stroke
      this.ctx.strokeStyle = color;
      this.ctx.lineWidth = isSelected ? 3 / scale : 2 / scale;
      this.ctx.stroke();

      // Selection indicator
      if (isSelected) {
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
    scale: number,
    showLabels: boolean,
    showShapes: boolean
  ): void {
    const { x, y, width, height, color, label } = rect;

    if (showShapes) {
      // Fill with semi-transparent color
      this.ctx.fillStyle = this.hexToRgba(color, 0.25);
      this.ctx.fillRect(x, y, width, height);

      // Stroke
      this.ctx.strokeStyle = color;
      this.ctx.lineWidth = isSelected ? 3 / scale : 2 / scale;
      this.ctx.strokeRect(x, y, width, height);

      // Selection indicator - corner handles
      if (isSelected) {
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
    }

    // Draw label above the rectangle
    if (showShapes && showLabels) {
      this.drawLabel(label, x + width / 2, y - 8 / scale, color, scale);
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
    } else {
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
    }
  }

  private hexToRgba(hex: string, alpha: number): string {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }
}
