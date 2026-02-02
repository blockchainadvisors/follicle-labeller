import { useState, useRef, useCallback, useEffect } from 'react';
import { X, Check } from 'lucide-react';
import type { RectangleAnnotation, FollicleOrigin, Point } from '../../types';
import './FollicleOriginDialog.css';

interface FollicleOriginDialogProps {
  annotation: RectangleAnnotation;
  imageBitmap: ImageBitmap | null;
  onSave: (origin: FollicleOrigin) => void;
  onCancel: () => void;
}

type Phase = 'origin' | 'direction';

export function FollicleOriginDialog({
  annotation,
  imageBitmap,
  onSave,
  onCancel,
}: FollicleOriginDialogProps) {
  const [phase, setPhase] = useState<Phase>('origin');
  const [originPoint, setOriginPoint] = useState<Point | null>(
    annotation.origin?.originPoint ?? null
  );
  const [directionAngle, setDirectionAngle] = useState<number>(
    annotation.origin?.directionAngle ?? 0
  );
  const [directionLength, setDirectionLength] = useState<number>(
    annotation.origin?.directionLength ?? 30
  );
  const [isDragging, setIsDragging] = useState(false);

  // Draggable dialog state
  const [position, setPosition] = useState<{ x: number; y: number } | null>(null);
  const dragRef = useRef<{ startX: number; startY: number; startPosX: number; startPosY: number } | null>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Padding around annotation in the zoomed view
  const PADDING = 20;

  // Handle dialog dragging
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if ((e.target as HTMLElement).closest('button')) return;
    if ((e.target as HTMLElement).closest('.origin-canvas-container')) return;

    const dialog = dialogRef.current;
    if (!dialog) return;

    const rect = dialog.getBoundingClientRect();
    const currentX = position?.x ?? rect.left + rect.width / 2 - window.innerWidth / 2;
    const currentY = position?.y ?? rect.top + rect.height / 2 - window.innerHeight / 2;

    dragRef.current = {
      startX: e.clientX,
      startY: e.clientY,
      startPosX: currentX,
      startPosY: currentY,
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  }, [position]);

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!dragRef.current) return;

    const deltaX = e.clientX - dragRef.current.startX;
    const deltaY = e.clientY - dragRef.current.startY;

    setPosition({
      x: dragRef.current.startPosX + deltaX,
      y: dragRef.current.startPosY + deltaY,
    });
  }, []);

  const handleMouseUp = useCallback(() => {
    dragRef.current = null;
    document.removeEventListener('mousemove', handleMouseMove);
    document.removeEventListener('mouseup', handleMouseUp);
  }, [handleMouseMove]);

  // Cleanup event listeners on unmount
  useEffect(() => {
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [handleMouseMove, handleMouseUp]);

  // Convert canvas coordinates to image coordinates
  const canvasToImage = useCallback((canvasX: number, canvasY: number): Point => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };

    const canvasWidth = canvas.width;
    const canvasHeight = canvas.height;

    // Calculate the view bounds with padding
    const viewWidth = annotation.width + PADDING * 2;
    const viewHeight = annotation.height + PADDING * 2;
    const viewX = annotation.x - PADDING;
    const viewY = annotation.y - PADDING;

    // Scale factor
    const scale = Math.min(canvasWidth / viewWidth, canvasHeight / viewHeight);
    const offsetX = (canvasWidth - viewWidth * scale) / 2;
    const offsetY = (canvasHeight - viewHeight * scale) / 2;

    // Convert canvas coords to image coords
    const imageX = (canvasX - offsetX) / scale + viewX;
    const imageY = (canvasY - offsetY) / scale + viewY;

    return { x: imageX, y: imageY };
  }, [annotation]);

  // Handle canvas click for setting origin point
  const handleCanvasClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const canvasX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const canvasY = (e.clientY - rect.top) * (canvas.height / rect.height);

    const imagePoint = canvasToImage(canvasX, canvasY);

    if (phase === 'origin') {
      setOriginPoint(imagePoint);
      setPhase('direction');
    }
  }, [phase, canvasToImage]);

  // Handle canvas mouse down for direction dragging
  const handleCanvasMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (phase !== 'direction' || !originPoint) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const canvasX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const canvasY = (e.clientY - rect.top) * (canvas.height / rect.height);

    const imagePoint = canvasToImage(canvasX, canvasY);

    // Calculate angle and length
    const dx = imagePoint.x - originPoint.x;
    const dy = imagePoint.y - originPoint.y;
    const angle = Math.atan2(dy, dx);
    const length = Math.sqrt(dx * dx + dy * dy);

    setDirectionAngle(angle);
    setDirectionLength(Math.min(Math.max(length, 10), 100));
    setIsDragging(true);
  }, [phase, originPoint, canvasToImage]);

  // Handle canvas mouse move for direction dragging
  const handleCanvasMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDragging || phase !== 'direction' || !originPoint) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const canvasX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const canvasY = (e.clientY - rect.top) * (canvas.height / rect.height);

    const imagePoint = canvasToImage(canvasX, canvasY);

    // Calculate angle and length
    const dx = imagePoint.x - originPoint.x;
    const dy = imagePoint.y - originPoint.y;
    const angle = Math.atan2(dy, dx);
    const length = Math.sqrt(dx * dx + dy * dy);

    setDirectionAngle(angle);
    setDirectionLength(Math.min(Math.max(length, 10), 100));
  }, [isDragging, phase, originPoint, canvasToImage]);

  // Handle canvas mouse up
  const handleCanvasMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  // Render the canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imageBitmap) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    const containerWidth = canvas.parentElement?.clientWidth ?? 400;
    canvas.width = containerWidth;
    canvas.height = containerWidth;

    const canvasWidth = canvas.width;
    const canvasHeight = canvas.height;

    // Clear canvas
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);

    // Calculate the view bounds with padding
    const viewWidth = annotation.width + PADDING * 2;
    const viewHeight = annotation.height + PADDING * 2;
    const viewX = annotation.x - PADDING;
    const viewY = annotation.y - PADDING;

    // Scale factor to fit annotation in canvas
    const scale = Math.min(canvasWidth / viewWidth, canvasHeight / viewHeight);
    const offsetX = (canvasWidth - viewWidth * scale) / 2;
    const offsetY = (canvasHeight - viewHeight * scale) / 2;

    // Draw the cropped region of the image
    ctx.save();
    ctx.translate(offsetX, offsetY);
    ctx.scale(scale, scale);
    ctx.translate(-viewX, -viewY);

    // Draw image
    ctx.drawImage(imageBitmap, 0, 0);

    // Draw rectangle border
    ctx.strokeStyle = annotation.color;
    ctx.lineWidth = 2 / scale;
    ctx.strokeRect(annotation.x, annotation.y, annotation.width, annotation.height);

    // Draw origin point if set
    if (originPoint) {
      // Origin dot
      ctx.beginPath();
      ctx.arc(originPoint.x, originPoint.y, 6 / scale, 0, Math.PI * 2);
      ctx.fillStyle = '#FF6B6B';
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2 / scale;
      ctx.stroke();

      // Draw direction arrow if in direction phase or if we have direction set
      if (phase === 'direction' || directionLength > 0) {
        const endX = originPoint.x + Math.cos(directionAngle) * directionLength;
        const endY = originPoint.y + Math.sin(directionAngle) * directionLength;

        // Arrow line
        ctx.strokeStyle = '#4ECDC4';
        ctx.lineWidth = 3 / scale;
        ctx.beginPath();
        ctx.moveTo(originPoint.x, originPoint.y);
        ctx.lineTo(endX, endY);
        ctx.stroke();

        // Arrow head
        const headLen = 10 / scale;
        ctx.beginPath();
        ctx.moveTo(endX, endY);
        ctx.lineTo(
          endX - headLen * Math.cos(directionAngle - Math.PI / 6),
          endY - headLen * Math.sin(directionAngle - Math.PI / 6)
        );
        ctx.lineTo(
          endX - headLen * Math.cos(directionAngle + Math.PI / 6),
          endY - headLen * Math.sin(directionAngle + Math.PI / 6)
        );
        ctx.closePath();
        ctx.fillStyle = '#4ECDC4';
        ctx.fill();
      }
    }

    ctx.restore();

  }, [imageBitmap, annotation, originPoint, directionAngle, directionLength, phase]);

  const handleReset = () => {
    setOriginPoint(null);
    setDirectionAngle(0);
    setDirectionLength(30);
    setPhase('origin');
  };

  const handleSave = () => {
    if (!originPoint) return;

    onSave({
      originPoint,
      directionAngle,
      directionLength,
    });
  };

  const canSave = originPoint !== null && phase === 'direction';

  return (
    <div className="follicle-origin-overlay" onClick={onCancel}>
      <div
        ref={dialogRef}
        className="follicle-origin-dialog"
        onClick={e => e.stopPropagation()}
        style={position ? { transform: `translate(${position.x}px, ${position.y}px)` } : undefined}
      >
        {/* Compact header with phase and instructions */}
        <div className="dialog-header" onMouseDown={handleMouseDown}>
          <div className="header-left">
            <div className={`step-badge ${originPoint ? 'completed' : 'active'}`}>
              {originPoint ? <Check size={12} /> : '1'}
            </div>
            <div className={`step-badge ${phase === 'direction' ? 'active' : ''}`}>2</div>
            <span className="header-instruction">
              {phase === 'origin' ? 'Click origin point' : 'Drag to set direction'}
            </span>
          </div>
          <button className="close-button" onClick={onCancel}>
            <X size={16} />
          </button>
        </div>

        {/* Canvas - maximum space */}
        <div className="dialog-content">
          <div className="origin-canvas-container">
            <canvas
              ref={canvasRef}
              className="origin-canvas"
              onClick={handleCanvasClick}
              onMouseDown={handleCanvasMouseDown}
              onMouseMove={handleCanvasMouseMove}
              onMouseUp={handleCanvasMouseUp}
              onMouseLeave={handleCanvasMouseUp}
            />
          </div>
        </div>

        {/* Compact footer with info and actions */}
        <div className="dialog-footer">
          <div className="origin-info">
            <span className="info-item">X: <strong>{originPoint ? originPoint.x.toFixed(0) : '-'}</strong></span>
            <span className="info-item">Y: <strong>{originPoint ? originPoint.y.toFixed(0) : '-'}</strong></span>
            <span className="info-item">Dir: <strong>{originPoint ? `${(directionAngle * 180 / Math.PI).toFixed(0)}°` : '-'}</strong></span>
          </div>
          <div className="footer-actions">
            <button className="btn-sm" onClick={handleReset}>Reset</button>
            <button className="btn-sm" onClick={onCancel}>Cancel</button>
            <button className="btn-sm btn-primary" onClick={handleSave} disabled={!canSave}>Save</button>
          </div>
        </div>
      </div>
    </div>
  );
}
