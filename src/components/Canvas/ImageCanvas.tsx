import React, { useRef, useEffect, useCallback, useState } from 'react';
import { useFollicleStore, useTemporalStore } from '../../store/follicleStore';
import { useCanvasStore } from '../../store/canvasStore';
import { CanvasRenderer } from './CanvasRenderer';
import { DragState, Point, Follicle, LinearAnnotation, isCircle, isRectangle, isLinear } from '../../types';
import {
  screenToImage,
  distance,
  isPointInCircle,
} from '../../utils/coordinate-transform';

// Check if point is inside a rectangle
function isPointInRectangle(point: Point, x: number, y: number, width: number, height: number): boolean {
  return point.x >= x && point.x <= x + width && point.y >= y && point.y <= y + height;
}

// Check if point is near a rectangle corner (for resize)
function getRectangleResizeHandle(
  point: Point,
  x: number,
  y: number,
  width: number,
  height: number,
  tolerance: number
): string | null {
  const corners = [
    { handle: 'nw', x: x, y: y },
    { handle: 'ne', x: x + width, y: y },
    { handle: 'sw', x: x, y: y + height },
    { handle: 'se', x: x + width, y: y + height },
  ];

  for (const corner of corners) {
    if (distance(point, { x: corner.x, y: corner.y }) <= tolerance) {
      return corner.handle;
    }
  }
  return null;
}

// Calculate distance from point to line (not segment - extends infinitely)
function pointToLineDistance(point: Point, lineStart: Point, lineEnd: Point): number {
  const dx = lineEnd.x - lineStart.x;
  const dy = lineEnd.y - lineStart.y;
  const lineLengthSquared = dx * dx + dy * dy;

  if (lineLengthSquared === 0) {
    return distance(point, lineStart);
  }

  const t = ((point.x - lineStart.x) * dx + (point.y - lineStart.y) * dy) / lineLengthSquared;
  const closestX = lineStart.x + t * dx;
  const closestY = lineStart.y + t * dy;

  return distance(point, { x: closestX, y: closestY });
}

// Check if point is inside a linear shape (rotated rectangle)
function isPointInLinear(point: Point, linear: LinearAnnotation): boolean {
  const { startPoint, endPoint, halfWidth } = linear;
  const dx = endPoint.x - startPoint.x;
  const dy = endPoint.y - startPoint.y;
  const length = Math.sqrt(dx * dx + dy * dy);

  if (length === 0) return false;

  // Transform point to local coordinates where the line is along the x-axis
  const ux = dx / length;
  const uy = dy / length;

  // Vector from start to point
  const px = point.x - startPoint.x;
  const py = point.y - startPoint.y;

  // Project onto line direction (along) and perpendicular (perp)
  const along = px * ux + py * uy;
  const perp = Math.abs(-px * uy + py * ux);

  return along >= 0 && along <= length && perp <= halfWidth;
}

// Get resize handle for linear shape
function getLinearResizeHandle(
  point: Point,
  linear: LinearAnnotation,
  tolerance: number
): string | null {
  const { startPoint, endPoint, halfWidth } = linear;

  // Check start point
  if (distance(point, startPoint) <= tolerance) {
    return 'start';
  }

  // Check end point
  if (distance(point, endPoint) <= tolerance) {
    return 'end';
  }

  // Check width handles (at midpoint, perpendicular to line)
  const midX = (startPoint.x + endPoint.x) / 2;
  const midY = (startPoint.y + endPoint.y) / 2;
  const dx = endPoint.x - startPoint.x;
  const dy = endPoint.y - startPoint.y;
  const length = Math.sqrt(dx * dx + dy * dy);

  if (length > 0) {
    const perpX = -dy / length;
    const perpY = dx / length;

    const widthHandle1 = { x: midX + perpX * halfWidth, y: midY + perpY * halfWidth };
    const widthHandle2 = { x: midX - perpX * halfWidth, y: midY - perpY * halfWidth };

    if (distance(point, widthHandle1) <= tolerance || distance(point, widthHandle2) <= tolerance) {
      return 'width';
    }
  }

  return null;
}

export const ImageCanvas: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<CanvasRenderer | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  const [dragState, setDragState] = useState<DragState>({
    isDragging: false,
    startPoint: null,
    currentPoint: null,
    dragType: null,
    targetId: null,
    resizeHandle: undefined,
  });

  const [canvasSize, setCanvasSize] = useState({ width: 800, height: 600 });

  // Store subscriptions
  const follicles = useFollicleStore(state => state.follicles);
  const selectedId = useFollicleStore(state => state.selectedId);
  const addCircle = useFollicleStore(state => state.addCircle);
  const addRectangle = useFollicleStore(state => state.addRectangle);
  const addLinear = useFollicleStore(state => state.addLinear);
  const selectFollicle = useFollicleStore(state => state.selectFollicle);
  const moveAnnotation = useFollicleStore(state => state.moveAnnotation);
  const resizeCircle = useFollicleStore(state => state.resizeCircle);
  const resizeRectangle = useFollicleStore(state => state.resizeRectangle);
  const resizeLinear = useFollicleStore(state => state.resizeLinear);
  const deleteFollicle = useFollicleStore(state => state.deleteFollicle);

  const viewport = useCanvasStore(state => state.viewport);
  const imageBitmap = useCanvasStore(state => state.imageBitmap);
  const mode = useCanvasStore(state => state.mode);
  const currentShapeType = useCanvasStore(state => state.currentShapeType);
  const pan = useCanvasStore(state => state.pan);
  const zoom = useCanvasStore(state => state.zoom);
  const zoomToFit = useCanvasStore(state => state.zoomToFit);
  const showLabels = useCanvasStore(state => state.showLabels);
  const showShapes = useCanvasStore(state => state.showShapes);

  const temporalStore = useTemporalStore();

  // Resize canvas to fit container
  useEffect(() => {
    const updateSize = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        setCanvasSize({ width: rect.width, height: rect.height });
      }
    };

    updateSize();
    window.addEventListener('resize', updateSize);
    return () => window.removeEventListener('resize', updateSize);
  }, []);

  // Initialize renderer
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    rendererRef.current = new CanvasRenderer(ctx);
  }, []);

  // Set image when bitmap changes (already pre-decoded for smooth rendering)
  useEffect(() => {
    if (!imageBitmap) {
      imageRef.current = null;
      if (rendererRef.current) {
        rendererRef.current.setImage(null);
      }
      return;
    }

    // Use pre-decoded ImageBitmap directly - no additional loading needed
    if (rendererRef.current) {
      rendererRef.current.setImage(imageBitmap);
    }
    zoomToFit(canvasSize.width, canvasSize.height);
  }, [imageBitmap, zoomToFit, canvasSize.width, canvasSize.height]);

  // Render loop
  useEffect(() => {
    const canvas = canvasRef.current;
    const renderer = rendererRef.current;
    if (!canvas || !renderer) return;

    const renderFrame = () => {
      renderer.render(
        canvas.width,
        canvas.height,
        viewport,
        follicles,
        selectedId,
        dragState,
        showLabels,
        showShapes,
        currentShapeType
      );
    };

    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    animationFrameRef.current = requestAnimationFrame(renderFrame);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [viewport, follicles, selectedId, dragState, canvasSize, showLabels, showShapes, currentShapeType]);

  // Get image coordinates from mouse event
  const getImagePoint = useCallback((e: React.MouseEvent): Point => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };

    const rect = canvas.getBoundingClientRect();
    return screenToImage(e.clientX, e.clientY, viewport, rect);
  }, [viewport]);

  // Find annotation at a point
  const findAnnotationAtPoint = useCallback((point: Point): Follicle | null => {
    for (let i = follicles.length - 1; i >= 0; i--) {
      const f = follicles[i];
      if (isCircle(f)) {
        if (isPointInCircle(point, f.center, f.radius)) {
          return f;
        }
      } else if (isRectangle(f)) {
        if (isPointInRectangle(point, f.x, f.y, f.width, f.height)) {
          return f;
        }
      } else if (isLinear(f)) {
        if (isPointInLinear(point, f)) {
          return f;
        }
      }
    }
    return null;
  }, [follicles]);

  // Mouse down handler
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    const point = getImagePoint(e);

    // Middle mouse button always triggers pan
    if (e.button === 1 || mode === 'pan') {
      e.preventDefault();
      setDragState({
        isDragging: true,
        startPoint: { x: e.clientX, y: e.clientY },
        currentPoint: { x: e.clientX, y: e.clientY },
        dragType: 'pan',
        targetId: null,
      });
      return;
    }

    if (mode === 'select') {
      const selected = follicles.find(f => f.id === selectedId);

      if (selected) {
        // Check for resize handles
        if (isCircle(selected)) {
          const handlePoint = { x: selected.center.x + selected.radius, y: selected.center.y };
          if (distance(point, handlePoint) <= 10 / viewport.scale) {
            setDragState({
              isDragging: true,
              startPoint: selected.center,
              currentPoint: point,
              dragType: 'resize',
              targetId: selected.id,
            });
            return;
          }
        } else if (isRectangle(selected)) {
          const handle = getRectangleResizeHandle(
            point,
            selected.x,
            selected.y,
            selected.width,
            selected.height,
            10 / viewport.scale
          );
          if (handle) {
            setDragState({
              isDragging: true,
              startPoint: point,
              currentPoint: point,
              dragType: 'resize',
              targetId: selected.id,
              resizeHandle: handle,
            });
            return;
          }
        } else if (isLinear(selected)) {
          const handle = getLinearResizeHandle(point, selected, 10 / viewport.scale);
          if (handle) {
            setDragState({
              isDragging: true,
              startPoint: point,
              currentPoint: point,
              dragType: 'resize',
              targetId: selected.id,
              resizeHandle: handle,
            });
            return;
          }
        }

        // Check if clicking inside selected annotation (for move)
        if (isCircle(selected) && isPointInCircle(point, selected.center, selected.radius)) {
          setDragState({
            isDragging: true,
            startPoint: point,
            currentPoint: point,
            dragType: 'move',
            targetId: selected.id,
          });
          return;
        } else if (isRectangle(selected) && isPointInRectangle(point, selected.x, selected.y, selected.width, selected.height)) {
          setDragState({
            isDragging: true,
            startPoint: point,
            currentPoint: point,
            dragType: 'move',
            targetId: selected.id,
          });
          return;
        } else if (isLinear(selected) && isPointInLinear(point, selected)) {
          setDragState({
            isDragging: true,
            startPoint: point,
            currentPoint: point,
            dragType: 'move',
            targetId: selected.id,
          });
          return;
        }
      }

      // Check if clicking inside any annotation
      const clicked = findAnnotationAtPoint(point);
      if (clicked) {
        selectFollicle(clicked.id);
        setDragState({
          isDragging: true,
          startPoint: point,
          currentPoint: point,
          dragType: 'move',
          targetId: clicked.id,
        });
        return;
      }

      selectFollicle(null);
      return;
    }

    if (mode === 'create') {
      // For linear shapes, check if we're in width-definition phase
      if (currentShapeType === 'linear' && dragState.createPhase === 'width' && dragState.startPoint && dragState.lineEndPoint) {
        // Click to finalize the linear shape
        const halfWidth = pointToLineDistance(point, dragState.startPoint, dragState.lineEndPoint);
        if (halfWidth > 5) {
          addLinear(dragState.startPoint, dragState.lineEndPoint, halfWidth);
        }
        setDragState({
          isDragging: false,
          startPoint: null,
          currentPoint: null,
          dragType: null,
          targetId: null,
          createPhase: undefined,
          lineEndPoint: undefined,
        });
        return;
      }

      // Start new shape creation
      setDragState({
        isDragging: true,
        startPoint: point,
        currentPoint: point,
        dragType: 'create',
        targetId: null,
        createPhase: currentShapeType === 'linear' ? 'line' : undefined,
      });
    }
  }, [mode, selectedId, follicles, viewport.scale, getImagePoint, selectFollicle, findAnnotationAtPoint, currentShapeType, dragState, addLinear]);

  // Mouse move handler
  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    // Handle width-definition phase for linear shapes (not dragging, but tracking mouse)
    if (dragState.createPhase === 'width' && dragState.startPoint && dragState.lineEndPoint) {
      const point = getImagePoint(e);
      setDragState(prev => ({ ...prev, currentPoint: point }));
      return;
    }

    if (!dragState.isDragging) return;

    if (dragState.dragType === 'pan') {
      const deltaX = e.clientX - (dragState.currentPoint?.x || 0);
      const deltaY = e.clientY - (dragState.currentPoint?.y || 0);
      pan(deltaX, deltaY);
      setDragState(prev => ({
        ...prev,
        currentPoint: { x: e.clientX, y: e.clientY },
      }));
      return;
    }

    const point = getImagePoint(e);
    setDragState(prev => ({ ...prev, currentPoint: point }));
  }, [dragState, pan, getImagePoint]);

  // Mouse up handler
  const handleMouseUp = useCallback(() => {
    if (!dragState.isDragging) return;

    if (dragState.dragType === 'create' && dragState.startPoint && dragState.currentPoint) {
      if (currentShapeType === 'circle') {
        const radius = distance(dragState.startPoint, dragState.currentPoint);
        if (radius > 5) {
          addCircle(dragState.startPoint, radius);
        }
      } else if (currentShapeType === 'rectangle') {
        const x = Math.min(dragState.startPoint.x, dragState.currentPoint.x);
        const y = Math.min(dragState.startPoint.y, dragState.currentPoint.y);
        const width = Math.abs(dragState.currentPoint.x - dragState.startPoint.x);
        const height = Math.abs(dragState.currentPoint.y - dragState.startPoint.y);
        if (width > 10 && height > 10) {
          addRectangle(x, y, width, height);
        }
      } else if (currentShapeType === 'linear' && dragState.createPhase === 'line') {
        // End of line definition phase - transition to width phase
        const lineLength = distance(dragState.startPoint, dragState.currentPoint);
        if (lineLength > 10) {
          setDragState({
            isDragging: false,
            startPoint: dragState.startPoint,
            currentPoint: dragState.currentPoint,
            dragType: 'create',
            targetId: null,
            createPhase: 'width',
            lineEndPoint: dragState.currentPoint,
          });
          return; // Don't reset drag state - continue to width phase
        }
      }
    }

    if (dragState.dragType === 'move' && dragState.targetId && dragState.currentPoint && dragState.startPoint) {
      const deltaX = dragState.currentPoint.x - dragState.startPoint.x;
      const deltaY = dragState.currentPoint.y - dragState.startPoint.y;
      moveAnnotation(dragState.targetId, deltaX, deltaY);
    }

    if (dragState.dragType === 'resize' && dragState.targetId && dragState.startPoint && dragState.currentPoint) {
      const target = follicles.find(f => f.id === dragState.targetId);
      if (target) {
        if (isCircle(target)) {
          const newRadius = distance(dragState.startPoint, dragState.currentPoint);
          resizeCircle(dragState.targetId, newRadius);
        } else if (isRectangle(target) && dragState.resizeHandle) {
          // Calculate new rectangle bounds based on which handle was dragged
          let newX = target.x;
          let newY = target.y;
          let newWidth = target.width;
          let newHeight = target.height;

          switch (dragState.resizeHandle) {
            case 'nw':
              newX = dragState.currentPoint.x;
              newY = dragState.currentPoint.y;
              newWidth = target.x + target.width - dragState.currentPoint.x;
              newHeight = target.y + target.height - dragState.currentPoint.y;
              break;
            case 'ne':
              newY = dragState.currentPoint.y;
              newWidth = dragState.currentPoint.x - target.x;
              newHeight = target.y + target.height - dragState.currentPoint.y;
              break;
            case 'sw':
              newX = dragState.currentPoint.x;
              newWidth = target.x + target.width - dragState.currentPoint.x;
              newHeight = dragState.currentPoint.y - target.y;
              break;
            case 'se':
              newWidth = dragState.currentPoint.x - target.x;
              newHeight = dragState.currentPoint.y - target.y;
              break;
          }

          resizeRectangle(dragState.targetId, newX, newY, newWidth, newHeight);
        } else if (isLinear(target) && dragState.resizeHandle) {
          // Handle linear resize based on which handle was dragged
          let newStart = target.startPoint;
          let newEnd = target.endPoint;
          let newHalfWidth = target.halfWidth;

          switch (dragState.resizeHandle) {
            case 'start':
              newStart = dragState.currentPoint;
              break;
            case 'end':
              newEnd = dragState.currentPoint;
              break;
            case 'width':
              newHalfWidth = pointToLineDistance(dragState.currentPoint, target.startPoint, target.endPoint);
              break;
          }

          resizeLinear(dragState.targetId, newStart, newEnd, newHalfWidth);
        }
      }
    }

    setDragState({
      isDragging: false,
      startPoint: null,
      currentPoint: null,
      dragType: null,
      targetId: null,
      resizeHandle: undefined,
      createPhase: undefined,
      lineEndPoint: undefined,
    });
  }, [dragState, currentShapeType, addCircle, addRectangle, moveAnnotation, resizeCircle, resizeRectangle, resizeLinear, follicles]);

  // Wheel zoom handler
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.1 : 0.1;
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const centerPoint = {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    };
    zoom(delta, centerPoint);
  }, [zoom]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.key === 'Delete' || e.key === 'Backspace') && selectedId) {
        e.preventDefault();
        deleteFollicle(selectedId);
      }

      if ((e.metaKey || e.ctrlKey) && e.key === 'z') {
        e.preventDefault();
        if (e.shiftKey) {
          temporalStore.getState().redo();
        } else {
          temporalStore.getState().undo();
        }
      }

      if (e.key === 'c' || e.key === 'C') {
        useCanvasStore.getState().setMode('create');
      }
      if (e.key === 'v' || e.key === 'V') {
        useCanvasStore.getState().setMode('select');
      }
      if (e.key === 'h' || e.key === 'H') {
        useCanvasStore.getState().setMode('pan');
      }

      if (e.key === 'Escape') {
        selectFollicle(null);
        // Cancel linear creation if in width phase
        setDragState({
          isDragging: false,
          startPoint: null,
          currentPoint: null,
          dragType: null,
          targetId: null,
          resizeHandle: undefined,
          createPhase: undefined,
          lineEndPoint: undefined,
        });
      }

      if (e.key === 'l' || e.key === 'L') {
        useCanvasStore.getState().toggleLabels();
      }

      if (e.key === 'o' || e.key === 'O') {
        useCanvasStore.getState().toggleShapes();
      }

      // Shape type shortcuts
      if (e.key === '1') {
        useCanvasStore.getState().setShapeType('circle');
      }
      if (e.key === '2') {
        useCanvasStore.getState().setShapeType('rectangle');
      }
      if (e.key === '3') {
        useCanvasStore.getState().setShapeType('linear');
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedId, deleteFollicle, selectFollicle, temporalStore]);

  // Get cursor based on mode and drag state
  const getCursor = (): string => {
    if (dragState.isDragging) {
      if (dragState.dragType === 'pan') return 'grabbing';
      if (dragState.dragType === 'move') return 'move';
      if (dragState.dragType === 'resize') return 'nwse-resize';
      return 'crosshair';
    }
    switch (mode) {
      case 'pan': return 'grab';
      case 'create': return 'crosshair';
      case 'select': return 'default';
      default: return 'default';
    }
  };

  return (
    <div
      ref={containerRef}
      className="canvas-container"
    >
      <canvas
        ref={canvasRef}
        width={canvasSize.width}
        height={canvasSize.height}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
        onAuxClick={(e) => e.preventDefault()}
        onContextMenu={(e) => e.preventDefault()}
        style={{ cursor: getCursor() }}
      />
    </div>
  );
};
