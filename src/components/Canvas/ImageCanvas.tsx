import React, { useRef, useEffect, useCallback, useState } from 'react';
import { useFollicleStore, useTemporalStore } from '../../store/follicleStore';
import { useCanvasStore } from '../../store/canvasStore';
import { CanvasRenderer } from './CanvasRenderer';
import { DragState, Point, Follicle, isCircle, isRectangle } from '../../types';
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
  const selectFollicle = useFollicleStore(state => state.selectFollicle);
  const moveAnnotation = useFollicleStore(state => state.moveAnnotation);
  const resizeCircle = useFollicleStore(state => state.resizeCircle);
  const resizeRectangle = useFollicleStore(state => state.resizeRectangle);
  const deleteFollicle = useFollicleStore(state => state.deleteFollicle);

  const viewport = useCanvasStore(state => state.viewport);
  const imageSrc = useCanvasStore(state => state.imageSrc);
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

  // Load image when source changes
  useEffect(() => {
    if (!imageSrc) {
      imageRef.current = null;
      return;
    }

    const img = new Image();
    img.onload = () => {
      imageRef.current = img;
      if (rendererRef.current) {
        rendererRef.current.setImage(img);
      }
      zoomToFit(canvasSize.width, canvasSize.height);
    };
    img.src = imageSrc;
  }, [imageSrc, zoomToFit, canvasSize.width, canvasSize.height]);

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
      setDragState({
        isDragging: true,
        startPoint: point,
        currentPoint: point,
        dragType: 'create',
        targetId: null,
      });
    }
  }, [mode, selectedId, follicles, viewport.scale, getImagePoint, selectFollicle, findAnnotationAtPoint]);

  // Mouse move handler
  const handleMouseMove = useCallback((e: React.MouseEvent) => {
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
      } else {
        const x = Math.min(dragState.startPoint.x, dragState.currentPoint.x);
        const y = Math.min(dragState.startPoint.y, dragState.currentPoint.y);
        const width = Math.abs(dragState.currentPoint.x - dragState.startPoint.x);
        const height = Math.abs(dragState.currentPoint.y - dragState.startPoint.y);
        if (width > 10 && height > 10) {
          addRectangle(x, y, width, height);
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
    });
  }, [dragState, currentShapeType, addCircle, addRectangle, moveAnnotation, resizeCircle, resizeRectangle, follicles]);

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
