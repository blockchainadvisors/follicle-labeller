import React, { useRef, useEffect, useCallback, useState } from 'react';
import { useFollicleStore, useTemporalStore } from '../../store/follicleStore';
import { useCanvasStore } from '../../store/canvasStore';
import { CanvasRenderer } from './CanvasRenderer';
import { DragState, Point, Follicle } from '../../types';
import {
  screenToImage,
  distance,
  isPointInCircle,
  isPointOnCircleEdge,
} from '../../utils/coordinate-transform';

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
  });

  const [canvasSize, setCanvasSize] = useState({ width: 800, height: 600 });

  // Store subscriptions
  const follicles = useFollicleStore(state => state.follicles);
  const selectedId = useFollicleStore(state => state.selectedId);
  const addFollicle = useFollicleStore(state => state.addFollicle);
  const selectFollicle = useFollicleStore(state => state.selectFollicle);
  const moveFollicle = useFollicleStore(state => state.moveFollicle);
  const resizeFollicle = useFollicleStore(state => state.resizeFollicle);
  const deleteFollicle = useFollicleStore(state => state.deleteFollicle);

  const viewport = useCanvasStore(state => state.viewport);
  const imageSrc = useCanvasStore(state => state.imageSrc);
  const mode = useCanvasStore(state => state.mode);
  const pan = useCanvasStore(state => state.pan);
  const zoom = useCanvasStore(state => state.zoom);
  const zoomToFit = useCanvasStore(state => state.zoomToFit);
  const showLabels = useCanvasStore(state => state.showLabels);
  const showCircles = useCanvasStore(state => state.showCircles);

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
      // Auto-fit image to canvas
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
        showCircles
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
  }, [viewport, follicles, selectedId, dragState, canvasSize, showLabels, showCircles]);

  // Get image coordinates from mouse event
  const getImagePoint = useCallback((e: React.MouseEvent): Point => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };

    const rect = canvas.getBoundingClientRect();
    return screenToImage(e.clientX, e.clientY, viewport, rect);
  }, [viewport]);

  // Find follicle at a point (for selection)
  const findFollicleAtPoint = useCallback((point: Point): Follicle | null => {
    // Check in reverse order (top-most first)
    for (let i = follicles.length - 1; i >= 0; i--) {
      const f = follicles[i];
      if (isPointInCircle(point, f.center, f.radius)) {
        return f;
      }
    }
    return null;
  }, [follicles]);

  // Mouse down handler
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    const point = getImagePoint(e);

    if (mode === 'pan') {
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
      // Check if clicking on resize handle of selected follicle
      const selected = follicles.find(f => f.id === selectedId);
      if (selected) {
        const handlePoint = { x: selected.center.x + selected.radius, y: selected.center.y };
        const handleDistance = distance(point, handlePoint);
        if (handleDistance <= 10 / viewport.scale) {
          setDragState({
            isDragging: true,
            startPoint: selected.center,
            currentPoint: point,
            dragType: 'resize',
            targetId: selected.id,
          });
          return;
        }

        // Check if clicking inside selected follicle (for move)
        if (isPointInCircle(point, selected.center, selected.radius)) {
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

      // Check if clicking inside any follicle
      const clicked = findFollicleAtPoint(point);
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
  }, [mode, selectedId, follicles, viewport.scale, getImagePoint, selectFollicle, findFollicleAtPoint]);

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
      const radius = distance(dragState.startPoint, dragState.currentPoint);
      if (radius > 5) {
        addFollicle(dragState.startPoint, radius);
      }
    }

    if (dragState.dragType === 'move' && dragState.targetId && dragState.currentPoint && dragState.startPoint) {
      const target = follicles.find(f => f.id === dragState.targetId);
      if (target) {
        const deltaX = dragState.currentPoint.x - dragState.startPoint.x;
        const deltaY = dragState.currentPoint.y - dragState.startPoint.y;
        const newCenter = {
          x: target.center.x + deltaX,
          y: target.center.y + deltaY,
        };
        moveFollicle(dragState.targetId, newCenter);
      }
    }

    if (dragState.dragType === 'resize' && dragState.targetId && dragState.startPoint && dragState.currentPoint) {
      const newRadius = distance(dragState.startPoint, dragState.currentPoint);
      resizeFollicle(dragState.targetId, newRadius);
    }

    setDragState({
      isDragging: false,
      startPoint: null,
      currentPoint: null,
      dragType: null,
      targetId: null,
    });
  }, [dragState, addFollicle, moveFollicle, resizeFollicle, follicles]);

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
      // Delete selected follicle
      if ((e.key === 'Delete' || e.key === 'Backspace') && selectedId) {
        e.preventDefault();
        deleteFollicle(selectedId);
      }

      // Undo/Redo
      if ((e.metaKey || e.ctrlKey) && e.key === 'z') {
        e.preventDefault();
        if (e.shiftKey) {
          temporalStore.getState().redo();
        } else {
          temporalStore.getState().undo();
        }
      }

      // Mode shortcuts
      if (e.key === 'c' || e.key === 'C') {
        useCanvasStore.getState().setMode('create');
      }
      if (e.key === 'v' || e.key === 'V') {
        useCanvasStore.getState().setMode('select');
      }
      if (e.key === 'h' || e.key === 'H') {
        useCanvasStore.getState().setMode('pan');
      }

      // Escape to deselect
      if (e.key === 'Escape') {
        selectFollicle(null);
      }

      // Toggle labels
      if (e.key === 'l' || e.key === 'L') {
        useCanvasStore.getState().toggleLabels();
      }

      // Toggle circles
      if (e.key === 'o' || e.key === 'O') {
        useCanvasStore.getState().toggleCircles();
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
        style={{ cursor: getCursor() }}
      />
    </div>
  );
};
