import React, { useRef, useEffect, useCallback, useState } from 'react';
import { ImagePlus } from 'lucide-react';
import { useFollicleStore, useTemporalStore } from '../../store/follicleStore';
import { useCanvasStore } from '../../store/canvasStore';
import { useProjectStore, generateImageId } from '../../store/projectStore';
import { useThemeStore } from '../../store/themeStore';
import { CanvasRenderer } from './CanvasRenderer';
import { DragState, Point, Follicle, LinearAnnotation, ProjectImage, isCircle, isRectangle, isLinear } from '../../types';
import {
  screenToImage,
  distance,
  isPointInCircle,
} from '../../utils/coordinate-transform';
import {
  createSelectionBounds,
  getFolliclesInBounds,
  getFolliclesInPolygon,
  simplifyPath,
} from '../../utils/selection-geometry';

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
  const allFollicles = useFollicleStore(state => state.follicles);
  const selectedIds = useFollicleStore(state => state.selectedIds);
  const addCircle = useFollicleStore(state => state.addCircle);
  const addRectangle = useFollicleStore(state => state.addRectangle);
  const addLinear = useFollicleStore(state => state.addLinear);
  const selectFollicle = useFollicleStore(state => state.selectFollicle);
  const toggleSelection = useFollicleStore(state => state.toggleSelection);
  const selectMultiple = useFollicleStore(state => state.selectMultiple);
  const clearSelection = useFollicleStore(state => state.clearSelection);
  const selectAll = useFollicleStore(state => state.selectAll);
  const moveSelected = useFollicleStore(state => state.moveSelected);
  const deleteSelected = useFollicleStore(state => state.deleteSelected);
  const moveAnnotation = useFollicleStore(state => state.moveAnnotation);
  const resizeCircle = useFollicleStore(state => state.resizeCircle);
  const resizeRectangle = useFollicleStore(state => state.resizeRectangle);
  const resizeLinear = useFollicleStore(state => state.resizeLinear);

  // Project store for multi-image support
  const images = useProjectStore(state => state.images);
  const imageOrder = useProjectStore(state => state.imageOrder);
  const activeImageId = useProjectStore(state => state.activeImageId);
  const addImage = useProjectStore(state => state.addImage);
  const pan = useProjectStore(state => state.pan);
  const zoom = useProjectStore(state => state.zoom);
  const zoomToFit = useProjectStore(state => state.zoomToFit);

  // Get active image data
  const activeImage = activeImageId ? images.get(activeImageId) : null;
  const viewport = activeImage?.viewport ?? { offsetX: 0, offsetY: 0, scale: 1 };
  const imageBitmap = activeImage?.imageBitmap ?? null;

  // Filter follicles by active image
  const follicles = activeImageId
    ? allFollicles.filter(f => f.imageId === activeImageId)
    : [];

  const mode = useCanvasStore(state => state.mode);
  const currentShapeType = useCanvasStore(state => state.currentShapeType);
  const selectionToolType = useCanvasStore(state => state.selectionToolType);
  const showLabels = useCanvasStore(state => state.showLabels);
  const showShapes = useCanvasStore(state => state.showShapes);

  // Subscribe to theme changes to trigger canvas re-render
  const themeBackground = useThemeStore(state => state.background);

  const temporalStore = useTemporalStore();

  // Resize canvas to fit container - use ResizeObserver to detect layout changes
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const updateSize = () => {
      const rect = container.getBoundingClientRect();
      // Only update if size actually changed to avoid unnecessary re-renders
      setCanvasSize(prev => {
        if (prev.width !== rect.width || prev.height !== rect.height) {
          return { width: rect.width, height: rect.height };
        }
        return prev;
      });
    };

    // Use ResizeObserver to detect container size changes (e.g., when sidebar appears)
    const resizeObserver = new ResizeObserver(updateSize);
    resizeObserver.observe(container);

    // Initial size
    updateSize();

    return () => resizeObserver.disconnect();
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
        selectedIds,
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
  }, [viewport, follicles, selectedIds, dragState, canvasSize, showLabels, showShapes, currentShapeType, themeBackground]);

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
      const isCtrlPressed = e.ctrlKey || e.metaKey;

      // First, check for resize handles (only for single selection)
      if (selectedIds.size === 1) {
        const selectedId = selectedIds.values().next().value;
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
        }
      }

      // Check if clicking inside any selected annotation (for move)
      const clickedSelected = [...selectedIds].map(id => follicles.find(f => f.id === id)).find(f => {
        if (!f) return false;
        if (isCircle(f) && isPointInCircle(point, f.center, f.radius)) return true;
        if (isRectangle(f) && isPointInRectangle(point, f.x, f.y, f.width, f.height)) return true;
        if (isLinear(f) && isPointInLinear(point, f)) return true;
        return false;
      });

      if (clickedSelected) {
        if (isCtrlPressed) {
          // Ctrl+click on selected item: remove from selection
          toggleSelection(clickedSelected.id);
        } else {
          // Clicking on an already selected item - start move (for all selected)
          setDragState({
            isDragging: true,
            startPoint: point,
            currentPoint: point,
            dragType: 'move',
            targetId: null, // null indicates moving all selected
          });
        }
        return;
      }

      // Check if clicking inside any annotation (not currently selected)
      const clicked = findAnnotationAtPoint(point);
      if (clicked) {
        if (isCtrlPressed) {
          // Ctrl+click: add to selection
          toggleSelection(clicked.id);
        } else {
          // Regular click: select only this one and start move
          selectFollicle(clicked.id);
          setDragState({
            isDragging: true,
            startPoint: point,
            currentPoint: point,
            dragType: 'move',
            targetId: clicked.id,
          });
        }
        return;
      }

      // Clicking on empty area - start marquee/lasso selection or clear
      if (selectionToolType === 'marquee') {
        if (!isCtrlPressed) {
          clearSelection();
        }
        setDragState({
          isDragging: true,
          startPoint: point,
          currentPoint: point,
          dragType: 'marquee',
          targetId: null,
        });
        return;
      }

      if (selectionToolType === 'lasso') {
        if (!isCtrlPressed) {
          clearSelection();
        }
        setDragState({
          isDragging: true,
          startPoint: point,
          currentPoint: point,
          dragType: 'lasso',
          targetId: null,
          lassoPoints: [point],
        });
        return;
      }

      // Fallback: clicking on empty area just clears selection
      if (!isCtrlPressed) {
        clearSelection();
      }
      return;
    }

    if (mode === 'create') {
      // Check if we're in the middle of creating a shape (second click to finalize)
      if (dragState.dragType === 'create' && dragState.startPoint && activeImageId) {
        if (currentShapeType === 'circle') {
          // Second click - finalize circle
          const radius = distance(dragState.startPoint, point);
          if (radius > 1) {
            addCircle(activeImageId, dragState.startPoint, radius);
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
        } else if (currentShapeType === 'rectangle') {
          // Second click - finalize rectangle
          const x = Math.min(dragState.startPoint.x, point.x);
          const y = Math.min(dragState.startPoint.y, point.y);
          const width = Math.abs(point.x - dragState.startPoint.x);
          const height = Math.abs(point.y - dragState.startPoint.y);
          if (width > 1 && height > 1) {
            addRectangle(activeImageId, x, y, width, height);
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
        } else if (currentShapeType === 'linear') {
          if (dragState.createPhase === 'line') {
            // Second click - finalize line, move to width phase
            const lineLength = distance(dragState.startPoint, point);
            if (lineLength > 1) {
              setDragState({
                isDragging: false,
                startPoint: dragState.startPoint,
                currentPoint: point,
                dragType: 'create',
                targetId: null,
                createPhase: 'width',
                lineEndPoint: point,
              });
            } else {
              // Line too short, cancel
              setDragState({
                isDragging: false,
                startPoint: null,
                currentPoint: null,
                dragType: null,
                targetId: null,
                createPhase: undefined,
                lineEndPoint: undefined,
              });
            }
            return;
          } else if (dragState.createPhase === 'width' && dragState.lineEndPoint) {
            // Third click - finalize linear shape
            const halfWidth = pointToLineDistance(point, dragState.startPoint, dragState.lineEndPoint);
            if (halfWidth > 1) {
              addLinear(activeImageId, dragState.startPoint, dragState.lineEndPoint, halfWidth);
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
        }
      }

      // First click - start new shape creation
      setDragState({
        isDragging: false,  // Not dragging, just tracking mouse
        startPoint: point,
        currentPoint: point,
        dragType: 'create',
        targetId: null,
        createPhase: currentShapeType === 'linear' ? 'line' : undefined,
      });
    }
  }, [mode, selectedIds, follicles, viewport.scale, getImagePoint, selectFollicle, toggleSelection, clearSelection, findAnnotationAtPoint, currentShapeType, selectionToolType, dragState, addLinear, addCircle, addRectangle, activeImageId]);

  // Mouse move handler
  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    // Handle shape creation - track mouse even when not dragging (click-to-click mode)
    if (dragState.dragType === 'create' && dragState.startPoint) {
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

    // Handle lasso - append points to the path
    if (dragState.dragType === 'lasso' && dragState.lassoPoints) {
      setDragState(prev => ({
        ...prev,
        currentPoint: point,
        lassoPoints: [...(prev.lassoPoints || []), point],
      }));
      return;
    }

    setDragState(prev => ({ ...prev, currentPoint: point }));
  }, [dragState, pan, getImagePoint]);

  // Mouse up handler
  const handleMouseUp = useCallback(() => {
    if (!dragState.isDragging) return;

    // Shape creation is now click-to-click, so don't finalize on mouse up
    // Just handle move, resize, pan, marquee, and lasso operations

    // Handle marquee selection
    if (dragState.dragType === 'marquee' && dragState.startPoint && dragState.currentPoint) {
      const bounds = createSelectionBounds(dragState.startPoint, dragState.currentPoint);
      const foundFollicles = getFolliclesInBounds(follicles, bounds);
      const foundIds = foundFollicles.map(f => f.id);
      if (foundIds.length > 0) {
        selectMultiple(foundIds);
      } else {
        clearSelection();
      }
    }

    // Handle lasso selection
    if (dragState.dragType === 'lasso' && dragState.lassoPoints && dragState.lassoPoints.length >= 3) {
      const simplifiedPath = simplifyPath(dragState.lassoPoints, 3);
      const foundFollicles = getFolliclesInPolygon(follicles, simplifiedPath);
      const foundIds = foundFollicles.map(f => f.id);
      if (foundIds.length > 0) {
        selectMultiple(foundIds);
      } else {
        clearSelection();
      }
    }

    // Handle multi-selection move (targetId is null)
    if (dragState.dragType === 'move' && dragState.targetId === null && dragState.currentPoint && dragState.startPoint) {
      const deltaX = dragState.currentPoint.x - dragState.startPoint.x;
      const deltaY = dragState.currentPoint.y - dragState.startPoint.y;
      moveSelected(deltaX, deltaY);
    }

    // Handle single annotation move (targetId is set)
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
      lassoPoints: undefined,
    });
  }, [dragState, moveAnnotation, moveSelected, resizeCircle, resizeRectangle, resizeLinear, follicles, selectMultiple, clearSelection]);

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

  // Open image dialog handler
  const handleOpenImage = useCallback(async () => {
    try {
      const result = await window.electronAPI.openImageDialog();
      if (result) {
        const blob = new Blob([result.data]);
        const url = URL.createObjectURL(blob);
        const bitmap = await createImageBitmap(blob);

        const newImage: ProjectImage = {
          id: generateImageId(),
          fileName: result.fileName,
          width: bitmap.width,
          height: bitmap.height,
          imageData: result.data,
          imageBitmap: bitmap,
          imageSrc: url,
          viewport: { offsetX: 0, offsetY: 0, scale: 1 },
          createdAt: Date.now(),
          sortOrder: imageOrder.length,
        };

        addImage(newImage);
      }
    } catch (error) {
      console.error('Failed to open image:', error);
    }
  }, [addImage, imageOrder.length]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't handle shortcuts when user is typing in an input field
      const target = e.target as HTMLElement | null;
      const tagName = target?.tagName?.toLowerCase();
      const isInputFocused = tagName === 'input' ||
        tagName === 'textarea' ||
        tagName === 'select' ||
        target?.isContentEditable === true;

      // Skip all shortcuts when typing in input fields
      if (isInputFocused) {
        // Only allow Escape to blur the input
        if (e.key === 'Escape' && target) {
          target.blur();
        }
        return;
      }

      if ((e.key === 'Delete' || e.key === 'Backspace') && selectedIds.size > 0) {
        e.preventDefault();
        deleteSelected();
      }

      if ((e.metaKey || e.ctrlKey) && e.key === 'z') {
        e.preventDefault();
        if (e.shiftKey) {
          temporalStore.getState().redo();
        } else {
          temporalStore.getState().undo();
        }
      }

      // Ctrl+A to select all on current image
      if ((e.metaKey || e.ctrlKey) && e.key === 'a' && activeImageId) {
        e.preventDefault();
        selectAll(activeImageId);
        return;
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

      // Selection tool shortcuts
      if (e.key === 'm' || e.key === 'M') {
        useCanvasStore.getState().setMode('select');
        useCanvasStore.getState().setSelectionToolType('marquee');
      }
      if (e.key === 'f' || e.key === 'F') {
        useCanvasStore.getState().setMode('select');
        useCanvasStore.getState().setSelectionToolType('lasso');
      }

      // View toggles
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

      // Help toggle
      if (e.key === '?') {
        useCanvasStore.getState().toggleHelp();
      }

      // Escape to deselect and cancel operations
      if (e.key === 'Escape') {
        clearSelection();
        setDragState({
          isDragging: false,
          startPoint: null,
          currentPoint: null,
          dragType: null,
          targetId: null,
          resizeHandle: undefined,
          createPhase: undefined,
          lineEndPoint: undefined,
          lassoPoints: undefined,
        });
      }

      // Image navigation: Ctrl+Tab / Ctrl+Shift+Tab
      if ((e.metaKey || e.ctrlKey) && e.key === 'Tab') {
        e.preventDefault();
        const projectState = useProjectStore.getState();
        const { imageOrder, activeImageId, setActiveImage } = projectState;
        if (imageOrder.length <= 1 || !activeImageId) return;

        const currentIndex = imageOrder.indexOf(activeImageId);
        let newIndex: number;
        if (e.shiftKey) {
          // Previous image
          newIndex = currentIndex <= 0 ? imageOrder.length - 1 : currentIndex - 1;
        } else {
          // Next image
          newIndex = currentIndex >= imageOrder.length - 1 ? 0 : currentIndex + 1;
        }
        setActiveImage(imageOrder[newIndex]);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedIds, deleteSelected, clearSelection, selectAll, activeImageId, temporalStore]);

  // Get cursor based on mode and drag state
  const getCursor = (): string => {
    if (dragState.isDragging) {
      if (dragState.dragType === 'pan') return 'grabbing';
      if (dragState.dragType === 'move') return 'move';
      if (dragState.dragType === 'resize') return 'nwse-resize';
      if (dragState.dragType === 'marquee') return 'crosshair';
      if (dragState.dragType === 'lasso') return 'crosshair';
      return 'crosshair';
    }
    switch (mode) {
      case 'pan': return 'grab';
      case 'create': return 'crosshair';
      case 'select':
        if (selectionToolType === 'marquee' || selectionToolType === 'lasso') {
          return 'crosshair';
        }
        return 'default';
      default: return 'default';
    }
  };

  const hasImage = images.size > 0;

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
        style={{ cursor: hasImage ? getCursor() : 'pointer' }}
      />
      {!hasImage && (
        <div className="canvas-empty-state" onClick={handleOpenImage}>
          <ImagePlus size={64} strokeWidth={1.5} />
          <p>Click to open an image</p>
          <span>or use File → Open Image (Ctrl+O)</span>
        </div>
      )}
    </div>
  );
};
