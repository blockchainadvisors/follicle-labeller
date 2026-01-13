import { create } from 'zustand';
import { Viewport, InteractionMode, Point, ShapeType } from '../types';

interface CanvasState {
  // Viewport
  viewport: Viewport;

  // Image state
  imageLoaded: boolean;
  imageWidth: number;
  imageHeight: number;
  imageSrc: string | null;
  imageData: ArrayBuffer | null;
  imageBitmap: ImageBitmap | null;  // Pre-decoded bitmap for smooth rendering
  fileName: string | null;

  // Interaction mode
  mode: InteractionMode;
  currentShapeType: ShapeType;

  // Display options
  showLabels: boolean;
  showShapes: boolean;

  // Actions
  setViewport: (viewport: Partial<Viewport>) => void;
  zoom: (delta: number, centerPoint?: Point) => void;
  zoomToFit: (canvasWidth: number, canvasHeight: number) => void;
  resetZoom: () => void;
  pan: (deltaX: number, deltaY: number) => void;
  setImage: (src: string, width: number, height: number, fileName: string, imageData: ArrayBuffer, bitmap: ImageBitmap) => void;
  clearImage: () => void;
  setMode: (mode: InteractionMode) => void;
  setShapeType: (shapeType: ShapeType) => void;
  toggleLabels: () => void;
  toggleShapes: () => void;
}

const ZOOM_MIN = 0.1;
const ZOOM_MAX = 10;

export const useCanvasStore = create<CanvasState>((set, get) => ({
  viewport: { offsetX: 0, offsetY: 0, scale: 1 },
  imageLoaded: false,
  imageWidth: 0,
  imageHeight: 0,
  imageSrc: null,
  imageData: null,
  imageBitmap: null,
  fileName: null,
  mode: 'create',
  currentShapeType: 'circle',
  showLabels: true,
  showShapes: true,

  setViewport: (viewport) => {
    set(state => ({
      viewport: { ...state.viewport, ...viewport },
    }));
  },

  zoom: (delta, centerPoint) => {
    set(state => {
      const newScale = Math.min(
        ZOOM_MAX,
        Math.max(ZOOM_MIN, state.viewport.scale * (1 + delta))
      );

      // If center point provided, zoom towards that point
      if (centerPoint) {
        const scaleFactor = newScale / state.viewport.scale;
        const newOffsetX = centerPoint.x - (centerPoint.x - state.viewport.offsetX) * scaleFactor;
        const newOffsetY = centerPoint.y - (centerPoint.y - state.viewport.offsetY) * scaleFactor;

        return {
          viewport: {
            scale: newScale,
            offsetX: newOffsetX,
            offsetY: newOffsetY,
          },
        };
      }

      return {
        viewport: { ...state.viewport, scale: newScale },
      };
    });
  },

  zoomToFit: (canvasWidth, canvasHeight) => {
    const state = get();
    if (!state.imageLoaded) return;

    const scaleX = canvasWidth / state.imageWidth;
    const scaleY = canvasHeight / state.imageHeight;
    const scale = Math.min(scaleX, scaleY) * 0.9; // 90% to add some padding

    const offsetX = (canvasWidth - state.imageWidth * scale) / 2;
    const offsetY = (canvasHeight - state.imageHeight * scale) / 2;

    set({
      viewport: { scale, offsetX, offsetY },
    });
  },

  resetZoom: () => {
    set({
      viewport: { offsetX: 0, offsetY: 0, scale: 1 },
    });
  },

  pan: (deltaX, deltaY) => {
    set(state => ({
      viewport: {
        ...state.viewport,
        offsetX: state.viewport.offsetX + deltaX,
        offsetY: state.viewport.offsetY + deltaY,
      },
    }));
  },

  setImage: (src, width, height, fileName, imageData, bitmap) => {
    // Close previous bitmap if exists to free memory
    const prevBitmap = get().imageBitmap;
    if (prevBitmap) {
      prevBitmap.close();
    }
    set({
      imageSrc: src,
      imageWidth: width,
      imageHeight: height,
      imageLoaded: true,
      imageData,
      imageBitmap: bitmap,
      fileName,
      viewport: { offsetX: 0, offsetY: 0, scale: 1 },
    });
  },

  clearImage: () => {
    // Close bitmap to free memory
    const bitmap = get().imageBitmap;
    if (bitmap) {
      bitmap.close();
    }
    set({
      imageSrc: null,
      imageData: null,
      imageBitmap: null,
      imageWidth: 0,
      imageHeight: 0,
      imageLoaded: false,
      fileName: null,
    });
  },

  setMode: (mode) => {
    set({ mode });
  },

  setShapeType: (shapeType) => {
    set({ currentShapeType: shapeType });
  },

  toggleLabels: () => {
    set(state => ({ showLabels: !state.showLabels }));
  },

  toggleShapes: () => {
    set(state => ({ showShapes: !state.showShapes }));
  },
}));
