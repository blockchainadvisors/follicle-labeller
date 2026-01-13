import { create } from 'zustand';
import { Viewport, InteractionMode, Point } from '../types';

interface CanvasState {
  // Viewport
  viewport: Viewport;

  // Image state
  imageLoaded: boolean;
  imageWidth: number;
  imageHeight: number;
  imageSrc: string | null;
  imageData: ArrayBuffer | null;
  fileName: string | null;

  // Interaction mode
  mode: InteractionMode;

  // Display options
  showLabels: boolean;
  showCircles: boolean;

  // Actions
  setViewport: (viewport: Partial<Viewport>) => void;
  zoom: (delta: number, centerPoint?: Point) => void;
  zoomToFit: (canvasWidth: number, canvasHeight: number) => void;
  resetZoom: () => void;
  pan: (deltaX: number, deltaY: number) => void;
  setImage: (src: string, width: number, height: number, fileName: string, imageData: ArrayBuffer) => void;
  clearImage: () => void;
  setMode: (mode: InteractionMode) => void;
  toggleLabels: () => void;
  toggleCircles: () => void;
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
  fileName: null,
  mode: 'create',
  showLabels: true,
  showCircles: true,

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

  setImage: (src, width, height, fileName, imageData) => {
    set({
      imageSrc: src,
      imageWidth: width,
      imageHeight: height,
      imageLoaded: true,
      imageData,
      fileName,
      viewport: { offsetX: 0, offsetY: 0, scale: 1 },
    });
  },

  clearImage: () => {
    set({
      imageSrc: null,
      imageData: null,
      imageWidth: 0,
      imageHeight: 0,
      imageLoaded: false,
      fileName: null,
    });
  },

  setMode: (mode) => {
    set({ mode });
  },

  toggleLabels: () => {
    set(state => ({ showLabels: !state.showLabels }));
  },

  toggleCircles: () => {
    set(state => ({ showCircles: !state.showCircles }));
  },
}));
