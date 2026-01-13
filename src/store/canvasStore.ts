import { create } from 'zustand';
import { InteractionMode, ShapeType } from '../types';

// Canvas store now only manages UI state
// Image/viewport state has moved to projectStore for multi-image support
interface CanvasState {
  // Interaction mode
  mode: InteractionMode;
  currentShapeType: ShapeType;

  // Display options
  showLabels: boolean;
  showShapes: boolean;

  // Help panel
  showHelp: boolean;

  // Canvas reference for screenshots
  canvasRef: HTMLCanvasElement | null;

  // Actions
  setMode: (mode: InteractionMode) => void;
  setShapeType: (shapeType: ShapeType) => void;
  toggleLabels: () => void;
  toggleShapes: () => void;
  toggleHelp: () => void;
  setCanvasRef: (ref: HTMLCanvasElement | null) => void;
}

export const useCanvasStore = create<CanvasState>((set) => ({
  mode: 'create',
  currentShapeType: 'circle',
  showLabels: true,
  showShapes: true,
  showHelp: false,
  canvasRef: null,

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

  toggleHelp: () => {
    set(state => ({ showHelp: !state.showHelp }));
  },

  setCanvasRef: (ref) => {
    set({ canvasRef: ref });
  },
}));
