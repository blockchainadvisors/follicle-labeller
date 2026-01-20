import { create } from 'zustand';
import { InteractionMode, ShapeType, SelectionToolType } from '../types';

// Canvas store now only manages UI state
// Image/viewport state has moved to projectStore for multi-image support
interface CanvasState {
  // Interaction mode
  mode: InteractionMode;
  currentShapeType: ShapeType;

  // Selection tool type (click, marquee, lasso)
  selectionToolType: SelectionToolType;

  // Display options
  showLabels: boolean;
  showShapes: boolean;

  // Help panel
  showHelp: boolean;

  // Actions
  setMode: (mode: InteractionMode) => void;
  setShapeType: (shapeType: ShapeType) => void;
  setSelectionToolType: (type: SelectionToolType) => void;
  toggleLabels: () => void;
  toggleShapes: () => void;
  toggleHelp: () => void;
}

export const useCanvasStore = create<CanvasState>((set) => ({
  mode: 'create',
  currentShapeType: 'circle',
  selectionToolType: 'marquee',
  showLabels: true,
  showShapes: true,
  showHelp: false,

  setMode: (mode) => {
    set({ mode });
  },

  setShapeType: (shapeType) => {
    set({ currentShapeType: shapeType });
  },

  setSelectionToolType: (type) => {
    set({ selectionToolType: type });
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
}));
