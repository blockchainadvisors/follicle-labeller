import { create } from 'zustand';
import { InteractionMode, ShapeType, SelectionToolType } from '../types';
import type { ColormapType, HeatmapOptions } from '../services/heatmapGenerator';

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

  // Heatmap display options
  showHeatmap: boolean;
  heatmapOptions: HeatmapOptions;

  // Statistics panel
  showStatistics: boolean;

  // Actions
  setMode: (mode: InteractionMode) => void;
  setShapeType: (shapeType: ShapeType) => void;
  setSelectionToolType: (type: SelectionToolType) => void;
  toggleLabels: () => void;
  toggleShapes: () => void;
  toggleHelp: () => void;
  toggleHeatmap: () => void;
  setHeatmapOptions: (options: Partial<HeatmapOptions>) => void;
  toggleStatistics: () => void;
}

export const useCanvasStore = create<CanvasState>((set) => ({
  mode: 'select',
  currentShapeType: 'circle',
  selectionToolType: 'marquee',
  showLabels: true,
  showShapes: true,
  showHelp: false,
  showHeatmap: false,
  heatmapOptions: {
    sigma: 30,
    colormap: 'jet' as ColormapType,
    alpha: 0.5,
    maxValue: 0,
    intensityScale: 1.0,
  },
  showStatistics: false,

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

  toggleHeatmap: () => {
    set(state => ({ showHeatmap: !state.showHeatmap }));
  },

  setHeatmapOptions: (options) => {
    set(state => ({
      heatmapOptions: { ...state.heatmapOptions, ...options },
    }));
  },

  toggleStatistics: () => {
    set(state => ({ showStatistics: !state.showStatistics }));
  },
}));
