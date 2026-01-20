import { create } from 'zustand';
import { ImageId, ProjectImage, Viewport, Point } from '../types';
import { useCanvasStore } from './canvasStore';

interface ProjectState {
  // Multi-image state
  images: Map<ImageId, ProjectImage>;
  imageOrder: ImageId[];  // Order for explorer display
  activeImageId: ImageId | null;
  currentProjectPath: string | null;  // Path to the currently loaded/saved project file
  isDirty: boolean;  // Track unsaved changes

  // Actions
  addImage: (image: ProjectImage) => void;
  removeImage: (imageId: ImageId) => void;
  setActiveImage: (imageId: ImageId | null) => void;
  setImageViewport: (imageId: ImageId, viewport: Partial<Viewport>) => void;
  reorderImages: (newOrder: ImageId[]) => void;
  setCurrentProjectPath: (path: string | null) => void;
  setDirty: (dirty: boolean) => void;
  markClean: () => void;
  clearProject: () => void;

  // Viewport actions for active image
  zoom: (delta: number, centerPoint?: Point) => void;
  zoomToFit: (canvasWidth: number, canvasHeight: number) => void;
  resetZoom: () => void;
  pan: (deltaX: number, deltaY: number) => void;

  // Selectors
  getActiveImage: () => ProjectImage | null;
  getImageCount: () => number;
}

const ZOOM_MIN = 0.01;
const ZOOM_MAX = 10;

export const useProjectStore = create<ProjectState>((set, get) => ({
  images: new Map(),
  imageOrder: [],
  activeImageId: null,
  currentProjectPath: null,
  isDirty: false,

  addImage: (image) => {
    set(state => {
      const newImages = new Map(state.images);
      newImages.set(image.id, image);
      const newOrder = [...state.imageOrder, image.id];

      // If this is the first image, make it active
      const newActiveId = state.activeImageId ?? image.id;

      return {
        images: newImages,
        imageOrder: newOrder,
        activeImageId: newActiveId,
        isDirty: true,
      };
    });
  },

  removeImage: (imageId) => {
    set(state => {
      const newImages = new Map(state.images);
      const imageToRemove = newImages.get(imageId);

      // Close bitmap to free memory
      if (imageToRemove?.imageBitmap) {
        imageToRemove.imageBitmap.close();
      }
      // Revoke object URL
      if (imageToRemove?.imageSrc) {
        URL.revokeObjectURL(imageToRemove.imageSrc);
      }

      newImages.delete(imageId);
      const newOrder = state.imageOrder.filter(id => id !== imageId);

      // Update active image if removed
      let newActiveId = state.activeImageId;
      if (state.activeImageId === imageId) {
        // Select next image, or previous, or null
        const removedIndex = state.imageOrder.indexOf(imageId);
        if (newOrder.length > 0) {
          newActiveId = newOrder[Math.min(removedIndex, newOrder.length - 1)];
        } else {
          newActiveId = null;
        }
      }

      return {
        images: newImages,
        imageOrder: newOrder,
        activeImageId: newActiveId,
        isDirty: true,
      };
    });
  },

  setActiveImage: (imageId) => {
    set({ activeImageId: imageId });
    // Reset to select mode when switching images
    useCanvasStore.getState().setMode('select');
  },

  setImageViewport: (imageId, viewport) => {
    set(state => {
      const newImages = new Map(state.images);
      const image = newImages.get(imageId);
      if (image) {
        newImages.set(imageId, {
          ...image,
          viewport: { ...image.viewport, ...viewport },
        });
      }
      return { images: newImages };
    });
  },

  reorderImages: (newOrder) => {
    set(state => {
      // Update sortOrder in images
      const newImages = new Map(state.images);
      newOrder.forEach((id, index) => {
        const image = newImages.get(id);
        if (image) {
          newImages.set(id, { ...image, sortOrder: index });
        }
      });
      return { images: newImages, imageOrder: newOrder };
    });
  },

  setCurrentProjectPath: (path) => {
    set({ currentProjectPath: path });
  },

  setDirty: (dirty) => {
    set({ isDirty: dirty });
  },

  markClean: () => {
    set({ isDirty: false });
  },

  clearProject: () => {
    // Close all bitmaps and revoke URLs
    const state = get();
    state.images.forEach(image => {
      if (image.imageBitmap) {
        image.imageBitmap.close();
      }
      if (image.imageSrc) {
        URL.revokeObjectURL(image.imageSrc);
      }
    });

    set({
      images: new Map(),
      imageOrder: [],
      activeImageId: null,
      currentProjectPath: null,
      isDirty: false,
    });
  },

  // Viewport actions for active image
  zoom: (delta, centerPoint) => {
    const state = get();
    if (!state.activeImageId) return;

    const image = state.images.get(state.activeImageId);
    if (!image) return;

    const currentScale = image.viewport.scale;
    const newScale = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, currentScale * (1 + delta)));

    let newViewport: Partial<Viewport>;
    if (centerPoint) {
      const scaleFactor = newScale / currentScale;
      const newOffsetX = centerPoint.x - (centerPoint.x - image.viewport.offsetX) * scaleFactor;
      const newOffsetY = centerPoint.y - (centerPoint.y - image.viewport.offsetY) * scaleFactor;
      newViewport = { scale: newScale, offsetX: newOffsetX, offsetY: newOffsetY };
    } else {
      newViewport = { scale: newScale };
    }

    get().setImageViewport(state.activeImageId, newViewport);
  },

  zoomToFit: (canvasWidth, canvasHeight) => {
    const state = get();
    if (!state.activeImageId) return;

    const image = state.images.get(state.activeImageId);
    if (!image) return;

    const scaleX = canvasWidth / image.width;
    const scaleY = canvasHeight / image.height;
    const scale = Math.min(scaleX, scaleY) * 0.9; // 90% padding

    const offsetX = (canvasWidth - image.width * scale) / 2;
    const offsetY = (canvasHeight - image.height * scale) / 2;

    get().setImageViewport(state.activeImageId, { scale, offsetX, offsetY });
  },

  resetZoom: () => {
    const state = get();
    if (!state.activeImageId) return;
    get().setImageViewport(state.activeImageId, { offsetX: 0, offsetY: 0, scale: 1 });
  },

  pan: (deltaX, deltaY) => {
    const state = get();
    if (!state.activeImageId) return;

    const image = state.images.get(state.activeImageId);
    if (!image) return;

    get().setImageViewport(state.activeImageId, {
      offsetX: image.viewport.offsetX + deltaX,
      offsetY: image.viewport.offsetY + deltaY,
    });
  },

  // Selectors
  getActiveImage: () => {
    const state = get();
    if (!state.activeImageId) return null;
    return state.images.get(state.activeImageId) ?? null;
  },

  getImageCount: () => {
    return get().images.size;
  },
}));

// Helper to generate unique image IDs
export function generateImageId(): ImageId {
  return `img_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}
