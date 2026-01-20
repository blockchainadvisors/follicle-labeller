import { create } from 'zustand';
import { temporal } from 'zundo';
import { Follicle, Point, CircleAnnotation, RectangleAnnotation, LinearAnnotation, ImageId, isCircle, isRectangle, isLinear } from '../types';
import { generateId } from '../utils/id-generator';

interface FollicleState {
  follicles: Follicle[];
  selectedIds: Set<string>;

  // Actions (with imageId for multi-image support)
  addCircle: (imageId: ImageId, center: Point, radius: number) => string;
  addRectangle: (imageId: ImageId, x: number, y: number, width: number, height: number) => string;
  addLinear: (imageId: ImageId, startPoint: Point, endPoint: Point, halfWidth: number) => string;
  updateFollicle: (id: string, updates: Partial<Follicle>) => void;
  deleteFollicle: (id: string) => void;

  // Selection actions
  selectFollicle: (id: string | null) => void;  // Clear and select single (backward compat)
  addToSelection: (id: string) => void;         // Add to current selection
  removeFromSelection: (id: string) => void;    // Remove from selection
  toggleSelection: (id: string) => void;        // Toggle for Ctrl+click
  selectMultiple: (ids: string[]) => void;      // Select multiple at once (for marquee/lasso)
  clearSelection: () => void;                   // Deselect all
  selectAll: (imageId: ImageId) => void;        // Select all on current image

  moveAnnotation: (id: string, deltaX: number, deltaY: number) => void;
  moveSelected: (deltaX: number, deltaY: number) => void;  // Move all selected
  deleteSelected: () => void;                   // Delete all selected
  resizeCircle: (id: string, newRadius: number) => void;
  resizeRectangle: (id: string, x: number, y: number, width: number, height: number) => void;
  resizeLinear: (id: string, startPoint: Point, endPoint: Point, halfWidth: number) => void;
  setLabel: (id: string, label: string) => void;
  setNotes: (id: string, notes: string) => void;

  // Bulk operations
  clearAll: () => void;
  importFollicles: (follicles: Follicle[]) => void;
  deleteFolliclesForImage: (imageId: ImageId) => void;

  // Selectors
  getSelected: () => Follicle | null;           // Returns first selected (backward compat)
  getSelectedFollicles: () => Follicle[];       // Returns all selected
  getFolliclesForImage: (imageId: ImageId) => Follicle[];
}

// Default colors for new annotations (cycles through)
const ANNOTATION_COLORS = [
  '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
  '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F',
  '#74B9FF', '#A29BFE', '#FD79A8', '#00CEC9',
];

let colorIndex = 0;

export const useFollicleStore = create<FollicleState>()(
  temporal(
    (set, get) => ({
      follicles: [],
      selectedIds: new Set<string>(),

      addCircle: (imageId, center, radius) => {
        const id = generateId();
        const newCircle: CircleAnnotation = {
          id,
          imageId,
          shape: 'circle',
          center,
          radius: Math.max(radius, 1),
          label: `Circle ${get().follicles.filter(f => isCircle(f)).length + 1}`,
          notes: '',
          color: ANNOTATION_COLORS[colorIndex++ % ANNOTATION_COLORS.length],
          createdAt: Date.now(),
          updatedAt: Date.now(),
        };

        set(state => ({
          follicles: [...state.follicles, newCircle],
          selectedIds: new Set([id]),
        }));

        return id;
      },

      addRectangle: (imageId, x, y, width, height) => {
        const id = generateId();
        const newRect: RectangleAnnotation = {
          id,
          imageId,
          shape: 'rectangle',
          x,
          y,
          width: Math.max(width, 1),
          height: Math.max(height, 1),
          label: `Rectangle ${get().follicles.filter(f => isRectangle(f)).length + 1}`,
          notes: '',
          color: ANNOTATION_COLORS[colorIndex++ % ANNOTATION_COLORS.length],
          createdAt: Date.now(),
          updatedAt: Date.now(),
        };

        set(state => ({
          follicles: [...state.follicles, newRect],
          selectedIds: new Set([id]),
        }));

        return id;
      },

      addLinear: (imageId, startPoint, endPoint, halfWidth) => {
        const id = generateId();
        const newLinear: LinearAnnotation = {
          id,
          imageId,
          shape: 'linear',
          startPoint,
          endPoint,
          halfWidth: Math.max(halfWidth, 1),
          label: `Linear ${get().follicles.filter(f => isLinear(f)).length + 1}`,
          notes: '',
          color: ANNOTATION_COLORS[colorIndex++ % ANNOTATION_COLORS.length],
          createdAt: Date.now(),
          updatedAt: Date.now(),
        };

        set(state => ({
          follicles: [...state.follicles, newLinear],
          selectedIds: new Set([id]),
        }));

        return id;
      },

      updateFollicle: (id, updates) => {
        set(state => ({
          follicles: state.follicles.map(f =>
            f.id === id
              ? { ...f, ...updates, updatedAt: Date.now() } as Follicle
              : f
          ),
        }));
      },

      deleteFollicle: (id) => {
        set(state => {
          const newSelectedIds = new Set(state.selectedIds);
          newSelectedIds.delete(id);
          return {
            follicles: state.follicles.filter(f => f.id !== id),
            selectedIds: newSelectedIds,
          };
        });
      },

      // Selection actions
      selectFollicle: (id) => {
        set({ selectedIds: id ? new Set([id]) : new Set() });
      },

      addToSelection: (id) => {
        set(state => {
          const newSelectedIds = new Set(state.selectedIds);
          newSelectedIds.add(id);
          return { selectedIds: newSelectedIds };
        });
      },

      removeFromSelection: (id) => {
        set(state => {
          const newSelectedIds = new Set(state.selectedIds);
          newSelectedIds.delete(id);
          return { selectedIds: newSelectedIds };
        });
      },

      toggleSelection: (id) => {
        set(state => {
          const newSelectedIds = new Set(state.selectedIds);
          if (newSelectedIds.has(id)) {
            newSelectedIds.delete(id);
          } else {
            newSelectedIds.add(id);
          }
          return { selectedIds: newSelectedIds };
        });
      },

      selectMultiple: (ids) => {
        set({ selectedIds: new Set(ids) });
      },

      clearSelection: () => {
        set({ selectedIds: new Set() });
      },

      selectAll: (imageId) => {
        const ids = get().follicles
          .filter(f => f.imageId === imageId)
          .map(f => f.id);
        set({ selectedIds: new Set(ids) });
      },

      moveAnnotation: (id, deltaX, deltaY) => {
        const follicle = get().follicles.find(f => f.id === id);
        if (!follicle) return;

        if (isCircle(follicle)) {
          get().updateFollicle(id, {
            center: {
              x: follicle.center.x + deltaX,
              y: follicle.center.y + deltaY,
            },
          } as Partial<CircleAnnotation>);
        } else if (isRectangle(follicle)) {
          get().updateFollicle(id, {
            x: follicle.x + deltaX,
            y: follicle.y + deltaY,
          } as Partial<RectangleAnnotation>);
        } else if (isLinear(follicle)) {
          get().updateFollicle(id, {
            startPoint: {
              x: follicle.startPoint.x + deltaX,
              y: follicle.startPoint.y + deltaY,
            },
            endPoint: {
              x: follicle.endPoint.x + deltaX,
              y: follicle.endPoint.y + deltaY,
            },
          } as Partial<LinearAnnotation>);
        }
      },

      moveSelected: (deltaX, deltaY) => {
        const { selectedIds, follicles, updateFollicle } = get();
        for (const id of selectedIds) {
          const follicle = follicles.find(f => f.id === id);
          if (!follicle) continue;

          if (isCircle(follicle)) {
            updateFollicle(id, {
              center: {
                x: follicle.center.x + deltaX,
                y: follicle.center.y + deltaY,
              },
            } as Partial<CircleAnnotation>);
          } else if (isRectangle(follicle)) {
            updateFollicle(id, {
              x: follicle.x + deltaX,
              y: follicle.y + deltaY,
            } as Partial<RectangleAnnotation>);
          } else if (isLinear(follicle)) {
            updateFollicle(id, {
              startPoint: {
                x: follicle.startPoint.x + deltaX,
                y: follicle.startPoint.y + deltaY,
              },
              endPoint: {
                x: follicle.endPoint.x + deltaX,
                y: follicle.endPoint.y + deltaY,
              },
            } as Partial<LinearAnnotation>);
          }
        }
      },

      deleteSelected: () => {
        set(state => ({
          follicles: state.follicles.filter(f => !state.selectedIds.has(f.id)),
          selectedIds: new Set(),
        }));
      },

      resizeCircle: (id, newRadius) => {
        get().updateFollicle(id, { radius: Math.max(newRadius, 1) } as Partial<CircleAnnotation>);
      },

      resizeRectangle: (id, x, y, width, height) => {
        get().updateFollicle(id, {
          x,
          y,
          width: Math.max(width, 1),
          height: Math.max(height, 1),
        } as Partial<RectangleAnnotation>);
      },

      resizeLinear: (id, startPoint, endPoint, halfWidth) => {
        get().updateFollicle(id, {
          startPoint,
          endPoint,
          halfWidth: Math.max(halfWidth, 1),
        } as Partial<LinearAnnotation>);
      },

      setLabel: (id, label) => {
        get().updateFollicle(id, { label });
      },

      setNotes: (id, notes) => {
        get().updateFollicle(id, { notes });
      },

      clearAll: () => {
        set({ follicles: [], selectedIds: new Set() });
      },

      importFollicles: (follicles) => {
        colorIndex = follicles.length;
        set({ follicles, selectedIds: new Set() });
      },

      deleteFolliclesForImage: (imageId) => {
        set(state => {
          const remainingFollicles = state.follicles.filter(f => f.imageId !== imageId);
          // Remove any selected IDs that were on the removed image
          const removedIds = state.follicles
            .filter(f => f.imageId === imageId)
            .map(f => f.id);
          const newSelectedIds = new Set(state.selectedIds);
          for (const id of removedIds) {
            newSelectedIds.delete(id);
          }
          return {
            follicles: remainingFollicles,
            selectedIds: newSelectedIds,
          };
        });
      },

      getSelected: () => {
        const { follicles, selectedIds } = get();
        // Return first selected for backward compatibility
        const firstId = selectedIds.values().next().value;
        return follicles.find(f => f.id === firstId) || null;
      },

      getSelectedFollicles: () => {
        const { follicles, selectedIds } = get();
        return follicles.filter(f => selectedIds.has(f.id));
      },

      getFolliclesForImage: (imageId) => {
        return get().follicles.filter(f => f.imageId === imageId);
      },
    }),
    {
      limit: 50,
      partialize: (state) => ({
        follicles: state.follicles,
      }),
    }
  )
);

export const useTemporalStore = () => useFollicleStore.temporal;
