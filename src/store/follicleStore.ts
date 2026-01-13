import { create } from 'zustand';
import { temporal } from 'zundo';
import { Follicle, Point, CircleAnnotation, RectangleAnnotation, LinearAnnotation, isCircle, isRectangle, isLinear } from '../types';
import { generateId } from '../utils/id-generator';

interface FollicleState {
  follicles: Follicle[];
  selectedId: string | null;

  // Actions
  addCircle: (center: Point, radius: number) => string;
  addRectangle: (x: number, y: number, width: number, height: number) => string;
  addLinear: (startPoint: Point, endPoint: Point, halfWidth: number) => string;
  updateFollicle: (id: string, updates: Partial<Follicle>) => void;
  deleteFollicle: (id: string) => void;
  selectFollicle: (id: string | null) => void;
  moveAnnotation: (id: string, deltaX: number, deltaY: number) => void;
  resizeCircle: (id: string, newRadius: number) => void;
  resizeRectangle: (id: string, x: number, y: number, width: number, height: number) => void;
  resizeLinear: (id: string, startPoint: Point, endPoint: Point, halfWidth: number) => void;
  setLabel: (id: string, label: string) => void;
  setNotes: (id: string, notes: string) => void;

  // Bulk operations
  clearAll: () => void;
  importFollicles: (follicles: Follicle[]) => void;

  // Computed
  getSelected: () => Follicle | null;
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
      selectedId: null,

      addCircle: (center, radius) => {
        const id = generateId();
        const newCircle: CircleAnnotation = {
          id,
          shape: 'circle',
          center,
          radius: Math.max(radius, 5),
          label: `Circle ${get().follicles.filter(f => isCircle(f)).length + 1}`,
          notes: '',
          color: ANNOTATION_COLORS[colorIndex++ % ANNOTATION_COLORS.length],
          createdAt: Date.now(),
          updatedAt: Date.now(),
        };

        set(state => ({
          follicles: [...state.follicles, newCircle],
          selectedId: id,
        }));

        return id;
      },

      addRectangle: (x, y, width, height) => {
        const id = generateId();
        const newRect: RectangleAnnotation = {
          id,
          shape: 'rectangle',
          x,
          y,
          width: Math.max(width, 10),
          height: Math.max(height, 10),
          label: `Rectangle ${get().follicles.filter(f => isRectangle(f)).length + 1}`,
          notes: '',
          color: ANNOTATION_COLORS[colorIndex++ % ANNOTATION_COLORS.length],
          createdAt: Date.now(),
          updatedAt: Date.now(),
        };

        set(state => ({
          follicles: [...state.follicles, newRect],
          selectedId: id,
        }));

        return id;
      },

      addLinear: (startPoint, endPoint, halfWidth) => {
        const id = generateId();
        const newLinear: LinearAnnotation = {
          id,
          shape: 'linear',
          startPoint,
          endPoint,
          halfWidth: Math.max(halfWidth, 5),
          label: `Linear ${get().follicles.filter(f => isLinear(f)).length + 1}`,
          notes: '',
          color: ANNOTATION_COLORS[colorIndex++ % ANNOTATION_COLORS.length],
          createdAt: Date.now(),
          updatedAt: Date.now(),
        };

        set(state => ({
          follicles: [...state.follicles, newLinear],
          selectedId: id,
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
        set(state => ({
          follicles: state.follicles.filter(f => f.id !== id),
          selectedId: state.selectedId === id ? null : state.selectedId,
        }));
      },

      selectFollicle: (id) => {
        set({ selectedId: id });
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

      resizeCircle: (id, newRadius) => {
        get().updateFollicle(id, { radius: Math.max(newRadius, 5) } as Partial<CircleAnnotation>);
      },

      resizeRectangle: (id, x, y, width, height) => {
        get().updateFollicle(id, {
          x,
          y,
          width: Math.max(width, 10),
          height: Math.max(height, 10),
        } as Partial<RectangleAnnotation>);
      },

      resizeLinear: (id, startPoint, endPoint, halfWidth) => {
        get().updateFollicle(id, {
          startPoint,
          endPoint,
          halfWidth: Math.max(halfWidth, 5),
        } as Partial<LinearAnnotation>);
      },

      setLabel: (id, label) => {
        get().updateFollicle(id, { label });
      },

      setNotes: (id, notes) => {
        get().updateFollicle(id, { notes });
      },

      clearAll: () => {
        set({ follicles: [], selectedId: null });
      },

      importFollicles: (follicles) => {
        colorIndex = follicles.length;
        set({ follicles, selectedId: null });
      },

      getSelected: () => {
        const { follicles, selectedId } = get();
        return follicles.find(f => f.id === selectedId) || null;
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
