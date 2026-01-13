import { create } from 'zustand';
import { temporal } from 'zundo';
import { Follicle, Point } from '../types';
import { generateId } from '../utils/id-generator';

interface FollicleState {
  follicles: Follicle[];
  selectedId: string | null;

  // Actions
  addFollicle: (center: Point, radius: number) => string;
  updateFollicle: (id: string, updates: Partial<Follicle>) => void;
  deleteFollicle: (id: string) => void;
  selectFollicle: (id: string | null) => void;
  moveFollicle: (id: string, newCenter: Point) => void;
  resizeFollicle: (id: string, newRadius: number) => void;
  setLabel: (id: string, label: string) => void;
  setNotes: (id: string, notes: string) => void;

  // Bulk operations
  clearAll: () => void;
  importFollicles: (follicles: Follicle[]) => void;

  // Computed
  getSelected: () => Follicle | null;
}

// Default colors for new follicles (cycles through)
const FOLLICLE_COLORS = [
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

      addFollicle: (center, radius) => {
        const id = generateId();
        const newFollicle: Follicle = {
          id,
          center,
          radius: Math.max(radius, 5), // Minimum radius of 5
          label: `Follicle ${get().follicles.length + 1}`,
          notes: '',
          color: FOLLICLE_COLORS[colorIndex++ % FOLLICLE_COLORS.length],
          createdAt: Date.now(),
          updatedAt: Date.now(),
        };

        set(state => ({
          follicles: [...state.follicles, newFollicle],
          selectedId: id,
        }));

        return id;
      },

      updateFollicle: (id, updates) => {
        set(state => ({
          follicles: state.follicles.map(f =>
            f.id === id
              ? { ...f, ...updates, updatedAt: Date.now() }
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

      moveFollicle: (id, newCenter) => {
        get().updateFollicle(id, { center: newCenter });
      },

      resizeFollicle: (id, newRadius) => {
        get().updateFollicle(id, { radius: Math.max(newRadius, 5) });
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
        // Reset color index when importing
        colorIndex = follicles.length;
        set({ follicles, selectedId: null });
      },

      getSelected: () => {
        const { follicles, selectedId } = get();
        return follicles.find(f => f.id === selectedId) || null;
      },
    }),
    {
      // Undo/redo configuration
      limit: 50,
      partialize: (state) => ({
        follicles: state.follicles,
      }),
    }
  )
);

// Export temporal controls for undo/redo
export const useTemporalStore = () => useFollicleStore.temporal;
