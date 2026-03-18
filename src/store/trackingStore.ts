import { create } from 'zustand';
import type { ImageId, TrackingSession } from '../types';

interface TrackingState {
  /** All tracking sessions */
  sessions: TrackingSession[];
  /** Currently active session ID */
  activeSessionId: string | null;
  /** Whether the comparison view is open */
  isComparisonViewOpen: boolean;

  // Actions
  addSession: (session: TrackingSession) => void;
  removeSession: (sessionId: string) => void;
  setActiveSession: (sessionId: string | null) => void;
  openComparisonView: (sessionId: string) => void;
  closeComparisonView: () => void;
  clearAll: () => void;

  // Selectors
  getActiveSession: () => TrackingSession | null;
  getSessionsForImage: (imageId: ImageId) => TrackingSession[];
}

export const useTrackingStore = create<TrackingState>((set, get) => ({
  sessions: [],
  activeSessionId: null,
  isComparisonViewOpen: false,

  addSession: (session) =>
    set((state) => ({
      sessions: [...state.sessions, session],
    })),

  removeSession: (sessionId) =>
    set((state) => ({
      sessions: state.sessions.filter((s) => s.id !== sessionId),
      activeSessionId:
        state.activeSessionId === sessionId ? null : state.activeSessionId,
      isComparisonViewOpen:
        state.activeSessionId === sessionId
          ? false
          : state.isComparisonViewOpen,
    })),

  setActiveSession: (sessionId) =>
    set({ activeSessionId: sessionId }),

  openComparisonView: (sessionId) =>
    set({
      activeSessionId: sessionId,
      isComparisonViewOpen: true,
    }),

  closeComparisonView: () =>
    set({
      isComparisonViewOpen: false,
    }),

  clearAll: () =>
    set({
      sessions: [],
      activeSessionId: null,
      isComparisonViewOpen: false,
    }),

  getActiveSession: () => {
    const { sessions, activeSessionId } = get();
    if (!activeSessionId) return null;
    return sessions.find((s) => s.id === activeSessionId) ?? null;
  },

  getSessionsForImage: (imageId) => {
    const { sessions } = get();
    return sessions.filter(
      (s) => s.sourceImageId === imageId || s.targetImageId === imageId
    );
  },
}));
