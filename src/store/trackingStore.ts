import { create } from 'zustand';
import type { ImageId, TrackingSession, FollicleCorrespondence } from '../types';

interface TrackingState {
  /** All tracking sessions */
  sessions: TrackingSession[];
  /** Currently active session ID */
  activeSessionId: string | null;
  /** Whether the comparison view is open */
  isComparisonViewOpen: boolean;
  /** Backend session ID for single-follicle matching */
  backendSessionId: string | null;

  // Actions
  addSession: (session: TrackingSession) => void;
  removeSession: (sessionId: string) => void;
  setActiveSession: (sessionId: string | null) => void;
  openComparisonView: (sessionId: string) => void;
  closeComparisonView: () => void;
  setBackendSessionId: (id: string | null) => void;
  addCorrespondence: (sessionId: string, correspondence: FollicleCorrespondence) => void;
  clearAll: () => void;

  // Selectors
  getActiveSession: () => TrackingSession | null;
  getSessionsForImage: (imageId: ImageId) => TrackingSession[];
}

export const useTrackingStore = create<TrackingState>((set, get) => ({
  sessions: [],
  activeSessionId: null,
  isComparisonViewOpen: false,
  backendSessionId: null,

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

  setBackendSessionId: (id) =>
    set({ backendSessionId: id }),

  addCorrespondence: (sessionId, correspondence) =>
    set((state) => ({
      sessions: state.sessions.map((s) =>
        s.id === sessionId
          ? { ...s, correspondences: [...s.correspondences, correspondence] }
          : s
      ),
    })),

  clearAll: () =>
    set({
      sessions: [],
      activeSessionId: null,
      isComparisonViewOpen: false,
      backendSessionId: null,
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
