import { create } from 'zustand';
import type { ImageId, TrackingSession, FollicleCorrespondence } from '../types';

export interface VideoSessionInfo {
  sessionId: string;
  videoFilePath: string;
  videoFileName: string;
  fps: number;
  frameCount: number;
  videoWidth: number;
  videoHeight: number;
  sourceImageId: ImageId;
  sourceFollicleId: string;
}

interface TrackingState {
  /** All tracking sessions */
  sessions: TrackingSession[];
  /** Currently active session ID */
  activeSessionId: string | null;
  /** Whether the comparison view is open */
  isComparisonViewOpen: boolean;
  /** Backend session ID for single-follicle matching */
  backendSessionId: string | null;
  /** Video tracking state */
  videoSession: VideoSessionInfo | null;
  isVideoTrackingOpen: boolean;

  // Actions
  addSession: (session: TrackingSession) => void;
  removeSession: (sessionId: string) => void;
  setActiveSession: (sessionId: string | null) => void;
  openComparisonView: (sessionId: string) => void;
  closeComparisonView: () => void;
  setBackendSessionId: (id: string | null) => void;
  addCorrespondence: (sessionId: string, correspondence: FollicleCorrespondence) => void;
  updateSession: (sessionId: string, updates: Partial<Pick<TrackingSession, 'targetImageId' | 'correspondences' | 'method'>>) => void;
  openVideoTracking: (session: VideoSessionInfo) => void;
  closeVideoTracking: () => void;
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
  videoSession: null,
  isVideoTrackingOpen: false,

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

  updateSession: (sessionId, updates) =>
    set((state) => ({
      sessions: state.sessions.map((s) =>
        s.id === sessionId ? { ...s, ...updates } : s
      ),
    })),

  openVideoTracking: (session) =>
    set({ videoSession: session, isVideoTrackingOpen: true }),

  closeVideoTracking: () =>
    set({ videoSession: null, isVideoTrackingOpen: false }),

  clearAll: () =>
    set({
      sessions: [],
      activeSessionId: null,
      isComparisonViewOpen: false,
      backendSessionId: null,
      videoSession: null,
      isVideoTrackingOpen: false,
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
