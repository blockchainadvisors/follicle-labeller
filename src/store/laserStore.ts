import { create } from 'zustand';
import { Point } from '../types';

export type LaserPhase = 'idle' | 'converging' | 'locked' | 'lost';

/**
 * Session mode. `static` is the original image-space workflow (spawn,
 * converge, lock). `tracking` is the video workflow where the target is
 * updated per-frame and the controller never enters `locked` — it keeps
 * servoing against a moving target until the caller stops the session.
 */
export type LaserMode = 'static' | 'tracking';

const TRAIL_CAP = 20;

interface LaserState {
  phase: LaserPhase;
  mode: LaserMode;
  targetFollicleId: string | null;
  targetPixel: Point | null;

  // Interpolation inputs: renderer lerps between prev and next over the tick period.
  prevPixel: Point | null;
  nextPixel: Point | null;
  nextPixelAt: number;
  tickPeriodMs: number;

  // Ring buffer of recent observed positions; oldest first, newest last.
  trail: Point[];

  // Session start time — used for pulse animation in `locked` phase.
  lockedAt: number | null;

  beginSession: (args: {
    targetFollicleId: string;
    targetPixel: Point;
    initialPixel: Point;
    tickPeriodMs: number;
    now: number;
    mode?: LaserMode;
  }) => void;
  pushObservation: (pixel: Point, now: number) => void;
  setPhase: (phase: LaserPhase, now: number) => void;
  endSession: () => void;
}

export const useLaserStore = create<LaserState>()((set) => ({
  phase: 'idle',
  mode: 'static',
  targetFollicleId: null,
  targetPixel: null,
  prevPixel: null,
  nextPixel: null,
  nextPixelAt: 0,
  tickPeriodMs: 100,
  trail: [],
  lockedAt: null,

  beginSession: ({ targetFollicleId, targetPixel, initialPixel, tickPeriodMs, now, mode = 'static' }) => {
    set({
      phase: 'converging',
      mode,
      targetFollicleId,
      targetPixel: { ...targetPixel },
      prevPixel: { ...initialPixel },
      nextPixel: { ...initialPixel },
      nextPixelAt: now,
      tickPeriodMs,
      trail: [{ ...initialPixel }],
      lockedAt: null,
    });
  },

  pushObservation: (pixel, now) => {
    set((state) => {
      const nextTrail = state.trail.length >= TRAIL_CAP
        ? [...state.trail.slice(state.trail.length - TRAIL_CAP + 1), { ...pixel }]
        : [...state.trail, { ...pixel }];
      return {
        prevPixel: state.nextPixel ? { ...state.nextPixel } : { ...pixel },
        nextPixel: { ...pixel },
        nextPixelAt: now,
        trail: nextTrail,
      };
    });
  },

  setPhase: (phase, now) => {
    set((state) => ({
      phase,
      lockedAt: phase === 'locked' ? (state.lockedAt ?? now) : null,
    }));
  },

  endSession: () => {
    set({
      phase: 'idle',
      mode: 'static',
      targetFollicleId: null,
      targetPixel: null,
      prevPixel: null,
      nextPixel: null,
      nextPixelAt: 0,
      trail: [],
      lockedAt: null,
    });
  },
}));
