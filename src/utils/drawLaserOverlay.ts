/**
 * Shared laser-dot overlay renderer.
 *
 * Used by both the static image canvas (`CanvasRenderer.drawLaser`) and
 * the video tracking canvas (`VideoTrackingView.redrawCanvas`). The two
 * call sites have different coordinate systems — the image canvas draws
 * in raw image pixels inside a viewport-scaled ctx, while the video
 * canvas draws in letterbox-scaled space — so this helper takes:
 *
 *   - `pixelToCanvas`: maps a point in the laser's pixel-space (the
 *     same space the controller commands in, identical to the image or
 *     video frame's pixel coords) to the current ctx's coord system.
 *   - `strokeScale`: a multiplier for stroke widths and geometric
 *     radii so they stay constant in screen pixels regardless of the
 *     ctx's current scale. Equivalent to `1 / viewport.scale`.
 */

import { Point } from '../types';
import { LaserPhase, LaserMode } from '../store/laserStore';

export interface LaserRenderSnapshot {
  phase: LaserPhase;
  mode: LaserMode;
  prevPixel: Point | null;
  nextPixel: Point | null;
  nextPixelAt: number;
  tickPeriodMs: number;
  trail: Point[];
  lockedAt: number | null;
}

export function drawLaserOverlay(
  ctx: CanvasRenderingContext2D,
  state: LaserRenderSnapshot,
  now: number,
  pixelToCanvas: (p: Point) => Point,
  strokeScale: number,
): void {
  const prev = state.prevPixel;
  const next = state.nextPixel;
  if (!prev || !next) return;

  // Linear interpolation between the last two controller ticks.
  const elapsed = now - state.nextPixelAt;
  const t = Math.max(0, Math.min(1, elapsed / Math.max(1, state.tickPeriodMs)));
  const displayPixel: Point = {
    x: prev.x + (next.x - prev.x) * t,
    y: prev.y + (next.y - prev.y) * t,
  };
  const display = pixelToCanvas(displayPixel);

  // Fading trail (oldest -> newest: low alpha -> high alpha).
  if (state.trail.length >= 2) {
    ctx.lineCap = 'round';
    ctx.lineWidth = 2 * strokeScale;
    for (let i = 1; i < state.trail.length; i++) {
      const a = pixelToCanvas(state.trail[i - 1]);
      const b = pixelToCanvas(state.trail[i]);
      const alpha = 0.5 * (i / state.trail.length);
      ctx.strokeStyle = `rgba(255, 42, 42, ${alpha})`;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }
    ctx.lineCap = 'butt';
  }

  // Halo — radial gradient approximating a soft glow.
  const haloRadius = 12 * strokeScale;
  const gradient = ctx.createRadialGradient(
    display.x, display.y, 0,
    display.x, display.y, haloRadius,
  );
  gradient.addColorStop(0, 'rgba(255, 42, 42, 0.6)');
  gradient.addColorStop(1, 'rgba(255, 42, 42, 0)');
  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(display.x, display.y, haloRadius, 0, Math.PI * 2);
  ctx.fill();

  // Solid core dot.
  const coreRadius = 4 * strokeScale;
  ctx.fillStyle = '#FF2A2A';
  ctx.beginPath();
  ctx.arc(display.x, display.y, coreRadius, 0, Math.PI * 2);
  ctx.fill();

  // Locked indicator: subtle pulsing ring to communicate "on target".
  // Static mode only — tracking never enters `locked` (moving target).
  if (state.mode === 'static' && state.phase === 'locked') {
    const pulse = 0.5 + 0.5 * Math.sin(now / 300);
    const ringRadius = 8 * strokeScale;
    ctx.strokeStyle = `rgba(255, 42, 42, ${0.3 + pulse * 0.5})`;
    ctx.lineWidth = 1.5 * strokeScale;
    ctx.beginPath();
    ctx.arc(display.x, display.y, ringRadius, 0, Math.PI * 2);
    ctx.stroke();
  }
}
