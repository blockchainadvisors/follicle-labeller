/**
 * Virtual Laser Pointer — uncalibrated visual servoing simulator.
 *
 * The service owns two internal objects:
 *
 *   - `VirtualArm`: the "real world" forward model. Given joint angles
 *     (theta, phi) it returns a pixel on the image. This is the thing
 *     we eventually replace with the physical arm + camera feedback.
 *
 *   - `VisualServoController`: the closed-loop controller. It only sees
 *     image observations and commands joint deltas — it never reads the
 *     forward model analytically. It maintains an online Broyden estimate
 *     of the 2×2 Jacobian `dpixel/djoint` and uses a damped pseudo-inverse
 *     to drive the observed pixel toward the target.
 *
 * This split is load-bearing: the real hardware will have the same
 * controller wired to real arm commands and real camera observations.
 * Only the `VirtualArm` is swapped.
 */

import { Point } from '../types';
import { useLaserStore, LaserMode } from '../store/laserStore';
import { useFollicleStore } from '../store/follicleStore';
import { distance } from '../utils/coordinate-transform';

const TICK_PERIOD_MS = 100;
const EPS_CONVERGED_PX = 2;
const MAX_ITERATIONS = 120;
const ALPHA = 0.6;
const MAX_STEP_RAD = 0.08;
const DLS_LAMBDA = 1e-3;
const PROBE_DELTA = 0.02;
const PROBE_MIN_RESPONSE_PX = 0.1;
const OFF_IMAGE_TOLERANCE = 10;
const SPAWN_OFFSET_PX = 500;
const SPAWN_TOLERANCE_PX = 30;
const MAX_SPAWN_BISECTION_ITERS = 10;

interface Vec2 {
  x: number;
  y: number;
}

interface Mat2 {
  a: number; // [0,0]
  b: number; // [0,1]
  c: number; // [1,0]
  d: number; // [1,1]
}

function sub(a: Vec2, b: Vec2): Vec2 {
  return { x: a.x - b.x, y: a.y - b.y };
}

function scale(v: Vec2, s: number): Vec2 {
  return { x: v.x * s, y: v.y * s };
}

function magnitude(v: Vec2): number {
  return Math.sqrt(v.x * v.x + v.y * v.y);
}

function clampMagnitude(v: Vec2, maxMag: number): Vec2 {
  const m = magnitude(v);
  if (m <= maxMag) return v;
  return scale(v, maxMag / m);
}

function matVec(m: Mat2, v: Vec2): Vec2 {
  return { x: m.a * v.x + m.b * v.y, y: m.c * v.x + m.d * v.y };
}

/**
 * Damped least-squares pseudo-inverse of a 2×2 matrix:
 *   (JᵀJ + λ²I)⁻¹ Jᵀ
 * Robust near singularities.
 */
function dampedPinv(m: Mat2, lambda: number): Mat2 {
  const { a, b, c, d } = m;
  // JᵀJ
  const ata = a * a + c * c;
  const atb = a * b + c * d;
  const btb = b * b + d * d;
  // JᵀJ + λ²I
  const l2 = lambda * lambda;
  const m00 = ata + l2;
  const m01 = atb;
  const m11 = btb + l2;
  const det = m00 * m11 - m01 * m01;
  if (Math.abs(det) < 1e-12) {
    return { a: 0, b: 0, c: 0, d: 0 };
  }
  // inv(JᵀJ + λ²I)
  const inv00 = m11 / det;
  const inv01 = -m01 / det;
  const inv11 = m00 / det;
  // inv * Jᵀ — Jᵀ is [[a,c],[b,d]]
  return {
    a: inv00 * a + inv01 * b,
    b: inv00 * c + inv01 * d,
    c: inv01 * a + inv11 * b,
    d: inv01 * c + inv11 * d,
  };
}

function det2(m: Mat2): number {
  return m.a * m.d - m.b * m.c;
}

/**
 * Private simulator: maps (theta, phi) to a pixel on the image.
 *
 * θ ∈ [-π, π]: rotation of the semicircle around the vertical axis.
 * φ ∈ [0, π/2]: arc position of the gripper along the semicircle.
 *
 * The controller does NOT read this function. It only observes the
 * pixel returned by `observe()` after commanding a delta.
 */
class VirtualArm {
  private theta: number;
  private phi: number;
  private readonly cx: number;
  private readonly cy: number;
  private readonly radius: number;
  private readonly yFactor: number;

  constructor(cx: number, cy: number, initialTheta: number, initialPhi: number) {
    this.cx = cx;
    this.cy = cy;
    this.theta = initialTheta;
    this.phi = initialPhi;
    // Radius is scaled so that the full workspace roughly matches the
    // image. In a real system this would be fixed by the physical arm
    // geometry and camera intrinsics — here we just make it plausible.
    this.radius = Math.min(cx, cy) * 0.9;
    this.yFactor = 0.7;
  }

  static projectAt(
    theta: number,
    phi: number,
    cx: number,
    cy: number,
    radius: number,
    yFactor: number,
  ): Point {
    return {
      x: cx + radius * Math.sin(phi) * Math.cos(theta),
      y: cy + radius * Math.sin(phi) * Math.sin(theta) * yFactor,
    };
  }

  command(deltaTheta: number, deltaPhi: number): void {
    this.theta += deltaTheta;
    // Wrap theta to keep it bounded
    if (this.theta > Math.PI) this.theta -= 2 * Math.PI;
    if (this.theta < -Math.PI) this.theta += 2 * Math.PI;
    // Clamp phi to physically meaningful range
    this.phi = Math.max(0, Math.min(Math.PI / 2, this.phi + deltaPhi));
  }

  observe(): Point {
    return VirtualArm.projectAt(this.theta, this.phi, this.cx, this.cy, this.radius, this.yFactor);
  }

  getImageCenter(): Point {
    return { x: this.cx, y: this.cy };
  }

  getRadius(): number {
    return this.radius;
  }

  getYFactor(): number {
    return this.yFactor;
  }
}

/**
 * Pick a plausible starting pose so the first observed pixel is roughly
 * `SPAWN_OFFSET_PX` from the target. We bisect phi at a random theta to
 * hit the offset, within tolerance. If the target is near the edge and
 * no 500px spawn is reachable, we fall back to the largest reachable
 * offset at that theta.
 */
function pickSpawnPose(
  target: Point,
  cx: number,
  cy: number,
  radius: number,
  yFactor: number,
): { theta: number; phi: number } {
  const theta = Math.random() * 2 * Math.PI - Math.PI;

  let lo = 0;
  let hi = Math.PI / 2;
  let bestPhi = hi;
  let bestDiff = Infinity;

  for (let i = 0; i < MAX_SPAWN_BISECTION_ITERS; i++) {
    const mid = (lo + hi) / 2;
    const p = VirtualArm.projectAt(theta, mid, cx, cy, radius, yFactor);
    const d = distance(p, target);
    const diff = Math.abs(d - SPAWN_OFFSET_PX);
    if (diff < bestDiff) {
      bestDiff = diff;
      bestPhi = mid;
    }
    if (diff < SPAWN_TOLERANCE_PX) {
      return { theta, phi: mid };
    }
    if (d < SPAWN_OFFSET_PX) {
      lo = mid;
    } else {
      hi = mid;
    }
  }

  return { theta, phi: bestPhi };
}

/**
 * Closed-loop image-space controller. Maintains an online Broyden
 * estimate of the 2×2 Jacobian dpixel/djoint. Never reads the arm's
 * forward model analytically — only acts through `command`/`observe`.
 */
class VisualServoController {
  private J: Mat2 = { a: 0, b: 0, c: 0, d: 0 };
  private jacobianInitialized = false;
  private iterations = 0;
  private consecutiveSingular = 0;
  private offImageCount = 0;

  constructor(
    private readonly arm: VirtualArm,
    private readonly imageWidth: number,
    private readonly imageHeight: number,
  ) {}

  /**
   * Seed the Jacobian with two orthogonal probe moves. Returns false if
   * either probe produced no detectable motion (would leave Ĵ undefined).
   */
  initJacobian(): boolean {
    const y0 = this.arm.observe();

    // probe θ
    this.arm.command(PROBE_DELTA, 0);
    const y1 = this.arm.observe();
    this.arm.command(-PROBE_DELTA, 0);

    // probe φ
    this.arm.command(0, PROBE_DELTA);
    const y2 = this.arm.observe();
    this.arm.command(0, -PROBE_DELTA);

    const dy1 = sub(y1, y0);
    const dy2 = sub(y2, y0);

    if (magnitude(dy1) < PROBE_MIN_RESPONSE_PX || magnitude(dy2) < PROBE_MIN_RESPONSE_PX) {
      return false;
    }

    this.J = {
      a: dy1.x / PROBE_DELTA,
      b: dy2.x / PROBE_DELTA,
      c: dy1.y / PROBE_DELTA,
      d: dy2.y / PROBE_DELTA,
    };
    this.jacobianInitialized = true;
    return true;
  }

  /**
   * Execute one control step. Returns `'converged'` when err < EPS,
   * `'lost'` on fatal conditions (too many iterations, off-image,
   * probe failure on re-init), or `'running'` otherwise.
   *
   * In `'tracking'` mode the target is expected to move between ticks,
   * so we never report `'converged'` (the controller keeps servoing)
   * and the open-ended termination conditions (`MAX_ITERATIONS`,
   * `offImageCount`) are suppressed. Jacobian re-seeding on sustained
   * singularity is kept — that's still a valuable recovery path.
   */
  step(target: Point, mode: LaserMode = 'static'): { status: 'running' | 'converged' | 'lost'; observed: Point } {
    if (!this.jacobianInitialized) {
      const ok = this.initJacobian();
      if (!ok) {
        return { status: 'lost', observed: this.arm.observe() };
      }
    }

    const observed = this.arm.observe();
    const err = sub(observed, target);

    if (mode === 'static' && magnitude(err) < EPS_CONVERGED_PX) {
      return { status: 'converged', observed };
    }

    // Watch for chronic singularity — if damped-LS can't get us anywhere
    // for 3 steps in a row, re-seed the Jacobian via probes.
    if (Math.abs(det2(this.J)) < 1e-6) {
      this.consecutiveSingular += 1;
      if (this.consecutiveSingular >= 3) {
        const ok = this.initJacobian();
        this.consecutiveSingular = 0;
        if (!ok) {
          return { status: 'lost', observed };
        }
      }
    } else {
      this.consecutiveSingular = 0;
    }

    const pinv = dampedPinv(this.J, DLS_LAMBDA);
    const step = clampMagnitude(scale(matVec(pinv, err), -ALPHA), MAX_STEP_RAD);

    this.arm.command(step.x, step.y);
    const newObserved = this.arm.observe();

    const dy = sub(newObserved, observed);
    const u2 = step.x * step.x + step.y * step.y;
    if (u2 > 1e-6) {
      // Broyden rank-1 update, guarded: reject outlier observations that
      // would poison the estimate (ratio threshold keeps Ĵ·u predictive).
      const predicted = matVec(this.J, step);
      const dyMag = magnitude(dy);
      const predMag = magnitude(predicted);
      const predictive = dyMag <= 3 * (predMag + 1e-6);
      if (predictive) {
        const residual = sub(dy, predicted);
        const scaleFactor = 1 / u2;
        this.J = {
          a: this.J.a + residual.x * step.x * scaleFactor,
          b: this.J.b + residual.x * step.y * scaleFactor,
          c: this.J.c + residual.y * step.x * scaleFactor,
          d: this.J.d + residual.y * step.y * scaleFactor,
        };
      }
    }

    this.iterations += 1;

    const isOffImage =
      newObserved.x < -OFF_IMAGE_TOLERANCE ||
      newObserved.x > this.imageWidth + OFF_IMAGE_TOLERANCE ||
      newObserved.y < -OFF_IMAGE_TOLERANCE ||
      newObserved.y > this.imageHeight + OFF_IMAGE_TOLERANCE;
    if (isOffImage) {
      this.offImageCount += 1;
    } else {
      this.offImageCount = 0;
    }

    // Open-ended tracking sessions don't time out or bail on off-image
    // excursions — the live target can legitimately drift near edges.
    if (mode === 'static') {
      if (this.offImageCount >= 10) {
        return { status: 'lost', observed: newObserved };
      }
      if (this.iterations >= MAX_ITERATIONS) {
        return { status: 'lost', observed: newObserved };
      }
    }

    return { status: 'running', observed: newObserved };
  }
}

export class LaserControlService {
  private static instance: LaserControlService | null = null;

  private intervalHandle: ReturnType<typeof setInterval> | null = null;
  private controller: VisualServoController | null = null;
  private currentTarget: Point | null = null;
  private currentTargetId: string | null = null;
  private mode: LaserMode = 'static';
  private follicleUnsubscribe: (() => void) | null = null;

  private constructor() {}

  static getInstance(): LaserControlService {
    if (!LaserControlService.instance) {
      LaserControlService.instance = new LaserControlService();
    }
    return LaserControlService.instance;
  }

  isActive(): boolean {
    return this.intervalHandle !== null;
  }

  /**
   * Begin a new laser session targeting the given follicle. Stops any
   * existing session first.
   *
   * `opts.mode` defaults to `'static'` for the image-canvas workflow.
   * Pass `'tracking'` for the video workflow: the session stays open
   * against a moving target updated via `setTarget`, never locks, and
   * ignores the follicle-store lifecycle (the video view owns it).
   */
  start(
    targetFollicleId: string,
    targetPixel: Point,
    imageWidth: number,
    imageHeight: number,
    opts: { mode?: LaserMode } = {},
  ): void {
    this.stop();

    const mode = opts.mode ?? 'static';
    this.mode = mode;

    const cx = imageWidth / 2;
    const cy = imageHeight / 2;
    const probeRadius = Math.min(cx, cy) * 0.9;
    const yFactor = 0.7;

    const { theta, phi } = pickSpawnPose(targetPixel, cx, cy, probeRadius, yFactor);

    const arm = new VirtualArm(cx, cy, theta, phi);
    const controller = new VisualServoController(arm, imageWidth, imageHeight);

    this.controller = controller;
    this.currentTarget = { ...targetPixel };
    this.currentTargetId = targetFollicleId;

    const initialPixel = arm.observe();
    const now = performance.now();
    useLaserStore.getState().beginSession({
      targetFollicleId,
      targetPixel,
      initialPixel,
      tickPeriodMs: TICK_PERIOD_MS,
      now,
      mode,
    });

    // In static mode, end the session if the target follicle is deleted
    // mid-flight. In tracking mode the video view owns lifecycle — the
    // sidebar follicle store isn't the source of truth for the session.
    if (mode === 'static') {
      this.follicleUnsubscribe = useFollicleStore.subscribe((state) => {
        if (!this.currentTargetId) return;
        const stillExists = state.follicles.some((f) => f.id === this.currentTargetId);
        if (!stillExists) {
          useLaserStore.getState().setPhase('lost', performance.now());
          this.stop();
        }
      });
    }

    this.intervalHandle = setInterval(() => this.tick(), TICK_PERIOD_MS);
  }

  /**
   * Update the current target pixel without tearing the session down.
   * Intended for `'tracking'` mode where the target moves per video
   * frame. The Broyden Jacobian estimate is preserved because the arm
   * geometry is unchanged; the next control step naturally steers
   * toward the new target.
   */
  setTarget(pixel: Point): void {
    if (!this.controller) return;
    this.currentTarget = { ...pixel };
    useLaserStore.setState({ targetPixel: { ...pixel } });
  }

  stop(): void {
    if (this.intervalHandle !== null) {
      clearInterval(this.intervalHandle);
      this.intervalHandle = null;
    }
    if (this.follicleUnsubscribe) {
      this.follicleUnsubscribe();
      this.follicleUnsubscribe = null;
    }
    this.controller = null;
    this.currentTarget = null;
    this.currentTargetId = null;
    this.mode = 'static';
    useLaserStore.getState().endSession();
  }

  private tick(): void {
    if (!this.controller || !this.currentTarget) return;

    const result = this.controller.step(this.currentTarget, this.mode);
    const now = performance.now();
    useLaserStore.getState().pushObservation(result.observed, now);

    if (result.status === 'converged') {
      useLaserStore.getState().setPhase('locked', now);
      if (this.intervalHandle !== null) {
        clearInterval(this.intervalHandle);
        this.intervalHandle = null;
      }
      return;
    }

    if (result.status === 'lost') {
      useLaserStore.getState().setPhase('lost', now);
      // Give the UI a beat to render the final frame before tearing down.
      setTimeout(() => {
        if (useLaserStore.getState().phase === 'lost') {
          this.stop();
        }
      }, 300);
    }
  }
}

export const laserControlService = LaserControlService.getInstance();
