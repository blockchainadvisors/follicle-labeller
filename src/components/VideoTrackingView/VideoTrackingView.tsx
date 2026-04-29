import React, { useEffect, useRef, useState, useCallback } from "react";
import { useTrackingStore, type VideoSessionInfo } from "../../store/trackingStore";
import { useProjectStore } from "../../store/projectStore";
import { useFollicleStore } from "../../store/follicleStore";
import { follicleTrackingService } from "../../services/follicleTrackingService";
import { laserControlService } from "../../services/laserControlService";
import { screenRecordingService } from "../../services/screenRecordingService";
import { useLaserStore } from "../../store/laserStore";
import { drawLaserOverlay } from "../../utils/drawLaserOverlay";
import { CanvasRenderer } from "../Canvas/CanvasRenderer";
import type {
  VideoFrameResult,
  Viewport,
  VideoFrameCacheEntry,
  RectangleAnnotation,
  DragState,
} from "../../types";
import "./VideoTrackingView.css";

const MATCH_COLOR = "#4ECDC4";
const ZOOM_MIN = 0.01;
const ZOOM_MAX = 100;
const CACHE_CAP_BYTES = 1.5 * 1024 * 1024 * 1024;
const PROCESSING_POLL_MS = 50;
const DOUBLE_CLICK_MS = 300;

async function decodeB64ToBitmap(b64: string): Promise<ImageBitmap> {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  const blob = new Blob([bytes], { type: "image/jpeg" });
  return await createImageBitmap(blob);
}

function formatTimecode(frameIndex: number, fps: number): string {
  if (fps <= 0 || frameIndex < 0) return "00:00.00";
  const fpsRound = Math.max(1, Math.round(fps));
  const totalSeconds = frameIndex / fps;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = Math.floor(totalSeconds % 60);
  const frames = frameIndex % fpsRound;
  return `${minutes.toString().padStart(2, "0")}:${seconds
    .toString()
    .padStart(2, "0")}.${frames.toString().padStart(2, "0")}`;
}

export const VideoTrackingView: React.FC = () => {
  const videoSession = useTrackingStore((s) => s.videoSession);
  const closeVideoTracking = useTrackingStore((s) => s.closeVideoTracking);
  if (!videoSession) return null;
  return (
    <VideoTrackingViewInner
      session={videoSession}
      onClose={closeVideoTracking}
    />
  );
};

interface InnerProps {
  session: VideoSessionInfo;
  onClose: () => void;
}

const VideoTrackingViewInner: React.FC<InnerProps> = ({ session, onClose }) => {
  const {
    sessionId,
    source,
    videoFileName,
    fps,
    frameCount,
    videoWidth,
    videoHeight,
    cameraDeviceId,
    sourceImageId,
    sourceFollicleId,
  } = session;
  const isCamera = source === 'camera';

  // Source scalp image and tracked follicle resolved from stores.
  // These may become null if the image/follicle is deleted mid-session;
  // the source pane renders a placeholder in that case.
  const sourceImage = useProjectStore((s) => s.images.get(sourceImageId) ?? null);
  const sourceFollicle = useFollicleStore((s) => {
    const f = s.follicles.find((x) => x.id === sourceFollicleId);
    return f && f.shape === "rectangle" ? (f as RectangleAnnotation) : null;
  });

  // ==================== REFS ====================
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  const scrubberTrackRef = useRef<HTMLDivElement>(null);

  const videoPaneRef = useRef<HTMLDivElement>(null);
  const sourcePaneRef = useRef<HTMLDivElement>(null);
  const sourceCanvasRef = useRef<HTMLCanvasElement>(null);
  const sourceRendererRef = useRef<CanvasRenderer | null>(null);
  const sourceViewportRef = useRef<Viewport>({ offsetX: 0, offsetY: 0, scale: 1 });
  const sourcePanelSizeRef = useRef<{ w: number; h: number }>({ w: 0, h: 0 });
  const sourcePanningRef = useRef<{ x: number; y: number } | null>(null);
  const sourceAutoFramedRef = useRef(false);

  // Frame-0 tracker anchor (in video pixel coords) for drift computation
  const frameZeroAnchorRef = useRef<{ x: number; y: number } | null>(null);

  const stoppedRef = useRef(false);

  // Virtual laser pointer: guards a single `start()` call for this
  // video session. Flipped on first trusted match; reset on stop.
  const laserStartedRef = useRef(false);

  const frameCacheRef = useRef<Map<number, VideoFrameCacheEntry>>(new Map());
  const cacheBytesRef = useRef(0);

  const processingMaxFrameRef = useRef(-1);
  const viewFrameRef = useRef(-1);

  const currentFrameRef = useRef<{
    bitmap: ImageBitmap;
    match: VideoFrameResult["match"];
    frameIndex: number;
    cooldownRemaining?: number | null;
    cooldownReason?: "seeking" | "origin_lost" | null;
  } | null>(null);

  const isPlayingRef = useRef(true);

  const viewportRef = useRef<Viewport>({ offsetX: 0, offsetY: 0, scale: 1 });
  const baseDisplayRef = useRef<{
    offsetX: number;
    offsetY: number;
    w: number;
    h: number;
  }>({ offsetX: 0, offsetY: 0, w: 0, h: 0 });

  const panPointRef = useRef<{ x: number; y: number } | null>(null);
  const lastMiddleClickTimeRef = useRef(0);

  const seekRequestIdRef = useRef(0);

  const scrubDraggingRef = useRef(false);
  const pendingScrubFrameRef = useRef<number | null>(null);

  // ==================== STATE ====================
  const [processingMaxFrame, setProcessingMaxFrame] = useState(-1);
  const [viewFrame, setViewFrame] = useState(-1);
  const [isPlaying, setIsPlaying] = useState(true);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [status, setStatus] = useState<"tracking" | "done" | "stopped">(
    "tracking",
  );
  const [viewport, setViewport] = useState<Viewport>({
    offsetX: 0,
    offsetY: 0,
    scale: 1,
  });
  const [scrubHoverFrame, setScrubHoverFrame] = useState<number | null>(null);
  const [scrubHoverX, setScrubHoverX] = useState(0);
  const [memoryWarning, setMemoryWarning] = useState(false);

  // Screen-recording state. ``isRecording`` drives the REC badge in the
  // header; ``recordingWarning`` is a non-blocking notice shown when the
  // OS denies screen-capture permission.
  const [isRecording, setIsRecording] = useState(false);
  const [recordingWarning, setRecordingWarning] = useState<{
    message: string;
    canOpenSettings: boolean;
  } | null>(null);

  const [sourcePanelCollapsed, setSourcePanelCollapsed] = useState(false);
  const [sourceViewport, setSourceViewport] = useState<Viewport>({
    offsetX: 0,
    offsetY: 0,
    scale: 1,
  });
  const [sourcePanelSize, setSourcePanelSize] = useState<{ w: number; h: number }>({
    w: 0,
    h: 0,
  });
  const [drift, setDrift] = useState<number | null>(null);

  useEffect(() => {
    isPlayingRef.current = isPlaying;
  }, [isPlaying]);

  // Virtual laser session subscription — drives a self-sustaining RAF
  // so the dot interpolates between 10 Hz controller ticks even when
  // video playback isn't advancing frames (paused, or after tracking
  // stopped). Scheduled only while the laser is non-idle.
  const laserPhase = useLaserStore((s) => s.phase);

  // ==================== DRAWING ====================
  const redrawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const containerW = canvas.width;
    const containerH = canvas.height;

    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, containerW, containerH);

    const frame = currentFrameRef.current;
    if (!frame) return;

    const videoAspect = frame.bitmap.width / frame.bitmap.height;
    const containerAspect = containerW / containerH;
    let baseW: number;
    let baseH: number;
    if (videoAspect > containerAspect) {
      baseW = containerW;
      baseH = containerW / videoAspect;
    } else {
      baseH = containerH;
      baseW = containerH * videoAspect;
    }
    const baseOffsetX = (containerW - baseW) / 2;
    const baseOffsetY = (containerH - baseH) / 2;
    baseDisplayRef.current = {
      offsetX: baseOffsetX,
      offsetY: baseOffsetY,
      w: baseW,
      h: baseH,
    };

    const v = viewportRef.current;

    ctx.save();
    ctx.translate(baseOffsetX + v.offsetX, baseOffsetY + v.offsetY);
    ctx.scale(v.scale, v.scale);

    ctx.drawImage(frame.bitmap, 0, 0, baseW, baseH);

    if (frame.match) {
      const sX = baseW / videoWidth;
      const sY = baseH / videoHeight;
      const inv = 1 / v.scale;

      // Virtual laser: start on first trusted origin, then follow the
      // moving origin via setTarget. Skip updates when the origin was
      // extrapolated (lostPoint === 'origin') OR the session is in
      // cooldown (lostPoint === 'both' with markers frozen on the last
      // trusted position) — matches the user's chosen policy.
      const originLostForLaser =
        frame.match.lostPoint === "origin" || frame.match.lostPoint === "both";
      if (!originLostForLaser) {
        const originPixel = {
          x: frame.match.transformedX,
          y: frame.match.transformedY,
        };
        if (!laserStartedRef.current) {
          laserControlService.start(
            sourceFollicleId,
            originPixel,
            videoWidth,
            videoHeight,
            { mode: "tracking" },
          );
          laserStartedRef.current = true;
        } else {
          laserControlService.setTarget(originPixel);
        }
      }

      const det = frame.match.targetDetection;
      const x = det.x * sX;
      const y = det.y * sY;
      const w = det.width * sX;
      const h = det.height * sY;

      ctx.strokeStyle = MATCH_COLOR;
      ctx.lineWidth = 2 * inv;
      ctx.strokeRect(x, y, w, h);

      // Dual-point tracking: origin and tip come from TWO independent NCC
      // matches with a rigid-consistency check. When one point fails
      // (lostPoint set), its position is extrapolated from the other and
      // rendered in a distinct style so the user can tell the tracker is
      // guessing. `lostPoint === 'both'` means the session is in cooldown
      // and both markers are frozen on the last trusted position.
      const originLost =
        frame.match.lostPoint === "origin" || frame.match.lostPoint === "both";
      const tipLost =
        frame.match.lostPoint === "tip" || frame.match.lostPoint === "both";

      // Origin marker — stroked ring (+ crosshair when trusted)
      const cx = frame.match.transformedX * sX;
      const cy = frame.match.transformedY * sY;

      ctx.save();
      ctx.strokeStyle = originLost ? "#FF6B6B" : MATCH_COLOR;
      ctx.lineWidth = 2 * inv;
      if (originLost) {
        ctx.setLineDash([6 * inv, 4 * inv]);
      }
      ctx.beginPath();
      ctx.arc(cx, cy, 6 * inv, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();

      if (!originLost) {
        ctx.strokeStyle = MATCH_COLOR;
        ctx.lineWidth = 2 * inv;
        ctx.beginPath();
        ctx.moveTo(cx - 10 * inv, cy);
        ctx.lineTo(cx + 10 * inv, cy);
        ctx.moveTo(cx, cy - 10 * inv);
        ctx.lineTo(cx, cy + 10 * inv);
        ctx.stroke();
      }

      // Direction line from origin to tip — drawn BEFORE the tip marker so
      // the marker sits on top. Teal normally; dashed when either end is
      // extrapolated so it's clear the line is anchored on a guess.
      const tx = frame.match.tipX * sX;
      const ty = frame.match.tipY * sY;

      ctx.save();
      ctx.strokeStyle = MATCH_COLOR;
      ctx.lineWidth = 1.5 * inv;
      if (originLost || tipLost) {
        ctx.setLineDash([5 * inv, 3 * inv]);
      }
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(tx, ty);
      ctx.stroke();
      ctx.restore();

      // Tip marker — filled red dot when trusted, hollow red ring when
      // extrapolated. Mirrors the "only origin gets the big circle"
      // convention in CanvasRenderer, with an extra state for lost.
      ctx.save();
      if (tipLost) {
        ctx.strokeStyle = "#FF6B6B";
        ctx.lineWidth = 1.5 * inv;
        ctx.setLineDash([3 * inv, 2 * inv]);
        ctx.beginPath();
        ctx.arc(tx, ty, 4 * inv, 0, Math.PI * 2);
        ctx.stroke();
      } else {
        ctx.fillStyle = "#FF6B6B";
        ctx.beginPath();
        ctx.arc(tx, ty, 4 * inv, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();

      // Virtual laser overlay — drawn inside the letterboxed + viewport
      // transform so the dot aligns with the origin in video-pixel space.
      // `pixelToCanvas` scales video-pixel coords into the letterbox
      // display space used by the markers above. `inv` keeps the dot a
      // constant screen size regardless of zoom.
      const laserSnapshot = useLaserStore.getState();
      if (laserSnapshot.phase !== "idle") {
        drawLaserOverlay(
          ctx,
          laserSnapshot,
          performance.now(),
          (p) => ({ x: p.x * sX, y: p.y * sY }),
          inv,
        );
      }
    }

    ctx.restore();

    // Draw confidence label in screen space (sharp text at any zoom)
    // Suppressed during cooldown — we render the cooldown label instead.
    const inCooldown =
      typeof frame.cooldownRemaining === "number" && frame.cooldownRemaining > 0;
    if (frame.match && !inCooldown) {
      const sX = baseW / videoWidth;
      const sY = baseH / videoHeight;
      const screenX =
        baseOffsetX + v.offsetX + frame.match.targetDetection.x * sX * v.scale;
      const screenY =
        baseOffsetY +
        v.offsetY +
        frame.match.targetDetection.y * sY * v.scale -
        6;
      ctx.fillStyle = MATCH_COLOR;
      ctx.font = "bold 12px sans-serif";
      ctx.fillText(
        `${(frame.match.confidence * 100).toFixed(0)}%`,
        screenX,
        screenY,
      );
    }

    // Cooldown countdown — shown anchored to the top-left of the
    // letterboxed video area in screen space. Colour is amber so it
    // doesn't clash with the teal match colour or the red laser.
    if (inCooldown) {
      const remaining = frame.cooldownRemaining as number;
      const label =
        frame.cooldownReason === "origin_lost"
          ? `Reacquiring in ${remaining.toFixed(1)} s…`
          : `Searching in ${remaining.toFixed(1)} s…`;
      ctx.fillStyle = "#FFB74D";
      ctx.font = "bold 14px sans-serif";
      ctx.fillText(label, baseOffsetX + 12, baseOffsetY + 22);
    }
  }, [videoWidth, videoHeight, sourceFollicleId]);

  // Keep the canvas repainting at 60 Hz while the laser is active so
  // the dot interpolates smoothly between 10 Hz controller ticks. The
  // loop reads `useLaserStore.getState()` inside `redrawCanvas` each
  // frame, so no dependency on tick-level state is needed here.
  useEffect(() => {
    if (laserPhase === "idle") return;
    let rafId = 0;
    const loop = () => {
      redrawCanvas();
      rafId = requestAnimationFrame(loop);
    };
    rafId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafId);
  }, [laserPhase, redrawCanvas]);

  // Stop the laser whenever the video session ends — covers handleStop
  // (which sets status='stopped'), camera disconnect (also 'stopped'),
  // and processing completion ('done'). Unmount is handled separately.
  useEffect(() => {
    if (status === "stopped" || status === "done") {
      laserControlService.stop();
      laserStartedRef.current = false;
    }
  }, [status]);

  // Reset the start-guard whenever the laser phase returns to idle.
  // This lets the next trusted match restart the session if the laser
  // died mid-tracking (e.g., probe failure on a bad initial pose).
  useEffect(() => {
    if (laserPhase === "idle") {
      laserStartedRef.current = false;
    }
  }, [laserPhase]);

  // ==================== DISPLAY FRAME ====================
  const displayFrameAtIndex = useCallback(
    async (idx: number) => {
      if (idx < 0) return;
      if (viewFrameRef.current === idx) return;
      const entry = frameCacheRef.current.get(idx);
      if (!entry) return;

      // Update view cursor synchronously to prevent races with playback tick
      viewFrameRef.current = idx;
      setViewFrame(idx);
      setConfidence(entry.match ? entry.match.confidence : null);

      // Capture frame-0 anchor on first sight; compute drift for every frame
      // from that anchor. Drift is in video pixel space.
      if (entry.match) {
        if (idx === 0 || frameZeroAnchorRef.current === null) {
          if (idx === 0) {
            frameZeroAnchorRef.current = {
              x: entry.match.transformedX,
              y: entry.match.transformedY,
            };
          }
        }
        const anchor = frameZeroAnchorRef.current;
        if (anchor) {
          const dx = entry.match.transformedX - anchor.x;
          const dy = entry.match.transformedY - anchor.y;
          setDrift(Math.hypot(dx, dy));
        } else {
          setDrift(null);
        }
      } else {
        setDrift(null);
      }

      const requestId = ++seekRequestIdRef.current;

      let bitmap: ImageBitmap;
      try {
        bitmap = await decodeB64ToBitmap(entry.frameDataB64);
      } catch (err) {
        console.error("Frame decode failed:", err);
        return;
      }

      // Stale check (rapid scrubbing replaced this request)
      if (requestId !== seekRequestIdRef.current) {
        bitmap.close();
        return;
      }

      if (currentFrameRef.current) {
        try {
          currentFrameRef.current.bitmap.close();
        } catch {
          // ignore
        }
      }

      currentFrameRef.current = {
        bitmap,
        match: entry.match,
        frameIndex: idx,
        cooldownRemaining: entry.cooldownRemaining,
        cooldownReason: entry.cooldownReason,
      };
      redrawCanvas();
    },
    [redrawCanvas],
  );

  // ==================== SEEK / ZOOM / PAN ====================
  const seekTo = useCallback(
    (target: number) => {
      const max = processingMaxFrameRef.current;
      if (max < 0) return;
      const clamped = Math.max(0, Math.min(target, max));
      if (clamped === viewFrameRef.current) return;
      void displayFrameAtIndex(clamped);
    },
    [displayFrameAtIndex],
  );

  const zoomAtPoint = useCallback(
    (delta: number, centerPoint: { x: number; y: number }) => {
      const v = viewportRef.current;
      const newScale = Math.min(
        ZOOM_MAX,
        Math.max(ZOOM_MIN, v.scale * (1 + delta)),
      );
      if (newScale === v.scale) return;
      const factor = newScale / v.scale;

      const base = baseDisplayRef.current;
      const totalOffsetX = base.offsetX + v.offsetX;
      const totalOffsetY = base.offsetY + v.offsetY;

      const newTotalOffsetX =
        centerPoint.x - (centerPoint.x - totalOffsetX) * factor;
      const newTotalOffsetY =
        centerPoint.y - (centerPoint.y - totalOffsetY) * factor;

      const newViewport: Viewport = {
        offsetX: newTotalOffsetX - base.offsetX,
        offsetY: newTotalOffsetY - base.offsetY,
        scale: newScale,
      };
      viewportRef.current = newViewport;
      setViewport(newViewport);
      redrawCanvas();
    },
    [redrawCanvas],
  );

  const panBy = useCallback(
    (dx: number, dy: number) => {
      const v = viewportRef.current;
      const newViewport: Viewport = {
        scale: v.scale,
        offsetX: v.offsetX + dx,
        offsetY: v.offsetY + dy,
      };
      viewportRef.current = newViewport;
      setViewport(newViewport);
      redrawCanvas();
    },
    [redrawCanvas],
  );

  const resetViewport = useCallback(() => {
    const newViewport: Viewport = { offsetX: 0, offsetY: 0, scale: 1 };
    viewportRef.current = newViewport;
    setViewport(newViewport);
    redrawCanvas();
  }, [redrawCanvas]);

  const togglePlay = useCallback(() => {
    if (isPlayingRef.current) {
      setIsPlaying(false);
      return;
    }
    // About to resume play
    if (
      status !== "tracking" &&
      processingMaxFrameRef.current >= 0 &&
      viewFrameRef.current >= processingMaxFrameRef.current
    ) {
      // At end of cached range and backend won't add more → restart from 0
      void displayFrameAtIndex(0);
    }
    setIsPlaying(true);
  }, [status, displayFrameAtIndex]);

  // ==================== EFFECTS ====================

  // ResizeObserver: keep each canvas in sync with its own pane.
  // Coalesced via requestAnimationFrame because the S-key collapse transition
  // fires the observer repeatedly during the 180ms flex-basis animation.
  useEffect(() => {
    const videoPane = videoPaneRef.current;
    const videoCanvas = canvasRef.current;
    if (!videoPane || !videoCanvas) return;

    let rafId = 0;
    let scheduled = false;

    const measure = () => {
      scheduled = false;

      const vRect = videoPane.getBoundingClientRect();
      const vw = Math.max(1, Math.floor(vRect.width));
      const vh = Math.max(1, Math.floor(vRect.height));
      if (videoCanvas.width !== vw || videoCanvas.height !== vh) {
        videoCanvas.width = vw;
        videoCanvas.height = vh;
      }
      redrawCanvas();

      const sourcePane = sourcePaneRef.current;
      const sourceCanvas = sourceCanvasRef.current;
      if (sourcePane && sourceCanvas) {
        // The source canvas lives inside .video-tracking-source-canvas-wrapper
        // below the pane header, so measure its parent for dimensions.
        const wrapper = sourceCanvas.parentElement;
        if (wrapper) {
          const sRect = wrapper.getBoundingClientRect();
          const sw = Math.max(1, Math.floor(sRect.width));
          const sh = Math.max(1, Math.floor(sRect.height));
          if (sourceCanvas.width !== sw || sourceCanvas.height !== sh) {
            sourceCanvas.width = sw;
            sourceCanvas.height = sh;
          }
          if (sw !== sourcePanelSizeRef.current.w || sh !== sourcePanelSizeRef.current.h) {
            sourcePanelSizeRef.current = { w: sw, h: sh };
            setSourcePanelSize({ w: sw, h: sh });
          }
        }
      }
    };

    const schedule = () => {
      if (scheduled) return;
      scheduled = true;
      rafId = requestAnimationFrame(measure);
    };

    measure();
    const observer = new ResizeObserver(schedule);
    observer.observe(videoPane);
    if (sourcePaneRef.current) observer.observe(sourcePaneRef.current);
    return () => {
      observer.disconnect();
      if (rafId) cancelAnimationFrame(rafId);
    };
  }, [redrawCanvas]);

  // Video panel wheel zoom
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();
      const delta = e.deltaY > 0 ? -0.1 : 0.1;
      const rect = canvas.getBoundingClientRect();
      const point = { x: e.clientX - rect.left, y: e.clientY - rect.top };
      zoomAtPoint(delta, point);
    };
    canvas.addEventListener("wheel", handleWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", handleWheel);
  }, [zoomAtPoint]);

  // Source renderer lifecycle: instantiate on mount, feed the scalp bitmap in.
  // Never call .close() on sourceImage.imageBitmap — the project store owns it.
  useEffect(() => {
    const canvas = sourceCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    sourceRendererRef.current = new CanvasRenderer(ctx);
    return () => {
      sourceRendererRef.current = null;
    };
  }, [sourceImage, sourceFollicle]);

  useEffect(() => {
    if (sourceRendererRef.current && sourceImage) {
      sourceRendererRef.current.setImage(sourceImage.imageBitmap);
    }
  }, [sourceImage]);

  // Auto-frame the source viewport on the tracked follicle when we know the
  // panel size. CONTEXT_MULT = 5.0 mirrors the template-crop logic in
  // Toolbar.tsx so the user sees the same patch the backend sees.
  useEffect(() => {
    if (sourceAutoFramedRef.current) return;
    if (!sourceFollicle || !sourceImage) return;
    if (sourcePanelSize.w <= 0 || sourcePanelSize.h <= 0) return;

    const CONTEXT_MULT = 5.0;
    const origin = sourceFollicle.origin?.originPoint;
    const cx = origin ? origin.x : sourceFollicle.x + sourceFollicle.width / 2;
    const cy = origin ? origin.y : sourceFollicle.y + sourceFollicle.height / 2;

    const contextW = Math.max(1, sourceFollicle.width * CONTEXT_MULT);
    const contextH = Math.max(1, sourceFollicle.height * CONTEXT_MULT);
    const scaleX = sourcePanelSize.w / contextW;
    const scaleY = sourcePanelSize.h / contextH;
    const scale = Math.min(scaleX, scaleY);

    const offsetX = sourcePanelSize.w / 2 - cx * scale;
    const offsetY = sourcePanelSize.h / 2 - cy * scale;

    const next: Viewport = { offsetX, offsetY, scale };
    sourceViewportRef.current = next;
    setSourceViewport(next);
    sourceAutoFramedRef.current = true;
  }, [sourceFollicle, sourceImage, sourcePanelSize]);

  // Reset auto-framing when the session switches to a different follicle.
  useEffect(() => {
    sourceAutoFramedRef.current = false;
  }, [sourceFollicleId]);

  // Render the source canvas. Dependencies are strictly source-side state, so
  // this effect NEVER runs on video frame ticks.
  useEffect(() => {
    const canvas = sourceCanvasRef.current;
    const renderer = sourceRendererRef.current;
    if (!canvas || !renderer || !sourceFollicle) return;
    const emptyDrag: DragState = {
      isDragging: false,
      startPoint: null,
      currentPoint: null,
      dragType: null,
      targetId: null,
    };
    renderer.render(
      canvas.width,
      canvas.height,
      sourceViewport,
      [sourceFollicle],
      new Set<string>(),
      emptyDrag,
      false,
      true,
      "rectangle",
      30,
    );
  }, [sourceViewport, sourceFollicle, sourcePanelSize, sourceImage]);

  const zoomSourceAtPoint = useCallback(
    (delta: number, centerPoint: { x: number; y: number }) => {
      const v = sourceViewportRef.current;
      const newScale = Math.min(
        ZOOM_MAX,
        Math.max(ZOOM_MIN, v.scale * (1 + delta)),
      );
      if (newScale === v.scale) return;
      const factor = newScale / v.scale;
      const newOffsetX = centerPoint.x - (centerPoint.x - v.offsetX) * factor;
      const newOffsetY = centerPoint.y - (centerPoint.y - v.offsetY) * factor;
      const next: Viewport = {
        offsetX: newOffsetX,
        offsetY: newOffsetY,
        scale: newScale,
      };
      sourceViewportRef.current = next;
      setSourceViewport(next);
    },
    [],
  );

  const panSourceBy = useCallback((dx: number, dy: number) => {
    const v = sourceViewportRef.current;
    const next: Viewport = {
      scale: v.scale,
      offsetX: v.offsetX + dx,
      offsetY: v.offsetY + dy,
    };
    sourceViewportRef.current = next;
    setSourceViewport(next);
  }, []);

  const fitSourceToFollicle = useCallback(() => {
    if (!sourceFollicle) return;
    if (sourcePanelSize.w <= 0 || sourcePanelSize.h <= 0) return;
    const CONTEXT_MULT = 5.0;
    const origin = sourceFollicle.origin?.originPoint;
    const cx = origin ? origin.x : sourceFollicle.x + sourceFollicle.width / 2;
    const cy = origin ? origin.y : sourceFollicle.y + sourceFollicle.height / 2;
    const contextW = Math.max(1, sourceFollicle.width * CONTEXT_MULT);
    const contextH = Math.max(1, sourceFollicle.height * CONTEXT_MULT);
    const scale = Math.min(
      sourcePanelSize.w / contextW,
      sourcePanelSize.h / contextH,
    );
    const offsetX = sourcePanelSize.w / 2 - cx * scale;
    const offsetY = sourcePanelSize.h / 2 - cy * scale;
    const next: Viewport = { offsetX, offsetY, scale };
    sourceViewportRef.current = next;
    setSourceViewport(next);
  }, [sourceFollicle, sourcePanelSize]);

  // Source panel wheel zoom
  useEffect(() => {
    const canvas = sourceCanvasRef.current;
    if (!canvas) return;
    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();
      const delta = e.deltaY > 0 ? -0.1 : 0.1;
      const rect = canvas.getBoundingClientRect();
      const point = { x: e.clientX - rect.left, y: e.clientY - rect.top };
      zoomSourceAtPoint(delta, point);
    };
    canvas.addEventListener("wheel", handleWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", handleWheel);
  }, [zoomSourceAtPoint, sourceImage, sourceFollicle]);

  // Source panel left-drag pan. Scoped to the source canvas so it cannot
  // collide with the video pan handler below.
  useEffect(() => {
    const canvas = sourceCanvasRef.current;
    if (!canvas) return;

    const handleMouseDown = (e: MouseEvent) => {
      if (e.button !== 0) return;
      e.preventDefault();
      e.stopPropagation();
      sourcePanningRef.current = { x: e.clientX, y: e.clientY };
      canvas.classList.add("is-panning");
    };

    const handleMouseMove = (e: MouseEvent) => {
      if (!sourcePanningRef.current) return;
      const dx = e.clientX - sourcePanningRef.current.x;
      const dy = e.clientY - sourcePanningRef.current.y;
      sourcePanningRef.current = { x: e.clientX, y: e.clientY };
      panSourceBy(dx, dy);
    };

    const handleMouseUp = () => {
      if (!sourcePanningRef.current) return;
      sourcePanningRef.current = null;
      canvas.classList.remove("is-panning");
    };

    canvas.addEventListener("mousedown", handleMouseDown);
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      canvas.removeEventListener("mousedown", handleMouseDown);
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [panSourceBy, sourceImage, sourceFollicle]);

  // Pan: left-mouse drag (any zoom) + middle-mouse drag, double middle-click resets
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const handleMouseDown = (e: MouseEvent) => {
      // Left button: start pan
      if (e.button === 0) {
        e.preventDefault();
        panPointRef.current = { x: e.clientX, y: e.clientY };
        canvas.classList.add("is-panning");
        return;
      }
      // Middle button: double-click resets, single click pans
      if (e.button === 1) {
        e.preventDefault();
        const now = Date.now();
        if (now - lastMiddleClickTimeRef.current < DOUBLE_CLICK_MS) {
          resetViewport();
          lastMiddleClickTimeRef.current = 0;
          panPointRef.current = null;
          canvas.classList.remove("is-panning");
          return;
        }
        lastMiddleClickTimeRef.current = now;
        panPointRef.current = { x: e.clientX, y: e.clientY };
        canvas.classList.add("is-panning");
      }
    };

    const handleMouseMove = (e: MouseEvent) => {
      if (!panPointRef.current) return;
      const dx = e.clientX - panPointRef.current.x;
      const dy = e.clientY - panPointRef.current.y;
      panPointRef.current = { x: e.clientX, y: e.clientY };
      panBy(dx, dy);
    };

    const handleMouseUp = (e: MouseEvent) => {
      if (e.button !== 0 && e.button !== 1) return;
      if (!panPointRef.current) return;
      panPointRef.current = null;
      canvas.classList.remove("is-panning");
    };

    canvas.addEventListener("mousedown", handleMouseDown);
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      canvas.removeEventListener("mousedown", handleMouseDown);
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [panBy, resetViewport]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const tagName = target?.tagName?.toLowerCase();
      const isInputFocused =
        tagName === "input" ||
        tagName === "textarea" ||
        tagName === "select" ||
        target?.isContentEditable === true;
      if (isInputFocused) return;

      // Zoom shortcuts
      if ((e.metaKey || e.ctrlKey) && (e.key === "=" || e.key === "+")) {
        e.preventDefault();
        const c = canvasRef.current;
        if (c) zoomAtPoint(0.1, { x: c.width / 2, y: c.height / 2 });
        return;
      }
      if ((e.metaKey || e.ctrlKey) && e.key === "-") {
        e.preventDefault();
        const c = canvasRef.current;
        if (c) zoomAtPoint(-0.1, { x: c.width / 2, y: c.height / 2 });
        return;
      }
      if ((e.metaKey || e.ctrlKey) && e.key === "0") {
        e.preventDefault();
        resetViewport();
        return;
      }

      // Playback shortcuts
      if (e.key === " " || e.code === "Space") {
        e.preventDefault();
        togglePlay();
        return;
      }
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        setIsPlaying(false);
        const step = e.shiftKey ? Math.max(1, Math.round(fps)) : 1;
        seekTo(viewFrameRef.current - step);
        return;
      }
      if (e.key === "ArrowRight") {
        e.preventDefault();
        setIsPlaying(false);
        const step = e.shiftKey ? Math.max(1, Math.round(fps)) : 1;
        seekTo(viewFrameRef.current + step);
        return;
      }
      if (e.key === "Home") {
        e.preventDefault();
        setIsPlaying(false);
        seekTo(0);
        return;
      }
      if (e.key === "End") {
        e.preventDefault();
        setIsPlaying(false);
        seekTo(processingMaxFrameRef.current);
        return;
      }

      // Source panel shortcuts (no modifiers)
      if (!e.metaKey && !e.ctrlKey && !e.altKey && !e.shiftKey) {
        if (e.key === "s" || e.key === "S") {
          e.preventDefault();
          setSourcePanelCollapsed((prev) => !prev);
          return;
        }
        if (e.key === "f" || e.key === "F") {
          e.preventDefault();
          fitSourceToFollicle();
          return;
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [zoomAtPoint, resetViewport, seekTo, togglePlay, fps, fitSourceToFollicle]);

  // Processing loop: fetches new frames from the backend.
  // File-source only — camera sessions use the capture loop below which
  // pushes frames FROM the frontend TO the backend.
  useEffect(() => {
    if (isCamera) return;
    if (status !== "tracking") return;
    stoppedRef.current = false;
    let cancelled = false;

    const loop = async () => {
      while (!cancelled && !stoppedRef.current) {
        // Wait until: playing AND at the live edge.
        if (
          !isPlayingRef.current ||
          viewFrameRef.current < processingMaxFrameRef.current
        ) {
          await new Promise((r) => setTimeout(r, PROCESSING_POLL_MS));
          continue;
        }

        if (cacheBytesRef.current > CACHE_CAP_BYTES) {
          if (!memoryWarning) setMemoryWarning(true);
          await new Promise((r) => setTimeout(r, 200));
          continue;
        }
        if (memoryWarning) setMemoryWarning(false);

        try {
          const result = await follicleTrackingService.videoMatchFrame(
            sessionId,
          );
          if (cancelled || stoppedRef.current) break;
          if (!result.success || result.done) {
            setStatus("done");
            break;
          }

          frameCacheRef.current.set(result.frameIndex, {
            frameIndex: result.frameIndex,
            frameDataB64: result.frameData || "",
            match: result.match,
            cooldownRemaining: result.cooldownRemaining,
            cooldownReason: result.cooldownReason,
          });
          cacheBytesRef.current += result.frameData?.length || 0;
          processingMaxFrameRef.current = result.frameIndex;
          setProcessingMaxFrame(result.frameIndex);

          if (
            isPlayingRef.current &&
            viewFrameRef.current <= result.frameIndex - 1
          ) {
            await displayFrameAtIndex(result.frameIndex);
          }
        } catch (err) {
          console.error("Processing loop error:", err);
          await new Promise((r) => setTimeout(r, 500));
        }
      }
    };

    void loop();
    return () => {
      cancelled = true;
    };
  }, [isCamera, sessionId, status, displayFrameAtIndex, memoryWarning]);

  // Playback tick: advance view through cache at FPS.
  // File-source only — camera sessions render each frame as it arrives,
  // driven by requestVideoFrameCallback rather than a fixed FPS cadence.
  useEffect(() => {
    if (isCamera) return;
    if (!isPlaying) return;
    if (fps <= 0) return;

    const frameInterval = 1000 / fps;
    let rafId = 0;
    let cancelled = false;
    let lastTick = performance.now();

    const tick = () => {
      if (cancelled) return;
      const now = performance.now();
      if (now - lastTick >= frameInterval) {
        lastTick = now;
        if (viewFrameRef.current < processingMaxFrameRef.current) {
          void displayFrameAtIndex(viewFrameRef.current + 1);
        }
        // else: at live edge, processing loop will push us forward
      }
      rafId = requestAnimationFrame(tick);
    };

    rafId = requestAnimationFrame(tick);
    return () => {
      cancelled = true;
      cancelAnimationFrame(rafId);
    };
  }, [isCamera, isPlaying, fps, displayFrameAtIndex]);

  // Camera capture loop: open getUserMedia, draw each <video> frame into
  // an OffscreenCanvas, push the JPEG to the backend for a per-frame
  // match, then render the resulting bitmap directly (no frame cache).
  // Back-pressure: drop intervening frames while a match is inflight.
  useEffect(() => {
    if (!isCamera) return;
    if (!cameraDeviceId) return;
    if (status !== "tracking") return;

    stoppedRef.current = false;
    let cancelled = false;
    let activeStream: MediaStream | null = null;
    let rafId = 0;
    let inflight = false;
    let frameCounter = 0;

    const video = document.createElement("video");
    video.muted = true;
    video.playsInline = true;
    const offCanvas =
      typeof OffscreenCanvas !== "undefined"
        ? new OffscreenCanvas(videoWidth || 640, videoHeight || 480)
        : null;

    const blobToBase64 = async (blob: Blob): Promise<string> => {
      const buf = await blob.arrayBuffer();
      const bytes = new Uint8Array(buf);
      let binary = "";
      for (let i = 0; i < bytes.length; i += 0x8000) {
        binary += String.fromCharCode.apply(
          null,
          Array.from(bytes.subarray(i, i + 0x8000)),
        );
      }
      return `data:image/jpeg;base64,${btoa(binary)}`;
    };

    const scheduleNext = () => {
      if (cancelled || stoppedRef.current) return;
      const v = video as HTMLVideoElement & {
        requestVideoFrameCallback?: (cb: () => void) => number;
      };
      if (typeof v.requestVideoFrameCallback === "function") {
        v.requestVideoFrameCallback(() => {
          void captureOnce();
        });
      } else {
        rafId = requestAnimationFrame(() => {
          void captureOnce();
        });
      }
    };

    const captureOnce = async () => {
      if (cancelled || stoppedRef.current) return;
      if (!isPlayingRef.current || inflight) {
        scheduleNext();
        return;
      }
      if (!video.videoWidth || !video.videoHeight || !offCanvas) {
        scheduleNext();
        return;
      }

      inflight = true;
      try {
        if (offCanvas.width !== video.videoWidth)
          offCanvas.width = video.videoWidth;
        if (offCanvas.height !== video.videoHeight)
          offCanvas.height = video.videoHeight;
        const ctx = offCanvas.getContext("2d");
        if (!ctx) {
          inflight = false;
          scheduleNext();
          return;
        }
        ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
        const blob = await offCanvas.convertToBlob({
          type: "image/jpeg",
          quality: 0.8,
        });
        const bitmap = await createImageBitmap(blob);
        const b64 = await blobToBase64(blob);

        const result = await follicleTrackingService.cameraMatchFrame(
          sessionId,
          b64,
        );

        if (cancelled || stoppedRef.current) {
          bitmap.close();
          return;
        }
        if (!result.success) {
          bitmap.close();
          return;
        }

        const frameIndex = frameCounter++;

        if (result.match) {
          if (frameIndex === 0 || frameZeroAnchorRef.current === null) {
            if (frameIndex === 0) {
              frameZeroAnchorRef.current = {
                x: result.match.transformedX,
                y: result.match.transformedY,
              };
            }
          }
          const anchor = frameZeroAnchorRef.current;
          if (anchor) {
            const dx = result.match.transformedX - anchor.x;
            const dy = result.match.transformedY - anchor.y;
            setDrift(Math.hypot(dx, dy));
          }
        } else {
          setDrift(null);
        }

        setConfidence(result.match ? result.match.confidence : null);
        viewFrameRef.current = frameIndex;
        processingMaxFrameRef.current = frameIndex;
        setViewFrame(frameIndex);
        setProcessingMaxFrame(frameIndex);

        if (currentFrameRef.current) {
          try {
            currentFrameRef.current.bitmap.close();
          } catch {
            // ignore
          }
        }
        currentFrameRef.current = {
          bitmap,
          match: result.match,
          frameIndex,
          cooldownRemaining: result.cooldownRemaining,
          cooldownReason: result.cooldownReason,
        };
        redrawCanvas();
      } catch (err) {
        console.error("Camera capture loop error:", err);
      } finally {
        inflight = false;
        scheduleNext();
      }
    };

    (async () => {
      try {
        activeStream = await navigator.mediaDevices.getUserMedia({
          video: { deviceId: { exact: cameraDeviceId } },
        });
        if (cancelled) {
          activeStream.getTracks().forEach((t) => t.stop());
          activeStream = null;
          return;
        }
        video.srcObject = activeStream;
        await video.play();
        if (!video.videoWidth || !video.videoHeight) {
          await new Promise<void>((resolve, reject) => {
            const timer = setTimeout(
              () => reject(new Error("Timed out waiting for camera metadata")),
              5000,
            );
            video.onloadedmetadata = () => {
              clearTimeout(timer);
              resolve();
            };
          });
        }
        scheduleNext();
      } catch (err) {
        console.error("Failed to open camera stream:", err);
        setStatus("stopped");
      }
    })();

    return () => {
      cancelled = true;
      cancelAnimationFrame(rafId);
      if (activeStream) {
        activeStream.getTracks().forEach((t) => t.stop());
        activeStream = null;
      }
      video.srcObject = null;
    };
  }, [
    isCamera,
    cameraDeviceId,
    sessionId,
    status,
    videoWidth,
    videoHeight,
    redrawCanvas,
  ]);

  // Mount cleanup: close all bitmaps on unmount
  useEffect(() => {
    return () => {
      stoppedRef.current = true;
      seekRequestIdRef.current++;
      // Tear down any in-flight laser session — the view is going away,
      // there is nowhere left to render the dot.
      laserControlService.stop();
      laserStartedRef.current = false;
      if (currentFrameRef.current) {
        try {
          currentFrameRef.current.bitmap.close();
        } catch {
          // ignore
        }
        currentFrameRef.current = null;
      }
      frameCacheRef.current.clear();
      cacheBytesRef.current = 0;
    };
  }, []);

  // Screen recording: start when the tracking view mounts, stop on
  // unmount. handleStop/handleClose also explicitly await stop() so the
  // recording is flushed before the user-initiated close path; this
  // effect's cleanup is the safety net for unmounts that don't go
  // through those handlers (e.g., the parent removing the view because
  // the session was closed elsewhere).
  useEffect(() => {
    let cancelled = false;

    const unsubscribe = screenRecordingService.onStateChange((rec) => {
      if (!cancelled) setIsRecording(rec);
    });

    (async () => {
      const result = await screenRecordingService.start(sessionId);
      if (cancelled) {
        // The view unmounted while we were starting — undo it.
        await screenRecordingService.stop().catch(() => undefined);
        return;
      }
      if (!result.started) {
        if (result.reason === 'permission') {
          setRecordingWarning({
            message:
              result.message ??
              'Screen recording disabled — grant access in System Settings.',
            canOpenSettings: true,
          });
        } else if (result.reason === 'unsupported') {
          // Silent in web mode; only surface this when running in
          // Electron with no compatible codec (extremely unlikely).
          if (window.electronAPI) {
            console.warn(
              '[recording] unsupported:',
              result.message ?? 'no compatible recorder',
            );
          }
        } else if (result.reason === 'error') {
          console.error('[recording] start failed:', result.message);
        }
      }
    })();

    return () => {
      cancelled = true;
      unsubscribe();
      // Fire-and-forget on cleanup. handleStop/handleClose already await
      // the same call when the user triggers them; this branch covers
      // unmounts that bypass those handlers.
      screenRecordingService
        .stop()
        .then((res) => {
          if (res.saved && res.filePath) {
            console.log('[recording] saved to', res.filePath);
          } else if (!res.saved && res.error && res.error !== 'Recorder not running') {
            console.warn('[recording] stop returned error:', res.error);
          }
        })
        .catch((err) => {
          console.error('[recording] cleanup stop failed', err);
        });
    };
  }, [sessionId]);

  const handleOpenScreenRecordingSettings = useCallback(async () => {
    if (window.electronAPI?.openScreenRecordingSettings) {
      await window.electronAPI.openScreenRecordingSettings();
    }
  }, []);

  const dismissRecordingWarning = useCallback(() => {
    setRecordingWarning(null);
  }, []);

  // ==================== HANDLERS ====================

  const handleStop = useCallback(async () => {
    stoppedRef.current = true;
    setStatus("stopped");
    setIsPlaying(false);
    // Stop the recorder first so the saved file lines up with what the
    // user just saw. Failure here shouldn't block the tracker stop.
    try {
      const recResult = await screenRecordingService.stop();
      if (recResult.saved && recResult.filePath) {
        console.log('[recording] saved to', recResult.filePath);
      }
    } catch (err) {
      console.warn('[recording] stop failed during handleStop', err);
    }
    try {
      await follicleTrackingService.videoStop(sessionId);
    } catch {
      // ignore
    }
  }, [sessionId]);

  const handleClose = useCallback(async () => {
    stoppedRef.current = true;
    seekRequestIdRef.current++;
    try {
      const recResult = await screenRecordingService.stop();
      if (recResult.saved && recResult.filePath) {
        console.log('[recording] saved to', recResult.filePath);
      }
    } catch (err) {
      console.warn('[recording] stop failed during handleClose', err);
    }
    try {
      await follicleTrackingService.videoStop(sessionId);
    } catch {
      // ignore
    }
    onClose();
  }, [sessionId, onClose]);

  // ==================== SCRUBBER ====================

  const computeScrubFrame = useCallback(
    (clientX: number, clamp: boolean): number => {
      const track = scrubberTrackRef.current;
      if (!track) return -1;
      if (frameCount <= 0) return -1;
      const rect = track.getBoundingClientRect();
      const ratio = (clientX - rect.left) / rect.width;
      const target = Math.round(ratio * (frameCount - 1));
      if (clamp) {
        return Math.max(0, Math.min(target, processingMaxFrameRef.current));
      }
      return Math.max(0, Math.min(target, frameCount - 1));
    },
    [frameCount],
  );

  const handleScrubMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (processingMaxFrameRef.current < 0) return;
      e.preventDefault();
      scrubDraggingRef.current = true;
      setIsPlaying(false);
      const target = computeScrubFrame(e.clientX, true);
      if (target >= 0) seekTo(target);
    },
    [computeScrubFrame, seekTo],
  );

  // Document-level scrub drag
  useEffect(() => {
    let rafScheduled = false;

    const flushPendingScrub = () => {
      rafScheduled = false;
      const pending = pendingScrubFrameRef.current;
      if (pending !== null && pending >= 0) {
        seekTo(pending);
        pendingScrubFrameRef.current = null;
      }
    };

    const handleMouseMove = (e: MouseEvent) => {
      if (!scrubDraggingRef.current) return;
      const target = computeScrubFrame(e.clientX, true);
      if (target < 0) return;
      pendingScrubFrameRef.current = target;
      if (!rafScheduled) {
        rafScheduled = true;
        requestAnimationFrame(flushPendingScrub);
      }
    };

    const handleMouseUp = () => {
      if (!scrubDraggingRef.current) return;
      scrubDraggingRef.current = false;
      flushPendingScrub();
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [computeScrubFrame, seekTo]);

  const handleScrubHover = useCallback(
    (e: React.MouseEvent) => {
      const track = scrubberTrackRef.current;
      if (!track) return;
      if (frameCount <= 0) return;
      const rect = track.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const ratio = Math.max(0, Math.min(1, x / rect.width));
      const frame = Math.round(ratio * (frameCount - 1));
      setScrubHoverFrame(frame);
      setScrubHoverX(x);
    },
    [frameCount],
  );

  const handleScrubLeave = useCallback(() => {
    setScrubHoverFrame(null);
  }, []);

  // ==================== DERIVED ====================

  const playedPercent =
    frameCount > 0 && viewFrame >= 0
      ? ((viewFrame + 1) / frameCount) * 100
      : 0;
  const bufferedPercent =
    frameCount > 0 && processingMaxFrame >= 0
      ? ((processingMaxFrame + 1) / frameCount) * 100
      : 0;
  const zoomPercent = Math.round(viewport.scale * 100);
  const displayFrameNum = Math.max(0, viewFrame);

  // Drift threshold = 20% of the source follicle diagonal (source-pixel space,
  // approximated as video-pixel space under the assumption of 1:1 scale).
  const driftThresholdPx: number | null = sourceFollicle
    ? 0.2 *
      Math.sqrt(
        sourceFollicle.width * sourceFollicle.width +
          sourceFollicle.height * sourceFollicle.height,
      )
    : null;

  return (
    <div className="video-tracking-view">
      {/* Header */}
      <div className="video-tracking-header">
        <div className="video-tracking-header-left">
          <div className="video-tracking-title">
            {isCamera ? "Live Camera Tracking" : "Video Tracking"}
            {isRecording && (
              <span
                className="video-tracking-rec-badge"
                title="Recording app window to disk"
              >
                <span className="video-tracking-rec-dot" aria-hidden="true" />
                REC
              </span>
            )}
          </div>
          <div className="video-tracking-stats">
            <span>
              Frame:{" "}
              <span className="video-tracking-stat-value">
                {isCamera ? displayFrameNum : `${displayFrameNum}/${frameCount}`}
              </span>
            </span>
            {!isCamera && (
              <span>
                FPS:{" "}
                <span className="video-tracking-stat-value">
                  {fps.toFixed(1)}
                </span>
              </span>
            )}
            <span>
              {isCamera ? "Camera: " : "File: "}
              <span className="video-tracking-stat-value">{videoFileName}</span>
            </span>
            {confidence !== null && (
              <span>
                Confidence:{" "}
                <span className="video-tracking-stat-value">
                  {(confidence * 100).toFixed(0)}%
                </span>
              </span>
            )}
            {drift !== null && driftThresholdPx !== null && (
              <span>
                Drift:{" "}
                <span
                  className={`video-tracking-stat-value${
                    drift > driftThresholdPx
                      ? " video-tracking-drift-high"
                      : ""
                  }`}
                >
                  {drift.toFixed(0)}px
                </span>
              </span>
            )}
            {memoryWarning && (
              <span className="video-tracking-memory-warning">
                ⚠ Cache full
              </span>
            )}
          </div>
        </div>
        <div className="video-tracking-header-right">
          {status === "tracking" && (
            <button className="video-tracking-stop-btn" onClick={handleStop}>
              Stop
            </button>
          )}
          <button
            className="video-tracking-stop-btn"
            onClick={handleClose}
            title="Close"
          >
            ✕
          </button>
        </div>
      </div>

      {recordingWarning && (
        <div className="video-tracking-rec-warning" role="status">
          <span className="video-tracking-rec-warning-message">
            {recordingWarning.message}
          </span>
          {recordingWarning.canOpenSettings && (
            <button
              className="video-tracking-rec-warning-action"
              onClick={handleOpenScreenRecordingSettings}
              type="button"
            >
              Open System Settings
            </button>
          )}
          <button
            className="video-tracking-rec-warning-dismiss"
            onClick={dismissRecordingWarning}
            type="button"
            aria-label="Dismiss"
          >
            ✕
          </button>
        </div>
      )}

      {/* Split content: source panel (left) + video panel (right) */}
      <div
        className={`video-tracking-content${
          sourcePanelCollapsed ? " source-collapsed" : ""
        }`}
        ref={contentRef}
      >
        <div className="video-tracking-source-pane" ref={sourcePaneRef}>
          <div className="video-tracking-source-header">
            <span className="video-tracking-source-label">Source</span>
            {sourceImage && (
              <span
                className="video-tracking-source-filename"
                title={sourceImage.fileName}
              >
                {sourceImage.fileName}
              </span>
            )}
          </div>
          <div className="video-tracking-source-canvas-wrapper">
            {sourceImage && sourceFollicle ? (
              <canvas
                ref={sourceCanvasRef}
                className="video-tracking-source-canvas"
              />
            ) : (
              <div className="video-tracking-source-placeholder">
                Source image no longer available
              </div>
            )}
          </div>
        </div>
        <div className="video-tracking-video-pane" ref={videoPaneRef}>
          <canvas ref={canvasRef} className="video-tracking-canvas" />
        </div>
      </div>

      {/* Footer / playback controls */}
      <div className="video-tracking-footer">
        <button
          className="video-tracking-play-btn"
          onClick={togglePlay}
          title={isPlaying ? "Pause (Space)" : "Play (Space)"}
        >
          {isPlaying ? "⏸" : "▶"}
        </button>

        {isCamera ? (
          // Live cameras have no known duration, so the scrubber is
          // hidden. Status text covers the tracking state.
          <div className="video-tracking-scrubber video-tracking-scrubber-live">
            <div className="scrubber-track" />
            <span className="video-tracking-live-label">● LIVE</span>
          </div>
        ) : (
          <div
            className="video-tracking-scrubber"
            ref={scrubberTrackRef}
            onMouseDown={handleScrubMouseDown}
            onMouseMove={handleScrubHover}
            onMouseLeave={handleScrubLeave}
          >
            <div className="scrubber-track" />
            <div
              className="scrubber-buffered"
              style={{ width: `${bufferedPercent}%` }}
            />
            <div
              className="scrubber-played"
              style={{ width: `${playedPercent}%` }}
            />
            <div
              className="scrubber-thumb"
              style={{ left: `${playedPercent}%` }}
            />
            {scrubHoverFrame !== null && (
              <div className="scrubber-tooltip" style={{ left: scrubHoverX }}>
                {formatTimecode(scrubHoverFrame, fps)}
              </div>
            )}
          </div>
        )}

        <div className="video-tracking-footer-right">
          <span className="video-tracking-zoom-display" title="Zoom level">
            {zoomPercent}%
          </span>
          <button
            className="video-tracking-reset-btn"
            onClick={resetViewport}
            title="Reset zoom (Cmd+0)"
          >
            Fit
          </button>
          <span className="video-tracking-status-text">
            {status === "tracking" && "Tracking"}
            {status === "done" && "Complete"}
            {status === "stopped" &&
              (isCamera ? "Stopped" : `Stopped at frame ${displayFrameNum}`)}
          </span>
        </div>
      </div>
    </div>
  );
};
