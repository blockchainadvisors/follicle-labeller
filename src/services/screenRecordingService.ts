/**
 * Screen Recording Service
 *
 * Records the app's BrowserWindow while a tracking session is active.
 *
 * Encoding pipeline (Electron-only):
 *   1. ``getDisplayMedia`` produces a full-screen MediaStream.
 *   2. The stream feeds a hidden <video>; a RAF loop crops to the
 *      window's bounds onto a hidden <canvas>.
 *   3. The mediabunny ``CanvasSource`` reads from that canvas, runs the
 *      WebCodecs ``VideoEncoder`` for H.264, and the
 *      ``Mp4OutputFormat`` muxes to a real .mp4 — all in pure JS.
 *
 * Why mediabunny? Electron 28's MediaRecorder doesn't have an MP4
 * muxer (only WebM). mediabunny is a small, zero-dependency TypeScript
 * library that wraps WebCodecs + an MP4 muxer so we can write a real
 * .mp4 file the user can drop into QuickTime, Premiere, etc.
 */

import {
  Output,
  Mp4OutputFormat,
  BufferTarget,
  CanvasSource,
  QUALITY_HIGH,
  canEncodeVideo,
} from 'mediabunny';
import { isElectron } from '../platform/detect';
import { useAppPreferencesStore } from '../store/appPreferencesStore';

type StartReason = 'permission' | 'unsupported' | 'error' | 'already-running';

export interface StartResult {
  started: boolean;
  reason?: StartReason;
  message?: string;
  /** Set when ``reason === 'permission'`` so the UI can route the user to System Settings. */
  permissionState?: 'denied' | 'restricted' | 'not-determined';
}

export interface StopResult {
  saved: boolean;
  filePath?: string;
  error?: string;
}

interface WindowGeometry {
  x: number;
  y: number;
  width: number;
  height: number;
  scaleFactor: number;
  displayWidth: number;
  displayHeight: number;
}

const TARGET_FRAMERATE = 30;
const TARGET_FRAME_INTERVAL_MS = 1000 / TARGET_FRAMERATE;

function timestampForFilename(): string {
  const d = new Date();
  const pad = (n: number) => n.toString().padStart(2, '0');
  return (
    `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}` +
    `-${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`
  );
}

class ScreenRecordingService {
  // Mediabunny output pipeline.
  private output: Output<Mp4OutputFormat, BufferTarget> | null = null;
  private target: BufferTarget | null = null;
  private canvasSource: CanvasSource | null = null;
  // Timestamp of the recording-start moment, used to compute per-frame
  // timestamps in seconds since start.
  private startTimeMs = 0;
  // Wall-clock of the last encoded frame, used to throttle to ~30 fps.
  private lastEncodeMs = 0;
  // True while an ``add`` call is in flight. Skip new frames during
  // backpressure rather than queueing.
  private encodeInFlight = false;

  // === Shared state ===
  // Full-screen capture stream from getDisplayMedia. We crop frames out
  // of it onto our canvas; this stream itself is never encoded directly.
  private screenStream: MediaStream | null = null;
  // Hidden <video> element backed by ``screenStream`` — needed because
  // CanvasRenderingContext2D.drawImage can sample frames from a video
  // element but not directly from a MediaStream.
  private video: HTMLVideoElement | null = null;
  // Crop region in screen device-pixel coords. Refreshed periodically
  // so moving/resizing the window during recording still produces a
  // window-aligned crop.
  private geometry: WindowGeometry | null = null;
  // requestAnimationFrame handle for the draw loop.
  private rafHandle: number | null = null;
  // setInterval handle that polls the main process for fresh window
  // geometry every second (so window moves/resizes follow the crop).
  private geometryPollHandle: ReturnType<typeof setInterval> | null = null;
  private currentSessionId: string | null = null;
  private listeners = new Set<(isRecording: boolean) => void>();
  // Bumped on every start() and on every stop(). When an in-flight start
  // resolves, it compares its captured generation against ``generation``;
  // if they differ, that start was superseded (e.g. React StrictMode
  // remounting in dev) and it tears down whatever it built instead of
  // installing it. Without this, a slow first start can overwrite a fast
  // second start's output.
  private generation = 0;
  // Set while a start() is in flight. Lets us serialize concurrent starts.
  private startInFlight: Promise<StartResult> | null = null;

  isRecording(): boolean {
    return this.output !== null && this.output.state === 'started';
  }

  /** Subscribe to recording-state changes. Returns an unsubscribe fn. */
  onStateChange(listener: (isRecording: boolean) => void): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  private notify(): void {
    const state = this.isRecording();
    for (const l of this.listeners) {
      try {
        l(state);
      } catch (err) {
        console.error('[recording] listener threw', err);
      }
    }
  }

  async start(sessionId: string): Promise<StartResult> {
    // Serialize concurrent starts. React StrictMode in dev mounts
    // components twice in rapid succession, so we get two start() calls
    // back-to-back; the second one waits for the first so we don't end
    // up with two overlapping outputs.
    if (this.startInFlight) {
      console.log('[recording] start: awaiting in-flight start');
      await this.startInFlight.catch(() => undefined);
    }

    const promise = this.startInner(sessionId);
    this.startInFlight = promise;
    try {
      return await promise;
    } finally {
      if (this.startInFlight === promise) this.startInFlight = null;
    }
  }

  private async startInner(sessionId: string): Promise<StartResult> {
    const myGen = ++this.generation;
    console.log(
      `[recording] start gen=${myGen} session=${sessionId} platform=${
        isElectron() ? 'electron' : 'web'
      }`,
    );

    if (!isElectron() || !window.electronAPI) {
      console.log('[recording] not running in Electron, skipping');
      return { started: false, reason: 'unsupported' };
    }

    if (this.output && this.output.state === 'started') {
      console.log('[recording] already recording, refusing to double-start');
      return { started: false, reason: 'already-running' };
    }

    // 1. Permission check.
    //
    // Only block on 'denied' / 'restricted' — those mean the user has
    // explicitly said no, or an MDM policy forbids it, and getDisplayMedia
    // will fail anyway. Don't block on 'not-determined': on macOS the
    // only way to trigger the system "Allow screen recording" prompt is
    // to actually attempt the capture.
    let permissionStatus:
      | 'not-determined' | 'granted' | 'denied' | 'restricted' | 'unknown'
      | undefined;
    try {
      permissionStatus = await window.electronAPI.checkScreenRecordingPermission();
      console.log(`[recording] permission status: ${permissionStatus}`);
      if (permissionStatus === 'denied' || permissionStatus === 'restricted') {
        return {
          started: false,
          reason: 'permission',
          permissionState: permissionStatus,
          message:
            'Screen recording permission is denied. Grant access in System Settings → Privacy & Security → Screen Recording, then relaunch the app.',
        };
      }
    } catch (err) {
      console.warn('[recording] permission check threw, continuing anyway', err);
    }

    // 2. Confirm the WebCodecs encoder can produce H.264.
    let avcSupported = false;
    try {
      avcSupported = await canEncodeVideo('avc');
    } catch (err) {
      console.warn('[recording] canEncodeVideo threw', err);
    }
    console.log(`[recording] AVC encode supported: ${avcSupported}`);
    if (!avcSupported) {
      return {
        started: false,
        reason: 'unsupported',
        message: 'This build does not support H.264 video encoding',
      };
    }

    // 3. Resolve the window's geometry on its display.
    let geometry: WindowGeometry | null;
    try {
      geometry = await window.electronAPI.getRecordingWindowGeometry();
      console.log('[recording] window geometry', geometry);
    } catch (err) {
      console.error('[recording] getRecordingWindowGeometry threw', err);
      return {
        started: false,
        reason: 'error',
        message: `Failed to resolve window geometry: ${err instanceof Error ? err.message : String(err)}`,
      };
    }
    if (!geometry) {
      return {
        started: false,
        reason: 'error',
        message: 'Could not resolve window geometry',
      };
    }

    // 4. Capture the full screen via getDisplayMedia. The main-process
    // setDisplayMediaRequestHandler returns a screen source for the
    // display the window lives on (window-type capture is unreliable
    // on macOS 13+). On macOS, this call is also what triggers the
    // system "Allow screen recording" prompt the first time.
    let screenStream: MediaStream;
    try {
      screenStream = await navigator.mediaDevices.getDisplayMedia({
        video: true,
        audio: false,
      });
      console.log('[recording] got screen MediaStream');
    } catch (err) {
      console.error('[recording] getDisplayMedia rejected', err);
      const message = err instanceof Error ? err.message : String(err);
      const isPermissionDenial = /permission|not allowed|NotAllowedError/i.test(
        message,
      );
      return {
        started: false,
        reason: isPermissionDenial ? 'permission' : 'error',
        permissionState: isPermissionDenial ? 'denied' : undefined,
        message: isPermissionDenial
          ? 'Screen recording permission was denied. Grant access in System Settings → Privacy & Security → Screen Recording, then relaunch the app.'
          : `getDisplayMedia failed: ${message}`,
      };
    }

    // 5. Was this start superseded while we were awaiting?
    if (myGen !== this.generation) {
      console.log(
        `[recording] gen ${myGen} superseded by ${this.generation}, discarding stream`,
      );
      screenStream.getTracks().forEach((t) => t.stop());
      return { started: false, reason: 'error', message: 'Start superseded' };
    }

    // 6. Pipe screen stream into a hidden <video>. ctx.drawImage can
    // sample frames from a video element; it can't read directly from
    // a MediaStream.
    const video = document.createElement('video');
    video.srcObject = screenStream;
    video.muted = true;
    video.playsInline = true;
    try {
      await video.play();
      if (!video.videoWidth || !video.videoHeight) {
        await new Promise<void>((resolve, reject) => {
          const t = setTimeout(
            () => reject(new Error('Timed out waiting for screen video metadata')),
            5000,
          );
          video.onloadedmetadata = () => {
            clearTimeout(t);
            resolve();
          };
        });
      }
      console.log(
        `[recording] screen video ready ${video.videoWidth}x${video.videoHeight}`,
      );
    } catch (err) {
      screenStream.getTracks().forEach((t) => t.stop());
      return {
        started: false,
        reason: 'error',
        message: `Screen video setup failed: ${err instanceof Error ? err.message : String(err)}`,
      };
    }

    // 7. Compute the effective scale between the display's logical
    // coordinate system and the captured video's pixel grid. We
    // intentionally do NOT use ``geometry.scaleFactor`` here — on
    // macOS, getDisplayMedia can hand us frames at either the
    // display's logical resolution OR its physical (Retina) one
    // depending on the OS version, the display, and how Chromium
    // negotiates with ScreenCaptureKit. Comparing video size to
    // logical display size tells us the real ratio.
    const captureScaleX = video.videoWidth / geometry.displayWidth;
    const captureScaleY = video.videoHeight / geometry.displayHeight;
    console.log(
      `[recording] capture scale: ${captureScaleX.toFixed(3)}×${captureScaleY.toFixed(3)} ` +
      `(video ${video.videoWidth}×${video.videoHeight}, ` +
      `display ${geometry.displayWidth}×${geometry.displayHeight}, ` +
      `reported scaleFactor ${geometry.scaleFactor})`,
    );

    // 8. Build the crop canvas at the window's effective capture-pixel
    // size so we copy 1:1 from the source video without scaling.
    // Locked for the rest of the session because the encoder has
    // fixed input dimensions. Both dimensions are rounded to the
    // nearest even number — H.264/AVC rejects odd width or height.
    const roundEven = (n: number) => Math.max(2, Math.round(n / 2) * 2);
    const canvas = document.createElement('canvas');
    canvas.width = roundEven(geometry.width * captureScaleX);
    canvas.height = roundEven(geometry.height * captureScaleY);
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      screenStream.getTracks().forEach((t) => t.stop());
      return {
        started: false,
        reason: 'error',
        message: 'Could not acquire 2D canvas context',
      };
    }

    // 8. Build the mediabunny output (real MP4 file). Order matters:
    // create Output → addVideoTrack(canvasSource) → start.
    const target = new BufferTarget();
    const output = new Output({
      format: new Mp4OutputFormat({ fastStart: 'in-memory' }),
      target,
    });
    const canvasSource = new CanvasSource(canvas, {
      codec: 'avc',
      bitrate: QUALITY_HIGH,
      // Allow the user to resize the window mid-recording — mediabunny
      // will containerize the frame change rather than throwing.
      sizeChangeBehavior: 'passThrough',
    });
    output.addVideoTrack(canvasSource, { frameRate: TARGET_FRAMERATE });
    try {
      await output.start();
    } catch (err) {
      console.error('[recording] output.start() threw', err);
      screenStream.getTracks().forEach((t) => t.stop());
      return {
        started: false,
        reason: 'error',
        message: `Output start failed: ${err instanceof Error ? err.message : String(err)}`,
      };
    }

    // 9. Install state and start the draw + encode loops.
    this.output = output;
    this.target = target;
    this.canvasSource = canvasSource;
    this.screenStream = screenStream;
    this.video = video;
    this.geometry = geometry;
    this.currentSessionId = sessionId;
    this.startTimeMs = performance.now();
    this.lastEncodeMs = 0;
    this.encodeInFlight = false;

    const drawFrame = () => {
      // Dropped frames once we've been torn down.
      if (this.video !== video) return;
      const geom = this.geometry;
      if (geom && video.readyState >= 2) {
        // Use the empirically-measured capture scale, not the display's
        // reported scaleFactor — see the long comment in startInner.
        const sx = Math.round(geom.x * captureScaleX);
        const sy = Math.round(geom.y * captureScaleY);
        const sw = Math.round(geom.width * captureScaleX);
        const sh = Math.round(geom.height * captureScaleY);
        try {
          ctx.drawImage(video, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
        } catch {
          // drawImage occasionally throws on the first frames before
          // the source surface is ready — ignore and try the next
          // frame.
        }
      }

      // Encode at most once per ~33ms (30 fps). Skip if a previous
      // ``add`` is still pending so we don't queue up encoder
      // backpressure.
      const now = performance.now();
      if (
        !this.encodeInFlight &&
        now - this.lastEncodeMs >= TARGET_FRAME_INTERVAL_MS &&
        this.canvasSource &&
        this.output?.state === 'started'
      ) {
        const elapsedSec = (now - this.startTimeMs) / 1000;
        const durationSec = TARGET_FRAME_INTERVAL_MS / 1000;
        this.encodeInFlight = true;
        this.lastEncodeMs = now;
        this.canvasSource
          .add(elapsedSec, durationSec)
          .catch((err) => {
            console.error('[recording] canvasSource.add threw', err);
          })
          .finally(() => {
            this.encodeInFlight = false;
          });
      }

      this.rafHandle = requestAnimationFrame(drawFrame);
    };
    drawFrame();

    // Refresh window geometry every second so moving/resizing during
    // recording follows the crop without per-frame IPC overhead.
    this.geometryPollHandle = setInterval(async () => {
      if (this.video !== video) return;
      try {
        const fresh = await window.electronAPI.getRecordingWindowGeometry();
        if (fresh) this.geometry = fresh;
      } catch {
        // ignore
      }
    }, 1000);

    this.notify();
    console.log(
      `[recording] started gen=${myGen} crop=${canvas.width}x${canvas.height}`,
    );
    return { started: true };
  }

  async stop(): Promise<StopResult> {
    // Invalidate any in-flight start. If one resolves after this point,
    // its generation check will fail and it'll discard its stream.
    this.generation++;

    const output = this.output;
    const target = this.target;
    const screenStream = this.screenStream;
    const video = this.video;
    const rafHandle = this.rafHandle;
    const geometryPollHandle = this.geometryPollHandle;
    const sessionId = this.currentSessionId;

    if (!output || !target) {
      console.log('[recording] stop: nothing to stop');
      return { saved: false, error: 'Recorder not running' };
    }
    console.log('[recording] stopping');

    // Cancel the draw loop and the geometry poller before finalizing
    // the output so no new frames are queued mid-finalize.
    if (rafHandle !== null) cancelAnimationFrame(rafHandle);
    if (geometryPollHandle !== null) clearInterval(geometryPollHandle);

    // Wait for any in-flight encode to settle before finalizing — its
    // ``add`` call still holds an open VideoFrame.
    if (this.encodeInFlight) {
      await new Promise<void>((resolve) => {
        const tick = () => {
          if (!this.encodeInFlight) resolve();
          else setTimeout(tick, 10);
        };
        tick();
      });
    }

    // Finalize the MP4 file.
    let buffer: ArrayBuffer | null = null;
    let finalizeError: string | undefined;
    try {
      await output.finalize();
      buffer = target.buffer;
    } catch (err) {
      console.error('[recording] output.finalize threw', err);
      finalizeError = err instanceof Error ? err.message : String(err);
    }

    // Tear down everything regardless of save outcome.
    if (screenStream) screenStream.getTracks().forEach((t) => t.stop());
    if (video) {
      video.pause();
      video.srcObject = null;
    }

    this.output = null;
    this.target = null;
    this.canvasSource = null;
    this.screenStream = null;
    this.video = null;
    this.geometry = null;
    this.rafHandle = null;
    this.geometryPollHandle = null;
    this.currentSessionId = null;
    this.notify();

    if (!buffer || buffer.byteLength === 0) {
      return {
        saved: false,
        error: finalizeError ?? 'No data captured',
      };
    }

    if (!window.electronAPI) {
      return { saved: false, error: 'Electron API unavailable' };
    }

    // Build filename and save.
    const folder = useAppPreferencesStore.getState().screenRecordingFolder;
    const sessionFragment = sessionId ? sessionId.slice(0, 8) : 'session';
    const filename = `tracking-${timestampForFilename()}-${sessionFragment}.mp4`;

    try {
      const result = await window.electronAPI.saveRecordingBuffer(
        buffer,
        folder,
        filename,
      );
      if (!result.success) {
        return { saved: false, error: result.error ?? 'Save failed' };
      }
      return { saved: true, filePath: result.filePath };
    } catch (err) {
      return {
        saved: false,
        error: err instanceof Error ? err.message : String(err),
      };
    }
  }
}

export const screenRecordingService = new ScreenRecordingService();
