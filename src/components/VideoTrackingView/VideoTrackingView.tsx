import React, { useEffect, useRef, useState, useCallback } from "react";
import { useTrackingStore } from "../../store/trackingStore";
import { follicleTrackingService } from "../../services/follicleTrackingService";
import type { VideoFrameResult } from "../../types";
import "./VideoTrackingView.css";

const MATCH_COLOR = "#4ECDC4";

export const VideoTrackingView: React.FC = () => {
  const videoSession = useTrackingStore((s) => s.videoSession);
  const closeVideoTracking = useTrackingStore((s) => s.closeVideoTracking);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  const stoppedRef = useRef(false);

  const [currentFrame, setCurrentFrame] = useState(0);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [status, setStatus] = useState<"tracking" | "done" | "stopped">("tracking");

  if (!videoSession) return null;

  const { sessionId, videoFileName, fps, frameCount, videoWidth, videoHeight } = videoSession;

  // Draw a frame + match overlay on the canvas
  const drawFrame = useCallback(async (result: VideoFrameResult) => {
    const canvas = canvasRef.current;
    const content = contentRef.current;
    if (!canvas || !content || !result.frameData) return;

    const contentRect = content.getBoundingClientRect();

    // Decode the JPEG frame
    const img = new Image();
    img.src = `data:image/jpeg;base64,${result.frameData}`;
    await new Promise<void>((resolve) => { img.onload = () => resolve(); });

    // Compute display size maintaining aspect ratio
    const videoAspect = img.width / img.height;
    const containerAspect = contentRect.width / contentRect.height;

    let displayW: number, displayH: number;
    if (videoAspect > containerAspect) {
      displayW = contentRect.width;
      displayH = contentRect.width / videoAspect;
    } else {
      displayH = contentRect.height;
      displayW = contentRect.height * videoAspect;
    }

    canvas.width = displayW;
    canvas.height = displayH;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Draw the video frame
    ctx.drawImage(img, 0, 0, displayW, displayH);

    // Draw match overlay
    if (result.match) {
      const scaleX = displayW / videoWidth;
      const scaleY = displayH / videoHeight;

      const det = result.match.targetDetection;
      const x = det.x * scaleX;
      const y = det.y * scaleY;
      const w = det.width * scaleX;
      const h = det.height * scaleY;

      // Rectangle
      ctx.strokeStyle = MATCH_COLOR;
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);

      // Crosshair at predicted origin
      const cx = result.match.transformedX * scaleX;
      const cy = result.match.transformedY * scaleY;
      ctx.beginPath();
      ctx.arc(cx, cy, 6, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(cx - 10, cy);
      ctx.lineTo(cx + 10, cy);
      ctx.moveTo(cx, cy - 10);
      ctx.lineTo(cx, cy + 10);
      ctx.stroke();

      // Confidence label
      ctx.fillStyle = MATCH_COLOR;
      ctx.font = "bold 12px sans-serif";
      ctx.fillText(`${(result.match.confidence * 100).toFixed(0)}%`, x, y - 6);
    }
  }, [videoWidth, videoHeight]);

  // Main frame-by-frame processing loop
  useEffect(() => {
    if (status !== "tracking") return;

    stoppedRef.current = false;
    const frameInterval = 1000 / fps;

    const processLoop = async () => {
      while (!stoppedRef.current) {
        const startTime = performance.now();

        const result = await follicleTrackingService.videoMatchFrame(sessionId);

        if (stoppedRef.current) break;

        if (!result.success || result.done) {
          setStatus("done");
          break;
        }

        // Draw frame + overlay
        await drawFrame(result);

        setCurrentFrame(result.frameIndex);
        setConfidence(result.match ? result.match.confidence : null);

        // Cap at video FPS
        const elapsed = performance.now() - startTime;
        if (elapsed < frameInterval) {
          await new Promise((r) => setTimeout(r, frameInterval - elapsed));
        }
      }
    };

    processLoop();

    return () => {
      stoppedRef.current = true;
    };
  }, [sessionId, fps, status, drawFrame]);

  // Stop tracking
  const handleStop = useCallback(async () => {
    stoppedRef.current = true;
    setStatus("stopped");
    await follicleTrackingService.videoStop(sessionId);
  }, [sessionId]);

  // Close and cleanup
  const handleClose = useCallback(async () => {
    stoppedRef.current = true;
    try {
      await follicleTrackingService.videoStop(sessionId);
    } catch { /* ignore */ }
    closeVideoTracking();
  }, [sessionId, closeVideoTracking]);

  const progress = frameCount > 0 ? (currentFrame / frameCount) * 100 : 0;

  return (
    <div className="video-tracking-view">
      {/* Header */}
      <div className="video-tracking-header">
        <div className="video-tracking-header-left">
          <div className="video-tracking-title">Video Tracking</div>
          <div className="video-tracking-stats">
            <span>
              Frame: <span className="video-tracking-stat-value">{currentFrame}/{frameCount}</span>
            </span>
            <span>
              FPS: <span className="video-tracking-stat-value">{fps.toFixed(1)}</span>
            </span>
            <span>
              File: <span className="video-tracking-stat-value">{videoFileName}</span>
            </span>
            {confidence !== null && (
              <span>
                Confidence: <span className="video-tracking-stat-value">{(confidence * 100).toFixed(0)}%</span>
              </span>
            )}
          </div>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          {status === "tracking" && (
            <button className="video-tracking-stop-btn" onClick={handleStop}>
              Stop
            </button>
          )}
          <button className="video-tracking-stop-btn" onClick={handleClose} title="Close">
            ✕
          </button>
        </div>
      </div>

      {/* Canvas (frame + overlay drawn together) */}
      <div className="video-tracking-content" ref={contentRef}>
        <canvas ref={canvasRef} />
        <img src="/kollestee-logo.png" alt="" className="video-tracking-watermark" />
      </div>

      {/* Footer */}
      <div className="video-tracking-footer">
        <div className="video-tracking-progress">
          <div className="video-tracking-progress-bar" style={{ width: `${progress}%` }} />
        </div>
        <span>
          {status === "tracking" && "Tracking..."}
          {status === "done" && "Complete"}
          {status === "stopped" && `Stopped at frame ${currentFrame}`}
        </span>
      </div>
    </div>
  );
};
