import React, { useEffect, useRef, useCallback, useState, useMemo } from "react";
import { useTrackingStore } from "../../store/trackingStore";
import { useProjectStore } from "../../store/projectStore";
import { useFollicleStore } from "../../store/follicleStore";
import { follicleTrackingService } from "../../services/follicleTrackingService";
import { CanvasRenderer } from "../Canvas/CanvasRenderer";
import type { Viewport, Follicle, RectangleAnnotation, FollicleCorrespondence } from "../../types";
import { generateId } from "../../utils/id-generator";
import "./ComparisonView.css";

const HIGHLIGHT_COLOR = "#FFD700";
const MATCH_ARROW_COLOR = "#FFD700";
const UNMATCHED_COLOR = "#808080";
const MATCHED_COLOR = "#4ECDC4";

export const ComparisonView: React.FC = () => {
  const activeSession = useTrackingStore((s) => s.getActiveSession());
  const closeComparisonView = useTrackingStore((s) => s.closeComparisonView);
  const backendSessionId = useTrackingStore((s) => s.backendSessionId);
  const addCorrespondence = useTrackingStore((s) => s.addCorrespondence);
  const images = useProjectStore((s) => s.images);
  const follicles = useFollicleStore((s) => s.follicles);
  const appendFollicles = useFollicleStore((s) => s.appendFollicles);
  const [matchingFollicleIds, setMatchingFollicleIds] = useState<Set<string>>(new Set());

  // Canvas refs
  const sourceCanvasRef = useRef<HTMLCanvasElement>(null);
  const targetCanvasRef = useRef<HTMLCanvasElement>(null);
  const sourcePanelRef = useRef<HTMLDivElement>(null);
  const targetPanelRef = useRef<HTMLDivElement>(null);

  // Renderers
  const sourceRendererRef = useRef<CanvasRenderer | null>(null);
  const targetRendererRef = useRef<CanvasRenderer | null>(null);

  // Viewport state per panel
  const [sourceViewport, setSourceViewport] = useState<Viewport>({ offsetX: 0, offsetY: 0, scale: 1 });
  const [targetViewport, setTargetViewport] = useState<Viewport>({ offsetX: 0, offsetY: 0, scale: 1 });

  // Pan state — use ref for isPanning to avoid stale closure issues in mouseUp
  const [isPanning, setIsPanning] = useState<"source" | "target" | null>(null);
  const isPanningRef = useRef<"source" | "target" | null>(null);
  const panStartRef = useRef<{ x: number; y: number; offsetX: number; offsetY: number }>({ x: 0, y: 0, offsetX: 0, offsetY: 0 });
  const panMovedRef = useRef(false);

  // Selected follicle for correspondence highlight
  const [selectedFollicleId, setSelectedFollicleId] = useState<string | null>(null);

  if (!activeSession) return null;

  const sourceImage = images.get(activeSession.sourceImageId);
  const targetImage = images.get(activeSession.targetImageId);

  if (!sourceImage || !targetImage) return null;

  const sourceFollicles = follicles.filter((f) => f.imageId === activeSession.sourceImageId);
  const targetFollicles = follicles.filter((f) => f.imageId === activeSession.targetImageId);

  // Build lookup maps for correspondences
  const correspondenceBySource = useMemo(() => {
    const map = new Map<string, FollicleCorrespondence>();
    activeSession.correspondences.forEach((c) => map.set(c.sourceAnnotationId, c));
    return map;
  }, [activeSession.correspondences]);

  const correspondenceByTarget = useMemo(() => {
    const map = new Map<string, FollicleCorrespondence>();
    activeSession.correspondences.forEach((c) => map.set(c.targetAnnotationId, c));
    return map;
  }, [activeSession.correspondences]);

  const matchedSourceIds = useMemo(
    () => new Set(activeSession.correspondences.map((c) => c.sourceAnnotationId)),
    [activeSession.correspondences]
  );
  const matchedTargetIds = useMemo(
    () => new Set(activeSession.correspondences.map((c) => c.targetAnnotationId)),
    [activeSession.correspondences]
  );

  // Find the paired follicle ID for the selected one
  const pairedFollicleId = useMemo(() => {
    if (!selectedFollicleId) return null;
    const fromSource = correspondenceBySource.get(selectedFollicleId);
    if (fromSource) return fromSource.targetAnnotationId;
    const fromTarget = correspondenceByTarget.get(selectedFollicleId);
    if (fromTarget) return fromTarget.sourceAnnotationId;
    return null;
  }, [selectedFollicleId, correspondenceBySource, correspondenceByTarget]);

  // Color follicles based on selection state
  const colorFollicles = useCallback(
    (list: Follicle[], matchedIds: Set<string>): Follicle[] => {
      return list.map((f) => {
        let color: string;
        if (f.id === selectedFollicleId || f.id === pairedFollicleId) {
          color = HIGHLIGHT_COLOR;
        } else if (matchedIds.has(f.id)) {
          color = MATCHED_COLOR;
        } else {
          color = UNMATCHED_COLOR;
        }
        return { ...f, color };
      });
    },
    [selectedFollicleId, pairedFollicleId]
  );

  const coloredSourceFollicles = colorFollicles(sourceFollicles, matchedSourceIds);
  const coloredTargetFollicles = colorFollicles(targetFollicles, matchedTargetIds);

  // Average confidence
  const avgConfidence =
    activeSession.correspondences.length > 0
      ? activeSession.correspondences.reduce((sum, c) => sum + c.confidence, 0) /
        activeSession.correspondences.length
      : 0;

  // Initialize renderers and fit images
  useEffect(() => {
    const sourceCanvas = sourceCanvasRef.current;
    const targetCanvas = targetCanvasRef.current;
    const sourcePanel = sourcePanelRef.current;
    const targetPanel = targetPanelRef.current;

    if (!sourceCanvas || !targetCanvas || !sourcePanel || !targetPanel) return;

    const sourceRect = sourcePanel.getBoundingClientRect();
    const targetRect = targetPanel.getBoundingClientRect();

    sourceCanvas.width = sourceRect.width;
    sourceCanvas.height = sourceRect.height;
    targetCanvas.width = targetRect.width;
    targetCanvas.height = targetRect.height;

    const sourceCtx = sourceCanvas.getContext("2d");
    const targetCtx = targetCanvas.getContext("2d");
    if (!sourceCtx || !targetCtx) return;

    sourceRendererRef.current = new CanvasRenderer(sourceCtx);
    targetRendererRef.current = new CanvasRenderer(targetCtx);

    sourceRendererRef.current.setImage(sourceImage.imageBitmap);
    targetRendererRef.current.setImage(targetImage.imageBitmap);

    const sourceFitScale = Math.min(
      sourceRect.width / sourceImage.width,
      sourceRect.height / sourceImage.height
    ) * 0.9;
    const targetFitScale = Math.min(
      targetRect.width / targetImage.width,
      targetRect.height / targetImage.height
    ) * 0.9;

    setSourceViewport({
      offsetX: (sourceRect.width - sourceImage.width * sourceFitScale) / 2,
      offsetY: (sourceRect.height - sourceImage.height * sourceFitScale) / 2,
      scale: sourceFitScale,
    });
    setTargetViewport({
      offsetX: (targetRect.width - targetImage.width * targetFitScale) / 2,
      offsetY: (targetRect.height - targetImage.height * targetFitScale) / 2,
      scale: targetFitScale,
    });
  }, [sourceImage, targetImage]);

  // Render canvases
  useEffect(() => {
    const sourceCanvas = sourceCanvasRef.current;
    const targetCanvas = targetCanvasRef.current;
    if (!sourceRendererRef.current || !targetRendererRef.current || !sourceCanvas || !targetCanvas) return;

    const emptyDragState = {
      isDragging: false, startPoint: null, currentPoint: null, dragType: null, targetId: null,
    };

    // Highlight selected follicle via selectedIds set
    const selectedSet = new Set<string>();
    if (selectedFollicleId) selectedSet.add(selectedFollicleId);
    if (pairedFollicleId) selectedSet.add(pairedFollicleId);

    sourceRendererRef.current.clear(sourceCanvas.width, sourceCanvas.height);
    sourceRendererRef.current.render(
      sourceCanvas.width, sourceCanvas.height,
      sourceViewport, coloredSourceFollicles,
      selectedSet, emptyDragState, false, true, "rectangle", 30
    );

    targetRendererRef.current.clear(targetCanvas.width, targetCanvas.height);
    targetRendererRef.current.render(
      targetCanvas.width, targetCanvas.height,
      targetViewport, coloredTargetFollicles,
      selectedSet, emptyDragState, false, true, "rectangle", 30
    );
  }, [sourceViewport, targetViewport, coloredSourceFollicles, coloredTargetFollicles, selectedFollicleId, pairedFollicleId]);

  // Zoom handler
  const handleWheel = useCallback(
    (e: React.WheelEvent, panel: "source" | "target") => {
      e.preventDefault();
      const setViewport = panel === "source" ? setSourceViewport : setTargetViewport;
      const viewport = panel === "source" ? sourceViewport : targetViewport;

      const delta = e.deltaY > 0 ? 0.9 : 1.1;
      const newScale = Math.max(0.01, Math.min(10, viewport.scale * delta));
      const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      setViewport({
        offsetX: mouseX - (mouseX - viewport.offsetX) * (newScale / viewport.scale),
        offsetY: mouseY - (mouseY - viewport.offsetY) * (newScale / viewport.scale),
        scale: newScale,
      });
    },
    [sourceViewport, targetViewport]
  );

  // Pan handlers
  const handleMouseDown = useCallback(
    (e: React.MouseEvent, panel: "source" | "target") => {
      const viewport = panel === "source" ? sourceViewport : targetViewport;
      isPanningRef.current = panel;
      setIsPanning(panel);
      panMovedRef.current = false;
      panStartRef.current = {
        x: e.clientX, y: e.clientY,
        offsetX: viewport.offsetX, offsetY: viewport.offsetY,
      };
    },
    [sourceViewport, targetViewport]
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      const panPanel = isPanningRef.current;
      if (!panPanel) return;
      const dx = e.clientX - panStartRef.current.x;
      const dy = e.clientY - panStartRef.current.y;
      if (Math.abs(dx) > 3 || Math.abs(dy) > 3) panMovedRef.current = true;

      const setViewport = panPanel === "source" ? setSourceViewport : setTargetViewport;
      setViewport({
        offsetX: panStartRef.current.offsetX + dx,
        offsetY: panStartRef.current.offsetY + dy,
        scale: (panPanel === "source" ? sourceViewport : targetViewport).scale,
      });
    },
    [sourceViewport, targetViewport]
  );

  const handleMouseUp = useCallback(
    (e: React.MouseEvent, panel: "source" | "target") => {
      // Read from ref to avoid stale closure — always reflects latest mouseDown
      const wasPanning = isPanningRef.current;
      isPanningRef.current = null;
      setIsPanning(null);

      // If it was a click (not a drag), try to select a follicle
      if (wasPanning === panel && !panMovedRef.current) {
        const viewport = panel === "source" ? sourceViewport : targetViewport;
        const list = panel === "source" ? sourceFollicles : targetFollicles;
        const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        // Convert screen coords to image coords
        const imgX = (mouseX - viewport.offsetX) / viewport.scale;
        const imgY = (mouseY - viewport.offsetY) / viewport.scale;

        // Find the follicle under the click
        let clicked: Follicle | null = null;
        for (const f of list) {
          if (f.shape === "rectangle") {
            const r = f as RectangleAnnotation;
            if (imgX >= r.x && imgX <= r.x + r.width && imgY >= r.y && imgY <= r.y + r.height) {
              clicked = f;
              break;
            }
          }
        }

        if (clicked) {
          // Toggle: click same follicle again to deselect
          if (clicked.id === selectedFollicleId) {
            setSelectedFollicleId(null);
          } else {
            setSelectedFollicleId(clicked.id);

            // If clicking a source follicle that isn't matched yet and
            // not already being matched, trigger single-follicle matching
            if (
              panel === "source" &&
              activeSession &&
              backendSessionId &&
              !matchedSourceIds.has(clicked.id) &&
              !matchingFollicleIds.has(clicked.id) &&
              clicked.shape === "rectangle"
            ) {
              const clickedRect = clicked as RectangleAnnotation;
              const clickedId = clicked.id;
              const sessionId = activeSession.id;
              const targetImageId = activeSession.targetImageId;
              const sourceImageId = activeSession.sourceImageId;

              setMatchingFollicleIds((prev) => new Set(prev).add(clickedId));

              follicleTrackingService
                .trackMatchSingle(backendSessionId, {
                  x: clickedRect.x,
                  y: clickedRect.y,
                  width: clickedRect.width,
                  height: clickedRect.height,
                })
                .then((result) => {
                  if (result.success && result.match) {
                    // Create target annotation from the matched detection
                    const det = result.match.targetDetection;
                    const now = Date.now();
                    const targetAnnotation: RectangleAnnotation = {
                      id: generateId(),
                      imageId: targetImageId,
                      shape: "rectangle",
                      x: det.x,
                      y: det.y,
                      width: det.width,
                      height: det.height,
                      label: `Match`,
                      notes: `Matched (conf: ${(result.match.confidence * 100).toFixed(0)}%)`,
                      color: "#4ECDC4",
                      createdAt: now,
                      updatedAt: now,
                    };

                    appendFollicles([targetAnnotation]);

                    const correspondence: FollicleCorrespondence = {
                      id: generateId(),
                      sourceAnnotationId: clickedId,
                      targetAnnotationId: targetAnnotation.id,
                      sourceImageId: sourceImageId,
                      targetImageId: targetImageId,
                      confidence: result.match.confidence,
                      transformedPosition: {
                        x: result.match.transformedX,
                        y: result.match.transformedY,
                      },
                    };

                    addCorrespondence(sessionId, correspondence);
                    console.log(
                      `Single follicle matched: confidence=${result.match.confidence.toFixed(2)}`
                    );
                  } else if (result.success && !result.match) {
                    console.log("No match found for selected follicle");
                  } else {
                    console.error("Match failed:", result.error);
                  }
                })
                .catch((err) => console.error("Match request failed:", err))
                .finally(() => {
                  setMatchingFollicleIds((prev) => {
                    const next = new Set(prev);
                    next.delete(clickedId);
                    return next;
                  });
                });
            }
          }
        } else {
          setSelectedFollicleId(null);
        }
      }
    },
    [sourceViewport, targetViewport, sourceFollicles, targetFollicles, selectedFollicleId,
     activeSession, backendSessionId, matchedSourceIds, matchingFollicleIds, appendFollicles, addCorrespondence]
  );

  // Compute arrow for selected correspondence only
  const arrow = useMemo(() => {
    if (!selectedFollicleId || !pairedFollicleId) return null;

    const sourceF = sourceFollicles.find((f) => f.id === selectedFollicleId || f.id === pairedFollicleId);
    const targetF = targetFollicles.find((f) => f.id === selectedFollicleId || f.id === pairedFollicleId);
    if (!sourceF || !targetF) return null;

    const getCenter = (f: Follicle) => {
      if (f.shape === "rectangle") {
        const r = f as RectangleAnnotation;
        return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
      }
      return { x: 0, y: 0 };
    };

    const sc = getCenter(sourceF);
    const tc = getCenter(targetF);

    const sourcePanel = sourcePanelRef.current;
    if (!sourcePanel) return null;
    const panelWidth = sourcePanel.getBoundingClientRect().width;

    return {
      x1: sc.x * sourceViewport.scale + sourceViewport.offsetX,
      y1: sc.y * sourceViewport.scale + sourceViewport.offsetY,
      x2: tc.x * targetViewport.scale + targetViewport.offsetX + panelWidth + 2,
      y2: tc.y * targetViewport.scale + targetViewport.offsetY,
    };
  }, [selectedFollicleId, pairedFollicleId, sourceFollicles, targetFollicles, sourceViewport, targetViewport]);

  // Find confidence for selected match
  const selectedConfidence = useMemo(() => {
    if (!selectedFollicleId) return null;
    const c = correspondenceBySource.get(selectedFollicleId) || correspondenceByTarget.get(selectedFollicleId);
    return c ? c.confidence : null;
  }, [selectedFollicleId, correspondenceBySource, correspondenceByTarget]);

  return (
    <div className="comparison-view" onMouseMove={handleMouseMove}>
      {/* Header */}
      <div className="comparison-header">
        <div className="comparison-header-left">
          <div className="comparison-title">Cross-Image Tracking</div>
          <div className="comparison-stats">
            <div className="comparison-stat">
              Matches: <span className="comparison-stat-value">{activeSession.correspondences.length}</span>
            </div>
            <div className="comparison-stat">
              Method: <span className="comparison-stat-value">{activeSession.method}</span>
            </div>
            <div className="comparison-stat">
              Avg Confidence: <span className="comparison-stat-value">{(avgConfidence * 100).toFixed(0)}%</span>
            </div>
            {selectedConfidence !== null && (
              <div className="comparison-stat">
                Selected: <span className="comparison-stat-value">{(selectedConfidence * 100).toFixed(0)}%</span>
              </div>
            )}
          </div>
        </div>
        <button className="comparison-close-btn" onClick={closeComparisonView} title="Close comparison">
          ✕
        </button>
      </div>

      {/* Side-by-side panels */}
      <div className="comparison-panels">
        {/* Source panel */}
        <div className="comparison-panel">
          <div className="comparison-panel-header">
            Source: <span>{sourceImage.fileName}</span> ({sourceFollicles.length} follicles)
          </div>
          <div
            className="comparison-canvas-wrapper"
            ref={sourcePanelRef}
            onWheel={(e) => handleWheel(e, "source")}
            onMouseDown={(e) => handleMouseDown(e, "source")}
            onMouseUp={(e) => handleMouseUp(e, "source")}
            style={{ cursor: isPanning === "source" ? "grabbing" : "grab" }}
          >
            <canvas ref={sourceCanvasRef} />
          </div>
        </div>

        {/* Target panel */}
        <div className="comparison-panel">
          <div className="comparison-panel-header">
            Target: <span>{targetImage.fileName}</span> ({targetFollicles.length} follicles)
          </div>
          <div
            className="comparison-canvas-wrapper"
            ref={targetPanelRef}
            onWheel={(e) => handleWheel(e, "target")}
            onMouseDown={(e) => handleMouseDown(e, "target")}
            onMouseUp={(e) => handleMouseUp(e, "target")}
            style={{ cursor: isPanning === "target" ? "grabbing" : "grab" }}
          >
            <canvas ref={targetCanvasRef} />
          </div>
        </div>

        {/* Single arrow overlay for selected correspondence */}
        {arrow && (
          <svg className="comparison-arrows-overlay">
            <defs>
              <marker id="arrowhead-selected" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill={MATCH_ARROW_COLOR} />
              </marker>
            </defs>
            <line
              x1={arrow.x1} y1={arrow.y1}
              x2={arrow.x2} y2={arrow.y2}
              stroke={MATCH_ARROW_COLOR}
              strokeWidth={2.5}
              strokeOpacity={0.9}
              markerEnd="url(#arrowhead-selected)"
            />
          </svg>
        )}
      </div>

      {/* Legend */}
      <div className="comparison-legend">
        <div className="comparison-legend-item">
          <div className="comparison-legend-swatch" style={{ background: HIGHLIGHT_COLOR }} />
          Selected pair
        </div>
        <div className="comparison-legend-item">
          <div className="comparison-legend-swatch" style={{ background: MATCHED_COLOR }} />
          Matched
        </div>
        <div className="comparison-legend-item">
          <div className="comparison-legend-swatch" style={{ background: UNMATCHED_COLOR }} />
          Unmatched
        </div>
        <span style={{ marginLeft: "auto" }}>
          Source: {matchedSourceIds.size}/{sourceFollicles.length} matched |
          Target: {matchedTargetIds.size}/{targetFollicles.length} matched
          {selectedFollicleId ? " | Click another follicle or empty area to change selection" : " | Click a follicle to see its match"}
        </span>
      </div>
    </div>
  );
};
