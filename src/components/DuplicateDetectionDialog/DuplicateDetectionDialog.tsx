import React from "react";
import { X, AlertTriangle, CheckCircle2, Copy, Sparkles } from "lucide-react";
import "./DuplicateDetectionDialog.css";

export interface DuplicateReport {
  totalDetected: number;
  duplicates: number;
  newDetections: number;
  duplicateIndices: Set<number>; // Indices of detections that are duplicates
}

export type DuplicateAction = "keepNew" | "keepAll" | "cancel";

interface DuplicateDetectionDialogProps {
  report: DuplicateReport;
  onAction: (action: DuplicateAction) => void;
  detectionMethod: string; // e.g., "YOLO", "BLOB", "Learned"
}

export const DuplicateDetectionDialog: React.FC<DuplicateDetectionDialogProps> = ({
  report,
  onAction,
  detectionMethod,
}) => {
  const hasDuplicates = report.duplicates > 0;
  const hasNewDetections = report.newDetections > 0;

  return (
    <div className="duplicate-detection-overlay" onClick={() => onAction("cancel")}>
      <div className="duplicate-detection-dialog" onClick={e => e.stopPropagation()}>
        <div className="dialog-header">
          <h2>
            {hasDuplicates ? (
              <><AlertTriangle size={20} /> Detection Results</>
            ) : (
              <><CheckCircle2 size={20} /> Detection Complete</>
            )}
          </h2>
          <button
            className="close-button"
            onClick={() => onAction("cancel")}
            aria-label="Close"
          >
            <X size={18} />
          </button>
        </div>

        <div className="dialog-content">
          <div className="detection-summary">
            <div className="summary-item">
              <Sparkles size={16} />
              <span className="summary-label">Detected by {detectionMethod}:</span>
              <span className="summary-value">{report.totalDetected}</span>
            </div>

            {hasDuplicates && (
              <div className="summary-item duplicates">
                <Copy size={16} />
                <span className="summary-label">Duplicates (overlap with existing):</span>
                <span className="summary-value duplicate-count">{report.duplicates}</span>
              </div>
            )}

            <div className="summary-item new-detections">
              <CheckCircle2 size={16} />
              <span className="summary-label">New unique detections:</span>
              <span className="summary-value new-count">{report.newDetections}</span>
            </div>
          </div>

          {hasDuplicates && (
            <p className="duplicate-info">
              {report.duplicates} detection{report.duplicates !== 1 ? "s" : ""} overlap
              significantly with existing annotations and may be duplicates.
            </p>
          )}

          {!hasNewDetections && hasDuplicates && (
            <p className="warning-text">
              All detections appear to be duplicates of existing annotations.
            </p>
          )}
        </div>

        <div className="dialog-footer">
          <button
            className="button-secondary"
            onClick={() => onAction("cancel")}
          >
            Cancel
          </button>

          <div className="footer-actions">
            {hasDuplicates && hasNewDetections && (
              <button
                className="button-primary"
                onClick={() => onAction("keepNew")}
              >
                Keep Only New ({report.newDetections})
              </button>
            )}

            <button
              className={hasDuplicates && hasNewDetections ? "button-secondary" : "button-primary"}
              onClick={() => onAction("keepAll")}
              disabled={report.totalDetected === 0}
            >
              {hasDuplicates ? `Keep All (${report.totalDetected})` : `Add All (${report.totalDetected})`}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

// Utility function to calculate IoU (Intersection over Union) between two bounding boxes
function calculateIoU(
  box1: { x: number; y: number; width: number; height: number },
  box2: { x: number; y: number; width: number; height: number }
): number {
  const x1 = Math.max(box1.x, box2.x);
  const y1 = Math.max(box1.y, box2.y);
  const x2 = Math.min(box1.x + box1.width, box2.x + box2.width);
  const y2 = Math.min(box1.y + box1.height, box2.y + box2.height);

  if (x2 <= x1 || y2 <= y1) {
    return 0; // No intersection
  }

  const intersection = (x2 - x1) * (y2 - y1);
  const area1 = box1.width * box1.height;
  const area2 = box2.width * box2.height;
  const union = area1 + area2 - intersection;

  return intersection / union;
}

// Convert any annotation shape to a bounding box
function annotationToBoundingBox(annotation: {
  shape: string;
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  center?: { x: number; y: number };
  radius?: number;
  startPoint?: { x: number; y: number };
  endPoint?: { x: number; y: number };
  halfWidth?: number;
}): { x: number; y: number; width: number; height: number } | null {
  if (annotation.shape === "rectangle") {
    return {
      x: annotation.x!,
      y: annotation.y!,
      width: annotation.width!,
      height: annotation.height!,
    };
  } else if (annotation.shape === "circle") {
    const r = annotation.radius!;
    return {
      x: annotation.center!.x - r,
      y: annotation.center!.y - r,
      width: r * 2,
      height: r * 2,
    };
  } else if (annotation.shape === "linear") {
    // Calculate bounding box for linear annotation
    const startX = annotation.startPoint!.x;
    const startY = annotation.startPoint!.y;
    const endX = annotation.endPoint!.x;
    const endY = annotation.endPoint!.y;
    const halfWidth = annotation.halfWidth!;

    // Get the perpendicular offset
    const dx = endX - startX;
    const dy = endY - startY;
    const len = Math.sqrt(dx * dx + dy * dy);
    if (len === 0) return null;

    const perpX = (-dy / len) * halfWidth;
    const perpY = (dx / len) * halfWidth;

    // Calculate all 4 corners
    const corners = [
      { x: startX + perpX, y: startY + perpY },
      { x: startX - perpX, y: startY - perpY },
      { x: endX + perpX, y: endY + perpY },
      { x: endX - perpX, y: endY - perpY },
    ];

    const minX = Math.min(...corners.map(c => c.x));
    const maxX = Math.max(...corners.map(c => c.x));
    const minY = Math.min(...corners.map(c => c.y));
    const maxY = Math.max(...corners.map(c => c.y));

    return {
      x: minX,
      y: minY,
      width: maxX - minX,
      height: maxY - minY,
    };
  }
  return null;
}

// Main function to find duplicates between new detections and existing annotations
export function findDuplicates(
  newDetections: Array<{ x: number; y: number; width: number; height: number }>,
  existingAnnotations: Array<{
    shape: string;
    x?: number;
    y?: number;
    width?: number;
    height?: number;
    center?: { x: number; y: number };
    radius?: number;
    startPoint?: { x: number; y: number };
    endPoint?: { x: number; y: number };
    halfWidth?: number;
  }>,
  iouThreshold: number = 0.5
): DuplicateReport {
  const duplicateIndices = new Set<number>();

  // Convert existing annotations to bounding boxes
  const existingBoxes = existingAnnotations
    .map(annotationToBoundingBox)
    .filter((box): box is NonNullable<typeof box> => box !== null);

  // Check each new detection against existing annotations
  newDetections.forEach((detection, index) => {
    for (const existingBox of existingBoxes) {
      const iou = calculateIoU(detection, existingBox);
      if (iou >= iouThreshold) {
        duplicateIndices.add(index);
        break; // Found a duplicate, no need to check more
      }
    }
  });

  return {
    totalDetected: newDetections.length,
    duplicates: duplicateIndices.size,
    newDetections: newDetections.length - duplicateIndices.size,
    duplicateIndices,
  };
}
