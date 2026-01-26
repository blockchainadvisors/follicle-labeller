import { useState, useMemo, useEffect } from "react";
import { X, ChevronDown, ChevronUp, Lightbulb, Loader2 } from "lucide-react";
import { blobService } from "../../services/blobService";
import "./LearnedDetectionDialog.css";

export interface LearnedStats {
  examplesAnalyzed: number;
  minWidth: number;
  maxWidth: number;
  minHeight: number;
  maxHeight: number;
  minAspectRatio: number;
  maxAspectRatio: number;
  meanIntensity: number;
}

export interface LearnedDetectionSettings {
  tolerance: number; // 0-100, percentage to expand size range
  darkBlobs: boolean;
}

export const DEFAULT_LEARNED_SETTINGS: LearnedDetectionSettings = {
  tolerance: 20,
  darkBlobs: true,
};

interface LearnedDetectionDialogProps {
  sessionId: string;
  settings: LearnedDetectionSettings;
  onRun: (settings: LearnedDetectionSettings) => void;
  onCancel: () => void;
}

export function LearnedDetectionDialog({
  sessionId,
  settings,
  onRun,
  onCancel,
}: LearnedDetectionDialogProps) {
  const [localSettings, setLocalSettings] = useState<LearnedDetectionSettings>({
    ...settings,
  });
  const [showBestPractices, setShowBestPractices] = useState(false);
  const [stats, setStats] = useState<LearnedStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch stats from server on mount
  useEffect(() => {
    const fetchStats = async () => {
      try {
        setLoading(true);
        setError(null);
        const response = await blobService.getLearnedStats(sessionId);
        setStats(response.stats);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to fetch stats");
      } finally {
        setLoading(false);
      }
    };
    fetchStats();
  }, [sessionId]);

  // Calculate effective size range based on tolerance
  const effectiveRange = useMemo(() => {
    if (!stats) return null;

    const toleranceMultiplier = localSettings.tolerance / 100;
    const widthRange = stats.maxWidth - stats.minWidth;
    const heightRange = stats.maxHeight - stats.minHeight;

    return {
      minWidth: Math.max(
        1,
        Math.round(stats.minWidth - widthRange * toleranceMultiplier),
      ),
      maxWidth: Math.round(stats.maxWidth + widthRange * toleranceMultiplier),
      minHeight: Math.max(
        1,
        Math.round(stats.minHeight - heightRange * toleranceMultiplier),
      ),
      maxHeight: Math.round(
        stats.maxHeight + heightRange * toleranceMultiplier,
      ),
    };
  }, [stats, localSettings.tolerance]);

  const handleRun = () => {
    onRun(localSettings);
  };

  // Loading state
  if (loading) {
    return (
      <div className="learned-detection-overlay" onClick={onCancel}>
        <div
          className="learned-detection-dialog"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="dialog-header">
            <h2>Learn from Selection</h2>
            <button className="close-button" onClick={onCancel}>
              <X size={18} />
            </button>
          </div>
          <div className="dialog-content loading-content">
            <Loader2 className="spinner" size={32} />
            <p>Analyzing annotations...</p>
          </div>
        </div>
      </div>
    );
  }

  // Error state
  if (error || !stats) {
    return (
      <div className="learned-detection-overlay" onClick={onCancel}>
        <div
          className="learned-detection-dialog"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="dialog-header">
            <h2>Learn from Selection</h2>
            <button className="close-button" onClick={onCancel}>
              <X size={18} />
            </button>
          </div>
          <div className="dialog-content error-content">
            <p className="error-message">
              {error || "Failed to analyze annotations"}
            </p>
            <button className="button-secondary" onClick={onCancel}>
              Close
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Not enough annotations
  if (stats.examplesAnalyzed < 3) {
    return (
      <div className="learned-detection-overlay" onClick={onCancel}>
        <div
          className="learned-detection-dialog"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="dialog-header">
            <h2>Learn from Selection</h2>
            <button className="close-button" onClick={onCancel}>
              <X size={18} />
            </button>
          </div>
          <div className="dialog-content error-content">
            <p className="error-message">
              Need at least 3 annotations to learn from. Currently have{" "}
              {stats.examplesAnalyzed}.
            </p>
            <p className="error-hint">
              Draw rectangle or circle annotations around follicles, then try
              again.
            </p>
            <button className="button-secondary" onClick={onCancel}>
              Close
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="learned-detection-overlay" onClick={onCancel}>
      <div
        className="learned-detection-dialog"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="dialog-header">
          <h2>Learn from Selection</h2>
          <button className="close-button" onClick={onCancel}>
            <X size={18} />
          </button>
        </div>

        <div className="dialog-content">
          {/* Learned Statistics */}
          <div className="stats-box">
            <div className="stat-row">
              <span className="stat-label">Examples analyzed</span>
              <span className="stat-value">
                {stats.examplesAnalyzed} annotations
              </span>
            </div>
            <div className="stat-row">
              <span className="stat-label">Learned size range</span>
              <span className="stat-value">
                {stats.minWidth}-{stats.maxWidth}px (W) × {stats.minHeight}-
                {stats.maxHeight}px (H)
              </span>
            </div>
            <div className="stat-row">
              <span className="stat-label">Aspect ratio</span>
              <span className="stat-value">
                {stats.minAspectRatio.toFixed(2)} -{" "}
                {stats.maxAspectRatio.toFixed(2)}
              </span>
            </div>
            <div className="stat-row">
              <span className="stat-label">Mean intensity</span>
              <span className="stat-value">{stats.meanIntensity}</span>
            </div>
          </div>

          {/* Tolerance Slider */}
          <div className="tolerance-section">
            <div className="tolerance-header">
              <span className="section-label">TOLERANCE</span>
              <span className="tolerance-value">
                {localSettings.tolerance}%
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={100}
              value={localSettings.tolerance}
              onChange={(e) =>
                setLocalSettings((prev) => ({
                  ...prev,
                  tolerance: parseInt(e.target.value),
                }))
              }
              className="tolerance-slider"
            />
            {effectiveRange && (
              <div className="effective-range">
                Effective size range: {effectiveRange.minWidth}-
                {effectiveRange.maxWidth}px (W) × {effectiveRange.minHeight}-
                {effectiveRange.maxHeight}px (H)
              </div>
            )}
          </div>

          {/* Blob Type */}
          <div className="blob-type-section">
            <span className="section-label">BLOB TYPE</span>
            <div className="blob-type-buttons">
              <button
                className={`blob-type-btn ${localSettings.darkBlobs ? "active" : ""}`}
                onClick={() =>
                  setLocalSettings((prev) => ({ ...prev, darkBlobs: true }))
                }
              >
                Dark blobs
              </button>
              <button
                className={`blob-type-btn ${!localSettings.darkBlobs ? "active" : ""}`}
                onClick={() =>
                  setLocalSettings((prev) => ({ ...prev, darkBlobs: false }))
                }
              >
                Light blobs
              </button>
            </div>
            <p className="blob-type-hint">
              {localSettings.darkBlobs
                ? "Detect dark regions on light background (typical for follicles)"
                : "Detect light regions on dark background"}
            </p>
          </div>

          {/* Best Practices Collapsible */}
          <div className="best-practices-section">
            <button
              className="best-practices-toggle"
              onClick={() => setShowBestPractices(!showBestPractices)}
            >
              <Lightbulb size={16} />
              <span>Best Practices for Selection</span>
              {showBestPractices ? (
                <ChevronUp size={16} />
              ) : (
                <ChevronDown size={16} />
              )}
            </button>
            {showBestPractices && (
              <div className="best-practices-content">
                <ul>
                  <li>Select at least 10-20 representative follicles</li>
                  <li>Include both small and large examples</li>
                  <li>Select follicles from different areas of the image</li>
                  <li>Avoid selecting artifacts or non-follicle structures</li>
                  <li>Use a higher tolerance if detection misses follicles</li>
                  <li>
                    Use a lower tolerance if detection finds too many false
                    positives
                  </li>
                </ul>
              </div>
            )}
          </div>
        </div>

        <div className="dialog-footer">
          <button className="button-secondary" onClick={onCancel}>
            Cancel
          </button>
          <button className="button-primary" onClick={handleRun}>
            Run Detection
          </button>
        </div>
      </div>
    </div>
  );
}

/**
 * Calculate learned statistics from annotations.
 */
export function calculateLearnedStats(
  annotations: Array<{
    shape: string;
    x?: number;
    y?: number;
    width?: number;
    height?: number;
    center?: { x: number; y: number };
    radius?: number;
  }>,
  _imageData?: ArrayBuffer, // Reserved for future mean intensity calculation
): LearnedStats {
  if (annotations.length === 0) {
    return {
      examplesAnalyzed: 0,
      minWidth: 10,
      maxWidth: 100,
      minHeight: 10,
      maxHeight: 100,
      minAspectRatio: 1,
      maxAspectRatio: 1,
      meanIntensity: 128,
    };
  }

  const sizes: { width: number; height: number }[] = [];

  for (const ann of annotations) {
    if (ann.shape === "rectangle" && ann.width && ann.height) {
      sizes.push({ width: ann.width, height: ann.height });
    } else if (ann.shape === "circle" && ann.radius) {
      const diameter = ann.radius * 2;
      sizes.push({ width: diameter, height: diameter });
    }
  }

  if (sizes.length === 0) {
    return {
      examplesAnalyzed: 0,
      minWidth: 10,
      maxWidth: 100,
      minHeight: 10,
      maxHeight: 100,
      minAspectRatio: 1,
      maxAspectRatio: 1,
      meanIntensity: 128,
    };
  }

  const widths = sizes.map((s) => s.width);
  const heights = sizes.map((s) => s.height);
  const aspectRatios = sizes.map((s) => s.width / s.height);

  return {
    examplesAnalyzed: sizes.length,
    minWidth: Math.round(Math.min(...widths)),
    maxWidth: Math.round(Math.max(...widths)),
    minHeight: Math.round(Math.min(...heights)),
    maxHeight: Math.round(Math.max(...heights)),
    minAspectRatio: Math.min(...aspectRatios),
    maxAspectRatio: Math.max(...aspectRatios),
    meanIntensity: 92, // Placeholder - would need image data to calculate
  };
}
