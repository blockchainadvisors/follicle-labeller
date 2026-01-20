import React, { useState, useCallback } from 'react';
import { X } from 'lucide-react';
import type { LearnedDetectionParams } from '../../types';
import { applyTolerance, formatSizeRange, formatAspectRatio } from '../../services/parameterLearner';

interface LearnedDetectionDialogProps {
  params: LearnedDetectionParams;
  onRun: (tolerance: number, darkBlobs: boolean) => void;
  onCancel: () => void;
}

export const LearnedDetectionDialog: React.FC<LearnedDetectionDialogProps> = ({
  params,
  onRun,
  onCancel,
}) => {
  const [tolerance, setTolerance] = useState(0.2); // 20% default
  const [darkBlobs, setDarkBlobs] = useState(true);

  const effectiveRange = applyTolerance(params, tolerance);
  const tolerancePercent = Math.round(tolerance * 100);

  const handleRun = useCallback(() => {
    onRun(tolerance, darkBlobs);
  }, [tolerance, darkBlobs, onRun]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      onCancel();
    } else if (e.key === 'Enter') {
      handleRun();
    }
  }, [onCancel, handleRun]);

  return (
    <div className="dialog-overlay" onKeyDown={handleKeyDown}>
      <div className="dialog-content learned-detection-dialog">
        <div className="dialog-header">
          <h3>Learn from Selection</h3>
          <button className="dialog-close" onClick={onCancel} title="Close (Esc)">
            <X size={18} />
          </button>
        </div>

        <div className="dialog-body">
          <div className="learned-summary">
            <div className="summary-item">
              <span className="summary-label">Examples analyzed</span>
              <span className="summary-value">{params.exampleCount} annotations</span>
            </div>
            <div className="summary-item">
              <span className="summary-label">Learned size range</span>
              <span className="summary-value">{formatSizeRange(params)}</span>
            </div>
            <div className="summary-item">
              <span className="summary-label">Aspect ratio</span>
              <span className="summary-value">{formatAspectRatio(params)}</span>
            </div>
            {params.meanIntensity !== undefined && (
              <div className="summary-item">
                <span className="summary-label">Mean intensity</span>
                <span className="summary-value">{Math.round(params.meanIntensity)}</span>
              </div>
            )}
          </div>

          <div className="dialog-section">
            <label className="section-label">Tolerance</label>
            <div className="tolerance-control">
              <input
                type="range"
                min="0"
                max="50"
                value={tolerancePercent}
                onChange={(e) => setTolerance(parseInt(e.target.value) / 100)}
                className="tolerance-slider"
              />
              <span className="tolerance-value">{tolerancePercent}%</span>
            </div>
            <div className="effective-range">
              <span className="range-label">Effective size range:</span>
              <span className="range-value">
                {effectiveRange.minWidth}-{effectiveRange.maxWidth}px (W) × {effectiveRange.minHeight}-{effectiveRange.maxHeight}px (H)
              </span>
            </div>
          </div>

          <div className="dialog-section">
            <label className="section-label">Blob Type</label>
            <div className="blob-type-toggle">
              <button
                className={`toggle-btn ${darkBlobs ? 'active' : ''}`}
                onClick={() => setDarkBlobs(true)}
              >
                Dark blobs
              </button>
              <button
                className={`toggle-btn ${!darkBlobs ? 'active' : ''}`}
                onClick={() => setDarkBlobs(false)}
              >
                Light blobs
              </button>
            </div>
            <p className="blob-type-hint">
              {darkBlobs
                ? 'Detect dark regions on light background (typical for follicles)'
                : 'Detect light regions on dark background'}
            </p>
          </div>
        </div>

        <div className="dialog-footer">
          <button className="dialog-btn cancel" onClick={onCancel}>
            Cancel
          </button>
          <button className="dialog-btn primary" onClick={handleRun}>
            Run Detection
          </button>
        </div>
      </div>
    </div>
  );
};
