import React, { useState, useCallback } from 'react';
import { Lock, Unlock, AlertTriangle, Crosshair, Loader2 } from 'lucide-react';
import { useFollicleStore } from '../../store/follicleStore';
import { useProjectStore } from '../../store/projectStore';
import { isCircle, isRectangle, isLinear, RectangleAnnotation } from '../../types';
import { blobService } from '../../services/blobService';
import { yoloKeypointService } from '../../services/yoloKeypointService';

export const PropertyPanel: React.FC = () => {
  const [showUnlockWarning, setShowUnlockWarning] = useState(false);
  const [showBatchUnlockWarning, setShowBatchUnlockWarning] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionError, setPredictionError] = useState<string | null>(null);

  const selectedIds = useFollicleStore(state => state.selectedIds);
  const follicles = useFollicleStore(state => state.follicles);
  const setLabel = useFollicleStore(state => state.setLabel);
  const setNotes = useFollicleStore(state => state.setNotes);
  const deleteFollicle = useFollicleStore(state => state.deleteFollicle);
  const deleteSelected = useFollicleStore(state => state.deleteSelected);
  const updateFollicle = useFollicleStore(state => state.updateFollicle);
  const selectMultiple = useFollicleStore(state => state.selectMultiple);

  const activeImageId = useProjectStore(state => state.activeImageId);

  // Get selected annotations that belong to the active image
  const selectedFollicles = follicles.filter(f => selectedIds.has(f.id) && f.imageId === activeImageId);

  // Get rectangles without origins (eligible for prediction)
  const rectanglesWithoutOrigins = selectedFollicles.filter(
    (f): f is RectangleAnnotation => isRectangle(f) && !f.origin
  );

  // Handle prediction for selected rectangles
  const handlePredictOrigins = useCallback(async () => {
    if (rectanglesWithoutOrigins.length === 0) return;

    const sessionId = blobService.getSessionId();
    if (!sessionId) {
      setPredictionError('No image session. Please load an image first.');
      return;
    }

    setIsPredicting(true);
    setPredictionError(null);

    try {
      // Check if models are available
      const models = await yoloKeypointService.listModels();
      if (models.length === 0) {
        setPredictionError('No trained YOLO models available. Train a model first.');
        return;
      }

      // Predict origins for rectangles without them
      const predictions = await blobService.predictOriginsForRectangles(
        sessionId,
        rectanglesWithoutOrigins.map(r => ({
          id: r.id,
          x: r.x,
          y: r.y,
          width: r.width,
          height: r.height,
        }))
      );

      // Update annotations with predicted origins
      let successCount = 0;
      for (const [id, origin] of predictions) {
        updateFollicle(id, { origin });
        successCount++;
      }

      if (successCount === 0) {
        setPredictionError('No origins could be predicted. Try with different annotations.');
      } else if (successCount < rectanglesWithoutOrigins.length) {
        setPredictionError(`Predicted ${successCount}/${rectanglesWithoutOrigins.length} origins.`);
      }
    } catch (error) {
      console.error('Failed to predict origins:', error);
      setPredictionError(error instanceof Error ? error.message : 'Prediction failed');
    } finally {
      setIsPredicting(false);
    }
  }, [rectanglesWithoutOrigins, updateFollicle]);

  // No selection
  if (selectedFollicles.length === 0) {
    return (
      <div className="property-panel empty">
        <div className="empty-state">
          <h3>No Selection</h3>
          <p>Select an annotation to edit its properties</p>
          <div className="tips">
            <h4>Tips:</h4>
            <ul>
              <li><strong>Create:</strong> Click to start, click again to finish</li>
              <li><strong>Rectangle (1):</strong> Click corner, click opposite corner</li>
              <li><strong>Circle (2):</strong> Click center, click for radius</li>
              <li><strong>Linear (3):</strong> Click start, click end, click for width</li>
              <li><strong>Select:</strong> Click on an annotation</li>
              <li><strong>Ctrl+Click:</strong> Add/remove from selection</li>
              <li><strong>Marquee (M):</strong> Drag to select multiple</li>
              <li><strong>Lasso (F):</strong> Draw to select multiple</li>
              <li><strong>Ctrl+A:</strong> Select all annotations</li>
              <li><strong>Move:</strong> Drag selected annotation(s)</li>
              <li><strong>Resize:</strong> Drag handles (single selection)</li>
              <li><strong>Cancel:</strong> Press Escape to cancel/deselect</li>
              <li><strong>Delete:</strong> Press Delete key</li>
            </ul>
          </div>
        </div>
      </div>
    );
  }

  // Multi-selection view
  if (selectedFollicles.length > 1) {
    // Count by type
    const circleCount = selectedFollicles.filter(isCircle).length;
    const rectangleCount = selectedFollicles.filter(isRectangle).length;
    const linearCount = selectedFollicles.filter(isLinear).length;

    // Count rectangles with/without origins
    const allRectangles = selectedFollicles.filter(isRectangle);
    const rectanglesWithOrigin = allRectangles.filter(r => r.origin);
    const rectanglesWithoutOrigin = allRectangles.filter(r => !r.origin);
    const hasRectangles = allRectangles.length > 0;

    // Filter handlers - modify selection to keep only matching annotations
    const handleKeepWithOrigin = () => {
      const toKeep = selectedFollicles
        .filter(f => !isRectangle(f) || f.origin !== undefined)
        .map(f => f.id);
      selectMultiple(toKeep);
    };

    const handleKeepWithoutOrigin = () => {
      const toKeep = selectedFollicles
        .filter(f => !isRectangle(f) || f.origin === undefined)
        .map(f => f.id);
      selectMultiple(toKeep);
    };

    // Batch unlock handler - removes origins from all selected rectangles
    const handleBatchUnlock = () => {
      for (const rect of rectanglesWithOrigin) {
        updateFollicle(rect.id, { origin: undefined });
      }
      setShowBatchUnlockWarning(false);
    };

    // Batch color change handler
    const handleBatchColorChange = (color: string) => {
      for (const f of selectedFollicles) {
        updateFollicle(f.id, { color });
      }
    };

    return (
      <div className="property-panel multi-selection">
        <h3>Multiple Selection</h3>

        <div className="property-group readonly">
          <label>Selected</label>
          <span className="type-display">{selectedFollicles.length} items</span>
        </div>

        <div className="property-group readonly">
          <label>Breakdown</label>
          <div className="selection-breakdown">
            {circleCount > 0 && <span>{circleCount} Circle{circleCount > 1 ? 's' : ''}</span>}
            {rectangleCount > 0 && <span>{rectangleCount} Rectangle{rectangleCount > 1 ? 's' : ''}</span>}
            {linearCount > 0 && <span>{linearCount} Linear{linearCount > 1 ? 's' : ''}</span>}
          </div>
        </div>

        {/* Origin filter for rectangles - modifies selection */}
        {hasRectangles && (
          <div className="property-group">
            <label>Filter Selection</label>
            <div className="origin-filter-buttons">
              <button
                className="filter-button"
                onClick={handleKeepWithOrigin}
                disabled={rectanglesWithOrigin.length === 0}
              >
                Keep With Origin ({rectanglesWithOrigin.length})
              </button>
              <button
                className="filter-button"
                onClick={handleKeepWithoutOrigin}
                disabled={rectanglesWithoutOrigin.length === 0}
              >
                Keep Without Origin ({rectanglesWithoutOrigin.length})
              </button>
            </div>
          </div>
        )}

        {/* Batch unlock for rectangles with origins */}
        {rectanglesWithOrigin.length > 0 && (
          <div className="property-group">
            <label>Batch Unlock</label>
            <button
              className="unlock-button batch-unlock"
              onClick={() => setShowBatchUnlockWarning(true)}
            >
              <Unlock size={14} />
              Unlock All ({rectanglesWithOrigin.length})
            </button>
          </div>
        )}

        <div className="property-group">
          <label htmlFor="batch-color">Batch Color</label>
          <div className="color-picker">
            <input
              id="batch-color"
              type="color"
              defaultValue="#4ECDC4"
              onChange={(e) => handleBatchColorChange(e.target.value)}
            />
            <span className="color-hint">Apply to all selected</span>
          </div>
        </div>

        {/* Predict Origins button for unlocked rectangles */}
        {rectanglesWithoutOrigins.length > 0 && (
          <div className="property-group">
            <label>Origin Prediction</label>
            <button
              className="predict-origins-button"
              onClick={handlePredictOrigins}
              disabled={isPredicting}
            >
              {isPredicting ? (
                <>
                  <Loader2 size={14} className="spin" />
                  Predicting...
                </>
              ) : (
                <>
                  <Crosshair size={14} />
                  Predict Origins ({rectanglesWithoutOrigins.length})
                </>
              )}
            </button>
            {predictionError && (
              <span className="prediction-error">{predictionError}</span>
            )}
          </div>
        )}

        <div className="property-actions">
          <button
            className="delete-button"
            onClick={() => deleteSelected()}
          >
            Delete All ({selectedFollicles.length})
          </button>
        </div>

        {/* Batch unlock confirmation dialog */}
        {showBatchUnlockWarning && (
          <div className="unlock-warning-overlay">
            <div className="unlock-warning-dialog">
              <AlertTriangle size={24} className="warning-icon" />
              <p className="warning-title">Unlock {rectanglesWithOrigin.length} Rectangles?</p>
              <p className="warning-text">
                This will delete the origin point and direction data from all {rectanglesWithOrigin.length} selected rectangles with origins.
                You will need to set them again.
              </p>
              <div className="warning-actions">
                <button
                  className="cancel-button"
                  onClick={() => setShowBatchUnlockWarning(false)}
                >
                  Cancel
                </button>
                <button
                  className="danger-button"
                  onClick={handleBatchUnlock}
                >
                  Unlock All
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  }

  // Single selection - original behavior
  const selected = selectedFollicles[0];

  const shapeLabel = isCircle(selected) ? 'Circle' : isRectangle(selected) ? 'Rectangle' : 'Linear';

  return (
    <div className="property-panel">
      <h3>{shapeLabel} Properties</h3>

      <div className="property-group readonly">
        <label>Type</label>
        <span className="type-display">{shapeLabel}</span>
      </div>

      <div className="property-group">
        <label htmlFor="label">Label</label>
        <input
          id="label"
          type="text"
          value={selected.label}
          onChange={(e) => setLabel(selected.id, e.target.value)}
          placeholder="Enter label..."
        />
      </div>

      <div className="property-group">
        <label htmlFor="notes">Notes</label>
        <textarea
          id="notes"
          value={selected.notes}
          onChange={(e) => setNotes(selected.id, e.target.value)}
          rows={4}
          placeholder="Add notes..."
        />
      </div>

      {isCircle(selected) && (
        <>
          <div className="property-group readonly">
            <label>Center</label>
            <div className="coordinate-display">
              <span>X: {Math.round(selected.center.x)}</span>
              <span>Y: {Math.round(selected.center.y)}</span>
            </div>
          </div>

          <div className="property-group readonly">
            <label>Radius</label>
            <span className="radius-display">{Math.round(selected.radius)} px</span>
          </div>
        </>
      )}

      {isRectangle(selected) && (
        <>
          <div className="property-group readonly">
            <label>Position</label>
            <div className="coordinate-display">
              <span>X: {Math.round(selected.x)}</span>
              <span>Y: {Math.round(selected.y)}</span>
            </div>
          </div>

          <div className="property-group readonly">
            <label>Size</label>
            <div className="coordinate-display">
              <span>W: {Math.round(selected.width)} px</span>
              <span>H: {Math.round(selected.height)} px</span>
            </div>
          </div>

          {/* Lock status and origin info */}
          {selected.origin ? (
            <>
              <div className="property-group lock-status">
                <div className="lock-info">
                  <Lock size={14} />
                  <span>Locked (origin set)</span>
                </div>
                <button
                  className="unlock-button"
                  onClick={() => setShowUnlockWarning(true)}
                >
                  <Unlock size={14} />
                  Unlock
                </button>
              </div>

              <div className="property-group readonly">
                <label>Origin Point</label>
                <div className="coordinate-display">
                  <span>X: {selected.origin.originPoint.x.toFixed(1)}</span>
                  <span>Y: {selected.origin.originPoint.y.toFixed(1)}</span>
                </div>
              </div>

              <div className="property-group readonly">
                <label>Direction</label>
                <span className="radius-display">
                  {(selected.origin.directionAngle * 180 / Math.PI).toFixed(0)}°
                </span>
              </div>
            </>
          ) : (
            <div className="property-group origin-actions">
              <span className="hint-text">Double-click to set origin manually</span>
              <button
                className="predict-origins-button small"
                onClick={handlePredictOrigins}
                disabled={isPredicting}
              >
                {isPredicting ? (
                  <>
                    <Loader2 size={14} className="spin" />
                    Predicting...
                  </>
                ) : (
                  <>
                    <Crosshair size={14} />
                    Predict Origin
                  </>
                )}
              </button>
              {predictionError && (
                <span className="prediction-error">{predictionError}</span>
              )}
            </div>
          )}
        </>
      )}

      {isLinear(selected) && (
        <>
          <div className="property-group readonly">
            <label>Start Point</label>
            <div className="coordinate-display">
              <span>X: {Math.round(selected.startPoint.x)}</span>
              <span>Y: {Math.round(selected.startPoint.y)}</span>
            </div>
          </div>

          <div className="property-group readonly">
            <label>End Point</label>
            <div className="coordinate-display">
              <span>X: {Math.round(selected.endPoint.x)}</span>
              <span>Y: {Math.round(selected.endPoint.y)}</span>
            </div>
          </div>

          <div className="property-group readonly">
            <label>Dimensions</label>
            <div className="coordinate-display">
              <span>L: {Math.round(Math.sqrt(
                Math.pow(selected.endPoint.x - selected.startPoint.x, 2) +
                Math.pow(selected.endPoint.y - selected.startPoint.y, 2)
              ))} px</span>
              <span>W: {Math.round(selected.halfWidth * 2)} px</span>
            </div>
          </div>

          <div className="property-group readonly">
            <label>Angle</label>
            <span className="radius-display">
              {Math.round(Math.atan2(
                selected.endPoint.y - selected.startPoint.y,
                selected.endPoint.x - selected.startPoint.x
              ) * 180 / Math.PI)}°
            </span>
          </div>
        </>
      )}

      <div className="property-group">
        <label htmlFor="color">Color</label>
        <div className="color-picker">
          <input
            id="color"
            type="color"
            value={selected.color}
            onChange={(e) => updateFollicle(selected.id, { color: e.target.value })}
          />
          <span className="color-value">{selected.color}</span>
        </div>
      </div>

      <div className="property-actions">
        <button
          className="delete-button"
          onClick={() => deleteFollicle(selected.id)}
        >
          Delete {shapeLabel}
        </button>
      </div>

      {/* Unlock confirmation dialog */}
      {showUnlockWarning && isRectangle(selected) && (
        <div className="unlock-warning-overlay">
          <div className="unlock-warning-dialog">
            <AlertTriangle size={24} className="warning-icon" />
            <p className="warning-title">Unlock Rectangle?</p>
            <p className="warning-text">
              Unlocking will delete the origin point and direction data.
              You will need to set them again.
            </p>
            <div className="warning-actions">
              <button
                className="cancel-button"
                onClick={() => setShowUnlockWarning(false)}
              >
                Cancel
              </button>
              <button
                className="danger-button"
                onClick={() => {
                  updateFollicle(selected.id, { origin: undefined });
                  setShowUnlockWarning(false);
                }}
              >
                Unlock & Delete Origin
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
