import React from 'react';
import { useFollicleStore } from '../../store/follicleStore';
import { useProjectStore } from '../../store/projectStore';
import { isCircle, isRectangle, isLinear } from '../../types';

export const PropertyPanel: React.FC = () => {
  const selectedIds = useFollicleStore(state => state.selectedIds);
  const follicles = useFollicleStore(state => state.follicles);
  const setLabel = useFollicleStore(state => state.setLabel);
  const setNotes = useFollicleStore(state => state.setNotes);
  const deleteFollicle = useFollicleStore(state => state.deleteFollicle);
  const deleteSelected = useFollicleStore(state => state.deleteSelected);
  const updateFollicle = useFollicleStore(state => state.updateFollicle);

  const activeImageId = useProjectStore(state => state.activeImageId);

  // Get selected annotations that belong to the active image
  const selectedFollicles = follicles.filter(f => selectedIds.has(f.id) && f.imageId === activeImageId);

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
              <li><strong>Circle (1):</strong> Click center, click for radius</li>
              <li><strong>Rectangle (2):</strong> Click corner, click opposite corner</li>
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

        <div className="property-actions">
          <button
            className="delete-button"
            onClick={() => deleteSelected()}
          >
            Delete All ({selectedFollicles.length})
          </button>
        </div>
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
    </div>
  );
};
