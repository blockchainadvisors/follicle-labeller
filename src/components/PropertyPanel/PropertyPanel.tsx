import React from 'react';
import { useFollicleStore } from '../../store/follicleStore';
import { isCircle, isRectangle, isLinear } from '../../types';

export const PropertyPanel: React.FC = () => {
  const selectedId = useFollicleStore(state => state.selectedId);
  const follicles = useFollicleStore(state => state.follicles);
  const setLabel = useFollicleStore(state => state.setLabel);
  const setNotes = useFollicleStore(state => state.setNotes);
  const deleteFollicle = useFollicleStore(state => state.deleteFollicle);
  const updateFollicle = useFollicleStore(state => state.updateFollicle);

  const selected = follicles.find(f => f.id === selectedId);

  if (!selected) {
    return (
      <div className="property-panel empty">
        <div className="empty-state">
          <h3>No Selection</h3>
          <p>Select an annotation to edit its properties</p>
          <div className="tips">
            <h4>Tips:</h4>
            <ul>
              <li><strong>Create:</strong> Click and drag to draw a shape</li>
              <li><strong>Circle (1):</strong> Click center, drag radius</li>
              <li><strong>Rectangle (2):</strong> Click corner, drag size</li>
              <li><strong>Linear (3):</strong> Drag line, then click for width</li>
              <li><strong>Select:</strong> Click on an annotation</li>
              <li><strong>Move:</strong> Drag a selected annotation</li>
              <li><strong>Resize:</strong> Drag the edge/corner handles</li>
              <li><strong>Delete:</strong> Press Delete key</li>
            </ul>
          </div>
        </div>
      </div>
    );
  }

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
