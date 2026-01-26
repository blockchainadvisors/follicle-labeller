import { X, Info } from "lucide-react";
import "./DetectionSettingsDialog.css";

export interface DetectionSettings {
  // These settings are kept for backwards compatibility but not used by the Python server
  minWidth: number;
  maxWidth: number;
  minHeight: number;
  maxHeight: number;
  darkBlobs: boolean;
  useCLAHE: boolean;
  claheClipLimit: number;
  claheTileSize: number;
  useSAHI: boolean;
  tileSize: number;
  tileOverlap: number;
  useSoftNMS: boolean;
  softNMSSigma: number;
  softNMSThreshold: number;
}

export const DEFAULT_DETECTION_SETTINGS: DetectionSettings = {
  minWidth: 10,
  maxWidth: 200,
  minHeight: 10,
  maxHeight: 200,
  darkBlobs: true,
  useCLAHE: true,
  claheClipLimit: 3.0,
  claheTileSize: 8,
  useSAHI: false,
  tileSize: 512,
  tileOverlap: 0.2,
  useSoftNMS: true,
  softNMSSigma: 0.5,
  softNMSThreshold: 0.3,
};

interface DetectionSettingsDialogProps {
  settings: DetectionSettings;
  onSave: (settings: DetectionSettings) => void;
  onCancel: () => void;
}

export function DetectionSettingsDialog({
  onCancel,
}: DetectionSettingsDialogProps) {
  return (
    <div className="detection-settings-overlay" onClick={onCancel}>
      <div
        className="detection-settings-dialog"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="dialog-header">
          <h2>Detection Settings</h2>
          <button className="close-button" onClick={onCancel}>
            <X size={18} />
          </button>
        </div>

        <div className="dialog-content">
          <section className="settings-section">
            <div className="info-box">
              <Info size={20} />
              <div>
                <h3>Automatic Parameter Learning</h3>
                <p>
                  The detection algorithm automatically learns follicle size
                  from your annotations. Draw at least{" "}
                  <strong>3 annotations</strong> to enable auto-detection.
                </p>
              </div>
            </div>
          </section>

          <section className="settings-section">
            <h3>How It Works</h3>
            <ol className="how-it-works-list">
              <li>Draw 3 or more follicle annotations on your image</li>
              <li>The detector learns the average size from your examples</li>
              <li>Click the Auto Detect button to find similar follicles</li>
              <li>
                Detection uses OpenCV SimpleBlobDetector + contour fallback
              </li>
            </ol>
          </section>

          <section className="settings-section">
            <h3>Detection Parameters (Automatic)</h3>
            <div className="param-info">
              <div className="param-row">
                <span className="param-label">CLAHE Clip Limit:</span>
                <span className="param-value">3.0</span>
              </div>
              <div className="param-row">
                <span className="param-label">Threshold Range:</span>
                <span className="param-value">10 - 220 (22 levels)</span>
              </div>
              <div className="param-row">
                <span className="param-label">Min Circularity:</span>
                <span className="param-value">0.2</span>
              </div>
              <div className="param-row">
                <span className="param-label">IoU Overlap Threshold:</span>
                <span className="param-value">0.3</span>
              </div>
              <div className="param-row">
                <span className="param-label">Size Tolerance:</span>
                <span className="param-value">3x learned size</span>
              </div>
            </div>
          </section>
        </div>

        <div className="dialog-footer">
          <button className="button-primary" onClick={onCancel}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
