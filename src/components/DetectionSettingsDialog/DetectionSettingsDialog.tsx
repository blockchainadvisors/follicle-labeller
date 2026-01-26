import { useState } from "react";
import { X, RotateCcw } from "lucide-react";
import "./DetectionSettingsDialog.css";

export interface DetectionSettings {
  // Size range settings
  minWidth: number;
  maxWidth: number;
  minHeight: number;
  maxHeight: number;
  darkBlobs: boolean;

  // CLAHE settings
  useCLAHE: boolean;
  claheClipLimit: number;
  claheTileSize: number;

  // SAHI settings (for future use)
  useSAHI: boolean;
  tileSize: number;
  tileOverlap: number;

  // Soft-NMS settings (for future use)
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
  settings,
  onSave,
  onCancel,
}: DetectionSettingsDialogProps) {
  const [localSettings, setLocalSettings] = useState<DetectionSettings>({
    ...settings,
  });

  const handleChange = (
    key: keyof DetectionSettings,
    value: number | boolean,
  ) => {
    setLocalSettings((prev) => ({ ...prev, [key]: value }));
  };

  const handleReset = () => {
    setLocalSettings({ ...DEFAULT_DETECTION_SETTINGS });
  };

  const handleSave = () => {
    onSave(localSettings);
  };

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
          {/* Size Range Section */}
          <section className="settings-section">
            <h3>Size Range</h3>
            <p className="section-description">
              Set the minimum and maximum follicle size to detect. If you have
              3+ annotations, size will be learned automatically unless you set
              manual values here.
            </p>

            <div className="settings-grid">
              <div className="setting-row">
                <label htmlFor="minWidth">Min Width</label>
                <input
                  type="number"
                  id="minWidth"
                  min={1}
                  max={500}
                  value={localSettings.minWidth}
                  onChange={(e) =>
                    handleChange("minWidth", parseInt(e.target.value) || 1)
                  }
                />
                <span className="setting-hint">px</span>
              </div>

              <div className="setting-row">
                <label htmlFor="maxWidth">Max Width</label>
                <input
                  type="number"
                  id="maxWidth"
                  min={1}
                  max={500}
                  value={localSettings.maxWidth}
                  onChange={(e) =>
                    handleChange("maxWidth", parseInt(e.target.value) || 1)
                  }
                />
                <span className="setting-hint">px</span>
              </div>

              <div className="setting-row">
                <label htmlFor="minHeight">Min Height</label>
                <input
                  type="number"
                  id="minHeight"
                  min={1}
                  max={500}
                  value={localSettings.minHeight}
                  onChange={(e) =>
                    handleChange("minHeight", parseInt(e.target.value) || 1)
                  }
                />
                <span className="setting-hint">px</span>
              </div>

              <div className="setting-row">
                <label htmlFor="maxHeight">Max Height</label>
                <input
                  type="number"
                  id="maxHeight"
                  min={1}
                  max={500}
                  value={localSettings.maxHeight}
                  onChange={(e) =>
                    handleChange("maxHeight", parseInt(e.target.value) || 1)
                  }
                />
                <span className="setting-hint">px</span>
              </div>
            </div>

            <div className="checkbox-row">
              <input
                type="checkbox"
                id="darkBlobs"
                checked={localSettings.darkBlobs}
                onChange={(e) => handleChange("darkBlobs", e.target.checked)}
              />
              <label htmlFor="darkBlobs">
                Dark Blobs (detect dark regions on light background)
              </label>
            </div>
          </section>

          {/* CLAHE Section */}
          <section className="settings-section">
            <div className="section-header">
              <div className="checkbox-row">
                <input
                  type="checkbox"
                  id="useCLAHE"
                  checked={localSettings.useCLAHE}
                  onChange={(e) => handleChange("useCLAHE", e.target.checked)}
                />
                <label htmlFor="useCLAHE">
                  <strong>CLAHE Preprocessing</strong>
                </label>
              </div>
            </div>
            <p className="section-description">
              Contrast Limited Adaptive Histogram Equalization improves
              detection in images with uneven lighting or low contrast.
            </p>

            {localSettings.useCLAHE && (
              <div className="settings-grid">
                <div className="setting-row">
                  <label htmlFor="claheClipLimit">Clip Limit</label>
                  <input
                    type="number"
                    id="claheClipLimit"
                    min={1}
                    max={10}
                    step={0.5}
                    value={localSettings.claheClipLimit}
                    onChange={(e) =>
                      handleChange(
                        "claheClipLimit",
                        parseFloat(e.target.value) || 2,
                      )
                    }
                  />
                  <span className="setting-hint">
                    1-10, higher = more contrast
                  </span>
                </div>

                <div className="setting-row">
                  <label htmlFor="claheTileSize">Tile Size</label>
                  <input
                    type="number"
                    id="claheTileSize"
                    min={2}
                    max={16}
                    value={localSettings.claheTileSize}
                    onChange={(e) =>
                      handleChange(
                        "claheTileSize",
                        parseInt(e.target.value) || 8,
                      )
                    }
                  />
                  <span className="setting-hint">
                    2-16, tiles per dimension
                  </span>
                </div>
              </div>
            )}
          </section>

          {/* SAHI Section (disabled for now) */}
          <section className="settings-section disabled-section">
            <div className="section-header">
              <div className="checkbox-row">
                <input
                  type="checkbox"
                  id="useSAHI"
                  checked={localSettings.useSAHI}
                  onChange={(e) => handleChange("useSAHI", e.target.checked)}
                  disabled
                />
                <label htmlFor="useSAHI">
                  <strong>SAHI Tiling</strong>{" "}
                  <span className="coming-soon">(Coming Soon)</span>
                </label>
              </div>
            </div>
            <p className="section-description">
              Sliced Aided Hyper Inference processes large images in overlapping
              tiles for better detection of small objects.
            </p>
          </section>

          {/* Soft-NMS Section (disabled for now) */}
          <section className="settings-section disabled-section">
            <div className="section-header">
              <div className="checkbox-row">
                <input
                  type="checkbox"
                  id="useSoftNMS"
                  checked={localSettings.useSoftNMS}
                  onChange={(e) => handleChange("useSoftNMS", e.target.checked)}
                  disabled
                />
                <label htmlFor="useSoftNMS">
                  <strong>Soft-NMS (Non-Maximum Suppression)</strong>{" "}
                  <span className="coming-soon">(Coming Soon)</span>
                </label>
              </div>
            </div>
            <p className="section-description">
              Soft-NMS reduces confidence of overlapping detections instead of
              removing them, helping detect closely packed follicles.
            </p>
          </section>
        </div>

        <div className="dialog-footer">
          <button className="button-secondary" onClick={handleReset}>
            <RotateCcw size={16} />
            Reset to Defaults
          </button>
          <div className="footer-right">
            <button className="button-secondary" onClick={onCancel}>
              Cancel
            </button>
            <button className="button-primary" onClick={handleSave}>
              Save Settings
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
