import { useState } from 'react';
import { X } from 'lucide-react';
import type { BlobDetectionOptions } from '../../types';
import './DetectionSettingsDialog.css';

export interface DetectionSettings {
  // Basic size parameters
  minWidth: number;
  maxWidth: number;
  minHeight: number;
  maxHeight: number;
  darkBlobs: boolean;

  // CLAHE preprocessing
  useCLAHE: boolean;
  claheClipLimit: number;
  claheTileSize: number;

  // SAHI-style tiling
  useSAHI: boolean;
  tileSize: number;
  tileOverlap: number;

  // Soft-NMS
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
  useCLAHE: false,
  claheClipLimit: 2.0,
  claheTileSize: 8,
  useSAHI: false,
  tileSize: 512,
  tileOverlap: 0.2,
  useSoftNMS: true,
  softNMSSigma: 0.5,
  softNMSThreshold: 0.1,
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
  const [localSettings, setLocalSettings] = useState<DetectionSettings>({ ...settings });

  const handleChange = <K extends keyof DetectionSettings>(
    key: K,
    value: DetectionSettings[K]
  ) => {
    setLocalSettings(prev => ({ ...prev, [key]: value }));
  };

  const handleReset = () => {
    setLocalSettings({ ...DEFAULT_DETECTION_SETTINGS });
  };

  const handleSave = () => {
    onSave(localSettings);
  };

  return (
    <div className="detection-settings-overlay" onClick={onCancel}>
      <div className="detection-settings-dialog" onClick={e => e.stopPropagation()}>
        <div className="dialog-header">
          <h2>Detection Settings</h2>
          <button className="close-button" onClick={onCancel}>
            <X size={18} />
          </button>
        </div>

        <div className="dialog-content">
          {/* Basic Size Parameters */}
          <section className="settings-section">
            <h3>Size Range</h3>
            <div className="settings-row">
              <label>Min Width</label>
              <input
                type="number"
                value={localSettings.minWidth}
                onChange={e => handleChange('minWidth', parseInt(e.target.value) || 0)}
                min={1}
                max={1000}
              />
            </div>
            <div className="settings-row">
              <label>Max Width</label>
              <input
                type="number"
                value={localSettings.maxWidth}
                onChange={e => handleChange('maxWidth', parseInt(e.target.value) || 0)}
                min={1}
                max={5000}
              />
            </div>
            <div className="settings-row">
              <label>Min Height</label>
              <input
                type="number"
                value={localSettings.minHeight}
                onChange={e => handleChange('minHeight', parseInt(e.target.value) || 0)}
                min={1}
                max={1000}
              />
            </div>
            <div className="settings-row">
              <label>Max Height</label>
              <input
                type="number"
                value={localSettings.maxHeight}
                onChange={e => handleChange('maxHeight', parseInt(e.target.value) || 0)}
                min={1}
                max={5000}
              />
            </div>
            <div className="settings-row checkbox">
              <label>
                <input
                  type="checkbox"
                  checked={localSettings.darkBlobs}
                  onChange={e => handleChange('darkBlobs', e.target.checked)}
                />
                Dark Blobs (detect dark regions on light background)
              </label>
            </div>
          </section>

          {/* CLAHE Preprocessing */}
          <section className="settings-section">
            <h3>
              <label className="section-toggle">
                <input
                  type="checkbox"
                  checked={localSettings.useCLAHE}
                  onChange={e => handleChange('useCLAHE', e.target.checked)}
                />
                CLAHE Preprocessing
              </label>
            </h3>
            <p className="section-description">
              Contrast Limited Adaptive Histogram Equalization improves detection in
              images with uneven lighting or low contrast.
            </p>
            {localSettings.useCLAHE && (
              <>
                <div className="settings-row">
                  <label>Clip Limit</label>
                  <input
                    type="number"
                    value={localSettings.claheClipLimit}
                    onChange={e => handleChange('claheClipLimit', parseFloat(e.target.value) || 1)}
                    min={1}
                    max={10}
                    step={0.5}
                  />
                  <span className="hint">1-10, higher = more contrast</span>
                </div>
                <div className="settings-row">
                  <label>Tile Size</label>
                  <input
                    type="number"
                    value={localSettings.claheTileSize}
                    onChange={e => handleChange('claheTileSize', parseInt(e.target.value) || 4)}
                    min={2}
                    max={16}
                  />
                  <span className="hint">2-16, tiles per dimension</span>
                </div>
              </>
            )}
          </section>

          {/* SAHI-style Tiling */}
          <section className="settings-section">
            <h3>
              <label className="section-toggle">
                <input
                  type="checkbox"
                  checked={localSettings.useSAHI}
                  onChange={e => handleChange('useSAHI', e.target.checked)}
                />
                SAHI Tiling
              </label>
            </h3>
            <p className="section-description">
              Sliced Aided Hyper Inference processes large images in overlapping tiles
              for better detection of small objects.
            </p>
            {localSettings.useSAHI && (
              <>
                <div className="settings-row">
                  <label>Tile Size (px)</label>
                  <input
                    type="number"
                    value={localSettings.tileSize}
                    onChange={e => handleChange('tileSize', parseInt(e.target.value) || 256)}
                    min={128}
                    max={2048}
                    step={64}
                  />
                  <span className="hint">128-2048px</span>
                </div>
                <div className="settings-row">
                  <label>Overlap</label>
                  <input
                    type="number"
                    value={localSettings.tileOverlap}
                    onChange={e => handleChange('tileOverlap', parseFloat(e.target.value) || 0.1)}
                    min={0}
                    max={0.5}
                    step={0.05}
                  />
                  <span className="hint">0-0.5, fraction of tile size</span>
                </div>
              </>
            )}
          </section>

          {/* Soft-NMS */}
          <section className="settings-section">
            <h3>
              <label className="section-toggle">
                <input
                  type="checkbox"
                  checked={localSettings.useSoftNMS}
                  onChange={e => handleChange('useSoftNMS', e.target.checked)}
                />
                Soft-NMS (Non-Maximum Suppression)
              </label>
            </h3>
            <p className="section-description">
              Reduces overlapping detections using soft suppression with Gaussian decay.
            </p>
            {localSettings.useSoftNMS && (
              <>
                <div className="settings-row">
                  <label>Sigma</label>
                  <input
                    type="number"
                    value={localSettings.softNMSSigma}
                    onChange={e => handleChange('softNMSSigma', parseFloat(e.target.value) || 0.3)}
                    min={0.1}
                    max={2}
                    step={0.1}
                  />
                  <span className="hint">0.1-2, Gaussian decay rate</span>
                </div>
                <div className="settings-row">
                  <label>Threshold</label>
                  <input
                    type="number"
                    value={localSettings.softNMSThreshold}
                    onChange={e => handleChange('softNMSThreshold', parseFloat(e.target.value) || 0.05)}
                    min={0.01}
                    max={0.5}
                    step={0.01}
                  />
                  <span className="hint">Min confidence to keep</span>
                </div>
              </>
            )}
          </section>
        </div>

        <div className="dialog-footer">
          <button className="button-secondary" onClick={handleReset}>
            Reset to Defaults
          </button>
          <div className="footer-actions">
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

/**
 * Convert DetectionSettings to BlobDetectionOptions
 */
export function settingsToOptions(settings: DetectionSettings): Partial<BlobDetectionOptions> {
  return {
    minWidth: settings.minWidth,
    maxWidth: settings.maxWidth,
    minHeight: settings.minHeight,
    maxHeight: settings.maxHeight,
    darkBlobs: settings.darkBlobs,
    useCLAHE: settings.useCLAHE,
    claheClipLimit: settings.claheClipLimit,
    claheTileSize: settings.claheTileSize,
    tileSize: settings.useSAHI ? settings.tileSize : 0,
    tileOverlap: settings.tileOverlap,
    useSoftNMS: settings.useSoftNMS,
    softNMSSigma: settings.softNMSSigma,
    softNMSThreshold: settings.softNMSThreshold,
    useGPU: true,
    workerCount: navigator.hardwareConcurrency || 4,
  };
}
