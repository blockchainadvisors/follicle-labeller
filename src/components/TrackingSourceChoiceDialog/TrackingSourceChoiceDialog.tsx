import { useEffect, useMemo, useRef, useState } from 'react';
import { AlertCircle, FileVideo, Video, X } from 'lucide-react';
import {
  useCameraDevices,
  useCameraPreview,
} from '../../hooks/useCameraDevices';
import { useSettingsStore } from '../../store/settingsStore';
import './TrackingSourceChoiceDialog.css';

interface Props {
  onPickCamera: (deviceId: string, deviceLabel: string) => void;
  onPickFile: () => void;
  onClose: () => void;
}

type Step = 'choice' | 'pick-camera';

/**
 * Modal that gates the optical-flow tracking flow: the user picks
 * between a live camera feed or a video file. If live camera is chosen
 * but no camera is configured in Inference Settings (or the configured
 * one is no longer connected), a secondary step shows a device picker
 * with the same `useCameraDevices` / `useCameraPreview` hooks used by
 * `DetectionSettingsDialog`. The pick is persisted to settingsStore so
 * the next invocation skips straight to tracking.
 */
export function TrackingSourceChoiceDialog({
  onPickCamera,
  onPickFile,
  onClose,
}: Props) {
  const settings = useSettingsStore(s => s.globalDetectionSettings);
  const setGlobalDetectionSettings = useSettingsStore(s => s.setGlobalDetectionSettings);

  const {
    devices: cameraDevices,
    permission: cameraPermission,
    refresh: refreshCameras,
    requestAccess: requestCameraAccess,
  } = useCameraDevices(true);

  const savedDeviceMatchesConnected = useMemo(
    () =>
      !!settings.liveCameraDeviceId &&
      cameraDevices.some(d => d.deviceId === settings.liveCameraDeviceId),
    [cameraDevices, settings.liveCameraDeviceId],
  );

  const [step, setStep] = useState<Step>('choice');
  const [pendingDeviceId, setPendingDeviceId] = useState<string | null>(
    settings.liveCameraDeviceId,
  );

  // Keep pendingDeviceId synced with the saved one until the user changes it
  // (so the dropdown preselects the saved camera once enumeration settles).
  useEffect(() => {
    if (step !== 'pick-camera') return;
    if (pendingDeviceId) return;
    if (savedDeviceMatchesConnected && settings.liveCameraDeviceId) {
      setPendingDeviceId(settings.liveCameraDeviceId);
      return;
    }
    const firstWithLabel = cameraDevices.find(d => d.label) ?? cameraDevices[0];
    if (firstWithLabel) setPendingDeviceId(firstWithLabel.deviceId);
  }, [
    step,
    pendingDeviceId,
    savedDeviceMatchesConnected,
    settings.liveCameraDeviceId,
    cameraDevices,
  ]);

  const cameraPreviewEnabled =
    step === 'pick-camera' &&
    !!pendingDeviceId &&
    cameraDevices.some(d => d.deviceId === pendingDeviceId);
  const {
    stream: cameraStream,
    error: cameraPreviewError,
    loading: cameraPreviewLoading,
  } = useCameraPreview(pendingDeviceId, cameraPreviewEnabled);

  const previewRef = useRef<HTMLVideoElement>(null);
  useEffect(() => {
    if (previewRef.current) {
      previewRef.current.srcObject = cameraStream;
    }
  }, [cameraStream]);

  const resolveCameraPick = (deviceId: string) => {
    const dev = cameraDevices.find(d => d.deviceId === deviceId);
    const label = dev?.label || settings.liveCameraLabel || 'Live camera';
    // Persist so the next invocation skips the picker entirely.
    setGlobalDetectionSettings({
      ...settings,
      liveCameraDeviceId: deviceId,
      liveCameraLabel: dev?.label || null,
    });
    onPickCamera(deviceId, label);
  };

  const handlePickCameraCard = () => {
    if (savedDeviceMatchesConnected && settings.liveCameraDeviceId) {
      resolveCameraPick(settings.liveCameraDeviceId);
      return;
    }
    setStep('pick-camera');
  };

  const handleContinueWithPending = () => {
    if (!pendingDeviceId) return;
    resolveCameraPick(pendingDeviceId);
  };

  const canContinue =
    !!pendingDeviceId &&
    cameraDevices.some(d => d.deviceId === pendingDeviceId);

  return (
    <div
      className="tracking-source-choice-overlay"
      onMouseDown={e => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="tracking-source-choice-dialog" role="dialog" aria-modal="true">
        <div className="dialog-header">
          <h2>
            {step === 'choice' ? 'Track Follicle' : 'Select a Camera'}
          </h2>
          <button className="close-button" onClick={onClose} aria-label="Close">
            <X size={18} />
          </button>
        </div>

        {step === 'choice' ? (
          <div className="dialog-content tracking-source-choice-grid">
            <button
              type="button"
              className="tracking-source-card"
              onClick={handlePickCameraCard}
            >
              <Video size={32} className="tracking-source-card-icon" />
              <div className="tracking-source-card-title">Use live camera</div>
              <div className="tracking-source-card-subtitle">
                {settings.liveCameraLabel
                  ? savedDeviceMatchesConnected
                    ? settings.liveCameraLabel
                    : `${settings.liveCameraLabel} (not connected)`
                  : "You'll pick a camera next"}
              </div>
            </button>

            <button
              type="button"
              className="tracking-source-card"
              onClick={onPickFile}
            >
              <FileVideo size={32} className="tracking-source-card-icon" />
              <div className="tracking-source-card-title">From file</div>
              <div className="tracking-source-card-subtitle">
                Choose a video from your computer
              </div>
            </button>
          </div>
        ) : (
          <div className="dialog-content">
            <div className="settings-section">
              <div className="settings-row">
                <label htmlFor="tracking-camera-select">Camera</label>
                <select
                  id="tracking-camera-select"
                  value={pendingDeviceId ?? ''}
                  disabled={
                    cameraPermission !== 'granted' || cameraDevices.length === 0
                  }
                  onChange={e => setPendingDeviceId(e.target.value || null)}
                >
                  <option value="" disabled>
                    {cameraPermission === 'granted'
                      ? cameraDevices.length
                        ? 'Select a camera…'
                        : 'No cameras detected'
                      : cameraPermission === 'denied'
                      ? 'Camera access denied'
                      : cameraPermission === 'unavailable'
                      ? 'Camera API unavailable'
                      : 'Requesting camera access…'}
                  </option>
                  {cameraDevices.map((d, i) => (
                    <option key={d.deviceId || `cam-${i}`} value={d.deviceId}>
                      {d.label || `Camera ${i + 1}`}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  className="camera-refresh-btn"
                  onClick={() => void refreshCameras()}
                  title="Refresh camera list"
                >
                  Refresh
                </button>
              </div>

              {cameraPermission === 'denied' && (
                <div className="settings-row camera-hint-row">
                  <AlertCircle size={14} />
                  <span>
                    Camera access denied. Enable it in System Settings,
                    then click Retry.
                  </span>
                  <button
                    type="button"
                    className="camera-refresh-btn"
                    onClick={() => void requestCameraAccess()}
                  >
                    Retry
                  </button>
                </div>
              )}

              {settings.liveCameraDeviceId &&
                !savedDeviceMatchesConnected &&
                cameraPermission === 'granted' && (
                  <div className="settings-row camera-warning-row">
                    <AlertCircle size={14} />
                    <span>
                      Previously selected camera "
                      {settings.liveCameraLabel ?? 'unknown'}" is not
                      connected. Pick another below.
                    </span>
                  </div>
                )}

              {cameraPreviewEnabled && (
                <div className="settings-row camera-preview-row">
                  {cameraPreviewLoading && (
                    <span className="setting-hint">Opening camera…</span>
                  )}
                  {cameraPreviewError && (
                    <span className="setting-hint camera-preview-error">
                      {cameraPreviewError}
                    </span>
                  )}
                  <video
                    ref={previewRef}
                    autoPlay
                    muted
                    playsInline
                    className="camera-preview"
                  />
                </div>
              )}
            </div>

            <div className="dialog-footer">
              <button
                type="button"
                className="button-secondary"
                onClick={() => setStep('choice')}
              >
                Back
              </button>
              <button
                type="button"
                className="button-primary"
                disabled={!canContinue}
                onClick={handleContinueWithPending}
              >
                Continue
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
