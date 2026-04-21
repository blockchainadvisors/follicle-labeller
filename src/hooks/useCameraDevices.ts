import { useCallback, useEffect, useRef, useState } from 'react';

export type CameraPermission =
  | 'unknown'
  | 'prompt'
  | 'granted'
  | 'denied'
  | 'unavailable';

export interface UseCameraDevicesResult {
  devices: MediaDeviceInfo[];
  permission: CameraPermission;
  refresh: () => Promise<void>;
  requestAccess: () => Promise<void>;
}

export interface UseCameraPreviewResult {
  stream: MediaStream | null;
  error: string | null;
  loading: boolean;
}

function mediaDevicesAvailable(): boolean {
  return typeof navigator !== 'undefined' && !!navigator.mediaDevices;
}

async function enumerateVideoInputs(): Promise<MediaDeviceInfo[]> {
  const all = await navigator.mediaDevices.enumerateDevices();
  return all.filter((d) => d.kind === 'videoinput');
}

function derivePermission(devices: MediaDeviceInfo[]): CameraPermission {
  if (devices.length === 0) return 'prompt';
  return devices.some((d) => d.label) ? 'granted' : 'prompt';
}

export function useCameraDevices(autoRequest: boolean): UseCameraDevicesResult {
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [permission, setPermission] = useState<CameraPermission>(
    mediaDevicesAvailable() ? 'unknown' : 'unavailable'
  );

  const permissionRef = useRef(permission);
  permissionRef.current = permission;

  const refresh = useCallback(async () => {
    if (!mediaDevicesAvailable()) {
      setPermission('unavailable');
      setDevices([]);
      return;
    }
    try {
      const list = await enumerateVideoInputs();
      setDevices(list);
      // Only downgrade permission from 'granted' if labels disappear.
      // Never upgrade from 'denied' silently — the user must retry.
      if (permissionRef.current !== 'denied') {
        setPermission(derivePermission(list));
      }
    } catch {
      setDevices([]);
    }
  }, []);

  const requestAccess = useCallback(async () => {
    if (!mediaDevicesAvailable()) {
      setPermission('unavailable');
      return;
    }
    try {
      const probe = await navigator.mediaDevices.getUserMedia({ video: true });
      probe.getTracks().forEach((t) => t.stop());
      setPermission('granted');
      await refresh();
    } catch (err) {
      const name = (err as DOMException)?.name ?? '';
      if (name === 'NotFoundError') {
        // No cameras present; permission state is indeterminate — leave at prompt.
        setPermission('prompt');
        setDevices([]);
      } else {
        setPermission('denied');
      }
    }
  }, [refresh]);

  // Initial enumeration + optional auto-request
  useEffect(() => {
    if (!mediaDevicesAvailable()) return;
    let cancelled = false;
    (async () => {
      try {
        const list = await enumerateVideoInputs();
        if (cancelled) return;
        setDevices(list);
        const initial = derivePermission(list);
        setPermission(initial);
        if (autoRequest && initial === 'prompt') {
          await requestAccess();
        }
      } catch {
        if (!cancelled) setDevices([]);
      }
    })();
    return () => {
      cancelled = true;
    };
    // Run once on mount. autoRequest and requestAccess are captured intentionally.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // devicechange listener for hot-plug
  useEffect(() => {
    if (!mediaDevicesAvailable()) return;
    const handler = () => {
      void refresh();
    };
    navigator.mediaDevices.addEventListener('devicechange', handler);
    return () => {
      navigator.mediaDevices.removeEventListener('devicechange', handler);
    };
  }, [refresh]);

  return { devices, permission, refresh, requestAccess };
}

export function useCameraPreview(
  deviceId: string | null,
  enabled: boolean
): UseCameraPreviewResult {
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!enabled || !deviceId || !mediaDevicesAvailable()) {
      setStream(null);
      setError(null);
      setLoading(false);
      return;
    }

    let cancelled = false;
    let activeStream: MediaStream | null = null;
    setLoading(true);
    setError(null);

    navigator.mediaDevices
      .getUserMedia({ video: { deviceId: { exact: deviceId } } })
      .then((s) => {
        if (cancelled) {
          s.getTracks().forEach((t) => t.stop());
          return;
        }
        activeStream = s;
        setStream(s);
        setLoading(false);
      })
      .catch((err: DOMException) => {
        if (cancelled) return;
        const name = err?.name ?? '';
        let message = 'Could not open camera.';
        if (name === 'NotReadableError')
          message = 'Camera is in use by another application.';
        else if (name === 'OverconstrainedError')
          message = 'Selected camera is no longer available.';
        else if (name === 'NotAllowedError')
          message = 'Camera permission was denied.';
        else if (name === 'NotFoundError') message = 'Camera not found.';
        setError(message);
        setStream(null);
        setLoading(false);
      });

    return () => {
      cancelled = true;
      if (activeStream) {
        activeStream.getTracks().forEach((t) => t.stop());
      }
    };
  }, [deviceId, enabled]);

  return { stream, error, loading };
}
