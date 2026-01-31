import React from "react";
import { Loader2 } from "lucide-react";
import { useLoadingStore } from "../../store/loadingStore";
import "./LoadingOverlay.css";

export const LoadingOverlay: React.FC = () => {
  const isLoading = useLoadingStore((state) => state.isLoading);
  const message = useLoadingStore((state) => state.message);
  const canCancel = useLoadingStore((state) => state.canCancel);
  const cancelLoading = useLoadingStore((state) => state.cancelLoading);
  const isPythonInitializing = useLoadingStore((state) => state.isPythonInitializing);
  const pythonStatus = useLoadingStore((state) => state.pythonStatus);

  // Priority: Python initialization takes precedence over regular loading
  if (isPythonInitializing) {
    return (
      <div className="loading-overlay">
        <div className="loading-dialog startup-dialog">
          <Loader2 size={48} className="loading-spinner" />
          <h2 className="startup-title">Starting up...</h2>
          <span className="startup-status">{pythonStatus}</span>
        </div>
      </div>
    );
  }

  if (!isLoading) {
    return null;
  }

  return (
    <div className="loading-overlay">
      <div className="loading-dialog">
        <Loader2 size={32} className="loading-spinner" />
        <span className="loading-message">{message}</span>
        {canCancel && (
          <button className="loading-cancel-btn" onClick={cancelLoading}>
            Cancel
          </button>
        )}
      </div>
    </div>
  );
};
