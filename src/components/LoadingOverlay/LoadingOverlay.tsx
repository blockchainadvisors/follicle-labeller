import React from "react";
import { Loader2 } from "lucide-react";
import { useLoadingStore } from "../../store/loadingStore";
import "./LoadingOverlay.css";

export const LoadingOverlay: React.FC = () => {
  const isLoading = useLoadingStore((state) => state.isLoading);
  const message = useLoadingStore((state) => state.message);
  const canCancel = useLoadingStore((state) => state.canCancel);
  const cancelLoading = useLoadingStore((state) => state.cancelLoading);

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
