import { useState } from 'react';
import { X, AlertTriangle, Copy, Check, RefreshCw } from 'lucide-react';
import './ServerErrorDialog.css';

interface ServerErrorDialogProps {
  error: string;
  errorDetails?: string;
  onRetry: () => void;
  onClose: () => void;
}

export function ServerErrorDialog({
  error,
  errorDetails,
  onRetry,
  onClose,
}: ServerErrorDialogProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    const textToCopy = errorDetails || error;
    try {
      await navigator.clipboard.writeText(textToCopy);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy to clipboard:', err);
    }
  };

  return (
    <div className="server-error-overlay" onClick={onClose}>
      <div className="server-error-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="dialog-header">
          <div className="header-title">
            <AlertTriangle size={20} className="error-icon" />
            <h2>Backend Server Error</h2>
          </div>
          <button className="close-button" onClick={onClose} title="Close">
            <X size={18} />
          </button>
        </div>

        <div className="dialog-content">
          <div className="error-summary">
            <span>{error}</span>
          </div>

          {errorDetails && (
            <div className="error-details-section">
              <div className="details-header">
                <span className="details-label">Error Details</span>
                <button
                  className="copy-button"
                  onClick={handleCopy}
                  title="Copy error details to clipboard"
                >
                  {copied ? (
                    <>
                      <Check size={14} />
                      <span>Copied!</span>
                    </>
                  ) : (
                    <>
                      <Copy size={14} />
                      <span>Copy to Clipboard</span>
                    </>
                  )}
                </button>
              </div>
              <pre className="error-output">{errorDetails}</pre>
            </div>
          )}

          <div className="error-help">
            <p>
              The Python backend server failed to start. This may be caused by:
            </p>
            <ul>
              <li>Python not installed or not in PATH</li>
              <li>Missing Python dependencies</li>
              <li>Port 5555 already in use</li>
              <li>Syntax error in server code</li>
            </ul>
            <p>
              Copy the error details above and include them when reporting issues.
            </p>
          </div>
        </div>

        <div className="dialog-footer">
          <button className="button-secondary" onClick={onClose}>
            Close
          </button>
          <button className="button-primary" onClick={onRetry}>
            <RefreshCw size={16} />
            Retry
          </button>
        </div>
      </div>
    </div>
  );
}
