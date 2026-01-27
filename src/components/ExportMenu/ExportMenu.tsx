import { useState, useRef, useEffect } from 'react';
import { Download, FileText, Table, Image, ChevronDown, Target } from 'lucide-react';
import './ExportMenu.css';

export type ExportType = 'images' | 'yolo' | 'yolo-keypoint' | 'csv';

interface ExportMenuProps {
  onExport: (type: ExportType) => void;
  disabled?: boolean;
  hasSelection?: boolean;
}

export function ExportMenu({ onExport, disabled, hasSelection }: ExportMenuProps) {
  const [isOpen, setIsOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  // Close menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen]);

  const handleExport = (type: ExportType) => {
    onExport(type);
    setIsOpen(false);
  };

  return (
    <div className="export-menu" ref={menuRef}>
      <button
        className={`export-button ${isOpen ? 'active' : ''}`}
        onClick={() => setIsOpen(!isOpen)}
        disabled={disabled}
        title="Export Options"
        aria-label="Export Options"
        aria-expanded={isOpen}
      >
        <Download size={18} />
        <ChevronDown size={12} className="chevron" />
      </button>

      {isOpen && (
        <div className="export-dropdown">
          <button
            className="export-option"
            onClick={() => handleExport('images')}
          >
            <Image size={16} />
            <div className="option-text">
              <span className="option-label">Follicle Images</span>
              <span className="option-description">
                {hasSelection ? 'Selected or all images as ZIP' : 'All follicle images as ZIP'}
              </span>
            </div>
          </button>

          <button
            className="export-option"
            onClick={() => handleExport('yolo')}
          >
            <FileText size={16} />
            <div className="option-text">
              <span className="option-label">YOLO Dataset</span>
              <span className="option-description">Images + labels for training</span>
            </div>
          </button>

          <button
            className="export-option"
            onClick={() => handleExport('yolo-keypoint')}
          >
            <Target size={16} />
            <div className="option-text">
              <span className="option-label">YOLO Keypoint Dataset</span>
              <span className="option-description">Origin + direction for pose training</span>
            </div>
          </button>

          <button
            className="export-option"
            onClick={() => handleExport('csv')}
          >
            <Table size={16} />
            <div className="option-text">
              <span className="option-label">CSV Export</span>
              <span className="option-description">Spreadsheet for analysis</span>
            </div>
          </button>
        </div>
      )}
    </div>
  );
}
