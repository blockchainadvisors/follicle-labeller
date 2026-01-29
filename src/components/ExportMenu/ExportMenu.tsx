import { useState, useRef, useEffect } from 'react';
import { Download, FileText, Table, Image, ChevronDown, Target, FileJson, ChevronRight, Database } from 'lucide-react';
import './ExportMenu.css';

export type ExportType = 'images' | 'yolo' | 'yolo-keypoint' | 'csv' | 'selected-json' | 'coco-json';

interface ExportMenuProps {
  onExport: (type: ExportType) => void;
  disabled?: boolean;
  hasSelection?: boolean;
}

interface Submenu {
  label: string;
  icon: React.ReactNode;
  items: {
    type: ExportType;
    label: string;
    description: string;
    icon: React.ReactNode;
    disabled?: boolean;
    disabledReason?: string;
  }[];
}

export function ExportMenu({ onExport, disabled, hasSelection }: ExportMenuProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [activeSubmenu, setActiveSubmenu] = useState<string | null>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  // Close menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsOpen(false);
        setActiveSubmenu(null);
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
    setActiveSubmenu(null);
  };

  // Define submenus
  const submenus: Submenu[] = [
    {
      label: 'Training Data',
      icon: <Database size={16} />,
      items: [
        {
          type: 'yolo',
          label: 'YOLO Detection Dataset',
          description: 'Images + bounding box labels for training',
          icon: <FileText size={16} />,
        },
        {
          type: 'yolo-keypoint',
          label: 'YOLO Keypoint Dataset',
          description: 'Origin + direction for pose training',
          icon: <Target size={16} />,
        },
        {
          type: 'coco-json',
          label: 'COCO JSON',
          description: 'Standard format with optional keypoints',
          icon: <FileJson size={16} />,
        },
      ],
    },
    {
      label: 'Analysis',
      icon: <Table size={16} />,
      items: [
        {
          type: 'csv',
          label: 'CSV Export',
          description: 'Spreadsheet for data analysis',
          icon: <Table size={16} />,
        },
        {
          type: 'images',
          label: 'Follicle Crops',
          description: hasSelection ? 'Selected or all images as ZIP' : 'All follicle images as ZIP',
          icon: <Image size={16} />,
        },
      ],
    },
  ];

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
          {/* Submenus */}
          {submenus.map((submenu) => (
            <div
              key={submenu.label}
              className="export-submenu-container"
              onMouseEnter={() => setActiveSubmenu(submenu.label)}
              onMouseLeave={() => setActiveSubmenu(null)}
            >
              <button className="export-option submenu-trigger">
                {submenu.icon}
                <div className="option-text">
                  <span className="option-label">{submenu.label}</span>
                </div>
                <ChevronRight size={14} className="submenu-arrow" />
              </button>

              {activeSubmenu === submenu.label && (
                <div className="export-submenu">
                  {submenu.items.map((item) => (
                    <button
                      key={item.type}
                      className={`export-option ${item.disabled ? 'disabled' : ''}`}
                      onClick={() => !item.disabled && handleExport(item.type)}
                      disabled={item.disabled}
                    >
                      {item.icon}
                      <div className="option-text">
                        <span className="option-label">{item.label}</span>
                        <span className="option-description">
                          {item.disabled && item.disabledReason
                            ? item.disabledReason
                            : item.description}
                        </span>
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>
          ))}

          {/* Divider */}
          <div className="export-divider" />

          {/* Selected Annotations (always at bottom) */}
          <button
            className={`export-option ${!hasSelection ? 'disabled' : ''}`}
            onClick={() => hasSelection && handleExport('selected-json')}
            disabled={!hasSelection}
          >
            <FileJson size={16} />
            <div className="option-text">
              <span className="option-label">Selected Annotations</span>
              <span className="option-description">
                {hasSelection ? 'Export selected as JSON' : 'Select annotations first'}
              </span>
            </div>
          </button>
        </div>
      )}
    </div>
  );
}
