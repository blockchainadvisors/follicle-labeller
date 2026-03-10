import { useState, useRef, useEffect } from 'react';
import { Square, Circle, Minus, ChevronDown } from 'lucide-react';
import { useCanvasStore } from '../../store/canvasStore';
import { ShapeType } from '../../types';
import './ShapeToolDropdown.css';

interface ShapeOption {
  type: ShapeType;
  label: string;
  shortcut: string;
  icon: React.ReactNode;
}

const shapeOptions: ShapeOption[] = [
  { type: 'rectangle', label: 'Rectangle', shortcut: '1', icon: <Square size={16} /> },
  { type: 'circle', label: 'Circle', shortcut: '2', icon: <Circle size={16} /> },
  { type: 'linear', label: 'Linear', shortcut: '3', icon: <Minus size={16} strokeWidth={3} /> },
];

export function ShapeToolDropdown() {
  const [isOpen, setIsOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  const currentShapeType = useCanvasStore((state) => state.currentShapeType);
  const setShapeType = useCanvasStore((state) => state.setShapeType);

  // Get current shape option for button display
  const currentOption = shapeOptions.find(opt => opt.type === currentShapeType) || shapeOptions[0];

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

  // Close on Escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener('keydown', handleKeyDown);
    }

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen]);

  const handleSelectShape = (type: ShapeType) => {
    setShapeType(type);
    setIsOpen(false);
  };

  return (
    <div className="shape-tool-dropdown" ref={menuRef}>
      <button
        className={`shape-tool-button ${isOpen ? 'active' : ''}`}
        onClick={() => setIsOpen(!isOpen)}
        title={`Shape: ${currentOption.label} (${currentOption.shortcut})`}
        aria-label={`Shape: ${currentOption.label}`}
        aria-expanded={isOpen}
      >
        {currentOption.icon}
        <ChevronDown size={12} className="chevron" />
      </button>

      {isOpen && (
        <div className="shape-dropdown">
          {shapeOptions.map((option) => (
            <button
              key={option.type}
              className={`shape-option ${currentShapeType === option.type ? 'active' : ''}`}
              onClick={() => handleSelectShape(option.type)}
            >
              <span className="shape-option-icon">{option.icon}</span>
              <span className="shape-option-label">{option.label}</span>
              <span className="shape-option-shortcut">{option.shortcut}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
