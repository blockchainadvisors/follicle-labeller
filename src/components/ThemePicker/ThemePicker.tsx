import React, { useState, useRef, useEffect } from 'react';
import { Sun, Moon, Palette } from 'lucide-react';
import { useThemeStore, accentColorOptions, AccentColor } from '../../store/themeStore';

export const ThemePicker: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const mode = useThemeStore(state => state.mode);
  const accent = useThemeStore(state => state.accent);
  const toggleMode = useThemeStore(state => state.toggleMode);
  const setAccent = useThemeStore(state => state.setAccent);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Close on Escape key
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener('keydown', handleKeyDown);
      return () => document.removeEventListener('keydown', handleKeyDown);
    }
  }, [isOpen]);

  return (
    <div className="theme-picker" ref={dropdownRef}>
      <button
        className="icon-button theme-toggle-btn"
        onClick={() => setIsOpen(!isOpen)}
        title="Theme Settings"
        aria-label="Theme Settings"
        aria-expanded={isOpen}
      >
        <Palette size={18} />
      </button>

      {isOpen && (
        <div className="theme-dropdown">
          {/* Mode toggle */}
          <div className="theme-section">
            <span className="theme-section-label">Mode</span>
            <button
              className="theme-mode-toggle"
              onClick={toggleMode}
              aria-label={`Switch to ${mode === 'dark' ? 'light' : 'dark'} mode`}
            >
              {mode === 'dark' ? (
                <>
                  <Moon size={16} />
                  <span>Dark</span>
                </>
              ) : (
                <>
                  <Sun size={16} />
                  <span>Light</span>
                </>
              )}
            </button>
          </div>

          {/* Accent color picker */}
          <div className="theme-section">
            <span className="theme-section-label">Accent Color</span>
            <div className="accent-color-grid">
              {accentColorOptions.map((option) => (
                <button
                  key={option.value}
                  className={`accent-color-btn ${accent === option.value ? 'active' : ''}`}
                  style={{ backgroundColor: option.color }}
                  onClick={() => setAccent(option.value as AccentColor)}
                  title={option.label}
                  aria-label={`${option.label} accent color`}
                  aria-pressed={accent === option.value}
                />
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
