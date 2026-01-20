import React, { useState, useRef, useEffect } from 'react';
import { Sun, Moon, Palette } from 'lucide-react';
import {
  useThemeStore,
  accentColorOptions,
  darkBackgroundOptions,
  lightBackgroundOptions,
  AccentColor,
  BackgroundTheme
} from '../../store/themeStore';

export const ThemePicker: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const accent = useThemeStore(state => state.accent);
  const background = useThemeStore(state => state.background);
  const setAccent = useThemeStore(state => state.setAccent);
  const setBackground = useThemeStore(state => state.setBackground);

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
          {/* Dark background themes */}
          <div className="theme-section">
            <span className="theme-section-label">
              <Moon size={12} /> Dark Themes
            </span>
            <div className="background-color-grid">
              {darkBackgroundOptions.map((option) => (
                <button
                  key={option.value}
                  className={`background-color-btn ${background === option.value ? 'active' : ''}`}
                  style={{ backgroundColor: option.color }}
                  onClick={() => setBackground(option.value as BackgroundTheme)}
                  title={option.label}
                  aria-label={`${option.label} background theme`}
                  aria-pressed={background === option.value}
                />
              ))}
            </div>
          </div>

          {/* Light background themes */}
          <div className="theme-section">
            <span className="theme-section-label">
              <Sun size={12} /> Light Themes
            </span>
            <div className="background-color-grid">
              {lightBackgroundOptions.map((option) => (
                <button
                  key={option.value}
                  className={`background-color-btn light ${background === option.value ? 'active' : ''}`}
                  style={{ backgroundColor: option.color }}
                  onClick={() => setBackground(option.value as BackgroundTheme)}
                  title={option.label}
                  aria-label={`${option.label} background theme`}
                  aria-pressed={background === option.value}
                />
              ))}
            </div>
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
