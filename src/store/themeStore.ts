import { create } from 'zustand';

// Theme color palettes
export interface ThemeColors {
  bgPrimary: string;
  bgSecondary: string;
  bgTertiary: string;
  textPrimary: string;
  textSecondary: string;
  accent: string;
  accentHover: string;
  danger: string;
  dangerHover: string;
  border: string;
  success: string;
}

export type ThemeMode = 'dark' | 'light';
export type AccentColor = 'teal' | 'blue' | 'purple' | 'orange' | 'pink';

// Accent color definitions
const accentColors: Record<AccentColor, { accent: string; accentHover: string }> = {
  teal: { accent: '#4ECDC4', accentHover: '#3dbdb5' },
  blue: { accent: '#3B82F6', accentHover: '#2563EB' },
  purple: { accent: '#8B5CF6', accentHover: '#7C3AED' },
  orange: { accent: '#F97316', accentHover: '#EA580C' },
  pink: { accent: '#EC4899', accentHover: '#DB2777' },
};

// Base theme palettes
const darkTheme: Omit<ThemeColors, 'accent' | 'accentHover'> = {
  bgPrimary: '#1a1a2e',
  bgSecondary: '#16213e',
  bgTertiary: '#0f3460',
  textPrimary: '#ffffff',
  textSecondary: '#a0a0a0',
  danger: '#e94560',
  dangerHover: '#d63d56',
  border: '#2a2a4a',
  success: '#4caf50',
};

const lightTheme: Omit<ThemeColors, 'accent' | 'accentHover'> = {
  bgPrimary: '#f5f5f5',
  bgSecondary: '#ffffff',
  bgTertiary: '#e8e8e8',
  textPrimary: '#1a1a2e',
  textSecondary: '#666666',
  danger: '#dc2626',
  dangerHover: '#b91c1c',
  border: '#d1d5db',
  success: '#16a34a',
};

// Get full theme colors by combining mode and accent
function getThemeColors(mode: ThemeMode, accent: AccentColor): ThemeColors {
  const baseTheme = mode === 'dark' ? darkTheme : lightTheme;
  const accentPair = accentColors[accent];
  return {
    ...baseTheme,
    ...accentPair,
  };
}

// Apply theme colors to CSS variables
function applyTheme(colors: ThemeColors): void {
  const root = document.documentElement;
  root.style.setProperty('--bg-primary', colors.bgPrimary);
  root.style.setProperty('--bg-secondary', colors.bgSecondary);
  root.style.setProperty('--bg-tertiary', colors.bgTertiary);
  root.style.setProperty('--text-primary', colors.textPrimary);
  root.style.setProperty('--text-secondary', colors.textSecondary);
  root.style.setProperty('--accent', colors.accent);
  root.style.setProperty('--accent-hover', colors.accentHover);
  root.style.setProperty('--danger', colors.danger);
  root.style.setProperty('--danger-hover', colors.dangerHover);
  root.style.setProperty('--border', colors.border);
  root.style.setProperty('--success', colors.success);

  // Also set a data attribute for any CSS that needs to know the mode
  root.setAttribute('data-theme', colors === getThemeColors('dark', 'teal') ? 'dark' : 'light');
}

// localStorage key
const THEME_STORAGE_KEY = 'follicle-labeller-theme';

// Load saved theme from localStorage
function loadSavedTheme(): { mode: ThemeMode; accent: AccentColor } {
  try {
    const saved = localStorage.getItem(THEME_STORAGE_KEY);
    if (saved) {
      const parsed = JSON.parse(saved);
      // Validate the values
      if (['dark', 'light'].includes(parsed.mode) &&
          Object.keys(accentColors).includes(parsed.accent)) {
        return parsed;
      }
    }
  } catch {
    // Ignore parse errors
  }
  return { mode: 'dark', accent: 'teal' }; // Default
}

// Save theme to localStorage
function saveTheme(mode: ThemeMode, accent: AccentColor): void {
  try {
    localStorage.setItem(THEME_STORAGE_KEY, JSON.stringify({ mode, accent }));
  } catch {
    // Ignore storage errors
  }
}

interface ThemeState {
  mode: ThemeMode;
  accent: AccentColor;
  colors: ThemeColors;

  // Actions
  setMode: (mode: ThemeMode) => void;
  setAccent: (accent: AccentColor) => void;
  toggleMode: () => void;
  initializeTheme: () => void;
}

export const useThemeStore = create<ThemeState>((set, get) => {
  const initial = loadSavedTheme();

  return {
    mode: initial.mode,
    accent: initial.accent,
    colors: getThemeColors(initial.mode, initial.accent),

    setMode: (mode) => {
      const { accent } = get();
      const colors = getThemeColors(mode, accent);
      applyTheme(colors);
      saveTheme(mode, accent);
      set({ mode, colors });
    },

    setAccent: (accent) => {
      const { mode } = get();
      const colors = getThemeColors(mode, accent);
      applyTheme(colors);
      saveTheme(mode, accent);
      set({ accent, colors });
    },

    toggleMode: () => {
      const { mode, accent } = get();
      const newMode = mode === 'dark' ? 'light' : 'dark';
      const colors = getThemeColors(newMode, accent);
      applyTheme(colors);
      saveTheme(newMode, accent);
      set({ mode: newMode, colors });
    },

    initializeTheme: () => {
      const { mode, accent } = get();
      const colors = getThemeColors(mode, accent);
      applyTheme(colors);
    },
  };
});

// Export accent color options for UI
export const accentColorOptions: { value: AccentColor; label: string; color: string }[] = [
  { value: 'teal', label: 'Teal', color: accentColors.teal.accent },
  { value: 'blue', label: 'Blue', color: accentColors.blue.accent },
  { value: 'purple', label: 'Purple', color: accentColors.purple.accent },
  { value: 'orange', label: 'Orange', color: accentColors.orange.accent },
  { value: 'pink', label: 'Pink', color: accentColors.pink.accent },
];
