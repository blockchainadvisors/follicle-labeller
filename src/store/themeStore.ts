import { create } from 'zustand';

// Theme color palettes
export interface ThemeColors {
  bgCanvas: string;    // Darkest - for main canvas area
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
export type AccentColor = 'teal' | 'blue' | 'purple' | 'orange' | 'pink' | 'red' | 'green' | 'yellow' | 'cyan' | 'indigo' | 'rose' | 'amber' | 'lime' | 'emerald' | 'sky' | 'violet' | 'fuchsia' | 'coral' | 'crimson' | 'sapphire' | 'forest' | 'plum' | 'gold';
export type BackgroundTheme = 'midnight' | 'charcoal' | 'forest' | 'ocean' | 'slate' | 'clean' | 'warm' | 'cool';

// Accent color definitions
const accentColors: Record<AccentColor, { accent: string; accentHover: string }> = {
  teal: { accent: '#4ECDC4', accentHover: '#3dbdb5' },
  blue: { accent: '#3B82F6', accentHover: '#2563EB' },
  purple: { accent: '#8B5CF6', accentHover: '#7C3AED' },
  orange: { accent: '#F97316', accentHover: '#EA580C' },
  pink: { accent: '#EC4899', accentHover: '#DB2777' },
  red: { accent: '#EF4444', accentHover: '#DC2626' },
  green: { accent: '#22C55E', accentHover: '#16A34A' },
  yellow: { accent: '#EAB308', accentHover: '#CA8A04' },
  cyan: { accent: '#06B6D4', accentHover: '#0891B2' },
  indigo: { accent: '#6366F1', accentHover: '#4F46E5' },
  rose: { accent: '#F43F5E', accentHover: '#E11D48' },
  amber: { accent: '#F59E0B', accentHover: '#D97706' },
  lime: { accent: '#84CC16', accentHover: '#65A30D' },
  emerald: { accent: '#10B981', accentHover: '#059669' },
  sky: { accent: '#0EA5E9', accentHover: '#0284C7' },
  violet: { accent: '#A855F7', accentHover: '#9333EA' },
  fuchsia: { accent: '#D946EF', accentHover: '#C026D3' },
  // Darker colors that work well on light themes
  coral: { accent: '#E85D4C', accentHover: '#D14836' },
  crimson: { accent: '#DC143C', accentHover: '#B91030' },
  sapphire: { accent: '#0F52BA', accentHover: '#0A3D8F' },
  forest: { accent: '#228B22', accentHover: '#1A6B1A' },
  plum: { accent: '#8E4585', accentHover: '#6E3467' },
  gold: { accent: '#B8860B', accentHover: '#996F09' },
};

// Background theme definitions
interface BackgroundColors {
  bgCanvas: string;   // Darkest - for main canvas area
  bgPrimary: string;
  bgSecondary: string;
  bgTertiary: string;
  textPrimary: string;
  textSecondary: string;
  border: string;
  danger: string;
  dangerHover: string;
  success: string;
  mode: ThemeMode;
}

const backgroundThemes: Record<BackgroundTheme, BackgroundColors> = {
  // Dark themes
  midnight: {
    bgCanvas: '#0d0d1a',
    bgPrimary: '#1a1a2e',
    bgSecondary: '#16213e',
    bgTertiary: '#0f3460',
    textPrimary: '#ffffff',
    textSecondary: '#a0a0a0',
    border: '#2a2a4a',
    danger: '#e94560',
    dangerHover: '#d63d56',
    success: '#4caf50',
    mode: 'dark',
  },
  charcoal: {
    bgCanvas: '#0f0f0f',
    bgPrimary: '#1a1a1a',
    bgSecondary: '#252525',
    bgTertiary: '#333333',
    textPrimary: '#ffffff',
    textSecondary: '#a0a0a0',
    border: '#404040',
    danger: '#ef4444',
    dangerHover: '#dc2626',
    success: '#22c55e',
    mode: 'dark',
  },
  forest: {
    bgCanvas: '#0f1a0f',
    bgPrimary: '#1a2e1a',
    bgSecondary: '#1e3e1e',
    bgTertiary: '#2a4a2a',
    textPrimary: '#ffffff',
    textSecondary: '#a0b0a0',
    border: '#3a5a3a',
    danger: '#e94560',
    dangerHover: '#d63d56',
    success: '#4ade80',
    mode: 'dark',
  },
  ocean: {
    bgCanvas: '#080c1a',
    bgPrimary: '#0f172a',
    bgSecondary: '#1e293b',
    bgTertiary: '#334155',
    textPrimary: '#f8fafc',
    textSecondary: '#94a3b8',
    border: '#475569',
    danger: '#f87171',
    dangerHover: '#ef4444',
    success: '#4ade80',
    mode: 'dark',
  },
  slate: {
    bgCanvas: '#11111b',
    bgPrimary: '#1e1e2e',
    bgSecondary: '#282838',
    bgTertiary: '#363648',
    textPrimary: '#cdd6f4',
    textSecondary: '#9399b2',
    border: '#45475a',
    danger: '#f38ba8',
    dangerHover: '#eba0ac',
    success: '#a6e3a1',
    mode: 'dark',
  },
  // Light themes
  clean: {
    bgCanvas: '#e0e0e0',
    bgPrimary: '#f5f5f5',
    bgSecondary: '#ffffff',
    bgTertiary: '#e8e8e8',
    textPrimary: '#1a1a2e',
    textSecondary: '#666666',
    border: '#d1d5db',
    danger: '#dc2626',
    dangerHover: '#b91c1c',
    success: '#16a34a',
    mode: 'light',
  },
  warm: {
    bgCanvas: '#ebe5d8',
    bgPrimary: '#faf8f5',
    bgSecondary: '#fffcf7',
    bgTertiary: '#f5f0e8',
    textPrimary: '#2d2a26',
    textSecondary: '#6b6560',
    border: '#e0d8cc',
    danger: '#c53030',
    dangerHover: '#9c2626',
    success: '#2f855a',
    mode: 'light',
  },
  cool: {
    bgCanvas: '#d8e0e8',
    bgPrimary: '#f0f4f8',
    bgSecondary: '#f8fafc',
    bgTertiary: '#e2e8f0',
    textPrimary: '#1a202c',
    textSecondary: '#4a5568',
    border: '#cbd5e0',
    danger: '#e53e3e',
    dangerHover: '#c53030',
    success: '#38a169',
    mode: 'light',
  },
};

// Get full theme colors by combining background and accent
function getThemeColors(background: BackgroundTheme, accent: AccentColor): ThemeColors {
  const bgTheme = backgroundThemes[background];
  const accentPair = accentColors[accent];
  return {
    bgCanvas: bgTheme.bgCanvas,
    bgPrimary: bgTheme.bgPrimary,
    bgSecondary: bgTheme.bgSecondary,
    bgTertiary: bgTheme.bgTertiary,
    textPrimary: bgTheme.textPrimary,
    textSecondary: bgTheme.textSecondary,
    border: bgTheme.border,
    danger: bgTheme.danger,
    dangerHover: bgTheme.dangerHover,
    success: bgTheme.success,
    ...accentPair,
  };
}

// Apply theme colors to CSS variables
function applyTheme(colors: ThemeColors, mode: ThemeMode): void {
  const root = document.documentElement;
  root.style.setProperty('--bg-canvas', colors.bgCanvas);
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
  root.setAttribute('data-theme', mode);
}

// Get mode for a background theme
function getModeForBackground(bg: BackgroundTheme): ThemeMode {
  return backgroundThemes[bg].mode;
}

// localStorage key
const THEME_STORAGE_KEY = 'follicle-labeller-theme';

// Load saved theme from localStorage
function loadSavedTheme(): { background: BackgroundTheme; accent: AccentColor } {
  try {
    const saved = localStorage.getItem(THEME_STORAGE_KEY);
    if (saved) {
      const parsed = JSON.parse(saved);
      // Validate the values
      if (Object.keys(backgroundThemes).includes(parsed.background) &&
          Object.keys(accentColors).includes(parsed.accent)) {
        return parsed;
      }
    }
  } catch {
    // Ignore parse errors
  }
  return { background: 'clean', accent: 'sapphire' }; // Default
}

// Save theme to localStorage
function saveTheme(background: BackgroundTheme, accent: AccentColor): void {
  try {
    localStorage.setItem(THEME_STORAGE_KEY, JSON.stringify({ background, accent }));
  } catch {
    // Ignore storage errors
  }
}

interface ThemeState {
  background: BackgroundTheme;
  accent: AccentColor;
  mode: ThemeMode; // Derived from background
  colors: ThemeColors;

  // Actions
  setBackground: (background: BackgroundTheme) => void;
  setAccent: (accent: AccentColor) => void;
  initializeTheme: () => void;
}

export const useThemeStore = create<ThemeState>((set, get) => {
  const initial = loadSavedTheme();
  const initialMode = getModeForBackground(initial.background);

  return {
    background: initial.background,
    accent: initial.accent,
    mode: initialMode,
    colors: getThemeColors(initial.background, initial.accent),

    setBackground: (background) => {
      const { accent } = get();
      const colors = getThemeColors(background, accent);
      const mode = getModeForBackground(background);
      applyTheme(colors, mode);
      saveTheme(background, accent);
      set({ background, mode, colors });
    },

    setAccent: (accent) => {
      const { background, mode } = get();
      const colors = getThemeColors(background, accent);
      applyTheme(colors, mode);
      saveTheme(background, accent);
      set({ accent, colors });
    },

    initializeTheme: () => {
      const { background, accent, mode } = get();
      const colors = getThemeColors(background, accent);
      applyTheme(colors, mode);
    },
  };
});

// Export accent color options for UI
export const accentColorOptions: { value: AccentColor; label: string; color: string }[] = [
  { value: 'teal', label: 'Teal', color: accentColors.teal.accent },
  { value: 'cyan', label: 'Cyan', color: accentColors.cyan.accent },
  { value: 'sky', label: 'Sky', color: accentColors.sky.accent },
  { value: 'blue', label: 'Blue', color: accentColors.blue.accent },
  { value: 'sapphire', label: 'Sapphire', color: accentColors.sapphire.accent },
  { value: 'indigo', label: 'Indigo', color: accentColors.indigo.accent },
  { value: 'violet', label: 'Violet', color: accentColors.violet.accent },
  { value: 'purple', label: 'Purple', color: accentColors.purple.accent },
  { value: 'plum', label: 'Plum', color: accentColors.plum.accent },
  { value: 'fuchsia', label: 'Fuchsia', color: accentColors.fuchsia.accent },
  { value: 'pink', label: 'Pink', color: accentColors.pink.accent },
  { value: 'rose', label: 'Rose', color: accentColors.rose.accent },
  { value: 'crimson', label: 'Crimson', color: accentColors.crimson.accent },
  { value: 'red', label: 'Red', color: accentColors.red.accent },
  { value: 'coral', label: 'Coral', color: accentColors.coral.accent },
  { value: 'orange', label: 'Orange', color: accentColors.orange.accent },
  { value: 'amber', label: 'Amber', color: accentColors.amber.accent },
  { value: 'gold', label: 'Gold', color: accentColors.gold.accent },
  { value: 'yellow', label: 'Yellow', color: accentColors.yellow.accent },
  { value: 'lime', label: 'Lime', color: accentColors.lime.accent },
  { value: 'green', label: 'Green', color: accentColors.green.accent },
  { value: 'forest', label: 'Forest', color: accentColors.forest.accent },
  { value: 'emerald', label: 'Emerald', color: accentColors.emerald.accent },
];

// Export background theme options for UI
export const darkBackgroundOptions: { value: BackgroundTheme; label: string; color: string }[] = [
  { value: 'midnight', label: 'Midnight', color: backgroundThemes.midnight.bgPrimary },
  { value: 'charcoal', label: 'Charcoal', color: backgroundThemes.charcoal.bgPrimary },
  { value: 'forest', label: 'Forest', color: backgroundThemes.forest.bgPrimary },
  { value: 'ocean', label: 'Ocean', color: backgroundThemes.ocean.bgPrimary },
  { value: 'slate', label: 'Slate', color: backgroundThemes.slate.bgPrimary },
];

export const lightBackgroundOptions: { value: BackgroundTheme; label: string; color: string }[] = [
  { value: 'clean', label: 'Clean', color: backgroundThemes.clean.bgPrimary },
  { value: 'warm', label: 'Warm', color: backgroundThemes.warm.bgPrimary },
  { value: 'cool', label: 'Cool', color: backgroundThemes.cool.bgPrimary },
];
