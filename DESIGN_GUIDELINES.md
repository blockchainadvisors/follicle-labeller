# Follicle Labeller Design Guidelines

**Version:** 1.0
**Last Updated:** January 2026
**Application:** Follicle Labeller - Image Annotation Tool

---

## Table of Contents

1. [Design Principles](#1-design-principles)
2. [Color System](#2-color-system)
3. [Typography](#3-typography)
4. [Spacing System](#4-spacing-system)
5. [Border Radius](#5-border-radius)
6. [Shadows & Elevation](#6-shadows--elevation)
7. [Z-Index Layers](#7-z-index-layers)
8. [Transitions & Animations](#8-transitions--animations)
9. [Component Specifications](#9-component-specifications)
10. [Layout Patterns](#10-layout-patterns)
11. [Icon System](#11-icon-system)
12. [Responsive Design](#12-responsive-design)
13. [Accessibility Guidelines](#13-accessibility-guidelines)
14. [Implementation Reference](#14-implementation-reference)

---

## 1. Design Principles

### Core Values

1. **Consistency** - All UI elements follow the same visual language
2. **Clarity** - Interface should be immediately understandable
3. **Efficiency** - Minimize user cognitive load and clicks
4. **Feedback** - Every interaction provides visual response
5. **Accessibility** - Usable by all users regardless of ability

### Visual Hierarchy

- Use size, weight, and color to establish hierarchy
- Primary actions should be visually prominent
- Secondary content uses muted colors
- Destructive actions use danger color

---

## 2. Color System

### CSS Custom Properties

All colors are defined as CSS custom properties (variables) and applied dynamically via JavaScript. Never use hardcoded color values in components.

### Semantic Color Variables

| Variable | Purpose | Dark Theme Default | Light Theme Default |
|----------|---------|-------------------|---------------------|
| `--bg-canvas` | Canvas/darkest background | `#0d0d1a` | `#e0e0e0` |
| `--bg-primary` | Main background | `#1a1a2e` | `#f5f5f5` |
| `--bg-secondary` | Panels, sidebars | `#16213e` | `#ffffff` |
| `--bg-tertiary` | Inputs, buttons | `#0f3460` | `#e8e8e8` |
| `--bg-hover` | Hover states | `#2a2a4a` | `#f0f0f0` |
| `--text-primary` | Primary text | `#ffffff` | `#1a1a2e` |
| `--text-secondary` | Secondary text | `#a0a0a0` | `#666666` |
| `--text-tertiary` | Muted text | `#6a6a8a` | - |
| `--accent` | Interactive elements | `#4ECDC4` (teal) | Configurable |
| `--accent-hover` | Accent hover state | `#3dbdb5` | Configurable |
| `--danger` | Destructive actions | `#e94560` | `#dc2626` |
| `--danger-hover` | Danger hover state | `#d63d56` | `#b91c1c` |
| `--warning` | Warning states | `#f59e0b` | `#f59e0b` |
| `--success` | Success states | `#4caf50` | `#16a34a` |
| `--border` | Dividers, borders | `#2a2a4a` | `#d1d5db` |

### Semantic Background Colors (Semi-transparent)

Use for status backgrounds and highlights:

```css
--success-bg: rgba(76, 175, 80, 0.1);
--danger-bg: rgba(233, 69, 96, 0.1);
--warning-bg: rgba(245, 158, 11, 0.1);
```

### Theme System

#### Background Themes (8 options)

**Dark Themes:**
| Theme | Primary BG | Description |
|-------|-----------|-------------|
| Midnight | `#1a1a2e` | Deep blue-purple, default dark |
| Charcoal | `#1a1a1a` | Pure neutral dark |
| Forest | `#1a2e1a` | Dark green tint |
| Ocean | `#0f172a` | Slate blue |
| Slate | `#1e1e2e` | Catppuccin-inspired |

**Light Themes:**
| Theme | Primary BG | Description |
|-------|-----------|-------------|
| Clean | `#f5f5f5` | Pure neutral light |
| Warm | `#faf8f5` | Cream/warm tint |
| Cool | `#f0f4f8` | Blue-gray tint |

#### Accent Colors (23 options)

Organized by hue for the color picker:

```
Teal → Cyan → Sky → Blue → Sapphire → Indigo → Violet → Purple → Plum
→ Fuchsia → Pink → Rose → Crimson → Red → Coral → Orange → Amber
→ Gold → Yellow → Lime → Green → Forest → Emerald
```

### Color Usage Rules

1. **Never** use hardcoded hex colors - always use CSS variables
2. **Always** use `--accent` for interactive elements
3. **Always** use `--danger` for destructive actions
4. **Always** test new components in both light and dark themes
5. Use `[data-theme="dark"]` or `[data-theme="light"]` selectors for theme-specific overrides

---

## 3. Typography

### Font Families

```css
--font-sans: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
--font-mono: 'SF Mono', 'Monaco', 'Consolas', monospace;
```

### Font Size Scale

| Variable | Size | Usage |
|----------|------|-------|
| `--text-2xs` | 10px | Zoom display, minimal UI |
| `--text-xs` | 11px | Captions, hints, chart labels |
| `--text-sm` | 12px | Uppercase labels, tips, secondary info |
| `--text-base` | 13px | Primary UI text, inputs, buttons |
| `--text-md` | 14px | Emphasized body, form labels |
| `--text-lg` | 16px | Panel headings, metrics |
| `--text-xl` | 18px | Dialog titles |
| `--text-2xl` | 22px | Major headings |

### Font Weights

| Variable | Weight | Usage |
|----------|--------|-------|
| `--font-normal` | 400 | Body text |
| `--font-medium` | 500 | Labels, emphasis |
| `--font-semibold` | 600 | Headings, buttons |

### Line Heights

| Variable | Value | Usage |
|----------|-------|-------|
| `--leading-tight` | 1.25 | Headings |
| `--leading-normal` | 1.5 | Body text |
| `--leading-relaxed` | 1.6 | Extended reading |

### Letter Spacing

| Variable | Value | Usage |
|----------|-------|-------|
| `--tracking-normal` | 0 | Default |
| `--tracking-wide` | 0.5px | Uppercase labels |

### Heading Hierarchy

```css
h1, .h1: 22px, semibold, tight line-height
h2, .h2: 18px, semibold, tight line-height
h3, .h3: 16px, semibold, tight line-height
h4, .h4: 14px, semibold, tight line-height
```

### Special Text Patterns

**Uppercase Label:**
```css
.uppercase-label {
  font-size: var(--text-sm);        /* 12px */
  font-weight: var(--font-medium);  /* 500 */
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: var(--tracking-wide);  /* 0.5px */
}
```

**Monospace Numbers:**
```css
.mono {
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
}
```

---

## 4. Spacing System

### Base Unit

The base spacing unit is **4px**. All spacing values should be multiples of 4.

### Spacing Scale

| Size | Value | Usage |
|------|-------|-------|
| 2xs | 4px | Minimal gaps, icon spacing |
| xs | 6px | Small padding, tight groups |
| sm | 8px | Standard small spacing |
| md | 10px | Padding, form elements |
| base | 12px | Default padding, moderate spacing |
| lg | 16px | Standard padding, panel margins |
| xl | 20px | Large padding, dialog content |
| 2xl | 24px | Major section spacing |
| 3xl | 32px | Empty state padding |

### Component-Specific Spacing

| Component | Padding | Gap |
|-----------|---------|-----|
| Toolbar | 6px 8px | 4px between groups |
| Toolbar Group | 2px | 1px between items |
| Dialog Header | 16px 20px | - |
| Dialog Body | 20px | - |
| Dialog Footer | 16px 20px | 10px between buttons |
| Property Panel | 16px | - |
| Property Group | margin-bottom: 16px | - |
| Image Explorer | 8px (list) | 10px item gap |
| Form Inputs | 8px 10px | - |
| Buttons | 8-10px 16-20px | - |

---

## 5. Border Radius

### Border Radius Scale

| Size | Value | Usage |
|------|-------|-------|
| none | 0 | No rounding |
| sm | 2px | Slider track |
| base | 3px | Code blocks |
| md | 4px | Form inputs, close buttons |
| lg | 5px | Icon buttons |
| xl | 6px | Toolbar groups, panels, tips |
| 2xl | 8px | Dialogs, dropdowns |
| 3xl | 12px | Large modals, empty states |
| full | 50% | Circular elements |

### Component Border Radius

| Component | Border Radius |
|-----------|---------------|
| Icon Button | 5px |
| Text Input | 4px |
| Dialog | 8px |
| Dropdown | 8px |
| Confirmation Dialog | 12px |
| Toolbar Group | 6px |
| Color Swatch | 4px |
| Scrollbar Thumb | 4px |
| Image Thumbnail | 4px |
| Active Image Item | 6px |

---

## 6. Shadows & Elevation

### Shadow Scale

| Level | Shadow | Usage |
|-------|--------|-------|
| None | - | Flat elements |
| Subtle | `0 1px 2px rgba(0,0,0,0.1)` | Subtle lift |
| Low | `0 2px 4px rgba(0,0,0,0.2)` | Slider thumb, small elevated elements |
| Medium | `0 4px 12px rgba(0,0,0,0.15)` | Cards, panels |
| High | `0 8px 24px rgba(0,0,0,0.3)` | Dropdowns, tooltips |
| Max | `0 8px 32px rgba(0,0,0,0.4)` | Dialogs, overlays |

### Component Shadows

| Component | Shadow |
|-----------|--------|
| Dialog | `0 8px 32px rgba(0,0,0,0.4)` |
| Dropdown | `0 8px 24px rgba(0,0,0,0.4)` |
| Help Panel | `-4px 0 20px rgba(0,0,0,0.3)` |
| Unlock Warning Dialog | `0 8px 32px rgba(0,0,0,0.24)` |

---

## 7. Z-Index Layers

### Z-Index Scale

| Layer | Value | Usage |
|-------|-------|-------|
| Base | 0-50 | Content, canvas elements |
| Panel | 100 | Side panels, help panel, stats panel, dropdowns |
| Dialog | 200 | Main dialogs |
| Modal | 1000 | Full-screen overlays, confirmation dialogs |

### Component Z-Index Values

| Component | Z-Index |
|-----------|---------|
| Help Panel | 100 |
| Statistics Panel | 100 |
| Theme Dropdown | 100 |
| Dialog Overlay | 200 |
| Delete Confirmation | 1000 |
| Unlock Warning | 1000 |

---

## 8. Transitions & Animations

### Standard Transition

```css
transition: all 0.15s ease;
```

Use `0.15s ease` for most UI interactions including:
- Button hover/active states
- Border color changes
- Background color changes
- Opacity changes

### Transition Variants

| Duration | Easing | Usage |
|----------|--------|-------|
| 0.1s | ease | Slider thumb scale |
| 0.15s | ease | Most UI interactions |
| 0.2s | ease | Panel slides, dialog appearance |
| 1s | linear | Loading spinners (infinite) |

### Keyframe Animations

**fadeIn** - For overlay backgrounds:
```css
@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}
/* Duration: 0.15s ease */
```

**scaleIn** - For dialog appearance:
```css
@keyframes scaleIn {
  from { transform: scale(0.95); opacity: 0; }
  to { transform: scale(1); opacity: 1; }
}
/* Duration: 0.15s ease */
```

**dialogSlideIn** - For dialog entrance:
```css
@keyframes dialogSlideIn {
  from { opacity: 0; transform: translateY(-20px); }
  to { opacity: 1; transform: translateY(0); }
}
/* Duration: 0.2s ease */
```

**helpSlideIn** - For side panel entrance:
```css
@keyframes helpSlideIn {
  from { transform: translateX(100%); }
  to { transform: translateX(0); }
}
/* Duration: 0.2s ease */
```

**dropdownFadeIn** - For dropdown menus:
```css
@keyframes dropdownFadeIn {
  from { opacity: 0; transform: translateY(-4px); }
  to { opacity: 1; transform: translateY(0); }
}
/* Duration: 0.15s ease */
```

**spin** - For loading indicators:
```css
@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}
/* Duration: 1s linear infinite */
```

---

## 9. Component Specifications

### 9.1 Buttons

#### Icon Button (Toolbar)

```css
.icon-button {
  width: 30px;
  height: 30px;
  padding: 0;
  border: 1px solid transparent;
  border-radius: 5px;
  background-color: transparent;
  color: var(--text-primary);
  cursor: pointer;
  transition: all 0.15s ease;
}

/* States */
.icon-button:hover:not(:disabled) {
  background-color: var(--bg-tertiary);
  border-color: var(--border);
}

.icon-button:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.icon-button.active {
  background-color: var(--accent);
  border-color: var(--accent);
  color: var(--bg-primary);
}
```

**Icon size:** 18px (16px on mobile)

#### Text Button (Dialog)

```css
.dialog-btn {
  padding: 8px 16px;
  border: 1px solid var(--border);
  border-radius: 4px;
  background-color: var(--bg-tertiary);
  color: var(--text-primary);
  font-size: var(--text-base);
  font-weight: var(--font-medium);
  cursor: pointer;
  transition: all 0.15s ease;
}

.dialog-btn:hover {
  background-color: var(--bg-primary);
}

/* Primary variant */
.dialog-btn.primary {
  background-color: var(--accent);
  border-color: var(--accent);
  color: var(--bg-primary);
}

.dialog-btn.primary:hover {
  background-color: var(--accent-hover);
  border-color: var(--accent-hover);
}
```

#### Danger Button

```css
.delete-button {
  width: 100%;
  padding: 10px 16px;
  border: 1px solid var(--danger);
  border-radius: 4px;
  background-color: transparent;
  color: var(--danger);
  font-size: var(--text-base);
  font-weight: var(--font-medium);
  cursor: pointer;
  transition: all 0.15s ease;
}

.delete-button:hover {
  background-color: var(--danger);
  color: var(--text-primary);
}
```

#### Filter Button

```css
.filter-button {
  flex: 1;
  min-width: 60px;
  padding: 6px 10px;
  border: 1px solid var(--border);
  border-radius: 4px;
  background-color: var(--bg-tertiary);
  color: var(--text-secondary);
  font-size: var(--text-sm);
  cursor: pointer;
  transition: all 0.15s ease;
}

.filter-button:hover:not(:disabled) {
  background-color: var(--bg-hover);
  color: var(--text-primary);
}

.filter-button.active {
  background-color: var(--accent-bg);
  border-color: var(--accent);
  color: var(--accent);
}
```

### 9.2 Inputs

#### Text Input

```css
input[type="text"],
textarea {
  width: 100%;
  padding: 8px 10px;
  border: 1px solid var(--border);
  border-radius: 4px;
  background-color: var(--bg-tertiary);
  color: var(--text-primary);
  font-size: var(--text-base);
  font-family: inherit;
  transition: border-color 0.15s ease;
}

input[type="text"]:focus,
textarea:focus {
  outline: none;
  border-color: var(--accent);
}
```

#### Color Picker

```css
input[type="color"] {
  width: 40px;
  height: 32px;
  padding: 2px;
  border: 1px solid var(--border);
  border-radius: 4px;
  background-color: var(--bg-tertiary);
  cursor: pointer;
}
```

#### Slider

```css
.slider {
  height: 4px;
  -webkit-appearance: none;
  appearance: none;
  background: var(--bg-tertiary);
  border-radius: 2px;
  outline: none;
}

.slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 16px;
  height: 16px;
  background: var(--accent);
  border-radius: 50%;
  cursor: pointer;
  box-shadow: 0 2px 4px rgba(0,0,0,0.2);
  transition: transform 0.1s ease;
}

.slider::-webkit-slider-thumb:hover {
  transform: scale(1.1);
}
```

### 9.3 Dialogs

#### Dialog Structure

```
┌─────────────────────────────────────┐
│ Header (16px 20px padding)      [X] │
├─────────────────────────────────────┤
│                                     │
│ Body (20px padding)                 │
│                                     │
├─────────────────────────────────────┤
│           [Cancel] [Primary] Footer │
└─────────────────────────────────────┘
```

#### Dialog Specifications

```css
.dialog-overlay {
  position: fixed;
  inset: 0;
  background-color: rgba(0, 0, 0, 0.6);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 200;
  animation: dialogFadeIn 0.15s ease;
}

.dialog-content {
  background-color: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 8px;
  min-width: 400px;
  max-width: 600px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
  animation: dialogSlideIn 0.2s ease;
}

.dialog-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  border-bottom: 1px solid var(--border);
}

.dialog-body {
  padding: 20px;
}

.dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  padding: 16px 20px;
  border-top: 1px solid var(--border);
}
```

#### Confirmation Dialog (Warning)

```css
.confirm-dialog {
  background: var(--bg-secondary);
  border-radius: 12px;
  padding: 24px;
  width: 320px;
  max-width: 90vw;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
  text-align: center;
}

.warning-icon {
  color: var(--warning);
  margin-bottom: 12px;
}
```

### 9.4 Panels

#### Property Panel (Right Sidebar)

```css
.property-panel {
  width: 280px;
  background-color: var(--bg-secondary);
  border-left: 1px solid var(--border);
  padding: 16px;
  overflow-y: auto;
}
```

#### Image Explorer (Left Sidebar)

```css
.image-explorer {
  width: 200px;
  background-color: var(--bg-secondary);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
```

#### Help Panel (Slide-in)

```css
.help-panel {
  position: fixed;
  top: 48px;
  right: 0;
  width: 420px;
  height: calc(100% - 48px);
  background-color: var(--bg-secondary);
  border-left: 1px solid var(--border);
  z-index: 100;
  box-shadow: -4px 0 20px rgba(0, 0, 0, 0.3);
  animation: helpSlideIn 0.2s ease;
}
```

### 9.5 Toolbar

```css
.toolbar {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 6px 8px;
  background-color: var(--bg-secondary);
  border-bottom: 1px solid var(--border);
  min-height: 44px;
  flex-wrap: wrap;
}

.toolbar-group {
  display: flex;
  align-items: center;
  gap: 1px;
  padding: 2px;
  background-color: rgba(0, 0, 0, 0.2);
  border-radius: 6px;
}

.toolbar-divider {
  width: 1px;
  height: 20px;
  background-color: var(--border);
  margin: 0 2px;
}
```

### 9.6 Dropdown

```css
.dropdown {
  position: absolute;
  top: calc(100% + 8px);
  right: 0;
  background-color: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px;
  min-width: 200px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
  z-index: 100;
  animation: dropdownFadeIn 0.15s ease;
}
```

### 9.7 Empty States

```css
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  color: var(--text-secondary);
  padding: 32px 48px;
  border-radius: 12px;
  cursor: pointer;
  transition: all 0.15s ease;
}

.empty-state:hover {
  background-color: var(--bg-tertiary);
  color: var(--text-primary);
}

.empty-state:hover svg {
  color: var(--accent);
}
```

### 9.8 Status Display

```css
.status-display {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: var(--text-xs);
  color: var(--text-secondary);
}

/* Status colors */
.status-file { color: var(--accent); }
.status-count { color: var(--success); }
.status-size { color: var(--text-secondary); }
```

### 9.9 Lock Status

```css
.lock-status {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 12px;
  background-color: rgba(78, 205, 196, 0.1);
  border: 1px solid rgba(78, 205, 196, 0.3);
  border-radius: 6px;
}

.lock-info {
  display: flex;
  align-items: center;
  gap: 6px;
  color: var(--accent);
  font-size: var(--text-base);
  font-weight: var(--font-medium);
}
```

---

## 10. Layout Patterns

### App Structure

```
┌──────────────────────────────────────────────────────────┐
│                        Toolbar                           │
├────────┬─────────────────────────────────────┬───────────┤
│        │                                     │           │
│ Image  │                                     │ Property  │
│ Explorer│           Canvas                   │ Panel     │
│ (200px)│           (flex: 1)                │ (280px)   │
│        │                                     │           │
│        │                                     │           │
└────────┴─────────────────────────────────────┴───────────┘
```

### Flexbox Patterns

**Main Layout:**
```css
.app {
  display: flex;
  flex-direction: column;
  height: 100%;
  width: 100%;
}

.main-content {
  display: flex;
  flex: 1;
  overflow: hidden;
}
```

**Toolbar:**
```css
.toolbar {
  display: flex;
  align-items: center;
  gap: 4px;
  flex-wrap: wrap;
}

.toolbar-spacer {
  flex: 1;
}
```

**Dialog Footer:**
```css
.dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
}
```

### Grid Patterns

**Color Grid (Theme Picker):**
```css
.color-grid {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 6px;
}
```

### Fixed Widths

| Component | Width |
|-----------|-------|
| Property Panel | 280px |
| Image Explorer | 200px |
| Dialog (min) | 400px |
| Dialog (max) | 600px |
| Help Panel | 420px |
| Dropdown (min) | 200px |

---

## 11. Icon System

### Library

**lucide-react** v0.469.0

### Icon Sizes

| Context | Size |
|---------|------|
| Toolbar (default) | 18px |
| Toolbar (mobile) | 16px |
| Dialog header close | 18px |
| Warning icon | 28px |
| Empty state | 48px |
| Small inline | 12px |
| Status indicator | 14px |

### Common Icons Reference

| Icon | Import | Usage |
|------|--------|-------|
| Pencil | `Pencil` | Create mode |
| MousePointer2 | `MousePointer2` | Select mode |
| Hand | `Hand` | Pan mode |
| ZoomIn/ZoomOut | `ZoomIn`, `ZoomOut` | Zoom controls |
| RotateCcw | `RotateCcw` | Reset zoom |
| HelpCircle | `HelpCircle` | Help button |
| Eye/EyeOff | `Eye`, `EyeOff` | Show/hide |
| Tag | `Tag` | Toggle labels |
| ImagePlus | `ImagePlus` | Add image |
| BoxSelect | `BoxSelect` | Marquee selection |
| Lasso | `Lasso` | Lasso selection |
| Sparkles | `Sparkles` | Auto detect |
| Loader2 | `Loader2` | Loading (with spin) |
| FolderOpen | `FolderOpen` | Open project |
| Save | `Save` | Save project |
| Flame | `Flame` | Heatmap |
| BarChart3 | `BarChart3` | Statistics |
| Settings | `Settings` | Settings |
| GraduationCap | `GraduationCap` | Learned detection |
| Brain | `Brain` | YOLO training |
| Database | `Database` | Model manager |
| FileUp | `FileUp` | Import |
| Download | `Download` | Export |
| Lock/Unlock | `Lock`, `Unlock` | Lock status |
| AlertTriangle | `AlertTriangle` | Warning |
| Crosshair | `Crosshair` | Origin prediction |
| Sun/Moon | `Sun`, `Moon` | Theme toggle |
| Palette | `Palette` | Theme picker |
| X | `X` | Close |
| Check | `Check` | Confirm |
| ChevronDown/Up | `ChevronDown`, `ChevronUp` | Dropdowns |

### Icon Button Pattern

```tsx
<button className="icon-button" title="Action description">
  <IconComponent size={18} />
</button>
```

---

## 12. Responsive Design

### Breakpoints

| Breakpoint | Width | Changes |
|------------|-------|---------|
| Large | > 900px | Full layout |
| Medium | ≤ 900px | Compact toolbar |
| Small | ≤ 700px | Hide status section |

### Responsive Changes

**≤ 900px:**
- Hide toolbar dividers
- Reduce toolbar gap to 3px
- Icon buttons: 28x28px (from 30x30px)
- Icon SVG: 16px (from 18px)
- Zoom display: 36px min-width, 10px font

**≤ 700px:**
- Hide `.toolbar-status`
- Hide `.toolbar-spacer`

### Media Query Pattern

```css
@media (max-width: 900px) {
  .toolbar .icon-button {
    width: 28px;
    height: 28px;
  }

  .toolbar .icon-button svg {
    width: 16px;
    height: 16px;
  }
}
```

---

## 13. Accessibility Guidelines

### Focus States

All interactive elements should have visible focus states:

```css
element:focus {
  outline: none;
  border-color: var(--accent);
}

/* Alternative ring style */
element:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 2px;
}
```

### Keyboard Navigation

- All buttons must be keyboard accessible
- Tab order should follow visual order
- Use `tabindex` appropriately

### ARIA Labels

```tsx
<button
  className="icon-button"
  title="Create annotation"
  aria-label="Create annotation"
>
  <Pencil size={18} />
</button>
```

### Color Contrast

- Ensure text has minimum 4.5:1 contrast ratio
- Interactive elements need 3:1 contrast
- Test with both light and dark themes

### Disabled States

```css
element:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}
```

---

## 14. Implementation Reference

### Adding a New Dialog

```tsx
// 1. Create overlay with animation
<div className="dialog-overlay" onClick={onClose}>

  // 2. Prevent close on content click
  <div className="dialog-content" onClick={e => e.stopPropagation()}>

    // 3. Header with title and close button
    <div className="dialog-header">
      <h3>Dialog Title</h3>
      <button className="dialog-close" onClick={onClose}>
        <X size={18} />
      </button>
    </div>

    // 4. Body content
    <div className="dialog-body">
      {/* Content here */}
    </div>

    // 5. Footer with actions
    <div className="dialog-footer">
      <button className="dialog-btn cancel" onClick={onClose}>
        Cancel
      </button>
      <button className="dialog-btn primary" onClick={onSubmit}>
        Confirm
      </button>
    </div>
  </div>
</div>
```

### Adding a New Panel Section

```tsx
<div className="property-group">
  <label className="uppercase-label">Section Label</label>
  {/* Content */}
</div>
```

### Adding a New Toolbar Button

```tsx
<button
  className={`icon-button ${isActive ? 'active' : ''}`}
  onClick={handleClick}
  title="Action tooltip"
  disabled={isDisabled}
>
  <IconName size={18} />
</button>
```

### Theme-Aware Styling

```css
/* Default styles (works for both themes) */
.component {
  background: var(--bg-secondary);
  color: var(--text-primary);
  border: 1px solid var(--border);
}

/* Dark theme specific */
[data-theme="dark"] .component {
  /* Dark-only overrides */
}

/* Light theme specific */
[data-theme="light"] .component {
  /* Light-only overrides */
}
```

### Creating Confirmation Dialogs

```tsx
<div className="delete-confirm-overlay">
  <div className="delete-confirm-dialog">
    <AlertTriangle size={28} className="delete-confirm-icon" />
    <h3>Confirmation Title</h3>
    <p className="delete-confirm-filename">Details here</p>
    <p className="delete-confirm-warning">
      <strong>Warning message</strong>
    </p>
    <div className="delete-confirm-actions">
      <button className="delete-confirm-cancel" onClick={onCancel}>
        Cancel
      </button>
      <button className="delete-confirm-delete" onClick={onConfirm}>
        Confirm
      </button>
    </div>
  </div>
</div>
```

---

## Quick Reference Cheat Sheet

### Colors
```css
/* Background */ --bg-primary, --bg-secondary, --bg-tertiary, --bg-hover
/* Text */       --text-primary, --text-secondary
/* Interactive */ --accent, --accent-hover
/* Status */     --danger, --success, --warning
/* Borders */    --border
```

### Typography
```css
/* Sizes */  --text-xs (11px), --text-base (13px), --text-lg (16px), --text-xl (18px)
/* Weights */ --font-medium (500), --font-semibold (600)
```

### Spacing
```css
/* Common */ 4px, 8px, 12px, 16px, 20px, 24px
```

### Border Radius
```css
/* Common */ 4px (inputs), 5px (icon-btn), 6px (groups), 8px (dialogs), 12px (confirms)
```

### Transitions
```css
transition: all 0.15s ease;  /* Standard */
```

### Z-Index
```css
/* Layers */ 100 (panels), 200 (dialogs), 1000 (confirms/overlays)
```

---

*This document should be updated whenever new UI patterns are introduced or existing patterns are modified.*
