# Design Guidelines

This document outlines the UI/UX design conventions used throughout the Follicle Labeller application.

## Theme System

The application supports multiple themes through CSS variables. See `src/store/themeStore.ts` for the complete theme configuration.

### CSS Variables

All colors should use CSS variables rather than hardcoded values:

| Variable | Purpose |
|----------|---------|
| `--bg-canvas` | Darkest background (main canvas area) |
| `--bg-primary` | Primary background |
| `--bg-secondary` | Secondary/elevated background |
| `--bg-tertiary` | Tertiary background |
| `--bg-hover` | Hover state background |
| `--text-primary` | Primary text color |
| `--text-secondary` | Secondary/muted text color |
| `--accent` | Accent/brand color |
| `--accent-hover` | Accent hover state |
| `--danger` | Danger/error color |
| `--danger-hover` | Danger hover state |
| `--border` | Border color |
| `--success` | Success color |

### Typography Variables

| Variable | Purpose |
|----------|---------|
| `--font-sans` | Sans-serif font stack |
| `--font-mono` | Monospace font stack |
| `--text-xs` | Extra small text (11px) |
| `--text-sm` | Small text (12px) |
| `--text-base` | Base text (13px) |
| `--text-md` | Medium text (14px) |
| `--text-lg` | Large text (16px) |
| `--text-xl` | Extra large text (18px) |
| `--font-normal` | Normal font weight (400) |
| `--font-medium` | Medium font weight (500) |
| `--font-semibold` | Semi-bold font weight (600) |

## Button Styles

### Standard Buttons

```css
.button {
  background: var(--bg-primary);
  color: var(--text-primary);
  border: 1px solid var(--border);
}

.button:hover:not(:disabled) {
  background: var(--bg-hover);
}

.button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
```

### Active/Selected Buttons

When a button represents an active or selected state, use the accent color as background with white text for contrast:

```css
.button.active {
  background-color: var(--accent);
  border-color: var(--accent);
  color: white;
}
```

This pattern is used consistently for:
- Tab buttons (`.tab-button.active`)
- Toggle buttons (`.toggle-btn.active`)
- Method selection buttons (`.method-btn.active`)
- Backend selection buttons (`.backend-btn.active`)
- Toolbar icon buttons (`.icon-button.active`)

### Danger Buttons

```css
.button.danger:hover:not(:disabled) {
  color: var(--danger);
  border-color: var(--danger);
}
```

### Accent/Primary Action Buttons

```css
.button.primary {
  background: var(--accent);
  color: white;
  border: none;
}

.button.primary:hover:not(:disabled) {
  background: var(--accent-hover);
}
```

## Tab Navigation

Tabs should follow this pattern:

```css
.tab-button {
  padding: 12px 24px;
  background: none;
  border: none;
  border-bottom: 2px solid transparent;
  color: var(--text-secondary);
  font-weight: var(--font-medium);
  cursor: pointer;
}

.tab-button:hover {
  color: var(--text-primary);
  background: var(--bg-hover);
}

.tab-button.active {
  background-color: var(--accent);
  border-color: var(--accent);
  color: white;
  font-weight: var(--font-semibold);
}
```

## Cards

```css
.card {
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px;
}

.card:hover {
  border-color: var(--accent);
}

.card.selected {
  border-color: var(--success);
  background: rgba(16, 185, 129, 0.1);
}
```

## Dialogs/Modals

```css
.dialog-overlay {
  background: rgba(0, 0, 0, 0.6);
}

.dialog {
  background: var(--bg-secondary);
  border-radius: 12px;
  box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
}

.dialog-header {
  padding: 16px 20px;
  border-bottom: 1px solid var(--border);
}

.dialog-content {
  padding: 20px;
}

.dialog-footer {
  padding: 16px 20px;
  border-top: 1px solid var(--border);
}
```

## Form Elements

### Input Fields

```css
input, select, textarea {
  background: var(--bg-primary);
  color: var(--text-primary);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 8px 12px;
}

input:focus, select:focus, textarea:focus {
  border-color: var(--accent);
  outline: none;
}
```

### Labels

```css
label {
  color: var(--text-secondary);
  font-size: var(--text-sm);
  font-weight: var(--font-medium);
}
```

## Status Indicators

### Success State
```css
.success {
  color: var(--success);
  background: rgba(16, 185, 129, 0.1);
}
```

### Error State
```css
.error {
  color: var(--danger);
  background: rgba(239, 68, 68, 0.1);
}
```

### Warning State
```css
.warning {
  color: #f59e0b;
  background: rgba(245, 158, 11, 0.1);
}
```

## Icons

- Use Lucide React icons consistently
- Icon size in buttons: 16px
- Icon size in headers: 18-24px
- Icons should inherit color from parent element

## Spacing

Use consistent spacing values:
- `4px` - Tight spacing (gap between icon and text)
- `8px` - Small spacing
- `12px` - Medium spacing
- `16px` - Standard spacing
- `20px` - Large spacing (dialog padding)
- `24px` - Extra large spacing

## Border Radius

- Small elements (badges, chips): `4px`
- Buttons, inputs: `6px`
- Cards: `8px`
- Dialogs: `12px`

## Transitions

Use consistent transition timing:
```css
transition: all 0.15s ease;
```

## Dark Theme Considerations

The application uses CSS variables that automatically adapt to dark/light themes. Avoid hardcoding colors. When specific dark theme adjustments are needed, use:

```css
[data-theme="dark"] .element {
  /* dark-specific styles */
}
```

## Accessibility

- Ensure sufficient color contrast (WCAG AA minimum)
- Disabled states should have `opacity: 0.5` and `cursor: not-allowed`
- Interactive elements should have visible focus states
- Use `title` attributes for icon-only buttons
