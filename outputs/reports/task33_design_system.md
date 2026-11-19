# ConvoSense Design System Specification

**Product Name:** ConvoSense  
**Version:** 2.0.0 (Refined Product Edition)  
**Date:** November 13, 2026  
**Target Platform:** Modern Web (Desktop & Mobile, Responsive, Dark/Light Theming)  
**Architectural Scope:** Five-Screen Conversational Emotion Dynamics & Model Comparison Suite

---

## 1. Executive Design Brief

### 1.1 Brand Identity & Purpose
**ConvoSense** is an advanced conversational intelligence platform designed for researchers, computational linguists, and product teams analyzing multi-turn interpersonal communication. Unlike generic AI dashboards that rely on decorative cards and neon purple gradients, ConvoSense is designed as a serious, credible analytical workstation. It balances intuitive consumer usability on conversational exploration screens with deep empirical rigor in its research inspection area.

### 1.2 Design Principles
1. **Clarity Over Clutter:** Eliminate nested card containers and redundant borders. Use whitespace, typographical weight, and subtle tonal shifts to define structure.
2. **Semantic Iconography:** Strictly ban decorative emojis from the user interface. Employ crisp, recognizable inline SVG vector icons with consistent line weights (1.75px) and sizing (16px, 18px, 20px).
3. **Distinctive Typography:** Use **Plus Jakarta Sans**—a geometric yet warm grotesque with exceptional legibility across dense data interfaces—paired with **JetBrains Mono** for quantitative scores, turn indices, matrix cells, and token representations.
4. **Restrained Color Hierarchy:** A single energetic Cyan-Azure accent (`#0284c7` in light mode, `#38bdf8` in dark mode) directs attention to active interactions. Emotional states are communicated via a desaturated, accessible 7-color taxonomy.
5. **Purposeful Motion:** Restrained 150–200ms ease transitions for screen switching, drawer expansion, and target utterance highlights. No distracting floating animations or bouncing effects.
6. **Robust States:** Comprehensive design for loading, empty, error, and partial validation states with actionable recovery suggestions.
7. **Empirical Authenticity:** No placeholder figures or simulated dynamics. Every metric, chart, and matrix cell is directly wired to verified backend inference.

---

## 2. Design Tokens

### 2.1 Color Palette

#### Base Neutrals (Dark Theme — Default)
| Token | Hex Value | Role |
| :--- | :--- | :--- |
| `--bg-base` | `#090d16` | Main viewport canvas, deep slate abyss |
| `--bg-surface-1` | `#0f172a` | Primary layout sections and major panels |
| `--bg-surface-2` | `#172033` | Secondary components, table headers, chat bubbles |
| `--bg-surface-3` | `#1e293b` | Interactive hover states, elevated controls |
| `--border-subtle` | `#1e293b` | Soft component separators |
| `--border-default` | `#334155` | Standard structural borders |
| `--border-strong` | `#475569` | Highlighted borders and active outlines |
| `--text-primary` | `#f8fafc` | High-contrast body text and headings |
| `--text-secondary` | `#94a3b8` | Informational labels, metadata, captions |
| `--text-tertiary` | `#64748b` | Disabled text, subtle hints, placeholders |

#### Base Neutrals (Light Theme — Mist)
| Token | Hex Value | Role |
| :--- | :--- | :--- |
| `--bg-base` | `#f8fafc` | Clean, subtle cool-grey canvas |
| `--bg-surface-1` | `#ffffff` | Elevated primary card surfaces |
| `--bg-surface-2` | `#f1f5f9` | Inset backgrounds, input fields, chat cards |
| `--bg-surface-3` | `#e2e8f0` | Hover states, tab backgrounds |
| `--border-subtle` | `#e2e8f0` | Inset borders and divider rules |
| `--border-default` | `#cbd5e1` | Standard container outlines |
| `--border-strong` | `#94a3b8` | Focus rings and emphasized borders |
| `--text-primary` | `#0f172a` | Deep slate high-contrast typography |
| `--text-secondary` | `#475569` | Secondary descriptions and column headers |
| `--text-tertiary` | `#64748b` | Muted counters and supplementary notes |

#### Primary Accent (Cyan-Azure)
| Token | Dark Theme | Light Theme | Role |
| :--- | :--- | :--- | :--- |
| `--accent-primary` | `#38bdf8` | `#0284c7` | Brand mark, primary CTA, active tab highlight |
| `--accent-hover` | `#0ea5e9` | `#0369a1` | Interactive button hover |
| `--accent-subtle` | `rgba(56, 189, 248, 0.12)` | `rgba(2, 132, 199, 0.08)` | Selected badge backgrounds, tab indicator |
| `--accent-contrast`| `#090d16` | `#ffffff` | Text on solid accent backgrounds |

#### Feedback Status Tokens
| State | Background | Border | Text |
| :--- | :--- | :--- | :--- |
| **Success** | `rgba(16, 185, 129, 0.15)` | `#059669` | `#34d399` |
| **Warning** | `rgba(245, 158, 11, 0.15)` | `#d97706` | `#fbbf24` |
| **Danger** | `rgba(239, 68, 68, 0.15)` | `#dc2626` | `#f87171` |

#### Emotion Categorical Palette (MELD 7-Class Standard)
| Emotion Class | Dark Color | Light Color | Dark Tint BG | Light Tint BG |
| :--- | :--- | :--- | :--- | :--- |
| **Anger** | `#ef4444` | `#dc2626` | `rgba(239, 68, 68, 0.14)` | `#fef2f2` |
| **Disgust** | `#a855f7` | `#9333ea` | `rgba(168, 85, 247, 0.14)`| `#faf5ff` |
| **Fear** | `#f59e0b` | `#d97706` | `rgba(245, 158, 11, 0.14)`| `#fffbeb` |
| **Joy** | `#10b981` | `#059669` | `rgba(16, 185, 129, 0.14)`| `#ecfdf5` |
| **Neutral** | `#94a3b8` | `#64748b` | `rgba(148, 163, 184, 0.14)`| `#f1f5f9` |
| **Sadness** | `#6366f1` | `#4f46e5` | `rgba(99, 102, 241, 0.14)`| `#eef2ff` |
| **Surprise** | `#06b6d4` | `#0284c7` | `rgba(6, 182, 212, 0.14)` | `#f0f9ff` |

---

### 2.2 Typography Scale

**Font Families:**
- **Primary Body & Headings:** `'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif`
- **Quantitative & Code Elements:** `'JetBrains Mono', ui-monospace, monospace`

| Level | Size | Line Height | Weight | Tracking | Usage |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `display-1` | `28px (1.75rem)` | `1.25` | `700` | `-0.025em` | Application Brand Title, Major Hero |
| `h1` | `22px (1.375rem)`| `1.3` | `700` | `-0.02em` | Screen Main Headings |
| `h2` | `18px (1.125rem)`| `1.35` | `600` | `-0.015em` | Section Titles, Panel Headers |
| `h3` | `15px (0.9375rem)`| `1.4` | `600` | `-0.01em` | Card Subheadings, Modal Titles |
| `body-base` | `14px (0.875rem)` | `1.5` | `400 / 500`| `0` | Standard Dialogue Utterances, Body Copy |
| `body-sm` | `13px (0.8125rem)`| `1.45` | `400 / 500`| `+0.005em` | Table Content, Descriptive Labels |
| `caption` | `11.5px (0.72rem)`| `1.4` | `500 / 600`| `+0.02em` | Badges, Timestamp Meta, KPI Subtitles |
| `mono-data` | `12.5px (0.78rem)`| `1.4` | `500` | `0` | Turn IDs, Probability Percentages, Stats |

---

### 2.3 Spacing Scale
ConvoSense uses a strict 4px baseline rhythm:
- `--space-1`: `4px` (Badge padding, micro gaps)
- `--space-2`: `8px` (Element margins, input internal padding)
- `--space-3`: `12px` (Standard component gutters, card padding)
- `--space-4`: `16px` (Panel padding, stream spacing)
- `--space-5`: `20px` (Section gaps)
- `--space-6`: `24px` (Major container padding)
- `--space-8`: `32px` (Screen section divisions)
- `--space-10`: `40px` (Macro screen spacing)
- `--space-12`: `48px` (Empty state padding)

---

### 2.4 Corner Radius System
- `--radius-xs`: `4px` (Tag badges, code tokens)
- `--radius-sm`: `6px` (Buttons, input controls)
- `--radius-md`: `10px` (Conversation bubbles, secondary panels)
- `--radius-lg`: `14px` (Primary application panels, modals)
- `--radius-xl`: `18px` (Dropzone containers)
- `--radius-full`: `9999px` (Pills, status dots, turn chips)

---

## 3. Reusable Component Specifications

### 3.1 Application Header & Navigation Bar
- **Brand Emblem:** Inline geometric SVG icon representing speech analysis (two overlapping conversational soundwave contours) paired with bold `ConvoSense` logotype.
- **System Liveness Indicator:** Real-time backend status pill displaying green pulse dot with `3 Models Online`.
- **Top-Level Navigation Tabs:** Five horizontal pill-style tabs:
  1. `Import` (Upload icon)
  2. `Conversation Viewer` (Chat bubble icon + dynamic count badge)
  3. `Emotion Dynamics` (Line chart trend icon)
  4. `Model Comparison` (Scale/balance icon)
  5. `Research & Benchmarks` (Microscope icon)
  No emojis. Active tab receives solid accent-subtle background with glowing text indicator.

### 3.2 Button Variants
- **Primary Action (`.btn-primary`):** Solid `--accent-primary` background, high-contrast text, 6px radius, subtle active scaling (`transform: scale(0.98)`).
- **Secondary Action (`.btn-secondary`):** Transparent surface with `--border-default`, `--text-primary` text, hover fill to `--bg-surface-3`.
- **Danger Action (`.btn-danger`):** Red boundary and subtle red tint for destructive actions (e.g. Session Clear).
- **Ghost Action (`.btn-ghost`):** Borderless button for inline utilities (drawer toggling, copy text).

### 3.3 Conversation Viewer Speech Bubbles
- **Speaker Distinction:** 
  - `Speaker A`: Left-aligned with subtle blue-tinted border indicator.
  - `Speaker B`: Right-aligned with emerald-tinted border indicator.
  - `Speaker C+`: Inset neutral alignment.
- **Emotion Pill:** Displays categorical emotion with a crisp status dot and predicted emotion text.
- **Confidence Rating:** Displays uncalibrated softmax score with informational tooltip (`Uncalibrated logit score; does not indicate psychological certainty`).
- **Expandable Probability Drawer:** Clean, toggleable breakdown displaying all 7 class probability progress bars.

### 3.4 Emotion Dynamics Timeline Graphic
- **Summary KPI Tiles:** 4 primary metrics in a compact grid: Total Transitions, Emotion Shifts, Emotional Persistence, Shift Rate (%).
- **Discrete Sequence Ribbon:** Interactive chronological sequence of turn pills colored by predicted emotion. Clicking any pill smoothly navigates to and highlights that turn in the viewer.
- **Chart.js Probability Timeline:** Clean stacked area visualization of the 7 emotion trajectories across conversation progression.
- **Turning Points Inspector:** Highlighted shift events showing the preceding emotion, trigger turn text, and target turn navigation CTA.

### 3.5 Model Comparison Matrix & Synchronized Cards
- **Consensus Rate Callout:** Unanimous agreement metric across DistilBERT, DialogueRNN, and DialogueGCN.
- **Pairwise Agreement Heatmap Table:** Compact 3x3 table with subtle opacity shading representing agreement rates between model pairs.
- **Synchronized Turn Stream:** Unified card per utterance showing text alongside side-by-side model predictions, highlighted with `✓ Unanimous Agreement` or `⚠️ Model Disagreement`.

### 3.6 Research & Benchmark Inspector
- **Cohort A Specification Card:** Highlights Matched Subset (N=2,328), parameter counts, and receptive fields.
- **Per-Class Breakdown:** Precision, Recall, F1-Score, and Support per emotion class.
- **Interactive Confusion Matrix:** Clean 7x7 matrix with diagonal saturation shading, selectable between models.
- **Significance Tests:** McNemar and permutation test p-values comparing DialogueGCN against baselines.
- **Methodological Boundaries Callout:** Explicit non-clinical warning and unimodal textual limitations.

---

## 4. State System

### 4.1 Loading & Progress State
- Full-viewport blurred backdrop overlay.
- Dual-ring SVG spinner in primary accent color.
- Action-specific live status message:
  - `"Tokenizing conversation sequence..."`
  - `"Extracting DistilBERT contextual embeddings..."`
  - `"Executing DialogueGCN graph convolution..."`
  - `"Calculating multi-turn emotional dynamics..."`

### 4.2 Empty States
- Custom illustrated SVG graphics for empty screens.
- Screen 2 (Viewer): `"No Conversation Loaded"` with one-click button to load sample or open import tab.
- Screen 3 (Dynamics): `"No Dynamics Computed"` explaining that model analysis must run first.
- Screen 4 (Comparison): `"Multi-Model Comparison Pending"` explaining the 3-model agreement protocol.

### 4.3 Error State
- Prominent non-intrusive alert banner with warning icon, actionable explanation (e.g. line number of formatting issue), and a "Dismiss" or "Retry" action.
