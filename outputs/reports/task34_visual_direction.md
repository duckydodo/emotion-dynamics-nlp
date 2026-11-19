# Emotion Dynamics — Visual Identity & Design Direction (Task 34)

**Product**: Emotion Dynamics  
**Subtitle**: Conversational Emotion Analysis and Research  
**Document Status**: Approved Specification  
**Scope**: Visual Identity, Design Tokens, and Home / Import Screen Redesign  

---

## 1. Executive Summary & Philosophy

The goal of Task 34 is to move **Emotion Dynamics** away from the ubiquitous "AI SaaS dashboard" aesthetic (characterized by dark navy backgrounds, electric cyan accents, floating glassmorphism cards, and decorative gradient meshes) toward a **restrained, scholarly, and professionally crafted computational research application**.

The application serves natural language processing researchers, computational linguists, and conversation analysts evaluating multi-party dialogues. The design reflects this scientific purpose:
- **Restraint over spectacle**: Elimination of glowing borders, electric cyan highlights, and excessive decorative badges.
- **Warm, natural materiality**: A grounded palette of warm charcoal (`#171B18`), muted sage (`#91A58B`), and restrained copper (`#C18A68`).
- **High typographic clarity**: Distinct hierarchy prioritizing conversation ingestion, clear model selection, and unambiguous data preview.
- **Zero generic branding**: Replaces startup-style neologisms ("ConvoSense") with the direct descriptive title **Emotion Dynamics** and subtitle *Conversational Emotion Analysis and Research*.

---

## 2. Analysis of Task 33 Design Choices: Retained vs. Replaced

| Area | Task 33 Implementation | Task 34 Direction | Rationale |
|---|---|---|---|
| **Product Branding** | "ConvoSense" with acoustic wave emblem | **Emotion Dynamics** with subtitle *Conversational Emotion Analysis and Research* | Unpretentious, descriptive research identity without commercial SaaS connotations. |
| **Color Theme** | Deep navy (`#090d16`) with electric cyan (`#38bdf8`) | **Warm charcoal (`#171B18`)** with **muted sage (`#91A58B`)** and **restrained copper (`#C18A68`)** | Reduces eye fatigue; creates an understated, natural research aesthetic while exceeding WCAG AAA contrast. |
| **Card Architecture** | Disjointed series of separate cards for sample bar, upload, paste, model selection, preview | **Unified Ingestion Workbench**: Single cohesive workspace using whitespace, subtle dividers, and typography instead of card walls | Improves visual flow and removes unnecessary container noise. |
| **Sample Dialogue Controls** | Prominent button bar across the top of the screen | **Discreet sample preset selector**: Compact, subtle pills placed contextually near the input workspace | Reduces distraction; sample inputs are secondary utilities, not the primary hero action. |
| **Model Selection** | Large separate card with oversized dropdown | **Integrated Architecture Selector**: Compact, contextual selector with concise, clear model descriptors | Aligns model choice directly with the input workflow before execution. |
| **Ingestion Preview** | 5 bulky stat cards + alert box | **Streamlined Ingestion Inspector**: Compact metadata strip with clean tabular alignment and copper warning callouts | Immediate clarity on sequence validity and token counts without visual bloat. |
| **Iconography** | Vector SVGs (improved from emojis in Task 33) | **Retained**: Crisp, thin-stroke inline SVGs with restrained stroke colors (`#9AA897`) | Professional, scalable, zero emoji clutter. |
| **Emotion Categories** | Standard 7-class MELD color codings | **Retained**: Category colors (anger, disgust, fear, joy, neutral, sadness, surprise) remain distinguishable | Preserves scientific visualization continuity across models. |

---

## 3. Revised Design Token System

### 3.1 Color Palette (Dark Theme — Default)

```css
:root {
    /* Base Surfaces */
    --bg-base: #171B18;             /* Warm charcoal background */
    --bg-surface-1: #222923;        /* Primary surface (panels, inputs) */
    --bg-surface-2: #2A322C;        /* Elevated surface (hover, active tabs) */
    --bg-surface-3: #343E36;        /* Subtle interactive fill */

    /* Neutral Borders */
    --border-subtle: #252D26;       /* Faint separator */
    --border-default: #354238;      /* Primary structural border (neutral grey-green) */
    --border-strong: #4B5C4F;       /* Focused / active border */

    /* Typography */
    --text-primary: #E8E5DB;        /* Warm off-white (primary headings & body) */
    --text-secondary: #9AA897;      /* Muted grey-green (labels, subtitles) */
    --text-tertiary: #6E7C6C;       /* Low-emphasis captions, metadata */

    /* Brand Accents */
    --accent-primary: #91A58B;      /* Muted sage */
    --accent-hover: #A2B69C;        /* Sage hover */
    --accent-subtle: rgba(145, 165, 139, 0.12); /* Sage background tint */
    --accent-contrast: #171B18;    /* Dark charcoal text on sage buttons */

    /* Secondary Highlight & Alerts */
    --highlight-secondary: #C18A68; /* Restrained copper */
    --highlight-subtle: rgba(193, 138, 104, 0.12);

    --success-bg: rgba(94, 173, 117, 0.12);
    --success-border: #4D8B60;
    --success-text: #74C68B;

    --warning-bg: rgba(193, 138, 104, 0.14);
    --warning-border: #A6704F;
    --warning-text: #DBA888;

    --danger-bg: rgba(224, 93, 82, 0.12);
    --danger-border: #AA473E;
    --danger-text: #F08E85;

    /* Categorical Emotion Palette (MELD 7-Class Standard) */
    --emo-anger: #E05D52;
    --emo-anger-bg: rgba(224, 93, 82, 0.14);
    --emo-anger-text: #F7A8A1;

    --emo-disgust: #9E67B8;
    --emo-disgust-bg: rgba(158, 103, 184, 0.14);
    --emo-disgust-text: #D2A8E5;

    --emo-fear: #D48D48;
    --emo-fear-bg: rgba(212, 141, 72, 0.14);
    --emo-fear-text: #ECC397;

    --emo-joy: #5EAD75;
    --emo-joy-bg: rgba(94, 173, 117, 0.14);
    --emo-joy-text: #96DDB0;

    --emo-neutral: #8C9B90;
    --emo-neutral-bg: rgba(140, 155, 144, 0.14);
    --emo-neutral-text: #C4CEC7;

    --emo-sadness: #6886C5;
    --emo-sadness-bg: rgba(104, 134, 197, 0.14);
    --emo-sadness-text: #A8C0F0;

    --emo-surprise: #49A6BA;
    --emo-surprise-bg: rgba(73, 166, 186, 0.14);
    --emo-surprise-text: #97DCEB;
}
```

### 3.2 Typography Scale

- **Primary Font**: `Plus Jakarta Sans`, `-apple-system`, `BlinkMacSystemFont`, `Segoe UI`, `sans-serif`
- **Monospace Font**: `JetBrains Mono`, `Consolas`, `Menlo`, `monospace`

| Role | Font Family | Size | Weight | Line Height | Tracking |
|---|---|---|---|---|---|
| **App Title** | Plus Jakarta Sans | 19px | 600 | 1.2 | -0.01em |
| **Subtitle / Tagline** | Plus Jakarta Sans | 12.5px | 400 | 1.3 | 0 |
| **Page Title** | Plus Jakarta Sans | 22px | 600 | 1.25 | -0.02em |
| **Section Title** | Plus Jakarta Sans | 14.5px | 600 | 1.3 | -0.01em |
| **Body Primary** | Plus Jakarta Sans | 13.5px | 400 | 1.55 | 0 |
| **Secondary / Labels** | Plus Jakarta Sans | 12.5px | 500 | 1.4 | 0.01em |
| **Small Utility / Pills** | Plus Jakarta Sans | 11.5px | 500 | 1.2 | 0.02em |
| **Quantitative Data / Code** | JetBrains Mono | 12.5px | 500 | 1.4 | 0 |

---

## 4. Redesigned Home / Import Screen Structure

The Home / Import screen has been reorganized into a unified, high-hierarchy workspace:

1. **Screen Intro Header**:
   - Title: `Conversation Ingestion & Sequence Analysis`
   - Description: Concise statement explaining that transcripts can be imported via file or direct text to run single-model inference or 3-model comparison.
2. **Preset Dialogue Strip**:
   - Discreet horizontal pill selector: `Load sample: [Conflict & Resolution] [WhatsApp export] [Instagram JSON] [Three-party meeting]`.
3. **Dual Ingestion Workbench**:
   - Two balanced, cleanly bordered columns without nested sub-cards:
     - **File Drag & Drop Area**: Restrained upload icon, dropzone with clean `#354238` dashed border, format tags (`Speaker: text`, `WhatsApp .txt`, `Instagram JSON`, `Structured CSV`).
     - **Direct Transcript Paste**: Native textarea with placeholder dialogue, JetBrains Mono character/line counter.
4. **Sequence Inference Model Selector**:
   - Streamlined horizontal bar directly underneath the input columns.
   - Dropdown offering strictly:
     - `DialogueGCN (Relational Graph Convolutional Network) [Default]`
     - `DialogueRNN (Causal Recurrent State Flow)`
     - `DistilBERT (Utterance-Only Transformer Baseline)`
   - Concise 1-line architecture helper note in muted grey-green.
5. **Ingestion Preview & Validation (Dynamic)**:
   - Activates automatically upon conversation input.
   - Displays detected format, message count, evaluable turns count, and speaker identities.
   - Non-evaluable items or timestamp notices rendered in restrained copper alert callouts.
   - Dual action CTAs:
     - **Analyze Conversation Dynamics** (Muted Sage `#91A58B` fill, `#171B18` text).
     - **Compare All 3 Models** (Neutral elevated `#2A322C` fill, `#E8E5DB` text).

---

## 5. Scope & Preservation Statement

As instructed, this redesign is strictly focused on:
- Global branding, header title, favicon, and root design tokens.
- **Screen 1 (Home / Import)** layout, components, and responsive behavior.
- Retaining functional compatibility for Screens 2–5 without altering their inner structures.
- Retaining all API routes (`/health`, `/predict`, `/analyze`, `/compare`, `/models`), all DOM identifiers, and the full test suite.
