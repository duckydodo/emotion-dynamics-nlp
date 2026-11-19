# Hybrid Conversational Emotion Analysis Platform: Architecture & Demo-Readiness Audit

**Repository:** `emotion-dynamics-nlp`  
**Working Directory:** `/home/chitta/Projects/emotion-dynamics-nlp`  
**Date:** November 10, 2026  
**Status:** Audit Completed — Pre-Implementation Planning Phase  
**Task Reference:** Task 16 (Hybrid Application Architecture and Demo-Readiness Audit)  
**Deliverable File:** [`outputs/reports/hybrid_application_readiness_audit.md`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/reports/hybrid_application_readiness_audit.md)  

---

## 1. Executive Summary & Readiness Verdict

This audit evaluates the readiness of the `emotion-dynamics-nlp` codebase and its FastAPI + web application for transformation into a **hybrid conversational emotion analysis platform**. 

The targeted application will expose three primary model families:
1. **DistilBERT:** Utterance-level emotion classification (fine-tuned transformer baseline).
2. **DialogueRNN:** Recurrent conversational context and speaker-state modeling (Majumder et al., AAAI 2019).
3. **DialogueGCN:** Relational graph convolutional context modeling (Ghosal et al., EMNLP 2019).

The application will feature five user-facing views:
1. **Home / Conversation Import** (Multi-format ingestion: pasted text, Instagram JSON, WhatsApp, CSV).
2. **Conversation Viewer** (Interactive chat-bubble dialogue viewer with emotion tags).
3. **Emotion Dynamics** (Trajectory charts, shift/inertia metrics, turning points).
4. **Model Comparison** (Side-by-side multi-model prediction and divergence analysis).
5. **Research / Model Inspection** (Interactive exploration of verified benchmark tables, ablations, and error analysis).

### Readiness Verdict: **PARTIALLY READY — ARCHITECTURAL FOUNDATIONS SOLID; TWO MAJOR GAPS**
- **Existing Strengths:** The backend core ([`src/api.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/api.py), [`src/application/analyzer.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/application/analyzer.py), [`src/models/inference.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/models/inference.py)), Docker containerization, dynamic metrics calculation, and verified DistilBERT and DialogueRNN checkpoints are robust, tested, and fully functional.
- **Gap 1 (DialogueGCN):** **Missing.** No trained checkpoint exists on disk for DialogueGCN. DialogueGCN exists only as an unverified, non-causal benchmarking skeleton in `src/benchmarking/models.py`. It requires causal architectural formalization, training pipeline implementation, checkpoint training on MELD, and canonical Cohort A evaluation.
- **Gap 2 (Multi-Format Ingestion & Multi-Screen UI):** **Partially Implemented.** The current frontend is a monolithic single-page scroll view limited to plain text `Speaker: utterance` pasted in a `<textarea>`. It lacks Instagram JSON and WhatsApp parsers, multi-screen navigation, chat-bubble conversation rendering, and multi-model comparison views.

---

## 2. Current Application Architecture

The existing application is structured as a lightweight Python/FastAPI service with a monolithic static HTML/JS frontend:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        CLIENT / WEB FRONTEND                           │
│ static/index.html (Vanilla JS, CSS3, Chart.js CDN)                     │
│ Single-page layout: Model Selector, Textarea, Trajectory, Turn Table   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP JSON Requests
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                       FASTAPI APPLICATION LAYER                        │
│ src/api.py                                                             │
│ ├── GET  /                 -> Serves static/index.html                 │
│ ├── GET  /health           -> Returns server status and modes          │
│ ├── POST /predict          -> Lightweight turn-level prediction        │
│ └── POST /analyze          -> End-to-end emotion & dynamics analysis   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    APPLICATION SERVICE & INFERENCE                     │
│ src/application/analyzer.py (ConversationAnalyzer)                    │
│ ├── Parser: src/data/conversation.py (parse_conversation)              │
│ ├── Dynamics: src/analysis/conversation_analysis.py                   │
│ └── Inference Orchestrator: src/models/inference.py                    │
│     ├── EmotionInference (DistilBERT: utterance, context_1)            │
│     └── DialogueRNNInference (Frozen DistilBERT CLS + DialogueRNN)    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        MODEL CHECKPOINTS ON DISK                       │
│ checkpoints/                                                           │
│ ├── transformer_baseline/best/   (Fine-tuned DistilBERT)               │
│ ├── transformer_context_1/best/  (Context-1 DistilBERT)                │
│ ├── dialoguernn_unweighted/      (Best Model Pt, Epoch 2)              │
│ └── ablations/                   (Ablations 1-4 Best Checkpoints)      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Existing Frontend Feature Inventory

**Source File:** [`static/index.html`](file:///home/chitta/Projects/emotion-dynamics-nlp/static/index.html) (2,521 lines, ~52.5 KB).

| Component / Feature | Description | Status | Reuse Assessment |
| :--- | :--- | :--- | :--- |
| **Theme System** | Dark mode (default) and Light mode toggle with CSS custom properties and `localStorage` persistence. | `Existing and verified` | **High Reuse:** Retain CSS variable design tokens (`--bg`, `--surface`, `--primary`, etc.). |
| **Model Dropdown** | `<select id="mode">` with 4 options: `utterance`, `context_1`, `context_1_speaker`, `dialoguernn`. | `Existing and verified` | **Redesign Needed:** Must be re-scoped to the 3 canonical model families (DistilBERT, DialogueRNN, DialogueGCN) plus an "All Models (Compare)" mode. |
| **Pasted Input Area** | `<textarea id="conversation">` expecting line-by-line `Speaker: utterance`. | `Existing and verified` | **Partial Reuse:** Keep as the manual paste option on Screen 1; add multi-format tabs and file drop zone. |
| **Load Example Button** | Pre-fills a 7-turn Alice/Bob sample dialogue. | `Existing and verified` | **High Reuse:** Retain; expand with multi-party (3+ speaker) example dialogues. |
| **Error Banner** | Dismissible notification container rendering API error messages. | `Existing and verified` | **High Reuse:** Directly reusable across all screens. |
| **Dynamics KPI Tiles** | Four metric cards: Total Transitions, Emotion Shifts, Persistence, Shift Rate. | `Existing and verified` | **High Reuse:** Transfer directly to Screen 3 (Emotion Dynamics). |
| **Probability Trajectory** | Chart.js stacked area line chart displaying 7-class probability flow across turns. | `Existing and verified` | **High Reuse:** Transfer to Screen 3; enhance with speaker-filtering and turn-scrubbing. |
| **Predicted Emotion Sequence** | Horizontal pill sequence displaying turn number, speaker, and emotion badge. | `Existing and verified` | **High Reuse:** Transfer to Screen 2 (Viewer) and Screen 3 (Dynamics). |
| **Distribution Bars** | Horizontal CSS percentage bar chart showing global emotion proportions. | `Existing and verified` | **High Reuse:** Transfer to Screen 3. |
| **Turn-by-Turn List** | Linear 4-column table rows: Speaker, Emotion badge, Utterance text, Confidence %. | `Existing and verified` | **Redesign Needed:** Replace with authentic chat-bubble dialogue viewer on Screen 2. |
| **Speaker Dynamics Tiles** | Metrics for Same-speaker vs Cross-speaker transitions and shifts. | `Existing and verified` | **High Reuse:** Transfer to Screen 3. |
| **Turning Points Section** | Lists detected 2-turn persistent emotional direction shifts. | `Existing and verified` | **High Reuse:** Transfer to Screen 3. |
| **Multi-Screen Navigation** | Tabbed or routed view switching between the 5 target screens. | `Missing` | **New Build:** Implement client-side tab navigation (Home/Import, Viewer, Dynamics, Comparison, Research). |
| **Instagram JSON Upload** | Drag-and-drop or file upload for Instagram `messages.json` exports. | `Missing` | **New Build:** Add file drop zone, JSON validation, and preview. |
| **Model Comparison View** | Side-by-side or overlaid multi-model comparison. | `Missing` | **New Build:** Implement Screen 4 with delta trajectories and disagreement matrix. |
| **Research Inspection View** | In-app exploration of canonical benchmark tables, ablations, and confusion matrices. | `Missing` | **New Build:** Implement Screen 5 to surface completed research artifacts. |

---

## 4. Existing Backend Endpoint Inventory

**Source File:** [`src/api.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/api.py)

| Endpoint | HTTP Method | Request Payload | Response Schema | Status | Action Required |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `/` | `GET` | None | HTML FileResponse (`static/index.html`) | `Existing and verified` | Retain as entry point for the single-page application. |
| `/health` | `GET` | None | `{"status": "ok", "available_modes": list}` | `Existing and verified` | Expand to report model status (loaded in memory vs available on disk). |
| `/analyze` | `POST` | `ConversationRequest(conversation, mode)` | Full payload: `turns`, `model_metadata`, `analysis` (distribution, transitions, shifts, speaker dynamics, turning points) | `Existing and verified` | Expand to accept `mode="all"` for multi-model comparison; accept structured turns directly. |
| `/predict` | `POST` | `ConversationRequest(conversation, mode)` | Lightweight payload: `mode`, `model_metadata`, `turns` (predictions and probabilities) | `Existing and verified` | Retain for fast prediction without analytics overhead. |
| `/import` | `POST` | Multipart/Form or JSON | Normalized `List[ConversationTurn]` | `Missing` | **New Endpoint:** Implement conversation parser API supporting plain text, Instagram JSON, WhatsApp, and CSV. |
| `/compare` | `POST` | `ConversationRequest(conversation, models)` | Multi-model aligned predictions, disagreement rate, per-model dynamics | `Missing` | **New Endpoint:** Execute all three models concurrently and compute comparative agreement. |
| `/research` | `GET` | Optional query params | Canonical benchmark tables, ablation comparisons, confusion matrices | `Missing` | **New Endpoint:** Serve pre-computed research artifacts (`outputs/tables/*.csv`). |

---

## 5. Verified Model and Checkpoint Inventory

| Model Family / Variant | Checkpoint Path | Architecture & Parameters | Trained on MELD? | Verified Evaluation Cohort | Verified Macro F1 | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **1. DistilBERT Baseline** | `checkpoints/transformer_baseline/best` | Fine-tuned DistilBERT ($66.96\text{M}$ params) | **YES** | Cohort A ($N=2,328$) | **0.4314** | `Existing and verified` |
| **2. DistilBERT Context-1** | `checkpoints/transformer_context_1/best` | Fine-tuned DistilBERT ($66.96\text{M}$ params) | **YES** | Cohort A ($N=2,328$) | **0.3912** | `Existing and verified` |
| **3. DistilBERT Context-1 + Spk** | `checkpoints/transformer_context_1_speaker/best` | Fine-tuned DistilBERT ($66.96\text{M}$ params) | **YES** | Cohort A ($N=2,328$) | **0.4078** | `Existing and verified` |
| **4. DialogueRNN (Unweighted)** | `checkpoints/dialoguernn_unweighted/best_model.pt` | Causal DialogueRNN ($670,855$ params) | **YES** | Cohort A ($N=2,328$) | **0.4342** | `Existing and verified` |
| **5. DialogueRNN (Weighted)** | `checkpoints/dialoguernn_weighted/best_model.pt` | Causal DialogueRNN ($670,855$ params) | **YES** | Cohort A ($N=2,328$) | **0.4284** | `Existing and verified` |
| **6. DialogueRNN Ablation 1** | `checkpoints/ablations/1_global_context_only/best_model.pt` | Global Context Only ($370,695$ params) | **YES** | Cohort A ($N=2,328$) | **0.4179** | `Existing and verified` |
| **7. DialogueRNN Ablation 2** | `checkpoints/ablations/2_global_party_states/best_model.pt` | Global + Party States ($617,223$ params) | **YES** | Cohort A ($N=2,328$) | **0.4352** | `Existing and verified` |
| **8. DialogueRNN Ablation 3** | `checkpoints/ablations/3_global_emotion_state/best_model.pt` | Global + Emotion State ($452,999$ params) | **YES** | Cohort A ($N=2,328$) | **0.4325** | `Existing and verified` |
| **9. DialogueRNN Ablation 4** | `checkpoints/ablations/4_full_dialoguernn/best_model.pt` | Full DialogueRNN ($670,855$ params) | **YES** | Cohort A ($N=2,328$) | **0.4342** | `Existing and verified` |
| **10. DialogueGCN** | *None on disk* | GCN Skeleton in `src/benchmarking/models.py` | **NO** | *None* (Benchmark placeholder only) | *None* | `Missing` |

> [!CAUTION]
> **DialogueGCN Status Warning:**  
> DialogueGCN does **not** exist as a trained model in this repository. The class in `src/benchmarking/models.py` is an unverified benchmarking prototype with non-causal BiGRU encoding. It has never been trained on MELD, has no checkpoint on disk, and has no verified predictions. It cannot be used in inference until implemented, trained, and audited.

---

## 6. Conversation Ingestion & Parser Capabilities

**Source File:** [`src/data/conversation.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/data/conversation.py)

| Format / Source | Current Parser Status | Parser Requirements & Failure Modes | Action Needed |
| :--- | :--- | :--- | :--- |
| **Plain Text (`Speaker: utterance`)** | `Existing and verified` | Enforces single colon separation; handles multi-line continuations; ignores blank lines. Rejects empty lines before first speaker. | Retain as default paste format. |
| **Unformatted Pasted Text** | `Missing` | Fails with `ConversationParseError: expected 'Speaker: utterance'`. | Implement heuristic fallback assigning turns to `User` or defaulting speaker. |
| **Instagram JSON Export** | `Missing` | **Requirements:**<br>1. JSON schema: `{"participants": [...], "messages": [...]}`.<br>2. **Reverse Chronological Sort:** Instagram exports newest message first; parser must sort ascending by `timestamp_ms`.<br>3. **Mojibake Encoding Fix:** Instagram exports Latin-1 encoded UTF-8 bytes (e.g. `\u00c3\u00a9` $\to$ `é`); requires `content.encode('latin1').decode('utf-8')`.<br>4. **Non-Text Filtering:** Skip/placeholder media turns (`photos`, `audio_files`, `reactions`).<br>5. Extract `sender_name` as `speaker` and `content` as `text`. | Implement dedicated `parse_instagram_json(data)` parser. |
| **WhatsApp Chat Export (.txt)** | `Missing` | **Requirements:** Match timestamp regex: `^\[?(\d{1,2}/\d{1,2}/\d{2,4},?\s\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?)\]?\s-?\s?([^:]+):\s(.+)$`. Handle multiline messages. | Implement `parse_whatsapp_text(raw_text)` parser. |
| **CSV / Tabular Export** | `Missing` | **Requirements:** Auto-detect columns (`speaker`, `text` / `utterance`, optional `timestamp`). | Implement `parse_csv_conversation(file_or_text)` parser. |

---

## 7. Docker & Deployment Status

**Source Files:** [`Dockerfile`](file:///home/chitta/Projects/emotion-dynamics-nlp/Dockerfile) and [`compose.yaml`](file:///home/chitta/Projects/emotion-dynamics-nlp/compose.yaml)

```
Dockerfile Audit:
- Base Image: python:3.12-slim (Clean, minimal base)
- PyTorch: CPU-only wheel (torch==2.13.0) via download.pytorch.org (Prevents CUDA bloat, keeps image ~1.5 GB)
- Dependencies: Installed via requirements-api.txt
- Exposed Port: 8000
- Server: uvicorn src.api:app --host 0.0.0.0 --port 8000

compose.yaml Audit:
- Image: emotion-dynamics-nlp:week11
- Port Mapping: 8001:8000 (Maps host 8001 to container 8000)
- Volumes: ./checkpoints:/app/checkpoints:ro
```

### Identified Deployment Limitations:
1. **Missing Volume Mounts:** Only `./checkpoints` is mounted. The directory `outputs/` (which contains all benchmark tables, research reports, and evaluation figures) is **not mounted**. The Research view on the frontend will need `outputs/` mounted or embedded.
2. **Static Asset Organization:** Currently, `src/api.py` only serves `static/index.html` on `GET /`. If the frontend is refactored into modular JS/CSS files, FastAPI must mount `app.mount("/static", StaticFiles(directory="static"), name="static")`.
3. **Container Healthcheck:** `compose.yaml` lacks a `healthcheck` definition targeting `http://localhost:8000/health`.
4. **Memory Sizing:** Hosting DistilBERT (~500 MB RAM) + DialogueRNN (~200 MB RAM) + future DialogueGCN (~300 MB RAM) requires at least 1.5–2.0 GB container RAM.

---

## 8. Feature Reuse Matrix

| Asset / Subsystem | Current State | Reusability | Target Role in Hybrid Platform |
| :--- | :--- | :---: | :--- |
| **CSS Theme & Design Tokens** | `static/index.html` lines 18–66 | **100% Reusable** | Provides the visual identity across all 5 screens (dark/light mode). |
| **Chart.js Trajectory Engine** | `static/index.html` lines 1759–2096 | **90% Reusable** | Primary visualization for Screen 3 (Emotion Dynamics). |
| **Turning Point Detector** | `src/analysis/conversation_analysis.py` | **100% Reusable** | Core analytics service for dynamics tracking. |
| **Summary Stats & Transition Engine** | `src/analysis/conversation_analysis.py` | **100% Reusable** | Feeds Screen 3 KPI cards and transition matrices. |
| **FastAPI Core & Error Handling** | `src/api.py` | **90% Reusable** | Base application router; needs endpoint additions. |
| **DistilBERT Inference Wrapper** | `src/models/inference.py` (`EmotionInference`) | **100% Reusable** | Serves Model Family 1 (DistilBERT). |
| **DialogueRNN Inference Wrapper** | `src/models/inference.py` (`DialogueRNNInference`) | **100% Reusable** | Serves Model Family 2 (DialogueRNN). |
| **Plain Text Parser** | `src/data/conversation.py` | **100% Reusable** | Parser for raw text paste in Screen 1. |
| **Ablation Models & Checkpoints** | `src/models/dialoguernn.py` & `checkpoints/ablations/` | **100% Reusable** | Serves technical inspection on Screen 5 (Research). |
| **Single-Page DOM Structure** | `static/index.html` lines 870–1270 | **Requires Redesign** | Must be refactored into 5 distinct screen views with a top navigation bar. |
| **Turn Table View** | `static/index.html` lines 2312–2374 | **Requires Redesign** | Must be replaced with interactive chat bubbles on Screen 2. |

---

## 9. Missing Functionality & Implementation Gaps

### High-Priority Gaps:
1. **DialogueGCN Implementation and Training:**
   - Causal relational graph construction (`window_past`, no future leakage).
   - Training pipeline with frozen DistilBERT features on MELD.
   - Checkpoint selection on Dev Macro F1 and verification on canonical Cohort A ($N=2,328$).
   - Inference wrapper `DialogueGCNInference` in `src/models/inference.py`.
2. **Multi-Format Ingestion Engine (`src/data/parsers/`):**
   - Instagram JSON export parser (`parse_instagram_json`).
   - WhatsApp chat export parser (`parse_whatsapp_text`).
   - CSV tabular dialogue parser (`parse_csv_conversation`).
3. **Frontend Multi-Screen Architecture (5 Screens):**
   - Screen 1: Home / Ingestion (Upload, Paste, Format Selector, Validation Preview).
   - Screen 2: Conversation Viewer (Chat bubbles, speaker coloring, emotion badges, turn scrubber).
   - Screen 3: Emotion Dynamics (Trajectory chart, transition statistics, turning points).
   - Screen 4: Model Comparison (Multi-model prediction alignment, divergence matrix).
   - Screen 5: Research & Model Inspection (Cohort A metrics table, ablation studies, error cases).
4. **Backend Comparison & Research Endpoints:**
   - `/compare`: Runs selected model families on the same dialogue and returns comparative agreement.
   - `/research`: Exposes pre-computed canonical benchmark metrics and ablation tables to the frontend.

---

## 10. Proposed Target Architecture

```
                                HYBRID PLATFORM ARCHITECTURE
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND INTERFACE (5 SCREENS)                            │
│  [Screen 1: Import]    [Screen 2: Chat]    [Screen 3: Dynamics]   [Screen 4: Compare]  [Screen 5: Research] │
│  - Text Paste          - Chat Bubbles      - Trajectory Chart     - 3-Way Prediction   - Benchmark Tables   │
│  - Instagram Upload    - Emotion Pills     - Transition Matrix    - Disagreement Diff  - Ablation Results   │
│  - WhatsApp Upload     - Turn Inspector    - Turning Points       - Trajectory Overlay - Confusion Matrices │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ HTTP REST API
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               FASTAPI BACKEND SERVICE                                  │
│  ├── POST /import    -> Detects format, parses turns, normalizes speakers, validates   │
│  ├── POST /analyze   -> Runs selected model + full conversational dynamics analysis   │
│  ├── POST /compare   -> Executes DistilBERT, DialogueRNN, and DialogueGCN in parallel  │
│  ├── GET  /research  -> Serves verified canonical benchmark data and report metrics    │
│  └── GET  /health    -> Reports model memory footprint and runtime availability        │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                            UNIFIED INFERENCE ORCHESTRATOR                              │
│                                                                                        │
│               Normalized Conversation: List[ConversationTurn(turn_id, speaker, text)]  │
│                                           │                                            │
│                                           ▼                                            │
│                        Frozen DistilBERT Feature Encoder (D=768)                       │
│                                           │                                            │
│                     ┌─────────────────────┼─────────────────────┐                      │
│                     ▼                     ▼                     ▼                      │
│             ┌───────────────┐     ┌───────────────┐     ┌───────────────┐              │
│             │  DistilBERT   │     │  DialogueRNN  │     │  DialogueGCN  │              │
│             │  Utterance    │     │  Recurrent    │     │  Relational   │              │
│             │  Classifier   │     │  Party States │     │  Graph Conv   │              │
│             └───────┬───────┘     └───────┬───────┘     └───────┬───────┘              │
│                     │                     │                     │                      │
│                     └─────────────────────┼─────────────────────┘                      │
│                                           ▼                                            │
│             Unified Turn Predictions: {turn_id, speaker, emotion, confidence, probs}   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 11. Recommended Implementation Order

To maintain stability and verify each layer before adding complexity, the following staged sequence is recommended:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                             RECOMMENDED IMPLEMENTATION ROADMAP                         │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ STAGE 1: INGESTION ENGINE                                                              │
│ 1. Implement and unit test Instagram JSON, WhatsApp, and CSV parsers.                  │
│ 2. Create the /import API endpoint with validation and preview payloads.               │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ STAGE 2: DIALOGUEGCN MODEL DEVELOPMENT                                                 │
│ 3. Implement causal DialogueGCN architecture (relational graph convolution).           │
│ 4. Train DialogueGCN on MELD frozen DistilBERT features (Seed 42, AdamW, Dev F1).      │
│ 5. Evaluate on canonical Cohort A (N=2,328) with paired permutation & bootstrap tests. │
│ 6. Implement DialogueGCNInference wrapper with dynamic graph construction.             │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ STAGE 3: BACKEND API CONSOLIDATION                                                     │
│ 7. Integrate DialogueGCN into ConversationAnalyzer.                                    │
│ 8. Implement the /compare endpoint (side-by-side execution & agreement rate).          │
│ 9. Implement the /research endpoint (serving benchmark tables & ablation metrics).     │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ STAGE 4: FRONTEND REDESIGN (5 SCREENS)                                                 │
│ 10. Implement navigation shell and Screen 1 (Multi-Format Import).                     │
│ 11. Implement Screen 2 (Chat-Bubble Conversation Viewer).                              │
│ 12. Migrate & enhance Screen 3 (Emotion Dynamics & Probability Trajectory).            │
│ 13. Implement Screen 4 (Model Comparison & Disagreement Highlights).                   │
│ 14. Implement Screen 5 (Research & Model Inspection Dashboard).                        │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ STAGE 5: DOCKER & INTEGRATION VERIFICATION                                             │
│ 15. Update Dockerfile, compose.yaml volume mounts, and test full stack end-to-end.     │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 12. Risks and Blockers

1. **DialogueGCN Training & Performance Risk:**  
   DialogueGCN requires graph adjacency construction over conversational turns. If graph convolutions overfit on MELD's imbalanced classes or fail to outperform DistilBERT Context-1, the model comparison may show negligible gains. Strict adherence to Dev Macro F1 selection and canonical Cohort A evaluation is essential.
2. **Instagram Export Schema Variability:**  
   Meta frequently alters data export formats. Older exports use `message_1.json`, while newer exports nest messages under `messages/inbox/...`. Latin-1 encoding bugs (`mojibake`) and missing `content` on media turns must be handled defensively with fallbacks.
3. **Frontend Complexity & Bundle Size:**  
   The current frontend has zero dependencies beyond Chart.js CDN. Rewriting the frontend in a heavy Node/React framework would introduce significant build tooling, Docker complexity, and maintenance overhead. **Recommendation:** Build the 5-screen interface using clean, modular vanilla JavaScript and CSS components, keeping the single-file or lightweight static structure with zero npm build step.
4. **Memory Footprint in Docker:**  
   Loading three PyTorch models concurrently (DistilBERT encoder + DistilBERT head + DialogueRNN + DialogueGCN) will require ~1.5 GB of RAM. The Docker host must have sufficient memory headroom.

---

## 13. Exact Next Implementation Task

**Recommended Next Step: Task 17 — Conversation Ingestion Engine & Parsers**
- Implement `src/data/parsers/instagram.py`, `src/data/parsers/whatsapp.py`, and `src/data/parsers/csv_parser.py`.
- Handle Instagram Latin-1 decoding, chronological sorting, and media filtering.
- Implement comprehensive unit tests under `tests/test_parsers.py`.
- Connect parsers to a new `/import` endpoint in `src/api.py`.
- **Do not train models or alter checkpoints until ingestion is tested.**
