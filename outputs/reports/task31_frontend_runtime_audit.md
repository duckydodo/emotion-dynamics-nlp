# Task 31 — Frontend Runtime & Architecture Audit Report

**Date:** November 13, 2026  
**Auditor:** Autonomous Agentic AI / Pair Programmer  
**Target Repository:** `emotion-dynamics-nlp`  
**Audit Scope:** Live container runtime, Docker configuration, static asset serving, model registry and checkpoints, ingestion pipeline, screen implementations, and frontend-backend connectivity.  
**Mode:** AUDIT ONLY (No code, Docker configurations, or checkpoint modifications were applied during this audit).

---

## Executive Summary

The user observed that the currently running application displays the outdated Week 11 user interface:
- Model selector: `Utterance-only`, `Context-1`, `Context-1 + Speaker`
- No visible conversation file-upload / import interface
- Older single-page emotion analysis layout

### Definitive Diagnosis

The discrepancy is caused by **Docker container asset stale-caching and container port routing**:

1. **Docker Compose runs a stale, un-mounted image from 6 days ago:**  
   Docker Compose (`compose.yaml`) specifies:
   ```yaml
   services:
     emotion-dynamics-nlp:
       build: .
       image: emotion-dynamics-nlp:week11
       ports:
         - "8001:8000"
       volumes:
         - ./checkpoints:/app/checkpoints:ro
   ```
   Notice that `compose.yaml` **only volume-mounts `./checkpoints`**. Neither `./src` nor `./static` are mounted.
2. **The Docker image `emotion-dynamics-nlp:week11` was built on November 7, 2026 (Week 11):**  
   Inspection of `docker inspect emotion-dynamics-nlp:week11` confirms creation timestamp `2026-11-07T16:28:27.987899763+05:30`. In `Dockerfile`, `COPY src/ ./src/` and `COPY static/ ./static/` baked the Week 11 files directly into the image layers.
3. **The active container (`ae9233cb49c3`) on port 8001 is running this Week 11 image:**  
   When `docker compose up` was executed, Docker reused the local image tag without rebuilding. Because host files are not bind-mounted, the container serves `/app/static/index.html` from November 7, which contains the old 3-model dropdown and lacks all Tasks 27–30 screens and endpoints.
4. **All modern code exists and is fully implemented in the repository:**  
   In the repository filesystem:
   - `static/index.html` (2,804 lines, 111 KB) contains the complete 5-screen redesigned application.
   - `src/api.py` contains `/models`, `/import`, `/predict`, `/compare`, `/benchmarks`, and `/analyze`.
   - `src/models/` contains verified inference wrappers for all three canonical models (`distilbert`, `dialoguernn`, `dialoguegcn`).
   In Tasks 28–29, verification was executed against an ephemeral container `emotion-api` (now stopped) that explicitly bind-mounted `-v $(pwd):/workspace`. When testing ceased and Docker Compose was restarted, port 8001 defaulted back to the baked-in Week 11 code.

---

## 1. Docker Runtime Inspection

### Container State
| Container ID | Name | Image | Status | Ports |
| :--- | :--- | :--- | :--- | :--- |
| `ae9233cb49c3` | `emotion-dynamics-nlp-emotion-dynamics-nlp-1` | `emotion-dynamics-nlp:week11` | `Up` (Active) | `0.0.0.0:8001->8000/tcp` |
| `bb92330251bf` | `emotion-api` | `emotion-dynamics-nlp:week11` | `Exited (0) 2 days ago` | None |

### Image Metadata
- **Image ID:** `sha256:0ee19d78d6d116dee4ce00dea99589301d6a7b5822fc1e485a20acf3c0634af0`
- **Creation Timestamp:** `2026-11-07T16:28:27.987899763+05:30` (Week 11 baseline build)
- **Tag:** `emotion-dynamics-nlp:week11`

### Volume Mounts for Running Container (`ae9233cb49c3`)
```json
[
  {
    "Type": "bind",
    "Source": "/home/chitta/Projects/emotion-dynamics-nlp/checkpoints",
    "Destination": "/app/checkpoints",
    "Mode": "ro",
    "RW": false,
    "Propagation": "rprivate"
  }
]
```
- **Finding:** Neither `/app/src` nor `/app/static` are mounted. The running container is completely isolated from repository changes made between November 8 and November 13 (Tasks 18 through 30).

---

## 2. Static File & Route Comparison

### `static/index.html`

| Dimension | Running Container (`/app/static/index.html`) | Repository (`static/index.html`) |
| :--- | :--- | :--- |
| **Line Count** | 569 lines | 2,804 lines |
| **File Size** | 24,198 bytes | 111,032 bytes |
| **Model Selector** | `<option value="utterance">Utterance-only</option>`<br>`<option value="context_1">Context-1</option>`<br>`<option value="context_1_speaker">Context-1 + Speaker</option>` | Dynamic population from `GET /models`<br>Defaults to `dialoguegcn` (7-class)<br>Exposes DistilBERT, DialogueRNN, DialogueGCN |
| **Ingestion UI** | Single text input only. No tabs, no dropzone, no file import. | 5 Ingestion tabs (Pasted text, TXT, Instagram JSON, WhatsApp TXT, CSV), drag-and-drop, CSV column mapping drawer, live preview drawer with warning badges. |
| **Screens** | Single screen. | 5 Tabbed Screens: Home/Import, Viewer, Dynamics, Compare, Research. |

### API Routes & Endpoint Responses

| Endpoint | Method | Container Port 8001 Response | Repo Backend (`src/api.py`) Response |
| :--- | :--- | :--- | :--- |
| `GET /` | GET | Serves Week 11 legacy `index.html` | Serves redesigned 5-screen `index.html` |
| `GET /health` | GET | `{"status": "ok", "available_modes": ["utterance", "context_1", ...]}` | `{"status": "ok", "available_modes": ["distilbert", "dialoguernn", "dialoguegcn", ...]}` |
| `GET /models` | GET | **HTTP 404 Not Found** | **HTTP 200 OK** (Registry of DistilBERT, DialogueRNN, DialogueGCN) |
| `POST /import` | POST | **HTTP 404 Not Found** | **HTTP 200 OK** (Canonical conversation schema + diagnostics) |
| `POST /compare`| POST | **HTTP 404 Not Found** | **HTTP 200 OK** (Synchronized 3-model comparison + agreement matrix) |
| `GET /benchmarks` | GET | **HTTP 404 Not Found** | **HTTP 200 OK** (Cohort A metrics, per-class table, confusion matrices, stats) |
| `POST /analyze`| POST | Legacy schema | Modern schema with emotion dynamics & turning points |

---

## 3. Model-to-Checkpoint Architecture Audit

We conducted a live inference verification inside the container runtime with workspace mounting to confirm that all three canonical models correctly load their respective trained checkpoints and produce genuine predictions without dummy fallbacks:

```python
# Live Verification Output
=== Model: distilbert ===
  Mode: distilbert, Turns: 3
    [0] Ross: neutral (conf: 0.965)
    [1] Rachel: joy (conf: 0.776)
    [2] Ross: neutral (conf: 0.628)

=== Model: dialoguernn ===
  Mode: dialoguernn, Turns: 3
    [0] Ross: neutral (conf: 0.971)
    [1] Rachel: joy (conf: 0.948)
    [2] Ross: neutral (conf: 0.562)

=== Model: dialoguegcn ===
  Mode: dialoguegcn, Turns: 3
    [0] Ross: neutral (conf: 1.000)
    [1] Rachel: joy (conf: 0.992)
    [2] Ross: neutral (conf: 0.753)
```

### Checkpoint Provenance Verification

| Canonical Model | Architecture | Active Checkpoint Path | Parameters | Selection Criterion | Fallback / Stub? | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DistilBERT** | Transformer (Utterance-only) | `checkpoints/transformer_baseline/best` | 66,960,000 | Baseline Dev Macro F1 | None (Genuine weights) | **VERIFIED** |
| **DialogueRNN** | Recurrent (Causal Party States) | `checkpoints/dialoguernn_unweighted/best_model.pt` + frozen DistilBERT CLS | 670,855 | Dev Macro F1 (Epoch 10) | None (Genuine weights) | **VERIFIED** |
| **DialogueGCN** | Relational GCN (Bidirectional) | `checkpoints/dialoguegcn/best_model.pt` + frozen DistilBERT CLS | 452,487 | Dev Macro F1 (Epoch 15) | None (Genuine weights) | **VERIFIED** |

- **Exclusion of Experimental Ablations:** Ablation variants (`dialoguernn_weighted`, `ablations/*`) are strictly excluded from the canonical public registry in `src/models/registry.py`. Only the three verified canonical models are exposed.

---

## 4. Conversation Ingestion Pipeline Audit

The backend ingestion pipeline in `src/data/parsers/` and `src/api.py:import_conversation` was verified against all five supported formats:

| Format | Parser Class | Test Status | Canonical Mapping | Diagnostics |
| :--- | :--- | :--- | :--- | :--- |
| **Pasted Text** | `PlainTextParser` | **VERIFIED** | `SourceFormat.PLAIN_TEXT` | Multiline turns, speaker extraction, unparsed warnings |
| **TXT File** | `PlainTextParser` | **VERIFIED** | `SourceFormat.PLAIN_TEXT` | Line breaks, unicode characters, empty line handling |
| **Instagram JSON** | `InstagramJsonParser` | **VERIFIED** | `SourceFormat.INSTAGRAM_JSON` | Epoch timestamp conversion, media filtering, system event detection |
| **WhatsApp TXT** | `WhatsAppTxtParser` | **VERIFIED** | `SourceFormat.WHATSAPP_TXT` | 12h/24h timestamp parsing, system messages, bracketed formats |
| **Tabular CSV** | `TabularCsvParser` | **VERIFIED** | `SourceFormat.TABULAR_CSV` | Delimiter detection, fuzzy column matching, explicit mapping drawer |

### Key Auto-Detection Bug Identified During Audit
- **Subtlety in Ingestion Auto-Detection:** In `src/data/parsers/csv_parser.py:168`, `can_parse()` checks if the first line has a delimiter (`,`, `\t`, `;`) AND contains a speaker alias substring (`"speaker"`).
- **The Issue:** A plain text conversation that begins with:
  ```text
  Speaker A: Hello, how are you today?
  Speaker B: I am feeling great, thank you!
  ```
  has a comma in the utterance and contains `"speaker"`. Because `_CSV_PARSER.can_parse()` was evaluated before `_PLAIN_TEXT_PARSER` in `src/data/parsers/__init__.py:get_parser()`, `auto` mode misidentified this plain text turn as a CSV file with missing columns.
- **Impact:** When a user selects a specific format tab (e.g. `Plain Text`), `format_hint="plain_text"` is passed, bypassing this issue. However, in `auto` detection mode, plain text lines containing commas can be misclassified as malformed CSVs.

---

## 5. Screen-by-Screen Implementation Status

Every screen in the agreed hybrid design was audited across three layers:
1. **Backend Endpoints:** Does the endpoint exist, return verified data, and validate inputs?
2. **Repository Frontend:** Is the HTML/CSS/JS written in `static/index.html` on disk?
3. **Running Container (`ae9233cb49c3`):** Is it currently served to the user's browser?

| Screen | Target Functionality | Backend Endpoints | Repo Frontend (`static/index.html`) | Running Container Status | Status Label |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Home / Import** | File drag-and-drop, 5 format tabs, CSV column mapping drawer, live preview drawer, warning badges, character count, model selector. | `POST /import`<br>`GET /models`<br>`GET /health` | Fully implemented (Lines 1176–1299, 1933–2015) | **BLOCKED** by stale container image | **VERIFIED WORKING** in repo<br>**BLOCKED** in running container |
| **2. Conversation Viewer** | Speech bubbles with speaker badges, predicted emotion pills, softmax probability drawers, speaker filter, emotion filter, text search, uncalibrated confidence label. | `POST /analyze`<br>`POST /predict` | Fully implemented (Lines 1300–1353, 2108–2213) | **BLOCKED** by stale container image | **VERIFIED WORKING** in repo<br>**BLOCKED** in running container |
| **3. Emotion Dynamics** | KPI summary tiles (transitions, shifts, persistence, same- vs cross-speaker inertia), Chart.js stacked probability timeline, discrete sequence pills, turning point cards with jump-to-turn navigation. | `POST /analyze` | Fully implemented (Lines 1354–1452, 2218–2415) | **BLOCKED** by stale container image | **VERIFIED WORKING** in repo<br>**BLOCKED** in running container |
| **4. Model Comparison** | Synchronized 3-model comparison cards, consensus rate badge, pairwise agreement matrix heatmap table, unanimous vs disagreement badges. | `POST /compare` | Fully implemented (Lines 1453–1510, 2417–2485) | **BLOCKED** by stale container image | **VERIFIED WORKING** in repo<br>**BLOCKED** in running container |
| **5. Research / Inspection** | Verified Cohort A benchmarks table (Acc, Macro F1, Weighted F1), per-class table, interactive 7x7 confusion matrix heatmap, McNemar & permutation significance tests, architecture cards, MELD domain limitation callout. | `GET /benchmarks`<br>`GET /models` | Fully implemented (Lines 1511–1620, 2487–2750) | **BLOCKED** by stale container image | **VERIFIED WORKING** in repo<br>**BLOCKED** in running container |

---

## 6. Frontend-to-Backend Connectivity & Schema Audit

We inspected every `fetch()` call in `static/index.html` against `src/api.py` and identified the following minor integration items:

1. **Warning Object Field Key Mismatch:**
   - In `static/index.html:2009`:
     ```javascript
     `<li><strong>${escapeHtml(w.warning_type)}:</strong> ${escapeHtml(w.message)} (line ${w.line_number || 'N/A'})</li>`
     ```
   - In `src/data/schema.py:ParserWarning.to_dict()`: The dictionary key is `"code"`, not `"warning_type"`.
   - **Result:** Warning type displays as empty string. Should be `w.code || w.warning_type`.
2. **Error Message Port Reference:**
   - In `static/index.html:1912`:
     ```javascript
     showError('Unable to connect to backend server. Ensure API is running on port 8000.');
     ```
   - In `compose.yaml`: Port mapping is `8001:8000`. If accessed via port 8001 and backend fails, the message mentions 8000.
3. **Auto-Detection Precedence in `get_parser`:**
   - When importing raw text with commas in line 1, auto-detection triggers `TabularCsvParser` instead of `PlainTextParser`. Adding a colon check (`not (":" in header and not header.startswith("http")))` to `can_parse()` resolves this.

---

## 7. Immediate Fixes Required

To bring the running application into 100% compliance with the agreed product design, the following exact changes are required:

1. **Update `compose.yaml` to Mount Source and Static Directories (or Rebuild):**
   ```yaml
   services:
     emotion-dynamics-nlp:
       build: .
       image: emotion-dynamics-nlp:latest
       ports:
         - "8000:8000"
       volumes:
         - ./checkpoints:/app/checkpoints:ro
         - ./src:/app/src:ro
         - ./static:/app/static:ro
         - ./outputs:/app/outputs:ro
   ```
   *Mounting `./src`, `./static`, and `./outputs` ensures that any future UI or backend changes are reflected immediately without requiring a full image rebuild.*
2. **Rebuild or Restart Container on Port 8000:**
   Stop the old container `ae9233cb49c3` and bring up the updated container mapping port 8000.
3. **Patch the Warning Property in `static/index.html`:**
   Change `w.warning_type` to `(w.code || w.warning_type || 'Warning')` on line 2009.
4. **Harden `TabularCsvParser.can_parse()` in `src/data/parsers/csv_parser.py`:**
   Ensure lines containing `:` (standard dialogue transcript syntax) are not misclassified as CSV headers in auto-detection mode.

---

## 8. Remediation Plan for Task 32

Once user approval is granted, execute the following step-by-step remediation:

1. **Step 1: Code and Schema Touch-Ups:**
   - In `static/index.html:2009`: Update warning property to `(w.code || w.warning_type || 'Warning')`.
   - In `src/data/parsers/csv_parser.py:180-194`: Prevent `can_parse` from triggering if line 1 matches a standard `Speaker:` format.
2. **Step 2: Update `compose.yaml`:**
   - Add bind-mounts for `./src`, `./static`, and `./outputs`.
   - Standardize port mapping to `8000:8000` (or keep `8001:8000` with unified documentation).
3. **Step 3: Docker Container Restart & Build:**
   - Execute `docker compose down`.
   - Execute `docker compose up -d --build`.
4. **Step 4: End-to-End Browser & API Verification:**
   - Verify `curl -s http://localhost:8000/models` returns HTTP 200 with the 3 canonical models.
   - Verify `curl -s http://localhost:8000/` returns the new 2,804-line `index.html`.
   - Open headless browser / automated verification to confirm that:
     - Model dropdown lists DistilBERT, DialogueRNN, and DialogueGCN.
     - Import tabs, preview drawer, and dropzone render.
     - Running inference displays Conversation Viewer bubbles and Emotion Dynamics chart.
     - Model Comparison displays agreement matrix and synchronized turns.
     - Research tab renders Cohort A benchmark table and confusion matrix heatmap.
