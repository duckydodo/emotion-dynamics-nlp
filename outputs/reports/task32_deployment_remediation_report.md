# Task 32 — Deployment Remediation and Live Application Verification Report

**Date:** November 13, 2026  
**Auditor / Engineer:** Autonomous Agentic AI / Pair Programmer  
**Target Repository:** `emotion-dynamics-nlp`  
**Host Port:** `http://localhost:8001`  
**Deployment Tooling:** Docker Compose (CPU-optimized PyTorch runtime)  
**Status:** **FULLY OPERATIONAL & VERIFIED** (100% Test & Browser Automation Pass Rate)

---

## Executive Summary

In Task 31, an audit revealed that Docker Compose was running a stale image (`emotion-dynamics-nlp:week11`) built on November 7, 2026, which lacked the five-screen redesigned frontend and the three-model unified backend implemented in Tasks 22–30.

In Task 32, we successfully remediated the deployment configuration, resolved two integration bugs in frontend warning rendering and CSV auto-detection, rebuilt the container from the repository using the standard Docker Compose workflow, and validated the live application on host port `8001` through both automated API suites and real headless browser automation via the Chrome DevTools Protocol (CDP).

---

## 1. Exact Docker Deployment Changes

To ensure Docker Compose builds from the active repository while keeping the deployment clean, hermetic, and fast:

### 1.1 `compose.yaml`
```yaml
services:
  emotion-dynamics-nlp:
    build: .
    image: emotion-dynamics-nlp:latest
    ports:
      - "8001:8000"
    volumes:
      - ./checkpoints:/app/checkpoints:ro
```
- **Image Tag Updated:** Replaced the legacy Week 11 image tag `emotion-dynamics-nlp:week11` with `emotion-dynamics-nlp:latest`.
- **Build Context Preserved:** Maintained `build: .` so Docker Compose builds directly from the current repository workspace.
- **Read-Only Checkpoint Mount:** Preserved `./checkpoints:/app/checkpoints:ro` so large model weights (~280MB) remain host-backed without container bloat.
- **Port Mapping:** Preserved the designated host port `8001:8000`.
- **No Unnecessary Bind Mounts:** Avoided mounting `src/` or `static/` as bind mounts; the container image is hermetic and self-contained.

### 1.2 `Dockerfile`
```dockerfile
# Copy application code, static UI, and benchmark research artifacts
COPY src/ ./src/
COPY static/ ./static/
COPY outputs/tables/ ./outputs/tables/
COPY outputs/experiments/dialoguernn_unweighted/confusion_matrix.json ./outputs/experiments/dialoguernn_unweighted/confusion_matrix.json
```
- **Benchmark Artifacts Included:** Added copies of `outputs/tables/` (308 KB) and `outputs/experiments/dialoguernn_unweighted/confusion_matrix.json` (436 bytes) so the `/benchmarks` endpoint can serve verified research evaluation tables without requiring an external volume mount.

### 1.3 `.dockerignore`
```dockerignore
outputs/*
!outputs/tables/
!outputs/tables/**
!outputs/experiments/
!outputs/experiments/dialoguernn_unweighted/
!outputs/experiments/dialoguernn_unweighted/confusion_matrix.json
```
- Configured exclusion rules so the Docker build context only includes the necessary research benchmark tables, ignoring all bulky temporary runs and figures.

---

## 2. Integration Bugs Fixed

### Bug 1: Parser Warning Rendering in `static/index.html`
- **Root Cause:** Line 2009 in `static/index.html` accessed `w.warning_type`, whereas the canonical `ParserWarning.to_dict()` outputs the standard field `"code"`.
- **Fix Applied:** Updated the template literal to support the canonical backend `code` field while maintaining fallback compatibility:
  ```javascript
  `<li><strong>${escapeHtml(w.code || w.warning_type || 'Warning')}:</strong> ${escapeHtml(w.message)} (line ${w.line_number || 'N/A'})</li>`
  ```
- **Result:** Warnings in the live import preview now render clear codes (e.g., `MISSING_TIMESTAMPS:`, `UNPARSED_LINE:`) rather than empty or `undefined` labels.

### Bug 2: CSV Auto-Detection False Positives on Dialogue with Commas
- **Root Cause:** In `src/data/parsers/csv_parser.py:can_parse()`, any line containing a delimiter (`,`) and a speaker keyword substring (`"speaker"`) was treated as CSV. Dialogue transcripts beginning with lines like:
  ```text
  Speaker A: Hello, how are you today?
  Speaker B: I am doing well, thank you!
  ```
  were incorrectly routed to `TabularCsvParser` instead of `PlainTextParser`.
- **Fix Applied:**
  1. Added dialogue turn detection: if `header` contains `:` where the candidate speaker prefix does not contain delimiters and is not a URL protocol, `can_parse()` returns `False`.
  2. Enhanced cell matching: discrete column tokens are matched against `SPEAKER_ALIASES` and `TEXT_ALIASES`, supporting compound column headers like `char_name` and `dialogue_text` while rejecting arbitrary dialogue text.
- **Regression Tests Added:** In `tests/test_csv_parser.py`:
  - `test_regression_can_parse_rejects_dialogue_with_commas`: Verifies that dialogue transcripts with commas resolve to `PlainTextParser` and parse 3 evaluable turns.
  - `test_regression_can_parse_preserves_legitimate_csv`: Verifies that standard CSV and TSV formats (`speaker,text`, `Speaker, Utterance`, `Sender, Message, Time`, `author\tbody\tcreated_at`, `name;dialogue`) remain correctly detected.
  - `test_regression_warning_schema_contains_code_field`: Verifies `ParserWarning.to_dict()` produces `"code"`.

---

## 3. Rebuild and Restart Verification

The service was restarted using the standard Docker Compose workflow:
```bash
docker compose down
docker compose up -d --build
```

### Container Runtime State
- **Container Name:** `emotion-dynamics-nlp-emotion-dynamics-nlp-1`
- **Container ID:** `59adf7199cda`
- **Image Name & Tag:** `emotion-dynamics-nlp:latest`
- **Image ID:** `sha256:ab7462834648272e167fac2c7822a8b47f1a14a2e284b33d612a22942a51d1cb`
- **Build Timestamp:** `2026-11-13T22:00:58.399458732+05:30`
- **Status:** `Up (healthy)`
- **Port Mapping:** `0.0.0.0:8001->8000/tcp, [::]:8001->8000/tcp`

---

## 4. Live Endpoint Status & Schema Verification

All endpoints were tested live against `http://localhost:8001`:

| Endpoint | Method | HTTP Status | Response Payload Summary | Verification |
| :--- | :--- | :--- | :--- | :--- |
| `GET /` | GET | **200 OK** | 111,069 bytes, redesigned 5-screen HTML document | **PASS** |
| `GET /health` | GET | **200 OK** | `{"status": "ok", "available_modes": ["distilbert", "dialoguernn", "dialoguegcn", ...]}` | **PASS** |
| `GET /models` | GET | **200 OK** | Canonical registry for `distilbert`, `dialoguernn`, `dialoguegcn` (all `is_available: true`) | **PASS** |
| `GET /benchmarks`| GET | **200 OK** | Cohort A overall metrics, per-class metrics, 7x7 confusion matrices, McNemar/permutation tests | **PASS** |
| `POST /import` | POST | **200 OK** | Canonical conversation parsing, evaluable turns count, date ranges, structured warnings | **PASS** |
| `POST /predict`| POST | **200 OK** | Turn-level predictions, uncalibrated confidence scores, 7-class probability distributions | **PASS** |
| `POST /analyze`| POST | **200 OK** | Full emotion dynamics: transitions, shifts, persistence, same- vs cross-speaker, turning points | **PASS** |
| `POST /compare`| POST | **200 OK** | Synchronized 3-model predictions on identical messages, consensus rate, 3x3 agreement matrix | **PASS** |

---

## 5. Model Loading and Inference Results

A canonical conversation was submitted to the live container on port `8001` across all three canonical model families:
```text
Ross: Hey Rachel how are you today?
Rachel: I am so happy to see you!
Ross: Me too, everything is wonderful.
```

### Inference Output on Port 8001
- **DistilBERT (Utterance Baseline):**
  - Checkpoint: `checkpoints/transformer_baseline/best` (66,960,000 params)
  - Turn 0 (Ross): `neutral` (conf: 0.953)
  - Turn 1 (Rachel): `joy` (conf: 0.916)
  - Turn 2 (Ross): `joy` (conf: 0.955)
  - Dynamics: 2 transitions, 1 shift, 1 persistence
- **DialogueRNN (Causal Attentive Recurrent):**
  - Checkpoint: `checkpoints/dialoguernn_unweighted/best_model.pt` + frozen DistilBERT CLS encoder (670,855 params)
  - Turn 0 (Ross): `neutral` (conf: 0.974)
  - Turn 1 (Rachel): `joy` (conf: 0.966)
  - Turn 2 (Ross): `joy` (conf: 0.979)
  - Dynamics: 2 transitions, 1 shift, 1 persistence
- **DialogueGCN (Relational Topology Graph):**
  - Checkpoint: `checkpoints/dialoguegcn/best_model.pt` + frozen DistilBERT CLS encoder (452,487 params)
  - Turn 0 (Ross): `neutral` (conf: 1.000)
  - Turn 1 (Rachel): `joy` (conf: 1.000)
  - Turn 2 (Ross): `joy` (conf: 1.000)
  - Dynamics: 2 transitions, 1 shift, 1 persistence
- **Multi-Model Comparison (`/compare`):**
  - Compared Models: `['distilbert', 'dialoguernn', 'dialoguegcn']`
  - Consensus Rate: `100.0%` (Unanimous agreement across all 3 turns)
  - Pairwise Agreement: 1.000 across all model pairs

---

## 6. Conversation Ingestion Format Results

Tested against the live container's `/import` endpoint:

| Format | Source Input Test | Detected Format | Messages | Evaluable Turns | Warnings / Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Plain Text** | Two speakers with conversational commas | `plain_text` | 2 | 2 | `MISSING_TIMESTAMPS` (Info) |
| **TXT File** | Header line + dialogue turns | `plain_text` | 2 | 2 | `UNPARSED_LINE` (Captured line 1) |
| **Instagram JSON** | JSON payload with epoch timestamps | `instagram_json` | 2 | 2 | Epoch ms converted to ISO |
| **WhatsApp TXT** | `12/10/2020, 10:00 - Alice: Hey` | `whatsapp_txt` | 2 | 2 | Timestamps parsed, ordered |
| **Tabular CSV** | `speaker,text,timestamp` | `tabular_csv` | 2 | 2 | Columns auto-resolved |
| **Validation Warnings**| Unparsed line + empty turn | `plain_text` | 2 | 1 evaluable | `code` rendered correctly |

---

## 7. Real Browser Automation Verification Evidence

We executed real browser automation using headless Google Chrome (`google-chrome --headless=new`) via the Chrome DevTools Protocol (`scripts/verify_task32_live_browser.py`) interacting directly with `http://localhost:8001/`:

```text
==================================================================
TASK 32: LIVE BROWSER AUTOMATION & COMPLIANCE VERIFICATION
Target URL: http://localhost:8001
==================================================================

[PASS] [Global] Page Title Verification
   Detail:   Served Title: Emotion Dynamics Analyzer | Multi-Model Conversation Intelligence
[PASS] [Global] System Status Pill
   Detail:   Confirmed live connection between frontend and backend on port 8001 (3 Models Online)
[PASS] [Home / Import] Model Selector Options
   Detail:   Options: ['distilbert', 'dialoguernn', 'dialoguegcn'], default selected: dialoguegcn
[PASS] [Home / Import] Navigation Tabs & Format Tags
   Detail:   Nav screens: 5, Formats: ['Speaker: text', 'WhatsApp .txt', 'Instagram JSON', 'Structured CSV']
[PASS] [Home / Import] Live Import Preview with Warnings
   Detail:   Rendered warning HTML: <li><strong>MISSING_TIMESTAMPS:</strong> Conversation does not contain timestamp metadata...
   [Screenshot Saved] Screen 1 -> outputs/figures/task32_screen1_import.png
[PASS] [Conversation Viewer] Inference Execution & Bubble Stream
   Detail:   First turn predicted: ● Predicted: neutral
[PASS] [Conversation Viewer] Softmax Probability Drawer
   Detail:   7-class probability tracks present in drawer
   [Screenshot Saved] Screen 2 -> outputs/figures/task32_screen2_viewer.png
[PASS] [Emotion Dynamics] KPIs, Sequence Bar, & Chart.js Timeline
   Detail:   Transitions: 2, Chart.js instance active
   [Screenshot Saved] Screen 3 -> outputs/figures/task32_screen3_dynamics.png
[PASS] [Model Comparison] Multi-Model Consensus & Agreement Matrix
   Detail:   Matrix size: 9 cells, rows: 3, prediction cards: 9
   [Screenshot Saved] Screen 4 -> outputs/figures/task32_screen4_comparison.png
[PASS] [Research Inspection] Cohort A Table, Per-Class Metrics, & Confusion Matrix
   Detail:   Cohort A rows: 6, Matrix cells: 49, Ethical notice verified
   [Screenshot Saved] Screen 5 -> outputs/figures/task32_screen5_research.png

==================================================================
BROWSER AUTOMATION VERIFICATION SUMMARY
==================================================================
Total Tests: 10 | Passed: 10 | Failed: 0
Results written to: outputs/tables/task32_browser_verification_results.json
```

### Visual Artifacts Captured from Port 8001
1. **Screen 1 (Home / Import):** [`outputs/figures/task32_screen1_import.png`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/figures/task32_screen1_import.png) (123 KB) — File dropzone, format tags, text input, model dropdown, live preview drawer with structured warnings.
2. **Screen 2 (Conversation Viewer):** [`outputs/figures/task32_screen2_viewer.png`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/figures/task32_screen2_viewer.png) (92 KB) — Conversation bubbles, speaker badges, predicted emotion pills, confidence scores, open 7-class probability drawer.
3. **Screen 3 (Emotion Dynamics):** [`outputs/figures/task32_screen3_dynamics.png`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/figures/task32_screen3_dynamics.png) (83 KB) — Transition and shift KPI tiles, speaker inertia rates, discrete sequence pills, Chart.js stacked timeline.
4. **Screen 4 (Model Comparison):** [`outputs/figures/task32_screen4_comparison.png`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/figures/task32_screen4_comparison.png) (104 KB) — Consensus rate badge (100%), 3x3 pairwise agreement matrix, synchronized turn comparison cards.
5. **Screen 5 (Research / Model Inspection):** [`outputs/figures/task32_screen5_research.png`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/figures/task32_screen5_research.png) (156 KB) — Benchmark summary cards, Cohort A table, per-class breakdown, 7x7 confusion matrix heatmap, McNemar significance tests, and ethical notice.

---

## 8. Remaining Blockers

**None.**
- The Docker Compose workflow builds and launches the verified application on port 8001.
- All five screens render real backend data.
- All three canonical model checkpoints load without error and return genuine predictions.
- All five ingestion formats operate smoothly with auto-detection.
- Warning rendering and CSV edge cases are tested and resolved.
