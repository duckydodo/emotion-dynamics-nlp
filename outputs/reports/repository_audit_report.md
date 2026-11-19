# Emotion Dynamics NLP — Comprehensive Repository Audit Report
**Baseline Verification, Model Inventory, Artifact Provenance, and Phase Feasibility**

**Document Version:** 1.0.0  
**Date:** November 15, 2026  
**Audit Executed By:** Antigravity Agentic Pair Programmer  
**Repository Working Directory:** `/home/chitta/Projects/emotion-dynamics-nlp`  
**Deployment Environment:** Docker container `emotion-dynamics-nlp:latest` (Port 8001 -> 8000) on Linux (`AMD Ryzen 5 7530U`, 12 vCPUs, 14 GiB RAM)

---

## 1. Executive Summary & Audit Scope

This audit report establishes an uncompromised, ground-truth baseline of the `emotion-dynamics-nlp` codebase prior to initiating new experiments, explainability modules, application refinements, or documentation consolidations.

### Key Audit Takeaways
1. **Model Implementation Status:** Three primary model families are implemented, trained, and verified:
   - **DistilBERT (Utterance-only baseline):** Checkpoint at `checkpoints/transformer_baseline/best`.
   - **DialogueRNN (Sequential Recurrent):** Checkpoint at `checkpoints/dialoguernn_unweighted/best_model.pt`.
   - **DialogueGCN (Relational Graph Convolutional):** Checkpoint at `checkpoints/dialoguegcn/best_model.pt`.
2. **Test Suite Integrity:** The test suite comprises 10 unittest modules containing **133 unit tests**, all of which pass (`OK`, 6.82s execution time inside Docker).
3. **Application & Deployment Status:** The FastAPI backend and five-screen frontend are actively deployed and running via Docker Compose (`emotion-dynamics-nlp-emotion-dynamics-nlp-1` on port 8001), serving the dark charcoal design system (`#171B18`) confirmed in Tasks 33 and 34.
4. **Resolution of Metric Conflicts (Full Test vs. Cohort A):** The historical difference between the full test results ($N=2,610$, Accuracy 60.31%, Macro F1 44.00%) and Cohort A matched test results ($N=2,328$, Accuracy 59.45%, Macro F1 43.14%) is thoroughly traced to **missing utterance-ID gaps and broken dialogue continuity** in raw MELD. DialogueRNN and DialogueGCN strictly require contiguous turns, necessitating Cohort A for apples-to-apples comparison.
5. **Phase 1 Feasibility Benchmark:** On this CPU environment (AMD Ryzen 5 7530U, PyTorch 2.13.0+cpu), forward+backward on DistilBERT (`batch_size=4`, `max_length=128`) executes at **1.3335 seconds per step**. A full 5-epoch run requires **4.63 to 5.1 hours** of dedicated CPU compute.

---

## 2. Existing Project Structure

```
emotion-dynamics-nlp/
├── checkpoints/                    # Model checkpoints and weights
│   ├── ablations/                  # DialogueRNN ablation checkpoints (6 variants)
│   ├── dialoguegcn/                # DialogueGCN best_model.pt (5.5 MB) & config.json
│   ├── dialoguernn_unweighted/     # DialogueRNN best_model.pt (8.1 MB)
│   ├── dialoguernn_weighted/       # DialogueRNN with class weights (8.1 MB)
│   ├── transformer_baseline/       # DistilBERT best/ (model.safetensors, 268 MB)
│   ├── transformer_combined_history/
│   ├── transformer_context_1/
│   ├── transformer_context_1_speaker/
│   ├── transformer_emotion_history/
│   └── transformer_weighted/
├── configs/                        # YAML/JSON ablation configurations
│   └── ablations/                  # 6 ablation config files (window, listener, attention, etc.)
├── data/
│   ├── raw/
│   │   ├── meld/                   # Official MELD train, dev, test CSVs (9,989 / 1,109 / 2,610)
│   │   ├── dailydialog/            # Raw DailyDialog dataset
│   │   └── emowoz/                 # Raw EmoWOZ dataset
│   └── processed/
│       └── dialoguernn/            # Pre-extracted frozen DistilBERT CLS features (.npz & .json)
├── notebooks/                      # 29 exploratory and validation Jupyter notebooks
├── outputs/
│   ├── benchmarks/                 # Latency, parameter, batch size benchmarks
│   ├── experiments/                # Metric logs and prediction CSVs for all runs
│   ├── figures/                    # QA screenshots, confusion matrices, architecture plots
│   ├── reports/                    # 40 formal Markdown research and implementation reports
│   └── tables/                     # Evaluation CSVs, statistical test results, bootstrap metrics
├── scripts/                        # Standalone training, evaluation, and QA automation scripts
│   ├── audit_dialoguernn_validation.py
│   ├── evaluate_dialoguernn_analysis.py
│   ├── evaluate_matched_test.py
│   ├── extract_dialogue_features.py
│   ├── run_benchmarks.py
│   ├── run_canonical_audit.py
│   ├── test_frontend_e2e.py
│   ├── train_combined_history.py
│   ├── train_dialoguegcn.py
│   ├── train_dialoguernn_ablations.py
│   ├── train_dialoguernn.py
│   ├── train_emotion_history.py
│   ├── train_weighted.py
│   ├── verify_extracted_features.py
│   ├── verify_task29_comprehensive.py
│   ├── verify_task32_live_browser.py
│   ├── verify_task33_visual_qa.py
│   └── verify_task34_home_qa.py
├── src/                            # Core Python package
│   ├── analysis/                   # Transition matrices, persistence, turning points
│   ├── api.py                      # FastAPI application and route endpoints
│   ├── application/                # Conversation analyzer and comparison service
│   ├── benchmarking/               # Profiling and latency benchmarking tools
│   ├── config.py                   # Global configuration and path anchors
│   ├── data/                       # Ingestion parsers (Text, WhatsApp, Instagram, CSV)
│   ├── evaluation/                 # Metrics calculation, permutation & bootstrap tests
│   ├── features/                   # Vocabulary and feature extraction utilities
│   └── models/                     # Architectures (DistilBERT, DialogueRNN, DialogueGCN, loss, registry)
├── static/                         # Frontend assets
│   ├── app.js                      # Application state controller and Chart.js integration
│   └── index.html                  # 5-screen interface with Dark Charcoal styling
├── tests/                          # 10 test modules (133 tests)
├── compose.yaml                    # Docker Compose configuration (port 8001:8000)
├── Dockerfile                      # Production container image definition
├── pyproject.toml                  # Python package configuration
└── requirements*.txt               # Dependencies (requirements.txt, requirements-api.txt, requirements-dev.txt)
```

---

## 3. Implemented Models and Actual Status

| Model Identifier | Architecture Family | Context Mechanism | Checkpoint Location | Parameter Count | Selection Criterion | Verification Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`distilbert`** | Transformer | Single utterance (no conversational context) | `checkpoints/transformer_baseline/best` | 66,960,000 | Highest Dev Macro F1 (Epoch 3) | **Verified & Active** |
| **`dialoguernn`** | Recurrent (GRU) | Causal sequential speaker & global state tracking | `checkpoints/dialoguernn_unweighted/best_model.pt` | 670,855 | Highest Dev Macro F1 (Epoch 10) | **Verified & Active** |
| **`dialoguegcn`** | Relational GCN | Bidirectional speaker graph with window=10 | `checkpoints/dialoguegcn/best_model.pt` | 452,487 | Highest Dev Macro F1 (Epoch 15) | **Verified & Active** |
| `context_1` | Transformer | Preceding utterance concatenation | `checkpoints/transformer_context_1` | 66,960,000 | Dev Macro F1 | Exploratory (Archived) |
| `context_1_speaker` | Transformer | Preceding utterance + speaker token | `checkpoints/transformer_context_1_speaker` | 66,960,000 | Dev Macro F1 | Exploratory (Archived) |
| `dialoguernn_weighted` | Recurrent (GRU) | Class-weighted cross entropy | `checkpoints/dialoguernn_weighted/best_model.pt` | 670,855 | Dev Macro F1 | Evaluated (Ablation) |
| `dialoguernn_ablations` | Recurrent (GRU) | 6 variants (`no_listener`, `no_context`, etc.) | `checkpoints/ablations/*` | 400K–700K | Dev Macro F1 | Validated (Ablation) |

### Key Observations:
* Only the three core models (`distilbert`, `dialoguernn`, `dialoguegcn`) are exposed via `src/models/registry.py` and the public inference API.
* All experimental and ablation variants remain isolated in checkpoints and experiment folders, preserving strict submission boundaries.

---

## 4. Existing Checkpoints and Experiment Outputs

### 4.1 Checkpoints
1. `checkpoints/transformer_baseline/best/`:
   - `model.safetensors` (267.8 MB), `config.json` (932 B), `tokenizer.json` (711 KB), `training_config.json` (299 B).
   - Trained on MELD train set (9,989 utterances).
2. `checkpoints/dialoguernn_unweighted/best_model.pt` (8.08 MB):
   - PyTorch checkpoint containing model state dict, optimizer state, epoch metadata, and development metrics.
3. `checkpoints/dialoguegcn/best_model.pt` (5.45 MB):
   - PyTorch checkpoint containing 2-layer GCN weights, similarity attention parameters, and optimizer states.
4. `checkpoints/ablations/`:
   - Contains 6 trained ablation models: `no_listener_best.pt`, `no_context_best.pt`, `no_attention_best.pt`, `window_3_best.pt`, `window_5_best.pt`, `window_10_best.pt`.

### 4.2 Experiment Outputs
* `outputs/experiments/transformer_baseline/`:
   - `metrics.json`: Test examples: 2,610, Accuracy: 0.6031, Macro F1: 0.4400, Weighted F1: 0.6054.
   - `predictions.csv`: 2,610 rows containing raw text, true label, predicted label, and 7-class probability distributions.
* `outputs/experiments/dialoguernn_unweighted/`:
   - Training history, development evaluation metrics across 20 epochs, and test predictions on Cohort A.
* `outputs/experiments/dialoguegcn/`:
   - Training logs across 30 epochs, development Macro F1 curve, and predictions on Cohort A.
* `outputs/tables/`:
   - `matched_test_evaluation.csv`: Direct side-by-side comparison on Cohort A ($N=2,328$).
   - `canonical_statistical_tests.csv`: Permutation test ($p$-values) and bootstrap confidence intervals.

---

## 5. Existing Application Features

The application is served by FastAPI (`src/api.py`) and a vanilla HTML5/ES6 frontend (`static/app.js`, `static/index.html`):

### 5.1 Route Endpoints
* `GET /`: Serves static frontend (`static/index.html`).
* `GET /health`: Health probe returning system status, timestamp, and active checkpoint availability.
* `GET /models`: Returns registry metadata for DistilBERT, DialogueRNN, and DialogueGCN.
* `POST /analyze`: Ingests conversation turns and runs inference using a selected model.
* `POST /compare`: Runs multi-model comparison across identical input turns.
* `POST /ingest`: Parses raw conversation text, WhatsApp, Instagram, or CSV into canonical schema.
* `GET /benchmarks`: Returns latency, parameter size, and batch sensitivity telemetry.
* `GET /demo`: Loads curated canonical MELD dialogue examples.

### 5.2 Frontend Screens (Dark Charcoal Visual Direction)
1. **Home / Import:** Format selector (Paste, TXT, WhatsApp, Instagram, CSV), drag-and-drop dropzone, MELD sample loader.
2. **Conversation Viewer:** Dual-column chat bubbles, speaker badges, predicted emotion pills with confidence badges.
3. **Emotion Dynamics:** Chart.js sentiment timeline, speaker emotional distribution bars, emotion transition summary tables.
4. **Model Comparison:** Utterance-by-utterance side-by-side prediction grid, model consensus badges, disagreement indicators.
5. **Research Inspection:** Architecture descriptions, parameter counts, matched-cohort metrics, and explicit MELD domain shift disclaimers.

---

## 6. Existing Tests and Verification Status

Running the complete unittest suite inside the active execution environment:
```bash
docker run --rm -v $(pwd):/app -w /app emotion-dynamics-nlp:latest python3 -m unittest discover -s tests
```
**Result:**
* **Test modules executed:** 10
* **Total unit tests:** 133
* **Pass count:** 133
* **Failures / Errors:** 0
* **Execution duration:** 6.818s
* **Status:** **PASS (`OK`)**

### Breakdown by Module:
1. `tests/test_parsers.py`: Ingestion logic for canonical conversations.
2. `tests/test_whatsapp_parser.py`: Timestamp parsing, 12h/24h formats, brackets, multiline messages.
3. `tests/test_instagram_parser.py`: JSON message extraction, sender alignment, encoding fixes.
4. `tests/test_csv_parser.py`: Header detection, comma/semicolon delimiters, quote handling.
5. `tests/test_ingestion_security_and_limits.py`: File size bounds, malicious payload rejection.
6. `tests/test_dialoguernn.py`: Sequential forward pass, party state updates, causal masking.
7. `tests/test_dialoguernn_ablations.py`: Verification that all 6 ablation variants initialize and run.
8. `tests/test_dialoguegcn.py`: Adjacency construction, GCN relational convolution, speaker edge logic.
9. `tests/test_three_model_integration.py`: Unified inference interface for DistilBERT, DialogueRNN, DialogueGCN.
10. `tests/test_application_integration.py`: End-to-end FastAPI endpoint validation.

---

## 7. Conflicting or Unverifiable Results Analysis

### Investigation of Full Test vs. Cohort A Metrics
An apparent discrepancy appears across past research notes:
* **Metric Set 1 (Full MELD Test Set, $N=2,610$):**
  - DistilBERT Accuracy: 60.31%
  - DistilBERT Macro F1: 44.00%
  - DistilBERT Weighted F1: 60.54%
* **Metric Set 2 (Canonical Cohort A Matched Set, $N=2,328$):**
  - DistilBERT Accuracy: 59.45%
  - DistilBERT Macro F1: 43.14%
  - DistilBERT Weighted F1: 57.94%
  - DialogueRNN Accuracy: 60.14% (Macro F1 43.42%, Weighted F1 58.92%)
  - DialogueGCN Accuracy: 60.91% (Macro F1 43.85%, Weighted F1 60.09%)

### Root Cause Analysis:
1. **MELD Dataset Structure:** The official MELD test split contains 280 dialogues totaling 2,610 utterances.
2. **Missing Utterance IDs:** In several dialogues, the original audio-visual data extraction resulted in non-consecutive utterance sequences (e.g., Utterance 0, followed immediately by Utterance 2, with Utterance 1 missing).
3. **Dialogue Boundary & Causal Validity:**
   - Utterance-only DistilBERT classifies each utterance in complete isolation; it can evaluate all 2,610 utterances regardless of gaps.
   - DialogueRNN and DialogueGCN rely on conversational continuity (speaker tracking, emotion inertia, turn-taking graphs). Constructing adjacency edges across missing utterance gaps introduces invalid dialogue transitions.
4. **Cohort Definition:**
   - **Full Test Set ($N=2,610$):** All utterances evaluated independently.
   - **Cohort A ($N=2,328$):** Strict contiguous dialogues without missing utterance gaps.
5. **Resolution:** Both metric sets are completely valid and verifiable when accompanied by their cohort specification. In all comparative analyses, **Cohort A is the gold-standard matched cohort**.

---

## 8. Hardware & Phase 1 Feasibility Assessment

### 8.1 Machine Specifications
* **CPU:** AMD Ryzen 5 7530U with Radeon Graphics
* **Topology:** 1 Socket, 6 Physical Cores, 12 Threads (SMT enabled)
* **RAM:** 14 GiB total (6.3 GiB currently available, 4.8 GiB free swap)
* **Storage:** NVMe SSD, 51 GiB free disk space on `/home`
* **GPU Availability:** None (integrated graphics only; PyTorch CPU execution)

### 8.2 DistilBERT Five-Epoch Benchmark Results
A live timing benchmark was executed on CPU using the exact experimental parameters:
- **Architecture:** `distilbert-base-uncased` (Sequence Classification, 7 labels)
- **Batch Size:** 4
- **Sequence Length:** 128 tokens
- **Optimizer:** AdamW (`lr=2e-5`)
- **Forward + Backward Step Duration:** **1.3335 seconds** per step
- **Steps per Epoch:** 2,498 steps ($9,989 \div 4$)
- **Time per Epoch:** **55.52 minutes** (0.93 hours)
- **Validation Time per Epoch (Dev $N=1,109$):** ~4.5 minutes
- **Estimated Total Runtime for 5 Epochs:** **4.63 hours** (pure training) + ~0.4 hours (evaluations) = **~5.0 hours**

### 8.3 Feasibility Determination
* **Feasibility:** **Feasible on CPU**, provided it runs in the background with incremental checkpointing (`latest_training_state.pt` and `epoch_*/`).
* **RAM Footprint:** ~2.4 GiB (well within 6.3 GiB available RAM; zero risk of OOM).
* **Disk Space Footprint:** ~268 MB per epoch checkpoint $\times$ 5 = ~1.34 GB (well within 51 GB free space).
* **Interruptibility:** Must maintain incremental epoch state to allow resuming if stopped.

---

## 9. Gap Analysis and Execution Plan

Mapped to the remaining project phases:

| Phase | Description | Pre-existing Code | Missing / Required Work |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Controlled 5-Epoch DistilBERT | Training pipeline in `trainer.py` | Dedicated script, 5-epoch run, epoch checkpoints, evaluation vs. baseline |
| **Phase 2A**| Temporal CNN Exploratory Run | Feature extractor exists (`.npz`) | Temporal CNN model, causal padding, kernel tuning, Cohort A evaluation |
| **Phase 2B**| CRF Exploratory Run | Linear-chain CRF formulation | Emission projection, Viterbi decoding, sequence metrics, Cohort A evaluation |
| **Phase 3** | Prediction Explainability | Softmax probabilities in API | Integrated Gradients text attribution, context ablation, model disagreement tool |
| **Phase 4** | Research Documentation | Individual reports in `reports/` | Consolidated findings master report (20 sections) & evaluation presentation |
| **Phase 5** | REA Setup & Verification | None installed | Official REA installation, isolated setup, harmless public site test |
| **Phase 6** | Application Improvements | 5-screen interface running | Timeline turning point navigation, disagreement inspector, import preview table |
| **Phase 7** | Multimodal Roadmap | Concept notes | Comprehensive vacation research specification (documentation only) |
| **Phase 8** | Final Verification & Handover | Comprehensive scripts exist | Full end-to-end validation, documentation reconciliation, status handover table |

---
*Audit successfully completed and verified against repository source code and runtime environment.*
