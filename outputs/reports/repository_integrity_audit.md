# NLP Repository Integrity Audit

**Repository:** `emotion-dynamics-nlp`  
**Location:** `~/Projects/emotion-dynamics-nlp`  
**Audit Date:** 2026-11-07  
**Commit Inspected:** `6bc76bb` (`main`, tracking `origin/main`)  

---

## 1. Current Repository State

### 1.1 Git Branch & Status
- **Current Branch:** `main` (up to date with `origin/main`).
- **Working Tree State:** Dirty.
  - **5 modified tracked files:** `Dockerfile`, `compose.yaml`, `src/evaluation/comparison.py`, `src/models/trainer.py`, `src/models/transformer.py`.
  - **18 untracked paths:**
    - Notebooks: `notebooks/26_class_weighted_distilbert.ipynb`, `notebooks/29_on_device_benchmarking.ipynb`
    - Scripts: `scripts/evaluate_matched_test.py`, `scripts/run_benchmarks.py`, `scripts/train_combined_history.py`, `scripts/train_emotion_history.py`, `scripts/train_weighted.py`
    - Core modules: `src/benchmarking/` (`__init__.py`, `models.py`, `profiler.py`), `src/models/loss.py`
    - Dependency files: `requirements-api.txt`, `requirements-dev.txt`
    - Experiment outputs: `outputs/experiments/transformer_weighted/`, `outputs/experiments/transformer_emotion_history/`, `outputs/experiments/transformer_combined_history/`
    - Benchmark tables & figures: `outputs/benchmarks/` (3 files), `outputs/figures/benchmark_*.png` (3 files)
    - Matched evaluation tables: `outputs/tables/matched_test_evaluation.csv`, `outputs/tables/matched_test_per_class.csv`
- **Recent Commit History:**
  1. `6bc76bb` (HEAD -> main, origin/main) Prepare repository for release: add web demo, training evaluation, and comprehensive README
  2. `d441166` Complete Week 10 higher-order forecasting
  3. `6d11d04` Extend forecasting with higher-order Markov models
  4. `2267c57` Complete Week 9 forecasting baselines
  5. `500fd5a` Complete Weeks 6-8 emotion dynamics analysis

### 1.2 Storage & Checkpoint Footprint
- **Tracked Checkpoints (`.gitignore` rule `checkpoints/`):**
  - `checkpoints/transformer_baseline/best` (256 MB DistilBERT safetensors, config, tokenizer, training_config)
  - `checkpoints/transformer_context_1/best` (256 MB)
  - `checkpoints/transformer_context_1_speaker/best` (256 MB)
  - `checkpoints/transformer_weighted/best` (256 MB) + `latest_training_state.pt` (767 MB)
  - `checkpoints/transformer_emotion_history/best` (256 MB) + `best_epoch2_backup/` + `latest_training_state.pt` + `latest_training_state.pt.bak`
  - `checkpoints/transformer_combined_history/best` (256 MB) + `latest_training_state.pt` (767 MB)
  - *Missing:* `transformer_context_3` and `transformer_context_5` have no saved checkpoints on disk.
- **Datasets:**
  - `data/raw/meld/`: All splits present (`train_sent_emo.csv` [9,989 turns], `dev_sent_emo.csv` [1,109 turns], `test_sent_emo.csv` [2,610 turns]).
  - `data/raw/emowoz/`: MultiWOZ, DialMAGE, and split JSON files present.
  - `data/raw/dailydialog/`: Directory exists but is empty.
  - `data/processed/`: Empty (all feature extractions and context window constructions execute dynamically in memory).

---

## 2. Existing Uncommitted Changes & Diff Analysis

All modifications currently in the working tree are functional, localized, and backward-compatible:

| Modified File | Summary of Changes | Rationale / Safety Assessment |
| :--- | :--- | :--- |
| `Dockerfile` | Switched from copying heavy `requirements.txt` and baking `checkpoints/` inside the image to installing `requirements-api.txt` with a CPU-only PyTorch wheel (`torch==2.13.0+cpu`). Removed `COPY checkpoints/...`. | Safe optimization. Reduces image size from 10.8 GB to 2.46 GB and eliminates CUDA bloat. |
| `compose.yaml` | Added host volume mount: `./checkpoints:/app/checkpoints:ro`. | Safe. Mounts local checkpoints read-only at runtime rather than baking them into layers. |
| `src/evaluation/comparison.py` | Added fallback handling for metric keys (`accuracy` vs `test_accuracy`, `macro_f1` vs `test_macro_f1`) and added an `Examples` count column to output tables. | Safe. Prevents `KeyError` when comparing classical baseline metric files with transformer experiment files. |
| `src/models/trainer.py` | 1. Added `class_weights`, `loss_fn`, `weighting_method`, and `weighting_params` to `TransformerTrainer`.<br>2. Implemented atomic checkpoint saving using temporary files (`latest_training_state.pt.tmp` replaced to `latest_training_state.pt`).<br>3. Serialized weighting parameters into `training_config.json`. | Critical improvement. Atomic saving protects against checkpoint corruption during power loss or kernel crashes. |
| `src/models/transformer.py` | 1. Added `LABEL_TO_ID` and `ID_TO_LABEL` module-level lookups.<br>2. Attached `class_weights` and `loss_fn` attributes to `TransformerEmotionClassifier`. | Safe. Backward-compatible extension used by the weighted trainer and inference scripts. |

---

## 3. Experiment Reproducibility Status

Across all 15 completed experiment configurations, reproducibility artifacts were audited across 9 standard scientific criteria:

| Experiment Name | Model Family | Config Saved | Random Seed | Training History | Best Checkpoint | Val Metrics | Test Metrics | Test Predictions | Per-Class Metrics | Confusion Matrix | Reproducibility Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `majority` | Classical | Hardcoded | None | N/A | No | No | Yes | No | Yes (embedded) | Yes | **Partial (Notebook-only)** |
| `logistic_unigram` | Classical | Hardcoded | 42 | N/A | No | No | Yes | No | Yes (embedded) | Yes | **Partial (Notebook-only)** |
| `logistic_bigram` | Classical | Hardcoded | 42 | N/A | No | No | Yes | No | Yes (embedded) | Yes | **Partial (Notebook-only)** |
| `logistic_trigram` | Classical | Hardcoded | 42 | N/A | No | No | Yes | No | Yes (embedded) | Yes | **Partial (Notebook-only)** |
| `svm_unigram` | Classical | Hardcoded | 42 | N/A | No | No | Yes | No | Yes (embedded) | Yes | **Partial (Notebook-only)** |
| `svm_bigram` | Classical | Hardcoded | 42 | N/A | No | No | Yes | No | Yes (embedded) | Yes | **Partial (Notebook-only)** |
| `svm_trigram` | Classical | Hardcoded | 42 | N/A | No | No | Yes | No | Yes (embedded) | Yes | **Partial (Notebook-only)** |
| `transformer_baseline` | Transformer | Yes | 42 | No | Yes | Best Val F1 | Yes | Yes (N=2,610) | Yes (CSV) | Yes | **High** |
| `transformer_context_1` | Transformer | Yes | 42 | No | Yes | Best Val F1 | Yes | Yes (N=2,610) | Yes (CSV) | Yes | **High** |
| `transformer_context_1_speaker` | Transformer | Yes | 42 | No | Yes | Best Val F1 | Yes | Yes (N=2,610) | Yes (CSV) | Yes | **High** |
| `transformer_context_3` | Transformer | No | (42) | No | **No** | No | Yes | Yes (N=1,807) | Yes (CSV) | Yes | **Moderate (Missing ckpt)** |
| `transformer_context_5` | Transformer | No | (42) | No | **No** | No | Yes | Yes (N=1,354) | Yes (CSV) | Yes | **Moderate (Missing ckpt)** |
| `transformer_weighted` | Transformer | Yes | 42 | Yes | Yes | Full (epochs) | Yes | Yes (full+matched)| Yes (CSV) | Yes (2 plots) | **Full / Complete** |
| `transformer_emotion_history` | Transformer | Yes | 42 | Yes | Yes | Full (epochs) | Yes (auto+oracle)| Yes (N=2,610) | Yes (matched) | Yes (3 plots) | **Full / Complete** |
| `transformer_combined_history` | Transformer | Yes | 42 | Yes | Yes | Full (epochs) | Yes (auto+oracle)| Yes (N=2,610) | Yes (matched) | Yes (3 plots) | **Full / Complete** |

---

## 4. Missing Artifacts & Discrepancies

1. **Missing Checkpoints for Context-3 and Context-5:**
   - Neither `checkpoints/transformer_context_3` nor `checkpoints/transformer_context_5` exists on disk.
   - Predictions (`predictions.csv`), classification reports, and confusion matrices exist, but model re-inference without retraining is not possible for these two configurations.
2. **Missing Classical Model Artifacts:**
   - No serialized `.joblib` or `.pkl` vectorizers or model weights exist for majority, logistic regression, or SVM.
   - No row-level prediction CSVs exist for the classical models (only summary `metrics.json` and confusion matrices).
3. **Absence of Epoch History in Early Transformers:**
   - `transformer_baseline`, `transformer_context_1`, and `transformer_context_1_speaker` only store the scalar `best_val_macro_f1` in `training_config.json`. Step-level or epoch-level training/validation loss curves were not saved to a `training_history.json`.
4. **Per-Class Granularity for Models A and B:**
   - `outputs/experiments/transformer_emotion_history` and `outputs/experiments/transformer_combined_history` contain `matched_classification_report.csv` (for subsequent turns, $N=2,330$), but lack a standalone `full_test_classification_report.csv` ($N=2,610$) and `turn_0_classification_report.csv` ($N=280$). Because `predictions_autoregressive.csv` stores all 2,610 rows, these reports can be derived directly without retraining.
5. **Evaluation Subset Sizing Distinction (2,328 vs 2,330):**
   - In `src/data/context.py` and `scripts/evaluate_matched_test.py`, turns immediately following dialogue ID gaps (dialogues 93 and 108) are dropped, yielding **$N=2,328$** context-1 turns.
   - In `scripts/train_emotion_history.py` and `scripts/train_combined_history.py`, sequence grouping strictly by dialogue partitions yields **$N=2,330$** subsequent turns ($2,610 - 280 = 2,330$). Both subsets are methodologically valid, but the 2-turn difference must be clearly reported to avoid ambiguity.

---

## 5. Documentation Gaps

1. **README Version Lag:**
   - `README.md` documents progress through Week 10 (September 2026).
   - It contains zero references to:
     - Class-weighted DistilBERT experiments (`notebooks/26`, `scripts/train_weighted.py`).
     - On-device CPU inference benchmarking (`notebooks/29`, `scripts/run_benchmarks.py`, `outputs/benchmarks/`).
     - Phase 1 Step 2A (Emotion-History DistilBERT / Model A) and Phase 1 Step 2B (Combined-History DistilBERT / Model B).
2. **Docker Instructions Divergence:**
   - `README.md` instructs the user to have checkpoints present *before* building the Docker image because it assumes Dockerfile copies them.
   - The current optimized setup uses host volume mounts (`./checkpoints:/app/checkpoints:ro`), which does not require checkpoints at image build time.
3. **Notebook Numbering Gaps:**
   - Notebooks `26` and `29` exist, but `27` and `28` do not exist. Model A and Model B were developed as modular CLI scripts (`scripts/train_emotion_history.py` and `scripts/train_combined_history.py`) rather than exploratory notebooks. A short note in `notebooks/README.md` or the main documentation should clarify this.

---

## 6. Recommended Action Items (Ranked by Importance)

### Priority 1: Version Control & Track Untracked Modules (High)
*Do not leave core research code untracked.* When authorized, stage and commit:
- `requirements-api.txt` and `requirements-dev.txt`
- `src/models/loss.py` and `src/benchmarking/`
- `scripts/train_emotion_history.py`, `scripts/train_combined_history.py`, `scripts/train_weighted.py`, `scripts/evaluate_matched_test.py`, `scripts/run_benchmarks.py`
- `notebooks/26_class_weighted_distilbert.ipynb` and `notebooks/29_on_device_benchmarking.ipynb`
- Modified source files (`src/models/trainer.py`, `src/models/transformer.py`, `src/evaluation/comparison.py`, `Dockerfile`, `compose.yaml`).

### Priority 2: Standardize Full-Test Per-Class Reports for Models A & B (Medium)
Generate standalone `classification_report.csv` for the full test set ($N=2,610$) and Turn-0 ($N=280$) from the existing `predictions_autoregressive.csv` to achieve complete reporting parity with earlier experiments.

### Priority 3: Update `README.md` & Document Subset Definitions (Medium)
1. Document the new Docker workflow (runtime checkpoint mount).
2. Add a section covering the Phase 1 research results (Model A, Model B, and Class Weighting).
3. Explicitly document the distinction between the contiguous matched subset ($N=2,328$) and the all-subsequent-turns subset ($N=2,330$).

### Priority 4: Classical Baseline Parity (Low)
If complete artifact parity is required across all 15 experiments, re-execute classical baseline notebooks (`10` through `14`) to export `predictions.csv` and serialized model pipelines. This is low priority since tabular metrics and confusion matrices are already recorded in `outputs/experiments/`.
