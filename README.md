# Emotion Dynamics in Multi-Party Conversations Using NLP

An empirical NLP research project and interactive web application investigating how conversational context and speaker identity affect **emotion recognition**, **emotion dynamics**, and **next-emotion forecasting** in multi-party dialogue.

> - **Project Type:** Individual Undergraduate Research Project
> - **Primary Dataset:** MELD (Multimodal EmotionLines Dataset, text modality)
> - **Core Architecture:** DistilBERT-base-uncased sequence classification with multi-turn context formatting
> - **Deployment:** Containerized FastAPI backend with an interactive vanilla JS/Chart.js single-page frontend

---

## Table of Contents
1. [Project Overview & Motivation](#project-overview--motivation)
2. [Research Questions & Objectives](#research-questions--objectives)
3. [Dataset Description & Preprocessing](#dataset-description--preprocessing)
4. [Model Architecture & Methodology](#model-architecture--methodology)
5. [Evaluation Results](#evaluation-results)
   - [Classical Baselines](#1-classical-n-gram-baselines-meld-test-split)
   - [Transformer Test Split Results](#2-transformer-test-split-results)
   - [Matched Context-1 Error Analysis](#3-matched-head-to-head-error-analysis-n--2328)
   - [Training Split Evaluation](#4-training-split-evaluation-model-fit--generalization-gap)
6. [Emotion Dynamics & Speaker Analysis](#emotion-dynamics--speaker-analysis)
7. [Next-Emotion Forecasting Experiments](#next-emotion-forecasting-experiments)
8. [Application Architecture & Features](#application-architecture--features)
9. [Installation & Execution Guide](#installation--execution-guide)
   - [Running with Docker Compose (Recommended)](#running-with-docker-compose-recommended)
   - [Running Locally with Python](#running-locally-with-python)
10. [API Specification & Examples](#api-specification--examples)
11. [Repository Structure](#repository-structure)
12. [Limitations & Open Challenges](#limitations--open-challenges)
13. [Reproducibility & Checkpoints](#reproducibility--checkpoints)
14. [References & Acknowledgments](#references--acknowledgments)

---

## Project Overview & Motivation

In human conversations, emotional expressions do not occur in isolation. An utterance such as *"Yeah, right"* can convey amusement, sarcasm, or hostility depending entirely on what was said immediately beforehand and who said it. While standard emotion classification models treat conversational turns as independent text classification problems, real-world dialogue requires understanding:
1. **Conversational Context:** How previous dialogue turns inform the emotional interpretation of the current utterance.
2. **Speaker Dynamics:** How transitions between the same speaker versus different interlocutors impact emotional stability or escalation.
3. **Temporal Dynamics:** How emotional trajectories unfold over time (inertia, shifts, and turning points).

This project investigates these phenomena using the text modality of the **MELD** corpus, comparing classical NLP baselines against fine-tuned **DistilBERT** models across various context window formulations, and deploying the resulting models into an interactive conversation analytics dashboard.

---

## Research Questions & Objectives

1. **Impact of Prior Context:** Does incorporating immediate conversational history ($t-1$) improve fine-grained emotion classification over single-utterance baselines?
2. **Speaker Information:** Does prepending explicit speaker identity tokens (`[SPEAKER_X]`) help resolve ambiguous cross-speaker transitions?
3. **The "Neutral Gravity" Dilemma:** In an imbalanced conversational dataset dominated by neutral expressions, does context help minority emotion detection or exacerbate majority bias?
4. **Trajectory & Dynamics Modeling:** What empirical transition patterns govern multi-party dialogues (e.g., self-persistence vs. cross-speaker emotion contagion)?
5. **Emotion Forecasting:** Can simple Markov and persistence models predict the *next* emotional state without access to future text?

---

## Dataset Description & Preprocessing

During exploratory data analysis (Gate 1), three candidate dialogue corpora were examined: **DailyDialog**, **EmoWOZ**, and **MELD**. MELD was selected as the primary corpus because of its multi-party structure, explicit speaker annotations, high transition density (88.5% of dialogues feature emotional transitions), and realistic conversational complexity.

### MELD Split Statistics

MELD is annotated across seven emotion classes: `anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, and `surprise`.

| Split | Utterances | Dialogues | Emotion Distribution |
| :--- | :---: | :---: | :--- |
| **Train** | 9,989 | 1,038 | Neutral: 4,710, Joy: 1,743, Surprise: 1,205, Anger: 1,109, Sadness: 683, Disgust: 271, Fear: 268 |
| **Validation (Dev)** | 1,109 | 114 | Neutral: 470, Joy: 163, Surprise: 150, Anger: 153, Sadness: 111, Disgust: 22, Fear: 40 |
| **Held-Out Test** | 2,610 | 280 | Neutral: 1,256, Joy: 402, Anger: 345, Surprise: 281, Sadness: 208, Disgust: 68, Fear: 50 |

### Preprocessing & Context Window Rules
Contextual examples are constructed using [`src/data/context.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/data/context.py):
- **Contiguity Requirement:** Context windows are formed strictly between contiguous utterance IDs within the same dialogue.
- **Dialogue Boundaries:** The initial utterance of each dialogue ($t=0$) has no preceding conversational turn and is excluded from multi-turn training/evaluation.
- **Gaps:** 69 dialogues in MELD contain non-contiguous utterance IDs (e.g., jump from turn 3 to 5). Any turn following a gap is excluded to prevent stitching unrelated turns together.

---

## Model Architecture & Methodology

All neural models are built on **DistilBERT-base-uncased** (6 transformer layers, 768 hidden dimensions, 12 attention heads, ~66M parameters) fine-tuned for sequence classification using cross-entropy loss:
- **Optimizer:** AdamW (`lr=2e-5`)
- **Batch Size:** 4
- **Epochs:** 3
- **Max Sequence Length:** 128
- **Checkpoint Selection:** Best epoch selected based on **Validation Macro F1**.

### Evaluated Configurations

1. **Utterance-only Baseline (`transformer_baseline`):**
   - Input: Current utterance text $u_t$.
   - Format: `u_t`
2. **Context-1 (`transformer_context_1`):**
   - Input: Concatenation of previous utterance $u_{t-1}$ and current utterance $u_t$.
   - Format: `u_{t-1}\nu_t`
3. **Context-1 + Speaker (`transformer_context_1_speaker`):**
   - Input: Speaker-prefixed previous and current turns.
   - Format: `[SPEAKER_{s_{t-1}}] u_{t-1}\n[SPEAKER_{s_t}] u_t`
4. **Context-3 & Context-5:**
   - Extended histories of 3 and 5 preceding turns concatenated sequentially.

---

## Evaluation Results

### 1. Classical N-Gram Baselines (MELD Test Split)
Evaluated on all 2,610 held-out test utterances ([`outputs/experiments/`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/experiments/)):

| Model | Representation | Accuracy | Macro F1 | Weighted F1 |
| :--- | :--- | :---: | :---: | :---: |
| **Majority Baseline** | Predicts `neutral` | 48.12% | 0.0928 | 0.3127 |
| **Logistic Regression** | TF-IDF Unigram | 52.68% | 0.2410 | 0.4444 |
| **Logistic Regression** | TF-IDF Bigram | **53.18%** | 0.2470 | 0.4432 |
| **Logistic Regression** | TF-IDF Trigram | 52.76% | 0.2327 | 0.4348 |
| **Linear SVM** | TF-IDF Unigram | 50.57% | 0.2788 | 0.4596 |
| **Linear SVM** | TF-IDF Bigram | 50.84% | **0.2874** | 0.4641 |
| **Linear SVM** | TF-IDF Trigram | 51.07% | 0.2822 | **0.4647** |

> **Note on Linear SVM:** Performance depends on the target metric. Linear SVM Bigram achieves the highest Macro F1 (**0.2874**; Accuracy 50.84%), while Linear SVM Trigram achieves the highest raw Accuracy (**51.07%**; Macro F1 0.2822) and Weighted F1 (**0.4647**) among the SVM baselines.

### 2. Transformer Test Split Results
Evaluated on the official MELD test split ([`outputs/tables/week6_transformer_comparison.csv`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/tables/week6_transformer_comparison.csv)):

| Model Configuration | Evaluated Test Utterances | Test Accuracy | Test Macro F1 | Test Weighted F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Transformer Baseline** (Utterance) | 2,610 | 60.31% | **0.4400** | **0.6054** |
| **Context-1** | 2,328 | 60.74% | 0.3912 | 0.5812 |
| **Context-1 + Speaker** | 2,328 | **61.94%** | 0.4078 | 0.5981 |
| **Context-3** | 1,807 | 58.44% | 0.3863 | 0.5622 |
| **Context-5** | 1,354 | 57.09% | 0.3798 | 0.5542 |

> **Important Evaluation Note:** Context-3 and Context-5 require 3 and 5 contiguous preceding turns respectively, excluding shorter dialogues and early turns. Their lower accuracy (58.44% and 57.09%) reflects both window dilution and evaluation on a smaller, dialogue-dense subset rather than a direct degradation across identical test turns.

### 3. Matched Head-to-Head Error Analysis ($N = 2,328$)
To ensure an exact apples-to-apples comparison between Baseline, Context-1, and Context-1 + Speaker, all models were evaluated on the **identical 2,328 test turns** where Context-1 is available ([`notebooks/21_week6_error_analysis.ipynb`](file:///home/chitta/Projects/emotion-dynamics-nlp/notebooks/21_week6_error_analysis.ipynb)):

| Metric | Utterance Baseline | Context-1 | Context-1 + Speaker |
| :--- | :---: | :---: | :---: |
| **Matched Accuracy** | 59.45% | 60.74% (+1.29%) | **61.94%** (+2.49%) |

#### The "Neutral Gravity" Phenomenon: Per-Class Accuracy Breakdown
* **Neutral ($n=1,116$):** 73.57% (Baseline) $\to$ **83.33%** (Context-1) $\to$ 82.35% (Context-1 + Speaker)
* **Anger ($n=307$):** **36.81%** (Baseline) $\to$ 30.94% (Context-1) $\to$ 31.60% (Context-1 + Speaker)
* **Disgust ($n=63$):** **20.63%** (Baseline) $\to$ 14.29% (Context-1) $\to$ 11.11% (Context-1 + Speaker)
* **Fear ($n=44$):** **25.00%** (Baseline) $\to$ 6.82% (Context-1) $\to$ 11.36% (Context-1 + Speaker)
* **Joy ($n=358$):** **58.66%** (Baseline) $\to$ 54.47% (Context-1) $\to$ 58.38% (Context-1 + Speaker)
* **Sadness ($n=187$):** **33.16%** (Baseline) $\to$ 16.04% (Context-1) $\to$ 20.32% (Context-1 + Speaker)
* **Surprise ($n=253$):** 60.87% (Baseline) $\to$ 60.08% (Context-1) $\to$ **66.01%** (Context-1 + Speaker)

**Core Scientific Finding:**
Adding raw conversational context ($u_{t-1}\nu_t$) causes the model to bias heavily toward the dominant majority class (`neutral`), increasing Neutral accuracy from 73.57% to 83.33%. However, this comes at the cost of minority emotion recall, causing Macro F1 to drop from 0.4400 to 0.3912. Adding explicit speaker tokens partially alleviates this issue: speaker identity distinguishes self-repetition from interlocutor responses, recovering minority performance (Surprise rises to 66.01%, Macro F1 recovers to 0.4078).

In qualitative error tracking:
- **Rescued Examples (259 turns):** Context-1 corrected an utterance misclassified by the baseline (e.g., short utterances like *"Push!"* disambiguated from fear to joy).
- **Broken Examples (229 turns):** Context-1 replaced a correct emotional prediction with neutral due to prior neutral context.

---

### 4. Training Split Evaluation (Model Fit & Generalization Gap)
To verify training convergence and quantify the generalization gap, all checkpoints were evaluated on the MELD training split ([`outputs/tables/training_split_evaluation.csv`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/tables/training_split_evaluation.csv)):

| Model Configuration | Evaluated Training Utterances | Accuracy | Macro F1 | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline** *(All Utterances)* | 9,989 | 85.88% | 0.7724 | 0.8602 |
| **Baseline** *(Matched Subset)* | 8,882 | 85.65% | 0.7703 | 0.8578 |
| **Context-1** | 8,882 | 87.99% | 0.7741 | 0.8747 |
| **Context-1 + Speaker** | 8,882 | **88.09%** | **0.7884** | **0.8772** |

#### Training Per-Class F1 Detail
- **Baseline (All):** Anger: 0.7615, Disgust: 0.7291, Fear: 0.5377, Joy: 0.8499, Neutral: 0.9314, Sadness: 0.7416, Surprise: 0.8559
- **Context-1 + Speaker:** Anger: 0.8087, Disgust: 0.6319, Fear: 0.6207, Joy: 0.8746, Neutral: 0.9409, Sadness: 0.7691, Surprise: 0.8730

**Analysis:** On the training set, the models exhibit high capacity (~88% accuracy, ~0.78 Macro F1), with speaker tags providing substantial gains on low-frequency classes (Fear F1 improved from 0.5312 to 0.6207). The difference between training Macro F1 (~0.78) and test Macro F1 (~0.41) highlights the challenge of generalizing subtle conversational emotion patterns in dialogue.

---

## Emotion Dynamics & Speaker Analysis

Using the sequential emotion predictions, empirical dialogue trajectories were analyzed ([`outputs/tables/week7_*`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/tables/) and [`outputs/tables/week8_*`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/tables/)):

### 1. Transition and Shift Rates
- **Overall Shift Rate on Test:** **58.08%** of contiguous turn transitions involve an emotion change (1,352 shifts across 2,328 transitions).
- **Dialogue Averages:** An average test dialogue has 5.04 emotion shifts (median 4.0).

### 2. Speaker Dynamics: Same-Speaker vs. Cross-Speaker

| Transition Type | Total Transitions | Emotion Shifts | Shift Rate | Persistence Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Same-Speaker** | 563 | 238 | 42.27% | **57.73%** |
| **Cross-Speaker** | 1,765 | 1,114 | **63.12%** | 36.88% |

**Insight:** Individuals exhibit emotional inertia when speaking multiple consecutive turns (57.73% persistence), whereas speaker switches trigger significantly higher emotional volatility (63.12% shift rate).

### 3. Turning Points
A **turning point** is defined as an emotional shift from emotion $A$ to $B$ that persists into turn $t+2$ ($A \to B \to B$). Top observed transitions include:
- `Joy → Neutral`: 61 occurrences
- `Anger → Neutral`: 47 occurrences
- `Neutral → Joy`: 44 occurrences
- `Surprise → Neutral`: 43 occurrences
- `Sadness → Neutral`: 32 occurrences

---

## Next-Emotion Forecasting Experiments

Forecasting aims to predict the emotional state of the *subsequent* turn $t+1$ without access to the future utterance text ([`outputs/tables/week9_forecasting_results.csv`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/tables/week9_forecasting_results.csv) & [`notebooks/25`](file:///home/chitta/Projects/emotion-dynamics-nlp/notebooks/25_week10_higher_order_markov.ipynb)):

| Forecasting Model | Accuracy | Macro F1 | Weighted F1 | Rationale / Behavior |
| :--- | :---: | :---: | :---: | :--- |
| **Majority Baseline** | 47.94% | 0.0926 | 0.3107 | Always predicts `neutral` |
| **Markov-1 (First Order)** | 47.94% | 0.0926 | 0.3107 | Transition matrix collapses to `neutral` due to dataset imbalance |
| **Persistence Baseline** | 41.92% | **0.2736** | **0.4186** | Predicts $e_{t+1} = e_t$; captures non-neutral persistence |
| **Markov-2 (Second Order)** | **48.81%** | 0.2003 | 0.3992 | Conditioned on previous 2 states $(e_{t-1}, e_t)$ ($N=2,059$) |
| **Markov-2 (Recursive)** | 44.24% | 0.1516 | 0.3488 | Multi-step unrolled forecast; 24.14% prediction divergence |

---

## Application Architecture & Features

The repository includes a production-grade FastAPI service and an interactive web interface:

```
[User Dialogue Input] (Speaker: Utterance)
         │
         ▼
[Conversation Parser] (src/data/conversation.py)
         │ Validates syntax, turns, contiguity
         ▼
[Inference Pipeline] (src/models/inference.py)
         │ Evaluates turns via DistilBERT Checkpoints
         ▼
[Dynamics Engine] (src/analysis/conversation_analysis.py)
         │ Computes transition matrix, shift rates, speaker dynamics, turning points
         ▼
[Interactive Dashboard] (static/index.html)
         ├─ Full Stacked Emotion Probability Trajectory (Chart.js)
         ├─ Conversation Dynamics & Shift Rate KPIs
         ├─ Turn-by-Turn Confidence Table
         ├─ Speaker Shift Breakdown
         └─ Sustained Turning Point Cards
```

---

## Installation & Execution Guide

### Running with Docker Compose (Recommended)

Docker Compose builds an isolated Python 3.12 environment containing all dependencies and checkpoints.

> **Checkpoint Prerequisite:** The three trained checkpoint directories must exist locally before building the image; they are excluded from Git (`.gitignore`):
> - `checkpoints/transformer_baseline/best/`
> - `checkpoints/transformer_context_1/best/`
> - `checkpoints/transformer_context_1_speaker/best/`

```bash
# 1. Clone the repository
git clone https://github.com/duckydodo/emotion-dynamics-nlp.git
cd emotion-dynamics-nlp

# 2. Launch the application
docker compose up -d

# 3. Access the web dashboard
Open http://localhost:8001 in your browser.

# 4. View API Documentation
Open http://localhost:8001/docs
```

To stop the container:
```bash
docker compose down
```

### Running Locally with Python

*Requires Python 3.12 (to match `pyproject.toml` and PyTorch 2.13 compatibility).*

```bash
# 1. Create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 2. Install pinned dependencies
pip install -r requirements.txt
pip install -e .

# 3. Start the application
uvicorn src.api:app --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000` in your browser.

---

## API Specification & Examples

### Endpoints
- `GET /health`: Health check and list of loaded checkpoints.
- `POST /predict`: Predicts turn-level emotions.
- `POST /analyze`: Performs full emotion classification and conversation dynamics analysis.

### Sample API Request (`POST /analyze`)
```bash
curl -X POST http://localhost:8001/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "context_1_speaker",
    "conversation": "Alice: I was really excited about the trip.\nBob: Me too. I thought everything was finally going well.\nAlice: Then why did you cancel the tickets?\nBob: I didn'\''t cancel them. The airline did."
  }'
```

### Sample Response Snippet
```json
{
  "mode": "context_1_speaker",
  "turns": [
    {"turn_id": 0, "speaker": "Alice", "text": "I was really excited about the trip.", "emotion": "joy", "confidence": 0.70},
    {"turn_id": 1, "speaker": "Bob", "text": "Me too. I thought everything was finally going well.", "emotion": "joy", "confidence": 0.74},
    {"turn_id": 2, "speaker": "Alice", "text": "Then why did you cancel the tickets?", "emotion": "anger", "confidence": 0.65},
    {"turn_id": 3, "speaker": "Bob", "text": "I didn't cancel them. The airline did.", "emotion": "neutral", "confidence": 0.71}
  ],
  "analysis": {
    "total_transitions": 3,
    "shifts": 2,
    "shift_rate": 0.667,
    "same_speaker_transitions": 0,
    "cross_speaker_transitions": 3,
    "turning_points": [
      {
        "from_emotion": "joy",
        "to_emotion": "anger",
        "turn_id": 2,
        "speaker_from": "Bob",
        "speaker_to": "Alice"
      }
    ]
  }
}
```

---

## Repository Structure

```
emotion-dynamics-nlp/
├── checkpoints/                        # Model weights (excluded from Git; included in Docker)
│   ├── transformer_baseline/best/      # DistilBERT utterance model (256MB)
│   ├── transformer_context_1/best/     # DistilBERT Context-1 model (256MB)
│   └── transformer_context_1_speaker/  # DistilBERT Context-1 + Speaker model (256MB)
├── data/                               # Local dataset storage (excluded from Git)
│   └── raw/meld/                       # Official MELD train/dev/test CSV files
├── docs/                               # Project documentation and decision logs
│   └── gate1_dataset_selection.md      # Candidate dataset comparison analysis
├── notebooks/                          # Jupyter experiment notebooks (Weeks 1–10)
│   ├── 01_dailydialog_eda.ipynb ... 03_emowoz_eda.ipynb
│   ├── 10_majority_baseline.ipynb ... 14_tfidf_ngram_experiments.ipynb
│   ├── 15_transformer_baseline.ipynb ... 19_context_1_speaker.ipynb
│   ├── 20_transformer_error_analysis.ipynb ... 21_week6_error_analysis.ipynb
│   └── 22_week7_trajectory_analysis.ipynb ... 25_week10_higher_order_markov.ipynb
├── outputs/                            # Persisted research outputs
│   ├── experiments/                    # Model metrics.json and confusion matrices
│   ├── figures/                        # Transition heatmaps and dynamics plots
│   └── tables/                         # CSV comparison tables and methodology notes
│       ├── training_split_evaluation.csv
│       ├── week6_transformer_comparison.csv
│       └── week9_forecasting_results.csv
├── scripts/                            # Reproducible automation scripts
│   └── evaluate_training_split.py      # Standalone training set evaluation script
├── src/                                # Core Python source library
│   ├── analysis/                       # Dynamics, trajectories, forecasting
│   ├── application/                    # Pipeline orchestration
│   ├── data/                           # Loaders, context builders, conversation parser
│   ├── evaluation/                     # Metrics, comparisons, confusion matrices
│   ├── features/                       # TF-IDF extractors
│   ├── models/                         # Classifiers, trainer, and inference engine
│   └── api.py                          # FastAPI web application
├── static/                             # Frontend assets
│   └── index.html                      # Interactive dashboard UI
├── compose.yaml                        # Docker Compose deployment specification
├── Dockerfile                          # Multi-stage Docker build recipe
└── requirements.txt                    # Pinned Python dependencies
```

---

## Limitations & Open Challenges

1. **Extreme Class Imbalance:**
   `neutral` accounts for 47% of MELD utterances. Models easily latch onto neutral context, suppressing rare emotions like `fear` (50 test instances) and `disgust` (68 test instances).
2. **Train–Test Generalization Gap:**
   While models achieve ~88% accuracy and ~0.78 Macro F1 on the training split, held-out test performance is ~61% accuracy and ~0.41 Macro F1. Multi-party dialogue exhibits significant stylistic variance across different conversational scenes.
3. **Domain Specificity & Sitcom Bias:**
   MELD consists of scripts from the TV sitcom *Friends*. The dialogue is theatrical, highly punchy, and dramatic. True out-of-distribution generalization across everyday conversation (DailyDialog) or task-oriented dialogue (EmoWOZ) has not been benchmarked.
4. **Speaker Tag Delimitation:**
   In `context_1_speaker`, speaker names observed during training were limited to recurring characters (e.g., Joey, Rachel). When tested on custom names (e.g., Alice, Bob), the tokens serve primarily as turn-boundary indicators rather than learned speaker representations.
5. **Acoustic and Visual Modalities Excluded:**
   This project focuses strictly on NLP text processing. Non-verbal signals (pitch, vocal intensity, facial expressions), which frequently carry the emotion in sarcastic or deadpan remarks, are not modeled.

---

## Reproducibility & Checkpoints

To replicate the training set evaluation reported in Section 5:
```bash
# Run the evaluation script inside Docker
docker run --rm \
  -e PYTHONPATH=/app \
  -w /app \
  -v $(pwd):/app_mount \
  emotion-dynamics-nlp:week11 \
  python /app_mount/scripts/evaluate_training_split.py
```
This regenerates [`outputs/tables/training_split_evaluation.csv`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/tables/training_split_evaluation.csv) and [`outputs/tables/training_split_evaluation_notes.txt`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/tables/training_split_evaluation_notes.txt).

---

## References & Acknowledgments

- **MELD Dataset:** Poria et al. (2019). *MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations*. ACL 2019.
- **DistilBERT:** Sanh et al. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*. NeurIPS EMC^2 Workshop.
- **Hugging Face Transformers:** Wolf et al. (2020). *Transformers: State-of-the-Art Natural Language Processing*. EMNLP 2020.
