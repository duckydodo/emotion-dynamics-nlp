# Emotion Dynamics
*Conversational Emotion Analysis and Research in Multi-Party Dialogue*

### Author
- **Chittaranjan Dutta** (B.Tech in Artificial Intelligence & Data Science, IIITDM Kurnool)

---

### Checkpoint & Dataset Notice
> **IMPORTANT**:
> In compliance with GitHub's file size limits, large model checkpoints (`checkpoints/`), weight tensors (`*.safetensors`, `*.pt`, `*.bin`, `*.npz`), and raw conversational corpora (`data/raw/`, `data/processed/`) are **strictly excluded from version control** via `.gitignore`.
> The repository provides full reproducible extraction and training pipelines (`scripts/`) alongside all verified benchmark tables, statistical tests, and research artifacts in `outputs/`.

---

### Quickstart (Docker Deployment)
To build and run the complete application in a single isolated command:

```bash
docker compose up --build
```

Access the interactive five-screen research application in your browser:
- **Application Interface:** `http://localhost:8001`
- **FastAPI OpenAPI Documentation:** `http://localhost:8001/docs`
- **API Health Check:** `http://localhost:8001/health`

---

## Table of Contents
1. [Project Overview & Motivation](#project-overview--motivation)
2. [Research Questions & Objectives](#research-questions--objectives)
3. [Dataset Description & Preprocessing](#dataset-description--preprocessing)
4. [Model Architecture & Methodology](#model-architecture--methodology)
   - [Transformer Context Models](#1-transformer-context-models-distilbert)
   - [Autoregressive History Models](#2-autoregressive-history-models-model-a--b)
   - [DialogueRNN: Attentive Recurrent Speaker Tracking](#3-dialoguernn-attentive-recurrent-speaker-tracking)
5. [Evaluation Results & Canonical Benchmarks](#evaluation-results--canonical-benchmarks)
   - [Classical Baselines](#1-classical-n-gram-baselines-meld-test-split)
   - [Canonical Matched Cross-Model Leaderboard (Cohort A: N=2,328)](#2-canonical-matched-cross-model-leaderboard-cohort-a-n--2328)
   - [The "Accuracy Illusion" & Minority Collapse](#3-the-accuracy-illusion--minority-collapse)
   - [Transition Dynamics: Shifts vs. Inertia](#4-transition-dynamics-shifts-vs-inertia)
   - [Statistical Significance Testing](#5-statistical-significance-testing)
   - [Training Split Evaluation](#6-training-split-evaluation-model-fit--generalization-gap)
6. [Emotion Dynamics & Speaker Analysis](#emotion-dynamics--speaker-analysis)
7. [Next-Emotion Forecasting Experiments](#next-emotion-forecasting-experiments)
8. [Application Architecture & Features](#application-architecture--features)
9. [Installation & Execution Guide](#installation--execution-guide)
   - [Running with Docker (Recommended)](#running-with-docker-recommended)
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

The project explores three distinct modeling paradigms for conversational emotion recognition:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        NLP MODELING PARADIGMS                          │
├─────────────────────────┬──────────────────────────┬───────────────────┤
│ 1. Context Concatenation│ 2. Autoregressive History│ 3. Recurrent State│
│    (DistilBERT Encoder) │    (Recursive History)   │    (DialogueRNN)  │
│  [u_{t-1} [SEP] u_t]    │ [prev_e [SEP] u_{t-1}..] │ Party GRU + Attn  │
└─────────────────────────┴──────────────────────────┴───────────────────┘
```

### 1. Transformer Context Models (DistilBERT)
Built on `distilbert-base-uncased` (6 layers, 768 hidden dimensions, 12 attention heads, ~66.96M parameters) fine-tuned with AdamW (`lr=2e-5`, batch size 4, max length 128, early stopping on dev Macro F1):
- **Utterance Baseline (`transformer_baseline`):** Independent turn $u_t$.
- **Context-1 (`transformer_context_1`):** Two-turn window $[u_{t-1} \text{ [SEP] } u_t]$.
- **Context-1 + Speaker (`transformer_context_1_speaker`):** Speaker-prefixed turns $[\text{Speaker}_{t-1}\text{: } u_{t-1} \text{ [SEP] Speaker}_t\text{: } u_t]$.
- **Context-3 & Context-5:** Multi-turn windows of 3 and 5 contiguous preceding turns.

### 2. Autoregressive History Models (Model A & Model B)
- **Model A (Emotion History):** Prepends the previous turn's emotion label token $\langle\text{prev\_}y_{t-1}\rangle$ before $u_t$.
- **Model B (Combined Text + Emotion History):** Prepends both $\langle\text{prev\_}y_{t-1}\rangle$ and $u_{t-1}$ before $u_t$.
- **Inference Evaluation:** Evaluated in both idealized *teacher-forced oracle* mode (using ground-truth prior emotions) and true *causal autoregressive* mode (recursively conditioning on the model's own predictions $\hat{y}_{t-1}$).

### 3. DialogueRNN: Attentive Recurrent Speaker Tracking
Implements the architecture from **Majumder et al. (AAAI 2019)** ([`src/models/dialoguernn.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/models/dialoguernn.py)):
- **Utterance Encoder:** Frozen 768-dimensional `[CLS]` embeddings pre-extracted from the fine-tuned DistilBERT baseline ([`scripts/extract_dialogue_features.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/scripts/extract_dialogue_features.py)).
- **Global Context GRU ($D_g = 150$):** Maintains the evolving dialogue context across all turns.
- **Party State GRUs ($D_p = 100$):** Maintains a separate latent state for each conversational participant $s \in \mathcal{S}$. Only the active speaker's state is updated at turn $t$.
- **Context Attention ($D_g = 150$):** Queries past global context states using the current party state vector.
- **Emotion GRU ($D_e = 50$):** Models the speaker's emotional trajectory conditioned on the party state.
- **Classification Head:** Linear projection on $[h_t \,\|\, q_t^s \,\|\, a_t]$ (300 dimensions $\to$ 7 classes, 670,855 total parameters).
- **Training Regimes:**
  - *Unweighted:* Trained with standard cross-entropy loss.
  - *Class-Weighted:* Trained with inverse class frequencies computed strictly from training split distributions.

### Research Scope & Roadmap
- **Completed Work:** Classical baselines, DistilBERT context models, Autoregressive history models, DialogueRNN (unweighted & weighted), Canonical evaluation audit, Web application & Docker containerization.
- **Planned / Future Work:** Relational graph neural networks (DialogueGCN) to address multi-party non-sequential dependencies; multimodal fusion (acoustic + visual features).

---

## Evaluation Results & Canonical Benchmarks

To eliminate sample-size discrepancies caused by dialogue gaps (`Utterance_ID` skips in Dialogue 93 and 108), all models are evaluated against the canonical cohorts established in the [Evaluation Audit](outputs/reports/canonical_evaluation_audit.md):
- **Cohort A ($N=2,328$):** Strict contiguous matched subset where $t \ge 1$ and `Utterance_ID == prev_Utterance_ID + 1`. **The canonical benchmark for comparing contextual models.**
- **Cohort B ($N=2,330$):** All subsequent sequential turns ($t \ge 1$).
- **Cohort C ($N=280$):** Dialogue-initial turns ($t = 0$).
- **Cohort D ($N=2,610$):** Full MELD test set.

### 1. Classical N-Gram Baselines (MELD Test Split)
Evaluated on all 2,610 held-out test utterances ([`outputs/experiments/`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/experiments/)):

| Model | Representation | Accuracy | Macro F1 | Weighted F1 |
| :--- | :--- | :---: | :---: | :---: |
| **Majority Baseline** | Predicts `neutral` | 48.12% | 0.0928 | 0.3127 |
| **Logistic Regression** | TF-IDF Unigram | 52.68% | 0.2410 | 0.4444 |
| **Logistic Regression** | TF-IDF Bigram | **53.18%** | 0.2470 | 0.4432 |
| **Linear SVM** | TF-IDF Bigram | 50.84% | **0.2874** | 0.4641 |
| **Linear SVM** | TF-IDF Trigram | 51.07% | 0.2822 | **0.4647** |

### 2. Canonical Matched Cross-Model Leaderboard (Cohort A: $N = 2,328$)
Evaluated on the exact matched subset ([`outputs/tables/dialoguernn_comparison.csv`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/tables/dialoguernn_comparison.csv)):

| Model Configuration | Architecture | Accuracy | Macro F1 | Weighted F1 | Neutral Rec | Fear Rec | Disgust Rec | Anger Rec | Joy Rec | Sadness Rec | Surprise Rec |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DistilBERT Baseline** | Utterance Only | 59.45% | 0.4314 | 0.5969 | 73.57% | 25.00% | 20.63% | 36.81% | 58.66% | 33.16% | 60.87% |
| **DistilBERT Context-1** | Text Context ($t-1$) | 60.74% | 0.3912 | 0.5812 | 83.33% | 6.82% | 14.29% | 30.94% | 54.47% | 16.04% | 60.08% |
| **DistilBERT Context-1+Spk** | Text Context + Speaker | 61.94% | 0.4078 | 0.5981 | 82.35% | 11.36% | 11.11% | 31.60% | 58.38% | 20.32% | 66.01% |
| **Model A (Autoregressive)** | History Token | 61.47% | 0.3930 | 0.5984 | 78.94% | 2.27% | 4.76% | 39.09% | 63.41% | 28.34% | 57.71% |
| **Model B (Autoregressive)** | Text + History Token | **62.59%** | 0.3848 | 0.6024 | 82.35% | **0.00%** | **3.17%** | 43.00% | 53.63% | 22.46% | 67.19% |
| **DialogueRNN (Unweighted)** | Attentive Recurrent | 60.14% | **0.4342** | **0.6025** | 72.85% | 15.91% | 15.87% | **46.91%** | 57.82% | **35.83%** | 60.08% |
| **DialogueRNN (Weighted)** | Recurrent + Inv. Weight | 59.88% | 0.4284 | 0.5944 | 77.96% | **25.00%** | **25.40%** | 35.50% | 50.84% | 27.81% | 60.87% |

### 3. The "Accuracy Illusion" & Minority Collapse
A critical finding of this research is that naive accuracy is heavily misleading on imbalanced conversational corpora:
1. **The Spurious Gain:** Model B achieves the highest raw accuracy (**62.59%**), yet exhibits the lowest Macro F1 (**0.3848**). It achieves high accuracy solely by exploiting majority bias: predicting `neutral` on 54.04% of turns and driving neutral recall to 82.35%.
2. **Minority Extinction:** Context-1 and Model B catastrophically collapse minority classes: Model B achieves **0.0% recall on fear** (0/44 turns detected) and **3.2% recall on disgust** (2/63 detected).
3. **DialogueRNN Resolution:** DialogueRNN explicitly decouples party states, predicting neutral on 44.63% of turns (matching ground truth 47.94%). It restores Macro F1 to **0.4342** (unweighted) and recovers minority recall to **25.00% for fear** and **25.40% for disgust** (class-weighted).

### 4. Transition Dynamics: Shifts vs. Inertia
Evaluated on Cohort A ($N=2,328$) partitioned into Emotion Shifts ($N=1,352$) and Emotion Inertia ($N=976$) ([`outputs/tables/dialoguernn_transition_metrics.csv`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/tables/dialoguernn_transition_metrics.csv)):

| Model Configuration | Shift Accuracy | Shift Macro F1 | Inertia Accuracy | Inertia Macro F1 | Speaker Change Macro F1 | Same Speaker Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DistilBERT Baseline** | 54.59% | 0.4299 | 66.19% | 0.3977 | 0.4327 | 0.3960 |
| **DistilBERT Context-1** | 53.11% | 0.3747 | 71.31% | 0.3882 | 0.3820 | 0.3899 |
| **DistilBERT Context-1 + Spk** | 56.14% | 0.4020 | 69.98% | 0.3746 | 0.3968 | 0.3976 |
| **Model B (Autoregressive)** | 55.18% | 0.3624 | 72.85% | 0.4064 | 0.3740 | 0.3922 |
| **DialogueRNN (Unweighted)** | **55.03%** | **0.4271** | 67.21% | **0.4155** | **0.4267** | **0.4230** |
| **DialogueRNN (Weighted)** | 53.33% | 0.4133 | 68.95% | 0.4135 | 0.4166 | 0.4099 |

*DialogueRNN maintains high discrimination during emotional turning points (Shift Macro F1 = 0.4271), whereas Context-1 and Model B collapse to 0.3747 and 0.3624 due to preceding neutral inertia.*

### 5. Statistical Significance Testing
Paired tests on Cohort A ($N=2,328$) using McNemar's test with Edwards continuity correction and paired bootstrap resampling (10,000 iterations) ([`outputs/tables/dialoguernn_statistical_tests.csv`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/tables/dialoguernn_statistical_tests.csv)):
- **DialogueRNN vs Context-1:** $\Delta \text{Macro F1} = \mathbf{+0.0430}$ ($p = 0.0004$, 95% CI: $[+0.0182, +0.0690]$) — **Highly Significant**
- **DialogueRNN vs Context-1 + Speaker:** $\Delta \text{Macro F1} = \mathbf{+0.0264}$ ($p = 0.0396$, 95% CI: $[+0.0014, +0.0516]$) — **Significant**
- **DialogueRNN vs Model B (Autoregressive):** $\Delta \text{Macro F1} = \mathbf{+0.0494}$ ($p = 0.0004$, 95% CI: $[+0.0215, +0.0773]$) — **Highly Significant**
- **DialogueRNN vs DistilBERT Baseline:** $\Delta \text{Macro F1} = +0.0028$ ($p = 0.7382$, 95% CI: $[-0.0143, +0.0196]$) — Equivalent Overall F1

### 6. Training Split Evaluation (Model Fit & Generalization Gap)
To verify training convergence and quantify the generalization gap, checkpoints were evaluated on the MELD training split ([`outputs/tables/training_split_evaluation.csv`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/tables/training_split_evaluation.csv)):

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

The repository includes a production-grade FastAPI service and an interactive web interface supporting four inference engines:

```
[User Dialogue Input] (Speaker: Utterance)
         │
         ▼
[Conversation Parser] (src/data/conversation.py)
         │ Validates syntax, turns, contiguity
         ▼
[Inference Pipeline] (src/models/inference.py)
         ├─ DistilBERT Baseline (Utterance only)
         ├─ Context-1 / Context-1 + Speaker (Transformer window)
         └─ DialogueRNN (Attentive Recurrent Party State Tracking)
         ▼
[Dynamics Engine] (src/analysis/conversation_analysis.py)
         │ Computes transition matrix, shift rates, speaker dynamics, turning points
         ▼
[Interactive Dashboard] (static/index.html)
         ├─ Model Switcher (Baseline, Context-1, Context-1+Spk, DialogueRNN)
         ├─ Emotion Trajectory & Probability Breakdown (Chart.js)
         ├─ Conversation Dynamics & Shift Rate KPIs
         ├─ Turn-by-Turn Confidence Table
         ├─ Speaker Shift Breakdown
         └─ Sustained Turning Point Cards
```

---

## Installation & Execution Guide

### Running with Docker (Recommended)

The project is packaged in a self-contained, CPU-optimized Docker image (`emotion-dynamics-nlp:week11`):

```bash
# 1. Run the containerized web application
docker run -d --name emotion-app -p 8000:8000 emotion-dynamics-nlp:week11

# 2. Access the web dashboard
# Open http://localhost:8000 in your browser.

# 3. View API Documentation
# Open http://localhost:8000/docs
```

Or using Docker Compose:
```bash
docker compose up --build
# Access at http://localhost:8001
```

### Running Locally with Python

*Requires Python 3.10+ (tested on Linux x86_64 with CPU-only PyTorch).*

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
pip install -e .

# 3. Start the application
uvicorn src.api:app --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000` in your browser.

---

## API Specification & Examples

### Supported Models (`mode`)
- `"baseline"`: Utterance-only DistilBERT baseline.
- `"context_1"`: DistilBERT with preceding utterance text.
- `"context_1_speaker"`: DistilBERT with speaker prefixes.
- `"dialoguernn"`: Attentive recurrent speaker-state tracking (DialogueRNN).

### Endpoints
- `GET /health`: Health check and list of loaded checkpoints.
- `POST /predict`: Predicts turn-level emotions and class probabilities.
- `POST /analyze`: Performs full emotion classification and conversation dynamics analysis.

### Sample API Request (`POST /analyze` with DialogueRNN)
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "dialoguernn",
    "conversation": "Alice: I was really excited about the trip.\nBob: Me too. I thought everything was finally going well.\nAlice: Then why did you cancel the tickets?\nBob: I didn'\''t cancel them. The airline did."
  }'
```

### Sample Response Snippet
```json
{
  "mode": "dialoguernn",
  "turns": [
    {"turn_id": 0, "speaker": "Alice", "text": "I was really excited about the trip.", "emotion": "joy", "confidence": 0.68},
    {"turn_id": 1, "speaker": "Bob", "text": "Me too. I thought everything was finally going well.", "emotion": "joy", "confidence": 0.72},
    {"turn_id": 2, "speaker": "Alice", "text": "Then why did you cancel the tickets?", "emotion": "anger", "confidence": 0.64},
    {"turn_id": 3, "speaker": "Bob", "text": "I didn't cancel them. The airline did.", "emotion": "neutral", "confidence": 0.70}
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
│   ├── transformer_context_1_speaker/  # DistilBERT Context-1 + Speaker model (256MB)
│   ├── dialoguernn_unweighted/         # DialogueRNN unweighted checkpoint (2.7MB)
│   └── dialoguernn_weighted/           # DialogueRNN class-weighted checkpoint (2.7MB)
├── data/                               # Local dataset storage (excluded from Git)
│   ├── raw/meld/                       # Official MELD train/dev/test CSV files
│   └── processed/dialoguernn/          # Pre-extracted DistilBERT CLS embeddings (.npz)
├── notebooks/                          # Jupyter experiment notebooks (Weeks 1–10)
├── outputs/                            # Persisted research outputs
│   ├── experiments/                    # Model metrics.json and training histories
│   ├── figures/                        # Transition heatmaps and multi-panel confusion matrices
│   ├── reports/                        # Formal research reports & audit documentation
│   │   ├── canonical_evaluation_audit.md
│   │   ├── dialoguernn_error_analysis.md
│   │   ├── dialoguernn_implementation_report.md
│   │   ├── dialoguernn_training_report.md
│   │   └── research_progress_report.md
│   └── tables/                         # CSV benchmark tables and statistical significance tests
│       ├── canonical_test_comparison.csv
│       ├── dialoguernn_comparison.csv
│       ├── dialoguernn_per_class.csv
│       ├── dialoguernn_statistical_tests.csv
│       └── dialoguernn_transition_metrics.csv
├── scripts/                            # Reproducible automation scripts
│   ├── extract_dialogue_features.py    # DistilBERT CLS feature extraction
│   ├── train_dialoguernn.py            # DialogueRNN training pipeline
│   ├── evaluate_dialoguernn_analysis.py# Canonical cohort evaluation & error analysis
│   └── evaluate_training_split.py      # Training set evaluation script
├── src/                                # Core Python source library
│   ├── analysis/                       # Dynamics, trajectories, forecasting
│   ├── application/                    # Pipeline orchestration & EmotionAnalyzer
│   ├── data/                           # Loaders, context builders, conversation parser
│   ├── evaluation/                     # Metrics, comparisons, confusion matrices
│   ├── models/                         # Classifiers, DialogueRNN, trainer, inference engine
│   └── api.py                          # FastAPI web application
├── static/                             # Frontend assets
│   └── index.html                      # Interactive dashboard UI with model switcher
├── tests/                              # Unit and integration test suite
│   ├── test_dialoguernn.py             # 11 unit tests for DialogueRNN architecture
│   └── test_application_integration.py # 10 integration tests for inference & API
├── compose.yaml                        # Docker Compose deployment specification
├── Dockerfile                          # Multi-stage Docker build recipe
└── requirements.txt                    # Pinned Python dependencies
```

---

## Limitations & Open Challenges

1. **Extreme Class Imbalance:**
   `neutral` accounts for 47% of MELD utterances. While DialogueRNN successfully avoids majority collapse, detecting rare emotions (`fear` and `disgust`) without false-positive inflation remains challenging.
2. **Train–Test Generalization Gap:**
   While models achieve ~88% accuracy and ~0.78 Macro F1 on the training split, held-out test performance is ~60% accuracy and ~0.43 Macro F1. Multi-party dialogue exhibits significant stylistic variance across different conversational scenes.
3. **Domain Specificity & Sitcom Bias:**
   MELD consists of scripts from the TV sitcom *Friends*. The dialogue is theatrical, highly punchy, and dramatic. True out-of-distribution generalization across everyday conversation (DailyDialog) or task-oriented dialogue (EmoWOZ) has not been benchmarked.
4. **Sequential Recurrent Window vs. Graph Structure:**
   DialogueRNN processes turns in a linear sequence ($t-1 \to t$). In multi-party conversations ($>3$ participants), an utterance often addresses a turn spoken several steps earlier by another speaker, motivating graph neural networks (DialogueGCN).
5. **Acoustic and Visual Modalities Excluded:**
   This project focuses strictly on NLP text processing. Non-verbal signals (pitch, vocal intensity, facial expressions), which frequently carry the emotion in sarcastic or deadpan remarks, are not modeled.

---

## Reproducibility & Checkpoints

To replicate the complete DialogueRNN research pipeline from scratch:

```bash
# 1. Feature Extraction (Extract 768-d DistilBERT CLS embeddings)
python scripts/extract_dialogue_features.py \
    --model_dir checkpoints/transformer_baseline/best \
    --data_dir data/raw/meld \
    --output_dir data/processed/dialoguernn

# 2. Train DialogueRNN (Unweighted)
python scripts/train_dialoguernn.py \
    --features_dir data/processed/dialoguernn \
    --raw_data_dir data/raw/meld \
    --output_dir outputs/experiments/dialoguernn_unweighted \
    --checkpoint_dir checkpoints/dialoguernn_unweighted \
    --seed 42

# 3. Train DialogueRNN (Class-Weighted)
python scripts/train_dialoguernn.py \
    --features_dir data/processed/dialoguernn \
    --raw_data_dir data/raw/meld \
    --output_dir outputs/experiments/dialoguernn_weighted \
    --checkpoint_dir checkpoints/dialoguernn_weighted \
    --class_weighted \
    --seed 42

# 4. Canonical Evaluation & Error Analysis
python scripts/evaluate_dialoguernn_analysis.py \
    --raw_test data/raw/meld/test_sent_emo.csv \
    --features_test data/processed/dialoguernn/test_features.npz \
    --unweighted_checkpoint checkpoints/dialoguernn_unweighted/best_model.pt \
    --weighted_checkpoint checkpoints/dialoguernn_weighted/best_model.pt \
    --output_dir outputs

# 5. Run Test Suite
python -m unittest tests/test_dialoguernn.py tests/test_application_integration.py
```

---

## References & Acknowledgments

- **DialogueRNN:** Majumder, N., Poria, S., Hazarika, D., Mihalcea, R., Gelbukh, A., & Cambria, E. (2019). *DialogueRNN: An Attentive RNN for Emotion Detection in Conversations*. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2019), 33(01), 6818-6825.
- **MELD Dataset:** Poria, S., Hazarika, D., Majumder, N., Naik, G., Cambria, E., & Mihalcea, R. (2019). *MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations*. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL 2019), 527-536.
- **DistilBERT:** Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*. NeurIPS EMC^2 Workshop.
- **Hugging Face Transformers:** Wolf et al. (2020). *Transformers: State-of-the-Art Natural Language Processing*. EMNLP 2020.
