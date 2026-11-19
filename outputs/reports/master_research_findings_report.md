# Emotion Recognition and Dynamics in Multi-Party Conversations
**Comprehensive Research Findings, Architectural Evaluations, and Clinical/Ethical Boundaries**

**Document Version:** 1.0.0 (Final Submission)  
**Date:** November 2026  
**Author:** Chittaranjan Dutta (B.Tech in Artificial Intelligence & Data Science, IIITDM Kurnool)  
**Target Dataset:** Multimodal EmotionLines Dataset (MELD, Text Modality)  
**Evaluated Architectures:** DistilBERT (Utterance-only), DialogueRNN (Recurrent), DialogueGCN (Relational Graph)  
**Exploratory Models:** Temporal CNN, Linear-Chain CRF  

---

## Table of Contents
1. [Research Motivation and Problem Definition](#1-research-motivation-and-problem-definition)
2. [Research Questions](#2-research-questions)
3. [Dataset Selection Rationale](#3-dataset-selection-rationale)
4. [Cross-Dataset Comparison: MELD, DailyDialog, and EmoWOZ](#4-cross-dataset-comparison-meld-dailydialog-and-emowoz)
5. [Exploratory Data Analysis (EDA)](#5-exploratory-data-analysis-eda)
6. [Official Dialogue-Level Splits](#6-official-dialogue-level-splits)
7. [Missing Utterance-ID Gaps and Evaluation Cohort Formulation](#7-missing-utterance-id-gaps-and-evaluation-cohort-formulation)
8. [Emotion Distribution and Transition Statistics](#8-emotion-distribution-and-transition-statistics)
9. [DistilBERT Architecture, Training, and Checkpoint Selection](#9-distilbert-architecture-training-and-checkpoint-selection)
10. [Contextual Modelling Experiments (Context-1, Context-1+Speaker)](#10-contextual-modelling-experiments)
11. [DialogueRNN Architecture, Party Tracking, and Implementation](#11-dialoguernn-architecture-party-tracking-and-implementation)
12. [DialogueGCN Architecture, Relational Topology, and Edge Logic](#12-dialoguegcn-architecture-relational-topology-and-edge-logic)
13. [Exploratory Sequence Models: Temporal CNN and Linear-Chain CRF](#13-exploratory-sequence-models-temporal-cnn-and-linear-chain-crf)
14. [Emotion Dynamics: Inertia, Resonance, and Turning Points](#14-emotion-dynamics-inertia-resonance-and-turning-points)
15. [Next-Emotion Forecasting and Empirical Sequence Baselines](#15-next-emotion-forecasting-and-empirical-sequence-baselines)
16. [Explainability Methods and Attributions](#16-explainability-methods-and-attributions)
17. [Application Architecture and Capabilities](#17-application-architecture-and-capabilities)
18. [Methodological and Domain Limitations](#18-methodological-and-domain-limitations)
19. [Future Multimodal Research Scope: Tri-Modal Affective Fusion](#19-future-multimodal-research-scope-tri-modal-affective-fusion)
20. [References](#20-references)

---

## 1. Research Motivation and Problem Definition

Most conventional emotion classification systems operate on isolated, single-turn text representations. In natural interpersonal dialogue, however, human emotional expression is fundamentally situated, dynamic, and interdependent:
* An utterance such as *"Yeah, right"* can express joyful agreement, neutral acknowledgment, or intense disgust/sarcasm depending entirely on preceding turns.
* Speakers maintain personal emotional momentum (**emotional inertia**) while simultaneously reacting to social stimuli introduced by interlocutors (**emotional resonance / cross-speaker influence**).
* Multi-party conversations introduce complex conversational topologies where turn-taking is asynchronous, multiple conversational threads intertwine, and speaker dynamics diverge.

This research investigates the degree to which multi-turn conversational context—modeled sequentially via Recurrent Neural Networks (DialogueRNN) and topologically via Graph Convolutional Networks (DialogueGCN)—statistically improves upon fine-tuned pretrained language models (DistilBERT) for conversational emotion recognition (ERC), and how these models illuminate temporal emotion dynamics.

---

## 2. Research Questions

* **RQ1 (Contextual Gain):** Does incorporating conversational history and speaker identity significantly improve emotion recognition over an isolated utterance-level transformer baseline?
* **RQ2 (Recurrent vs. Graph Topology):** Does explicit graph-relational modeling of speaker interaction (DialogueGCN) outperform sequential recurrent party-state tracking (DialogueRNN)?
* **RQ3 (Temporal Dynamics Modeling):** How accurately can conversational dynamics (emotional inertia, transition matrices, and turning points) be identified and traced across multi-party exchanges?
* **RQ4 (Model Disagreement & Context Sensitivity):** On which specific dialogue turns and emotion classes does conversational context overturn utterance-only misclassifications?
* **RQ5 (Domain Shift & Explainability):** How do token-level and context-level attribution methods perform when explaining emotion predictions on real-world conversations outside the scripted television domain?

---

## 3. Dataset Selection Rationale

The project standardizes on the **Multimodal EmotionLines Dataset (MELD)** (Chen et al., 2018; Poria et al., 2019):
* **Multi-Party Interaction:** Features conversations involving 2 to 9 concurrent speakers (from the television series *Friends*), reflecting authentic multi-party group dynamics rather than synthetic dyadic scripts.
* **Emotion Taxonomy:** Annotates utterances across the 7 Ekman-derived affective categories: *Neutral, Joy, Surprise, Anger, Sadness, Disgust, Fear*.
* **Dialogue Granularity:** Entire conversational scenes are preserved in chronological sequence, enabling rigorous evaluation of sequential and graph-based dialogue models.
* **Textual Modality Focus:** While MELD provides audio and video clips, our primary investigation isolates the text modality to establish rigorous NLP baselines before introducing multimodal acoustic fusion.

---

## 4. Cross-Dataset Comparison: MELD, DailyDialog, and EmoWOZ

| Dataset Dimension | MELD (Primary Benchmark) | DailyDialog (Li et al., 2017) | EmoWOZ (Feng et al., 2022) |
| :--- | :--- | :--- | :--- |
| **Domain** | Multi-party television comedy (*Friends*) | Everyday daily communication | Task-oriented customer service wizard-of-oz |
| **Interaction Structure**| Multi-party (up to 9 speakers per dialogue) | Strictly dyadic (2 speakers alternating) | Strictly dyadic (User and System) |
| **Emotion Classes** | 7 classes (Ekman + Neutral) | 7 classes (Ekman + Neutral) | 7 classes (Task-adapted: dissatisfaction, etc.)|
| **Class Imbalance** | Moderate (*Neutral* ~47%, *Disgust* ~2.7%) | Extreme (*Neutral* >83%, *Fear* <0.2%) | High (*Neutral* >70%) |
| **Dialogue Count** | 1,433 dialogues (13,708 utterances) | 13,118 dialogues (102,980 utterances) | 11,057 dialogues (83,568 utterances) |
| **Why MELD?** | **Only multi-party dataset with complex group speaker graphs** | Too skewed toward Neutral; only 2 speakers | Constrained task-oriented dialogue; no group dynamics |

---

## 5. Exploratory Data Analysis (EDA)

### 5.1 Utterance Length Distribution
* **Mean Utterance Length:** 8.02 words (median: 6 words).
* **Token Length (DistilBERT WordPiece):** 99.4% of all utterances contain $\le 64$ tokens. Setting `max_length=128` comfortably covers the maximum dialogue turn without truncation.
* **Monosyllabic Utterances:** Over 18% of utterances consist of 3 words or fewer (e.g., *"Okay"*, *"Push!"*, *"What?"*), representing the highest-ambiguity turns where conversational context is indispensable.

### 5.2 Speaker Distribution
* Six primary core characters (Rachel, Monica, Phoebe, Joey, Chandler, Ross) account for 73.4% of all utterances.
* Over 250 guest/auxiliary speakers appear throughout the dataset, necessitating speaker-independent role encoding rather than static one-hot speaker embeddings.

---

## 6. Official Dialogue-Level Splits

To prevent conversational leakage, all train, development, and test splits are strictly partitioned at the **dialogue level**:

| Partition Split | Dialogues | Utterances | Mean Turns / Dialogue |
| :--- | :---: | :---: | :---: |
| **Train** | 1,038 | 9,989 | 9.62 |
| **Development (Dev)** | 114 | 1,109 | 9.73 |
| **Held-Out Test** | 280 | 2,610 | 9.32 |
| **Total Corpus** | 1,432 | 13,708 | 9.57 |

*Note: Random utterance-level splitting is strictly prohibited as it destroys conversational history and leaks sequential party states across splits.*

---

## 7. Missing Utterance-ID Gaps and Evaluation Cohort Formulation

A critical methodological finding in this research relates to missing utterance IDs in the raw MELD CSV files:

### 7.1 Root Cause of Missing Gaps
During MELD's video extraction from *Friends* DVDs, certain intermediate utterances were discarded due to visual occlusion or audio overlap. For example, a dialogue might transition directly from `Utterance_ID = 0` to `Utterance_ID = 2`, omitting `Utterance_ID = 1`.

### 7.2 The Two Test Cohorts
* **Full Test Set ($N=2,610$ across 280 dialogues):** Contains all utterances regardless of internal gaps. Utterance-level models (DistilBERT) can score every utterance independently.
* **Canonical Cohort A ($N=2,328$ across contiguous dialogues):** Excludes dialogue-initial turns and breaks across missing ID gaps where sequential adjacency cannot be validated.

| Test Evaluation Cohort | Turn Count ($N$) | Inclusion Criteria | Research Function |
| :--- | :---: | :--- | :--- |
| **Full Test Set** | 2,610 | All utterances across all 280 test dialogues | Unconstrained baseline benchmark |
| **Excluded: Initial Turns** | 280 | First utterance of each dialogue ($t=0$) | No preceding history for context models |
| **Excluded: Broken Gaps** | 2 | Turns immediately following non-contiguous IDs | Prevents invalid cross-gap context construction |
| **Canonical Cohort A** | **2,328** | Contiguous turns with verified $t-1$ context | **Strict matched evaluation benchmark** |

**Methodological Guarantee:** All statistical hypothesis tests and model comparisons (DistilBERT vs. DialogueRNN vs. DialogueGCN) are conducted on **Cohort A ($N=2,328$)** to guarantee matched-sample validity.

---

## 8. Emotion Distribution and Transition Statistics

### 8.1 Class Distribution (Train Split, $N=9,989$)
* **Neutral:** 4,710 (47.15%)
* **Joy:** 1,743 (17.45%)
* **Surprise:** 1,205 (12.06%)
* **Anger:** 1,109 (11.10%)
* **Sadness:** 683 (6.84%)
* **Disgust:** 271 (2.71%)
* **Fear:** 268 (2.68%)

### 8.2 Transition Probabilities
* **Neutral Inertia:** A speaker currently in *Neutral* has a 62.4% probability of remaining in *Neutral* on their next turn.
* **Negative Resonance:** An *Anger* utterance by Speaker A elevates the probability of an immediate *Anger* response from Speaker B from the base rate of 11.1% to **38.7%** ($p < 0.001$).
* **Emotion Stickiness:** *Sadness* exhibits high self-transition probability (34.2%) compared to its low marginal base rate (6.84%).

---

## 9. DistilBERT Architecture, Training, and Checkpoint Selection

### 9.1 Baseline Architecture
* **Backbone:** `distilbert-base-uncased` (6 transformer layers, 12 attention heads, 768 hidden dimension, 66.96M parameters).
* **Classification Head:** Linear projection ($768 \to 7$) with dropout ($p=0.2$).
* **Training Hyperparameters:** AdamW, learning rate $2 \times 10^{-5}$, batch size 4, maximum sequence length 128, seed 42.

### 9.2 Baseline 3-Epoch Results
* **Dev Epoch 1:** Accuracy 59.24%, Macro F1 34.93%, Weighted F1 54.59%
* **Dev Epoch 2:** Accuracy 60.96%, Macro F1 41.74%, Weighted F1 57.96%
* **Dev Epoch 3 (Best Checkpoint):** Accuracy 58.70%, **Macro F1 44.29%**, Weighted F1 57.64%
* **Full Test Performance ($N=2,610$):**
  * Accuracy: **60.31%**
  * Macro F1: **44.00%**
  * Weighted F1: **60.54%**
* **Cohort A Matched Performance ($N=2,328$):**
  * Accuracy: **59.45%**
  * Macro F1: **43.14%**
  * Weighted F1: **57.94%**

### 9.3 Controlled 5-Epoch Experiment and Overfitting Analysis
To test whether extended fine-tuning improves representation learning on minority classes, a rigorous 5-epoch training experiment was executed (`scripts/train_distilbert_5epochs.py`) with all 6 physical CPU cores dedicated (`torch.set_num_threads(6)`):
* **CPU Step Latency:** 1.3335s/step on AMD Ryzen 5 7530U; total training duration: 240.2 minutes (4.00 hours).
* **Validation Progression across Epochs:**
  * Epoch 1: Loss 1.1578 | Val Acc 58.70% | Val Macro F1 39.58% | Val Weighted F1 56.76%
  * Epoch 2: Loss 0.8172 | Val Acc 59.87% | Val Macro F1 41.59% | Val Weighted F1 58.05%
  * Epoch 3: Loss 0.6019 | Val Acc 59.51% | Val Macro F1 40.85% | Val Weighted F1 57.77%
  * Epoch 4: Loss 0.4439 | Val Acc 60.59% | **Val Macro F1 42.15%** | Val Weighted F1 58.55% *(Selected as best 5-epoch checkpoint)*
  * Epoch 5: Loss 0.3587 | Val Acc 60.50% | Val Macro F1 39.66% | Val Weighted F1 58.46% *(Severe overfitting; dev Macro F1 collapsed by -2.49%)*
* **Test Performance on Canonical Cohort A ($N=2,328$):**
  * Accuracy: **61.90%** (+2.45% over 3-epoch baseline)
  * Macro F1: **41.70%** (-1.44% drop vs. 3-epoch baseline)
  * Weighted F1: **59.86%** (+1.92% over 3-epoch baseline)
* **Empirical Takeaway:** Extended fine-tuning creates an illusion of performance improvement if evaluated solely by accuracy (+2.45%), but this gain is achieved entirely by collapsing minority class recall (*Fear* recall dropped from 15.69% to 10.00%; *Disgust* recall from 23.53% to 14.71%) while greedily over-predicting the majority *Neutral* class (Neutral recall surged to 82.96%). Consequently, the 3-epoch checkpoint remains the mathematically sound baseline for multi-party conversational emotion recognition.

---

## 10. Contextual Modelling Experiments

Prior to graph and recurrent models, simple concatenation baselines were evaluated:
* **Context-1 (Concatenation):** Preceding utterance prepended to target utterance: `[CLS] Utterance_{t-1} [SEP] Utterance_t [SEP]`.
* **Context-1 + Speaker:** Speaker identity tokens prepended: `[SPEAKER_A] Utterance_{t-1} [SPEAKER_B] Utterance_t`.

### Findings:
Naive concatenation of preceding tokens yields modest gains in macro F1 (+0.8%) but suffers from attention dispersion over long dialogues and fails to distinguish individual party states when 4+ speakers interact simultaneously.

---

## 11. DialogueRNN Architecture, Party Tracking, and Implementation

### 11.1 Architectural Composition
DialogueRNN (Majumder et al., 2019) decomposes multi-party dialogue into three specialized GRU components:
1. **Global GRU ($G$):** Tracks overall conversational context and dialogue flow across all speakers.
2. **Party GRU ($P$):** Maintains individual psychological states for each active participant, updating only when that specific speaker takes a turn.
3. **Emotion GRU ($E$):** Synthesizes the global context, party state, and listener context into an emotion representation vector.

### 11.2 Recurrent Information Flow Walkthrough
DialogueRNN processes conversational turns sequentially through a multi-stage recurrent pipeline:
* **Turn Ingestion ($u_t \to \mathbf{x}_t$):** The current turn's text is converted into a frozen 768-dimensional DistilBERT [CLS] embedding vector $\mathbf{x}_t$.
* **Global Context Update ($g_t$):** The Global GRU aggregates overall conversation flow by concatenating the incoming utterance vector with the speaking participant's prior party state: $g_t = \text{GRU}_G(g_{t-1}, [\mathbf{x}_t \mathbin{\Vert} q_t^{(s_t)}])$.
* **Speaker Party State Update ($q_t^{(s)}$):** The active speaker's private Party GRU updates its internal psychological state using the new utterance and the updated global context: $q_t^{(s_t)} = \text{GRU}_P(q_{t-1}^{(s_t)}, [\mathbf{x}_t \mathbin{\Vert} g_t])$. All listening interlocutors retain their existing party states ($q_t^{(s)} = q_{t-1}^{(s)}$ for $s \ne s_t$).
* **Interlocutor Context Attention ($c_t$):** An attentive query generated from the speaker's state attends dynamically across all preceding global states $\{g_1, \dots, g_{t-1}\}$ to extract relevant past dialogue context $c_t$.
* **Emotion Representation & Prediction ($e_t \to \hat{y}_t$):** The Emotion GRU synthesizes the speaker's party state with attentive context ($e_t = \text{GRU}_E(e_{t-1}, [q_t^{(s_t)} \mathbin{\Vert} c_t])$) and passes $e_t$ through a linear projection layer to produce 7-class softmax probabilities.

### 11.3 Mathematical Formulation
* **Input Feature:** Frozen 768-dimensional DistilBERT [CLS] vector $\mathbf{x}_t$.
* **Global State Update:**
  $$g_t = \text{GRU}_G(g_{t-1}, [\mathbf{x}_t \mathbin{\Vert} q_t^{(s_t)}])$$
* **Party State Update:**
  $$q_t^{(s)} = \begin{cases} \text{GRU}_P(q_{t-1}^{(s)}, [\mathbf{x}_t \mathbin{\Vert} g_t]), & \text{if } s = s_t \\ q_{t-1}^{(s)}, & \text{otherwise} \end{cases}$$
* **Attention Mechanism:** Soft attention over preceding global states $\{g_1, \dots, g_{t-1}\}$ computes attentive context $c_t$.
* **Emotion Representation:** $e_t = \text{GRU}_E(e_{t-1}, [q_t^{(s_t)} \mathbin{\Vert} c_t])$.

### 11.4 Verified Results (Cohort A, $N=2,328$)
* **Accuracy:** **60.14%** (+0.69% over DistilBERT)
* **Macro F1:** **43.42%** (+0.28% over DistilBERT)
* **Weighted F1:** **58.92%** (+0.98% over DistilBERT)
* **Parameter Count:** 670,855 trainable parameters.

---

## 12. DialogueGCN Architecture, Relational Topology, and Edge Logic

### 12.1 Relational Graph Construction
DialogueGCN (Ghosal et al., 2019) represents each dialogue as an undirected graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{R})$:
* **Nodes $\mathcal{V}$:** Utterances $u_1, \dots, u_N$ represented by frozen DistilBERT CLS embeddings.
* **Relations $\mathcal{R}$:** Two relation types:
  1. $r = \text{self}$ (Intra-speaker edge: same speaker connecting past and future turns).
  2. $r = \text{other}$ (Inter-speaker edge: conversational exchanges between different interlocutors).
* **Context Window:** Windowed temporal bounds ($w=10$ past turns, $w=10$ future turns).

### 12.2 Graph Convolution and Similarity Attention
The relational edge weight $\alpha_{ij}$ between turn $i$ and turn $j$ is dynamically computed using scaled dot-product similarity:
$$\alpha_{ij} = \text{Softmax}\left(\frac{(u_i \mathbf{W}_Q)(u_j \mathbf{W}_K)^T}{\sqrt{d}}\right)$$

The 2-layer Graph Convolution updates node representations:
$$h_i^{(l+1)} = \sigma \left( \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_i^r} \frac{\alpha_{ij}}{c_{i,r}} \mathbf{W}_r^{(l)} h_j^{(l)} + \mathbf{W}_0^{(l)} h_i^{(l)} \right)$$

### 12.3 Verified Results (Cohort A, $N=2,328$)
* **Accuracy:** **60.91%** (+1.46% over DistilBERT, +0.77% over DialogueRNN)
* **Macro F1:** **43.85%** (+0.71% over DistilBERT, +0.43% over DialogueRNN)
* **Weighted F1:** **60.09%** (+2.15% over DistilBERT, +1.17% over DialogueRNN)
* **Parameter Count:** 452,487 trainable parameters.

---

## 13. Exploratory Sequence Models: Temporal CNN and Linear-Chain CRF

To evaluate alternative sequential paradigms against recurrent and graph architectures, two classic sequence modeling formulations were implemented and evaluated on pre-extracted frozen DistilBERT CLS features ($d=768$):

### 13.1 Causal Temporal CNN (1D Dilated Convolution)
* **Architecture:** 3 dilated residual blocks with kernel size $k=3$ and exponential dilations $d \in \{1, 2, 4\}$.
* **Causality & Receptive Field:** Strict left-padding of $(k-1) \times d$ turns prevents future token leakage, achieving an effective causal receptive field of **15 conversational turns**.
* **Parameter Count:** 823,303 trainable parameters.
* **Training Benchmark:** 34.7s on 6 CPU cores across 25 epochs.
* **Cohort A Test Results ($N=2,328$):**
  * Accuracy: **60.27%** (+0.82% over DistilBERT baseline)
  * Macro F1: **42.47%** (-0.67% vs. DistilBERT baseline)
  * Weighted F1: **59.52%** (+1.58% over DistilBERT baseline)

### 13.2 Linear-Chain Conditional Random Field (CRF)
* **Architecture:** Learns an emission projection ($768 \to 7$, 5,383 parameters) and a $7 \times 7$ emotion transition matrix $\mathbf{A}_{j, k}$ ($49$ parameters) with start/end transitions. Total parameters: **5,446**.
* **Training Benchmark:** 11.3s on 6 CPU cores across 25 epochs under Negative Log-Likelihood loss.
* **Cohort A Test Results ($N=2,328$):**
  * **Global Viterbi Decoding (Non-Causal):** Accuracy **60.01%**, Macro F1 **42.44%**, Weighted F1 **59.47%**
  * **Causal Online Greedy Decoding:** Accuracy **60.18%**, Macro F1 **43.15%**, Weighted F1 **59.63%**
* **Learned Transition Matrix Dynamics:**
  * Diagonal elements exhibit positive emotional inertia for high-frequency states (*Neutral $\to$ Neutral* $+1.84$, *Joy $\to$ Joy* $+0.92$), while penalizing rare self-transitions (*Fear $\to$ Fear* $-0.48$, *Disgust $\to$ Disgust* $-0.31$).
  * Cross-emotion transitions penalize polarity reversals (*Disgust $\to$ Joy* $-1.12$, *Anger $\to$ Joy* $-0.89$), confirming that immediate polarity reversals are statistically improbable without an intervening neutral or explanatory turn.

### 13.3 Architectural Synthesis: Inductive Biases in Conversational NLP
The matched Cohort A comparison illuminates why graph and recurrent models maintain a performance ceiling over temporal convolutions and sequence CRFs:
1. **Speaker Agnosticism of Temporal Convolutions:** Dilated convolutions apply uniform receptive field kernels across time steps regardless of *speaker turns*. Turn $t-1$ by Speaker B affects Speaker A differently than Turn $t-2$ by Speaker A (interpersonal influence vs. self-inertia). Convolutions lack the dynamic indexing required to separate interlocutor influence from personal emotional inertia.
2. **Homogeneous Markov Transitions:** A single transition matrix $\mathbf{A}_{j, k}$ assumes transitions are identical whether speakers switch or remain the same.
3. **The Superiority of Relational Speaker Inductive Biases:** DialogueGCN and DialogueRNN condition state updates directly on speaker identity (relational edges or speaker-specific party GRUs), allowing them to break through the 43.4% Macro F1 barrier.

---

## 14. Emotion Dynamics: Inertia, Resonance, and Turning Points

### 14.1 Quantifying Emotional Inertia
Emotional inertia is defined as the conditional probability that Speaker $S$ expresses emotion $E$ at turn $t$, given that Speaker $S$'s previous utterance at turn $t - k$ was also emotion $E$:
$$\mathcal{I}(E, S) = P(y_t = E \mid y_{t-k} = E, \text{speaker}(t) = \text{speaker}(t-k) = S)$$

### 14.2 Quantifying Emotional Resonance
Resonance measures cross-speaker contagion:
$$\mathcal{R}(E_A \to E_B) = P(y_t = E_B \mid y_{t-1} = E_A, \text{speaker}(t) \ne \text{speaker}(t-1))$$

### 14.3 Turning Point Detection
A conversational turn $t^*$ is classified as an **affective turning point** if:
1. Significant emotional shift: $y_{t^*} \ne y_{t^* - 1}$.
2. Valence polarity reversal: $\text{Valence}(y_{t^*}) \times \text{Valence}(y_{t^* - 1}) < 0$.
3. Divergence persistence: The new emotion state persists for at least 2 subsequent turns.

---

## 15. Next-Emotion Forecasting and Empirical Sequence Baselines

In conversational interaction, next-emotion forecasting evaluates whether past dialogue history alone can predict the upcoming emotion of the next turn **before that utterance is spoken or perceived**. Unlike emotion recognition (which conditions on the current utterance representation $u_t$), forecasting evaluates pure emotional inertia, interpersonal contagion, and affective transition dynamics.

### 15.1 Matched Sequence Evaluation Results

We evaluated sequential baselines across official MELD test splits. Because higher-order models require preceding context turns, metrics are benchmarked on corresponding valid history cohorts (Order 1 / standard test $N=2,610$; Order 2 $N=2,059$; Order 3 $N=1,807$):

| Forecasting Model | History Size ($k$) | Conditioning Features | Test Split ($N$) | Accuracy | Macro F1 | Weighted F1 |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **Majority Baseline** | 0 | None (constant *Neutral*) | 2,610 | 47.94% | 9.26% | 31.07% |
| **Speaker Persistence** | 1 (Speaker) | Speaker's previous emotion $y_{\text{last}}(S)$ | 2,610 | 41.92% | **27.36%** | **41.86%** |
| **First-Order Markov Chain** | 1 (Turn) | Preceding turn's emotion $y_{t-1}$ | 2,610 | 47.94% | 9.26% | 31.07% |
| **Higher-Order Markov-2** | 2 (Turns) | Preceding 2 turns $(y_{t-1}, y_{t-2})$ | 2,059 | **48.81%** | 20.03% | 39.92% |
| **Higher-Order Markov-3** | 3 (Turns) | Preceding 3 turns $(y_{t-1}, y_{t-2}, y_{t-3})$ | 1,807 | 47.65% | 20.80% | 39.84% |
| **Recursive Markov-2** | 2 (Predicted) | Model's own rolled-out predictions | 2,059 | 44.24% | 15.16% | 34.88% |
| **Recursive Markov-3** | 3 (Predicted) | Model's own rolled-out predictions | 1,807 | 44.38% | 13.45% | 33.33% |

### 15.2 Empirical Analysis of Affective Forecasting Dynamics

1. **The Inertia Advantage of Speaker Persistence (Macro F1 27.36%):**
   Despite posting lower raw accuracy (41.92%), the Speaker Persistence baseline achieves the highest Macro F1 among all standalone historical sequence baselines. Because human speakers exhibit strong emotional inertia across their own conversational turns (*Joy $\to$ Joy*, *Anger $\to$ Anger*), Persistence successfully predicts minority emotion bursts. It yields non-zero recall across multiple emotion classes, preventing the minority-class erasure that cripples transition-probability models.

2. **First-Order Markov Degeneracy:**
   The First-Order Markov chain ($P(y_t \mid y_{t-1})$) produces performance **identical to the trivial Majority baseline** (47.94% Accuracy, 9.26% Macro F1). Because *Neutral* constitutes 47.1% of all turns in MELD, the maximum likelihood transition from *every* emotion $y_{t-1} \in \mathcal{Y}$ leads to *Neutral*:
   $$\arg\max_{y \in \mathcal{Y}} P(y \mid y_{t-1}) = \text{Neutral}, \quad \forall y_{t-1} \in \mathcal{Y}$$
   Consequently, the 1st-order Markov model predicts *Neutral* unconditionally for every turn, achieving an F1 of 0.648 on Neutral and 0.00 on all six non-neutral emotions ($\text{Macro F1} = 0.648 / 7 = 9.26\%$).

3. **Higher-Order Markov Transitions and State Sparsity:**
   Conditioning on higher-order histories ($k=2, 3$) successfully breaks the *Neutral* attractor. In Markov-2, conditioning on 49 unique $(y_{t-1}, y_{t-2})$ bigram pairs lifts Macro F1 to **20.03%**, capturing strong emotional continuities such as *Anger $\to$ Anger* (F1 = 0.39), *Joy $\to$ Joy* (F1 = 0.20), and *Sadness* (F1 = 0.18).
   However, moving to Markov-3 yields only a marginal Macro F1 gain (**20.80%**) while overall accuracy declines to 47.65%. This plateau is driven by severe **combinatorial state sparsity**: while $7^3 = 343$ transition states are theoretically possible, only 311 are observed in training, with over 60% of observed states having support $N \le 2$.

4. **Compounding Error in Recursive Forecasting:**
   When forecasting multiple turns into the future in a closed loop (recursive generation without teacher forcing), prediction divergence reaches **24.14%** for Markov-2 and **24.52%** for Markov-3. As small forecasting errors accumulate, the sequential trajectory quickly collapses back into the stationary *Neutral* distribution, degrading Macro F1 to 15.16% and 13.45% respectively.

---

## 16. Explainability Methods and Attributions

To avoid deceptive certainty, explanations are decoupled into three empirical layers:

### 16.1 Text-Level Token Attribution (Integrated Gradients)
Following Sundararajan et al. (2017), path-integrated gradients ($M=25$ steps) compute the attribution of subword token embeddings toward classification logits:
$$\text{Attribution}_i^c(x) \approx (x_i - x'_i) \times \frac{1}{M} \sum_{m=1}^{M} \left. \frac{\partial F_c}{\partial x_i} \right|_{x' + \frac{m}{M}(x - x')}$$
* **Sarcasm Lexical Trap:** On the utterance *"Oh that's just fantastic, another flat tire in the middle of a thunderstorm."*, DistilBERT falsely predicts **Joy** with $89.9\%$ confidence because `"fantastic"` receives the highest positive attribution ($+0.242$). This empirically proves the vulnerability of surface text transformers to sarcastic irony in the absence of vocal prosody or conversational context.
* **Affective Token Salience:** Across true positive classes, attribution concentrates on semantic pivots: `"disgusting"` ($+0.339$) for *Disgust*, `"terrified"` ($+0.159$) and `"leave"` ($+0.135$) for *Fear*, and discourse markers `"wait"` ($+0.159$) and `"serious"` ($+0.160$) for *Surprise*.

### 16.2 Conversational Context Perturbation ($\Delta P$, $\Delta \ell$)
Evaluating turns under full conversational history $\mathcal{H}_{<t}$ versus isolated utterance conditions ($0$-context) isolates conversational sensitivity:
$$\Delta P_t(y) = P(y \mid u_t, \mathcal{H}_{<t}) - P(y \mid u_t, \emptyset)$$
* In an emotional escalation dialogue between Ross and Rachel (*"I made a mistake, okay? It didn't mean anything!"*), isolated DialogueRNN predicts *Anger* with $0.77$ confidence. Under full context, preceding rejection (*"We are done"*) introduces conflicting affective history into Ross's party state, shifting confidence down by **$\Delta P = -0.314$** ($0.46$) as probability distributes toward distress.
* Conversely, DialogueGCN leverages relational speaker edges to reinforce conflict dynamics, boosting *Anger* confidence by **$+0.113$** ($1.00$).

### 16.3 Cross-Model Disagreement Taxonomy (Cohort A, $N=2,328$)
Matching test predictions across DistilBERT, DialogueRNN, and DialogueGCN reveals high pairwise concordance and illuminates exact context failure modes:
* **Unanimous Agreement:** All 3 models agree on **1,772 out of 2,328 turns ($76.12\%$)**.
* **Pairwise Cohen's Kappa:**
  * DistilBERT vs. DialogueRNN: **$\kappa = 0.8412$** (Concordance: $88.66\%$)
  * DialogueRNN vs. DialogueGCN: **$\kappa = 0.7468$** (Concordance: $81.44\%$)
  * DistilBERT vs. DialogueGCN: **$\kappa = 0.7116$** (Concordance: $78.91\%$)
* **Taxonomy Distribution:**
  * **All Correct:** $1,210$ turns ($51.98\%$)
  * **Context Recovery (Net):** $197$ turns ($8.46\%$) where DistilBERT failed but context models recovered ground truth ($107$ unique DialogueGCN wins, $30$ DialogueRNN wins, and $60$ mutual wins).
  * **Context Dilution:** $174$ turns ($7.47\%$) where isolated DistilBERT was correct, but conversational history over-smoothed the prediction into an error.
  * **Consensus Failure:** $10$ turns ($0.43\%$) where all 3 models predicted different emotions.
  * **Universal Error:** $737$ turns ($31.66\%$) where all 3 models failed, reflecting the ceiling of textual ERC on short backchannels and subtle sarcasm without multimodal audio.

---

## 17. Application Architecture and Capabilities

The deployed system (`Emotion Dynamics`) comprises:
* **Backend:** FastAPI asynchronously serving DistilBERT, DialogueRNN, and DialogueGCN inference engines.
* **Frontend:** 5-screen desktop/mobile interface in restrained Dark Charcoal (`#171B18`):
  1. *Home / Import:* Drag-and-drop parsing for Raw Text, WhatsApp, Instagram, and CSV.
  2. *Conversation Viewer:* Dual-speaker bubbles with predicted emotion pills.
  3. *Emotion Dynamics:* Timeline charts, speaker inertia bars, transition statistics.
  4. *Model Comparison:* Side-by-side prediction grid with consensus badges.
  5. *Research Inspection:* Model cards, parameter counts, and benchmark reports.

---

## 18. Methodological and Domain Limitations

1. **MELD Domain Shift:** Scripted situation comedy dialogue features elevated rates of sarcasm, exaggerated vocalizations, and comedic timing that do not mirror clinical or casual interpersonal text.
2. **Annotation Ambiguity:** Human inter-annotator agreement on MELD achieves Fleiss' $\kappa \approx 0.43$, reflecting genuine affective ambiguity.
3. **Non-Clinical Boundary:** Predictions reflect **perceived expressed emotion in text**, not internal psychological reality.

---

## 19. Future Multimodal Research Scope: Tri-Modal Affective Fusion

While the core contributions of this study establish textual and contextual baselines for multi-party ERC, MELD natively provides paired audiovisual recordings for every dialogue turn (`.mp4` clips). As demonstrated by our Integrated Gradients error analysis, text-only ERC suffers from fundamental ambiguity in sarcastic irony, backchannel vocalizations (*"Oh"*, *"Yeah"*), and non-verbal affective leakage. We establish a concrete architectural blueprint for extending the current pipeline into full **Tri-Modal Affective Fusion (Text + Audio + Video)**:

### 19.1 Acoustic and Prosodic Feature Engineering
Acoustic signals capture vocal arousal, pitch inflection, and affective intensity that are invisible in text transcriptions:
1. **Expert Engineered Prosody (openSMILE eGeMAPS):** Extraction of the 88-dimensional extended Geneva Minimalistic Acoustic Parameter Set (`eGeMAPS`), including fundamental frequency ($F_0$ pitch trajectory, jitter), energy dynamics (loudness, shimmer), spectral balance (formants $F_1-F_3$, alpha ratio), and harmonic-to-noise ratio (HNR).
2. **Self-Supervised Speech Representations (WavLM / HuBERT):** Dense acoustic representations extracted from pre-trained `WavLM-Large` or `HuBERT-Large` (1024-dim), pooling frame-level transformer layers to preserve paralinguistic tone, pauses, and emotional tremors.
3. **Temporal ASR Word Alignment:** Utilizing OpenAI `Whisper-Large-v3` forced alignment to timestamp phoneme-to-word boundaries, ensuring microsecond-level synchronization between acoustic pitch contours and subword textual tokens.

### 19.2 Visual Facial Expression and Gestural Dynamics
Visual information resolves ambiguity through micro-expressions and gaze dynamics:
1. **Facial Action Coding System (OpenFace 2.0 FACS):** Extraction of 18 continuous Action Unit (`AU`) intensities (e.g., AU01 Inner Brow Raiser, AU04 Brow Lowerer, AU12 Lip Corner Puller, AU15 Lip Corner Depressor), alongside 3D head pose orientation (pitch, yaw, roll) and eye gaze vectors.
2. **Spatiotemporal Visual Embeddings (VideoMAE / CLIP):** Extracting frame-level face-crop representations using `CLIP-ViT-B/16` and temporal transformer encoders (`VideoMAE-v2`), capturing dynamic facial gestures and muscle tension across the utterance duration.

### 19.3 Tri-Modal Relational Fusion Architecture
Rather than simple late-stage heuristic voting, multimodal features are fused at the utterance level before sequential dialogue modeling:
$$\mathbf{u}_t = \text{LayerNorm}\left( \mathbf{W}_f \left[ \mathbf{e}_{\text{text},t} \mathbin{\Vert} \mathbf{a}_{\text{audio},t} \mathbin{\Vert} \mathbf{v}_{\text{video},t} \right] + \mathbf{b}_f \right)$$

Alternatively, **Cross-Modal Multi-Head Attention** allows textual tokens to query acoustic and visual feature maps, selectively amplifying vocal stress cues when textual sentiment is ambiguous:
$$\mathbf{h}_{\text{cross}, t} = \text{MultiHeadAttention}\left(\mathbf{Q}=\mathbf{e}_{\text{text},t}, \, \mathbf{K}=\mathbf{a}_{\text{audio},t}, \, \mathbf{V}=\mathbf{a}_{\text{audio},t}\right)$$

These fused tri-modal utterance representations $\mathbf{u}_t$ directly replace unimodal DistilBERT features in:
* **DialogueRNN:** Modulating speaker party state GRUs ($q_t^{(s_t)} = \text{GRU}(q_{t-1}^{(s_t)}, [\mathbf{u}_t \mathbin{\Vert} c_t])$), capturing how a speaker's vocal tension accumulates over time.
* **DialogueGCN:** Modulating relational edge weights ($\alpha_{ij} = \text{Softmax}(\mathbf{u}_i^\top \mathbf{W}_e \mathbf{u}_j)$), allowing interlocutor graph attention to weight speakers who express elevated vocal or visual agitation.

### 19.4 Real-World Application: Conversational De-Escalation Index
For deployment in customer resolution, conflict mediation, and mental health support, tri-modal emotion dynamics provide an actionable **De-Escalation Index** ($D_{\text{idx}}$):
$$D_{\text{idx}}(t) = \sum_{\tau=t-k}^{t} w_\tau \cdot \left[ \Delta \text{Valence}(\tau) - \lambda \cdot \Delta \text{Arousal}(\tau) \right]$$
Where $w_\tau$ is an exponential decay recency weight, $\text{Valence}$ is derived from text and facial smiles (AU12), and $\text{Arousal}$ is derived from acoustic volume and pitch velocity. A consistently negative $D_{\text{idx}}$ triggers real-time early-warning intervention prompts before conversational breakdown occurs.

---

## 20. References

1. Chen, S. Y., Hsu, C. C., Kuo, C. C., & Ku, L. W. (2018). *EmotionLines: An Emotion Corpus of Multi-Party Conversations*. In Proceedings of LREC.
2. Poria, S., Hazarika, D., Majumder, N., Naik, G., Cambria, E., & Mihalcea, R. (2019). *MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations*. In Proceedings of ACL.
3. Majumder, N., Poria, S., Hazarika, D., Mihalcea, R., Gelbukh, A., & Cambria, E. (2019). *DialogueRNN: An Attentive RNN for Emotion Detection in Conversation*. In Proceedings of AAAI.
4. Ghosal, D., Majumder, N., Poria, S., Chhaya, N., & Gelbukh, A. (2019). *DialogueGCN: A Graph Convolutional Network for Emotion Recognition in Conversation*. In Proceedings of EMNLP.
5. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. In Proceedings of NAACL.
6. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*. arXiv preprint arXiv:1910.01108.

---
*Comprehensive master research findings report compiled for the Emotion Dynamics NLP initiative.*
