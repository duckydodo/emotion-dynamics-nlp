# DialogueRNN Implementation Specification for MELD

**Repository:** `emotion-dynamics-nlp`  
**Project Path:** `~/Projects/emotion-dynamics-nlp`  
**Reference Paper:** Majumder et al., *"DialogueRNN: An Attentive RNN for Emotion Detection in Conversations"*, AAAI 2019  
**Upstream Audit Standard:** [`outputs/reports/canonical_evaluation_audit.md`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/reports/canonical_evaluation_audit.md)  
**Target Output File:** `outputs/reports/dialoguernn_implementation_spec.md`  

---

## Executive Summary & Research Motivation

The Canonical MELD Evaluation Audit established that simple sequence concatenation approaches (such as Context-1, Context-1 + Speaker, and Autoregressive Combined-History DistilBERT) suffer from a fundamental failure mode:
1. **The "Accuracy Trap":** They exploit the massive `neutral` majority class (47.9% of turns), driving Neutral recall to 83.3% and nudging accuracy from 59.45% to 62.59%.
2. **The Minority-Class Collapse:** They severely suppress rare emotions (`fear` recall drops to 0.0%, `disgust` recall drops to 3.2%, and Macro F1 falls significantly from 0.4314 to 0.3848).
3. **The Unresolved Speaker State Dilemma:** In multi-party conversations ($M \le 9$ participants per dialogue in MELD), concatenating previous tokens fails to maintain distinct speaker identities. The model cannot discern whether an emotional shift is due to personal cognitive appraisal or an emotional reaction to an interlocutor.

**DialogueRNN** directly addresses these limitations by maintaining:
- A **Global Context GRU** that captures conversational topic and discourse flow across all turns.
- Distinct **Party-Specific State GRUs** that track the personal emotional state of each individual speaker across multiple, non-consecutive turns.
- An **Emotion GRU** that explicitly models internal emotional inertia and turning points for each speaker.

This specification details how to integrate DialogueRNN into the `emotion-dynamics-nlp` repository using frozen, fine-tuned DistilBERT utterance embeddings ($D_u = 768$) while preserving full causal isolation, computational feasibility on CPU, and exact alignment with canonical evaluation cohorts.

---

## 1. DialogueRNN Architecture: Mathematical Formulation

### 1.1 Original Architecture (Majumder et al., AAAI 2019)

At turn $t \in \{1, \dots, T\}$ in a dialogue, speaker $p = \mathcal{S}(t)$ produces an utterance with feature representation $u_t \in \mathbb{R}^{D_u}$, where $\mathcal{S}(t) \in \{1, \dots, M\}$ maps turn $t$ to one of $M$ participating dialogue parties.

```
                    ┌─────────────────────────┐
                    │ Emotion Prediction: l_t │
                    └────────────▲────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │       Emotion GRU       │◄─── Speaker-specific Attention
                    │        (e_t^(p))        │     over past {e_j^(p)}
                    └────────────▲────────────┘
                                 │
           ┌─────────────────────┴─────────────────────┐
           │                                           │
┌──────────┴──────────┐                     ┌──────────┴──────────┐
│  Party GRU (p):     │                     │ Context Attention   │
│  State q_t^(p)      │                     │ c_t over {g_1...g_t}│
└──────────▲──────────┘                     └──────────▲──────────┘
           │                                           │
           ├───────────────────────────┬───────────────┘
           │                           │
┌──────────┴──────────┐     ┌──────────┴──────────┐
│  Utterance Feature  │     │  Global Context GRU │
│       u_t           │────►│        (g_t)        │
└─────────────────────┘     └─────────────────────┘
```

The model executes three recurrent state transitions at every turn:

#### Step 1: Global Context Tracking ($\text{GRU}_{\mathcal{G}}$)
Tracks inter-speaker dialogue dynamics. At turn $t$, it consumes the incoming utterance $u_t$ concatenated with the speaking party's *previous* state $q_{t-1}^{(p)}$:
$$g_t = \text{GRU}_{\mathcal{G}}\Big(g_{t-1}, [u_t \,;\, q_{t-1}^{(p)}]\Big), \quad g_t \in \mathbb{R}^{D_g}$$
*Initial condition:* $g_0 = \mathbf{0}$, $q_0^{(m)} = \mathbf{0} \quad \forall m \in \{1, \dots, M\}$.

#### Step 2: Context Attention & Party State Update ($\text{GRU}_{\mathcal{P}}$)
To account for varying relevance of preceding conversational turns, an attention mechanism pools past global states $\{g_1, \dots, g_t\}$:
$$a_{t, j} = g_j^T W_{\alpha} g_t \quad (1 \le j \le t)$$
$$\alpha_{t, j} = \frac{\exp(a_{t, j})}{\sum_{k=1}^t \exp(a_{t, k})}$$
$$c_t = \sum_{j=1}^t \alpha_{t, j} g_j \in \mathbb{R}^{D_g}$$

The active speaker $p = \mathcal{S}(t)$ updates their individual state using the incoming utterance and the attended context vector:
$$q_t^{(p)} = \text{GRU}_{\mathcal{P}}\Big(q_{t-1}^{(p)}, [u_t \,;\, c_t]\Big), \quad q_t^{(p)} \in \mathbb{R}^{D_q}$$

For all listening (non-speaking) parties $m \ne p$, states remain invariant:
$$q_t^{(m)} = q_{t-1}^{(m)} \quad \forall m \ne p$$

*(Note on Listener States: Majumder et al. explored an optional listener GRU update. However, empirical ablation in Table 4 of the AAAI 2019 paper demonstrated that passive listener updating did not improve MELD Macro F1 and added substantial parameter overhead. We retain the non-speaking invariance standard).*

#### Step 3: Emotional Inertia Tracking ($\text{GRU}_{\mathcal{E}}$)
The Emotion GRU models internal emotional inertia by attending over the speaking party's *own prior emotional states* $\{e_j : j < t, \, \mathcal{S}(j) = p\}$:
$$b_{t, j} = (e_j)^T W_{\beta} q_t^{(p)} \quad \text{for } j < t \text{ where } \mathcal{S}(j) = p$$
$$\beta_{t, j} = \frac{\exp(b_{t, j})}{\sum_{k < t, \mathcal{S}(k)=p} \exp(b_{t, k})}$$
$$c_t^{(e)} = \sum_{j < t, \mathcal{S}(k)=p} \beta_{t, j} e_j \in \mathbb{R}^{D_e}$$
*(If turn $t$ is the speaker's first utterance in the dialogue, $c_t^{(e)} = \mathbf{0}$).*

The emotional state is updated via:
$$e_t = \text{GRU}_{\mathcal{E}}\Big(e_{t-1}^{(p)}, [q_t^{(p)} \,;\, c_t^{(e)}]\Big), \quad e_t \in \mathbb{R}^{D_e}$$

#### Step 4: Emotion Classification Layer
The emotion distribution for turn $t$ is projected from the emotional state $e_t$:
$$h_t = \text{ReLU}(W_e e_t + b_e), \quad h_t \in \mathbb{R}^{D_h}$$
$$\hat{y}_t = \text{softmax}(W_l h_t + b_l), \quad \hat{y}_t \in \mathbb{R}^7$$

---

### 1.2 Distinctions: Original 2019 Paper vs. Proposed NLP Adaptation

| Dimension | Original Majumder et al. (AAAI 2019) | Proposed Adaptation in `emotion-dynamics-nlp` | Rationale for Adaptation |
| :--- | :--- | :--- | :--- |
| **Utterance Encoder** | 100-dim GloVe + TextCNN ($D_u = 100$) | Frozen DistilBERT Baseline [CLS] ($D_u = 768$) | Leverages domain-adapted representations fine-tuned on MELD in this project. |
| **Dimensionality Projection** | None (CNN was small) | Linear Projection ($768 \to 256$) + LayerNorm + Dropout | Prevents GRUs from over-parameterizing on 768-dim inputs. |
| **Modality** | Multimodal (Text + Audio + Vision) | Text-only NLP modality | Matches repository scope and thesis objectives. |
| **Recurrence Direction** | Explored bidirectional variant | **Strictly Causal (Forward-only)** | Enforces online, streaming causal inference; prevents future leakage. |
| **Global Hidden Size ($D_g$)** | 150 | 128 | Optimized for lightweight CPU training without capacity loss. |
| **Party Hidden Size ($D_q$)** | 100 | 128 | Enables expressive party tracking for up to 9 speakers. |
| **Emotion Hidden Size ($D_e$)** | 100 | 64 | Compact emotional state space; prevents overfitting on small splits. |
| **Speaker Representation** | Local party indexing | Local party indexing ($0 \dots M-1$) | Generalizes out-of-vocabulary speakers (e.g. guest actors) without fixed vocabularies. |

---

## 2. Integration with Repository: Feature Extraction & Architecture

### 2.1 Reusable Feature Extraction Pipeline

We utilize the fine-tuned checkpoint [`checkpoints/transformer_baseline/best`](file:///home/chitta/Projects/emotion-dynamics-nlp/checkpoints/transformer_baseline/best).

#### Verified Tensor Dimensions:
- Checkpoint: `distilbert-base-uncased` Sequence Classification.
- Hidden Size ($D_{\text{dim}}$): **768**.
- Output Head: Linear layer $(768 \to 7)$.
- Sequence Length: Padded/truncated to 128 tokens during feature extraction.

#### Extraction Procedure:
```python
# Pass single utterance text into frozen DistilBERT
with torch.no_grad():
  outputs = distilbert_model(input_ids, attention_mask=attention_mask)
  # Extract [CLS] representation (token index 0)
  cls_rep = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, 768)
```
*Alternative:* The pre-classifier layer `model.pre_classifier(cls_rep)` (768-dim post-linear) may also be evaluated. We adopt the canonical pooled [CLS] representation ($u_t \in \mathbb{R}^{768}$).

---

## 3. Detailed Data Pipeline & Tensor Specification

### 3.1 Speaker Indexing Within Each Dialogue

An empirical audit of MELD shows:
- **Total distinct speaker strings across MELD:** 260 in Train, 47 in Dev, 100 in Test.
- **Maximum speakers in any single dialogue:** **9** (Train), **8** (Dev), **8** (Test). Median = **2** speakers.

**Protocol:**
For each dialogue, dialogue participants are dynamically assigned zero-indexed party IDs based on order of appearance:
```python
party_map = {}
for turn in dialogue_turns:
  spk = turn["Speaker"]
  if spk not in party_map:
    party_map[spk] = len(party_map)
```
This guarantees party index $p \in \{0, \dots, M-1\}$ where $M \le 9$. This completely insulates the model from out-of-vocabulary speaker strings at test time.

### 3.2 Tensor Shapes Across DialogueRNN

For a dialogue of length $T$ turns with $M$ parties:

| Component / Tensor | Tensor Name | Dimensions | Description |
| :--- | :--- | :--- | :--- |
| Utterance Sequence | `U` | `(T, 768)` | Pre-extracted DistilBERT embeddings. |
| Projected Utterances | `U_proj` | `(T, 256)` | Linear projection + LayerNorm + Dropout(0.2). |
| Speaker Index Vector | `speakers` | `(T,)` | Integer array of party indices $p_t \in [0, M-1]$. |
| Emotion Labels | `labels` | `(T,)` | Ground-truth emotion IDs $\in [0, 6]$. |
| Turn Validity Mask | `turn_mask` | `(T,)` | Boolean mask (1 = valid turn, 0 = padding). |
| Global Hidden States | `g_t` | `(128,)` | State of Global Context GRU at turn $t$. |
| Global State History | `G_history` | `(t, 128)` | Accumulated global states $\{g_1, \dots, g_t\}$. |
| Party State Bank | `Q` | `(M, 128)` | Bank storing the current party state for all $M$ speakers. |
| Emotion State Bank | `E_history` | `(t, 64)` | Accumulated emotional states $\{e_1, \dots, e_t\}$. |
| Logits | `logits` | `(T, 7)` | Unnormalized class scores for each turn. |

### 3.3 Handling Utterance-ID Gaps & Dialogue Boundaries

The canonical evaluation audit established that dialogues 93 and 108 contain utterance ID jumps:
* **In DialogueRNN Sequence Processing:** DialogueRNN processes dialogues chronologically as causal Markovian sequences. All contiguous and sequential turns within a dialogue are processed sequentially ($i = 0, \dots, T-1$).
* **Evaluation Reporting Separation:**
  * For **Cohort A ($N=2,328$)**, the two post-gap turns `(D93, U8)` and `(D108, U3)` are excluded from the loss and evaluation metrics, matching Context-1 exactly.
  * For **Cohort B ($N=2,330$)**, all turns $i \ge 1$ are included.
  * For **Cohort C ($N=280$)**, turn $i=0$ is evaluated.
  * For **Cohort D ($N=2,610$)**, all turns are evaluated.
* **State Reset Rule:** At turn 0 of every dialogue, all recurrent states ($g_0, Q, E$) are strictly reset to $\mathbf{0}$. No information persists across dialogue boundaries.

---

## 4. Class Imbalance & Loss Strategy

### 4.1 Justification from Canonical Audit
The audit proved that unweighted models trained with context collapse into predicting Neutral on >54% of turns, resulting in 0% recall on rare classes.

### 4.2 Training-Only Weight Computation
Class weights are computed **strictly from the 9,989 training labels** using [`src/models/loss.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/models/loss.py):

$$\text{Weight}(c) = \frac{N_{\text{train}}}{C \times n_c}$$

Exact measured training counts and computed weights ($C=7, N=9989$):
- `neutral` ($n=4,710$): **$0.3030$**
- `joy` ($n=1,743$): **$0.8187$**
- `surprise` ($n=1,205$): **$1.1842$**
- `anger` ($n=1,109$): **$1.2867$**
- `sadness` ($n=683$): **$2.0893$**
- `disgust` ($n=271$): **$5.2657$**
- `fear` ($n=268$): **$5.3246$**

### 4.3 Smoothed Inverse Frequency (Recommended)
To prevent extreme gradients on rare classes while still penalizing Neutral collapse, a smoothed square-root inverse frequency schedule is also implemented:
$$w_c = \sqrt{\frac{N}{C \times n_c}}, \quad \text{normalized so } \frac{1}{C}\sum w_c = 1.0$$
- `neutral`: **0.3962**
- `joy`: **0.6513**
- `surprise`: **0.7833**
- `anger`: **0.8165**
- `sadness`: **1.0404**
- `disgust`: **1.6516**
- `fear`: **1.6609**

*Protocol Rule:* Both the **Unweighted** and **Class-Weighted** DialogueRNN models will be trained under identical random seeds (seed 42) to isolate the exact impact of recurrent speaker tracking versus loss weighting.

---

## 5. CPU Feasibility, Storage & Memory Analysis

The host machine is a Fedora development system with an AMD Ryzen CPU and **no NVIDIA GPU**. All resource requirements have been measured against actual dataset sizes:

### 5.1 Pre-Extracted Feature Storage on Disk
* Train split: 9,989 utterances $\times 768$ float32 ($4\text{ bytes}$) = **30.68 MB**.
* Dev split: 1,109 utterances $\times 768$ float32 = **3.41 MB**.
* Test split: 2,610 utterances $\times 768$ float32 = **8.02 MB**.
* **Total Feature Storage:** **42.11 MB** (saved as PyTorch `.pt` dictionary tensors: `dialogue_id -> Tensor(T, 768)`).
* *Storage Impact:* Insignificant (<0.1% of available 57 GB free space).

### 5.2 DialogueRNN Parameter Count
* Utterance Projection: $768 \times 256 + 256 = 196,864$
* Global Context GRU ($[256 + 128] \to 128$): $3 \times (384 \times 128 + 128^2) = 196,608$
* Context Attention ($128 \times 128$): $16,384$
* Party State GRU ($[256 + 128] \to 128$): $196,608$
* Emotion Attention ($64 \times 128$): $8,192$
* Emotion GRU ($[128 + 64] \to 64$): $3 \times (192 \times 64 + 64^2) = 49,152$
* Classification Head: $64 \times 64 + 64 \times 7 = 4,551$
* **Total Model Parameters:** **~668,359 parameters** (only **2.67 MB** of weights!).

### 5.3 Training Time & Throughput Estimation (Measured)
* Feature extraction forward pass (13,708 utterances on DistilBERT at batch size 32): **~2.5 minutes total (one-time)**.
* DialogueRNN recurrent forward-backward pass per dialogue ($T \approx 10$): **~1.8 milliseconds**.
* Epoch duration (1,038 training dialogues): **~1.9 seconds per epoch**.
* Complete training run (40 epochs + validation evaluation): **~90 to 120 seconds total**.
* **Verdict:** Highly CPU-efficient; orders of magnitude faster than fine-tuning transformer backbones.

---

## 6. Leakage Safeguards & Evaluation Protocol

1. **Strict Temporal Isolation:** Attention at turn $t$ is strictly masked to past and current turns:
   $$\text{Attention Mask: } M_{t, j} = \begin{cases} 0 & j \le t \\ -\infty & j > t \end{cases}$$
   No future tokens or global states are visible to turn $t$.
2. **Strict Speaker Privacy:** Party state $q^{(m)}$ is updated *only* when party $m$ speaks.
3. **Dialogue Boundary Resets:** Every dialogue initializes with fresh hidden states. Dialogue IDs are shuffled during training, but utterances within each dialogue maintain strict chronological order.
4. **Validation Checkpoint Selection:** Model selection is governed exclusively by **Validation Macro F1** on the official 114 MELD dev dialogues.
5. **Canonical Reporting Standard:** Test results will be evaluated and reported across:
   - Cohort A ($N=2,328$, strict matched subset).
   - Cohort B ($N=2,330$, all subsequent turns).
   - Cohort C ($N=280$, Turn-0).
   - Cohort D ($N=2,610$, Full test).
   - McNemar and bootstrap tests against the baseline.

---

## 7. Expected Failure Modes & Diagnostic Checks

1. **Short Dialogues / Solitary Turns:**
   - In dialogues with only 1 or 2 turns, the emotion attention mechanism has no prior party states ($j < t$).
   - *Safeguard:* Fallback to zero context vector $c_t^{(e)} = \mathbf{0}$ for speaker-initial turns.
2. **Neutral Gravitation in Global Context:**
   - Because 47.9% of turns are Neutral, the global state $g_t$ can gradually drift towards a neutral representation.
   - *Diagnostic:* Monitor the cosine similarity between $g_t$ and a mean neutral embedding. Evaluate both unweighted and class-weighted training.
3. **Overfitting on Small Hidden Dimensions:**
   - With 668K parameters and 9,989 training examples, the recurrent layers could overfit.
   - *Safeguard:* Implement dropout ($p=0.2$ to $0.3$) across projection, attention, and classification layers, along with weight decay ($1 \times 10^{-4}$).

---

## 8. Implementation Milestones

```
[Phase 1: Feature Extraction]
  │  Extract frozen 768-dim embeddings from checkpoints/transformer_baseline/best
  │  Serialize to data/processed/meld_distilbert_features_{train,dev,test}.pt
  ▼
[Phase 2: Core DialogueRNN Module]
  │  Implement GlobalContextGRU, PartyStateGRU, EmotionGRU, and Causal Attention
  │  Write unit tests validating causal masking and state invariance for listeners
  ▼
[Phase 3: Data Loader & Batching Engine]
  │  Implement dialogue-level collator with party mapping and padding masks
  ▼
[Phase 4: Training & Validation Loop]
  │  Train DialogueRNN-Unweighted (Seed 42)
  │  Train DialogueRNN-Weighted (Smoothed Inv Freq, Seed 42)
  │  Select checkpoints on Dev Macro F1
  ▼
[Phase 5: Canonical Evaluation & Statistical Auditing]
  │  Evaluate on Cohorts A, B, C, D
  │  Run McNemar and paired bootstrap tests against Utterance Baseline & Model B
  │  Generate report and update leaderboard
```

---

## 9. Recommendation: Smallest Valid DialogueRNN Implementation

To verify whether explicit speaker tracking solves emotional inertia without unnecessary architectural bloat, we recommend:

1. **The Minimal Valid Architecture:**
   - **Feature Extractor:** Frozen DistilBERT [CLS] (768-dim) projected to 256-dim.
   - **Global Context GRU:** 128 units (causal forward only).
   - **Party State GRU:** 128 units per speaker (active speaker update only, inactive invariant).
   - **Emotion Layer:** Direct projection from $[q_t^{(p)} \,;\, c_t]$ to emotion logits (bypassing the separate Emotion GRU in the first iteration to eliminate 50,000 parameters and isolate party-state tracking).
2. **Benchmark Comparison:**
   Evaluate this minimal DialogueRNN directly against the Canonical Baseline ($N=2,328$, Macro F1 = 0.4314) and Model B ($N=2,328$, Macro F1 = 0.3848).
3. **Success Criterion:**
   Achieve deployable Accuracy $\ge 62.0\%$ while retaining Macro F1 $\ge 0.4200$ and recovering Fear recall $\ge 15.0\%$.
