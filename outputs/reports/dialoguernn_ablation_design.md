# Controlled DialogueRNN Architectural Ablation Study: Design Specification

**Repository:** `emotion-dynamics-nlp`  
**Working Directory:** `~/Projects/emotion-dynamics-nlp`  
**Date:** November 10, 2026  
**Status:** Pre-Experimental Design & Configuration Specification (Pending Approval)  
**Task Reference:** Task 12 (Part C)  
**Deliverable:** [`outputs/reports/dialoguernn_ablation_design.md`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/reports/dialoguernn_ablation_design.md)  
**Machine-Readable Configuration Files:**
- Ablation 1: [`configs/ablations/1_global_context_only.json`](file:///home/chitta/Projects/emotion-dynamics-nlp/configs/ablations/1_global_context_only.json)
- Ablation 2: [`configs/ablations/2_global_party_states.json`](file:///home/chitta/Projects/emotion-dynamics-nlp/configs/ablations/2_global_party_states.json)
- Ablation 3: [`configs/ablations/3_global_emotion_state.json`](file:///home/chitta/Projects/emotion-dynamics-nlp/configs/ablations/3_global_emotion_state.json)
- Ablation 4: [`configs/ablations/4_full_dialoguernn.json`](file:///home/chitta/Projects/emotion-dynamics-nlp/configs/ablations/4_full_dialoguernn.json)

---

## 1. Executive Summary & Research Motivation

The independent audit (Tasks 11 & 12) established that DialogueRNN achieves **Macro F1 = 0.4342** on Cohort A ($N=2,328$), statistically significantly outperforming standard text concatenation (Context-1, Macro F1 = 0.3912, $p_{\text{adj}} = 0.0216$) and text+emotion autoregression (Model B, Macro F1 = 0.3848, $p_{\text{adj}} = 0.0395$).

However, DialogueRNN is a compound architecture comprising three distinct recurrent/attentive sub-networks:
1. **Global Context GRU ($g_t$) & Context Attention ($c_t$):** Tracks the global sequence of utterances across all dialogue participants.
2. **Speaker-Specific Party State GRUs ($q_t^{(p)}$):** Maintains individual latent states for each interlocutor, updated only when that speaker speaks.
3. **Internal Emotion GRU ($e_t$) & Emotion Attention ($c_t^{(e)}$):** Models the personal emotional trajectory of the active speaker.

To rigorously determine **which architectural components drive performance** and whether speaker state tracking or emotion tracking is the primary mechanism preventing minority-class collapse, we have designed a **controlled four-way ablation study**.

---

## 2. Controlled Experimental Framework

To isolate architectural effects without introducing confounding variables, all four ablations maintain **strict parity** across all non-architectural factors:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        EXPERIMENTAL CONTROLS                           │
├─────────────────────────┬──────────────────────────────────────────────┤
│ Input Features          │ Frozen DistilBERT [CLS] embeddings (D=768)   │
│ Feature Storage         │ data/processed/dialoguernn/*.npz             │
│ Dataset Partitions      │ MELD Official Splits: Train (9,989 turns),   │
│                         │ Dev (1,109 turns), Test (2,610 turns)        │
│ Evaluation Cohort       │ Cohort A: Strict Contiguous Matched (N=2,328)│
│ Random Seed             │ Fixed seed = 42 for all runs                 │
│ Optimization Algorithm  │ AdamW (lr = 0.001, weight_decay = 0.0001)    │
│ Gradient Regularization │ Gradient norm clipping at max_norm = 1.0     │
│ Batching Strategy       │ 16 dialogues per batch with zero-padding     │
│ Training Budget         │ Max 40 epochs; Early stopping patience = 10  │
│ Checkpoint Selection    │ Best validation Macro F1 on Dev split        │
│ Test-Set Isolation      │ Test split strictly isolated during training │
│ Execution Environment   │ CPU-only Docker container (week11)           │
└─────────────────────────┴──────────────────────────────────────────────┘
```

---

## 3. Four Ablation Architectures in Detail

```
Configuration Matrix:
┌──────────────────────────────┬──────────────┬──────────────┬──────────────┬──────────────────┐
│ Ablation Configuration       │ Global GRU   │ Party GRUs   │ Emotion GRU  │ Parameter Count  │
├──────────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────────┤
│ 1. Global Context Only       │ YES (c_t)    │ NO           │ NO           │ 370,695 params   │
│ 2. Global + Party States     │ YES (c_t)    │ YES (q_t^p)  │ NO           │ 617,223 params   │
│ 3. Global + Emotion State    │ YES (c_t)    │ NO           │ YES (e_t)    │ 453,000 params   │
│ 4. Full DialogueRNN          │ YES (c_t)    │ YES (q_t^p)  │ YES (e_t)    │ 670,855 params   │
└──────────────────────────────┴──────────────┴──────────────┴──────────────┴──────────────────┘
```

### 3.1 Ablation 1: Global Context Only (`ablation_global_context_only`)
- **Research Question:** Can conversational context alone—modeled by a global causal GRU with attention over past turns—prevent neutral gravitation without tracking speaker identities?
- **Architectural Formulation:**
  1. *Projection:* $u_t' = \text{Dropout}(\text{LayerNorm}(W_{\text{in}} u_t + b_{\text{in}})) \in \mathbb{R}^{256}$
  2. *Global Context GRU:* $g_t = \text{GRU}_G(g_{t-1}, u_t') \in \mathbb{R}^{128}$  
     *(Note: No party state feedback $q_{t-1}^{(p)}$, so input size is 256).*
  3. *Context Attention:* $c_t = \sum_{\tau=0}^t \alpha_{t, \tau} g_\tau \in \mathbb{R}^{128}$ where $\alpha_{t, \tau} = \text{softmax}(g_\tau^T W_\alpha g_t)$.
  4. *Classification Head:* The context vector $c_t$ is projected directly to emotion logits:
     $$h_t = \text{ReLU}(W_1 c_t + b_1) \in \mathbb{R}^{64}$$
     $$\text{logits}_t = W_2 h_t + b_2 \in \mathbb{R}^7$$
- **Ablated Elements:** Party State GRU bank, Emotion Attention, and Emotion GRU bank completely omitted. No speaker IDs are consumed.
- **Parameter Count:** **370,695** parameters.

---

### 3.2 Ablation 2: Global Context + Party-Specific States (`ablation_global_party_states`)
- **Research Question:** Does maintaining separate recurrent party states for individual interlocutors account for the majority of DialogueRNN's performance, rendering the secondary Emotion GRU redundant?
- **Architectural Formulation:**
  1. *Projection:* $u_t' \in \mathbb{R}^{256}$.
  2. *Global Context GRU:* $g_t = \text{GRU}_G(g_{t-1}, [u_t'; q_{t-1}^{(p)}]) \in \mathbb{R}^{128}$  
     *(Input size: $256 + 128 = 384$).*
  3. *Context Attention:* $c_t = \text{Attn}(g_t, \{g_0 \dots g_t\}) \in \mathbb{R}^{128}$.
  4. *Speaker Party GRU Bank:* Maintains a separate latent state $q_t^{(p)} \in \mathbb{R}^{128}$ for each dialogue participant. Only the active speaker $p_t$ is updated:
     $$q_t^{(p_t)} = \text{GRU}_P(q_{t-1}^{(p_t)}, [u_t'; c_t]) \in \mathbb{R}^{128}$$
     $$q_t^{(k)} = q_{t-1}^{(k)} \quad \forall k \ne p_t$$
  5. *Classification Head:* Classification is performed directly from the active speaker's updated party state $q_t^{(p)}$:
     $$h_t = \text{ReLU}(W_1 q_t^{(p)} + b_1) \in \mathbb{R}^{64}$$
     $$\text{logits}_t = W_2 h_t + b_2 \in \mathbb{R}^7$$
- **Ablated Elements:** Emotion Attention and Emotion GRU bank omitted.
- **Parameter Count:** **617,223** parameters.

---

### 3.3 Ablation 3: Global Context + Emotion State (`ablation_global_emotion_state`)
- **Research Question:** Does modeling internal emotional persistence via an Emotion GRU help if the model cannot distinguish who is speaking (i.e., when emotional inertia is unconditioned on speaker identity)?
- **Architectural Formulation:**
  1. *Projection:* $u_t' \in \mathbb{R}^{256}$.
  2. *Global Context GRU:* $g_t = \text{GRU}_G(g_{t-1}, u_t') \in \mathbb{R}^{128}$.
  3. *Context Attention:* $c_t = \text{Attn}(g_t, \{g_0 \dots g_t\}) \in \mathbb{R}^{128}$.
  4. *Shared Emotion GRU:* A single, shared Emotion GRU models conversation-level emotional trajectory conditioned on the current utterance and context vector:
     $$e_t = \text{GRU}_E(e_{t-1}, [u_t'; c_t]) \in \mathbb{R}^{64}$$
     *(Input size: $256 + 128 = 384$, hidden size: 64).*
  5. *Classification Head:*
     $$h_t = \text{ReLU}(W_1 e_t + b_1) \in \mathbb{R}^{64}$$
     $$\text{logits}_t = W_2 h_t + b_2 \in \mathbb{R}^7$$
- **Ablated Elements:** Speaker-specific Party GRU bank and speaker-specific Emotion Attention omitted. All dialogue turns share a single emotional trajectory.
- **Parameter Count:** **453,000** parameters.

---

### 3.4 Ablation 4: Full DialogueRNN Reference (`ablation_full_dialoguernn`)
- **Architectural Formulation:**
  Full architecture as specified in Majumder et al. (AAAI 2019) and implemented in [`src/models/dialoguernn.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/models/dialoguernn.py):
  1. Projection ($768 \to 256$, LayerNorm, Dropout $p=0.2$).
  2. Global Context GRU ($g_t \in \mathbb{R}^{128}$, conditioned on $u_t'$ and $q_{t-1}^{(p)}$).
  3. Context Attention ($c_t \in \mathbb{R}^{128}$).
  4. Speaker Party GRU Bank ($q_t^{(p)} \in \mathbb{R}^{128}$, updated selectively for active speaker).
  5. Emotion Attention ($c_t^{(e)} \in \mathbb{R}^{64}$, attending over active speaker's own prior emotion states).
  6. Emotion GRU Bank ($e_t \in \mathbb{R}^{64}$, conditioned on $q_t^{(p)}$ and $c_t^{(e)}$).
  7. Classification Head: $h_t = \text{ReLU}(W_1 e_t + b_1) \in \mathbb{R}^{64} \to W_2 h_t + b_2 \in \mathbb{R}^7$.
- **Parameter Count:** **670,855** parameters.

---

## 4. Parameter Count and Sub-Network Comparison

| Layer / Sub-Network | Ablation 1 (Global Only) | Ablation 2 (Global + Party) | Ablation 3 (Global + Emotion) | Ablation 4 (Full Model) | Dimension Rationale |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Input Linear Projection** | 196,864 | 196,864 | 196,864 | 196,864 | $768 \times 256 + 256$ |
| **LayerNorm (256)** | 512 | 512 | 512 | 512 | $256 \times 2$ |
| **Global Context GRU** | 148,224 | 197,376 | 148,224 | 197,376 | Input 256 vs 384, Hidden 128 |
| **Context Attention** | 16,384 | 16,384 | 16,384 | 16,384 | $128 \times 128$ |
| **Speaker Party GRU Bank** | *Omitted* | 197,376 | *Omitted* | 197,376 | Input 384, Hidden 128 |
| **Emotion Attention** | *Omitted* | *Omitted* | *Omitted* | 8,192 | $128 \times 64$ |
| **Emotion GRU Bank** | *Omitted* | *Omitted* | 86,400 | 49,536 | Input 384 vs 192, Hidden 64 |
| **Classification FC1** | 8,256 | 8,256 | 4,160 | 4,160 | In 128 (Abl 1, 2) vs In 64 (Abl 3, 4) |
| **Classification FC2** | 455 | 455 | 455 | 455 | $64 \times 7 + 7$ |
| **Total Trainable Parameters** | **370,695** | **617,223** | **453,000** | **670,855** | Strictly accounted for |

---

## 5. Machine-Readable Configuration Registry

Each ablation configuration has been validated and persisted as an immutable JSON specification in `configs/ablations/`:

1. **`configs/ablations/1_global_context_only.json`**
   - Identifier: `ablation_global_context_only` (370,695 parameters)
   - Checkpoint path: `checkpoints/ablations/1_global_context_only/best_model.pt`
   - Output path: `outputs/experiments/ablations/1_global_context_only/`
2. **`configs/ablations/2_global_party_states.json`**
   - Identifier: `ablation_global_party_states` (617,223 parameters)
   - Checkpoint path: `checkpoints/ablations/2_global_party_states/best_model.pt`
   - Output path: `outputs/experiments/ablations/2_global_party_states/`
3. **`configs/ablations/3_global_emotion_state.json`**
   - Identifier: `ablation_global_emotion_state` (453,000 parameters)
   - Checkpoint path: `checkpoints/ablations/3_global_emotion_state/best_model.pt`
   - Output path: `outputs/experiments/ablations/3_global_emotion_state/`
4. **`configs/ablations/4_full_dialoguernn.json`**
   - Identifier: `ablation_full_dialoguernn` (670,855 parameters)
   - Checkpoint path: `checkpoints/ablations/4_full_dialoguernn/best_model.pt`
   - Output path: `outputs/experiments/ablations/4_full_dialoguernn/`

---

## 6. Pre-Registered Hypotheses and Success Criteria

When executed, the ablations will evaluate three formal hypotheses on the canonical matched test cohort (Cohort A, $N=2,328$):

* **Hypothesis 1 (Context vs. Speaker Disambiguation):**  
  *Prediction:* Ablation 1 (Global Context Only) will suffer from neutral gravitation and reduced minority recall, exhibiting performance closer to Context-1 (Macro F1 $\approx 0.39$) than DialogueRNN. This will demonstrate that conversational context alone does not prevent majority bias.
* **Hypothesis 2 (Primary Driver of Minority Recovery):**  
  *Prediction:* Ablation 2 (Global Context + Party States) will account for $\ge 85\%$ of the total Macro F1 lift from Context-1 to Full DialogueRNN. Maintaining interlocutor-specific states prevents an interlocutor's neutral demeanor from overwriting a partner's heightened state.
* **Hypothesis 3 (Incremental Benefit of Emotion Attention & GRU):**  
  *Prediction:* Ablation 4 (Full DialogueRNN) will outperform Ablation 2 primarily on **sustained emotional inertia** turns (turns where the same speaker remains in an emotion for $\ge 2$ consecutive utterances), while demonstrating comparable performance on cross-speaker shift turns.

---

## 7. Execution Readiness & Protocol

The ablation design is complete and fully specified. In compliance with task instructions:
- **Zero training has been executed.**
- **No model weights or code files have been modified.**
- All experiments are ready to launch upon user review and approval of this design.
