# DialogueRNN Architectural Ablation Study: Implementation and Verification Report

**Repository:** `emotion-dynamics-nlp`  
**Working Directory:** `/home/chitta/Projects/emotion-dynamics-nlp`  
**Date:** November 10, 2026  
**Status:** Implementation & Verification Complete — Pre-Training Phase  
**Task Reference:** Task 14 (Implement and Test DialogueRNN Ablation Architectures)  
**Deliverable Document:** [`outputs/reports/dialoguernn_ablation_implementation.md`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/reports/dialoguernn_ablation_implementation.md)  
**Code Implementations:**
- Architecture Classes & Factory: [`src/models/dialoguernn.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/models/dialoguernn.py#L310-L650)
- Ablation Test Suite: [`tests/test_dialoguernn_ablations.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/tests/test_dialoguernn_ablations.py)
**Configuration Files:**
- Ablation 1: [`configs/ablations/1_global_context_only.json`](file:///home/chitta/Projects/emotion-dynamics-nlp/configs/ablations/1_global_context_only.json)
- Ablation 2: [`configs/ablations/2_global_party_states.json`](file:///home/chitta/Projects/emotion-dynamics-nlp/configs/ablations/2_global_party_states.json)
- Ablation 3: [`configs/ablations/3_global_emotion_state.json`](file:///home/chitta/Projects/emotion-dynamics-nlp/configs/ablations/3_global_emotion_state.json)
- Ablation 4: [`configs/ablations/4_full_dialoguernn.json`](file:///home/chitta/Projects/emotion-dynamics-nlp/configs/ablations/4_full_dialoguernn.json)

---

## 1. Executive Summary & Verification Verdict

In accordance with the Task 13 validation specifications, the four DialogueRNN architectural ablations have been implemented and comprehensively tested.

### Key Verification Milestones:
1. **Preserved Backward Compatibility:** The original [`DialogueRNN`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/models/dialoguernn.py#L58-L308) class and its existing behavior, forward methods, and checkpoint loading contracts were preserved 100% intact.
2. **Dedicated Modular Ablation Class:** Implemented [`DialogueRNNAblation`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/models/dialoguernn.py#L370-L620), [`DialogueRNNAblationConfig`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/models/dialoguernn.py#L315-L368), and factory function [`build_ablation_model`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/models/dialoguernn.py#L622-L645).
3. **Exact Parameter Counts:** Independently verified against PyTorch modules to the exact integer:
   - **Ablation 1 (Global Context Only):** **370,695** parameters.
   - **Ablation 2 (Global Context + Party States):** **617,223** parameters.
   - **Ablation 3 (Global Context + Emotion State):** **452,999** parameters.
   - **Ablation 4 (Full DialogueRNN Reference):** **670,855** parameters.
4. **Genuine Absence of Ablated Components:** Disabled components are not instantiated as unused PyTorch submodules; their references are `None` and have 0 parameters in `named_parameters()`.
5. **Exact Full-Model Reproduction:** When initialized with identical weights or loaded from `checkpoints/dialoguernn_unweighted/best_model.pt`, [`DialogueRNNAblation`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/models/dialoguernn.py#L370-L620) (Mode 4) achieves:
   - State dictionary loading with `strict=True` (0 missing keys, 0 unexpected keys).
   - Exact numerical logit match with [`DialogueRNN`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/models/dialoguernn.py#L58-L308) (maximum absolute difference = $0.0$).
   - Exact reproduction of **2,610 / 2,610 (100.00%)** test predictions from the canonical benchmark.
6. **Test Suite Status:** 33 of 33 unit tests (21 existing regression tests + 12 new ablation tests) pass with zero errors.
7. **Strict Restrictions Observed:** Zero model training has occurred, existing checkpoints remain unmodified, and no Git commits or pushes were executed.

---

## 2. Architecture Specifications & Component Flows

### 2.1 Ablation 1: Global Context Only (`ablation_global_context_only`)
- **Ablation Objective:** Evaluates conversational emotion recognition relying exclusively on sequential global conversational flow without tracking speaker identities.
- **Component Flow Diagram:**
  ```
  u_t (768) ──► Linear(768->256) ──► LayerNorm ──► Dropout(0.2) ──► u'_t (256)
                                                                       │
                         ┌─────────────────────────────────────────────┘
                         ▼
  g_{t-1} (128) ──► Global GRU(256 -> 128) ──► g_t (128) ──► Context History {g_0 ... g_t}
                                                                       │
                                              Bilinear Attention ◄─────┘
                                                       │
                                                       ▼
                                                   c_t (128)
                                                       │
                                                       ▼
                                            Linear(128 -> 64) [FC1]
                                                       │
                                                       ▼
                                                     ReLU
                                                       │
                                                       ▼
                                            Linear(64 -> 7) [FC2] ──► Logits (7)
  ```
- **Active Sub-Networks:** Input Projection ($196,864$ params), LayerNorm ($512$), Global Context GRU ($148,224$), Context Attention ($16,384$), Classifier Head ($8,711$).
- **Ablated Sub-Networks:** Speaker Party GRUs (`None`), Emotion Attention (`None`), Emotion GRU (`None`).
- **Speaker IDs Consumed:** None.
- **Total Parameters:** **370,695**.

---

### 2.2 Ablation 2: Global Context + Party-Specific States (`ablation_global_party_states`)
- **Ablation Objective:** Tests whether maintaining individual latent states for each interlocutor captures the primary performance benefit of DialogueRNN, omitting the secondary emotion recurrence module.
- **Component Flow Diagram:**
  ```
  u'_t (256) ──┐
               ├─► Concat(384) ──► Global GRU(384 -> 128) ──► g_t ──► Context Attn ──► c_t (128)
  q_{t-1}^{(p)}┘                                                                        │
                                                                                         │
  u'_t (256) ──┐                                                                         │
               ├─► Concat(384) ──────────────────────────────────────────────────────────┤
  c_t (128)  ──┘                                                                         │
                                                                                         ▼
  q_{t-1}^{(p)} ──────────────────────────────────────► Party GRU(384 -> 128) ──► q_t^{(p)} (128)
                                                                                         │
                                                                                         ▼
                                                                             Linear(128 -> 64) [FC1]
                                                                                         │
                                                                                         ▼
                                                                                       ReLU
                                                                                         │
                                                                                         ▼
                                                                             Linear(64 -> 7) [FC2] ──► Logits (7)
  ```
- **Active Sub-Networks:** Input Projection ($196,864$), LayerNorm ($512$), Global Context GRU ($197,376$), Context Attention ($16,384$), Party State GRU Bank ($197,376$), Classifier Head ($8,711$).
- **Ablated Sub-Networks:** Emotion Attention (`None`), Emotion GRU (`None`).
- **Speaker IDs Consumed:** Active speaker index $p_t$ selectively updates the $p_t$-th state slot in party bank $Q \in \mathbb{R}^{B \times P \times 128}$.
- **Total Parameters:** **617,223**.

---

### 2.3 Ablation 3: Global Context + Emotion State (`ablation_global_emotion_state`)
- **Ablation Objective:** Tests whether an unconditioned emotion GRU provides utility when emotional persistence is tracked across conversational turns without attributing utterances to specific speakers.
- **Component Flow Diagram:**
  ```
  u'_t (256) ──► Global GRU(256 -> 128) ──► g_t ──► Context Attn ──► c_t (128)
                                                                        │
  u'_t (256) ──┐                                                        │
               ├─► Concat(384) ─────────────────────────────────────────┘
  c_t (128)  ──┘                                                        │
                                                                        ▼
  e_{t-1} (64) ───────────────────────────────► Emotion GRU(384 -> 64) ──► e_t (64)
                                                                             │
                                                                             ▼
                                                                   Linear(64 -> 64) [FC1]
                                                                             │
                                                                             ▼
                                                                           ReLU
                                                                             │
                                                                             ▼
                                                                   Linear(64 -> 7) [FC2] ──► Logits (7)
  ```
- **Active Sub-Networks:** Input Projection ($196,864$), LayerNorm ($512$), Global Context GRU ($148,224$), Context Attention ($16,384$), Shared Emotion GRU ($86,400$), Classifier Head ($4,615$).
- **Ablated Sub-Networks:** Speaker Party GRUs (`None`), Speaker-Specific Emotion Attention (`None`).
- **Speaker IDs Consumed:** None.
- **Total Parameters:** **452,999**.

---

### 2.4 Ablation 4: Full DialogueRNN Reference (`ablation_full_dialoguernn`)
- **Ablation Objective:** Complete reference architecture integrating Global Context GRU, Context Attention, Speaker Party State GRUs, Emotion Attention, and Emotion GRU.
- **Component Flow Diagram:**
  ```
  u'_t (256) + q_{t-1}^{(p)} (128) ──► Global GRU(384 -> 128) ──► g_t ──► Context Attn ──► c_t (128)
                                                                                             │
  u'_t (256) + c_t (128) ────────────► Party GRU(384 -> 128) ──► q_t^{(p)} (128)            │
                                                                      │                      │
                                                                      ├─► Emotion Attn ──────┤
                                                                      │   over E_past(p)     │
                                                                      │                      ▼
  q_t^{(p)} (128) + c_t^{(e)} (64) ──► Emotion GRU(192 -> 64) ────────┴──────────────► e_t^{(p)} (64)
                                                                                             │
                                                                                             ▼
                                                                                   Linear(64 -> 64) [FC1]
                                                                                             │
                                                                                             ▼
                                                                                           ReLU
                                                                                             │
                                                                                             ▼
                                                                                   Linear(64 -> 7) [FC2] ──► Logits (7)
  ```
- **Active Sub-Networks:** Input Projection ($196,864$), LayerNorm ($512$), Global Context GRU ($197,376$), Context Attention ($16,384$), Party State GRU Bank ($197,376$), Emotion Attention ($8,192$), Emotion GRU Bank ($49,536$), Classifier Head ($4,615$).
- **Ablated Sub-Networks:** None.
- **Total Parameters:** **670,855**.

---

## 3. Parameter Accounting and Component Inventory

| Architectural Layer / Sub-Network | Layer Specification | Ablation 1 (Global Only) | Ablation 2 (Global + Party) | Ablation 3 (Global + Emotion) | Ablation 4 (Full DialogueRNN) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Input Linear Projection** | Linear(768, 256) | 196,864 | 196,864 | 196,864 | 196,864 |
| **Input LayerNorm** | LayerNorm(256) | 512 | 512 | 512 | 512 |
| **Global Context GRU** | GRUCell($I_G$, 128) | 148,224 ($I_G=256$) | 197,376 ($I_G=384$) | 148,224 ($I_G=256$) | 197,376 ($I_G=384$) |
| **Context Attention** | Linear(128, 128, bias=False) | 16,384 | 16,384 | 16,384 | 16,384 |
| **Party State GRU Bank** | GRUCell(384, 128) | *Omitted (None)* | 197,376 | *Omitted (None)* | 197,376 |
| **Emotion Attention** | Linear(128, 64, bias=False) | *Omitted (None)* | *Omitted (None)* | *Omitted (None)* | 8,192 |
| **Emotion GRU Bank** | GRUCell($I_E$, 64) | *Omitted (None)* | *Omitted (None)* | 86,400 ($I_E=384$) | 49,536 ($I_E=192$) |
| **Classification FC1** | Linear($D_{\text{in}}$, 64) | 8,256 ($D_{\text{in}}=128$) | 8,256 ($D_{\text{in}}=128$) | 4,160 ($D_{\text{in}}=64$) | 4,160 ($D_{\text{in}}=64$) |
| **Classification FC2** | Linear(64, 7) | 455 | 455 | 455 | 455 |
| **Total Trainable Parameters** | **Exact Count** | **370,695** | **617,223** | **452,999** | **670,855** |

---

## 4. Unavoidable Architectural Confounds & Methodological Constraints

> [!WARNING]
> **Model Capacity vs. Inductive Bias Confounding:**
> These ablations **do not isolate architectural effects independently of model capacity**. Adding or removing recurrent sub-networks unavoidably alters total parameter count:
> - Ablation 1 (370,695 params) has **44.7% fewer parameters** than Full DialogueRNN.
> - Ablation 2 (617,223 params) has **8.0% fewer parameters** than Full DialogueRNN.
> - Ablation 3 (452,999 params) has **32.5% fewer parameters** than Full DialogueRNN.
> 
> Any observed empirical performance differences between configurations reflect the joint impact of **structural inductive bias** (e.g., party state segregation) and **parameter capacity**. This limitation must be explicitly reported in all research conclusions.

### Secondary Structural Confounds:
1. **Global GRU Input Dimension Shift:**  
   In Ablations 1 & 3, the Global Context GRU receives only projected utterance $u_t' \in \mathbb{R}^{256}$ ($148,224$ params). In Ablations 2 & 4, it receives $[u_t'; q_{t-1}^{(p)}] \in \mathbb{R}^{384}$ ($197,376$ params). The Global GRU in Ablations 1 & 3 has slightly lower expressive capacity.
2. **Emotion GRU Input Dimension Shift:**  
   In Ablation 3, lacking party states, the shared Emotion GRU receives $[u_t'; c_t] \in \mathbb{R}^{384}$ ($86,400$ params). In Ablation 4, it receives $[q_t^{(p)}; c_t^{(e)}] \in \mathbb{R}^{192}$ ($49,536$ params).
3. **Classification Head Dimension Shift:**  
   In Ablations 1 & 2, the classifier input is 128-dimensional ($8,711$ params); in Ablations 3 & 4, it is 64-dimensional ($4,615$ params).

---

## 5. Test Suite Results & Audit Findings

The complete unit test suite was executed inside the project Docker container (`emotion-dynamics-nlp:week11`).

### 5.1 Ablation Test Suite (`tests/test_dialoguernn_ablations.py`)
All 12 tests passed:
```
test_causality_future_perturbations ....................................... OK
test_checkpoint_loading_strict_keys ....................................... OK
test_disabled_components_genuinely_absent ................................ OK
test_forward_pass_output_shape_and_finite_logits .......................... OK
test_full_model_numerical_equivalence_with_dialoguernn .................... OK
test_gradient_propagation_all_enabled_components ......................... OK
test_multiple_speakers_and_speaker_changes ................................ OK
test_parameter_counts_exact_match ......................................... OK
test_recurrent_state_reset_between_dialogues .............................. OK
test_saved_predictions_exact_reproduction ................................. OK
test_single_dialogue_forward .............................................. OK
test_variable_dialogue_lengths_and_padding_masks .......................... OK
----------------------------------------------------------------------
Ran 12 tests in 0.606s (All passed)
```

### 5.2 Test Audit Highlights:
1. **Causality Invariance:** When an utterance at index $t^* = 4$ was perturbed by $+10.0$ in feature space, logits for all prior turns $t < 4$ showed **$\Delta_{\max} = 0.0$** across all four configurations.
2. **State Reset Invariance:** Evaluating Dialogue B after Dialogue A vs. evaluating Dialogue B in isolation produced identical logits (**$\Delta_{\max} = 0.0$**).
3. **Gradient Flow:** Backward pass through cross-entropy loss produced valid, non-zero gradients across all active parameters in all four configurations.
4. **Checkpoint Equivalence:** Loading `checkpoints/dialoguernn_unweighted/best_model.pt` into [`DialogueRNNAblation`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/models/dialoguernn.py#L370-L620) (Mode 4) with `strict=True` succeeded without missing or unexpected keys, reproducing **2,610 / 2,610 (100.00%)** test predictions from `outputs/experiments/dialoguernn_unweighted/test_predictions.csv`.

### 5.3 Full Regression Suite Status
Running the complete test discovery suite:
```
Ran 33 tests in 1.990s (21 existing regression tests + 12 new ablation tests)
Status: OK (Zero failures, Zero errors)
```

---

## 6. Pre-Training Readiness & Deviations

- **Deviations from Task 13 Validation:** **None.** All parameter counts, configuration mappings, and structural specifications conform precisely to the Task 13 design.
- **Current Repository State:**
  - `src/models/dialoguernn.py`: Implements both `DialogueRNN` and `DialogueRNNAblation`.
  - `tests/test_dialoguernn_ablations.py`: Comprehensive test suite.
  - `configs/ablations/*.json`: Fully verified machine-readable configurations.
  - `outputs/experiments/*`: Historical checkpoints and prediction files preserved intact.

**Task 14 is complete. In strict adherence to restrictions, zero training has been launched. Awaiting explicit user approval before proceeding to the training phase.**
