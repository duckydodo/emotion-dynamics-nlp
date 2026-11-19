# Frozen DistilBERT Feature Extraction Pipeline for DialogueRNN

**Repository:** `emotion-dynamics-nlp`  
**Project Path:** `~/Projects/emotion-dynamics-nlp`  
**Source Checkpoint:** [`checkpoints/transformer_baseline/best`](file:///home/chitta/Projects/emotion-dynamics-nlp/checkpoints/transformer_baseline/best)  
**Extraction Script:** [`scripts/extract_dialogue_features.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/scripts/extract_dialogue_features.py)  
**Verification Script:** [`scripts/verify_extracted_features.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/scripts/verify_extracted_features.py)  
**Data Loader Module:** [`src/data/feature_loader.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/data/feature_loader.py)  
**Feature Archive Directory:** `data/processed/dialoguernn/`  
**Execution Environment:** Docker image `emotion-dynamics-nlp:week11` (CPU execution)  

---

## 1. Executive Summary

To support the DialogueRNN sequential modeling experiment without fine-tuning the 66-million-parameter DistilBERT backbone repeatedly on CPU, we constructed and executed a deterministic, frozen utterance representation extraction pipeline.

### Core Achievements
1. **Canonical Representation Selection:** Evaluated the internal representation topology of the fine-tuned sequence classification model. Extracted the unbounded 768-dimensional final-layer `[CLS]` token hidden state ($h_{\text{CLS}}^{(6)}$), rather than post-ReLU pre-classifier projections or classification logits.
2. **Deterministic & Identical Coverage:** Processed all 13,708 MELD utterances across Train ($N=9,989$), Dev ($N=1,109$), and Test ($N=2,610$). Re-extraction determinism was verified with maximum absolute difference $< 10^{-5}$ ($0.0$).
3. **Storage & Disk Budget:** Produced compressed NumPy `.npz` archives and accompanying JSON sidecar metadata. Total disk footprint for all splits is **36.83 MB**, consuming $<0.05\%$ of disk space (vastly below the 50 MB budget).
4. **Dynamic Party Indexing:** Built a local dialogue party mapper that translates 260 distinct raw speaker strings into compact per-dialogue party indices ($p \in [0, M-1]$, where $1 \le M \le 9$), preventing parameter explosion while preserving multi-party conversational interaction.
5. **Independent Verification:** A standalone test suite verified 100% schema compliance, zero NaN/Inf values, identical row-key alignment to original MELD splits, and valid party-sequence reconstruction.

---

## 2. Representation Architecture & Rationale

### 2.1 Model Architecture Topology

The upstream fine-tuned checkpoint (`checkpoints/transformer_baseline/best`) uses the Hugging Face `DistilBertForSequenceClassification` architecture with the following forward pathway:

```
[Raw Utterance Text]
         │
         ▼
[DistilBERT Tokenizer] (max_length=96, truncation=True, padding=False/batch)
         │
         ▼
[DistilBERT 6-Transformer-Layer Backbone]
         │
         ├──► outputs.last_hidden_state: (B, L, 768)
         │           │
         │           ▼
         │    [CLS] Token Hidden Vector: outputs.last_hidden_state[:, 0, :]  <── [EXTRACTED REPRESENTATION]
         │           │                                                            Shape: (B, 768)
         │           ▼                                                            Manifold: Unbounded ℝ^768
         │    [pre_classifier Linear + ReLU]: (B, 768) ── [Disallowed: clips negative subspace]
         │           │
         │           ▼
         │    [classifier Linear]: (B, 7) ─────────────── [Disallowed: Task-specific logits, destroys context]
```

### 2.2 Rationale for Final-Layer `[CLS]` State

We evaluated three potential extraction layers:
- **Disallowed — Classifier Logits ($\mathbb{R}^7$):**
  Extracting logits collapses the rich 768-dimensional semantic and affective state into a 7-dimensional probability vector tuned specifically to independent utterance emotion classification. This destroys nuanced syntactic and pragmatic cues necessary for conversational tracking and exacerbates majority-class collapse.
- **Disallowed — Post-ReLU Pre-classifier State ($\mathbb{R}_+^{768}$):**
  The pre-classifier applies a `ReLU` non-linearity, projecting all negative activation coordinates to zero. This induces artificial sparsity and destroys directional information across half of the embedding manifold.
- **Selected — Final-Layer `[CLS]` Hidden State ($h_{\text{CLS}}^{(6)} \in \mathbb{R}^{768}$):**
  The un-pooled final transformer hidden representation of `[CLS]` provides an unbounded, continuous vector space that encodes the contextual semantics of the entire utterance as shaped by fine-tuning on the MELD dataset. It serves as the standard input representation $u_t$ required by the DialogueRNN architecture (Majumder et al., AAAI 2019).

### 2.3 Tokenizer Configuration & Batching

- **Max Sequence Length:** 96 tokens (exact configuration used during baseline fine-tuning; avoids artificial truncation while encompassing 99.8% of MELD utterances).
- **Batching:** Mini-batches of $B=64$ utterances on CPU with dynamic padding to the longest sequence in each batch, followed by PyTorch `torch.no_grad()` inference and explicit evaluation mode (`model.eval()`).
- **Reproducibility:** Seed 42 set across Python `random`, NumPy, and PyTorch.

---

## 3. Dynamic Local Party Indexing

In the original MELD corpus, speakers are designated by variable string labels (e.g., `"Chandler"`, `"Joey"`, `"Monica"`, `"Phoebe"`, `"Rachel"`, `"Ross"`, `"Mark"`, `"Waitress"`, `"Gunther"`, etc.). Across the dataset, there are 260 unique speaker strings.

### 3.1 The Global vs. Local Mapping Problem

- **Global Party Mapping:** Assigning a separate GRU cell or projection matrix to every unique speaker in the corpus would require 260 distinct party state modules, the vast majority of which appear in fewer than 5 utterances. This would lead to severe parameter bloat and un-trainable parameters.
- **Local Dynamic Party Mapping:** DialogueRNN tracks speakers *within* a conversation. In any single MELD dialogue:
  - Minimum number of participants: $M_{\min} = 1$
  - Maximum number of participants: $M_{\max} = 9$ (Train), $8$ (Dev/Test)
  - Mean number of participants: $\approx 2.7$

### 3.2 Implemented Mapping Specification

In [`src/data/feature_loader.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/data/feature_loader.py), each dialogue $d$ dynamically maps raw speaker names to local indices $p \in \{0, \dots, M^{(d)} - 1\}$ based on the order of their appearance:

$$\mathcal{S}_d(\text{speaker}) = \text{index of first appearance in dialogue } d$$

For example, in a 4-turn dialogue:
```
Turn 0: Chandler  ──► Party 0
Turn 1: Monica    ──► Party 1
Turn 2: Chandler  ──► Party 0
Turn 3: Joey      ──► Party 2
Total Parties M = 3
```

This local indexing allows DialogueRNN to initialize party state tensors with fixed dimension $M_{\text{max\_batch}} \le 9$, or dynamically index party GRUs without parameter bloat.

---

## 4. Storage Architecture & Archive Specifications

Features are saved under `data/processed/dialoguernn/` using compressed NumPy format (`.npz`) paired with human-readable JSON sidecar metadata.

### 4.1 Schema Definition

Each `.npz` archive contains 5 aligned arrays:
1. `features` (`float32`, shape: $(N, 768)$): Dense utterance embedding matrix.
2. `dialogue_ids` (`int32`, shape: $(N,)$): Original MELD `Dialogue_ID`.
3. `utterance_ids` (`int32`, shape: $(N,)$): Original MELD `Utterance_ID`.
4. `emotions` (`int32`, shape: $(N,)$): Canonical numeric emotion label ($0 \dots 6$).
5. `party_indices` (`int32`, shape: $(N,)$): Dynamic local party index ($0 \dots M^{(d)}-1$).

The companion `.json` sidecar stores:
- `split`: Split name (`train`, `dev`, `test`).
- `utterance_count`: Total utterance count $N$.
- `dialogue_count`: Total dialogue count $D$.
- `feature_dim`: Embedding dimensionality (768).
- `speakers`: List of $N$ original speaker strings.
- `dialogue_to_speakers`: Dictionary mapping each `dialogue_id` to its ordered list of distinct speaker names.
- `summary_statistics`: Numerical distribution moments (min, max, mean, std, L2 norm).

---

## 5. Verification & Empirical Feature Statistics

### 5.1 Verification Results Table

The standalone verification script [`scripts/verify_extracted_features.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/scripts/verify_extracted_features.py) executed inside Docker container `emotion-dynamics-nlp:week11` confirmed complete integrity across all splits:

| Split | Utterance Count $N$ | Dialogue Count $D$ | Feature Dim $D_u$ | Min Value | Max Value | Mean | Std Dev | Mean L2 Norm | File Size (.npz) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 9,989 | 1,038 | 768 | -7.4121 | 3.8592 | -0.0170 | 0.5850 | 16.1971 | 26.85 MB |
| **Dev** | 1,109 | 114 | 768 | -4.4240 | 3.2791 | -0.0164 | 0.5782 | 16.0057 | 2.98 MB |
| **Test** | 2,610 | 280 | 768 | -5.0476 | 3.1817 | -0.0166 | 0.5801 | 16.0599 | 7.00 MB |
| **Total** | **13,708** | **1,432** | **768** | — | — | — | — | — | **36.83 MB** |

### 5.2 Verification Checks Summary

1. **Count & Dimension Invariant:** Exact row counts match original MELD CSV splits ($N_{\text{train}}=9,989$, $N_{\text{dev}}=1,109$, $N_{\text{test}}=2,610$). Feature dimension is exactly 768 for every utterance.
2. **Missing Utterances & Gaps:** All 13,708 utterances are preserved. Utterance ID gaps (e.g., Dialogue 93 and 108 in Test) are strictly preserved in their original natural sequence order without artificial omission.
3. **Finite Representation Invariant:** Checked $13,708 \times 768 = 10,527,744$ float coordinates: 0 NaNs, 0 Infs, 0 zero-norm vectors.
4. **Distributional Stability:** Feature distributions are virtually identical across Train, Dev, and Test:
   - Means are centered near zero ($\mu \approx -0.0166$ to $-0.0170$).
   - Standard deviations are stable ($\sigma \approx 0.578$ to $0.585$).
   - Mean L2 norms are tightly bounded ($\|\mathbf{u}\|_2 \approx 16.01$ to $16.20$).
5. **Determinism Verification:** Re-extracting a 32-sample slice for Train, Dev, and Test produced identical values ($\max |\Delta| = 0.0 < 10^{-5}$).
6. **Key Alignment:** 100% match between extracted `(dialogue_id, utterance_id, emotion)` tuples and the canonical raw MELD CSV files.

---

## 6. Integration Guide for DialogueRNN Training

### 6.1 Loading Dialogue Sequences

[`src/data/feature_loader.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/data/feature_loader.py) provides two primary interfaces for consuming extracted features:

#### Direct Dialogue Loading
```python
from src.data.feature_loader import load_dialogue_features

# Loads a list of DialogueSequence objects
dialogues, metadata = load_dialogue_features("data/processed/dialoguernn/meld_train_features.npz")

for diag in dialogues:
    # diag.dialogue_id: int
    # diag.features: torch.Tensor of shape (T, 768)
    # diag.emotions: torch.Tensor of shape (T,)
    # diag.party_indices: torch.Tensor of shape (T,)
    # diag.speakers: list of length T
    # diag.num_parties: int (number of unique parties in this dialogue)
    # diag.length: int (number of utterances T)
    pass
```

#### PyTorch Dataset & DataLoader
```python
from src.data.feature_loader import DialogueFeatureDataset, dialogue_collate_fn
from torch.utils.data import DataLoader

train_dataset = DialogueFeatureDataset("data/processed/dialoguernn/meld_train_features.npz")

# Batching dialogues (e.g., batch_size=8 dialogues)
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=dialogue_collate_fn
)

for batch in train_loader:
    # batch['features']: (B, T_max, 768) padded tensor
    # batch['emotions']: (B, T_max) padded tensor (-100 for pad)
    # batch['party_indices']: (B, T_max) padded tensor
    # batch['mask']: (B, T_max) boolean attention mask
    # batch['lengths']: (B,) sequence lengths
    # batch['num_parties']: (B,) number of parties per dialogue
    pass
```

### 6.2 Alignment with Canonical Evaluation Cohorts

When evaluating trained DialogueRNN checkpoints on the test set:
- **Cohort A (Strict Contiguous Matched Subset, $N=2,328$):** Filter predictions using the canonical key map `canonical_cohort_a_keys.json` to directly compare against Context-1 and Context-1 + Speaker baselines.
- **Cohort B (All Subsequent Sequential Turns, $N=2,330$):** Evaluate turns with $t \ge 1$ within each dialogue to compare against autoregressive recurrent models.
- **Full Test Set ($N=2,610$):** DialogueRNN naturally predicts emotions for *all* utterances ($t \ge 0$), including the initial turn of each dialogue (where party state is initialized to zero).

---

## 7. Next Research Steps

With the feature extraction pipeline completed, verified, and documented:
1. Implement the modular PyTorch DialogueRNN architecture specified in [`outputs/reports/dialoguernn_implementation_spec.md`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/reports/dialoguernn_implementation_spec.md).
2. Construct the training script (`scripts/train_dialoguernn.py`) with class-weighted loss and gradient clipping.
3. Train DialogueRNN on CPU inside the Docker container using the cached 36.83 MB feature archives.
4. Evaluate predictions on Cohort A, Cohort B, and the full test set, running paired bootstrap statistical significance tests against the verified baselines.
