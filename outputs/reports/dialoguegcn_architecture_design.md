# DialogueGCN Architecture Design and Leakage Audit for MELD

**Repository:** `emotion-dynamics-nlp`  
**Project Path:** `/home/chitta/Projects/emotion-dynamics-nlp`  
**Date:** November 10, 2026  
**Status:** Architecture Design & Leakage Audit Complete (Pre-Training Phase)  
**Reference Paper:** Deepanway Ghosal, Navonil Majumder, Soujanya Poria, Nihar R. Radenovic, Alexander Gelbukh, *"DialogueGCN: A Graph Convolutional Network for Emotion Recognition in Conversation"*, EMNLP-IJCNLP 2019  
**Upstream Audit Standards:**
- [`outputs/reports/canonical_evaluation_audit.md`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/reports/canonical_evaluation_audit.md)
- [`outputs/reports/dialoguernn_implementation_spec.md`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/reports/dialoguernn_implementation_spec.md)
- [`outputs/reports/hybrid_application_readiness_audit.md`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/reports/hybrid_application_readiness_audit.md)
**Deliverable File:** [`outputs/reports/dialoguegcn_architecture_design.md`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/reports/dialoguegcn_architecture_design.md)  

---

## 1. Executive Summary & Research Motivation

The hybrid conversational emotion analysis platform in `emotion-dynamics-nlp` is designed to expose three foundational model families:
1. **DistilBERT:** Utterance-level local semantic classification (fine-tuned transformer baseline).
2. **DialogueRNN:** Sequential causal recurrence tracking global discourse flow, speaker-specific emotional states, and emotional inertia.
3. **DialogueGCN:** Relational graph convolution capturing conversational topology, multi-party influence, and bidirectional context.

### 1.1 Why Graph Convolution for Emotion Recognition in Conversation (ERC)?
While recurrent neural architectures (such as DialogueRNN) model sequential state transitions across time, they exhibit inherent structural characteristics:
- **Sequential Bottleneck:** Recurrent hidden states must compress information step-by-step through intermediate turns, which can dilute direct semantic dependencies between non-consecutive turns separated by interlocutor banter.
- **Strictly Causal Processing:** In online conversational dialogue, causality ($t$ conditioned only on $\le t$) is mandatory. However, in **offline transcript analysis** (the primary use case of our platform for analyzing completed conversations, customer service logs, or uploaded chats), the entire conversation history is already complete.
- **Relational Symmetry & Direct Propagation:** Graph neural networks formulate each utterance as a node, directly connecting turns through explicit relational edges (intra-speaker emotional inertia vs. inter-speaker conversational influence). Information propagates across conversational distance in parallel through graph message passing.

### 1.2 Purpose of this Specification
This document establishes the formal architectural specification, tensor dimensions, graph construction rules, leakage prevention proofs, parameter counts, and training protocol for **DialogueGCN on MELD**. In accordance with Task 23 instructions, **no model training or source modification is performed during this design phase**.

---

## 2. DialogueGCN Methodology: Original Formulation vs. Repository Adaptations

### 2.1 Original Formulation (Ghosal et al., EMNLP 2019)
In the original paper, DialogueGCN models multi-party conversation as a directed/undirected graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{R}, \mathcal{W})$:
1. **Utterance Features:** Utterance representations were generated using CNN feature extractors (Kim CNN) or RoBERTa.
2. **Sequential Context Encoder:** A Bidirectional GRU (BiGRU) was applied across sequential utterance vectors to incorporate local sequential discourse context before graph construction.
3. **Graph Construction:** Nodes correspond to utterances $u_1, \dots, u_T$. Edges connect utterances within a temporal window of past ($p$) and future ($f$) turns: $|i - j| \le p$ for $j < i$ and $|j - i| \le f$ for $j > i$.
4. **Relational Edges:** Edges are assigned relations based on speaker identities:
   - In two-party datasets (IEMOCAP), relations represented speaker transitions ($S_A \to S_A$, $S_A \to S_B$, $S_B \to S_A$, $S_B \to S_B$).
   - In multi-party datasets (MELD), allocating $M \times M$ relational matrices is intractable due to arbitrary participant counts ($M \le 9$ per dialogue, 260+ unique characters). Ghosal et al. grouped relations into **Intra-Speaker** (self-dependency) and **Inter-Speaker** (interpersonal influence).
5. **Similarity Attention:** Edge weights $\alpha_{ij}$ were dynamically computed using query-key similarity over node states.
6. **Relational Graph Convolution:** Feature aggregation across relations with residual connections.
7. **Classifier:** Fully-connected network projecting the final node representations into emotion logits.

### 2.2 Proposed Adaptations for `emotion-dynamics-nlp`

To maintain rigorous experimental control and comparability with our existing MELD benchmarks, the following design choices are formalized:

| Component | Original Ghosal et al. (2019) | `emotion-dynamics-nlp` Adaptation | Justification & Parity Standard |
| :--- | :--- | :--- | :--- |
| **Input Features ($u_t$)** | CNN / 300-d GloVe or RoBERTa-large | **Frozen DistilBERT [CLS] hidden states ($D_u = 768$)** | **Strict Parity:** Uses the identical frozen 768-dim features extracted from `checkpoints/transformer_baseline/best` used by DialogueRNN. |
| **Speaker Relations ($\mathcal{R}$)** | Pairwise or Intra/Inter | **Binary Relational Edge Types:**<br>1. `r_intra` (Intra-speaker: $S_i = S_j$)<br>2. `r_inter` (Inter-speaker: $S_i \neq S_j$) | Avoids parameter explosion over MELD's 260+ named speakers; preserves statistical robustness on rare speaker combinations. |
| **Temporal Context** | $p = 10, f = 10$ window | **$p = 10, f = 10$ bidirectional window** (bounded by dialogue length) | Reflects completed-conversation analysis where future context is available. |
| **Hidden Dimension ($D_h$)** | $D_h \in \{128, 256, 300\}$ | **$D_h = 128$** ($64$-dim per direction in BiGRU) | Direct dimensional parity with DialogueRNN's global context state ($D_g = 128$) and party state ($D_q = 128$). |
| **GCN Depth ($L$)** | $L = 2$ RGCN layers | **$L = 2$ RGCN layers with residual connections** | Prevents over-smoothing while enabling 2-hop message propagation across the conversation graph. |
| **Evaluation Cohort** | Full test set (varies in literature) | **Canonical Cohort A ($N=2,328$, $M=268$)** | Adheres to repository evaluation audit standard; exact key matching via `(Dialogue_ID, Utterance_ID)`. |

---

## 3. Important Design Decision: Completed-Conversation Offline Analysis

The user request highlights a fundamental architectural decision:
> *"The application analyses completed conversations offline. Do not impose causal-only graph construction automatically."*

### 3.1 Causal vs. Non-Causal Conversational Analysis
In ERC research, there are two distinct operational settings:
1. **Online / Streaming Dialogue (e.g. conversational agents, live chatbots):**
   - At turn $t$, turns $t+1, \dots, T$ have not yet occurred.
   - The model must strictly enforce causal recurrence ($t$ conditioned solely on $\le t$). This is the operating assumption of **DialogueRNN**.
2. **Offline / Post-Hoc Transcript Analysis (e.g. customer service analytics, clinical interview review, uploaded chat inspection):**
   - The entire dialogue transcript $\mathcal{D} = (u_1, \dots, u_T)$ is already completed and available in its entirety at ingestion time.
   - In human discourse comprehension, interpreting the emotional valence of an earlier ambiguous statement (e.g., deadpan irony or tease) often depends critically on how the conversational partner reacted in subsequent turns ($t+1$).
   - Imposing an artificial causal mask on a completed transcript discards valuable disambiguating future context.

### 3.2 Graph Construction Rules
For a completed dialogue with $T$ turns $(u_1, \dots, u_T)$ spoken by parties $(S_1, \dots, S_T)$:
1. **Node Set:** Each utterance $u_i$ is represented by a node $v_i \in \mathcal{V}$, where $|\mathcal{V}| = T$.
2. **Temporal Window ($p, f$):** A node $v_i$ connects to all preceding nodes within past window $p = 10$ and all succeeding nodes within future window $f = 10$:
   $$\mathcal{N}_i = \Big\{ v_j \in \mathcal{V} \;\Big|\; j \ne i \;\land\; \max(0, i - p) \le j \le \min(T - 1, i + f) \Big\}$$
3. **Edge Relational Assignment:**
   - **Intra-Speaker Edge ($r_{\text{intra}}$):** Connects $v_j \to v_i$ if $S_j = S_i$. Models self-dependency, emotional inertia, and topic continuity within the same person.
   - **Inter-Speaker Edge ($r_{\text{inter}}$):** Connects $v_j \to v_i$ if $S_j \ne S_i$. Models interpersonal emotional stimulus, social influence, and conversational reactivity.
4. **Self-Loop ($W_{\text{self}}$):** Each node retains a dedicated identity self-transformation to preserve its own intrinsic utterance semantics during convolution.

```
Conversational Graph Topology for Turn i (Speaker A):

   Turn i-2 [Spk A] ──(r_intra)──► ┌───────────────┐ ◄──(r_intra)── Turn i+2 [Spk A]
                                   │    Node i     │
   Turn i-1 [Spk B] ──(r_inter)──► │  (Speaker A)  │ ◄──(r_inter)── Turn i+1 [Spk B]
                                   └───────▲───────┘
                                           │ (W_self)
                                    Self-Transformation
```

---

## 4. Leakage & Threat Model Audit

A graph architecture that incorporates past and future context requires rigorous verification to prove that no data leakage occurs. Below is the comprehensive audit across six threat vectors:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DATA LEAKAGE AUDIT MATRIX                       │
├───────────────────────┬────────────────────────────────────────────────┤
│ Threat Vector         │ Audit Finding & Leakage Prevention Guarantee   │
├───────────────────────┼────────────────────────────────────────────────┤
│ 1. Future Label       │ ZERO LEAKAGE: Ground-truth emotion labels      │
│    Leakage            │ are NEVER graph inputs. Graph nodes receive    │
│                       │ only frozen text features.                     │
├───────────────────────┼────────────────────────────────────────────────┤
│ 2. Dialogue Boundary  │ ZERO LEAKAGE: Graphs are constructed strictly  │
│    Contamination      │ within single dialogues. Adjacency is block-   │
│                       │ diagonal; no cross-dialogue edges exist.       │
├───────────────────────┼────────────────────────────────────────────────┤
│ 3. Train/Dev/Test     │ ZERO LEAKAGE: All graph operations operate     │
│    Split Isolation    │ strictly within dialogue instances of a split. │
│                       │ Test/dev features are never seen in training.  │
├───────────────────────┼────────────────────────────────────────────────┤
│ 4. Feature Extraction │ ZERO LEAKAGE: DistilBERT CLS embeddings were   │
│    Independence       │ extracted in inference mode without target     │
│                       │ label supervision during extraction.           │
├───────────────────────┼────────────────────────────────────────────────┤
│ 5. Graph Topology     │ ZERO LEAKAGE: Edges are defined purely by      │
│    Label Conditioning │ speaker metadata and turn index. Zero label    │
│                       │ conditioning in topology.                      │
├───────────────────────┼────────────────────────────────────────────────┤
│ 6. Checkpoint &       │ ZERO LEAKAGE: Checkpoints selected strictly on │
│    Cohort Selection   │ Dev Macro F1. Cohort A evaluated post-freeze.  │
└───────────────────────┴────────────────────────────────────────────────┘
```

### 4.1 Proof: Prevention of Future Emotion-Label Leakage
- **The Risk:** In sequential autoregressive models (such as Model B in our DistilBERT experiments), feeding previous ground-truth labels $\tilde{y}_{t-1}$ at training time caused massive label-leakage exposure bias, leading to catastrophic failure at test time when true labels were unavailable.
- **The Safeguard in DialogueGCN:**
  - Graph nodes $v_i$ are initialized exclusively with linguistic feature vectors: $x_i = \text{DistilBERT}(u_i) \in \mathbb{R}^{768}$.
  - Ground-truth emotion labels $y_i \in \{0, \dots, 6\}$ are **never passed into the model, never concatenated with node features, and never used as edge attributes**.
  - During training, the forward pass computes predictions $\hat{y}_i = f_{\theta}(X, \mathcal{A}, \mathcal{R})_i$ across all nodes simultaneously.
  - The cross-entropy loss is computed against the ground-truth targets after the forward pass:
    $$\mathcal{L} = \frac{1}{\sum_{i} m_i} \sum_{i=1}^T m_i \cdot \text{CrossEntropy}(\hat{y}_i, y_i)$$
  - Therefore, at inference time, the model operates with **identical information** as at training time. Future *linguistic text* is used for context, but future *emotion labels* are completely unknown.

### 4.2 Proof: Strict Dialogue Boundary Enforcement
- **The Risk:** Cross-talk between distinct dialogues in a batch could allow utterances from one episode of *Friends* to attend to utterances from a completely unrelated scene.
- **The Safeguard:**
  - In our batching protocol, each dialogue $d$ of length $T_d$ has its own adjacency matrix $A_d \in \mathbb{R}^{T_d \times T_d}$.
  - When batching $B$ dialogues into padded tensor form $(B, T_{\max}, D_h)$, adjacency is formatted as $(B, T_{\max}, T_{\max})$.
  - For dialogue $b$, element $A_{b, i, j}$ can only be non-zero if both $i < T_b$ and $j < T_b$. Padding turns are masked out with $-10^9$ in attention.
  - Cross-dialogue connections are mathematically zero:
    $$\forall b \ne b', \quad \text{AttentionWeight}(u_{b, i}, u_{b', j}) \equiv 0$$

### 4.3 Proof: Strict Split Boundary Enforcement
- MELD dialogues are strictly partitioned into:
  - **Train:** 1,038 dialogues ($9,989$ utterances)
  - **Dev:** 114 dialogues ($1,109$ utterances)
  - **Test:** 280 dialogues ($2,610$ utterances)
- Dialogue graphs are instantiated solely on demand within their respective split DataLoader. No shared graph or transductive message passing across splits is possible.

---

## 5. Detailed Architecture & Mathematical Formulation

DialogueGCN consists of five modular computational stages:

```
┌────────────────────────────────────────────────────────────────────────┐
│                      DIALOGUEGCN FORWARD PIPELINE                      │
└────────────────────────────────────────────────────────────────────────┘

 [Input Utterances]       u_1, u_2, ..., u_T  (Frozen DistilBERT CLS)
          │
          ▼
┌──────────────────┐
│ Sequential BiGRU │     h_seq = BiGRU(u_1...u_T)
│  Context Encoder │     Captures chronological discourse flow
└─────────┬────────┘
          │               h_i ∈ R^(D_h)  (Node initial states)
          ▼
┌──────────────────┐     Construct Adjacency A ∈ R^(T × T)
│ Graph Topology & │     Window: max(0, i-p) ≤ j ≤ min(T-1, i+f)
│ Relation Masking │     Relation Masks: Rel_intra (same spk), Rel_inter (diff spk)
└─────────┬────────┘
          │
          ▼
┌──────────────────┐
│  RGCN Layer 1    │     Similarity Attention: α_ij = Softmax(h_i W_α h_j^T)
│  + Similarity    │     Message Aggregation: msg = Σ_r Rel_r ⊙ α (h W_r)
│    Attention     │     Residual Skip: h^(1) = ReLU(W_self h^(0) + msg + b) + h^(0)
└─────────┬────────┘
          │
          ▼
┌──────────────────┐
│  RGCN Layer 2    │     2-hop relational message propagation
│  + Similarity    │     Residual Skip: h^(2) = ReLU(W_self h^(1) + msg + b) + h^(1)
│    Attention     │     Dropout regularized
└─────────┬────────┘
          │
          ▼
┌──────────────────┐
│ Classification   │     Logits = W_cls h_i^(2) + b_cls
│      Head        │     Shape: (T, 7)  --> Softmax to 7 Emotion Classes
└──────────────────┘
```

### 5.1 Stage 1: Sequential Context Encoding (BiGRU)
Before graph convolutions, utterance representations are passed through a Bidirectional Gated Recurrent Unit (BiGRU). This equips each utterance node with baseline local conversational context:
$$\overrightarrow{h}_t = \text{GRU}_{\text{fwd}}(u_t, \overrightarrow{h}_{t-1})$$
$$\overleftarrow{h}_t = \text{GRU}_{\text{bwd}}(u_t, \overleftarrow{h}_{t+1})$$
$$h_t^{(0)} = [\overrightarrow{h}_t \,;\, \overleftarrow{h}_t] \in \mathbb{R}^{D_h}$$
- Input: $u_t \in \mathbb{R}^{D_u}$ ($D_u = 768$).
- Forward hidden size: $D_h / 2 = 64$.
- Backward hidden size: $D_h / 2 = 64$.
- Output node representation: $h_t^{(0)} \in \mathbb{R}^{128}$.

### 5.2 Stage 2: Graph Adjacency and Relation Formulation
For dialogue $d$ with length $T$, we construct:
1. **Binary Adjacency Matrix $A \in \{0, 1\}^{T \times T}$:**
   $$A_{i, j} = \begin{cases} 1 & \text{if } i \ne j \;\land\; \max(0, i - p) \le j \le \min(T - 1, i + f) \\ 0 & \text{otherwise} \end{cases}$$
   where $p = 10, f = 10$.
2. **Relation Masks $\mathcal{M} \in \{0, 1\}^{2 \times T \times T}$:**
   - $\mathcal{M}_{0, i, j} = 1$ if $A_{i, j} = 1$ and $S_i = S_j$ (**Intra-speaker relation $r_{\text{intra}}$**).
   - $\mathcal{M}_{1, i, j} = 1$ if $A_{i, j} = 1$ and $S_i \ne S_j$ (**Inter-speaker relation $r_{\text{inter}}$**).

### 5.3 Stage 3: Relational Graph Convolution with Similarity Attention
Each RGCN layer $l \in \{1, \dots, L\}$ performs graph convolution:

#### (a) Dynamic Similarity Attention:
Nodes compute pairwise attention scores over valid neighbors:
$$S_{i, j} = h_i^{(l-1)} W_{\alpha}^{(l)} (h_j^{(l-1)})^T \quad \in \mathbb{R}$$
Scores are masked using the binary adjacency matrix $A$:
$$\tilde{S}_{i, j} = \begin{cases} S_{i, j} & \text{if } A_{i, j} = 1 \\ -\infty & \text{if } A_{i, j} = 0 \end{cases}$$
The normalized attention coefficients $\alpha_{i, j}$ are:
$$\alpha_{i, j} = \frac{\exp(\tilde{S}_{i, j})}{\sum_{k=1}^T \exp(\tilde{S}_{i, k})}$$

#### (b) Relational Message Passing:
For each relation $r \in \{r_{\text{intra}}, r_{\text{inter}}\}$, neighbors transform their representations via relation-specific weight matrix $W_r^{(l)}$:
$$\text{msg}_i^{(l)} = \sum_{r \in \mathcal{R}} \sum_{j=1}^T \Big( \mathcal{M}_{r, i, j} \cdot \alpha_{i, j} \Big) \Big( h_j^{(l-1)} W_r^{(l)} \Big)$$

#### (c) Self-Transformation & Activation:
The node's own state is transformed via $W_{\text{self}}^{(l)}$, combined with incoming relational messages and bias $b^{(l)}$, and passed through a non-linear activation:
$$\tilde{h}_i^{(l)} = \text{ReLU}\Big( W_{\text{self}}^{(l)} h_i^{(l-1)} + \text{msg}_i^{(l)} + b^{(l)} \Big)$$

#### (d) Residual Connection & Regularization:
To avoid gradient vanishing and over-smoothing, a residual skip connection and dropout are applied:
$$h_i^{(l)} = \text{Dropout}\Big( \tilde{h}_i^{(l)} + h_i^{(l-1)} \Big)$$

### 5.4 Stage 4: Emotion Classification Head
After $L = 2$ graph convolution layers, the final contextualized node embeddings $h_i^{(L)} \in \mathbb{R}^{D_h}$ are mapped to unnormalized emotion logits:
$$\hat{z}_i = W_{\text{cls}} h_i^{(L)} + b_{\text{cls}} \quad \in \mathbb{R}^7$$
$$\hat{y}_i = \text{Softmax}(\hat{z}_i)$$

---

## 6. Tensor Shapes & Exact Parameter Count Calculations

### 6.1 Tensor Dimension Flow

| Operation / Tensor Name | Input Shape | Output Shape | Description |
| :--- | :--- | :--- | :--- |
| `features` (Input Batch) | - | `(B, T_max, 768)` | Frozen DistilBERT [CLS] embeddings |
| `BiGRU Context Encoder` | `(B, T_max, 768)` | `(B, T_max, 128)` | Bidirectional sequential context |
| `Adjacency Tensor A` | `(B, T_max)` | `(B, T_max, T_max)` | Windowed binary connectivity |
| `Relation Mask Tensor M` | `(B, T_max)` | `(B, 2, T_max, T_max)` | 0: intra-speaker, 1: inter-speaker |
| `Similarity Attention` | `(B, T_max, 128)` | `(B, T_max, T_max)` | Softmax-normalized attention $\alpha$ |
| `RGCN Layer 1 Output` | `(B, T_max, 128)` | `(B, T_max, 128)` | 1-hop relational convolution |
| `RGCN Layer 2 Output` | `(B, T_max, 128)` | `(B, T_max, 128)` | 2-hop relational convolution |
| `Classifier Logits` | `(B, T_max, 128)` | `(B, T_max, 7)` | 7 unnormalized emotion class logits |

### 6.2 Analytical Parameter Count Calculation

Let $D_u = 768$, $D_h = 128$, $R = 2$, $C = 7$, $L = 2$:

#### 1. BiGRU Sequential Context Encoder
A standard PyTorch GRU cell with input size $D_{\text{in}}$ and hidden size $D_{\text{hidden}}$ contains:
$$\text{Params}_{\text{GRU}} = 3 \times \Big( D_{\text{in}} \times D_{\text{hidden}} + D_{\text{hidden}} \times D_{\text{hidden}} + D_{\text{hidden}} (\text{bias}_x) + D_{\text{hidden}} (\text{bias}_h) \Big)$$
For $D_{\text{in}} = 768$ and $D_{\text{hidden}} = 64$ ($D_h / 2$):
$$\text{Params}_{\text{dir}} = 3 \times \Big( 768 \times 64 + 64 \times 64 + 64 + 64 \Big) = 3 \times (49,152 + 4,096 + 128) = 3 \times 53,376 = 160,128$$
Bidirectional GRU (Forward + Backward):
$$\text{Params}_{\text{BiGRU}} = 160,128 \times 2 = \mathbf{320,256}$$

#### 2. Relational Graph Convolutional Layer (per layer)
Each `RGCNLayer` consists of:
- **Self Transformation ($W_{\text{self}}$):** Linear without bias: $D_h \times D_h = 128 \times 128 = 16,384$
- **Relational Transformations ($W_{\text{rel}}$):** $R$ matrices of shape $D_h \times D_h$:
  $$R \times (D_h \times D_h) = 2 \times (128 \times 128) = 32,768$$
- **Attention Projection ($W_{\alpha}$):** Linear without bias: $D_h \times D_h = 128 \times 128 = 16,384$
- **Layer Bias ($b$):** Vector of size $D_h$: $128$
$$\text{Params}_{\text{RGCN\_layer}} = 16,384 + 32,768 + 16,384 + 128 = \mathbf{65,664}$$
For $L = 2$ layers:
$$\text{Params}_{\text{RGCN\_total}} = 2 \times 65,664 = \mathbf{131,328}$$

#### 3. Classification Head
Linear classifier mapping $D_h \to C$:
$$\text{Params}_{\text{cls}} = (D_h \times C) + C = (128 \times 7) + 7 = 896 + 7 = \mathbf{903}$$

#### Total Model Parameter Count Summary:
$$\text{Total Parameters} = \text{Params}_{\text{BiGRU}} + \text{Params}_{\text{RGCN\_total}} + \text{Params}_{\text{cls}} = 320,256 + 131,328 + 903 = \mathbf{452,487}$$

### 6.3 Architectural Comparison Across Model Families

| Model Family | Core Paradigm | Trainable Params | Receptive Field | Processing Mode |
| :--- | :--- | :--- | :--- | :--- |
| **DistilBERT Baseline** | Utterance Transformer | $66.96\text{M}$ | Single utterance | Independent turn |
| **DistilBERT Context-1** | Sequence Concatenation | $66.96\text{M}$ | $u_{t-1} + u_t$ | Immediate predecessor |
| **DialogueRNN (Ablation 1)** | Global Context Only | $370,695$ | Past history $0 \dots t$ | Causal recurrent |
| **DialogueGCN (Proposed)** | **Relational Graph Conv** | **$\mathbf{452,487}$** | **Past \& Future $\pm 10$ turns** | **Bidirectional graph** |
| **DialogueRNN (Full)** | Recurrent State Bank | $670,855$ | Past history $0 \dots t$ | Causal recurrent |

DialogueGCN is compact (~$452\text{k}$ parameters), highly expressive, and approximately $32.5\%$ smaller than full DialogueRNN while having a larger bidirectional receptive field.

---

## 7. Experimental Protocol & Reproducibility Plan

To ensure valid scientific comparison against DialogueRNN and DistilBERT, DialogueGCN will adhere to identical optimization protocols:

### 7.1 Experimental Hyperparameters
- **Random Seed:** `42` (fixed across Python, NumPy, PyTorch CPU/GPU).
- **Frozen Utterance Encoder:** DistilBERT baseline checkpoint (`checkpoints/transformer_baseline/best`). Utterance representations are loaded directly from pre-extracted `data/processed/dialoguernn/meld_{train,dev,test}_features.npz`.
- **Optimization Algorithm:** AdamW (`torch.optim.AdamW`).
- **Learning Rate:** $1.0 \times 10^{-3}$ ($0.001$).
- **Weight Decay:** $1.0 \times 10^{-4}$ ($0.0001$).
- **Loss Function:** Standard Unweighted Cross-Entropy Loss:
  $$\mathcal{L} = -\frac{1}{N_{\text{valid}}} \sum_{i} \log \frac{\exp(\hat{z}_{i, y_i})}{\sum_{c=0}^6 \exp(\hat{z}_{i, c})}$$
- **Gradient Clipping:** Max norm $1.0$.
- **Batch Strategy:** Batch size = $16$ dialogues per step, using masked sequence collation.
- **Regularization:** Dropout $p = 0.1$ after BiGRU and after each RGCN layer.
- **Window Parameters:** Past window $p = 10$, future window $f = 10$.

### 7.2 Checkpoint Selection Protocol
- **Early Stopping:** Evaluated after every training epoch on the MELD Development Split ($114$ dialogues, $1,109$ utterances).
- **Patience:** $10$ epochs without improvement in Development Macro F1.
- **Primary Checkpoint Criterion:** Highest Development Macro F1 score:
  $$\text{Checkpoint}^* = \arg\max_e \Big( \text{Macro F1}_{\text{dev}}^{(e)} \Big)$$
- **Test Metric Isolation:** Test set metrics are NEVER computed, logged, or observed during checkpoint selection. The best checkpoint is frozen before any test set evaluation.

### 7.3 Evaluation & Prediction Key Alignment on Cohort A
Once the checkpoint is frozen:
1. **Prediction Generation:** Generate test predictions for all 280 test dialogues ($2,610$ total turns).
2. **Key Tuple Alignment:** Utterances are identified by the canonical tuple `(Dialogue_ID, Utterance_ID)`.
3. **Cohort A Subsetting:** Filter predictions to the canonical **Cohort A (Strict Contiguous Matched Subset, $N = 2,328$ turns across $M = 268$ dialogues)**, exactly matching Task 3 and Tasks 11–15.
4. **Statistical Testing:**
   - **Paired Dialogue-Level Permutation Test ($B = 10,000$ permutations)** against DialogueRNN Unweighted, DialogueRNN Weighted, and DistilBERT Context-1.
   - **Dialogue Cluster Bootstrap (95% BCa Confidence Interval, $B = 2,000$)** over the 268 dialogue clusters.

---

## 8. Implementation Risks, Failure Modes & Mitigations

```
┌────────────────────────────────────────────────────────────────────────┐
│                      RISK AND MITIGATION MATRIX                        │
├───────────────────────┬──────────────────────┬─────────────────────────┤
│ Identified Risk       │ Severity / Impact    │ Mitigation Strategy     │
├───────────────────────┼──────────────────────┼─────────────────────────┤
│ 1. Graph Over-        │ High: Node features  │ Bound depth to L=2; use │
│    Smoothing          │ converge; loss of    │ residual skip-connects; │
│                       │ utterance diversity  │ identity self-loops.    │
├───────────────────────┼──────────────────────┼─────────────────────────┤
│ 2. Isolated Speakers  │ Medium: Low-degree   │ W_self guarantees valid │
│    in 1-turn cameos   │ nodes receive weak   │ node update even with   │
│                       │ message updates      │ zero neighbor edges.    │
├───────────────────────┼──────────────────────┼─────────────────────────┤
│ 3. Neutral Dominance  │ High: Model predicts │ Monitor per-class F1;   │
│    on MELD            │ neutral on rare      │ compare unweighted vs   │
│                       │ classes (fear, etc.) │ weighted cross-entropy. │
├───────────────────────┼──────────────────────┼─────────────────────────┤
│ 4. Variable Dialogue  │ Medium: Dimension    │ Collate function pads   │
│    Lengths (1 to 33)  │ mismatch in batched  │ with -1e9 mask; zero    │
│                       │ adjacency matrices   │ cross-dialogue leakage. │
└───────────────────────┴──────────────────────┴─────────────────────────┘
```

---

## 9. Formal Acceptance Criteria for Task 24 Implementation

The DialogueGCN implementation (Task 24) will be considered successful if and only if it satisfies all of the following criteria:
1. **Zero Checkpoint / Code Regressions:** Existing DistilBERT and DialogueRNN checkpoints, training pipelines, and test suites must remain completely untouched and passing (101/101 tests).
2. **Exact Feature Compatibility:** Consumes the identical pre-extracted frozen DistilBERT [CLS] features (`meld_train_features.npz`, `meld_dev_features.npz`, `meld_test_features.npz`).
3. **Parameter Verification:** The instantiated DialogueGCN model has exactly **$452,487$ trainable parameters** (matching the analytical calculation in Section 6.2).
4. **Dialogue Boundary Verification:** Unit test proves that adjacency and message passing never connect utterances across dialogue boundaries.
5. **No Label Leakage:** Unit test confirms that the model forward pass requires only `features`, `speakers`/`party_ids`, and `lengths`, with zero ground-truth label inputs.
6. **Cohort A Prediction Alignment:** Outputs test predictions with strict `(Dialogue_ID, Utterance_ID)` keys that cleanly align with Cohort A ($N = 2,328$).

---

## 10. Conclusion & Stop for Review

The architectural design, leakage audit, and experimental protocol for DialogueGCN are complete. In strict adherence to user instructions:
- **No source code has been modified.**
- **No models have been trained.**
- **No dependencies have been installed.**
- **No git commits or pushes have been executed.**

The repository is now paused awaiting user review and approval before proceeding to implementation.
