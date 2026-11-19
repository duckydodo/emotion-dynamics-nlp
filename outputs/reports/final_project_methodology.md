# Final Project Methodology: Context-Aware Emotion Recognition and Dynamics in Conversation

**Repository**: `emotion-dynamics-nlp`  
**Deliverable**: `outputs/reports/final_project_methodology.md`  
**Target Audience**: Academic Reviewers, Research Advisors, and NLP Practitioners  
**Status**: Completed & Grounded in Verified Artifacts  

---

## 1. Research Problem and Motivation

### 1.1 The Challenge of Emotion Recognition in Conversation (ERC)
Emotion Recognition in Conversation (ERC) is a fundamental task at the intersection of natural language processing, affective computing, and cognitive science. Unlike standard sentence-level sentiment analysis, which classifies isolated text fragments without situational grounding, conversational emotion classification is inherently dynamic, relational, and context-dependent.

An utterance such as *"Fine, do whatever you want"* can express neutral compliance, passive anger, or deep sadness depending entirely on:
1. **Conversational History**: What preceding events or statements triggered the utterance?
2. **Speaker State & Inertia**: Is the speaker maintaining an ongoing affective state (emotional inertia) or reacting to partner input?
3. **Inter-Speaker Influence**: How do the participants influence each other's emotional trajectories across discourse turns?

Traditional natural language processing models treat dialogue turns as independent text samples (bag-of-words or vanilla sentence transformers). This introduces severe contextual blindness, failing to recognize sarcasm, emotional escalation, or subtle affective shifts.

### 1.2 Mathematical Formulation of Emotion Dynamics
We model a dialogue $\mathcal{D}$ as an ordered sequence of $N$ conversational turns:
$$\mathcal{D} = \{(u_1, s_1), (u_2, s_2), \dots, (u_N, s_N)\}$$
where $u_t$ represents the textual transcript of turn $t$, and $s_t \in \mathcal{S} = \{p_1, p_2, \dots, p_M\}$ identifies the speaker from a set of $M$ participants.

The goal is to assign to each turn $u_t$ an expressed categorical emotion label $y_t \in \mathcal{Y}$, where $\mathcal{Y} = \{\text{anger}, \text{disgust}, \text{fear}, \text{joy}, \text{neutral}, \text{sadness}, \text{surprise}\}$, while tracking the affective transition dynamics across time:
- **Total Transitions**: $T = N - 1$
- **Emotion Shift**: $S = \sum_{t=2}^{N} \mathbb{I}(y_t \ne y_{t-1})$
- **Emotion Persistence**: $P = \sum_{t=2}^{N} \mathbb{I}(y_t = y_{t-1}) = T - S$
- **Shift Rate**: $R_{\text{shift}} = \frac{S}{T}$ (for $T > 0$)
- **Intra-Speaker Inertia**: $R_{\text{inertia}} = \frac{T_{\text{same}} - S_{\text{same}}}{T_{\text{same}}}$
- **Inter-Speaker Influence**: $R_{\text{cross-shift}} = \frac{S_{\text{cross}}}{T_{\text{cross}}}$

---

## 2. Dataset and Domain Characteristics

### 2.1 The MELD Benchmark
Experiments are conducted on the **Multimodal EmotionLines Dataset (MELD)** (Poria et al., 2019), an established conversational benchmark derived from multi-party television dialogue (*Friends*). 

| Split | Dialogues | Total Utterances | Mean Turns / Dialogue |
| :--- | :---: | :---: | :---: |
| **Train** | 1,039 | 9,989 | 9.61 |
| **Development** | 114 | 1,109 | 9.73 |
| **Test** | 280 | 2,610 | 9.32 |
| **Total** | 1,433 | 13,708 | 9.57 |

Each turn is annotated with one of seven Ekman-derived emotion categories:
$$\mathcal{Y} = \{\text{neutral}, \text{joy}, \text{surprise}, \text{anger}, \text{sadness}, \text{disgust}, \text{fear}\}$$

### 2.2 Class Imbalance and Sampling Geometry
The label distribution in MELD exhibits extreme class imbalance:
- **Neutral**: ~47.1% (majority class)
- **Joy**: ~13.5%
- **Surprise**: ~9.5%
- **Anger**: ~11.8%
- **Sadness**: ~7.2%
- **Disgust**: ~2.7% (minority class)
- **Fear**: ~1.9% (extreme minority class)

Due to this disparity, raw accuracy is an uninformative metric; a naive majority-class classifier predicting `neutral` achieves nearly 50% accuracy while failing completely on active emotions. Consequently, **Macro F1** (the unweighted mean of per-class F1 scores) serves as the primary optimization and model selection criterion across all experiments:
$$\text{Macro F1} = \frac{1}{|\mathcal{Y}|} \sum_{c \in \mathcal{Y}} \text{F1}_c$$

### 2.3 MELD Domain Limitations
1. **Scripted Television Domain**: Dialogues originate from television comedy series. Utterances reflect comedic banter, theatrical deliveries, and exaggerated emotional reactions that differ from real-world customer service, mental health counseling, or workplace communication.
2. **Text-Only Modality Subset**: Although MELD provides raw audio and video streams, this research focuses on the NLP challenge of extracting emotion dynamics purely from text transcripts. Nuances carried exclusively by vocal prosody or facial grimaces are inherently inaccessible to text models.

---

## 3. Data Preprocessing and Canonical Schema

### 3.1 Unified Canonical Conversation Representation
To support arbitrary external conversation exports (plain text, WhatsApp, Instagram, CSV) while guaranteeing dialogue isolation and schema integrity, we implemented a canonical conversation layer ([`src/data/schema.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/data/schema.py)):
- **`CanonicalMessage`**: Encapsulates unique `message_id`, `source_index`, `speaker`, `text`, `timestamp` (ISO-8601), `message_type`, and `is_evaluable` boolean flag.
- **`CanonicalConversation`**: Encapsulates `conversation_id`, `source_format`, `participants` array, ordered `messages` list, and audit `warnings`.

### 3.2 Parsing Rules and Edge Sanitization
1. **Dialogue Boundary Isolation**: Conversations are processed in strict isolation. No hidden states, graph edges, or attention weights cross conversation boundaries.
2. **Whitespace & Non-Evaluable Filtering**: Empty lines and non-evaluable system tokens (e.g., WhatsApp `<Media omitted>`, attachment notices) are parsed with `is_evaluable = False` and excluded from model classification while preserving sequence indices.
3. **Timestamp Normalization**: Diverse datetime formats (WhatsApp 12h/24h, Instagram epoch milliseconds, CSV ISO strings) are normalized to timezone-aware UTC objects.

---

## 4. Utterance Feature Extraction: DistilBERT

### 4.1 Transformer Feature Backbone
To decouple sentence-level semantic encoding from sequence-level recurrence and graph propagation, we employ **DistilBERT** (`distilbert-base-uncased`, 66.96M parameters) (Sanh et al., 2019). DistilBERT offers a 40% reduction in size and 60% faster inference compared to BERT-base while retaining 97% of its language understanding capabilities.

### 4.2 Feature Representation Strategy
1. **Supervised Baseline Fine-Tuning**: A sequence classification head is trained on MELD training utterances to classify emotions independently, establishing the standalone sentence-level baseline.
2. **Frozen Dense Utterance Embeddings**: For multi-turn recurrent and graph models, the fine-tuned DistilBERT backbone is frozen. For each utterance $u_t$, we extract the 768-dimensional hidden state corresponding to the special `[CLS]` token:
$$x_t = \text{DistilBERT}(u_t)_{[\text{CLS}]} \in \mathbb{R}^{768}$$
3. **Memory Optimization**: Rather than re-instantiating transformer weights in each model, an in-memory singleton feature cache ([`SharedDistilBertEncoder`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/models/inference.py#L29-L73)) is shared across inference pipelines, saving $> 500\text{ MB}$ of memory.

---

## 5. Model Architecture 1: DialogueRNN

### 5.1 Formulation and State Tracking
**DialogueRNN** (Majumder et al., 2019) is a causal, recurrent neural network specifically designed to track discourse flow and emotional inertia in conversation:

```
[Utterance x_t] ────┐
                    ▼
           ┌─────────────────┐
  q_{t-1} ─┤ Global GRU (q_t)├─► Context Attention e_t
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────┐
  p_{s,t-1}┤Party GRU (p_{s,t})├─► Speaker State
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────┐
  e_{s,t-1}┤Emotion GRU(e_{s,t})├─► Classification Logits
           └─────────────────┘
```

1. **Global State GRU ($q_t$)**: Encodes the conversational discourse flow given the incoming utterance $x_t$ and party state $p_{s_t, t-1}$:
$$q_t = \text{GRU}_G(q_{t-1}, [x_t \oplus p_{s_t, t-1}])$$
2. **Context Attention Mechanism ($c_t$)**: Attends over past global states $q_1, \dots, q_t$ using attention query $q_t$:
$$\alpha_i = \text{softmax}(q_t^T W_{\alpha} q_i), \quad c_t = \sum_{i=1}^{t} \alpha_i q_i$$
3. **Party State GRU ($p_{s, t}$)**: Tracks individual speaker state, modeling how speaker $s_t$ updates their mental state given context:
$$p_{s_t, t} = \text{GRU}_P(p_{s_t, t-1}, [x_t \oplus c_t])$$
For speakers $s \ne s_t$, the party state remains unchanged: $p_{s, t} = p_{s, t-1}$.
4. **Emotion State GRU ($e_{s, t}$)**: Captures emotional inertia:
$$e_{s_t, t} = \text{GRU}_E(e_{s_t, t-1}, p_{s_t, t})$$
5. **Emotion Classifier**: Logits are computed via a two-layer feedforward network over $e_{s_t, t}$:
$$\hat{y}_t = \text{softmax}(W_2 \text{ReLU}(W_1 e_{s_t, t} + b_1) + b_2)$$

### 5.2 Architectural Specifications
- Hidden Dimensions: $D_g = 150$, $D_p = 150$, $D_e = 100$
- Recurrent Layers: Single-layer unidirectional GRUs (strictly causal)
- Total Trainable Parameters: **670,855** (excluding frozen DistilBERT)

---

## 6. Model Architecture 2: DialogueGCN

### 6.1 Formulation and Conversational Graph Topology
**DialogueGCN** (Ghosal et al., 2019; implemented in [`src/models/dialoguegcn.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/models/dialoguegcn.py)) re-conceptualizes conversation as a directed, relational graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{R})$.

```
           [u_1 (Alice)] <=========> [u_3 (Alice)]  (Intra-Speaker Relation r_intra)
                | \                     / |
                |   \                 /   |
                |     \             /     |
(r_inter)       |       \         /       |         (Inter-Speaker Relation r_inter)
                |         \     /         |
                ▼           ▼ ▼           ▼
           [u_2 (Bob)]   <=========> [u_4 (Bob)]    (Intra-Speaker Relation r_intra)
```

1. **Nodes ($\mathcal{V}$)**: Every utterance turn $t \in \{1, \dots, N\}$ forms a graph node initialized with DistilBERT features $x_t \in \mathbb{R}^{768}$.
2. **Context Window**: Each node $i$ connects to surrounding nodes $j$ within a bidirectional window $\max(1, i - p) \le j \le \min(N, i + f)$, where past window $p = 10$ and future window $f = 10$.
3. **Edge Relations ($\mathcal{R}$)**: Exactly two relation types are defined:
   - **$r_{\text{intra}}$**: Intra-speaker edge connecting turns uttered by the same participant ($s_i = s_j$), modeling emotional inertia.
   - **$r_{\text{inter}}$**: Inter-speaker edge connecting turns uttered by different participants ($s_i \ne s_j$), modeling partner influence and conversational stimulus.

### 6.2 Relational Graph Convolution and Attention
1. **Sequential Context Encoding**: Node representations are contextualized via a bidirectional GRU:
$$H^{(0)} = \text{BiGRU}(X) \in \mathbb{R}^{N \times 2 D_h}$$
2. **Similarity Attention Weights ($\alpha_{ij}$)**: Relational edge weights are dynamically computed via similarity attention:
$$\alpha_{ij} = \text{softmax}_j \left( \frac{h_i^T W_{\text{att}} h_j}{\sqrt{D_h}} \right)$$
3. **Relational Message Passing**: For layer $l \in \{1, \dots, L\}$ (where $L = 2$):
$$h_i^{(l)} = \sigma \left( W_{\text{root}}^{(l)} h_i^{(l-1)} + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_i^r} \alpha_{ij} W_r^{(l)} h_j^{(l-1)} \right)$$
4. **Classification Head**: Final representations $h_i^{(L)}$ pass through a two-layer MLP with residual skip connections to produce 7-class logits.

### 6.3 Architectural Specifications
- Graph Embedding Dimension: $D_h = 100$
- GCN Layers: $L = 2$
- Relation Types: 2 ($r_{\text{intra}}, r_{\text{inter}}$)
- Receptive Field: Past 10 turns, Future 10 turns (offline bidirectional)
- Total Trainable Parameters: **452,487** (32.5% fewer parameters than DialogueRNN)

---

## 7. Experimental and Training Protocol

### 7.1 Optimization Configuration
All models were trained under identical conditions to eliminate confounding variables:
- **Optimizer**: AdamW ($\beta_1 = 0.9, \beta_2 = 0.999$, weight decay $= 10^{-4}$)
- **Learning Rate**: $10^{-4}$ with linear warmup and cosine decay
- **Batch Size**: 32 dialogues (variable turn sequence lengths handled via masking)
- **Loss Function**: Cross-Entropy Loss with label smoothing ($0.05$)
- **Random Seed**: Fixed to `42` across PyTorch, NumPy, and Python standard library
- **Gradient Clipping**: Norm capped at $1.0$

### 7.2 Checkpoint Selection Protocol
To prevent data snooping and test set overfitting:
1. Checkpoints were saved after every epoch during training.
2. **Model Selection Criterion**: The checkpoint achieving the highest **Macro F1 on the Development split** was selected.
   - DialogueRNN selected at **Epoch 10** (Dev Macro F1 = 0.4342)
   - DialogueGCN selected at **Epoch 15** (Dev Macro F1 = 0.4528)
3. The selected best checkpoints were evaluated **exactly once** on the test set.

---

## 8. Evaluation Cohort Design: Canonical Cohort A

### 8.1 The Sequence Length Confound
In dialogue datasets, test dialogues vary significantly in length (from 1 turn up to $> 30$ turns). Comparing recurrent models (which require sequential history) against windowed baselines (which require $t-1$) on disjoint subsets introduces severe sampling bias.

### 8.2 Strict Matched Subset Definition
To guarantee strict statistical comparability across all models, we defined **Cohort A: Strict Contiguous Matched Subset**:
- **Sample Size**: Exactly **$N = 2,328$ turns** (280 dialogues).
- **Inclusion Criteria**: All turns having contiguous preceding dialogue context ($t \ge 1$), ensuring that DistilBERT Baseline, DistilBERT Context-1, DialogueRNN, and DialogueGCN predict on the exact same utterances.
- Every metric, table, and statistical significance test in this project is calculated strictly on this canonical cohort.

---

## 9. Statistical Significance Testing Framework

To separate genuine architectural improvements from random sampling noise, three rigorous statistical tests were implemented:

1. **Non-Parametric Permutation Test (Monte Carlo, $B = 10,000$ iterations)**:
   Tests the null hypothesis $H_0$ that Model A and Model B have identical Macro F1 distributions by repeatedly swapping predictions turn-by-turn and computing the empirical null distribution:
   $$p_{\text{perm}} = \frac{1}{B} \sum_{b=1}^{B} \mathbb{I}(|\Delta \text{F1}^{(b)}| \ge |\Delta \text{F1}_{\text{observed}}|)$$
2. **Paired Bootstrap Confidence Intervals ($B = 1,000$ resamples)**:
   Resamples dialogues with replacement to calculate the 95% empirical bootstrap confidence interval for $\Delta \text{Macro F1}$.
3. **McNemar's Test**:
   Evaluates discordance in paired classification correctness on identical utterances ($b$ = Model A correct & Model B incorrect; $c$ = Model A incorrect & Model B correct):
   $$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}, \quad p = P(\chi_1^2 \ge \chi^2)$$
