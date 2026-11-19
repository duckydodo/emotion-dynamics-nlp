# Independent DialogueRNN Results Validation & Statistical Audit Report

**Repository:** `emotion-dynamics-nlp`  
**Working Directory:** `~/Projects/emotion-dynamics-nlp`  
**Audit Date:** November 10, 2026  
**Auditor:** Antigravity Autonomous Code & Statistical Auditor  
**Task:** Task 11 — Independent DialogueRNN Results Validation  
**Deliverable:** [`outputs/reports/dialoguernn_independent_validation.md`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/reports/dialoguernn_independent_validation.md)  
**Associated Audit Artifacts:**
- Audit Script: [`scripts/audit_dialoguernn_validation.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/scripts/audit_dialoguernn_validation.py)
- Prediction Integrity Verification: [`outputs/tables/validation_prediction_integrity.json`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/tables/validation_prediction_integrity.json)
- Recalculated Comparison Benchmarks: [`outputs/tables/validation_recalculated_metrics.csv`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/tables/validation_recalculated_metrics.csv)
- Recalculated Per-Class Metrics: [`outputs/tables/validation_per_class_recalculated.csv`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/tables/validation_per_class_recalculated.csv)
- Dialogue-Level Cluster Bootstrap Results: [`outputs/tables/validation_dialogue_cluster_bootstrap.csv`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/tables/validation_dialogue_cluster_bootstrap.csv)

---

## 1. Executive Summary

This independent audit evaluated the validity, reproducibility, and statistical robustness of the DialogueRNN experiments completed on the MELD conversational emotion dataset. The audit re-examined raw ground-truth data, inspected all underlying prediction CSV files, recalculated every summary and per-class metric independently, evaluated model architectures and training configurations, and executed an independent **dialogue-level paired cluster bootstrap** (10,000 resamples) with **Holm-Bonferroni family-wise error rate adjustment**.

### Audit Verdict: VALIDATED WITH QUALIFICATIONS

1. **Prediction Integrity (100% Validated):**
   All 7 model prediction files are authentic, have zero duplicate keys, zero missing turns, exact row counts, and 100% agreement with the official `data/raw/meld/test_sent_emo.csv` ground truth. Dialogue utterance gaps (Dialogue 93: 4 $\to$ 8; Dialogue 108: 0 $\to$ 3) are strictly excluded from Cohort A ($N=2,328$). All model comparisons evaluate on the identical ordered sequence of examples.
2. **Metric Recalculation (100% Validated):**
   Independent recalculation of Accuracy, Macro F1, Weighted F1, and per-class Precision, Recall, and F1 across all 22 model/cohort combinations produced **exact numerical agreement** (0 discrepancies across all metrics at $10^{-6}$ precision).
3. **Core Hypotheses (Validated):**
   - DialogueRNN (Unweighted) achieves **Macro F1 = 0.4342**, significantly outperforming Context-1 ($\Delta = +0.0430$, Holm-adjusted $p = 0.0072$) and Model B Autoregressive ($\Delta = +0.0494$, Holm-adjusted $p = 0.0072$).
   - The "Accuracy Illusion" and minority collapse of concatenative and autoregressive models are mathematically verified.
4. **Primary Statistical Qualification (Crucial Correction):**
   The previous report claimed DialogueRNN was statistically significantly superior to **Context-1 + Speaker** ($p = 0.0396$). That finding was an artifact of **utterance-level resampling**, which violated the independence assumption by ignoring within-dialogue clustering. When properly evaluated using a **dialogue-level paired cluster bootstrap**, the 95% confidence interval widens to **$[-0.0013, +0.0535]$** ($p = 0.0614$ unadjusted), and under Holm-Bonferroni correction, **$p_{\text{adj}} = 0.1842$**. Therefore, DialogueRNN is **not statistically significantly superior** to Context-1 + Speaker in Macro F1.
5. **Methodological Clarification on McNemar Tests:**
   McNemar's test assesses marginal homogeneity of 0/1 correctness (overall accuracy), **not Macro F1**. Non-significant McNemar tests do not contradict significant Macro F1 improvements, but McNemar tests must never be cited as evidence for or against Macro F1 differences.

---

## 2. Prediction and Cohort Integrity Results

Every prediction file was loaded, parsed, and verified row-by-row against the canonical ground truth file `data/raw/meld/test_sent_emo.csv`.

### 2.1 Test Split Dialogue Structure and Gaps
- Total test dialogues: **280**.
- Total test utterances: **2,610**.
- Opening turns ($t = 0$, Cohort C): **280**.
- Subsequent sequential turns ($t \ge 1$, Cohort B): **2,330**.
- Dialogues containing Utterance_ID gaps:
  - **Dialogue 93:** `Utterance_ID` jumps from `4` to `8` (turns 5, 6, 7 absent in source).
  - **Dialogue 108:** `Utterance_ID` jumps from `0` to `3` (turns 1, 2 absent in source).
- Strict contiguous subsequent turns ($t \ge 1$ and `Utterance_ID == prev + 1`, Cohort A): **$2,330 - 2 = 2,328$**.
- Verification: Neither `(93, 8)` nor `(108, 3)` appears in Cohort A. Both are retained in Cohort B and Cohort D.

### 2.2 Model Prediction File Integrity Audit

| Model / Prediction File | File Path | Actual Rows | Unique Keys | Duplicate Keys | Ground Truth Mismatches | Cohort A Exact Sequence Match | Integrity Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DistilBERT Utterance Baseline** | `outputs/experiments/transformer_baseline/predictions.csv` | 2,610 | 2,610 | 0 | 0 / 2,610 | True (100% aligned) | **PASS** |
| **DistilBERT Context-1** | `outputs/experiments/transformer_context_1/predictions.csv` | 2,328 | 2,328 | 0 | 0 / 2,328 | True (100% aligned) | **PASS** |
| **DistilBERT Context-1 + Speaker** | `outputs/experiments/transformer_context_1_speaker/predictions.csv` | 2,328 | 2,328 | 0 | 0 / 2,328 | True (100% aligned) | **PASS** |
| **Model A: Emotion-History (Auto)** | `outputs/experiments/transformer_emotion_history/predictions_autoregressive.csv` | 2,610 | 2,610 | 0 | 0 / 2,610 | True (100% aligned) | **PASS** |
| **Model B: Combined-History (Auto)**| `outputs/experiments/transformer_combined_history/predictions_autoregressive.csv` | 2,610 | 2,610 | 0 | 0 / 2,610 | True (100% aligned) | **PASS** |
| **DialogueRNN (Unweighted)** | `outputs/experiments/dialoguernn_unweighted/test_predictions.csv` | 2,610 | 2,610 | 0 | 0 / 2,610 | True (100% aligned) | **PASS** |
| **DialogueRNN (Class-Weighted)** | `outputs/experiments/dialoguernn_weighted/test_predictions.csv` | 2,610 | 2,610 | 0 | 0 / 2,610 | True (100% aligned) | **PASS** |

*Artifact Generated: [`outputs/tables/validation_prediction_integrity.json`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/tables/validation_prediction_integrity.json)*

**Integrity Findings:**
- No predictions are missing.
- No keys are duplicated.
- When filtered to Cohort A, all 7 models contain exactly the identical 2,328 examples in the identical chronological order.
- Cohort alignment is verified without leakage or indexing misalignment.

---

## 3. Independently Recalculated Benchmark Table

All summary metrics and per-class metrics were recomputed from scratch using the verified raw predictions and the canonical label alphabetical order:
`["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]`.

### 3.1 Canonical Benchmark Table (Cohort A: $N = 2,328$ Strict Contiguous Matched Subset)

| Model Configuration | Architecture | Accuracy | Macro F1 | Weighted F1 | Neutral Pred % | Neutral Rec | Fear Rec | Disgust Rec | Anger Rec | Joy Rec | Sadness Rec | Surprise Rec | Audit Match |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DistilBERT Baseline** | Utterance Only | 59.45% | 0.4314 | 0.5969 | 45.19% | 73.57% | 25.00% | 20.63% | 36.81% | 58.66% | 33.16% | 60.87% | **Exact (0 diff)** |
| **DistilBERT Context-1** | Text Context ($t-1$) | 60.74% | 0.3912 | 0.5812 | 57.26% | 83.33% | 6.82% | 14.29% | 30.94% | 54.47% | 16.04% | 60.08% | **Exact (0 diff)** |
| **DistilBERT Context-1 + Spk** | Text Context + Speaker | 61.94% | 0.4078 | 0.5981 | 53.78% | 82.35% | 11.36% | 11.11% | 31.60% | 58.38% | 20.32% | 66.01% | **Exact (0 diff)** |
| **Model A (Autoregressive)** | History Token | 61.47% | 0.3930 | 0.5984 | 51.03% | 78.94% | 2.27% | 4.76% | 39.09% | 63.41% | 28.34% | 57.71% | **Exact (0 diff)** |
| **Model B (Autoregressive)** | Text + History Token | **62.59%** | 0.3848 | 0.6024 | 54.04% | 82.35% | **0.00%** | **3.17%** | 43.00% | 53.63% | 22.46% | 67.19% | **Exact (0 diff)** |
| **DialogueRNN (Unweighted)** | Attentive Recurrent | 60.14% | **0.4342** | **0.6025** | 44.63% | 72.85% | 15.91% | 15.87% | **46.91%** | 57.82% | **35.83%** | 60.08% | **Exact (0 diff)** |
| **DialogueRNN (Weighted)** | Recurrent + Inv. Weight | 59.88% | 0.4284 | 0.5944 | 49.74% | 77.96% | **25.00%** | **25.40%** | 35.50% | 50.84% | 27.81% | 60.87% | **Exact (0 diff)** |

### 3.2 Per-Class F1 and Recall Recalculation Check
Every per-class metric in [`outputs/tables/dialoguernn_per_class.csv`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/tables/dialoguernn_per_class.csv) was audited against independent recalculations ([`outputs/tables/validation_per_class_recalculated.csv`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/tables/validation_per_class_recalculated.csv)):
- **Total checked cells:** 154 rows $\times$ 4 metric columns (Precision, Recall, F1, Support) = 616 data points.
- **Discrepancies:** Exactly **0**.
- **Minority class collapse verification:**
  - `fear` recall in Model B is confirmed at **0.0000** (0 / 44 detected; F1 = 0.0000).
  - `disgust` recall in Model B is confirmed at **0.0317** (2 / 63 detected; F1 = 0.0606).
  - DialogueRNN Unweighted restores `fear` to **0.1591** (F1 = 0.1667) and `disgust` to **0.1587** (F1 = 0.2128).
  - DialogueRNN Class-Weighted further elevates `fear` to **0.2500** (F1 = 0.1964) and `disgust` to **0.2540** (F1 = 0.2712).

---

## 4. Statistical Validation Methodology

### 4.1 Weakness of Prior Utterance-Level Bootstrap
In prior tasks, the paired bootstrap was conducted by sampling **individual utterances** with replacement:
$$\text{Sample } \{u_1^*, \dots, u_N^*\} \sim \text{Uniform}(\{u_1, \dots, u_N\})$$

While this preserves model pairing on individual turns, it assumes that all 2,328 utterances are mutually independent. In conversational dialogue:
- Utterances within dialogue $d$ share interlocutors, conversational context, and emotion dynamics.
- In Cohort A, the 2,328 utterances originate from **268 unique dialogues** (ranging from 1 to 32 turns per dialogue, mean = 8.69 turns).
- Independent utterance resampling overstates the effective degrees of freedom, artificially deflating standard errors and underestimating p-values.

### 4.2 Independent Dialogue-Level Paired Cluster Bootstrap
To establish statistically reliable confidence intervals and hypothesis tests:
1. **Cluster Unit:** Resample complete dialogues $d_1^*, \dots, d_M^*$ with replacement from the set of 268 unique dialogues containing Cohort A turns ($M = 268$).
2. **Cluster Integrity:** For each selected dialogue $d_k^*$, all eligible Cohort A utterances belonging to $d_k^*$ are included in the bootstrap replicate.
3. **Paired Integrity:** Both models in each comparison are evaluated on the **exact same dialogue clusters and utterance sequence** for that bootstrap replication.
4. **Iterations & Seed:** Exactly $B = 10,000$ cluster resamples using fixed random seed `seed=42`.
5. **Two-Sided Empirical $p$-value:**
   $$p = 2 \times \min\left(\frac{1}{B}\sum_{b=1}^B \mathbb{I}(\Delta_b^* \le 0), \, \frac{1}{B}\sum_{b=1}^B \mathbb{I}(\Delta_b^* \ge 0)\right), \quad \text{bounded in } \left[\frac{1}{B}, 1.0\right]$$
6. **Family-Wise Error Rate Correction:** Apply the **Holm-Bonferroni method** across the pre-specified family of 6 pairwise comparisons against DialogueRNN (Unweighted).

---

## 5. Dialogue-Level Cluster Bootstrap Results

The table below contrasts the prior utterance-level bootstrap results against the independent dialogue-level cluster bootstrap (10,000 iterations each):

| Comparison (vs. DialogueRNN Unweighted) | Observed $\Delta$ Macro F1 | Utterance Bootstrap 95% CI | Utterance $p$-value (Unadj.) | Dialogue Cluster Bootstrap 95% CI | Cluster Std. Error | Cluster $p$-value (Unadj.) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **vs. DistilBERT Baseline** | $+0.0028$ | $[-0.0143, +0.0196]$ | 0.7382 | $[-0.0152, +0.0198]$ | 0.0088 | 0.7496 |
| **vs. DistilBERT Context-1** | **$+0.0430$** | $[+0.0182, +0.0690]$ | 0.0004 | **$[+0.0161, +0.0716]$** | 0.0140 | **0.0012** |
| **vs. DistilBERT Context-1 + Speaker** | $+0.0264$ | $[+0.0014, +0.0516]$ | 0.0396 | **$[-0.0013, +0.0535]$** | 0.0140 | **0.0614** |
| **vs. Model A: Emotion-History (Auto)** | **$+0.0412$** | $[+0.0149, +0.0686]$ | 0.0024 | **$[+0.0114, +0.0717]$** | 0.0153 | **0.0076** |
| **vs. Model B: Combined-History (Auto)**| **$+0.0494$** | $[+0.0215, +0.0773]$ | 0.0004 | **$[+0.0194, +0.0781]$** | 0.0151 | **0.0014** |
| **vs. DialogueRNN (Class-Weighted)** | $+0.0058$ | $[-0.0170, +0.0284]$ | 0.6108 | $[-0.0150, +0.0266]$ | 0.0107 | 0.5888 |

### Impact of Dialogue Clustering on Uncertainty
- For **Context-1**, the cluster standard error increases ($SE = 0.0140$ vs $0.0129$). The confidence interval widens from $[+0.0182, +0.0690]$ to $[+0.0161, +0.0716]$, but the gain remains **robustly significant** ($p = 0.0012$).
- For **Model B Autoregressive**, the confidence interval widens from $[+0.0215, +0.0773]$ to $[+0.0194, +0.0781]$, remaining **highly significant** ($p = 0.0014$).
- For **Context-1 + Speaker**, the standard error increases from $0.0129$ to $0.0140$. This shifts the lower bound from $+0.0014$ to **$-0.0013$**, crossing zero! Consequently, the unadjusted p-value increases from $p = 0.0396$ to **$p = 0.0614$**.

---

## 6. Multiple-Comparison-Adjusted Results

To control the family-wise error rate across the $K = 6$ planned comparisons against DialogueRNN (Unweighted), the Holm-Bonferroni step-down correction was applied:

| Rank ($i$) | Comparison (vs. DialogueRNN Unweighted) | $\Delta$ Macro F1 | Unadjusted Cluster $p$ | Holm Multiplier ($K - i + 1$) | Holm-Adjusted $p$ | Significant at $\alpha = 0.05$? |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | **vs. DistilBERT Context-1** | $+0.0430$ | 0.0012 | 6 | **0.0072** | **YES (Statistically Significant)** |
| 2 | **vs. Model B: Combined-History (Auto)** | $+0.0494$ | 0.0014 | 5 | **0.0072** | **YES (Statistically Significant)** |
| 3 | **vs. Model A: Emotion-History (Auto)** | $+0.0412$ | 0.0076 | 4 | **0.0304** | **YES (Statistically Significant)** |
| 4 | **vs. DistilBERT Context-1 + Speaker** | $+0.0264$ | 0.0614 | 3 | **0.1842** | **NO (Not Significant)** |
| 5 | **vs. DialogueRNN (Class-Weighted)** | $+0.0058$ | 0.5888 | 2 | **1.0000** | **NO (Equivalent)** |
| 6 | **vs. DistilBERT Utterance Baseline** | $+0.0028$ | 0.7496 | 1 | **1.0000** | **NO (Equivalent)** |

*Artifact Generated: [`outputs/tables/validation_dialogue_cluster_bootstrap.csv`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/tables/validation_dialogue_cluster_bootstrap.csv)*

### Clarification on McNemar Tests vs. Macro F1
The audit examined the McNemar tests recorded in earlier reports:
- Context-1: $\chi^2 = 0.35, p = 0.5529$
- Context-1 + Speaker: $\chi^2 = 3.70, p = 0.0543$
- Model B: $\chi^2 = 7.31, p = 0.0069$ (Model B has higher accuracy)
- Baseline: $\chi^2 = 1.37, p = 0.2415$

**Critical Finding:** McNemar's test is a test of **overall classification accuracy** (marginal homogeneity of binary 0/1 correctness across instances). It does not test Macro F1. In imbalanced datasets, a model that predicts `neutral` on 54% of turns can achieve higher accuracy ($p = 0.0069$ in McNemar) while completely failing to identify minority classes (`fear` recall 0.0%). The paired cluster bootstrap is the appropriate statistical test for Macro F1.

---

## 7. Reproducibility & Implementation Audit

### 7.1 Checkpoint Loading & Architecture
- Code inspected: [`src/models/dialoguernn.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/models/dialoguernn.py).
- Trainable parameter count: **670,855** parameters.
- Architecture dimension audit:
  - Input Projection: $768 \to 256$ with LayerNorm and Dropout ($p = 0.2$).
  - Global Context GRU: Input dimension $256 + 128 = 384$, Hidden dimension $D_g = 128$.
  - Context Attention: Linear projection $128 \to 128$ over past global states.
  - Party State GRU: Input dimension $256 + 128 = 384$, Hidden dimension $D_p = 128$.
  - Emotion Attention: Linear projection $128 \to 64$ over active speaker's own prior emotion states.
  - Emotion GRU: Input dimension $128 + 64 = 192$, Hidden dimension $D_e = 64$.
  - Classification Head: Linear $64 \to 64 \to 7$ emotion logits.
- **Discrepancy Noted in Prior Report:** Section 5 of `research_progress_report.md` mentioned $(D_g = 150, D_p = 100, D_e = 50)$ from the original Majumder et al. paper, whereas the actual implemented code used $(D_g = 128, D_p = 128, D_e = 64)$ to optimize power-of-two CPU vectorization. The actual parameter count (670,855) was correctly reported.

### 7.2 Training-Only Class Weights Verification
- Code inspected: `scripts/train_dialoguernn.py` (lines 262–274).
- The class weights were computed strictly from `train_dataset.emotions` (1,039 training dialogues, 9,989 utterances):
  - `anger: 1.2856, disgust: 5.2609, fear: 5.3198, joy: 0.8180, neutral: 0.3027, sadness: 2.0874, surprise: 1.1832`.
- Zero validation or test set label leakage occurred during weight calculation.

### 7.3 Model Selection and Test-Set Isolation
- Code inspected: `scripts/train_dialoguernn.py` (lines 391–419).
- Checkpoints were saved strictly upon obtaining a new best **Development Set Macro F1**:
  - `dialoguernn_unweighted`: Best epoch 7 (Dev Macro F1 = 0.4045, Dev Loss = 1.3444).
  - `dialoguernn_weighted`: Best epoch 35 (Dev Macro F1 = 0.4011, Dev Loss = 1.6366).
- Test set data was loaded and evaluated **only once**, after all training and early stopping had terminated. Zero test-set optimization or hyperparameter leakage was detected.

---

## 8. Confirmed Issues and Unresolved Concerns

1. **Overstated Significance over Context-1 + Speaker:**
   *Issue:* Earlier task summaries claimed DialogueRNN was statistically significantly superior to `Context-1 + Speaker` ($p = 0.0396$).  
   *Audit Finding:* That claim fails when within-dialogue clustering is accounted for ($p = 0.0614$ unadjusted, $p_{\text{adj}} = 0.1842$ under Holm correction). While DialogueRNN beats Context-1 + Speaker by $+0.0264$ in Macro F1 and provides substantially better neutral balance (44.63% vs 53.78% neutral prediction), the Macro F1 difference is not statistically significant at $\alpha = 0.05$.
2. **Text-Narrative Dimension Typo:**
   *Issue:* In `research_progress_report.md`, DialogueRNN dimensions were cited as $(150, 100, 50)$ rather than the implemented $(128, 128, 64)$.  
   *Audit Finding:* Minor documentation typo; actual code and saved weights are consistent and parameter counts match.
3. **Sitcom Specificity Limitation (Unresolved):**
   *Issue:* All models are trained and tested on MELD (conversations from *Friends*). Whether DialogueRNN's speaker tracking generalizes to non-sitcom conversational domains (e.g., DailyDialog or clinical transcripts) remains unvalidated.

---

## 9. Final Decision & Status

### Decision: **RESULTS VALIDATED WITH QUALIFICATIONS**

- **Validated Elements:**
  - Prediction files, data splits, and canonical cohort definitions are 100% authentic and error-free.
  - All metrics (Accuracy, Macro F1, Weighted F1, Per-class scores) are independently verified to exact precision.
  - DialogueRNN significantly outperforms naive Context-1 ($p_{\text{adj}} = 0.0072$) and Model B Autoregressive ($p_{\text{adj}} = 0.0072$).
  - DialogueRNN successfully mitigates neutral gravitation and restores minority-class recall.
- **Required Qualifications:**
  - DialogueRNN does **not** achieve statistically significant Macro F1 superiority over **Context-1 + Speaker** under cluster bootstrap testing ($p_{\text{adj}} = 0.1842$).
  - McNemar's test must be clearly designated as an accuracy test on 0/1 correctness, not a test of Macro F1.

---

## 10. Recommended Next Experiment

**Experiment: DialogueGCN (Graph Convolutional Networks for Multi-Party Relational Dialogue)**

*Scientific Rationale:*
The audit proves that sequential speaker tracking (DialogueRNN) prevents minority collapse, but fails to achieve statistically significant separation from Context-1 + Speaker ($p = 0.1842$). This occurs because DialogueRNN processes turns as a linear recurrent chain ($t-1 \to t$). In multi-party conversations ($>2$ speakers), speaker $A$ at turn $t$ is frequently responding to speaker $C$ from turn $t-4$, while speaker $B$ spoke at turns $t-1$ through $t-3$. A sequential GRU forces the latent state through irrelevant interlocutors.

*Implementation Plan:*
Implement **DialogueGCN** (Ghosal et al., EMNLP 2019), representing the conversation as a directed relational graph where:
1. Nodes are utterance vectors $e_t \in \mathbb{R}^{768}$.
2. Edge weights model intra-speaker and inter-speaker relational dependencies directly across arbitrary conversational distances.
3. Graph convolution updates representations across conversation topology rather than strict sequential adjacency.
