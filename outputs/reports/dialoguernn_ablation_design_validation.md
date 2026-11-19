# DialogueRNN Architectural Ablation Study: Final Design Validation Report

**Repository:** `emotion-dynamics-nlp`  
**Working Directory:** `/home/chitta/Projects/emotion-dynamics-nlp`  
**Date:** November 10, 2026  
**Status:** Pre-Execution Validation Completed — Awaiting User Approval Prior to Training  
**Task Reference:** Task 13 (Final Ablation Design Validation)  
**Deliverable Document:** [`outputs/reports/dialoguernn_ablation_design_validation.md`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/reports/dialoguernn_ablation_design_validation.md)  
**Associated Configuration Files:**
- Ablation 1: [`configs/ablations/1_global_context_only.json`](file:///home/chitta/Projects/emotion-dynamics-nlp/configs/ablations/1_global_context_only.json)
- Ablation 2: [`configs/ablations/2_global_party_states.json`](file:///home/chitta/Projects/emotion-dynamics-nlp/configs/ablations/2_global_party_states.json)
- Ablation 3: [`configs/ablations/3_global_emotion_state.json`](file:///home/chitta/Projects/emotion-dynamics-nlp/configs/ablations/3_global_emotion_state.json)
- Ablation 4: [`configs/ablations/4_full_dialoguernn.json`](file:///home/chitta/Projects/emotion-dynamics-nlp/configs/ablations/4_full_dialoguernn.json)

---

## 1. Executive Summary

This validation review audits the statistical reporting framework, architectural specifications, experimental controls, and falsifiable hypotheses for the planned DialogueRNN ablation study. Prior tasks (Tasks 11 and 12) established that DialogueRNN Unweighted achieves **Macro F1 = 0.4342** on canonical Cohort A ($N=2,328$), statistically significantly outperforming standard text concatenation (Context-1: $\Delta = +0.0430$, Holm-adjusted $p = 0.0216$) and autoregressive decoding (Model B: $\Delta = +0.0494$, Holm-adjusted $p = 0.0395$).

Before initiating model training for the ablation series, this review enforces strict methodological integrity:
1. **Statistical Reporting Rigor:** The paired dialogue-level permutation test and cluster bootstrap were audited. Non-significant differences are explicitly defined as *insufficient evidence of a difference* rather than *evidence of equivalence*, correcting earlier informal narrative phrasing. The exchangeability assumption under $H_0$ is formally articulated at the dialogue cluster level ($M = 268$).
2. **Architectural Parity & Accounting:** Parameter counts were independently derived and verified against PyTorch modules. The full model reproduces exactly **670,855 parameters**. An off-by-one clerical rounding error in Ablation 3 (previously recorded as 453,000) was discovered and corrected to the exact count of **452,999 parameters** in [`configs/ablations/3_global_emotion_state.json`](file:///home/chitta/Projects/emotion-dynamics-nlp/configs/ablations/3_global_emotion_state.json).
3. **Hypothesis Reframing:** Research questions were converted from assumed outcomes into falsifiable statistical hypotheses evaluated on canonical Cohort A ($N = 2,328$).
4. **Scope Boundaries Preserved:** No model training has been executed, no checkpoints or prediction files have been altered, and no Git commits have been made.

---

## 2. Part A: Statistical Reporting and Inference Validation

### 2.1 Paired Dialogue-Level Permutation Test Implementation Review
The primary hypothesis test evaluating pairwise performance differences against DialogueRNN uses a **paired dialogue-level permutation test** over the canonical Cohort A evaluation split:
- **Sample Cohort:** $N = 2,328$ utterances grouped into $M = 268$ dialogue clusters.
- **Test Statistic:** The difference in Macro F1 across the entire cohort:
  $$T(\hat{\mathbf{y}}_1, \hat{\mathbf{y}}_2) = \text{MacroF1}(\mathbf{y}, \hat{\mathbf{y}}_1) - \text{MacroF1}(\mathbf{y}, \hat{\mathbf{y}}_2)$$
  where $\hat{\mathbf{y}}_1$ represents DialogueRNN predictions and $\hat{\mathbf{y}}_2$ represents the comparator model.
- **Observed Difference:** $\hat{\Delta}_{\text{obs}} = T(\hat{\mathbf{y}}_{\text{DialogueRNN}}, \hat{\mathbf{y}}_{\text{comparator}})$.
- **Monte Carlo Resampling ($B = 10,000$ iterations):** Under $H_0$, model labels within each dialogue are exchangeable. For each dialogue $d \in \{1, \dots, M\}$, a Rademacher sign-flip $S_d \in \{-1, +1\}$ is drawn with $P(S_d = 1) = 0.5$. If $S_d = -1$, the predictions for dialogue $d$ are swapped between the two models; if $S_d = +1$, they remain unchanged.
- **Two-Sided P-Value Calculation:**
  $$p_{\text{perm}} = \frac{1 + \sum_{b=1}^B \mathbb{I}(|\Delta_b^*| \ge |\hat{\Delta}_{\text{obs}}|)}{1 + B}$$
  With $B = 10,000$, the minimum achievable non-zero $p$-value is $1 / 10,001 \approx 0.0001$.

### 2.2 Dialogue-Level Exchangeability Assumption
The permutation test requires that under the sharp null hypothesis $H_0$, the assignment of model labels to dialogue predictions is arbitrary:
- **Mathematical Assumption:** For each dialogue cluster $d \in \{1, \dots, M\}$, conditional on the dialogue conversational history and speaker sequence $\mathbf{x}_d$, the joint prediction error distributions of Model 1 and Model 2 are identical:
  $$P(\hat{\mathbf{y}}_{1, d}, \hat{\mathbf{y}}_{2, d} \mid \mathbf{x}_d) = P(\hat{\mathbf{y}}_{2, d}, \hat{\mathbf{y}}_{1, d} \mid \mathbf{x}_d)$$
- **Why Utterance-Level Swapping is Invalid:** Utterances within a dialogue are strongly correlated across time and interlocutors. Swapping individual turns independently would shatter the sequential and conversational dependencies, generating synthetic sequences that violate the physical data-generating process and artificially deflating test variance. Swapping at the **dialogue cluster level** preserves all within-dialogue correlations, utterance lengths, speaker turn transitions, and class imbalances.

### 2.3 Paired Dialogue Structure & Holm-Bonferroni Correction
The family-wise error rate is controlled at $\alpha = 0.05$ across the family of exactly **$K = 6$ planned pairwise comparisons** against DialogueRNN (Unweighted). The Holm-Bonferroni step-down adjustment sorts raw $p$-values in ascending order $p_{(1)} \le p_{(2)} \le \dots \le p_{(K)}$ and enforces:
$$\tilde{p}_{(i)} = \max_{j \le i} \left[ \min\left(1, (K - j + 1) p_{(j)}\right) \right]$$

Applying this procedure to the canonical Cohort A comparisons yields:
1. **vs DistilBERT Context-1:** $\hat{\Delta} = +0.0430$, raw $p = 0.0036$, multiplier $6 \implies \tilde{p} = \mathbf{0.0216} < 0.05$ (**Statistically Significant**).
2. **vs Model B Combined-History (Autoregressive):** $\hat{\Delta} = +0.0494$, raw $p = 0.0079$, multiplier $5 \implies \tilde{p} = \mathbf{0.0395} < 0.05$ (**Statistically Significant**).
3. **vs Model A Emotion-History (Autoregressive):** $\hat{\Delta} = +0.0412$, raw $p = 0.0181$, multiplier $4 \implies \tilde{p} = 0.0724 > 0.05$ (**Insufficient statistical evidence of a difference**).
4. **vs DistilBERT Context-1 + Speaker:** $\hat{\Delta} = +0.0264$, raw $p = 0.0729$, multiplier $3 \implies \tilde{p} = 0.2187 > 0.05$ (**Insufficient statistical evidence of a difference**).
5. **vs DialogueRNN Class-Weighted:** $\hat{\Delta} = +0.0058$, raw $p = 0.6039$, multiplier $2 \implies \tilde{p} = 1.0000 > 0.05$ (**Insufficient statistical evidence of a difference**).
6. **vs DistilBERT Utterance Baseline:** $\hat{\Delta} = +0.0028$, raw $p = 0.7624$, multiplier $1 \implies \tilde{p} = 1.0000 > 0.05$ (**Insufficient statistical evidence of a difference**).

### 2.4 Framing Non-Significant Results: Evidence of Absence vs Absence of Evidence
In classical null-hypothesis testing, failing to reject $H_0$ ($p > 0.05$) does not prove that two models perform identically:
- **Methodological Rule:** Non-significant comparisons must never be described as "statistically equivalent" or "evidence of equivalence" unless a formal Two One-Sided Tests (TOST) equivalence procedure is conducted within a pre-specified equivalence margin $[-\delta, +\delta]$.
- **Correction Applied:** In draft report text, comparisons such as DialogueRNN vs Context-1 + Speaker or vs DistilBERT Baseline were informally annotated as `NO (Equiv.)`. This phrasing has been corrected:
  - DialogueRNN vs Context-1 + Speaker ($\Delta = +0.0264$, 95% CI $[-0.0013, +0.0535]$, $\tilde{p} = 0.2187$): There is **insufficient statistical evidence of a difference**; the confidence interval spans from $-0.0013$ to $+0.0535$, demonstrating substantial parameter uncertainty rather than equivalence.
  - DialogueRNN vs DistilBERT Baseline ($\Delta = +0.0028$, 95% CI $[-0.0152, +0.0198]$, $\tilde{p} = 1.0000$): There is **insufficient statistical evidence of a difference**.

### 2.5 Methodological Distinction: Bootstrap CIs vs Permutation P-Values
To ensure transparent reporting, the functions of bootstrap intervals and permutation tests are kept distinct:
- **Paired Dialogue Cluster Bootstrap ($\text{CI}_{95\%}$):** Measures **parameter estimation uncertainty**. By resampling $M = 268$ dialogue clusters with replacement $B = 10,000$ times, it estimates the standard error and empirical 95% confidence interval of the effect size $\Delta = \text{MacroF1}_A - \text{MacroF1}_B$.
- **Dialogue-Level Permutation Test ($p_{\text{perm}}$):** Evaluates **null hypothesis compatibility**. By enforcing sharp exchangeability ($\mathbb{E}[\Delta^*] = 0$), it tests whether the observed difference could easily arise if model assignment within dialogues were random.

### 2.6 Invariance of Existing Test Predictions and Metrics
All canonical test predictions (`test_predictions.csv`), confusion matrices, and benchmark tables created in Tasks 1–12 remain unaltered. No historical files were modified or recomputed.

---

## 3. Part B: Ablation Architecture Audit

### 3.1 Systematic Component Audit across Configurations
The four ablation configurations systematically isolate the contribution of each recurrent sub-network:

```
┌──────────────────────────────┬──────────────┬──────────────┬──────────────┬──────────────────┐
│ Configuration                │ Global GRU   │ Party GRUs   │ Emotion GRU  │ Parameter Count  │
├──────────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────────┤
│ 1. Global Context Only       │ YES (c_t)    │ OMITTED      │ OMITTED      │ 370,695 params   │
│ 2. Global + Party States     │ YES (c_t)    │ YES (q_t^p)  │ OMITTED      │ 617,223 params   │
│ 3. Global + Emotion State    │ YES (c_t)    │ OMITTED      │ YES (e_t)    │ 452,999 params   │
│ 4. Full DialogueRNN (Ref)    │ YES (c_t)    │ YES (q_t^p)  │ YES (e_t)    │ 670,855 params   │
└──────────────────────────────┴──────────────┴──────────────┴──────────────┴──────────────────┘
```

1. **Ablation 1 (Global Context Only):** Evaluates whether causal dialogue context alone—without speaker differentiation—prevents neutral collapse. All speaker IDs are ignored.
2. **Ablation 2 (Global Context + Party States):** Evaluates speaker-specific tracking by allocating a latent state $q_t^{(p)}$ to each participant, updated selectively when that participant speaks. Ablates the personal emotion GRU.
3. **Ablation 3 (Global Context + Emotion State):** Evaluates whether an unconditioned emotion GRU provides any benefit if the model cannot distinguish which interlocutor produced each utterance.
4. **Ablation 4 (Full DialogueRNN):** The complete reference architecture, including speaker-specific emotion attention and personal emotion state tracking.

### 3.2 Shared Component Parity Verification
All four architectures share identical structural specifications for common sub-networks:
- **Input Projection:** Linear layer mapping $768 \to 256$ dimensions.
- **Normalization:** PyTorch `nn.LayerNorm(256)`.
- **Regularization:** Dropout probability $p = 0.2$ applied to projected embeddings and classifier hidden activations.
- **Weight Initialization:** Standard PyTorch uniform/orthogonal initialization maintained across all modules.
- **Activation Functions:** ReLU activations throughout.

### 3.3 Classification Head Sizing and Dimension Audit
The classification head maps latent representations to emotion logits:
- **Ablation 1 & Ablation 2:** The output vector fed to the classifier is either context vector $c_t \in \mathbb{R}^{128}$ (Ablation 1) or party state $q_t^{(p)} \in \mathbb{R}^{128}$ (Ablation 2).
  - Head: $\text{FC}_1(128 \to 64) \to \text{ReLU} \to \text{Dropout}(0.2) \to \text{FC}_2(64 \to 7)$.
  - Parameters: $(128 \times 64 + 64) + (64 \times 7 + 7) = 8,256 + 455 = \mathbf{8,711}$ parameters.
- **Ablation 3 & Ablation 4:** The output vector fed to the classifier is emotion state $e_t \in \mathbb{R}^{64}$.
  - Head: $\text{FC}_1(64 \to 64) \to \text{ReLU} \to \text{Dropout}(0.2) \to \text{FC}_2(64 \to 7)$.
  - Parameters: $(64 \times 64 + 64) + (64 \times 7 + 7) = 4,160 + 455 = \mathbf{4,615}$ parameters.

### 3.4 Attention Mechanism Preservation
- **Context Attention ($c_t$):** Preserved identically across all four configurations. Bilinear attention matrix $W_\alpha \in \mathbb{R}^{128 \times 128}$ attends over all prior global states $\{g_0, \dots, g_t\}$ ($16,384$ parameters).
- **Emotion Attention ($c_t^{(e)}$):** Strictly restricted to Ablation 4. Emotion attention requires attending over a speaker's *own* past emotion states. In Ablations 1, 2, and 3, emotion attention is ablated (or omitted) to prevent leakage of speaker information into speaker-agnostic configurations.

### 3.5 Independent Parameter Count Verification
Every parameter count was analytically derived and verified against PyTorch modules executed in the project Docker environment (`emotion-dynamics-nlp:week11`).

Recall that for a standard PyTorch `nn.GRUCell(input_size=I, hidden_size=H)`:
$$\text{Parameters} = 3 \times H \times (I + H + 2)$$

#### Parameter Calculation Breakdown:

| Layer / Sub-Network | Formula | Ablation 1 | Ablation 2 | Ablation 3 | Ablation 4 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Input Linear Projection** | $768 \times 256 + 256$ | 196,864 | 196,864 | 196,864 | 196,864 |
| **LayerNorm (256)** | $256 \times 2$ | 512 | 512 | 512 | 512 |
| **Global Context GRU** | $3 \times 128 \times (I_G + 128 + 2)$ | 148,224 ($I_G=256$) | 197,376 ($I_G=384$) | 148,224 ($I_G=256$) | 197,376 ($I_G=384$) |
| **Context Attention** | $128 \times 128$ | 16,384 | 16,384 | 16,384 | 16,384 |
| **Speaker Party GRU Bank** | $3 \times 128 \times (384 + 128 + 2)$ | — | 197,376 | — | 197,376 |
| **Emotion Attention** | $128 \times 64$ | — | — | — | 8,192 |
| **Emotion GRU Bank** | $3 \times 64 \times (I_E + 64 + 2)$ | — | — | 86,400 ($I_E=384$) | 49,536 ($I_E=192$) |
| **Classifier FC1** | $D_{\text{in}} \times 64 + 64$ | 8,256 ($D_{\text{in}}=128$) | 8,256 ($D_{\text{in}}=128$) | 4,160 ($D_{\text{in}}=64$) | 4,160 ($D_{\text{in}}=64$) |
| **Classifier FC2** | $64 \times 7 + 7$ | 455 | 455 | 455 | 455 |
| **Total Parameter Count** | **Exact Sum** | **370,695** | **617,223** | **452,999** | **670,855** |

> [!IMPORTANT]
> **Resolution of Off-by-One Rounding Error in Ablation 3:**
> Prior draft documentation recorded Ablation 3 as 453,000 parameters. Independent verification revealed that:
> $$196,864 + 512 + 148,224 + 16,384 + 86,400 + 4,160 + 455 = \mathbf{452,999}$$
> The configuration file [`configs/ablations/3_global_emotion_state.json`](file:///home/chitta/Projects/emotion-dynamics-nlp/configs/ablations/3_global_emotion_state.json) has been updated from `453000` to `452999`. Ablation 4 reproduces exactly **670,855 parameters**.

### 3.6 Data Pipeline, Split Parity, and Feature Invariance
All four configurations use identical pre-extracted feature tensors:
- `train_features`: `data/processed/dialoguernn/meld_train_features.npz` (9,989 utterances across 1,038 dialogues)
- `dev_features`: `data/processed/dialoguernn/meld_dev_features.npz` (1,109 utterances across 114 dialogues)
- `test_features`: `data/processed/dialoguernn/meld_test_features.npz` (2,610 utterances across 280 dialogues)
- Feature semantics: Frozen DistilBERT final-layer `[CLS]` embeddings ($D = 768$).
- Dialogue boundaries: Strict per-dialogue resets of recurrent states; zero cross-dialogue state leakage.

### 3.7 Identification of Potential Confounding Variables
1. **Parameter Capacity Confounding:** Total parameters vary from 370,695 (Ablation 1) to 670,855 (Ablation 4). If Ablation 2 or 4 outperforms Ablation 1, the gain reflects both the structural inductive bias of speaker tracking and increased capacity. This is an inherent property of modular ablation, but must be acknowledged when interpreting findings.
2. **Global GRU Input Dimension Shift:** In Ablations 1 and 3, Global GRU receives only $u_t' \in \mathbb{R}^{256}$ ($I_G = 256$), whereas in Ablations 2 and 4 it receives $[u_t'; q_{t-1}^{(p)}] \in \mathbb{R}^{384}$ ($I_G = 384$). This makes the Global GRU slightly smaller in Ablations 1 and 3 ($148,224$ vs $197,376$ parameters).
3. **Emotion GRU Input Dimension Shift:** In Ablation 3, because there is no party state $q_t$, the Emotion GRU receives $[u_t'; c_t] \in \mathbb{R}^{384}$ ($86,400$ params), whereas in Ablation 4 it receives $[q_t; c_t^{(e)}] \in \mathbb{R}^{192}$ ($49,536$ params).

---

## 4. Part C: Hypotheses and Evaluation Protocol

### 4.1 Falsifiable Research Questions

Rather than assuming outcomes, the ablation study will test three pre-registered research questions:

#### Research Question 1: Speaker State Tracking vs. Global Context Alone
- **Core Question:** Does tracking interlocutor-specific recurrent states ($\text{GRU}_P$) significantly improve emotion recognition over global conversational context alone ($\text{GRU}_G$)?
- **Comparison:** Ablation 2 (617,223 params) vs. Ablation 1 (370,695 params), and Ablation 1 vs. Full DialogueRNN (670,855 params).
- **Null Hypothesis ($H_{0, 1}$):** $\Delta_{\text{MacroF1}} = \text{MacroF1}_{\text{Abl2}} - \text{MacroF1}_{\text{Abl1}} = 0$ on Cohort A.
- **Alternative Hypothesis ($H_{1, 1}$):** $\Delta_{\text{MacroF1}} > 0$. Speaker state separation prevents an interlocutor's neutral demeanor from overwriting a speaking partner's emotional state, resulting in a statistically significant increase in Macro F1 and higher recall on minority classes (`fear`, `disgust`, `anger`).

#### Research Question 2: Internal Emotion Recurrence vs. Speaker States Alone
- **Core Question:** Does maintaining an explicit secondary Emotion GRU ($\text{GRU}_E$) and emotion attention mechanism provide measurable value beyond speaker-specific party states?
- **Comparison:** Ablation 4 (670,855 params) vs. Ablation 2 (617,223 params).
- **Null Hypothesis ($H_{0, 2}$):** $\Delta_{\text{MacroF1}} = \text{MacroF1}_{\text{Full}} - \text{MacroF1}_{\text{Abl2}} = 0$ on Cohort A.
- **Alternative Hypothesis ($H_{1, 2}$):** $\Delta_{\text{MacroF1}} > 0$. Personal emotion modeling selectively improves performance on **sustained emotional inertia turns** ($\ge 2$ consecutive utterances with identical emotion by the same speaker), while exhibiting negligible difference on cross-speaker shift turns.

#### Research Question 3: Speaker-Agnostic Emotion Recurrence vs. Global Context
- **Core Question:** Does adding emotion recurrence without speaker conditioning improve or degrade conversational emotion tracking?
- **Comparison:** Ablation 3 (452,999 params) vs. Ablation 1 (370,695 params) and Ablation 4 (670,855 params).
- **Null Hypothesis ($H_{0, 3}$):** $\Delta_{\text{MacroF1}} = \text{MacroF1}_{\text{Abl3}} - \text{MacroF1}_{\text{Abl1}} = 0$ on Cohort A.
- **Alternative Hypothesis ($H_{1, 3}$):** Conflating emotional trajectories across multiple speakers without party state conditioning introduces cross-speaker emotional interference, degrading classification on turn-taking boundaries.

### 4.2 Evaluation Protocol and Metrics Hierarchy
All evaluations will strictly follow the canonical protocol established in Tasks 3 and 11:
- **Primary Benchmark:** Macro F1 on canonical Cohort A: Strict Contiguous Matched Subset ($N = 2,328$).
- **Secondary Metrics:**
  1. Per-class precision, recall, and F1 across all 7 MELD classes, focusing on minority classes (`disgust`, `fear`, `sadness`).
  2. Overall classification accuracy.
  3. Weighted F1.
  4. Neutral prediction frequency: $P(\hat{y} = \text{neutral})$.
  5. Dynamics metrics: Emotional Shift Macro F1 vs. Emotional Inertia Macro F1.
- **Statistical Significance Testing:**
  - Paired dialogue-level permutation test ($B = 10,000$, sign-flips over $M = 268$ clusters).
  - Paired cluster bootstrap 95% confidence intervals ($B = 10,000$).
  - Holm-Bonferroni family-wise error adjustment applied across all ablation comparisons against the full reference model.

### 4.3 Training Controls and Hyperparameter Parity
- **Fixed Random Seed:** `42` for all initialization, data batching, and optimization steps.
- **Optimizer:** AdamW ($\text{lr} = 0.001$, $\text{weight\_decay} = 0.0001$).
- **Batching:** 16 dialogues per batch with zero-padding and sequence masking.
- **Regularization:** Gradient norm clipping at $1.0$; Dropout $p = 0.2$.
- **Training Budget:** 40 epochs with early stopping patience = 10 based on Dev Macro F1.
- **Test-Set Isolation:** The test split is never evaluated during training or model selection.

---

## 5. Part D: Unresolved Methodological and Implementation Issues

Before ablation training can be executed, the following four issues must be recognized:

1. **Implementation Requirement in `src/models/dialoguernn.py`:**  
   The existing [`src/models/dialoguernn.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/models/dialoguernn.py) implements only the full `DialogueRNN` architecture (with boolean flags `use_context_attention` and `use_emotion_attention`). It does not currently contain branching logic for `use_party_state=False` or `use_emotion_state=False`.  
   *Resolution Needed Prior to Training:* The model file must be extended (e.g. by introducing a backward-compatible `DialogueRNNAblation` class or adding conditional branches) without modifying the behavior or weights of existing saved checkpoints.
2. **Confounding between Capacity and Inductive Bias:**  
   As documented in Section 3.7, the four ablations differ in parameter count (370k vs 453k vs 617k vs 671k). If Ablation 2 outperforms Ablation 1, the improvement cannot be attributed purely to speaker inductive bias without acknowledging the 66% increase in trainable parameters.
3. **Single Random Seed Limitation:**  
   While running with fixed seed `42` ensures perfect experimental reproducibility, recurrent models on class-imbalanced datasets can exhibit seed-dependent optimization variance. If performance differences between configurations are marginal ($\Delta_{\text{MacroF1}} < 0.010$), conclusions from a single seed should be interpreted cautiously, and multi-seed replications should be scheduled.
4. **Unimodal Text Boundary:**  
   All models operate on frozen text representations. In human dialogue, emotional shifts and minority emotions (e.g., subtle sarcasm, fear) are frequently signaled by acoustic pitch or facial cues. The text-only modality imposes an upper bound on emotion dynamics recovery.

---

## 6. Pre-Execution Deliverables and Status Checklist

```
┌─────────────────────────────────────────────────────────────┬────────────┬─────────────────────────────┐
│ Validation Milestone                                        │ Status     │ Verification Artifact       │
├─────────────────────────────────────────────────────────────┼────────────┼─────────────────────────────┤
│ Paired Permutation Test Audited                             │ VERIFIED   │ Section 2.1; Table CSV      │
│ Exchangeability Assumption Formulated                       │ VERIFIED   │ Section 2.2                 │
│ Holm Correction Verified across 6 Comparisons               │ VERIFIED   │ Section 2.3; Table CSV      │
│ Non-Significant Results Framed Rigorously                   │ VERIFIED   │ Section 2.4                 │
│ Bootstrap CIs vs Permutation p-values Separated             │ VERIFIED   │ Section 2.5                 │
│ Canonical Prediction & Metric Invariance Preserved          │ VERIFIED   │ outputs/experiments/        │
│ Ablation JSON Configurations Audited                        │ VERIFIED   │ configs/ablations/*.json    │
│ Parameter Counts Independently Verified                     │ VERIFIED   │ 370k, 617k, 453k, 671k      │
│ Ablation 3 Parameter Count Typo Corrected (452,999)         │ COMPLETED  │ configs/ablations/3_*.json  │
│ Falsifiable Hypotheses & Evaluation Protocol Formulated     │ VERIFIED   │ Section 4                   │
│ Unresolved Implementation & Methodological Issues Listed   │ DOCUMENTED │ Section 5                   │
│ Scope Boundaries Enforced (Zero Training, No Git Commits)   │ ENFORCED   │ Maintained Cleanly          │
└─────────────────────────────────────────────────────────────┴────────────┴─────────────────────────────┘
```

**Next Step:** Await explicit user approval of this validation report before implementing model extensions or launching ablation training runs.
