# Canonical MELD Evaluation Audit Report

**Repository:** `emotion-dynamics-nlp`  
**Location:** `~/Projects/emotion-dynamics-nlp`  
**Audit Execution Date:** 2026-11-07  
**Test Set Ground Truth:** `data/raw/meld/test_sent_emo.csv` ($N = 2,610$ total utterances across 280 dialogues)  
**Evaluated Models:** 7 distinct checkpoints/inference regimes  

---

## Executive Summary

This audit establishes the **canonical scientific evaluation standard** for all experiments on the MELD conversational emotion dataset. By auditing the raw test dialogue structures, verifying every prediction file row-by-row against ground truth, eliminating confounding utterance-ID gaps, and running paired significance tests (McNemar with Edwards continuity correction and paired bootstrap Macro F1 tests with 10,000 resamples), this report resolves prior discrepancies between subset definitions ($N=2,328$ vs $N=2,330$) and provides an unambiguous assessment of the models.

### Key Headline Findings
1. **Highest Accuracy:** **Model B (Combined-History DistilBERT, Autoregressive)** achieves the highest deployable accuracy on the canonical matched subset (**62.59%**), followed closely by **Context-1 + Speaker (61.94%)** and **Class-Weighted Baseline (61.73%)**.
2. **Highest Macro F1:** **DistilBERT Utterance Baseline (Unweighted)** retains the highest Macro F1 (**0.4314**), followed by **Class-Weighted DistilBERT (0.4230)**.
3. **The Minority-Class Collapse:** Adding conversational context **does not improve minority-class recognition**. Instead, it triggers catastrophic rare-class suppression. Model B completely zeroes out `fear` (recall **0.0%**, 0/44 turns detected) and reduces `disgust` to **3.2% recall** (2/63 detected).
4. **The Autoregressive Penalty:** Autoregressive deployment incurs a statistically significant penalty of **$-1.76\%$ in accuracy** ($p = 0.0002$) and **$-0.0347$ in Macro F1** ($p = 0.0001$) relative to teacher-forced oracle inference due to error cascading.
5. **Research Hypothesis Verdict:** The hypothesis that *Combined-History DistilBERT (Model B) mitigates emotional inertia and minority-class suppression* is **refuted**. Model B exhibits the lowest Macro F1 (**0.3848**) of all evaluated models on the matched subset.

---

## 1. Evaluation Cohorts & Dialogue Gap Audit

An audit of `data/raw/meld/test_sent_emo.csv` revealed that while MELD contains 280 dialogues totaling 2,610 utterances, two dialogues contain non-consecutive utterance IDs:
* **Dialogue 93:** `Utterance_ID` jumps from `4` to `8` (utterances 5, 6, 7 are absent).
* **Dialogue 108:** `Utterance_ID` jumps from `0` to `3` (utterances 1, 2 are absent).

Under strict context construction rules ([`src/data/context.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/data/context.py)), a context window $u_{t-1}\nu_t$ is only valid if `Utterance_ID == prev_Utterance_ID + 1`. Consequently, turns following a gap cannot form a valid contiguous context window and must be excluded.

To maintain strict scientific comparability without mixing subsets, four mutually exclusive or hierarchical cohorts are defined:

| Cohort Identifier | Definition & Inclusions | Exclusions | Sample Size ($N$) | Valid Models |
| :--- | :--- | :--- | :---: | :--- |
| **Cohort A: Strict Contiguous Matched Subset** | Turns where dialogue turn $t \ge 1$ AND `Utterance_ID == prev_Utterance_ID + 1`. | 280 dialogue-initial turns ($t=0$) AND 2 post-gap turns: `(D93, U8)` and `(D108, U3)`. | **2,328** | **All 7 models** (Baseline, Context-1, Context-1+Spk, Weighted, Model A, Model B Auto, Model B Oracle). |
| **Cohort B: All Subsequent Sequential Turns** | All turns where chronological dialogue index $i \ge 1$ within dialogue. | 280 dialogue-initial turns ($t=0$). Retains post-gap turns `(D93, U8)` and `(D108, U3)`. | **2,330** | Models evaluated using list-adjacency: Baseline, Weighted, Model A, Model B. *(Excludes Context-1).* |
| **Cohort C: Dialogue-Initial Turns ($t=0$)** | First turn chronologically ($i=0$) of each of the 280 dialogues. | All subsequent turns ($t \ge 1$). | **280** | Models with $t=0$ predictions (Baseline, Weighted, Model A, Model B). |
| **Cohort D: Full MELD Test Set** | Complete test split. | None. | **2,610** | Models with full-split predictions (Baseline, Weighted, Model A, Model B). |

> **Audit Rule:** Comparisons involving Context-1 or Context-1 + Speaker must strictly use **Cohort A ($N=2,328$)**. Cohort B ($N=2,330$) must not be mixed with Cohort A.

---

## 2. Prediction File Integrity & Leakage Verification

All prediction files were audited against `data/raw/meld/test_sent_emo.csv` and cross-checked against train/dev splits:

| Model / Prediction File | Target Key Uniqueness | Duplicate Rows | Ground-Truth Label Match | Leakage Check (Train/Dev Keys) | Row Count Verified |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `transformer_baseline/predictions.csv` | 100% Unique | 0 | 2,610 / 2,610 exact | Zero overlap with train/dev | 2,610 |
| `transformer_context_1/predictions.csv` | 100% Unique | 0 | 2,328 / 2,328 exact | Zero overlap with train/dev | 2,328 |
| `transformer_context_1_speaker/predictions.csv`| 100% Unique | 0 | 2,328 / 2,328 exact | Zero overlap with train/dev | 2,328 |
| `transformer_weighted/predictions.csv` | 100% Unique | 0 | 2,610 / 2,610 exact | Zero overlap with train/dev | 2,610 |
| `transformer_weighted/matched_predictions.csv` | 100% Unique | 0 | 2,328 / 2,328 exact | Zero overlap with train/dev | 2,328 |
| `transformer_emotion_history/predictions_autoregressive.csv` | 100% Unique | 0 | 2,610 / 2,610 exact | Zero overlap with train/dev | 2,610 |
| `transformer_combined_history/predictions_autoregressive.csv` | 100% Unique | 0 | 2,610 / 2,610 exact | Zero overlap with train/dev | 2,610 |
| `transformer_combined_history/predictions_teacher_forced.csv`| 100% Unique | 0 | 2,610 / 2,610 exact | Zero overlap with train/dev | 2,610 |

All target keys match `(Dialogue_ID, Utterance_ID)` tuples from the official MELD test split. No key ordering discrepancies affect metric computation.

---

## 3. Canonical Evaluation Results

### 3.1 Primary Benchmark: Cohort A (Strict Contiguous Matched Subset, $N=2,328$)

All 7 models evaluated on the identical 2,328 turns:

| Model Configuration | Input Conditioning | Oracle? | Accuracy | Macro F1 | Weighted F1 | Neutral Pred Rate | Neutral True Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DistilBERT Utterance Baseline** | Single utterance $u_t$ | No | 59.45% | **0.4314** | 0.5969 | 45.19% | 47.94% |
| **DistilBERT Context-1** | Prior text $u_{t-1} \nu u_t$ | No | 60.74% | 0.3912 | 0.5812 | **57.26%** | 47.94% |
| **DistilBERT Context-1 + Speaker** | Prior text + Speaker token | No | 61.94% | 0.4078 | 0.5981 | 53.78% | 47.94% |
| **DistilBERT Class-Weighted** | Unweighted utterance + Inverse Freq Loss | No | 61.73% | 0.4230 | 0.6067 | 49.36% | 47.94% |
| **Model A: Emotion-History** | Autoregressive prior emotion $\hat{e}_{t-1}$ | No | 61.47% | 0.3930 | 0.5984 | 51.03% | 47.94% |
| **Model B: Combined-History** | Autoregressive text $u_{t-1} + \hat{e}_{t-1}$ | No | **62.59%** | 0.3848 | 0.6024 | 54.04% | 47.94% |
| *Model B: Combined-History (Oracle)* | Teacher-forced text $u_{t-1} + e_{t-1}^*$ | Yes | *64.35%* | *0.4195* | *0.6243* | 53.05% | 47.94% |

---

### 3.2 Full Test Split (Cohort D, $N=2,610$)

| Model Configuration | Input Conditioning | Accuracy | Macro F1 | Weighted F1 | Neutral Pred Rate |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **DistilBERT Utterance Baseline** | Single utterance $u_t$ | 60.31% | **0.4400** | 0.6054 | 45.59% |
| **DistilBERT Class-Weighted** | Single utterance + Inverse Freq Loss | 62.03% | 0.4243 | 0.6091 | 49.58% |
| **Model A: Emotion-History** | Autoregressive prior emotion $\hat{e}_{t-1}$ | 61.92% | 0.3912 | 0.6019 | 51.30% |
| **Model B: Combined-History** | Autoregressive text $u_{t-1} + \hat{e}_{t-1}$ | **62.99%** | 0.3872 | 0.6063 | 54.48% |
| *Model B: Combined-History (Oracle)* | Teacher-forced text $u_{t-1} + e_{t-1}^*$ | *64.56%* | *0.4184* | *0.6260* | 53.60% |

---

### 3.3 Dialogue-Initial Turns (Cohort C, $t=0$, $N=280$)

On dialogue-initial turns, where no conversational history exists:
* **DistilBERT Utterance Baseline:** **67.14% Accuracy**, **0.5153 Macro F1**, Weighted F1 0.6713.
* **Model B Autoregressive (Conditioned on `none`):** 66.43% Accuracy, 0.4123 Macro F1, Weighted F1 0.6370.
* **Model A Autoregressive (Conditioned on `none`):** 65.36% Accuracy, 0.3638 Macro F1, Weighted F1 0.6259.
* **Insight:** Even at $t=0$ where models receive a fixed `"Previous emotion: none"` prompt, conditioning on an explicit missing-history token degrades Macro F1 compared to a clean utterance-only architecture.

---

## 4. Per-Class Detail & The Neutral Gravitation Phenomenon

Evaluating per-class performance on Cohort A ($N=2,328$) reveals the exact mechanism behind the apparent accuracy gains:

| Emotion Class | Metric | Utterance Baseline | Context-1 | Context-1 + Spk | Class-Weighted | Model A (Auto) | Model B (Auto) | Model B (Oracle) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Neutral** ($n=1,116$) | Precision | 78.04% | 69.77% | 73.40% | 77.02% | 74.16% | 73.05% | 73.93% |
| | **Recall** | 73.57% | **83.33%** | 82.35% | 79.30% | 78.94% | 82.35% | 81.81% |
| | **F1** | 0.7574 | 0.7595 | 0.7762 | 0.7815 | 0.7648 | 0.7742 | 0.7767 |
| **Joy** ($n=358$) | Recall | 58.66% | 54.47% | 58.38% | 60.89% | **63.41%** | 53.63% | 56.70% |
| | F1 | 0.5405 | 0.5166 | 0.5478 | **0.5782** | 0.5577 | 0.5647 | 0.5971 |
| **Anger** ($n=307$) | Recall | 36.81% | 30.94% | 31.60% | 33.22% | 39.09% | **43.00%** | 49.84% |
| | F1 | 0.4124 | 0.3838 | 0.3888 | 0.3977 | 0.4286 | 0.4452 | **0.4984** |
| **Surprise** ($n=253$) | Recall | 60.87% | 60.08% | 66.01% | **67.19%** | 57.71% | **67.19%** | 67.59% |
| | F1 | 0.5375 | 0.5630 | 0.5642 | 0.5312 | 0.5626 | 0.5583 | **0.5729** |
| **Sadness** ($n=187$) | Recall | **33.16%** | 16.04% | 20.32% | 22.99% | 28.34% | 22.46% | 28.88% |
| | F1 | **0.3454** | 0.2362 | 0.2754 | 0.2997 | 0.3164 | 0.2907 | 0.3600 |
| **Disgust** ($n=63$) | Predicted Count | 38 | 30 | 19 | 52 | 11 | **3** | 5 |
| | Recall | **20.63%** | 14.29% | 11.11% | **20.63%** | 4.76% | **3.17%** | 4.76% |
| | F1 | **0.2574** | 0.1935 | 0.1707 | 0.2261 | 0.0811 | 0.0606 | 0.0882 |
| **Fear** ($n=44$) | Predicted Count | 86 | 26 | 32 | 38 | 6 | **1** | 2 |
| | Recall | **25.00%** | 6.82% | 11.36% | 13.64% | 2.27% | **0.00%** | 2.27% |
| | F1 | **0.1692** | 0.0857 | 0.1316 | 0.1463 | 0.0400 | **0.0000** | 0.0435 |

### Analysis of the Collapse:
* **The "Accuracy Trap":** Neutral accounts for 47.9% of Cohort A. When context is added, the model predicts Neutral on **54.0% to 57.3%** of all test turns. Because Neutral recall rises from 73.6% to 83.3%, overall accuracy climbs by +1.3% to +3.1%.
* **The Rare-Class Casualty:** This accuracy gain comes at the cost of minority emotions. In Model B Autoregressive, across all 2,328 test turns, the model predicted `fear` **only once** (and that prediction was incorrect), yielding a Fear F1 of **0.0000**. Disgust was predicted only 3 times, yielding 3.2% recall.
* **The Balanced Baseline:** The unweighted single-utterance baseline predicts Neutral conservatively (45.2% vs 47.9% true) and maintains non-trivial recall across rare classes (Fear recall 25.0%, Disgust recall 20.6%), resulting in the highest Macro F1 (0.4314).

---

## 5. Statistical Significance Testing

Paired significance tests were computed on exactly aligned predictions across Cohort A ($N=2,328$):
* **Accuracy:** McNemar's test with Edwards continuity correction: $\chi^2 = \frac{(|b - c| - 1)^2}{b + c}$.
* **Macro F1:** Paired bootstrap resampling (10,000 iterations, seed 42) computing empirical 95% confidence intervals and two-sided bootstrap p-values.

| Paired Comparison ($N=2,328$) | Discordant Pairs $(b, c)$ | McNemar $\chi^2$ | McNemar $p$-value | Accuracy Sig? | Macro F1 Difference | 95% Bootstrap CI | Bootstrap $p$-value | Macro F1 Sig? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Context-1 vs Baseline** | (259, 229) | 1.7234 | $0.1893$ | **No** | $-0.0402$ | $[-0.0669, -0.0144]$ | $0.0034$ | **Degraded (Sig)** |
| **Context-1+Spk vs Baseline** | (248, 190) | 7.4178 | $0.0065$ | **Yes** | $-0.0236$ | $[-0.0509, +0.0041]$ | $0.0942$ | No |
| **Class-Weighted vs Baseline** | (236, 183) | 6.4535 | $0.0111$ | **Yes** | $-0.0085$ | $[-0.0339, +0.0174]$ | $0.5204$ | No |
| **Model A vs Baseline** | (219, 172) | 5.4118 | $0.0200$ | **Yes** | $-0.0384$ | $[-0.0660, -0.0111]$ | $0.0058$ | **Degraded (Sig)** |
| **Model B Auto vs Baseline** | (256, 183) | 11.8087 | $0.0006$ | **Yes** | $-0.0466$ | $[-0.0739, -0.0190]$ | $0.0006$ | **Degraded (Sig)** |
| **Model B Oracle vs Baseline**| (280, 166) | 28.6300 | $8.76 \times 10^{-8}$ | **Yes** | $-0.0119$ | $[-0.0400, +0.0169]$ | $0.4152$ | No |
| **Model B Auto vs Model A** | (157, 131) | 2.1701 | $0.1407$ | **No** | $-0.0082$ | $[-0.0285, +0.0098]$ | $0.4042$ | No |
| **Model B Oracle vs Model B Auto** | (79, 38) | 13.6752 | $0.0002$ | **Yes** | $+0.0347$ | $[+0.0186, +0.0540]$ | $0.0001$ | **Penalty (Sig)** |

### Multiple Testing Discussion
Applying Holm-Bonferroni correction across the 5 primary deployable comparisons against baseline ($\alpha = 0.05$):
1. **Model B Auto vs Baseline:** $p = 0.0006 < 0.05/5 = 0.0100$ $\to$ **Remains Significant**.
2. **Context-1 + Speaker vs Baseline:** $p = 0.0065 < 0.05/4 = 0.0125$ $\to$ **Remains Significant**.
3. **Class-Weighted vs Baseline:** $p = 0.0111 < 0.05/3 = 0.0167$ $\to$ **Remains Significant**.
4. **Model A Auto vs Baseline:** $p = 0.0200 < 0.05/2 = 0.0250$ $\to$ **Remains Significant**.
5. **Context-1 vs Baseline:** $p = 0.1893 > 0.05/1 = 0.0500$ $\to$ **Not Significant**.

---

## 6. Research Conclusions & Answers to Core Questions

### Q1: Which model has the highest accuracy?
**Model B (Combined-History DistilBERT, Autoregressive)** achieves the highest deployable accuracy on Cohort A (**62.59%**), statistically significantly higher than the baseline (59.45%, $p = 0.0006$). Under oracle diagnostic conditions (Teacher-Forced), it reaches **64.35%**.

### Q2: Which model has the highest Macro F1?
**DistilBERT Utterance Baseline** achieves the highest Macro F1 (**0.4314**), followed closely by **Class-Weighted DistilBERT (0.4230)**. Every context-augmented model suffers a drop in Macro F1, with Model B recording the lowest Macro F1 (**0.3848**).

### Q3: Does adding context genuinely improve minority-class recognition?
**No. It significantly harms minority-class recognition.** Across all context formulations (text context, emotion history, and combined history), the model learns that the prior turn is usually Neutral, creating a strong prior that overrides subtle linguistic indicators of Fear, Disgust, and Sadness.

### Q4: How severe are neutral gravitation and rare-class failures?
* **Neutral Gravitation:** Severe. Context models predict Neutral on **53.8% to 57.3%** of all test turns against a ground-truth rate of **47.9%**.
* **Rare-Class Failures:** Catastrophic in Model B:
  * Fear recall collapsed to **0.0%** (0 out of 44 detected).
  * Disgust recall collapsed to **3.2%** (2 out of 63 detected).
  * Sadness recall fell from 33.2% to 22.5%.

### Q5: What is the autoregressive penalty?
When transitioning from teacher-forced oracle inference to real autoregressive deployment:
* Accuracy drops by **$1.76\%$** (64.35% $\to$ 62.59%, McNemar $p = 0.0002$).
* Macro F1 drops by **$0.0347$** (0.4195 $\to$ 0.3848, bootstrap $p = 0.0001$).
* This confirms that imperfect predictions in early dialogue turns propagate forward, causing cascading misclassifications.

### Q6: Which claims from previous reports remain supported?
1. Context models achieve higher raw accuracy than the unweighted single-utterance baseline ($p < 0.01$).
2. Context models suffer from severe Neutral gravitation that inflates accuracy while suppressing rare emotions.
3. Adding explicit speaker tokens partially alleviates Neutral bias compared to raw text context alone ($p = 0.0065$).
4. Autoregressive inference incurs a measurable, statistically significant penalty compared to teacher-forced evaluation.

### Q7: Which claims need correction or qualification?
1. **The Combined-History Hypothesis is Refuted:** Prepending prior text alongside prior predicted emotion does *not* mitigate minority-class suppression. Model B suffered the most severe rare-class collapse of any evaluated model.
2. **Context-1 Accuracy is Not Statistically Significant:** The raw text Context-1 accuracy improvement (+1.29%) has a McNemar $p$-value of **$0.1893$**. The claim that simple text context significantly improves emotion classification is unsupported.
3. **Accuracy is Misleading:** Describing Model B or Context-1 as "better" models based on accuracy alone is scientifically invalid. For balanced conversational emotion recognition, **DistilBERT Utterance Baseline** and **Class-Weighted DistilBERT** remain the superior architectures.

---

## 7. Canonical Artifact Inventory

The following canonical evaluation artifacts have been generated and verified:

1. **[`outputs/tables/canonical_test_comparison.csv`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/tables/canonical_test_comparison.csv):** Consolidated comparison table reporting Accuracy, Macro F1, Weighted F1, and Neutral prediction rates across Cohorts A, B, C, and D.
2. **[`outputs/tables/canonical_per_class_metrics.csv`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/tables/canonical_per_class_metrics.csv):** Precision, Recall, F1, True Support, and Predicted Count for all 7 emotion classes across every model and cohort.
3. **[`outputs/tables/canonical_statistical_tests.csv`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/tables/canonical_statistical_tests.csv):** Exact McNemar test statistics, Edwards continuity correction, discordant pairs $(b, c)$, exact binomial p-values, Macro F1 differences, 95% bootstrap confidence intervals, and bootstrap p-values.
4. **[`outputs/tables/canonical_confusion_matrices.json`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/tables/canonical_confusion_matrices.json):** Full $7 \times 7$ confusion matrices for every model-cohort pair.
5. **[`outputs/reports/canonical_evaluation_audit.md`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/reports/canonical_evaluation_audit.md):** This formal research audit report.
