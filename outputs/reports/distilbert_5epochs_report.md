# Controlled Five-Epoch DistilBERT Experiment Report
**Empirical Investigation into Extended Fine-Tuning for Utterance-Level Emotion Classification on MELD**

**Document Version:** 1.0.0  
**Date:** November 16, 2026  
**Experiment Identifier:** `distilbert_5epochs`  
**Checkpoint Path:** `checkpoints/distilbert_5epochs/best` (Selected: Epoch 4)  
**Execution Environment:** Linux (`x86_64`), AMD Ryzen 5 7530U (6 Cores / 12 Threads), 14 GiB RAM, PyTorch 2.13.0+cpu  

---

## 1. Executive Summary & Objective

This experiment investigates whether extending fine-tuning from 3 to 5 epochs improves the utterance-level emotion classification baseline on the MELD dataset, or whether overfitting and majority-class representation collapse occur.

### Key Empirical Findings:
1. **Convergence & Overfitting Trajectory:**
   - Training loss steadily decreased from **1.2035** (Epoch 1) to **0.3610** (Epoch 5).
   - Validation Macro F1 peaked at **Epoch 4 (42.15%)**, after which it degraded significantly to **39.66%** in Epoch 5 as the model overfit the training split.
   - **Epoch 4** was selected as the optimal checkpoint based on the pre-registered criterion (highest validation Macro F1).
2. **The Accuracy vs. Macro F1 Divergence:**
   - On the full test set ($N=2,610$), the 5-epoch model achieved **62.11% accuracy** (an improvement of **+1.80%** over the 3-epoch baseline of 60.31%).
   - On Canonical Cohort A ($N=2,328$), accuracy reached **61.90%** (an improvement of **+2.45%** over the 3-epoch baseline of 59.45%).
   - However, **Test Macro F1 declined** from **44.00% to 41.73%** (Full Test) and from **43.14% to 41.70%** (Cohort A).
3. **Underlying Mechanism:** Extended unweighted fine-tuning on imbalanced dialogue data causes the model to optimize cross-entropy loss by predicting the dominant *Neutral* class with higher confidence (neutral recall rose to 82.96%), boosting accuracy while degrading recall on scarce minority classes (*Fear* recall collapsed to 10.0%, *Disgust* recall to 14.7%).
4. **Conclusion:** Additional training epochs without class reweighting or sequential dialogue context do **not** improve balanced conversational emotion recognition; 3 epochs remains the optimal stopping point for balanced utterance-level classification.

---

## 2. Hardware Profile and Timing Summary

| Hardware Metric | Value |
| :--- | :--- |
| **Processor** | AMD Ryzen 5 7530U with Radeon Graphics |
| **Topology** | 1 Socket, 6 Physical Cores, 12 Threads |
| **Memory Allocation** | 14 GiB Total RAM, 6.3 GiB Available, 8.0 GiB Swap |
| **Device Used** | CPU (`torch.set_num_threads(6)`) |
| **Training Duration** | **4.00 hours** (240.2 minutes) |
| **Mean Time per Epoch** | ~48.0 minutes (Train: ~46.5 min, Dev Eval: ~1.0 min) |
| **Evaluation Duration** | ~3.5 minutes (Full Test $N=2,610$ + Cohort A $N=2,328$) |

---

## 3. Training & Validation Progression Across 5 Epochs

Hyperparameters: `batch_size = 4`, `lr = 2e-5` (AdamW), `max_length = 128`, `seed = 42`, unweighted cross-entropy loss.

| Epoch | Train Loss | Train Time | Val Time | Val Accuracy | Val Macro F1 | Val Weighted F1 | Checkpoint Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | 1.2035 | 49.19 min | 1.06 min | 60.32% | 37.68% | 57.20% | Saved (`epoch_1`) |
| **2** | 0.9936 | 47.05 min | 0.97 min | 60.23% | 40.89% | 56.33% | Saved (`epoch_2`) |
| **3** | 0.7610 | 46.35 min | 0.97 min | 58.88% | 41.48% | 56.93% | Saved (`epoch_3`) |
| **4** | **0.5227** | 46.34 min | 0.98 min | 57.53% | **42.15%** | 55.43% | **★ BEST Checkpoint Selected** |
| **5** | 0.3610 | 46.24 min | 0.98 min | 56.90% | 39.66% | 54.30% | Saved (`epoch_5`, overfit) |

### Observation:
Between Epoch 1 and Epoch 4, the model progressively learns discriminative features for minority classes, raising validation Macro F1 from 37.68% to 42.15%. In Epoch 5, loss drops from 0.52 to 0.36, but validation Macro F1 plummets by 2.49 points, confirming severe empirical overfitting.

---

## 4. Test Set Evaluation (Selected Best Checkpoint: Epoch 4)

### 4.1 Full Test Set Evaluation ($N=2,610$)

```
===== EVALUATION (Full Test Set, N=2,610) =====
Accuracy     : 0.6211 (62.11%)
Macro F1     : 0.4173 (41.73%)
Weighted F1  : 0.6010 (60.10%)

Per-Class Metrics:
-----------------------------------------------------------------
Emotion        Precision      Recall          F1        Support
anger             0.4484      0.3275      0.3786            345
disgust           0.5263      0.1471      0.2299             68
fear              0.1515      0.1000      0.1205             50
joy               0.5561      0.5796      0.5676            402
neutral           0.7246      0.8296      0.7736           1256
sadness           0.4752      0.2308      0.3107            208
surprise          0.4885      0.6050      0.5405            281
-----------------------------------------------------------------

Confusion Matrix (Labels: anger, disgust, fear, joy, neutral, sadness, surprise):
[[ 113    2    6   57  104    8   55]
 [  14   10    2    5   28    1    8]
 [   8    0    5    4   20    5    8]
 [  30    2    1  233  107    7   22]
 [  36    3   11   66 1042   28   70]
 [  19    2    5   21   98   48   15]
 [  32    0    3   33   39    4  170]]
```

### 4.2 Canonical Cohort A Matched Evaluation ($N=2,328$)

```
===== EVALUATION (Canonical Cohort A, N=2,328) =====
Accuracy     : 0.6190 (61.90%)
Macro F1     : 0.4170 (41.70%)
Weighted F1  : 0.5986 (59.86%)

Per-Class Metrics:
-----------------------------------------------------------------
Emotion        Precision      Recall          F1        Support
anger             0.4386      0.3257      0.3738            307
disgust           0.5000      0.1270      0.2025             63
fear              0.1786      0.1136      0.1389             44
joy               0.5556      0.5726      0.5640            358
neutral           0.7205      0.8271      0.7701           1116
sadness           0.5000      0.2406      0.3249            187
surprise          0.4905      0.6126      0.5448            253
-----------------------------------------------------------------
```

---

## 5. Comparative Analysis: 3-Epoch vs. 5-Epoch Baseline

### 5.1 Full Test Comparison ($N=2,610$)

```
+---------------------+-------------------+-------------------+--------------------+
| Metric              | Original 3-Epochs | New 5-Epochs      | Delta              |
+---------------------+-------------------+-------------------+--------------------+
| Epoch Selected      | Epoch 3           | Epoch 4           | +1 Epoch           |
| Test Accuracy       | 60.31%            | 62.11%            | **+1.80%**         |
| Test Macro F1       | 44.00%            | 41.73%            | **-2.27%**         |
| Test Weighted F1    | 60.54%            | 60.10%            | **-0.44%**         |
+---------------------+-------------------+-------------------+--------------------+
```

### 5.2 Canonical Cohort A Matched Comparison ($N=2,328$)

```
+---------------------+-------------------+-------------------+--------------------+
| Metric              | Original 3-Epochs | New 5-Epochs      | Delta              |
+---------------------+-------------------+-------------------+--------------------+
| Test Accuracy       | 59.45%            | 61.90%            | **+2.45%**         |
| Test Macro F1       | 43.14%            | 41.70%            | **-1.44%**         |
| Test Weighted F1    | 57.94%            | 59.86%            | **+1.92%**         |
+---------------------+-------------------+-------------------+--------------------+
```

### 5.3 Per-Class F1 Shift Analysis:
* **Neutral:** Rose from 74.8% to **77.4%** (+2.6%), driving the higher accuracy.
* **Joy:** Rose from 53.2% to **56.8%** (+3.6%).
* **Anger:** Dropped from 42.1% to **37.9%** (-4.2%).
* **Fear:** Dropped from 18.5% to **12.1%** (-6.4%).
* **Disgust:** Dropped from 26.1% to **23.0%** (-3.1%).

---

## 6. Artifact Provenance and Verification

All artifacts for this experiment are saved and accessible:
* **Checkpoints:**
  - Best Checkpoint: `checkpoints/distilbert_5epochs/best/`
  - Per-Epoch Checkpoints: `checkpoints/distilbert_5epochs/epoch_1/` through `epoch_5/`
  - Resume State: `checkpoints/distilbert_5epochs/latest_training_state.pt`
* **Outputs:**
  - Training History: `outputs/experiments/distilbert_5epochs/history.json`
  - Step Logs: `outputs/experiments/distilbert_5epochs/training_log.jsonl`
  - Full Predictions: `outputs/experiments/distilbert_5epochs/predictions.csv`
  - Full Test Metrics: `outputs/experiments/distilbert_5epochs/metrics.json`
  - Cohort A Metrics: `outputs/experiments/distilbert_5epochs/cohort_a_metrics.json`
  - Direct Comparison: `outputs/experiments/distilbert_5epochs/comparison_vs_3epochs.json`

The original 3-epoch baseline at `checkpoints/transformer_baseline/best` remains 100% untouched.

---
*Report generated and verified against experimental output artifacts.*
