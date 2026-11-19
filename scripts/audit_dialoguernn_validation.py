#!/usr/bin/env python3
"""
Independent DialogueRNN Results Validation & Statistical Audit Script
Task 11 — emotion-dynamics-nlp repository audit.

Performs:
1. Exhaustive prediction and cohort integrity verification against raw MELD.
2. Independent recalculation of all benchmark metrics and per-class metrics.
3. Dialogue-level paired cluster bootstrap (10,000 resamples) preserving within-dialogue clusters.
4. Comparison of utterance-level vs dialogue-level bootstrap and Holm-Bonferroni correction.
5. Saves machine-readable audit artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MELD_TEST_PATH = PROJECT_ROOT / "data" / "raw" / "meld" / "test_sent_emo.csv"
MELD_TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "meld" / "train_sent_emo.csv"
MELD_DEV_PATH = PROJECT_ROOT / "data" / "raw" / "meld" / "dev_sent_emo.csv"

LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
LABEL_TO_ID = {l: i for i, l in enumerate(LABELS)}
ID_TO_LABEL = {i: l for i, l in enumerate(LABELS)}


def run_mcnemar_test(y_true, y_pred_a, y_pred_b):
    """
    McNemar test of classification accuracy on paired instances.
    Evaluates marginal homogeneity of 0/1 correctness.
    Note: Does NOT test Macro F1!
    """
    correct_a = (np.array(y_pred_a) == np.array(y_true))
    correct_b = (np.array(y_pred_b) == np.array(y_true))

    b = int(np.sum(correct_a & (~correct_b)))  # Model A correct, Model B incorrect
    c = int(np.sum((~correct_a) & correct_b))  # Model A incorrect, Model B correct
    n_discordant = b + c

    if n_discordant == 0:
        return {"b": 0, "c": 0, "n_discordant": 0, "chi2": 0.0, "p_value": 1.0, "binom_p": 1.0}

    # Edwards continuity correction: (|b - c| - 1)^2 / (b + c)
    chi2 = ((abs(b - c) - 1.0) ** 2) / n_discordant
    p_val = float(stats.chi2.sf(chi2, df=1))
    binom_res = stats.binomtest(b, n_discordant, 0.5, alternative="two-sided")
    binom_p = float(binom_res.pvalue)

    return {
        "b": b,
        "c": c,
        "n_discordant": n_discordant,
        "chi2": float(chi2),
        "p_value": p_val,
        "binom_p": binom_p,
    }


def fast_macro_f1(y_t_ids: np.ndarray, y_p_ids: np.ndarray, num_classes: int = 7) -> float:
    """Fast calculation of Macro F1 from class ID integer arrays."""
    f1s = []
    for c in range(num_classes):
        tp = np.sum((y_t_ids == c) & (y_p_ids == c))
        denom = np.sum(y_p_ids == c) + np.sum(y_t_ids == c)
        f1s.append(2.0 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s))


def holm_bonferroni(p_values: List[float]) -> List[float]:
    """
    Apply Holm-Bonferroni step-down correction to a list of p-values.
    Returns adjusted p-values in original list order.
    """
    p_arr = np.array(p_values, dtype=float)
    n = len(p_arr)
    sorted_indices = np.argsort(p_arr)
    sorted_p = p_arr[sorted_indices]

    adj_p = np.zeros(n)
    running_max = 0.0
    for i in range(n):
        multiplier = n - i
        val = min(1.0, sorted_p[i] * multiplier)
        running_max = max(running_max, val)
        adj_p[i] = min(1.0, running_max)

    out = np.zeros(n)
    out[sorted_indices] = adj_p
    return out.tolist()


def main():
    print("=" * 80)
    print("TASK 11: INDEPENDENT DIALOGUERNN RESULTS VALIDATION AUDIT")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. Prediction and Cohort Integrity Audit
    # ---------------------------------------------------------
    print("\n--- 1. AUDITING PREDICTION AND COHORT INTEGRITY ---")
    test_raw = pd.read_csv(MELD_TEST_PATH)
    assert len(test_raw) == 2610, f"Expected 2610 test rows, got {len(test_raw)}"
    test_raw["dialogue_id"] = test_raw["Dialogue_ID"].astype(int)
    test_raw["utterance_id"] = test_raw["Utterance_ID"].astype(int)
    test_raw["emotion"] = test_raw["Emotion"].str.lower().str.strip()

    raw_gt_map = {(int(r["dialogue_id"]), int(r["utterance_id"])): r["emotion"] for _, r in test_raw.iterrows()}
    assert len(raw_gt_map) == 2610, "Duplicate keys found in raw test CSV!"

    # Identify all utterance ID gaps across test dialogues
    identified_gaps = []
    cohort_d_keys = []
    cohort_c_keys = []
    cohort_b_keys = []
    cohort_a_keys = []

    for d_id, group in test_raw.groupby("dialogue_id", sort=False):
        group = group.sort_values("utterance_id").reset_index(drop=True)
        for i in range(len(group)):
            key = (int(d_id), int(group.loc[i, "utterance_id"]))
            cohort_d_keys.append(key)
            if i == 0:
                cohort_c_keys.append(key)
            else:
                cohort_b_keys.append(key)
                curr_u = group.loc[i, "utterance_id"]
                prev_u = group.loc[i - 1, "utterance_id"]
                if curr_u == prev_u + 1:
                    cohort_a_keys.append(key)
                else:
                    identified_gaps.append({
                        "dialogue_id": int(d_id),
                        "prev_utterance_id": int(prev_u),
                        "curr_utterance_id": int(curr_u),
                        "missing_turns": list(range(prev_u + 1, curr_u)),
                    })

    print(f"Identified Utterance Gaps in Test Set: {len(identified_gaps)}")
    for g in identified_gaps:
        print(f"  Dialogue {g['dialogue_id']}: Utterance_ID jumps from {g['prev_utterance_id']} to {g['curr_utterance_id']} (missing: {g['missing_turns']})")

    assert len(cohort_a_keys) == 2328, f"Cohort A expected 2328, got {len(cohort_a_keys)}"
    assert len(cohort_b_keys) == 2330, f"Cohort B expected 2330, got {len(cohort_b_keys)}"
    assert len(cohort_c_keys) == 280, f"Cohort C expected 280, got {len(cohort_c_keys)}"
    assert len(cohort_d_keys) == 2610, f"Cohort D expected 2610, got {len(cohort_d_keys)}"

    # Check that gap turns (93, 8) and (108, 3) are strictly excluded from Cohort A
    assert (93, 8) not in set(cohort_a_keys), "Gap turn (93, 8) erroneously in Cohort A!"
    assert (108, 3) not in set(cohort_a_keys), "Gap turn (108, 3) erroneously in Cohort A!"
    assert (93, 8) in set(cohort_b_keys), "Gap turn (93, 8) missing from Cohort B!"
    assert (108, 3) in set(cohort_b_keys), "Gap turn (108, 3) missing from Cohort B!"

    # Inspect all model prediction files
    model_spec = {
        "DistilBERT Utterance Baseline": {
            "path": PROJECT_ROOT / "outputs/experiments/transformer_baseline/predictions.csv",
            "pcol": "predicted_emotion",
            "tcol": "emotion",
            "expected_rows": 2610,
        },
        "DistilBERT Context-1": {
            "path": PROJECT_ROOT / "outputs/experiments/transformer_context_1/predictions.csv",
            "pcol": "predicted_emotion",
            "tcol": "true_emotion",
            "expected_rows": 2328,
        },
        "DistilBERT Context-1 + Speaker": {
            "path": PROJECT_ROOT / "outputs/experiments/transformer_context_1_speaker/predictions.csv",
            "pcol": "predicted_emotion",
            "tcol": "true_emotion",
            "expected_rows": 2328,
        },
        "Model A: Emotion-History DistilBERT (Autoregressive)": {
            "path": PROJECT_ROOT / "outputs/experiments/transformer_emotion_history/predictions_autoregressive.csv",
            "pcol": "predicted_emotion",
            "tcol": "true_emotion",
            "expected_rows": 2610,
        },
        "Model B: Combined-History DistilBERT (Autoregressive)": {
            "path": PROJECT_ROOT / "outputs/experiments/transformer_combined_history/predictions_autoregressive.csv",
            "pcol": "predicted_emotion",
            "tcol": "true_emotion",
            "expected_rows": 2610,
        },
        "DialogueRNN (Unweighted)": {
            "path": PROJECT_ROOT / "outputs/experiments/dialoguernn_unweighted/test_predictions.csv",
            "pcol": "pred_emotion",
            "tcol": "true_emotion",
            "expected_rows": 2610,
        },
        "DialogueRNN (Class-Weighted)": {
            "path": PROJECT_ROOT / "outputs/experiments/dialoguernn_weighted/test_predictions.csv",
            "pcol": "pred_emotion",
            "tcol": "true_emotion",
            "expected_rows": 2610,
        },
    }

    prediction_integrity_report = {}
    loaded_predictions = {}
    cohort_a_set = set(cohort_a_keys)

    for m_name, spec in model_spec.items():
        p_path = spec["path"]
        assert p_path.exists(), f"Prediction file not found: {p_path}"
        df = pd.read_csv(p_path)
        d_col = "dialogue_id" if "dialogue_id" in df.columns else "Dialogue_ID"
        u_col = "utterance_id" if "utterance_id" in df.columns else "Utterance_ID"

        keys = list(zip(df[d_col].astype(int), df[u_col].astype(int)))
        unique_keys = set(keys)

        # Check duplicates
        n_duplicates = len(keys) - len(unique_keys)
        assert n_duplicates == 0, f"{m_name} contains {n_duplicates} duplicate prediction keys!"

        # Check row count
        assert len(df) == spec["expected_rows"], f"{m_name} expected {spec['expected_rows']} rows, got {len(df)}"

        # Check ground truth match
        mismatches = 0
        if spec["tcol"] in df.columns:
            for k, val in zip(keys, df[spec["tcol"]]):
                if str(val).lower().strip() != raw_gt_map[k]:
                    mismatches += 1
        assert mismatches == 0, f"{m_name} has {mismatches} ground-truth label discrepancies!"

        # Check key membership vs raw test
        in_raw_count = sum(k in raw_gt_map for k in keys)
        assert in_raw_count == len(keys), f"{m_name} has keys not present in raw test!"

        # Extract predictions
        p_map = {k: str(v).lower().strip() for k, v in zip(keys, df[spec["pcol"]])}
        loaded_predictions[m_name] = p_map

        # Check Cohort A alignment
        if len(keys) == 2328:
            seq_match = (keys == cohort_a_keys)
        else:
            filtered_keys = [k for k in keys if k in cohort_a_set]
            seq_match = (filtered_keys == cohort_a_keys)

        assert seq_match, f"{m_name} does not match Cohort A key sequence ordering exactly!"

        prediction_integrity_report[m_name] = {
            "path": str(p_path.relative_to(PROJECT_ROOT)),
            "row_count": len(df),
            "expected_rows": spec["expected_rows"],
            "unique_keys": len(unique_keys),
            "duplicate_keys": n_duplicates,
            "ground_truth_mismatches": mismatches,
            "cohort_a_key_exact_sequence_match": seq_match,
        }
        print(f"  Verified {m_name:<50}: {len(df)} rows, 0 duplicates, 0 GT mismatches, exact Cohort A sequence match.")

    # Save integrity report
    tables_dir = PROJECT_ROOT / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    with open(tables_dir / "validation_prediction_integrity.json", "w") as f:
        json.dump(prediction_integrity_report, f, indent=2)

    # ---------------------------------------------------------
    # 2. Metric Verification & Recalculation
    # ---------------------------------------------------------
    print("\n--- 2. METRIC VERIFICATION & RECALCULATION ---")
    cohorts_dict = {
        "Cohort A: Strict Contiguous Matched Subset": cohort_a_keys,
        "Cohort B: All Subsequent Sequential Turns": cohort_b_keys,
        "Cohort C: Dialogue Initial Turns (t=0)": cohort_c_keys,
        "Cohort D: Full Test Set": cohort_d_keys,
    }

    recalculated_comp_rows = []
    recalculated_per_class_rows = []

    for m_name in model_spec:
        p_map = loaded_predictions[m_name]
        for c_name, keys in cohorts_dict.items():
            if not all(k in p_map for k in keys):
                continue

            y_true = [raw_gt_map[k] for k in keys]
            y_pred = [p_map[k] for k in keys]

            acc = accuracy_score(y_true, y_pred)
            macro_f1 = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
            weighted_f1 = f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)

            p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(
                y_true, y_pred, labels=LABELS, zero_division=0
            )

            p_dict = {l: {"p": float(p_per[i]), "r": float(r_per[i]), "f1": float(f1_per[i]), "sup": int(sup_per[i])}
                      for i, l in enumerate(LABELS)}

            pred_counts = Counter(y_pred)
            neutral_pct = (pred_counts.get("neutral", 0) / len(y_pred)) * 100

            recalculated_comp_rows.append({
                "Model": m_name,
                "Cohort": c_name,
                "N": len(keys),
                "Accuracy": float(acc),
                "Macro_F1": float(macro_f1),
                "Weighted_F1": float(weighted_f1),
                "Neutral_Pred_Pct": float(neutral_pct),
                "Neutral_Recall": p_dict["neutral"]["r"],
                "Fear_Recall": p_dict["fear"]["r"],
                "Disgust_Recall": p_dict["disgust"]["r"],
                "Anger_Recall": p_dict["anger"]["r"],
                "Joy_Recall": p_dict["joy"]["r"],
                "Sadness_Recall": p_dict["sadness"]["r"],
                "Surprise_Recall": p_dict["surprise"]["r"],
                "Fear_F1": p_dict["fear"]["f1"],
                "Disgust_F1": p_dict["disgust"]["f1"],
            })

            for l in LABELS:
                recalculated_per_class_rows.append({
                    "Model": m_name,
                    "Cohort": c_name,
                    "Emotion": l,
                    "Precision": p_dict[l]["p"],
                    "Recall": p_dict[l]["r"],
                    "F1": p_dict[l]["f1"],
                    "Support": p_dict[l]["sup"],
                })

    recalc_comp_df = pd.DataFrame(recalculated_comp_rows)
    recalc_pc_df = pd.DataFrame(recalculated_per_class_rows)

    # Compare against existing outputs/tables/dialoguernn_comparison.csv
    existing_comp_path = tables_dir / "dialoguernn_comparison.csv"
    existing_pc_path = tables_dir / "dialoguernn_per_class.csv"
    assert existing_comp_path.exists(), f"Existing comparison table not found: {existing_comp_path}"
    assert existing_pc_path.exists(), f"Existing per-class table not found: {existing_pc_path}"

    existing_comp = pd.read_csv(existing_comp_path)
    existing_pc = pd.read_csv(existing_pc_path)

    print(f"Comparing {len(recalc_comp_df)} recalculated rows against existing {len(existing_comp)} rows...")
    comp_diff_count = 0
    for _, row in recalc_comp_df.iterrows():
        match = existing_comp[(existing_comp["Model"] == row["Model"]) & (existing_comp["Cohort"] == row["Cohort"])]
        assert len(match) == 1, f"Missing row in existing comp: {row['Model']} | {row['Cohort']}"
        ex_row = match.iloc[0]
        for col in ["Accuracy", "Macro_F1", "Weighted_F1", "Neutral_Pred_Pct", "Neutral_Recall", "Fear_Recall", "Disgust_Recall"]:
            diff = abs(row[col] - ex_row[col])
            if diff > 1e-5:
                print(f"  DISCREPANCY in {row['Model']} [{row['Cohort']}] {col}: recalc={row[col]}, existing={ex_row[col]}")
                comp_diff_count += 1

    print(f"Comparison table verification: {comp_diff_count} discrepancies found (0 expected).")
    assert comp_diff_count == 0, "Comparison table has numerical discrepancies!"

    pc_diff_count = 0
    for _, row in recalc_pc_df.iterrows():
        match = existing_pc[(existing_pc["Model"] == row["Model"]) & (existing_pc["Cohort"] == row["Cohort"]) & (existing_pc["Emotion"] == row["Emotion"])]
        assert len(match) == 1
        ex_row = match.iloc[0]
        for col in ["Precision", "Recall", "F1"]:
            diff = abs(row[col] - ex_row[col])
            if diff > 1e-5:
                print(f"  DISCREPANCY in {row['Model']} [{row['Cohort']}] {row['Emotion']} {col}: recalc={row[col]}, existing={ex_row[col]}")
                pc_diff_count += 1

    print(f"Per-class table verification: {pc_diff_count} discrepancies found (0 expected).")
    assert pc_diff_count == 0, "Per-class table has numerical discrepancies!"

    recalc_comp_df.to_csv(tables_dir / "validation_recalculated_metrics.csv", index=False)
    recalc_pc_df.to_csv(tables_dir / "validation_per_class_recalculated.csv", index=False)

    # ---------------------------------------------------------
    # 3. Statistical Validation: Utterance vs Dialogue-Cluster Bootstrap
    # ---------------------------------------------------------
    print("\n--- 3. STATISTICAL VALIDATION & DIALOGUE-CLUSTER BOOTSTRAP ---")
    y_true_a = [raw_gt_map[k] for k in cohort_a_keys]
    yt_ids = np.array([LABEL_TO_ID[l] for l in y_true_a], dtype=np.int32)
    dialogue_ids_a = np.array([k[0] for k in cohort_a_keys], dtype=np.int32)

    # Map dialogue IDs to utterance indices
    d_to_indices = {}
    for idx, d_id in enumerate(dialogue_ids_a):
        d_to_indices.setdefault(int(d_id), []).append(idx)
    for d_id in d_to_indices:
        d_to_indices[d_id] = np.array(d_to_indices[d_id], dtype=np.int32)

    cohort_dialogue_ids = np.array(sorted(d_to_indices.keys()), dtype=np.int32)
    all_test_dialogue_ids = np.array(sorted(test_raw["dialogue_id"].unique()), dtype=np.int32)

    print(f"Cohort A Utterances: N={len(cohort_a_keys)}")
    print(f"Unique Dialogues with Cohort A turns: M={len(cohort_dialogue_ids)} (out of {len(all_test_dialogue_ids)} total test dialogues)")

    # Prepare integer prediction arrays for fast F1 computation
    pred_id_arrays = {}
    for m_name in model_spec:
        p_map = loaded_predictions[m_name]
        preds_a = [p_map[k] for k in cohort_a_keys]
        pred_id_arrays[m_name] = np.array([LABEL_TO_ID[p] for p in preds_a], dtype=np.int32)

    target_model = "DialogueRNN (Unweighted)"
    target_preds = pred_id_arrays[target_model]
    target_f1 = fast_macro_f1(yt_ids, target_preds)

    comparisons = [
        "DistilBERT Utterance Baseline",
        "DistilBERT Context-1",
        "DistilBERT Context-1 + Speaker",
        "Model A: Emotion-History DistilBERT (Autoregressive)",
        "Model B: Combined-History DistilBERT (Autoregressive)",
        "DialogueRNN (Class-Weighted)",
    ]

    n_bootstraps = 10000
    fixed_seed = 42

    # Pre-generate bootstrap samples to ensure EXACT same dialogue selections across all pairwise comparisons
    rng_cluster = np.random.RandomState(fixed_seed)
    sampled_cluster_indices = []
    for _ in range(n_bootstraps):
        sampled_dialogues = rng_cluster.choice(cohort_dialogue_ids, size=len(cohort_dialogue_ids), replace=True)
        idx_concat = np.concatenate([d_to_indices[d] for d in sampled_dialogues])
        sampled_cluster_indices.append(idx_concat)

    # Pre-generate utterance-level bootstrap samples for direct comparison
    rng_utt = np.random.RandomState(fixed_seed)
    sampled_utt_indices = [rng_utt.choice(len(cohort_a_keys), size=len(cohort_a_keys), replace=True) for _ in range(n_bootstraps)]

    statistical_records = []
    p_vals_cluster_for_holm = []
    p_vals_utt_for_holm = []

    for comp_model in comparisons:
        comp_preds = pred_id_arrays[comp_model]
        comp_f1 = fast_macro_f1(yt_ids, comp_preds)
        delta_macro_f1 = target_f1 - comp_f1

        # McNemar test on paired accuracy
        mcnemar = run_mcnemar_test(y_true_a, [ID_TO_LABEL[p] for p in target_preds], [ID_TO_LABEL[p] for p in comp_preds])

        # 1. Dialogue-Level Paired Cluster Bootstrap
        diffs_cluster = []
        for idx in sampled_cluster_indices:
            b_yt = yt_ids[idx]
            b_target = target_preds[idx]
            b_comp = comp_preds[idx]
            diffs_cluster.append(fast_macro_f1(b_yt, b_target) - fast_macro_f1(b_yt, b_comp))

        diffs_cluster = np.array(diffs_cluster)
        ci_lower_c = float(np.percentile(diffs_cluster, 2.5))
        ci_upper_c = float(np.percentile(diffs_cluster, 97.5))
        se_c = float(np.std(diffs_cluster, ddof=1))
        p_val_c = 2 * min(np.mean(diffs_cluster <= 0), np.mean(diffs_cluster >= 0))
        p_val_c = min(1.0, max(1.0 / n_bootstraps, p_val_c))

        # 2. Utterance-Level Paired Bootstrap
        diffs_utt = []
        for idx in sampled_utt_indices:
            b_yt = yt_ids[idx]
            b_target = target_preds[idx]
            b_comp = comp_preds[idx]
            diffs_utt.append(fast_macro_f1(b_yt, b_target) - fast_macro_f1(b_yt, b_comp))

        diffs_utt = np.array(diffs_utt)
        ci_lower_u = float(np.percentile(diffs_utt, 2.5))
        ci_upper_u = float(np.percentile(diffs_utt, 97.5))
        se_u = float(np.std(diffs_utt, ddof=1))
        p_val_u = 2 * min(np.mean(diffs_utt <= 0), np.mean(diffs_utt >= 0))
        p_val_u = min(1.0, max(1.0 / n_bootstraps, p_val_u))

        p_vals_cluster_for_holm.append(p_val_c)
        p_vals_utt_for_holm.append(p_val_u)

        statistical_records.append({
            "Comparison": f"DialogueRNN (Unweighted) vs {comp_model}",
            "Delta_Macro_F1": delta_macro_f1,
            "Cluster_CI_Lower": ci_lower_c,
            "Cluster_CI_Upper": ci_upper_c,
            "Cluster_Std_Error": se_c,
            "Cluster_p_unadjusted": p_val_c,
            "Utterance_CI_Lower": ci_lower_u,
            "Utterance_CI_Upper": ci_upper_u,
            "Utterance_Std_Error": se_u,
            "Utterance_p_unadjusted": p_val_u,
            "McNemar_b": mcnemar["b"],
            "McNemar_c": mcnemar["c"],
            "McNemar_chi2": mcnemar["chi2"],
            "McNemar_p": mcnemar["p_value"],
            "McNemar_binom_p": mcnemar["binom_p"],
        })

    # Apply Holm-Bonferroni correction
    holm_cluster_p = holm_bonferroni(p_vals_cluster_for_holm)
    holm_utt_p = holm_bonferroni(p_vals_utt_for_holm)

    for i in range(len(statistical_records)):
        statistical_records[i]["Cluster_p_Holm_adjusted"] = holm_cluster_p[i]
        statistical_records[i]["Utterance_p_Holm_adjusted"] = holm_utt_p[i]
        statistical_records[i]["Significant_Cluster_Holm_05"] = (holm_cluster_p[i] < 0.05)

    stat_df = pd.DataFrame(statistical_records)
    stat_df.to_csv(tables_dir / "validation_dialogue_cluster_bootstrap.csv", index=False)
    print(f"Saved {tables_dir / 'validation_dialogue_cluster_bootstrap.csv'}")

    print("\n" + "=" * 130)
    print(f"{'Comparison':<50} | {'Delta':<7} | {'Cluster 95% CI':<22} | {'Unadj p':<7} | {'Holm p':<7} | {'Sig (α=.05)':<11} | {'McNemar p (Acc)'}")
    print("-" * 130)
    for r in statistical_records:
        ci_str = f"[{r['Cluster_CI_Lower']:+.4f}, {r['Cluster_CI_Upper']:+.4f}]"
        sig_str = "YES" if r["Significant_Cluster_Holm_05"] else "NO"
        print(f"{r['Comparison']:<50} | {r['Delta_Macro_F1']:+.4f} | {ci_str:<22} | {r['Cluster_p_unadjusted']:.4f}  | {r['Cluster_p_Holm_adjusted']:.4f} | {sig_str:<11} | {r['McNemar_p']:.4f}")
    print("=" * 130)

    print("\nAUDIT DATA EXTRACTION AND RECALCULATION COMPLETE.")


if __name__ == "__main__":
    main()
