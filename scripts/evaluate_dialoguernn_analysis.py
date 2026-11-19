#!/usr/bin/env python3
"""
Comprehensive Evaluation, Statistical Comparison, and Error Analysis Script
for DialogueRNN against Baseline and Context-1 Models on MELD.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MELD_TEST_PATH = PROJECT_ROOT / "data" / "raw" / "meld" / "test_sent_emo.csv"
LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
LABEL_TO_ID = {l: i for i, l in enumerate(LABELS)}


def run_mcnemar_test(y_true, y_pred_a, y_pred_b):
    correct_a = (np.array(y_pred_a) == np.array(y_true))
    correct_b = (np.array(y_pred_b) == np.array(y_true))

    b = int(np.sum(correct_a & (~correct_b)))  # Model A correct, Model B incorrect
    c = int(np.sum((~correct_a) & correct_b))  # Model A incorrect, Model B correct
    n_discordant = b + c

    if n_discordant == 0:
        return {"b": 0, "c": 0, "n_discordant": 0, "chi2": 0.0, "p_value": 1.0, "binom_p": 1.0}

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


def fast_macro_f1(y_t_ids, y_p_ids, num_classes=7):
    f1s = []
    for c in range(num_classes):
        tp = np.sum((y_t_ids == c) & (y_p_ids == c))
        denom = np.sum(y_p_ids == c) + np.sum(y_t_ids == c)
        f1s.append(2.0 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s))


def paired_bootstrap_macro_f1(y_true, y_pred_a, y_pred_b, n_bootstraps=10000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    yt_ids = np.array([LABEL_TO_ID[l] for l in y_true])
    ypa_ids = np.array([LABEL_TO_ID[l] for l in y_pred_a])
    ypb_ids = np.array([LABEL_TO_ID[l] for l in y_pred_b])

    orig_f1_a = fast_macro_f1(yt_ids, ypa_ids)
    orig_f1_b = fast_macro_f1(yt_ids, ypb_ids)
    orig_diff = orig_f1_a - orig_f1_b

    diffs = []
    indices = np.arange(n)
    for _ in range(n_bootstraps):
        boot_idx = rng.choice(indices, size=n, replace=True)
        boot_true = yt_ids[boot_idx]
        boot_a = ypa_ids[boot_idx]
        boot_b = ypb_ids[boot_idx]
        f1_a = fast_macro_f1(boot_true, boot_a)
        f1_b = fast_macro_f1(boot_true, boot_b)
        diffs.append(f1_a - f1_b)

    diffs = np.array(diffs)
    ci_lower = float(np.percentile(diffs, 2.5))
    ci_upper = float(np.percentile(diffs, 97.5))
    p_val = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    p_val = min(1.0, max(1.0 / n_bootstraps, p_val))

    return {
        "diff": float(orig_diff),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": float(p_val),
    }


def main():
    print("=" * 80)
    print("TASK 8: DIALOGUERNN CANONICAL EVALUATION & ERROR ANALYSIS")
    print("=" * 80)

    # 1. Load MELD raw test data
    test_raw = pd.read_csv(MELD_TEST_PATH)
    test_raw["dialogue_id"] = test_raw["Dialogue_ID"].astype(int)
    test_raw["utterance_id"] = test_raw["Utterance_ID"].astype(int)
    test_raw["emotion"] = test_raw["Emotion"].str.lower().str.strip()
    test_raw["speaker"] = test_raw["Speaker"].str.strip()
    test_raw["utterance"] = test_raw["Utterance"].astype(str)

    ground_truth_map = {(r["dialogue_id"], r["utterance_id"]): r["emotion"] for _, r in test_raw.iterrows()}
    speaker_map = {(r["dialogue_id"], r["utterance_id"]): r["speaker"] for _, r in test_raw.iterrows()}
    text_map = {(r["dialogue_id"], r["utterance_id"]): r["utterance"] for _, r in test_raw.iterrows()}

    # Construct Cohorts exactly matching Task 3
    cohort_d_keys = [(r["dialogue_id"], r["utterance_id"]) for _, r in test_raw.iterrows()]
    cohort_c_keys = []
    cohort_b_keys = []
    cohort_a_keys = []

    # Map previous turn information for transition dynamics
    prev_turn_map = {}

    for d_id, group in test_raw.groupby("dialogue_id", sort=False):
        group = group.sort_values("utterance_id").reset_index(drop=True)
        for i, row in group.iterrows():
            key = (row["dialogue_id"], row["utterance_id"])
            if i == 0:
                cohort_c_keys.append(key)
                prev_turn_map[key] = None
            else:
                cohort_b_keys.append(key)
                prev_row = group.iloc[i - 1]
                prev_turn_map[key] = {
                    "dialogue_id": int(prev_row["dialogue_id"]),
                    "utterance_id": int(prev_row["utterance_id"]),
                    "speaker": prev_row["speaker"],
                    "emotion": prev_row["emotion"],
                    "utterance": prev_row["utterance"],
                    "contiguous": (row["utterance_id"] == prev_row["utterance_id"] + 1),
                }
                if row["utterance_id"] == prev_row["utterance_id"] + 1:
                    cohort_a_keys.append(key)

    cohort_keys_dict = {
        "Cohort A: Strict Contiguous Matched Subset": cohort_a_keys,
        "Cohort B: All Subsequent Sequential Turns": cohort_b_keys,
        "Cohort C: Dialogue Initial Turns (t=0)": cohort_c_keys,
        "Cohort D: Full Test Set": cohort_d_keys,
    }

    assert len(cohort_a_keys) == 2328, f"Cohort A expected 2328, got {len(cohort_a_keys)}"
    assert len(cohort_b_keys) == 2330, f"Cohort B expected 2330, got {len(cohort_b_keys)}"
    assert len(cohort_c_keys) == 280, f"Cohort C expected 280, got {len(cohort_c_keys)}"
    assert len(cohort_d_keys) == 2610, f"Cohort D expected 2610, got {len(cohort_d_keys)}"

    # 2. Load model predictions
    model_paths = {
        "DistilBERT Utterance Baseline": PROJECT_ROOT / "outputs/experiments/transformer_baseline/predictions.csv",
        "DistilBERT Context-1": PROJECT_ROOT / "outputs/experiments/transformer_context_1/predictions.csv",
        "DistilBERT Context-1 + Speaker": PROJECT_ROOT / "outputs/experiments/transformer_context_1_speaker/predictions.csv",
        "Model A: Emotion-History DistilBERT (Autoregressive)": PROJECT_ROOT / "outputs/experiments/transformer_emotion_history/predictions_autoregressive.csv",
        "Model B: Combined-History DistilBERT (Autoregressive)": PROJECT_ROOT / "outputs/experiments/transformer_combined_history/predictions_autoregressive.csv",
        "DialogueRNN (Unweighted)": PROJECT_ROOT / "outputs/experiments/dialoguernn_unweighted/test_predictions.csv",
        "DialogueRNN (Class-Weighted)": PROJECT_ROOT / "outputs/experiments/dialoguernn_weighted/test_predictions.csv",
    }

    loaded_preds = {}
    for name, path in model_paths.items():
        assert path.exists(), f"Missing prediction file: {path}"
        df = pd.read_csv(path)
        col = "predicted_emotion" if "predicted_emotion" in df.columns else "pred_emotion"
        df["dialogue_id"] = df["dialogue_id"].astype(int)
        df["utterance_id"] = df["utterance_id"].astype(int)
        p_map = {(int(r["dialogue_id"]), int(r["utterance_id"])): str(r[col]).lower().strip() for _, r in df.iterrows()}
        loaded_preds[name] = p_map
        print(f"Loaded {name} (N={len(p_map)} turns)")

    # 3. Compute Metrics for each Model and Cohort
    comparison_rows = []
    per_class_rows = []
    confusion_matrices = {}

    models_to_evaluate = [
        # Model Name, Architecture Category, Deployable?
        ("DistilBERT Utterance Baseline", "Independent Utterance", True),
        ("DistilBERT Context-1", "Text Context (t-1)", True),
        ("DistilBERT Context-1 + Speaker", "Text Context + Speaker Token", True),
        ("Model A: Emotion-History DistilBERT (Autoregressive)", "Autoregressive Emotion History", True),
        ("Model B: Combined-History DistilBERT (Autoregressive)", "Autoregressive Text + Emotion", True),
        ("DialogueRNN (Unweighted)", "Recurrent Speaker State Tracking", True),
        ("DialogueRNN (Class-Weighted)", "Recurrent Speaker Tracking + Weighted Loss", True),
    ]

    for model_name, arch_desc, is_deployable in models_to_evaluate:
        p_map = loaded_preds[model_name]

        # Evaluate on each applicable cohort
        for cohort_name, keys in cohort_keys_dict.items():
            # Check if all keys exist in model predictions
            if not all(k in p_map for k in keys):
                continue

            y_true = [ground_truth_map[k] for k in keys]
            y_pred = [p_map[k] for k in keys]

            acc = accuracy_score(y_true, y_pred)
            macro_f1 = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
            weighted_f1 = f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)

            p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(
                y_true, y_pred, labels=LABELS, zero_division=0
            )
            cm = confusion_matrix(y_true, y_pred, labels=LABELS)

            per_class_dict = {
                l: {"p": float(p_per[i]), "r": float(r_per[i]), "f1": float(f1_per[i]), "sup": int(sup_per[i])}
                for i, l in enumerate(LABELS)
            }

            pred_counts = Counter(y_pred)
            neutral_pred_pct = (pred_counts.get("neutral", 0) / len(y_pred)) * 100

            comparison_rows.append({
                "Model": model_name,
                "Architecture": arch_desc,
                "Cohort": cohort_name,
                "N": len(keys),
                "Accuracy": acc,
                "Macro_F1": macro_f1,
                "Weighted_F1": weighted_f1,
                "Neutral_Pred_Pct": neutral_pred_pct,
                "Neutral_Recall": per_class_dict["neutral"]["r"],
                "Fear_Recall": per_class_dict["fear"]["r"],
                "Disgust_Recall": per_class_dict["disgust"]["r"],
                "Anger_Recall": per_class_dict["anger"]["r"],
                "Joy_Recall": per_class_dict["joy"]["r"],
                "Sadness_Recall": per_class_dict["sadness"]["r"],
                "Surprise_Recall": per_class_dict["surprise"]["r"],
                "Fear_F1": per_class_dict["fear"]["f1"],
                "Disgust_F1": per_class_dict["disgust"]["f1"],
            })

            for l in LABELS:
                per_class_rows.append({
                    "Model": model_name,
                    "Cohort": cohort_name,
                    "Emotion": l,
                    "Precision": per_class_dict[l]["p"],
                    "Recall": per_class_dict[l]["r"],
                    "F1": per_class_dict[l]["f1"],
                    "Support": per_class_dict[l]["sup"],
                })

            confusion_matrices[f"{model_name} | {cohort_name}"] = cm.tolist()

    comp_df = pd.DataFrame(comparison_rows)
    per_class_df = pd.DataFrame(per_class_rows)

    # Save deliverables
    tables_dir = PROJECT_ROOT / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    comp_df.to_csv(tables_dir / "dialoguernn_comparison.csv", index=False)
    per_class_df.to_csv(tables_dir / "dialoguernn_per_class.csv", index=False)
    print(f"Saved {tables_dir / 'dialoguernn_comparison.csv'}")
    print(f"Saved {tables_dir / 'dialoguernn_per_class.csv'}")

    # 4. Statistical Tests on Cohort A (N=2,328)
    print("\n--- Running Statistical Significance Tests (Cohort A, N=2328) ---")
    y_true_a = [ground_truth_map[k] for k in cohort_a_keys]
    y_drnn_unw = [loaded_preds["DialogueRNN (Unweighted)"][k] for k in cohort_a_keys]
    y_drnn_wtd = [loaded_preds["DialogueRNN (Class-Weighted)"][k] for k in cohort_a_keys]

    ref_models = [
        "DistilBERT Utterance Baseline",
        "DistilBERT Context-1",
        "DistilBERT Context-1 + Speaker",
        "Model A: Emotion-History DistilBERT (Autoregressive)",
        "Model B: Combined-History DistilBERT (Autoregressive)",
        "DialogueRNN (Class-Weighted)",
    ]

    stat_records = []
    for ref_name in ref_models:
        y_ref = [loaded_preds[ref_name][k] for k in cohort_a_keys]
        mcnemar = run_mcnemar_test(y_true_a, y_drnn_unw, y_ref)
        bootstrap = paired_bootstrap_macro_f1(y_true_a, y_drnn_unw, y_ref, n_bootstraps=10000, seed=42)

        stat_records.append({
            "Comparison": f"DialogueRNN (Unweighted) vs {ref_name}",
            "N": len(cohort_a_keys),
            "McNemar_b": mcnemar["b"],
            "McNemar_c": mcnemar["c"],
            "McNemar_chi2": mcnemar["chi2"],
            "McNemar_p": mcnemar["p_value"],
            "Delta_Macro_F1": bootstrap["diff"],
            "Bootstrap_CI_Lower": bootstrap["ci_lower"],
            "Bootstrap_CI_Upper": bootstrap["ci_upper"],
            "Bootstrap_p": bootstrap["p_value"],
        })
        print(f"vs {ref_name:<45} | McNemar p={mcnemar['p_value']:.4f} | Delta F1={bootstrap['diff']:+.4f} (95% CI: [{bootstrap['ci_lower']:+.4f}, {bootstrap['ci_upper']:+.4f}], p={bootstrap['p_value']:.4f})")

    stat_df = pd.DataFrame(stat_records)
    stat_df.to_csv(tables_dir / "dialoguernn_statistical_tests.csv", index=False)

    # 5. Transition Dynamics Performance on Cohort A
    print("\n--- Evaluating Transition Dynamics on Cohort A (N=2328) ---")
    transition_records = []
    for k in cohort_a_keys:
        pt = prev_turn_map[k]
        is_shift = (ground_truth_map[k] != pt["emotion"])
        is_spk_change = (speaker_map[k] != pt["speaker"])
        transition_records.append({
            "key": k,
            "dialogue_id": k[0],
            "utterance_id": k[1],
            "speaker": speaker_map[k],
            "utterance": text_map[k],
            "true_emotion": ground_truth_map[k],
            "prev_speaker": pt["speaker"],
            "prev_emotion": pt["emotion"],
            "prev_utterance": pt["utterance"],
            "is_shift": is_shift,
            "is_spk_change": is_spk_change,
        })
    trans_df = pd.DataFrame(transition_records)

    trans_eval_rows = []
    for model_name, _, _ in models_to_evaluate:
        p_map = loaded_preds[model_name]
        preds = [p_map[k] for k in cohort_a_keys]
        trans_df[model_name] = preds

        y_true = trans_df["true_emotion"]
        y_pred = trans_df[model_name]

        # Shift turns
        m_shift = trans_df["is_shift"]
        acc_shift = accuracy_score(y_true[m_shift], y_pred[m_shift])
        macro_shift = f1_score(y_true[m_shift], y_pred[m_shift], labels=LABELS, average="macro", zero_division=0)

        # Inertia turns
        acc_inertia = accuracy_score(y_true[~m_shift], y_pred[~m_shift])
        macro_inertia = f1_score(y_true[~m_shift], y_pred[~m_shift], labels=LABELS, average="macro", zero_division=0)

        # Speaker change turns
        m_spk = trans_df["is_spk_change"]
        acc_spk = accuracy_score(y_true[m_spk], y_pred[m_spk])
        macro_spk = f1_score(y_true[m_spk], y_pred[m_spk], labels=LABELS, average="macro", zero_division=0)

        # Same speaker turns
        acc_same = accuracy_score(y_true[~m_spk], y_pred[~m_spk])
        macro_same = f1_score(y_true[~m_spk], y_pred[~m_spk], labels=LABELS, average="macro", zero_division=0)

        trans_eval_rows.append({
            "Model": model_name,
            "Shift_Acc": acc_shift,
            "Shift_Macro_F1": macro_shift,
            "Inertia_Acc": acc_inertia,
            "Inertia_Macro_F1": macro_inertia,
            "Spk_Change_Acc": acc_spk,
            "Spk_Change_Macro_F1": macro_spk,
            "Same_Spk_Acc": acc_same,
            "Same_Spk_Macro_F1": macro_same,
        })

    trans_eval_df = pd.DataFrame(trans_eval_rows)
    trans_eval_df.to_csv(tables_dir / "dialoguernn_transition_metrics.csv", index=False)
    print(f"Saved {tables_dir / 'dialoguernn_transition_metrics.csv'}")

    # 6. Extract Systematic Representative Error Cases
    print("\n--- Extracting Systematic Representative Error Cases ---")
    error_cases = {}

    # Category 1: Neutral vs Non-Neutral Confusion
    # (True is non-neutral, but Context-1 predicted Neutral, while DialogueRNN predicted correctly OR vice versa)
    neut_recovered = trans_df[
        (trans_df["true_emotion"] != "neutral") &
        (trans_df["DistilBERT Context-1"] == "neutral") &
        (trans_df["DialogueRNN (Unweighted)"] == trans_df["true_emotion"])
    ]
    error_cases["neutral_gravitation_recovered"] = neut_recovered.head(3).to_dict(orient="records")

    neut_false_gravitation = trans_df[
        (trans_df["true_emotion"] != "neutral") &
        (trans_df["DialogueRNN (Unweighted)"] == "neutral")
    ]
    error_cases["neutral_gravitation_failures"] = neut_false_gravitation.head(3).to_dict(orient="records")

    # Category 2: Minority Emotion Suppression (Fear & Disgust)
    fear_cases = trans_df[
        (trans_df["true_emotion"] == "fear") &
        (trans_df["DialogueRNN (Unweighted)"] == "fear") &
        (trans_df["DistilBERT Context-1"] != "fear")
    ]
    error_cases["fear_recovered"] = fear_cases.head(3).to_dict(orient="records")

    disgust_cases = trans_df[
        (trans_df["true_emotion"] == "disgust") &
        (trans_df["DialogueRNN (Unweighted)"] == "disgust") &
        (trans_df["DistilBERT Context-1"] != "disgust")
    ]
    error_cases["disgust_recovered"] = disgust_cases.head(3).to_dict(orient="records")

    # Category 3: Emotion Persistence (True Inertia)
    inertia_correct = trans_df[
        (~trans_df["is_shift"]) &
        (trans_df["true_emotion"] != "neutral") &
        (trans_df["DialogueRNN (Unweighted)"] == trans_df["true_emotion"]) &
        (trans_df["DistilBERT Utterance Baseline"] != trans_df["true_emotion"])
    ]
    error_cases["emotion_persistence_recovered"] = inertia_correct.head(3).to_dict(orient="records")

    # Category 4: Emotion Shifts (Turning Points)
    shift_correct = trans_df[
        (trans_df["is_shift"]) &
        (trans_df["true_emotion"] != "neutral") &
        (trans_df["DialogueRNN (Unweighted)"] == trans_df["true_emotion"]) &
        (trans_df["DistilBERT Context-1"] == trans_df["prev_emotion"])  # Context-1 trapped in inertia
    ]
    error_cases["emotion_shift_turning_points"] = shift_correct.head(3).to_dict(orient="records")

    # Category 5: Cross-Speaker Transitions
    cross_spk_cases = trans_df[
        (trans_df["is_spk_change"]) &
        (trans_df["true_emotion"] != "neutral") &
        (trans_df["DialogueRNN (Unweighted)"] == trans_df["true_emotion"]) &
        (trans_df["DistilBERT Context-1"] != trans_df["true_emotion"])
    ]
    error_cases["cross_speaker_transitions"] = cross_spk_cases.head(3).to_dict(orient="records")

    # Category 6: Ambiguous Short Utterances
    short_utterances = trans_df[
        (trans_df["utterance"].str.split().str.len() <= 3) &
        (trans_df["true_emotion"] != "neutral")
    ]
    error_cases["ambiguous_short_utterances"] = short_utterances.head(4).to_dict(orient="records")

    # Category 7: Context-Dependent Emotion
    context_dependent = trans_df[
        (trans_df["true_emotion"] != "neutral") &
        (trans_df["DialogueRNN (Unweighted)"] == trans_df["true_emotion"]) &
        (trans_df["DistilBERT Utterance Baseline"] != trans_df["true_emotion"])
    ]
    error_cases["context_dependent_success"] = context_dependent.head(3).to_dict(orient="records")

    # Category 8: Missing Utterance ID Gaps (Dialogue 93 turn 8, Dialogue 108 turn 3)
    gap_cases = []
    for d_id, u_id in [(93, 8), (108, 3)]:
        row = test_raw[(test_raw["dialogue_id"] == d_id) & (test_raw["utterance_id"] == u_id)].iloc[0]
        gap_cases.append({
            "dialogue_id": d_id,
            "utterance_id": u_id,
            "speaker": row["speaker"],
            "utterance": row["utterance"],
            "true_emotion": row["emotion"],
            "DialogueRNN (Unweighted)": loaded_preds["DialogueRNN (Unweighted)"].get((d_id, u_id), "N/A"),
            "DialogueRNN (Class-Weighted)": loaded_preds["DialogueRNN (Class-Weighted)"].get((d_id, u_id), "N/A"),
            "DistilBERT Utterance Baseline": loaded_preds["DistilBERT Utterance Baseline"].get((d_id, u_id), "N/A"),
            "DistilBERT Context-1": loaded_preds["DistilBERT Context-1"].get((d_id, u_id), "OMITTED_BY_GAP"),
        })
    error_cases["gap_turns"] = gap_cases

    # Save error cases JSON for report generation
    with open(tables_dir / "dialoguernn_error_cases.json", "w") as f:
        json.dump(error_cases, f, indent=2)
    print(f"Saved {tables_dir / 'dialoguernn_error_cases.json'}")

    # 7. Generate Multi-Panel Confusion Matrix Figure
    print("\n--- Generating Confusion Matrix Figure ---")
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    cm_models = [
        ("DistilBERT Utterance Baseline", "Baseline (Utterance-only)"),
        ("DistilBERT Context-1", "Context-1 (Model B)"),
        ("DialogueRNN (Unweighted)", "DialogueRNN (Unweighted)"),
        ("DialogueRNN (Class-Weighted)", "DialogueRNN (Class-Weighted)"),
    ]

    for ax, (m_key, title) in zip(axes, cm_models):
        cm = np.array(confusion_matrices[f"{m_key} | Cohort A: Strict Contiguous Matched Subset"])
        # Normalize by row (true label) to show recall proportions
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        tick_marks = np.arange(len(LABELS))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=11)
        ax.set_yticklabels(LABELS, fontsize=11)
        ax.set_xlabel("Predicted Emotion", fontsize=12, labelpad=8)
        ax.set_ylabel("True Emotion", fontsize=12, labelpad=8)

        # Annotate cells with percentages
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                val = cm_norm[i, j]
                text_color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val*100:.1f}%\n({cm[i, j]})", ha="center", va="center", color=text_color, fontsize=8)

    plt.tight_layout()
    fig_dir = PROJECT_ROOT / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "dialoguernn_confusion_matrix.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {fig_path}")

    print("\nALL EVALUATION TASKS COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
