import json
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
LABEL_TO_ID = {l: i for i, l in enumerate(LABELS)}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MELD_TEST_PATH = PROJECT_ROOT / "data" / "raw" / "meld" / "test_sent_emo.csv"
MELD_TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "meld" / "train_sent_emo.csv"
MELD_DEV_PATH = PROJECT_ROOT / "data" / "raw" / "meld" / "dev_sent_emo.csv"

def run_mcnemar_test(y_true, y_pred_a, y_pred_b):
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
    
    # Exact two-sided binomial test
    binom_res = stats.binomtest(b, n_discordant, 0.5, alternative="two-sided")
    binom_p = float(binom_res.pvalue)

    return {
        "b": b,
        "c": c,
        "n_discordant": n_discordant,
        "chi2": float(chi2),
        "p_value": p_val,
        "binom_p": binom_p
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
    lbl_to_id = {l: i for i, l in enumerate(LABELS)}
    yt_ids = np.array([lbl_to_id[l] for l in y_true])
    ypa_ids = np.array([lbl_to_id[l] for l in y_pred_a])
    ypb_ids = np.array([lbl_to_id[l] for l in y_pred_b])

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

    # Two-sided empirical p-value under H0: delta == 0
    p_val = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    p_val = min(1.0, max(1.0 / n_bootstraps, p_val))

    return {
        "diff": float(orig_diff),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": float(p_val)
    }


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)

    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    per_class = {}
    for i, label in enumerate(LABELS):
        per_class[label] = {
            "precision": float(p_per[i]),
            "recall": float(r_per[i]),
            "f1": float(f1_per[i]),
            "support": int(sup_per[i])
        }

    pred_counts = Counter(y_pred)
    true_counts = Counter(y_true)

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "pred_distribution": {l: int(pred_counts.get(l, 0)) for l in LABELS},
        "true_distribution": {l: int(true_counts.get(l, 0)) for l in LABELS},
    }

def main():
    print("=" * 80)
    print("CANONICAL MELD EVALUATION AUDIT")
    print("=" * 80)

    test_raw = pd.read_csv(MELD_TEST_PATH)
    train_raw = pd.read_csv(MELD_TRAIN_PATH)
    dev_raw = pd.read_csv(MELD_DEV_PATH)

    test_raw["dialogue_id"] = test_raw["Dialogue_ID"]
    test_raw["utterance_id"] = test_raw["Utterance_ID"]
    test_raw["emotion"] = test_raw["Emotion"].str.lower().str.strip()

    # Ground truth mapping
    ground_truth_map = { (r["dialogue_id"], r["utterance_id"]): r["emotion"] for _, r in test_raw.iterrows() }

    # Construct Cohorts
    cohort_d_keys = [(r["dialogue_id"], r["utterance_id"]) for _, r in test_raw.iterrows()]
    cohort_c_keys = []
    cohort_b_keys = []
    cohort_a_keys = []

    for d_id, group in test_raw.groupby("dialogue_id", sort=False):
        group = group.sort_values("utterance_id").reset_index(drop=True)
        for i, row in group.iterrows():
            key = (row["dialogue_id"], row["utterance_id"])
            if i == 0:
                cohort_c_keys.append(key)
            else:
                cohort_b_keys.append(key)
                prev_row = group.iloc[i - 1]
                if row["utterance_id"] == prev_row["utterance_id"] + 1:
                    cohort_a_keys.append(key)

    cohort_keys_dict = {
        "Cohort A: Strict Contiguous Matched Subset": cohort_a_keys,
        "Cohort B: All Subsequent Sequential Turns": cohort_b_keys,
        "Cohort C: Dialogue Initial Turns (t=0)": cohort_c_keys,
        "Cohort D: Full Test Set": cohort_d_keys,
    }

    cohort_sizes = {
        "Cohort A: Strict Contiguous Matched Subset": 2328,
        "Cohort B: All Subsequent Sequential Turns": 2330,
        "Cohort C: Dialogue Initial Turns (t=0)": 280,
        "Cohort D: Full Test Set": 2610,
    }

    for name, keys in cohort_keys_dict.items():
        assert len(keys) == cohort_sizes[name], f"Size mismatch for {name}: {len(keys)} vs {cohort_sizes[name]}"

    # Load predictions
    pred_files = {
        "DistilBERT Utterance Baseline": PROJECT_ROOT / "outputs/experiments/transformer_baseline/predictions.csv",
        "DistilBERT Context-1": PROJECT_ROOT / "outputs/experiments/transformer_context_1/predictions.csv",
        "DistilBERT Context-1 + Speaker": PROJECT_ROOT / "outputs/experiments/transformer_context_1_speaker/predictions.csv",
        "DistilBERT Class-Weighted (Baseline)": PROJECT_ROOT / "outputs/experiments/transformer_weighted/predictions.csv",
        "Model A: Emotion-History DistilBERT (Autoregressive)": PROJECT_ROOT / "outputs/experiments/transformer_emotion_history/predictions_autoregressive.csv",
        "Model B: Combined-History DistilBERT (Autoregressive)": PROJECT_ROOT / "outputs/experiments/transformer_combined_history/predictions_autoregressive.csv",
        "Model B: Combined-History DistilBERT (Teacher-Forced Oracle)": PROJECT_ROOT / "outputs/experiments/transformer_combined_history/predictions_teacher_forced.csv",
    }

    loaded_preds = {}
    for name, path in pred_files.items():
        assert path.exists(), f"Missing file: {path}"
        df = pd.read_csv(path)
        df["dialogue_id"] = df["dialogue_id"].astype(int)
        df["utterance_id"] = df["utterance_id"].astype(int)
        true_col = "true_emotion" if "true_emotion" in df.columns else "emotion"
        pred_col = "predicted_emotion"

        # Verification: unique keys, no duplicate rows
        keys = list(zip(df["dialogue_id"], df["utterance_id"]))
        assert len(keys) == len(set(keys)), f"Duplicate keys in {name}"

        # Verification: true labels match MELD ground truth exactly
        for _, row in df.iterrows():
            k = (int(row["dialogue_id"]), int(row["utterance_id"]))
            gt_label = ground_truth_map[k]
            file_true = str(row[true_col]).lower().strip()
            assert gt_label == file_true, f"Label mismatch in {name} for {k}: {gt_label} vs {file_true}"

        pred_map = { (int(row["dialogue_id"]), int(row["utterance_id"])): str(row[pred_col]).lower().strip() for _, row in df.iterrows() }
        loaded_preds[name] = pred_map
        print(f"Verified {name} (N={len(pred_map)} predictions loaded).")

    # 3. Compute Metrics for each Model and applicable Cohort
    print("\n--- Computing Canonical Metrics ---")
    comparison_rows = []
    per_class_rows = []
    cm_dict = {}

    # Define model-cohort validity matrix
    model_cohort_evals = [
        # Cohort A (N=2328) - all 7 models valid
        ("DistilBERT Utterance Baseline", "Cohort A: Strict Contiguous Matched Subset", "Independent Utterance", False),
        ("DistilBERT Context-1", "Cohort A: Strict Contiguous Matched Subset", "Prior Text Context (t-1)", False),
        ("DistilBERT Context-1 + Speaker", "Cohort A: Strict Contiguous Matched Subset", "Prior Text + Speaker Token", False),
        ("DistilBERT Class-Weighted (Baseline)", "Cohort A: Strict Contiguous Matched Subset", "Class-Weighted Loss (Inv Freq)", False),
        ("Model A: Emotion-History DistilBERT (Autoregressive)", "Cohort A: Strict Contiguous Matched Subset", "Prior Predicted Emotion (Autoregressive)", False),
        ("Model B: Combined-History DistilBERT (Autoregressive)", "Cohort A: Strict Contiguous Matched Subset", "Prior Text + Predicted Emotion (Autoregressive)", False),
        ("Model B: Combined-History DistilBERT (Teacher-Forced Oracle)", "Cohort A: Strict Contiguous Matched Subset", "Prior Text + Ground-Truth Emotion (Oracle)", True),

        # Cohort B (N=2330) - only models evaluated on sequential turns
        ("DistilBERT Utterance Baseline", "Cohort B: All Subsequent Sequential Turns", "Independent Utterance", False),
        ("DistilBERT Class-Weighted (Baseline)", "Cohort B: All Subsequent Sequential Turns", "Class-Weighted Loss (Inv Freq)", False),
        ("Model A: Emotion-History DistilBERT (Autoregressive)", "Cohort B: All Subsequent Sequential Turns", "Prior Predicted Emotion (Autoregressive)", False),
        ("Model B: Combined-History DistilBERT (Autoregressive)", "Cohort B: All Subsequent Sequential Turns", "Prior Text + Predicted Emotion (Autoregressive)", False),
        ("Model B: Combined-History DistilBERT (Teacher-Forced Oracle)", "Cohort B: All Subsequent Sequential Turns", "Prior Text + Ground-Truth Emotion (Oracle)", True),

        # Cohort C: Turn 0 (N=280)
        ("DistilBERT Utterance Baseline", "Cohort C: Dialogue Initial Turns (t=0)", "Independent Utterance", False),
        ("DistilBERT Class-Weighted (Baseline)", "Cohort C: Dialogue Initial Turns (t=0)", "Class-Weighted Loss (Inv Freq)", False),
        ("Model A: Emotion-History DistilBERT (Autoregressive)", "Cohort C: Dialogue Initial Turns (t=0)", "Conditioned on 'none' history", False),
        ("Model B: Combined-History DistilBERT (Autoregressive)", "Cohort C: Dialogue Initial Turns (t=0)", "Conditioned on 'none' history", False),
        ("Model B: Combined-History DistilBERT (Teacher-Forced Oracle)", "Cohort C: Dialogue Initial Turns (t=0)", "Conditioned on 'none' history", True),

        # Cohort D: Full Test (N=2610)
        ("DistilBERT Utterance Baseline", "Cohort D: Full Test Set", "Independent Utterance", False),
        ("DistilBERT Class-Weighted (Baseline)", "Cohort D: Full Test Set", "Class-Weighted Loss (Inv Freq)", False),
        ("Model A: Emotion-History DistilBERT (Autoregressive)", "Cohort D: Full Test Set", "Autoregressive (Turn 0 + Turns >= 1)", False),
        ("Model B: Combined-History DistilBERT (Autoregressive)", "Cohort D: Full Test Set", "Autoregressive (Turn 0 + Turns >= 1)", False),
        ("Model B: Combined-History DistilBERT (Teacher-Forced Oracle)", "Cohort D: Full Test Set", "Teacher-Forced Oracle (Turn 0 + Turns >= 1)", True),
    ]

    for model_name, cohort_name, input_desc, is_oracle in model_cohort_evals:
        target_keys = cohort_keys_dict[cohort_name]
        y_true = [ground_truth_map[k] for k in target_keys]
        y_pred = [loaded_preds[model_name][k] for k in target_keys]

        metrics = compute_metrics(y_true, y_pred)
        n_examples = len(target_keys)

        comp_row = {
            "Cohort": cohort_name,
            "Sample Size (N)": n_examples,
            "Model": model_name,
            "Input Conditioning": input_desc,
            "Oracle Diagnostic": is_oracle,
            "Accuracy": metrics["accuracy"],
            "Macro F1": metrics["macro_f1"],
            "Weighted F1": metrics["weighted_f1"],
            "Neutral Pred Rate": metrics["pred_distribution"]["neutral"] / n_examples,
            "Neutral True Rate": metrics["true_distribution"]["neutral"] / n_examples,
        }
        comparison_rows.append(comp_row)

        for label in LABELS:
            pm = metrics["per_class"][label]
            per_class_rows.append({
                "Cohort": cohort_name,
                "Sample Size (N)": n_examples,
                "Model": model_name,
                "Emotion": label,
                "Precision": pm["precision"],
                "Recall": pm["recall"],
                "F1": pm["f1"],
                "True Support": pm["support"],
                "Predicted Count": metrics["pred_distribution"][label],
            })

        cm_dict[f"{model_name}___{cohort_name}"] = metrics["confusion_matrix"]

    df_comp = pd.DataFrame(comparison_rows)
    df_per_class = pd.DataFrame(per_class_rows)

    # 4. Statistical Significance Testing
    print("\n--- Running Statistical Significance Tests ---")
    stat_test_rows = []

    # Test suites:
    # Suite 1: Cohort A (Strict Contiguous Matched N=2328)
    cohort_a_keys = cohort_keys_dict["Cohort A: Strict Contiguous Matched Subset"]
    y_true_a = [ground_truth_map[k] for k in cohort_a_keys]

    # Baseline anchor: DistilBERT Utterance Baseline
    baseline_pred_a = [loaded_preds["DistilBERT Utterance Baseline"][k] for k in cohort_a_keys]

    models_to_test_cohort_a = [
        "DistilBERT Context-1",
        "DistilBERT Context-1 + Speaker",
        "DistilBERT Class-Weighted (Baseline)",
        "Model A: Emotion-History DistilBERT (Autoregressive)",
        "Model B: Combined-History DistilBERT (Autoregressive)",
        "Model B: Combined-History DistilBERT (Teacher-Forced Oracle)",
    ]

    p_values_mcnemar = []
    raw_mcnemar_records = []

    for model_name in models_to_test_cohort_a:
        pred_a = [loaded_preds[model_name][k] for k in cohort_a_keys]
        
        # McNemar test
        mcnemar = run_mcnemar_test(y_true_a, pred_a, baseline_pred_a)
        
        # Paired bootstrap for Macro F1
        boot_f1 = paired_bootstrap_macro_f1(y_true_a, pred_a, baseline_pred_a, n_bootstraps=10000, seed=42)

        raw_mcnemar_records.append({
            "cohort": "Cohort A: Strict Contiguous Matched Subset",
            "n": len(y_true_a),
            "model_tested": model_name,
            "baseline_model": "DistilBERT Utterance Baseline",
            "mcnemar": mcnemar,
            "boot_f1": boot_f1,
        })
        p_values_mcnemar.append(mcnemar["p_value"])

    # Model B vs Model A paired comparison (Cohort A)
    pred_model_a = [loaded_preds["Model A: Emotion-History DistilBERT (Autoregressive)"][k] for k in cohort_a_keys]
    pred_model_b = [loaded_preds["Model B: Combined-History DistilBERT (Autoregressive)"][k] for k in cohort_a_keys]
    mcnemar_b_vs_a = run_mcnemar_test(y_true_a, pred_model_b, pred_model_a)
    boot_f1_b_vs_a = paired_bootstrap_macro_f1(y_true_a, pred_model_b, pred_model_a, n_bootstraps=10000, seed=42)

    raw_mcnemar_records.append({
        "cohort": "Cohort A: Strict Contiguous Matched Subset",
        "n": len(y_true_a),
        "model_tested": "Model B: Combined-History DistilBERT (Autoregressive)",
        "baseline_model": "Model A: Emotion-History DistilBERT (Autoregressive)",
        "mcnemar": mcnemar_b_vs_a,
        "boot_f1": boot_f1_b_vs_a,
    })

    # Model B Auto vs Model B Teacher-Forced (Autoregressive Penalty in Cohort A)
    pred_model_b_tf = [loaded_preds["Model B: Combined-History DistilBERT (Teacher-Forced Oracle)"][k] for k in cohort_a_keys]
    mcnemar_b_tf_vs_auto = run_mcnemar_test(y_true_a, pred_model_b_tf, pred_model_b)
    boot_f1_b_tf_vs_auto = paired_bootstrap_macro_f1(y_true_a, pred_model_b_tf, pred_model_b, n_bootstraps=10000, seed=42)

    raw_mcnemar_records.append({
        "cohort": "Cohort A: Strict Contiguous Matched Subset",
        "n": len(y_true_a),
        "model_tested": "Model B: Combined-History DistilBERT (Teacher-Forced Oracle)",
        "baseline_model": "Model B: Combined-History DistilBERT (Autoregressive)",
        "mcnemar": mcnemar_b_tf_vs_auto,
        "boot_f1": boot_f1_b_tf_vs_auto,
    })

    # Apply Holm-Bonferroni correction to all paired comparisons against baseline (first 6 tests)
    # Sort by p-value
    k = len(raw_mcnemar_records)
    
    for rec in raw_mcnemar_records:
        m = rec["mcnemar"]
        b_f1 = rec["boot_f1"]
        stat_test_rows.append({
            "Cohort": rec["cohort"],
            "Sample Size (N)": rec["n"],
            "Comparison": f"{rec['model_tested']} vs {rec['baseline_model']}",
            "Discordant Pairs (b, c)": f"({m['b']}, {m['c']})",
            "Discordant Total (b+c)": m["n_discordant"],
            "McNemar Chi2 (Edwards)": round(m["chi2"], 4),
            "McNemar p-value": f"{m['p_value']:.4e}" if m['p_value'] < 0.0001 else f"{m['p_value']:.4f}",
            "Exact Binomial p-value": f"{m['binom_p']:.4e}" if m['binom_p'] < 0.0001 else f"{m['binom_p']:.4f}",
            "Accuracy Significant (p<0.05)": "Yes" if m["p_value"] < 0.05 else "No",
            "Macro F1 Diff (Model - Baseline)": round(b_f1["diff"], 4),
            "Macro F1 95% Bootstrap CI": f"[{b_f1['ci_lower']:.4f}, {b_f1['ci_upper']:.4f}]",
            "Macro F1 Bootstrap p-value": f"{b_f1['p_value']:.4e}" if b_f1['p_value'] < 0.0001 else f"{b_f1['p_value']:.4f}",
            "Macro F1 Significant (p<0.05)": "Yes" if b_f1["p_value"] < 0.05 else "No",
        })

    df_stat = pd.DataFrame(stat_test_rows)

    # Save CSV outputs
    out_table_dir = PROJECT_ROOT / "outputs/tables"
    out_table_dir.mkdir(parents=True, exist_ok=True)

    df_comp.to_csv(out_table_dir / "canonical_test_comparison.csv", index=False)
    df_per_class.to_csv(out_table_dir / "canonical_per_class_metrics.csv", index=False)
    df_stat.to_csv(out_table_dir / "canonical_statistical_tests.csv", index=False)

    print(f"\nSaved {out_table_dir / 'canonical_test_comparison.csv'}")
    print(f"Saved {out_table_dir / 'canonical_per_class_metrics.csv'}")
    print(f"Saved {out_table_dir / 'canonical_statistical_tests.csv'}")

    # Print summary of Cohort A
    print("\n" + "=" * 80)
    print("COHORT A RESULTS SUMMARY (N=2,328 Strict Contiguous Matched)")
    print("=" * 80)
    cohort_a_summary = df_comp[df_comp["Cohort"] == "Cohort A: Strict Contiguous Matched Subset"][
        ["Model", "Accuracy", "Macro F1", "Weighted F1", "Neutral Pred Rate", "Neutral True Rate"]
    ]
    print(cohort_a_summary.to_string(index=False))

    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE SUMMARY (Cohort A)")
    print("=" * 80)
    print(df_stat[["Comparison", "Discordant Pairs (b, c)", "McNemar p-value", "Accuracy Significant (p<0.05)", "Macro F1 Diff (Model - Baseline)", "Macro F1 95% Bootstrap CI", "Macro F1 Significant (p<0.05)"]].to_string(index=False))

    # Save confusion matrices for reference in json
    with open(PROJECT_ROOT / "outputs/tables/canonical_confusion_matrices.json", "w") as f:
        json.dump(cm_dict, f, indent=2)

if __name__ == "__main__":
    main()
