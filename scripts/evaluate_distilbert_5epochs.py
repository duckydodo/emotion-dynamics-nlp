#!/usr/bin/env python3
"""
Evaluate the trained 5-Epoch DistilBERT best checkpoint on Full Test (N=2,610)
and Canonical Cohort A (N=2,328), saving metrics.json, cohort_a_metrics.json,
and comparison_vs_3epochs.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_meld
from src.data.context import build_context_examples
from src.models.transformer import (
    TransformerEmotionClassifier,
    LABELS,
    LABEL_TO_ID,
    ID_TO_LABEL,
)
from src.models.trainer import TransformerTrainer
from src.evaluation.metrics import evaluate, print_results


def main():
    print("=" * 80)
    print("EVALUATING 5-EPOCH DISTILBERT BEST CHECKPOINT")
    print("=" * 80)

    checkpoint_dir = PROJECT_ROOT / "checkpoints" / "distilbert_5epochs"
    experiment_dir = PROJECT_ROOT / "outputs" / "experiments" / "distilbert_5epochs"
    best_checkpoint_path = checkpoint_dir / "best"

    dataset = load_meld(PROJECT_ROOT / "data" / "raw" / "meld")
    test_df = dataset.test
    test_texts = test_df["text"].tolist()
    test_labels = test_df["emotion"].str.lower().tolist()

    # Load history to get best val macro f1
    with open(experiment_dir / "history.json") as f:
        history_data = json.load(f)
    best_val_macro_f1 = history_data.get("best_val_macro_f1", 0.4215)

    print(f"Loading best checkpoint from: {best_checkpoint_path}")
    classifier = TransformerEmotionClassifier(
        model_name="distilbert-base-uncased",
        model_path=str(best_checkpoint_path),
        device="cpu",
    )

    trainer = TransformerTrainer(
        classifier=classifier,
        learning_rate=2e-5,
        batch_size=16,
        epochs=5,
        max_length=128,
        seed=42,
    )

    # 1. Full Test Evaluation (N=2,610)
    print("\n[1/3] Evaluating on Full Test Set (N=2,610)...")
    test_loader = trainer._create_loader(texts=test_texts, labels=test_labels, shuffle=False)
    y_true_indices, y_pred_indices, test_probs = trainer._predict_loader(test_loader)
    y_true = [ID_TO_LABEL[i] for i in y_true_indices]
    y_pred = [ID_TO_LABEL[i] for i in y_pred_indices]

    test_results = evaluate(y_true, y_pred, labels=LABELS)
    print("\nFull Test Set Results (N=2,610):")
    print_results(test_results)

    # Save predictions.csv
    pred_df = test_df.copy()
    pred_df["model_text"] = test_texts
    pred_df["true_emotion"] = y_true
    pred_df["label_id"] = y_true_indices
    pred_df["predicted_id"] = y_pred_indices
    pred_df["predicted_emotion"] = y_pred
    pred_df["correct"] = [t == p for t, p in zip(y_true, y_pred)]
    for i, emotion in enumerate(LABELS):
        pred_df[f"prob_{emotion}"] = test_probs[:, i]

    pred_file = experiment_dir / "predictions.csv"
    pred_df.to_csv(pred_file, index=False)
    print(f"  Saved full test predictions to {pred_file}")

    # Save metrics.json
    metrics_data = {
        "experiment": "distilbert_5epochs",
        "selected_checkpoint": str(best_checkpoint_path),
        "best_val_macro_f1": float(best_val_macro_f1),
        "test_examples": len(test_df),
        "test_accuracy": float(test_results.accuracy),
        "test_macro_f1": float(test_results.macro_f1),
        "test_weighted_f1": float(test_results.weighted_f1),
        "per_class": {
            cls: {
                "precision": float(test_results.per_class[cls].precision),
                "recall": float(test_results.per_class[cls].recall),
                "f1": float(test_results.per_class[cls].f1),
                "support": int(sum(1 for y in y_true if y == cls)),
            }
            for cls in LABELS
        },
        "confusion_matrix": test_results.confusion_matrix.tolist(),
    }
    with open(experiment_dir / "metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"  Saved metrics to {experiment_dir / 'metrics.json'}")

    # 2. Canonical Cohort A Matched Evaluation (N=2,328)
    print("\n[2/3] Evaluating on Canonical Cohort A (N=2,328)...")
    test_ctx1 = build_context_examples(test_df, context_turns=1, include_speaker=False)
    merged_cohort_a = pd.merge(
        test_df,
        test_ctx1[["dialogue_id", "utterance_id"]],
        on=["dialogue_id", "utterance_id"],
        how="inner",
    )
    assert len(merged_cohort_a) == 2328, f"Expected 2,328 Cohort A examples, got {len(merged_cohort_a)}"

    cohort_a_texts = merged_cohort_a["text"].tolist()
    cohort_a_labels = merged_cohort_a["emotion"].str.lower().tolist()
    cohort_a_loader = trainer._create_loader(texts=cohort_a_texts, labels=cohort_a_labels, shuffle=False)
    c_true_idx, c_pred_idx, _ = trainer._predict_loader(cohort_a_loader)
    c_true = [ID_TO_LABEL[i] for i in c_true_idx]
    c_pred = [ID_TO_LABEL[i] for i in c_pred_idx]
    cohort_a_results = evaluate(c_true, c_pred, labels=LABELS)

    print("\nCohort A Results (N=2,328):")
    print_results(cohort_a_results)

    cohort_a_metrics = {
        "cohort": "Cohort A (Contiguous Matched Subset, N=2,328)",
        "accuracy": float(cohort_a_results.accuracy),
        "macro_f1": float(cohort_a_results.macro_f1),
        "weighted_f1": float(cohort_a_results.weighted_f1),
        "per_class": {
            cls: {
                "precision": float(cohort_a_results.per_class[cls].precision),
                "recall": float(cohort_a_results.per_class[cls].recall),
                "f1": float(cohort_a_results.per_class[cls].f1),
                "support": int(sum(1 for y in c_true if y == cls)),
            }
            for cls in LABELS
        },
    }
    with open(experiment_dir / "cohort_a_metrics.json", "w") as f:
        json.dump(cohort_a_metrics, f, indent=2)
    print(f"  Saved Cohort A metrics to {experiment_dir / 'cohort_a_metrics.json'}")

    # 3. Baseline Comparison
    print("\n[3/3] Generating comparative analysis against 3-epoch baseline...")
    baseline_metrics_path = PROJECT_ROOT / "outputs" / "experiments" / "transformer_baseline" / "metrics.json"
    if baseline_metrics_path.exists():
        with open(baseline_metrics_path) as f:
            base_data = json.load(f)
        
        comparison = {
            "full_test": {
                "original_3_epochs": {
                    "epochs": 3,
                    "test_accuracy": base_data.get("test_accuracy", 0.6031),
                    "test_macro_f1": base_data.get("test_macro_f1", 0.4400),
                    "test_weighted_f1": base_data.get("test_weighted_f1", 0.6054),
                },
                "new_5_epochs_best_at_4": {
                    "epochs": 5,
                    "selected_epoch": 4,
                    "test_accuracy": float(test_results.accuracy),
                    "test_macro_f1": float(test_results.macro_f1),
                    "test_weighted_f1": float(test_results.weighted_f1),
                },
                "deltas": {
                    "accuracy_delta": float(test_results.accuracy - base_data.get("test_accuracy", 0.6031)),
                    "macro_f1_delta": float(test_results.macro_f1 - base_data.get("test_macro_f1", 0.4400)),
                    "weighted_f1_delta": float(test_results.weighted_f1 - base_data.get("test_weighted_f1", 0.6054)),
                },
            },
            "cohort_a_matched": {
                "original_3_epochs": {
                    "accuracy": 0.5945,
                    "macro_f1": 0.4314,
                    "weighted_f1": 0.5794,
                },
                "new_5_epochs_best_at_4": {
                    "accuracy": float(cohort_a_results.accuracy),
                    "macro_f1": float(cohort_a_results.macro_f1),
                    "weighted_f1": float(cohort_a_results.weighted_f1),
                },
                "deltas": {
                    "accuracy_delta": float(cohort_a_results.accuracy - 0.5945),
                    "macro_f1_delta": float(cohort_a_results.macro_f1 - 0.4314),
                    "weighted_f1_delta": float(cohort_a_results.weighted_f1 - 0.5794),
                },
            },
        }
        with open(experiment_dir / "comparison_vs_3epochs.json", "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"  Saved comparison to {experiment_dir / 'comparison_vs_3epochs.json'}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
