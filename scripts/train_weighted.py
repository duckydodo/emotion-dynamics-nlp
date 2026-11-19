"""
Train class-weighted DistilBERT for emotion classification on MELD.

Supports:
- Conditions: utterance, context_1, context_1_speaker
- Weighting methods: inverse_frequency (with alpha smoothing), effective_samples (Cui et al., with beta), unweighted
- Hyperparameter tuning on validation Macro F1 only
- Automatic best checkpoint selection and structured output generation

Usage:
    # Smoke test on tiny slice:
    python scripts/train_weighted.py --condition utterance --weighting_method inverse_frequency --tiny

    # Full training:
    python scripts/train_weighted.py --condition context_1 --weighting_method effective_samples --beta 0.999 --epochs 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
from src.models.trainer import TransformerTrainer, set_seed
from src.models.loss import compute_class_weights
from src.evaluation.metrics import evaluate, print_results, plot_confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Train class-weighted DistilBERT on MELD.")
    parser.add_argument(
        "--condition",
        type=str,
        default="utterance",
        choices=["utterance", "context_1", "context_1_speaker"],
        help="Input format: 'utterance', 'context_1', or 'context_1_speaker'.",
    )
    parser.add_argument(
        "--weighting_method",
        type=str,
        default="inverse_frequency",
        choices=["inverse_frequency", "effective_samples", "unweighted"],
        help="Class weighting method.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Smoothing exponent for inverse_frequency (alpha=1.0 is standard, alpha=0.5 is sqrt).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.999,
        help="Hyperparameter for effective_samples (typically 0.99 or 0.999).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate for AdamW.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum token length for tokenizer.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Run on a tiny subset (32 train, 16 dev) for quick smoke testing.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw" / "meld"),
        help="Path to raw MELD data directory.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory to save checkpoints. Defaults to checkpoints/<experiment_name>.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save results. Defaults to outputs/experiments/<experiment_name>.",
    )
    parser.add_argument(
        "--evaluate_test",
        action="store_true",
        default=True,
        help="Evaluate on test split after training and model selection.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of PyTorch CPU threads to utilize.",
    )

    return parser.parse_args()


def prepare_dataset_for_condition(
    dataset,
    condition: str,
):
    """
    Prepare input texts and labels according to the specified condition.
    """
    if condition == "utterance":
        train_texts = dataset.train["text"].tolist()
        train_labels = dataset.train["emotion"].tolist()

        dev_texts = dataset.dev["text"].tolist()
        dev_labels = dataset.dev["emotion"].tolist()

        test_texts = dataset.test["text"].tolist()
        test_labels = dataset.test["emotion"].tolist()

        train_meta = dataset.train[["dialogue_id", "utterance_id"]].copy()
        dev_meta = dataset.dev[["dialogue_id", "utterance_id"]].copy()
        test_meta = dataset.test[["dialogue_id", "utterance_id"]].copy()

    elif condition == "context_1":
        train_ctx = build_context_examples(dataset.train, context_turns=1, include_speaker=False)
        dev_ctx = build_context_examples(dataset.dev, context_turns=1, include_speaker=False)
        test_ctx = build_context_examples(dataset.test, context_turns=1, include_speaker=False)

        train_texts = train_ctx["context"].tolist()
        train_labels = train_ctx["emotion"].tolist()

        dev_texts = dev_ctx["context"].tolist()
        dev_labels = dev_ctx["emotion"].tolist()

        test_texts = test_ctx["context"].tolist()
        test_labels = test_ctx["emotion"].tolist()

        train_meta = train_ctx[["dialogue_id", "utterance_id"]].copy()
        dev_meta = dev_ctx[["dialogue_id", "utterance_id"]].copy()
        test_meta = test_ctx[["dialogue_id", "utterance_id"]].copy()

    elif condition == "context_1_speaker":
        train_ctx = build_context_examples(dataset.train, context_turns=1, include_speaker=True)
        dev_ctx = build_context_examples(dataset.dev, context_turns=1, include_speaker=True)
        test_ctx = build_context_examples(dataset.test, context_turns=1, include_speaker=True)

        train_texts = train_ctx["context"].tolist()
        train_labels = train_ctx["emotion"].tolist()

        dev_texts = dev_ctx["context"].tolist()
        dev_labels = dev_ctx["emotion"].tolist()

        test_texts = test_ctx["context"].tolist()
        test_labels = test_ctx["emotion"].tolist()

        train_meta = train_ctx[["dialogue_id", "utterance_id"]].copy()
        dev_meta = dev_ctx[["dialogue_id", "utterance_id"]].copy()
        test_meta = test_ctx[["dialogue_id", "utterance_id"]].copy()

    else:
        raise ValueError(f"Unknown condition '{condition}'")

    return {
        "train": (train_texts, train_labels, train_meta),
        "dev": (dev_texts, dev_labels, dev_meta),
        "test": (test_texts, test_labels, test_meta),
    }


def main():
    args = parse_args()

    if args.threads is not None and args.threads > 0:
        torch.set_num_threads(args.threads)

    set_seed(args.seed)

    # Determine experiment name
    weight_desc = args.weighting_method
    if args.weighting_method == "inverse_frequency":
        weight_desc += f"_a{args.alpha}"
    elif args.weighting_method == "effective_samples":
        weight_desc += f"_b{args.beta}"

    exp_name = f"transformer_{args.condition}_{weight_desc}"
    if args.tiny:
        exp_name += "_tiny"

    print("=" * 70)
    print(f"EXPERIMENT: {exp_name}")
    print(f"Condition: {args.condition} | Method: {args.weighting_method}")
    print("=" * 70)

    # Paths
    data_path = Path(args.data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found at: {data_path}")

    checkpoint_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else PROJECT_ROOT / "checkpoints" / exp_name
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "outputs" / "experiments" / exp_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading MELD from: {data_path}")
    dataset = load_meld(data_path)
    splits = prepare_dataset_for_condition(dataset, args.condition)

    train_texts, train_labels, train_meta = splits["train"]
    dev_texts, dev_labels, dev_meta = splits["dev"]
    test_texts, test_labels, test_meta = splits["test"]

    print(f"Full counts — Train: {len(train_texts)}, Dev: {len(dev_texts)}, Test: {len(test_texts)}")

    # Compute class weights STRICTLY from training split
    weighting_params = {}
    if args.weighting_method == "inverse_frequency":
        weighting_params["alpha"] = args.alpha
    elif args.weighting_method == "effective_samples":
        weighting_params["beta"] = args.beta

    class_weights = compute_class_weights(
        labels=train_labels,
        method=args.weighting_method,
        alpha=args.alpha,
        beta=args.beta,
        class_order=LABELS,
    )

    print("\n--- Training Class Weights ---")
    for emo, w in zip(LABELS, class_weights.tolist()):
        print(f"  {emo:10s}: {w:.4f}")

    if args.tiny:
        print("\n[TINY MODE] Truncating to 32 train, 16 dev, 16 test examples.")
        train_texts = train_texts[:32]
        train_labels = train_labels[:32]
        train_meta = train_meta.iloc[:32]

        dev_texts = dev_texts[:16]
        dev_labels = dev_labels[:16]
        dev_meta = dev_meta.iloc[:16]

        test_texts = test_texts[:16]
        test_labels = test_labels[:16]
        test_meta = test_meta.iloc[:16]

    # Initialize model
    classifier = TransformerEmotionClassifier(
        model_name="distilbert-base-uncased",
        num_labels=len(LABELS),
        class_weights=class_weights,
    )

    trainer = TransformerTrainer(
        classifier=classifier,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_length=args.max_length,
        seed=args.seed,
        class_weights=class_weights,
        weighting_method=args.weighting_method,
        weighting_params=weighting_params,
    )

    # Train and select best model on validation Macro F1
    print("\nStarting training loop...")
    history = trainer.fit(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=dev_texts,
        val_labels=dev_labels,
        checkpoint_dir=checkpoint_dir,
        resume=False,
    )

    # Save training history
    history_file = output_dir / "training_history.json"
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved training history to {history_file}")

    best_checkpoint = checkpoint_dir / "best"
    print(f"Best validation model checkpoint: {best_checkpoint}")

    # Evaluate on test data ONLY AFTER model selection is complete
    if args.evaluate_test:
        print("\n" + "=" * 70)
        print("EVALUATING BEST MODEL ON TEST DATA (Held-Out)")
        print("=" * 70)

        # Load best validation model
        best_classifier = TransformerEmotionClassifier(
            model_path=best_checkpoint,
        )

        test_predictions = best_classifier.predict(
            test_texts,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        test_probabilities = best_classifier.predict_proba(
            test_texts,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

        eval_result = evaluate(
            y_true=test_labels,
            y_pred=test_predictions,
            labels=LABELS,
        )

        print("\n--- Full Test Evaluation (N = {}) ---".format(len(test_texts)))
        print_results(eval_result)

        # Plot and save confusion matrix
        plot_confusion_matrix(
            results=eval_result,
            save_path=output_dir / "confusion_matrix.png",
            show=False,
        )

        from collections import Counter
        test_counts = Counter(test_labels)

        # Save structured results
        metrics_dict = {
            "experiment": exp_name,
            "condition": args.condition,
            "weighting_method": args.weighting_method,
            "weighting_params": weighting_params,
            "class_weights": {emo: float(w) for emo, w in zip(LABELS, class_weights.tolist())},
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "test_examples": len(test_texts),
            "test_accuracy": eval_result.accuracy,
            "test_macro_f1": eval_result.macro_f1,
            "test_weighted_f1": eval_result.weighted_f1,
            "accuracy": eval_result.accuracy,
            "macro_f1": eval_result.macro_f1,
            "weighted_f1": eval_result.weighted_f1,
            "confusion_matrix": eval_result.confusion_matrix.tolist(),
        }

        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics_dict, f, indent=4)

        # Save classification report CSV (with support)
        per_class_rows = []
        for emo, m in eval_result.per_class.items():
            per_class_rows.append(
                {
                    "emotion": emo,
                    "precision": m.precision,
                    "recall": m.recall,
                    "f1": m.f1,
                    "support": test_counts.get(emo, 0),
                }
            )
        pd.DataFrame(per_class_rows).to_csv(
            output_dir / "classification_report.csv", index=False
        )

        # Save predictions CSV
        pred_df = test_meta.copy()
        pred_df["text"] = test_texts
        pred_df["true_emotion"] = test_labels
        pred_df["predicted_emotion"] = test_predictions
        pred_df["correct"] = pred_df["true_emotion"] == pred_df["predicted_emotion"]

        for idx, emo in enumerate(LABELS):
            pred_df[f"prob_{emo}"] = test_probabilities[:, idx]

        pred_df.to_csv(output_dir / "predictions.csv", index=False)

        # --------------------------------------------------------------------
        # Matched Context Subset Evaluation (N = 2,328)
        # --------------------------------------------------------------------
        if args.condition == "utterance":
            print("\n" + "=" * 70)
            print("EVALUATING ON MATCHED CONTEXT SUBSET (N = 2,328)")
            print("=" * 70)

            test_ctx1 = build_context_examples(dataset.test, context_turns=1, include_speaker=False)
            matched_keys = set(zip(test_ctx1["dialogue_id"], test_ctx1["utterance_id"]))

            matched_indices = [
                i for i, (d_id, u_id) in enumerate(zip(test_meta["dialogue_id"], test_meta["utterance_id"]))
                if (d_id, u_id) in matched_keys
            ]

            matched_y_true = [test_labels[i] for i in matched_indices]
            matched_y_pred = [test_predictions[i] for i in matched_indices]
            matched_probs = test_probabilities[matched_indices]
            matched_counts = Counter(matched_y_true)

            eval_matched = evaluate(
                y_true=matched_y_true,
                y_pred=matched_y_pred,
                labels=LABELS,
            )

            print_results(eval_matched)

            plot_confusion_matrix(
                results=eval_matched,
                save_path=output_dir / "matched_confusion_matrix.png",
                show=False,
            )

            matched_metrics_dict = {
                "experiment": f"{exp_name}_matched",
                "condition": "utterance_matched",
                "test_examples": len(matched_indices),
                "test_accuracy": eval_matched.accuracy,
                "test_macro_f1": eval_matched.macro_f1,
                "test_weighted_f1": eval_matched.weighted_f1,
                "accuracy": eval_matched.accuracy,
                "macro_f1": eval_matched.macro_f1,
                "weighted_f1": eval_matched.weighted_f1,
                "confusion_matrix": eval_matched.confusion_matrix.tolist(),
            }

            with open(output_dir / "matched_metrics.json", "w") as f:
                json.dump(matched_metrics_dict, f, indent=4)

            matched_per_class_rows = []
            for emo, m in eval_matched.per_class.items():
                matched_per_class_rows.append(
                    {
                        "emotion": emo,
                        "precision": m.precision,
                        "recall": m.recall,
                        "f1": m.f1,
                        "support": matched_counts.get(emo, 0),
                    }
                )
            pd.DataFrame(matched_per_class_rows).to_csv(
                output_dir / "matched_classification_report.csv", index=False
            )

            matched_pred_df = pred_df.iloc[matched_indices].copy()
            matched_pred_df.to_csv(output_dir / "matched_predictions.csv", index=False)

            # ----------------------------------------------------------------
            # Comparison Against Baseline on Identical Subsets
            # ----------------------------------------------------------------
            print("\n" + "=" * 70)
            print("SIDE-BY-SIDE COMPARISON: BASELINE vs CLASS-WEIGHTED")
            print("=" * 70)

            # Baseline metrics (verified)
            # Full: Acc 60.31%, Macro F1 0.4400, Weighted F1 0.6054
            # Matched: Acc 59.45%, Macro F1 0.4314, Weighted F1 0.5969
            comp_rows = [
                {
                    "subset": "Full Test (N=2,610)",
                    "metric": "Accuracy",
                    "baseline_unweighted": 0.603065,
                    "class_weighted": eval_result.accuracy,
                    "delta": eval_result.accuracy - 0.603065,
                },
                {
                    "subset": "Full Test (N=2,610)",
                    "metric": "Macro F1",
                    "baseline_unweighted": 0.440029,
                    "class_weighted": eval_result.macro_f1,
                    "delta": eval_result.macro_f1 - 0.440029,
                },
                {
                    "subset": "Full Test (N=2,610)",
                    "metric": "Weighted F1",
                    "baseline_unweighted": 0.605354,
                    "class_weighted": eval_result.weighted_f1,
                    "delta": eval_result.weighted_f1 - 0.605354,
                },
                {
                    "subset": "Matched Test (N=2,328)",
                    "metric": "Accuracy",
                    "baseline_unweighted": 0.594502,
                    "class_weighted": eval_matched.accuracy,
                    "delta": eval_matched.accuracy - 0.594502,
                },
                {
                    "subset": "Matched Test (N=2,328)",
                    "metric": "Macro F1",
                    "baseline_unweighted": 0.431416,
                    "class_weighted": eval_matched.macro_f1,
                    "delta": eval_matched.macro_f1 - 0.431416,
                },
                {
                    "subset": "Matched Test (N=2,328)",
                    "metric": "Weighted F1",
                    "baseline_unweighted": 0.596910,
                    "class_weighted": eval_matched.weighted_f1,
                    "delta": eval_matched.weighted_f1 - 0.596910,
                },
            ]
            comp_df = pd.DataFrame(comp_rows)
            comp_df.to_csv(output_dir / "baseline_comparison.csv", index=False)
            print(comp_df.to_string(index=False))

        print(f"\nAll artifacts successfully saved to {output_dir}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
