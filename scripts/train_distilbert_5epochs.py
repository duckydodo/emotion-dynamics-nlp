#!/usr/bin/env python3
"""
Controlled Five-Epoch DistilBERT Experiment for Utterance-Level Emotion Classification on MELD.

Investigation:
    Evaluate whether extending training from 3 to 5 epochs improves validation and test
    Macro F1 and accuracy, or whether overfitting/representation degradation occurs on the
    imbalanced MELD dataset.

Configuration:
    Model: distilbert-base-uncased (fresh initialization)
    Epochs: 5
    Learning Rate: 2e-5 (AdamW)
    Batch Size: 4
    Max Sequence Length: 128
    Seed: 42
    Selection Criterion: Highest Validation Macro F1
    Checkpoints: Saved per epoch (epoch_1 through epoch_5) plus best and latest_training_state.pt
    Output Directory: outputs/experiments/distilbert_5epochs
    Checkpoint Directory: checkpoints/distilbert_5epochs

Usage:
    docker run --rm -v $(pwd):/app -v /home/chitta/.cache/huggingface:/root/.cache/huggingface \
        -w /app emotion-dynamics-nlp:latest python3 scripts/train_distilbert_5epochs.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

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
from src.evaluation.metrics import evaluate, print_results, plot_confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Controlled 5-Epoch DistilBERT Experiment on MELD.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5).")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and eval (default: 4).")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for AdamW (default: 2e-5).")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length (default: 128).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42).")
    parser.add_argument("--num_threads", type=int, default=6, help="PyTorch CPU thread count (default: 6).")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest_training_state.pt if available.")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(args.num_threads)
    set_seed(args.seed)

    print("=" * 80)
    print("PHASE 1: CONTROLLED FIVE-EPOCH DISTILBERT EXPERIMENT")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: CPU (threads={args.num_threads})")
    print(f"Hyperparameters: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.learning_rate}, max_length={args.max_length}, seed={args.seed}")
    print("=" * 80)

    # Output and Checkpoint Directories
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / "distilbert_5epochs"
    experiment_dir = PROJECT_ROOT / "outputs" / "experiments" / "distilbert_5epochs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load official MELD splits
    print("\n[1/5] Loading official MELD splits from data/raw/meld/...")
    dataset = load_meld(PROJECT_ROOT / "data" / "raw" / "meld")
    train_df = dataset.train
    dev_df = dataset.dev
    test_df = dataset.test

    print(f"  Train: {len(train_df):,} utterances ({train_df['dialogue_id'].nunique()} dialogues)")
    print(f"  Dev:   {len(dev_df):,} utterances ({dev_df['dialogue_id'].nunique()} dialogues)")
    print(f"  Test:  {len(test_df):,} utterances ({test_df['dialogue_id'].nunique()} dialogues)")

    train_texts = train_df["text"].tolist()
    train_labels = train_df["emotion"].str.lower().tolist()
    dev_texts = dev_df["text"].tolist()
    dev_labels = dev_df["emotion"].str.lower().tolist()
    test_texts = test_df["text"].tolist()
    test_labels = test_df["emotion"].str.lower().tolist()

    # 2. Initialize fresh DistilBERT model
    print("\n[2/5] Initializing fresh distilbert-base-uncased classifier...")
    classifier = TransformerEmotionClassifier(
        model_name="distilbert-base-uncased",
        num_labels=7,
        device="cpu",
    )

    trainer = TransformerTrainer(
        classifier=classifier,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_length=args.max_length,
        seed=args.seed,
        weighting_method="unweighted",
    )

    # 3. Check for resumption
    start_epoch = 1
    best_macro_f1 = -1.0
    training_history = []
    log_file = experiment_dir / "training_log.jsonl"

    state_path = checkpoint_dir / "latest_training_state.pt"
    if args.resume and state_path.exists():
        print(f"  Found existing training state at {state_path}. Resuming...")
        state = trainer.load_training_checkpoint(checkpoint_dir)
        start_epoch = state["epoch"] + 1
        best_macro_f1 = state["best_macro_f1"]
        training_history = trainer.history
        print(f"  Resumed at epoch {start_epoch}, previous best val Macro F1: {best_macro_f1:.4f}")
    else:
        print("  Starting fresh training run from Epoch 1...")
        if log_file.exists():
            log_file.unlink()

    # Create train dataloader
    train_loader = trainer._create_loader(
        texts=train_texts,
        labels=train_labels,
        shuffle=True,
    )

    # 4. Training Loop
    print("\n[3/5] Commencing training across 5 epochs...")
    total_start_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time()
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        
        # Train epoch
        train_loss = trainer._run_epoch(train_loader)
        train_duration = time.time() - epoch_start_time

        # Validate on Dev
        val_start_time = time.time()
        val_results, _ = trainer.validate(dev_texts, dev_labels)
        val_duration = time.time() - val_start_time
        epoch_total_duration = time.time() - epoch_start_time

        epoch_record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_duration_sec": round(train_duration, 2),
            "val_duration_sec": round(val_duration, 2),
            "epoch_total_duration_sec": round(epoch_total_duration, 2),
            "val_accuracy": float(val_results.accuracy),
            "val_macro_f1": float(val_results.macro_f1),
            "val_weighted_f1": float(val_results.weighted_f1),
            "is_best": bool(val_results.macro_f1 > best_macro_f1),
        }
        training_history.append(epoch_record)
        trainer.history = training_history

        print(f"  Train Loss:     {train_loss:.4f} ({train_duration/60:.2f} min)")
        print(f"  Val Accuracy:   {val_results.accuracy*100:.3f}%")
        print(f"  Val Macro F1:   {val_results.macro_f1*100:.3f}%")
        print(f"  Val Weighted F1:{val_results.weighted_f1*100:.3f}%")
        print(f"  Epoch Duration: {epoch_total_duration/60:.2f} min")

        # Save incremental epoch checkpoint
        epoch_ckpt_dir = checkpoint_dir / f"epoch_{epoch}"
        epoch_ckpt_dir.mkdir(parents=True, exist_ok=True)
        classifier.save(epoch_ckpt_dir)
        with open(epoch_ckpt_dir / "metrics.json", "w") as f:
            json.dump(epoch_record, f, indent=2)
        print(f"  Saved epoch checkpoint: {epoch_ckpt_dir}")

        # Check for best model
        if val_results.macro_f1 > best_macro_f1:
            best_macro_f1 = val_results.macro_f1
            best_dir = trainer._save_best_model(checkpoint_dir, best_macro_f1)
            print(f"  ★ New best model saved to {best_dir} (Macro F1 = {best_macro_f1*100:.3f}%)")

        # Save training state for resumption
        trainer._save_training_checkpoint(checkpoint_dir, epoch, best_macro_f1)

        # Write to logfile
        with open(log_file, "a") as f:
            f.write(json.dumps(epoch_record) + "\n")

    total_training_duration = time.time() - total_start_time
    print(f"\nTraining completed in {total_training_duration/3600:.2f} hours ({total_training_duration/60:.1f} minutes).")

    # 5. Save complete history
    with open(experiment_dir / "history.json", "w") as f:
        json.dump(
            {
                "experiment": "distilbert_5epochs",
                "total_duration_sec": round(total_training_duration, 2),
                "epochs": args.epochs,
                "best_val_macro_f1": best_macro_f1,
                "history": training_history,
            },
            f,
            indent=2,
        )

    # 6. Evaluate Best Checkpoint on Test Set
    print("\n[4/5] Evaluating selected best checkpoint on Test set...")
    best_checkpoint_path = checkpoint_dir / "best"
    best_classifier = TransformerEmotionClassifier(
        model_name="distilbert-base-uncased",
        model_path=str(best_checkpoint_path),
        device="cpu",
    )

    # Full Test Evaluation (N=2,610)
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
        "best_val_macro_f1": float(best_macro_f1),
        "test_examples": len(test_df),
        "test_accuracy": float(test_results.accuracy),
        "test_macro_f1": float(test_results.macro_f1),
        "test_weighted_f1": float(test_results.weighted_f1),
        "per_class": {
            cls: {
                "precision": float(test_results.per_class[cls].precision),
                "recall": float(test_results.per_class[cls].recall),
                "f1": float(test_results.per_class[cls].f1),
                "support": int(test_results.per_class[cls].support),
            }
            for cls in LABELS
        },
        "confusion_matrix": test_results.confusion_matrix.tolist(),
    }
    with open(experiment_dir / "metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=2)

    # Cohort A Matched Evaluation (Contiguous dialogues, N=2,328)
    print("\nEvaluating on Canonical Cohort A (Matched subset without ID gaps, N=2,328)...")
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
                "support": int(cohort_a_results.per_class[cls].support),
            }
            for cls in LABELS
        },
    }
    with open(experiment_dir / "cohort_a_metrics.json", "w") as f:
        json.dump(cohort_a_metrics, f, indent=2)

    print(f"  Cohort A Accuracy:    {cohort_a_results.accuracy*100:.3f}%")
    print(f"  Cohort A Macro F1:    {cohort_a_results.macro_f1*100:.3f}%")
    print(f"  Cohort A Weighted F1: {cohort_a_results.weighted_f1*100:.3f}%")

    # 7. Comparison with Original 3-Epoch Baseline
    print("\n[5/5] Generating comparative analysis against 3-epoch baseline...")
    baseline_metrics_path = PROJECT_ROOT / "outputs" / "experiments" / "transformer_baseline" / "metrics.json"
    if baseline_metrics_path.exists():
        with open(baseline_metrics_path) as f:
            base_data = json.load(f)
        
        comparison = {
            "original_3_epochs": {
                "epochs": 3,
                "test_accuracy": base_data.get("test_accuracy", 0.6031),
                "test_macro_f1": base_data.get("test_macro_f1", 0.4400),
                "test_weighted_f1": base_data.get("test_weighted_f1", 0.6054),
            },
            "new_5_epochs": {
                "epochs": 5,
                "test_accuracy": float(test_results.accuracy),
                "test_macro_f1": float(test_results.macro_f1),
                "test_weighted_f1": float(test_results.weighted_f1),
            },
            "deltas": {
                "accuracy_delta": float(test_results.accuracy - base_data.get("test_accuracy", 0.6031)),
                "macro_f1_delta": float(test_results.macro_f1 - base_data.get("test_macro_f1", 0.4400)),
                "weighted_f1_delta": float(test_results.weighted_f1 - base_data.get("test_weighted_f1", 0.6054)),
            },
        }
        with open(experiment_dir / "comparison_vs_3epochs.json", "w") as f:
            json.dump(comparison, f, indent=2)
        print("  Comparison saved to outputs/experiments/distilbert_5epochs/comparison_vs_3epochs.json")

    print("\n" + "=" * 80)
    print("PHASE 1 COMPLETE: All checkpoints and evaluation artifacts saved.")
    print("=" * 80)


if __name__ == "__main__":
    main()
