#!/usr/bin/env python3
"""
Training and Canonical Evaluation Pipeline for DialogueRNN.
Supports causal inference, variable-length dialogue batching, atomic checkpointing,
class weighting, and evaluation on all canonical MELD cohorts (A, B, C, D).
"""

from __future__ import annotations

import argparse
import json
import os
import time
import resource
import shutil
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from src.models.dialoguernn import DialogueRNN, DialogueRNNConfig
from src.data.feature_loader import (
    load_dialogue_features,
    DialogueBatchDataset,
    dialogue_collate_fn,
)
from src.models.transformer import LABELS, LABEL_TO_ID, ID_TO_LABEL
from src.models.loss import compute_class_weights
from src.evaluation.results import save_metrics, save_metadata
from src.data.types import EvaluationResult, ClassMetrics


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_model(
    model: DialogueRNN,
    dataloader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor | None = None,
) -> Tuple[float, Dict[str, Any], pd.DataFrame]:
    """
    Evaluate DialogueRNN across a dataset, returning loss, metrics dictionary,
    and a DataFrame containing utterance-level predictions.
    """
    model.eval()
    total_loss = 0.0
    total_turns = 0

    records: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            party_ids = batch["party_ids"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["label_ids"].to(device)
            lengths = batch["lengths"]
            dialogue_ids = batch["dialogue_ids"]
            utterance_ids = batch["utterance_ids"]
            speakers = batch["speakers"]
            emotions = batch["emotions"]

            logits = model(features, party_ids, mask=mask, lengths=lengths)
            loss = model.compute_loss(logits, targets, mask=mask, class_weights=class_weights)
            batch_valid_turns = int(mask.sum().item())
            total_loss += loss.item() * batch_valid_turns
            total_turns += batch_valid_turns

            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            for b in range(len(lengths)):
                seq_len = lengths[b].item()
                d_id = dialogue_ids[b].item()
                for t in range(seq_len):
                    u_id = utterance_ids[b, t].item()
                    spk = speakers[b][t]
                    true_emo = emotions[b][t]
                    pred_id = int(preds[b, t])
                    pred_emo = ID_TO_LABEL[pred_id]
                    prob_vec = probs[b, t].tolist()

                    records.append({
                        "dialogue_id": d_id,
                        "utterance_id": u_id,
                        "turn_index": t,
                        "speaker": spk,
                        "true_emotion": true_emo,
                        "true_id": LABEL_TO_ID.get(true_emo, -1),
                        "pred_emotion": pred_emo,
                        "pred_id": pred_id,
                        "probabilities": prob_vec,
                    })

    avg_loss = total_loss / total_turns if total_turns > 0 else 0.0
    pred_df = pd.DataFrame(records)

    # Compute standard summary metrics across all evaluated turns
    y_true = pred_df["true_emotion"].tolist()
    y_pred = pred_df["pred_emotion"].tolist()

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
            "support": int(sup_per[i]),
        }

    metrics = {
        "loss": float(avg_loss),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "total_turns": total_turns,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "pred_distribution": {l: int((pred_df["pred_emotion"] == l).sum()) for l in LABELS},
        "true_distribution": {l: int((pred_df["true_emotion"] == l).sum()) for l in LABELS},
    }

    return avg_loss, metrics, pred_df


def evaluate_cohorts(pred_df: pd.DataFrame, meld_test_raw_path: Path) -> Dict[str, Any]:
    """
    Evaluate predictions on all canonical cohorts:
    - Cohort A (Strict Contiguous Matched Subset, N=2,328)
    - Cohort B (All Subsequent Sequential Turns, N=2,330)
    - Cohort C (Dialogue Initial Turns t=0, N=280)
    - Cohort D (Full Test Set, N=2,610)
    """
    test_raw = pd.read_csv(meld_test_raw_path)
    test_raw["dialogue_id"] = test_raw["Dialogue_ID"]
    test_raw["utterance_id"] = test_raw["Utterance_ID"]

    # Build cohort keys exactly matching the Task 3 canonical definition
    cohort_d_keys = set(zip(test_raw["dialogue_id"], test_raw["utterance_id"]))
    cohort_c_keys = set()
    cohort_b_keys = set()
    cohort_a_keys = set()

    for _, group in test_raw.groupby("dialogue_id", sort=False):
        group = group.sort_values("utterance_id").reset_index(drop=True)
        for i, row in group.iterrows():
            key = (row["dialogue_id"], row["utterance_id"])
            if i == 0:
                cohort_c_keys.add(key)
            else:
                cohort_b_keys.add(key)
                prev_row = group.iloc[i - 1]
                if row["utterance_id"] == prev_row["utterance_id"] + 1:
                    cohort_a_keys.add(key)

    cohorts_dict = {
        "Cohort A (Strict Contiguous Matched Subset)": cohort_a_keys,
        "Cohort B (All Subsequent Sequential Turns)": cohort_b_keys,
        "Cohort C (Dialogue Initial Turns t=0)": cohort_c_keys,
        "Cohort D (Full Test Set)": cohort_d_keys,
    }

    results = {}
    pred_df["key"] = list(zip(pred_df["dialogue_id"], pred_df["utterance_id"]))

    for cohort_name, keys in cohorts_dict.items():
        subset = pred_df[pred_df["key"].isin(keys)].copy()
        y_true = subset["true_emotion"].tolist()
        y_pred = subset["pred_emotion"].tolist()

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
                "support": int(sup_per[i]),
            }

        results[cohort_name] = {
            "n_turns": len(subset),
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
            "pred_distribution": {l: int((subset["pred_emotion"] == l).sum()) for l in LABELS},
            "true_distribution": {l: int((subset["true_emotion"] == l).sum()) for l in LABELS},
        }

    return results


def train_dialoguernn(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cpu")  # Strictly CPU execution

    print("=" * 80)
    print(f"TRAINING DIALOGUERNN: {args.experiment_name}")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Random Seed: {args.seed}")
    print(f"Class Weighting Method: {args.class_weight_method}")
    print(f"Hyperparameters: Batch Size={args.batch_size}, LR={args.lr}, Epochs={args.epochs}, Dropout={args.dropout}")
    print(f"Dimensions: Proj={args.projected_dim}, Global={args.global_hidden_dim}, Party={args.party_hidden_dim}, Emotion={args.emotion_hidden_dim}")

    # 1. Load pre-extracted feature archives
    train_dataset = load_dialogue_features(args.train_features)
    dev_dataset = load_dialogue_features(args.dev_features)
    test_dataset = load_dialogue_features(args.test_features)

    train_dialogues = train_dataset.get_dialogues()
    dev_dialogues = dev_dataset.get_dialogues()
    test_dialogues = test_dataset.get_dialogues()

    if args.dry_run:
        print("[DRY RUN] Truncating dialogues for pipeline validation...")
        train_dialogues = train_dialogues[:16]
        dev_dialogues = dev_dialogues[:8]
        test_dialogues = test_dialogues[:8]
        args.epochs = 2

    print(f"Loaded Dialogues: Train={len(train_dialogues)}, Dev={len(dev_dialogues)}, Test={len(test_dialogues)}")

    # 2. Compute class weights exclusively from training labels
    class_weights: torch.Tensor | None = None
    if args.class_weight_method != "unweighted":
        if args.class_weight_method == "inverse_frequency":
            class_weights = compute_class_weights(train_dataset.emotions, method="inverse_frequency", alpha=1.0)
        elif args.class_weight_method == "smoothed":
            class_weights = compute_class_weights(train_dataset.emotions, method="inverse_frequency", alpha=0.5)
        else:
            raise ValueError(f"Unknown class weight method: {args.class_weight_method}")
        print(f"Calculated Training-Only Class Weights ({args.class_weight_method}):")
        for lbl, w in zip(LABELS, class_weights.tolist()):
            print(f"  {lbl:<10}: {w:.4f}")

    # 3. DataLoaders
    train_loader = DataLoader(
        DialogueBatchDataset(train_dialogues),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dialogue_collate_fn,
    )
    dev_loader = DataLoader(
        DialogueBatchDataset(dev_dialogues),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dialogue_collate_fn,
    )
    test_loader = DataLoader(
        DialogueBatchDataset(test_dialogues),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dialogue_collate_fn,
    )

    # 4. Initialize DialogueRNN model
    config = DialogueRNNConfig(
        feature_dim=train_dataset.feature_dim,
        projected_dim=args.projected_dim,
        global_hidden_dim=args.global_hidden_dim,
        party_hidden_dim=args.party_hidden_dim,
        emotion_hidden_dim=args.emotion_hidden_dim,
        fc_hidden_dim=args.fc_hidden_dim,
        num_classes=len(LABELS),
        dropout=args.dropout,
        use_context_attention=not args.no_context_attn,
        use_emotion_attention=not args.no_emotion_attn,
    )
    model = DialogueRNN(config).to(device)
    print(f"Model Initialized. Total Trainable Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 5. Training loop with early stopping on Dev Macro F1
    best_dev_macro_f1 = -1.0
    best_epoch = -1
    patience_counter = 0

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / "best_model.pt"

    training_history: List[Dict[str, Any]] = []
    total_start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_train_loss = 0.0
        total_train_turns = 0

        for batch in train_loader:
            features = batch["features"].to(device)
            party_ids = batch["party_ids"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["label_ids"].to(device)
            lengths = batch["lengths"]

            optimizer.zero_grad()
            logits = model(features, party_ids, mask=mask, lengths=lengths)
            loss = model.compute_loss(logits, targets, mask=mask, class_weights=class_weights)
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

            optimizer.step()

            valid_turns = int(mask.sum().item())
            total_train_loss += loss.item() * valid_turns
            total_train_turns += valid_turns

        avg_train_loss = total_train_loss / total_train_turns if total_train_turns > 0 else 0.0

        # Evaluate on Dev set
        dev_loss, dev_metrics, _ = evaluate_model(model, dev_loader, device, class_weights=class_weights)
        dev_acc = dev_metrics["accuracy"]
        dev_macro_f1 = dev_metrics["macro_f1"]
        dev_weighted_f1 = dev_metrics["weighted_f1"]

        epoch_duration = time.time() - t0
        peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

        training_history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "dev_loss": dev_loss,
            "dev_accuracy": dev_acc,
            "dev_macro_f1": dev_macro_f1,
            "dev_weighted_f1": dev_weighted_f1,
            "duration_sec": epoch_duration,
            "peak_rss_mb": peak_rss_mb,
        })

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Dev Loss: {dev_loss:.4f} | "
            f"Dev Acc: {dev_acc:.4f} | "
            f"Dev Macro F1: {dev_macro_f1:.4f} | "
            f"Dev Weighted F1: {dev_weighted_f1:.4f} | "
            f"Time: {epoch_duration:.2f}s | "
            f"RAM: {peak_rss_mb:.1f} MB"
        )

        # Save training history after every epoch
        exp_output_dir = Path("outputs/experiments") / args.experiment_name
        exp_output_dir.mkdir(parents=True, exist_ok=True)
        with open(exp_output_dir / "training_history.json", "w") as f:
            json.dump(training_history, f, indent=2)

        # Atomic checkpoint save on Dev Macro F1 improvement
        if dev_macro_f1 > best_dev_macro_f1:
            best_dev_macro_f1 = dev_macro_f1
            best_epoch = epoch
            patience_counter = 0

            checkpoint_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config.to_dict(),
                "best_dev_macro_f1": best_dev_macro_f1,
                "best_dev_metrics": dev_metrics,
                "class_weights": class_weights.tolist() if class_weights is not None else None,
                "seed": args.seed,
            }

            temp_path = checkpoint_dir / "temp_best_model.pt"
            torch.save(checkpoint_state, temp_path)
            os.replace(temp_path, best_checkpoint_path)
            # Also save copy in experiment output directory
            shutil.copyfile(best_checkpoint_path, exp_output_dir / "best_model.pt")
            print(f"  --> [BEST] New best Dev Macro F1: {best_dev_macro_f1:.4f} (Saved to {best_checkpoint_path})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered after {epoch} epochs (patience={args.patience}).")
                break

    total_training_time = time.time() - total_start_time
    print("=" * 80)
    print(f"TRAINING COMPLETE. Best Epoch: {best_epoch} with Dev Macro F1: {best_dev_macro_f1:.4f}")
    print(f"Total Training Duration: {total_training_time:.2f}s ({total_training_time / 60.0:.2f} min)")
    print("=" * 80)

    # 6. Load best checkpoint for validation & canonical test evaluation
    print(f"Loading best checkpoint from {best_checkpoint_path}...")
    saved_state = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(saved_state["model_state_dict"])
    model.eval()

    # 7. Evaluate Best Model on Dev set (to save dev_predictions.csv)
    print("Generating validation set predictions with best checkpoint...")
    dev_loss, dev_eval_metrics, dev_pred_df = evaluate_model(model, dev_loader, device)
    dev_pred_path = exp_output_dir / "dev_predictions.csv"
    dev_pred_df.to_csv(dev_pred_path, index=False)
    with open(exp_output_dir / "dev_metrics.json", "w") as f:
        json.dump(dev_eval_metrics, f, indent=2)
    print(f"Saved validation predictions to: {dev_pred_path}")

    # 8. Evaluate Best Model on Test set
    print("Running canonical test evaluation across all cohorts...")
    test_loss, test_metrics, pred_df = evaluate_model(model, test_loader, device)

    # Evaluate all canonical cohorts
    meld_test_raw_path = Path("data/raw/meld/test_sent_emo.csv")
    cohort_results = evaluate_cohorts(pred_df, meld_test_raw_path)

    # Print summary table for all cohorts
    print("\n" + "=" * 80)
    print(f"CANONICAL COHORT EVALUATION RESULTS: {args.experiment_name}")
    print("=" * 80)
    for cohort_name, c_res in cohort_results.items():
        print(f"\n[{cohort_name}] (N = {c_res['n_turns']})")
        print(f"  Accuracy    : {c_res['accuracy'] * 100:.2f}%")
        print(f"  Macro F1    : {c_res['macro_f1']:.4f}")
        print(f"  Weighted F1 : {c_res['weighted_f1']:.4f}")
        print("  Per-class metrics:")
        print(f"    {'Emotion':<12}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Support':>10}")
        for em, p_data in c_res["per_class"].items():
            print(f"    {em:<12}{p_data['precision']:>10.4f}{p_data['recall']:>10.4f}{p_data['f1']:>10.4f}{p_data['support']:>10d}")

    # 9. Save artifacts to outputs/experiments/{experiment_name}
    pred_path = exp_output_dir / "test_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"\nSaved test predictions to: {pred_path}")

    # Save canonical cohorts evaluation results
    cohort_path = exp_output_dir / "canonical_cohort_evaluation.json"
    with open(cohort_path, "w") as f:
        json.dump(cohort_results, f, indent=2)

    # Save confusion matrix directly
    cm_path = exp_output_dir / "confusion_matrix.json"
    with open(cm_path, "w") as f:
        json.dump(cohort_results["Cohort A (Strict Contiguous Matched Subset)"]["confusion_matrix"], f, indent=2)

    # Save primary metrics via repository save_metrics / save_metadata
    eval_result_obj = EvaluationResult(
        accuracy=cohort_results["Cohort A (Strict Contiguous Matched Subset)"]["accuracy"],
        macro_f1=cohort_results["Cohort A (Strict Contiguous Matched Subset)"]["macro_f1"],
        weighted_f1=cohort_results["Cohort A (Strict Contiguous Matched Subset)"]["weighted_f1"],
        per_class={
            k: ClassMetrics(precision=v["precision"], recall=v["recall"], f1=v["f1"])
            for k, v in cohort_results["Cohort A (Strict Contiguous Matched Subset)"]["per_class"].items()
        },
        confusion_matrix=np.array(cohort_results["Cohort A (Strict Contiguous Matched Subset)"]["confusion_matrix"]),
    )
    save_metrics(eval_result_obj, args.experiment_name)
    save_metadata(
        args.experiment_name,
        best_epoch=best_epoch,
        best_dev_macro_f1=best_dev_macro_f1,
        total_training_time_sec=total_training_time,
        peak_rss_mb=peak_rss_mb,
        test_accuracy=test_metrics["accuracy"],
        test_macro_f1=test_metrics["macro_f1"],
        test_weighted_f1=test_metrics["weighted_f1"],
        cohort_a_accuracy=cohort_results["Cohort A (Strict Contiguous Matched Subset)"]["accuracy"],
        cohort_a_macro_f1=cohort_results["Cohort A (Strict Contiguous Matched Subset)"]["macro_f1"],
        cohort_a_weighted_f1=cohort_results["Cohort A (Strict Contiguous Matched Subset)"]["weighted_f1"],
        cohort_b_accuracy=cohort_results["Cohort B (All Subsequent Sequential Turns)"]["accuracy"],
        cohort_b_macro_f1=cohort_results["Cohort B (All Subsequent Sequential Turns)"]["macro_f1"],
        cohort_b_weighted_f1=cohort_results["Cohort B (All Subsequent Sequential Turns)"]["weighted_f1"],
        hyperparameters=vars(args),
        config=config.to_dict(),
    )
    print(f"Saved evaluation metrics and metadata to: {exp_output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate DialogueRNN on MELD.")
    parser.add_argument("--experiment_name", type=str, default="dialoguernn_unweighted")
    parser.add_argument("--train_features", type=str, default="data/processed/dialoguernn/meld_train_features.npz")
    parser.add_argument("--dev_features", type=str, default="data/processed/dialoguernn/meld_dev_features.npz")
    parser.add_argument("--test_features", type=str, default="data/processed/dialoguernn/meld_test_features.npz")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/dialoguernn_unweighted")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class_weight_method", type=str, default="unweighted", choices=["unweighted", "inverse_frequency", "smoothed"])
    parser.add_argument("--projected_dim", type=int, default=256)
    parser.add_argument("--global_hidden_dim", type=int, default=128)
    parser.add_argument("--party_hidden_dim", type=int, default=128)
    parser.add_argument("--emotion_hidden_dim", type=int, default=64)
    parser.add_argument("--fc_hidden_dim", type=int, default=64)
    parser.add_argument("--no_context_attn", action="store_true")
    parser.add_argument("--no_emotion_attn", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_dialoguernn(args)
