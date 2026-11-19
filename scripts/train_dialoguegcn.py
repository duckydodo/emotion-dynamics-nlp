#!/usr/bin/env python3
"""
Training, Canonical Cohort Evaluation, and Statistical Benchmarking for DialogueGCN.
Uses frozen DistilBERT [CLS] features, variable-length dialogue batching with sequence packing,
early stopping on Development Macro F1, and canonical Cohort A evaluation.

Usage:
    docker run --rm -v $(pwd):/app_mount -w /app_mount -e PYTHONPATH=/app_mount emotion-dynamics-nlp:week11 \
        python scripts/train_dialoguegcn.py
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
from scipy import stats
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.dialoguegcn import DialogueGCN, DialogueGCNConfig
from src.data.feature_loader import (
    load_dialogue_features,
    DialogueBatchDataset,
    dialogue_collate_fn,
)
from src.models.transformer import LABELS, LABEL_TO_ID, ID_TO_LABEL
from src.models.loss import compute_class_weights


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def verify_alignment(
    features_npz: Path,
    raw_csv: Path,
    split_name: str,
) -> None:
    """Verify feature archive alignment with raw MELD CSV."""
    raw_df = pd.read_csv(raw_csv)
    archive = np.load(features_npz, allow_pickle=True)
    d_ids = archive["dialogue_ids"]
    u_ids = archive["utterance_ids"]
    spks = archive["speakers"]

    assert len(d_ids) == len(raw_df), f"{split_name}: length mismatch {len(d_ids)} vs {len(raw_df)}"
    for i in range(len(d_ids)):
        r_d = int(raw_df.iloc[i]["Dialogue_ID"])
        r_u = int(raw_df.iloc[i]["Utterance_ID"])
        a_d = int(d_ids[i])
        a_u = int(u_ids[i])
        assert (r_d, r_u) == (a_d, a_u), f"{split_name}: alignment mismatch at row {i}: raw ({r_d}, {r_u}) != feat ({a_d}, {a_u})"
    print(f"Alignment Verified: {split_name} ({len(d_ids)} turns across {len(np.unique(d_ids))} dialogues) matches raw CSV 100%.")


def evaluate_model(
    model: DialogueGCN,
    dataloader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor | None = None,
) -> Tuple[float, Dict[str, Any], pd.DataFrame]:
    """
    Evaluate DialogueGCN across a dataset, returning loss, metrics dictionary,
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

            logits = model(features, party_ids=party_ids, mask=mask, lengths=lengths)
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
    """Evaluate predictions on all canonical cohorts (A, B, C, D)."""
    test_raw = pd.read_csv(meld_test_raw_path)
    test_raw["dialogue_id"] = test_raw["Dialogue_ID"]
    test_raw["utterance_id"] = test_raw["Utterance_ID"]

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
        "Cohort A: Strict Contiguous Matched Subset": cohort_a_keys,
        "Cohort B: All Subsequent Sequential Turns": cohort_b_keys,
        "Cohort C: Dialogue Initial Turns (t=0)": cohort_c_keys,
        "Cohort D: Full Test Set": cohort_d_keys,
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
            "num_turns": len(subset),
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
        }

    return results


def run_paired_dialogue_permutation_test(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    n_permutations: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Paired dialogue-cluster permutation test evaluating Macro F1 differences.
    Permutes dialogue assignment (swapping all utterances in dialogue d between model A and B).
    """
    rng = np.random.RandomState(seed)

    merged = pd.merge(
        df_a[["dialogue_id", "utterance_id", "true_emotion", "pred_emotion"]].rename(columns={"pred_emotion": "pred_a"}),
        df_b[["dialogue_id", "utterance_id", "pred_emotion"]].rename(columns={"pred_emotion": "pred_b"}),
        on=["dialogue_id", "utterance_id"],
    )

    y_true = merged["true_emotion"].values
    pred_a = merged["pred_a"].values
    pred_b = merged["pred_b"].values
    dialogue_ids = merged["dialogue_id"].values

    orig_f1_a = f1_score(y_true, pred_a, labels=LABELS, average="macro", zero_division=0)
    orig_f1_b = f1_score(y_true, pred_b, labels=LABELS, average="macro", zero_division=0)
    orig_diff = orig_f1_a - orig_f1_b

    unique_dialogues = np.unique(dialogue_ids)
    n_dialogues = len(unique_dialogues)

    # Pre-map dialogue to indices
    d_indices = {d: np.where(dialogue_ids == d)[0] for d in unique_dialogues}

    diffs = np.zeros(n_permutations)
    for i in range(n_permutations):
        swap_mask = rng.rand(n_dialogues) < 0.5
        perm_a = pred_a.copy()
        perm_b = pred_b.copy()

        for d_idx, should_swap in enumerate(swap_mask):
            if should_swap:
                idx = d_indices[unique_dialogues[d_idx]]
                perm_a[idx] = pred_b[idx]
                perm_b[idx] = pred_a[idx]

        f1_a_p = f1_score(y_true, perm_a, labels=LABELS, average="macro", zero_division=0)
        f1_b_p = f1_score(y_true, perm_b, labels=LABELS, average="macro", zero_division=0)
        diffs[i] = f1_a_p - f1_b_p

    p_val = float(np.mean(np.abs(diffs) >= np.abs(orig_diff)))
    return {
        "orig_diff": float(orig_diff),
        "f1_a": float(orig_f1_a),
        "f1_b": float(orig_f1_b),
        "p_value": p_val,
    }


def run_dialogue_cluster_bootstrap(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    n_bootstraps: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """Dialogue cluster bootstrap 95% Confidence Interval for Macro F1 difference."""
    rng = np.random.RandomState(seed)

    merged = pd.merge(
        df_a[["dialogue_id", "utterance_id", "true_emotion", "pred_emotion"]].rename(columns={"pred_emotion": "pred_a"}),
        df_b[["dialogue_id", "utterance_id", "pred_emotion"]].rename(columns={"pred_emotion": "pred_b"}),
        on=["dialogue_id", "utterance_id"],
    )

    dialogue_groups = [group for _, group in merged.groupby("dialogue_id")]
    n_dialogues = len(dialogue_groups)

    diffs = []
    for _ in range(n_bootstraps):
        sampled_indices = rng.choice(n_dialogues, size=n_dialogues, replace=True)
        boot_df = pd.concat([dialogue_groups[idx] for idx in sampled_indices], ignore_index=True)
        f1_a = f1_score(boot_df["true_emotion"], boot_df["pred_a"], labels=LABELS, average="macro", zero_division=0)
        f1_b = f1_score(boot_df["true_emotion"], boot_df["pred_b"], labels=LABELS, average="macro", zero_division=0)
        diffs.append(f1_a - f1_b)

    ci_lower = float(np.percentile(diffs, 2.5))
    ci_upper = float(np.percentile(diffs, 97.5))
    return {
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def measure_inference_latency(
    model: DialogueGCN,
    dialogues: list,
    device: torch.device,
    warmup: int = 5,
    runs: int = 20,
) -> Dict[str, float]:
    """Measure inference latency using standard protocol."""
    model.eval()
    sample_dialogue = dialogues[0]
    feat = torch.from_numpy(sample_dialogue.features).float().to(device)
    p_ids = torch.from_numpy(sample_dialogue.party_ids).long().to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(feat, party_ids=p_ids)

    # Timing
    timings = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(feat, party_ids=p_ids)
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)  # ms

    return {
        "mean_latency_ms": float(np.mean(timings)),
        "std_latency_ms": float(np.std(timings)),
        "median_latency_ms": float(np.median(timings)),
        "p95_latency_ms": float(np.percentile(timings, 95)),
        "turns_in_sample": int(len(p_ids)),
    }


def main():
    parser = argparse.ArgumentParser(description="Train DialogueGCN on MELD.")
    parser.add_argument("--feature_dir", type=str, default="data/processed/dialoguernn")
    parser.add_argument("--data_dir", type=str, default="data/raw/meld")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/dialoguegcn")
    parser.add_argument("--output_dir", type=str, default="outputs/experiments/dialoguegcn")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--window_past", type=int, default=10)
    parser.add_argument("--window_future", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    print("=" * 80)
    print("TASK 25: DIALOGUEGCN MELD TRAINING AND CANONICAL EVALUATION")
    print(f"Random Seed: {args.seed} | Device: {device} | Batch Size: {args.batch_size}")
    print("=" * 80)

    # 1. Verify alignment
    train_npz = Path(args.feature_dir) / "meld_train_features.npz"
    dev_npz = Path(args.feature_dir) / "meld_dev_features.npz"
    test_npz = Path(args.feature_dir) / "meld_test_features.npz"

    train_csv = Path(args.data_dir) / "train_sent_emo.csv"
    dev_csv = Path(args.data_dir) / "dev_sent_emo.csv"
    test_csv = Path(args.data_dir) / "test_sent_emo.csv"

    verify_alignment(train_npz, train_csv, "Train")
    verify_alignment(dev_npz, dev_csv, "Dev")
    verify_alignment(test_npz, test_csv, "Test")

    # 2. Load feature datasets
    train_dataset = load_dialogue_features(train_npz)
    dev_dataset = load_dialogue_features(dev_npz)
    test_dataset = load_dialogue_features(test_npz)

    train_dialogues = train_dataset.get_dialogues()
    dev_dialogues = dev_dataset.get_dialogues()
    test_dialogues = test_dataset.get_dialogues()

    if args.dry_run:
        print("[DRY RUN] Truncating dialogues for test...")
        train_dialogues = train_dialogues[:16]
        dev_dialogues = dev_dialogues[:8]
        test_dialogues = test_dialogues[:8]
        args.epochs = 2

    print(f"Loaded Dialogues: Train={len(train_dialogues)}, Dev={len(dev_dialogues)}, Test={len(test_dialogues)}")

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

    # 4. Initialize DialogueGCN
    config = DialogueGCNConfig(
        feature_dim=train_dataset.feature_dim,
        hidden_dim=args.hidden_dim,
        num_classes=len(LABELS),
        num_relations=2,
        num_layers=args.num_layers,
        dropout=args.dropout,
        window_past=args.window_past,
        window_future=args.window_future,
    )
    model = DialogueGCN(config).to(device)
    param_count = model.count_parameters()
    print(f"Model Initialized. Total Trainable Parameters: {param_count:,}")
    assert param_count == 452487, f"Expected 452487 params, got {param_count}"

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 5. Training loop
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / "best_model.pt"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_dev_macro_f1 = -1.0
    best_epoch = -1
    patience_counter = 0
    training_history: List[Dict[str, Any]] = []
    total_start_time = time.time()

    print("\nStarting Training with Early Stopping (Patience = 10 on Dev Macro F1)...")
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
            logits = model(features, party_ids=party_ids, mask=mask, lengths=lengths)
            loss = model.compute_loss(logits, targets, mask=mask)
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

            optimizer.step()

            valid_turns = int(mask.sum().item())
            total_train_loss += loss.item() * valid_turns
            total_train_turns += valid_turns

        avg_train_loss = total_train_loss / total_train_turns if total_train_turns > 0 else 0.0

        # Evaluate on Dev set
        dev_loss, dev_metrics, _ = evaluate_model(model, dev_loader, device)
        dev_acc = dev_metrics["accuracy"]
        dev_macro_f1 = dev_metrics["macro_f1"]
        dev_weighted_f1 = dev_metrics["weighted_f1"]
        epoch_duration = time.time() - t0

        training_history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "dev_loss": dev_loss,
            "dev_accuracy": dev_acc,
            "dev_macro_f1": dev_macro_f1,
            "dev_weighted_f1": dev_weighted_f1,
            "duration_sec": epoch_duration,
        })

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Dev Loss: {dev_loss:.4f} | "
            f"Dev Acc: {dev_acc:.4f} | "
            f"Dev Macro F1: {dev_macro_f1:.4f} | "
            f"Dev Weighted F1: {dev_weighted_f1:.4f} | "
            f"Time: {epoch_duration:.2f}s"
        )

        # Checkpoint selection strictly on Dev Macro F1
        if dev_macro_f1 > best_dev_macro_f1:
            best_dev_macro_f1 = dev_macro_f1
            best_epoch = epoch
            patience_counter = 0

            checkpoint_meta = {
                "epoch": epoch,
                "best_dev_macro_f1": best_dev_macro_f1,
                "best_dev_metrics": dev_metrics,
                "seed": args.seed,
                "architecture": "DialogueGCN",
            }
            model.save_checkpoint(best_checkpoint_path, optimizer=optimizer, metadata=checkpoint_meta)
            shutil.copyfile(best_checkpoint_path, output_dir / "best_model.pt")
            config.to_json(checkpoint_dir / "config.json")
            config.to_json(output_dir / "config.json")
            print(f"  --> [BEST] New best Dev Macro F1: {best_dev_macro_f1:.4f} (Saved to {best_checkpoint_path})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered after {epoch} epochs (patience={args.patience}).")
                break

    total_training_time = time.time() - total_start_time
    print("=" * 80)
    print(f"TRAINING COMPLETE. Best Epoch: {best_epoch} with Dev Macro F1: {best_dev_macro_f1:.4f}")
    print(f"Total Duration: {total_training_time:.2f}s ({total_training_time / 60.0:.2f} min)")
    print("=" * 80)

    # 6. Save training log
    with open(output_dir / "training_log.json", "w") as f:
        json.dump(training_history, f, indent=2)

    # 7. Load best frozen checkpoint for canonical evaluation
    print(f"\nLoading best frozen checkpoint from {best_checkpoint_path}...")
    best_model, meta = DialogueGCN.load_checkpoint(best_checkpoint_path, device=device)

    # 8. Generate validation predictions
    print("Evaluating best model on Dev split...")
    dev_loss, dev_eval_metrics, dev_pred_df = evaluate_model(best_model, dev_loader, device)
    dev_pred_df.to_csv(output_dir / "dev_predictions.csv", index=False)
    with open(output_dir / "dev_metrics.json", "w") as f:
        json.dump(dev_eval_metrics, f, indent=2)

    # 9. Generate test predictions across all cohorts
    print("Evaluating best model on Test split across all cohorts...")
    test_loss, test_metrics, pred_df = evaluate_model(best_model, test_loader, device)
    pred_df.to_csv(output_dir / "test_predictions.csv", index=False)

    cohort_results = evaluate_cohorts(pred_df, test_csv)
    with open(output_dir / "cohort_metrics.json", "w") as f:
        json.dump(cohort_results, f, indent=2)

    print("\n" + "=" * 80)
    print("CANONICAL TEST EVALUATION RESULTS ACROSS COHORTS")
    print("=" * 80)
    print(f"{'Cohort':<45} | {'Turns':<6} | {'Accuracy':<8} | {'Macro F1':<8} | {'Weighted F1':<10}")
    print("-" * 86)
    for c_name, c_res in cohort_results.items():
        print(f"{c_name:<45} | {c_res['num_turns']:<6} | {c_res['accuracy']:<8.4f} | {c_res['macro_f1']:<8.4f} | {c_res['weighted_f1']:<10.4f}")
    print("=" * 80)

    # 10. Per-class metrics table for Cohort A
    cohort_a_metrics = cohort_results["Cohort A: Strict Contiguous Matched Subset"]
    per_class_rows = []
    for emotion, metrics in cohort_a_metrics["per_class"].items():
        per_class_rows.append({
            "Model": "DialogueGCN",
            "Cohort": "Cohort A: Strict Contiguous Matched Subset",
            "Emotion": emotion,
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1"],
            "Support": metrics["support"],
        })
    per_class_df = pd.DataFrame(per_class_rows)
    tables_dir = Path("outputs/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    per_class_df.to_csv(tables_dir / "dialoguegcn_per_class.csv", index=False)

    # 11. Confusion matrix for Cohort A
    cm_a = cohort_a_metrics["confusion_matrix"]
    with open(tables_dir / "dialoguegcn_confusion_matrix.json", "w") as f:
        json.dump({"Cohort A": cm_a}, f, indent=2)

    # Plot Confusion Matrix
    figures_dir = Path("outputs/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_a, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(LABELS)),
        yticks=np.arange(len(LABELS)),
        xticklabels=LABELS,
        yticklabels=LABELS,
        title="DialogueGCN Confusion Matrix (Cohort A, N=2,328)",
        ylabel="True Emotion",
        xlabel="Predicted Emotion",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, format(cm_a[i][j], "d"), ha="center", va="center",
                    color="white" if cm_a[i][j] > np.max(cm_a) / 2 else "black")
    fig.tight_layout()
    fig.savefig(figures_dir / "dialoguegcn_confusion_matrix.png", dpi=300)
    plt.close(fig)
    print(f"Saved Confusion Matrix figure to: {figures_dir / 'dialoguegcn_confusion_matrix.png'}")

    # 12. Statistical Benchmarking against existing models on Cohort A
    print("\nRunning Statistical Benchmarking on Cohort A (N=2,328)...")
    pred_df["key"] = list(zip(pred_df["dialogue_id"], pred_df["utterance_id"]))
    test_raw = pd.read_csv(test_csv)
    cohort_a_keys = set()
    for _, group in test_raw.groupby("Dialogue_ID", sort=False):
        group = group.sort_values("Utterance_ID").reset_index(drop=True)
        for i, row in group.iterrows():
            if i > 0 and row["Utterance_ID"] == group.iloc[i - 1]["Utterance_ID"] + 1:
                cohort_a_keys.add((int(row["Dialogue_ID"]), int(row["Utterance_ID"])))

    gcn_cohort_a = pred_df[pred_df["key"].isin(cohort_a_keys)].copy()
    if not args.dry_run:
        assert len(gcn_cohort_a) == 2328, f"Expected 2328, got {len(gcn_cohort_a)}"

    benchmarks = {
        "DistilBERT Utterance Baseline": PROJECT_ROOT / "outputs/experiments/transformer_baseline/predictions.csv",
        "DistilBERT Context-1": PROJECT_ROOT / "outputs/experiments/transformer_context_1/predictions.csv",
        "DistilBERT Context-1 + Speaker": PROJECT_ROOT / "outputs/experiments/transformer_context_1_speaker/predictions.csv",
        "DialogueRNN (Unweighted)": PROJECT_ROOT / "outputs/experiments/dialoguernn_unweighted/test_predictions.csv",
        "DialogueRNN (Class-Weighted)": PROJECT_ROOT / "outputs/experiments/dialoguernn_weighted/test_predictions.csv",
    }

    stat_rows = []
    for name, path in benchmarks.items():
        if not path.exists():
            print(f"Skipping {name}: file {path} not found.")
            continue

        b_df = pd.read_csv(path)
        pred_col = "predicted_emotion" if "predicted_emotion" in b_df.columns else "pred_emotion"
        b_df["pred_emotion"] = b_df[pred_col]
        b_df["key"] = list(zip(b_df["dialogue_id"].astype(int), b_df["utterance_id"].astype(int)))
        b_cohort_a = b_df[b_df["key"].isin(cohort_a_keys)].copy()
        if not args.dry_run:
            assert len(b_cohort_a) == 2328, f"{name}: expected 2328 turns, got {len(b_cohort_a)}"

        # McNemar test
        merged_test = pd.merge(
            gcn_cohort_a[["key", "true_emotion", "pred_emotion"]].rename(columns={"pred_emotion": "pred_gcn"}),
            b_cohort_a[["key", "pred_emotion"]].rename(columns={"pred_emotion": "pred_b"}),
            on="key",
        )
        correct_gcn = (merged_test["pred_gcn"] == merged_test["true_emotion"])
        correct_b = (merged_test["pred_b"] == merged_test["true_emotion"])
        b_count = int(np.sum(correct_gcn & (~correct_b)))
        c_count = int(np.sum((~correct_gcn) & correct_b))
        n_disc = b_count + c_count
        chi2 = ((abs(b_count - c_count) - 1.0) ** 2) / n_disc if n_disc > 0 else 0.0
        mcnemar_p = float(stats.chi2.sf(chi2, df=1)) if n_disc > 0 else 1.0

        # Permutation test
        perm_res = run_paired_dialogue_permutation_test(gcn_cohort_a, b_cohort_a, n_permutations=10000, seed=42)
        boot_res = run_dialogue_cluster_bootstrap(gcn_cohort_a, b_cohort_a, n_bootstraps=2000, seed=42)

        stat_rows.append({
            "Comparison": f"DialogueGCN vs {name}",
            "N": 2328,
            "DialogueGCN_Macro_F1": perm_res["f1_a"],
            "Benchmark_Macro_F1": perm_res["f1_b"],
            "Delta_Macro_F1": perm_res["orig_diff"],
            "Bootstrap_CI_Lower": boot_res["ci_lower"],
            "Bootstrap_CI_Upper": boot_res["ci_upper"],
            "Permutation_p_value": perm_res["p_value"],
            "McNemar_b": b_count,
            "McNemar_c": c_count,
            "McNemar_p": mcnemar_p,
        })

    stat_df = pd.DataFrame(stat_rows)
    stat_df.to_csv(tables_dir / "dialoguegcn_statistical_tests.csv", index=False)
    print("\n" + "=" * 80)
    print("STATISTICAL BENCHMARKING RESULTS (COHORT A, N=2,328)")
    print("=" * 80)
    for _, row in stat_df.iterrows():
        print(f"{row['Comparison']}:")
        print(f"  Delta Macro F1: {row['Delta_Macro_F1']:+.4f} (95% CI: [{row['Bootstrap_CI_Lower']:+.4f}, {row['Bootstrap_CI_Upper']:+.4f}])")
        print(f"  Dialogue-Cluster Permutation p-value: {row['Permutation_p_value']:.4f}")
        print(f"  McNemar p-value: {row['McNemar_p']:.4f} (b={row['McNemar_b']}, c={row['McNemar_c']})\n")

    # 13. Inference Latency Benchmark
    print("Measuring inference latency on CPU...")
    latency_res = measure_inference_latency(best_model, test_dialogues, device)
    print(f"CPU Latency (per dialogue, {latency_res['turns_in_sample']} turns): {latency_res['mean_latency_ms']:.2f} ms ± {latency_res['std_latency_ms']:.2f} ms")

    # 14. Save metadata
    reproducibility_meta = {
        "model_architecture": "DialogueGCN",
        "random_seed": args.seed,
        "parameters": param_count,
        "epochs_trained": best_epoch,
        "best_dev_macro_f1": best_dev_macro_f1,
        "cohort_a_accuracy": cohort_a_metrics["accuracy"],
        "cohort_a_macro_f1": cohort_a_metrics["macro_f1"],
        "cohort_a_weighted_f1": cohort_a_metrics["weighted_f1"],
        "total_training_duration_sec": total_training_time,
        "cpu_latency": latency_res,
        "hyperparameters": vars(args),
    }
    with open(output_dir / "reproducibility_metadata.json", "w") as f:
        json.dump(reproducibility_meta, f, indent=2)

    # 15. Save comparison summary table
    summary_rows = [
        {
            "Model": "DistilBERT Utterance Baseline",
            "Parameters": 66960000,
            "Receptive_Field": "Single utterance",
            "Cohort_A_Accuracy": 0.5945,
            "Cohort_A_Macro_F1": 0.4314,
            "Cohort_A_Weighted_F1": 0.5794,
        },
        {
            "Model": "DistilBERT Context-1",
            "Parameters": 66960000,
            "Receptive_Field": "t-1 + t",
            "Cohort_A_Accuracy": 0.6259,
            "Cohort_A_Macro_F1": 0.3912,
            "Cohort_A_Weighted_F1": 0.6053,
        },
        {
            "Model": "DistilBERT Context-1 + Speaker",
            "Parameters": 66960000,
            "Receptive_Field": "t-1 + t + Speaker",
            "Cohort_A_Accuracy": 0.6134,
            "Cohort_A_Macro_F1": 0.4078,
            "Cohort_A_Weighted_F1": 0.5986,
        },
        {
            "Model": "DialogueRNN (Unweighted)",
            "Parameters": 670855,
            "Receptive_Field": "Past 0..t (causal)",
            "Cohort_A_Accuracy": 0.6014,
            "Cohort_A_Macro_F1": 0.4342,
            "Cohort_A_Weighted_F1": 0.5892,
        },
        {
            "Model": "DialogueRNN (Class-Weighted)",
            "Parameters": 670855,
            "Receptive_Field": "Past 0..t (causal)",
            "Cohort_A_Accuracy": 0.5988,
            "Cohort_A_Macro_F1": 0.4284,
            "Cohort_A_Weighted_F1": 0.5857,
        },
        {
            "Model": "DialogueGCN (Ours)",
            "Parameters": param_count,
            "Receptive_Field": "Past & Future +/-10 (graph)",
            "Cohort_A_Accuracy": cohort_a_metrics["accuracy"],
            "Cohort_A_Macro_F1": cohort_a_metrics["macro_f1"],
            "Cohort_A_Weighted_F1": cohort_a_metrics["weighted_f1"],
        },
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(tables_dir / "dialoguegcn_cohort_a_metrics.csv", index=False)
    print("\n" + "=" * 80)
    print("CANONICAL COHORT A MODEL COMPARISON SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)
    print("Task 25 Training, Evaluation, and Statistical Benchmarking Completed Successfully.")


if __name__ == "__main__":
    main()
