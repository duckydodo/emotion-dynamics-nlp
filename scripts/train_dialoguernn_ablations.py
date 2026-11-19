"""
Controlled DialogueRNN Architectural Ablation Training & Evaluation Script (Task 15).

Trains and evaluates four approved DialogueRNN ablation configurations:
1. Global Context Only (370,695 params)
2. Global Context + Party States (617,223 params)
3. Global Context + Emotion State (452,999 params)
4. Full DialogueRNN Reference (670,855 params)

Experimental Controls:
- Random Seed: 42 (fixed across all runs)
- Features: Frozen DistilBERT [CLS] representations (D=768)
- Loss: Unweighted Cross-Entropy
- Optimizer: AdamW (lr=0.001, weight_decay=0.0001)
- Batch Size: 16 dialogues with padding and sequence masking
- Gradient Clipping: max_norm = 1.0
- Early Stopping: patience = 10 epochs based on Dev Macro F1
- Canonical Evaluation: Cohort A (Strict Contiguous Matched Subset, N=2,328)
- Statistical Inference: Paired dialogue cluster bootstrap (B=10,000) & permutation tests (B=10,000)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from src.data.feature_loader import (
    DialogueBatchDataset,
    dialogue_collate_fn,
    load_dialogue_features,
)
from src.models.dialoguernn import (
    DialogueRNNAblation,
    DialogueRNNAblationConfig,
    build_ablation_model,
)

LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
LABEL_TO_ID = {l: i for i, l in enumerate(LABELS)}
ID_TO_LABEL = {i: l for i, l in enumerate(LABELS)}


def set_seed(seed: int = 42) -> None:
    """Set random seed for strict determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_split(
    model: DialogueRNNAblation,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, Dict[str, Any], pd.DataFrame]:
    """
    Evaluate DialogueRNNAblation over a dataloader.
    Returns average loss, metrics dictionary, and prediction DataFrame.
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
            loss = model.compute_loss(logits, targets, mask=mask)

            batch_valid_turns = int(mask.sum().item())
            total_loss += loss.item() * batch_valid_turns
            total_turns += batch_valid_turns

            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            for b in range(len(lengths)):
                seq_len = lengths[b].item()
                d_id = dialogue_ids[b].item() if torch.is_tensor(dialogue_ids[b]) else int(dialogue_ids[b])
                for t in range(seq_len):
                    u_id = utterance_ids[b, t].item() if torch.is_tensor(utterance_ids[b, t]) else int(utterance_ids[b, t])
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
                        "key": f"({d_id}, {u_id})",
                    })

    avg_loss = total_loss / total_turns if total_turns > 0 else 0.0
    pred_df = pd.DataFrame(records)

    y_true = pred_df["true_emotion"].tolist()
    y_pred = pred_df["pred_emotion"].tolist()

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0))

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
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "total_turns": int(total_turns),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }
    return avg_loss, metrics, pred_df


def train_single_ablation(
    config_path: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Train a single ablation configuration strictly following Task 15 protocol.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    exp_name = config_data["experiment_name"]
    abl_id = config_data["ablation_id"]
    expected_params = config_data["parameter_count"]
    train_cfg = config_data["training"]
    data_cfg = config_data["data"]

    seed = train_cfg.get("seed", 42)
    epochs = train_cfg.get("epochs", 40)
    batch_size = train_cfg.get("batch_size", 16)
    lr = train_cfg.get("lr", 0.001)
    weight_decay = train_cfg.get("weight_decay", 0.0001)
    grad_clip = train_cfg.get("grad_clip", 1.0)
    patience = train_cfg.get("patience", 10)

    checkpoint_dir = Path(train_cfg["checkpoint_dir"])
    output_dir = Path(train_cfg["output_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"STARTING ABLATION {abl_id}: {exp_name}")
    print(f"Checkpoint Dir: {checkpoint_dir}")
    print(f"Output Dir:     {output_dir}")
    print(f"Hyperparameters: seed={seed}, lr={lr}, weight_decay={weight_decay}, batch_size={batch_size}, epochs={epochs}, patience={patience}")
    print("=" * 80)

    set_seed(seed)

    # 1. Load feature datasets
    print("Loading pre-extracted DistilBERT features...")
    train_dataset = load_dialogue_features(data_cfg["train_features"])
    dev_dataset = load_dialogue_features(data_cfg["dev_features"])
    test_dataset = load_dialogue_features(data_cfg["test_features"])

    train_dialogues = train_dataset.get_dialogues()
    dev_dialogues = dev_dataset.get_dialogues()
    test_dialogues = test_dataset.get_dialogues()
    print(f"Dialogues: Train={len(train_dialogues)}, Dev={len(dev_dialogues)}, Test={len(test_dialogues)}")

    train_loader = DataLoader(
        DialogueBatchDataset(train_dialogues),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dialogue_collate_fn,
    )
    dev_loader = DataLoader(
        DialogueBatchDataset(dev_dialogues),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dialogue_collate_fn,
    )
    test_loader = DataLoader(
        DialogueBatchDataset(test_dialogues),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dialogue_collate_fn,
    )

    # 2. Build model and verify parameters
    model = build_ablation_model(config_path).to(device)
    actual_params = model.count_parameters()
    print(f"Model Initialized. Parameters: {actual_params:,} (Expected: {expected_params:,})")
    assert actual_params == expected_params, f"Parameter mismatch! Actual: {actual_params}, Expected: {expected_params}"

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 3. Training loop with early stopping on Dev Macro F1
    best_dev_macro_f1 = -1.0
    best_epoch = -1
    patience_counter = 0
    best_checkpoint_path = checkpoint_dir / "best_model.pt"

    training_history: List[Dict[str, Any]] = []
    total_start_time = time.time()

    for epoch in range(1, epochs + 1):
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
            loss = model.compute_loss(logits, targets, mask=mask)
            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()

            valid_turns = int(mask.sum().item())
            total_train_loss += loss.item() * valid_turns
            total_train_turns += valid_turns

        avg_train_loss = total_train_loss / total_train_turns if total_train_turns > 0 else 0.0

        # Evaluate on dev set
        dev_loss, dev_metrics, _ = evaluate_split(model, dev_loader, device)
        dev_acc = dev_metrics["accuracy"]
        dev_macro_f1 = dev_metrics["macro_f1"]
        dev_weighted_f1 = dev_metrics["weighted_f1"]
        epoch_sec = time.time() - t0

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "dev_loss": dev_loss,
            "dev_acc": dev_acc,
            "dev_macro_f1": dev_macro_f1,
            "dev_weighted_f1": dev_weighted_f1,
            "epoch_duration_sec": epoch_sec,
        }
        training_history.append(epoch_record)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Dev Loss: {dev_loss:.4f} | "
            f"Dev Acc: {dev_acc:.4f} | "
            f"Dev Macro F1: {dev_macro_f1:.4f} | "
            f"Time: {epoch_sec:.2f}s"
        )

        # Check for improvement
        if dev_macro_f1 > best_dev_macro_f1:
            best_dev_macro_f1 = dev_macro_f1
            best_epoch = epoch
            patience_counter = 0

            checkpoint_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config_data,
                "best_dev_macro_f1": best_dev_macro_f1,
                "best_dev_metrics": dev_metrics,
                "seed": seed,
            }

            temp_path = checkpoint_dir / "temp_best_model.pt"
            torch.save(checkpoint_state, temp_path)
            os.replace(temp_path, best_checkpoint_path)
            # Copy to output directory
            shutil.copyfile(best_checkpoint_path, output_dir / "best_model.pt")
            print(f"  --> [BEST] New best Dev Macro F1: {best_dev_macro_f1:.4f} (Saved to {best_checkpoint_path})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs (patience={patience}).")
                break

    total_training_time = time.time() - total_start_time
    print(f"TRAINING COMPLETE. Best Epoch: {best_epoch} with Dev Macro F1: {best_dev_macro_f1:.4f} in {total_training_time:.2f}s")

    # Save training history and configuration in output directory
    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(training_history, f, indent=2)
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    # 4. Load best checkpoint for final evaluation
    print(f"Loading best checkpoint from {best_checkpoint_path} for final evaluation...")
    best_state = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_state["model_state_dict"])
    model.eval()

    # Re-evaluate dev set to verify exact match
    _, final_dev_metrics, _ = evaluate_split(model, dev_loader, device)
    with open(output_dir / "dev_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_dev_metrics, f, indent=2)

    # 5. Generate full test predictions
    print("Generating test predictions on held-out test split...")
    test_loss, test_metrics, test_pred_df = evaluate_split(model, test_loader, device)
    test_pred_path = output_dir / "test_predictions.csv"
    test_pred_df.to_csv(test_pred_path, index=False)
    print(f"Saved {len(test_pred_df)} test predictions to {test_pred_path}")

    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    return {
        "ablation_id": abl_id,
        "experiment_name": exp_name,
        "best_epoch": best_epoch,
        "best_dev_macro_f1": best_dev_macro_f1,
        "total_training_time_sec": total_training_time,
        "parameter_count": actual_params,
        "checkpoint_path": str(best_checkpoint_path),
        "output_dir": str(output_dir),
        "test_metrics": test_metrics,
        "pred_df": test_pred_df,
    }


def compute_cluster_bootstrap_ci(
    y_true: np.ndarray,
    preds_ref: np.ndarray,
    preds_abl: np.ndarray,
    dialogue_ids: np.ndarray,
    b_iterations: int = 10000,
    seed: int = 42,
) -> Tuple[float, float, float, float]:
    """
    Paired dialogue cluster bootstrap for Delta Macro F1 (Ref - Abl).
    Returns (delta_obs, ci_lower, ci_upper, std_err).
    """
    rng = np.random.RandomState(seed)
    unique_dialogues = np.unique(dialogue_ids)
    num_dialogues = len(unique_dialogues)

    def calc_macro_f1(t: np.ndarray, p: np.ndarray) -> float:
        f1s = []
        for c in range(7):
            tp = np.sum((t == c) & (p == c))
            denom = np.sum(p == c) + np.sum(t == c)
            f1s.append(2.0 * tp / denom if denom > 0 else 0.0)
        return float(np.mean(f1s))

    delta_obs = calc_macro_f1(y_true, preds_ref) - calc_macro_f1(y_true, preds_abl)

    # Map dialogue IDs to turn indices
    d_to_indices = {d: np.where(dialogue_ids == d)[0] for d in unique_dialogues}

    deltas = np.zeros(b_iterations)
    for b in range(b_iterations):
        sampled_d = rng.choice(unique_dialogues, size=num_dialogues, replace=True)
        idx_list = [d_to_indices[d] for d in sampled_d]
        boot_indices = np.concatenate(idx_list)

        t_boot = y_true[boot_indices]
        ref_boot = preds_ref[boot_indices]
        abl_boot = preds_abl[boot_indices]

        deltas[b] = calc_macro_f1(t_boot, ref_boot) - calc_macro_f1(t_boot, abl_boot)

    ci_lower = float(np.percentile(deltas, 2.5))
    ci_upper = float(np.percentile(deltas, 97.5))
    std_err = float(np.std(deltas))

    return delta_obs, ci_lower, ci_upper, std_err


def compute_paired_permutation_test(
    y_true: np.ndarray,
    preds_ref: np.ndarray,
    preds_abl: np.ndarray,
    dialogue_ids: np.ndarray,
    b_iterations: int = 10000,
    seed: int = 42,
) -> float:
    """
    Paired dialogue-level sign-flip permutation test for Delta Macro F1.
    """
    rng = np.random.RandomState(seed)
    unique_dialogues = np.unique(dialogue_ids)
    num_dialogues = len(unique_dialogues)

    def calc_macro_f1(t: np.ndarray, p: np.ndarray) -> float:
        f1s = []
        for c in range(7):
            tp = np.sum((t == c) & (p == c))
            denom = np.sum(p == c) + np.sum(t == c)
            f1s.append(2.0 * tp / denom if denom > 0 else 0.0)
        return float(np.mean(f1s))

    delta_obs = calc_macro_f1(y_true, preds_ref) - calc_macro_f1(y_true, preds_abl)

    d_to_indices = {d: np.where(dialogue_ids == d)[0] for d in unique_dialogues}
    flips = rng.choice([-1, 1], size=(b_iterations, num_dialogues))

    count_extreme = 0
    for b in range(b_iterations):
        p_ref_perm = preds_ref.copy()
        p_abl_perm = preds_abl.copy()

        flip_mask = flips[b] == -1
        swap_dialogues = unique_dialogues[flip_mask]

        if len(swap_dialogues) > 0:
            swap_indices = np.concatenate([d_to_indices[d] for d in swap_dialogues])
            p_ref_perm[swap_indices] = preds_abl[swap_indices]
            p_abl_perm[swap_indices] = preds_ref[swap_indices]

        delta_perm = calc_macro_f1(y_true, p_ref_perm) - calc_macro_f1(y_true, p_abl_perm)
        if abs(delta_perm) >= abs(delta_obs):
            count_extreme += 1

    p_val = (1.0 + count_extreme) / (1.0 + b_iterations)
    return float(p_val)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate DialogueRNN ablations")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "configs/ablations/1_global_context_only.json",
            "configs/ablations/2_global_party_states.json",
            "configs/ablations/3_global_emotion_state.json",
            "configs/ablations/4_full_dialoguernn.json",
        ],
        help="Path to ablation configuration files",
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"Using execution device: {device}")

    # Train each configuration sequentially
    ablation_results = {}
    for cfg_str in args.configs:
        cfg_path = Path(cfg_str)
        assert cfg_path.exists(), f"Configuration file not found: {cfg_path}"
        res = train_single_ablation(cfg_path, device)
        ablation_results[res["ablation_id"]] = res

    # Perform canonical Cohort A evaluation across all trained ablations
    print("\n" + "=" * 80)
    print("CANONICAL COHORT A EVALUATION (Strict Contiguous Matched Subset, N=2,328)")
    print("=" * 80)

    # 1. Load raw MELD test CSV to construct exact Cohort A keys
    raw_meld_test_path = Path("data/raw/meld/test_sent_emo.csv")
    test_raw = pd.read_csv(raw_meld_test_path)
    test_raw["dialogue_id"] = test_raw["Dialogue_ID"]
    test_raw["utterance_id"] = test_raw["Utterance_ID"]

    cohort_a_keys = []
    for d_id, group in test_raw.groupby("dialogue_id", sort=False):
        group = group.sort_values("utterance_id").reset_index(drop=True)
        for i in range(1, len(group)):
            curr_u = group.loc[i, "utterance_id"]
            prev_u = group.loc[i - 1, "utterance_id"]
            if curr_u == prev_u + 1:
                cohort_a_keys.append((int(d_id), int(curr_u)))

    assert len(cohort_a_keys) == 2328, f"Cohort A expected 2328 keys, got {len(cohort_a_keys)}"
    cohort_a_set = set(cohort_a_keys)
    print(f"Verified canonical Cohort A key count: {len(cohort_a_keys)}")

    # 2. Extract Cohort A predictions for each ablation
    cohort_a_dfs = {}
    for abl_id, res in ablation_results.items():
        df = res["pred_df"].copy()
        df["key_tuple"] = list(zip(df["dialogue_id"], df["utterance_id"]))
        df_a = df[df["key_tuple"].isin(cohort_a_set)].copy()
        # Sort by Cohort A ordered keys
        key_to_order = {k: i for i, k in enumerate(cohort_a_keys)}
        df_a["order"] = df_a["key_tuple"].map(key_to_order)
        df_a = df_a.sort_values("order").reset_index(drop=True)
        assert len(df_a) == 2328, f"Expected 2328 rows for Ablation {abl_id}, got {len(df_a)}"
        cohort_a_dfs[abl_id] = df_a

    # Reference model: Ablation 4 (Full DialogueRNN)
    ref_df = cohort_a_dfs[4]
    y_true_str = ref_df["true_emotion"].values
    y_true_ids = np.array([LABEL_TO_ID[lbl] for lbl in y_true_str])
    dialogue_ids = ref_df["dialogue_id"].values
    ref_pred_ids = np.array([LABEL_TO_ID[lbl] for lbl in ref_df["pred_emotion"].values])

    # 3. Compute metrics for each ablation on Cohort A
    comparison_records = []
    per_class_records = []

    names = {
        1: "Ablation 1: Global Context Only",
        2: "Ablation 2: Global + Party States",
        3: "Ablation 3: Global + Emotion State",
        4: "Ablation 4: Full DialogueRNN Reference",
    }

    raw_p_values = {}
    bootstrap_results = {}

    for abl_id in [1, 2, 3, 4]:
        df_a = cohort_a_dfs[abl_id]
        abl_preds_str = df_a["pred_emotion"].values
        abl_preds_ids = np.array([LABEL_TO_ID[lbl] for lbl in abl_preds_str])

        acc = accuracy_score(y_true_str, abl_preds_str)
        macro_f1 = f1_score(y_true_str, abl_preds_str, labels=LABELS, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true_str, abl_preds_str, labels=LABELS, average="weighted", zero_division=0)

        p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(
            y_true_str, abl_preds_str, labels=LABELS, zero_division=0
        )

        for i, lbl in enumerate(LABELS):
            per_class_records.append({
                "ablation_id": abl_id,
                "model_name": names[abl_id],
                "emotion": lbl,
                "precision": float(p_per[i]),
                "recall": float(r_per[i]),
                "f1": float(f1_per[i]),
                "support": int(sup_per[i]),
            })

        # Paired statistics against Ablation 4 (Full DialogueRNN)
        if abl_id != 4:
            delta_obs, ci_low, ci_high, std_err = compute_cluster_bootstrap_ci(
                y_true_ids, ref_pred_ids, abl_preds_ids, dialogue_ids, b_iterations=10000, seed=42
            )
            p_val = compute_paired_permutation_test(
                y_true_ids, ref_pred_ids, abl_preds_ids, dialogue_ids, b_iterations=10000, seed=42
            )
            raw_p_values[abl_id] = p_val
            bootstrap_results[abl_id] = (delta_obs, ci_low, ci_high, std_err)
        else:
            raw_p_values[abl_id] = 1.0
            bootstrap_results[abl_id] = (0.0, 0.0, 0.0, 0.0)

    # Apply Holm correction across the 3 comparisons against Full Model (Ablations 1, 2, 3)
    p_to_correct = [raw_p_values[1], raw_p_values[2], raw_p_values[3]]
    sorted_order = np.argsort(p_to_correct)
    adj_p = np.zeros(3)
    running_max = 0.0
    for rank, orig_idx in enumerate(sorted_order):
        mult = 3 - rank
        val = min(1.0, p_to_correct[orig_idx] * mult)
        running_max = max(running_max, val)
        adj_p[orig_idx] = min(1.0, running_max)

    holm_dict = {1: adj_p[0], 2: adj_p[1], 3: adj_p[2], 4: 1.0}

    for abl_id in [1, 2, 3, 4]:
        df_a = cohort_a_dfs[abl_id]
        abl_preds_str = df_a["pred_emotion"].values
        acc = accuracy_score(y_true_str, abl_preds_str)
        macro_f1 = f1_score(y_true_str, abl_preds_str, labels=LABELS, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true_str, abl_preds_str, labels=LABELS, average="weighted", zero_division=0)

        delta_obs, ci_low, ci_high, std_err = bootstrap_results[abl_id]
        raw_p = raw_p_values[abl_id]
        holm_p = holm_dict[abl_id]
        sig_05 = (holm_p < 0.05) if abl_id != 4 else False

        comparison_records.append({
            "ablation_id": abl_id,
            "model_name": names[abl_id],
            "parameter_count": ablation_results[abl_id]["parameter_count"],
            "best_epoch": ablation_results[abl_id]["best_epoch"],
            "cohort_a_accuracy": float(acc),
            "cohort_a_macro_f1": float(macro_f1),
            "cohort_a_weighted_f1": float(weighted_f1),
            "delta_macro_f1_vs_full": float(delta_obs),
            "cluster_ci_lower_95": float(ci_low) if abl_id != 4 else 0.0,
            "cluster_ci_upper_95": float(ci_high) if abl_id != 4 else 0.0,
            "cluster_std_err": float(std_err) if abl_id != 4 else 0.0,
            "permutation_p_raw": float(raw_p) if abl_id != 4 else 1.0,
            "permutation_p_holm": float(holm_p) if abl_id != 4 else 1.0,
            "statistically_significant": sig_05,
        })

    # Save CSV artifacts
    tables_dir = Path("outputs/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    df_comp = pd.DataFrame(comparison_records)
    comp_csv_path = tables_dir / "dialoguernn_ablation_comparison.csv"
    df_comp.to_csv(comp_csv_path, index=False)
    print(f"\nSaved comparison table to: {comp_csv_path}")

    df_per_class = pd.DataFrame(per_class_records)
    per_class_csv_path = tables_dir / "dialoguernn_ablation_per_class.csv"
    df_per_class.to_csv(per_class_csv_path, index=False)
    print(f"Saved per-class table to:    {per_class_csv_path}")

    # Print summary table
    print("\n" + "=" * 110)
    print("FINAL ABLATION SUMMARY (Cohort A: N=2,328)")
    print("=" * 110)
    print(f"{'Model Name':<38} | {'Params':<8} | {'Ep':<3} | {'Acc':<6} | {'Macro F1':<8} | {'Delta vs Full':<13} | {'95% Cluster CI':<18} | {'Holm p':<7} | {'Sig'}")
    print("-" * 110)
    for r in comparison_records:
        ci_str = f"[{r['cluster_ci_lower_95']:+.4f}, {r['cluster_ci_upper_95']:+.4f}]" if r["ablation_id"] != 4 else "Reference"
        delta_str = f"{r['delta_macro_f1_vs_full']:+.4f}" if r["ablation_id"] != 4 else "0.0000"
        sig_str = "YES" if r["statistically_significant"] else "NO"
        print(
            f"{r['model_name']:<38} | "
            f"{r['parameter_count']:<8} | "
            f"{r['best_epoch']:<3} | "
            f"{r['cohort_a_accuracy']:.4f} | "
            f"{r['cohort_a_macro_f1']:.4f} | "
            f"{delta_str:<13} | "
            f"{ci_str:<18} | "
            f"{r['permutation_p_holm']:.4f}  | "
            f"{sig_str}"
        )
    print("=" * 110)


if __name__ == "__main__":
    main()
