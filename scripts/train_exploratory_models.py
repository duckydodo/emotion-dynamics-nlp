#!/usr/bin/env python3
"""
Phase 2 — Exploratory Conversational Models: Temporal CNN and Linear-Chain CRF.
Trains and evaluates both exploratory architectures on pre-extracted DistilBERT features,
reporting performance on Canonical Cohort A (N=2,328) against established baselines.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.feature_loader import (
    load_dialogue_features,
    DialogueBatchDataset,
    dialogue_collate_fn,
)
from src.data.loader import load_meld
from src.data.context import build_context_examples
from src.models.transformer import LABELS, LABEL_TO_ID, ID_TO_LABEL
from src.models.temporal_cnn import TemporalCNN
from src.models.crf import LinearChainCRF
from src.evaluation.metrics import evaluate, print_results


def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_cohort_a_turn_mask(test_dataset, test_df) -> Tuple[np.ndarray, int]:
    """
    Construct a boolean mask over flat test utterances indicating membership in Cohort A.
    """
    test_ctx1 = build_context_examples(test_df, context_turns=1, include_speaker=False)
    cohort_a_pairs = set(zip(test_ctx1["dialogue_id"], test_ctx1["utterance_id"]))

    cohort_mask = np.zeros(len(test_dataset), dtype=bool)
    for idx in range(len(test_dataset)):
        d_id = int(test_dataset.dialogue_ids[idx])
        u_id = int(test_dataset.utterance_ids[idx])
        if (d_id, u_id) in cohort_a_pairs:
            cohort_mask[idx] = True

    assert int(cohort_mask.sum()) == 2328, f"Expected 2,328 Cohort A turns, got {cohort_mask.sum()}"
    return cohort_mask, 2328


# ==============================================================================
# TEMPORAL CNN TRAINING & EVALUATION
# ==============================================================================
def train_temporal_cnn(
    train_loader: DataLoader,
    dev_loader: DataLoader,
    test_loader: DataLoader,
    cohort_mask: np.ndarray,
    device: torch.device,
    epochs: int = 25,
    lr: float = 1e-3,
    checkpoint_dir: Path = None,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("TRAINING TEMPORAL CNN (CAUSAL DILATED CONVOLUTION)")
    print("=" * 80)

    model = TemporalCNN(
        input_dim=768,
        hidden_dim=256,
        num_classes=7,
        kernel_size=3,
        dilations=(1, 2, 4),
        dropout=0.2,
    ).to(device)

    print(f"Model Architecture: 3 Causal Dilated Blocks (k=3, d=[1, 2, 4])")
    print(f"Receptive Field:    {model.receptive_field} turns")
    print(f"Trainable Params:   {model.parameter_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val_macro_f1 = -1.0
    best_model_state = None
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_turns = 0

        for batch in train_loader:
            features = batch["features"].to(device)
            targets = batch["label_ids"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            logits = model(features, mask=mask)
            loss = model.compute_loss(logits, targets, mask=mask)
            loss.backward()
            optimizer.step()

            valid_turns = int(mask.sum().item())
            total_loss += loss.item() * valid_turns
            total_turns += valid_turns

        avg_train_loss = total_loss / max(total_turns, 1)

        # Validate on Dev
        model.eval()
        dev_true, dev_pred = [], []
        with torch.no_grad():
            for batch in dev_loader:
                features = batch["features"].to(device)
                targets = batch["label_ids"].to(device)
                mask = batch["mask"].to(device)

                logits = model(features, mask=mask)
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                targs = targets.cpu().numpy()
                m = mask.cpu().numpy()

                for b in range(len(m)):
                    for t in range(len(m[b])):
                        if m[b, t]:
                            dev_true.append(ID_TO_LABEL[targs[b, t]])
                            dev_pred.append(ID_TO_LABEL[preds[b, t]])

        val_res = evaluate(dev_true, dev_pred, labels=LABELS)
        print(f"Epoch {epoch:2d}/{epochs:2d} | Loss: {avg_train_loss:.4f} | Val Acc: {val_res.accuracy*100:.2f}% | Val Macro F1: {val_res.macro_f1*100:.2f}% | Val Weighted F1: {val_res.weighted_f1*100:.2f}%")

        if val_res.macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_res.macro_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_duration = time.time() - start_time
    print(f"\nTemporal CNN Training Finished in {train_duration:.1f}s. Best Val Macro F1: {best_val_macro_f1*100:.2f}%")

    # Load best checkpoint
    model.load_state_dict(best_model_state)
    if checkpoint_dir:
        torch.save(best_model_state, checkpoint_dir / "temporal_cnn_best.pt")

    # Evaluate on Test (both full and Cohort A)
    model.eval()
    test_true, test_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            targets = batch["label_ids"].to(device)
            mask = batch["mask"].to(device)

            logits = model(features, mask=mask)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            targs = targets.cpu().numpy()
            m = mask.cpu().numpy()

            for b in range(len(m)):
                for t in range(len(m[b])):
                    if m[b, t]:
                        test_true.append(ID_TO_LABEL[targs[b, t]])
                        test_pred.append(ID_TO_LABEL[preds[b, t]])

    full_results = evaluate(test_true, test_pred, labels=LABELS)

    # Filter for Cohort A
    c_true = [test_true[i] for i in range(len(test_true)) if cohort_mask[i]]
    c_pred = [test_pred[i] for i in range(len(test_pred)) if cohort_mask[i]]
    cohort_results = evaluate(c_true, c_pred, labels=LABELS)

    print("\n--- Temporal CNN Cohort A Evaluation (N=2,328) ---")
    print_results(cohort_results)

    return {
        "model": "Temporal CNN (Causal Dilated Conv1D)",
        "parameters": model.parameter_count,
        "receptive_field": model.receptive_field,
        "train_time_sec": round(train_duration, 2),
        "best_val_macro_f1": float(best_val_macro_f1),
        "full_test": {
            "accuracy": float(full_results.accuracy),
            "macro_f1": float(full_results.macro_f1),
            "weighted_f1": float(full_results.weighted_f1),
        },
        "cohort_a": {
            "accuracy": float(cohort_results.accuracy),
            "macro_f1": float(cohort_results.macro_f1),
            "weighted_f1": float(cohort_results.weighted_f1),
            "per_class": {
                cls: {
                    "precision": float(cohort_results.per_class[cls].precision),
                    "recall": float(cohort_results.per_class[cls].recall),
                    "f1": float(cohort_results.per_class[cls].f1),
                }
                for cls in LABELS
            },
            "confusion_matrix": cohort_results.confusion_matrix.tolist(),
        },
    }


# ==============================================================================
# LINEAR-CHAIN CRF TRAINING & EVALUATION
# ==============================================================================
def train_crf(
    train_loader: DataLoader,
    dev_loader: DataLoader,
    test_loader: DataLoader,
    cohort_mask: np.ndarray,
    device: torch.device,
    epochs: int = 25,
    checkpoint_dir: Path = None,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("TRAINING LINEAR-CHAIN CRF")
    print("=" * 80)

    model = LinearChainCRF(
        input_dim=768,
        num_classes=7,
    ).to(device)

    print(f"Model: Linear-Chain CRF on Frozen DistilBERT [CLS] Embeddings")
    print(f"Trainable Params: {model.parameter_count:,} (Emissions: 5,383, Transitions: 63)")

    optimizer = torch.optim.AdamW([
        {"params": model.emission_proj.parameters(), "lr": 2e-3, "weight_decay": 1e-4},
        {"params": [model.transitions, model.start_transitions, model.end_transitions], "lr": 1e-2},
    ])

    best_val_macro_f1 = -1.0
    best_model_state = None
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_turns = 0

        for batch in train_loader:
            features = batch["features"].to(device)
            targets = batch["label_ids"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            emissions = model(features)
            loss = model.compute_loss(emissions, targets, mask=mask)
            loss.backward()
            optimizer.step()

            valid_turns = int(mask.sum().item())
            total_loss += loss.item() * valid_turns
            total_turns += valid_turns

        avg_train_loss = total_loss / max(total_turns, 1)

        # Validate on Dev via Viterbi decoding
        model.eval()
        dev_true, dev_pred = [], []
        with torch.no_grad():
            for batch in dev_loader:
                features = batch["features"].to(device)
                targets = batch["label_ids"].to(device)
                mask = batch["mask"].to(device)

                emissions = model(features)
                decoded_paths = model.decode(emissions, mask=mask)
                targs = targets.cpu().numpy()
                m = mask.cpu().numpy()

                for b in range(len(decoded_paths)):
                    seq_len = len(decoded_paths[b])
                    for t in range(seq_len):
                        dev_true.append(ID_TO_LABEL[targs[b, t]])
                        dev_pred.append(ID_TO_LABEL[decoded_paths[b][t]])

        val_res = evaluate(dev_true, dev_pred, labels=LABELS)
        print(f"Epoch {epoch:2d}/{epochs:2d} | NLL Loss: {avg_train_loss:.4f} | Val Acc: {val_res.accuracy*100:.2f}% | Val Macro F1: {val_res.macro_f1*100:.2f}% | Val Weighted F1: {val_res.weighted_f1*100:.2f}%")

        if val_res.macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_res.macro_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_duration = time.time() - start_time
    print(f"\nCRF Training Finished in {train_duration:.1f}s. Best Val Macro F1: {best_val_macro_f1*100:.2f}%")

    model.load_state_dict(best_model_state)
    if checkpoint_dir:
        torch.save(best_model_state, checkpoint_dir / "crf_best.pt")

    # 1. Global Viterbi Evaluation
    model.eval()
    test_true, test_pred_viterbi, test_pred_causal = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            targets = batch["label_ids"].to(device)
            mask = batch["mask"].to(device)

            emissions = model(features)
            viterbi_paths = model.decode(emissions, mask=mask)
            causal_paths = model.decode_causal(emissions, mask=mask)

            targs = targets.cpu().numpy()
            m = mask.cpu().numpy()

            for b in range(len(viterbi_paths)):
                seq_len = len(viterbi_paths[b])
                for t in range(seq_len):
                    test_true.append(ID_TO_LABEL[targs[b, t]])
                    test_pred_viterbi.append(ID_TO_LABEL[viterbi_paths[b][t]])
                    test_pred_causal.append(ID_TO_LABEL[causal_paths[b][t]])

    # Filter Cohort A for both decoders
    c_true = [test_true[i] for i in range(len(test_true)) if cohort_mask[i]]
    c_pred_viterbi = [test_pred_viterbi[i] for i in range(len(test_pred_viterbi)) if cohort_mask[i]]
    c_pred_causal = [test_pred_causal[i] for i in range(len(test_pred_causal)) if cohort_mask[i]]

    cohort_viterbi_res = evaluate(c_true, c_pred_viterbi, labels=LABELS)
    cohort_causal_res = evaluate(c_true, c_pred_causal, labels=LABELS)

    print("\n--- Linear-Chain CRF Cohort A Evaluation (Global Viterbi, N=2,328) ---")
    print_results(cohort_viterbi_res)

    print("\n--- Linear-Chain CRF Cohort A Evaluation (Causal Online, N=2,328) ---")
    print_results(cohort_causal_res)

    # Transition Matrix summary
    learned_transitions = model.transitions.detach().cpu().numpy().tolist()

    return {
        "model": "Linear-Chain CRF",
        "parameters": model.parameter_count,
        "train_time_sec": round(train_duration, 2),
        "best_val_macro_f1": float(best_val_macro_f1),
        "cohort_a_viterbi": {
            "decoding": "Global Viterbi (Non-Causal Sequence Decoding)",
            "accuracy": float(cohort_viterbi_res.accuracy),
            "macro_f1": float(cohort_viterbi_res.macro_f1),
            "weighted_f1": float(cohort_viterbi_res.weighted_f1),
            "per_class": {
                cls: {
                    "precision": float(cohort_viterbi_res.per_class[cls].precision),
                    "recall": float(cohort_viterbi_res.per_class[cls].recall),
                    "f1": float(cohort_viterbi_res.per_class[cls].f1),
                }
                for cls in LABELS
            },
            "confusion_matrix": cohort_viterbi_res.confusion_matrix.tolist(),
        },
        "cohort_a_causal": {
            "decoding": "Online Greedy (Causal Past Transitions Only)",
            "accuracy": float(cohort_causal_res.accuracy),
            "macro_f1": float(cohort_causal_res.macro_f1),
            "weighted_f1": float(cohort_causal_res.weighted_f1),
        },
        "learned_transitions": learned_transitions,
    }


def main():
    set_seed(42)
    device = torch.device("cpu")
    torch.set_num_threads(6)

    checkpoint_dir = PROJECT_ROOT / "checkpoints" / "exploratory"
    experiment_dir = PROJECT_ROOT / "outputs" / "experiments" / "exploratory"
    tables_dir = PROJECT_ROOT / "outputs" / "tables"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 2: EXPLORATORY CONVERSATIONAL MODELS (TEMPORAL CNN & CRF)")
    print("=" * 80)

    # 1. Load pre-extracted frozen DistilBERT CLS features
    features_dir = PROJECT_ROOT / "data" / "processed" / "dialoguernn"
    print(f"Loading dialogue features from {features_dir}...")
    train_features = load_dialogue_features(features_dir / "meld_train_features.npz")
    dev_features = load_dialogue_features(features_dir / "meld_dev_features.npz")
    test_features = load_dialogue_features(features_dir / "meld_test_features.npz")

    # Load raw test CSV for Cohort A mask derivation
    dataset = load_meld(PROJECT_ROOT / "data" / "raw" / "meld")
    cohort_mask, n_cohort = get_cohort_a_turn_mask(test_features, dataset.test)
    print(f"Verified Cohort A Mask: {n_cohort} turns across {test_features.num_dialogues} dialogues.")

    # DataLoaders (dialogue-batching)
    train_loader = DataLoader(
        DialogueBatchDataset(train_features.get_dialogues()),
        batch_size=32,
        shuffle=True,
        collate_fn=dialogue_collate_fn,
    )
    dev_loader = DataLoader(
        DialogueBatchDataset(dev_features.get_dialogues()),
        batch_size=32,
        shuffle=False,
        collate_fn=dialogue_collate_fn,
    )
    test_loader = DataLoader(
        DialogueBatchDataset(test_features.get_dialogues()),
        batch_size=32,
        shuffle=False,
        collate_fn=dialogue_collate_fn,
    )

    # 2. Run Temporal CNN Experiment
    tcn_results = train_temporal_cnn(
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        cohort_mask=cohort_mask,
        device=device,
        epochs=25,
        lr=1e-3,
        checkpoint_dir=checkpoint_dir,
    )
    with open(experiment_dir / "temporal_cnn_metrics.json", "w") as f:
        json.dump(tcn_results, f, indent=2)

    # 3. Run Linear-Chain CRF Experiment
    crf_results = train_crf(
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        cohort_mask=cohort_mask,
        device=device,
        epochs=25,
        checkpoint_dir=checkpoint_dir,
    )
    with open(experiment_dir / "crf_metrics.json", "w") as f:
        json.dump(crf_results, f, indent=2)

    # 4. Generate Comparative Summary Table
    comparison_rows = [
        {
            "Model": "DistilBERT Baseline (3-Epoch)",
            "Type": "Utterance-only Transformer",
            "Causal": "Yes (Turn Isolated)",
            "Parameters": 66960000,
            "Accuracy": 0.5945,
            "Macro_F1": 0.4314,
            "Weighted_F1": 0.5794,
        },
        {
            "Model": "DialogueRNN",
            "Type": "Recurrent (GRU Party Tracking)",
            "Causal": "Yes (Sequential History)",
            "Parameters": 670855,
            "Accuracy": 0.6014,
            "Macro_F1": 0.4342,
            "Weighted_F1": 0.5892,
        },
        {
            "Model": "DialogueGCN",
            "Type": "Graph Convolutional (Relational)",
            "Causal": "No (Bidirectional Window=10)",
            "Parameters": 452487,
            "Accuracy": 0.6091,
            "Macro_F1": 0.4385,
            "Weighted_F1": 0.6009,
        },
        {
            "Model": "Temporal CNN (Causal)",
            "Type": "Dilated 1D Conv (k=3, d=[1,2,4])",
            "Causal": "Yes (Left-Padded RF=15)",
            "Parameters": tcn_results["parameters"],
            "Accuracy": tcn_results["cohort_a"]["accuracy"],
            "Macro_F1": tcn_results["cohort_a"]["macro_f1"],
            "Weighted_F1": tcn_results["cohort_a"]["weighted_f1"],
        },
        {
            "Model": "Linear-Chain CRF (Viterbi)",
            "Type": "Structured Sequence CRF",
            "Causal": "No (Global Viterbi Decoding)",
            "Parameters": crf_results["parameters"],
            "Accuracy": crf_results["cohort_a_viterbi"]["accuracy"],
            "Macro_F1": crf_results["cohort_a_viterbi"]["macro_f1"],
            "Weighted_F1": crf_results["cohort_a_viterbi"]["weighted_f1"],
        },
        {
            "Model": "Linear-Chain CRF (Causal)",
            "Type": "Structured Sequence CRF",
            "Causal": "Yes (Online Greedy Decoding)",
            "Parameters": crf_results["parameters"],
            "Accuracy": crf_results["cohort_a_causal"]["accuracy"],
            "Macro_F1": crf_results["cohort_a_causal"]["macro_f1"],
            "Weighted_F1": crf_results["cohort_a_causal"]["weighted_f1"],
        },
    ]

    comp_df = pd.DataFrame(comparison_rows)
    comp_file = tables_dir / "exploratory_models_comparison.csv"
    comp_df.to_csv(comp_file, index=False)
    print("\n" + "=" * 80)
    print("EXPLORATORY EXPERIMENTS SUMMARY (CANONICAL COHORT A, N=2,328):")
    print("=" * 80)
    print(comp_df.to_string(index=False))
    print(f"\nSaved comparison table to {comp_file}")


if __name__ == "__main__":
    main()
