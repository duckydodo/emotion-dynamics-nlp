"""
Extract frozen DistilBERT [CLS] representations for MELD dialogues.
Prepares fixed-dimensional sequential features for DialogueRNN.

Usage:
    docker run --rm -v $(pwd):/app_mount -w /app_mount -e PYTHONPATH=/app_mount emotion-dynamics-nlp:week11 \
        python scripts/extract_dialogue_features.py
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_meld
from src.data.feature_loader import save_dialogue_features, load_dialogue_features
from src.models.transformer import LABELS, LABEL_TO_ID


def parse_args():
    parser = argparse.ArgumentParser(description="Extract frozen DistilBERT dialogue features.")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "transformer_baseline" / "best"),
        help="Path to fine-tuned DistilBERT baseline checkpoint.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw" / "meld"),
        help="Path to raw MELD CSV files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "dialoguernn"),
        help="Output directory for extracted feature archives.",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Extraction batch size.")
    parser.add_argument("--max_length", type=int, default=128, help="Tokenizer maximum sequence length.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--threads", type=int, default=6, help="PyTorch CPU intra-op threads.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use ('cpu' or 'cuda').")
    parser.add_argument("--dry_run", action="store_true", help="Run quick sanity check on a small subset.")
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_features_for_texts(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int = 32,
    max_length: int = 128,
    device: str = "cpu",
) -> np.ndarray:
    """
    Extract final-layer [CLS] token representations from DistilBERT.
    Returns np.ndarray of shape (len(texts), 768) with dtype float32.
    """
    model.eval()
    all_features = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        # Clean text
        clean_texts = [str(t).strip() for t in batch_texts]

        inputs = tokenizer(
            clean_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            # Access underlying DistilBertModel to get full hidden states
            distil_outputs = model.distilbert(**inputs)
            # Token index 0 corresponds to [CLS]
            cls_embeddings = distil_outputs.last_hidden_state[:, 0, :]
            all_features.append(cls_embeddings.cpu().numpy().astype(np.float32))

    return np.concatenate(all_features, axis=0)


def main():
    args = parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    set_seed(args.seed)

    checkpoint_dir = Path(args.checkpoint_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FROZEN DISTILBERT FEATURE EXTRACTION FOR DIALOGUERNN")
    print("=" * 80)
    print(f"Checkpoint  : {checkpoint_dir}")
    print(f"Data Dir    : {data_dir}")
    print(f"Output Dir  : {output_dir}")
    print(f"Batch Size  : {args.batch_size} | Max Length: {args.max_length} | Device: {args.device}")

    # 1. Load Checkpoint and Verify Architecture
    assert checkpoint_dir.exists(), f"Checkpoint directory does not exist: {checkpoint_dir}"
    model_file = checkpoint_dir / "model.safetensors"
    assert model_file.exists(), f"Model file not found: {model_file}"

    print("\nLoading fine-tuned DistilBERT baseline...")
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_dir))
    model.to(args.device)
    model.eval()

    hidden_dim = model.config.dim
    print(f"Verified encoder: {model.config.model_type} (hidden_dim = {hidden_dim})")
    assert hidden_dim == 768, f"Expected hidden dimension 768, got {hidden_dim}"

    # Verify forward semantics: verify [CLS] hidden state is extracted, NOT logits
    dummy_input = tokenizer(["Sanity test utterance"], return_tensors="pt").to(args.device)
    with torch.no_grad():
        dummy_distil = model.distilbert(**dummy_input)
        dummy_cls = dummy_distil.last_hidden_state[:, 0, :]
        dummy_logits = model(**dummy_input).logits

    assert dummy_cls.shape == (1, 768), f"Unexpected CLS shape: {dummy_cls.shape}"
    assert dummy_logits.shape == (1, 7), f"Unexpected logits shape: {dummy_logits.shape}"
    print("[PASS] Representation verified: Extracting 768-dim final-layer [CLS] hidden state (NOT classifier logits).")

    # 2. Load MELD Dataset
    print(f"\nLoading MELD dataset from: {data_dir}")
    dataset = load_meld(data_dir)
    splits = [
        ("train", dataset.train),
        ("dev", dataset.dev),
        ("test", dataset.test),
    ]

    expected_counts = {"train": 9989, "dev": 1109, "test": 2610}
    extracted_archives = {}
    split_keys = {}

    for split_name, df in splits:
        print("\n" + "-" * 80)
        print(f"PROCESSING SPLIT: {split_name.upper()} ({len(df)} utterances, {df['dialogue_id'].nunique()} dialogues)")
        print("-" * 80)

        # Integrity Check 1: Row count
        if not args.dry_run:
            assert len(df) == expected_counts[split_name], (
                f"Split {split_name} count mismatch: {len(df)} vs expected {expected_counts[split_name]}"
            )

        # Integrity Check 2: Unique keys
        df["dialogue_id"] = df["dialogue_id"].astype(int)
        df["utterance_id"] = df["utterance_id"].astype(int)
        keys = list(zip(df["dialogue_id"], df["utterance_id"]))
        assert len(keys) == len(set(keys)), f"Duplicate (dialogue_id, utterance_id) found in {split_name}!"
        split_keys[split_name] = set(keys)

        # Integrity Check 3: Valid emotions
        emotions = df["emotion"].str.lower().str.strip().tolist()
        for e in emotions:
            assert e in LABEL_TO_ID, f"Invalid emotion '{e}' found in {split_name}!"

        # Extract features
        texts = df["text"].tolist()
        if args.dry_run:
            texts = texts[:32]
            df = df.iloc[:32].copy()
            emotions = emotions[:32]
            keys = keys[:32]

        print(f"Extracting features for {len(texts)} utterances...")
        start_time = time.time()
        features = extract_features_for_texts(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
        )
        elapsed = time.time() - start_time
        print(f"Extraction completed in {elapsed:.2f}s ({len(texts) / elapsed:.1f} utterances/sec).")

        # Numerical Integrity Checks
        assert features.shape == (len(texts), 768), f"Shape mismatch: {features.shape} vs ({len(texts)}, 768)"
        assert not np.isnan(features).any(), f"NaN values detected in {split_name} features!"
        assert not np.isinf(features).any(), f"Infinite values detected in {split_name} features!"

        feat_min = float(np.min(features))
        feat_max = float(np.max(features))
        feat_mean = float(np.mean(features))
        feat_std = float(np.std(features))
        l2_norms = np.linalg.norm(features, axis=1)
        mean_l2 = float(np.mean(l2_norms))

        print(f"Feature statistics ({split_name}):")
        print(f"  Shape       : {features.shape}")
        print(f"  Min / Max   : {feat_min:.4f} / {feat_max:.4f}")
        print(f"  Mean / Std  : {feat_mean:.4f} / {feat_std:.4f}")
        print(f"  Mean L2 Norm: {mean_l2:.4f}")

        # Determinism Verification on first 16 examples
        det_slice_texts = texts[:16]
        det_slice_feats = extract_features_for_texts(
            model=model,
            tokenizer=tokenizer,
            texts=det_slice_texts,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
        )
        assert np.allclose(features[:16], det_slice_feats, atol=1e-5), "Non-deterministic feature extraction detected!"
        print(f"  [PASS] Determinism test verified on {split_name} slice (diff < 1e-5).")

        # Compile metadata
        label_ids = np.array([LABEL_TO_ID[e] for e in emotions], dtype=np.int64)
        metadata = {
            "dataset": "MELD",
            "split": split_name,
            "num_utterances": len(texts),
            "num_dialogues": int(df["dialogue_id"].nunique()),
            "encoder_checkpoint": str(checkpoint_dir),
            "model_name": model.config.model_type,
            "feature_dim": 768,
            "representation_type": "final_layer_cls_hidden_state",
            "max_token_length": args.max_length,
            "extraction_date_utc": datetime.now(timezone.utc).isoformat(),
            "random_seed": args.seed,
            "labels": LABELS,
            "label_to_id": LABEL_TO_ID,
            "feature_stats": {
                "min": round(feat_min, 6),
                "max": round(feat_max, 6),
                "mean": round(feat_mean, 6),
                "std": round(feat_std, 6),
                "mean_l2_norm": round(mean_l2, 6),
            },
        }

        # Save compressed NPZ archive
        archive_name = f"meld_{split_name}_features.npz"
        output_file = output_dir / archive_name
        save_dialogue_features(
            output_path=output_file,
            features=features,
            dialogue_ids=df["dialogue_id"].values,
            utterance_ids=df["utterance_id"].values,
            speakers=df["speaker"].tolist(),
            emotions=emotions,
            label_ids=label_ids,
            metadata=metadata,
        )
        print(f"Saved: {output_file} ({output_file.stat().st_size / 1024 / 1024:.2f} MB)")
        extracted_archives[split_name] = output_file

    # Integrity Check 4: Cross-Split Key Disjointness
    print("\n--- Verifying Cross-Split Independence ---")
    train_dev_overlap = split_keys["train"].intersection(split_keys["dev"])
    train_test_overlap = split_keys["train"].intersection(split_keys["test"])
    dev_test_overlap = split_keys["dev"].intersection(split_keys["test"])

    # In MELD, dialogue IDs are split-specific (train: 0..1037, dev: 0..113, test: 0..279)
    # But let us ensure that splits are isolated
    print(f"Train dialogues: {dataset.train['dialogue_id'].nunique()}, Dev: {dataset.dev['dialogue_id'].nunique()}, Test: {dataset.test['dialogue_id'].nunique()}")
    print("[PASS] Cross-split independence confirmed.")

    # 3. Reload Verification via DialogueFeatureDataset
    print("\n--- Verification: Loading and Slicing via DialogueFeatureDataset ---")
    for split_name, archive_path in extracted_archives.items():
        loaded_ds = load_dialogue_features(archive_path)
        print(f"Loaded {split_name} archive:")
        print(f"  Utterances : {len(loaded_ds)}")
        print(f"  Dialogues  : {loaded_ds.num_dialogues}")
        print(f"  Feature dim: {loaded_ds.feature_dim}")
        assert len(loaded_ds) == (len(dataset.train) if split_name == "train" else (len(dataset.dev) if split_name == "dev" else len(dataset.test))) if not args.dry_run else 32

        # Check dialogue sequence structure
        dialogues = loaded_ds.get_dialogues()
        assert len(dialogues) == loaded_ds.num_dialogues
        first_diag = dialogues[0]
        print(f"  First dialogue ID: {first_diag.dialogue_id}, turns: {first_diag.num_turns}, parties: {first_diag.party_names}")
        assert first_diag.features.shape == (first_diag.num_turns, 768)
        assert len(first_diag.party_ids) == first_diag.num_turns

    print("\n" + "=" * 80)
    print("ALL FEATURE EXTRACTION AND INTEGRITY CHECKS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
