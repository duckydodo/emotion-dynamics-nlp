"""
Independent verification script for extracted DialogueRNN features.
Loads the extracted .npz feature archives, validates schema, shape, numerical integrity,
and dialogue sequence reconstruction.

Usage:
    docker run --rm -v $(pwd):/app_mount -w /app_mount -e PYTHONPATH=/app_mount emotion-dynamics-nlp:week11 \
        python scripts/verify_extracted_features.py
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_meld
from src.data.feature_loader import load_dialogue_features
from src.models.transformer import LABELS, LABEL_TO_ID


def main():
    print("=" * 80)
    print("INDEPENDENT VERIFICATION OF EXTRACTED DIALOGUERNN FEATURES")
    print("=" * 80)

    feature_dir = PROJECT_ROOT / "data" / "processed" / "dialoguernn"
    raw_dir = PROJECT_ROOT / "data" / "raw" / "meld"

    assert feature_dir.exists(), f"Feature directory does not exist: {feature_dir}"
    dataset = load_meld(raw_dir)

    splits = [
        ("train", dataset.train, feature_dir / "meld_train_features.npz"),
        ("dev", dataset.dev, feature_dir / "meld_dev_features.npz"),
        ("test", dataset.test, feature_dir / "meld_test_features.npz"),
    ]

    expected_counts = {"train": 9989, "dev": 1109, "test": 2610}
    expected_dialogues = {"train": 1038, "dev": 114, "test": 280}

    results = {}

    for split_name, raw_df, archive_path in splits:
        print(f"\n[Verifying Split: {split_name.upper()}]")
        print(f"Archive: {archive_path}")
        assert archive_path.exists(), f"Missing archive file: {archive_path}"

        ds = load_dialogue_features(archive_path)

        # 1. Count checks
        n_utterances = len(ds)
        n_dialogues = ds.num_dialogues
        print(f"  Utterance Count : {n_utterances} (Expected: {expected_counts[split_name]})")
        print(f"  Dialogue Count  : {n_dialogues} (Expected: {expected_dialogues[split_name]})")
        assert n_utterances == expected_counts[split_name], f"Utterance count mismatch in {split_name}"
        assert n_dialogues == expected_dialogues[split_name], f"Dialogue count mismatch in {split_name}"

        # 2. Shape and Dimension checks
        assert ds.feature_dim == 768, f"Feature dimension mismatch: {ds.feature_dim} vs 768"
        assert ds.features.shape == (expected_counts[split_name], 768)

        # 3. Numerical integrity
        assert not np.isnan(ds.features).any(), f"NaN values detected in {split_name} features!"
        assert not np.isinf(ds.features).any(), f"Inf values detected in {split_name} features!"

        feat_min = float(np.min(ds.features))
        feat_max = float(np.max(ds.features))
        feat_mean = float(np.mean(ds.features))
        feat_std = float(np.std(ds.features))
        l2_norms = np.linalg.norm(ds.features, axis=1)
        mean_l2 = float(np.mean(l2_norms))

        print(f"  Numerical Range : [{feat_min:.4f}, {feat_max:.4f}]")
        print(f"  Mean / Std      : {feat_mean:.4f} / {feat_std:.4f}")
        print(f"  Mean L2 Norm    : {mean_l2:.4f}")

        # 4. Key alignment and ground truth matching
        raw_df["dialogue_id"] = raw_df["dialogue_id"].astype(int)
        raw_df["utterance_id"] = raw_df["utterance_id"].astype(int)
        raw_emotions = raw_df["emotion"].str.lower().str.strip().tolist()

        assert (ds.dialogue_ids == raw_df["dialogue_id"].values).all(), f"Dialogue ID sequence mismatch in {split_name}"
        assert (ds.utterance_ids == raw_df["utterance_id"].values).all(), f"Utterance ID sequence mismatch in {split_name}"
        assert ds.emotions == raw_emotions, f"Emotion labels mismatch in {split_name}"
        assert (ds.label_ids == np.array([LABEL_TO_ID[e] for e in raw_emotions])).all(), f"Label IDs mismatch in {split_name}"

        # 5. Dialogue sequence reconstruction check
        dialogues = ds.get_dialogues()
        assert len(dialogues) == n_dialogues
        total_turns = sum(d.num_turns for d in dialogues)
        assert total_turns == n_utterances

        max_parties = max(len(d.party_names) for d in dialogues)
        min_parties = min(len(d.party_names) for d in dialogues)
        print(f"  Parties per Dialogue: min={min_parties}, max={max_parties}")

        # Check party IDs are valid
        for d in dialogues:
            assert np.min(d.party_ids) == 0
            assert np.max(d.party_ids) == len(d.party_names) - 1
            assert d.features.shape == (d.num_turns, 768)

        print(f"  [PASS] All schema, alignment, numerical and dialogue sequence checks passed.")
        results[split_name] = {
            "utterances": n_utterances,
            "dialogues": n_dialogues,
            "feature_dim": ds.feature_dim,
            "min": round(feat_min, 4),
            "max": round(feat_max, 4),
            "mean": round(feat_mean, 4),
            "std": round(feat_std, 4),
            "mean_l2": round(mean_l2, 4),
            "file_size_mb": round(archive_path.stat().st_size / 1024 / 1024, 2),
        }

    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY TABLE")
    print("=" * 80)
    summary_df = pd.DataFrame(results).T
    print(summary_df.to_string())
    print("\nALL VERIFICATION TESTS COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
