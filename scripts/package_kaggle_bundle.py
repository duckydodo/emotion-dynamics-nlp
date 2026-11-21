#!/usr/bin/env python3
"""
Build lightweight Kaggle dataset upload bundle (~40 MB total).

Bundles:
1. Frozen DistilBERT [CLS] features (meld_train_features.npz, meld_dev_features.npz, meld_test_features.npz)
2. Explainability taxonomy (disagreement_taxonomy.json)
3. Text-only baseline test predictions for Cohort A (DistilBERT, DialogueRNN, DialogueGCN)
"""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_DIR = PROJECT_ROOT / "kaggle_upload_bundle"
ZIP_PATH = PROJECT_ROOT / "kaggle_upload_bundle.zip"

FILES_TO_COPY = [
    # Feature npz files
    ("data/processed/dialoguernn/meld_train_features.npz", "meld_train_features.npz"),
    ("data/processed/dialoguernn/meld_dev_features.npz", "meld_dev_features.npz"),
    ("data/processed/dialoguernn/meld_test_features.npz", "meld_test_features.npz"),
    # Explainability taxonomy
    ("outputs/experiments/explainability/disagreement_taxonomy.json", "disagreement_taxonomy.json"),
    # Baseline test predictions
    ("outputs/experiments/transformer_baseline/predictions.csv", "distilbert_test_predictions.csv"),
    ("outputs/experiments/dialoguernn_unweighted/test_predictions.csv", "dialoguernn_unweighted_test_predictions.csv"),
    ("outputs/experiments/dialoguegcn/test_predictions.csv", "dialoguegcn_test_predictions.csv"),
]


def format_size(size_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def main():
    print("=" * 80)
    print("PACKAGING KAGGLE UPLOAD BUNDLE")
    print("=" * 80)

    if BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

    copied_files = []
    total_uncompressed = 0

    for rel_src, dst_name in FILES_TO_COPY:
        src = PROJECT_ROOT / rel_src
        dst = BUNDLE_DIR / dst_name

        if not src.exists():
            raise FileNotFoundError(f"Required file not found: {src}")

        shutil.copy2(src, dst)
        sz = dst.stat().st_size
        total_uncompressed += sz
        copied_files.append((dst_name, sz))
        print(f"  [+] Copied: {dst_name:<45} ({format_size(sz)})")

    # Add dataset metadata description for Kaggle
    readme_content = """# MELD Multimodal Experiment Companion Bundle
This bundle contains pre-extracted text representations and baseline predictions for the MELD dataset:
- `meld_train_features.npz`, `meld_dev_features.npz`, `meld_test_features.npz`: 768-d frozen DistilBERT [CLS] representations.
- `disagreement_taxonomy.json`: Cross-model disagreement audit across Cohort A (N=2,328).
- `distilbert_test_predictions.csv`, `dialoguernn_unweighted_test_predictions.csv`, `dialoguegcn_test_predictions.csv`: Baseline text-only predictions.
"""
    readme_path = BUNDLE_DIR / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    print("\nCreating zip archive...")
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in BUNDLE_DIR.iterdir():
            zf.write(file_path, arcname=file_path.name)

    zip_size = ZIP_PATH.stat().st_size
    print(f"\n[SUCCESS] Kaggle bundle created successfully:")
    print(f"  Directory: {BUNDLE_DIR} ({format_size(total_uncompressed)})")
    print(f"  Zip File:  {ZIP_PATH} ({format_size(zip_size)})")


if __name__ == "__main__":
    main()
