"""
Evaluate existing trained DistilBERT checkpoints on the matched MELD test subset (N=2,328).

Usage:
    docker run --rm -v $(pwd):/app_mount -w /app_mount -e PYTHONPATH=/app_mount emotion-dynamics-nlp:week11 python scripts/evaluate_matched_test.py
"""

from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_meld
from src.data.context import build_context_examples
from src.models.transformer import TransformerEmotionClassifier, LABELS
from src.evaluation.metrics import evaluate, print_results


def main():
    print("=" * 70)
    print("MELD Matched Test Split Evaluation (N = 2,328)")
    print("=" * 70)

    data_dir = PROJECT_ROOT / "data" / "raw" / "meld"
    output_dir = PROJECT_ROOT / "outputs" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading MELD dataset from: {data_dir}")
    dataset = load_meld(data_dir)
    test_df = dataset.test

    # Build context-1 test examples
    test_ctx1 = build_context_examples(test_df, context_turns=1, include_speaker=False)
    test_ctx1_spk = build_context_examples(test_df, context_turns=1, include_speaker=True)

    assert len(test_ctx1) == 2328
    assert len(test_ctx1_spk) == 2328
    print(f"Verified matched test subset: {len(test_ctx1)} utterances (282 excluded: 280 dialogue-initial, 2 gap turns)")

    # Matched baseline targets
    merged_baseline = pd.merge(
        test_df,
        test_ctx1[["dialogue_id", "utterance_id"]],
        on=["dialogue_id", "utterance_id"],
        how="inner",
    )
    assert len(merged_baseline) == 2328

    models_to_eval = [
        {
            "name": "transformer_baseline (matched)",
            "checkpoint": PROJECT_ROOT / "checkpoints" / "transformer_baseline" / "best",
            "texts": merged_baseline["text"].tolist(),
            "labels": merged_baseline["emotion"].tolist(),
            "count": len(merged_baseline),
        },
        {
            "name": "transformer_context_1 (matched)",
            "checkpoint": PROJECT_ROOT / "checkpoints" / "transformer_context_1" / "best",
            "texts": test_ctx1["context"].tolist(),
            "labels": test_ctx1["emotion"].tolist(),
            "count": len(test_ctx1),
        },
        {
            "name": "transformer_context_1_speaker (matched)",
            "checkpoint": PROJECT_ROOT / "checkpoints" / "transformer_context_1_speaker" / "best",
            "texts": test_ctx1_spk["context"].tolist(),
            "labels": test_ctx1_spk["emotion"].tolist(),
            "count": len(test_ctx1_spk),
        },
    ]

    summary_rows = []
    per_class_data = {}

    for item in models_to_eval:
        print(f"\nEvaluating {item['name']}...")
        clf = TransformerEmotionClassifier(model_path=item["checkpoint"])
        preds = clf.predict(item["texts"], batch_size=16)

        res = evaluate(y_true=item["labels"], y_pred=preds, labels=LABELS)

        summary_rows.append({
            "model": item["name"],
            "evaluated_examples": item["count"],
            "accuracy": res.accuracy,
            "macro_f1": res.macro_f1,
            "weighted_f1": res.weighted_f1,
        })

        per_class_data[item["name"]] = res.per_class

    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 70)
    print("MATCHED TEST EVALUATION RESULTS (N = 2,328)")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    summary_csv = output_dir / "matched_test_evaluation.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved matched test summary to {summary_csv}")

    # Build per-class comparison table
    per_class_rows = []
    for emo in LABELS:
        row = {"emotion": emo}
        for item in models_to_eval:
            m_name = item["name"].split()[0].replace("transformer_", "")
            metrics = per_class_data[item["name"]][emo]
            row[f"{m_name}_precision"] = metrics.precision
            row[f"{m_name}_recall"] = metrics.recall
            row[f"{m_name}_f1"] = metrics.f1
        per_class_rows.append(row)

    per_class_df = pd.DataFrame(per_class_rows)
    per_class_csv = output_dir / "matched_test_per_class.csv"
    per_class_df.to_csv(per_class_csv, index=False)
    print(f"Saved matched test per-class results to {per_class_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
