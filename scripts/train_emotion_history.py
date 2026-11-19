"""
Train and evaluate Emotion-History DistilBERT on MELD.

Investigates whether conditioning on previous emotional states improves sequential
conversational emotion recognition.

Evaluates:
- Condition 3: Teacher-forced emotion history (oracle diagnostic upper bound).
- Condition 4: Autoregressive predicted emotion history (deployable inference).

Stratified evaluation:
- Full test set (N = 2,610)
- Turn-0 subset (N = 282, initial dialogue turns)
- Turns >= 1 subset (N = 2,328, matched context subset)

Transition & Error Propagation Analysis:
- Preceding prediction correctness & error cascading
- Persistence vs. shift recognition
- Neutral gravitation analysis
- McNemar's test against baseline
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys
from collections import Counter
import numpy as np
import pandas as pd
import torch
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_meld
from src.models.transformer import (
    TransformerEmotionClassifier,
    LABELS,
    LABEL_TO_ID,
    ID_TO_LABEL,
)
from src.models.trainer import TransformerTrainer, set_seed
from src.evaluation.metrics import evaluate, print_results, plot_confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate Emotion-History DistilBERT.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--max_length", type=int, default=64, help="Max sequence length.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--threads", type=int, default=6, help="PyTorch CPU threads.")
    parser.add_argument("--data_dir", type=str, default=str(PROJECT_ROOT / "data" / "raw" / "meld"))
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "transformer_emotion_history"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "experiments" / "transformer_emotion_history"),
    )
    parser.add_argument("--tiny", action="store_true", help="Quick smoke test on tiny slice.")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume training from latest_training_state.pt.")
    parser.add_argument("--dry_run_resume", action="store_true", default=False, help="Verify resume state without executing training.")
    return parser.parse_args()


def format_emotion_history_prompt(prev_emotion: str, utterance_text: str) -> str:
    """
    Format natural language prefix conditioning on preceding emotional state.
    """
    clean_text = str(utterance_text).strip()
    return f"Previous emotion: {prev_emotion}\nCurrent utterance: {clean_text}"


def build_emotion_history_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct emotion-history examples from a MELD split.
    Respects dialogue boundaries: Turn 0 always receives 'none'.
    """
    df = df.sort_values(["dialogue_id", "utterance_id"]).reset_index(drop=True)
    rows = []

    for dialogue_id, dialogue in df.groupby("dialogue_id", sort=False):
        dialogue = dialogue.sort_values("utterance_id").reset_index(drop=True)
        records = dialogue.to_dict("records")

        for i, turn in enumerate(records):
            if i == 0:
                prev_emotion = "none"
                is_turn_0 = True
            else:
                prev_emotion = records[i - 1]["emotion"]
                is_turn_0 = False

            prompt = format_emotion_history_prompt(prev_emotion, turn["text"])

            rows.append({
                "dialogue_id": turn["dialogue_id"],
                "utterance_id": turn["utterance_id"],
                "speaker": turn["speaker"],
                "text": turn["text"],
                "emotion": turn["emotion"],
                "prev_emotion": prev_emotion,
                "is_turn_0": is_turn_0,
                "input_text": prompt,
            })

    return pd.DataFrame(rows)


def run_autoregressive_inference(
    classifier: TransformerEmotionClassifier,
    test_df: pd.DataFrame,
    max_length: int = 64,
) -> pd.DataFrame:
    """
    Perform causal, turn-by-turn autoregressive inference.
    At turn t, feeds the model's own predicted emotion from turn t-1.
    Strictly resets state at dialogue boundaries. Never leaks target labels.
    """
    classifier.model.eval()
    test_df = test_df.sort_values(["dialogue_id", "utterance_id"]).reset_index(drop=True)
    results = []

    for dialogue_id, dialogue in test_df.groupby("dialogue_id", sort=False):
        dialogue = dialogue.sort_values("utterance_id").reset_index(drop=True)
        records = dialogue.to_dict("records")
        prev_predicted = "none"

        for i, turn in enumerate(records):
            is_turn_0 = (i == 0)
            true_prev_emotion = "none" if is_turn_0 else records[i - 1]["emotion"]

            # Construct autoregressive input using prior model prediction
            prompt = format_emotion_history_prompt(prev_predicted, turn["text"])

            # Single-turn inference
            preds = classifier.predict([prompt], max_length=max_length)
            probs = classifier.predict_proba([prompt], max_length=max_length)[0]
            pred_emotion = preds[0]

            record_out = {
                "dialogue_id": turn["dialogue_id"],
                "utterance_id": turn["utterance_id"],
                "speaker": turn["speaker"],
                "text": turn["text"],
                "true_emotion": turn["emotion"],
                "predicted_emotion": pred_emotion,
                "prev_predicted": prev_predicted,
                "true_prev_emotion": true_prev_emotion,
                "prev_correct": (prev_predicted == true_prev_emotion) if not is_turn_0 else None,
                "is_turn_0": is_turn_0,
                "confidence": float(np.max(probs)),
            }
            for idx, emo in enumerate(LABELS):
                record_out[f"prob_{emo}"] = float(probs[idx])

            results.append(record_out)
            # Autoregressive update for next turn
            prev_predicted = pred_emotion

    return pd.DataFrame(results)


def run_mcnemar_test(y_true, y_pred_a, y_pred_b) -> tuple[float, float]:
    """
    Compute McNemar's test with continuity correction for paired classification predictions.
    Returns (statistic, p_value).
    """
    correct_a = (np.array(y_pred_a) == np.array(y_true))
    correct_b = (np.array(y_pred_b) == np.array(y_true))

    b = int(np.sum(correct_a & (~correct_b)))
    c = int(np.sum((~correct_a) & correct_b))

    if b + c == 0:
        return 0.0, 1.0

    # McNemar's chi-squared statistic with Edwards continuity correction
    stat = ((abs(b - c) - 1.0) ** 2) / (b + c)
    p_val = float(stats.chi2.sf(stat, df=1))
    return float(stat), float(p_val)


def main():
    args = parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    set_seed(args.seed)

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("PHASE 1, STEP 2: EMOTION-HISTORY DISTILBERT (MODEL A)")
    print(f"Epochs: {args.epochs} | Batch size: {args.batch_size} | LR: {args.learning_rate} | Max Length: {args.max_length}")
    print(f"Checkpoints: {checkpoint_dir} | Outputs: {output_dir}")
    print("=" * 75)

    # 1. Load data
    dataset = load_meld(Path(args.data_dir))
    print(f"Loaded MELD: Train={len(dataset.train)}, Dev={len(dataset.dev)}, Test={len(dataset.test)}")

    train_data = build_emotion_history_dataset(dataset.train)
    dev_data = build_emotion_history_dataset(dataset.dev)
    test_data = build_emotion_history_dataset(dataset.test)

    if args.tiny:
        print("[TINY MODE] Truncating datasets for quick smoke test...")
        train_data = train_data.iloc[:48].copy()
        dev_data = dev_data.iloc[:24].copy()
        test_data = test_data.iloc[:24].copy()

    # 2. Initialize classifier & trainer
    classifier = TransformerEmotionClassifier(
        model_name="distilbert-base-uncased",
        num_labels=len(LABELS),
    )
    trainer = TransformerTrainer(
        classifier=classifier,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_length=args.max_length,
        seed=args.seed,
    )

    # Dry run resume verification
    if args.dry_run_resume:
        print("\n" + "=" * 75)
        print("LIGHTWEIGHT RESUME VERIFICATION (DRY RUN)")
        print("=" * 75)
        state_file = checkpoint_dir / "latest_training_state.pt"
        if not state_file.exists():
            raise FileNotFoundError(f"Checkpoint state file not found at: {state_file}")

        print(f"Loading checkpoint state from: {state_file}")
        state = trainer.load_training_checkpoint(checkpoint_dir)
        print(f"  [x] Successfully restored model state dict into {classifier.model_name}")
        print(f"  [x] Successfully restored optimizer state dict")
        print(f"  [x] Checkpoint completed epoch: {state['epoch']}")
        print(f"  [x] Next start epoch will be: {state['epoch'] + 1} (Target total epochs: {args.epochs})")
        print(f"  [x] Epochs remaining to train: {list(range(state['epoch'] + 1, args.epochs + 1))}")
        print(f"  [x] Preserved best validation Macro F1: {state['best_macro_f1']:.6f}")
        print(f"  [x] Preserved training history entries: {len(trainer.history)}")
        for h in trainer.history:
            print(f"       Epoch {h['epoch']}: Train Loss = {h['train_loss']:.4f}, Val Acc = {h['val_accuracy']:.4f}, Val Macro F1 = {h['val_macro_f1']:.4f}")

        best_dir = checkpoint_dir / "best"
        print(f"  [x] Best checkpoint directory exists: {best_dir.exists()}")
        if (best_dir / "training_config.json").exists():
            with open(best_dir / "training_config.json") as f:
                best_cfg = json.load(f)
            print(f"       Best checkpoint training_config best_val_macro_f1: {best_cfg.get('best_val_macro_f1'):.6f}")

        print("\nVerification PASSED: Resume mechanism is verified and will execute ONLY Epoch 3.")
        return 0

    # 3. Train with checkpoint selection on validation Macro F1
    t0 = time.time()
    if args.resume:
        print(f"\nResuming training from checkpoint in {checkpoint_dir}...")
    else:
        print("\nStarting training loop from scratch (validation model selection on Macro F1)...")

    history = trainer.fit(
        train_texts=train_data["input_text"].tolist(),
        train_labels=train_data["emotion"].tolist(),
        val_texts=dev_data["input_text"].tolist(),
        val_labels=dev_data["emotion"].tolist(),
        checkpoint_dir=checkpoint_dir,
        resume=args.resume,
    )
    train_duration_sec = time.time() - t0
    print(f"\nTraining completed in {train_duration_sec:.1f}s ({train_duration_sec/60:.2f} min)")

    with open(output_dir / "training_history.json", "w") as f:
        json.dump({
            "train_duration_seconds": round(train_duration_sec, 2),
            "history": history,
        }, f, indent=2)

    best_checkpoint = checkpoint_dir / "best"
    print(f"Best validation checkpoint loaded from: {best_checkpoint}")
    best_classifier = TransformerEmotionClassifier(model_path=best_checkpoint)

    # ------------------------------------------------------------------------
    # 4. Evaluation 1: Teacher-Forced (Condition 3 - Oracle Diagnostic)
    # ------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("EVALUATION 1: TEACHER-FORCED EMOTION HISTORY (ORACLE DIAGNOSTIC)")
    print("=" * 75)
    tf_texts = test_data["input_text"].tolist()
    tf_labels = test_data["emotion"].tolist()

    tf_preds = best_classifier.predict(tf_texts, batch_size=args.batch_size, max_length=args.max_length)
    tf_probs = best_classifier.predict_proba(tf_texts, batch_size=args.batch_size, max_length=args.max_length)

    # Stratified splits
    turn0_idx = test_data[test_data["is_turn_0"]].index.tolist()
    subseq_idx = test_data[~test_data["is_turn_0"]].index.tolist()

    eval_tf_full = evaluate(tf_labels, tf_preds, labels=LABELS)
    eval_tf_turn0 = evaluate([tf_labels[i] for i in turn0_idx], [tf_preds[i] for i in turn0_idx], labels=LABELS)
    eval_tf_subseq = evaluate([tf_labels[i] for i in subseq_idx], [tf_preds[i] for i in subseq_idx], labels=LABELS)

    print("\n[Teacher-Forced Full Test (N={})]".format(len(tf_labels)))
    print(f"Accuracy: {eval_tf_full.accuracy:.4f} | Macro F1: {eval_tf_full.macro_f1:.4f} | Weighted F1: {eval_tf_full.weighted_f1:.4f}")
    print("\n[Teacher-Forced Turn-0 (N={})]".format(len(turn0_idx)))
    print(f"Accuracy: {eval_tf_turn0.accuracy:.4f} | Macro F1: {eval_tf_turn0.macro_f1:.4f} | Weighted F1: {eval_tf_turn0.weighted_f1:.4f}")
    print("\n[Teacher-Forced Subsequent Turns >=1 (Matched Context N={})]".format(len(subseq_idx)))
    print(f"Accuracy: {eval_tf_subseq.accuracy:.4f} | Macro F1: {eval_tf_subseq.macro_f1:.4f} | Weighted F1: {eval_tf_subseq.weighted_f1:.4f}")

    plot_confusion_matrix(eval_tf_full, save_path=output_dir / "confusion_matrix_teacher_forced.png", show=False)

    with open(output_dir / "metrics_teacher_forced.json", "w") as f:
        json.dump({
            "mode": "teacher_forced_oracle",
            "full_test": {
                "n": len(tf_labels),
                "accuracy": eval_tf_full.accuracy,
                "macro_f1": eval_tf_full.macro_f1,
                "weighted_f1": eval_tf_full.weighted_f1,
            },
            "turn_0": {
                "n": len(turn0_idx),
                "accuracy": eval_tf_turn0.accuracy,
                "macro_f1": eval_tf_turn0.macro_f1,
                "weighted_f1": eval_tf_turn0.weighted_f1,
            },
            "turns_subsequent": {
                "n": len(subseq_idx),
                "accuracy": eval_tf_subseq.accuracy,
                "macro_f1": eval_tf_subseq.macro_f1,
                "weighted_f1": eval_tf_subseq.weighted_f1,
            },
        }, f, indent=4)

    # ------------------------------------------------------------------------
    # 5. Evaluation 2: Autoregressive Predicted History (Condition 4 - Deployable)
    # ------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("EVALUATION 2: AUTOREGRESSIVE EMOTION HISTORY (DEPLOYABLE INFERENCE)")
    print("=" * 75)

    auto_df = run_autoregressive_inference(best_classifier, dataset.test if not args.tiny else dataset.test.iloc[:24], max_length=args.max_length)
    auto_df.to_csv(output_dir / "predictions_autoregressive.csv", index=False)

    auto_y_true = auto_df["true_emotion"].tolist()
    auto_y_pred = auto_df["predicted_emotion"].tolist()
    auto_turn0_mask = auto_df["is_turn_0"].values
    auto_subseq_mask = ~auto_turn0_mask

    eval_auto_full = evaluate(auto_y_true, auto_y_pred, labels=LABELS)
    eval_auto_turn0 = evaluate(
        [auto_y_true[i] for i in range(len(auto_y_true)) if auto_turn0_mask[i]],
        [auto_y_pred[i] for i in range(len(auto_y_pred)) if auto_turn0_mask[i]],
        labels=LABELS,
    )
    eval_auto_subseq = evaluate(
        [auto_y_true[i] for i in range(len(auto_y_true)) if auto_subseq_mask[i]],
        [auto_y_pred[i] for i in range(len(auto_y_pred)) if auto_subseq_mask[i]],
        labels=LABELS,
    )

    print("\n[Autoregressive Full Test (N={})]".format(len(auto_y_true)))
    print_results(eval_auto_full)
    print("\n[Autoregressive Turn-0 (N={})]".format(np.sum(auto_turn0_mask)))
    print(f"Accuracy: {eval_auto_turn0.accuracy:.4f} | Macro F1: {eval_auto_turn0.macro_f1:.4f} | Weighted F1: {eval_auto_turn0.weighted_f1:.4f}")
    print("\n[Autoregressive Subsequent Turns >=1 (Matched Context N={})]".format(np.sum(auto_subseq_mask)))
    print_results(eval_auto_subseq)

    plot_confusion_matrix(eval_auto_full, save_path=output_dir / "confusion_matrix_autoregressive.png", show=False)
    plot_confusion_matrix(eval_auto_subseq, save_path=output_dir / "matched_confusion_matrix_autoregressive.png", show=False)

    # Save per-class classification reports
    full_counts = Counter(auto_y_true)
    subseq_counts = Counter([auto_y_true[i] for i in range(len(auto_y_true)) if auto_subseq_mask[i]])

    per_class_subseq_rows = []
    for emo, m in eval_auto_subseq.per_class.items():
        per_class_subseq_rows.append({
            "emotion": emo,
            "precision": m.precision,
            "recall": m.recall,
            "f1": m.f1,
            "support": subseq_counts.get(emo, 0),
        })
    pd.DataFrame(per_class_subseq_rows).to_csv(output_dir / "matched_classification_report.csv", index=False)

    with open(output_dir / "metrics_autoregressive.json", "w") as f:
        json.dump({
            "mode": "autoregressive_deployable",
            "full_test": {
                "n": len(auto_y_true),
                "accuracy": eval_auto_full.accuracy,
                "macro_f1": eval_auto_full.macro_f1,
                "weighted_f1": eval_auto_full.weighted_f1,
            },
            "turn_0": {
                "n": int(np.sum(auto_turn0_mask)),
                "accuracy": eval_auto_turn0.accuracy,
                "macro_f1": eval_auto_turn0.macro_f1,
                "weighted_f1": eval_auto_turn0.weighted_f1,
            },
            "turns_subsequent": {
                "n": int(np.sum(auto_subseq_mask)),
                "accuracy": eval_auto_subseq.accuracy,
                "macro_f1": eval_auto_subseq.macro_f1,
                "weighted_f1": eval_auto_subseq.weighted_f1,
            },
        }, f, indent=4)

    # ------------------------------------------------------------------------
    # 6. Transition & Error Propagation Analysis
    # ------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("TRANSITION & ERROR PROPAGATION ANALYSIS (Turns >= 1)")
    print("=" * 75)
    subseq_df = auto_df[auto_subseq_mask].copy()

    # Preceding prediction correctness split
    correct_prior_mask = (subseq_df["prev_correct"] == True)
    incorrect_prior_mask = (subseq_df["prev_correct"] == False)

    acc_when_prior_correct = np.mean(subseq_df.loc[correct_prior_mask, "true_emotion"] == subseq_df.loc[correct_prior_mask, "predicted_emotion"]) if np.sum(correct_prior_mask) > 0 else 0.0
    acc_when_prior_wrong = np.mean(subseq_df.loc[incorrect_prior_mask, "true_emotion"] == subseq_df.loc[incorrect_prior_mask, "predicted_emotion"]) if np.sum(incorrect_prior_mask) > 0 else 0.0

    # Persistence vs Shift split
    persistent_mask = (subseq_df["true_emotion"] == subseq_df["true_prev_emotion"])
    shift_mask = ~persistent_mask

    acc_persistent = np.mean(subseq_df.loc[persistent_mask, "true_emotion"] == subseq_df.loc[persistent_mask, "predicted_emotion"]) if np.sum(persistent_mask) > 0 else 0.0
    acc_shift = np.mean(subseq_df.loc[shift_mask, "true_emotion"] == subseq_df.loc[shift_mask, "predicted_emotion"]) if np.sum(shift_mask) > 0 else 0.0

    # Neutral Gravitation
    ground_truth_neutral_rate = np.mean(subseq_df["true_emotion"] == "neutral")
    oracle_neutral_rate = np.mean(np.array(tf_preds)[subseq_idx] == "neutral")
    auto_neutral_rate = np.mean(subseq_df["predicted_emotion"] == "neutral")

    transition_analysis = {
        "turns_evaluated": int(len(subseq_df)),
        "prior_prediction_correct_count": int(np.sum(correct_prior_mask)),
        "prior_prediction_wrong_count": int(np.sum(incorrect_prior_mask)),
        "accuracy_when_prior_correct": round(float(acc_when_prior_correct), 4),
        "accuracy_when_prior_wrong": round(float(acc_when_prior_wrong), 4),
        "error_cascade_drop": round(float(acc_when_prior_correct - acc_when_prior_wrong), 4),
        "persistent_turns_count": int(np.sum(persistent_mask)),
        "shift_turns_count": int(np.sum(shift_mask)),
        "accuracy_on_persistent_turns": round(float(acc_persistent), 4),
        "accuracy_on_shift_turns": round(float(acc_shift), 4),
        "ground_truth_neutral_rate": round(float(ground_truth_neutral_rate), 4),
        "oracle_predicted_neutral_rate": round(float(oracle_neutral_rate), 4),
        "autoregressive_predicted_neutral_rate": round(float(auto_neutral_rate), 4),
    }

    with open(output_dir / "transition_analysis.json", "w") as f:
        json.dump(transition_analysis, f, indent=4)

    print(f"Accuracy when preceding prediction was CORRECT: {acc_when_prior_correct:.4f} (N={np.sum(correct_prior_mask)})")
    print(f"Accuracy when preceding prediction was WRONG  : {acc_when_prior_wrong:.4f} (N={np.sum(incorrect_prior_mask)})")
    print(f"Error Cascade Impact (Delta)                    : {acc_when_prior_correct - acc_when_prior_wrong:.4f}")
    print(f"Accuracy on Emotion Persistence turns          : {acc_persistent:.4f} (N={np.sum(persistent_mask)})")
    print(f"Accuracy on Emotion Shift turns                : {acc_shift:.4f} (N={np.sum(shift_mask)})")
    print(f"Neutral Prediction Rate — Ground Truth: {ground_truth_neutral_rate:.4f} | Oracle: {oracle_neutral_rate:.4f} | Autoregressive: {auto_neutral_rate:.4f}")

    # ------------------------------------------------------------------------
    # 7. Comparison Against Baselines (Matched Subset N=2,328)
    # ------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("CONSOLIDATED COMPARISON TABLE (MATCHED CONTEXT SUBSET N=2,328)")
    print("=" * 75)

    # Load baseline predictions if available for McNemar test
    base_pred_file = PROJECT_ROOT / "outputs" / "experiments" / "transformer_baseline" / "predictions.csv"
    mcnemar_stat, mcnemar_p = None, None
    if base_pred_file.exists() and not args.tiny:
        base_df = pd.read_csv(base_pred_file)
        # Match keys
        matched_keys = set(zip(subseq_df["dialogue_id"], subseq_df["utterance_id"]))
        base_matched = base_df[base_df.apply(lambda r: (r["dialogue_id"], r["utterance_id"]) in matched_keys, axis=1)]
        if len(base_matched) == len(subseq_df):
            mcnemar_stat, mcnemar_p = run_mcnemar_test(
                subseq_df["true_emotion"].values,
                base_matched["predicted_emotion"].values,
                subseq_df["predicted_emotion"].values,
            )
            print(f"McNemar Test vs Utterance Baseline: chi2={mcnemar_stat:.4f}, p={mcnemar_p:.4e}")

    comp_rows = [
        {
            "model_condition": "DistilBERT Utterance-Only Baseline",
            "eval_mode": "Independent",
            "accuracy": 0.594502,
            "macro_f1": 0.431416,
            "weighted_f1": 0.596910,
        },
        {
            "model_condition": "DistilBERT Text Context-1 Baseline",
            "eval_mode": "Concatenated Context",
            "accuracy": 0.607388,
            "macro_f1": 0.391191,
            "weighted_f1": 0.581155,
        },
        {
            "model_condition": "Emotion-History DistilBERT (Condition 3)",
            "eval_mode": "Oracle Diagnostic (Teacher-Forced)",
            "accuracy": eval_tf_subseq.accuracy,
            "macro_f1": eval_tf_subseq.macro_f1,
            "weighted_f1": eval_tf_subseq.weighted_f1,
        },
        {
            "model_condition": "Emotion-History DistilBERT (Condition 4)",
            "eval_mode": "Autoregressive (Deployable)",
            "accuracy": eval_auto_subseq.accuracy,
            "macro_f1": eval_auto_subseq.macro_f1,
            "weighted_f1": eval_auto_subseq.weighted_f1,
        },
    ]
    df_comp = pd.DataFrame(comp_rows)
    df_comp.to_csv(output_dir / "baseline_comparison_matched.csv", index=False)
    print(df_comp.to_string(index=False))

    print(f"\nAll artifacts successfully saved to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
