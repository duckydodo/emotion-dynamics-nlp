#!/usr/bin/env python3
"""
Phase 3: Prediction Explainability and Model Disagreement Study.

Executes:
1. Cross-model disagreement analysis on Canonical Cohort A (N=2,328) across DistilBERT,
   DialogueRNN, and DialogueGCN.
2. Integrated Gradients token attribution for DistilBERT across representative affective turns.
3. Context perturbation ablation measuring probability shift (Delta P) and prediction flips.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.transformer import TransformerEmotionClassifier, LABELS, ID_TO_LABEL
from src.evaluation.attribution import IntegratedGradientsExplainer
from src.evaluation.context_attribution import ContextPerturbationAnalyzer
from src.evaluation.disagreement import analyze_model_disagreements
from src.application.analyzer import ConversationAnalyzer
from src.data.conversation import ConversationTurn
from src.data.loader import load_meld
from src.data.context import build_context_examples


def main():
    print("=" * 80)
    print("PHASE 3: PREDICTION EXPLAINABILITY & MODEL DISAGREEMENT STUDY")
    print("=" * 80)

    out_dir = PROJECT_ROOT / "outputs" / "experiments" / "explainability"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------
    # 1. CROSS-MODEL DISAGREEMENT ANALYSIS (COHORT A, N=2,328)
    # --------------------------------------------------------------------------
    print("\n[1/3] Running Cross-Model Disagreement Analysis on Cohort A (N=2,328)...")

    # Load test sets and Cohort A turn keys
    meld = load_meld(PROJECT_ROOT / "data" / "raw" / "meld")
    test_ctx1 = build_context_examples(meld.test, context_turns=1, include_speaker=False)
    cohort_a_keys = set(zip(test_ctx1["dialogue_id"].astype(int), test_ctx1["utterance_id"].astype(int)))
    print(f"Cohort A target size: {len(cohort_a_keys)} turns.")

    # Load predictions from all three models
    db_df = pd.read_csv(PROJECT_ROOT / "outputs" / "experiments" / "transformer_baseline" / "predictions.csv")
    rnn_df = pd.read_csv(PROJECT_ROOT / "outputs" / "experiments" / "dialoguernn_unweighted" / "test_predictions.csv")
    gcn_df = pd.read_csv(PROJECT_ROOT / "outputs" / "experiments" / "dialoguegcn" / "test_predictions.csv")

    # Filter to Cohort A
    db_df["d_id"] = db_df["dialogue_id"].astype(int)
    db_df["u_id"] = db_df["utterance_id"].astype(int)
    db_cohort = db_df[db_df.apply(lambda r: (r["d_id"], r["u_id"]) in cohort_a_keys, axis=1)].copy()

    rnn_df["d_id"] = rnn_df["dialogue_id"].astype(int)
    rnn_df["u_id"] = rnn_df["utterance_id"].astype(int)
    rnn_cohort = rnn_df[rnn_df.apply(lambda r: (r["d_id"], r["u_id"]) in cohort_a_keys, axis=1)].copy()

    gcn_df["d_id"] = gcn_df["dialogue_id"].astype(int)
    gcn_df["u_id"] = gcn_df["utterance_id"].astype(int)
    gcn_cohort = gcn_df[gcn_df.apply(lambda r: (r["d_id"], r["u_id"]) in cohort_a_keys, axis=1)].copy()

    # Sort deterministically
    db_cohort = db_cohort.sort_values(by=["d_id", "u_id"]).reset_index(drop=True)
    rnn_cohort = rnn_cohort.sort_values(by=["d_id", "u_id"]).reset_index(drop=True)
    gcn_cohort = gcn_cohort.sort_values(by=["d_id", "u_id"]).reset_index(drop=True)

    assert len(db_cohort) == 2328, f"DistilBERT Cohort A size is {len(db_cohort)}"
    assert len(rnn_cohort) == 2328, f"DialogueRNN Cohort A size is {len(rnn_cohort)}"
    assert len(gcn_cohort) == 2328, f"DialogueGCN Cohort A size is {len(gcn_cohort)}"

    gold = db_cohort["true_emotion"].tolist()
    pred_db = db_cohort["predicted_emotion"].tolist()
    pred_rnn = rnn_cohort["pred_emotion"].tolist()
    pred_gcn = gcn_cohort["pred_emotion"].tolist()

    metadata = [
        {
            "dialogue_id": int(r["d_id"]),
            "utterance_id": int(r["u_id"]),
            "speaker": str(r["speaker"]),
            "text": str(r["text"]),
        }
        for _, r in db_cohort.iterrows()
    ]

    disagreement_results = analyze_model_disagreements(
        gold_labels=gold,
        distilbert_preds=pred_db,
        dialoguernn_preds=pred_rnn,
        dialoguegcn_preds=pred_gcn,
        utterance_metadata=metadata,
    )

    disagreement_json_path = out_dir / "disagreement_taxonomy.json"
    with open(disagreement_json_path, "w") as f:
        json.dump(disagreement_results.to_dict(), f, indent=2)

    print(f"Saved Disagreement Taxonomy to {disagreement_json_path}")
    print(f"Unanimous Agreement: {disagreement_results.unanimous_agreement_count}/{disagreement_results.total_turns} "
          f"({disagreement_results.unanimous_agreement_rate*100:.2f}%)")
    print("Pairwise Cohen's Kappa:")
    for k, v in disagreement_results.pairwise_kappa.items():
        print(f"  {k}: {v:.4f}")
    print("Taxonomy Distribution:")
    for cat, count in disagreement_results.taxonomy_counts.items():
        print(f"  {cat:22s}: {count:4d} ({count/2328*100:.2f}%)")

    # --------------------------------------------------------------------------
    # 2. TOKEN-LEVEL INTEGRATED GRADIENTS ATTRIBUTION
    # --------------------------------------------------------------------------
    print("\n[2/3] Computing Integrated Gradients Token Attributions for DistilBERT...")
    classifier = TransformerEmotionClassifier(
        model_path=PROJECT_ROOT / "checkpoints" / "transformer_baseline" / "best"
    )
    explainer = IntegratedGradientsExplainer(classifier, steps=25)

    sample_texts = [
        ("I cannot believe you lied to me after everything I did for you!", "anger"),
        ("Oh that's just fantastic, another flat tire in the middle of a thunderstorm.", "anger"),
        ("I am so genuinely thrilled and excited for your new promotion!", "joy"),
        ("I really miss him, the entire house feels completely empty and quiet now.", "sadness"),
        ("Wait, are you serious? Did the dean actually call you back already?", "surprise"),
        ("Oh. That's so Monica can keep track. That way if one is missing she can scream.", "anger"),
        ("Yeah, I'll probably just grab a cup of coffee and head to work.", "neutral"),
        ("Ew, what is that disgusting smell coming from the refrigerator?", "disgust"),
        ("Please don't leave me alone here, I'm really terrified of what might happen.", "fear"),
    ]

    attribution_cases = []
    for text, exp_emotion in sample_texts:
        res = explainer.explain(text)
        top_words = sorted(res.word_attributions, key=lambda w: abs(w.score), reverse=True)[:5]
        top_summary = [f"{w.word} ({w.score:+.3f})" for w in top_words]
        print(f"\nText: \"{text}\"")
        print(f"  Pred: {res.predicted_emotion} ({res.confidence*100:.1f}%) | Salient: {', '.join(top_summary)}")
        attribution_cases.append(res.to_dict())

    attr_json_path = out_dir / "token_attributions.json"
    with open(attr_json_path, "w") as f:
        json.dump(attribution_cases, f, indent=2)
    print(f"\nSaved Token Attributions to {attr_json_path}")

    # --------------------------------------------------------------------------
    # 3. CONTEXT PERTURBATION ANALYSIS (DELTA P, DELTA L)
    # --------------------------------------------------------------------------
    print("\n[3/3] Running Context Perturbation Ablation on Multi-Turn Dialogue...")
    analyzer = ConversationAnalyzer()
    dialoguernn_model = analyzer._get_model("dialoguernn")
    dialoguegcn_model = analyzer._get_model("dialoguegcn")

    # Select representative multi-turn test dialogue
    # E.g. Dialogue 0 from MELD test
    d0_turns = [
        ConversationTurn(turn_id=0, speaker="Mark", text="Why do all you're coffee mugs have numbers on the bottom?"),
        ConversationTurn(turn_id=1, speaker="Rachel", text="Oh. That's so Monica can keep track. That way if one on them is missing, she can be like, 'Where's number 27?!'"),
        ConversationTurn(turn_id=2, speaker="Rachel", text="Y'know what?"),
    ]

    # Additional high-context escalation dialogue
    escalation_dialogue = [
        ConversationTurn(turn_id=0, speaker="Ross", text="Rachel, can we please talk about last night?"),
        ConversationTurn(turn_id=1, speaker="Rachel", text="There is nothing to talk about, Ross. We are done."),
        ConversationTurn(turn_id=2, speaker="Ross", text="I made a mistake, okay? It didn't mean anything!"),
        ConversationTurn(turn_id=3, speaker="Rachel", text="It meant everything to me."),
        ConversationTurn(turn_id=4, speaker="Ross", text="Fine, walk away! See if I care!"),
        ConversationTurn(turn_id=5, speaker="Rachel", text="I will."),
    ]

    pert_analyzer_rnn = ContextPerturbationAnalyzer(dialoguernn_model)
    pert_analyzer_gcn = ContextPerturbationAnalyzer(dialoguegcn_model)

    rnn_pert_results = pert_analyzer_rnn.analyze_dialogue(escalation_dialogue)
    gcn_pert_results = pert_analyzer_gcn.analyze_dialogue(escalation_dialogue)

    perturbation_export = {
        "dialogue_length": len(escalation_dialogue),
        "dialoguernn": [r.to_dict() for r in rnn_pert_results],
        "dialoguegcn": [r.to_dict() for r in gcn_pert_results],
    }

    pert_json_path = out_dir / "context_perturbation.json"
    with open(pert_json_path, "w") as f:
        json.dump(perturbation_export, f, indent=2)
    print(f"Saved Context Perturbation results to {pert_json_path}")

    print("\nContext Perturbation Effects (Escalation Dialogue):")
    for r_rnn, r_gcn in zip(rnn_pert_results, gcn_pert_results):
        print(f"Turn {r_rnn.turn_id} [{r_rnn.speaker}]: \"{r_rnn.text}\"")
        print(f"  DialogueRNN: Full={r_rnn.full_prediction} ({r_rnn.full_confidence:.2f}) vs Iso={r_rnn.isolated_prediction} ({r_rnn.isolated_confidence:.2f}) | Flipped: {r_rnn.prediction_flipped} | Delta P: {r_rnn.target_prob_delta:+.3f}")
        print(f"  DialogueGCN: Full={r_gcn.full_prediction} ({r_gcn.full_confidence:.2f}) vs Iso={r_gcn.isolated_prediction} ({r_gcn.isolated_confidence:.2f}) | Flipped: {r_gcn.prediction_flipped} | Delta P: {r_gcn.target_prob_delta:+.3f}")

    print("\n" + "=" * 80)
    print("PHASE 3 COMPLETE: ALL EXPLAINABILITY ARTIFACTS GENERATED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    main()
