"""
Cross-Model Disagreement and Error Taxonomy Analyzer.

Compares predictions across DistilBERT, DialogueRNN, and DialogueGCN to quantify
pairwise concordance (Cohen's Kappa), identify context-driven shifts, and categorize
systematic conversational failure modes.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.models.transformer import LABELS


def cohens_kappa(preds_a: List[str], preds_b: List[str], labels: List[str] = LABELS) -> float:
    """Computes Cohen's Kappa inter-annotator agreement between two categorical sequences."""
    assert len(preds_a) == len(preds_b)
    n = len(preds_a)
    if n == 0:
        return 0.0

    po = sum(1 for a, b in zip(preds_a, preds_b) if a == b) / n

    counts_a = Counter(preds_a)
    counts_b = Counter(preds_b)
    pe = sum((counts_a[lbl] / n) * (counts_b[lbl] / n) for lbl in labels)

    if pe >= 1.0:
        return 1.0
    return float((po - pe) / (1.0 - pe))


@dataclass
class DisagreementTaxonomy:
    total_turns: int
    unanimous_agreement_count: int
    unanimous_agreement_rate: float
    pairwise_kappa: Dict[str, float]
    pairwise_concordance: Dict[str, float]
    taxonomy_counts: Dict[str, int]
    taxonomy_rates: Dict[str, float]
    representative_cases: Dict[str, List[Dict[str, Any]]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_turns": self.total_turns,
            "unanimous_agreement_count": self.unanimous_agreement_count,
            "unanimous_agreement_rate": round(self.unanimous_agreement_rate, 4),
            "pairwise_kappa": {k: round(v, 4) for k, v in self.pairwise_kappa.items()},
            "pairwise_concordance": {k: round(v, 4) for k, v in self.pairwise_concordance.items()},
            "taxonomy_counts": self.taxonomy_counts,
            "taxonomy_rates": {k: round(v, 4) for k, v in self.taxonomy_rates.items()},
            "representative_cases": self.representative_cases,
        }


def analyze_model_disagreements(
    gold_labels: List[str],
    distilbert_preds: List[str],
    dialoguernn_preds: List[str],
    dialoguegcn_preds: List[str],
    utterance_metadata: Optional[List[Dict[str, Any]]] = None,
) -> DisagreementTaxonomy:
    """
    Analyze cross-model disagreement across the three canonical models on identical turns.
    """
    n = len(gold_labels)
    assert len(distilbert_preds) == n and len(dialoguernn_preds) == n and len(dialoguegcn_preds) == n

    # 1. Pairwise metrics
    pairs = {
        "distilbert_vs_dialoguernn": (distilbert_preds, dialoguernn_preds),
        "distilbert_vs_dialoguegcn": (distilbert_preds, dialoguegcn_preds),
        "dialoguernn_vs_dialoguegcn": (dialoguernn_preds, dialoguegcn_preds),
    }

    kappa_map = {pair_name: cohens_kappa(p1, p2) for pair_name, (p1, p2) in pairs.items()}
    concordance_map = {
        pair_name: sum(1 for a, b in zip(p1, p2) if a == b) / max(n, 1)
        for pair_name, (p1, p2) in pairs.items()
    }

    # 2. Taxonomy categorization
    unanimous_count = 0
    categories = {
        "all_correct": 0,
        "context_recovery_both": 0, # DistilBERT wrong, both DialogueRNN & DialogueGCN correct
        "graph_relational_win": 0,  # DialogueGCN correct, DistilBERT & DialogueRNN wrong
        "recurrent_state_win": 0,   # DialogueRNN correct, DistilBERT & DialogueGCN wrong
        "context_dilution": 0,      # DistilBERT correct, DialogueRNN & DialogueGCN wrong
        "all_disagree": 0,          # 3 distinct predictions
        "universal_error": 0,       # All wrong
    }

    cases: Dict[str, List[Dict[str, Any]]] = {k: [] for k in categories}

    for idx in range(n):
        y_true = gold_labels[idx]
        y_db = distilbert_preds[idx]
        y_rnn = dialoguernn_preds[idx]
        y_gcn = dialoguegcn_preds[idx]

        meta = utterance_metadata[idx] if utterance_metadata else {"index": idx}
        case_info = {
            "index": idx,
            "speaker": meta.get("speaker", "Unknown"),
            "text": meta.get("text", ""),
            "gold": y_true,
            "distilbert": y_db,
            "dialoguernn": y_rnn,
            "dialoguegcn": y_gcn,
        }

        # Check unanimity
        if y_db == y_rnn == y_gcn:
            unanimous_count += 1

        # Classify taxonomy
        if y_db == y_true and y_rnn == y_true and y_gcn == y_true:
            categories["all_correct"] += 1
            if len(cases["all_correct"]) < 5:
                cases["all_correct"].append(case_info)
        elif y_db != y_true and y_rnn == y_true and y_gcn == y_true:
            categories["context_recovery_both"] += 1
            if len(cases["context_recovery_both"]) < 5:
                cases["context_recovery_both"].append(case_info)
        elif y_gcn == y_true and y_db != y_true and y_rnn != y_true:
            categories["graph_relational_win"] += 1
            if len(cases["graph_relational_win"]) < 5:
                cases["graph_relational_win"].append(case_info)
        elif y_rnn == y_true and y_db != y_true and y_gcn != y_true:
            categories["recurrent_state_win"] += 1
            if len(cases["recurrent_state_win"]) < 5:
                cases["recurrent_state_win"].append(case_info)
        elif y_db == y_true and (y_rnn != y_true or y_gcn != y_true):
            categories["context_dilution"] += 1
            if len(cases["context_dilution"]) < 5:
                cases["context_dilution"].append(case_info)
        elif y_db != y_rnn and y_db != y_gcn and y_rnn != y_gcn:
            categories["all_disagree"] += 1
            if len(cases["all_disagree"]) < 5:
                cases["all_disagree"].append(case_info)
        elif y_db != y_true and y_rnn != y_true and y_gcn != y_true:
            categories["universal_error"] += 1
            if len(cases["universal_error"]) < 5:
                cases["universal_error"].append(case_info)

    rates = {k: v / max(n, 1) for k, v in categories.items()}

    return DisagreementTaxonomy(
        total_turns=n,
        unanimous_agreement_count=unanimous_count,
        unanimous_agreement_rate=unanimous_count / max(n, 1),
        pairwise_kappa=kappa_map,
        pairwise_concordance=concordance_map,
        taxonomy_counts=categories,
        taxonomy_rates=rates,
        representative_cases=cases,
    )
