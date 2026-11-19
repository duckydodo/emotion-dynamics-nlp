"""
Canonical Model Registry for emotion-dynamics-nlp.

Exposes exactly three verified model families:
1. DistilBERT (utterance-level local baseline)
2. DialogueRNN (causal recurrent state tracking)
3. DialogueGCN (relational graph convolution)

Experimental variants and ablations are excluded from the public application registry.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


CANONICAL_MODEL_IDS = ("distilbert", "dialoguernn", "dialoguegcn")

MODEL_ALIASES: Dict[str, str] = {
    "distilbert": "distilbert",
    "utterance": "distilbert",
    "transformer_baseline": "distilbert",
    "dialoguernn": "dialoguernn",
    "dialoguernn_unweighted": "dialoguernn",
    "dialoguegcn": "dialoguegcn",
}


@dataclass
class ModelMetadata:
    model_id: str
    display_name: str
    description: str
    architecture_type: str
    checkpoint_path: str
    checkpoint_metadata: Dict[str, Any]
    verified_metrics: Dict[str, Any]
    context_window: str
    causal: bool
    is_available: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_RAW_MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "distilbert": {
        "model_id": "distilbert",
        "display_name": "DistilBERT (Utterance-Only Baseline)",
        "description": "Fine-tuned transformer classifying emotional state per utterance independently without multi-turn conversational context.",
        "architecture_type": "transformer",
        "checkpoint_path": "checkpoints/transformer_baseline/best",
        "checkpoint_metadata": {
            "checkpoint_dir": "checkpoints/transformer_baseline/best",
            "base_model": "distilbert-base-uncased",
            "parameters": 66960000,
            "max_sequence_length": 128,
        },
        "verified_metrics": {
            "cohort": "Cohort A: Strict Contiguous Matched Subset (N=2,328)",
            "accuracy": 0.5945,
            "macro_f1": 0.4314,
            "weighted_f1": 0.5794,
        },
        "context_window": "Single utterance",
        "causal": True,
    },
    "dialoguernn": {
        "model_id": "dialoguernn",
        "display_name": "DialogueRNN",
        "description": "Causal attentive recurrent network tracking global discourse flow, speaker-specific party states, and emotional inertia across conversation turns.",
        "architecture_type": "recurrent",
        "checkpoint_path": "checkpoints/dialoguernn_unweighted/best_model.pt",
        "checkpoint_metadata": {
            "checkpoint_path": "checkpoints/dialoguernn_unweighted/best_model.pt",
            "encoder": "Frozen DistilBERT (CLS hidden vector, 768-dim)",
            "parameters": 670855,
            "selection_criterion": "Development Macro F1 (Epoch 10)",
        },
        "verified_metrics": {
            "cohort": "Cohort A: Strict Contiguous Matched Subset (N=2,328)",
            "accuracy": 0.6014,
            "macro_f1": 0.4342,
            "weighted_f1": 0.5892,
        },
        "context_window": "Full conversational history (causal)",
        "causal": True,
    },
    "dialoguegcn": {
        "model_id": "dialoguegcn",
        "display_name": "DialogueGCN",
        "description": "Relational graph convolutional network with similarity attention modeling conversational topology, intra-speaker emotional inertia, and inter-speaker influence across past and future turns.",
        "architecture_type": "graph_convolutional",
        "checkpoint_path": "checkpoints/dialoguegcn/best_model.pt",
        "checkpoint_metadata": {
            "checkpoint_path": "checkpoints/dialoguegcn/best_model.pt",
            "encoder": "Frozen DistilBERT (CLS hidden vector, 768-dim)",
            "parameters": 452487,
            "layers": 2,
            "relations": 2,
            "selection_criterion": "Development Macro F1 (Epoch 15)",
        },
        "verified_metrics": {
            "cohort": "Cohort A: Strict Contiguous Matched Subset (N=2,328)",
            "accuracy": 0.6091,
            "macro_f1": 0.4385,
            "weighted_f1": 0.6009,
        },
        "context_window": "Bidirectional window (past=10, future=10)",
        "causal": False,
    },
}


def normalize_model_id(model_id: str) -> str:
    """Normalize input mode/model string to canonical model ID or raise ValueError."""
    key = model_id.strip().lower()
    if key in MODEL_ALIASES:
        return MODEL_ALIASES[key]
    raise ValueError(
        f"Unknown model identifier '{model_id}'. "
        f"Choose from canonical model families: {', '.join(CANONICAL_MODEL_IDS)}"
    )


def get_model_metadata(model_id: str) -> ModelMetadata:
    """Retrieve verified metadata for a canonical model family."""
    canonical_id = normalize_model_id(model_id)
    spec = _RAW_MODEL_SPECS[canonical_id]
    ckpt_path = Path(spec["checkpoint_path"])
    is_available = ckpt_path.exists()

    return ModelMetadata(
        model_id=canonical_id,
        display_name=spec["display_name"],
        description=spec["description"],
        architecture_type=spec["architecture_type"],
        checkpoint_path=spec["checkpoint_path"],
        checkpoint_metadata=spec["checkpoint_metadata"],
        verified_metrics=spec["verified_metrics"],
        context_window=spec["context_window"],
        causal=spec["causal"],
        is_available=is_available,
    )


def get_canonical_registry() -> Dict[str, Dict[str, Any]]:
    """Return dictionary of all three canonical model families with availability status."""
    registry = {}
    for m_id in CANONICAL_MODEL_IDS:
        meta = get_model_metadata(m_id)
        registry[m_id] = meta.to_dict()
    return registry
