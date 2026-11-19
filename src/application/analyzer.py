from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional

from src.analysis.conversation_analysis import analyze_predictions
from src.data.conversation import parse_conversation, parse_canonical_conversation
from src.data.schema import CanonicalConversation
from src.models.inference import (
    EmotionInference,
    DialogueRNNInference,
    DialogueGCNInference,
)
from src.models.registry import (
    get_canonical_registry,
    get_model_metadata,
    normalize_model_id,
    CANONICAL_MODEL_IDS,
)


MODEL_CHECKPOINTS = {
    "distilbert": "checkpoints/transformer_baseline/best",
    "dialoguernn": "checkpoints/dialoguernn_unweighted/best_model.pt",
    "dialoguegcn": "checkpoints/dialoguegcn/best_model.pt",
    # Legacy aliases preserved for regression compatibility
    "utterance": "checkpoints/transformer_baseline/best",
    "context_1": "checkpoints/transformer_context_1/best",
    "context_1_speaker": "checkpoints/transformer_context_1_speaker/best",
}

MODEL_METADATA = {
    "distilbert": {
        "architecture": "DistilBERT (Utterance-only)",
        "description": "Fine-tuned DistilBERT predicting emotion independently per turn.",
        "context_window": "Single utterance",
        "causal": True,
        "parameters": 66960000,
    },
    "dialoguernn": {
        "architecture": "DialogueRNN (AAAI 2019)",
        "description": "Attentive RNN with global context tracking and speaker-specific party states.",
        "context_window": "Full conversational history",
        "causal": True,
        "parameters": 670855,
        "utterance_encoder": "Frozen DistilBERT (CLS hidden vector, 768-dim)",
    },
    "dialoguegcn": {
        "architecture": "DialogueGCN (EMNLP 2019)",
        "description": "Relational GCN with similarity attention over conversational topology.",
        "context_window": "Bidirectional window (past=10, future=10)",
        "causal": False,
        "parameters": 452487,
        "utterance_encoder": "Frozen DistilBERT (CLS hidden vector, 768-dim)",
    },
    # Legacy aliases
    "utterance": {
        "architecture": "DistilBERT (Utterance-only)",
        "description": "Fine-tuned DistilBERT predicting emotion independently per turn.",
        "context_window": "Single utterance",
        "causal": True,
    },
    "context_1": {
        "architecture": "DistilBERT (Context-1)",
        "description": "Fine-tuned DistilBERT concatenated with immediate preceding utterance text.",
        "context_window": "Immediate previous turn (t-1)",
        "causal": True,
    },
    "context_1_speaker": {
        "architecture": "DistilBERT (Context-1 + Speaker)",
        "description": "Fine-tuned DistilBERT with preceding utterance and explicit speaker role tokens.",
        "context_window": "Immediate previous turn + speaker tokens",
        "causal": True,
    },
}


class ConversationAnalyzer:
    """
    Application-level service that combines:
        raw conversation
            -> parsing
            -> emotion inference
            -> dynamics analysis
            -> multi-model comparison
    """

    def __init__(self):
        self._models = {}

    def parse_conversation(self, conversation: str):
        return parse_conversation(conversation)

    def parse_canonical(
        self,
        conversation: str,
        format_hint: str = "auto",
        **kwargs: Any,
    ) -> CanonicalConversation:
        return parse_canonical_conversation(conversation, format_hint=format_hint, **kwargs)

    def get_model_registry(self) -> Dict[str, Dict[str, Any]]:
        """Return the canonical model registry."""
        return get_canonical_registry()

    def _get_model(self, mode: str):
        mode_key = mode.strip().lower()
        if mode_key not in MODEL_CHECKPOINTS:
            raise ValueError(
                f"Unknown model mode '{mode}'. "
                f"Choose from: {', '.join(MODEL_CHECKPOINTS)}"
            )

        # Normalize canonical key
        canonical_key = mode_key
        if mode_key == "utterance":
            canonical_key = "distilbert"

        if canonical_key not in self._models:
            if canonical_key == "dialoguegcn":
                self._models[canonical_key] = DialogueGCNInference(
                    checkpoint_path=MODEL_CHECKPOINTS["dialoguegcn"],
                    encoder_path=MODEL_CHECKPOINTS["distilbert"],
                )
            elif canonical_key == "dialoguernn":
                self._models[canonical_key] = DialogueRNNInference(
                    checkpoint_path=MODEL_CHECKPOINTS["dialoguernn"],
                    encoder_path=MODEL_CHECKPOINTS["distilbert"],
                )
            else:
                self._models[canonical_key] = EmotionInference(
                    MODEL_CHECKPOINTS[mode_key]
                )

        return self._models[canonical_key]

    def analyze(
        self,
        conversation: str,
        mode: str = "utterance",
    ) -> dict:
        canonical = parse_canonical_conversation(conversation)
        turns = canonical.get_evaluable_turns()
        if not turns:
            raise ValueError("No valid conversation turns were found.")

        model = self._get_model(mode)
        predictions = model.predict(turns)

        analysis = analyze_predictions(predictions)

        return {
            "mode": mode,
            "model_metadata": MODEL_METADATA.get(mode, MODEL_METADATA.get(mode.lower(), {})),
            "turns": predictions,
            "analysis": {
                "emotion_distribution": analysis.emotion_distribution,
                "transition_counts": analysis.transition_counts,
                "total_transitions": analysis.total_transitions,
                "shifts": analysis.shifts,
                "persistence": analysis.persistence,
                "shift_rate": analysis.shift_rate,
                "persistence_rate": analysis.persistence_rate,
                "same_speaker_transitions": analysis.same_speaker_transitions,
                "same_speaker_shifts": analysis.same_speaker_shifts,
                "cross_speaker_transitions": analysis.cross_speaker_transitions,
                "cross_speaker_shifts": analysis.cross_speaker_shifts,
                "turning_points": analysis.turning_points,
            },
            "canonical": {
                "conversation_id": canonical.conversation_id,
                "source_format": canonical.source_format.value,
                "num_messages": len(canonical.messages),
                "num_speakers": len(canonical.participants),
                "warnings": [w.to_dict() for w in canonical.warnings],
            },
        }

    def compare(
        self,
        conversation: str | CanonicalConversation,
        model_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Execute multi-model comparison across selected models on identical messages."""
        from src.application.comparison_service import ModelComparisonService
        service = ModelComparisonService(self)
        return service.compare(conversation, model_ids=model_ids)
