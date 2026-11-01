from dataclasses import asdict, is_dataclass

from src.analysis.conversation_analysis import analyze_predictions
from src.data.conversation import parse_conversation
from src.models.inference import EmotionInference


MODEL_CHECKPOINTS = {
    "utterance": "checkpoints/transformer_baseline/best",
    "context_1": "checkpoints/transformer_context_1/best",
    "context_1_speaker": "checkpoints/transformer_context_1_speaker/best",
}


class ConversationAnalyzer:
    """
    Application-level service that combines:

        raw conversation
            -> parsing
            -> emotion inference
            -> dynamics analysis
    """

    def __init__(self):
        self._models = {}

    def parse_conversation(self, conversation):
        return parse_conversation(conversation)

    def _get_model(self, mode: str) -> EmotionInference:
        if mode not in MODEL_CHECKPOINTS:
            raise ValueError(
                f"Unknown model mode '{mode}'. "
                f"Choose from: {', '.join(MODEL_CHECKPOINTS)}"
            )

        if mode not in self._models:
            self._models[mode] = EmotionInference(
                MODEL_CHECKPOINTS[mode]
            )

        return self._models[mode]

    def analyze(
        self,
        conversation: str,
        mode: str = "utterance",
    ) -> dict:
        turns = parse_conversation(conversation)

        model = self._get_model(mode)
        predictions = model.predict(turns)

        analysis = analyze_predictions(predictions)

        return {
            "mode": mode,
            "turns": predictions,
            "analysis": {
                "emotion_distribution": analysis.emotion_distribution,
                "transition_counts": analysis.transition_counts,
                "total_transitions": analysis.total_transitions,
                "shifts": analysis.shifts,
                "persistence": analysis.persistence,
                "shift_rate": analysis.shift_rate,
                "persistence_rate": analysis.persistence_rate,
                "same_speaker_transitions": (
                    analysis.same_speaker_transitions
                ),
                "same_speaker_shifts": analysis.same_speaker_shifts,
                "cross_speaker_transitions": (
                    analysis.cross_speaker_transitions
                ),
                "cross_speaker_shifts": (
                    analysis.cross_speaker_shifts
                ),
                "turning_points": analysis.turning_points,
            },
        }
