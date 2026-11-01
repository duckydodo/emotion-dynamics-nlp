from pathlib import Path

from src.data.conversation import ConversationTurn
from src.models.transformer import TransformerEmotionClassifier


CHECKPOINTS = {
    "utterance": Path("checkpoints/transformer_baseline/best"),
    "context_1": Path("checkpoints/transformer_context_1/best"),
    "context_1_speaker": Path(
        "checkpoints/transformer_context_1_speaker/best"
    ),
}


class EmotionInference:
    """
    Application inference wrapper for the trained MELD models.

    Supported modes:
        utterance
        context_1
        context_1_speaker
    """

    def __init__(self, checkpoint_path):
        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}"
            )

        self.classifier = TransformerEmotionClassifier(
            model_path=self.checkpoint_path
        )

    def predict(self, turns):
        if not turns:
            return []

        # Context models cannot make a context-based prediction
        # for the first turn, so use the utterance-only model for it.
        if self.checkpoint_path.parent.name == "transformer_baseline":
            texts = [turn.text.strip() for turn in turns]
            predictions = self.classifier.predict(texts)
            probabilities = self.classifier.predict_proba(texts)
        else:
            utterance_model = EmotionInference(CHECKPOINTS["utterance"])
            first_prediction = utterance_model.predict([turns[0]])[0]

            context_turns = turns[1:]
            texts = self._build_model_inputs(context_turns, turns[0])

            context_predictions = self.classifier.predict(texts)
            context_probabilities = self.classifier.predict_proba(texts)

            results = [first_prediction]

            for turn, prediction, probability in zip(
                context_turns,
                context_predictions,
                context_probabilities,
            ):
                probability_dict = {
                    label: float(probability[index])
                    for index, label in enumerate(self.classifier.model.config.id2label.values())
                }

                results.append({
                    "turn_id": turn.turn_id,
                    "speaker": turn.speaker,
                    "text": turn.text,
                    "emotion": prediction,
                    "confidence": float(probability.max()),
                    "probabilities": probability_dict,
                })

            return results

        results = []

        for turn, prediction, probability in zip(
            turns,
            predictions,
            probabilities,
        ):
            probability_dict = {
                label: float(probability[index])
                for index, label in enumerate(self.classifier.model.config.id2label.values())
            }

            results.append({
                "turn_id": turn.turn_id,
                "speaker": turn.speaker,
                "text": turn.text,
                "emotion": prediction,
                "confidence": float(probability.max()),
                "probabilities": probability_dict,
            })

        return results


    def _build_model_inputs(self, turns, previous_turn):
        include_speaker = (
            self.checkpoint_path.parent.name == "transformer_context_1_speaker"
        )

        inputs = []

        for turn in turns:
            if include_speaker:
                previous_text = (
                    f"[SPEAKER_{previous_turn.speaker}] "
                    f"{previous_turn.text.strip()}"
                )
                current_text = (
                    f"[SPEAKER_{turn.speaker}] "
                    f"{turn.text.strip()}"
                )
            else:
                previous_text = previous_turn.text.strip()
                current_text = turn.text.strip()

            inputs.append(f"{previous_text}\n{current_text}")
            previous_turn = turn

        return inputs
