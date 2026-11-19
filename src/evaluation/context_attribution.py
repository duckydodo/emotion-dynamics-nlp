"""
Context Perturbation and Context Attribution Analysis.

Measures conversational context sensitivity by perturbing or ablating prior dialogue history:
Delta L = logit_full(y) - logit_isolated(y)
Delta P = P_full(y) - P_isolated(y)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

from src.data.conversation import ConversationTurn
from src.models.transformer import LABELS


@dataclass
class TurnPerturbationResult:
    turn_id: int
    speaker: str
    text: str
    full_prediction: str
    full_confidence: float
    full_probabilities: Dict[str, float]
    isolated_prediction: str
    isolated_confidence: float
    isolated_probabilities: Dict[str, float]
    prediction_flipped: bool
    confidence_delta: float
    target_prob_delta: float
    entropy_full: float
    entropy_isolated: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "speaker": self.speaker,
            "text": self.text,
            "full_prediction": self.full_prediction,
            "full_confidence": round(self.full_confidence, 4),
            "full_probabilities": {k: round(v, 4) for k, v in self.full_probabilities.items()},
            "isolated_prediction": self.isolated_prediction,
            "isolated_confidence": round(self.isolated_confidence, 4),
            "isolated_probabilities": {k: round(v, 4) for k, v in self.isolated_probabilities.items()},
            "prediction_flipped": self.prediction_flipped,
            "confidence_delta": round(self.confidence_delta, 4),
            "target_prob_delta": round(self.target_prob_delta, 4),
            "entropy_full": round(self.entropy_full, 4),
            "entropy_isolated": round(self.entropy_isolated, 4),
        }


def compute_entropy(probs: Dict[str, float]) -> float:
    """Shannon entropy in nats over emotion class distribution."""
    p_vals = np.array(list(probs.values()), dtype=np.float64)
    p_vals = p_vals[p_vals > 1e-12]
    return float(-np.sum(p_vals * np.log(p_vals)))


class ContextPerturbationAnalyzer:
    """
    Evaluates conversational models under full context vs. isolated (0-context) turn conditions.
    """

    def __init__(self, model_instance):
        self.model = model_instance

    def analyze_dialogue(
        self,
        turns: List[ConversationTurn],
    ) -> List[TurnPerturbationResult]:
        if not turns:
            return []

        # 1. Full context predictions
        full_preds = self.model.predict(turns)

        # 2. Isolated predictions (each turn evaluated independently)
        isolated_preds = []
        for turn in turns:
            iso_turn = ConversationTurn(speaker=turn.speaker, text=turn.text, turn_id=0)
            iso_res = self.model.predict([iso_turn])
            isolated_preds.append(iso_res[0])

        results: List[TurnPerturbationResult] = []
        for full_p, iso_p, turn in zip(full_preds, isolated_preds, turns):
            full_pred = full_p["emotion"]
            full_conf = full_p["confidence"]
            full_probs = full_p["probabilities"]

            iso_pred = iso_p["emotion"]
            iso_conf = iso_p["confidence"]
            iso_probs = iso_p["probabilities"]

            target_prob_delta = full_probs.get(full_pred, 0.0) - iso_probs.get(full_pred, 0.0)
            conf_delta = full_conf - iso_conf
            flipped = (full_pred != iso_pred)

            results.append(TurnPerturbationResult(
                turn_id=turn.turn_id,
                speaker=turn.speaker,
                text=turn.text,
                full_prediction=full_pred,
                full_confidence=full_conf,
                full_probabilities=full_probs,
                isolated_prediction=iso_pred,
                isolated_confidence=iso_conf,
                isolated_probabilities=iso_probs,
                prediction_flipped=flipped,
                confidence_delta=conf_delta,
                target_prob_delta=target_prob_delta,
                entropy_full=compute_entropy(full_probs),
                entropy_isolated=compute_entropy(iso_probs),
            ))

        return results
