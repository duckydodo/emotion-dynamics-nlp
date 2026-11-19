"""
Unified Multi-Model Comparison Service.
Executes inference across the three canonical model families (DistilBERT, DialogueRNN, DialogueGCN)
on an identical set of conversational messages, verifies message alignment, and generates
cross-model agreement diagnostics.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Set, Tuple

from src.data.schema import CanonicalConversation, CanonicalMessage
from src.data.conversation import parse_canonical_conversation, ConversationTurn
from src.models.registry import CANONICAL_MODEL_IDS, normalize_model_id, get_model_metadata


class AlignmentMismatchError(Exception):
    """Raised when predictions from different models do not match the identical message sequence."""
    pass


@dataclass
class MessagePrediction:
    message_id: str
    message_index: int
    speaker: str
    text: str
    predicted_emotion: Optional[str]
    confidence: Optional[float]
    class_probabilities: Optional[Dict[str, float]]
    model_identifier: str
    is_evaluable: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelComparisonService:
    """
    Comparison service that executes selected canonical models on identical conversation messages.
    Guarantees strict message alignment and never silently compares disjoint or misaligned inputs.
    """

    def __init__(self, analyzer=None):
        from src.application.analyzer import ConversationAnalyzer
        self.analyzer = analyzer or ConversationAnalyzer()

    def compare(
        self,
        conversation_input: str | CanonicalConversation,
        model_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run multi-model comparison across selected models.

        Parameters
        ----------
        conversation_input : str or CanonicalConversation
            Raw conversation string or CanonicalConversation instance.
        model_ids : Optional[List[str]]
            List of model identifiers to compare. Defaults to all 3 canonical models.

        Returns
        -------
        Dict[str, Any] containing message-level comparison, consensus diagnostics,
        pairwise agreement metrics, and model status.
        """
        # 1. Parse into CanonicalConversation
        if isinstance(conversation_input, CanonicalConversation):
            canonical = conversation_input
        elif isinstance(conversation_input, str):
            canonical = self.analyzer.parse_canonical(conversation_input)
        else:
            raise ValueError(f"Unsupported conversation input type: {type(conversation_input)}")

        evaluable_turns = canonical.get_evaluable_turns()
        if not evaluable_turns:
            raise ValueError("No valid evaluable conversation turns were found to compare.")

        # 2. Normalize and validate requested models
        if model_ids is None or len(model_ids) == 0:
            target_models = list(CANONICAL_MODEL_IDS)
        else:
            target_models = [normalize_model_id(m) for m in model_ids]
            # Deduplicate while preserving order
            seen: Set[str] = set()
            target_models = [m for m in target_models if not (m in seen or seen.add(m))]

        # 3. Collect evaluable messages from canonical representation
        evaluable_messages: List[CanonicalMessage] = [
            m for m in canonical.messages if m.is_evaluable and m.text.strip()
        ]
        expected_message_ids = [m.message_id for m in evaluable_messages]

        assert len(evaluable_messages) == len(evaluable_turns), (
            f"Message count mismatch: {len(evaluable_messages)} messages vs {len(evaluable_turns)} turns"
        )

        # 4. Execute inference per model
        model_predictions: Dict[str, List[Dict[str, Any]]] = {}
        model_statuses: Dict[str, Dict[str, Any]] = {}

        for m_id in target_models:
            meta = get_model_metadata(m_id)
            if not meta.is_available:
                model_statuses[m_id] = {
                    "status": "unavailable",
                    "error": f"Checkpoint not found at {meta.checkpoint_path}",
                }
                continue

            try:
                model_instance = self.analyzer._get_model(m_id)
                raw_preds = model_instance.predict(evaluable_turns)

                # Alignment verification: exact length match
                if len(raw_preds) != len(evaluable_messages):
                    raise AlignmentMismatchError(
                        f"Model '{m_id}' returned {len(raw_preds)} predictions, "
                        f"expected {len(evaluable_messages)} matching canonical messages."
                    )

                # Format predictions aligned to canonical messages
                formatted_preds = []
                for turn_pred, msg in zip(raw_preds, evaluable_messages):
                    formatted_preds.append({
                        "message_id": msg.message_id,
                        "message_index": msg.source_index,
                        "speaker": msg.speaker,
                        "text": msg.text,
                        "predicted_emotion": turn_pred["emotion"],
                        "confidence": turn_pred["confidence"],
                        "class_probabilities": turn_pred["probabilities"],
                        "model_identifier": m_id,
                        "is_evaluable": True,
                    })

                model_predictions[m_id] = formatted_preds
                model_statuses[m_id] = {
                    "status": "success",
                    "num_predictions": len(formatted_preds),
                }

            except Exception as exc:
                model_statuses[m_id] = {
                    "status": "error",
                    "error": str(exc),
                }

        successful_models = [m for m in target_models if m in model_predictions]
        if not successful_models:
            raise RuntimeError("All requested models failed during inference.")

        # 5. Strict Cross-Model Alignment Assertion
        # Verify that all successful models predicted on the identical message IDs in identical order
        for m_id in successful_models:
            m_ids = [p["message_id"] for p in model_predictions[m_id]]
            if m_ids != expected_message_ids:
                raise AlignmentMismatchError(
                    f"Model '{m_id}' message IDs do not match canonical sequence: {m_ids} != {expected_message_ids}"
                )

        # 6. Build per-message comparison table
        num_turns = len(evaluable_messages)
        message_comparisons = []
        full_agreement_count = 0

        for t_idx in range(num_turns):
            msg = evaluable_messages[t_idx]
            turn_preds: Dict[str, Dict[str, Any]] = {}
            emotions_here = []

            for m_id in successful_models:
                p = model_predictions[m_id][t_idx]
                turn_preds[m_id] = {
                    "predicted_emotion": p["predicted_emotion"],
                    "confidence": p["confidence"],
                    "class_probabilities": p["class_probabilities"],
                }
                emotions_here.append(p["predicted_emotion"])

            # Consensus & agreement
            is_unanimous = (len(set(emotions_here)) == 1) if emotions_here else False
            if is_unanimous:
                full_agreement_count += 1

            most_common_emo, count = Counter(emotions_here).most_common(1)[0]
            consensus_emotion = most_common_emo if count > len(emotions_here) // 2 else "no_consensus"

            message_comparisons.append({
                "message_id": msg.message_id,
                "message_index": msg.source_index,
                "speaker": msg.speaker,
                "text": msg.text,
                "models": turn_preds,
                "is_unanimous": is_unanimous,
                "consensus_emotion": consensus_emotion,
            })

        # 7. Pairwise Agreement Matrix
        pairwise_agreement: Dict[str, Dict[str, float]] = {}
        for m1 in successful_models:
            pairwise_agreement[m1] = {}
            preds1 = [p["predicted_emotion"] for p in model_predictions[m1]]
            for m2 in successful_models:
                preds2 = [p["predicted_emotion"] for p in model_predictions[m2]]
                matches = sum(1 for e1, e2 in zip(preds1, preds2) if e1 == e2)
                rate = float(matches / num_turns) if num_turns > 0 else 1.0
                pairwise_agreement[m1][m2] = round(rate, 4)

        # 8. Emotion distributions per model
        model_distributions: Dict[str, Dict[str, int]] = {}
        for m_id in successful_models:
            emotions = [p["predicted_emotion"] for p in model_predictions[m_id]]
            counts = Counter(emotions)
            model_distributions[m_id] = {emo: counts.get(emo, 0) for emo in set(emotions)}

        consensus_rate = float(full_agreement_count / num_turns) if num_turns > 0 else 1.0

        return {
            "conversation_id": canonical.conversation_id,
            "num_messages_evaluated": num_turns,
            "compared_models": successful_models,
            "model_statuses": model_statuses,
            "consensus_rate": round(consensus_rate, 4),
            "pairwise_agreement": pairwise_agreement,
            "model_distributions": model_distributions,
            "messages": message_comparisons,
        }
