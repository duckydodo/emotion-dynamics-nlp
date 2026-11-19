"""
Unified Model Inference Module.
Provides inference wrappers for the three canonical model families:
1. DistilBERT (Utterance-only baseline)
2. DialogueRNN (Causal recurrent state tracking)
3. DialogueGCN (Relational graph convolution)

Includes SharedDistilBertEncoder to avoid duplicating the 67M-parameter DistilBERT
feature extraction weights in memory.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import torch

from src.data.conversation import ConversationTurn
from src.models.transformer import TransformerEmotionClassifier, LABELS, ID_TO_LABEL


CHECKPOINTS = {
    "distilbert": Path("checkpoints/transformer_baseline/best"),
    "utterance": Path("checkpoints/transformer_baseline/best"),
    "context_1": Path("checkpoints/transformer_context_1/best"),
    "context_1_speaker": Path("checkpoints/transformer_context_1_speaker/best"),
    "dialoguernn": Path("checkpoints/dialoguernn_unweighted/best_model.pt"),
    "dialoguegcn": Path("checkpoints/dialoguegcn/best_model.pt"),
}


class SharedDistilBertEncoder:
    """
    Singleton cache for the frozen DistilBERT baseline encoder.
    Prevents loading duplicate transformer weights into RAM across DialogueRNN and DialogueGCN.
    """
    _tokenizer = None
    _encoder = None
    _loaded_path: Optional[Path] = None

    @classmethod
    def get_encoder(cls, model_path: Path | str = CHECKPOINTS["distilbert"]):
        path = Path(model_path)
        if cls._encoder is None or cls._loaded_path != path:
            if not path.exists():
                raise FileNotFoundError(f"DistilBERT encoder not found: {path}")
            from transformers import AutoTokenizer, DistilBertModel
            cls._tokenizer = AutoTokenizer.from_pretrained(path)
            cls._encoder = DistilBertModel.from_pretrained(path)
            cls._encoder.eval()
            cls._loaded_path = path
        return cls._tokenizer, cls._encoder

    @classmethod
    def encode_texts(
        cls,
        texts: List[str],
        max_length: int = 96,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Extracts 768-dim final-layer [CLS] hidden states for an utterance sequence.
        """
        tokenizer, encoder = cls.get_encoder()
        clean_texts = [str(t).strip() if t else "" for t in texts]

        with torch.no_grad():
            encoded = tokenizer(
                clean_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls_features = outputs.last_hidden_state[:, 0, :]  # (T, 768)

        return cls_features


class EmotionInference:
    """
    Application inference wrapper for DistilBERT transformer models.
    Supports canonical 'distilbert' (utterance-only) as well as legacy context modes.
    """

    def __init__(self, checkpoint_path: Path | str = CHECKPOINTS["distilbert"]):
        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}"
            )

        self.classifier = TransformerEmotionClassifier(
            model_path=self.checkpoint_path
        )
        self.model_identifier = "distilbert"

    def predict(self, turns: List[ConversationTurn]) -> List[Dict[str, Any]]:
        if not turns:
            return []

        # Canonical DistilBERT utterance-only path
        if self.checkpoint_path.parent.name in ("transformer_baseline", "best"):
            texts = [turn.text.strip() if turn.text else "" for turn in turns]
            predictions = self.classifier.predict(texts)
            probabilities = self.classifier.predict_proba(texts)

            results = []
            for turn, prediction, probability in zip(turns, predictions, probabilities):
                prob_dict = {
                    label: float(probability[idx])
                    for idx, label in enumerate(LABELS)
                }
                confidence = float(probability.max())
                results.append({
                    "turn_id": turn.turn_id,
                    "speaker": turn.speaker,
                    "text": turn.text,
                    "emotion": prediction,
                    "confidence": confidence,
                    "probabilities": prob_dict,
                    "model_identifier": self.model_identifier,
                })
            return results

        # Legacy context concatenation modes
        utterance_model = EmotionInference(CHECKPOINTS["distilbert"])
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
            prob_dict = {
                label: float(probability[idx])
                for idx, label in enumerate(LABELS)
            }
            results.append({
                "turn_id": turn.turn_id,
                "speaker": turn.speaker,
                "text": turn.text,
                "emotion": prediction,
                "confidence": float(probability.max()),
                "probabilities": prob_dict,
                "model_identifier": self.model_identifier,
            })
        return results

    def _build_model_inputs(self, turns, previous_turn):
        include_speaker = (
            self.checkpoint_path.parent.name == "transformer_context_1_speaker"
        )
        inputs = []
        for turn in turns:
            if include_speaker:
                previous_text = f"[SPEAKER_{previous_turn.speaker}] {previous_turn.text.strip()}"
                current_text = f"[SPEAKER_{turn.speaker}] {turn.text.strip()}"
            else:
                previous_text = previous_turn.text.strip()
                current_text = turn.text.strip()

            inputs.append(f"{previous_text}\n{current_text}")
            previous_turn = turn
        return inputs


class DialogueRNNInference:
    """
    Application inference wrapper for DialogueRNN (Majumder et al., AAAI 2019).
    Uses the SharedDistilBertEncoder for feature extraction, maps participants to
    dynamic party states, and executes causal recurrent state tracking across turns.
    """

    def __init__(
        self,
        checkpoint_path: Path | str = CHECKPOINTS["dialoguernn"],
        encoder_path: Path | str = CHECKPOINTS["distilbert"],
        device: str = "cpu",
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.encoder_path = Path(encoder_path)
        self.device = torch.device(device)
        self.model_identifier = "dialoguernn"

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"DialogueRNN checkpoint not found: {self.checkpoint_path}")

        from src.models.dialoguernn import DialogueRNN, DialogueRNNConfig

        saved_state = torch.load(self.checkpoint_path, map_location=self.device)
        self.config = DialogueRNNConfig.from_dict(saved_state["config"])
        self.model = DialogueRNN(self.config)
        self.model.load_state_dict(saved_state["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, turns: List[ConversationTurn]) -> List[Dict[str, Any]]:
        if not turns:
            return []

        texts = [turn.text.strip() if turn.text else "" for turn in turns]

        # 1. Shared feature extraction
        cls_features = SharedDistilBertEncoder.encode_texts(
            texts,
            device=str(self.device),
        )  # (T, 768)

        # 2. Dynamic party mapping
        party_map: Dict[str, int] = {}
        party_ids: List[int] = []
        for turn in turns:
            spk = turn.speaker.strip() if (turn.speaker and turn.speaker.strip()) else "Speaker"
            if spk not in party_map:
                party_map[spk] = len(party_map)
            party_ids.append(party_map[spk])

        party_tensor = torch.tensor(party_ids, dtype=torch.long, device=self.device)

        # 3. Causal recurrent forward pass
        with torch.no_grad():
            features_b = cls_features.unsqueeze(0).to(self.device)  # (1, T, 768)
            party_b = party_tensor.unsqueeze(0)                     # (1, T)
            logits_b = self.model(features_b, party_b)              # (1, T, num_classes)
            logits = logits_b.squeeze(0)                            # (T, num_classes)
            probs = torch.softmax(logits, dim=-1)                  # (T, num_classes)
            preds = torch.argmax(logits, dim=-1)                   # (T,)

        results: List[Dict[str, Any]] = []
        for i, turn in enumerate(turns):
            pred_id = int(preds[i].item())
            prob_vec = probs[i].tolist()
            prob_dict = {label: float(prob_vec[idx]) for idx, label in enumerate(LABELS)}

            results.append({
                "turn_id": turn.turn_id,
                "speaker": turn.speaker,
                "text": turn.text,
                "emotion": ID_TO_LABEL[pred_id],
                "confidence": float(max(prob_vec)),
                "probabilities": prob_dict,
                "model_identifier": self.model_identifier,
            })
        return results


class DialogueGCNInference:
    """
    Application inference wrapper for DialogueGCN (Ghosal et al., EMNLP 2019).
    Uses the SharedDistilBertEncoder for feature extraction, constructs windowed
    relational conversational graphs, and computes 7-class emotion predictions.
    """

    def __init__(
        self,
        checkpoint_path: Path | str = CHECKPOINTS["dialoguegcn"],
        encoder_path: Path | str = CHECKPOINTS["distilbert"],
        device: str = "cpu",
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.encoder_path = Path(encoder_path)
        self.device = torch.device(device)
        self.model_identifier = "dialoguegcn"

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"DialogueGCN checkpoint not found: {self.checkpoint_path}")

        from src.models.dialoguegcn import DialogueGCN, DialogueGCNConfig

        self.model, self.metadata = DialogueGCN.load_checkpoint(
            self.checkpoint_path,
            device=self.device,
        )
        self.model.eval()

    def predict(self, turns: List[ConversationTurn]) -> List[Dict[str, Any]]:
        if not turns:
            return []

        texts = [turn.text.strip() if turn.text else "" for turn in turns]

        # 1. Shared feature extraction
        cls_features = SharedDistilBertEncoder.encode_texts(
            texts,
            device=str(self.device),
        )  # (T, 768)

        # 2. Dynamic party mapping
        party_map: Dict[str, int] = {}
        party_ids: List[int] = []
        for turn in turns:
            spk = turn.speaker.strip() if (turn.speaker and turn.speaker.strip()) else "Speaker"
            if spk not in party_map:
                party_map[spk] = len(party_map)
            party_ids.append(party_map[spk])

        party_tensor = torch.tensor(party_ids, dtype=torch.long, device=self.device)

        # 3. Graph convolution forward pass
        with torch.no_grad():
            features_b = cls_features.unsqueeze(0).to(self.device)  # (1, T, 768)
            party_b = party_tensor.unsqueeze(0)                     # (1, T)
            lengths = torch.tensor([len(turns)], device=self.device)
            logits_b = self.model(features_b, party_ids=party_b, lengths=lengths)
            logits = logits_b.squeeze(0)                            # (T, num_classes)
            probs = torch.softmax(logits, dim=-1)                  # (T, num_classes)
            preds = torch.argmax(logits, dim=-1)                   # (T,)

        results: List[Dict[str, Any]] = []
        for i, turn in enumerate(turns):
            pred_id = int(preds[i].item())
            prob_vec = probs[i].tolist()
            prob_dict = {label: float(prob_vec[idx]) for idx, label in enumerate(LABELS)}

            results.append({
                "turn_id": turn.turn_id,
                "speaker": turn.speaker,
                "text": turn.text,
                "emotion": ID_TO_LABEL[pred_id],
                "confidence": float(max(prob_vec)),
                "probabilities": prob_dict,
                "model_identifier": self.model_identifier,
            })
        return results
