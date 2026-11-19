"""
Prediction Explainability: Text-Level Feature Attribution for DistilBERT.

Implements Integrated Gradients (Sundararajan et al., 2017) and Gradient x Input
for DistilBERT utterance classifications, projecting salience back to subwords and words.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

from src.models.transformer import TransformerEmotionClassifier, LABELS, LABEL_TO_ID, ID_TO_LABEL


@dataclass
class TokenAttribution:
    token: str
    score: float
    is_special: bool = False


@dataclass
class WordAttribution:
    word: str
    score: float
    tokens: List[str]


@dataclass
class ExplanationResult:
    text: str
    predicted_emotion: str
    target_emotion: str
    confidence: float
    token_attributions: List[TokenAttribution]
    word_attributions: List[WordAttribution]
    probabilities: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "predicted_emotion": self.predicted_emotion,
            "target_emotion": self.target_emotion,
            "confidence": round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "word_attributions": [
                {"word": wa.word, "score": round(wa.score, 4), "tokens": wa.tokens}
                for wa in self.word_attributions
            ],
            "token_attributions": [
                {"token": ta.token, "score": round(ta.score, 4), "is_special": ta.is_special}
                for ta in self.token_attributions
            ],
        }


class IntegratedGradientsExplainer:
    """
    Computes path-integrated gradients over the DistilBERT embedding layer.
    """

    def __init__(
        self,
        classifier: TransformerEmotionClassifier,
        steps: int = 20,
    ):
        self.classifier = classifier
        self.model = classifier.model
        self.tokenizer = classifier.tokenizer
        self.device = classifier.device
        self.steps = steps
        self.embedding_layer = self.model.distilbert.embeddings.word_embeddings

    def explain(
        self,
        text: str,
        target_emotion: Optional[str] = None,
    ) -> ExplanationResult:
        self.model.eval()
        clean_text = text.strip()

        # Tokenize
        encoded = self.tokenizer(
            clean_text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        ).to(self.device)

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        # Baseline: zero embeddings or pad token
        orig_embeddings = self.embedding_layer(input_ids).detach()
        baseline_embeddings = torch.zeros_like(orig_embeddings)

        # Forward pass for base predictions
        with torch.no_grad():
            base_logits = self.model(
                inputs_embeds=orig_embeddings,
                attention_mask=attention_mask,
            ).logits
            probs = F.softmax(base_logits, dim=-1).squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_label = ID_TO_LABEL[pred_idx]
        confidence = float(probs[pred_idx])

        if target_emotion is not None and target_emotion.lower() in LABEL_TO_ID:
            target_idx = LABEL_TO_ID[target_emotion.lower()]
            target_label = target_emotion.lower()
        else:
            target_idx = pred_idx
            target_label = pred_label

        # Accumulate gradients across interpolation steps
        grads_sum = torch.zeros_like(orig_embeddings)
        diff = orig_embeddings - baseline_embeddings

        for step in range(1, self.steps + 1):
            alpha = float(step) / float(self.steps)
            interp_embeddings = baseline_embeddings + alpha * diff
            interp_embeddings.requires_grad_(True)

            logits = self.model(
                inputs_embeds=interp_embeddings,
                attention_mask=attention_mask,
            ).logits

            target_logit = logits[0, target_idx]
            self.model.zero_grad()
            grad = torch.autograd.grad(target_logit, interp_embeddings)[0]
            grads_sum += grad.detach()

        avg_grads = grads_sum / float(self.steps)
        integrated_grads = (diff * avg_grads).sum(dim=-1).squeeze(0).cpu().numpy()

        # Normalize score magnitude for readability (L1 norm scaling)
        norm_factor = np.linalg.norm(integrated_grads, ord=1)
        if norm_factor > 1e-8:
            normalized_scores = integrated_grads / norm_factor
        else:
            normalized_scores = integrated_grads

        # Construct token attributions
        token_attrs: List[TokenAttribution] = []
        for tok, score in zip(tokens, normalized_scores):
            is_special = tok in (self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token)
            token_attrs.append(TokenAttribution(
                token=tok,
                score=float(score),
                is_special=is_special,
            ))

        # Aggregate subwords into whole words
        word_attrs = self._aggregate_tokens_to_words(tokens, normalized_scores)

        prob_dict = {label: float(probs[idx]) for idx, label in enumerate(LABELS)}

        return ExplanationResult(
            text=clean_text,
            predicted_emotion=pred_label,
            target_emotion=target_label,
            confidence=confidence,
            token_attributions=token_attrs,
            word_attributions=word_attrs,
            probabilities=prob_dict,
        )

    def _aggregate_tokens_to_words(
        self,
        tokens: List[str],
        scores: np.ndarray,
    ) -> List[WordAttribution]:
        """
        Merge WordPiece subwords ('un', '##believable') into full words with summed attribution.
        """
        words: List[WordAttribution] = []
        curr_word_tokens: List[str] = []
        curr_score = 0.0

        for tok, score in zip(tokens, scores):
            if tok in (self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token):
                continue

            if tok.startswith("##"):
                curr_word_tokens.append(tok)
                curr_score += float(score)
            else:
                if curr_word_tokens:
                    merged = "".join([t.replace("##", "") for t in curr_word_tokens])
                    words.append(WordAttribution(word=merged, score=curr_score, tokens=list(curr_word_tokens)))
                curr_word_tokens = [tok]
                curr_score = float(score)

        if curr_word_tokens:
            merged = "".join([t.replace("##", "") for t in curr_word_tokens])
            words.append(WordAttribution(word=merged, score=curr_score, tokens=list(curr_word_tokens)))

        return words
