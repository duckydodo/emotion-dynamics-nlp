"""
Unit tests for Phase 2 Exploratory Models (Temporal CNN, CRF) and Phase 3 Explainability Modules.
"""

from __future__ import annotations

import unittest
import torch
import numpy as np

from src.models.temporal_cnn import TemporalCNN
from src.models.crf import LinearChainCRF
from src.evaluation.attribution import TokenAttribution, WordAttribution, ExplanationResult
from src.evaluation.context_attribution import compute_entropy
from src.evaluation.disagreement import cohens_kappa, analyze_model_disagreements


class TestTemporalCNN(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 5
        self.input_dim = 768
        self.num_classes = 7
        self.model = TemporalCNN(
            input_dim=self.input_dim,
            hidden_dim=64,
            num_classes=self.num_classes,
            kernel_size=3,
            dilations=(1, 2, 4),
        )

    def test_forward_shape(self):
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.bool)
        mask[0, 3:] = False

        logits = self.model(x, mask=mask)
        self.assertEqual(logits.shape, (self.batch_size, self.seq_len, self.num_classes))

    def test_loss_computation(self):
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.bool)
        targets = torch.tensor([[0, 1, 4, -100, -100], [4, 4, 3, 2, 0]], dtype=torch.long)
        mask[0, 3:] = False

        logits = self.model(x, mask=mask)
        loss = self.model.compute_loss(logits, targets, mask=mask)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)

    def test_receptive_field(self):
        # k=3, dilations=(1, 2, 4) -> rf = 1 + 2 + 4 + 8 = 15
        self.assertEqual(self.model.receptive_field, 15)


class TestLinearChainCRF(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 4
        self.input_dim = 768
        self.num_classes = 7
        self.model = LinearChainCRF(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
        )

    def test_forward_and_loss(self):
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.bool)
        targets = torch.tensor([[0, 4, -100, -100], [1, 2, 3, 4]], dtype=torch.long)
        mask[0, 2:] = False

        emissions = self.model(x)
        self.assertEqual(emissions.shape, (self.batch_size, self.seq_len, self.num_classes))

        loss = self.model.compute_loss(emissions, targets, mask=mask)
        self.assertTrue(torch.isfinite(loss))

    def test_viterbi_and_causal_decoding(self):
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.bool)
        mask[0, 2:] = False

        emissions = self.model(x)
        viterbi_paths = self.model.decode(emissions, mask=mask)
        causal_paths = self.model.decode_causal(emissions, mask=mask)

        # Sequence 0 has 2 turns, Sequence 1 has 4 turns
        self.assertEqual(len(viterbi_paths[0]), 2)
        self.assertEqual(len(viterbi_paths[1]), 4)
        self.assertEqual(len(causal_paths[0]), 2)
        self.assertEqual(len(causal_paths[1]), 4)


class TestExplainabilityAndAttribution(unittest.TestCase):
    def test_entropy_computation(self):
        # Decisive probability
        p_decisive = {"anger": 0.98, "neutral": 0.02}
        h_decisive = compute_entropy(p_decisive)
        # Uniform probability
        p_uniform = {k: 1/7 for k in ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]}
        h_uniform = compute_entropy(p_uniform)

        self.assertLess(h_decisive, h_uniform)
        self.assertAlmostEqual(h_uniform, float(np.log(7)), places=3)

    def test_explanation_result_serialization(self):
        res = ExplanationResult(
            text="Hello world",
            predicted_emotion="joy",
            target_emotion="joy",
            confidence=0.85,
            token_attributions=[
                TokenAttribution(token="hello", score=0.4),
                TokenAttribution(token="world", score=0.6),
            ],
            word_attributions=[
                WordAttribution(word="hello", score=0.4, tokens=["hello"]),
                WordAttribution(word="world", score=0.6, tokens=["world"]),
            ],
            probabilities={"joy": 0.85, "neutral": 0.15},
        )
        d = res.to_dict()
        self.assertEqual(d["predicted_emotion"], "joy")
        self.assertEqual(len(d["word_attributions"]), 2)
        self.assertEqual(d["word_attributions"][0]["word"], "hello")


class TestDisagreementTaxonomy(unittest.TestCase):
    def test_cohens_kappa_perfect_and_imperfect(self):
        p1 = ["joy", "anger", "neutral", "sadness"]
        p2 = ["joy", "anger", "neutral", "sadness"]
        self.assertAlmostEqual(cohens_kappa(p1, p2), 1.0, places=4)

        p3 = ["joy", "joy", "joy", "joy"]
        self.assertLess(cohens_kappa(p1, p3), 0.5)

    def test_disagreement_taxonomy_classification(self):
        gold = ["anger", "joy", "surprise", "disgust"]
        db = ["neutral", "joy", "surprise", "neutral"]    # Turn 0 wrong, Turn 3 wrong
        rnn = ["anger", "joy", "surprise", "neutral"]     # Turn 0 correct (Context recovery)
        gcn = ["anger", "joy", "anger", "disgust"]        # Turn 0 correct, Turn 2 wrong (Dilution), Turn 3 correct (GCN win)

        tax = analyze_model_disagreements(gold, db, rnn, gcn)
        self.assertEqual(tax.total_turns, 4)
        self.assertIn("context_recovery_both", tax.taxonomy_counts)
        self.assertIn("graph_relational_win", tax.taxonomy_counts)
        self.assertIn("all_correct", tax.taxonomy_counts)


if __name__ == "__main__":
    unittest.main()
