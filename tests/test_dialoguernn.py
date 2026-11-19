"""
Unit tests for DialogueRNN architecture, causality, batching, and gradient flow.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np

from src.models.dialoguernn import DialogueRNN, DialogueRNNConfig
from src.data.feature_loader import DialogueSequence, DialogueBatchDataset, dialogue_collate_fn


class TestDialogueRNN(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.config = DialogueRNNConfig(
            feature_dim=768,
            projected_dim=64,
            global_hidden_dim=32,
            party_hidden_dim=32,
            emotion_hidden_dim=16,
            fc_hidden_dim=16,
            num_classes=7,
            dropout=0.0,  # 0.0 for deterministic numerical tests
        )
        self.model = DialogueRNN(self.config)
        self.model.eval()

    def test_tensor_dimensions(self):
        """Test output tensor dimensions for batched and single dialogue inputs."""
        B, T, D = 4, 8, 768
        features = torch.randn(B, T, D)
        party_ids = torch.randint(0, 3, (B, T))

        logits = self.model(features, party_ids)
        self.assertEqual(logits.shape, (B, T, 7))

        # Single dialogue forward
        single_feats = torch.randn(T, D)
        single_parties = torch.randint(0, 2, (T,))
        single_logits = self.model.forward_dialogue(single_feats, single_parties)
        self.assertEqual(single_logits.shape, (T, 7))

    def test_forward_pass_finite_values(self):
        """Test forward pass produces finite, non-NaN, non-Inf logits."""
        B, T, D = 2, 6, 768
        features = torch.randn(B, T, D)
        party_ids = torch.randint(0, 2, (B, T))

        logits = self.model(features, party_ids)
        self.assertFalse(torch.isnan(logits).any().item())
        self.assertFalse(torch.isinf(logits).any().item())

    def test_variable_length_batches(self):
        """Test batching dialogues of varying lengths."""
        B = 3
        lengths = [3, 7, 10]
        max_len = max(lengths)
        features = torch.randn(B, max_len, 768)
        party_ids = torch.randint(0, 3, (B, max_len))
        mask = torch.zeros(B, max_len, dtype=torch.bool)
        for i, length in enumerate(lengths):
            mask[i, :length] = True

        logits = self.model(features, party_ids, mask=mask)
        self.assertEqual(logits.shape, (B, max_len, 7))
        self.assertFalse(torch.isnan(logits).any().item())

    def test_multiple_speakers(self):
        """Test dialogues with up to 9 distinct interlocutors."""
        T = 12
        features = torch.randn(1, T, 768)
        # 6 distinct speakers
        party_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 1, 4, 2, 3, 5]])

        logits = self.model(features, party_ids)
        self.assertEqual(logits.shape, (1, T, 7))
        self.assertFalse(torch.isnan(logits).any().item())

    def test_single_speaker_dialogues(self):
        """Test single-speaker monologue dialogues (all party_ids = 0)."""
        T = 5
        features = torch.randn(1, T, 768)
        party_ids = torch.zeros(1, T, dtype=torch.long)

        logits = self.model(features, party_ids)
        self.assertEqual(logits.shape, (1, T, 7))
        self.assertFalse(torch.isnan(logits).any().item())

    def test_padding_masks_loss(self):
        """Test that padding mask correctly excludes padded turns from loss."""
        features = torch.randn(2, 4, 768)
        party_ids = torch.zeros(2, 4, dtype=torch.long)
        targets = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 6]])
        mask = torch.tensor([[True, True, False, False], [True, True, True, True]])

        logits = self.model(features, party_ids)
        loss = self.model.compute_loss(logits, targets, mask=mask)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(loss.item(), 0.0)

        # Loss should be identical if padded target entries are altered
        targets_altered = targets.clone()
        targets_altered[0, 2] = 5
        targets_altered[0, 3] = 4
        loss_altered = self.model.compute_loss(logits, targets_altered, mask=mask)
        self.assertAlmostEqual(loss.item(), loss_altered.item(), places=5)

    def test_gradient_flow(self):
        """Test backpropagation ensures gradients flow to all architectural modules."""
        self.model.train()
        features = torch.randn(2, 4, 768, requires_grad=False)
        party_ids = torch.randint(0, 2, (2, 4))
        targets = torch.randint(0, 7, (2, 4))

        logits = self.model(features, party_ids)
        loss = self.model.compute_loss(logits, targets)
        loss.backward()

        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad, f"Parameter {name} has no gradient.")
            grad_norm = param.grad.norm().item()
            self.assertFalse(np.isnan(grad_norm), f"Parameter {name} gradient is NaN.")
            self.assertGreater(grad_norm, 0.0, f"Parameter {name} gradient is exactly zero.")

    def test_state_reset_between_dialogues(self):
        """Test that recurrent states reset cleanly between successive dialogues."""
        features_a = torch.randn(1, 5, 768)
        party_ids_a = torch.tensor([[0, 1, 0, 1, 0]])

        features_b = torch.randn(1, 4, 768)
        party_ids_b = torch.tensor([[0, 0, 1, 1]])

        logits_a1 = self.model(features_a, party_ids_a)
        # Interleaved forward on dialogue B
        _ = self.model(features_b, party_ids_b)
        # Re-run dialogue A
        logits_a2 = self.model(features_a, party_ids_a)

        self.assertTrue(
            torch.allclose(logits_a1, logits_a2, atol=1e-6),
            "Dialogue states leaked across calls! State reset failed.",
        )

    def test_causality_future_perturbations(self):
        """
        Critical causal test: Perturbing a future turn t* MUST NOT alter predictions
        at any prior turn t < t*.
        """
        T = 6
        features = torch.randn(1, T, 768)
        party_ids = torch.tensor([[0, 1, 0, 1, 0, 1]])

        base_logits = self.model(features, party_ids).squeeze(0)  # (T, 7)

        # Perturb turn t* = 3 (4th turn)
        perturbed_features = features.clone()
        perturbed_features[0, 3, :] += 50.0  # Large perturbation

        perturbed_logits = self.model(perturbed_features, party_ids).squeeze(0)  # (T, 7)

        # Check prior turns (t < 3: turns 0, 1, 2)
        for t in range(3):
            self.assertTrue(
                torch.allclose(base_logits[t], perturbed_logits[t], atol=1e-6),
                f"Causality violation! Prediction at prior turn {t} changed when turn 3 was perturbed.",
            )

        # Check turn 3 itself changed
        self.assertFalse(
            torch.allclose(base_logits[3], perturbed_logits[3], atol=1e-3),
            "Perturbed turn did not produce different logits.",
        )

    def test_deterministic_behavior_fixed_seeds(self):
        """Test identical initialization and forward passes with fixed seeds."""
        torch.manual_seed(1234)
        m1 = DialogueRNN(self.config)
        m1.eval()

        torch.manual_seed(1234)
        m2 = DialogueRNN(self.config)
        m2.eval()

        x = torch.randn(2, 5, 768)
        p = torch.randint(0, 2, (2, 5))

        out1 = m1(x, p)
        out2 = m2(x, p)
        self.assertTrue(torch.equal(out1, out2), "Models initialized with same seed produced different outputs.")

    def test_data_loader_batch_collate(self):
        """Test dataset wrapper and dialogue_collate_fn."""
        seq1 = DialogueSequence(
            dialogue_id=1,
            num_turns=3,
            utterance_ids=np.array([0, 1, 2]),
            speakers=["A", "B", "A"],
            party_ids=np.array([0, 1, 0]),
            party_names=["A", "B"],
            emotions=["neutral", "joy", "anger"],
            label_ids=np.array([0, 1, 2]),
            features=np.random.randn(3, 768).astype(np.float32),
        )
        seq2 = DialogueSequence(
            dialogue_id=2,
            num_turns=5,
            utterance_ids=np.array([0, 1, 2, 3, 4]),
            speakers=["C", "D", "C", "D", "C"],
            party_ids=np.array([0, 1, 0, 1, 0]),
            party_names=["C", "D"],
            emotions=["neutral", "surprise", "fear", "sadness", "disgust"],
            label_ids=np.array([0, 3, 4, 5, 6]),
            features=np.random.randn(5, 768).astype(np.float32),
        )

        batch_data = dialogue_collate_fn([seq1, seq2])
        self.assertEqual(batch_data["features"].shape, (2, 5, 768))
        self.assertEqual(batch_data["party_ids"].shape, (2, 5))
        self.assertEqual(batch_data["label_ids"].shape, (2, 5))
        self.assertEqual(batch_data["mask"].shape, (2, 5))
        self.assertTrue(batch_data["mask"][0, 2].item())
        self.assertFalse(batch_data["mask"][0, 3].item())
        self.assertEqual(batch_data["label_ids"][0, 3].item(), -100)


if __name__ == "__main__":
    unittest.main()
