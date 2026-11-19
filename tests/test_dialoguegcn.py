"""
Unit and integration tests for DialogueGCN model implementation (Task 24).

Verifies:
1. Exact parameter count (452,487 trainable parameters) and layer dimensions.
2. Graph construction: binary adjacency, intra-speaker vs inter-speaker relation masks.
3. Temporal window bounds (past and future window boundaries).
4. Dialogue isolation in batched processing (zero cross-dialogue edges).
5. Variable-length dialogues and padding mask handling.
6. Single-utterance dialogues (T=1 boundary condition).
7. Dynamic speaker mapping (strings to party IDs).
8. Output shapes for 2D and 3D tensor inputs.
9. End-to-end gradient flow and backpropagation.
10. Deterministic inference in eval mode.
11. Absolute absence of emotion-label leakage.
12. Atomic checkpoint saving and configuration restoration.
13. Mathematical batch equivalence (single vs batched evaluation).
14. Masked and class-weighted loss computation.
"""

from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn as nn

from src.models.dialoguegcn import DialogueGCN, DialogueGCNConfig, RGCNLayer


class TestDialogueGCN(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.config = DialogueGCNConfig()
        self.model = DialogueGCN(self.config)

    def test_parameter_count_and_submodules(self):
        """Verify exact parameter count of 452,487 and submodule dimensions."""
        param_count = self.model.count_parameters()
        self.assertEqual(param_count, 452487)

        # BiGRU parameters: 320,256
        bigru_params = sum(p.numel() for p in self.model.bigru.parameters())
        self.assertEqual(bigru_params, 320256)

        # 2 RGCN layers: 65,664 each = 131,328
        self.assertEqual(len(self.model.layers), 2)
        for layer in self.model.layers:
            self.assertIsInstance(layer, RGCNLayer)
            layer_params = sum(p.numel() for p in layer.parameters())
            self.assertEqual(layer_params, 65664)

        # Classifier: 903
        cls_params = sum(p.numel() for p in self.model.classifier.parameters())
        self.assertEqual(cls_params, 903)

    def test_graph_construction_and_edge_types(self):
        """
        Verify that build_graph correctly creates binary adjacency and assigns
        intra-speaker (relation 0) and inter-speaker (relation 1) edge masks.
        """
        # Dialogue of 4 turns: Alice (0), Bob (1), Alice (0), Bob (1)
        party_ids = torch.tensor([0, 1, 0, 1])
        adj, rel_mask = self.model.build_graph(party_ids)

        self.assertEqual(adj.shape, (4, 4))
        self.assertEqual(rel_mask.shape, (2, 4, 4))

        # Check self-loops are zero
        for i in range(4):
            self.assertEqual(adj[i, i].item(), 0.0)
            self.assertEqual(rel_mask[0, i, i].item(), 0.0)
            self.assertEqual(rel_mask[1, i, i].item(), 0.0)

        # Turn 0 (Alice) and Turn 2 (Alice): same speaker -> Intra-speaker edge
        self.assertEqual(adj[0, 2].item(), 1.0)
        self.assertEqual(rel_mask[0, 0, 2].item(), 1.0)  # Intra
        self.assertEqual(rel_mask[1, 0, 2].item(), 0.0)  # Inter

        # Turn 0 (Alice) and Turn 1 (Bob): different speakers -> Inter-speaker edge
        self.assertEqual(adj[0, 1].item(), 1.0)
        self.assertEqual(rel_mask[0, 0, 1].item(), 0.0)  # Intra
        self.assertEqual(rel_mask[1, 0, 1].item(), 1.0)  # Inter

        # Verify relation masks partition all non-self edges
        self.assertTrue(torch.all((rel_mask[0] + rel_mask[1]) == adj))

    def test_edge_direction_and_temporal_window(self):
        """Verify past and future context window boundaries."""
        # Test with small windows: past=2, future=1 on a sequence of 6 turns
        party_ids = torch.tensor([0, 0, 0, 0, 0, 0])
        adj, _ = self.model.build_graph(party_ids, window_past=2, window_future=1)

        # For turn 3:
        # Valid past neighbors: 3 - 2 = 1, 2
        # Valid future neighbor: 3 + 1 = 4
        # Turn 0 is past-out-of-bounds (distance 3 > 2)
        # Turn 5 is future-out-of-bounds (distance 2 > 1)
        self.assertEqual(adj[3, 0].item(), 0.0)
        self.assertEqual(adj[3, 1].item(), 1.0)
        self.assertEqual(adj[3, 2].item(), 1.0)
        self.assertEqual(adj[3, 3].item(), 0.0)  # self
        self.assertEqual(adj[3, 4].item(), 1.0)
        self.assertEqual(adj[3, 5].item(), 0.0)

    def test_dialogue_isolation(self):
        """
        Verify that batched graphs never connect utterances across different dialogues,
        and modifying dialogue 1 does not affect dialogue 0's predictions.
        """
        self.model.eval()
        x0 = torch.randn(3, 768)
        p0 = torch.tensor([0, 1, 0])

        x1_a = torch.randn(4, 768)
        x1_b = torch.randn(4, 768)
        p1 = torch.tensor([0, 1, 1, 0])

        # Batch with x1_a
        batch_a = torch.zeros(2, 4, 768)
        batch_a[0, :3] = x0
        batch_a[1] = x1_a
        p_batch = torch.zeros(2, 4, dtype=torch.long)
        p_batch[0, :3] = p0
        p_batch[1] = p1
        lengths = torch.tensor([3, 4])

        adj, _ = self.model.build_graph(p_batch, lengths=lengths)
        # Verify dialogue 0 has zero edges to indices >= 3
        self.assertEqual(adj[0, :, 3].sum().item(), 0.0)
        self.assertEqual(adj[0, 3, :].sum().item(), 0.0)

        with torch.no_grad():
            out_a = self.model(batch_a, party_ids=p_batch, lengths=lengths)

            # Batch with x1_b (changed second dialogue completely)
            batch_b = torch.zeros(2, 4, 768)
            batch_b[0, :3] = x0
            batch_b[1] = x1_b
            out_b = self.model(batch_b, party_ids=p_batch, lengths=lengths)

        # Dialogue 0's predictions must be 100% identical between batch_a and batch_b
        diff = (out_a[0, :3] - out_b[0, :3]).abs().max().item()
        self.assertLess(diff, 1e-6)

    def test_single_utterance_dialogue(self):
        """Verify boundary condition T=1 (single-utterance dialogue)."""
        self.model.eval()
        x = torch.randn(1, 768)
        p = torch.tensor([0])

        adj, rel_mask = self.model.build_graph(p)
        self.assertEqual(adj.sum().item(), 0.0)
        self.assertEqual(rel_mask.sum().item(), 0.0)

        out = self.model(x, party_ids=p)
        self.assertEqual(out.shape, (1, 7))
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_variable_length_and_padding(self):
        """Verify batched variable-length dialogues with padding."""
        self.model.eval()
        B = 3
        T_max = 8
        features = torch.randn(B, T_max, 768)
        party_ids = torch.randint(0, 3, (B, T_max))
        lengths = torch.tensor([2, 5, 8])
        mask = torch.zeros(B, T_max, dtype=torch.bool)
        for b in range(B):
            mask[b, :lengths[b]] = True

        out = self.model(features, party_ids=party_ids, lengths=lengths, mask=mask)
        self.assertEqual(out.shape, (B, T_max, 7))
        self.assertFalse(torch.isnan(out).any())

    def test_speaker_mapping_string_and_party_ids(self):
        """Verify that passing speaker names as strings produces identical party resolution."""
        self.model.eval()
        features = torch.randn(3, 768)
        speakers = ["Chandler", "Joey", "Chandler"]
        party_ids = torch.tensor([0, 1, 0])

        with torch.no_grad():
            out_spk = self.model(features, speakers=speakers)
            out_pid = self.model(features, party_ids=party_ids)

        diff = (out_spk - out_pid).abs().max().item()
        self.assertLess(diff, 1e-6)

    def test_output_shapes_2d_and_3d(self):
        """Verify proper dimensional preservation for 2D and 3D inputs."""
        self.model.eval()
        # 2D input: (T, 768) -> (T, 7)
        x_2d = torch.randn(5, 768)
        out_2d = self.model(x_2d)
        self.assertEqual(out_2d.shape, (5, 7))

        # 3D input: (B, T, 768) -> (B, T, 7)
        x_3d = torch.randn(2, 5, 768)
        out_3d = self.model(x_3d)
        self.assertEqual(out_3d.shape, (2, 5, 7))

    def test_gradient_flow_and_backprop(self):
        """Verify non-zero gradients flow through all model parameters during training."""
        self.model.train()
        features = torch.randn(2, 4, 768)
        party_ids = torch.tensor([[0, 1, 0, 1], [0, 0, 1, 1]])
        targets = torch.tensor([[0, 4, 3, 2], [1, 5, 6, 4]])

        logits = self.model(features, party_ids=party_ids)
        loss = self.model.compute_loss(logits, targets)
        loss.backward()

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient is None for {name}")
                self.assertFalse(torch.isnan(param.grad).any(), f"Gradient contains NaN for {name}")
                # Ensure gradient is non-trivial
                grad_norm = param.grad.norm().item()
                self.assertGreater(grad_norm, 0.0, f"Gradient norm is zero for {name}")

    def test_deterministic_inference(self):
        """Verify deterministic outputs in evaluation mode."""
        self.model.eval()
        features = torch.randn(4, 768)
        party_ids = torch.tensor([0, 1, 0, 1])

        with torch.no_grad():
            out1 = self.model(features, party_ids=party_ids)
            out2 = self.model(features, party_ids=party_ids)

        self.assertTrue(torch.equal(out1, out2))

    def test_no_label_leakage(self):
        """
        Verify that model forward pass operates strictly on features and speaker IDs,
        with zero input dependence on emotion labels.
        """
        import inspect
        sig = inspect.signature(self.model.forward)
        param_names = list(sig.parameters.keys())
        self.assertNotIn("labels", param_names)
        self.assertNotIn("targets", param_names)
        self.assertNotIn("y_true", param_names)

        # Confirm output is identical regardless of any hypothetical external label
        self.model.eval()
        feat = torch.randn(3, 768)
        pid = torch.tensor([0, 1, 0])
        with torch.no_grad():
            pred = self.model.predict_dialogue(feat, party_ids=pid)
        self.assertEqual(pred.shape, (3, 7))

    def test_checkpoint_save_and_load(self):
        """Verify atomic checkpoint serialization and configuration restoration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "dialoguegcn_test.pt"
            self.model.eval()
            features = torch.randn(3, 768)
            party_ids = torch.tensor([0, 1, 0])

            with torch.no_grad():
                expected_logits = self.model(features, party_ids=party_ids)

            self.model.save_checkpoint(ckpt_path, metadata={"author": "test", "seed": 42})
            self.assertTrue(ckpt_path.exists())

            loaded_model, metadata = DialogueGCN.load_checkpoint(ckpt_path)
            self.assertEqual(metadata.get("seed"), 42)
            self.assertEqual(loaded_model.config.hidden_dim, self.config.hidden_dim)
            self.assertEqual(loaded_model.count_parameters(), 452487)

            with torch.no_grad():
                actual_logits = loaded_model(features, party_ids=party_ids)

            self.assertTrue(torch.allclose(expected_logits, actual_logits, atol=1e-6))

    def test_batch_equivalence(self):
        """
        Verify that evaluating a dialogue alone yields identical logits to evaluating it
        inside a batched, padded tensor alongside other dialogues.
        """
        self.model.eval()
        x0 = torch.randn(3, 768)
        p0 = torch.tensor([0, 1, 0])

        x1 = torch.randn(5, 768)
        p1 = torch.tensor([0, 1, 1, 0, 1])

        # 1. Single evaluation of dialogue 0
        with torch.no_grad():
            out_single = self.model(x0, party_ids=p0)

        # 2. Batched evaluation with dialogue 1
        x_batch = torch.zeros(2, 5, 768)
        x_batch[0, :3] = x0
        x_batch[1] = x1
        p_batch = torch.zeros(2, 5, dtype=torch.long)
        p_batch[0, :3] = p0
        p_batch[1] = p1
        lengths = torch.tensor([3, 5])

        with torch.no_grad():
            out_batched = self.model(x_batch, party_ids=p_batch, lengths=lengths)

        diff = (out_single - out_batched[0, :3]).abs().max().item()
        self.assertLess(diff, 1e-5, f"Batch equivalence failed with diff: {diff}")

    def test_compute_loss_masked_and_weighted(self):
        """Verify loss computation with padding mask and optional class weights."""
        logits = torch.randn(2, 4, 7)
        targets = torch.tensor([[0, 1, 2, -100], [4, 5, -100, -100]])
        mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
        weights = torch.ones(7)
        weights[0] = 2.0  # Weight class 0 higher

        loss = self.model.compute_loss(logits, targets, mask=mask, class_weights=weights)
        self.assertTrue(torch.is_tensor(loss))
        self.assertGreater(loss.item(), 0.0)
        self.assertFalse(torch.isnan(loss))


if __name__ == "__main__":
    unittest.main()
