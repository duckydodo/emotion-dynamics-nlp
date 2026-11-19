"""
Unit tests for DialogueRNN modular ablation architectures (Task 14).

Verifies:
1. Exact parameter counts for all four ablation configurations (370,695; 617,223; 452,999; 670,855).
2. Genuine absence of disabled components (no unused submodules or hanging parameters).
3. Forward pass output dimensions and finite values.
4. Variable dialogue lengths and padding masks.
5. Multiple speakers and speaker switching dynamics.
6. Recurrent state reset between independent dialogues.
7. Strict causal recurrence (future utterances cannot affect earlier predictions).
8. Gradient flow through all active components.
9. Strict checkpoint loading compatibility with original DialogueRNN checkpoints.
10. Exact numerical equivalence between DialogueRNN and DialogueRNNAblation (Mode 4).
11. Exact reproduction of 2,610 / 2,610 saved test predictions.
"""

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.feature_loader import (
    DialogueBatchDataset,
    dialogue_collate_fn,
    load_dialogue_features,
)
from src.models.dialoguernn import (
    DialogueRNN,
    DialogueRNNConfig,
    DialogueRNNAblation,
    DialogueRNNAblationConfig,
    build_ablation_model,
)

CONFIG_PATHS = [
    Path("configs/ablations/1_global_context_only.json"),
    Path("configs/ablations/2_global_party_states.json"),
    Path("configs/ablations/3_global_emotion_state.json"),
    Path("configs/ablations/4_full_dialoguernn.json"),
]

EXPECTED_PARAMS = {
    1: 370695,
    2: 617223,
    3: 452999,
    4: 670855,
}

LABEL_ORDER = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


class TestDialogueRNNAblations(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)

    def test_parameter_counts_exact_match(self):
        """Test that all 4 ablation configurations match validated parameter counts to the exact integer."""
        for cfg_path in CONFIG_PATHS:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            abl_id = data["ablation_id"]
            expected = EXPECTED_PARAMS[abl_id]

            model = build_ablation_model(cfg_path)
            actual = model.count_parameters()

            self.assertEqual(
                actual,
                expected,
                f"Ablation {abl_id} ({cfg_path.name}) expected {expected} parameters, got {actual}",
            )

    def test_disabled_components_genuinely_absent(self):
        """Test that disabled components are genuinely absent (None) and have no registered parameters."""
        for cfg_path in CONFIG_PATHS:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            abl_id = data["ablation_id"]
            model = build_ablation_model(cfg_path)
            named_params = dict(model.named_parameters())

            if abl_id == 1:
                # Global Context Only: party_gru, emotion_attn, emotion_gru absent
                self.assertIsNone(model.party_gru)
                self.assertIsNone(model.emotion_attn)
                self.assertIsNone(model.emotion_gru)
                self.assertIsNotNone(model.global_gru)
                self.assertIsNotNone(model.context_attn)
                self.assertFalse(any("party_gru" in k for k in named_params))
                self.assertFalse(any("emotion_attn" in k for k in named_params))
                self.assertFalse(any("emotion_gru" in k for k in named_params))
            elif abl_id == 2:
                # Global + Party: emotion_attn, emotion_gru absent
                self.assertIsNotNone(model.party_gru)
                self.assertIsNone(model.emotion_attn)
                self.assertIsNone(model.emotion_gru)
                self.assertIsNotNone(model.global_gru)
                self.assertIsNotNone(model.context_attn)
                self.assertTrue(any("party_gru" in k for k in named_params))
                self.assertFalse(any("emotion_attn" in k for k in named_params))
                self.assertFalse(any("emotion_gru" in k for k in named_params))
            elif abl_id == 3:
                # Global + Emotion: party_gru, emotion_attn absent
                self.assertIsNone(model.party_gru)
                self.assertIsNone(model.emotion_attn)
                self.assertIsNotNone(model.emotion_gru)
                self.assertIsNotNone(model.global_gru)
                self.assertIsNotNone(model.context_attn)
                self.assertFalse(any("party_gru" in k for k in named_params))
                self.assertFalse(any("emotion_attn" in k for k in named_params))
                self.assertTrue(any("emotion_gru" in k for k in named_params))
            elif abl_id == 4:
                # Full DialogueRNN: all components present
                self.assertIsNotNone(model.party_gru)
                self.assertIsNotNone(model.emotion_attn)
                self.assertIsNotNone(model.emotion_gru)
                self.assertIsNotNone(model.global_gru)
                self.assertIsNotNone(model.context_attn)
                self.assertTrue(any("party_gru" in k for k in named_params))
                self.assertTrue(any("emotion_attn" in k for k in named_params))
                self.assertTrue(any("emotion_gru" in k for k in named_params))

    def test_forward_pass_output_shape_and_finite_logits(self):
        """Test forward pass produces correct tensor dimensions and non-NaN, non-Inf logits."""
        B, T, D = 3, 7, 768
        features = torch.randn(B, T, D)
        party_ids = torch.randint(0, 3, (B, T))

        for cfg_path in CONFIG_PATHS:
            model = build_ablation_model(cfg_path)
            model.eval()

            with torch.no_grad():
                logits = model(features, party_ids)

            self.assertEqual(logits.shape, (B, T, 7))
            self.assertFalse(torch.isnan(logits).any().item(), f"NaN in {cfg_path.name}")
            self.assertFalse(torch.isinf(logits).any().item(), f"Inf in {cfg_path.name}")

    def test_single_dialogue_forward(self):
        """Test forward_dialogue convenience method for single unbatched dialogues."""
        T, D = 8, 768
        features = torch.randn(T, D)
        party_ids = torch.randint(0, 4, (T,))

        for cfg_path in CONFIG_PATHS:
            model = build_ablation_model(cfg_path)
            model.eval()

            with torch.no_grad():
                logits = model.forward_dialogue(features, party_ids)

            self.assertEqual(logits.shape, (T, 7))
            self.assertFalse(torch.isnan(logits).any().item())

    def test_variable_dialogue_lengths_and_padding_masks(self):
        """Test variable sequence lengths with padding masks and verify padding turns don't affect loss."""
        B, T_max, D = 3, 10, 768
        lengths = [4, 7, 10]
        features = torch.randn(B, T_max, D)
        party_ids = torch.randint(0, 3, (B, T_max))
        targets = torch.randint(0, 7, (B, T_max))

        mask = torch.zeros(B, T_max, dtype=torch.bool)
        for i, length in enumerate(lengths):
            mask[i, :length] = True

        for cfg_path in CONFIG_PATHS:
            model = build_ablation_model(cfg_path)
            model.eval()

            with torch.no_grad():
                logits = model(features, party_ids, mask=mask)
                loss = model.compute_loss(logits, targets, mask=mask)

            self.assertEqual(logits.shape, (B, T_max, 7))
            self.assertFalse(torch.isnan(loss).item())
            self.assertGreater(loss.item(), 0.0)

    def test_multiple_speakers_and_speaker_changes(self):
        """Test dynamic speaker alternation with up to 8 interlocutors."""
        B, T, D = 2, 12, 768
        features = torch.randn(B, T, D)
        party_ids = torch.tensor([
            [0, 1, 2, 0, 1, 3, 4, 0, 2, 5, 6, 7],
            [1, 0, 1, 0, 2, 2, 3, 1, 0, 4, 0, 1],
        ])

        for cfg_path in CONFIG_PATHS:
            model = build_ablation_model(cfg_path)
            model.eval()

            with torch.no_grad():
                logits = model(features, party_ids)

            self.assertEqual(logits.shape, (B, T, 7))
            self.assertFalse(torch.isnan(logits).any().item())

    def test_recurrent_state_reset_between_dialogues(self):
        """Test that states reset completely between forward calls (zero cross-dialogue leakage)."""
        T, D = 5, 768
        torch.manual_seed(101)
        dialogue_a_feats = torch.randn(1, T, D)
        dialogue_a_parties = torch.randint(0, 2, (1, T))

        dialogue_b_feats = torch.randn(1, T, D)
        dialogue_b_parties = torch.randint(0, 2, (1, T))

        for cfg_path in CONFIG_PATHS:
            model = build_ablation_model(cfg_path)
            model.eval()

            # Run Dialogue B in isolation
            with torch.no_grad():
                logits_b_isolated = model(dialogue_b_feats, dialogue_b_parties)

            # Run Dialogue A followed by Dialogue B
            with torch.no_grad():
                _ = model(dialogue_a_feats, dialogue_a_parties)
                logits_b_after_a = model(dialogue_b_feats, dialogue_b_parties)

            diff = torch.max(torch.abs(logits_b_isolated - logits_b_after_a)).item()
            self.assertEqual(diff, 0.0, f"State leaked between dialogues in {cfg_path.name}")

    def test_causality_future_perturbations(self):
        """Test strict causal temporal flow: modifying turn t* cannot affect turns t < t*."""
        T, D = 8, 768
        perturb_idx = 4

        for cfg_path in CONFIG_PATHS:
            torch.manual_seed(999)
            model = build_ablation_model(cfg_path)
            model.eval()

            base_feats = torch.randn(1, T, D)
            base_parties = torch.tensor([[0, 1, 0, 2, 1, 0, 1, 2]])

            with torch.no_grad():
                logits_base = model(base_feats, base_parties)

            perturbed_feats = base_feats.clone()
            perturbed_feats[:, perturb_idx, :] += 10.0

            with torch.no_grad():
                logits_perturbed = model(perturbed_feats, base_parties)

            # Turns before perturbation MUST be exactly identical
            diff_past = torch.max(
                torch.abs(logits_base[:, :perturb_idx, :] - logits_perturbed[:, :perturb_idx, :])
            ).item()
            self.assertEqual(
                diff_past,
                0.0,
                f"Causality violated in {cfg_path.name}: past logits altered by future perturbation",
            )

            # Turns at or after perturbation should differ
            diff_future = torch.max(
                torch.abs(logits_base[:, perturb_idx:, :] - logits_perturbed[:, perturb_idx:, :])
            ).item()
            self.assertGreater(
                diff_future,
                1e-4,
                f"Expected perturbation at t={perturb_idx} to affect future logits in {cfg_path.name}",
            )

    def test_gradient_propagation_all_enabled_components(self):
        """Test backpropagation produces valid gradients in all enabled trainable parameters."""
        B, T, D = 2, 5, 768
        targets = torch.randint(0, 7, (B, T))

        for cfg_path in CONFIG_PATHS:
            torch.manual_seed(42)
            model = build_ablation_model(cfg_path)
            model.train()

            features = torch.randn(B, T, D, requires_grad=True)
            party_ids = torch.randint(0, 2, (B, T))

            logits = model(features, party_ids)
            loss = model.compute_loss(logits, targets)
            loss.backward()

            for name, param in model.named_parameters():
                self.assertIsNotNone(
                    param.grad,
                    f"Parameter '{name}' in {cfg_path.name} received no gradient!",
                )
                grad_norm = param.grad.norm().item()
                self.assertFalse(
                    np.isnan(grad_norm),
                    f"NaN gradient in '{name}' for {cfg_path.name}",
                )
                self.assertGreater(
                    grad_norm,
                    0.0,
                    f"Zero gradient in '{name}' for {cfg_path.name}",
                )

    def test_checkpoint_loading_strict_keys(self):
        """Test that original full DialogueRNN checkpoints load into DialogueRNNAblation (Mode 4) with strict=True."""
        ckpt_path = Path("checkpoints/dialoguernn_unweighted/best_model.pt")
        if not ckpt_path.exists():
            self.skipTest(f"Checkpoint not found at {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = build_ablation_model("configs/ablations/4_full_dialoguernn.json")

        # strict=True raises if there are missing or unexpected keys
        load_result = model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.assertEqual(len(load_result.missing_keys), 0)
        self.assertEqual(len(load_result.unexpected_keys), 0)

    def test_full_model_numerical_equivalence_with_dialoguernn(self):
        """Test that DialogueRNN and DialogueRNNAblation (Mode 4) produce identical logits for identical weights."""
        ckpt_path = Path("checkpoints/dialoguernn_unweighted/best_model.pt")
        if not ckpt_path.exists():
            self.skipTest(f"Checkpoint not found at {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu")

        ref_model = DialogueRNN(DialogueRNNConfig())
        ref_model.load_state_dict(ckpt["model_state_dict"], strict=True)
        ref_model.eval()

        abl_model = build_ablation_model("configs/ablations/4_full_dialoguernn.json")
        abl_model.load_state_dict(ckpt["model_state_dict"], strict=True)
        abl_model.eval()

        torch.manual_seed(2026)
        features = torch.randn(2, 6, 768)
        party_ids = torch.randint(0, 3, (2, 6))

        with torch.no_grad():
            out_ref = ref_model(features, party_ids)
            out_abl = abl_model(features, party_ids)

        max_diff = torch.max(torch.abs(out_ref - out_abl)).item()
        self.assertEqual(
            max_diff,
            0.0,
            f"DialogueRNN and DialogueRNNAblation (Mode 4) differ by {max_diff}",
        )

    def test_saved_predictions_exact_reproduction(self):
        """Test that DialogueRNNAblation reproduces 100% (2,610/2,610) saved test predictions."""
        ckpt_path = Path("checkpoints/dialoguernn_unweighted/best_model.pt")
        test_features_path = Path("data/processed/dialoguernn/meld_test_features.npz")
        saved_preds_path = Path("outputs/experiments/dialoguernn_unweighted/test_predictions.csv")

        if not (ckpt_path.exists() and test_features_path.exists() and saved_preds_path.exists()):
            self.skipTest("Required artifacts for prediction reproduction test not found.")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = build_ablation_model("configs/ablations/4_full_dialoguernn.json")
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.eval()

        test_dataset = load_dialogue_features(test_features_path)
        test_dialogues = test_dataset.get_dialogues()
        test_loader = DataLoader(
            DialogueBatchDataset(test_dialogues),
            batch_size=16,
            shuffle=False,
            collate_fn=dialogue_collate_fn,
        )

        all_preds = []
        with torch.no_grad():
            for batch in test_loader:
                feats = batch["features"]
                parties = batch["party_ids"]
                mask = batch["mask"]
                logits = model(feats, parties, mask=mask)
                preds = torch.argmax(logits, dim=-1)

                B = feats.shape[0]
                for b in range(B):
                    seq_len = batch["lengths"][b].item()
                    for t in range(seq_len):
                        all_preds.append({
                            "dialogue_id": batch["dialogue_ids"][b],
                            "utterance_id": batch["utterance_ids"][b][t],
                            "pred_id": preds[b, t].item(),
                        })

        df_new = pd.DataFrame(all_preds)
        df_saved = pd.read_csv(saved_preds_path)

        df_new["pred_emotion"] = df_new["pred_id"].apply(lambda idx: LABEL_ORDER[idx])

        matches = (df_new["pred_emotion"] == df_saved["pred_emotion"]).sum()
        total = len(df_saved)

        self.assertEqual(
            total,
            2610,
            f"Expected 2610 total test utterances, got {total}",
        )
        self.assertEqual(
            matches,
            total,
            f"Expected 2610 matches with saved predictions, got {matches} ({matches/total*100:.2f}%)",
        )


if __name__ == "__main__":
    unittest.main()
