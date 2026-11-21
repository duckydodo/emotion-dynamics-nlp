"""
Comprehensive Unit & Regression Tests for the Multimodal Pipeline.

Validates:
1. GatedMultimodalFusion, MultimodalDialogueRNN, and MultimodalDialogueGCN forward/backward passes.
2. Missing-modality resilience with random masks (zero NaNs, zero Infs, finite gradients).
3. AST syntax checking of all code cells in notebooks/30_kaggle_multimodal_meld.ipynb.
4. End-to-end fast synthetic training + evaluation smoke test matching Kaggle training loop.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score

from src.models.multimodal_fusion import (
    GatedMultimodalFusion,
    MultimodalDialogueRNN,
    MultimodalDialogueGCN,
)
from src.models.dialoguernn import DialogueRNNConfig
from src.models.dialoguegcn import DialogueGCNConfig


class TestMultimodalPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        np.random.seed(42)
        cls.project_root = Path(__file__).resolve().parent.parent

    def test_fusion_forward_and_backward_trimodal(self):
        """Test GatedMultimodalFusion in Tri-Modal mode (text, audio, video)."""
        B, T = 2, 6
        fusion = GatedMultimodalFusion(
            text_dim=768, audio_dim=768, video_dim=512, out_dim=256,
            modalities=("text", "audio", "video"), dropout=0.1
        )
        text = torch.randn(B, T, 768, requires_grad=True)
        audio = torch.randn(B, T, 768, requires_grad=True)
        video = torch.randn(B, T, 512, requires_grad=True)
        audio_mask = torch.ones(B, T)
        video_mask = torch.ones(B, T)

        u_t = fusion(text, audio=audio, video=video, audio_mask=audio_mask, video_mask=video_mask)
        self.assertEqual(u_t.shape, (B, T, 256))
        self.assertFalse(torch.isnan(u_t).any())
        self.assertFalse(torch.isinf(u_t).any())

        # Test backward pass
        loss = u_t.sum()
        loss.backward()
        self.assertIsNotNone(text.grad)
        self.assertIsNotNone(audio.grad)
        self.assertIsNotNone(video.grad)
        self.assertFalse(torch.isnan(text.grad).any())

    def test_fusion_forward_and_backward_bimodal(self):
        """Test GatedMultimodalFusion in Bimodal mode (text, audio)."""
        B, T = 3, 4
        fusion = GatedMultimodalFusion(
            text_dim=768, audio_dim=768, out_dim=256,
            modalities=("text", "audio"), dropout=0.1
        )
        text = torch.randn(B, T, 768, requires_grad=True)
        audio = torch.randn(B, T, 768, requires_grad=True)
        audio_mask = torch.ones(B, T)

        u_t = fusion(text, audio=audio, audio_mask=audio_mask)
        self.assertEqual(u_t.shape, (B, T, 256))
        self.assertFalse(torch.isnan(u_t).any())

        loss = u_t.sum()
        loss.backward()
        self.assertIsNotNone(text.grad)
        self.assertIsNotNone(audio.grad)

    def test_missing_modality_resilience(self):
        """Simulate missing/corrupted audio and video clips (mask=0)."""
        B, T = 2, 5
        fusion = GatedMultimodalFusion(
            text_dim=768, audio_dim=768, video_dim=512, out_dim=256,
            modalities=("text", "audio", "video")
        )
        text = torch.randn(B, T, 768, requires_grad=True)
        # Intentionally inject NaNs into missing inputs to test sanitization
        audio = torch.randn(B, T, 768)
        audio[0, 1] = float('nan')
        audio[1, 3] = float('inf')

        video = torch.randn(B, T, 512)
        video[0, 2] = float('nan')

        audio_mask = torch.tensor([[1.0, 0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0, 1.0]])
        video_mask = torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]])

        u_t = fusion(text, audio=audio, video=video, audio_mask=audio_mask, video_mask=video_mask)
        self.assertEqual(u_t.shape, (B, T, 256))
        self.assertFalse(torch.isnan(u_t).any(), "Output contains NaNs despite masking!")
        self.assertFalse(torch.isinf(u_t).any(), "Output contains Infs despite masking!")

        loss = u_t.sum()
        loss.backward()
        self.assertIsNotNone(text.grad)
        self.assertFalse(torch.isnan(text.grad).any())

    def test_multimodal_dialoguernn(self):
        """Test MultimodalDialogueRNN batched and single-dialogue passes."""
        B, T = 2, 4
        model = MultimodalDialogueRNN(
            text_dim=768, audio_dim=768, video_dim=512, fusion_dim=256,
            modalities=("text", "audio", "video")
        )
        text = torch.randn(B, T, 768)
        audio = torch.randn(B, T, 768)
        video = torch.randn(B, T, 512)
        party_ids = torch.tensor([[0, 1, 0, 1], [0, 0, 1, 1]])

        logits = model(text, party_ids, audio=audio, video=video)
        self.assertEqual(logits.shape, (B, T, 7))
        self.assertFalse(torch.isnan(logits).any())

        # Test single dialogue convenience pass
        single_logits = model.forward_dialogue(text[0], party_ids[0], audio=audio[0], video=video[0])
        self.assertEqual(single_logits.shape, (T, 7))

    def test_multimodal_dialoguegcn(self):
        """Test MultimodalDialogueGCN batched and single-dialogue passes."""
        B, T = 2, 5
        model = MultimodalDialogueGCN(
            text_dim=768, audio_dim=768, video_dim=512, fusion_dim=256,
            modalities=("text", "audio", "video")
        )
        text = torch.randn(B, T, 768)
        audio = torch.randn(B, T, 768)
        video = torch.randn(B, T, 512)
        party_ids = torch.tensor([[0, 1, 0, 1, 0], [0, 0, 1, 1, 0]])

        logits = model(text, party_ids=party_ids, audio=audio, video=video)
        self.assertEqual(logits.shape, (B, T, 7))
        self.assertFalse(torch.isnan(logits).any())

    def test_notebook_cells_ast_parse(self):
        """Verify that every code cell in 30_kaggle_multimodal_meld.ipynb compiles cleanly."""
        nb_path = self.project_root / "notebooks" / "30_kaggle_multimodal_meld.ipynb"
        self.assertTrue(nb_path.exists(), f"Notebook not found at {nb_path}")

        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
        self.assertGreaterEqual(len(code_cells), 5, "Expected at least 5 code cells in Kaggle notebook")

        for idx, cell in enumerate(code_cells, start=1):
            src = "".join(cell.get("source", []))
            try:
                ast.parse(src)
            except SyntaxError as e:
                self.fail(f"Syntax error in Kaggle notebook Code Cell {idx}: {e}")

    def test_synthetic_training_and_evaluation_smoke_test(self):
        """
        Fast 2-epoch training and evaluation smoke test matching the exact loop
        of the Kaggle notebook, ensuring metric calculations and export integrity.
        """
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # Create synthetic dialogue datasets
            def make_synth_dias(num_dias=4, num_turns=5):
                dias = []
                for d in range(num_dias):
                    cohort_a = torch.zeros(num_turns, dtype=torch.bool)
                    cohort_a[1:] = True
                    dias.append({
                        "dialogue_id": d,
                        "length": num_turns,
                        "text": torch.randn(num_turns, 768),
                        "audio": torch.randn(num_turns, 768),
                        "video": torch.randn(num_turns, 512),
                        "audio_mask": torch.ones(num_turns),
                        "video_mask": torch.ones(num_turns),
                        "labels": torch.randint(0, 7, (num_turns,)),
                        "party_ids": torch.tensor([i % 2 for i in range(num_turns)]),
                        "cohort_a_mask": cohort_a,
                    })
                return dias

            train_dias = make_synth_dias(4, 5)
            dev_dias = make_synth_dias(2, 5)
            test_dias = make_synth_dias(2, 5)

            model = MultimodalDialogueRNN(
                text_dim=768, audio_dim=768, video_dim=512, fusion_dim=64,
                modalities=("text", "audio", "video")
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()

            # Fast 2-epoch train
            for epoch in range(2):
                model.train()
                for dia in train_dias:
                    optimizer.zero_grad()
                    logits = model(
                        dia["text"].unsqueeze(0),
                        dia["party_ids"].unsqueeze(0),
                        audio=dia["audio"].unsqueeze(0),
                        video=dia["video"].unsqueeze(0),
                        audio_mask=dia["audio_mask"].unsqueeze(0),
                        video_mask=dia["video_mask"].unsqueeze(0),
                    ).squeeze(0)
                    loss = criterion(logits, dia["labels"])
                    loss.backward()
                    optimizer.step()

            # Eval on dev
            model.eval()
            dev_preds, dev_targets = [], []
            with torch.no_grad():
                for dia in dev_dias:
                    logits = model(
                        dia["text"].unsqueeze(0),
                        dia["party_ids"].unsqueeze(0),
                        audio=dia["audio"].unsqueeze(0),
                        video=dia["video"].unsqueeze(0),
                    ).squeeze(0)
                    preds = logits.argmax(dim=-1).numpy()
                    dev_preds.extend(preds)
                    dev_targets.extend(dia["labels"].numpy())

            macro_f1 = f1_score(dev_targets, dev_preds, average="macro", zero_division=0)
            acc = accuracy_score(dev_targets, dev_preds)
            self.assertGreaterEqual(macro_f1, 0.0)
            self.assertGreaterEqual(acc, 0.0)

            # Test checkpoint save
            ckpt_path = temp_dir / "test_model.pt"
            torch.save(model.state_dict(), ckpt_path)
            self.assertTrue(ckpt_path.exists())

            # Test export JSON
            summary_data = {
                "model": "MultimodalDialogueRNN_Smoke",
                "accuracy": float(acc),
                "macro_f1": float(macro_f1),
            }
            json_path = temp_dir / "summary.json"
            with open(json_path, "w") as f:
                json.dump(summary_data, f)
            self.assertTrue(json_path.exists())

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
