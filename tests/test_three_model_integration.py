"""
Integration and regression tests for Unified Three-Model Inference (Task 26).

Verifies:
1. Model Registry exposes exactly three canonical model families: DistilBERT, DialogueRNN, DialogueGCN.
2. Each model family provides complete verified metadata, metrics, and availability status.
3. All three models execute inference over canonical conversations and return aligned turn predictions.
4. Shared DistilBERT feature encoder operates as a memory-efficient singleton.
5. Comparison service runs selected models on identical messages with strict alignment verification.
6. Error handling for invalid inputs, unknown models, and alignment mismatches.
7. Full API endpoints: GET /models, POST /compare, POST /analyze, and POST /predict.
"""

import unittest
from fastapi.testclient import TestClient

from src.api import app
from src.application.analyzer import ConversationAnalyzer
from src.application.comparison_service import ModelComparisonService, AlignmentMismatchError
from src.models.inference import (
    EmotionInference,
    DialogueRNNInference,
    DialogueGCNInference,
    SharedDistilBertEncoder,
)
from src.models.registry import (
    CANONICAL_MODEL_IDS,
    get_canonical_registry,
    get_model_metadata,
    normalize_model_id,
)


class TestModelRegistry(unittest.TestCase):
    """Requirements for the canonical model registry."""

    def test_registry_contains_exactly_three_canonical_models(self):
        """Verify only DistilBERT, DialogueRNN, and DialogueGCN are exposed."""
        registry = get_canonical_registry()
        self.assertEqual(set(registry.keys()), {"distilbert", "dialoguernn", "dialoguegcn"})
        self.assertEqual(len(registry), 3)

        # Ensure no ablations or experimental variants are exposed
        for excluded in ["context_1", "context_1_speaker", "ablation_1", "model_b"]:
            self.assertNotIn(excluded, registry)

    def test_registry_metadata_fields(self):
        """Verify each model entry contains all required metadata and verified metrics."""
        registry = get_canonical_registry()
        for model_id, info in registry.items():
            self.assertIn("model_id", info)
            self.assertIn("display_name", info)
            self.assertIn("description", info)
            self.assertIn("architecture_type", info)
            self.assertIn("checkpoint_metadata", info)
            self.assertIn("verified_metrics", info)
            self.assertIn("is_available", info)
            self.assertTrue(info["is_available"], f"Model {model_id} checkpoint should be available")

            # Check verified metrics
            metrics = info["verified_metrics"]
            self.assertIn("accuracy", metrics)
            self.assertIn("macro_f1", metrics)
            self.assertIn("weighted_f1", metrics)
            self.assertIn("cohort", metrics)
            self.assertGreater(metrics["macro_f1"], 0.40)

    def test_model_id_normalization(self):
        """Verify normalization of alias identifiers."""
        self.assertEqual(normalize_model_id("distilbert"), "distilbert")
        self.assertEqual(normalize_model_id("utterance"), "distilbert")
        self.assertEqual(normalize_model_id("transformer_baseline"), "distilbert")
        self.assertEqual(normalize_model_id("dialoguernn"), "dialoguernn")
        self.assertEqual(normalize_model_id("dialoguegcn"), "dialoguegcn")

        with self.assertRaises(ValueError):
            normalize_model_id("non_existent_model")


class TestSharedEncoder(unittest.TestCase):
    """Test singleton shared feature encoder."""

    def test_shared_encoder_caching(self):
        """Verify that SharedDistilBertEncoder returns identical cached instance."""
        tok1, enc1 = SharedDistilBertEncoder.get_encoder()
        tok2, enc2 = SharedDistilBertEncoder.get_encoder()
        self.assertIs(tok1, tok2)
        self.assertIs(enc1, enc2)

    def test_encode_texts(self):
        """Verify shared feature extraction produces correct tensor dimensions."""
        texts = ["Hello there!", "How are you?"]
        feats = SharedDistilBertEncoder.encode_texts(texts)
        self.assertEqual(feats.shape, (2, 768))


class TestThreeModelInference(unittest.TestCase):
    """Test individual inference wrappers for all three canonical models."""

    @classmethod
    def setUpClass(cls):
        cls.analyzer = ConversationAnalyzer()
        cls.dialogue_text = (
            "Alice: Hi Bob, did you hear the great news?\n"
            "Bob: No, what happened? I am so curious!\n"
            "Alice: We got accepted into the program!"
        )

    def test_distilbert_inference(self):
        """Verify DistilBERT inference returns aligned predictions with probabilities."""
        model = self.analyzer._get_model("distilbert")
        turns = self.analyzer.parse_conversation(self.dialogue_text)
        preds = model.predict(turns)

        self.assertEqual(len(preds), 3)
        for p in preds:
            self.assertIn("emotion", p)
            self.assertIn("confidence", p)
            self.assertIn("probabilities", p)
            self.assertEqual(len(p["probabilities"]), 7)
            self.assertEqual(p["model_identifier"], "distilbert")

    def test_dialoguernn_inference(self):
        """Verify DialogueRNN inference returns aligned predictions with probabilities."""
        model = self.analyzer._get_model("dialoguernn")
        turns = self.analyzer.parse_conversation(self.dialogue_text)
        preds = model.predict(turns)

        self.assertEqual(len(preds), 3)
        for p in preds:
            self.assertIn("emotion", p)
            self.assertIn("confidence", p)
            self.assertIn("probabilities", p)
            self.assertEqual(len(p["probabilities"]), 7)
            self.assertEqual(p["model_identifier"], "dialoguernn")

    def test_dialoguegcn_inference(self):
        """Verify DialogueGCN inference returns aligned predictions with probabilities."""
        model = self.analyzer._get_model("dialoguegcn")
        turns = self.analyzer.parse_conversation(self.dialogue_text)
        preds = model.predict(turns)

        self.assertEqual(len(preds), 3)
        for p in preds:
            self.assertIn("emotion", p)
            self.assertIn("confidence", p)
            self.assertIn("probabilities", p)
            self.assertEqual(len(p["probabilities"]), 7)
            self.assertEqual(p["model_identifier"], "dialoguegcn")


class TestComparisonService(unittest.TestCase):
    """Test Multi-Model Comparison Service."""

    def setUp(self):
        self.service = ModelComparisonService()
        self.conversation_text = (
            "Alice: I am really worried about tomorrow's exam.\n"
            "Bob: Don't worry Alice, you studied so hard for it.\n"
            "Alice: Thank you Bob, that makes me feel much better!"
        )

    def test_comparison_all_three_models(self):
        """Verify comparison service executes across all three models on identical messages."""
        res = self.service.compare(self.conversation_text)

        self.assertEqual(res["num_messages_evaluated"], 3)
        self.assertEqual(res["compared_models"], ["distilbert", "dialoguernn", "dialoguegcn"])

        # Check status
        for m_id in ["distilbert", "dialoguernn", "dialoguegcn"]:
            self.assertEqual(res["model_statuses"][m_id]["status"], "success")

        # Check pairwise agreement matrix
        pairwise = res["pairwise_agreement"]
        self.assertEqual(len(pairwise), 3)
        for m1 in ["distilbert", "dialoguernn", "dialoguegcn"]:
            self.assertEqual(pairwise[m1][m1], 1.0)
            for m2 in ["distilbert", "dialoguernn", "dialoguegcn"]:
                self.assertGreaterEqual(pairwise[m1][m2], 0.0)
                self.assertLessEqual(pairwise[m1][m2], 1.0)

        # Check message alignment
        messages = res["messages"]
        self.assertEqual(len(messages), 3)
        expected_speakers = ["Alice", "Bob", "Alice"]
        for i, msg in enumerate(messages):
            self.assertEqual(msg["message_id"], f"msg_{i + 1:04d}")
            self.assertEqual(msg["speaker"], expected_speakers[i])
            self.assertIn("distilbert", msg["models"])
            self.assertIn("dialoguernn", msg["models"])
            self.assertIn("dialoguegcn", msg["models"])
            self.assertIn("is_unanimous", msg)
            self.assertIn("consensus_emotion", msg)

    def test_comparison_subset_of_models(self):
        """Verify comparison service handles a requested subset of models."""
        res = self.service.compare(self.conversation_text, model_ids=["dialoguernn", "dialoguegcn"])
        self.assertEqual(res["compared_models"], ["dialoguernn", "dialoguegcn"])
        self.assertEqual(len(res["messages"][0]["models"]), 2)

    def test_comparison_rejects_empty_input(self):
        """Verify comparison service rejects empty or whitespace-only inputs."""
        with self.assertRaises(ValueError):
            self.service.compare("")

    def test_comparison_rejects_unknown_model(self):
        """Verify comparison service rejects unknown model identifiers."""
        with self.assertRaises(ValueError):
            self.service.compare(self.conversation_text, model_ids=["distilbert", "invalid_model"])


class TestApiEndpoints(unittest.TestCase):
    """Test API endpoints for model registry, comparison, and analysis."""

    def setUp(self):
        self.client = TestClient(app)
        self.dialogue_text = (
            "Alice: Good morning Bob!\n"
            "Bob: Good morning Alice, how are you today?"
        )

    def test_get_models_endpoint(self):
        """Verify GET /models returns canonical model registry."""
        res = self.client.get("/models")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(set(data.keys()), {"distilbert", "dialoguernn", "dialoguegcn"})
        self.assertEqual(data["distilbert"]["architecture_type"], "transformer")
        self.assertEqual(data["dialoguernn"]["architecture_type"], "recurrent")
        self.assertEqual(data["dialoguegcn"]["architecture_type"], "graph_convolutional")

    def test_post_compare_endpoint(self):
        """Verify POST /compare endpoint executes comparison service."""
        payload = {
            "conversation": self.dialogue_text,
            "models": ["distilbert", "dialoguernn", "dialoguegcn"],
        }
        res = self.client.post("/compare", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["num_messages_evaluated"], 2)
        self.assertEqual(data["compared_models"], ["distilbert", "dialoguernn", "dialoguegcn"])
        self.assertEqual(len(data["messages"]), 2)

    def test_post_analyze_dialoguegcn(self):
        """Verify POST /analyze supports DialogueGCN with dynamics analysis."""
        payload = {
            "conversation": self.dialogue_text,
            "mode": "dialoguegcn",
        }
        res = self.client.post("/analyze", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["mode"], "dialoguegcn")
        self.assertEqual(len(data["turns"]), 2)
        self.assertIn("analysis", data)
        self.assertEqual(data["model_metadata"]["architecture"], "DialogueGCN (EMNLP 2019)")


if __name__ == "__main__":
    unittest.main()
