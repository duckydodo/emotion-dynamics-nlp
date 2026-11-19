"""
Integration tests for the NLP Application, API, and DialogueRNN inference mode.
"""

import unittest
from fastapi.testclient import TestClient

from src.api import app
from src.application.analyzer import ConversationAnalyzer, MODEL_METADATA


class TestApplicationIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.sample_conversation = (
            "Chandler: Hey, what are you doing?\n"
            "Joey: I'm practicing my audition! It's so exciting!\n"
            "Monica: Joey, you're being way too loud, stop it!\n"
            "Rachel: Oh, leave him alone, he's just happy."
        )

    def test_health_endpoint(self):
        """Verify /health reports all available inference modes."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("utterance", data["available_modes"])
        self.assertIn("context_1", data["available_modes"])
        self.assertIn("context_1_speaker", data["available_modes"])
        self.assertIn("dialoguernn", data["available_modes"])

    def test_existing_modes_regression(self):
        """Verify all three existing modes remain functional with their response contracts."""
        for mode in ["utterance", "context_1", "context_1_speaker"]:
            response = self.client.post(
                "/analyze",
                json={"conversation": self.sample_conversation, "mode": mode},
            )
            self.assertEqual(response.status_code, 200, f"Mode {mode} failed with {response.text}")
            data = response.json()
            self.assertEqual(data["mode"], mode)
            self.assertEqual(len(data["turns"]), 4)
            for turn in data["turns"]:
                self.assertIn("turn_id", turn)
                self.assertIn("speaker", turn)
                self.assertIn("text", turn)
                self.assertIn("emotion", turn)
                self.assertIn("confidence", turn)
                self.assertIn("probabilities", turn)
                self.assertEqual(len(turn["probabilities"]), 7)

            self.assertIn("analysis", data)
            self.assertIn("emotion_distribution", data["analysis"])
            self.assertIn("shifts", data["analysis"])
            self.assertIn("persistence", data["analysis"])

    def test_dialoguernn_analyze_endpoint(self):
        """Verify DialogueRNN mode executes end-to-end on /analyze."""
        response = self.client.post(
            "/analyze",
            json={"conversation": self.sample_conversation, "mode": "dialoguernn"},
        )
        self.assertEqual(response.status_code, 200, f"DialogueRNN failed with {response.text}")
        data = response.json()
        self.assertEqual(data["mode"], "dialoguernn")
        self.assertEqual(len(data["turns"]), 4)

        # Check metadata
        self.assertIn("model_metadata", data)
        self.assertEqual(data["model_metadata"]["architecture"], "DialogueRNN (AAAI 2019)")
        self.assertTrue(data["model_metadata"]["causal"])

        # Check turn outputs
        for turn in data["turns"]:
            self.assertIn("emotion", turn)
            self.assertIn("confidence", turn)
            self.assertGreaterEqual(turn["confidence"], 0.0)
            self.assertLessEqual(turn["confidence"], 1.0)
            self.assertEqual(len(turn["probabilities"]), 7)
            # Sum of probabilities must equal ~1.0
            prob_sum = sum(turn["probabilities"].values())
            self.assertAlmostEqual(prob_sum, 1.0, places=4)

        # Check trajectory analysis
        self.assertIn("analysis", data)
        self.assertEqual(data["analysis"]["total_transitions"], 3)
        self.assertIn("shift_rate", data["analysis"])

    def test_dialoguernn_predict_endpoint(self):
        """Verify DialogueRNN mode on /predict lightweight endpoint."""
        response = self.client.post(
            "/predict",
            json={"conversation": self.sample_conversation, "mode": "dialoguernn"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["mode"], "dialoguernn")
        self.assertIn("model_metadata", data)
        self.assertEqual(len(data["turns"]), 4)

    def test_multispeaker_conversation(self):
        """Verify DialogueRNN tracks dynamic party states for multiple speakers."""
        multi_spk = (
            "Alice: Hello everyone!\n"
            "Bob: Hey Alice, good to see you.\n"
            "Charlie: Did you hear about the news?\n"
            "David: No, what happened?\n"
            "Alice: We won the prize!"
        )
        response = self.client.post(
            "/analyze",
            json={"conversation": multi_spk, "mode": "dialoguernn"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["turns"]), 5)

    def test_state_reset_between_requests(self):
        """Verify recurrent states strictly reset between separate conversation requests."""
        conv_a = "Ross: Hey Rachel.\nRachel: Hi Ross, what is it?"
        conv_b = "Joey: I'm freaking out!\nChandler: Calm down Joey!"

        # Call A
        res_a1 = self.client.post("/predict", json={"conversation": conv_a, "mode": "dialoguernn"}).json()
        # Interleave Call B
        _ = self.client.post("/predict", json={"conversation": conv_b, "mode": "dialoguernn"}).json()
        # Call A again
        res_a2 = self.client.post("/predict", json={"conversation": conv_a, "mode": "dialoguernn"}).json()

        for t1, t2 in zip(res_a1["turns"], res_a2["turns"]):
            self.assertEqual(t1["emotion"], t2["emotion"])
            self.assertAlmostEqual(t1["confidence"], t2["confidence"], places=6)
            for k in t1["probabilities"]:
                self.assertAlmostEqual(t1["probabilities"][k], t2["probabilities"][k], places=6)

    def test_deterministic_behavior(self):
        """Verify repeated calls on the exact same dialogue return deterministic values."""
        res1 = self.client.post("/predict", json={"conversation": self.sample_conversation, "mode": "dialoguernn"}).json()
        res2 = self.client.post("/predict", json={"conversation": self.sample_conversation, "mode": "dialoguernn"}).json()

        for t1, t2 in zip(res1["turns"], res2["turns"]):
            self.assertEqual(t1["emotion"], t2["emotion"])
            self.assertAlmostEqual(t1["confidence"], t2["confidence"], places=6)

    def test_malformed_and_unknown_speakers(self):
        """Verify robust handling of unusual formatting, blank lines, or single turns."""
        conv = (
            "\n"
            "UnknownSpeaker_99:    A very strange sentence here.\n"
            "\n"
            "AnotherGuest: Wow, really?\n"
        )
        response = self.client.post("/analyze", json={"conversation": conv, "mode": "dialoguernn"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["turns"]), 2)

    def test_empty_conversation_error(self):
        """Verify empty conversation raises 400."""
        response = self.client.post("/analyze", json={"conversation": "   ", "mode": "dialoguernn"})
        self.assertEqual(response.status_code, 400)

    def test_frontend_home(self):
        """Verify GET / returns static index.html containing DialogueRNN option and ethical notice."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.text
        self.assertIn("value=\"dialoguernn\"", html)
        self.assertIn("DialogueRNN", html)
        self.assertIn("Research note", html)


if __name__ == "__main__":
    unittest.main()
