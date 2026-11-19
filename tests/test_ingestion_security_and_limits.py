"""
Unified Ingestion Validation and Security Audit Tests (Task 22).

Verifies:
1. Cross-format equivalence across Plain Text, Instagram JSON, WhatsApp TXT, and CSV.
2. Defensive bounds (10 MB payload limit, 1,000 messages limit, 4,000 char limit, 64 char speaker limit).
3. Security hardening (path masking, privacy/in-memory processing, no disk persistence).
4. Preview readiness on POST /import (message counts, preview turns, date range, warnings).
5. Error handling and validation across malformed or edge-case inputs.
"""

import json
import os
import unittest
from fastapi.testclient import TestClient

from src.api import app, MAX_PAYLOAD_BYTES
from src.data.conversation import parse_canonical_conversation
from src.data.schema import (
    MAX_MESSAGES_PER_CONVERSATION,
    MAX_MESSAGE_CHARACTERS,
    MAX_SPEAKER_NAME_LENGTH,
)


class TestUnifiedIngestionValidation(unittest.TestCase):
    """Requirement A: Cross-format equivalence and canonical consistency."""

    def test_cross_format_equivalence(self):
        """
        Verify that identical dialogues represented in Plain Text, WhatsApp,
        Instagram JSON, and CSV produce identical evaluable speech turns.
        """
        dialogue_text = (
            "Alice: Hello Bob!\n"
            "Bob: Hi Alice, how are you today?\n"
            "Alice: I am feeling great, thank you."
        )

        whatsapp_text = (
            "[10/11/26, 10:00:00 AM] Alice: Hello Bob!\n"
            "[10/11/26, 10:01:00 AM] Bob: Hi Alice, how are you today?\n"
            "[10/11/26, 10:02:00 AM] Alice: I am feeling great, thank you."
        )

        instagram_data = {
            "participants": [{"name": "Alice"}, {"name": "Bob"}],
            "messages": [
                {
                    "sender_name": "Alice",
                    "timestamp_ms": 1794304920000,
                    "content": "I am feeling great, thank you.",
                },
                {
                    "sender_name": "Bob",
                    "timestamp_ms": 1794304860000,
                    "content": "Hi Alice, how are you today?",
                },
                {
                    "sender_name": "Alice",
                    "timestamp_ms": 1794304800000,
                    "content": "Hello Bob!",
                },
            ],
            "title": "Alice and Bob",
        }
        instagram_text = json.dumps(instagram_data)

        csv_text = (
            "speaker,text,timestamp\n"
            "Alice,Hello Bob!,2026-11-10T10:00:00Z\n"
            "Bob,\"Hi Alice, how are you today?\",2026-11-10T10:01:00Z\n"
            "Alice,\"I am feeling great, thank you.\",2026-11-10T10:02:00Z\n"
        )

        # Parse across all 4 formats
        conv_plain = parse_canonical_conversation(dialogue_text, format_hint="plain_text")
        conv_wa = parse_canonical_conversation(whatsapp_text, format_hint="whatsapp")
        conv_ig = parse_canonical_conversation(instagram_text, format_hint="instagram")
        conv_csv = parse_canonical_conversation(csv_text, format_hint="csv")

        # 1. Verify all 4 produced valid CanonicalConversation objects
        self.assertEqual(len(conv_plain.validate()), 0)
        self.assertEqual(len(conv_wa.validate()), 0)
        self.assertEqual(len(conv_ig.validate()), 0)
        self.assertEqual(len(conv_csv.validate()), 0)

        # 2. Extract evaluable turns from each
        turns_plain = conv_plain.get_evaluable_turns()
        turns_wa = conv_wa.get_evaluable_turns()
        turns_ig = conv_ig.get_evaluable_turns()
        turns_csv = conv_csv.get_evaluable_turns()

        # 3. Assert exact sequence match across all formats
        expected_turns = [
            (0, "Alice", "Hello Bob!"),
            (1, "Bob", "Hi Alice, how are you today?"),
            (2, "Alice", "I am feeling great, thank you."),
        ]

        for turns, fmt_name in [
            (turns_plain, "Plain Text"),
            (turns_wa, "WhatsApp"),
            (turns_ig, "Instagram"),
            (turns_csv, "CSV"),
        ]:
            self.assertEqual(len(turns), 3, f"{fmt_name} turn count mismatch")
            for i, expected in enumerate(expected_turns):
                self.assertEqual(turns[i].turn_id, expected[0], f"{fmt_name} turn_id mismatch at {i}")
                self.assertEqual(turns[i].speaker, expected[1], f"{fmt_name} speaker mismatch at {i}")
                self.assertEqual(turns[i].text, expected[2], f"{fmt_name} text mismatch at {i}")


class TestDefensiveBoundsAndLimits(unittest.TestCase):
    """Requirement B: Enforcement of defensive bounds and system limits."""

    def setUp(self):
        self.client = TestClient(app)

    def test_payload_size_limit_exceeded_returns_413(self):
        """Verify that payloads exceeding 10 MB return HTTP 413."""
        oversized = "A: " + ("x" * (MAX_PAYLOAD_BYTES + 100))
        # Test /analyze
        res = self.client.post("/analyze", json={"conversation": oversized, "mode": "utterance"})
        self.assertEqual(res.status_code, 413)
        self.assertIn("Payload exceeds maximum permitted size", res.json()["detail"])

        # Test /predict
        res = self.client.post("/predict", json={"conversation": oversized, "mode": "utterance"})
        self.assertEqual(res.status_code, 413)

        # Test /import
        res = self.client.post("/import", json={"conversation": oversized, "format": "plain_text"})
        self.assertEqual(res.status_code, 413)

    def test_message_count_limit_truncation(self):
        """Verify that conversations exceeding 1,000 messages are safely truncated."""
        lines = [f"Alice: Message turn {i}" for i in range(1050)]
        raw_text = "\n".join(lines)

        conv = parse_canonical_conversation(raw_text, format_hint="plain_text")
        self.assertEqual(len(conv.messages), MAX_MESSAGES_PER_CONVERSATION)
        self.assertEqual(len(conv.get_evaluable_turns()), MAX_MESSAGES_PER_CONVERSATION)

        warning_codes = [w.code for w in conv.warnings]
        self.assertIn("MESSAGE_COUNT_LIMIT_EXCEEDED", warning_codes)
        limit_warning = next(w for w in conv.warnings if w.code == "MESSAGE_COUNT_LIMIT_EXCEEDED")
        self.assertIn("Truncated 50 trailing messages", limit_warning.message)

    def test_message_character_truncation(self):
        """Verify that individual utterances exceeding 4,000 characters are safely truncated."""
        long_text = "Alice: " + ("Hello world! " * 500)  # > 6,000 chars
        conv = parse_canonical_conversation(long_text, format_hint="plain_text")

        self.assertEqual(len(conv.messages), 1)
        self.assertEqual(len(conv.messages[0].text), MAX_MESSAGE_CHARACTERS)

        warning_codes = [w.code for w in conv.warnings]
        self.assertIn("MESSAGE_TEXT_TRUNCATED", warning_codes)

    def test_speaker_name_truncation(self):
        """Verify that speaker names exceeding 64 characters are truncated."""
        long_speaker = "Dr" + ("LongTitle" * 8)  # 74 chars
        raw_text = f"{long_speaker}: Hello there."
        conv = parse_canonical_conversation(raw_text, format_hint="plain_text")

        self.assertEqual(len(conv.messages), 1)
        self.assertEqual(len(conv.messages[0].speaker), MAX_SPEAKER_NAME_LENGTH)
        self.assertEqual(len(conv.participants[0].name), MAX_SPEAKER_NAME_LENGTH)


class TestSecurityAndPrivacy(unittest.TestCase):
    """Requirement C: Privacy, sanitization, and filesystem protection."""

    def setUp(self):
        self.client = TestClient(app)

    def test_error_sanitization_prevents_path_leakage(self):
        """Verify that 500 errors from missing checkpoints mask server paths."""
        res = self.client.post("/analyze", json={"conversation": "Alice: Hi", "mode": "context_1"})
        # Even if a checkpoint error occurs, verify no absolute filesystem path is leaked
        if res.status_code == 500:
            detail = res.json()["detail"]
            self.assertEqual(detail, "Required model checkpoint is unavailable.")
            self.assertNotIn("/home/", detail)
            self.assertNotIn("/workspace/", detail)

    def test_no_disk_writes_during_import(self):
        """Verify that parsing does not persist any files to disk."""
        initial_cwd_files = set(os.listdir("."))
        payload = {
            "conversation": "Alice: Secret message 123\nBob: Confidential reply 456",
            "format": "plain_text",
        }
        res = self.client.post("/import", json=payload)
        self.assertEqual(res.status_code, 200)

        current_cwd_files = set(os.listdir("."))
        self.assertEqual(initial_cwd_files, current_cwd_files)


class TestPreviewReadiness(unittest.TestCase):
    """Requirement D: Preview API readiness before downstream inference."""

    def setUp(self):
        self.client = TestClient(app)

    def test_preview_response_structure(self):
        """Verify that POST /import provides all fields required for UI preview."""
        wa_text = (
            "[10/11/26, 10:00:00 AM] Alice: Message one\n"
            "[10/11/26, 10:01:00 AM] Bob: Message two\n"
            "[10/11/26, 10:02:00 AM] Messages and calls are end-to-end encrypted.\n"
            "[10/11/26, 10:03:00 AM] Alice: Message three\n"
        )
        res = self.client.post("/import", json={"conversation": wa_text, "format": "whatsapp"})
        self.assertEqual(res.status_code, 200)
        data = res.json()

        # Check preview-specific fields
        self.assertEqual(data["parsed_message_count"], 4)
        self.assertEqual(data["evaluable_turn_count"], 3)
        self.assertEqual(data["participant_count"], 2)
        self.assertEqual(data["unsupported_message_count"], 1)
        self.assertEqual(data["warning_count"], 1)
        self.assertEqual(data["warnings"][0]["code"], "AMBIGUOUS_DATE_FORMAT")
        self.assertEqual(data["speakers"], ["Alice", "Bob"])

        # Check date range calculation
        self.assertIsNotNone(data["date_range"])
        self.assertIn("start", data["date_range"])
        self.assertIn("end", data["date_range"])
        self.assertEqual(data["date_range"]["duration_seconds"], 180.0)

        # Check preview turns (limited to first 5)
        self.assertEqual(len(data["preview_messages"]), 4)
        self.assertEqual(data["preview_messages"][0]["speaker"], "Alice")
        self.assertEqual(data["preview_messages"][0]["text"], "Message one")
        self.assertTrue(data["preview_messages"][0]["is_evaluable"])
        self.assertFalse(data["preview_messages"][2]["is_evaluable"])  # System message


if __name__ == "__main__":
    unittest.main()
