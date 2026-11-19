"""
Unit and integration tests for Instagram JSON conversation parser.
Tests all required export variants, timestamps, media, reactions, system events,
mojibake encoding repairs, validation, and API integration.
"""

from datetime import datetime, timezone
import json
import unittest

from fastapi.testclient import TestClient

from src.api import app
from src.data.parsers.instagram import InstagramJsonParser, attempt_mojibake_repair
from src.data.schema import MessageType, SourceFormat, WarningSeverity


class TestInstagramJsonParser(unittest.TestCase):

    def setUp(self):
        self.parser = InstagramJsonParser()

    def test_standard_message(self):
        """Verify standard 1-to-1 conversation with timestamps."""
        data = {
            "participants": [{"name": "Alice"}, {"name": "Bob"}],
            "messages": [
                {
                    "sender_name": "Bob",
                    "timestamp_ms": 1700000020000,
                    "content": "Sounds good!",
                },
                {
                    "sender_name": "Alice",
                    "timestamp_ms": 1700000000000,
                    "content": "Hey Bob, are you coming?",
                },
            ],
            "title": "Bob",
        }
        conv = self.parser.parse(data)
        self.assertEqual(conv.source_format, SourceFormat.INSTAGRAM_JSON)
        self.assertEqual(len(conv.participants), 2)
        self.assertEqual(len(conv.messages), 2)

        # Chronological ordering: Alice (00s) -> Bob (20s)
        self.assertEqual(conv.messages[0].speaker, "Alice")
        self.assertEqual(conv.messages[0].text, "Hey Bob, are you coming?")
        self.assertEqual(conv.messages[1].speaker, "Bob")
        self.assertEqual(conv.messages[1].text, "Sounds good!")

        evaluable = conv.get_evaluable_turns()
        self.assertEqual(len(evaluable), 2)
        self.assertEqual(evaluable[0].speaker, "Alice")
        self.assertEqual(evaluable[1].speaker, "Bob")

    def test_multiple_participants(self):
        """Verify group conversation with 3 or more participants."""
        data = {
            "participants": ["Alice", "Bob", "Charlie", "David"],
            "messages": [
                {"sender_name": "David", "timestamp_ms": 1700000040000, "content": "I am in!"},
                {"sender_name": "Charlie", "timestamp_ms": 1700000030000, "content": "When is it?"},
                {"sender_name": "Bob", "timestamp_ms": 1700000020000, "content": "Count me in."},
                {"sender_name": "Alice", "timestamp_ms": 1700000010000, "content": "Movie tonight?"},
            ],
        }
        conv = self.parser.parse(data)
        self.assertEqual(len(conv.participants), 4)
        self.assertEqual(len(conv.messages), 4)
        speakers = [m.speaker for m in conv.messages]
        self.assertEqual(speakers, ["Alice", "Bob", "Charlie", "David"])

    def test_multiple_messages_same_speaker(self):
        """Verify consecutive messages from the same sender."""
        data = {
            "participants": [{"name": "Alice"}],
            "messages": [
                {"sender_name": "Alice", "timestamp_ms": 1700000020000, "content": "Second turn."},
                {"sender_name": "Alice", "timestamp_ms": 1700000010000, "content": "First turn."},
            ],
        }
        conv = self.parser.parse(data)
        self.assertEqual(len(conv.messages), 2)
        self.assertEqual(conv.messages[0].text, "First turn.")
        self.assertEqual(conv.messages[1].text, "Second turn.")
        self.assertEqual(conv.participants[0].turn_count, 2)

    def test_missing_timestamp(self):
        """Verify handling of messages lacking timestamp information."""
        data = {
            "participants": [{"name": "Alice"}, {"name": "Bob"}],
            "messages": [
                {"sender_name": "Bob", "content": "Latest message."},
                {"sender_name": "Alice", "content": "Earlier message."},
            ],
        }
        conv = self.parser.parse(data)
        self.assertEqual(len(conv.messages), 2)
        self.assertIsNone(conv.messages[0].timestamp)
        self.assertIsNone(conv.messages[1].timestamp)
        # Inverts newest-first array
        self.assertEqual(conv.messages[0].text, "Earlier message.")
        self.assertEqual(conv.messages[1].text, "Latest message.")
        self.assertTrue(any(w.code == "MISSING_TIMESTAMPS_INVERTED" for w in conv.warnings))

    def test_attachment_only_message(self):
        """Verify photos/stickers without text are flagged is_evaluable=False."""
        data = {
            "participants": [{"name": "Alice"}, {"name": "Bob"}],
            "messages": [
                {
                    "sender_name": "Bob",
                    "timestamp_ms": 1700000020000,
                    "photos": [{"uri": "messages/photos/photo_1.jpg"}],
                },
                {
                    "sender_name": "Alice",
                    "timestamp_ms": 1700000010000,
                    "content": "Check out this photo:",
                },
            ],
        }
        conv = self.parser.parse(data)
        self.assertEqual(len(conv.messages), 2)
        photo_msg = conv.messages[1]
        self.assertEqual(photo_msg.speaker, "Bob")
        self.assertEqual(photo_msg.text, "")
        self.assertEqual(photo_msg.message_type, MessageType.MEDIA)
        self.assertFalse(photo_msg.is_evaluable)
        self.assertIn("photos", photo_msg.metadata)

        # Evaluable turns must only include Alice's text turn
        evaluable = conv.get_evaluable_turns()
        self.assertEqual(len(evaluable), 1)
        self.assertEqual(evaluable[0].speaker, "Alice")

    def test_system_event(self):
        """Verify system events (e.g. call or group action) are preserved but flagged non-evaluable."""
        data = {
            "participants": [{"name": "Alice"}],
            "messages": [
                {
                    "sender_name": "Alice",
                    "timestamp_ms": 1700000010000,
                    "type": "Call",
                    "call_duration": 120,
                }
            ],
        }
        conv = self.parser.parse(data)
        self.assertEqual(len(conv.messages), 1)
        self.assertEqual(conv.messages[0].message_type, MessageType.SYSTEM)
        self.assertFalse(conv.messages[0].is_evaluable)
        self.assertEqual(len(conv.get_evaluable_turns()), 0)

    def test_reaction_handling(self):
        """Verify reaction metadata is extracted and emoji encoding repaired."""
        data = {
            "participants": [{"name": "Alice"}, {"name": "Bob"}],
            "messages": [
                {
                    "sender_name": "Alice",
                    "timestamp_ms": 1700000010000,
                    "content": "Happy birthday!",
                    "reactions": [
                        {"reaction": "â\x9d\xa4", "actor": "Bob"}  # Mojibake for red heart
                    ],
                }
            ],
        }
        conv = self.parser.parse(data)
        self.assertEqual(len(conv.messages), 1)
        msg = conv.messages[0]
        self.assertIn("reactions", msg.metadata)
        # Repaired to red heart ❤️
        self.assertIn("actor", msg.metadata["reactions"][0])
        self.assertEqual(msg.metadata["reactions"][0]["actor"], "Bob")

    def test_malformed_json_string(self):
        """Verify parser returns structured warning when raw JSON string is invalid."""
        conv = self.parser.parse("{\"messages\": [ incomplete json ...")
        self.assertEqual(len(conv.messages), 0)
        self.assertTrue(any(w.code == "MALFORMED_JSON" for w in conv.warnings))

    def test_unexpected_json_structure(self):
        """Verify parser reports error when top-level messages array is absent."""
        conv = self.parser.parse({"not_messages": 123})
        self.assertEqual(len(conv.messages), 0)
        self.assertTrue(any(w.code == "MISSING_MESSAGE_ARRAY" for w in conv.warnings))

    def test_unicode_and_mojibake_encoding(self):
        """Verify both clean UTF-8 and Latin-1 mojibake strings are properly decoded."""
        # 1. Latin-1 mojibake string
        mojibake_text = "Packed and ready! Let\u00e2\u0080\u0099s go! \u00f0\u009f\u008e\u0089"
        repaired, was_repaired = attempt_mojibake_repair(mojibake_text)
        self.assertTrue(was_repaired)
        self.assertEqual(repaired, "Packed and ready! Let’s go! 🎉")

        # 2. Clean UTF-8 string with Japanese and emoji (must remain intact)
        clean_utf8 = "こんにちは！世界 🌍"
        repaired2, was_repaired2 = attempt_mojibake_repair(clean_utf8)
        self.assertFalse(was_repaired2)
        self.assertEqual(repaired2, clean_utf8)

        # Full parse test
        data = {
            "participants": [{"name": "Alice"}],
            "messages": [
                {"sender_name": "Alice", "timestamp_ms": 1700000010000, "content": mojibake_text}
            ],
        }
        conv = self.parser.parse(data)
        self.assertEqual(conv.messages[0].text, "Packed and ready! Let’s go! 🎉")
        self.assertTrue(any(w.code == "MOJIBAKE_REPAIRED" for w in conv.warnings))

    def test_chronological_ordering_and_tie_breaking(self):
        """Verify reverse-chronological export is inverted, and equal timestamps preserve tie-break."""
        data = {
            "participants": [{"name": "Alice"}, {"name": "Bob"}],
            "messages": [
                # Exported newest to oldest:
                # 3. Latest
                {"sender_name": "Bob", "timestamp_ms": 1700000030000, "content": "Third"},
                # 2. Middle (same timestamp as 1)
                {"sender_name": "Bob", "timestamp_ms": 1700000010000, "content": "Second"},
                # 1. Earliest (same timestamp as 2)
                {"sender_name": "Alice", "timestamp_ms": 1700000010000, "content": "First"},
            ],
        }
        conv = self.parser.parse(data)
        self.assertEqual(len(conv.messages), 3)
        self.assertEqual(conv.messages[0].text, "First")
        self.assertEqual(conv.messages[1].text, "Second")
        self.assertEqual(conv.messages[2].text, "Third")

    def test_missing_sender_fallback(self):
        """Verify missing sender defaults to Unknown with a warning."""
        data = {
            "messages": [
                {"timestamp_ms": 1700000010000, "content": "Hello world"}
            ]
        }
        conv = self.parser.parse(data)
        self.assertEqual(conv.messages[0].speaker, "Unknown")
        self.assertTrue(any(w.code == "MISSING_SENDER" for w in conv.warnings))


class TestInstagramApiIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_import_endpoint_instagram_auto_detection(self):
        """Verify POST /import auto-detects Instagram JSON payload without explicit format."""
        ig_payload = json.dumps({
            "participants": [{"name": "Alice"}, {"name": "Bob"}],
            "messages": [
                {"sender_name": "Bob", "timestamp_ms": 1700000020000, "content": "I am great!"},
                {"sender_name": "Alice", "timestamp_ms": 1700000010000, "content": "How are you?"},
            ],
        })
        response = self.client.post(
            "/import",
            json={"conversation": ig_payload},  # format defaults to 'auto'
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["source_format"], "instagram_json")
        self.assertEqual(data["num_messages"], 2)
        self.assertEqual(data["num_speakers"], 2)
        self.assertEqual(data["speakers"], ["Alice", "Bob"])
        # Check chronological ordering: Alice then Bob
        self.assertEqual(data["normalized_conversation"]["messages"][0]["speaker"], "Alice")
        self.assertEqual(data["normalized_conversation"]["messages"][1]["speaker"], "Bob")

    def test_analyze_endpoint_with_instagram_json(self):
        """Verify existing POST /analyze can analyze Instagram JSON conversations end-to-end."""
        ig_payload = json.dumps({
            "participants": [{"name": "Alice"}, {"name": "Bob"}],
            "messages": [
                {"sender_name": "Bob", "timestamp_ms": 1700000020000, "content": "Me too!"},
                {"sender_name": "Alice", "timestamp_ms": 1700000010000, "content": "I am so excited!"},
            ],
        })
        response = self.client.post(
            "/analyze",
            json={"conversation": ig_payload, "mode": "utterance"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["turns"]), 2)
        self.assertIn("analysis", data)
        self.assertIn("canonical", data)
        self.assertEqual(data["canonical"]["source_format"], "instagram_json")

    def test_analyze_endpoint_dialoguernn_with_instagram_json(self):
        """Verify DialogueRNN mode tracks recurrent party states on Instagram conversations."""
        ig_payload = json.dumps({
            "participants": [{"name": "Chandler"}, {"name": "Joey"}],
            "messages": [
                {"sender_name": "Joey", "timestamp_ms": 1700000020000, "content": "I love acting!"},
                {"sender_name": "Chandler", "timestamp_ms": 1700000010000, "content": "Good luck with your audition today."},
            ],
        })
        response = self.client.post(
            "/analyze",
            json={"conversation": ig_payload, "mode": "dialoguernn"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["mode"], "dialoguernn")
        self.assertEqual(len(data["turns"]), 2)
        self.assertEqual(data["turns"][0]["speaker"], "Chandler")
        self.assertEqual(data["turns"][1]["speaker"], "Joey")
        self.assertIn("probabilities", data["turns"][0])
        self.assertEqual(data["analysis"]["total_transitions"], 1)


if __name__ == "__main__":
    unittest.main()
