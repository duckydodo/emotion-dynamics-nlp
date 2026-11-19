"""
Tests for canonical conversation schema, plain text parser, and /import API endpoint.
"""

from datetime import datetime
import unittest

from fastapi.testclient import TestClient

from src.api import app
from src.data.conversation import (
    ConversationParseError,
    ConversationTurn,
    parse_canonical_conversation,
    parse_conversation,
)
from src.data.parsers.plain_text import PlainTextParser
from src.data.schema import (
    CanonicalConversation,
    CanonicalMessage,
    MessageType,
    SourceFormat,
    WarningSeverity,
)


class TestCanonicalParser(unittest.TestCase):

    def setUp(self):
        self.parser = PlainTextParser()

    def test_basic_two_speaker_conversation(self):
        """Verify standard alternating two-speaker dialogue."""
        text = (
            "Alice: Hi Bob!\n"
            "Bob: Hello Alice, how are you?\n"
            "Alice: I am doing great."
        )
        conv = self.parser.parse(text)
        self.assertEqual(len(conv.messages), 3)
        self.assertEqual(len(conv.participants), 2)
        self.assertEqual(conv.participants[0].name, "Alice")
        self.assertEqual(conv.participants[0].speaker_id, "spk_0")
        self.assertEqual(conv.participants[0].turn_count, 2)
        self.assertEqual(conv.participants[1].name, "Bob")
        self.assertEqual(conv.participants[1].speaker_id, "spk_1")
        self.assertEqual(conv.participants[1].turn_count, 1)

        evaluable = conv.get_evaluable_turns()
        self.assertEqual(len(evaluable), 3)
        self.assertEqual(evaluable[0].turn_id, 0)
        self.assertEqual(evaluable[1].turn_id, 1)
        self.assertEqual(evaluable[2].turn_id, 2)
        self.assertEqual(evaluable[0].speaker, "Alice")
        self.assertEqual(evaluable[1].speaker, "Bob")
        self.assertEqual(conv.validate(), [])

    def test_multiple_speakers(self):
        """Verify conversations with 3 or more dynamic speakers."""
        text = (
            "Chandler: Hey guys!\n"
            "Joey: How you doing?\n"
            "Monica: Joey, please clean up.\n"
            "Rachel: I can help too."
        )
        conv = self.parser.parse(text)
        self.assertEqual(len(conv.participants), 4)
        speaker_names = [p.name for p in conv.participants]
        self.assertEqual(speaker_names, ["Chandler", "Joey", "Monica", "Rachel"])
        self.assertEqual(len(conv.messages), 4)
        for i, msg in enumerate(conv.messages):
            self.assertEqual(msg.sequence_index, i)
            self.assertEqual(msg.source_index, i)
            self.assertTrue(msg.is_evaluable)

    def test_multiline_messages(self):
        """Verify messages spanning multiple lines without speaker prefixes."""
        text = (
            "Alice: This is paragraph one.\n"
            "This is continuation of paragraph one.\n"
            "Bob: Got it.\n"
            "Here is another continuation line for Bob."
        )
        conv = self.parser.parse(text)
        self.assertEqual(len(conv.messages), 2)
        self.assertEqual(
            conv.messages[0].text,
            "This is paragraph one. This is continuation of paragraph one.",
        )
        self.assertEqual(
            conv.messages[1].text,
            "Got it. Here is another continuation line for Bob.",
        )

    def test_colons_in_message_content(self):
        """Verify colons inside message text are preserved without breaking utterance."""
        text = (
            "Alice: Meeting update: please join at 10:30:00 AM sharp!\n"
            "Bob: URL reference: https://meet.google.com/abc-defg-hij: join now."
        )
        conv = self.parser.parse(text)
        self.assertEqual(len(conv.messages), 2)
        self.assertEqual(
            conv.messages[0].text,
            "Meeting update: please join at 10:30:00 AM sharp!",
        )
        self.assertEqual(
            conv.messages[1].text,
            "URL reference: https://meet.google.com/abc-defg-hij: join now.",
        )

    def test_blank_lines(self):
        """Verify arbitrary blank lines and whitespace are handled cleanly."""
        text = (
            "\n\n"
            "Alice: First turn.\n"
            "\n"
            "\n"
            "Bob: Second turn.\n"
            "\n"
        )
        conv = self.parser.parse(text)
        self.assertEqual(len(conv.messages), 2)
        self.assertEqual(conv.messages[0].text, "First turn.")
        self.assertEqual(conv.messages[1].text, "Second turn.")

    def test_unicode_and_non_english(self):
        """Verify UTF-8 characters, emojis, and non-Latin scripts are uncorrupted."""
        text = (
            "Elena: Bonjour! Ça va bien? ✨\n"
            "Kenji: こんにちは！元気です 🌸\n"
            "Fatima: مرحبا بكم جميعاً! 🚀"
        )
        conv = self.parser.parse(text)
        self.assertEqual(len(conv.messages), 3)
        self.assertEqual(conv.messages[0].text, "Bonjour! Ça va bien? ✨")
        self.assertEqual(conv.messages[1].text, "こんにちは！元気です 🌸")
        self.assertEqual(conv.messages[2].text, "مرحبا بكم جميعاً! 🚀")

    def test_missing_speaker_and_unrecognized_lines(self):
        """Verify unrecognized lines before speaker are captured as warnings, not crashing parser."""
        text = (
            "--- Chat Export 2026-11-10 ---\n"
            "Alice: Hello there!\n"
            "Bob: Hi Alice!"
        )
        conv = self.parser.parse(text)
        # Message 0 is unparsed header
        self.assertEqual(len(conv.messages), 3)
        self.assertEqual(conv.messages[0].message_type, MessageType.UNPARSED)
        self.assertFalse(conv.messages[0].is_evaluable)
        self.assertEqual(conv.messages[1].speaker, "Alice")
        self.assertEqual(conv.messages[2].speaker, "Bob")

        # Warnings should record the unparsed line
        unparsed_warns = [w for w in conv.warnings if w.code == "UNPARSED_LINE"]
        self.assertEqual(len(unparsed_warns), 1)

        # Evaluable turns only include the 2 real turns
        evaluable = conv.get_evaluable_turns()
        self.assertEqual(len(evaluable), 2)
        self.assertEqual(evaluable[0].turn_id, 0)
        self.assertEqual(evaluable[1].turn_id, 1)

    def test_empty_input(self):
        """Verify empty or whitespace-only input behavior."""
        conv = self.parser.parse("   \n\t  ")
        self.assertEqual(len(conv.messages), 0)
        val_errors = conv.validate()
        self.assertTrue(any("Empty conversation" in err for err in val_errors))

        # parse_conversation raises ConversationParseError
        with self.assertRaises(ConversationParseError):
            parse_conversation("   \n  ")

    def test_malformed_lines_and_empty_utterance(self):
        """Verify speaker with empty utterance generates warning and is_evaluable=False."""
        text = (
            "Alice:\n"
            "Bob: Where are you?"
        )
        conv = self.parser.parse(text)
        self.assertEqual(len(conv.messages), 2)
        self.assertEqual(conv.messages[0].speaker, "Alice")
        self.assertEqual(conv.messages[0].text, "")
        self.assertFalse(conv.messages[0].is_evaluable)
        self.assertTrue(conv.messages[1].is_evaluable)

        evaluable = conv.get_evaluable_turns()
        self.assertEqual(len(evaluable), 1)
        self.assertEqual(evaluable[0].speaker, "Bob")

    def test_repeated_speaker_names(self):
        """Verify consecutive turns by the same speaker are distinctly preserved."""
        text = (
            "Alice: I have a quick question.\n"
            "Alice: Actually, never mind, I solved it!\n"
            "Bob: Glad to hear."
        )
        conv = self.parser.parse(text)
        self.assertEqual(len(conv.messages), 3)
        self.assertEqual(conv.messages[0].speaker, "Alice")
        self.assertEqual(conv.messages[1].speaker, "Alice")
        self.assertEqual(conv.messages[2].speaker, "Bob")
        self.assertEqual(conv.messages[0].message_id, "msg_0001")
        self.assertEqual(conv.messages[1].message_id, "msg_0002")
        self.assertEqual(conv.messages[2].message_id, "msg_0003")
        self.assertEqual(len(conv.participants), 2)
        self.assertEqual(conv.participants[0].turn_count, 2)


class TestImportApiEndpoint(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_import_endpoint_success(self):
        """Verify POST /import returns canonical schema and metadata."""
        conv_text = (
            "Alice: Good morning!\n"
            "Bob: Morning Alice, ready for the demo?\n"
            "Alice: Absolutely!"
        )
        response = self.client.post(
            "/import",
            json={"conversation": conv_text, "format": "plain_text"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("conversation_id", data)
        self.assertEqual(data["source_format"], "plain_text")
        self.assertEqual(data["num_messages"], 3)
        self.assertEqual(data["num_evaluable_turns"], 3)
        self.assertEqual(data["num_speakers"], 2)
        self.assertEqual(data["speakers"], ["Alice", "Bob"])
        self.assertIn("normalized_conversation", data)
        self.assertIn("import_metadata", data)
        self.assertEqual(data["validation_errors"], [])

        # Verify normalized messages
        msgs = data["normalized_conversation"]["messages"]
        self.assertEqual(len(msgs), 3)
        self.assertEqual(msgs[0]["speaker"], "Alice")
        self.assertEqual(msgs[0]["speaker_id"], "spk_0")
        self.assertEqual(msgs[1]["speaker"], "Bob")
        self.assertEqual(msgs[1]["speaker_id"], "spk_1")

    def test_import_endpoint_empty_error(self):
        """Verify POST /import on empty text returns HTTP 400."""
        response = self.client.post(
            "/import",
            json={"conversation": "   ", "format": "plain_text"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Conversation is empty", response.json()["detail"])

    def test_import_endpoint_no_speakers_validation(self):
        """Verify POST /import on text without speakers returns status warning and validation errors."""
        response = self.client.post(
            "/import",
            json={"conversation": "Unrecognized line without colon", "format": "plain_text"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "warning")
        self.assertEqual(data["num_evaluable_turns"], 0)
        self.assertTrue(len(data["validation_errors"]) > 0)


if __name__ == "__main__":
    unittest.main()
