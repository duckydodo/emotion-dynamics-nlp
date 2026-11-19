"""
Unit and integration tests for WhatsApp plain-text conversation parser.
Tests Android dash, iOS bracketed, 12h/24h timestamps, multiline utterances,
colons in message text, media placeholders, system events, phone number senders,
and API integration.
"""

from datetime import datetime, timezone
import json
import unittest

from fastapi.testclient import TestClient

from src.api import app
from src.data.parsers.whatsapp import WhatsAppTxtParser
from src.data.schema import MessageType, SourceFormat


class TestWhatsAppTxtParser(unittest.TestCase):

    def setUp(self):
        self.parser = WhatsAppTxtParser()

    def test_android_dash_format_24h(self):
        """Verify standard Android format: 'DD/MM/YYYY, HH:MM - Speaker: text'."""
        text = (
            "15/05/2026, 14:30 - Alice: Hey Bob, are you ready?\n"
            "15/05/2026, 14:32 - Bob: Yes, leaving right now!"
        )
        conv = self.parser.parse(text)
        self.assertEqual(conv.source_format, SourceFormat.WHATSAPP_TXT)
        self.assertEqual(len(conv.participants), 2)
        self.assertEqual(len(conv.messages), 2)

        self.assertEqual(conv.messages[0].speaker, "Alice")
        self.assertEqual(conv.messages[0].text, "Hey Bob, are you ready?")
        self.assertEqual(conv.messages[0].timestamp, datetime(2026, 5, 15, 14, 30, tzinfo=timezone.utc))

        self.assertEqual(conv.messages[1].speaker, "Bob")
        self.assertEqual(conv.messages[1].text, "Yes, leaving right now!")
        self.assertEqual(conv.messages[1].timestamp, datetime(2026, 5, 15, 14, 32, tzinfo=timezone.utc))

        evaluable = conv.get_evaluable_turns()
        self.assertEqual(len(evaluable), 2)

    def test_android_dash_format_12h_ampm(self):
        """Verify Android 12-hour format with AM/PM."""
        text = (
            "15/05/2026, 9:15 am - Alice: Good morning!\n"
            "15/05/2026, 1:45 pm - Bob: Good afternoon!"
        )
        conv = self.parser.parse(text)
        self.assertEqual(len(conv.messages), 2)
        self.assertEqual(conv.messages[0].timestamp, datetime(2026, 5, 15, 9, 15, tzinfo=timezone.utc))
        self.assertEqual(conv.messages[1].timestamp, datetime(2026, 5, 15, 13, 45, tzinfo=timezone.utc))

    def test_ios_bracketed_format(self):
        """Verify iOS / macOS bracketed format: '[DD/MM/YYYY, HH:MM:SS] Speaker: text'."""
        text = (
            "[15/05/2026, 14:32:15] Alice: Meeting starting soon.\n"
            "[15/05/2026, 14:33:00] Bob: Joining now."
        )
        conv = self.parser.parse(text)
        self.assertEqual(len(conv.messages), 2)
        self.assertEqual(conv.messages[0].speaker, "Alice")
        self.assertEqual(conv.messages[0].timestamp, datetime(2026, 5, 15, 14, 32, 15, tzinfo=timezone.utc))
        self.assertEqual(conv.messages[1].speaker, "Bob")
        self.assertEqual(conv.messages[1].timestamp, datetime(2026, 5, 15, 14, 33, 0, tzinfo=timezone.utc))

    def test_ios_bracketed_format_with_invisible_lrm_and_ampm(self):
        """Verify iOS bracketed format with hidden direction marks (\\u200e) and narrow no-break space."""
        text = (
            "\u200e[15/05/2026, 2:32:15\u202fPM] Alice: Hello!\n"
            "\u200e[15/05/2026, 2:33:00\u202fPM] Bob: Hi there!"
        )
        conv = self.parser.parse(text)
        self.assertEqual(len(conv.messages), 2)
        self.assertEqual(conv.messages[0].speaker, "Alice")
        self.assertEqual(conv.messages[0].timestamp, datetime(2026, 5, 15, 14, 32, 15, tzinfo=timezone.utc))

    def test_multiline_messages(self):
        """Verify lines without timestamp markers are appended to previous message."""
        text = (
            "15/05/2026, 14:30 - Alice: This is paragraph one.\n"
            "This is continuation of paragraph one.\n"
            "And line three.\n"
            "15/05/2026, 14:32 - Bob: Got it all."
        )
        conv = self.parser.parse(text)
        self.assertEqual(len(conv.messages), 2)
        self.assertEqual(
            conv.messages[0].text,
            "This is paragraph one. This is continuation of paragraph one. And line three.",
        )
        self.assertEqual(conv.messages[1].text, "Got it all.")

    def test_messages_with_colons_in_text(self):
        """Verify colons inside text content (URLs, timestamps) do not break parsing."""
        text = (
            "15/05/2026, 14:30 - Alice: Meeting update: join at 10:30:00 AM sharp!\n"
            "15/05/2026, 14:31 - Bob: Check url: https://example.com/meeting: passcode is 1234."
        )
        conv = self.parser.parse(text)
        self.assertEqual(len(conv.messages), 2)
        self.assertEqual(conv.messages[0].speaker, "Alice")
        self.assertEqual(conv.messages[0].text, "Meeting update: join at 10:30:00 AM sharp!")
        self.assertEqual(conv.messages[1].speaker, "Bob")
        self.assertEqual(conv.messages[1].text, "Check url: https://example.com/meeting: passcode is 1234.")

    def test_system_notifications(self):
        """Verify system messages (encryption, group updates) are classified MessageType.SYSTEM."""
        text = (
            "15/05/2026, 14:00 - Messages and calls are end-to-end encrypted. No one outside can read them.\n"
            "15/05/2026, 14:01 - Alice added Bob\n"
            "15/05/2026, 14:05 - Alice: Welcome Bob!\n"
            "15/05/2026, 14:06 - Bob: Thanks Alice."
        )
        conv = self.parser.parse(text)
        self.assertEqual(len(conv.messages), 4)

        # Message 0: Encryption
        self.assertEqual(conv.messages[0].message_type, MessageType.SYSTEM)
        self.assertEqual(conv.messages[0].speaker, "System")
        self.assertFalse(conv.messages[0].is_evaluable)

        # Message 1: Added user
        self.assertEqual(conv.messages[1].message_type, MessageType.SYSTEM)
        self.assertFalse(conv.messages[1].is_evaluable)

        # Message 2 & 3: Real speech turns
        self.assertEqual(conv.messages[2].message_type, MessageType.TEXT)
        self.assertTrue(conv.messages[2].is_evaluable)
        self.assertEqual(conv.messages[3].message_type, MessageType.TEXT)
        self.assertTrue(conv.messages[3].is_evaluable)

        evaluable = conv.get_evaluable_turns()
        self.assertEqual(len(evaluable), 2)
        self.assertEqual(evaluable[0].speaker, "Alice")
        self.assertEqual(evaluable[1].speaker, "Bob")

    def test_media_placeholders_omitted(self):
        """Verify '<Media omitted>', 'image omitted', 'sticker omitted' become MessageType.MEDIA with empty text."""
        text = (
            "15/05/2026, 14:10 - Alice: Look at this photo:\n"
            "15/05/2026, 14:11 - Alice: <Media omitted>\n"
            "15/05/2026, 14:12 - Bob: image omitted\n"
            "15/05/2026, 14:13 - Bob: Nice picture!"
        )
        conv = self.parser.parse(text)
        self.assertEqual(len(conv.messages), 4)

        # Message 1: <Media omitted>
        self.assertEqual(conv.messages[1].message_type, MessageType.MEDIA)
        self.assertEqual(conv.messages[1].text, "")
        self.assertFalse(conv.messages[1].is_evaluable)
        self.assertIn("media_placeholder", conv.messages[1].metadata)

        # Message 2: image omitted
        self.assertEqual(conv.messages[2].message_type, MessageType.MEDIA)
        self.assertEqual(conv.messages[2].text, "")
        self.assertFalse(conv.messages[2].is_evaluable)

        # Evaluable turns only include speech turns 0 and 3
        evaluable = conv.get_evaluable_turns()
        self.assertEqual(len(evaluable), 2)
        self.assertEqual(evaluable[0].turn_id, 0)
        self.assertEqual(evaluable[0].text, "Look at this photo:")
        self.assertEqual(evaluable[1].turn_id, 1)
        self.assertEqual(evaluable[1].text, "Nice picture!")

    def test_attached_prefix_with_caption(self):
        """Verify '<attached: photo.jpg> Caption here' extracts attachment and retains text caption."""
        text = (
            "[15/05/2026, 14:20:00] Alice: <attached: sunset.jpg> Look at this sunset!\n"
            "[15/05/2026, 14:21:00] Bob: <attached: sticker.webp>"
        )
        conv = self.parser.parse(text)
        self.assertEqual(len(conv.messages), 2)

        # Message 0: Caption extracted
        self.assertEqual(conv.messages[0].message_type, MessageType.TEXT)
        self.assertEqual(conv.messages[0].text, "Look at this sunset!")
        self.assertTrue(conv.messages[0].is_evaluable)
        self.assertEqual(conv.messages[0].metadata["attachment"], "sunset.jpg")

        # Message 1: No caption
        self.assertEqual(conv.messages[1].message_type, MessageType.MEDIA)
        self.assertEqual(conv.messages[1].text, "")
        self.assertFalse(conv.messages[1].is_evaluable)
        self.assertEqual(conv.messages[1].metadata["attachment"], "sticker.webp")

    def test_sender_with_phone_number_and_punctuation(self):
        """Verify senders with phone numbers, parentheses, or titles."""
        text = (
            "15/05/2026, 14:30 - +1 (555) 123-4567: Hello from phone number!\n"
            "15/05/2026, 14:31 - Dr. Smith (Work): Acknowledged."
        )
        conv = self.parser.parse(text)
        self.assertEqual(len(conv.participants), 2)
        self.assertEqual(conv.messages[0].speaker, "+1 (555) 123-4567")
        self.assertEqual(conv.messages[1].speaker, "Dr. Smith (Work)")

    def test_unicode_and_emojis(self):
        """Verify UTF-8 characters and multi-byte emojis are preserved."""
        text = (
            "15/05/2026, 14:30 - Elena: Bonjour! Ça va bien? ✨\n"
            "15/05/2026, 14:31 - Kenji: こんにちは！世界 🌸"
        )
        conv = self.parser.parse(text)
        self.assertEqual(len(conv.messages), 2)
        self.assertEqual(conv.messages[0].text, "Bonjour! Ça va bien? ✨")
        self.assertEqual(conv.messages[1].text, "こんにちは！世界 🌸")

    def test_date_convention_inference(self):
        """Verify parser infers Month-first (MDY) when second number exceeds 12."""
        text = (
            "05/25/2026, 14:30 - Alice: Day is 25, month is 5!\n"
            "05/25/2026, 14:31 - Bob: Correct."
        )
        conv = self.parser.parse(text)
        self.assertEqual(conv.metadata["date_convention"], "MDY")
        self.assertEqual(conv.messages[0].timestamp, datetime(2026, 5, 25, 14, 30, tzinfo=timezone.utc))

    def test_unrecognized_line_before_first_message(self):
        """Verify lines before the first timestamped message are preserved as UNPARSED warnings."""
        text = (
            "Chat history exported on 2026-11-10\n"
            "15/05/2026, 14:30 - Alice: Hi!\n"
            "15/05/2026, 14:31 - Bob: Hello!"
        )
        conv = self.parser.parse(text)
        self.assertEqual(len(conv.messages), 3)
        self.assertEqual(conv.messages[0].message_type, MessageType.UNPARSED)
        self.assertFalse(conv.messages[0].is_evaluable)
        self.assertTrue(any(w.code == "UNPARSED_LINE" for w in conv.warnings))


class TestWhatsAppApiIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_import_endpoint_whatsapp_auto_detection(self):
        """Verify POST /import auto-detects WhatsApp text payload without explicit format."""
        wa_text = (
            "15/05/2026, 14:30 - Alice: Good morning!\n"
            "15/05/2026, 14:31 - Bob: Morning Alice!"
        )
        response = self.client.post(
            "/import",
            json={"conversation": wa_text},  # format defaults to 'auto'
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["source_format"], "whatsapp_txt")
        self.assertEqual(data["num_messages"], 2)
        self.assertEqual(data["num_speakers"], 2)
        self.assertEqual(data["speakers"], ["Alice", "Bob"])

    def test_analyze_endpoint_with_whatsapp_text(self):
        """Verify existing POST /analyze can analyze WhatsApp text conversations end-to-end."""
        wa_text = (
            "15/05/2026, 14:30 - Alice: I am really happy about this!\n"
            "15/05/2026, 14:31 - Bob: Me too, congratulations!"
        )
        response = self.client.post(
            "/analyze",
            json={"conversation": wa_text, "mode": "utterance"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["turns"]), 2)
        self.assertEqual(data["turns"][0]["speaker"], "Alice")
        self.assertEqual(data["turns"][1]["speaker"], "Bob")
        self.assertIn("analysis", data)
        self.assertIn("canonical", data)
        self.assertEqual(data["canonical"]["source_format"], "whatsapp_txt")

    def test_analyze_endpoint_dialoguernn_with_whatsapp_text(self):
        """Verify DialogueRNN tracks dynamic party states on WhatsApp conversations."""
        wa_text = (
            "[15/05/2026, 10:00:00] Chandler: Could this be any more exciting?\n"
            "[15/05/2026, 10:01:00] Joey: I love it!"
        )
        response = self.client.post(
            "/analyze",
            json={"conversation": wa_text, "mode": "dialoguernn"},
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
