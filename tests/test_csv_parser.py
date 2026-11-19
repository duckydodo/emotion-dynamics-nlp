"""
Unit and integration tests for TabularCsvParser.
Tests standard CSV, alternative column names, explicit column mappings,
quoted commas, multiline fields, UTF-8 BOM, missing columns, empty rows,
invalid timestamps, duplicate IDs, different delimiters, and API integration.
"""

from datetime import datetime, timezone
import json
import unittest

from fastapi.testclient import TestClient

from src.api import app
from src.data.parsers.csv_parser import TabularCsvParser
from src.data.schema import MessageType, SourceFormat


class TestTabularCsvParser(unittest.TestCase):

    def setUp(self):
        self.parser = TabularCsvParser()

    def test_standard_csv(self):
        """Verify standard CSV with 'speaker,text' columns."""
        csv_text = (
            "speaker,text\n"
            "Alice,Good morning Bob!\n"
            'Bob,"Morning Alice, how are you?"\n'
            "Alice,I am doing well."
        )
        conv = self.parser.parse(csv_text)
        self.assertEqual(conv.source_format, SourceFormat.TABULAR_CSV)
        self.assertEqual(len(conv.participants), 2)
        self.assertEqual(len(conv.messages), 3)

        self.assertEqual(conv.messages[0].speaker, "Alice")
        self.assertEqual(conv.messages[0].text, "Good morning Bob!")
        self.assertEqual(conv.messages[1].speaker, "Bob")
        self.assertEqual(conv.messages[1].text, "Morning Alice, how are you?")
        self.assertEqual(conv.metadata["rows_imported"], 3)
        self.assertEqual(conv.metadata["rows_rejected"], 0)

        evaluable = conv.get_evaluable_turns()
        self.assertEqual(len(evaluable), 3)
        self.assertEqual(evaluable[0].turn_id, 0)
        self.assertEqual(evaluable[1].turn_id, 1)

    def test_alternative_column_names_auto_detection(self):
        """Verify automatic matching of common column aliases (e.g. Sender, Utterance)."""
        csv_text = (
            "Sender,Utterance,CreatedAt\n"
            "Alice,Are you ready?,2026-11-10T14:30:00Z\n"
            "Bob,Yes I am!,2026-11-10T14:31:00Z\n"
        )
        conv = self.parser.parse(csv_text)
        self.assertEqual(len(conv.messages), 2)
        self.assertEqual(conv.messages[0].speaker, "Alice")
        self.assertEqual(conv.messages[0].text, "Are you ready?")
        self.assertEqual(conv.messages[0].timestamp, datetime(2026, 11, 10, 14, 30, 0, tzinfo=timezone.utc))
        self.assertEqual(conv.messages[1].speaker, "Bob")
        self.assertEqual(conv.messages[1].text, "Yes I am!")

    def test_explicit_column_mapping(self):
        """Verify user-provided column_mapping overrides and maps arbitrary column headers."""
        csv_text = (
            "person_name,spoken_line,time_stamp\n"
            "Chandler,Could this BE any cooler?,1700000010\n"
            "Joey,I know right!,1700000020\n"
        )
        mapping = {
            "speaker": "person_name",
            "text": "spoken_line",
            "timestamp": "time_stamp",
        }
        conv = self.parser.parse(csv_text, column_mapping=mapping)
        self.assertEqual(len(conv.messages), 2)
        self.assertEqual(conv.messages[0].speaker, "Chandler")
        self.assertEqual(conv.messages[0].text, "Could this BE any cooler?")
        self.assertEqual(conv.messages[1].speaker, "Joey")
        self.assertEqual(conv.messages[1].text, "I know right!")

    def test_quoted_commas(self):
        """Verify fields containing embedded commas inside quotes are preserved intact."""
        csv_text = (
            'speaker,text\n'
            'Alice,"Hello, my friend, how are you today?"\n'
            'Bob,"I am fine, thank you, ready to go."\n'
        )
        conv = self.parser.parse(csv_text)
        self.assertEqual(len(conv.messages), 2)
        self.assertEqual(conv.messages[0].text, "Hello, my friend, how are you today?")
        self.assertEqual(conv.messages[1].text, "I am fine, thank you, ready to go.")

    def test_multiline_fields(self):
        """Verify multiline quoted text fields preserve paragraph content."""
        csv_text = (
            'speaker,text\n'
            'Alice,"Line one.\nLine two.\nLine three."\n'
            'Bob,Single line response.\n'
        )
        conv = self.parser.parse(csv_text)
        self.assertEqual(len(conv.messages), 2)
        self.assertEqual(conv.messages[0].text, "Line one.\nLine two.\nLine three.")
        self.assertEqual(conv.messages[1].text, "Single line response.")

    def test_utf8_bom(self):
        """Verify leading UTF-8 Byte Order Mark (\\ufeff) does not corrupt column 0."""
        csv_text = (
            "\ufeffspeaker,text\n"
            "Alice,Testing BOM\n"
            "Bob,BOM works!\n"
        )
        conv = self.parser.parse(csv_text)
        self.assertEqual(len(conv.messages), 2)
        self.assertEqual(conv.messages[0].speaker, "Alice")
        self.assertEqual(conv.messages[0].text, "Testing BOM")

    def test_missing_required_columns(self):
        """Verify CSV lacking speaker or text columns returns COLUMN_MAPPING_REQUIRED error."""
        csv_text = (
            "product_id,price,quantity\n"
            "101,29.99,5\n"
            "102,99.50,2\n"
        )
        conv = self.parser.parse(csv_text)
        self.assertEqual(len(conv.messages), 0)
        self.assertTrue(any(w.code == "COLUMN_MAPPING_REQUIRED" for w in conv.warnings))
        val_errors = conv.validate()
        self.assertTrue(len(val_errors) > 0)

    def test_empty_rows(self):
        """Verify empty and whitespace-only rows are skipped cleanly."""
        csv_text = (
            "speaker,text\n"
            "\n"
            "Alice,Turn 1\n"
            "   \n"
            "Bob,Turn 2\n"
            "\n\n"
        )
        conv = self.parser.parse(csv_text)
        self.assertEqual(len(conv.messages), 2)
        self.assertEqual(conv.metadata["rows_imported"], 2)

    def test_invalid_timestamps(self):
        """Verify invalid timestamps generate a warning and set timestamp to None without failing."""
        csv_text = (
            "speaker,text,timestamp\n"
            "Alice,Hello,2026-11-10T14:30:00Z\n"
            "Bob,Hi,NOT_A_VALID_DATE_TIME\n"
        )
        conv = self.parser.parse(csv_text)
        self.assertEqual(len(conv.messages), 2)
        self.assertIsNotNone(conv.messages[0].timestamp)
        self.assertIsNone(conv.messages[1].timestamp)
        self.assertEqual(conv.metadata["invalid_timestamps"], 1)
        self.assertTrue(any(w.code == "INVALID_TIMESTAMP" for w in conv.warnings))

    def test_duplicate_message_ids(self):
        """Verify duplicate IDs in CSV trigger DUPLICATE_MESSAGE_ID warning and local unique ID."""
        csv_text = (
            "id,speaker,text\n"
            "turn_1,Alice,First message\n"
            "turn_1,Bob,Duplicate ID message\n"
        )
        conv = self.parser.parse(csv_text)
        self.assertEqual(len(conv.messages), 2)
        self.assertEqual(conv.metadata["duplicate_ids"], 1)
        self.assertTrue(any(w.code == "DUPLICATE_MESSAGE_ID" for w in conv.warnings))
        self.assertNotEqual(conv.messages[0].message_id, conv.messages[1].message_id)

    def test_different_delimiters_tsv_and_semicolon(self):
        """Verify tab-separated (TSV) and semicolon-separated files are auto-detected."""
        # TSV
        tsv_text = (
            "speaker\ttext\n"
            "Alice\tHello tab world\n"
            "Bob\tTSV works\n"
        )
        conv_tsv = self.parser.parse(tsv_text)
        self.assertEqual(conv_tsv.metadata["delimiter"], "\t")
        self.assertEqual(len(conv_tsv.messages), 2)

        # Semicolon
        semi_text = (
            "speaker;text\n"
            "Alice;Hello semicolon\n"
            "Bob;Semicolon works\n"
        )
        conv_semi = self.parser.parse(semi_text)
        self.assertEqual(conv_semi.metadata["delimiter"], ";")
        self.assertEqual(len(conv_semi.messages), 2)

    def test_missing_speakers_and_empty_text_diagnostics(self):
        """Verify rejected rows (missing speaker or empty text) are reported accurately."""
        csv_text = (
            "speaker,text\n"
            ",Missing speaker text\n"  # Missing speaker -> rejected
            "Alice,\n"                 # Empty text -> un-evaluable
            "Bob,Valid speech turn\n"  # Valid
        )
        conv = self.parser.parse(csv_text)
        self.assertEqual(conv.metadata["rows_rejected"], 1)
        self.assertEqual(conv.metadata["missing_speakers"], 1)
        self.assertEqual(conv.metadata["empty_text_rows"], 1)
        self.assertEqual(conv.metadata["rows_imported"], 1)

        evaluable = conv.get_evaluable_turns()
        self.assertEqual(len(evaluable), 1)
        self.assertEqual(evaluable[0].speaker, "Bob")

    def test_meld_format_csv_import(self):
        """Verify standard MELD CSV format is parsed directly with column aliases."""
        meld_text = (
            'Sr No.,Utterance,Speaker,Emotion,Sentiment,Dialogue_ID,Utterance_ID,Season,Episode,StartTime,EndTime\n'
            '1,"Oh my God, he’s lost it.",Phoebe,sadness,negative,0,0,4,7,"00:20:57,256","00:21:00,049"\n'
            '2,What?,Monica,surprise,negative,0,1,4,7,"00:21:01,927","00:21:03,261"\n'
        )
        conv = self.parser.parse(meld_text)
        self.assertEqual(len(conv.messages), 2)
        self.assertEqual(conv.messages[0].speaker, "Phoebe")
        self.assertEqual(conv.messages[0].text, "Oh my God, he’s lost it.")
        self.assertEqual(conv.messages[1].speaker, "Monica")
        self.assertEqual(conv.messages[1].text, "What?")


class TestCsvApiIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.parser = TabularCsvParser()

    def test_import_endpoint_csv_auto_detection(self):
        """Verify POST /import auto-detects CSV payload and returns canonical representation."""
        csv_text = (
            "speaker,text\n"
            "Alice,Good morning!\n"
            "Bob,Good morning Alice!\n"
        )
        response = self.client.post(
            "/import",
            json={"conversation": csv_text},  # format defaults to 'auto'
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["source_format"], "tabular_csv")
        self.assertEqual(data["num_messages"], 2)
        self.assertEqual(data["num_speakers"], 2)
        self.assertEqual(data["speakers"], ["Alice", "Bob"])

    def test_import_endpoint_csv_with_explicit_mapping(self):
        """Verify POST /import accepts custom column_mapping."""
        csv_text = (
            "char_name,dialogue_text\n"
            "Alice,Hello custom mapping\n"
            "Bob,Confirmed\n"
        )
        response = self.client.post(
            "/import",
            json={
                "conversation": csv_text,
                "column_mapping": {"speaker": "char_name", "text": "dialogue_text"},
            },
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["source_format"], "tabular_csv")
        self.assertEqual(data["num_messages"], 2)
        self.assertEqual(data["speakers"], ["Alice", "Bob"])

    def test_analyze_endpoint_with_csv(self):
        """Verify POST /analyze can analyze CSV conversations end-to-end with DialogueRNN."""
        csv_text = (
            "speaker,text\n"
            "Alice,I am so happy about the results!\n"
            "Bob,Me too, everything went great!\n"
        )
        response = self.client.post(
            "/analyze",
            json={"conversation": csv_text, "mode": "dialoguernn"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["mode"], "dialoguernn")
        self.assertEqual(len(data["turns"]), 2)
        self.assertEqual(data["turns"][0]["speaker"], "Alice")
        self.assertEqual(data["turns"][1]["speaker"], "Bob")
        self.assertIn("analysis", data)
        self.assertEqual(data["canonical"]["source_format"], "tabular_csv")

    def test_regression_can_parse_rejects_dialogue_with_commas(self):
        """Task 32 regression: ensure ordinary 'Speaker: utterance' with commas is not identified as CSV."""
        from src.data.parsers import get_parser, parse_canonical
        from src.data.parsers.plain_text import PlainTextParser

        dialogue = (
            "Speaker A: Hello, how are you today?\n"
            "Speaker B: I am doing well, thank you!\n"
            "Speaker A: That sounds wonderful, let's proceed."
        )
        # 1. Direct can_parse check
        self.assertFalse(self.parser.can_parse(dialogue))

        # 2. get_parser auto-detection resolution
        resolved_parser = get_parser(format_hint="auto", raw_input=dialogue)
        self.assertIsInstance(resolved_parser, PlainTextParser)

        # 3. End-to-end canonical parsing
        canonical = parse_canonical(dialogue, format_hint="auto")
        self.assertEqual(canonical.source_format, SourceFormat.PLAIN_TEXT)
        self.assertEqual(len(canonical.messages), 3)
        self.assertEqual(len(canonical.get_evaluable_turns()), 3)

    def test_regression_can_parse_preserves_legitimate_csv(self):
        """Task 32 regression: ensure legitimate CSV/TSV headers remain correctly detected."""
        legit_cases = [
            "speaker,text,timestamp\nAlice,Hello,2026-11-10T14:30:00Z\nBob,Hi,2026-11-10T14:31:00Z",
            "Speaker,Utterance\nAlice,Good morning\nBob,Morning",
            "Sender,Message,Time\nAlice,Are we ready?,10:00\nBob,Yes,10:01",
            "author\tbody\tcreated_at\nAlice\tHello\t2026-11-10",
            "name;dialogue;timestamp\nAlice;Greetings;2026-11-10",
        ]
        for case in legit_cases:
            self.assertTrue(self.parser.can_parse(case), f"Expected can_parse=True for:\n{case[:50]}")

    def test_regression_warning_schema_contains_code_field(self):
        """Task 32 regression: verify ParserWarning outputs 'code' field for frontend consumption."""
        from src.data.schema import ParserWarning, WarningSeverity

        warning = ParserWarning(
            code="TEST_CODE",
            message="Test message",
            severity=WarningSeverity.WARNING,
            line_number=42,
        )
        w_dict = warning.to_dict()
        self.assertIn("code", w_dict)
        self.assertEqual(w_dict["code"], "TEST_CODE")
        self.assertEqual(w_dict["message"], "Test message")
        self.assertEqual(w_dict["line_number"], 42)


if __name__ == "__main__":
    unittest.main()
