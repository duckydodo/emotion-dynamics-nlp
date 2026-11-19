# Task 21 — Structured CSV Conversation Ingestion Report

**Repository:** `emotion-dynamics-nlp`  
**Working Directory:** `/home/chitta/Projects/emotion-dynamics-nlp`  
**Date:** November 10, 2026  
**Status:** Implemented & Verified (93/93 Tests Passing)  
**Task Reference:** Task 21 (Structured CSV Conversation Import)  
**Deliverable File:** [`outputs/reports/task21_csv_ingestion_report.md`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/reports/task21_csv_ingestion_report.md)  

---

## 1. Executive Summary

Task 21 implements a **Structured Tabular CSV/TSV Conversation Parser** ([`src/data/parsers/csv_parser.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/data/parsers/csv_parser.py)) that converts tabular dialogue exports into the canonical conversation schema ([`src/data/schema.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/data/schema.py)).

### Key Achievements
1. **Configurable & Auto-Detected Column Mapping:** Supports both automatic alias matching and explicit user-specified column mapping (`column_mapping`) for logical fields (`speaker`, `text`, `timestamp`, `message_id`). If required fields (`speaker`, `text`) cannot be determined with confidence, the parser refuses to guess unrelated columns and raises a descriptive `COLUMN_MAPPING_REQUIRED` warning.
2. **Robust RFC 4180 Dialect & Delimiter Handling:** Automatically identifies delimiters (comma `,`, tab `\t`, semicolon `;`, pipe `|`) and handles quoted fields with embedded commas and embedded newlines without corrupting row alignment.
3. **Encoding & BOM Safety:** Transparently handles UTF-8 Byte Order Marks (`\ufeff`) without corrupting column 0 headers.
4. **Validation & Non-Destructive Rejection Reporting:** Tracks and reports granular row statistics: `rows_imported`, `rows_rejected`, `missing_speakers`, `empty_text_rows`, `invalid_timestamps`, and `duplicate_ids`. Rejected rows are retained as non-evaluable messages rather than being silently dropped.
5. **Full Suite Verification:** Added 16 new unit and integration tests in [`tests/test_csv_parser.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/tests/test_csv_parser.py). All 93 tests across the entire repository test suite passed with 100% success in Docker.

---

## 2. Supported CSV Contract & Logical Mapping

The CSV parser binds tabular data to the canonical schema through four logical fields:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        SUPPORTED CSV CONTRACT                          │
│                                                                        │
│  Logical Field    Requirement   Sample Aliases                         │
│  ────────────────────────────────────────────────────────────────────  │
│  speaker          REQUIRED      speaker, sender, character, user, from │
│  text             REQUIRED      text, utterance, message, content, line│
│  timestamp        OPTIONAL      timestamp, time, date, created_at      │
│  message_id       OPTIONAL      id, msg_id, turn_id, utterance_id      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
       ┌────────────────────────────┴────────────────────────────┐
       ▼                                                         ▼
Case A: Automatic Matching                      Case B: Explicit User Mapping
Heuristic normalized matching                   POST /import with:
(e.g., "Utterance" -> "text")                   "column_mapping": {
                                                  "speaker": "char_name",
                                                  "text": "dialogue_text"
                                                }
```

### 2.1 Auto-Detection Heuristic

When no explicit mapping is provided, the parser matches case-insensitive, punctuation-stripped header names against canonical alias dictionaries:

| Logical Field | Priority Aliases | Normalized Match Examples |
| :--- | :--- | :--- |
| **`speaker`** | `["speaker", "sender", "author", "user", "from", "character", "name", "person", "party"]` | `Speaker`, `sender_name`, `From`, `character` |
| **`text`** | `["text", "utterance", "message", "content", "body", "dialogue", "line", "sentence"]` | `Utterance`, `text_content`, `Message`, `dialogue` |
| **`timestamp`** | `["timestamp", "time", "date", "datetime", "created_at", "timestamp_ms", "starttime"]` | `CreatedAt`, `start_time`, `Timestamp_ms`, `date` |
| **`message_id`** | `["message_id", "msg_id", "id", "turn_id", "utterance_id", "index", "sr no."]` | `Utterance_ID`, `Sr No.`, `msg_id`, `turn_id` |

### 2.2 Safety Guarantee Against Unrelated CSV Data
If neither `speaker` nor `text` can be determined from the headers, the parser **does not guess or pick random columns**. It emits:
- `ParserWarning(code="COLUMN_MAPPING_REQUIRED", message="Required column(s) ['speaker', 'text'] could not be determined. Available columns: [...]. Please provide an explicit column_mapping.", severity=ERROR)`
- Conversation metadata with `"error": "Column mapping required"`.
- Zero imported speech turns, preventing corrupt data from entering the emotion model.

---

## 3. Delimiter Detection & RFC 4180 Handling

1. **Delimiter Detection (`detect_delimiter`):**
   - Counts delimiter candidates (`,`, `\t`, `;`, `|`) in the header line while strictly ignoring occurrences inside double quotes (`"`).
   - Selects the dominant delimiter outside quotes.
2. **Quoted Embedded Commas:**
   - Evaluates standard RFC 4180 quotes:
     `Bob,"Morning Alice, how are you today?"`
     Correctly parses as 2 columns: speaker `Bob`, text `Morning Alice, how are you today?`.
3. **Quoted Embedded Newlines (Multiline Fields):**
   - Handles text spanning multiple lines inside quotes:
     `Alice,"Line one.\nLine two.\nLine three."`
     Preserves the multiline utterance intact within `CanonicalMessage.text`.
4. **UTF-8 Byte Order Mark (BOM):**
   - Automatically detects and strips leading `\ufeff`. Prevents the first column from being misinterpreted as `\ufeffspeaker`.

---

## 4. Timestamp & ID Processing

### 4.1 Timestamp Extraction & Normalization
The parser interprets diverse timestamp formats into timezone-aware UTC `datetime`:
1. **Numeric Epoch Timestamps:** Supports milliseconds ($> 10^{10}$) and seconds ($> 946{,}684{,}800$).
2. **ISO-8601 Strings:** Parses `YYYY-MM-DDTHH:MM:SSZ` or with timezone offset.
3. **Standard Date/Time Strings:** Supports `YYYY-MM-DD HH:MM:SS`, `DD/MM/YYYY HH:MM:SS`, `MM/DD/YYYY HH:MM:SS`.
4. **MELD Time Format:** Supports millisecond comma format `HH:MM:SS,mmm` (e.g. `00:20:57,256`).
5. **Invalid Timestamps:** If timestamp parsing fails, the turn is retained, `timestamp` is set to `None`, and an `INVALID_TIMESTAMP` warning is logged without rejecting the speech content.

### 4.2 Message ID & Duplicate Handling
- If a column is mapped to `message_id`, its value is checked against a `seen_ids` set.
- If a collision occurs (duplicate ID), an informative warning `DUPLICATE_MESSAGE_ID` is recorded and a unique local ID (`{id}_dup_{row_idx}`) is assigned to avoid schema validation errors.
- If no ID column is present, standard zero-padded sequential IDs (`msg_0001`, `msg_0002`) are automatically generated.

---

## 5. Row Validation & Diagnostics

The parser tracks granular row-level processing metrics:

| Metric | Condition | Pipeline Outcome |
| :--- | :--- | :--- |
| `rows_imported` | Valid row with non-empty speaker and text | Stored as `CanonicalMessage(is_evaluable=True)` |
| `rows_rejected` | Missing speaker | Stored as `CanonicalMessage(message_type=UNPARSED, is_evaluable=False)` with `MISSING_SPEAKER` warning |
| `missing_speakers` | Column cell is empty | Warning logged; row rejected from inference |
| `empty_text_rows` | Speaker exists, text is empty string | Stored with `is_evaluable=False` with `EMPTY_TEXT` warning |
| `invalid_timestamps` | Value present in timestamp column, unparseable | Stored with `timestamp=None` with `INVALID_TIMESTAMP` warning |
| `duplicate_ids` | Value in ID column appears more than once | Assigned local suffixed unique ID with warning |

---

## 6. API Integration

### 6.1 `POST /import` Integration
The `POST /import` endpoint in [`src/api.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/api.py) supports CSV imports:
- **Auto-Detection (`format: "auto"`):** Header containing delimiter and speaker/text keywords automatically routes to `TabularCsvParser`.
- **Explicit Mapping:** Accepts optional `column_mapping: Optional[Dict[str, str]]`:
  ```json
  {
    "conversation": "char_name,dialogue_text\nAlice,Hello custom mapping\nBob,Confirmed\n",
    "column_mapping": {
      "speaker": "char_name",
      "text": "dialogue_text"
    }
  }
  ```
- **Response:** Returns `source_format: "tabular_csv"`, `normalized_conversation`, discovered speakers, message counts, warnings, validation errors, and metadata including row statistics (`rows_imported`, `rows_rejected`, `delimiter`).

### 6.2 `POST /analyze` Integration
CSV strings can be submitted directly to `POST /analyze`. Evaluable rows are extracted, passed to the selected model (DistilBERT or DialogueRNN), and analyzed for conversational emotion dynamics.

---

## 7. Test Execution & Verification

Tests were executed inside the project container `emotion-dynamics-nlp:week11`:
```bash
docker run --rm -v $(pwd):/workspace -w /workspace -e PYTHONPATH=/workspace emotion-dynamics-nlp:week11 \
    python3 -m unittest discover -s tests
```

### 7.1 Results
```
Ran 93 tests in 2.289s
OK
```

### 7.2 Full Repository Test Breakdown

| Test Suite File | Tests | Result | Coverage |
| :--- | :---: | :---: | :--- |
| [`tests/test_csv_parser.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/tests/test_csv_parser.py) | **16** | **PASSED** | Standard CSV, alternative column names auto-matching, explicit `column_mapping`, quoted embedded commas, multiline quoted fields, UTF-8 BOM, missing required columns, empty rows, invalid timestamps, duplicate IDs, TSV/semicolon delimiters, missing speaker rejection, MELD format CSV import, `/import` auto-detection, `/import` with custom mapping, `/analyze` end-to-end DialogueRNN. |
| [`tests/test_whatsapp_parser.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/tests/test_whatsapp_parser.py) | **16** | **PASSED** | Android/iOS formats, 12h/24h timestamps, multiline messages, system notifications, media placeholders, auto-detection, API. |
| [`tests/test_instagram_parser.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/tests/test_instagram_parser.py) | **15** | **PASSED** | Meta JSON export structures, mojibake repair, media, reactions, system events, auto-detection, API. |
| [`tests/test_parsers.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/tests/test_parsers.py) | **13** | **PASSED** | Plain text multi-speaker, multiline, colons, Unicode, unrecognized lines, empty input, repeated speakers, `/import` validation. |
| [`tests/test_application_integration.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/tests/test_application_integration.py) | **8** | **PASSED** | Core application integration, `/health`, regression testing across `utterance`, `context_1`, `context_1_speaker`, and `dialoguernn`. |
| [`tests/test_dialoguernn.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/tests/test_dialoguernn.py) | **13** | **PASSED** | Reference DialogueRNN architecture, party states, causal recurrence. |
| [`tests/test_dialoguernn_ablations.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/tests/test_dialoguernn_ablations.py) | **12** | **PASSED** | DialogueRNN ablation architectures (all 4 Task 14/15 configurations). |
| **Total** | **93** | **PASSED** | **100% Pass Rate** |

---

## 8. Compatibility Findings & Guarantees

1. **MELD Dataset Compatibility:** Direct test confirmation that MELD CSV format (`Dialogue_ID`, `Utterance_ID`, `Speaker`, `Utterance`, `StartTime`) parses cleanly with zero manual configuration.
2. **Inference Pipeline Stability:** Only valid turns with non-empty text and speakers are returned by `get_evaluable_turns()`, with contiguous 0-indexed `turn_id` values.
3. **Preservation of Non-Destructive Principles:** No model weights, configurations, training scripts, or frontend files were modified.

---

## 9. Conclusion

Task 21 is complete. The structured CSV conversation ingestion parser is fully operational, thoroughly tested, and integrated into the application API. The platform now supports all four primary ingestion formats: **Plain Text**, **Instagram Direct Message JSON**, **WhatsApp Chat Exports**, and **Structured CSV/TSV**.
