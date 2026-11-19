# Task 20 — WhatsApp Text Export Ingestion Report

**Repository:** `emotion-dynamics-nlp`  
**Working Directory:** `/home/chitta/Projects/emotion-dynamics-nlp`  
**Date:** November 10, 2026  
**Status:** Implemented & Verified (77/77 Tests Passing)  
**Task Reference:** Task 20 (WhatsApp Text Export Import)  
**Deliverable File:** [`outputs/reports/task20_whatsapp_ingestion_report.md`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/reports/task20_whatsapp_ingestion_report.md)  

---

## 1. Executive Summary

Task 20 implements a **WhatsApp Plain-Text Chat Export Parser** ([`src/data/parsers/whatsapp.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/data/parsers/whatsapp.py)) that ingests exported `.txt` chat logs from Android, iOS, macOS, Windows, and Web clients and normalizes them into the approved canonical conversation schema ([`src/data/schema.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/data/schema.py)).

### Key Achievements
1. **Multi-Platform Pattern Recognition:** Supports both Android dash-separated format (`DD/MM/YYYY, HH:MM - Sender: text`) and iOS/macOS bracketed format (`[DD/MM/YYYY, HH:MM:SS] Sender: text`), including 12-hour (with `AM`/`PM`, `am`/`pm`) and 24-hour time variations.
2. **Adaptive Date Format Inference:** Avoids hardcoding a single date format. Analyzes day vs. month components across the dialogue to automatically distinguish `DD/MM/YYYY` (global WhatsApp standard), `MM/DD/YYYY` (US standard), and `YYYY-MM-DD` (ISO standard).
3. **Clean Handling of Hidden Unicode Formatting:** Strips invisible directional marks (Left-to-Right Mark `\u200e`, Right-to-Left Mark `\u200f`) and normalizes non-breaking spaces (`\u202f`, `\u00a0`) commonly inserted by WhatsApp on iOS without corrupting message text or emojis.
4. **Media Placeholder & Attachment Extraction:** Identifies media placeholders (`<Media omitted>`, `image omitted`, `sticker omitted`, `<attached: photo.jpg>`), extracting attachment filenames into metadata and flagging records as `MessageType.MEDIA` with `is_evaluable = False` so downstream models do not receive empty strings.
5. **System Notification Isolation:** Correctly separates system events (encryption notices, user additions/removals, group subject updates, missed calls, deleted messages) into `MessageType.SYSTEM` without misidentifying system phrases as senders.
6. **Full Test Suite Verification:** Built 16 new unit and integration tests in [`tests/test_whatsapp_parser.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/tests/test_whatsapp_parser.py). All 77 tests in the repository pass with 100% success in Docker.

---

## 2. Supported Export Formats & Syntax Patterns

WhatsApp exports vary depending on the operating system and locale settings:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   SUPPORTED WHATSAPP EXPORT PATTERNS                   │
│                                                                        │
│  Pattern 1: Android / Web / Windows (Hyphen-separated)                │
│  15/05/2026, 14:30 - Alice: Hey Bob, are you ready?                   │
│  15/05/2026, 2:32 pm - Bob: Leaving right now!                         │
│  15/05/2026, 14:35 - Messages and calls are end-to-end encrypted.     │
│                                                                        │
│  Pattern 2: iOS / macOS (Bracketed, often with \u200e and \u202f)     │
│  [15/05/2026, 14:32:15] Alice: Meeting starting soon.                  │
│  [15/05/2026, 2:32:15 PM] Bob: Joining now.                           │
│  [15/05/2026, 14:40:00] Bob: <attached: report.pdf> Look at page 3!   │
│                                                                        │
│  Multiline Continuation:                                               │
│  15/05/2026, 14:30 - Alice: Line one of message.                       │
│  Line two without timestamp continues Alice's utterance.              │
└────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Pattern Matching Specifications

| Pattern Variant | Regular Expression Anchor | Example Line |
| :--- | :--- | :--- |
| **Android Standard** | `^(\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp]\.?[Mm]\.?)?)\s*-\s*(.*)$` | `15/05/2026, 14:30 - Alice: Hello` |
| **iOS / macOS Bracketed** | `^\[(\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp]\.?[Mm]\.?)?)\]\s*(.*)$` | `[15/05/2026, 14:30:00] Alice: Hello` |
| **Hidden Direction Marks** | Strip `[\u200e\u200f\u202a-\u202e]` before regex evaluation | `\u200e[15/05/2026, 2:30:00 PM] ...` |

---

## 3. Timestamp Normalization & Date Convention Inference

WhatsApp does not explicitly declare the date format in export files. The parser employs a two-pass convention inference:

### 3.1 Date Convention Inference (`infer_date_convention`)
1. **First Pass:** Scans all timestamped lines in the file and extracts `(part1, part2, part3)`.
2. **Ambiguity Resolution:**
   - If `part1 > 12`: `part1` must be Day $\implies$ Format is **`DMY`** (`DD/MM/YYYY`).
   - If `part2 > 12`: `part2` must be Day $\implies$ Format is **`MDY`** (`MM/DD/YYYY`).
   - If `part1 > 1000`: Format is **`YMD`** (`YYYY-MM-DD`).
3. **Fallback Convention:**
   - If both `part1 \le 12` and `part2 \le 12` across the entire chat, the parser defaults to **`DMY`** (the standard outside the United States) and issues a `ParserWarning(code="AMBIGUOUS_DATE_FORMAT", message="Date numbers could be either Day/Month or Month/Day. Inferred 'DMY'.", severity=INFO)`.

### 3.2 Time Parsing & Normalization
- Normalizes narrow no-break space (`\u202f`) and non-breaking space (`\u00a0`) between time digits and `AM`/`PM`.
- Evaluates 24-hour time directly.
- For 12-hour time: adds 12 to hour for `PM` ($< 12$), sets hour to 0 for $12\text{ AM}$.
- Generates timezone-aware UTC `datetime`:
  $$\text{datetime}(\text{year}, \text{month}, \text{day}, \text{hour}, \text{minute}, \text{second}, \text{tzinfo}=\text{timezone.utc})$$

---

## 4. Message Classification & Content Extraction

### 4.1 Sender Identification
- In a valid user message, the remainder after the timestamp begins with `Speaker: `.
- The parser splits on the first colon (`remainder.split(":", 1)`).
- Senders with non-alphanumeric characters are supported:
  - Phone numbers: `+1 (555) 123-4567: Hello!`
  - Titles & Parentheses: `Dr. Smith (Cardiology): Call me.`
- Senders are dynamically mapped to canonical `SpeakerProfile` (`spk_0`, `spk_1`).

### 4.2 Distinguishing User Messages from System Notifications
Lines without colons after the timestamp or containing system-announcement phrases are classified as `MessageType.SYSTEM`:
- Phrased indicators: `"messages and calls are end-to-end encrypted"`, `"added"`, `"removed"`, `"left"`, `"changed the group"`, `"changed the subject"`, `"deleted this message"`, `"missed voice call"`, `"missed video call"`, `"security code changed"`.
- System messages receive `speaker = "System"`, `speaker_id = "spk_system"`, and `is_evaluable = False`.

### 4.3 Media Placeholders & Caption Extraction
WhatsApp replaces attachments with text placeholders:

| Exported Text | Canonical Message Type | `text` Field | `is_evaluable` | Stored Metadata |
| :--- | :---: | :---: | :---: | :--- |
| `<Media omitted>` | `MessageType.MEDIA` | `""` | `False` | `{"media_placeholder": "<Media omitted>"}` |
| `image omitted`, `sticker omitted` | `MessageType.MEDIA` | `""` | `False` | `{"media_placeholder": "image omitted"}` |
| `<attached: photo.jpg>` | `MessageType.MEDIA` | `""` | `False` | `{"attachment": "photo.jpg"}` |
| `<attached: photo.jpg> Look at this!` | `MessageType.TEXT` | `"Look at this!"` | `True` | `{"attachment": "photo.jpg"}` |

By setting `text = ""` and `is_evaluable = False` for pure media placeholders, the downstream emotion models (DistilBERT / DialogueRNN) are protected from tokenizing non-speech strings like `<Media omitted>`.

---

## 5. API Auto-Detection & Routing

The parser is integrated into [`src/data/parsers/__init__.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/data/parsers/__init__.py):

```python
def get_parser(format_hint="auto", raw_input=None):
    if format_hint in ("whatsapp", "whatsapp_txt", "wa"):
        return _WHATSAPP_PARSER
    ...
    if raw_input is not None:
        if _INSTAGRAM_PARSER.can_parse(raw_input):
            return _INSTAGRAM_PARSER
        if _WHATSAPP_PARSER.can_parse(raw_input):
            return _WHATSAPP_PARSER
    return _PLAIN_TEXT_PARSER
```

- **`POST /import`:** Accepts WhatsApp text directly. Auto-detection identifies WhatsApp timestamp patterns and returns normalized canonical JSON, discovered speakers, evaluable turn counts, warnings, and metadata.
- **`POST /analyze`:** Accepts WhatsApp text directly. Evaluates valid speech turns through DistilBERT or DialogueRNN, calculates dynamics metrics, and attaches canonical conversation metadata.

---

## 6. Test Suite Execution & Verification

Tests were executed inside the project container `emotion-dynamics-nlp:week11`:
```bash
docker run --rm -v $(pwd):/workspace -w /workspace -e PYTHONPATH=/workspace emotion-dynamics-nlp:week11 \
    python3 -m unittest discover -s tests
```

### 6.1 Results
```
Ran 77 tests in 2.341s
OK
```

### 6.2 Test Inventory by Suite

| Test Suite File | Tests | Result | Coverage |
| :--- | :---: | :---: | :--- |
| [`tests/test_whatsapp_parser.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/tests/test_whatsapp_parser.py) | **16** | **PASSED** | Android 24h, Android 12h AM/PM, iOS bracketed, iOS with LRM/narrow space, multiline messages, colons in text, system notifications, media omitted, `<attached: file>` with caption, phone number senders, Unicode/emojis, date convention inference (MDY vs DMY), unparsed lines before chat, `/import` auto-detection, `/analyze` DistilBERT mode, `/analyze` DialogueRNN mode. |
| [`tests/test_instagram_parser.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/tests/test_instagram_parser.py) | **15** | **PASSED** | Instagram JSON exports, mojibake repair, media, reactions, system events, auto-detection, inference. |
| [`tests/test_parsers.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/tests/test_parsers.py) | **13** | **PASSED** | Plain text multi-speaker, multiline, colons, Unicode, unrecognized lines, empty input, repeated speakers, `/import` validation. |
| [`tests/test_application_integration.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/tests/test_application_integration.py) | **8** | **PASSED** | Core application integration, `/health`, regression testing across `utterance`, `context_1`, `context_1_speaker`, and `dialoguernn`. |
| [`tests/test_dialoguernn.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/tests/test_dialoguernn.py) | **13** | **PASSED** | Reference DialogueRNN architecture, causal recurrence, loss function. |
| [`tests/test_dialoguernn_ablations.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/tests/test_dialoguernn_ablations.py) | **12** | **PASSED** | DialogueRNN ablation architectures (all 4 Task 14/15 configurations). |
| **Total** | **77** | **PASSED** | **100% Pass Rate** |

---

## 7. Compatibility Findings & Guarantees

1. **Schema Integrity:** The canonical schema was unaltered (`CanonicalConversation`, `CanonicalMessage`, `MessageType`, `SourceFormat.WHATSAPP_TXT` used as specified).
2. **Inference Decoupling:** `get_evaluable_turns()` guarantees contiguous 0-indexed `turn_id` values for valid speech turns, keeping DistilBERT tokenization, DialogueRNN party tensors, and dynamics adjacency checks 100% stable.
3. **MELD Dataset Isolation:** Training scripts in `scripts/` and dataset loaders in `src/data/loader.py` remain untouched and isolated from runtime parsers.
4. **No Destructive Operations:** No files were deleted, no checkpoints modified, and no commits or pushes performed.

---

## 8. Known Limitations

1. **Non-Standard Localized System Messages:**
   - WhatsApp system messages translated into languages other than English (e.g. German `"Nachrichten und Anrufe sind Ende-zu-Ende-verschlüsselt"`) will be classified as `MessageType.SYSTEM` if they lack a speaker colon, but localized phrase heuristics can be expanded for complete multi-lingual system filtering.
2. **Omitted Voice Notes Audio:**
   - Voice note indicators (`audio omitted` or `<attached: PTT-*.opus>`) are flagged as media records. Transcribing spoken content from audio files would require integration of an ASR model.

---

## 9. Conclusion

Task 20 is complete. The WhatsApp plain-text conversation ingestion engine is fully operational, thoroughly tested, and integrated into the FastAPI application with auto-detection. All 77 unit and integration tests are passing.
