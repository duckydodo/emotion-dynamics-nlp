# Task 18 — Canonical Conversation Layer and Basic Text Ingestion Report

**Repository:** `emotion-dynamics-nlp`  
**Working Directory:** `/home/chitta/Projects/emotion-dynamics-nlp`  
**Date:** November 10, 2026  
**Status:** Implemented & Verified (46/46 Tests Passing)  
**Task Reference:** Task 18 (Canonical Conversation Layer and Basic Text Import)  
**Deliverable File:** [`outputs/reports/task18_text_ingestion_report.md`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/reports/task18_text_ingestion_report.md)  

---

## 1. Executive Summary

Task 18 implements the **Canonical Conversation Layer** and **Plain-Text Ingestion Engine** designed and approved in Task 17 ([`outputs/reports/conversation_schema_and_parser_audit.md`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/reports/conversation_schema_and_parser_audit.md)). 

The primary objectives achieved in this task are:
1. **Canonical Schema Implementation:** Created [`src/data/schema.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/data/schema.py) defining strongly-typed dataclasses for `CanonicalConversation`, `CanonicalMessage`, `SpeakerProfile`, `ParserWarning`, and `ConversationTurn`.
2. **Robust Plain-Text Parser:** Implemented [`src/data/parsers/plain_text.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/data/parsers/plain_text.py) supporting pasted conversations and `.txt` files in `Speaker: utterance` format with full handling of consecutive same-speaker turns, multiline utterances, colons in text, Unicode/emojis, and non-fatal preservation of unparsed lines.
3. **API Integration:** Exposed a lightweight dedicated ingestion endpoint `POST /import` in [`src/api.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/api.py) and enriched `POST /analyze` with canonical conversation metadata without breaking existing clients.
4. **Adapter Pattern for Inference:** Guaranteed 100% backward compatibility with [`src/models/inference.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/models/inference.py) and [`src/analysis/conversation_analysis.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/analysis/conversation_analysis.py) via `get_evaluable_turns()`, which yields contiguous 0-indexed `ConversationTurn` instances.
5. **Rigorous Verification:** Built 13 new unit tests in [`tests/test_parsers.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/tests/test_parsers.py). All 46 tests across the entire test suite passed cleanly in the Docker execution environment.

---

## 2. Architecture & Implementation Summary

The ingestion subsystem was reorganized from a monolithic 111-line script into a modular architecture:

```
src/data/
├── schema.py                   # Canonical data structures, Enums, and validation
├── conversation.py             # Backward-compatible facade (re-exports ConversationTurn)
├── parsers/
│   ├── __init__.py             # Parser factory and parse_canonical() entrypoint
│   ├── base.py                 # Abstract BaseConversationParser
│   └── plain_text.py           # PlainTextParser for pasted text & .txt files
```

### 2.1 Component Overview

| File | Purpose | Lines of Code | Key Interfaces |
| :--- | :--- | :---: | :--- |
| [`src/data/schema.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/data/schema.py) | Core domain model & validation | 196 | `CanonicalConversation`, `CanonicalMessage`, `SpeakerProfile`, `ParserWarning`, `ConversationTurn` |
| [`src/data/parsers/base.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/data/parsers/base.py) | Abstract parser interface | 22 | `BaseConversationParser.can_parse()`, `.parse()` |
| [`src/data/parsers/plain_text.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/data/parsers/plain_text.py) | Plain text parser & heuristics | 240 | `PlainTextParser` |
| [`src/data/parsers/__init__.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/data/parsers/__init__.py) | Package factory | 30 | `get_parser()`, `parse_canonical()` |
| [`src/data/conversation.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/data/conversation.py) | Facade maintaining legacy API | 58 | `parse_conversation()`, `ConversationParseError` |
| [`src/application/analyzer.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/application/analyzer.py) | Application service integration | 125 | `ConversationAnalyzer.parse_canonical()`, enriched `analyze()` |
| [`src/api.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/api.py) | FastAPI routing layer | 115 | `POST /import`, `ImportRequest` |
| [`tests/test_parsers.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/tests/test_parsers.py) | Test suite for parsing & import | 275 | 13 test methods covering all edge cases |

---

## 3. Canonical Schema Specification

The canonical schema represents conversation state across import formats while maintaining complete decoupling from offline MELD training structures.

### 3.1 Field-by-Field Definition

#### `CanonicalMessage`
- `conversation_id: str` — Unique conversation identifier (`conv_<hex8>`).
- `message_id: str` — 1-indexed, zero-padded unique message identifier (`msg_0001`, `msg_0002`).
- `source_index: int` — 0-indexed position in raw input (preserves exact source appearance order).
- `sequence_index: int` — 0-indexed position in verified chronological sequence.
- `speaker: str` — Display name of speaker (e.g., `"Alice"` or `"System"` for unparsed items).
- `speaker_id: str` — Dynamically allocated normalized ID (`spk_0`, `spk_1`).
- `text: str` — Utterance content (clean string).
- `timestamp: Optional[datetime]` — ISO-8601 UTC timestamp (`None` for plain text).
- `message_type: MessageType` — Enum (`text`, `media`, `system`, `reaction`, `unparsed`).
- `metadata: Dict[str, Any]` — Format-specific metadata (e.g. line number, raw length).
- `is_evaluable: bool` — Boolean flag indicating whether the message contains speech suitable for emotion inference.

#### `CanonicalConversation`
- `conversation_id: str` — Identifier for conversation thread.
- `source_format: SourceFormat` — Ingestion format (`plain_text`, `instagram_json`, `whatsapp_txt`, `tabular_csv`).
- `title: Optional[str]` — Optional title/subject.
- `participants: List[SpeakerProfile]` — List of dynamically discovered speakers with turn counts.
- `messages: List[CanonicalMessage]` — Ordered list of canonical message records.
- `warnings: List[ParserWarning]` — Diagnostic warnings encountered during ingestion.
- `metadata: Dict[str, Any]` — Aggregate metadata (`has_timestamps`, `total_messages`, `evaluable_messages`).

### 3.2 Canonical JSON Representation (Realistic Example)

```json
{
  "conversation_id": "conv_9b3e1a04",
  "source_format": "plain_text",
  "title": null,
  "participants": [
    {
      "speaker_id": "spk_0",
      "name": "Alice",
      "normalized_name": "alice",
      "turn_count": 2
    },
    {
      "speaker_id": "spk_1",
      "name": "Bob",
      "normalized_name": "bob",
      "turn_count": 1
    }
  ],
  "messages": [
    {
      "conversation_id": "conv_9b3e1a04",
      "message_id": "msg_0001",
      "source_index": 0,
      "sequence_index": 0,
      "speaker": "Alice",
      "speaker_id": "spk_0",
      "text": "Hi Bob! Are we meeting at 10:30 AM?",
      "timestamp": null,
      "message_type": "text",
      "metadata": { "line_number": 1 },
      "is_evaluable": true
    },
    {
      "conversation_id": "conv_9b3e1a04",
      "message_id": "msg_0002",
      "source_index": 1,
      "sequence_index": 1,
      "speaker": "Bob",
      "speaker_id": "spk_1",
      "text": "Yes, join via https://meet.google.com/xyz: see you there! 🚀",
      "timestamp": null,
      "message_type": "text",
      "metadata": { "line_number": 2 },
      "is_evaluable": true
    },
    {
      "conversation_id": "conv_9b3e1a04",
      "message_id": "msg_0003",
      "source_index": 2,
      "sequence_index": 2,
      "speaker": "Alice",
      "speaker_id": "spk_0",
      "text": "Sounds perfect.",
      "timestamp": null,
      "message_type": "text",
      "metadata": { "line_number": 3 },
      "is_evaluable": true
    }
  ],
  "warnings": [
    {
      "code": "MISSING_TIMESTAMPS",
      "message": "Conversation does not contain timestamp metadata; ordering is based on input sequence.",
      "severity": "info",
      "line_number": null,
      "message_id": null
    }
  ],
  "metadata": {
    "has_timestamps": false,
    "total_messages": 3,
    "evaluable_messages": 3,
    "num_speakers": 2
  }
}
```

---

## 4. Plain-Text Parsing & Edge Case Handling

The `PlainTextParser` implements a robust parsing state machine that handles real-world text variations without discarding context:

### 4.1 Edge Case Solutions

1. **Consecutive Messages from Same Speaker:**
   - When a known speaker marker appears consecutively (e.g. `Alice: First message.\nAlice: Second message.`), the previous turn is immediately flushed and a distinct second turn is recorded with a new `message_id`.
   - Speaker turn count is properly incremented (`turn_count = 2`).

2. **Multiline Utterances:**
   - Lines that do not contain a valid speaker marker are appended to the active turn buffer `current_text_parts`.
   - Blank lines separate thoughts within an utterance without prematurely closing the turn.
   - Text parts are stripped and joined with single whitespace.

3. **Colons in Message Content:**
   - Only the first colon splits speaker from utterance (`line.split(":", 1)`).
   - Subsequent colons inside timestamps (`10:30:00`), URLs (`https://...`), or explanatory text (`Note: ...`) remain fully intact in `text`.

4. **False-Positive Speaker Prevention:**
   - Not all lines containing colons represent new speakers.
   - The parser checks `CONTINUATION_PREFIXES` (`note`, `ps`, `warning`, `tip`, `hint`, `e.g.`, `i.e.`, `ref`, `reason`, `step`, `time`, `date`). When inside an ongoing speaker turn, lines starting with these keywords are treated as continuation text rather than creating bogus speakers.
   - Lines matching URL protocol schemes (`http://`, `https://`, `ftp://`) are rejected as speaker turns.
   - Speaker names are constrained to reasonable lengths ($1 \le \text{len} \le 40$) and cannot contain sentence punctuation (`. `, `? `, `! `, `;`).

5. **Unicode and Non-English Preservation:**
   - Strings are parsed using standard Python 3 UTF-8 native decoding.
   - French accents, Japanese kanji/kana, Arabic script, and multi-byte emojis (e.g. `Ça va bien? ✨`, `こんにちは 🌸`, `مرحبا بكم 🚀`) are preserved verbatim with zero corruption.

6. **Unrecognized Lines / Headers:**
   - Non-empty lines preceding the first valid speaker (e.g. `--- Chat Export 2026-11-10 ---`) are captured as `MessageType.UNPARSED` with `is_evaluable = False` and an associated `ParserWarning(code="UNPARSED_LINE")`.
   - They are never silently discarded, allowing complete auditing in the canonical representation.

---

## 5. API Integration & Request/Response Contracts

### 5.1 Route Inventory

The application layer in [`src/api.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/api.py) exposes the following verified routes:

| Method | Path | Request Body | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | None | Serves web frontend ([`static/index.html`](file:///home/chitta/Projects/emotion-dynamics-nlp/static/index.html)) |
| `GET` | `/health` | None | Returns service status and available model modes |
| `POST` | `/import` | `ImportRequest` | Ingests, parses, and validates conversation without running inference |
| `POST` | `/predict` | `ConversationRequest` | Executes lightweight turn-by-turn emotion inference |
| `POST` | `/analyze` | `ConversationRequest` | Executes full emotion inference, dynamics analysis, and returns canonical summary |

### 5.2 `POST /import` Contract

#### Request
```json
{
  "conversation": "Alice: Hello!\nBob: Hi Alice!",
  "format": "plain_text"
}
```

#### Response (`HTTP 200 OK`)
```json
{
  "status": "warning",
  "conversation_id": "conv_a81f3d2e",
  "source_format": "plain_text",
  "normalized_conversation": { ... },
  "num_messages": 2,
  "num_evaluable_turns": 2,
  "num_speakers": 2,
  "speakers": ["Alice", "Bob"],
  "warnings": [
    {
      "code": "MISSING_TIMESTAMPS",
      "message": "Conversation does not contain timestamp metadata; ordering is based on input sequence.",
      "severity": "info",
      "line_number": null,
      "message_id": null
    }
  ],
  "validation_errors": [],
  "import_metadata": {
    "has_timestamps": false,
    "total_messages": 2,
    "evaluable_messages": 2,
    "num_speakers": 2
  }
}
```

- When the conversation input is empty, `POST /import` raises `HTTP 400 Bad Request` with detail `"Conversation is empty."`, consistent with `/analyze` and `/predict`.
- When input has text but no speakers or invalid structure, `POST /import` returns `HTTP 200 OK` with `status: "warning"` and detailed `validation_errors: [...]`.

---

## 6. Validation Framework & Diagnostic Rules

The validation method (`CanonicalConversation.validate()`) performs structural and semantic checks:

1. **Empty Conversation:** Flags conversations containing zero messages (`"Empty conversation: conversation contains no messages."`).
2. **Duplicate Message IDs:** Checks `seen_ids` set and flags any collision (`"Duplicate message_id '{msg_id}' at index {idx}."`).
3. **Missing Speaker:** Flags evaluable messages where `speaker.strip()` is empty (`"Message '{msg_id}' (index {idx}) is missing a speaker."`).
4. **Empty Utterance Text:** Flags evaluable messages where `text.strip()` is empty (`"Evaluable message '{msg_id}' from '{msg.speaker}' has empty text."`).
5. **Speaker Count:** Verifies that at least one distinct non-empty speaker exists (`"No valid speakers identified in evaluable conversation turns."`).

---

## 7. Test Results & Verification

All tests were executed within the Docker container `emotion-dynamics-nlp:week11` under Python 3.12:

```bash
docker run --rm -v $(pwd):/workspace -w /workspace -e PYTHONPATH=/workspace emotion-dynamics-nlp:week11 \
    python3 -m unittest discover -s tests
```

### 7.1 Test Execution Summary

```
Ran 46 tests in 2.158s
OK
```

### 7.2 Test Breakdown by Module

| Test Module | Tests | Status | Scope |
| :--- | :---: | :---: | :--- |
| [`tests/test_parsers.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/tests/test_parsers.py) | **13** | **PASSED** | Two-speaker, multi-speaker, multiline, colons in text, blank lines, Unicode/emojis, unrecognized lines, empty input, malformed lines, repeated speaker names, `/import` success, `/import` empty 400 error, `/import` unparsed validation. |
| [`tests/test_application_integration.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/tests/test_application_integration.py) | **8** | **PASSED** | `/health` endpoint, existing modes regression (`utterance`, `context_1`, `context_1_speaker`), DialogueRNN on `/analyze` and `/predict`, multi-speaker dynamic party tracking, state reset between requests, deterministic behavior, malformed speakers, empty conversation 400, frontend home. |
| [`tests/test_dialoguernn.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/tests/test_dialoguernn.py) | **13** | **PASSED** | Full DialogueRNN reference architecture, party states, causal recurrence, loss function. |
| [`tests/test_dialoguernn_ablations.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/tests/test_dialoguernn_ablations.py) | **12** | **PASSED** | Exact parameter counts and causal forward passes across all 4 Task 14/15 ablation configurations. |
| **Total** | **46** | **PASSED** | **100% Pass Rate** |

---

## 8. Compatibility Findings & Backward Compatibility Guarantees

1. **Inference Pipeline Compatibility:**
   - Both `EmotionInference` and `DialogueRNNInference` receive `ConversationTurn` items via `canonical.get_evaluable_turns()`.
   - Contiguous 0-indexing ($0, 1, \dots, K-1$) is strictly preserved, preventing indexing out-of-bounds in DistilBERT or party tensor construction in DialogueRNN.
2. **Emotion Dynamics Adjacency Preservation:**
   - In [`src/analysis/conversation_analysis.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/analysis/conversation_analysis.py), transitions are computed only between turns where `following["turn_id"] == current["turn_id"] + 1`.
   - Because `get_evaluable_turns()` assigns contiguous turn IDs to valid speech turns, dynamics metrics (shifts, persistence, turning points) operate accurately without phantom gaps caused by non-text records.
3. **MELD Dataset Isolation Boundary:**
   - Offline MELD training loaders ([`src/data/loader.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/data/loader.py)) and feature extraction scripts remain 100% decoupled and untouched.
4. **Frontend Non-Breaking Contract:**
   - Existing responses from `POST /analyze` retain `turns` and `analysis` keys expected by [`static/index.html`](file:///home/chitta/Projects/emotion-dynamics-nlp/static/index.html). The new `canonical` field is purely additive.

---

## 9. Known Limitations & Next Steps

1. **Missing Formats (Scheduled for Future Tasks):**
   - Instagram Direct Message JSON parser (`messages/inbox/.../message_1.json` with reverse-chronological sorting and Latin-1 mojibake fix) is planned for subsequent tasks.
   - WhatsApp chat export parser (`.txt` with timestamp regex matching) is planned for subsequent tasks.
   - Tabular CSV/TSV parser (dynamic header mapping) is planned for subsequent tasks.
2. **Time-Based Dynamics:**
   - Because plain-text inputs have no timestamps, dynamics calculations currently evaluate transitions strictly turn-by-turn rather than as a function of wall-clock time intervals.

---

## 10. Conclusion

Task 18 is complete. The canonical conversation layer is fully operational, thoroughly tested, and integrated into the FastAPI application with complete backward compatibility. No trained weights, Docker configurations, or frontend layouts were modified.
