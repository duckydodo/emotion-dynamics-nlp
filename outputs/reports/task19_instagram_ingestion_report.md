# Task 19 — Instagram JSON Conversation Ingestion Report

**Repository:** `emotion-dynamics-nlp`  
**Working Directory:** `/home/chitta/Projects/emotion-dynamics-nlp`  
**Date:** November 10, 2026  
**Status:** Implemented & Verified (61/61 Tests Passing)  
**Task Reference:** Task 19 (Instagram JSON Conversation Import)  
**Deliverable File:** [`outputs/reports/task19_instagram_ingestion_report.md`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/reports/task19_instagram_ingestion_report.md)  

---

## 1. Executive Summary

Task 19 implements an **Instagram Direct Message JSON Parser** ([`src/data/parsers/instagram.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/data/parsers/instagram.py)) that ingests Meta "Download Your Information" exports (`message_1.json`) and converts them into the canonical conversation schema ([`src/data/schema.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/data/schema.py)).

### Key Achievements
1. **Multi-Variant Instagram Export Parsing:** Supports standard and legacy Meta message export structures, extracting participants (objects or strings), senders, text content, multi-type media attachments (photos, videos, audio, stickers, shares), reactions, and system events.
2. **Chronological Inversion & Equal-Timestamp Tie-Breaking:** Correctly identifies that Meta exports messages in reverse chronological order (newest first). Reorders them into ascending chronological sequence using normalized UTC timestamps, with stable inverted source-order tie-breaking for equal or missing timestamps.
3. **Adaptive Encoding & Mojibake Repair:** Accurately diagnoses and repairs legacy Meta export encoding quirks where UTF-8 bytes were exported as Latin-1 string representations (e.g., `\u00e2\u0080\u0099` $\to$ `’` and `\u00f0\u009f...` $\to$ emojis), while leaving clean UTF-8 text untouched.
4. **Unified API Ingestion & Auto-Detection:** Fully integrated into `POST /import` and `POST /analyze` in [`src/api.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/api.py). Incoming payloads with `"messages"` are auto-detected and parsed without needing a second API endpoint.
5. **Full Suite Verification:** Added 15 new unit and integration tests in [`tests/test_instagram_parser.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/tests/test_instagram_parser.py). All 61 tests across the entire repository test suite passed with 100% success in Docker.

---

## 2. Supported Export Structures

Meta's export format has evolved across different versions of "Download Your Information" (JSON format). The parser accommodates both modern and legacy export variants:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   SUPPORTED INSTAGRAM EXPORT SCHEMA                    │
│                                                                        │
│  Top-level Object:                                                     │
│  ├── participants: [ {"name": "Alice"}, ... ] OR [ "Alice", ... ]      │
│  ├── title: str (Thread / participant display name)                    │
│  ├── is_still_participant: bool                                        │
│  ├── thread_path: str (e.g., "inbox/alice_123")                        │
│  └── messages: [                                                       │
│        {                                                               │
│          "sender_name": "Alice",                                       │
│          "timestamp_ms": 1700000020000,                                │
│          "content": "Hey! Check this out:",                            │
│          "photos": [ {"uri": "messages/inbox/.../photo.jpg"} ],       │
│          "reactions": [ {"reaction": "❤️", "actor": "Bob"} ],          │
│          "type": "Generic",                                            │
│          "is_geoblocked_for_viewer": false                             │
│        }, ...                                                          │
│      ]                                                                 │
└────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Variant Differences Handled

| Field | Primary Specification | Detected Variations & Fallback Strategy |
| :--- | :--- | :--- |
| **Participants** | List of dicts: `[{"name": "Alice"}]` | Also accepts flat string lists `["Alice", "Bob"]`. In addition, all distinct `sender_name` values across messages are dynamically aggregated so participants are never omitted. |
| **Sender** | `"sender_name"` | Checks `"sender_name"`, falls back to `"sender"`, then `"actor"`. Defaults to `"Unknown"` with a `MISSING_SENDER` warning if omitted. |
| **Timestamp** | `"timestamp_ms"` (integer milliseconds) | Supports `"timestamp_ms"`, `"timestamp"` (detects whether milliseconds or seconds based on integer magnitude $> 10^{10}$), and ISO-8601 string dates (`"created_at"`). |
| **Content** | `"content"` (string) | Checks `"content"`, falls back to `"text"`. If absent or null, checked for media attachments. |
| **Attachments** | Multiple dedicated arrays | Identifies `"photos"`, `"videos"`, `"audio_files"`, `"files"`, `"sticker"`, and `"share"`. Extracted into message metadata. |
| **Reactions** | `"reactions": [{"reaction": "...", "actor": "..."}]` | Extracted into metadata; emoji reactions undergo Latin-1 mojibake repair. |
| **System Events** | `"type": "Call"`, `"type": "Generic"` | System events without text or media are classified as `MessageType.SYSTEM` with `is_evaluable = False`. |

---

## 3. Timestamp Normalization & Ordering Rules

Instagram exports present two temporal challenges: timestamps in milliseconds and reverse-chronological ordering.

### 3.1 Normalization Process
1. **Unit Interpretation:**
   - If timestamp $\ge 946{,}684{,}800{,}000$ (Jan 1, 2000 in ms), it is treated as milliseconds:
     $$\text{epoch\_seconds} = \frac{\text{timestamp\_ms}}{1000.0}$$
   - If timestamp $< 10{,}000{,}000{,}000$, it is interpreted directly as seconds.
   - Converted to timezone-aware UTC `datetime`: `datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)`.
2. **Original Value Preservation:**
   - Raw integer timestamps and detected units (`"milliseconds"`, `"seconds"`, `"iso"`) are preserved in `CanonicalMessage.metadata`.

### 3.2 Chronological Inversion & Equal Timestamp Tie-Breaking
- **Export Direction:** Meta exports the `messages` array starting from the most recent turn (`messages[0]`) back to the oldest turn (`messages[N-1]`).
- **Sorting Algorithm:**
  - When timestamps are present, messages are sorted into ascending chronological order using:
    $$\text{sort\_key}(m) = (\text{timestamp},\; N - 1 - m.\text{source\_index})$$
  - The tie-breaker $N - 1 - m.\text{source\_index}$ inverts the array order: for two messages sent within the same second/millisecond, the one that appeared later in the export array was sent earlier in conversation time.
  - When no timestamps are present in the entire payload, the array is inverted directly, and a `MISSING_TIMESTAMPS_INVERTED` warning is generated.
- After sorting, `sequence_index` is reassigned sequentially ($0, 1, \dots, N-1$) and `message_id` is updated to reflect chronological order (`msg_0001`, `msg_0002`).

---

## 4. Encoding & Latin-1 Mojibake Handling

### 4.1 Root Cause of Mojibake
Meta's export pipeline has historically suffered from an encoding defect: UTF-8 encoded text was mistakenly decoded as Latin-1 (`iso-8859-1`) before escaping as JSON.
For example:
- Apostrophe `’` (UTF-8 bytes `\xe2\x80\x99`) was exported as the three Latin-1 characters `â\x80\x99` (JSON `\u00e2\u0080\u0099`).
- Party popper emoji 🎉 (UTF-8 bytes `\xf0\x9f\x8e\x89`) was exported as `\u00f0\u009f\u008e\u0089`.

### 4.2 Non-Destructive Repair Strategy
Rather than indiscriminately forcing Latin-1 decoding across all inputs, [`src/data/parsers/instagram.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/data/parsers/instagram.py) implements an adaptive strategy (`attempt_mojibake_repair`):
1. **ASCII Skip:** If all characters in a string have ordinal $< 128$, no decoding is required.
2. **Safe Re-encoding Check:**
   ```python
   try:
       repaired = text.encode("latin1").decode("utf-8")
       if repaired != text:
           return repaired, True
       return text, False
   except (UnicodeEncodeError, UnicodeDecodeError):
       return text, False
   ```
3. **Safety Guarantee:** If the input string is already valid clean UTF-8 (e.g. Japanese text `こんにちは` or clean Unicode emojis `🌍`), encoding to Latin-1 raises `UnicodeEncodeError`. The exception is caught, and the text is preserved untouched.
4. **Diagnostic Reporting:** The conversation metadata records the detected encoding strategy (`"encoding_strategy": "latin1_mojibake_repair"` vs `"utf8_direct"`) and reports the total repaired count via a `ParserWarning(code="MOJIBAKE_REPAIRED")`.

---

## 5. Message Representation & Downstream Inference Safety

A major risk identified in Task 17 was that empty text from media attachments or reactions could cause DistilBERT and DialogueRNN to crash or produce spurious neutral logits.

### 5.1 Message Type Classification

| Message Category | Conditions in Export | Canonical `message_type` | `text` Value | `is_evaluable` | Downstream Handling |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Standard Utterance** | `"content"` is non-empty string | `MessageType.TEXT` | Repaired text | `True` | Dispatched to DistilBERT / DialogueRNN |
| **Photo / Video / Sticker** | Media keys present, `"content"` absent/empty | `MessageType.MEDIA` | `""` (empty) | `False` | Excluded from emotion models; metadata retained |
| **Caption + Photo** | Both text and media keys present | `MessageType.TEXT` | Repaired text | `True` | Dispatched to emotion models; media in metadata |
| **System Event** | `"type": "Call"`, call duration, etc. | `MessageType.SYSTEM` | `""` (empty) | `False` | Excluded from emotion models |
| **Pure Reaction** | Only `"reactions"` present without text | `MessageType.REACTION` | `""` (empty) | `False` | Excluded from emotion models |

### 5.2 Contiguous Turn Re-Indexing
`CanonicalConversation.get_evaluable_turns()` filters exclusively for `msg.is_evaluable and msg.text.strip()`. It re-indexes evaluable turns with contiguous 0-indexed integers ($0, 1, \dots, K-1$). 

This ensures that:
1. `EmotionInference` tokenizes only actual spoken/written utterances.
2. `DialogueRNNInference` builds dynamic speaker party vectors only across valid dialogue steps.
3. `analyze_predictions` strict adjacency verification (`following["turn_id"] == current["turn_id"] + 1`) executes cleanly across real conversational turns.

---

## 6. Validation Framework & Warning Diagnostics

The parser detects irregularities and records structured `ParserWarning` instances:

| Warning Code | Severity | Trigger Condition | Pipeline Action |
| :--- | :---: | :--- | :--- |
| `MALFORMED_JSON` | `ERROR` | Input string is invalid JSON | Returns empty conversation container with error metadata |
| `MISSING_MESSAGE_ARRAY` | `ERROR` | Missing top-level `"messages"` list | Returns empty conversation container with error metadata |
| `UNSUPPORTED_STRUCTURE` | `ERROR` | Payload is neither string nor dict | Returns empty conversation container with error metadata |
| `MISSING_SENDER` | `WARNING` | Message has no sender field | Assigns speaker to `"Unknown"` (`spk_unknown`) |
| `MISSING_TIMESTAMP` | `WARNING` | Message has no parseable timestamp | Sets `timestamp = None`; preserves inverted source position |
| `EMPTY_MESSAGE` | `INFO` | Item has neither text nor attachments | Retained in log with `is_evaluable = False` |
| `DUPLICATE_RECORD` | `INFO` | Duplicate `(sender, timestamp, text)` tuple | Recorded in warnings; original preserved |
| `MOJIBAKE_REPAIRED` | `INFO` | Latin-1 escaped UTF-8 bytes decoded | Text repaired; total repair count reported |
| `CHRONOLOGICAL_SORT_APPLIED` | `INFO` | Reverse-order messages reordered | Notifies client of chronological sequence adjustment |

---

## 7. Unified API Routing & Auto-Detection

The parser is integrated into the application layer:
1. **Parser Factory ([`src/data/parsers/__init__.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/data/parsers/__init__.py)):**
   - Automatically inspects the raw payload. If the input is a JSON string or dictionary containing `"messages"`, it routes to `InstagramJsonParser`.
   - Otherwise, falls back to `PlainTextParser`.
2. **`POST /import` Endpoint:**
   - Setting `format: "auto"` (the default) seamlessly parses both Instagram exports and plain text files.
   - Returns normalized canonical JSON, speaker list, message counts, evaluable turn counts, warnings, and validation errors.
3. **`POST /analyze` Endpoint:**
   - Accepts Instagram JSON export directly. Automatically parses, extracts evaluable speech turns, runs DistilBERT or DialogueRNN inference, computes dynamics metrics, and attaches canonical metadata.

---

## 8. Test Execution & Verification

Tests were run in the project container `emotion-dynamics-nlp:week11`:
```bash
docker run --rm -v $(pwd):/workspace -w /workspace -e PYTHONPATH=/workspace emotion-dynamics-nlp:week11 \
    python3 -m unittest discover -s tests
```

### 8.1 Results
```
Ran 61 tests in 2.730s
OK
```

### 8.2 Breakdown of All Test Modules

| Test Module | Tests | Result | Coverage |
| :--- | :---: | :---: | :--- |
| [`tests/test_instagram_parser.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/tests/test_instagram_parser.py) | **15** | **PASSED** | Standard message, multiple participants, consecutive messages, missing timestamp, attachment-only photo message, system event, reactions, malformed JSON, unexpected structure, Latin-1 mojibake vs clean UTF-8, chronological ordering & equal timestamp tie-break, missing sender fallback, `/import` auto-detection, `/analyze` DistilBERT mode, `/analyze` DialogueRNN mode. |
| [`tests/test_parsers.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/tests/test_parsers.py) | **13** | **PASSED** | Plain text multi-speaker, multiline, colons, Unicode, unrecognized lines, empty input, repeated speakers, `/import` validation. |
| [`tests/test_application_integration.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/tests/test_application_integration.py) | **8** | **PASSED** | `/health`, existing inference modes regression (`utterance`, `context_1`, `context_1_speaker`), DialogueRNN integration, party state resets, deterministic checks. |
| [`tests/test_dialoguernn.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/tests/test_dialoguernn.py) | **13** | **PASSED** | DialogueRNN reference architecture, party states, causal recurrence. |
| [`tests/test_dialoguernn_ablations.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/tests/test_dialoguernn_ablations.py) | **12** | **PASSED** | Exact parameter counts & causal passes for all 4 ablation models. |
| **Total** | **61** | **PASSED** | **100% Pass Rate** |

---

## 9. Known Limitations

1. **ZIP Archive Direct Ingestion:**
   - Meta exports often ship as a `.zip` archive containing media directories and `message_1.json`, `message_2.json`. The current parser accepts the unzipped JSON payload directly; automated multi-file ZIP extraction can be added in a future enhancement.
2. **Audio Voice Notes Transcription:**
   - Audio files in `"audio_files"` are preserved in canonical message metadata with `is_evaluable = False`. Automatic speech-to-text (e.g. Whisper) would be required to extract spoken text from voice messages.

---

## 10. Conclusion

Task 19 is complete. The Instagram JSON conversation ingestion engine is fully implemented, verified, and integrated into the FastAPI application. All 61 unit and integration tests are passing with zero regressions to the existing models or API contracts.
