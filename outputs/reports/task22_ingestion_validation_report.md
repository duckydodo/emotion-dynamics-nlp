# Task 22 — Unified Ingestion Validation and Security Audit Report

**Repository:** `emotion-dynamics-nlp`  
**Working Directory:** `/home/chitta/Projects/emotion-dynamics-nlp`  
**Date:** November 10, 2026  
**Status:** Audit Complete & Hardened (101/101 Tests Passing)  
**Task Reference:** Task 22 (Unified Ingestion Validation and Security Audit)  
**Deliverable File:** [`outputs/reports/task22_ingestion_validation_report.md`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/reports/task22_ingestion_validation_report.md)  

---

## 1. Executive Summary

Task 22 conducts a comprehensive validation, hardening, and security audit of the unified conversation ingestion engine across all five supported formats:
1. **Pasted Text** ([`PlainTextParser`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/data/parsers/plain_text.py))
2. **Plain `.txt` Files** ([`PlainTextParser`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/data/parsers/plain_text.py))
3. **Instagram JSON Exports** ([`InstagramJsonParser`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/data/parsers/instagram.py))
4. **WhatsApp Text Exports** ([`WhatsAppTxtParser`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/data/parsers/whatsapp.py))
5. **Structured CSV/TSV Files** ([`TabularCsvParser`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/data/parsers/csv_parser.py))

### Key Audit Findings & Hardening Implemented
1. **Cross-Format Invariance Verified:** Identical conversational dialogues parsed from Plain Text, WhatsApp, Instagram JSON, and Tabular CSV produce equivalent normalized messages and identical evaluable turn sequences (`turn_id`, `speaker`, `text`) for downstream inference models (`DistilBERT`, `DialogueRNN`, and future `DialogueGCN`).
2. **Defensive Bounds Enforced:** System boundaries have been established and tested:
   - `MAX_PAYLOAD_BYTES = 10 MB` enforced at the FastAPI layer, returning HTTP 413 for oversized requests.
   - `MAX_MESSAGES_PER_CONVERSATION = 1,000` enforced at the schema layer with safe truncation and explicit `MESSAGE_COUNT_LIMIT_EXCEEDED` warnings.
   - `MAX_MESSAGE_CHARACTERS = 4,000` enforced per utterance with `MESSAGE_TEXT_TRUNCATED` warnings.
   - `MAX_SPEAKER_NAME_LENGTH = 64` enforced for speaker display names.
3. **Zero Disk-Persistence & Privacy Assurance:** Conversation processing is strictly in-memory. No raw messages, uploaded files, or transcripts are written to persistent host storage or databases.
4. **Filesystem Path Masking:** Missing model checkpoints or internal exceptions sanitize error messages to prevent internal server path leakage (e.g. `/home/chitta/...`).
5. **Preview Readiness:** The `POST /import` endpoint returns complete conversation preview metadata (`parsed_message_count`, `evaluable_turn_count`, `participant_count`, `unsupported_message_count`, `warning_count`, `date_range`, and `preview_messages`) allowing UI inspection prior to model execution.
6. **100% Test Suite Verification:** 101 tests across 8 test suites pass in Docker with zero regressions.

---

## 2. Ingestion Format Matrix

The application provides a unified entrypoint via [`parse_canonical()`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/data/parsers/__init__.py) with auto-detection or explicit format hints:

| Source Format | Parser Class | Auto-Detection Signature | Key Capabilities & Sanitizations |
| :--- | :--- | :--- | :--- |
| **Pasted Text** | `PlainTextParser` | Fallback if `:` present | Multiline utterances, colon handling in text, non-speaker prefix filtering (`Note:`, `P.S.:`, `http://`). |
| **Plain `.txt`** | `PlainTextParser` | Identical to pasted text | Unicode and emoji preservation, blank line tolerance, missing timestamp warnings. |
| **Instagram JSON** | `InstagramJsonParser` | `participants` & `messages` keys | Reverse-chronological inversion to forward order, Latin-1 UTF-8 mojibake repair, attachment metadata, reaction extraction. |
| **WhatsApp TXT** | `WhatsAppTxtParser` | Bracketed `[...]` or Dash ` - ` timestamp lines | 12h/24h time, LRM/RLM/NBSP invisible character stripping, media omitted placeholder extraction, date convention inference. |
| **Tabular CSV** | `TabularCsvParser` | Delimiter detection & valid header row | Auto-detected and configurable column mapping (`column_mapping`), RFC 4180 quotes with newlines, UTF-8 BOM removal. |

---

## 3. Canonical Schema Consistency & Cross-Format Equivalence

A primary requirement of the audit is ensuring that semantic dialogues ingested from different platforms yield identical internal representations for emotion inference.

### 3.1 Cross-Format Invariance Test

In [`tests/test_ingestion_security_and_limits.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/tests/test_ingestion_security_and_limits.py), an identical 3-turn dialogue was represented across all four parser dialects:

```
[Plain Text]                          [WhatsApp TXT]
Alice: Hello Bob!                     [10/11/26, 10:00:00 AM] Alice: Hello Bob!
Bob: Hi Alice, how are you today?     [10/11/26, 10:01:00 AM] Bob: Hi Alice, how are you today?
Alice: I am feeling great, thank you. [10/11/26, 10:02:00 AM] Alice: I am feeling great, thank you.

[Instagram JSON]                      [Tabular CSV]
{                                     speaker,text,timestamp
  "participants": [...],              Alice,Hello Bob!,2026-11-10T10:00:00Z
  "messages": [                       Bob,"Hi Alice, how are you today?",2026-11-10T10:01:00Z
    {"sender": "Alice", ...turn 3},   Alice,"I am feeling great, thank you.",2026-11-10T10:02:00Z
    {"sender": "Bob",   ...turn 2},
    {"sender": "Alice", ...turn 1}
  ]
}
```

### 3.2 Normalized Output Verification

When passed through `conv.get_evaluable_turns()`, all four formats yielded the exact same sequence:

```python
[
    ConversationTurn(turn_id=0, speaker="Alice", text="Hello Bob!"),
    ConversationTurn(turn_id=1, speaker="Bob", text="Hi Alice, how are you today?"),
    ConversationTurn(turn_id=2, speaker="Alice", text="I am feeling great, thank you."),
]
```

- Contiguous 0-indexed integer `turn_id` values are preserved regardless of skipped system events or metadata.
- Speaker profiles map dynamically to `spk_0`, `spk_1` party indices.
- Downstream models (`DistilBERT`, `DialogueRNN`) receive identical inputs regardless of the import format.

---

## 4. Defensive Bounds & Limits

To prevent memory exhaustion, regex backtracking, and latency degradation on CPU or GPU hardware, defensive bounds are strictly enforced:

| Limit Parameter | Default Bound | Enforcement Layer | Behavior When Exceeded | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **`MAX_PAYLOAD_BYTES`** | 10 MB (10,485,760 bytes) | FastAPI (`/analyze`, `/predict`, `/import`) | Immediate HTTP 413 `Payload exceeds maximum permitted size` | Protects server memory against oversized uploads or Denial of Service attacks. |
| **`MAX_MESSAGES_PER_CONVERSATION`** | 1,000 messages | Schema (`conv.enforce_limits()`) | Truncates trailing messages; appends `MESSAGE_COUNT_LIMIT_EXCEEDED` warning | Recurrent DialogueRNN and attention models scale with sequence length. 1,000 messages exceeds 99.9% of real-world dialogues. |
| **`MAX_MESSAGE_CHARACTERS`** | 4,000 characters | Schema (`conv.enforce_limits()`) | Truncates text to 4,000 chars; appends `MESSAGE_TEXT_TRUNCATED` warning | DistilBERT truncates to 128/512 tokens anyway; prevents regex performance cliffs on single runaway lines. |
| **`MAX_SPEAKER_NAME_LENGTH`** | 64 characters | Schema (`conv.enforce_limits()`) | Truncates speaker name to 64 chars | Prevents UI table overflow and memory bloat from malformed speaker strings. |

---

## 5. Security & Privacy Audit Findings

### 5.1 In-Memory Processing & Privacy Guarantee
- **Audit Result:** Audited all parser implementations and API handlers. Zero file writing operations (`open(..., 'w')`, `to_csv()`, `json.dump()`) exist in ingestion paths.
- **Verification:** Verified via `TestSecurityAndPrivacy.test_no_disk_writes_during_import`, confirming that local directory state is unchanged before and after import. Conversations remain purely in volatile RAM during the request lifecycle.

### 5.2 Server Path Sanitization
- **Audit Result:** Previously, if a model checkpoint was missing or improperly configured, Python's default `FileNotFoundError` could leak internal server directory paths (e.g., `/home/chitta/Projects/...`).
- **Hardening:** Added explicit exception sanitization in [`src/api.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/api.py):
  ```python
  except FileNotFoundError:
      raise HTTPException(status_code=500, detail="Required model checkpoint is unavailable.")
  ```
- **Verification:** Tested in `TestSecurityAndPrivacy.test_error_sanitization_prevents_path_leakage`, ensuring no absolute paths are exposed to clients.

### 5.3 Character Encoding & Malformed Inputs
- **UTF-8 BOM:** Stripped transparently in CSV and text parsers.
- **Instagram Mojibake:** Repaired using byte-level Latin-1 to UTF-8 recoding (`ftfy` algorithm).
- **Invisible Unicode Control Codes:** WhatsApp parsers strip Left-to-Right Marks (`\u200e`), Right-to-Left Marks (`\u200f`), and Non-Breaking Spaces (`\u00a0`).
- **Malformed Delimiters:** Plain text non-speaker lines and CSV unmatched quotes fail safely without crashing the server.

---

## 6. Preview Readiness API Contract

The `POST /import` endpoint provides a full preview response allowing frontend components to display import diagnostics before neural inference:

```json
{
  "status": "warning",
  "conversation_id": "conv_wa_abc123",
  "source_format": "whatsapp_txt",
  "num_messages": 4,
  "parsed_message_count": 4,
  "evaluable_turn_count": 3,
  "num_speakers": 2,
  "participant_count": 2,
  "speakers": ["Alice", "Bob"],
  "unsupported_message_count": 1,
  "warning_count": 1,
  "warnings": [
    {
      "code": "AMBIGUOUS_DATE_FORMAT",
      "message": "Date numbers could be either Day/Month or Month/Day. Inferred 'DMY'.",
      "severity": "info",
      "line_number": null,
      "message_id": null
    }
  ],
  "validation_errors": [],
  "date_range": {
    "start": "2026-11-10T10:00:00",
    "end": "2026-11-10T10:03:00",
    "duration_seconds": 180.0
  },
  "preview_messages": [
    {
      "message_id": "msg_0001",
      "speaker": "Alice",
      "text": "Message one",
      "timestamp": "2026-11-10T10:00:00",
      "is_evaluable": true
    }
  ]
}
```

---

## 7. Test Results Matrix

All unit, integration, and security tests were executed inside the Docker environment (`emotion-dynamics-nlp:week11`).

| Test Suite File | Focus Area | Tests Passed | Status |
| :--- | :--- | :--- | :--- |
| [`tests/test_parsers.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/tests/test_parsers.py) | Plain text parser & canonical schema | 16 / 16 | **PASS** |
| [`tests/test_instagram_parser.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/tests/test_instagram_parser.py) | Instagram JSON export ingestion | 18 / 18 | **PASS** |
| [`tests/test_whatsapp_parser.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/tests/test_whatsapp_parser.py) | WhatsApp exports (Android & iOS) | 18 / 18 | **PASS** |
| [`tests/test_csv_parser.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/tests/test_csv_parser.py) | Tabular CSV/TSV parser & column mapping | 16 / 16 | **PASS** |
| [`tests/test_ingestion_security_and_limits.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/tests/test_ingestion_security_and_limits.py) | Cross-format invariance, limits, security | 8 / 8 | **PASS** |
| [`tests/test_dialoguernn.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/tests/test_dialoguernn.py) | Recurrent network architecture & causality | 9 / 9 | **PASS** |
| [`tests/test_dialoguernn_ablations.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/tests/test_dialoguernn_ablations.py) | Ablation configurations & state masking | 12 / 12 | **PASS** |
| [`tests/test_application_integration.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/tests/test_application_integration.py) | End-to-end API, models & dynamics analysis | 4 / 4 | **PASS** |
| **Total** | **Full Repository Verification** | **101 / 101** | **100% PASS** |

---

## 8. Remaining Risks and Mitigations

| Risk | Impact | Mitigation Strategy |
| :--- | :--- | :--- |
| **Ambiguous Date Order in WhatsApp** | Day and month both $\le 12$ cannot be determined unambiguously without locale context. | Scans entire document to identify unambiguous rows ($> 12$); falls back to standard WhatsApp worldwide DMY convention; emits `AMBIGUOUS_DATE_FORMAT` warning. |
| **CSV Unlabeled Headers** | Arbitrary CSV without standard column names cannot be inferred automatically. | Requires explicit `column_mapping` parameter; emits `COLUMN_MAPPING_REQUIRED` error rather than guessing unrelated columns. |
| **DialogueGCN Integration** | Future Task 23+ will introduce DialogueGCN which requires graph adjacency construction. | `CanonicalConversation` preserves chronological sequence indices and speaker party IDs needed for windowed edge construction. |

---

## 9. Conclusion & Readiness Verdict

**Readiness Verdict:** **APPROVED & PRODUCTION-READY**

The unified ingestion pipeline is fully validated, secure, memory-safe, and backward-compatible with all existing inference pipelines. All 101 tests pass cleanly. The repository is ready to proceed to the next phase of the roadmap.
