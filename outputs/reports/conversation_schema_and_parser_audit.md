# Conversation Parser Audit and Canonical Schema Specification

**Repository:** `emotion-dynamics-nlp`  
**Working Directory:** `/home/chitta/Projects/emotion-dynamics-nlp`  
**Date:** November 10, 2026  
**Status:** Audit & Design Approved — Pre-Implementation Phase  
**Task Reference:** Task 17 (Conversation Parser Audit and Canonical Schema)  
**Deliverable File:** [`outputs/reports/conversation_schema_and_parser_audit.md`](file:///home/chitta/Projects/emotion-dynamics-nlp/outputs/reports/conversation_schema_and_parser_audit.md)  

---

## Executive Summary

This report performs a comprehensive architectural audit of the conversation ingestion pipeline in `emotion-dynamics-nlp` and defines a **canonical internal conversation schema** (`CanonicalConversation` / `CanonicalMessage`). 

The current system relies entirely on a monolithic, plain-text parser ([`src/data/conversation.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/data/conversation.py)) that expects unadorned `Speaker: utterance` text. This parser lacks support for timestamps, non-text media events, structured chat exports (such as Instagram Direct Message JSON, WhatsApp `.txt`, and tabular CSVs), and parser diagnostic warnings.

To enable multi-format conversational analysis without degrading or breaking the verified downstream inference models (DistilBERT, DialogueRNN) and dynamics metrics ([`src/analysis/conversation_analysis.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/analysis/conversation_analysis.py)), this audit defines:
1. Complete inventory of the 7 downstream locations consuming conversational structures.
2. Root-cause analysis of current parser limitations (e.g., hard errors on empty text, lack of chronological awareness, absence of dynamic speaker ID generation).
3. A formal, extensible canonical schema supporting timestamps, dynamic speaker IDs, message types, media attachments, parser diagnostics, and deterministic tie-breaking for out-of-order or timestamp-deficient conversations.
4. An adapter pattern that guarantees 100% backward compatibility with existing API contracts, model inference wrappers, and MELD training pipelines.

---

## 1. Current Parser Architecture

The current conversation ingestion mechanism is implemented in a single module: [`src/data/conversation.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/data/conversation.py#L1-L111).

```
┌────────────────────────────────────────────────────────────────────────┐
│                        CURRENT PARSER PIPELINE                         │
│                                                                        │
│   Raw String Input ("Alice: Hello\nBob: Hi")                          │
│         │                                                              │
│         ▼                                                              │
│   src/data/conversation.py :: parse_conversation(raw_text: str)        │
│         │                                                              │
│         ├─ Splits on newline: for line in raw_text.splitlines():       │
│         ├─ Checks for ":": line.split(":", 1)                          │
│         ├─ Multiline continuation if no colon                          │
│         ├─ Validates speaker is not empty                              │
│         ├─ Validates text is not empty (throws error if empty)         │
│         │                                                              │
│         ▼                                                              │
│   list[ConversationTurn]                                               │
│   [ ConversationTurn(turn_id=0, speaker="Alice", text="Hello"), ... ] │
└────────────────────────────────────────────────────────────────────────┘
```

### 1.1 Implementation Details

The current data structure is a minimal dataclass ([`src/data/conversation.py#L4-L8`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/data/conversation.py#L4-L8)):
```python
@dataclass
class ConversationTurn:
    turn_id: int
    speaker: str
    text: str
```

The parsing algorithm ([`src/data/conversation.py#L15-L110`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/data/conversation.py#L15-L110)) operates as a line-oriented procedural state machine with the following specific rules:
1. **Type Checking:** Validates `isinstance(raw_text, str)`. Non-strings raise `ConversationParseError("Conversation input must be text.")`.
2. **Empty Input Check:** Validates `raw_text.strip()`. Empty or whitespace-only strings raise `ConversationParseError("Conversation is empty.")`.
3. **Turn ID Generation:** Generates 0-indexed sequential integers `turn_id = len(turns)` strictly based on line arrival order.
4. **Delimiter Splitting:** Scans each line for `:`. Only the first colon splits `speaker` from `text` (`line.split(":", 1)`).
5. **Utterance Continuation:** Lines lacking a colon are treated as continuation lines for the preceding speaker and appended to `current_text_parts`.
6. **No-Speaker Error:** If a continuation line appears before any line containing a colon, raises `ConversationParseError(f"Line {line_number}: expected 'Speaker: utterance'.")`.
7. **Empty Speaker Error:** If `speaker.strip()` is empty (`: text`), raises `ConversationParseError(f"Line {line_number}: speaker name is empty.")`.
8. **Empty Text Error:** If `text.strip()` is empty (`Speaker:`), raises `ConversationParseError(f"Line {line_number}: speaker '{speaker}' has no utterance.")`.
9. **Final Flush & Validation:** Upon loop termination, flushes the final turn. If `turns` is empty, raises `ConversationParseError("No valid conversation turns were found.")`.

### 1.2 Limitations of the Current Architecture

| Architectural Area | Current Behavior | Limitation & Impact |
| :--- | :--- | :--- |
| **Input Format** | Plain string `Speaker: text` only | Cannot parse JSON (Instagram, Telegram, Slack), WhatsApp exports, or tabular CSV/TSV. |
| **Temporal Data** | Timestamps completely absent | Cannot compute duration, inter-utterance latency, or chronological ordering when logs are out of order. |
| **Media & Events** | Empty text raises fatal `ConversationParseError` | Silently discarding or rejecting media attachments, reactions, stickers, and system messages breaks analysis of real-world chat exports. |
| **Message Ordering** | Assumes line order is chronological order | Incompatible with formats exported in reverse chronological order (e.g., Meta/Instagram `message_1.json`, where `messages[0]` is the most recent message). |
| **Encoding Support** | Standard Python UTF-8 decoding only | Breaks on legacy mojibake exports (e.g., Instagram's Latin-1 escaped UTF-8 characters like `\u00f0\u009f\u0098\u0082` for emojis). |
| **Speaker Identification** | Raw string literal only | No normalized, machine-readable `speaker_id` (e.g. `spk_0`, `spk_1`); downstream models must independently map strings. |
| **Diagnostics & Warnings** | Fail-stop on any irregularity | No concept of non-fatal warnings (e.g., malformed timestamp recovered, non-text message skipped for emotion inference). |

---

## 2. Current Input and Output Structures

The conversation data structures traverse three layers: the API transport layer, the internal application service layer, and the web frontend presentation layer.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           API REQUEST PAYLOAD                            │
│                                                                          │
│  ConversationRequest (pydantic in src/api.py)                            │
│  ├── conversation: str = "Alice: Excited!\nBob: Me too."                │
│  └── mode: str = "dialoguernn"                                           │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         INTERNAL INFERENCE LAYER                         │
│                                                                          │
│  list[ConversationTurn] (dataclass in src/data/conversation.py)          │
│  └── [ ConversationTurn(turn_id=0, speaker="Alice", text="Excited!"),   │
│        ConversationTurn(turn_id=1, speaker="Bob", text="Me too.") ]     │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                          API RESPONSE PAYLOAD                            │
│                                                                          │
│  AnalyzeResponse (JSON dict produced by src/application/analyzer.py)     │
│  ├── mode: "dialoguernn"                                                 │
│  ├── model_metadata: { architecture: "...", causal: true, ... }          │
│  ├── turns: [                                                            │
│  │     { turn_id: 0, speaker: "Alice", text: "Excited!",                 │
│  │       emotion: "joy", confidence: 0.942,                              │
│  │       probabilities: { anger: 0.01, ..., joy: 0.94 } }               │
│  │   ]                                                                   │
│  └── analysis: {                                                         │
│        total_transitions: 1, shifts: 0, persistence: 1,                  │
│        shift_rate: 0.0, persistence_rate: 1.0,                           │
│        same_speaker_transitions: 0, cross_speaker_transitions: 1,       │
│        emotion_distribution: { joy: 2 }, turning_points: []              │
│      }                                                                   │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.1 API Request Schema ([`src/api.py#L22-L30`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/api.py#L22-L30))
```python
class ConversationRequest(BaseModel):
    conversation: str = Field(
        ...,
        description="Conversation using one Speaker: utterance per turn.",
    )
    mode: str = Field(
        default="utterance",
        description="Model mode: utterance, context_1, context_1_speaker, or dialoguernn.",
    )
```

### 2.2 Internal Turn Output Contract ([`src/models/inference.py#L96-L103`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/models/inference.py#L96-L103), [`#L218-L225`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/models/inference.py#L218-L225))
Both `EmotionInference` and `DialogueRNNInference` output a list of dictionaries with the exact keys:
- `turn_id: int` — 0-indexed turn position.
- `speaker: str` — verbatim speaker label.
- `text: str` — verbatim utterance text.
- `emotion: str` — predicted emotion label from MELD vocabulary (`anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, `surprise`).
- `confidence: float` — maximum softmax probability ($0.0 \le p \le 1.0$).
- `probabilities: dict[str, float]` — dictionary mapping each of the 7 MELD emotion keys to its softmax float value.

### 2.3 Analysis Output Contract ([`src/analysis/conversation_analysis.py#L6-L19`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/analysis/conversation_analysis.py#L6-L19))
The dynamics calculation yields:
- Transition counts (`transition_counts: dict[str, int]`) keyed by `"from_emotion → to_emotion"`.
- Binary transition counts: `shifts: int` ($E_{t} \ne E_{t-1}$) and `persistence: int` ($E_{t} = E_{t-1}$).
- Transition rates: `shift_rate: float` and `persistence_rate: float`.
- Speaker-conditioned dynamics: `same_speaker_transitions`, `same_speaker_shifts`, `cross_speaker_transitions`, `cross_speaker_shifts`.
- Turning points: `turning_points: list[dict]` recording persistent shift events ($E_{t-1} \ne E_t$ and $E_{t+1} = E_t$).

---

## 3. Downstream Dependency Map

An audit of the entire codebase reveals seven primary consumer locations that depend on the conversation structures.

```
                               ┌─────────────────────────┐
                               │  src/data/conversation  │
                               │    ConversationTurn     │
                               │   parse_conversation    │
                               └────────────┬────────────┘
                                            │
        ┌───────────────────────────────────┼───────────────────────────────────┐
        │                                   │                                   │
        ▼                                   ▼                                   ▼
┌───────────────────────┐       ┌───────────────────────┐       ┌───────────────────────┐
│  src/application/     │       │  src/models/          │       │  src/analysis/        │
│  analyzer.py          │       │  inference.py         │       │  conversation_        │
│  ConversationAnalyzer │       │  EmotionInference     │       │  analysis.py          │
│  - parse_conversation │       │  DialogueRNNInference │       │  analyze_predictions  │
│  - analyze            │       │  - turn.text          │       │  - item["turn_id"]    │
└───────────┬───────────┘       │  - turn.speaker       │       │  - item["speaker"]    │
            │                   │  - turn.turn_id       │       │  - item["emotion"]    │
            ▼                   └───────────────────────┘       └───────────────────────┘
┌───────────────────────┐
│  src/api.py           │
│  - POST /analyze      │
│  - POST /predict      │
└───────────┬───────────┘
            │
            ├───────────────────────────────────┐
            ▼                                   ▼
┌───────────────────────┐       ┌───────────────────────┐
│  static/index.html    │       │  tests/test_          │
│  - textarea input     │       │  application_         │
│  - renderTurns()      │       │  integration.py       │
│  - renderTimeline()   │       │  - test contracts     │
└───────────────────────┘       └───────────────────────┘
```

### Detailed Dependency Breakdown

1. **[`src/application/analyzer.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/application/analyzer.py#L58-L92)**
   - Line 58–59: Exposes `parse_conversation(self, conversation)` delegating to `src.data.conversation`.
   - Line 86: Calls `parse_conversation(conversation)` directly in `analyze()`.
   - Line 89: Passes `turns` directly into `model.predict(turns)`.
   - Line 91: Passes `predictions` to `analyze_predictions(predictions)`.
   - *Dependency Sensitivity:* Expects `turns` to be an iterable sequence with `turn.text`, `turn.speaker`, and `turn.turn_id`.

2. **[`src/models/inference.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/models/inference.py#L41-L133), [`#L169-L228`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/models/inference.py#L169-L228)**
   - **`EmotionInference.predict(turns)`**:
     - Extracts `turn.text.strip()` for sequence tokenization.
     - For `context_1` and `context_1_speaker`, reads `previous_turn.text` and formats `[SPEAKER_{turn.speaker}] {turn.text}`.
     - Constructs result dictionaries pairing `turn.turn_id`, `turn.speaker`, and `turn.text` with model predictions.
   - **`DialogueRNNInference.predict(turns)`**:
     - Line 174: Cleans `turn.text.strip() if turn.text else ""`.
     - Line 178–189: Encodes text batches through `DistilBertModel` extracting `[CLS]` embeddings.
     - Line 192–200: Constructs a dynamic `party_map: Dict[str, int]` mapping each unique `turn.speaker` string to party indices $0 \dots P-1$.
     - Line 219–225: Formats output dictionaries using `turn.turn_id`, `turn.speaker`, `turn.text`.
   - *Dependency Sensitivity:* Both wrappers require `turn.text` as a string. If `turns` contains media events with empty text, DistilBERT encodes empty strings `""` (just `[CLS] [SEP]`), which generates arbitrary uncalibrated logits unless filtered or flagged.

3. **[`src/analysis/conversation_analysis.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/analysis/conversation_analysis.py#L42-L118)**
   - Line 42–45: Sorts predictions by `item["turn_id"]`.
   - Line 69–70: **Strict Adjacency Check:**
     ```python
     if following["turn_id"] != current["turn_id"] + 1:
         continue
     ```
   - Line 101–102: **Turning Point Adjacency Check:**
     ```python
     if next_turn["turn_id"] != following["turn_id"] + 1:
         continue
     ```
   - Line 83: Compares `current["speaker"] == following["speaker"]`.
   - *Dependency Sensitivity:* **Crucial.** If non-text messages are omitted and leave gaps in `turn_id`, the dynamics engine will treat subsequent text turns as non-adjacent and skip calculating transitions across the gap. The schema design must explicitly account for turn numbering vs. original message numbering.

4. **[`src/api.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/api.py#L22-L71)**
   - Declares `ConversationRequest(conversation: str, mode: str)`.
   - Endpoints `POST /analyze` and `POST /predict` accept `ConversationRequest`.
   - *Dependency Sensitivity:* Adding multi-format support must maintain backwards compatibility with existing clients posting `{"conversation": "...", "mode": "..."}`.

5. **[`static/index.html`](file:///home/chitta/Projects/emotion-dynamics-nlp/static/index.html#L1472-L1519), [`#L1657-L1671`](file:///home/chitta/Projects/emotion-dynamics-nlp/static/index.html#L1657-L1671), [`#L2312-L2374`](file:///home/chitta/Projects/emotion-dynamics-nlp/static/index.html#L2312-L2374)**
   - Sends `{ conversation: string, mode: string }` via `fetch("/analyze")`.
   - `renderTimeline(turns)`: plots `turn.turn_id` as X-axis labels and `turn.probabilities` as stacked areas.
   - `renderTurns(turns)`: injects `turn.speaker`, `turn.emotion`, `turn.text`, `turn.confidence` into the DOM.
   - *Dependency Sensitivity:* The frontend relies directly on `turns` containing `speaker`, `emotion`, `text`, `confidence`, and `probabilities`.

6. **[`tests/test_application_integration.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/tests/test_application_integration.py#L38-L165)**
   - Contains 8 integration test methods exercising `/analyze` and `/predict` across `utterance`, `context_1`, `context_1_speaker`, and `dialoguernn`.
   - Explicitly asserts presence of `turn_id`, `speaker`, `text`, `emotion`, `confidence`, `probabilities`.
   - *Dependency Sensitivity:* Any breaking modification to `ConversationTurn` or the API payload keys causes regression failures across the integration test suite.

7. **MELD Training Pipeline Isolation Boundary**
   - MELD dataset training is driven by [`src/data/loader.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/src/data/loader.py#L7-L44) and [`scripts/extract_dialogue_features.py`](file:///home/chitta/Projects/emotion-dynamics-nlp/scripts/extract_dialogue_features.py#L29-L32).
   - These scripts load raw CSVs (`Dialogue_ID`, `Utterance_ID`, `Speaker`, `Utterance`, `Emotion`) directly into Pandas DataFrames.
   - **Isolation Guarantee:** The conversation parser in `src/data/conversation.py` is **NOT** used during MELD offline training or canonical feature extraction. Therefore, extending the runtime ingestion parser does not affect MELD model training or canonical evaluation benchmarks.

---

## 4. Proposed Canonical Internal Schema

To unify plain text, Instagram Direct Message JSON, WhatsApp exports, and tabular CSVs into a single consistent structure, we specify the **`CanonicalConversation`** data model.

```
┌────────────────────────────────────────────────────────────────────────┐
│                          CANONICAL CONVERSATION                        │
│                                                                        │
│  CanonicalConversation                                                 │
│  ├── conversation_id: str ("conv_20261110_a1b2")                       │
│  ├── source_format: SourceFormat (TEXT | INSTAGRAM | WHATSAPP | CSV)   │
│  ├── title: Optional[str] ("Project Discussion")                       │
│  ├── participants: list[SpeakerProfile]                                │
│  │     └── [ SpeakerProfile(speaker_id="spk_0", name="Alice"), ... ]   │
│  ├── messages: list[CanonicalMessage]                                  │
│  │     ├── message_id: str ("msg_0001")                                │
│  │     ├── source_index: int (original raw arrival order: 0, 1, ...)   │
│  │     ├── sequence_index: int (chronological order: 0, 1, ...)        │
│  │     ├── speaker: str ("Alice")                                      │
│  │     ├── speaker_id: str ("spk_0")                                   │
│  │     ├── text: str ("Are you free?")                                 │
│  │     ├── timestamp: Optional[datetime] (UTC ISO-8601)                │
│  │     ├── message_type: MessageType (TEXT | MEDIA | SYSTEM | REACTION)│
│  │     ├── metadata: dict[str, Any]                                    │
│  │     └── is_evaluable: bool (True for text, False for media)         │
│  ├── warnings: list[ParserWarning]                                     │
│  └── metadata: dict[str, Any]                                          │
└────────────────────────────────────────────────────────────────────────┘
```

### 4.1 Formal Schema Specification (Python Dataclasses / Pydantic)

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class MessageType(str, Enum):
    """Categorical classification of conversational items."""
    TEXT = "text"          # Standard spoken/written utterance
    MEDIA = "media"        # Photo, video, audio voice note, sticker, file
    SYSTEM = "system"      # Chat event (joined, left, call missed, pinned)
    REACTION = "reaction"  # Emoji reaction attached to an utterance


class SourceFormat(str, Enum):
    """Supported source ingestion formats."""
    PLAIN_TEXT = "plain_text"
    INSTAGRAM_JSON = "instagram_json"
    WHATSAPP_TXT = "whatsapp_txt"
    TABULAR_CSV = "tabular_csv"


class WarningSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ParserWarning:
    """Non-fatal diagnostic produced during conversation ingestion."""
    code: str                  # e.g., "MOJIBAKE_REPAIRED", "MEDIA_OMITTED"
    message: str               # Human-readable explanation
    severity: WarningSeverity
    line_number: Optional[int] = None
    message_id: Optional[str] = None


@dataclass
class SpeakerProfile:
    """Dynamically generated speaker profile."""
    speaker_id: str            # Normalized machine ID: "spk_0", "spk_1"
    name: str                  # Human-readable display name: "Alice"
    normalized_name: str       # Lowercase stripped: "alice"
    turn_count: int = 0


@dataclass
class CanonicalMessage:
    """Single canonical message item representing an utterance or event."""
    conversation_id: str
    message_id: str                        # Globally unique ID (e.g., "msg_0001")
    source_index: int                      # Position in raw input file (0-indexed)
    sequence_index: int                    # Verified chronological position (0-indexed)
    speaker: str                           # Display name ("Alice")
    speaker_id: str                        # Normalized party ID ("spk_0")
    text: str                              # Utterance text (empty string for pure media)
    timestamp: Optional[datetime]          # ISO-8601 UTC timestamp or None
    message_type: MessageType = MessageType.TEXT
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_evaluable: bool = True              # True if contains evaluable NLP text

    def to_inference_turn(self, inference_turn_id: int) -> ConversationTurn:
        """
        Adapter method producing backward-compatible ConversationTurn
        for EmotionInference and DialogueRNNInference.
        """
        return ConversationTurn(
            turn_id=inference_turn_id,
            speaker=self.speaker,
            text=self.text,
        )


@dataclass
class CanonicalConversation:
    """Complete canonical conversation container."""
    conversation_id: str
    source_format: SourceFormat
    title: Optional[str] = None
    participants: List[SpeakerProfile] = field(default_factory=list)
    messages: List[CanonicalMessage] = field(default_factory=list)
    warnings: List[ParserWarning] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_evaluable_turns(self) -> List[ConversationTurn]:
        """
        Returns strictly evaluable text turns with contiguous 0-indexed
        turn_id values for downstream inference and dynamics analysis.
        """
        evaluable_turns: List[ConversationTurn] = []
        for msg in self.messages:
            if msg.is_evaluable and msg.text.strip():
                evaluable_turns.append(
                    msg.to_inference_turn(inference_turn_id=len(evaluable_turns))
                )
        return evaluable_turns

    def get_speaker_party_mapping(self) -> Dict[str, int]:
        """Returns dynamic speaker string to integer party mapping for DialogueRNN."""
        mapping: Dict[str, int] = {}
        for p in self.participants:
            if p.name not in mapping:
                mapping[p.name] = len(mapping)
        return mapping
```

---

### 4.2 Realistic Multi-Format Example

Below is a complete, realistic example of a `CanonicalConversation` resulting from parsing an **Instagram Direct Message JSON export** containing text, an omitted photo, an emoji reaction, and a recovered Latin-1 mojibake character:

```json
{
  "conversation_id": "conv_ig_20261110_7f3b89",
  "source_format": "instagram_json",
  "title": "Weekend Trip Planning",
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
      "turn_count": 2
    }
  ],
  "messages": [
    {
      "conversation_id": "conv_ig_20261110_7f3b89",
      "message_id": "msg_0001",
      "source_index": 3,
      "sequence_index": 0,
      "speaker": "Alice",
      "speaker_id": "spk_0",
      "text": "Hey! Are you still coming to the cabin?",
      "timestamp": "2026-11-10T09:30:00Z",
      "message_type": "text",
      "metadata": {
        "client_timestamp_ms": 1794297000000
      },
      "is_evaluable": true
    },
    {
      "conversation_id": "conv_ig_20261110_7f3b89",
      "message_id": "msg_0002",
      "source_index": 2,
      "sequence_index": 1,
      "speaker": "Bob",
      "speaker_id": "spk_1",
      "text": "",
      "timestamp": "2026-11-10T09:31:15Z",
      "message_type": "media",
      "metadata": {
        "media_type": "photo",
        "uri": "messages/inbox/alice_123/photos/101.jpg",
        "caption": null
      },
      "is_evaluable": false
    },
    {
      "conversation_id": "conv_ig_20261110_7f3b89",
      "message_id": "msg_0003",
      "source_index": 1,
      "sequence_index": 2,
      "speaker": "Bob",
      "speaker_id": "spk_1",
      "text": "Packed and ready! Let’s go! 🎉",
      "timestamp": "2026-11-10T09:31:45Z",
      "message_type": "text",
      "metadata": {
        "encoding_repaired": true,
        "raw_content_preview": "Packed and ready! Let\u00e2\u0080\u0099s go! \u00f0\u009f\u008e\u0089"
      },
      "is_evaluable": true
    },
    {
      "conversation_id": "conv_ig_20261110_7f3b89",
      "message_id": "msg_0004",
      "source_index": 0,
      "sequence_index": 3,
      "speaker": "Alice",
      "speaker_id": "spk_0",
      "text": "",
      "timestamp": "2026-11-10T09:32:00Z",
      "message_type": "reaction",
      "metadata": {
        "reaction": "❤️",
        "target_message_id": "msg_0003"
      },
      "is_evaluable": false
    }
  ],
  "warnings": [
    {
      "code": "IG_REVERSE_CHRONOLOGICAL_INVERTED",
      "message": "Instagram export array detected in reverse chronological order. Inverted 4 messages to chronological order.",
      "severity": "info",
      "message_id": null
    },
    {
      "code": "MOJIBAKE_REPAIRED",
      "message": "Repaired Latin-1 encoded UTF-8 bytes for 1 message.",
      "severity": "info",
      "message_id": "msg_0003"
    },
    {
      "code": "NON_TEXT_MEDIA_PRESERVED",
      "message": "Message contains photo without caption. Retained in canonical log but flagged is_evaluable=False for emotion inference.",
      "severity": "info",
      "message_id": "msg_0002"
    }
  ],
  "metadata": {
    "total_raw_messages": 4,
    "evaluable_messages": 2,
    "has_timestamps": true
  }
}
```

---

### 4.3 Deterministic Ordering and Tie-Breaking Rules

A critical requirement is clearly distinguishing **original source order** (`source_index`) from **chronological sequence order** (`sequence_index`), especially when timestamps are missing, identical, or partially present.

```
┌────────────────────────────────────────────────────────────────────────┐
│                      DETERMINISTIC ORDERING RULES                      │
│                                                                        │
│   Case A: Full Timestamps Available                                    │
│   Primary Sort: timestamp (ascending)                                  │
│   Tie-Breaker: source_index (original arrival order)                   │
│                                                                        │
│   Case B: Reverse Chronological Export (Instagram JSON)                │
│   Source Array: [latest, ..., earliest]                                │
│   Normalized Source Index: len(raw) - 1 - original_index               │
│   Sort: (timestamp_ms, normalized_source_index)                        │
│                                                                        │
│   Case C: No Timestamps Available (Plain Text)                         │
│   sequence_index = source_index (strictly line arrival order)          │
│                                                                        │
│   Case D: Partial / Missing Timestamps in Thread                       │
│   Messages inherit upper-bound timestamp of bounding preceding turn    │
│   Sub-sort: (effective_timestamp, source_index)                        │
└────────────────────────────────────────────────────────────────────────┘
```

1. **Definitions:**
   - `source_index`: An integer ($0, 1, \dots, N-1$) denoting the physical appearance order in the source payload.
   - `sequence_index`: An integer ($0, 1, \dots, N-1$) denoting the validated chronological order for conversational analysis.
2. **Rule 1 — Full Timestamps Available:**
   - Messages are sorted with a stable Python sort using key:
     $$\text{sort\_key}(m) = (\text{timestamp},\; m.\text{source\_index})$$
   - If two messages possess identical timestamps (common in rapid messaging apps where precision is seconds rather than milliseconds), the tie is broken strictly by `source_index`.
3. **Rule 2 — Reverse Chronological Formats (Meta / Instagram JSON):**
   - Instagram exports messages from newest (`index 0`) to oldest (`index N-1`).
   - If timestamps are present: sort ascending by `timestamp_ms`. In the event of identical timestamps, tie-break by inverted array index ($N - 1 - \text{index}$), which preserves the sender's original transmission sequence.
4. **Rule 3 — Missing Timestamps (Plain Text):**
   - When no timestamps exist (e.g. plain text `Speaker: text` input):
     $$m.\text{sequence\_index} = m.\text{source\_index}$$
     The source order is the chronological sequence.
5. **Rule 4 — Sparse / Mixed Timestamps:**
   - If a conversation has intermittent timestamps (e.g., WhatsApp date headers that appear only once per date block, or multiline text without repeat timestamps), intermediate messages inherit the timestamp of the immediately preceding timestamped message with a microsecond-increment tie-breaker, guaranteeing:
     $$t_{\text{prev}} \le t_{\text{intermed}} \le t_{\text{next}}$$
     and preserving `source_index` as the invariant tie-breaker.

---

## 5. Ingestion Validation Rules

To prevent malformed inputs from crashing inference models or producing corrupt emotion dynamics, the canonical parser pipeline enforces four tiers of validation rules:

### 5.1 Structural & Syntactic Validation
1. **Input Size & Depth:**
   - Enforce maximum payload size: $10\text{ MB}$ for raw JSON/text files.
   - Maximum turns per conversation: $1{,}000$ turns (prevents memory exhaustion during transformer forward passes). Conversations exceeding this limit are truncated with a fatal or non-fatal warning.
2. **Dynamic Speaker ID Assignment:**
   - Speakers are identified by trimmed display names.
   - Never assume MELD character identities (`Chandler`, `Joey`, etc.).
   - Assign sequential IDs `spk_0`, `spk_1`, $\dots$ on first discovery.
   - Case-insensitive deduplication: `"alice"` and `"Alice"` map to the same `speaker_id` (`spk_0`), retaining the first encountered capitalization for display.
   - Unknown/unlabeled speakers default to `"Speaker"` or `"Unknown"` and receive a dedicated party ID.

### 5.2 Content & Semantic Validation
1. **Empty Text Handling:**
   - For `MessageType.TEXT`: If stripped text is empty, the parser issues a warning (`EMPTY_TEXT_DROPPED`) and drops the turn if no metadata exists, OR converts it to `is_evaluable = False`.
   - For `MessageType.MEDIA` or `MessageType.SYSTEM`: Empty text is valid. The item is preserved in `CanonicalConversation.messages` with `is_evaluable = False` and appropriate metadata (e.g. `{"attachment": "photo"}`).
2. **Text Normalization:**
   - Strips excessive whitespace and carriage returns (`\r\n` $\to$ `\n`).
   - Instagram Latin-1 Mojibake fix:
     ```python
     def repair_mojibake(text: str) -> str:
         try:
             return text.encode("latin1").decode("utf-8")
         except (UnicodeEncodeError, UnicodeDecodeError):
             return text
     ```
3. **Timestamp Normalization:**
   - All parsed timestamps are converted to UTC `datetime` objects.
   - String representations serialize to standard ISO-8601 (`YYYY-MM-DDTHH:MM:SSZ`).
   - Unparseable date strings generate a `TIMESTAMP_PARSE_FAILED` warning and fall back to `None`, invoking Rule 3 (source order fallback).

### 5.3 Diagnostic Severity Policy

| Diagnostic Code | Condition | Severity | Pipeline Action |
| :--- | :--- | :--- | :--- |
| `EMPTY_CONVERSATION` | Zero messages in input payload | `ERROR` | Raise 400 Bad Request |
| `UNRECOGNIZED_FORMAT` | Payload matches no known format parser | `ERROR` | Raise 400 Bad Request |
| `MOJIBAKE_REPAIRED` | Fixed Latin-1 UTF-8 byte escape | `INFO` | Text repaired; recorded in warnings |
| `MEDIA_OMITTED` | Photo/Audio without caption encountered | `INFO` | Preserved in log; flagged `is_evaluable=False` |
| `SYSTEM_EVENT_SKIPPED`| Encryption or join notice parsed | `INFO` | Stored in metadata; excluded from emotion models |
| `TIMESTAMP_GAP` | Temporal gap $> 24\text{ hours}$ between turns | `WARNING` | Recorded in metadata (potential session split) |
| `SPEAKER_NAME_TRUNCATED`| Speaker string $> 50$ characters | `WARNING` | Trimmed to 50 chars; recorded in warnings |

---

## 6. Compatibility Risks and Mitigation Strategies

Transforming conversation ingestion carries specific compatibility risks across the verified system components. Below is a risk matrix and precise architectural mitigation strategy for each.

| Risk ID | Component | Failure Mode / Compatibility Risk | Severity | Mitigation Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **R-1** | `src/models/inference.py` | DistilBERT and DialogueRNN crash or produce spurious neutral predictions on empty media text. | **HIGH** | Expose `get_evaluable_turns()` on `CanonicalConversation`. Only items with `is_evaluable=True` and non-empty text are dispatched to `model.predict()`. Non-text events remain in the canonical log for UI display. |
| **R-2** | `src/analysis/conversation_analysis.py` | Strict adjacency check (`following["turn_id"] != current["turn_id"] + 1`) fails if non-text turns create gaps in turn numbering. | **HIGH** | Ensure `get_evaluable_turns()` re-indexes evaluable turns with contiguous IDs ($0, 1, \dots, K-1$) while preserving `canonical_message_id` references. This guarantees dynamics transitions only evaluate actual adjacent conversational speech. |
| **R-3** | `src/api.py` | Existing `/analyze` and `/predict` endpoints break for existing client scripts sending `{ "conversation": "..." }`. | **CRITICAL** | Keep `ConversationRequest` backward-compatible. Add optional `format: Optional[str] = "auto"` and optional `raw_payload: Optional[dict] = None`. Auto-detect plain text vs JSON transparently. |
| **R-4** | `static/index.html` | Chart.js and Turn Table expect flat list of turns with `probabilities` and `turn_id`. | **MEDIUM** | Preserve the existing `turns` response array structure in `POST /analyze`. Add extended canonical metadata (`messages`, `warnings`, `participants`) under an optional top-level `canonical` key. |
| **R-5** | `tests/test_application_integration.py` | Regression tests fail due to altered payload keys or changed error messages. | **HIGH** | Preserve existing HTTP 400 exception handling and verify that all 8 existing integration tests pass unchanged. |
| **R-6** | Offline MELD Training Scripts | Training pipelines inadvertently modified or broken. | **LOW** | Strict boundary: MELD training scripts in `scripts/` and `src/data/loader.py` never import `src/data/conversation.py`. Leave training loaders completely decoupled. |

---

## 7. Recommended Implementation Architecture

The planned implementation for Stage 1 (Multi-Format Ingestion) will be structured cleanly across modular files under `src/data/`:

```
src/data/
├── __init__.py
├── conversation.py             <- Backward-compatible facade (re-exports Canonical types)
├── schema.py                   <- CanonicalConversation, CanonicalMessage, MessageType
├── parsers/
│   ├── __init__.py             <- Parser factory (auto_detect_and_parse)
│   ├── base.py                 <- BaseConversationParser abstract class
│   ├── plain_text.py           <- PlainTextParser (refactored from conversation.py)
│   ├── instagram.py            <- InstagramJsonParser (Latin-1 fix, ascending sort)
│   ├── whatsapp.py             <- WhatsAppTxtParser (regex timestamp matching)
│   └── csv_parser.py           <- TabularCsvParser (column mapping heuristic)
```

### 7.1 Unified Parsing Flow

```python
class BaseConversationParser(ABC):
    @abstractmethod
    def can_parse(self, raw_input: Any) -> bool:
        """Heuristic check whether input matches this parser format."""
        pass

    @abstractmethod
    def parse(self, raw_input: Any) -> CanonicalConversation:
        """Parse raw input into CanonicalConversation."""
        pass
```

1. **Auto-Detection Strategy:**
   - If input is a dictionary or JSON string with `"messages"` and `"participants"` $\to$ **`InstagramJsonParser`**.
   - If input is a CSV string or contains commas/tabs with a header row containing `speaker` and `text`/`utterance` $\to$ **`TabularCsvParser`**.
   - If input lines match WhatsApp timestamp regex `^\[?\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}` $\to$ **`WhatsAppTxtParser`**.
   - Otherwise $\to$ fallback to **`PlainTextParser`** (`Speaker: utterance`).

2. **Integration into `ConversationAnalyzer`:**
   ```python
   # In src/application/analyzer.py
   def analyze(self, conversation: Any, mode: str = "utterance", format: str = "auto"):
       canonical_conv = parse_canonical(conversation, format=format)
       evaluable_turns = canonical_conv.get_evaluable_turns()
       
       model = self._get_model(mode)
       predictions = model.predict(evaluable_turns)
       analysis = analyze_predictions(predictions)
       
       return {
           "mode": mode,
           "model_metadata": MODEL_METADATA.get(mode, {}),
           "turns": predictions,
           "analysis": asdict(analysis),
           "canonical": {
               "conversation_id": canonical_conv.conversation_id,
               "source_format": canonical_conv.source_format.value,
               "warnings": [asdict(w) for w in canonical_conv.warnings],
               "total_messages": len(canonical_conv.messages),
               "evaluable_messages": len(evaluable_turns),
           }
       }
   ```

3. **Verification Plan:**
   - Create comprehensive unit test suite in `tests/test_parsers.py` testing:
     - Plain text multiline parsing and continuation.
     - Instagram JSON parsing with reverse-order inversion and Latin-1 mojibake repair.
     - WhatsApp iOS and Android timestamp variations.
     - Tabular CSV with arbitrary column orders.
     - Re-indexing and `to_inference_turn()` adapter contracts.
   - Run `tests/test_application_integration.py` to confirm zero regression across all 4 inference modes (`utterance`, `context_1`, `context_1_speaker`, `dialoguernn`).

---

## 8. Summary & Status

* **Audit Completed:** Verified every consumer in backend, inference, analysis, API, and frontend.
* **Canonical Schema Formulated:** Defined `CanonicalConversation` and `CanonicalMessage` addressing all constraints (timestamps, dynamic speaker IDs, media events, deterministic ordering, parser diagnostics).
* **Zero Disruption to Model Weights or MELD Training:** Model inference contracts and MELD data loaders remain 100% decoupled and preserved.
* **Constraints Respected:** No source code, Docker configs, or checkpoints were modified during this audit.

Awaiting user review and authorization before proceeding with the implementation of the parser suite.
