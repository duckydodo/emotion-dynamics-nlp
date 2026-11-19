from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class MessageType(str, Enum):
    """Categorical classification of conversational items."""
    TEXT = "text"          # Standard spoken/written utterance
    MEDIA = "media"        # Photo, video, audio, sticker, file
    SYSTEM = "system"      # System events (join, leave, call, encryption)
    REACTION = "reaction"  # Emoji reactions
    UNPARSED = "unparsed"  # Unparsed or malformed lines preserved for transparency


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
    code: str
    message: str
    severity: WarningSeverity = WarningSeverity.WARNING
    line_number: Optional[int] = None
    message_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value if isinstance(self.severity, WarningSeverity) else str(self.severity),
            "line_number": self.line_number,
            "message_id": self.message_id,
        }


@dataclass
class SpeakerProfile:
    """Dynamically generated speaker profile."""
    speaker_id: str            # Normalized machine ID: "spk_0", "spk_1"
    name: str                  # Human-readable display name: "Alice"
    normalized_name: str       # Lowercase stripped: "alice"
    turn_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConversationTurn:
    """
    Downstream inference turn format.
    Preserved for 100% backward compatibility with EmotionInference and DialogueRNNInference.
    """
    turn_id: int
    speaker: str
    text: str


@dataclass
class CanonicalMessage:
    """Single canonical message item representing an utterance, event, or unparsed line."""
    conversation_id: str
    message_id: str                        # Unique ID (e.g., "msg_0001")
    source_index: int                      # Position in raw input (0-indexed)
    sequence_index: int                    # Chronological position (0-indexed)
    speaker: str                           # Display name ("Alice" or "System")
    speaker_id: str                        # Normalized party ID ("spk_0")
    text: str                              # Utterance text
    timestamp: Optional[datetime] = None   # ISO-8601 UTC timestamp or None
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "message_id": self.message_id,
            "source_index": self.source_index,
            "sequence_index": self.sequence_index,
            "speaker": self.speaker,
            "speaker_id": self.speaker_id,
            "text": self.text,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "message_type": self.message_type.value if isinstance(self.message_type, MessageType) else str(self.message_type),
            "metadata": self.metadata,
            "is_evaluable": self.is_evaluable,
        }


MAX_MESSAGES_PER_CONVERSATION: int = 1000
MAX_MESSAGE_CHARACTERS: int = 4000
MAX_SPEAKER_NAME_LENGTH: int = 64


@dataclass
class CanonicalConversation:
    """Complete canonical conversation container."""
    conversation_id: str
    source_format: SourceFormat = SourceFormat.PLAIN_TEXT
    title: Optional[str] = None
    participants: List[SpeakerProfile] = field(default_factory=list)
    messages: List[CanonicalMessage] = field(default_factory=list)
    warnings: List[ParserWarning] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def enforce_limits(
        self,
        max_messages: int = MAX_MESSAGES_PER_CONVERSATION,
        max_chars: int = MAX_MESSAGE_CHARACTERS,
        max_speaker_len: int = MAX_SPEAKER_NAME_LENGTH,
    ) -> None:
        """
        Enforces defensive system bounds to prevent memory exhaustion
        and excessive latency during neural inference.
        """
        # 1. Truncate message count if exceeded
        if len(self.messages) > max_messages:
            excess = len(self.messages) - max_messages
            self.messages = self.messages[:max_messages]
            self.warnings.append(
                ParserWarning(
                    code="MESSAGE_COUNT_LIMIT_EXCEEDED",
                    message=f"Conversation exceeded maximum limit of {max_messages} messages. Truncated {excess} trailing messages.",
                    severity=WarningSeverity.WARNING,
                )
            )

        # 2. Truncate text and speaker lengths
        for msg in self.messages:
            if len(msg.text) > max_chars:
                msg.text = msg.text[:max_chars]
                self.warnings.append(
                    ParserWarning(
                        code="MESSAGE_TEXT_TRUNCATED",
                        message=f"Message {msg.message_id} text exceeded {max_chars} characters and was truncated.",
                        severity=WarningSeverity.INFO,
                        message_id=msg.message_id,
                    )
                )

            if len(msg.speaker) > max_speaker_len:
                msg.speaker = msg.speaker[:max_speaker_len]

        # 3. Truncate participant names
        for p in self.participants:
            if len(p.name) > max_speaker_len:
                p.name = p.name[:max_speaker_len]

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

    def validate(self) -> List[str]:
        """
        Validates conversation consistency.
        Returns a list of validation error descriptions (empty list if valid).
        """
        errors: List[str] = []

        if not self.messages:
            errors.append("Empty conversation: conversation contains no messages.")
            return errors

        seen_ids: set[str] = set()
        for idx, msg in enumerate(self.messages):
            # Check duplicate message IDs
            if msg.message_id in seen_ids:
                errors.append(f"Duplicate message_id '{msg.message_id}' at index {idx}.")
            seen_ids.add(msg.message_id)

            # Check missing speaker for evaluable text messages
            if msg.is_evaluable and not msg.speaker.strip():
                errors.append(f"Message '{msg.message_id}' (index {idx}) is missing a speaker.")

            # Check empty text for evaluable messages
            if msg.is_evaluable and not msg.text.strip():
                errors.append(f"Evaluable message '{msg.message_id}' from '{msg.speaker}' has empty text.")

        # Check speaker count
        evaluable_speakers = {
            msg.speaker.strip()
            for msg in self.messages
            if msg.is_evaluable and msg.speaker.strip()
        }
        if not evaluable_speakers:
            errors.append("No valid speakers identified in evaluable conversation turns.")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Safe JSON-serializable representation."""
        return {
            "conversation_id": self.conversation_id,
            "source_format": self.source_format.value if isinstance(self.source_format, SourceFormat) else str(self.source_format),
            "title": self.title,
            "participants": [p.to_dict() for p in self.participants],
            "messages": [m.to_dict() for m in self.messages],
            "warnings": [w.to_dict() for w in self.warnings],
            "metadata": self.metadata,
        }
