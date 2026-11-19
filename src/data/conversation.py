from __future__ import annotations

from typing import Any, List

from src.data.schema import (
    CanonicalConversation,
    CanonicalMessage,
    ConversationTurn,
    MessageType,
    ParserWarning,
    SourceFormat,
    SpeakerProfile,
    WarningSeverity,
)
from src.data.parsers import parse_canonical


class ConversationParseError(ValueError):
    """Raised when conversation input cannot be parsed safely."""


def parse_canonical_conversation(
    raw_text: str,
    format_hint: str = "auto",
    **kwargs: Any,
) -> CanonicalConversation:
    """
    Parse raw text into a CanonicalConversation.
    Validates input type and non-empty content.
    """
    if not isinstance(raw_text, str):
        raise ConversationParseError("Conversation input must be text.")

    if not raw_text.strip():
        raise ConversationParseError("Conversation is empty.")

    return parse_canonical(raw_text, format_hint=format_hint, **kwargs)


def parse_conversation(raw_text: str) -> List[ConversationTurn]:
    """
    Parse a conversation written as 'Speaker: utterance'.
    Returns list of ConversationTurn for backward compatibility.

    Rules:
    - Turn IDs are generated automatically (0-indexed contiguous).
    - Blank lines are ignored.
    - Lines without a speaker marker continue the previous utterance.
    - Unparsed lines before the first speaker are captured as warnings.
    - If no evaluable turns are found, raises ConversationParseError.
    """
    canonical = parse_canonical_conversation(raw_text, format_hint="plain_text")
    turns = canonical.get_evaluable_turns()

    if not turns:
        raise ConversationParseError("No valid conversation turns were found.")

    return turns
