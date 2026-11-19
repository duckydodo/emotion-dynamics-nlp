from __future__ import annotations

from typing import Any, Optional

from src.data.parsers.base import BaseConversationParser
from src.data.parsers.plain_text import PlainTextParser
from src.data.parsers.instagram import InstagramJsonParser
from src.data.parsers.whatsapp import WhatsAppTxtParser
from src.data.parsers.csv_parser import TabularCsvParser
from src.data.schema import CanonicalConversation, SourceFormat

_INSTAGRAM_PARSER = InstagramJsonParser()
_WHATSAPP_PARSER = WhatsAppTxtParser()
_CSV_PARSER = TabularCsvParser()
_PLAIN_TEXT_PARSER = PlainTextParser()


def get_parser(
    format_hint: str = "auto",
    raw_input: Optional[Any] = None,
) -> BaseConversationParser:
    """
    Return appropriate parser based on format hint or content auto-detection.
    """
    norm = format_hint.lower().strip() if format_hint else "auto"

    if norm in ("csv", "tabular_csv", "tsv"):
        return _CSV_PARSER
    elif norm in ("whatsapp", "whatsapp_txt", "wa"):
        return _WHATSAPP_PARSER
    elif norm in ("instagram", "instagram_json", "ig", "ig_json"):
        return _INSTAGRAM_PARSER
    elif norm in ("plain_text", "text", "txt"):
        return _PLAIN_TEXT_PARSER

    # Auto-detection
    if raw_input is not None:
        if _INSTAGRAM_PARSER.can_parse(raw_input):
            return _INSTAGRAM_PARSER
        if _WHATSAPP_PARSER.can_parse(raw_input):
            return _WHATSAPP_PARSER
        if _CSV_PARSER.can_parse(raw_input):
            return _CSV_PARSER

    # Default fallback
    return _PLAIN_TEXT_PARSER


def parse_canonical(
    raw_input: Any,
    format_hint: str = "auto",
    conversation_id: Optional[str] = None,
    **kwargs: Any,
) -> CanonicalConversation:
    """
    Parse raw conversation content into a CanonicalConversation.
    Auto-detects format if format_hint is 'auto'.
    Passes optional kwargs (e.g. column_mapping) to the parser.
    """
    parser = get_parser(format_hint=format_hint, raw_input=raw_input)
    conv = parser.parse(raw_input, conversation_id=conversation_id, **kwargs)
    conv.enforce_limits()
    return conv
