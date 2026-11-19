from __future__ import annotations

from datetime import datetime, timezone
import re
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from src.data.parsers.base import BaseConversationParser
from src.data.schema import (
    CanonicalConversation,
    CanonicalMessage,
    MessageType,
    ParserWarning,
    SourceFormat,
    SpeakerProfile,
    WarningSeverity,
)

# Strip invisible directional marks and non-standard spaces
INVISIBLE_CHARS_REGEX = re.compile(r"[\u200e\u200f\u202a\u202b\u202c\u202d\u202e]")
WHITESPACE_CLEAN_REGEX = re.compile(r"[\u202f\u00a0\xa0]")

# 1. Bracketed iOS/macOS pattern: [12/05/2026, 14:32:15] Speaker: Message
# or [12/05/2026, 2:32:15 PM] Speaker: Message
BRACKETED_START_REGEX = re.compile(
    r"^\["
    r"(\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4})"        # Date group (1)
    r",?\s+"
    r"(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp]\.?[Mm]\.?)?)" # Time group (2)
    r"\]\s*"
    r"(.*)$"                                     # Remainder group (3)
)

# 2. Dash Android/Windows/Web pattern: 12/05/2026, 14:32 - Speaker: Message
# or 12/05/2026, 2:32 pm - Speaker: Message
DASH_START_REGEX = re.compile(
    r"^"
    r"(\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4})"        # Date group (1)
    r",?\s+"
    r"(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp]\.?[Mm]\.?)?)" # Time group (2)
    r"\s*-\s*"
    r"(.*)$"                                     # Remainder group (3)
)

# Media omitted placeholders across Android / iOS
MEDIA_OMITTED_REGEX = re.compile(
    r"^(?:"
    r"<Media omitted>|"
    r"<attached:\s*[^>]+>|"
    r"image omitted|"
    r"video omitted|"
    r"audio omitted|"
    r"sticker omitted|"
    r"GIF omitted|"
    r"document omitted|"
    r"Contact card omitted|"
    r"Location:\s*https?://\S+"
    r")$",
    re.IGNORECASE,
)

# Attached prefix with caption: <attached: photo.jpg> Caption here
ATTACHED_CAPTION_REGEX = re.compile(r"^<attached:\s*([^>]+)>\s*(.*)$", re.IGNORECASE)

# System message indicator phrases
SYSTEM_PHRASES = [
    "messages and calls are end-to-end encrypted",
    "created group",
    "added",
    "removed",
    "left",
    "changed the group",
    "changed the subject",
    "changed the description",
    "changed this group's icon",
    "deleted this message",
    "you deleted this message",
    "this message was deleted",
    "missed voice call",
    "missed video call",
    "missed group call",
    "security code changed",
    "waiting for this message",
]


def clean_line_markers(line: str) -> str:
    """Strip invisible Unicode directional marks and normalize non-breaking spaces."""
    cleaned = INVISIBLE_CHARS_REGEX.sub("", line)
    cleaned = WHITESPACE_CLEAN_REGEX.sub(" ", cleaned)
    return cleaned.strip()


def infer_date_convention(raw_lines: List[str]) -> Tuple[str, bool]:
    """
    Scans timestamped lines to determine whether date format is
    Day-first (DD/MM/YYYY) or Month-first (MM/DD/YYYY).
    Returns (convention: 'DMY' | 'MDY' | 'YMD', is_ambiguous: bool).
    """
    saw_d_gt_12 = False
    saw_m_gt_12 = False
    saw_ymd = False

    for raw_line in raw_lines:
        line = clean_line_markers(raw_line)
        m = BRACKETED_START_REGEX.match(line) or DASH_START_REGEX.match(line)
        if not m:
            continue

        date_str = m.group(1)
        delims = [c for c in date_str if c in "/-."]
        if not delims:
            continue
        parts = date_str.replace("-", "/").replace(".", "/").split("/")
        if len(parts) != 3:
            continue

        try:
            p1, p2, p3 = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue

        if p1 > 1000:
            saw_ymd = True
            break
        if p1 > 12:
            saw_d_gt_12 = True
        if p2 > 12:
            saw_m_gt_12 = True

    if saw_ymd:
        return "YMD", False
    if saw_d_gt_12 and not saw_m_gt_12:
        return "DMY", False
    if saw_m_gt_12 and not saw_d_gt_12:
        return "MDY", False

    # If neither exceeded 12, worldwide WhatsApp convention defaults to DMY (Day/Month/Year)
    return "DMY", True


def parse_timestamp_components(
    date_str: str,
    time_str: str,
    convention: str,
) -> Optional[datetime]:
    """
    Converts WhatsApp date and time strings into timezone-aware UTC datetime.
    """
    parts = date_str.replace("-", "/").replace(".", "/").split("/")
    if len(parts) != 3:
        return None

    try:
        p1, p2, p3 = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None

    if convention == "YMD" or p1 > 1000:
        year, month, day = p1, p2, p3
    elif convention == "MDY":
        month, day, year = p1, p2, p3
    else:  # DMY
        day, month, year = p1, p2, p3

    # Handle 2-digit years
    if year < 100:
        year += 2000

    # Parse time: 14:32, 14:32:15, 2:32 PM, 02:32:15 am
    time_clean = WHITESPACE_CLEAN_REGEX.sub(" ", time_str).strip()
    is_pm = bool(re.search(r"[Pp]\.?[Mm]\.?", time_clean))
    is_am = bool(re.search(r"[Aa]\.?[Mm]\.?", time_clean))
    time_digits = re.sub(r"[^\d:]", "", time_clean)
    t_parts = time_digits.split(":")

    if len(t_parts) < 2:
        return None

    try:
        hour = int(t_parts[0])
        minute = int(t_parts[1])
        second = int(t_parts[2]) if len(t_parts) > 2 else 0
    except ValueError:
        return None

    if is_pm and hour < 12:
        hour += 12
    elif is_am and hour == 12:
        hour = 0

    try:
        return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
    except (ValueError, OverflowError):
        return None


class WhatsAppTxtParser(BaseConversationParser):
    """
    Parser for exported WhatsApp .txt chat logs (Android, iOS, macOS, Web).
    Handles:
    - Bracketed '[DD/MM/YYYY, HH:MM:SS] Speaker: text' (iOS)
    - Dash-separated 'DD/MM/YYYY, HH:MM - Speaker: text' (Android)
    - 12-hour and 24-hour time variations
    - Multiline utterances
    - System notifications (encryption notice, join/leave, deleted messages)
    - Media omitted placeholders (<Media omitted>, <attached: photo.jpg>, etc.)
    - Phone number senders and names with punctuation
    - Preserves source and chronological ordering
    """

    def can_parse(self, raw_input: Any) -> bool:
        if not isinstance(raw_input, str):
            return False
        if not raw_input.strip():
            return False

        # Check if any of the first 20 non-empty lines matches a WhatsApp timestamp pattern
        matches = 0
        for line in raw_input.splitlines()[:25]:
            sline = clean_line_markers(line)
            if BRACKETED_START_REGEX.match(sline) or DASH_START_REGEX.match(sline):
                matches += 1
                if matches >= 1:
                    return True
        return False

    def parse(
        self,
        raw_input: Any,
        conversation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> CanonicalConversation:
        if not isinstance(raw_input, str):
            raise ValueError("Conversation input must be text.")

        raw_text = raw_input
        conv_id = conversation_id or f"conv_wa_{uuid.uuid4().hex[:8]}"
        warnings: List[ParserWarning] = []

        if not raw_text.strip():
            warnings.append(
                ParserWarning(
                    code="EMPTY_INPUT",
                    message="Conversation input is empty or whitespace only.",
                    severity=WarningSeverity.ERROR,
                )
            )
            return CanonicalConversation(
                conversation_id=conv_id,
                source_format=SourceFormat.WHATSAPP_TXT,
                participants=[],
                messages=[],
                warnings=warnings,
                metadata={"has_timestamps": False, "raw_length": len(raw_text)},
            )

        raw_lines = raw_text.splitlines()
        date_convention, is_ambiguous = infer_date_convention(raw_lines)
        if is_ambiguous:
            warnings.append(
                ParserWarning(
                    code="AMBIGUOUS_DATE_FORMAT",
                    message=f"Date numbers could be either Day/Month or Month/Day. Inferred '{date_convention}'.",
                    severity=WarningSeverity.INFO,
                )
            )

        speaker_profiles: Dict[str, SpeakerProfile] = {}

        def get_or_create_speaker(name: str) -> str:
            norm = name.strip().lower()
            if norm not in speaker_profiles:
                spk_id = f"spk_{len(speaker_profiles)}"
                speaker_profiles[norm] = SpeakerProfile(
                    speaker_id=spk_id,
                    name=name.strip(),
                    normalized_name=norm,
                    turn_count=0,
                )
            speaker_profiles[norm].turn_count += 1
            return speaker_profiles[norm].speaker_id

        messages: List[CanonicalMessage] = []

        # State tracking for multi-line utterances
        current_speaker: Optional[str] = None
        current_speaker_id: Optional[str] = None
        current_text_parts: List[str] = []
        current_timestamp: Optional[datetime] = None
        current_message_type: MessageType = MessageType.TEXT
        current_metadata: Dict[str, Any] = {}
        current_line_start: Optional[int] = None
        current_is_evaluable: bool = True

        def flush_current_turn() -> None:
            nonlocal current_speaker, current_speaker_id, current_text_parts
            nonlocal current_timestamp, current_message_type, current_metadata
            nonlocal current_line_start, current_is_evaluable

            if current_speaker is None:
                return

            text = " ".join(part.strip() for part in current_text_parts if part.strip()).strip()
            msg_id = f"msg_{len(messages) + 1:04d}"
            src_idx = len(messages)

            # Check attached prefix: <attached: photo.jpg> Caption (or without caption)
            att_m = ATTACHED_CAPTION_REGEX.match(text)
            if att_m:
                att_file = att_m.group(1).strip()
                caption = att_m.group(2).strip()
                current_metadata["attachment"] = att_file
                if caption:
                    current_message_type = MessageType.TEXT
                    current_is_evaluable = True
                    text = caption
                else:
                    current_message_type = MessageType.MEDIA
                    current_metadata["media_placeholder"] = text
                    current_is_evaluable = False
                    text = ""
            elif MEDIA_OMITTED_REGEX.match(text):
                current_message_type = MessageType.MEDIA
                current_metadata["media_placeholder"] = text
                current_is_evaluable = False
                text = ""

            messages.append(
                CanonicalMessage(
                    conversation_id=conv_id,
                    message_id=msg_id,
                    source_index=src_idx,
                    sequence_index=src_idx,
                    speaker=current_speaker,
                    speaker_id=current_speaker_id or "spk_unknown",
                    text=text,
                    timestamp=current_timestamp,
                    message_type=current_message_type,
                    metadata=current_metadata,
                    is_evaluable=current_is_evaluable and bool(text),
                )
            )

            current_speaker = None
            current_speaker_id = None
            current_text_parts = []
            current_timestamp = None
            current_message_type = MessageType.TEXT
            current_metadata = {}
            current_line_start = None
            current_is_evaluable = True

        for line_number, raw_line in enumerate(raw_lines, start=1):
            line = clean_line_markers(raw_line)
            if not line:
                # Empty lines in multi-line messages
                continue

            match = BRACKETED_START_REGEX.match(line) or DASH_START_REGEX.match(line)

            if match:
                date_str = match.group(1)
                time_str = match.group(2)
                remainder = match.group(3).strip()

                dt = parse_timestamp_components(date_str, time_str, date_convention)
                if dt is None:
                    warnings.append(
                        ParserWarning(
                            code="MALFORMED_TIMESTAMP",
                            message=f"Line {line_number}: Unable to parse timestamp '{date_str} {time_str}'.",
                            severity=WarningSeverity.WARNING,
                            line_number=line_number,
                        )
                    )

                # Check if remainder has a sender delimiter
                # Format is typically "Speaker: utterance"
                is_user_message = False
                speaker_cand: Optional[str] = None
                text_cand: Optional[str] = None

                if ":" in remainder:
                    parts = remainder.split(":", 1)
                    cand_spk = parts[0].strip()
                    cand_txt = parts[1].strip()

                    # Validate that cand_spk does not match system announcement keywords
                    lower_rem = remainder.lower()
                    if not any(phrase in lower_rem for phrase in SYSTEM_PHRASES):
                        # Reasonable sender name length (1 to 50 characters)
                        if 1 <= len(cand_spk) <= 50:
                            is_user_message = True
                            speaker_cand = cand_spk
                            text_cand = cand_txt

                if is_user_message and speaker_cand:
                    # Flush preceding turn before starting a new user turn
                    flush_current_turn()

                    current_speaker = speaker_cand
                    current_speaker_id = get_or_create_speaker(speaker_cand)
                    current_text_parts = [text_cand] if text_cand else []
                    current_timestamp = dt
                    current_message_type = MessageType.TEXT
                    current_metadata = {"line_number": line_number, "raw_time": time_str}
                    current_line_start = line_number
                    current_is_evaluable = True

                else:
                    # System notification (e.g. encryption notice, user added, call missed)
                    flush_current_turn()

                    msg_id = f"msg_{len(messages) + 1:04d}"
                    src_idx = len(messages)
                    messages.append(
                        CanonicalMessage(
                            conversation_id=conv_id,
                            message_id=msg_id,
                            source_index=src_idx,
                            sequence_index=src_idx,
                            speaker="System",
                            speaker_id="spk_system",
                            text=remainder,
                            timestamp=dt,
                            message_type=MessageType.SYSTEM,
                            metadata={"line_number": line_number, "system_event": True},
                            is_evaluable=False,
                        )
                    )

            else:
                # Continuation line of current turn or unparsed header
                if current_speaker is not None:
                    current_text_parts.append(line)
                else:
                    # Line before first valid WhatsApp timestamp
                    msg_id = f"msg_{len(messages) + 1:04d}"
                    src_idx = len(messages)
                    warnings.append(
                        ParserWarning(
                            code="UNPARSED_LINE",
                            message=f"Line {line_number}: line before first WhatsApp message ignored: '{line[:60]}...'",
                            severity=WarningSeverity.WARNING,
                            line_number=line_number,
                            message_id=msg_id,
                        )
                    )
                    messages.append(
                        CanonicalMessage(
                            conversation_id=conv_id,
                            message_id=msg_id,
                            source_index=src_idx,
                            sequence_index=src_idx,
                            speaker="System",
                            speaker_id="spk_system",
                            text=line,
                            timestamp=None,
                            message_type=MessageType.UNPARSED,
                            metadata={"line_number": line_number},
                            is_evaluable=False,
                        )
                    )

        # Flush final turn
        flush_current_turn()

        # WhatsApp logs are natively exported chronologically; ensure sequence_index matches
        for idx, m in enumerate(messages):
            m.sequence_index = idx

        return CanonicalConversation(
            conversation_id=conv_id,
            source_format=SourceFormat.WHATSAPP_TXT,
            title=None,
            participants=list(speaker_profiles.values()),
            messages=messages,
            warnings=warnings,
            metadata={
                "has_timestamps": True,
                "date_convention": date_convention,
                "total_messages": len(messages),
                "evaluable_messages": len([m for m in messages if m.is_evaluable]),
                "num_speakers": len(speaker_profiles),
            },
        )
