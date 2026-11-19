from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List, Optional, Set

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

# Non-speaker continuation keywords that should not be treated as speaker turns
CONTINUATION_PREFIXES: Set[str] = {
    "note",
    "notes",
    "ps",
    "p.s.",
    "warning",
    "caution",
    "tip",
    "hint",
    "example",
    "e.g.",
    "i.e.",
    "ref",
    "reference",
    "source",
    "reason",
    "step",
    "time",
    "date",
}

# URL scheme patterns that should never be parsed as speakers
URL_SCHEME_PATTERN = re.compile(r"^(https?|ftp|file|mailto|ssh)://", re.IGNORECASE)


class PlainTextParser(BaseConversationParser):
    """
    Parser for plain text conversations in 'Speaker: utterance' format.
    Supports multiline utterances, multiple speakers, colons in utterances,
    blank lines, Unicode/emojis, and preserves unparsed lines as warnings.
    """

    def can_parse(self, raw_input: Any) -> bool:
        if not isinstance(raw_input, str):
            return False
        if not raw_input.strip():
            return False
        # If there's at least one line with ":" not matching a URL pattern
        for line in raw_input.splitlines():
            sline = line.strip()
            if ":" in sline and not URL_SCHEME_PATTERN.match(sline):
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
        conv_id = conversation_id or f"conv_{uuid.uuid4().hex[:8]}"

        warnings: List[ParserWarning] = []
        messages: List[CanonicalMessage] = []
        speaker_profiles: Dict[str, SpeakerProfile] = {}
        known_speakers_lower: Set[str] = set()

        if not raw_text.strip():
            # Return empty canonical conversation with warning
            warnings.append(
                ParserWarning(
                    code="EMPTY_INPUT",
                    message="Conversation input is empty or whitespace only.",
                    severity=WarningSeverity.ERROR,
                )
            )
            return CanonicalConversation(
                conversation_id=conv_id,
                source_format=SourceFormat.PLAIN_TEXT,
                participants=[],
                messages=[],
                warnings=warnings,
                metadata={"has_timestamps": False, "raw_length": len(raw_text)},
            )

        # Informational warning: plain text has no native timestamps
        warnings.append(
            ParserWarning(
                code="MISSING_TIMESTAMPS",
                message="Conversation does not contain timestamp metadata; ordering is based on input sequence.",
                severity=WarningSeverity.INFO,
            )
        )

        current_speaker: Optional[str] = None
        current_speaker_id: Optional[str] = None
        current_text_parts: List[str] = []
        current_line_start: Optional[int] = None

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
                known_speakers_lower.add(norm)
            speaker_profiles[norm].turn_count += 1
            return speaker_profiles[norm].speaker_id

        def flush_current_turn() -> None:
            nonlocal current_speaker, current_speaker_id, current_text_parts, current_line_start

            if current_speaker is None:
                return

            text = " ".join(part.strip() for part in current_text_parts if part.strip()).strip()
            msg_id = f"msg_{len(messages) + 1:04d}"
            src_idx = len(messages)

            if not text:
                warnings.append(
                    ParserWarning(
                        code="EMPTY_UTTERANCE",
                        message=f"Speaker '{current_speaker}' has empty text utterance at line {current_line_start}.",
                        severity=WarningSeverity.WARNING,
                        line_number=current_line_start,
                        message_id=msg_id,
                    )
                )
                # Keep record in messages with is_evaluable=False
                messages.append(
                    CanonicalMessage(
                        conversation_id=conv_id,
                        message_id=msg_id,
                        source_index=src_idx,
                        sequence_index=src_idx,
                        speaker=current_speaker,
                        speaker_id=current_speaker_id or "spk_unknown",
                        text="",
                        timestamp=None,
                        message_type=MessageType.TEXT,
                        metadata={"line_number": current_line_start},
                        is_evaluable=False,
                    )
                )
            else:
                messages.append(
                    CanonicalMessage(
                        conversation_id=conv_id,
                        message_id=msg_id,
                        source_index=src_idx,
                        sequence_index=src_idx,
                        speaker=current_speaker,
                        speaker_id=current_speaker_id or "spk_unknown",
                        text=text,
                        timestamp=None,
                        message_type=MessageType.TEXT,
                        metadata={"line_number": current_line_start},
                        is_evaluable=True,
                    )
                )

            current_speaker = None
            current_speaker_id = None
            current_text_parts = []
            current_line_start = None

        lines = raw_text.splitlines()
        for line_number, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            if not line:
                # Blank lines separate paragraphs or utterances
                continue

            # Check if this line introduces a speaker turn
            is_new_speaker_turn = False
            candidate_speaker: Optional[str] = None
            candidate_text: Optional[str] = None

            if ":" in line:
                parts = line.split(":", 1)
                cand_spk = parts[0].strip()
                cand_txt = parts[1].strip()

                # Determine if cand_spk is a genuine speaker name
                cand_lower = cand_spk.lower()

                # Rule 1: Cannot be URL scheme
                if URL_SCHEME_PATTERN.match(line):
                    is_new_speaker_turn = False

                # Rule 2: If we are already inside an utterance, check continuation keywords
                elif current_speaker is not None and cand_lower in CONTINUATION_PREFIXES:
                    is_new_speaker_turn = False

                # Rule 3: Speaker name must be reasonable length (1 to 80 chars)
                elif 0 < len(cand_spk) <= 80:
                    # If already known speaker, definitely a turn
                    if cand_lower in known_speakers_lower:
                        is_new_speaker_turn = True
                        candidate_speaker = cand_spk
                        candidate_text = cand_txt
                    # If candidate doesn't contain newline or sentence delimiters
                    elif not any(char in cand_spk for char in [". ", "? ", "! ", ";"]):
                        is_new_speaker_turn = True
                        candidate_speaker = cand_spk
                        candidate_text = cand_txt

            if is_new_speaker_turn and candidate_speaker is not None:
                # Flush previous turn before starting new one
                flush_current_turn()

                current_speaker = candidate_speaker
                current_speaker_id = get_or_create_speaker(candidate_speaker)
                current_line_start = line_number
                current_text_parts = [candidate_text] if candidate_text else []

            else:
                # Continuation of current utterance or unparsed line before any speaker
                if current_speaker is not None:
                    current_text_parts.append(line)
                else:
                    # Unrecognized line before any speaker was established
                    msg_id = f"msg_{len(messages) + 1:04d}"
                    src_idx = len(messages)
                    warnings.append(
                        ParserWarning(
                            code="UNPARSED_LINE",
                            message=f"Line {line_number}: line before first speaker turn ignored for inference: '{line[:60]}...'",
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

        # Build final CanonicalConversation
        conversation = CanonicalConversation(
            conversation_id=conv_id,
            source_format=SourceFormat.PLAIN_TEXT,
            participants=list(speaker_profiles.values()),
            messages=messages,
            warnings=warnings,
            metadata={
                "has_timestamps": False,
                "total_messages": len(messages),
                "evaluable_messages": len([m for m in messages if m.is_evaluable]),
                "num_speakers": len(speaker_profiles),
            },
        )

        return conversation
