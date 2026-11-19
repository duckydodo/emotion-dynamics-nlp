from __future__ import annotations

from datetime import datetime, timezone
import json
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


def attempt_mojibake_repair(text: str) -> Tuple[str, bool]:
    """
    Attempts to repair legacy Meta export encoding where UTF-8 bytes were
    erroneously decoded as Latin-1 (e.g. \u00e2\u0080\u0099 -> ’ or \u00f0\u009f... -> emoji).
    Returns (repaired_text, was_repaired).
    Leaves clean UTF-8 or ASCII strings untouched.
    """
    if not text:
        return text, False

    # If all characters are standard ASCII (< 128), no repair needed
    if all(ord(c) < 128 for c in text):
        return text, False

    try:
        # If text contains Latin-1 representation of UTF-8 bytes:
        # encoding to latin1 recovers original bytes, decoding as utf-8 recovers real text.
        repaired = text.encode("latin1").decode("utf-8")
        if repaired != text:
            return repaired, True
        return text, False
    except (UnicodeEncodeError, UnicodeDecodeError):
        # Cannot be cleanly re-encoded to Latin-1 or decoded as UTF-8; keep original text
        return text, False


def parse_timestamp(msg_dict: Dict[str, Any]) -> Tuple[Optional[datetime], Optional[Dict[str, Any]], Optional[str]]:
    """
    Extracts and normalizes timestamp to timezone-aware UTC datetime.
    Supports timestamp_ms (integer ms), timestamp (integer seconds or ms), or ISO string.
    Returns (normalized_datetime, metadata_dict, warning_message).
    """
    # 1. timestamp_ms (standard Meta Instagram export format)
    if "timestamp_ms" in msg_dict and msg_dict["timestamp_ms"] is not None:
        try:
            ts_ms = int(msg_dict["timestamp_ms"])
            # Validate plausible millisecond range (between year 2000 and 2100)
            if 946684800000 <= ts_ms <= 4102444800000:
                dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
                return dt, {"timestamp_ms": ts_ms, "timestamp_unit": "milliseconds"}, None
            elif ts_ms < 946684800000:
                # Value was actually in seconds despite the key name
                dt = datetime.fromtimestamp(ts_ms, tz=timezone.utc)
                return dt, {"timestamp_raw": ts_ms, "timestamp_unit": "seconds_fallback"}, None
        except (ValueError, TypeError, OverflowError):
            pass

    # 2. timestamp (could be seconds or ms)
    if "timestamp" in msg_dict and msg_dict["timestamp"] is not None:
        val = msg_dict["timestamp"]
        if isinstance(val, (int, float)):
            try:
                # Milliseconds range
                if val > 10_000_000_000:
                    dt = datetime.fromtimestamp(val / 1000.0, tz=timezone.utc)
                    return dt, {"timestamp_raw": val, "timestamp_unit": "milliseconds"}, None
                else:
                    # Seconds range
                    dt = datetime.fromtimestamp(val, tz=timezone.utc)
                    return dt, {"timestamp_raw": val, "timestamp_unit": "seconds"}, None
            except (ValueError, TypeError, OverflowError):
                pass
        elif isinstance(val, str):
            try:
                dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt, {"timestamp_iso": val}, None
            except ValueError:
                pass

    # 3. created_at (string or timestamp)
    if "created_at" in msg_dict and msg_dict["created_at"] is not None:
        val = msg_dict["created_at"]
        if isinstance(val, str):
            try:
                dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt, {"timestamp_iso": val}, None
            except ValueError:
                pass

    return None, None, "Message is missing a valid timestamp."


class InstagramJsonParser(BaseConversationParser):
    """
    Parser for Meta Instagram Direct Message JSON exports (e.g. message_1.json).
    Supports:
    - Multiple participants list formats (objects or strings)
    - Chronological reordering (Meta exports are reverse-chronological newest-first)
    - Timezone-aware UTC timestamp normalization
    - Latin-1 mojibake encoding repair
    - Media attachments (photos, videos, audio, stickers, shares) flagged is_evaluable=False
    - System events and reactions
    - Duplicate detection and missing sender/text validation
    """

    def can_parse(self, raw_input: Any) -> bool:
        if isinstance(raw_input, dict):
            return "messages" in raw_input and isinstance(raw_input["messages"], list)

        if isinstance(raw_input, str):
            s = raw_input.strip()
            if s.startswith("{") and s.endswith("}"):
                try:
                    data = json.loads(s)
                    return isinstance(data, dict) and "messages" in data and isinstance(data["messages"], list)
                except Exception:
                    return False
        return False

    def parse(
        self,
        raw_input: Any,
        conversation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> CanonicalConversation:
        conv_id = conversation_id or f"conv_ig_{uuid.uuid4().hex[:8]}"
        warnings: List[ParserWarning] = []

        # 1. Parse JSON if string
        data: Dict[str, Any] = {}
        if isinstance(raw_input, str):
            try:
                data = json.loads(raw_input)
            except json.JSONDecodeError as exc:
                warnings.append(
                    ParserWarning(
                        code="MALFORMED_JSON",
                        message=f"JSON decoding error: {exc.msg} at line {exc.lineno} column {exc.colno}",
                        severity=WarningSeverity.ERROR,
                    )
                )
                return CanonicalConversation(
                    conversation_id=conv_id,
                    source_format=SourceFormat.INSTAGRAM_JSON,
                    participants=[],
                    messages=[],
                    warnings=warnings,
                    metadata={"error": "Malformed JSON payload"},
                )
        elif isinstance(raw_input, dict):
            data = raw_input
        else:
            warnings.append(
                ParserWarning(
                    code="UNSUPPORTED_STRUCTURE",
                    message="Instagram import payload must be a JSON string or dict.",
                    severity=WarningSeverity.ERROR,
                )
            )
            return CanonicalConversation(
                conversation_id=conv_id,
                source_format=SourceFormat.INSTAGRAM_JSON,
                participants=[],
                messages=[],
                warnings=warnings,
                metadata={"error": "Invalid payload type"},
            )

        # 2. Validate top-level keys
        if "messages" not in data or not isinstance(data["messages"], list):
            warnings.append(
                ParserWarning(
                    code="MISSING_MESSAGE_ARRAY",
                    message="Instagram JSON export is missing a top-level 'messages' array.",
                    severity=WarningSeverity.ERROR,
                )
            )
            return CanonicalConversation(
                conversation_id=conv_id,
                source_format=SourceFormat.INSTAGRAM_JSON,
                participants=[],
                messages=[],
                warnings=warnings,
                metadata={"error": "Missing messages list"},
            )

        raw_messages: List[Dict[str, Any]] = data["messages"]
        title: Optional[str] = data.get("title")
        if title:
            title_repaired, _ = attempt_mojibake_repair(title)
            title = title_repaired

        # 3. Discover participants
        speaker_profiles: Dict[str, SpeakerProfile] = {}

        def get_or_create_speaker(name: str) -> str:
            clean_name = name.strip() if name and name.strip() else "Unknown"
            repaired_name, _ = attempt_mojibake_repair(clean_name)
            norm = repaired_name.lower()
            if norm not in speaker_profiles:
                spk_id = f"spk_{len(speaker_profiles)}"
                speaker_profiles[norm] = SpeakerProfile(
                    speaker_id=spk_id,
                    name=repaired_name,
                    normalized_name=norm,
                    turn_count=0,
                )
            return speaker_profiles[norm].speaker_id

        # Register top-level participants if present
        raw_participants = data.get("participants", [])
        if isinstance(raw_participants, list):
            for p in raw_participants:
                if isinstance(p, dict) and "name" in p:
                    get_or_create_speaker(p["name"])
                elif isinstance(p, str):
                    get_or_create_speaker(p)

        # 4. Parse messages
        parsed_items: List[CanonicalMessage] = []
        mojibake_repairs_count = 0
        mojibake_active = False
        seen_signatures: Set[Tuple[str, Optional[int], str]] = set()

        for src_idx, raw_msg in enumerate(raw_messages):
            msg_id = f"msg_{src_idx + 1:04d}"

            if not isinstance(raw_msg, dict):
                warnings.append(
                    ParserWarning(
                        code="INVALID_MESSAGE_RECORD",
                        message=f"Item at index {src_idx} is not a JSON object.",
                        severity=WarningSeverity.WARNING,
                        message_id=msg_id,
                    )
                )
                continue

            # Sender identification
            sender_name = raw_msg.get("sender_name") or raw_msg.get("sender") or raw_msg.get("actor")
            if not sender_name or not str(sender_name).strip():
                warnings.append(
                    ParserWarning(
                        code="MISSING_SENDER",
                        message=f"Message {msg_id} is missing a sender; defaulting to 'Unknown'.",
                        severity=WarningSeverity.WARNING,
                        message_id=msg_id,
                    )
                )
                sender_name = "Unknown"

            spk_id = get_or_create_speaker(str(sender_name))
            display_speaker = speaker_profiles[str(sender_name).strip().lower()].name

            # Timestamp extraction
            dt, ts_meta, ts_warn = parse_timestamp(raw_msg)
            if ts_warn:
                warnings.append(
                    ParserWarning(
                        code="MISSING_TIMESTAMP",
                        message=f"Message {msg_id}: {ts_warn}",
                        severity=WarningSeverity.WARNING,
                        message_id=msg_id,
                    )
                )

            # Content & Attachment extraction
            content_raw = raw_msg.get("content") or raw_msg.get("text")
            content_text = ""
            if content_raw is not None:
                repaired_content, was_repaired = attempt_mojibake_repair(str(content_raw))
                if was_repaired:
                    mojibake_repairs_count += 1
                    mojibake_active = True
                content_text = repaired_content.strip()

            msg_meta: Dict[str, Any] = {}
            if ts_meta:
                msg_meta.update(ts_meta)

            # Check media attachments
            has_media = False
            media_types: List[str] = []
            for media_key in ("photos", "videos", "audio_files", "files"):
                if media_key in raw_msg and isinstance(raw_msg[media_key], list) and raw_msg[media_key]:
                    has_media = True
                    media_types.append(media_key)
                    msg_meta[media_key] = raw_msg[media_key]

            if "sticker" in raw_msg and raw_msg["sticker"]:
                has_media = True
                media_types.append("sticker")
                msg_meta["sticker"] = raw_msg["sticker"]

            if "share" in raw_msg and raw_msg["share"]:
                has_media = True
                media_types.append("share")
                msg_meta["share"] = raw_msg["share"]

            # Reactions extraction
            if "reactions" in raw_msg and isinstance(raw_msg["reactions"], list):
                reactions_list = []
                for rx in raw_msg["reactions"]:
                    if isinstance(rx, dict):
                        rx_reaction = rx.get("reaction", "")
                        rx_actor = rx.get("actor", "")
                        repaired_rx, _ = attempt_mojibake_repair(rx_reaction)
                        repaired_actor, _ = attempt_mojibake_repair(rx_actor)
                        reactions_list.append({"reaction": repaired_rx, "actor": repaired_actor})
                if reactions_list:
                    msg_meta["reactions"] = reactions_list

            # System events / Unsent flags
            is_system = False
            if raw_msg.get("is_geoblocked_for_viewer") or raw_msg.get("is_unsent"):
                msg_meta["is_unsent"] = raw_msg.get("is_unsent", False)

            if raw_msg.get("type") in ("Generic", "Call", "Subscribe", "Unsubscribe") and not content_text and not has_media:
                is_system = True

            # Determine MessageType and is_evaluable
            if is_system:
                msg_type = MessageType.SYSTEM
                is_eval = False
            elif content_text:
                msg_type = MessageType.TEXT
                is_eval = True
                speaker_profiles[str(sender_name).strip().lower()].turn_count += 1
            elif has_media:
                msg_type = MessageType.MEDIA
                is_eval = False
                msg_meta["media_types"] = media_types
            elif "reactions" in msg_meta and not content_text:
                msg_type = MessageType.REACTION
                is_eval = False
            else:
                msg_type = MessageType.TEXT
                is_eval = False
                warnings.append(
                    ParserWarning(
                        code="EMPTY_MESSAGE",
                        message=f"Message {msg_id} from '{display_speaker}' has no text content or attachments.",
                        severity=WarningSeverity.INFO,
                        message_id=msg_id,
                    )
                )

            # Duplicate detection check
            ts_sig = raw_msg.get("timestamp_ms") or raw_msg.get("timestamp")
            sig = (display_speaker, ts_sig, content_text)
            if sig in seen_signatures:
                warnings.append(
                    ParserWarning(
                        code="DUPLICATE_RECORD",
                        message=f"Potential duplicate message {msg_id} detected for speaker '{display_speaker}'.",
                        severity=WarningSeverity.INFO,
                        message_id=msg_id,
                    )
                )
            seen_signatures.add(sig)

            parsed_items.append(
                CanonicalMessage(
                    conversation_id=conv_id,
                    message_id=msg_id,
                    source_index=src_idx,
                    sequence_index=src_idx,  # Updated after sorting
                    speaker=display_speaker,
                    speaker_id=spk_id,
                    text=content_text,
                    timestamp=dt,
                    message_type=msg_type,
                    metadata=msg_meta,
                    is_evaluable=is_eval,
                )
            )

        if mojibake_active:
            warnings.append(
                ParserWarning(
                    code="MOJIBAKE_REPAIRED",
                    message=f"Repaired Latin-1 mojibake encoding for {mojibake_repairs_count} text items.",
                    severity=WarningSeverity.INFO,
                )
            )

        # 5. Chronological Sorting
        # Meta exports are newest-first (reverse chronological).
        # We sort messages ascending by timestamp.
        # Tie-breaker: if timestamps are identical or missing, invert source index
        # (len(parsed_items) - 1 - source_index) so that earlier messages in the
        # conversation arrive before later messages.
        total_items = len(parsed_items)
        has_any_timestamps = any(m.timestamp is not None for m in parsed_items)

        if has_any_timestamps:
            def sort_key(m: CanonicalMessage):
                ts_val = m.timestamp.timestamp() if m.timestamp is not None else float("-inf")
                # Inverted source index ensures stable chronological sequence for identical timestamps
                inv_src = total_items - 1 - m.source_index
                return (ts_val, inv_src)

            sorted_messages = sorted(parsed_items, key=sort_key)
            # Check if order changed
            if any(sorted_messages[i].source_index != i for i in range(total_items)):
                warnings.append(
                    ParserWarning(
                        code="CHRONOLOGICAL_SORT_APPLIED",
                        message="Reordered messages into ascending chronological sequence using timestamps.",
                        severity=WarningSeverity.INFO,
                    )
                )
        else:
            # If no timestamps exist, invert source array assuming standard Meta newest-first export
            sorted_messages = list(reversed(parsed_items))
            warnings.append(
                ParserWarning(
                    code="MISSING_TIMESTAMPS_INVERTED",
                    message="No timestamps found; inverted newest-first array to restore chronological sequence.",
                    severity=WarningSeverity.WARNING,
                )
            )

        # Re-assign sequential sequence_index and message_id
        for seq_idx, msg in enumerate(sorted_messages):
            msg.sequence_index = seq_idx
            msg.message_id = f"msg_{seq_idx + 1:04d}"

        # 6. Build final CanonicalConversation
        return CanonicalConversation(
            conversation_id=conv_id,
            source_format=SourceFormat.INSTAGRAM_JSON,
            title=title,
            participants=list(speaker_profiles.values()),
            messages=sorted_messages,
            warnings=warnings,
            metadata={
                "has_timestamps": has_any_timestamps,
                "total_messages": len(sorted_messages),
                "evaluable_messages": len([m for m in sorted_messages if m.is_evaluable]),
                "num_speakers": len(speaker_profiles),
                "encoding_strategy": "latin1_mojibake_repair" if mojibake_active else "utf8_direct",
            },
        )
