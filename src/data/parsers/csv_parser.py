from __future__ import annotations

import csv
from datetime import datetime, timezone
import io
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

# Canonical target fields
TARGET_SPEAKER = "speaker"
TARGET_TEXT = "text"
TARGET_TIMESTAMP = "timestamp"
TARGET_MESSAGE_ID = "message_id"

# Aliases for automatic column matching (case-insensitive)
SPEAKER_ALIASES = [
    "speaker",
    "sender",
    "author",
    "user",
    "from",
    "character",
    "name",
    "person",
    "party",
]

TEXT_ALIASES = [
    "text",
    "utterance",
    "message",
    "content",
    "body",
    "dialogue",
    "line",
    "sentence",
]

TIMESTAMP_ALIASES = [
    "timestamp",
    "time",
    "date",
    "datetime",
    "created_at",
    "timestamp_ms",
    "starttime",
    "start_time",
]

MESSAGE_ID_ALIASES = [
    "message_id",
    "msg_id",
    "id",
    "turn_id",
    "utterance_id",
    "index",
    "sr no.",
    "sr no",
    "sr_no",
]


def detect_delimiter(sample_text: str) -> str:
    """
    Heuristically determines CSV delimiter (',', '\t', ';', '|')
    from sample text while respecting quotes.
    """
    # Count occurrences outside quotes on first line
    first_line = sample_text.splitlines()[0] if sample_text.splitlines() else ""
    in_quote = False
    counts = {",": 0, "\t": 0, ";": 0, "|": 0}

    for char in first_line:
        if char == '"':
            in_quote = not in_quote
        elif not in_quote and char in counts:
            counts[char] += 1

    # Choose delimiter with highest occurrence
    best_delim = max(counts, key=counts.get)
    if counts[best_delim] > 0:
        return best_delim

    # Default fallback
    return ","


def parse_csv_timestamp(ts_val: Any) -> Tuple[Optional[datetime], Optional[Dict[str, Any]], Optional[str]]:
    """
    Parses timestamps from CSV cell values (ISO strings, epoch seconds/ms, MELD time formats).
    Returns (datetime, metadata_dict, error_message).
    """
    if ts_val is None:
        return None, None, None

    s = str(ts_val).strip()
    if not s:
        return None, None, None

    # 1. Numeric epoch (seconds or milliseconds)
    try:
        num = float(s)
        if num > 10_000_000_000:
            # Milliseconds
            dt = datetime.fromtimestamp(num / 1000.0, tz=timezone.utc)
            return dt, {"timestamp_raw": s, "unit": "milliseconds"}, None
        elif num > 946684800:
            # Seconds (after year 2000)
            dt = datetime.fromtimestamp(num, tz=timezone.utc)
            return dt, {"timestamp_raw": s, "unit": "seconds"}, None
    except (ValueError, OverflowError):
        pass

    # 2. ISO-8601 strings
    try:
        clean_iso = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean_iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt, {"timestamp_raw": s, "unit": "iso"}, None
    except ValueError:
        pass

    # 3. Common date-time formats
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%d-%m-%Y %H:%M:%S",
        "%H:%M:%S,%f",  # MELD format: 00:20:57,256
        "%H:%M:%S",
    ):
        try:
            dt = datetime.strptime(s, fmt)
            dt = dt.replace(tzinfo=timezone.utc)
            return dt, {"timestamp_raw": s, "format": fmt}, None
        except ValueError:
            continue

    return None, None, f"Unrecognized timestamp format: '{s}'"


class TabularCsvParser(BaseConversationParser):
    """
    Parser for tabular CSV/TSV conversational exports.
    Supports:
    - Delimiter detection (comma, tab, semicolon, pipe)
    - Quoted fields with embedded commas and newlines
    - UTF-8 and UTF-8 BOM
    - Auto-detection or explicit column mapping
    - Rejection diagnostics (missing speaker, empty text, invalid timestamps)
    - Duplicate ID handling and dynamic ID generation
    """

    def can_parse(self, raw_input: Any) -> bool:
        if not isinstance(raw_input, str):
            return False
        clean = raw_input.lstrip("\ufeff").strip()
        if not clean:
            return False

        lines = clean.splitlines()
        if not lines:
            return False

        header = lines[0].strip()
        # Quick check for delimiters
        if not any(delim in header for delim in (",", "\t", ";", "|")):
            return False

        # Exclude WhatsApp (starts with digits or [)
        if re.match(r"^[\u200e\u200f]?\[?\d{1,4}[/\-.]", header):
            return False

        # Exclude standard dialogue turns: 'Speaker: utterance'
        # If the header line contains ':' where prefix looks like a speaker name without delimiters,
        # it is an ordinary conversation turn containing commas, not a CSV header.
        if ":" in header:
            cand_spk, _, _ = header.partition(":")
            cand_spk_clean = cand_spk.strip()
            if (
                cand_spk_clean
                and len(cand_spk_clean) <= 80
                and not cand_spk_clean.lower().startswith(("http", "ftp", "mailto", "file", "ssh"))
                and not any(delim in cand_spk_clean for delim in (",", "\t", ";", "|"))
            ):
                return False

        delim = detect_delimiter(header)
        if delim not in header:
            return False

        try:
            reader = csv.reader([header], delimiter=delim)
            row = next(reader)
        except Exception:
            return False

        if len(row) < 2:
            return False

        # Match discrete cell tokens against known speaker and text aliases
        norm_cells = [re.sub(r"[_\- .]", "", c.strip().lower()) for c in row if c.strip()]
        has_spk = any(any(a.replace("_", "").replace("-", "") in cell for a in SPEAKER_ALIASES) for cell in norm_cells)
        has_txt = any(any(a.replace("_", "").replace("-", "") in cell for a in TEXT_ALIASES) for cell in norm_cells)

        return has_spk or has_txt

    def resolve_columns(
        self,
        headers: List[str],
        user_mapping: Optional[Dict[str, str]] = None,
    ) -> Tuple[Dict[str, int], List[ParserWarning]]:
        """
        Resolves column indices for target fields:
        'speaker', 'text', 'timestamp', 'message_id'.
        Returns (resolved_indices: Dict[target_name, col_index], warnings: List[ParserWarning]).
        """
        warnings: List[ParserWarning] = []
        lower_headers = [h.strip().strip('"').strip("'").lower() for h in headers]
        header_map = {name: idx for idx, name in enumerate(lower_headers)}

        resolved: Dict[str, int] = {}

        # Normalized map without underscores, hyphens, spaces, or dots
        def normalize_name(s: str) -> str:
            return s.lower().replace("_", "").replace("-", "").replace(" ", "").replace(".", "")

        normalized_header_map = {normalize_name(name): idx for name, idx in header_map.items()}

        def match_alias(aliases: List[str]) -> Optional[int]:
            for alias in aliases:
                if alias in header_map:
                    return header_map[alias]
                norm_a = normalize_name(alias)
                if norm_a in normalized_header_map:
                    return normalized_header_map[norm_a]
            return None

        # 1. Apply user-specified column mapping
        if user_mapping:
            for k, v in user_mapping.items():
                k_clean = k.strip().lower()
                v_clean = v.strip().lower()

                # Check if k is target and v is CSV column
                if k_clean in (TARGET_SPEAKER, TARGET_TEXT, TARGET_TIMESTAMP, TARGET_MESSAGE_ID):
                    target = k_clean
                    csv_col = v_clean
                elif v_clean in (TARGET_SPEAKER, TARGET_TEXT, TARGET_TIMESTAMP, TARGET_MESSAGE_ID):
                    target = v_clean
                    csv_col = k_clean
                else:
                    continue

                if csv_col in header_map:
                    resolved[target] = header_map[csv_col]
                elif normalize_name(csv_col) in normalized_header_map:
                    resolved[target] = normalized_header_map[normalize_name(csv_col)]
                else:
                    warnings.append(
                        ParserWarning(
                            code="MAPPED_COLUMN_NOT_FOUND",
                            message=f"Mapped column '{csv_col}' for target '{target}' was not found in CSV header.",
                            severity=WarningSeverity.WARNING,
                        )
                    )

        # 2. Heuristic auto-detection for unresolved target fields
        if TARGET_SPEAKER not in resolved:
            m_spk = match_alias(SPEAKER_ALIASES)
            if m_spk is not None:
                resolved[TARGET_SPEAKER] = m_spk

        if TARGET_TEXT not in resolved:
            m_txt = match_alias(TEXT_ALIASES)
            if m_txt is not None:
                resolved[TARGET_TEXT] = m_txt

        if TARGET_TIMESTAMP not in resolved:
            m_ts = match_alias(TIMESTAMP_ALIASES)
            if m_ts is not None:
                resolved[TARGET_TIMESTAMP] = m_ts

        if TARGET_MESSAGE_ID not in resolved:
            m_id = match_alias(MESSAGE_ID_ALIASES)
            if m_id is not None:
                resolved[TARGET_MESSAGE_ID] = m_id

        return resolved, warnings

    def parse(
        self,
        raw_input: Any,
        conversation_id: Optional[str] = None,
        column_mapping: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> CanonicalConversation:
        if not isinstance(raw_input, str):
            raise ValueError("CSV input must be text.")

        conv_id = conversation_id or f"conv_csv_{uuid.uuid4().hex[:8]}"
        warnings: List[ParserWarning] = []

        # Strip UTF-8 BOM if present
        text_content = raw_input.lstrip("\ufeff")

        if not text_content.strip():
            warnings.append(
                ParserWarning(
                    code="EMPTY_INPUT",
                    message="CSV input is empty or whitespace only.",
                    severity=WarningSeverity.ERROR,
                )
            )
            return CanonicalConversation(
                conversation_id=conv_id,
                source_format=SourceFormat.TABULAR_CSV,
                participants=[],
                messages=[],
                warnings=warnings,
                metadata={"has_timestamps": False, "rows_imported": 0, "rows_rejected": 0},
            )

        # Detect delimiter
        delimiter = detect_delimiter(text_content)

        # Parse CSV rows
        f = io.StringIO(text_content)
        reader = csv.reader(f, delimiter=delimiter, quotechar='"')

        try:
            headers = next(reader)
        except StopIteration:
            warnings.append(
                ParserWarning(
                    code="EMPTY_CSV",
                    message="CSV contains no header row.",
                    severity=WarningSeverity.ERROR,
                )
            )
            return CanonicalConversation(
                conversation_id=conv_id,
                source_format=SourceFormat.TABULAR_CSV,
                participants=[],
                messages=[],
                warnings=warnings,
                metadata={"has_timestamps": False, "rows_imported": 0, "rows_rejected": 0},
            )

        # Clean headers
        clean_headers = [h.strip().strip('"').strip("'") for h in headers]

        # Resolve columns
        resolved_cols, map_warnings = self.resolve_columns(clean_headers, user_mapping=column_mapping)
        warnings.extend(map_warnings)

        # Check required columns (speaker, text)
        missing_req: List[str] = []
        if TARGET_SPEAKER not in resolved_cols:
            missing_req.append("speaker")
        if TARGET_TEXT not in resolved_cols:
            missing_req.append("text")

        if missing_req:
            warnings.append(
                ParserWarning(
                    code="COLUMN_MAPPING_REQUIRED",
                    message=(
                        f"Required column(s) {missing_req} could not be determined. "
                        f"Available columns: {clean_headers}. "
                        f"Please provide an explicit column_mapping."
                    ),
                    severity=WarningSeverity.ERROR,
                )
            )
            return CanonicalConversation(
                conversation_id=conv_id,
                source_format=SourceFormat.TABULAR_CSV,
                participants=[],
                messages=[],
                warnings=warnings,
                metadata={
                    "error": "Column mapping required",
                    "available_columns": clean_headers,
                    "missing_targets": missing_req,
                    "rows_imported": 0,
                    "rows_rejected": 0,
                },
            )

        spk_idx = resolved_cols[TARGET_SPEAKER]
        txt_idx = resolved_cols[TARGET_TEXT]
        ts_idx = resolved_cols.get(TARGET_TIMESTAMP)
        id_idx = resolved_cols.get(TARGET_MESSAGE_ID)

        speaker_profiles: Dict[str, SpeakerProfile] = {}

        def get_or_create_speaker(name: str) -> str:
            clean = name.strip() if name and name.strip() else "Unknown"
            norm = clean.lower()
            if norm not in speaker_profiles:
                spk_id = f"spk_{len(speaker_profiles)}"
                speaker_profiles[norm] = SpeakerProfile(
                    speaker_id=spk_id,
                    name=clean,
                    normalized_name=norm,
                    turn_count=0,
                )
            speaker_profiles[norm].turn_count += 1
            return speaker_profiles[norm].speaker_id

        messages: List[CanonicalMessage] = []
        seen_ids: Set[str] = set()

        rows_imported = 0
        rows_rejected = 0
        missing_speakers = 0
        empty_text_rows = 0
        invalid_timestamps = 0
        duplicate_ids = 0

        for row_idx, row in enumerate(reader, start=2):
            # Check empty row
            if not row or all(not cell.strip() for cell in row):
                continue

            # Ensure row has enough columns
            max_needed = max(filter(lambda x: x is not None, [spk_idx, txt_idx, ts_idx, id_idx]))  # type: ignore
            if len(row) <= max_needed:
                # Pad row with empty strings
                row = row + [""] * (max_needed + 1 - len(row))

            raw_spk = row[spk_idx].strip()
            raw_txt = row[txt_idx].strip()

            # Message ID extraction
            custom_id: Optional[str] = None
            if id_idx is not None and len(row) > id_idx and row[id_idx].strip():
                cand_id = row[id_idx].strip()
                if cand_id in seen_ids:
                    duplicate_ids += 1
                    warnings.append(
                        ParserWarning(
                            code="DUPLICATE_MESSAGE_ID",
                            message=f"Row {row_idx}: Duplicate message ID '{cand_id}'. Assigned local unique ID.",
                            severity=WarningSeverity.WARNING,
                            line_number=row_idx,
                            message_id=cand_id,
                        )
                    )
                    custom_id = f"{cand_id}_dup_{row_idx}"
                else:
                    custom_id = cand_id
                    seen_ids.add(cand_id)
            else:
                custom_id = f"msg_{len(messages) + 1:04d}"
                seen_ids.add(custom_id)

            # Timestamp extraction
            dt: Optional[datetime] = None
            ts_meta: Optional[Dict[str, Any]] = None
            if ts_idx is not None and len(row) > ts_idx and row[ts_idx].strip():
                dt, ts_meta, ts_err = parse_csv_timestamp(row[ts_idx].strip())
                if ts_err:
                    invalid_timestamps += 1
                    warnings.append(
                        ParserWarning(
                            code="INVALID_TIMESTAMP",
                            message=f"Row {row_idx}: {ts_err}",
                            severity=WarningSeverity.WARNING,
                            line_number=row_idx,
                            message_id=custom_id,
                        )
                    )

            # Row validity checks
            is_valid_turn = True
            msg_type = MessageType.TEXT
            metadata: Dict[str, Any] = {"row_number": row_idx}
            if ts_meta:
                metadata.update(ts_meta)

            if not raw_spk:
                missing_speakers += 1
                rows_rejected += 1
                is_valid_turn = False
                msg_type = MessageType.UNPARSED
                warnings.append(
                    ParserWarning(
                        code="MISSING_SPEAKER",
                        message=f"Row {row_idx}: missing speaker name.",
                        severity=WarningSeverity.WARNING,
                        line_number=row_idx,
                        message_id=custom_id,
                    )
                )
                display_speaker = "Unknown"
                spk_id = "spk_unknown"
            else:
                display_speaker = raw_spk
                spk_id = get_or_create_speaker(raw_spk)

            if not raw_txt:
                empty_text_rows += 1
                is_valid_turn = False
                if raw_spk:
                    warnings.append(
                        ParserWarning(
                            code="EMPTY_TEXT",
                            message=f"Row {row_idx}: Speaker '{raw_spk}' has empty text.",
                            severity=WarningSeverity.INFO,
                            line_number=row_idx,
                            message_id=custom_id,
                        )
                    )

            if is_valid_turn:
                rows_imported += 1

            src_idx = len(messages)
            messages.append(
                CanonicalMessage(
                    conversation_id=conv_id,
                    message_id=custom_id,
                    source_index=src_idx,
                    sequence_index=src_idx,
                    speaker=display_speaker,
                    speaker_id=spk_id,
                    text=raw_txt,
                    timestamp=dt,
                    message_type=msg_type,
                    metadata=metadata,
                    is_evaluable=is_valid_turn,
                )
            )

        has_timestamps = any(m.timestamp is not None for m in messages)
        if not has_timestamps and messages:
            warnings.append(
                ParserWarning(
                    code="MISSING_TIMESTAMPS",
                    message="CSV does not contain timestamp metadata; ordering is based on row sequence.",
                    severity=WarningSeverity.INFO,
                )
            )

        return CanonicalConversation(
            conversation_id=conv_id,
            source_format=SourceFormat.TABULAR_CSV,
            title=None,
            participants=list(speaker_profiles.values()),
            messages=messages,
            warnings=warnings,
            metadata={
                "has_timestamps": has_timestamps,
                "delimiter": delimiter,
                "resolved_columns": resolved_cols,
                "rows_imported": rows_imported,
                "rows_rejected": rows_rejected,
                "missing_speakers": missing_speakers,
                "empty_text_rows": empty_text_rows,
                "invalid_timestamps": invalid_timestamps,
                "duplicate_ids": duplicate_ids,
                "total_messages": len(messages),
                "evaluable_messages": len([m for m in messages if m.is_evaluable]),
                "num_speakers": len(speaker_profiles),
            },
        )
