from dataclasses import dataclass


@dataclass
class ConversationTurn:
    turn_id: int
    speaker: str
    text: str


class ConversationParseError(ValueError):
    """Raised when conversation input cannot be parsed safely."""


def parse_conversation(raw_text: str) -> list[ConversationTurn]:
    """
    Parse a conversation written as:

        Speaker: utterance

    Rules:
    - Turn IDs are generated automatically.
    - Only the first ':' separates speaker from utterance.
    - Blank lines are ignored.
    - Lines without a speaker marker continue the previous utterance.
    - A continuation line before the first speaker is invalid.
    - A speaker marker with no utterance is invalid.
    """

    if not isinstance(raw_text, str):
        raise ConversationParseError("Conversation input must be text.")

    if not raw_text.strip():
        raise ConversationParseError("Conversation is empty.")

    turns: list[ConversationTurn] = []
    current_speaker: str | None = None
    current_text_parts: list[str] = []

    def finish_current_turn() -> None:
        nonlocal current_speaker, current_text_parts

        if current_speaker is None:
            return

        text = " ".join(
            part.strip()
            for part in current_text_parts
            if part.strip()
        ).strip()

        if not text:
            raise ConversationParseError(
                f"Speaker '{current_speaker}' has no utterance."
            )

        turns.append(
            ConversationTurn(
                turn_id=len(turns),
                speaker=current_speaker,
                text=text,
            )
        )

        current_speaker = None
        current_text_parts = []

    for line_number, raw_line in enumerate(raw_text.splitlines(), start=1):
        line = raw_line.strip()

        if not line:
            continue

        if ":" in line:
            speaker, text = line.split(":", 1)
            speaker = speaker.strip()
            text = text.strip()

            if not speaker:
                raise ConversationParseError(
                    f"Line {line_number}: speaker name is empty."
                )

            if not text:
                raise ConversationParseError(
                    f"Line {line_number}: speaker '{speaker}' "
                    "has no utterance."
                )

            finish_current_turn()

            current_speaker = speaker
            current_text_parts = [text]

        else:
            if current_speaker is None:
                raise ConversationParseError(
                    f"Line {line_number}: expected 'Speaker: utterance'."
                )

            current_text_parts.append(line)

    finish_current_turn()

    if not turns:
        raise ConversationParseError(
            "No valid conversation turns were found."
        )

    return turns
