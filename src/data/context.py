import pandas as pd


def build_context_examples(
    df: pd.DataFrame,
    context_turns: int = 1,
    include_speaker: bool = False,
) -> pd.DataFrame:
    """
    Build context-aware emotion classification examples from MELD.

    Each output row represents a target utterance for which the
    complete requested conversational history is available.

    Parameters
    ----------
    df : pd.DataFrame
        MELD split containing:
        dialogue_id, utterance_id, speaker, text, emotion.

    context_turns : int
        Number of previous valid conversational turns to include.

    include_speaker : bool
        Whether to prefix each utterance with its speaker.

    Returns
    -------
    pd.DataFrame
        Columns:
        dialogue_id
        utterance_id
        speaker
        text
        emotion
        context
    """

    if context_turns < 1:
        raise ValueError("context_turns must be at least 1.")

    required_columns = {
        "dialogue_id",
        "utterance_id",
        "speaker",
        "text",
        "emotion",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df = df.sort_values(
        ["dialogue_id", "utterance_id"]
    ).reset_index(drop=True)

    examples = []

    for dialogue_id, dialogue in df.groupby(
        "dialogue_id",
        sort=False,
    ):
        dialogue = dialogue.sort_values(
            "utterance_id"
        ).reset_index(drop=True)

        rows = dialogue.to_dict("records")

        for target_index in range(len(rows)):

            # We need exactly `context_turns` previous turns.
            if target_index < context_turns:
                continue

            window = rows[
                target_index - context_turns:
                target_index + 1
            ]

            # The entire context window must be contiguous.
            valid_window = True

            for i in range(len(window) - 1):

                current_id = window[i]["utterance_id"]
                next_id = window[i + 1]["utterance_id"]

                if next_id != current_id + 1:
                    valid_window = False
                    break

            if not valid_window:
                continue

            target = window[-1]

            formatted_parts = []

            for row in window:

                text = str(row["text"]).strip()

                if include_speaker:
                    text = (
                        f"[SPEAKER_{row['speaker']}] "
                        f"{text}"
                    )

                formatted_parts.append(text)

            context = "\n".join(formatted_parts)

            examples.append(
                {
                    "dialogue_id": target["dialogue_id"],
                    "utterance_id": target["utterance_id"],
                    "speaker": target["speaker"],
                    "text": target["text"],
                    "emotion": target["emotion"],
                    "context": context,
                }
            )

    return pd.DataFrame(examples)
