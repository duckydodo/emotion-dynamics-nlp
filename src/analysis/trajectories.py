from __future__ import annotations

from collections import Counter

import pandas as pd


EMOTIONS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
]


def build_valid_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build valid adjacent emotion transitions within MELD dialogues.

    A transition is valid only when:
        next.dialogue_id == current.dialogue_id
        next.utterance_id == current.utterance_id + 1

    This prevents transitions from crossing dialogue boundaries
    or missing Utterance_ID gaps.
    """

    required = {
        "dialogue_id",
        "utterance_id",
        "speaker",
        "emotion",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = df.sort_values(
        ["dialogue_id", "utterance_id"]
    ).reset_index(drop=True)

    next_rows = data.shift(-1)

    valid = (
        (next_rows["dialogue_id"] == data["dialogue_id"])
        & (next_rows["utterance_id"] == data["utterance_id"] + 1)
    )

    transitions = pd.DataFrame(
        {
            "dialogue_id": data.loc[valid, "dialogue_id"].values,
            "utterance_id": data.loc[valid, "utterance_id"].values,
            "emotion_from": data.loc[valid, "emotion"].values,
            "emotion_to": next_rows.loc[valid, "emotion"].values,
            "speaker_from": data.loc[valid, "speaker"].values,
            "speaker_to": next_rows.loc[valid, "speaker"].values,
        }
    )

    return transitions.reset_index(drop=True)


def transition_counts(transitions: pd.DataFrame) -> pd.DataFrame:
    """
    Return a 7x7 matrix of emotion transition counts.
    """

    counts = pd.crosstab(
        transitions["emotion_from"],
        transitions["emotion_to"],
    )

    return counts.reindex(
        index=EMOTIONS,
        columns=EMOTIONS,
        fill_value=0,
    )


def transition_probabilities(transitions: pd.DataFrame) -> pd.DataFrame:
    """
    Return row-normalized emotion transition probabilities.
    """

    counts = transition_counts(transitions)

    probabilities = counts.div(
        counts.sum(axis=1),
        axis=0,
    ).fillna(0.0)

    return probabilities


def calculate_persistence(transitions: pd.DataFrame) -> dict:
    """
    Calculate overall and per-emotion persistence.

    Persistence means the next valid utterance has the
    same emotion as the current utterance.
    """

    total = len(transitions)

    if total == 0:
        return {
            "total_transitions": 0,
            "persistent_transitions": 0,
            "overall_persistence_rate": 0.0,
            "per_emotion": {},
        }

    same = transitions["emotion_from"] == transitions["emotion_to"]

    per_emotion = {}

    for emotion in EMOTIONS:
        subset = transitions[
            transitions["emotion_from"] == emotion
        ]

        if len(subset) == 0:
            per_emotion[emotion] = {
                "transitions": 0,
                "persistent": 0,
                "persistence_rate": 0.0,
            }
            continue

        persistent = (
            subset["emotion_from"] == subset["emotion_to"]
        ).sum()

        per_emotion[emotion] = {
            "transitions": len(subset),
            "persistent": int(persistent),
            "persistence_rate": float(persistent / len(subset)),
        }

    return {
        "total_transitions": total,
        "persistent_transitions": int(same.sum()),
        "overall_persistence_rate": float(same.mean()),
        "per_emotion": per_emotion,
    }


def calculate_dialogue_shifts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate emotion changes for each dialogue.

    Only valid adjacent utterance pairs are considered.
    """

    transitions = build_valid_transitions(df)

    if transitions.empty:
        return pd.DataFrame(
            columns=[
                "dialogue_id",
                "valid_transitions",
                "emotion_shifts",
                "shift_rate",
            ]
        )

    transitions["is_shift"] = (
        transitions["emotion_from"]
        != transitions["emotion_to"]
    )

    result = (
        transitions.groupby("dialogue_id")
        .agg(
            valid_transitions=("emotion_from", "size"),
            emotion_shifts=("is_shift", "sum"),
        )
        .reset_index()
    )

    result["shift_rate"] = (
        result["emotion_shifts"]
        / result["valid_transitions"]
    )

    return result


def summarize_trajectories(df: pd.DataFrame) -> dict:
    """
    Produce high-level trajectory statistics for one MELD split.
    """

    transitions = build_valid_transitions(df)
    dialogue_stats = calculate_dialogue_shifts(df)

    if dialogue_stats.empty:
        return {
            "num_dialogues": int(df["dialogue_id"].nunique()),
            "valid_transitions": len(transitions),
            "emotion_shifts": 0,
            "overall_shift_rate": 0.0,
            "mean_shifts_per_dialogue": 0.0,
            "median_shifts_per_dialogue": 0.0,
        }

    shifts = dialogue_stats["emotion_shifts"]

    return {
        "num_dialogues": int(df["dialogue_id"].nunique()),
        "valid_transitions": len(transitions),
        "emotion_shifts": int(shifts.sum()),
        "overall_shift_rate": float(
            transitions["emotion_from"]
            .ne(transitions["emotion_to"])
            .mean()
        ),
        "mean_shifts_per_dialogue": float(shifts.mean()),
        "median_shifts_per_dialogue": float(shifts.median()),
    }
