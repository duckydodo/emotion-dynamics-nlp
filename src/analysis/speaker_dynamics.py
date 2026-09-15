from __future__ import annotations

import pandas as pd

from src.analysis.trajectories import (
    EMOTIONS,
    build_valid_transitions,
    transition_counts,
    transition_probabilities,
    calculate_persistence,
)


def classify_speaker_transitions(
    transitions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add speaker-transition information to valid emotion transitions.

    A transition is:
        - same_speaker: speaker remains the same
        - cross_speaker: speaker changes
    """

    required = {
        "speaker_from",
        "speaker_to",
    }

    missing = required - set(transitions.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    result = transitions.copy()

    result["speaker_changed"] = (
        result["speaker_from"] != result["speaker_to"]
    )

    result["speaker_transition"] = result["speaker_changed"].map(
        {
            False: "same_speaker",
            True: "cross_speaker",
        }
    )

    return result


def split_by_speaker_transition(
    transitions: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Split valid transitions into same-speaker and cross-speaker groups.
    """

    transitions = classify_speaker_transitions(transitions)

    return {
        "same_speaker": transitions[
            transitions["speaker_transition"] == "same_speaker"
        ].reset_index(drop=True),
        "cross_speaker": transitions[
            transitions["speaker_transition"] == "cross_speaker"
        ].reset_index(drop=True),
    }


def calculate_speaker_transition_summary(
    transitions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare emotion dynamics between same-speaker and
    cross-speaker transitions.
    """

    groups = split_by_speaker_transition(transitions)

    rows = []

    for transition_type, subset in groups.items():

        total = len(subset)

        if total == 0:
            rows.append(
                {
                    "speaker_transition": transition_type,
                    "transitions": 0,
                    "emotion_shifts": 0,
                    "shift_rate": 0.0,
                    "persistence_rate": 0.0,
                }
            )
            continue

        shifts = (
            subset["emotion_from"]
            != subset["emotion_to"]
        )

        rows.append(
            {
                "speaker_transition": transition_type,
                "transitions": total,
                "emotion_shifts": int(shifts.sum()),
                "shift_rate": float(shifts.mean()),
                "persistence_rate": float((~shifts).mean()),
            }
        )

    return pd.DataFrame(rows)


def speaker_transition_counts(
    transitions: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Return emotion transition-count matrices separately for
    same-speaker and cross-speaker transitions.
    """

    groups = split_by_speaker_transition(transitions)

    return {
        transition_type: transition_counts(subset)
        for transition_type, subset in groups.items()
    }


def speaker_transition_probabilities(
    transitions: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Return emotion transition-probability matrices separately
    for same-speaker and cross-speaker transitions.
    """

    groups = split_by_speaker_transition(transitions)

    return {
        transition_type: transition_probabilities(subset)
        for transition_type, subset in groups.items()
    }


def speaker_persistence(
    transitions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate persistence separately for same-speaker and
    cross-speaker transitions.
    """

    groups = split_by_speaker_transition(transitions)

    rows = []

    for transition_type, subset in groups.items():

        result = calculate_persistence(subset)

        rows.append(
            {
                "speaker_transition": transition_type,
                "transitions": result["total_transitions"],
                "persistent_transitions": result[
                    "persistent_transitions"
                ],
                "persistence_rate": result[
                    "overall_persistence_rate"
                ],
            }
        )

    return pd.DataFrame(rows)


def build_speaker_trajectories(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build valid utterance-level trajectories while retaining
    speaker identity.

    Each row represents one valid consecutive transition.
    """

    transitions = build_valid_transitions(df)

    if transitions.empty:
        return transitions

    transitions = classify_speaker_transitions(transitions)

    return transitions[
        [
            "dialogue_id",
            "utterance_id",
            "speaker_from",
            "speaker_to",
            "speaker_changed",
            "speaker_transition",
            "emotion_from",
            "emotion_to",
        ]
    ].reset_index(drop=True)
