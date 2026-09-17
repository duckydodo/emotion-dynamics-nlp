from __future__ import annotations

import pandas as pd

from src.analysis.trajectories import EMOTIONS, build_valid_transitions, transition_probabilities


REQUIRED_COLUMNS = {"dialogue_id", "utterance_id", "emotion"}


def build_forecasting_examples(df: pd.DataFrame) -> pd.DataFrame:
    """Build valid one-step emotion forecasting examples."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = df.copy()

    # build_valid_transitions expects speaker, even though forecasting
    # itself does not use speaker information.
    if "speaker" not in data.columns:
        data["speaker"] = "unknown"

    transitions = build_valid_transitions(data)

    return transitions.rename(
        columns={
            "emotion_from": "current_emotion",
            "emotion_to": "next_emotion",
        }
    )[
        [
            "dialogue_id",
            "utterance_id",
            "current_emotion",
            "next_emotion",
            "speaker_from",
            "speaker_to",
        ]
    ].reset_index(drop=True)


def majority_forecast(
    train_examples: pd.DataFrame,
    test_examples: pd.DataFrame,
) -> list[str]:
    """Always predict the most common next emotion in training."""
    if train_examples.empty:
        raise ValueError("Training examples cannot be empty.")

    majority = (
        train_examples["next_emotion"]
        .value_counts()
        .reindex(EMOTIONS, fill_value=0)
        .idxmax()
    )

    return [majority] * len(test_examples)


def persistence_forecast(test_examples: pd.DataFrame) -> list[str]:
    """Predict that the next emotion equals the current emotion."""
    return test_examples["current_emotion"].astype(str).tolist()


def fit_markov_model(train_examples: pd.DataFrame) -> pd.DataFrame:
    """Fit P(next_emotion | current_emotion) using training transitions only."""
    required = {"current_emotion", "next_emotion"}
    missing = required - set(train_examples.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    transitions = train_examples.rename(
        columns={
            "current_emotion": "emotion_from",
            "next_emotion": "emotion_to",
        }
    )

    return transition_probabilities(transitions)


def markov_forecast(
    test_examples: pd.DataFrame,
    transition_model: pd.DataFrame,
) -> list[str]:
    """Predict the most probable next emotion for each current emotion."""
    model = transition_model.reindex(
        index=EMOTIONS,
        columns=EMOTIONS,
        fill_value=0.0,
    )

    predictions = []
    for emotion in test_examples["current_emotion"]:
        if emotion in model.index:
            predictions.append(model.loc[emotion].idxmax())
        else:
            predictions.append("neutral")

    return predictions


def forecast_with_markov(
    train_examples: pd.DataFrame,
    test_examples: pd.DataFrame,
) -> list[str]:
    """Fit Markov model on train and forecast test."""
    return markov_forecast(
        test_examples,
        fit_markov_model(train_examples),
    )


def markov_transition_summary(
    transition_model: pd.DataFrame,
) -> pd.DataFrame:
    """Return the most likely next emotion and probability per current emotion."""
    model = transition_model.reindex(
        index=EMOTIONS,
        columns=EMOTIONS,
        fill_value=0.0,
    )

    rows = []
    for emotion in EMOTIONS:
        next_emotion = model.loc[emotion].idxmax()
        rows.append(
            {
                "current_emotion": emotion,
                "predicted_next_emotion": next_emotion,
                "probability": float(model.loc[emotion, next_emotion]),
            }
        )

    return pd.DataFrame(rows)


def build_emotion_sequences(
    df: pd.DataFrame,
    history_size: int,
) -> pd.DataFrame:
    """Build valid fixed-length emotion-history forecasting examples."""
    if history_size < 1:
        raise ValueError("history_size must be at least 1.")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = df.copy()
    if "speaker" not in data.columns:
        data["speaker"] = "unknown"

    data = data.sort_values(["dialogue_id", "utterance_id"]).reset_index(drop=True)
    rows = []

    for dialogue_id, dialogue in data.groupby("dialogue_id", sort=False):
        dialogue = dialogue.reset_index(drop=True)

        for i in range(history_size, len(dialogue)):
            window = dialogue.iloc[i - history_size:i + 1]
            ids = window["utterance_id"].tolist()

            if any(ids[j + 1] != ids[j] + 1 for j in range(len(ids) - 1)):
                continue

            emotions = window["emotion"].astype(str).tolist()
            row = {
                "dialogue_id": dialogue_id,
                "target_utterance_id": ids[-1],
                "next_emotion": emotions[-1],
            }

            for j in range(history_size):
                row[f"history_{j + 1}"] = emotions[j]

            rows.append(row)

    columns = (
        ["dialogue_id", "target_utterance_id"]
        + [f"history_{j + 1}" for j in range(history_size)]
        + ["next_emotion"]
    )
    return pd.DataFrame(rows, columns=columns)


def fit_higher_order_markov_model(
    sequence_examples: pd.DataFrame,
    history_size: int,
) -> dict[tuple[str, ...], pd.Series]:
    """Fit P(next_emotion | emotion history) using training examples only."""
    if history_size < 1:
        raise ValueError("history_size must be at least 1.")

    history_columns = [f"history_{j + 1}" for j in range(history_size)]
    required = set(history_columns + ["next_emotion"])
    missing = required - set(sequence_examples.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if sequence_examples.empty:
        raise ValueError("Training sequence examples cannot be empty.")

    model = {}
    for history, group in sequence_examples.groupby(history_columns, sort=False):
        if history_size == 1:
            history = (history,)

        counts = (
            group["next_emotion"]
            .value_counts()
            .reindex(EMOTIONS, fill_value=0)
        )
        model[tuple(history)] = counts / counts.sum()

    return model


def higher_order_markov_forecast(
    sequence_examples: pd.DataFrame,
    transition_model: dict[tuple[str, ...], pd.Series],
    history_size: int,
    fallback: str = "neutral",
) -> list[str]:
    """Forecast next emotion using a higher-order Markov model."""
    if history_size < 1:
        raise ValueError("history_size must be at least 1.")
    if fallback not in EMOTIONS:
        raise ValueError(f"Invalid fallback emotion: {fallback}")

    history_columns = [f"history_{j + 1}" for j in range(history_size)]
    missing = set(history_columns) - set(sequence_examples.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    predictions = []
    for _, row in sequence_examples.iterrows():
        history = tuple(str(row[column]) for column in history_columns)
        predictions.append(
            transition_model[history].idxmax()
            if history in transition_model
            else fallback
        )

    return predictions
