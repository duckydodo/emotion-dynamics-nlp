from collections import Counter
from dataclasses import dataclass


@dataclass
class ConversationAnalysis:
    emotion_distribution: dict[str, int]
    transition_counts: dict[str, int]
    total_transitions: int
    shifts: int
    persistence: int
    shift_rate: float
    persistence_rate: float
    same_speaker_transitions: int
    same_speaker_shifts: int
    cross_speaker_transitions: int
    cross_speaker_shifts: int
    turning_points: list[dict]


def analyze_predictions(
    predictions: list[dict],
) -> ConversationAnalysis:
    """Analyze emotion dynamics from ordered prediction results."""

    if not predictions:
        return ConversationAnalysis(
            emotion_distribution={},
            transition_counts={},
            total_transitions=0,
            shifts=0,
            persistence=0,
            shift_rate=0.0,
            persistence_rate=0.0,
            same_speaker_transitions=0,
            same_speaker_shifts=0,
            cross_speaker_transitions=0,
            cross_speaker_shifts=0,
            turning_points=[],
        )

    ordered = sorted(
        predictions,
        key=lambda item: item["turn_id"],
    )

    emotions = [item["emotion"] for item in ordered]

    distribution = dict(Counter(emotions))

    transition_counts: Counter = Counter()

    shifts = 0
    persistence = 0

    same_speaker_transitions = 0
    same_speaker_shifts = 0

    cross_speaker_transitions = 0
    cross_speaker_shifts = 0

    turning_points = []

    for i in range(len(ordered) - 1):
        current = ordered[i]
        following = ordered[i + 1]

        # Only analyze actual consecutive turn IDs.
        if following["turn_id"] != current["turn_id"] + 1:
            continue

        from_emotion = current["emotion"]
        to_emotion = following["emotion"]

        transition_key = f"{from_emotion} → {to_emotion}"
        transition_counts[transition_key] += 1

        if from_emotion == to_emotion:
            persistence += 1
        else:
            shifts += 1

        same_speaker = current["speaker"] == following["speaker"]

        if same_speaker:
            same_speaker_transitions += 1

            if from_emotion != to_emotion:
                same_speaker_shifts += 1
        else:
            cross_speaker_transitions += 1

            if from_emotion != to_emotion:
                cross_speaker_shifts += 1

        # A turning point is a change that persists into
        # the following valid turn.
        if i + 2 < len(ordered):
            next_turn = ordered[i + 2]

            if next_turn["turn_id"] != following["turn_id"] + 1:
                continue

            if (
                from_emotion != to_emotion
                and next_turn["emotion"] == to_emotion
            ):
                turning_points.append(
                    {
                        "from_emotion": from_emotion,
                        "to_emotion": to_emotion,
                        "turn_id": following["turn_id"],
                        "speaker_from": current["speaker"],
                        "speaker_to": following["speaker"],
                        "text": following["text"],
                    }
                )

    total_transitions = shifts + persistence

    if total_transitions:
        shift_rate = shifts / total_transitions
        persistence_rate = persistence / total_transitions
    else:
        shift_rate = 0.0
        persistence_rate = 0.0

    return ConversationAnalysis(
        emotion_distribution=distribution,
        transition_counts=dict(transition_counts),
        total_transitions=total_transitions,
        shifts=shifts,
        persistence=persistence,
        shift_rate=shift_rate,
        persistence_rate=persistence_rate,
        same_speaker_transitions=same_speaker_transitions,
        same_speaker_shifts=same_speaker_shifts,
        cross_speaker_transitions=cross_speaker_transitions,
        cross_speaker_shifts=cross_speaker_shifts,
        turning_points=turning_points,
    )
