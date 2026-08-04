from collections import Counter

from src.data.types import ValidationReport


def validate_meld(df, split_name="train"):

    missing_text = df["text"].isna().sum()

    missing_emotion = df["emotion"].isna().sum()

    duplicate_rows = (
        df
        .duplicated(["dialogue_id", "utterance_id"])
        .sum()
    )

    report = ValidationReport(
        split_name=split_name,

        num_rows=len(df),
        num_dialogues=df["dialogue_id"].nunique(),
        num_speakers=df["speaker"].nunique(),

        missing_text=missing_text,
        missing_emotion=missing_emotion,
        duplicate_rows=duplicate_rows,

        emotion_distribution=Counter(df["emotion"]),
    )

    return report


def print_report(report):

    print(f"\n===== {report.split_name.upper()} =====")

    print(f"Rows          : {report.num_rows}")
    print(f"Dialogues     : {report.num_dialogues}")
    print(f"Speakers      : {report.num_speakers}")

    print(f"Missing text  : {report.missing_text}")
    print(f"Missing label : {report.missing_emotion}")
    print(f"Duplicates    : {report.duplicate_rows}")

    print("\nEmotion distribution")

    for emotion, count in report.emotion_distribution.most_common():
        print(f"{emotion:<10} {count}")
