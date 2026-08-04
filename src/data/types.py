from dataclasses import dataclass
from collections import Counter

import pandas as pd


@dataclass(slots=True)
class MELDDataset:
    train: pd.DataFrame
    dev: pd.DataFrame
    test: pd.DataFrame


@dataclass(slots=True)
class ValidationReport:
    split_name: str

    num_rows: int
    num_dialogues: int
    num_speakers: int

    missing_text: int
    missing_emotion: int
    duplicate_rows: int

    emotion_distribution: Counter
