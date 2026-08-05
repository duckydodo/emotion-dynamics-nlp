from dataclasses import dataclass
from collections import Counter

import numpy as np
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


@dataclass(slots=True)
class ClassMetrics:
    precision: float
    recall: float
    f1: float


@dataclass(slots=True)
class EvaluationResult:
    accuracy: float
    macro_f1: float
    weighted_f1: float

    per_class: dict[str, ClassMetrics]

    confusion_matrix: np.ndarray
