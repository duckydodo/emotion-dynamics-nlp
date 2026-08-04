from src.data.types import MELDDataset
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "Dialogue_ID",
    "Utterance_ID",
    "Speaker",
    "Utterance",
    "Emotion",
]


COLUMN_MAP = {
    "Dialogue_ID": "dialogue_id",
    "Utterance_ID": "utterance_id",
    "Speaker": "speaker",
    "Utterance": "text",
    "Emotion": "emotion",
}


def _load_split(csv_path: Path) -> pd.DataFrame:
    """
    Load a single MELD split.
    """

    df = pd.read_csv(csv_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name} is missing columns: {missing}")

    df = df.rename(columns=COLUMN_MAP)

    df = df.sort_values(
        ["dialogue_id", "utterance_id"],
        ignore_index=True,
    )

    return df


def load_meld(data_dir: Path):
    """
    Load official MELD train/dev/test splits.

    Returns
    -------
    train_df, dev_df, test_df
    """

    data_dir = Path(data_dir)

    train = _load_split(data_dir / "train_sent_emo.csv")
    dev = _load_split(data_dir / "dev_sent_emo.csv")
    test = _load_split(data_dir / "test_sent_emo.csv")

    return MELDDataset(
    train=train,
    dev=dev,
    test=test,
)
