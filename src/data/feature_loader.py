from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.models.transformer import LABELS, LABEL_TO_ID, ID_TO_LABEL


@dataclass
class DialogueSequence:
    """
    Container for a single dialogue's extracted sequential features and metadata.
    """
    dialogue_id: int
    num_turns: int
    utterance_ids: np.ndarray  # Shape: (T,)
    speakers: List[str]        # Length: T
    party_ids: np.ndarray      # Shape: (T,), values in [0, num_parties - 1]
    party_names: List[str]     # Ordered unique speaker names for this dialogue
    emotions: List[str]        # Length: T
    label_ids: np.ndarray      # Shape: (T,), values in [0, 6]
    features: np.ndarray       # Shape: (T, feature_dim)


class DialogueFeatureDataset:
    """
    In-memory representation of extracted dialogue features for a MELD split.
    Supports both flat utterance-level access and dialogue-level sequence access.
    """

    def __init__(
        self,
        features: np.ndarray,
        dialogue_ids: np.ndarray,
        utterance_ids: np.ndarray,
        speakers: np.ndarray | List[str],
        emotions: np.ndarray | List[str],
        label_ids: np.ndarray,
        metadata: Dict,
    ):
        self.features = np.asarray(features, dtype=np.float32)
        self.dialogue_ids = np.asarray(dialogue_ids, dtype=np.int64)
        self.utterance_ids = np.asarray(utterance_ids, dtype=np.int64)
        self.speakers = [str(s) for s in speakers]
        self.emotions = [str(e).lower().strip() for e in emotions]
        self.label_ids = np.asarray(label_ids, dtype=np.int64)
        self.metadata = metadata

        assert len(self.features) == len(self.dialogue_ids), "Mismatched features and dialogue_ids length"
        assert len(self.features) == len(self.utterance_ids), "Mismatched features and utterance_ids length"
        assert len(self.features) == len(self.speakers), "Mismatched features and speakers length"
        assert len(self.features) == len(self.emotions), "Mismatched features and emotions length"
        assert len(self.features) == len(self.label_ids), "Mismatched features and label_ids length"

        self._dialogues_cache: Optional[List[DialogueSequence]] = None

    def __len__(self) -> int:
        return len(self.features)

    @property
    def feature_dim(self) -> int:
        return self.features.shape[1]

    @property
    def num_dialogues(self) -> int:
        return len(np.unique(self.dialogue_ids))

    def get_dialogues(self) -> List[DialogueSequence]:
        """
        Group flat utterances into chronological dialogue sequences.
        Party IDs are dynamically assigned 0..M-1 per dialogue in order of appearance.
        """
        if self._dialogues_cache is not None:
            return self._dialogues_cache

        dialogue_sequences: List[DialogueSequence] = []
        unique_dialogues = pd.unique(self.dialogue_ids)

        for d_id in unique_dialogues:
            mask = (self.dialogue_ids == d_id)
            d_u_ids = self.utterance_ids[mask]
            d_spks = [self.speakers[i] for i, m in enumerate(mask) if m]
            d_emos = [self.emotions[i] for i, m in enumerate(mask) if m]
            d_lbls = self.label_ids[mask]
            d_feats = self.features[mask]

            # Dynamic local party mapping
            party_map: Dict[str, int] = {}
            party_names: List[str] = []
            party_ids = np.zeros(len(d_spks), dtype=np.int64)

            for idx, spk in enumerate(d_spks):
                if spk not in party_map:
                    party_map[spk] = len(party_map)
                    party_names.append(spk)
                party_ids[idx] = party_map[spk]

            dialogue_sequences.append(
                DialogueSequence(
                    dialogue_id=int(d_id),
                    num_turns=len(d_u_ids),
                    utterance_ids=d_u_ids,
                    speakers=d_spks,
                    party_ids=party_ids,
                    party_names=party_names,
                    emotions=d_emos,
                    label_ids=d_lbls,
                    features=d_feats,
                )
            )

        self._dialogues_cache = dialogue_sequences
        return dialogue_sequences

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return metadata and labels as a pandas DataFrame.
        """
        return pd.DataFrame({
            "dialogue_id": self.dialogue_ids,
            "utterance_id": self.utterance_ids,
            "speaker": self.speakers,
            "emotion": self.emotions,
            "label_id": self.label_ids,
        })


def save_dialogue_features(
    output_path: Path,
    features: np.ndarray,
    dialogue_ids: np.ndarray,
    utterance_ids: np.ndarray,
    speakers: List[str],
    emotions: List[str],
    label_ids: np.ndarray,
    metadata: Dict,
) -> None:
    """
    Save extracted features and metadata into a compressed .npz archive.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        features=np.asarray(features, dtype=np.float32),
        dialogue_ids=np.asarray(dialogue_ids, dtype=np.int64),
        utterance_ids=np.asarray(utterance_ids, dtype=np.int64),
        speakers=np.asarray(speakers, dtype=object),
        emotions=np.asarray(emotions, dtype=object),
        label_ids=np.asarray(label_ids, dtype=np.int64),
        metadata_json=json.dumps(metadata, indent=2),
    )

    # Also save standalone metadata.json alongside for quick inspection
    meta_path = output_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)


def load_dialogue_features(archive_path: Path | str) -> DialogueFeatureDataset:
    """
    Load a compressed .npz archive produced by save_dialogue_features.
    """
    archive_path = Path(archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(f"Feature archive not found: {archive_path}")

    with np.load(archive_path, allow_pickle=True) as data:
        features = data["features"]
        dialogue_ids = data["dialogue_ids"]
        utterance_ids = data["utterance_ids"]
        speakers = data["speakers"].tolist()
        emotions = data["emotions"].tolist()
        label_ids = data["label_ids"]
        metadata = json.loads(str(data["metadata_json"]))

    return DialogueFeatureDataset(
        features=features,
        dialogue_ids=dialogue_ids,
        utterance_ids=utterance_ids,
        speakers=speakers,
        emotions=emotions,
        label_ids=label_ids,
        metadata=metadata,
    )


class DialogueBatchDataset:
    """
    PyTorch-compatible Dataset wrapper around a list of DialogueSequence instances.
    """

    def __init__(self, sequences: List[DialogueSequence]):
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> DialogueSequence:
        return self.sequences[idx]


def dialogue_collate_fn(batch: List[DialogueSequence]) -> Dict[str, Any]:
    """
    Collate variable-length dialogue sequences into a padded batch tensor dictionary.
    """
    import torch

    batch_size = len(batch)
    lengths = [seq.num_turns for seq in batch]
    max_len = max(lengths)
    feature_dim = batch[0].features.shape[1]

    padded_features = torch.zeros((batch_size, max_len, feature_dim), dtype=torch.float32)
    padded_party_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    padded_label_ids = torch.full((batch_size, max_len), -100, dtype=torch.long)
    padded_utterance_ids = torch.full((batch_size, max_len), -1, dtype=torch.long)
    turn_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    dialogue_ids = torch.tensor([seq.dialogue_id for seq in batch], dtype=torch.long)
    length_tensor = torch.tensor(lengths, dtype=torch.long)
    num_parties = torch.tensor([len(seq.party_names) for seq in batch], dtype=torch.long)

    speakers: List[List[str]] = []
    emotions: List[List[str]] = []

    for i, seq in enumerate(batch):
        t = seq.num_turns
        padded_features[i, :t, :] = torch.from_numpy(seq.features)
        padded_party_ids[i, :t] = torch.from_numpy(seq.party_ids)
        padded_label_ids[i, :t] = torch.from_numpy(seq.label_ids)
        padded_utterance_ids[i, :t] = torch.from_numpy(seq.utterance_ids)
        turn_mask[i, :t] = True
        speakers.append(seq.speakers)
        emotions.append(seq.emotions)

    return {
        "features": padded_features,
        "party_ids": padded_party_ids,
        "label_ids": padded_label_ids,
        "utterance_ids": padded_utterance_ids,
        "mask": turn_mask,
        "dialogue_ids": dialogue_ids,
        "lengths": length_tensor,
        "num_parties": num_parties,
        "speakers": speakers,
        "emotions": emotions,
    }
