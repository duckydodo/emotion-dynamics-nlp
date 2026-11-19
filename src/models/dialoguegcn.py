"""
DialogueGCN: Graph Convolutional Network for Emotion Recognition in Conversation.
Adapted for MELD in emotion-dynamics-nlp based on Ghosal et al. (EMNLP 2019).

Components:
- Sequential BiGRU Context Encoder: Captures local discourse flow across utterances.
- Dialogue Graph Construction: Connects utterances within past and future temporal windows.
- Relational Graph Convolution (RGCN) with Similarity Attention:
  - Intra-speaker relation (r_intra): captures speaker self-dependency and emotional inertia.
  - Inter-speaker relation (r_inter): captures interpersonal emotional influence.
- Residual connections and dropout regularization.
- Emotion Classification Head: Projects final node embeddings to 7 MELD emotion logits.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DialogueGCNConfig:
    """
    Configuration hyperparameters for DialogueGCN.
    """
    feature_dim: int = 768         # Frozen DistilBERT [CLS] embedding size
    hidden_dim: int = 128          # Node hidden state size (64 per direction in BiGRU)
    num_classes: int = 7           # Number of emotion classes in MELD
    num_relations: int = 2         # 0: intra-speaker, 1: inter-speaker
    num_layers: int = 2            # Number of RGCN layers
    dropout: float = 0.1           # Dropout probability
    window_past: int = 10          # Context window into past turns
    window_future: int = 10        # Context window into future turns

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DialogueGCNConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_json(self, path: Union[str, Path]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "DialogueGCNConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


class RGCNLayer(nn.Module):
    """
    Relational Graph Convolution layer with dynamic similarity attention
    (Schlichtkrull et al. 2018; Ghosal et al. 2019).
    Supports both 2D (T, D) and 3D batched (B, T, D) inputs.
    """

    def __init__(self, in_features: int, out_features: int, num_relations: int = 2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations

        self.w_self = nn.Linear(in_features, out_features, bias=False)
        self.w_rel = nn.Parameter(
            torch.empty(num_relations, in_features, out_features)
        )
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.attn_proj = nn.Linear(in_features, in_features, bias=False)

        nn.init.xavier_uniform_(self.w_rel)

    def forward(
        self,
        h: torch.Tensor,
        adj: torch.Tensor,
        rel_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        h : torch.Tensor of shape (B, T, in_features) or (T, in_features)
            Node feature representations.
        adj : torch.Tensor of shape (B, T, T) or (T, T)
            Binary adjacency matrix indicating valid edges.
        rel_mask : torch.Tensor of shape (B, num_relations, T, T) or (num_relations, T, T)
            Binary relational masks for each relation type.

        Returns
        -------
        out : torch.Tensor of shape (B, T, out_features) or (T, out_features)
            Updated node representations after relational convolution and ReLU.
        """
        is_2d = (h.dim() == 2)
        if is_2d:
            h = h.unsqueeze(0)
            adj = adj.unsqueeze(0)
            rel_mask = rel_mask.unsqueeze(0)

        B, T, _ = h.shape

        # 1. Dynamic Similarity Attention: S_{i,j} = h_i W_alpha h_j^T
        proj = self.attn_proj(h)  # (B, T, in_features)
        scores = torch.matmul(h, proj.transpose(-1, -2))  # (B, T, T)
        scores = scores.masked_fill(~adj.bool(), -1e9)
        alpha = torch.softmax(scores, dim=-1)  # (B, T, T)
        alpha = alpha.masked_fill(~adj.bool(), 0.0)
        alpha = torch.nan_to_num(alpha, nan=0.0)

        # 2. Self-transformation
        out = self.w_self(h)  # (B, T, out_features)

        # 3. Message passing across relations
        for r in range(self.num_relations):
            w_r = self.w_rel[r]  # (in_features, out_features)
            r_mask = rel_mask[:, r, :, :]  # (B, T, T)
            weighted_adj = alpha * r_mask  # (B, T, T)
            h_transformed = torch.matmul(h, w_r)  # (B, T, out_features)
            msg = torch.matmul(weighted_adj, h_transformed)  # (B, T, out_features)
            out = out + msg

        out = torch.relu(out + self.bias)

        if is_2d:
            out = out.squeeze(0)
        return out


class DialogueGCN(nn.Module):
    """
    DialogueGCN architecture for Conversational Emotion Recognition (EMNLP 2019).
    Combines sequential BiGRU context encoding with relational graph convolution
    and dynamic similarity attention over multi-party conversational dialogue.
    """

    def __init__(self, config: Optional[DialogueGCNConfig] = None):
        super().__init__()
        self.config = config or DialogueGCNConfig()

        # 1. Sequential BiGRU Context Encoder
        self.bigru = nn.GRU(
            input_size=self.config.feature_dim,
            hidden_size=self.config.hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )

        # 2. Graph Convolution Layers
        self.layers = nn.ModuleList([
            RGCNLayer(
                in_features=self.config.hidden_dim,
                out_features=self.config.hidden_dim,
                num_relations=self.config.num_relations,
            )
            for _ in range(self.config.num_layers)
        ])

        # 3. Regularization & Classification Head
        self.dropout = nn.Dropout(self.config.dropout)
        self.classifier = nn.Linear(self.config.hidden_dim, self.config.num_classes)

    def count_parameters(self) -> int:
        """Returns total count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def build_graph(
        self,
        party_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        window_past: Optional[int] = None,
        window_future: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Constructs adjacency and relational masks for batched or single dialogues.

        Parameters
        ----------
        party_ids : torch.Tensor of shape (B, T_max) or (T,)
            Local party indices (0 <= party_id < num_parties).
        lengths : Optional[torch.Tensor] of shape (B,)
            Lengths of each dialogue in the batch. If None, assumes full length T_max.
        window_past : Optional[int]
            Past context window (defaults to config.window_past).
        window_future : Optional[int]
            Future context window (defaults to config.window_future).
        device : Optional[torch.device]
            Device on which to allocate graph tensors.

        Returns
        -------
        adj : torch.Tensor of shape (B, T_max, T_max) or (T, T)
            Binary adjacency matrix.
        rel_mask : torch.Tensor of shape (B, num_relations, T_max, T_max) or (num_relations, T, T)
            Relational masks (0: intra-speaker, 1: inter-speaker).
        """
        is_2d = (party_ids.dim() == 1)
        if is_2d:
            party_ids = party_ids.unsqueeze(0)  # (1, T)

        B, T_max = party_ids.shape
        target_device = device or party_ids.device
        wp = window_past if window_past is not None else self.config.window_past
        wf = window_future if window_future is not None else self.config.window_future

        if lengths is None:
            lengths = torch.full((B,), T_max, dtype=torch.long, device=target_device)
        else:
            lengths = lengths.to(target_device)

        adj = torch.zeros(B, T_max, T_max, device=target_device)
        rel_mask = torch.zeros(B, self.config.num_relations, T_max, T_max, device=target_device)

        for b in range(B):
            L = int(lengths[b].item())
            if L <= 1:
                # Single-utterance dialogue has no neighbor edges
                continue

            for i in range(L):
                start = max(0, i - wp)
                end = min(L, i + wf + 1)
                for j in range(start, end):
                    if i != j:
                        adj[b, i, j] = 1.0
                        if party_ids[b, i] == party_ids[b, j]:
                            rel_mask[b, 0, i, j] = 1.0  # Intra-speaker (self-dependency)
                        else:
                            rel_mask[b, 1, i, j] = 1.0  # Inter-speaker (interpersonal)

        if is_2d:
            adj = adj.squeeze(0)
            rel_mask = rel_mask.squeeze(0)

        return adj, rel_mask

    def forward(
        self,
        features: torch.Tensor,
        party_ids: Optional[torch.Tensor] = None,
        speakers: Optional[Union[List[List[str]], List[str]]] = None,
        mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        adj: Optional[torch.Tensor] = None,
        rel_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for DialogueGCN supporting single and batched dialogues.

        Parameters
        ----------
        features : torch.Tensor of shape (B, T_max, feature_dim) or (T, feature_dim)
            Frozen utterance representations.
        party_ids : Optional[torch.Tensor] of shape (B, T_max) or (T,)
            Local party indices.
        speakers : Optional[List[List[str]] or List[str]]
            Speaker names used to derive party_ids if party_ids is None.
        mask : Optional[torch.Tensor] of shape (B, T_max) or (T,)
            Boolean mask (True for valid turns, False for padding).
        lengths : Optional[torch.Tensor] of shape (B,)
            Lengths of each dialogue sequence in the batch.
        adj : Optional[torch.Tensor]
            Pre-computed adjacency matrix (optional optimization).
        rel_mask : Optional[torch.Tensor]
            Pre-computed relation mask (optional optimization).

        Returns
        -------
        logits : torch.Tensor of shape (B, T_max, num_classes) or (T, num_classes)
            Unnormalized emotion prediction logits for every turn.
        """
        is_2d = (features.dim() == 2)
        if is_2d:
            features = features.unsqueeze(0)  # (1, T, D)
            if party_ids is not None and party_ids.dim() == 1:
                party_ids = party_ids.unsqueeze(0)
            if mask is not None and mask.dim() == 1:
                mask = mask.unsqueeze(0)

        B, T_max, D = features.shape
        device = features.device

        # Infer lengths if not explicitly supplied
        if lengths is None:
            if mask is not None:
                lengths = mask.sum(dim=1).long()
            else:
                lengths = torch.full((B,), T_max, dtype=torch.long, device=device)
        else:
            lengths = lengths.to(device)

        # Resolve party_ids from speaker strings if needed
        if party_ids is None:
            if speakers is not None:
                party_ids = self._speakers_to_party_ids(speakers, B, T_max, device)
            else:
                party_ids = torch.zeros(B, T_max, dtype=torch.long, device=device)
        else:
            party_ids = party_ids.to(device)

        # 1. Sequential BiGRU Context Encoding with sequence packing for exact padding isolation
        if B > 1 and lengths is not None and (lengths < T_max).any():
            lengths_cpu = lengths.cpu().clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                features,
                lengths_cpu,
                batch_first=True,
                enforce_sorted=False,
            )
            packed_out, _ = self.bigru(packed)
            h_seq, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out,
                batch_first=True,
                total_length=T_max,
            )
        else:
            h_seq, _ = self.bigru(features)  # (B, T_max, hidden_dim)

        # 2. Build graph topology if not provided
        if adj is None or rel_mask is None:
            adj, rel_mask = self.build_graph(
                party_ids=party_ids,
                lengths=lengths,
                device=device,
            )

        # 3. Relational Graph Convolution Layers with Residual Connections
        h = h_seq
        for layer in self.layers:
            h_next = layer(h, adj, rel_mask) + h
            h = self.dropout(h_next)

        # 4. Classification Head: (B, T_max, num_classes)
        logits = self.classifier(h)

        if is_2d:
            logits = logits.squeeze(0)
        return logits

    def _speakers_to_party_ids(
        self,
        speakers: Union[List[List[str]], List[str]],
        B: int,
        T_max: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Helper to map speaker names to dynamic integer party IDs."""
        party_ids = torch.zeros(B, T_max, dtype=torch.long, device=device)
        if B == 1 and isinstance(speakers[0], str):
            # Single dialogue list of strings
            spk_list: List[str] = speakers  # type: ignore
            party_map: Dict[str, int] = {}
            for t, spk in enumerate(spk_list):
                if spk not in party_map:
                    party_map[spk] = len(party_map)
                party_ids[0, t] = party_map[spk]
        else:
            # Batch of dialogues
            for b in range(B):
                spk_list_b = speakers[b] if b < len(speakers) else []
                party_map = {}
                for t, spk in enumerate(spk_list_b):
                    if spk not in party_map:
                        party_map[spk] = len(party_map)
                    party_ids[b, t] = party_map[spk]
        return party_ids

    def predict_dialogue(
        self,
        features: torch.Tensor,
        party_ids: Optional[torch.Tensor] = None,
        speakers: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Convenience method for inference on a single dialogue sequence.

        Parameters
        ----------
        features : torch.Tensor of shape (T, feature_dim)
        party_ids : Optional[torch.Tensor] of shape (T,)
        speakers : Optional[List[str]] of length T

        Returns
        -------
        logits : torch.Tensor of shape (T, num_classes)
        """
        self.eval()
        with torch.no_grad():
            return self.forward(features=features, party_ids=party_ids, speakers=speakers)

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute masked cross-entropy loss over active dialogue turns.

        Parameters
        ----------
        logits : torch.Tensor of shape (B, T_max, num_classes) or (T, num_classes)
        targets : torch.Tensor of shape (B, T_max) or (T,)
        mask : Optional[torch.Tensor] of shape (B, T_max) or (T,)
        class_weights : Optional[torch.Tensor] of shape (num_classes,)

        Returns
        -------
        loss : scalar torch.Tensor
        """
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
            targets = targets.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        B, T_max, C = logits.shape
        loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

        targets_flat = targets.reshape(B * T_max).clone()
        if mask is not None:
            targets_flat[~mask.reshape(B * T_max)] = -100

        logits_flat = logits.reshape(B * T_max, C)
        return loss_fn(logits_flat, targets_flat)

    def save_checkpoint(
        self,
        path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save model state dict, configuration, and metadata atomically.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint: Dict[str, Any] = {
            "architecture": "DialogueGCN",
            "config": self.config.to_dict(),
            "model_state_dict": self.state_dict(),
            "metadata": metadata or {},
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        temp_path = path.with_suffix(".tmp")
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, path)

    @classmethod
    def load_checkpoint(
        cls,
        path: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
    ) -> Tuple["DialogueGCN", Dict[str, Any]]:
        """
        Load checkpoint, restore configuration, and return initialized model and metadata.
        """
        checkpoint = torch.load(path, map_location=device)
        config_data = checkpoint.get("config", {})
        config = DialogueGCNConfig.from_dict(config_data)

        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(device)
        model.eval()

        metadata = checkpoint.get("metadata", {})
        return model, metadata
