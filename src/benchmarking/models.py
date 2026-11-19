"""
Model definitions and benchmarking adapters for conversational emotion recognition architectures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.models.transformer import LABELS


# ============================================================================
# DialogueRNN PyTorch Implementation
# ============================================================================
class DialogueRNN(nn.Module):
    """
    DialogueRNN architecture (Majumder et al., AAAI 2019) for conversational emotion recognition.
    Tracks global conversational state (Global GRU), individual speaker states (Party GRU),
    and emotional state with attentive historical context.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 128,
        num_classes: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Global GRU: encodes current utterance + previous party state
        self.global_gru = nn.GRUCell(input_dim + hidden_dim, hidden_dim)

        # Party GRU: encodes current utterance + attention context
        self.party_gru = nn.GRUCell(input_dim + hidden_dim, hidden_dim)

        # Emotion GRU: encodes party state into emotion representation
        self.emotion_gru = nn.GRUCell(hidden_dim, hidden_dim)

        # Attention layer over past global states
        self.attention_linear = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        features: torch.Tensor,
        speakers: list[str],
    ) -> torch.Tensor:
        """
        features: (seq_len, input_dim) tensor of dialogue utterances.
        speakers: list of speaker IDs of length seq_len.
        Returns: (seq_len, num_classes) logits.
        """
        seq_len = features.size(0)
        device = features.device

        # Initialize global state and speaker states
        g_t = torch.zeros(1, self.hidden_dim, device=device)
        speaker_states: dict[str, torch.Tensor] = {}
        global_history: list[torch.Tensor] = []
        emotion_states: list[torch.Tensor] = []

        for t in range(seq_len):
            u_t = features[t : t + 1]  # (1, input_dim)
            spk = speakers[t]
            q_spk = speaker_states.get(
                spk, torch.zeros(1, self.hidden_dim, device=device)
            )

            # 1. Update Global GRU
            global_input = torch.cat([u_t, q_spk], dim=-1)
            g_t = self.global_gru(global_input, g_t)
            global_history.append(g_t)

            # 2. Compute Attention over past global states
            if len(global_history) > 1:
                hist_tensor = torch.cat(global_history[:-1], dim=0)  # (t, hidden_dim)
                attn_scores = torch.softmax(
                    torch.matmul(hist_tensor, self.attention_linear(g_t).T), dim=0
                )  # (t, 1)
                context_t = torch.sum(hist_tensor * attn_scores, dim=0, keepdim=True)
            else:
                context_t = torch.zeros(1, self.hidden_dim, device=device)

            # 3. Update Party GRU
            party_input = torch.cat([u_t, context_t], dim=-1)
            q_spk_next = self.party_gru(party_input, q_spk)
            speaker_states[spk] = q_spk_next

            # 4. Update Emotion GRU
            e_t = self.emotion_gru(q_spk_next, q_spk)
            emotion_states.append(e_t)

        all_emotions = torch.cat(emotion_states, dim=0)  # (seq_len, hidden_dim)
        logits = self.classifier(self.dropout(all_emotions))
        return logits


# ============================================================================
# DialogueGCN PyTorch Implementation
# ============================================================================
class RGCNLayer(nn.Module):
    """
    Relational Graph Convolution layer with similarity attention (Schlichtkrull et al. 2018; Ghosal et al. 2019).
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
        h: (N, in_features) node representations
        adj: (N, N) binary adjacency matrix
        rel_mask: (num_relations, N, N) binary relation masks (e.g. intra, inter)
        """
        N = h.size(0)
        # Compute similarity attention weights
        proj = self.attn_proj(h)  # (N, in_features)
        scores = torch.matmul(h, proj.T)  # (N, N)
        scores = scores.masked_fill(~adj.bool(), -1e9)
        alpha = torch.softmax(scores, dim=-1)  # (N, N)
        alpha = torch.nan_to_num(alpha, nan=0.0)

        out = self.w_self(h)  # (N, out_features)

        for r in range(self.num_relations):
            w_r = self.w_rel[r]  # (in_features, out_features)
            weighted_adj = alpha * rel_mask[r]  # (N, N)
            msg = torch.matmul(weighted_adj, torch.matmul(h, w_r))
            out = out + msg

        return torch.relu(out + self.bias)


class DialogueGCN(nn.Module):
    """
    DialogueGCN architecture (Ghosal et al., EMNLP 2019) with sequential BiGRU context encoding,
    relational graph convolution, and similarity attention.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 128,
        num_classes: int = 7,
        num_relations: int = 2,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Sequential context encoder
        self.bigru = nn.GRU(
            input_dim,
            hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )

        # Graph convolution layers
        self.layers = nn.ModuleList([
            RGCNLayer(hidden_dim, hidden_dim, num_relations=num_relations)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        features: torch.Tensor,
        speakers: list[str],
        window_past: int = 10,
        window_future: int = 10,
    ) -> torch.Tensor:
        """
        features: (seq_len, input_dim)
        speakers: list of speaker IDs of length seq_len
        """
        seq_len = features.size(0)
        device = features.device

        # 1. Sequential BiGRU encoding
        h_seq, _ = self.bigru(features.unsqueeze(0))  # (1, seq_len, hidden_dim)
        h = h_seq.squeeze(0)  # (seq_len, hidden_dim)

        # 2. Build adjacency and relation masks
        adj = torch.zeros(seq_len, seq_len, device=device)
        rel_mask = torch.zeros(2, seq_len, seq_len, device=device)  # 0: intra, 1: inter

        for i in range(seq_len):
            start = max(0, i - window_past)
            end = min(seq_len, i + window_future + 1)
            for j in range(start, end):
                if i != j:
                    adj[i, j] = 1.0
                    if speakers[i] == speakers[j]:
                        rel_mask[0, i, j] = 1.0  # Intra-speaker
                    else:
                        rel_mask[1, i, j] = 1.0  # Inter-speaker

        # 3. Graph convolutions with residual connections
        for layer in self.layers:
            h = layer(h, adj, rel_mask) + h
            h = self.dropout(h)

        # 4. Classification
        logits = self.classifier(h)
        return logits
