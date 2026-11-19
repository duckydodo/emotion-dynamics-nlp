"""
Linear-Chain Conditional Random Field (CRF) for Conversational Emotion Recognition.

Features:
- Structured sequence modeling over dialogue turns.
- Separates emission potentials (DistilBERT projection) from transition potentials (A_{j, k}).
- Exact negative log-likelihood training via the Forward algorithm (log-sum-exp).
- Global Viterbi sequence decoding (full dialogue context).
- Causal online greedy decoding (turn-by-turn with past transition constraints only).
- Strictly preserves dialogue boundaries and masks padding.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearChainCRF(nn.Module):
    """
    Linear-Chain Conditional Random Field on top of utterance representations.
    """

    def __init__(
        self,
        input_dim: int = 768,
        num_classes: int = 7,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        # Linear emission layer
        self.emission_proj = nn.Linear(input_dim, num_classes)

        # Transition parameters: transitions[from_tag, to_tag]
        self.transitions = nn.Parameter(torch.empty(num_classes, num_classes))
        self.start_transitions = nn.Parameter(torch.empty(num_classes))
        self.end_transitions = nn.Parameter(torch.empty(num_classes))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize transition parameters uniformly."""
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.xavier_uniform_(self.emission_proj.weight)
        nn.init.zeros_(self.emission_proj.bias)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        features: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute emission logits from input features.
        Args:
            features: (batch_size, seq_len, input_dim)
        Returns:
            emissions: (batch_size, seq_len, num_classes)
        """
        return self.emission_proj(features)

    def _compute_gold_score(
        self,
        emissions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the unnormalized score of the true label sequence:
        Score(x, y) = sum_t (emission(y_t)) + sum_t (transition(y_{t-1}, y_t))
        """
        batch_size, seq_len, num_classes = emissions.shape
        score = torch.zeros(batch_size, device=emissions.device)

        # Clamp targets to avoid negative index (-100 padding) errors during indexing/gather
        safe_targets = targets.clamp(min=0)

        # Start transition + first emission
        first_tags = safe_targets[:, 0]
        score += self.start_transitions[first_tags]
        score += emissions[:, 0].gather(1, first_tags.unsqueeze(1)).squeeze(1)

        for t in range(1, seq_len):
            curr_mask = mask[:, t]
            prev_tags = safe_targets[:, t - 1]
            curr_tags = safe_targets[:, t]

            # Emission score at turn t
            emit_score = emissions[:, t].gather(1, curr_tags.unsqueeze(1)).squeeze(1)

            # Transition score from t-1 to t
            trans_score = self.transitions[prev_tags, curr_tags]

            step_score = emit_score + trans_score
            score = torch.where(curr_mask, score + step_score, score)

        # End transitions for each sequence based on true lengths
        lengths = mask.sum(dim=1).long()
        last_tags = safe_targets.gather(1, (lengths - 1).unsqueeze(1)).squeeze(1)
        score += self.end_transitions[last_tags]

        return score

    def _compute_partition_function(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log Z(x) using the forward algorithm in log-space with logsumexp.
        """
        batch_size, seq_len, num_classes = emissions.shape

        # Initial forward state: start_transitions + first emission
        # Shape: (batch_size, num_classes)
        forward_var = self.start_transitions.unsqueeze(0) + emissions[:, 0]

        for t in range(1, seq_len):
            curr_mask = mask[:, t].unsqueeze(1)  # (batch_size, 1)

            # Broadcast forward_var to (batch_size, num_classes, 1)
            # Broadcast transitions to (1, num_classes, num_classes) [from, to]
            # Broadcast emissions to (batch_size, 1, num_classes)
            emit_t = emissions[:, t].unsqueeze(1)  # (batch_size, 1, num_classes)

            # Total score moving from 'from_tag' to 'to_tag':
            # forward_var (from) + transitions (from -> to) + emit_t (to)
            next_var = forward_var.unsqueeze(2) + self.transitions.unsqueeze(0) + emit_t
            next_var = torch.logsumexp(next_var, dim=1)  # (batch_size, num_classes)

            # If masked out, keep previous forward_var
            forward_var = torch.where(curr_mask, next_var, forward_var)

        # Add end transitions
        forward_var = forward_var + self.end_transitions.unsqueeze(0)
        return torch.logsumexp(forward_var, dim=1)

    def compute_loss(
        self,
        emissions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute Negative Log-Likelihood (NLL) Loss:
        NLL = log Z(x) - Score(x, y)
        """
        log_z = self._compute_partition_function(emissions, mask)
        gold_score = self._compute_gold_score(emissions, targets, mask)
        loss = torch.mean(log_z - gold_score)
        return loss

    def decode(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> List[List[int]]:
        """
        Global Viterbi sequence decoding: returns the highest-scoring label sequence.
        """
        batch_size, seq_len, num_classes = emissions.shape
        device = emissions.device

        # viterbi_vars: (batch_size, num_classes)
        viterbi_vars = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        backpointers = []

        for t in range(1, seq_len):
            curr_mask = mask[:, t].unsqueeze(1)

            # (batch_size, from_tag, 1) + (1, from_tag, to_tag)
            broadcast_viterbi = viterbi_vars.unsqueeze(2) + self.transitions.unsqueeze(0)
            max_vars, best_tags = torch.max(broadcast_viterbi, dim=1)  # (batch_size, to_tag)

            # Add emission scores
            max_vars = max_vars + emissions[:, t]

            # If masked out, preserve previous
            next_viterbi = torch.where(curr_mask, max_vars, viterbi_vars)
            backpointers.append(best_tags)
            viterbi_vars = next_viterbi

        viterbi_vars = viterbi_vars + self.end_transitions.unsqueeze(0)
        lengths = mask.sum(dim=1).long()

        # Backtrack best path
        best_paths: List[List[int]] = []
        for b in range(batch_size):
            seq_l = lengths[b].item()
            best_last_tag = torch.argmax(viterbi_vars[b]).item()
            best_path = [best_last_tag]

            for t in range(seq_l - 2, -1, -1):
                best_tag = backpointers[t][b, best_path[-1]].item()
                best_path.append(best_tag)

            best_path.reverse()
            best_paths.append(best_path)

        return best_paths

    def decode_causal(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> List[List[int]]:
        """
        Causal online greedy decoding:
        At turn t, selects y_t = argmax (emission_{t, k} + transition_{y_{t-1}, k})
        without conditioning on future turns.
        """
        batch_size, seq_len, num_classes = emissions.shape
        lengths = mask.sum(dim=1).long()
        causal_paths: List[List[int]] = []

        for b in range(batch_size):
            seq_l = lengths[b].item()
            path: List[int] = []

            # Turn 0: start transition + emission
            first_scores = self.start_transitions + emissions[b, 0]
            first_tag = int(torch.argmax(first_scores).item())
            path.append(first_tag)

            # Subsequent turns: only transition from previous tag + current emission
            for t in range(1, seq_l):
                prev_tag = path[-1]
                step_scores = self.transitions[prev_tag] + emissions[b, t]
                step_tag = int(torch.argmax(step_scores).item())
                path.append(step_tag)

            causal_paths.append(path)

        return causal_paths
