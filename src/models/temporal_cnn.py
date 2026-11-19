"""
Causal Temporal Convolutional Network (TCN) for Conversational Emotion Recognition.

Key Design Properties:
- Operates on pre-extracted 768-dim DistilBERT [CLS] embeddings.
- Strictly causal left-padding: prevents future conversational turns from leaking into past predictions.
- Dilated 1D convolutions with residual connections and layer normalization.
- Receptive field: RF = 15 turns across 3 dilated blocks (k=3, d=[1, 2, 4]).
- Dialogue-bounded: padded and masked per dialogue sequence.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1dBlock(nn.Module):
    """
    Dilated 1D Convolution with causal left-padding, residual connection, and layer norm.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation  # Left padding amount for causality

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=True,
        )
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Residual shortcut if channel dimensions differ
        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (batch_size, in_channels, seq_len)
        Output: (batch_size, out_channels, seq_len)
        """
        # Causal left padding: pad (left, right) on the last dimension (seq_len)
        padded = F.pad(x, (self.padding, 0))
        out = self.conv(padded)

        # LayerNorm expects (batch_size, seq_len, channels)
        out = out.transpose(1, 2)
        out = self.norm(out)
        out = out.transpose(1, 2)

        out = self.activation(out)
        out = self.dropout(out)

        res = self.shortcut(x)
        return out + res


class TemporalCNN(nn.Module):
    """
    Causal Temporal CNN for Conversational Emotion Recognition on Dialogue Sequences.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 7,
        kernel_size: int = 3,
        dilations: Tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.dropout_rate = dropout

        # Calculate theoretical receptive field
        # RF = 1 + sum_l (k - 1) * d_l
        self.receptive_field = 1 + sum((kernel_size - 1) * d for d in dilations)

        # Input linear projection to hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Stack of causal dilated convolution blocks
        self.blocks = nn.ModuleList([
            CausalConv1dBlock(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                dilation=d,
                dropout=dropout,
            )
            for d in dilations
        ])

        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Forward pass.
        Args:
            features: (batch_size, seq_len, input_dim)
            mask: (batch_size, seq_len) boolean tensor
            lengths: (batch_size,) sequence lengths
        Returns:
            logits: (batch_size, seq_len, num_classes)
        """
        # 1. Project input features
        h = self.input_proj(features)  # (batch_size, seq_len, hidden_dim)

        # 2. Reshape to (batch_size, hidden_dim, seq_len) for 1D convolution
        h = h.transpose(1, 2)

        # 3. Pass through causal dilated convolution blocks
        for block in self.blocks:
            h = block(h)

        # 4. Transpose back to (batch_size, seq_len, hidden_dim)
        h = h.transpose(1, 2)

        # 5. Classification head
        logits = self.classifier(h)  # (batch_size, seq_len, num_classes)

        return logits

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute masked cross-entropy loss over active dialogue turns.
        """
        batch_size, seq_len, num_classes = logits.shape
        flat_logits = logits.view(-1, num_classes)
        flat_targets = targets.view(-1)

        if mask is not None:
            flat_mask = mask.view(-1)
            flat_targets = flat_targets.clone()
            flat_targets[~flat_mask] = -100

        loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
        return loss_fn(flat_logits, flat_targets)
