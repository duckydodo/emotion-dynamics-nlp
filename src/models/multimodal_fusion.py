"""
Isolated Multimodal Fusion Architecture for Emotion Dynamics in Conversation.

Provides:
- GatedMultimodalFusion: Robust gated multimodal projection with missing-modality masking.
- MultimodalDialogueRNN: Sequential speaker-state tracking on fused multimodal representations.
- MultimodalDialogueGCN: Relational graph convolution on fused multimodal representations.

Supports both Bimodal (Text + Audio) and Tri-Modal (Text + Audio + Video) configurations.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.dialoguernn import DialogueRNN, DialogueRNNConfig
from src.models.dialoguegcn import DialogueGCN, DialogueGCNConfig


class GatedMultimodalFusion(nn.Module):
    """
    Gated Multimodal Fusion with Modality Availability Masking.
    
    Transforms individual textual (768-d), acoustic (768-d), and visual (512-d)
    embeddings into an aligned shared latent space (out_dim=256).
    Applies learned feature-wise sigmoid gating modulated by binary modality
    availability masks (audio_mask, video_mask, text_mask) so missing or corrupt
    clips cleanly fall back to available modalities without injecting NaNs or noise.
    """

    def __init__(
        self,
        text_dim: int = 768,
        audio_dim: int = 768,
        video_dim: int = 512,
        out_dim: int = 256,
        modalities: Tuple[str, ...] = ("text", "audio", "video"),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.video_dim = video_dim
        self.out_dim = out_dim
        self.modalities = tuple(modalities)
        self.dropout_p = dropout

        dim_map = {
            "text": text_dim,
            "audio": audio_dim,
            "video": video_dim,
        }

        # Per-modality projection: LayerNorm -> Linear -> GELU -> Dropout
        self.input_norms = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        self.gates = nn.ModuleDict()

        for m in self.modalities:
            if m not in dim_map:
                raise ValueError(f"Unsupported modality '{m}'. Supported: {list(dim_map.keys())}")
            in_d = dim_map[m]
            self.input_norms[m] = nn.LayerNorm(in_d)
            self.projections[m] = nn.Sequential(
                nn.Linear(in_d, out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            # Feature-wise gating projection: maps projected feature to gating scores
            self.gates[m] = nn.Linear(out_dim, out_dim)

        # Final fusion layer: residual projection + LayerNorm
        self.fusion_norm = nn.LayerNorm(out_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        text: torch.Tensor,
        audio: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        video_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for Gated Multimodal Fusion.

        Parameters
        ----------
        text : torch.Tensor of shape (B, T, text_dim) or (T, text_dim)
        audio : Optional[torch.Tensor] of shape (B, T, audio_dim) or (T, audio_dim)
        video : Optional[torch.Tensor] of shape (B, T, video_dim) or (T, video_dim)
        audio_mask : Optional[torch.Tensor] of shape (B, T) or (T,) or (B, T, 1)
            1.0 if audio is valid, 0.0 if missing/corrupt.
        video_mask : Optional[torch.Tensor] of shape (B, T) or (T,) or (B, T, 1)
            1.0 if video is valid, 0.0 if missing/corrupt.
        text_mask : Optional[torch.Tensor] of shape (B, T) or (T,) or (B, T, 1)
            1.0 if text is valid (defaults to 1.0 everywhere).

        Returns
        -------
        u_t : torch.Tensor of shape (B, T, out_dim) or (T, out_dim)
            Fused multimodal utterance representations.
        """
        is_2d = (text.dim() == 2)
        if is_2d:
            text = text.unsqueeze(0)
            if audio is not None:
                audio = audio.unsqueeze(0)
            if video is not None:
                video = video.unsqueeze(0)
            if audio_mask is not None:
                audio_mask = audio_mask.unsqueeze(0)
            if video_mask is not None:
                video_mask = video_mask.unsqueeze(0)
            if text_mask is not None:
                text_mask = text_mask.unsqueeze(0)

        B, T, _ = text.shape
        device = text.device
        dtype = text.dtype

        modality_inputs: Dict[str, torch.Tensor] = {"text": text}
        if "audio" in self.modalities:
            if audio is not None:
                modality_inputs["audio"] = audio
            else:
                modality_inputs["audio"] = torch.zeros(B, T, self.audio_dim, device=device, dtype=dtype)
        if "video" in self.modalities:
            if video is not None:
                modality_inputs["video"] = video
            else:
                modality_inputs["video"] = torch.zeros(B, T, self.video_dim, device=device, dtype=dtype)

        # Standardize masks to (B, T, 1)
        modality_masks: Dict[str, torch.Tensor] = {}
        for m in self.modalities:
            if m == "text":
                raw_m = text_mask if text_mask is not None else torch.ones(B, T, device=device, dtype=dtype)
            elif m == "audio":
                raw_m = audio_mask if audio_mask is not None else (
                    torch.ones(B, T, device=device, dtype=dtype) if audio is not None else torch.zeros(B, T, device=device, dtype=dtype)
                )
            elif m == "video":
                raw_m = video_mask if video_mask is not None else (
                    torch.ones(B, T, device=device, dtype=dtype) if video is not None else torch.zeros(B, T, device=device, dtype=dtype)
                )
            else:
                raw_m = torch.ones(B, T, device=device, dtype=dtype)

            # Ensure shape is (B, T, 1)
            raw_m = raw_m.to(device=device, dtype=dtype)
            if raw_m.dim() == 2:
                raw_m = raw_m.unsqueeze(-1)
            elif raw_m.dim() == 1:
                raw_m = raw_m.view(1, -1, 1)
            modality_masks[m] = raw_m

        # Project, gate, and mask each modality
        gated_features: List[torch.Tensor] = []
        active_counts = torch.zeros(B, T, 1, device=device, dtype=dtype)

        for m in self.modalities:
            inp = modality_inputs[m].to(device=device, dtype=dtype)
            mask_m = modality_masks[m]  # (B, T, 1)

            # Replace any stray NaNs or Infs in raw missing inputs with zero
            inp = torch.nan_to_num(inp, nan=0.0, posinf=0.0, neginf=0.0)

            # 1. Project
            normed = self.input_norms[m](inp)
            h_m = self.projections[m](normed)  # (B, T, out_dim)

            # 2. Learned feature-wise sigmoid gate
            g_m = torch.sigmoid(self.gates[m](h_m))  # (B, T, out_dim)

            # 3. Apply availability mask
            effective_gate = g_m * mask_m  # Broadcasts (B, T, out_dim) * (B, T, 1)
            h_gated = effective_gate * h_m  # (B, T, out_dim)

            gated_features.append(h_gated)
            active_counts = active_counts + mask_m

        # Sum gated features across available modalities
        fused = torch.stack(gated_features, dim=0).sum(dim=0)  # (B, T, out_dim)

        # Scale by number of active modalities to maintain stable variance
        safe_counts = torch.clamp(active_counts, min=1.0)
        scaled_fused = fused / torch.sqrt(safe_counts)

        # Residual transformation + LayerNorm
        u_t = self.fusion_norm(scaled_fused + self.fusion_mlp(scaled_fused))  # (B, T, out_dim)

        if is_2d:
            u_t = u_t.squeeze(0)

        return u_t


class MultimodalDialogueRNN(nn.Module):
    """
    Multimodal DialogueRNN.
    Fuses textual, acoustic, and visual utterance representations via GatedMultimodalFusion,
    then executes causal conversational state tracking via DialogueRNN.
    """

    def __init__(
        self,
        text_dim: int = 768,
        audio_dim: int = 768,
        video_dim: int = 512,
        fusion_dim: int = 256,
        modalities: Tuple[str, ...] = ("text", "audio", "video"),
        fusion_dropout: float = 0.3,
        rnn_config: Optional[DialogueRNNConfig] = None,
        num_classes: int = 7,
    ):
        super().__init__()
        self.modalities = tuple(modalities)
        self.fusion = GatedMultimodalFusion(
            text_dim=text_dim,
            audio_dim=audio_dim,
            video_dim=video_dim,
            out_dim=fusion_dim,
            modalities=modalities,
            dropout=fusion_dropout,
        )

        config = rnn_config or DialogueRNNConfig()
        config.feature_dim = fusion_dim
        config.num_classes = num_classes
        self.config = config
        self.rnn = DialogueRNN(config=self.config)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward_dialogue(
        self,
        text: torch.Tensor,
        party_ids: torch.Tensor,
        audio: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        video_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Single unbatched dialogue inference."""
        u_t = self.fusion(
            text=text,
            audio=audio,
            video=video,
            audio_mask=audio_mask,
            video_mask=video_mask,
            text_mask=text_mask,
        )
        return self.rnn.forward_dialogue(u_t, party_ids=party_ids)

    def forward(
        self,
        text: torch.Tensor,
        party_ids: torch.Tensor,
        audio: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        video_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Batched forward pass.
        
        Parameters
        ----------
        text : (B, T, text_dim)
        party_ids : (B, T)
        audio : Optional (B, T, audio_dim)
        video : Optional (B, T, video_dim)
        """
        u_t = self.fusion(
            text=text,
            audio=audio,
            video=video,
            audio_mask=audio_mask,
            video_mask=video_mask,
            text_mask=text_mask,
        )
        return self.rnn(u_t, party_ids=party_ids, mask=mask, lengths=lengths)


class MultimodalDialogueGCN(nn.Module):
    """
    Multimodal DialogueGCN.
    Fuses textual, acoustic, and visual utterance representations via GatedMultimodalFusion,
    then executes relational graph convolution with similarity attention via DialogueGCN.
    """

    def __init__(
        self,
        text_dim: int = 768,
        audio_dim: int = 768,
        video_dim: int = 512,
        fusion_dim: int = 256,
        modalities: Tuple[str, ...] = ("text", "audio", "video"),
        fusion_dropout: float = 0.3,
        gcn_config: Optional[DialogueGCNConfig] = None,
        num_classes: int = 7,
    ):
        super().__init__()
        self.modalities = tuple(modalities)
        self.fusion = GatedMultimodalFusion(
            text_dim=text_dim,
            audio_dim=audio_dim,
            video_dim=video_dim,
            out_dim=fusion_dim,
            modalities=modalities,
            dropout=fusion_dropout,
        )

        config = gcn_config or DialogueGCNConfig()
        config.feature_dim = fusion_dim
        config.num_classes = num_classes
        self.config = config
        self.gcn = DialogueGCN(config=self.config)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def build_graph(
        self,
        party_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        window_past: Optional[int] = None,
        window_future: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.gcn.build_graph(
            party_ids=party_ids,
            lengths=lengths,
            window_past=window_past,
            window_future=window_future,
            device=device,
        )

    def forward(
        self,
        text: torch.Tensor,
        party_ids: Optional[torch.Tensor] = None,
        speakers: Optional[Union[List[List[str]], List[str]]] = None,
        audio: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        video_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        adj: Optional[torch.Tensor] = None,
        rel_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for Multimodal DialogueGCN.
        """
        u_t = self.fusion(
            text=text,
            audio=audio,
            video=video,
            audio_mask=audio_mask,
            video_mask=video_mask,
            text_mask=text_mask,
        )
        return self.gcn(
            features=u_t,
            party_ids=party_ids,
            speakers=speakers,
            mask=mask,
            lengths=lengths,
            adj=adj,
            rel_mask=rel_mask,
        )
