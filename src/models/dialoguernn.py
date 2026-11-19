"""
DialogueRNN: An Attentive RNN for Emotion Detection in Conversations (AAAI 2019).
Adapted for frozen DistilBERT representations on MELD in emotion-dynamics-nlp.

Components:
- Global Context GRU: Tracks conversational discourse flow across all interlocutors.
- Context Attention: Attends over past global context states.
- Speaker-Specific Party State GRU: Tracks individual emotional states for each dialogue party.
- Emotion Attention: Attends over the active speaker's own prior emotional states.
- Emotion GRU: Models speaker-specific internal emotional inertia.
- Emotion Classifier: Fully-connected projection to 7 emotion logits.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DialogueRNNConfig:
    """
    Configuration hyperparameters for DialogueRNN.
    """
    feature_dim: int = 768         # Frozen DistilBERT [CLS] embedding size
    projected_dim: int = 256       # Linear projection size with LayerNorm
    global_hidden_dim: int = 128   # Global Context GRU state size (D_g)
    party_hidden_dim: int = 128    # Party State GRU size (D_q)
    emotion_hidden_dim: int = 64   # Emotion GRU state size (D_e)
    fc_hidden_dim: int = 64        # Classification hidden size (D_h)
    num_classes: int = 7           # Number of emotion classes in MELD
    dropout: float = 0.2           # Dropout probability
    use_context_attention: bool = True
    use_emotion_attention: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "projected_dim": self.projected_dim,
            "global_hidden_dim": self.global_hidden_dim,
            "party_hidden_dim": self.party_hidden_dim,
            "emotion_hidden_dim": self.emotion_hidden_dim,
            "fc_hidden_dim": self.fc_hidden_dim,
            "num_classes": self.num_classes,
            "dropout": self.dropout,
            "use_context_attention": self.use_context_attention,
            "use_emotion_attention": self.use_emotion_attention,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DialogueRNNConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class DialogueRNN(nn.Module):
    """
    Causal DialogueRNN model for multi-party conversational emotion recognition.
    """

    def __init__(self, config: Optional[DialogueRNNConfig] = None):
        super().__init__()
        self.config = config or DialogueRNNConfig()

        # 1. Utterance Input Projection & Regularization
        self.input_proj = nn.Linear(self.config.feature_dim, self.config.projected_dim)
        self.layer_norm = nn.LayerNorm(self.config.projected_dim)
        self.dropout = nn.Dropout(self.config.dropout)

        # 2. Global Context GRU: g_t = GRU_G(g_{t-1}, [u_t'; q_{t-1}^{(p)}])
        self.global_gru = nn.GRUCell(
            input_size=self.config.projected_dim + self.config.party_hidden_dim,
            hidden_size=self.config.global_hidden_dim,
        )

        # 3. Context Attention: a_{t, j} = g_j^T W_alpha g_t
        if self.config.use_context_attention:
            self.context_attn = nn.Linear(
                self.config.global_hidden_dim,
                self.config.global_hidden_dim,
                bias=False,
            )

        # 4. Speaker-Specific Party State GRU: q_t^{(p)} = GRU_P(q_{t-1}^{(p)}, [u_t'; c_t])
        context_dim = self.config.global_hidden_dim if self.config.use_context_attention else self.config.global_hidden_dim
        self.party_gru = nn.GRUCell(
            input_size=self.config.projected_dim + context_dim,
            hidden_size=self.config.party_hidden_dim,
        )

        # 5. Emotion Attention: b_{t, j} = e_j^T W_beta q_t^{(p)}
        if self.config.use_emotion_attention:
            self.emotion_attn = nn.Linear(
                self.config.party_hidden_dim,
                self.config.emotion_hidden_dim,
                bias=False,
            )

        # 6. Emotion GRU: e_t = GRU_E(e_{t-1}^{(p)}, [q_t^{(p)}; c_t^{(e)}])
        emotion_ctx_dim = self.config.emotion_hidden_dim if self.config.use_emotion_attention else 0
        self.emotion_gru = nn.GRUCell(
            input_size=self.config.party_hidden_dim + emotion_ctx_dim,
            hidden_size=self.config.emotion_hidden_dim,
        )

        # 7. Classification Head: h_t = ReLU(W_e e_t + b_e); logits = W_l h_t + b_l
        self.fc1 = nn.Linear(self.config.emotion_hidden_dim, self.config.fc_hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.config.fc_hidden_dim, self.config.num_classes)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward_dialogue(
        self,
        features: torch.Tensor,
        party_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convenience inference method for a single unbatched dialogue.
        
        Parameters
        ----------
        features : torch.Tensor of shape (T, feature_dim)
        party_ids : torch.Tensor of shape (T,)

        Returns
        -------
        logits : torch.Tensor of shape (T, num_classes)
        """
        if features.dim() != 2:
            raise ValueError(f"Expected 2D tensor for single dialogue features (T, D), got {features.shape}")
        if party_ids.dim() != 1:
            raise ValueError(f"Expected 1D tensor for single dialogue party_ids (T,), got {party_ids.shape}")

        features_b = features.unsqueeze(0)    # (1, T, D)
        party_ids_b = party_ids.unsqueeze(0)  # (1, T)
        logits_b = self.forward(features_b, party_ids_b)
        return logits_b.squeeze(0)            # (T, num_classes)

    def forward(
        self,
        features: torch.Tensor,
        party_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Batched forward pass supporting variable-length dialogues.

        Parameters
        ----------
        features : torch.Tensor of shape (B, T_max, feature_dim)
            Frozen utterance representations.
        party_ids : torch.Tensor of shape (B, T_max)
            Local party indices (0 <= party_id < num_parties_in_dialogue).
        mask : Optional[torch.Tensor] of shape (B, T_max)
            Boolean mask (True for valid turns, False for padding).
        lengths : Optional[torch.Tensor] of shape (B,)
            Lengths of each dialogue sequence in the batch.

        Returns
        -------
        logits : torch.Tensor of shape (B, T_max, num_classes)
            Unnormalized emotion prediction logits for each turn.
        """
        B, T_max, _ = features.shape
        device = features.device

        # Determine number of party slots to allocate across this batch
        max_party_id = int(party_ids.max().item()) if party_ids.numel() > 0 else 0
        num_parties = max(max_party_id + 1, 1)

        # 1. Project input features
        proj_features = self.dropout(self.layer_norm(self.input_proj(features)))  # (B, T_max, projected_dim)

        # 2. Initialize recurrent state banks for all dialogues in this batch
        # Fresh zero state guarantee - no information carries over across calls
        g_prev = torch.zeros(B, self.config.global_hidden_dim, device=device)
        Q = torch.zeros(B, num_parties, self.config.party_hidden_dim, device=device)
        E_party = torch.zeros(B, num_parties, self.config.emotion_hidden_dim, device=device)
        E_all = torch.zeros(B, T_max, self.config.emotion_hidden_dim, device=device)

        g_history: List[torch.Tensor] = []
        all_logits: List[torch.Tensor] = []
        batch_indices = torch.arange(B, device=device)

        # 3. Step chronologically through turns t = 0 .. T_max - 1
        for t in range(T_max):
            u_t = proj_features[:, t, :]         # (B, projected_dim)
            p_t = party_ids[:, t]                # (B,)

            # (a) Previous party state of the active speaker
            q_prev = Q[batch_indices, p_t]       # (B, party_hidden_dim)

            # (b) Global Context GRU update: g_t = GRU_G(g_{t-1}, [u_t; q_{t-1}^{(p)}])
            g_input = torch.cat([u_t, q_prev], dim=-1)
            g_t = self.global_gru(g_input, g_prev)  # (B, global_hidden_dim)
            g_history.append(g_t)
            g_prev = g_t

            # (c) Context Attention over past and present global states: {g_0, ..., g_t}
            if self.config.use_context_attention:
                G_stack = torch.stack(g_history, dim=1)  # (B, t + 1, global_hidden_dim)
                # Query: g_t W_alpha
                query_g = self.context_attn(g_t).unsqueeze(2)  # (B, global_hidden_dim, 1)
                scores_g = torch.bmm(G_stack, query_g).squeeze(2)  # (B, t + 1)
                alpha_t = F.softmax(scores_g, dim=-1)  # (B, t + 1)
                c_t = torch.bmm(alpha_t.unsqueeze(1), G_stack).squeeze(1)  # (B, global_hidden_dim)
            else:
                c_t = g_t

            # (d) Party State GRU update: q_t^{(p)} = GRU_P(q_{t-1}^{(p)}, [u_t; c_t])
            party_input = torch.cat([u_t, c_t], dim=-1)
            q_t = self.party_gru(party_input, q_prev)  # (B, party_hidden_dim)

            # Invariant listener states: Update only the active speaker's party state
            Q_next = Q.clone()
            Q_next[batch_indices, p_t] = q_t
            Q = Q_next

            # (e) Emotion Attention over speaker's own prior emotional states
            e_prev = E_party[batch_indices, p_t]  # (B, emotion_hidden_dim)

            if self.config.use_emotion_attention:
                if t == 0:
                    # First turn in dialogue: no prior emotion states exist
                    c_t_e = torch.zeros(B, self.config.emotion_hidden_dim, device=device)
                else:
                    # Mask turns j < t where speaker was the current active party p_t
                    past_speakers = party_ids[:, :t]  # (B, t)
                    speaker_mask = (past_speakers == p_t.unsqueeze(1))  # (B, t)

                    E_past = E_all[:, :t, :]  # (B, t, emotion_hidden_dim)
                    query_e = self.emotion_attn(q_t).unsqueeze(2)  # (B, emotion_hidden_dim, 1)
                    scores_e = torch.bmm(E_past, query_e).squeeze(2)  # (B, t)

                    # Mask out turns from other interlocutors
                    scores_e = scores_e.masked_fill(~speaker_mask, -1e9)
                    has_prior = speaker_mask.any(dim=1)  # (B,)

                    # Where speaker has spoken previously, compute softmax; otherwise context is 0
                    attn_weights_e = F.softmax(scores_e, dim=-1)
                    attn_weights_e = torch.where(
                        has_prior.unsqueeze(1),
                        attn_weights_e,
                        torch.zeros_like(attn_weights_e),
                    )
                    c_t_e = torch.bmm(attn_weights_e.unsqueeze(1), E_past).squeeze(1)  # (B, emotion_hidden_dim)

                emotion_input = torch.cat([q_t, c_t_e], dim=-1)
            else:
                emotion_input = q_t

            # (f) Emotion GRU update: e_t = GRU_E(e_{t-1}^{(p)}, [q_t^{(p)}; c_t^{(e)}])
            e_t = self.emotion_gru(emotion_input, e_prev)  # (B, emotion_hidden_dim)

            # Record emotion states
            E_all = E_all.clone()
            E_all[:, t, :] = e_t

            E_party_next = E_party.clone()
            E_party_next[batch_indices, p_t] = e_t
            E_party = E_party_next

            # (g) Classification Layer: h_t = ReLU(W_e e_t + b_e); logits = W_l h_t + b_l
            h_t = self.dropout(self.relu(self.fc1(e_t)))
            logits_t = self.fc2(h_t)  # (B, num_classes)
            all_logits.append(logits_t)

        # 4. Stack into full dialogue sequence tensor: (B, T_max, num_classes)
        logits = torch.stack(all_logits, dim=1)
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

        Parameters
        ----------
        logits : torch.Tensor of shape (B, T_max, num_classes)
        targets : torch.Tensor of shape (B, T_max)
        mask : Optional[torch.Tensor] of shape (B, T_max)
        class_weights : Optional[torch.Tensor] of shape (num_classes,)

        Returns
        -------
        loss : scalar torch.Tensor
        """
        B, T_max, C = logits.shape
        loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

        targets_flat = targets.reshape(B * T_max).clone()
        if mask is not None:
            targets_flat[~mask.reshape(B * T_max)] = -100

        logits_flat = logits.reshape(B * T_max, C)
        return loss_fn(logits_flat, targets_flat)


# ==============================================================================
# DialogueRNN Ablation Study Architectures
# ==============================================================================


@dataclass
class DialogueRNNAblationConfig:
    """
    Configuration hyperparameters for DialogueRNN architectural ablations.

    Supports four validated experimental modes:
    1. Global Context Only: use_party_state=False, use_emotion_state=False (370,695 params)
    2. Global + Party States: use_party_state=True, use_emotion_state=False (617,223 params)
    3. Global + Emotion State: use_party_state=False, use_emotion_state=True (452,999 params)
    4. Full DialogueRNN: use_party_state=True, use_emotion_state=True (670,855 params)
    """
    feature_dim: int = 768         # Frozen DistilBERT [CLS] embedding size
    projected_dim: int = 256       # Linear projection size with LayerNorm
    global_hidden_dim: int = 128   # Global Context GRU state size (D_g)
    party_hidden_dim: int = 128    # Party State GRU size (D_q)
    emotion_hidden_dim: int = 64   # Emotion GRU state size (D_e)
    fc_hidden_dim: int = 64        # Classification hidden size (D_h)
    num_classes: int = 7           # Number of emotion classes in MELD
    dropout: float = 0.2           # Dropout probability
    use_party_state: bool = True   # Enable speaker-specific party state GRU bank
    use_emotion_state: bool = True # Enable emotion state recurrence
    use_context_attention: bool = True # Context attention over global dialogue history
    use_emotion_attention: bool = True # Emotion attention over speaker's own emotional history
    ablation_id: Optional[int] = None
    experiment_name: Optional[str] = None
    model_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DialogueRNNAblationConfig":
        """
        Create configuration from either a raw dictionary or a structured ablation JSON spec.
        """
        arch = data.get("architecture", data)
        valid_fields = set(cls.__dataclass_fields__.keys())
        params = {k: v for k, v in arch.items() if k in valid_fields}
        if "ablation_id" in data and "ablation_id" not in params:
            params["ablation_id"] = data["ablation_id"]
        if "experiment_name" in data and "experiment_name" not in params:
            params["experiment_name"] = data["experiment_name"]
        return cls(**params)

    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> "DialogueRNNAblationConfig":
        """
        Load configuration directly from an ablation JSON specification file.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


class DialogueRNNAblation(nn.Module):
    """
    Controlled modular DialogueRNN ablation architecture.

    Allows clean, isolated component ablation while guaranteeing exact parameter counts,
    causal recurrence, dialogue state reset, and strict 100% equivalence to full
    DialogueRNN when all components are active.
    """

    def __init__(self, config: Optional[DialogueRNNAblationConfig] = None):
        super().__init__()
        self.config = config or DialogueRNNAblationConfig()

        # 1. Utterance Input Projection & Regularization (Shared across all 4 modes)
        self.input_proj = nn.Linear(self.config.feature_dim, self.config.projected_dim)
        self.layer_norm = nn.LayerNorm(self.config.projected_dim)
        self.dropout = nn.Dropout(self.config.dropout)

        party_dim = (
            self.config.party_hidden_dim
            if (self.config.use_party_state and self.config.party_hidden_dim > 0)
            else 128
        )
        emotion_dim = (
            self.config.emotion_hidden_dim
            if (self.config.use_emotion_state and self.config.emotion_hidden_dim > 0)
            else 64
        )

        # 2. Global Context GRU
        if self.config.use_party_state:
            # Conditioned on projected utterance and speaker's prior party state
            self.global_gru = nn.GRUCell(
                input_size=self.config.projected_dim + party_dim,
                hidden_size=self.config.global_hidden_dim,
            )
        else:
            # Conditioned only on projected utterance (speaker-agnostic context)
            self.global_gru = nn.GRUCell(
                input_size=self.config.projected_dim,
                hidden_size=self.config.global_hidden_dim,
            )

        # 3. Context Attention over global context history
        if self.config.use_context_attention:
            self.context_attn = nn.Linear(
                self.config.global_hidden_dim,
                self.config.global_hidden_dim,
                bias=False,
            )
        else:
            self.context_attn = None

        # 4. Speaker-Specific Party State GRU Bank
        if self.config.use_party_state:
            self.party_gru = nn.GRUCell(
                input_size=self.config.projected_dim + self.config.global_hidden_dim,
                hidden_size=party_dim,
            )
        else:
            self.party_gru = None

        # 5. Emotion Attention over speaker's own emotional history
        # Only active when both party states and emotion state are enabled
        if self.config.use_party_state and self.config.use_emotion_state and self.config.use_emotion_attention:
            self.emotion_attn = nn.Linear(
                party_dim,
                emotion_dim,
                bias=False,
            )
        else:
            self.emotion_attn = None

        # 6. Emotion GRU
        if self.config.use_emotion_state:
            if self.config.use_party_state:
                # Mode 4: Full DialogueRNN (conditioned on party state and emotion attention)
                emotion_ctx_dim = emotion_dim if self.emotion_attn is not None else 0
                self.emotion_gru = nn.GRUCell(
                    input_size=party_dim + emotion_ctx_dim,
                    hidden_size=emotion_dim,
                )
            else:
                # Mode 3: Global + Emotion State without party states (conditioned on [u_t; c_t])
                self.emotion_gru = nn.GRUCell(
                    input_size=self.config.projected_dim + self.config.global_hidden_dim,
                    hidden_size=emotion_dim,
                )
        else:
            self.emotion_gru = None

        # 7. Classification Head
        # Classifier input dimension correctly reflects each ablation's terminal representation:
        # - Modes 3 & 4: Emotion state e_t (emotion_dim = 64)
        # - Mode 2: Active speaker party state q_t (party_dim = 128)
        # - Mode 1: Global context vector c_t (global_hidden_dim = 128)
        if self.config.use_emotion_state:
            classifier_in_dim = emotion_dim
        elif self.config.use_party_state:
            classifier_in_dim = party_dim
        else:
            classifier_in_dim = self.config.global_hidden_dim

        self.fc1 = nn.Linear(classifier_in_dim, self.config.fc_hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.config.fc_hidden_dim, self.config.num_classes)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_enabled_components(self) -> List[str]:
        """Return names of enabled components."""
        enabled = ["input_proj", "layer_norm", "global_gru"]
        if self.context_attn is not None:
            enabled.append("context_attn")
        if self.party_gru is not None:
            enabled.append("party_gru")
        if self.emotion_attn is not None:
            enabled.append("emotion_attn")
        if self.emotion_gru is not None:
            enabled.append("emotion_gru")
        enabled.extend(["fc1", "fc2"])
        return enabled

    def get_disabled_components(self) -> List[str]:
        """Return names of ablated/disabled components."""
        all_possible = ["party_gru", "emotion_attn", "emotion_gru", "context_attn"]
        return [c for c in all_possible if getattr(self, c, None) is None]

    def forward_dialogue(
        self,
        features: torch.Tensor,
        party_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convenience inference method for a single unbatched dialogue.
        """
        if features.dim() != 2:
            raise ValueError(f"Expected 2D tensor for single dialogue features (T, D), got {features.shape}")
        if party_ids.dim() != 1:
            raise ValueError(f"Expected 1D tensor for single dialogue party_ids (T,), got {party_ids.shape}")

        features_b = features.unsqueeze(0)    # (1, T, D)
        party_ids_b = party_ids.unsqueeze(0)  # (1, T)
        logits_b = self.forward(features_b, party_ids_b)
        return logits_b.squeeze(0)            # (T, num_classes)

    def forward(
        self,
        features: torch.Tensor,
        party_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Batched causal forward pass supporting variable-length dialogues.
        """
        B, T_max, _ = features.shape
        device = features.device

        party_dim = (
            self.config.party_hidden_dim
            if (self.config.use_party_state and self.config.party_hidden_dim > 0)
            else 128
        )
        emotion_dim = (
            self.config.emotion_hidden_dim
            if (self.config.use_emotion_state and self.config.emotion_hidden_dim > 0)
            else 64
        )

        if self.config.use_party_state:
            max_party_id = int(party_ids.max().item()) if party_ids.numel() > 0 else 0
            num_parties = max(max_party_id + 1, 1)
            Q = torch.zeros(B, num_parties, party_dim, device=device)
            if self.config.use_emotion_state:
                E_party = torch.zeros(B, num_parties, emotion_dim, device=device)
                E_all = torch.zeros(B, T_max, emotion_dim, device=device)

        if not self.config.use_party_state and self.config.use_emotion_state:
            e_shared = torch.zeros(B, emotion_dim, device=device)

        # 1. Project input features
        proj_features = self.dropout(self.layer_norm(self.input_proj(features)))  # (B, T_max, projected_dim)

        # 2. Initialize global recurrent state (clean dialogue start guarantee)
        g_prev = torch.zeros(B, self.config.global_hidden_dim, device=device)
        g_history: List[torch.Tensor] = []
        all_logits: List[torch.Tensor] = []
        batch_indices = torch.arange(B, device=device)

        # 3. Step chronologically through turns t = 0 .. T_max - 1
        for t in range(T_max):
            u_t = proj_features[:, t, :]  # (B, projected_dim)

            if self.config.use_party_state:
                p_t = party_ids[:, t]     # (B,)
                q_prev = Q[batch_indices, p_t]  # (B, party_dim)
                g_input = torch.cat([u_t, q_prev], dim=-1)
            else:
                g_input = u_t

            # Global Context GRU update
            g_t = self.global_gru(g_input, g_prev)
            g_history.append(g_t)
            g_prev = g_t

            # Context Attention over {g_0, ..., g_t}
            if self.context_attn is not None:
                G_stack = torch.stack(g_history, dim=1)  # (B, t + 1, global_hidden_dim)
                query_g = self.context_attn(g_t).unsqueeze(2)  # (B, global_hidden_dim, 1)
                scores_g = torch.bmm(G_stack, query_g).squeeze(2)  # (B, t + 1)
                alpha_t = F.softmax(scores_g, dim=-1)  # (B, t + 1)
                c_t = torch.bmm(alpha_t.unsqueeze(1), G_stack).squeeze(1)  # (B, global_hidden_dim)
            else:
                c_t = g_t

            # Determine representation for classification based on active components
            if self.config.use_party_state:
                # Update speaker-specific party state: q_t = GRU_P(q_{prev}, [u_t; c_t])
                party_input = torch.cat([u_t, c_t], dim=-1)
                q_t = self.party_gru(party_input, q_prev)

                Q_next = Q.clone()
                Q_next[batch_indices, p_t] = q_t
                Q = Q_next

                if self.config.use_emotion_state:
                    # Full DialogueRNN path
                    e_prev = E_party[batch_indices, p_t]
                    if self.emotion_attn is not None:
                        if t == 0:
                            c_t_e = torch.zeros(B, emotion_dim, device=device)
                        else:
                            past_speakers = party_ids[:, :t]  # (B, t)
                            speaker_mask = (past_speakers == p_t.unsqueeze(1))  # (B, t)
                            E_past = E_all[:, :t, :]  # (B, t, emotion_dim)
                            query_e = self.emotion_attn(q_t).unsqueeze(2)
                            scores_e = torch.bmm(E_past, query_e).squeeze(2)
                            scores_e = scores_e.masked_fill(~speaker_mask, -1e9)
                            has_prior = speaker_mask.any(dim=1)
                            attn_weights_e = F.softmax(scores_e, dim=-1)
                            attn_weights_e = torch.where(
                                has_prior.unsqueeze(1),
                                attn_weights_e,
                                torch.zeros_like(attn_weights_e),
                            )
                            c_t_e = torch.bmm(attn_weights_e.unsqueeze(1), E_past).squeeze(1)
                        emotion_input = torch.cat([q_t, c_t_e], dim=-1)
                    else:
                        emotion_input = q_t

                    e_t = self.emotion_gru(emotion_input, e_prev)
                    E_all = E_all.clone()
                    E_all[:, t, :] = e_t

                    E_party_next = E_party.clone()
                    E_party_next[batch_indices, p_t] = e_t
                    E_party = E_party_next

                    rep_t = e_t
                else:
                    # Ablation 2: Classify directly from active party state q_t
                    rep_t = q_t
            else:
                if self.config.use_emotion_state:
                    # Ablation 3: Shared unconditioned emotion GRU
                    emotion_input = torch.cat([u_t, c_t], dim=-1)
                    e_t = self.emotion_gru(emotion_input, e_shared)
                    e_shared = e_t
                    rep_t = e_t
                else:
                    # Ablation 1: Global context only, classify directly from c_t
                    rep_t = c_t

            # Classification
            h_t = self.dropout(self.relu(self.fc1(rep_t)))
            logits_t = self.fc2(h_t)
            all_logits.append(logits_t)

        logits = torch.stack(all_logits, dim=1)
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
        B, T_max, C = logits.shape
        loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

        targets_flat = targets.reshape(B * T_max).clone()
        if mask is not None:
            targets_flat[~mask.reshape(B * T_max)] = -100

        logits_flat = logits.reshape(B * T_max, C)
        return loss_fn(logits_flat, targets_flat)


def build_ablation_model(
    config_or_path: Union[str, Path, Dict[str, Any], DialogueRNNAblationConfig],
) -> DialogueRNNAblation:
    """
    Factory function to construct a DialogueRNNAblation model from a file path,
    dictionary, or DialogueRNNAblationConfig instance.
    """
    if isinstance(config_or_path, DialogueRNNAblationConfig):
        cfg = config_or_path
    elif isinstance(config_or_path, (str, Path)):
        cfg = DialogueRNNAblationConfig.from_json(config_or_path)
    elif isinstance(config_or_path, dict):
        cfg = DialogueRNNAblationConfig.from_dict(config_or_path)
    else:
        raise TypeError(f"Unsupported config type: {type(config_or_path)}")
    return DialogueRNNAblation(cfg)

