from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.models.transformer import LABELS


def compute_class_weights(
    labels: Sequence[str] | pd.Series | np.ndarray,
    method: str = "inverse_frequency",
    alpha: float = 1.0,
    beta: float = 0.999,
    class_order: Sequence[str] = LABELS,
) -> torch.FloatTensor:
    """
    Compute class weights for cross-entropy loss exclusively from training labels.

    Parameters
    ----------
    labels : Sequence[str]
        Collection of training emotion labels. Must ONLY come from the training split.
    method : str
        Weighting strategy:
        - 'inverse_frequency': Standard balanced inverse frequency (w_c = N / (C * n_c)),
          or smoothed with power alpha (w_c ~ (1 / n_c)^alpha).
        - 'effective_samples': Class-balanced loss based on effective number of samples
          (Cui et al., CVPR 2019), where E_n = (1 - beta^n) / (1 - beta).
        - 'unweighted': Uniform weights (1.0 for all classes).
    alpha : float, default=1.0
        Smoothing exponent for inverse frequency. alpha=1.0 is standard inverse frequency;
        alpha=0.5 corresponds to square-root inverse frequency smoothing.
    beta : float, default=0.999
        Hyperparameter for effective number of samples, typically in [0.9, 0.9999].
    class_order : Sequence[str], default=LABELS
        Fixed ordering of emotion classes matching model head label2id indices.

    Returns
    -------
    torch.FloatTensor
        1D tensor of class weights of shape (num_classes,), normalized so sum(weights) == num_classes.
    """
    if isinstance(labels, (pd.Series, pd.DataFrame)):
        labels_series = labels.dropna().astype(str)
    else:
        labels_series = pd.Series(list(labels)).dropna().astype(str)

    counts = labels_series.value_counts()
    num_classes = len(class_order)
    total_samples = len(labels_series)

    if total_samples == 0:
        raise ValueError("Cannot compute class weights from an empty labels sequence.")

    count_array = np.array(
        [counts.get(cls_name, 0) for cls_name in class_order],
        dtype=np.float64,
    )

    if np.any(count_array == 0):
        missing = [cls for cls, cnt in zip(class_order, count_array) if cnt == 0]
        raise ValueError(f"Classes with 0 occurrences in training labels: {missing}")

    if method == "unweighted" or method == "none":
        weights = np.ones(num_classes, dtype=np.float64)

    elif method == "inverse_frequency":
        if alpha <= 0.0 or alpha > 2.0:
            raise ValueError(f"alpha must be in (0.0, 2.0], got {alpha}")
        # When alpha=1.0, this is standard balanced inverse frequency: N / (C * n_c)
        raw_weights = np.power(1.0 / count_array, alpha)
        # Normalize so mean is 1.0 (sum is num_classes)
        weights = raw_weights / np.sum(raw_weights) * num_classes

    elif method == "effective_samples":
        if beta <= 0.0 or beta >= 1.0:
            raise ValueError(f"beta must be in (0.0, 1.0), got {beta}")
        effective_num = (1.0 - np.power(beta, count_array)) / (1.0 - beta)
        raw_weights = 1.0 / effective_num
        # Normalize so mean is 1.0 (sum is num_classes)
        weights = raw_weights / np.sum(raw_weights) * num_classes

    else:
        raise ValueError(
            f"Unknown class weighting method '{method}'. "
            f"Supported methods: 'inverse_frequency', 'effective_samples', 'unweighted'."
        )

    return torch.tensor(weights, dtype=torch.float32)


def get_weighted_loss_fn(
    class_weights: torch.Tensor | None = None,
    device: torch.device | str = "cpu",
) -> nn.CrossEntropyLoss:
    """
    Construct a PyTorch CrossEntropyLoss criterion with optional class weights.
    """
    if class_weights is not None:
        if not isinstance(class_weights, torch.Tensor):
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        weight = class_weights.to(device)
    else:
        weight = None

    return nn.CrossEntropyLoss(weight=weight)
