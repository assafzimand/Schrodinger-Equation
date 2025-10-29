"""Metrics for evaluating model predictions.

This module provides metrics for comparing predicted and true solutions to the
Schrödinger equation, including phase-invariant relative L² error.
"""

import numpy as np
import torch


def phase_aligned_rel_l2_torch(
    h_pred: torch.Tensor, h_true: torch.Tensor
) -> torch.Tensor:
    """Compute phase-aligned relative L² error (PyTorch version for training).

    The global phase is optimized to best align h_pred with h_true before
    computing the relative L² error. This accounts for the gauge freedom in
    the Schrödinger equation where solutions differing by a constant phase
    are equivalent.

    Args:
        h_pred: Predicted complex field, shape [N] or [N, ...]
        h_true: True complex field, shape [N] or [N, ...]

    Returns:
        Scalar tensor: relative L² error after optimal phase alignment
    """
    # Flatten to [N] for phase estimation
    hp = h_pred.reshape(-1)
    ht = h_true.reshape(-1)

    # Compute inner product <ht, hp> = sum conj(ht)*hp
    inner = torch.sum(torch.conj(ht) * hp)

    # Phase that best aligns hp to ht
    phase = torch.angle(inner)

    # Align prediction by this phase
    aligned = h_pred * torch.exp(-1j * phase)

    # Compute relative L² error
    num = torch.linalg.norm(aligned - h_true)
    den = torch.linalg.norm(h_true)

    return num / den


def phase_aligned_rel_l2_numpy(h_pred: np.ndarray, h_true: np.ndarray) -> float:
    """Compute phase-aligned relative L² error (NumPy version for evaluation).

    The global phase is optimized to best align h_pred with h_true before
    computing the relative L² error. This accounts for the gauge freedom in
    the Schrödinger equation where solutions differing by a constant phase
    are equivalent.

    Args:
        h_pred: Predicted complex field, shape [N] or [N, ...]
        h_true: True complex field, shape [N] or [N, ...]

    Returns:
        Float: relative L² error after optimal phase alignment
    """
    # Flatten to [N] for phase estimation
    hp = h_pred.reshape(-1)
    ht = h_true.reshape(-1)

    # Compute inner product <ht, hp> = np.vdot(ht, hp) = sum conj(ht)*hp
    inner = np.vdot(ht, hp)

    # Phase that best aligns hp to ht
    phase = np.angle(inner)

    # Align prediction by this phase
    aligned = h_pred * np.exp(-1j * phase)

    # Compute relative L² error
    num = np.linalg.norm(aligned - h_true)
    den = np.linalg.norm(h_true)

    return float(num / den)

