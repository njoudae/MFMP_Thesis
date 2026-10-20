"""Exact MFMP masking and aggregation operations from the final Arabic run.

The experiment notebooks remain the provenance-bearing executable artifacts.
This module exposes their core tensor operations for import and unit testing.
"""

from __future__ import annotations

from typing import Tuple

import torch


EPSILON = 1e-12


def calculate_sentence_statistics(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return valid-token mean, L2 magnitudes, absolute local z, and counts.

    Population variance (``ddof=0``) is calculated independently within each
    sentence over all attended tokens. Padding is excluded; attended special
    tokens are included, matching the final experiments.
    """
    valid = attention_mask.bool()
    valid_f = valid.to(hidden_states.dtype)
    counts = valid_f.sum(dim=1).clamp_min(1.0)

    mean_vectors = (hidden_states * valid_f.unsqueeze(-1)).sum(dim=1) / counts.unsqueeze(-1)
    magnitudes = torch.linalg.vector_norm(hidden_states, ord=2, dim=-1)
    mu = (magnitudes * valid_f).sum(dim=1) / counts
    centered = (magnitudes - mu.unsqueeze(1)) * valid_f
    variance = centered.pow(2).sum(dim=1) / counts
    sigma = torch.sqrt(variance).clamp_min(EPSILON)
    z = (magnitudes - mu.unsqueeze(1)) / sigma.unsqueeze(1)
    abs_z = torch.abs(z)
    return mean_vectors, magnitudes, abs_z, counts


def calculate_mfmp_from_hidden(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    mean_vectors: torch.Tensor,
    abs_z: torch.Tensor,
    k_lower: float,
    k_upper: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pool original contextual vectors retained by the inclusive MFMP mask.

    The rule is ``k_lower <= abs(z_i) <= k_upper``. If a sentence retains no
    token, its unfiltered attended-token mean is used as the fallback.
    """
    if k_lower > k_upper:
        raise ValueError("k_lower must not exceed k_upper")
    valid = attention_mask.bool()
    retained = valid & (abs_z >= k_lower) & (abs_z <= k_upper)
    retained_counts = retained.sum(dim=1)
    summed = (hidden_states * retained.unsqueeze(-1).to(hidden_states.dtype)).sum(dim=1)
    safe_counts = retained_counts.clamp_min(1).to(hidden_states.dtype).unsqueeze(-1)
    pooled = summed / safe_counts
    fallback_mask = retained_counts == 0
    pooled = torch.where(fallback_mask.unsqueeze(-1), mean_vectors, pooled)
    return pooled, retained_counts, fallback_mask
