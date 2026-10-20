import torch

K_VALUES = [0.5, 1, 1.5, 2, 2.5, 3]
L2_FILTER_BASE_METHODS = ["mean", "std", "mean_std", "mean_plus_std"]
SPECIAL_TOKEN_EXCLUSION_METHODS = [
    "filtered_mean_pooling",
    "filtered_std_pooling",
    "filtered_mean_std_pooling",
    "filtered_mean_plus_std_pooling",
]
L2_CLS_BASE_METHODS = ["mean_cls", "mean_plus_cls"]
MEDIAN_POOLING_METHODS = [
    "median_pooling",
    "median_std_pooling",
    "mean_median_pooling",
    "mean_plus_median_pooling",
]
STABLE_DIMENSION_POOLING_METHODS = [
    "mean_div_std_pooling",
    "mean_inv_std_weighted_pooling",
    "mean_sigmoid_std_weighted_pooling",
]
def _expanded_mask(attention_mask: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
    return attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()


def _k_suffix(k_value: float) -> str:
    return str(k_value).replace(".", "_")


def _l2_zscore_filtered_mask(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    k_value: float,
) -> torch.Tensor:
    """Keep valid tokens whose sentence-local L2 z-score is within +/- K."""
    valid_mask = attention_mask.bool()
    valid_float = valid_mask.float()
    counts = torch.clamp(valid_float.sum(dim=1, keepdim=True), min=1.0)

    token_l2 = torch.linalg.vector_norm(hidden_states, ord=2, dim=-1)
    mean_l2 = torch.sum(token_l2 * valid_float, dim=1, keepdim=True) / counts
    centered_l2 = (token_l2 - mean_l2) * valid_float
    variance_l2 = torch.sum(centered_l2 * centered_l2, dim=1, keepdim=True) / counts
    std_l2 = torch.sqrt(torch.clamp(variance_l2, min=0.0))
    z_scores = (token_l2 - mean_l2) / torch.clamp(std_l2, min=1e-12)

    filtered_mask = valid_mask & (torch.abs(z_scores) <= k_value)
    has_remaining_tokens = filtered_mask.any(dim=1, keepdim=True)
    filtered_mask = torch.where(has_remaining_tokens, filtered_mask, valid_mask)
    return filtered_mask.to(dtype=attention_mask.dtype)


def _exclude_special_token_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Exclude CLS and final non-padding SEP-like token; keep other non-PAD tokens."""
    filtered_mask = attention_mask.clone()
    filtered_mask[:, 0] = 0
    lengths = attention_mask.sum(dim=1)
    for row_index, length in enumerate(lengths.tolist()):
        if length > 1:
            filtered_mask[row_index, length - 1] = 0

    has_remaining_tokens = filtered_mask.bool().any(dim=1, keepdim=True)
    return torch.where(has_remaining_tokens, filtered_mask, attention_mask)


def cls_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Use the first token representation."""
    return hidden_states[:, 0, :]


def mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean over non-padding tokens."""
    mask = _expanded_mask(attention_mask, hidden_states)
    summed = torch.sum(hidden_states * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def mean_cls_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Concatenate masked mean and CLS vectors."""
    return torch.cat(
        [
            mean_pooling(hidden_states, attention_mask),
            cls_pooling(hidden_states, attention_mask),
        ],
        dim=1,
    )


def mean_plus_cls_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Add masked mean and CLS vectors element-wise."""
    return mean_pooling(hidden_states, attention_mask) + cls_pooling(hidden_states, attention_mask)


def _dimension_zone_pooling_with_stats(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    k: int | float,
    mode: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Dimension-level value filtering with aggregate logging stats."""
    if mode not in {"mean", "std", "mean_std", "mean_plus_std"}:
        raise ValueError(f"Unsupported dimension-zone pooling mode: {mode}")

    valid_mask = attention_mask.bool().unsqueeze(-1)
    mask = valid_mask.float()
    valid_mask_expanded = valid_mask.expand_as(last_hidden_state)
    valid_counts = torch.clamp(mask.sum(dim=1), min=1e-9).expand(-1, last_hidden_state.shape[-1])

    original_mean = torch.sum(last_hidden_state * mask, dim=1) / valid_counts
    centered = (last_hidden_state - original_mean.unsqueeze(1)) * mask
    variance = torch.sum(centered * centered, dim=1) / valid_counts
    original_std = torch.sqrt(torch.clamp(variance, min=0.0))

    lower = original_mean.unsqueeze(1) - float(k) * original_std.unsqueeze(1)
    upper = original_mean.unsqueeze(1) + float(k) * original_std.unsqueeze(1)
    keep_mask = valid_mask & (last_hidden_state >= lower) & (last_hidden_state <= upper)

    kept_counts = keep_mask.sum(dim=1)
    fallback_mask = kept_counts == 0
    effective_mask = torch.where(fallback_mask.unsqueeze(1), valid_mask_expanded, keep_mask)
    effective_counts = torch.where(fallback_mask, valid_counts, kept_counts.float())

    filtered_mean = torch.sum(last_hidden_state * effective_mask.float(), dim=1) / torch.clamp(
        effective_counts,
        min=1e-9,
    )
    filtered_centered = (last_hidden_state - filtered_mean.unsqueeze(1)) * effective_mask.float()
    filtered_variance = torch.sum(filtered_centered * filtered_centered, dim=1) / torch.clamp(
        effective_counts,
        min=1e-9,
    )
    filtered_std = torch.sqrt(torch.clamp(filtered_variance, min=0.0))

    if mode == "mean":
        pooled = filtered_mean
    elif mode == "std":
        pooled = filtered_std
    elif mode == "mean_std":
        pooled = torch.cat([filtered_mean, filtered_std], dim=1)
    else:
        pooled = filtered_mean + filtered_std

    total_values = int(valid_mask_expanded.sum().item())
    kept_values = int(keep_mask.sum().item())
    removed_values = total_values - kept_values
    sentence_dimension_count = int(kept_counts.numel())
    stats = {
        "total_values": total_values,
        "kept_values": kept_values,
        "removed_values": removed_values,
        "sentence_dimension_count": sentence_dimension_count,
        "fallback_count": int(fallback_mask.sum().item()),
    }
    return pooled, stats


def dimension_mean_zone_pooling_with_stats(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    k: int | float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Dimension-level filtered mean pooling with aggregate logging stats."""
    return _dimension_zone_pooling_with_stats(last_hidden_state, attention_mask, k, mode="mean")


def dimension_mean_zone_pooling(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    k: int | float,
) -> torch.Tensor:
    """
    Dimension-level mean-zone filtering.

    Inputs:
    - last_hidden_state: [B, T, H]
    - attention_mask: [B, T]
    - k: 1, 2, or 3

    For each sentence and each dimension:
    1. Ignore PAD tokens.
    2. Compute mean and std across tokens for each dimension.
    3. Keep values inside mean +/- k*std.
    4. Compute mean of kept values.
    5. If all values are removed for a dimension, fallback to original mean.
    6. Return sentence vectors with shape [B, H].
    """
    pooled, _ = dimension_mean_zone_pooling_with_stats(last_hidden_state, attention_mask, k)
    return pooled


def dimension_std_zone_pooling_with_stats(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    k: int | float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Dimension-level filtered std pooling with aggregate logging stats."""
    return _dimension_zone_pooling_with_stats(last_hidden_state, attention_mask, k, mode="std")


def dimension_std_zone_pooling(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    k: int | float,
) -> torch.Tensor:
    """
    Dimension-level std-zone filtering.

    Inputs:
    - last_hidden_state: [B, T, H]
    - attention_mask: [B, T]
    - k: 1, 2, or 3

    For each sentence and each dimension:
    1. Ignore PAD tokens.
    2. Compute mean and std across tokens for each dimension.
    3. Keep values inside mean +/- k*std.
    4. Compute std of kept values.
    5. If all values are removed for a dimension, fallback to original values.
    6. Return sentence vectors with shape [B, H].
    """
    pooled, _ = dimension_std_zone_pooling_with_stats(last_hidden_state, attention_mask, k)
    return pooled


def dimension_mean_std_zone_pooling_with_stats(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    k: int | float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Dimension-level filtered mean_std pooling with aggregate logging stats."""
    return _dimension_zone_pooling_with_stats(last_hidden_state, attention_mask, k, mode="mean_std")


def dimension_mean_std_zone_pooling(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    k: int | float,
) -> torch.Tensor:
    """
    Dimension-level mean_std-zone filtering.

    Inputs:
    - last_hidden_state: [B, T, H]
    - attention_mask: [B, T]
    - k: 1, 2, or 3

    For each sentence and each dimension:
    1. Ignore PAD tokens.
    2. Compute mean and std across tokens for each dimension.
    3. Keep values inside mean +/- k*std.
    4. Compute mean and std of kept values.
    5. If all values are removed for a dimension, fallback to original values.
    6. Return sentence vectors with shape [B, 2H].
    """
    pooled, _ = dimension_mean_std_zone_pooling_with_stats(last_hidden_state, attention_mask, k)
    return pooled


def dimension_mean_plus_std_zone_pooling_with_stats(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    k: int | float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Dimension-level filtered mean_plus_std pooling with aggregate logging stats."""
    return _dimension_zone_pooling_with_stats(last_hidden_state, attention_mask, k, mode="mean_plus_std")


def dimension_mean_plus_std_zone_pooling(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    k: int | float,
) -> torch.Tensor:
    """
    Dimension-level mean_plus_std-zone filtering.

    For each sentence and each dimension:
    1. Ignore PAD tokens.
    2. Compute mean and std across tokens for each dimension.
    3. Keep values inside mean +/- k*std.
    4. Compute mean and std of kept values.
    5. Return mean + std element-wise with shape [B, H].
    """
    pooled, _ = dimension_mean_plus_std_zone_pooling_with_stats(last_hidden_state, attention_mask, k)
    return pooled


def max_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Max over non-padding tokens."""
    mask = _expanded_mask(attention_mask, hidden_states).bool()
    masked_hidden = hidden_states.masked_fill(~mask, -1e9)
    return torch.max(masked_hidden, dim=1).values


def mean_max_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Concatenate mean and max pooled vectors."""
    return torch.cat(
        [
            mean_pooling(hidden_states, attention_mask),
            max_pooling(hidden_states, attention_mask),
        ],
        dim=1,
    )


def max_mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Concatenate max and mean pooled vectors."""
    return torch.cat(
        [
            max_pooling(hidden_states, attention_mask),
            mean_pooling(hidden_states, attention_mask),
        ],
        dim=1,
    )


def mean_plus_max_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Add masked mean and masked max element-wise."""
    return mean_pooling(hidden_states, attention_mask) + max_pooling(hidden_states, attention_mask)


def std_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Standard deviation over non-padding tokens."""
    mask = _expanded_mask(attention_mask, hidden_states)
    mean = mean_pooling(hidden_states, attention_mask)
    centered = (hidden_states - mean.unsqueeze(1)) * mask
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    variance = torch.sum(centered * centered, dim=1) / counts
    return torch.sqrt(torch.clamp(variance, min=0.0))


def mean_std_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Concatenate masked mean and masked standard deviation."""
    return torch.cat(
        [
            mean_pooling(hidden_states, attention_mask),
            std_pooling(hidden_states, attention_mask),
        ],
        dim=1,
    )


def mean_plus_std_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Add masked mean and masked standard deviation element-wise."""
    return mean_pooling(hidden_states, attention_mask) + std_pooling(hidden_states, attention_mask)


def filtered_mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling that excludes CLS and the final non-padding SEP-like token."""
    return mean_pooling(hidden_states, _exclude_special_token_mask(attention_mask))


def filtered_std_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Std pooling that excludes CLS and the final non-padding SEP-like token."""
    return std_pooling(hidden_states, _exclude_special_token_mask(attention_mask))


def filtered_mean_std_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Concatenate filtered mean and filtered std after excluding CLS/SEP."""
    filtered_mask = _exclude_special_token_mask(attention_mask)
    return torch.cat(
        [
            mean_pooling(hidden_states, filtered_mask),
            std_pooling(hidden_states, filtered_mask),
        ],
        dim=1,
    )


def filtered_mean_plus_std_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Add filtered mean and filtered std after excluding CLS/SEP."""
    filtered_mask = _exclude_special_token_mask(attention_mask)
    return mean_pooling(hidden_states, filtered_mask) + std_pooling(hidden_states, filtered_mask)


def median_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Median over non-padding tokens."""
    mask = _expanded_mask(attention_mask, hidden_states).bool()
    masked_hidden = hidden_states.masked_fill(~mask, float("inf"))
    sorted_hidden = torch.sort(masked_hidden, dim=1).values
    counts = torch.clamp(attention_mask.sum(dim=1), min=1).long()
    hidden_size = hidden_states.shape[-1]

    lower_index = ((counts - 1) // 2).view(-1, 1, 1).expand(-1, 1, hidden_size)
    upper_index = (counts // 2).view(-1, 1, 1).expand(-1, 1, hidden_size)
    lower = torch.gather(sorted_hidden, dim=1, index=lower_index).squeeze(1)
    upper = torch.gather(sorted_hidden, dim=1, index=upper_index).squeeze(1)
    return (lower + upper) / 2.0


def median_std_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Concatenate masked median and masked std."""
    return torch.cat(
        [
            median_pooling(hidden_states, attention_mask),
            std_pooling(hidden_states, attention_mask),
        ],
        dim=1,
    )


def mean_median_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Concatenate masked mean and masked median."""
    return torch.cat(
        [
            mean_pooling(hidden_states, attention_mask),
            median_pooling(hidden_states, attention_mask),
        ],
        dim=1,
    )


def mean_plus_median_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Add masked mean and masked median element-wise."""
    return mean_pooling(hidden_states, attention_mask) + median_pooling(hidden_states, attention_mask)


def mean_div_std_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Scale each mean dimension by inverse token-level std."""
    return mean_pooling(hidden_states, attention_mask) / torch.clamp(std_pooling(hidden_states, attention_mask), min=1e-6)


def mean_inv_std_weighted_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Weight mean dimensions more when token-level std is lower."""
    std = std_pooling(hidden_states, attention_mask)
    return mean_pooling(hidden_states, attention_mask) * (1.0 / (1.0 + std))


def mean_sigmoid_std_weighted_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Weight mean dimensions by sigmoid(-std)."""
    std = std_pooling(hidden_states, attention_mask)
    return mean_pooling(hidden_states, attention_mask) * torch.sigmoid(-std)


def _make_l2_filtered_pooling(pooling_fn, k_value: float, method_name: str):
    def l2_filtered_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        filtered_mask = _l2_zscore_filtered_mask(hidden_states, attention_mask, k_value)
        return pooling_fn(hidden_states, filtered_mask)

    l2_filtered_pooling.__name__ = method_name
    return l2_filtered_pooling


def _make_l2_filtered_cls_pooling(mode: str, k_value: float, method_name: str):
    def l2_filtered_cls_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        filtered_mask = _l2_zscore_filtered_mask(hidden_states, attention_mask, k_value)
        filtered_mean = mean_pooling(hidden_states, filtered_mask)
        cls_vector = cls_pooling(hidden_states, attention_mask)
        if mode == "mean_cls":
            return torch.cat([filtered_mean, cls_vector], dim=1)
        if mode == "mean_plus_cls":
            return filtered_mean + cls_vector
        raise ValueError(f"Unsupported L2 CLS mode: {mode}")

    l2_filtered_cls_pooling.__name__ = method_name
    return l2_filtered_cls_pooling


BASELINE_POOLING_METHODS = {
    "mean": mean_pooling,
    "std": std_pooling,
    "mean_std": mean_std_pooling,
    "mean_plus_std": mean_plus_std_pooling,
}


def get_l2_filter_method_names() -> list[str]:
    names = list(BASELINE_POOLING_METHODS)
    for k_value in K_VALUES:
        suffix = _k_suffix(k_value)
        for base_method in L2_FILTER_BASE_METHODS:
            names.append(f"{base_method}_{suffix}")
    return names


POOLING_METHODS = {
    **BASELINE_POOLING_METHODS,
    "cls_pooling": cls_pooling,
    "mean_pooling": mean_pooling,
    "mean_baseline": mean_pooling,
    "mean_cls": mean_cls_pooling,
    "mean_cls_pooling": mean_cls_pooling,
    "mean_plus_cls": mean_plus_cls_pooling,
    "mean_plus_cls_pooling": mean_plus_cls_pooling,
    "mean_pls_cls": mean_plus_cls_pooling,
    "mean_pls_cls_pooling": mean_plus_cls_pooling,
    "max_pooling": max_pooling,
    "mean_max_pooling": mean_max_pooling,
    "max_mean_pooling": max_mean_pooling,
    "mean_plus_max_pooling": mean_plus_max_pooling,
    "std_pooling": std_pooling,
    "mean_std_pooling": mean_std_pooling,
    "mean_plus_std_pooling": mean_plus_std_pooling,
    "filtered_mean_pooling": filtered_mean_pooling,
    "filtered_std_pooling": filtered_std_pooling,
    "filtered_mean_std_pooling": filtered_mean_std_pooling,
    "filtered_mean_plus_std_pooling": filtered_mean_plus_std_pooling,
    "median_pooling": median_pooling,
    "median_std_pooling": median_std_pooling,
    "mean_median_pooling": mean_median_pooling,
    "mean_plus_median_pooling": mean_plus_median_pooling,
    "mean_div_std_pooling": mean_div_std_pooling,
    "mean_inv_std_weighted_pooling": mean_inv_std_weighted_pooling,
    "mean_sigmoid_std_weighted_pooling": mean_sigmoid_std_weighted_pooling,
}

for _k_value in K_VALUES:
    _suffix = _k_suffix(_k_value)
    for _base_name, _pooling_fn in BASELINE_POOLING_METHODS.items():
        _method_name = f"{_base_name}_{_suffix}"
        POOLING_METHODS[_method_name] = _make_l2_filtered_pooling(_pooling_fn, _k_value, _method_name)

    POOLING_METHODS[f"mean_cls_{_suffix}"] = _make_l2_filtered_cls_pooling(
        "mean_cls",
        _k_value,
        f"mean_cls_{_suffix}",
    )
    POOLING_METHODS[f"mean_plus_cls_{_suffix}"] = _make_l2_filtered_cls_pooling(
        "mean_plus_cls",
        _k_value,
        f"mean_plus_cls_{_suffix}",
    )
    POOLING_METHODS[f"mean_pls_cls_{_suffix}"] = POOLING_METHODS[f"mean_plus_cls_{_suffix}"]

for _dim_zone_k in [0.5, 1, 1.5, 2, 2.5, 3]:
    _dim_zone_suffix = _k_suffix(_dim_zone_k)
    POOLING_METHODS[f"dim_mean_zone_{_dim_zone_suffix}std"] = (
        lambda hidden_states, attention_mask, k=_dim_zone_k: dimension_mean_zone_pooling(
            hidden_states,
            attention_mask,
            k,
        )
    )
    POOLING_METHODS[f"dim_std_zone_{_dim_zone_suffix}std"] = (
        lambda hidden_states, attention_mask, k=_dim_zone_k: dimension_std_zone_pooling(
            hidden_states,
            attention_mask,
            k,
        )
    )
    POOLING_METHODS[f"dim_mean_std_zone_{_dim_zone_suffix}std"] = (
        lambda hidden_states, attention_mask, k=_dim_zone_k: dimension_mean_std_zone_pooling(
            hidden_states,
            attention_mask,
            k,
        )
    )
    POOLING_METHODS[f"dim_mean_plus_std_zone_{_dim_zone_suffix}std"] = (
        lambda hidden_states, attention_mask, k=_dim_zone_k: dimension_mean_plus_std_zone_pooling(
            hidden_states,
            attention_mask,
            k,
        )
    )


def get_pooling_method(method_name: str):
    if method_name not in POOLING_METHODS:
        available = ", ".join(sorted(POOLING_METHODS))
        raise KeyError(f"Unknown pooling method {method_name!r}. Available: {available}")
    return POOLING_METHODS[method_name]


def get_l2_cls_filter_method_names() -> list[str]:
    names = []
    for k_value in K_VALUES:
        suffix = _k_suffix(k_value)
        names.extend([f"mean_cls_{suffix}", f"mean_plus_cls_{suffix}"])
    return names


def get_mean_cls_l2_method_names() -> list[str]:
    """Return the exact baseline, filtered-mean, concat, and plus suite."""
    suffixes = [_k_suffix(k_value) for k_value in K_VALUES]
    return [
        "cls_pooling",
        "mean_pooling",
        *[f"mean_{suffix}" for suffix in suffixes],
        *[f"mean_cls_{suffix}" for suffix in suffixes],
        *[f"mean_plus_cls_{suffix}" for suffix in suffixes],
    ]
