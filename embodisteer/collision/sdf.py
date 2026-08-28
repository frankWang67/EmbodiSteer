"""Whole-body signed-distance reductions used by collision guidance."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def curobo_signed_distance_to_cbf_h(dist_signed: torch.Tensor) -> torch.Tensor:
    """Convert cuRobo ESDF sign convention to the CBF safety function.

    cuRobo uses positive values inside an obstacle; EmbodiSteer's CBF uses
    ``h >= 0`` for safe states. Therefore ``h = -distance``.
    """
    return -dist_signed


def aggregate_signed_distance(
    distances: torch.Tensor,
    mode: str = "topk",
    topk: int = 4,
    temperature: float = 20.0,
) -> torch.Tensor:
    """Reduce per-sphere ESDF values while preserving penetrations exactly."""
    normalized_mode = str(mode).lower().strip()
    if normalized_mode == "softmax":
        normalized_mode = "topk"
    if normalized_mode == "max":
        return distances.max(dim=-1).values
    if normalized_mode != "topk":
        raise ValueError("SDF aggregation mode must be 'max' or 'topk'")
    if distances.shape[-1] < 1:
        raise ValueError("SDF aggregation requires at least one robot sphere")

    k = min(max(int(topk), 1), int(distances.shape[-1]))
    top_values = torch.topk(distances, k=k, dim=-1).values
    max_value = top_values[..., 0]
    if k == 1:
        return max_value
    temp = torch.as_tensor(
        max(float(temperature), 1e-6),
        device=distances.device,
        dtype=distances.dtype,
    )
    shifted = top_values - max_value.unsqueeze(-1)
    smooth_topk = max_value + torch.log(
        torch.exp(shifted * temp).mean(dim=-1)
    ) / temp
    return torch.where(max_value > 0, max_value, smooth_topk)


def collision_penalty(
    signed_distance: torch.Tensor,
    safety_margin: float,
    loss_power: float = 2.0,
) -> torch.Tensor:
    """Return a hinge penalty for penetration into the safety margin."""
    penalty = functional.relu(signed_distance + float(safety_margin))
    if float(loss_power) != 1.0:
        penalty = penalty.pow(float(loss_power))
    return penalty


__all__ = [
    "aggregate_signed_distance",
    "collision_penalty",
    "curobo_signed_distance_to_cbf_h",
]
