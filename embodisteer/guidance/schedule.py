"""Diffusion-step guidance schedules."""

from __future__ import annotations

import torch


def logistic_guidance_strength(
    step: torch.Tensor,
    num_diffusion_steps: torch.Tensor,
    *,
    midpoint: float = 0.7,
    steepness: float = 50.0,
) -> torch.Tensor:
    """Return the logistic schedule used by the paper implementation."""
    normalized = step / num_diffusion_steps
    return 1.0 / (
        1.0 + torch.exp(-float(steepness) * (float(midpoint) - normalized))
    )


def guidance_scale_at(
    index: int,
    num_steps: int,
    base_scale: float,
    *,
    use_schedule: bool,
    dtype: torch.dtype,
    device: torch.device,
    midpoint: float = 0.7,
    steepness: float = 50.0,
) -> torch.Tensor:
    """Evaluate the scalar guidance multiplier at a denoising index."""
    scale = torch.tensor(base_scale, device=device, dtype=dtype)
    if not use_schedule:
        return scale
    if num_steps <= 1:
        step = torch.tensor(0.0, device=device, dtype=dtype)
        count = torch.tensor(1.0, device=device, dtype=dtype)
    else:
        step = torch.tensor(float((num_steps - 1) - index), device=device, dtype=dtype)
        count = torch.tensor(float(num_steps - 1), device=device, dtype=dtype)
    return scale * logistic_guidance_strength(
        step, count, midpoint=midpoint, steepness=steepness
    )


__all__ = ["guidance_scale_at", "logistic_guidance_strength"]
