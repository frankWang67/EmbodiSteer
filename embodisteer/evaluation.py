"""Shared evaluation result naming helpers."""

from __future__ import annotations

import os


def evaluation_subdir(
    *,
    inference_space: str,
    guidance: str,
    baseline_method: str,
    reverse_cbf_task_threshold,
    obstacle: bool,
    obstacle_observation_noise=None,
) -> str:
    """Return a collision-resistant result directory for one policy method.

    The method name is part of the path so runs with the same checkpoint and
    environment cannot silently overwrite one another (for example joint CBF
    and joint GD).
    """
    inference_space = str(inference_space).lower().strip()
    guidance = str(guidance).lower().strip()
    baseline_method = str(baseline_method).lower().strip()
    if baseline_method:
        subdir = f"obstacle_baseline_{baseline_method}"
    elif inference_space == "joint":
        if guidance:
            if reverse_cbf_task_threshold is not None:
                threshold = f"{float(reverse_cbf_task_threshold):.8g}"
                subdir = (
                    "obstacle_joint_space_guidance_"
                    f"{guidance}_reverse_cbf_task_threshold_{threshold}"
                )
            else:
                subdir = f"obstacle_joint_space_guidance_{guidance}"
        else:
            subdir = (
                "obstacle_joint_space" if obstacle
                else "no_obstacle_joint_space"
            )
    elif guidance:
        subdir = f"obstacle_ee_space_guidance_{guidance}"
    else:
        subdir = "obstacle_ee_space" if obstacle else "no_obstacle_ee_space"

    if obstacle_observation_noise is not None:
        pos_std, size_std, rot_std = obstacle_observation_noise
        subdir = os.path.join(
            subdir,
            f"obs_noise_pos{float(pos_std):.6g}_"
            f"size{float(size_std):.6g}_rot{float(rot_std):.6g}",
        )
    return subdir


__all__ = ["evaluation_subdir"]
