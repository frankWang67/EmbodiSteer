"""Batch adapters for the shared pose-representation utilities."""

from __future__ import annotations

import numpy as np

from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep


def batched_convert_pose_mat_rep(
    pose_mat: np.ndarray,
    base_pose_mat: np.ndarray,
    pose_rep: str = "abs",
    backward: bool = False,
) -> np.ndarray:
    """Apply pose conversion independently to each batch element."""
    if pose_mat.shape[0] != base_pose_mat.shape[0]:
        raise ValueError("pose_mat and base_pose_mat must have the same batch size")
    return np.stack(
        [
            convert_pose_mat_rep(
                pose_mat[index], base_pose_mat[index], pose_rep, backward
            )
            for index in range(pose_mat.shape[0])
        ],
        axis=0,
    )


__all__ = ["batched_convert_pose_mat_rep"]
