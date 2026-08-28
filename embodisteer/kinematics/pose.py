"""Differentiable pose and Jacobian utilities used by EmbodiSteer.

These functions are deliberately independent of a policy object.  They form
the small kinematic contract shared by simulation, real-world inference and
unit tests; robot-specific FK/IK remains supplied by cuRobo.
"""

from __future__ import annotations

import torch

from .rotation import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)


def pose9d_to_mat(pose9d: torch.Tensor) -> torch.Tensor:
    """Convert ``[x, y, z, rot6d]`` poses to homogeneous matrices."""
    pos = pose9d[..., :3]
    rot = rotation_6d_to_matrix(pose9d[..., 3:])
    out = torch.zeros(
        (*pose9d.shape[:-1], 4, 4), device=pose9d.device, dtype=pose9d.dtype
    )
    out[..., :3, :3] = rot
    out[..., :3, 3] = pos
    out[..., 3, 3] = 1.0
    return out


def inv_se3(mat: torch.Tensor) -> torch.Tensor:
    """Invert rigid transforms without constructing a general matrix inverse."""
    rot = mat[..., :3, :3]
    pos = mat[..., :3, 3]
    rot_t = rot.transpose(-2, -1)
    out = torch.zeros_like(mat)
    out[..., :3, :3] = rot_t
    out[..., :3, 3] = -(rot_t @ pos.unsqueeze(-1)).squeeze(-1)
    out[..., 3, 3] = 1.0
    return out


def relative_pose9_to_absolute(
    rel_pose9: torch.Tensor, chunk_start_pose: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lift relative position/rotation-6D actions into a base frame."""
    base_pos = chunk_start_pose[:, :3]
    base_rot = axis_angle_to_matrix(chunk_start_pose[:, 3:6])
    rel_pos = rel_pose9[..., :3]
    rel_rot = rotation_6d_to_matrix(rel_pose9[..., 3:])
    abs_pos = (
        base_rot.unsqueeze(1) @ rel_pos.unsqueeze(-1)
    ).squeeze(-1) + base_pos.unsqueeze(1)
    abs_rot = base_rot.unsqueeze(1) @ rel_rot
    return abs_pos, abs_rot


def absolute_pose_to_relative9(
    abs_pos: torch.Tensor,
    abs_rot: torch.Tensor,
    chunk_start_pose: torch.Tensor,
) -> torch.Tensor:
    """Express absolute pose trajectories relative to the chunk start."""
    base_pos = chunk_start_pose[:, :3]
    base_rot = axis_angle_to_matrix(chunk_start_pose[:, 3:6])
    base_rot_t = base_rot.transpose(-2, -1)
    rel_pos = (
        base_rot_t.unsqueeze(1)
        @ (abs_pos - base_pos.unsqueeze(1)).unsqueeze(-1)
    ).squeeze(-1)
    rel_rot = base_rot_t.unsqueeze(1) @ abs_rot
    rel_rot6d = matrix_to_rotation_6d(rel_rot.reshape(-1, 3, 3)).reshape(
        *rel_rot.shape[:2], 6
    )
    return torch.cat([rel_pos, rel_rot6d], dim=-1)


def twist6_from_matrices(
    abs_pos_curr: torch.Tensor,
    abs_rot_curr: torch.Tensor,
    abs_pos_tgt: torch.Tensor,
    abs_rot_tgt: torch.Tensor,
) -> torch.Tensor:
    """Compute translational and rotational residuals for two pose batches."""
    batch, horizon = abs_pos_curr.shape[:2]
    pos_cur = abs_pos_curr.reshape(-1, 3)
    rot_cur = abs_rot_curr.reshape(-1, 3, 3)
    pos_tgt = abs_pos_tgt.reshape(-1, 3)
    rot_tgt = abs_rot_tgt.reshape(-1, 3, 3)
    dpos = pos_tgt - pos_cur
    drot = matrix_to_axis_angle(rot_tgt @ rot_cur.transpose(-2, -1))
    return torch.cat([dpos, drot], dim=-1).reshape(batch, horizon, 6)


def absolute_pose_delta_to_twist6(
    abs_pose9_curr: torch.Tensor,
    abs_pose9_tgt: torch.Tensor,
    cartesian_delta_mode: str = "geometric",
) -> torch.Tensor:
    """Convert two pose-9D trajectories to translational/rotational twists."""
    batch, horizon, _ = abs_pose9_curr.shape
    cur_flat = abs_pose9_curr.reshape(-1, 9)
    tgt_flat = abs_pose9_tgt.reshape(-1, 9)
    if cartesian_delta_mode == "se3_delta":
        t_cur = pose9d_to_mat(cur_flat)
        t_tgt = pose9d_to_mat(tgt_flat)
        t_delta = t_tgt @ inv_se3(t_cur)
        dpos = t_delta[:, :3, 3]
        drot = matrix_to_axis_angle(t_delta[:, :3, :3])
    elif cartesian_delta_mode == "geometric":
        pos_cur = cur_flat[:, :3]
        rot_cur = rotation_6d_to_matrix(cur_flat[:, 3:])
        pos_tgt = tgt_flat[:, :3]
        rot_tgt = rotation_6d_to_matrix(tgt_flat[:, 3:])
        dpos = pos_tgt - pos_cur
        drot = matrix_to_axis_angle(rot_tgt @ rot_cur.transpose(-2, -1))
    else:
        raise ValueError(
            "cartesian_delta_mode must be 'geometric' or 'se3_delta'"
        )
    return torch.cat([dpos, drot], dim=-1).reshape(batch, horizon, 6)


def damped_least_squares_pinv(
    twist: torch.Tensor, jacobian: torch.Tensor, damping: float
) -> torch.Tensor:
    """Map task-space twists to joint updates with a damped pseudoinverse."""
    j_t = jacobian.transpose(-2, -1)
    jjt = jacobian @ j_t
    eye = torch.eye(6, device=jacobian.device, dtype=jacobian.dtype).unsqueeze(0)
    system = jjt + float(damping) * eye
    solved = torch.linalg.solve(system, twist.unsqueeze(-1))
    return (j_t @ solved).squeeze(-1)


__all__ = [
    "absolute_pose_delta_to_twist6",
    "absolute_pose_to_relative9",
    "damped_least_squares_pinv",
    "inv_se3",
    "pose9d_to_mat",
    "relative_pose9_to_absolute",
    "twist6_from_matrices",
]
