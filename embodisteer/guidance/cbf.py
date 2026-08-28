"""Closed-form batched CBF guidance solvers.

The constraints used by EmbodiSteer contain one linear collision inequality per
trajectory state.  This permits a deterministic closed-form solve and avoids a
runtime dependency on a generic QP package.
"""

from __future__ import annotations

import torch


def _task_metric(
    jacobian: torch.Tensor,
    position_weight: float,
    rotation_weight: float,
    regularization: float,
) -> torch.Tensor:
    _, task_dim, controlled_dof = jacobian.shape
    weights = torch.ones(
        (task_dim,), device=jacobian.device, dtype=jacobian.dtype
    )
    weights[: min(3, task_dim)] = float(position_weight)
    if task_dim >= 6 and float(rotation_weight) > 0.0:
        weights[3:6] = float(rotation_weight)
    identity = torch.eye(
        controlled_dof, device=jacobian.device, dtype=jacobian.dtype
    ).unsqueeze(0)
    metric = (
        jacobian * weights.view(1, task_dim, 1)
    ).transpose(-2, -1) @ jacobian
    return metric + float(regularization) * identity


def _prepare_problem(
    jacobian: torch.Tensor,
    grad_h: torch.Tensor,
    h_value: torch.Tensor,
    constraint_scale: torch.Tensor,
    *,
    arm_dof: int,
    safety_margin: float,
    position_weight: float,
    rotation_weight: float,
    regularization: float,
):
    batch, horizon, task_dim, total_dof = jacobian.shape
    count = batch * horizon
    controlled_dof = int(min(max(int(arm_dof), 1), total_dof))
    task_jacobian = jacobian[..., :controlled_dof].reshape(
        count, task_dim, controlled_dof
    )
    collision_grad = grad_h[..., :controlled_dof].reshape(count, controlled_dof)
    h_flat = h_value.reshape(count)
    scale = constraint_scale if constraint_scale.ndim == 0 else constraint_scale.reshape(-1)[0]
    rhs = torch.clamp((float(safety_margin) - h_flat) * scale, min=0.0)
    metric = _task_metric(
        task_jacobian, position_weight, rotation_weight, regularization
    )
    metric_inv_grad = torch.linalg.solve(
        metric, collision_grad.unsqueeze(-1)
    ).squeeze(-1)
    denominator = (collision_grad * metric_inv_grad).sum(dim=-1)
    return (
        batch,
        horizon,
        total_dof,
        controlled_dof,
        rhs,
        metric,
        metric_inv_grad,
        denominator,
    )


def _restore_total_dof(
    dq_controlled: torch.Tensor,
    *,
    batch: int,
    horizon: int,
    total_dof: int,
    controlled_dof: int,
) -> torch.Tensor:
    if controlled_dof == total_dof:
        dq = dq_controlled
    else:
        dq = torch.zeros(
            (batch * horizon, total_dof),
            device=dq_controlled.device,
            dtype=dq_controlled.dtype,
        )
        dq[:, :controlled_dof] = dq_controlled
    return dq.reshape(batch, horizon, total_dof)


def solve_batched_cbf_qp(
    jacobian: torch.Tensor,
    grad_h: torch.Tensor,
    h_value: torch.Tensor,
    constraint_scale: torch.Tensor,
    *,
    arm_dof: int,
    safety_margin: float,
    position_weight: float,
    rotation_weight: float,
    regularization: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve the single-inequality EmbodiSteer CBF-QP in closed form."""
    (
        batch,
        horizon,
        total_dof,
        controlled_dof,
        rhs,
        _,
        metric_inv_grad,
        denominator,
    ) = _prepare_problem(
        jacobian,
        grad_h,
        h_value,
        constraint_scale,
        arm_dof=arm_dof,
        safety_margin=safety_margin,
        position_weight=position_weight,
        rotation_weight=rotation_weight,
        regularization=regularization,
    )
    epsilon = torch.tensor(
        1e-9, device=denominator.device, dtype=denominator.dtype
    )
    feasible = denominator > epsilon
    alpha = torch.zeros_like(rhs)
    alpha[feasible] = rhs[feasible] / denominator[feasible]
    dq_controlled = metric_inv_grad * alpha.unsqueeze(-1)
    dq = _restore_total_dof(
        dq_controlled,
        batch=batch,
        horizon=horizon,
        total_dof=total_dof,
        controlled_dof=controlled_dof,
    )
    return (
        dq,
        rhs.reshape(batch, horizon),
        denominator.reshape(batch, horizon),
        feasible.reshape(batch, horizon),
    )


def solve_batched_reverse_cbf_qcqp(
    jacobian: torch.Tensor,
    grad_h: torch.Tensor,
    h_value: torch.Tensor,
    constraint_scale: torch.Tensor,
    *,
    arm_dof: int,
    safety_margin: float,
    position_weight: float,
    rotation_weight: float,
    regularization: float,
    task_threshold: float,
    joint_clip: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve the reverse-CBF task-disturbance-constrained QCQP."""
    (
        batch,
        horizon,
        total_dof,
        controlled_dof,
        rhs,
        metric,
        metric_inv_grad,
        denominator,
    ) = _prepare_problem(
        jacobian,
        grad_h,
        h_value,
        constraint_scale,
        arm_dof=arm_dof,
        safety_margin=safety_margin,
        position_weight=position_weight,
        rotation_weight=rotation_weight,
        regularization=regularization,
    )
    epsilon = torch.tensor(
        1e-9, device=denominator.device, dtype=denominator.dtype
    )
    feasible = denominator > epsilon
    denominator_safe = torch.clamp(denominator, min=epsilon)
    threshold = torch.tensor(
        float(task_threshold), device=denominator.device, dtype=denominator.dtype
    )
    alpha_to_zero_cost = rhs / denominator_safe
    alpha_task_budget = threshold / torch.sqrt(denominator_safe)
    alpha = torch.minimum(alpha_to_zero_cost, alpha_task_budget)
    alpha = torch.where(feasible, alpha, torch.zeros_like(alpha))
    dq_controlled = metric_inv_grad * alpha.unsqueeze(-1)

    if float(joint_clip) > 0.0:
        dq_controlled = torch.clamp(
            dq_controlled, min=-float(joint_clip), max=float(joint_clip)
        )
    count = batch * horizon
    task_energy = (
        dq_controlled.unsqueeze(-2) @ metric @ dq_controlled.unsqueeze(-1)
    ).reshape(count)
    task_norm = torch.sqrt(torch.clamp(task_energy, min=0.0))
    task_rescale = torch.clamp(
        threshold / torch.clamp(task_norm, min=epsilon), max=1.0
    )
    dq_controlled = dq_controlled * task_rescale.unsqueeze(-1)
    dq = _restore_total_dof(
        dq_controlled,
        batch=batch,
        horizon=horizon,
        total_dof=total_dof,
        controlled_dof=controlled_dof,
    )
    return (
        dq,
        rhs.reshape(batch, horizon),
        denominator.reshape(batch, horizon),
        feasible.reshape(batch, horizon),
    )


__all__ = ["solve_batched_cbf_qp", "solve_batched_reverse_cbf_qcqp"]
