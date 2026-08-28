import os
import sys

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

from embodisteer.policies.ee2joint import (
    EmbodiSteerJointPolicy,
)


def _make_solver(task_threshold, *, arm_dof=4, grad_clip=0.0):
    policy = EmbodiSteerJointPolicy.__new__(
        EmbodiSteerJointPolicy
    )
    torch.nn.Module.__init__(policy)
    policy.arm_dof = arm_dof
    policy.guidance_safety_margin = 0.5
    policy.guidance_task_pos_weight = 1.0
    policy.guidance_task_rot_weight = 0.4
    policy.guidance_cbf_lambda = 0.2
    policy.guidance_cbf_reverse_task_threshold = task_threshold
    policy.guidance_grad_clip = grad_clip
    return policy


def _task_metric(policy, jac):
    task_dim = jac.shape[-2]
    ctrl_dof = min(policy.arm_dof, jac.shape[-1])
    J = jac[..., :ctrl_dof].reshape(-1, task_dim, ctrl_dof)
    weights = torch.ones(task_dim, dtype=jac.dtype, device=jac.device)
    weights[:3] = policy.guidance_task_pos_weight
    if task_dim >= 6:
        weights[3:6] = policy.guidance_task_rot_weight
    eye = torch.eye(ctrl_dof, dtype=jac.dtype, device=jac.device).unsqueeze(0)
    return (J * weights.view(1, task_dim, 1)).transpose(-2, -1) @ J + (
        policy.guidance_cbf_lambda * eye
    )


def test_reverse_matches_standard_qp_when_budget_is_sufficient():
    torch.manual_seed(7)
    policy = _make_solver(task_threshold=100.0)
    jac = torch.randn(1, 3, 6, 4, dtype=torch.float64)
    grad_h = torch.randn(1, 3, 4, dtype=torch.float64)
    h_value = torch.tensor([[0.1, 0.35, 0.7]], dtype=torch.float64)
    scale = torch.tensor(1.0, dtype=torch.float64)

    dq_standard, rhs_standard, _, _ = policy._solve_batched_cbf_qp(
        jac, grad_h, h_value, scale
    )
    dq_reverse, rhs_reverse, _, _ = policy._solve_batched_reverse_cbf_qcqp(
        jac, grad_h, h_value, scale
    )

    torch.testing.assert_close(rhs_reverse, rhs_standard)
    torch.testing.assert_close(dq_reverse, dq_standard)


def test_reverse_saturates_task_budget_and_reduces_collision_cost():
    torch.manual_seed(11)
    threshold = 0.03
    policy = _make_solver(task_threshold=threshold)
    jac = torch.randn(1, 2, 6, 4, dtype=torch.float64)
    grad_h = torch.randn(1, 2, 4, dtype=torch.float64)
    h_value = torch.zeros(1, 2, dtype=torch.float64)
    scale = torch.tensor(1.0, dtype=torch.float64)

    dq, rhs, _, feasible = policy._solve_batched_reverse_cbf_qcqp(
        jac, grad_h, h_value, scale
    )

    H = _task_metric(policy, jac)
    dq_ctrl = dq[..., : policy.arm_dof].reshape(-1, policy.arm_dof)
    task_norm = torch.sqrt(
        (dq_ctrl.unsqueeze(-2) @ H @ dq_ctrl.unsqueeze(-1)).reshape(-1)
    )
    collision_improvement = (
        grad_h[..., : policy.arm_dof] * dq[..., : policy.arm_dof]
    ).sum(dim=-1)

    assert torch.all(feasible)
    assert torch.all(task_norm <= threshold + 1e-10)
    torch.testing.assert_close(
        task_norm,
        torch.full_like(task_norm, threshold),
        atol=1e-10,
        rtol=1e-10,
    )
    assert torch.all(collision_improvement > 0.0)
    assert torch.all(torch.relu(rhs - collision_improvement) < rhs)


def test_reverse_returns_zero_when_safe_or_collision_gradient_is_degenerate():
    torch.manual_seed(19)
    policy = _make_solver(task_threshold=0.1)
    jac = torch.randn(1, 2, 6, 4, dtype=torch.float64)
    grad_h = torch.randn(1, 2, 4, dtype=torch.float64)
    grad_h[:, 1] = 0.0
    h_value = torch.tensor([[0.6, 0.0]], dtype=torch.float64)

    dq, rhs, denom, feasible = policy._solve_batched_reverse_cbf_qcqp(
        jac, grad_h, h_value, torch.tensor(1.0, dtype=torch.float64)
    )

    assert rhs[0, 0] == 0.0
    assert not feasible[0, 1]
    assert denom[0, 1] == 0.0
    assert torch.all(dq == 0.0)
    assert torch.all(torch.isfinite(dq))


def test_reverse_keeps_uncontrolled_joints_zero_and_limit_after_joint_clip():
    torch.manual_seed(23)
    threshold = 0.04
    policy = _make_solver(task_threshold=threshold, arm_dof=3, grad_clip=0.002)
    jac = torch.randn(1, 1, 6, 5, dtype=torch.float64)
    grad_h = torch.randn(1, 1, 5, dtype=torch.float64)
    h_value = torch.zeros(1, 1, dtype=torch.float64)

    dq, _, _, _ = policy._solve_batched_reverse_cbf_qcqp(
        jac, grad_h, h_value, torch.tensor(1.0, dtype=torch.float64)
    )

    assert torch.all(dq[..., 3:] == 0.0)
    assert torch.all(dq[..., :3].abs() <= policy.guidance_grad_clip)
    H = _task_metric(policy, jac)
    dq_ctrl = dq[..., :3].reshape(-1, 3)
    task_norm = torch.sqrt(
        (dq_ctrl.unsqueeze(-2) @ H @ dq_ctrl.unsqueeze(-1)).reshape(-1)
    )
    assert torch.all(task_norm <= threshold + 1e-10)


if __name__ == "__main__":
    test_reverse_matches_standard_qp_when_budget_is_sufficient()
    test_reverse_saturates_task_budget_and_reduces_collision_cost()
    test_reverse_returns_zero_when_safe_or_collision_gradient_is_degenerate()
    test_reverse_keeps_uncontrolled_joints_zero_and_limit_after_joint_clip()
    print("reverse CBF QCQP tests passed")
