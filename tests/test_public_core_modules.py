import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embodisteer.collision import (
    aggregate_signed_distance,
    curobo_signed_distance_to_cbf_h,
)
from embodisteer.guidance import (
    logistic_guidance_strength,
    solve_batched_cbf_qp,
)
from embodisteer.kinematics import (
    absolute_pose_delta_to_twist6,
    damped_least_squares_pinv,
)
from embodisteer.policies import EmbodiSteerEESpacePolicy, EmbodiSteerJointPolicy


def test_public_policy_classes_have_public_modules():
    assert EmbodiSteerJointPolicy.__module__ == "embodisteer.policies.ee2joint"
    assert EmbodiSteerEESpacePolicy.__module__ == "embodisteer.policies.ee_space"


def test_identity_pose_has_zero_twist():
    pose = torch.zeros((2, 3, 9), dtype=torch.float64)
    pose[..., 3] = 1.0
    twist = absolute_pose_delta_to_twist6(pose, pose)
    torch.testing.assert_close(twist, torch.zeros_like(twist))


def test_damped_pseudoinverse_identity_jacobian():
    jacobian = torch.eye(6, dtype=torch.float64).unsqueeze(0)
    twist = torch.arange(1, 7, dtype=torch.float64).unsqueeze(0)
    damping = 0.2
    actual = damped_least_squares_pinv(twist, jacobian, damping)
    torch.testing.assert_close(actual, twist / (1.0 + damping))


def test_collision_sign_and_topk_penetration_preservation():
    distance = torch.tensor([[-0.3, -0.1, -0.2], [0.2, -0.1, 0.1]])
    h = curobo_signed_distance_to_cbf_h(distance)
    torch.testing.assert_close(h, -distance)
    reduced = aggregate_signed_distance(distance, mode="topk", topk=2)
    assert reduced[0] <= 0.0
    assert reduced[1] == distance[1].max()


def test_public_cbf_solver_satisfies_active_constraint():
    jacobian = torch.eye(6, dtype=torch.float64).reshape(1, 1, 6, 6)
    grad_h = torch.ones((1, 1, 6), dtype=torch.float64)
    h = torch.zeros((1, 1), dtype=torch.float64)
    dq, rhs, _, feasible = solve_batched_cbf_qp(
        jacobian,
        grad_h,
        h,
        torch.tensor(1.0, dtype=torch.float64),
        arm_dof=6,
        safety_margin=0.1,
        position_weight=1.0,
        rotation_weight=0.1,
        regularization=0.01,
    )
    improvement = (grad_h * dq).sum(dim=-1)
    assert torch.all(feasible)
    assert torch.all(improvement >= rhs - 1e-12)


def test_guidance_schedule_decreases_with_forward_step_index():
    early = logistic_guidance_strength(torch.tensor(0.0), torch.tensor(15.0))
    late = logistic_guidance_strength(torch.tensor(15.0), torch.tensor(15.0))
    assert early > late


if __name__ == "__main__":
    test_public_policy_classes_have_public_modules()
    test_identity_pose_has_zero_twist()
    test_damped_pseudoinverse_identity_jacobian()
    test_collision_sign_and_topk_penetration_preservation()
    test_public_cbf_solver_satisfies_active_constraint()
    test_guidance_schedule_decreases_with_forward_step_index()
    print("Public core module tests passed.")
