from types import SimpleNamespace

import pytest
import torch

from embodisteer.kinematics import twist6_from_matrices, twist6_from_matrices_fast
from embodisteer.kinematics.rotation import axis_angle_to_matrix
from embodisteer.policies import EmbodiSteerJointPolicy


@pytest.mark.parametrize("use_policy_compile", [False, True])
def test_joint_policy_twist_uses_paper_fast_rotation_path(monkeypatch, use_policy_compile):
    twist_fn = twist6_from_matrices_fast
    if use_policy_compile:
        compile_fn = torch.compile

        def compile_on_cpu(fn, **kwargs):
            return compile_fn(fn, fullgraph=kwargs["fullgraph"], backend="eager")

        monkeypatch.setattr(torch, "compile", compile_on_cpu)
        policy = SimpleNamespace(
            arm_dof=6,
            _pk_chain=SimpleNamespace(
                jacobian_tensor=lambda q: torch.eye(6).expand(q.shape[0], 6, 6)
            ),
        )
        EmbodiSteerJointPolicy._compile_hot_functions(policy, torch.device("cpu"))
        twist_fn = policy._twist_fn
    current_position = torch.zeros((1, 1, 3), dtype=torch.float32)
    target_position = torch.tensor(
        [[[0.01, -0.02, 0.03]]], dtype=torch.float32
    )
    current_rotation = torch.eye(3, dtype=torch.float32).reshape(1, 1, 3, 3)
    target_axis_angle = torch.tensor(
        [[1.0e-4, 0.0, 0.0]], dtype=torch.float32
    )
    target_rotation = axis_angle_to_matrix(target_axis_angle).reshape(1, 1, 3, 3)

    actual = twist_fn(
        current_position,
        current_rotation,
        target_position,
        target_rotation,
    )
    # The paper's float32 acos path rounds this tiny rotation to zero. Keep
    # this expected value independent of the helper under test.
    expected = torch.cat(
        [target_position - current_position, torch.zeros_like(target_position)], dim=-1
    )

    assert torch.equal(actual, expected)

    generic = twist6_from_matrices(
        current_position,
        current_rotation,
        target_position,
        target_rotation,
    )
    assert not torch.equal(actual, generic)
