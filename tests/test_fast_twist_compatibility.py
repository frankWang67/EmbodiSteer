import torch

from embodisteer.kinematics import twist6_from_matrices
from embodisteer.kinematics.rotation import axis_angle_to_matrix
from embodisteer.policies.ee2joint import EmbodiSteerJointPolicy


def test_joint_policy_twist_uses_paper_fast_rotation_path():
    policy = object.__new__(EmbodiSteerJointPolicy)
    current_position = torch.zeros((1, 1, 3), dtype=torch.float32)
    target_position = torch.tensor(
        [[[0.01, -0.02, 0.03]]], dtype=torch.float32
    )
    current_rotation = torch.eye(3, dtype=torch.float32).reshape(1, 1, 3, 3)
    target_axis_angle = torch.tensor(
        [[1.0e-4, 0.0, 0.0]], dtype=torch.float32
    )
    target_rotation = axis_angle_to_matrix(target_axis_angle).reshape(1, 1, 3, 3)

    actual = policy._twist6_from_matrices(
        current_position,
        current_rotation,
        target_position,
        target_rotation,
    )
    expected_rotation = policy._matrix_to_axis_angle_fast(
        target_rotation.reshape(-1, 3, 3)
    ).reshape(1, 1, 3)
    expected = torch.cat(
        [target_position - current_position, expected_rotation], dim=-1
    )

    assert torch.equal(actual, expected)

    generic = twist6_from_matrices(
        current_position,
        current_rotation,
        target_position,
        target_rotation,
    )
    assert not torch.equal(actual, generic)
