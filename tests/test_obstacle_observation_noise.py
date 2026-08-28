import os
import sys

import gymnasium as gym
import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

from scripts_maniskill.obstacle_observation_noise import (
    ObstacleObservationNoiseWrapper,
)


class DummyObstacleEnv(gym.Env):
    metadata = {}

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.obstacles = [
            {
                "center": torch.tensor(
                    [[0.3, -0.2, 0.1], [0.4, 0.2, 0.15]], dtype=torch.float32
                ),
                "quat": torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0], [0.9238795, 0.0, 0.0, 0.3826834]],
                    dtype=torch.float32,
                ),
                "extent": torch.tensor(
                    [[0.05, 0.04, 0.10], [0.03, 0.08, 0.12]], dtype=torch.float32
                ),
            }
        ]

    def reset(self, *, seed=None, options=None):
        gym_seed = seed[0] if isinstance(seed, (list, tuple)) else seed
        super().reset(seed=gym_seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(1, dtype=np.float32), 0.0, False, False, {}

    def get_obstacles_info(self):
        return [
            {key: value.clone() for key, value in obstacle.items()}
            for obstacle in self.obstacles
        ]


def assert_obstacles_equal(lhs, rhs):
    assert len(lhs) == len(rhs)
    for lhs_obstacle, rhs_obstacle in zip(lhs, rhs):
        for key in ("center", "quat", "extent"):
            torch.testing.assert_close(lhs_obstacle[key], rhs_obstacle[key])


def test_noise_is_fixed_within_episode_and_reproducible_from_seed():
    wrapped = ObstacleObservationNoiseWrapper(
        DummyObstacleEnv(),
        position_std_m=0.01,
        size_std_m=0.02,
        rotation_std_rad=0.05,
    )

    wrapped.reset(seed=123)
    first = wrapped.get_obstacles_info()
    second = wrapped.get_obstacles_info()
    assert_obstacles_equal(first, second)

    wrapped.reset(seed=123)
    repeated = wrapped.get_obstacles_info()
    assert_obstacles_equal(first, repeated)

    wrapped.reset(seed=124)
    different_seed = wrapped.get_obstacles_info()
    assert not torch.equal(first[0]["center"], different_seed[0]["center"])


def test_reset_without_seed_draws_new_episode_noise():
    wrapped = ObstacleObservationNoiseWrapper(
        DummyObstacleEnv(),
        position_std_m=0.01,
        size_std_m=0.0,
        rotation_std_rad=0.0,
    )

    wrapped.reset(seed=123)
    first = wrapped.get_obstacles_info()
    wrapped.reset()
    second = wrapped.get_obstacles_info()
    assert not torch.equal(first[0]["center"], second[0]["center"])


def test_noise_preserves_true_geometry_and_tensor_invariants():
    env = DummyObstacleEnv()
    original = env.get_obstacles_info()
    wrapped = ObstacleObservationNoiseWrapper(
        env,
        position_std_m=0.5,
        size_std_m=1.0,
        rotation_std_rad=0.2,
        min_edge_length_m=1e-3,
    )

    wrapped.reset(seed=[2022, 2023])
    noisy = wrapped.get_obstacles_info()
    assert_obstacles_equal(original, env.get_obstacles_info())

    for key in ("center", "quat", "extent"):
        assert noisy[0][key].shape == original[0][key].shape
        assert noisy[0][key].dtype == original[0][key].dtype
        assert noisy[0][key].device == original[0][key].device
    assert torch.all(noisy[0]["extent"] >= 5e-4)
    torch.testing.assert_close(
        torch.linalg.vector_norm(noisy[0]["quat"], dim=-1),
        torch.ones(2),
    )


def test_zero_std_fields_remain_exact():
    env = DummyObstacleEnv()
    original = env.get_obstacles_info()
    wrapped = ObstacleObservationNoiseWrapper(
        env,
        position_std_m=0.01,
        size_std_m=0.0,
        rotation_std_rad=0.0,
    )

    wrapped.reset(seed=123)
    noisy = wrapped.get_obstacles_info()
    assert not torch.equal(noisy[0]["center"], original[0]["center"])
    torch.testing.assert_close(noisy[0]["extent"], original[0]["extent"])
    torch.testing.assert_close(noisy[0]["quat"], original[0]["quat"])


def test_negative_std_is_rejected():
    try:
        ObstacleObservationNoiseWrapper(
            DummyObstacleEnv(),
            position_std_m=-0.01,
            size_std_m=0.0,
            rotation_std_rad=0.0,
        )
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("Negative obstacle noise std should be rejected.")


if __name__ == "__main__":
    test_noise_is_fixed_within_episode_and_reproducible_from_seed()
    test_reset_without_seed_draws_new_episode_noise()
    test_noise_preserves_true_geometry_and_tensor_invariants()
    test_zero_std_fields_remain_exact()
    test_negative_std_is_rejected()
    print("Obstacle observation noise tests passed.")
