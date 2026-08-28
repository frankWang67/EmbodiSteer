"""Observation-only noise for ManiSkill cuboid obstacle geometry."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np
import torch


class ObstacleObservationNoiseWrapper(gym.Wrapper):
    """Perturb ``get_obstacles_info`` without changing simulated obstacles.

    A noise realization is sampled on the first obstacle query after every reset
    and is reused for the rest of that episode. Positions and full edge lengths
    receive additive Gaussian noise. Orientations receive a Gaussian angle error
    around a uniformly sampled axis.

    The wrapper expects the ManiSkill obstacle representation used by the
    evaluation code: a list of dictionaries containing ``center`` (``[..., 3]``),
    ``quat`` (``[..., 4]`` in wxyz order), and ``extent`` (``[..., 3]`` half
    edge lengths).
    """

    _SEED_NAMESPACE = 0x4F42534E  # "OBSN"

    def __init__(
        self,
        env: gym.Env,
        position_std_m: float,
        size_std_m: float,
        rotation_std_rad: float,
        min_edge_length_m: float = 1e-4,
    ):
        super().__init__(env)
        self.position_std_m = float(position_std_m)
        self.size_std_m = float(size_std_m)
        self.rotation_std_rad = float(rotation_std_rad)
        self.min_edge_length_m = float(min_edge_length_m)

        stds = (
            self.position_std_m,
            self.size_std_m,
            self.rotation_std_rad,
        )
        if any(std < 0.0 for std in stds):
            raise ValueError(f"Obstacle observation noise stds must be non-negative, got {stds}.")
        if self.min_edge_length_m <= 0.0:
            raise ValueError("min_edge_length_m must be positive.")

        self._rng: np.random.Generator | None = None
        self._noise_cache: list[dict[str, torch.Tensor]] | None = None

    @staticmethod
    def _seed_values(seed: Any) -> list[int]:
        if isinstance(seed, torch.Tensor):
            seed = seed.detach().cpu().reshape(-1).tolist()
        elif isinstance(seed, np.ndarray):
            seed = seed.reshape(-1).tolist()

        if isinstance(seed, Sequence) and not isinstance(seed, (str, bytes)):
            values = seed
        else:
            values = [seed]
        return [int(value) & 0xFFFFFFFF for value in values]

    def _reset_rng(self, seed: Any) -> None:
        entropy = [self._SEED_NAMESPACE, *self._seed_values(seed)]
        self._rng = np.random.default_rng(np.random.SeedSequence(entropy))

    def reset(self, *, seed=None, options=None):
        result = self.env.reset(seed=seed, options=options)
        if seed is not None:
            self._reset_rng(seed)
        elif self._rng is None:
            self._reset_rng(0)
        self._noise_cache = None
        return result

    def _normal_like(self, reference: torch.Tensor, std: float) -> torch.Tensor:
        if std == 0.0:
            return torch.zeros_like(reference)
        if self._rng is None:
            self._reset_rng(0)
        values = self._rng.normal(loc=0.0, scale=std, size=tuple(reference.shape))
        return torch.as_tensor(values, dtype=reference.dtype, device=reference.device)

    def _orientation_delta_like(self, quat: torch.Tensor) -> torch.Tensor:
        batch_shape = tuple(quat.shape[:-1])
        if self.rotation_std_rad == 0.0:
            identity = torch.zeros_like(quat)
            identity[..., 0] = 1.0
            return identity
        if self._rng is None:
            self._reset_rng(0)

        axes = self._rng.normal(size=(*batch_shape, 3))
        axis_norm = np.linalg.norm(axes, axis=-1, keepdims=True)
        fallback = np.zeros_like(axes)
        fallback[..., 0] = 1.0
        axes = np.divide(axes, axis_norm, out=fallback, where=axis_norm > 1e-12)
        angles = self._rng.normal(
            loc=0.0,
            scale=self.rotation_std_rad,
            size=(*batch_shape, 1),
        )
        half_angles = 0.5 * angles
        delta_quat = np.concatenate(
            [np.cos(half_angles), axes * np.sin(half_angles)],
            axis=-1,
        )
        return torch.as_tensor(delta_quat, dtype=quat.dtype, device=quat.device)

    @staticmethod
    def _quaternion_multiply_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        aw, ax, ay, az = a.unbind(dim=-1)
        bw, bx, by, bz = b.unbind(dim=-1)
        return torch.stack(
            (
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
            ),
            dim=-1,
        )

    @staticmethod
    def _validate_obstacle(obstacle: dict[str, Any]) -> None:
        for key, last_dim in (("center", 3), ("quat", 4), ("extent", 3)):
            value = obstacle.get(key)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Obstacle field {key!r} must be a torch.Tensor.")
            if value.ndim < 1 or value.shape[-1] != last_dim:
                raise ValueError(
                    f"Obstacle field {key!r} must have shape [..., {last_dim}], "
                    f"got {tuple(value.shape)}."
                )

    def _cache_matches(self, obstacles: list[dict[str, Any]]) -> bool:
        if self._noise_cache is None or len(self._noise_cache) != len(obstacles):
            return False
        for obstacle, noise in zip(obstacles, self._noise_cache):
            for key in ("center", "quat", "extent"):
                value = obstacle[key]
                cached = noise[key]
                if (
                    value.shape != cached.shape
                    or value.dtype != cached.dtype
                    or value.device != cached.device
                ):
                    return False
        return True

    def _sample_noise(self, obstacles: list[dict[str, Any]]) -> None:
        self._noise_cache = []
        for obstacle in obstacles:
            self._validate_obstacle(obstacle)
            self._noise_cache.append(
                {
                    "center": self._normal_like(obstacle["center"], self.position_std_m),
                    # size_std_m is defined on the full edge length, while ManiSkill
                    # exposes half edge lengths in the extent field.
                    "extent": self._normal_like(obstacle["extent"], self.size_std_m),
                    "quat": self._orientation_delta_like(obstacle["quat"]),
                }
            )

    def _get_true_obstacles_info(self):
        try:
            getter = self.env.get_wrapper_attr("get_obstacles_info")
        except AttributeError:
            getter = getattr(self.env.unwrapped, "get_obstacles_info")
        return getter()

    def get_obstacles_info(self):
        obstacles = self._get_true_obstacles_info()
        if not isinstance(obstacles, (list, tuple)):
            raise TypeError(
                "get_obstacles_info() must return a list or tuple of obstacle dictionaries."
            )
        obstacles = list(obstacles)
        if len(obstacles) == 0:
            self._noise_cache = None
            return []

        for obstacle in obstacles:
            if not isinstance(obstacle, dict):
                raise TypeError(
                    "Each obstacle returned by get_obstacles_info() must be a dictionary."
                )
            self._validate_obstacle(obstacle)
        if not self._cache_matches(obstacles):
            self._sample_noise(obstacles)

        noisy_obstacles = []
        for obstacle, noise in zip(obstacles, self._noise_cache):
            noisy = dict(obstacle)
            noisy["center"] = obstacle["center"] + noise["center"]

            full_edge_length = 2.0 * obstacle["extent"] + noise["extent"]
            full_edge_length = torch.clamp(
                full_edge_length,
                min=self.min_edge_length_m,
            )
            noisy["extent"] = 0.5 * full_edge_length
            # noisy["extent"] += 0.01

            if self.rotation_std_rad > 0.0:
                quat = self._quaternion_multiply_wxyz(noise["quat"], obstacle["quat"])
                quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1e-12)
                noisy["quat"] = torch.where(quat[..., :1] < 0.0, -quat, quat)
            noisy_obstacles.append(noisy)
        return noisy_obstacles
