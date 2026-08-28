"""Configuration helpers for real-world launchers.

These helpers perform no device I/O. They exist so a release can validate site
configuration and obstacle geometry before importing it into a policy call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml


PLACEHOLDER_ADDRESSES = {
    "CHANGE_ME",
    "YOUR_ROBOT_IP",
    "YOUR_GRIPPER_IP",
    "YOUR_FRANKA_HOST",
}


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    return value


def validate_robot_config(config: dict[str, Any], require_addresses: bool = True) -> None:
    robots = config.get("robots")
    grippers = config.get("grippers")
    if not isinstance(robots, list) or not robots:
        raise ValueError("robot config must contain a non-empty 'robots' list")
    if not isinstance(grippers, list) or not grippers:
        raise ValueError("robot config must contain a non-empty 'grippers' list")
    if len(robots) != len(grippers):
        raise ValueError("robot and gripper counts must match")

    for idx, robot in enumerate(robots):
        for field in ("robot_type", "robot_ip"):
            if not robot.get(field):
                raise ValueError(f"robots[{idx}] is missing {field}")
        if require_addresses and str(robot["robot_ip"]).upper() in PLACEHOLDER_ADDRESSES:
            raise ValueError(f"robots[{idx}].robot_ip is still a template placeholder")

    for idx, gripper in enumerate(grippers):
        if not gripper.get("gripper_ip"):
            raise ValueError(f"grippers[{idx}] is missing gripper_ip")
        if require_addresses and str(gripper["gripper_ip"]).upper() in PLACEHOLDER_ADDRESSES:
            raise ValueError(f"grippers[{idx}].gripper_ip is still a template placeholder")

    transform = config.get("tx_left_right")
    if not isinstance(transform, list) or len(transform) != 4:
        raise ValueError("tx_left_right must be a 4x4 matrix")
    if any(not isinstance(row, list) or len(row) != 4 for row in transform):
        raise ValueError("tx_left_right must be a 4x4 matrix")


def load_obstacle_config(path: str | Path | None) -> list[dict[str, torch.Tensor]]:
    if path is None:
        return []
    config = load_yaml_mapping(path)
    entries = config.get("obstacles", [])
    if not isinstance(entries, list):
        raise ValueError("obstacle config must contain an 'obstacles' list")

    obstacles: list[dict[str, torch.Tensor]] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"obstacles[{idx}] must be a mapping")
        center = torch.as_tensor(entry.get("center"), dtype=torch.float32)
        quat = torch.as_tensor(entry.get("quat_wxyz"), dtype=torch.float32)
        extent = torch.as_tensor(entry.get("half_extent"), dtype=torch.float32)
        if center.shape != (3,) or quat.shape != (4,) or extent.shape != (3,):
            raise ValueError(
                f"obstacles[{idx}] requires center[3], quat_wxyz[4], half_extent[3]"
            )
        if torch.any(extent <= 0):
            raise ValueError(f"obstacles[{idx}].half_extent must be positive")
        quat_norm = torch.linalg.vector_norm(quat)
        if not torch.isfinite(quat_norm) or float(quat_norm) < 1e-8:
            raise ValueError(f"obstacles[{idx}].quat_wxyz must be non-zero")
        quat = quat / quat_norm
        obstacles.append(
            {
                "center": center.unsqueeze(0),
                "quat": quat.unsqueeze(0),
                "extent": extent.unsqueeze(0),
            }
        )
    return obstacles


__all__ = [
    "load_obstacle_config",
    "load_yaml_mapping",
    "validate_robot_config",
]
