"""Dependency-light validation for the real-world launcher.

PyYAML is its only third-party dependency, so offline preflight and schema
tests need not import runtime modules. The ``eval_real.py`` entry point,
including ``--dry_run``, assumes the real runtime profile is installed.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

PLACEHOLDER_ADDRESSES = {
    "CHANGE_ME", "YOUR_ROBOT_IP", "YOUR_GRIPPER_IP", "YOUR_FRANKA_HOST",
}
REAL_JOINT_ROBOT_CONFIGS = {
    "ur5": "ur5_robotiq_umi.yml",
    "franka": "panda_robotiq_umi.yml",
}
SUPPORTED_INPUT_DEVICES = {"keyboard", "spacemouse"}


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    return value


def _number(value: Any, location: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{location} must be a finite number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{location} must be >= {minimum}")
    return float(value)


def _vector(value: Any, size: int, location: str) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{location} must contain {size} numbers")
    return [_number(item, location) for item in value]


def _integer(value: Any, location: str, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{location} must be an integer in [{minimum}, {maximum}]")


def validate_robot_config(config: dict[str, Any], require_addresses: bool = True) -> None:
    input_device = config.get("input_device", "keyboard")
    if not isinstance(input_device, str) or input_device not in SUPPORTED_INPUT_DEVICES:
        raise ValueError(
            f"input_device must be 'keyboard' or 'spacemouse'; got {input_device!r}"
        )
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
            if not isinstance(robot, dict) or not robot.get(field):
                raise ValueError(f"robots[{idx}] is missing {field}")
        if require_addresses and str(robot["robot_ip"]).strip().upper() in PLACEHOLDER_ADDRESSES:
            raise ValueError(f"robots[{idx}].robot_ip is still a template placeholder")
        if robot["robot_type"] not in ("ur5", "ur5e", "franka"):
            raise ValueError(f"robots[{idx}].robot_type must be 'ur5', 'ur5e' or 'franka'")
        if not isinstance(robot["robot_ip"], str) or not robot["robot_ip"].strip():
            raise ValueError(f"robots[{idx}].robot_ip must be a non-empty string")
        for field in ("robot_obs_latency", "robot_action_latency"):
            _number(robot.get(field), f"robots[{idx}].{field}", 0)
        for field in ("tcp_offset", "height_threshold", "sphere_radius"):
            _number(robot.get(field), f"robots[{idx}].{field}", 0 if field == "sphere_radius" else None)
        _vector(robot.get("sphere_center"), 3, f"robots[{idx}].sphere_center")

    for idx, gripper in enumerate(grippers):
        if not isinstance(gripper, dict):
            raise ValueError(f"grippers[{idx}] must be a mapping")
        gripper_type = gripper.get("gripper_type", "robotiq")
        if gripper_type == "robotiq":
            address_field = "gripper_serial_port"
        elif gripper_type == "wsg50":
            address_field = "gripper_ip"
        else:
            raise ValueError(
                f"grippers[{idx}].gripper_type must be 'robotiq' or 'wsg50', "
                f"got {gripper_type!r}"
            )
        if not gripper.get(address_field):
            raise ValueError(f"grippers[{idx}] is missing {address_field}")
        if require_addresses and str(gripper[address_field]).strip().upper() in PLACEHOLDER_ADDRESSES:
            raise ValueError(f"grippers[{idx}].{address_field} is still a template placeholder")
        if not isinstance(gripper[address_field], str) or not gripper[address_field].strip():
            raise ValueError(f"grippers[{idx}].{address_field} must be a non-empty string")
        for field in ("gripper_obs_latency", "gripper_action_latency"):
            _number(gripper.get(field), f"grippers[{idx}].{field}", 0)
        if gripper_type == "robotiq":
            _integer(gripper.get("gripper_slave_id", 9), f"grippers[{idx}].gripper_slave_id", 1, 247)
        else:
            _integer(gripper.get("gripper_port", 1000), f"grippers[{idx}].gripper_port", 1, 65535)

    transform = config.get("tx_left_right")
    if not isinstance(transform, list) or len(transform) != 4:
        raise ValueError("tx_left_right must be a 4x4 matrix")
    if any(not isinstance(row, list) or len(row) != 4 for row in transform):
        raise ValueError("tx_left_right must be a 4x4 matrix")
    rows = [_vector(row, 4, "tx_left_right") for row in transform]
    if any(abs(a - b) > 1e-5 for a, b in zip(rows[3], (0, 0, 0, 1))):
        raise ValueError("tx_left_right must have homogeneous last row [0, 0, 0, 1]")
    rotation = [row[:3] for row in rows[:3]]
    for i in range(3):
        for j in range(3):
            dot = sum(rotation[i][k] * rotation[j][k] for k in range(3))
            if abs(dot - (1 if i == j else 0)) > 1e-3:
                raise ValueError("tx_left_right rotation must be orthonormal")
    a, b, c = rotation
    det = a[0] * (b[1]*c[2] - b[2]*c[1]) - a[1] * (b[0]*c[2] - b[2]*c[0]) + a[2] * (b[0]*c[1] - b[1]*c[0])
    if abs(det - 1) > 1e-3:
        raise ValueError("tx_left_right rotation must have determinant +1")


def load_obstacle_geometry(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    config = load_yaml_mapping(path)
    entries = config.get("obstacles", [])
    if not isinstance(entries, list):
        raise ValueError("obstacle config must contain an 'obstacles' list")
    result = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"obstacles[{idx}] must be a mapping")
        normalized = {
            field: _vector(entry.get(field), size, f"obstacles[{idx}].{field}")
            for field, size in (("center", 3), ("quat_wxyz", 4), ("half_extent", 3))
        }
        if any(value <= 0 for value in normalized["half_extent"]):
            raise ValueError(f"obstacles[{idx}].half_extent must be positive")
        norm = math.hypot(*normalized["quat_wxyz"])
        if not math.isfinite(norm) or norm < 1e-8:
            raise ValueError(f"obstacles[{idx}].quat_wxyz must be non-zero")
        normalized["quat_wxyz"] = [value / norm for value in normalized["quat_wxyz"]]
        result.append(normalized)
    return result


def validate_real_policy(config: dict[str, Any], policy_settings: dict[str, Any]) -> None:
    """Check policy choices that affect the real launcher without importing torch."""
    if policy_settings.get("baseline_method"):
        raise ValueError(
            "Real-world evaluation does not support baseline policies; use a "
            "config with baseline.method='' and select the desired inference space."
        )
    if policy_settings.get("inference_space") == "joint":
        if len(config["robots"]) != 1:
            raise ValueError("Joint-space real-world evaluation currently supports a single robot only.")
        if config["robots"][0]["robot_type"] not in REAL_JOINT_ROBOT_CONFIGS:
            raise ValueError("No joint-space real-world model for this robot_type")
        if config["grippers"][0].get("gripper_type", "robotiq") != "robotiq":
            raise ValueError(
                "Joint-space real-world inference currently requires Robotiq geometry; "
                "WSG50 needs matching kinematic/collision assets and finger conversion."
            )
