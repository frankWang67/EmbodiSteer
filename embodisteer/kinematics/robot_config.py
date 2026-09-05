"""Resolve ManiSkill robot UIDs to cuRobo configuration filenames."""

from __future__ import annotations

import os

from curobo.util_file import get_robot_configs_path


_ROBOT_CONFIG_NAMES = {
    "panda_robotiq_wristcam": "panda_robotiq_wristcam.yml",
    "ur5_robotiq_wristcam": "ur5_robotiq_wristcam.yml",
    "xarm6_robotiq_wristcam": "xarm6_robotiq_wristcam.yml",
    "xarm7_robotiq_wristcam": "xarm7_robotiq_wristcam.yml",
    "floating_robotiq_2f_85_gripper_wristcam": "floating_robotiq_wristcam.yml",
    "floating_robotiq_wristcam": "floating_robotiq_wristcam.yml",
}


def infer_robot_cfg_name(robot_uid: str | None) -> str:
    """Return a known alias or an installed UID-named YAML; default to Panda."""
    if robot_uid is None:
        return "panda_robotiq_wristcam.yml"
    if robot_uid in _ROBOT_CONFIG_NAMES:
        return _ROBOT_CONFIG_NAMES[robot_uid]
    candidate = f"{robot_uid}.yml"
    cfg_path = os.path.join(get_robot_configs_path(), candidate)
    if os.path.exists(cfg_path):
        return candidate
    raise ValueError(f"Cannot infer cuRobo robot config for robot_uid={robot_uid}")


__all__ = ["infer_robot_cfg_name"]
