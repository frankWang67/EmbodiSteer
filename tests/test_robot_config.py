import pytest

from embodisteer.kinematics import robot_config


@pytest.mark.parametrize(
    ("robot_uid", "expected"),
    [
        (None, "panda_robotiq_wristcam.yml"),
        ("panda_robotiq_wristcam", "panda_robotiq_wristcam.yml"),
        ("ur5_robotiq_wristcam", "ur5_robotiq_wristcam.yml"),
        ("xarm6_robotiq_wristcam", "xarm6_robotiq_wristcam.yml"),
        ("xarm7_robotiq_wristcam", "xarm7_robotiq_wristcam.yml"),
        ("floating_robotiq_2f_85_gripper_wristcam", "floating_robotiq_wristcam.yml"),
        ("floating_robotiq_wristcam", "floating_robotiq_wristcam.yml"),
    ],
)
def test_known_robot_configs_resolve_without_installation(monkeypatch, robot_uid, expected):
    def unexpected_lookup():
        raise AssertionError("Known robot aliases should not query the filesystem")

    monkeypatch.setattr(robot_config, "get_robot_configs_path", unexpected_lookup)
    assert robot_config.infer_robot_cfg_name(robot_uid) == expected


def test_custom_robot_config_resolves_from_curobo_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(robot_config, "get_robot_configs_path", lambda: str(tmp_path))
    (tmp_path / "custom_arm.yml").touch()
    assert robot_config.infer_robot_cfg_name("custom_arm") == "custom_arm.yml"


def test_missing_robot_config_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(robot_config, "get_robot_configs_path", lambda: str(tmp_path))
    with pytest.raises(ValueError, match="Cannot infer cuRobo robot config.*missing_arm"):
        robot_config.infer_robot_cfg_name("missing_arm")
