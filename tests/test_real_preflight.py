"""Offline real preflight tests; all network/driver operations are mocked."""

from pathlib import Path
from unittest.mock import Mock, MagicMock

import pytest
import yaml

from embodisteer.real_config import load_obstacle_geometry, validate_robot_config, validate_real_policy
from scripts import preflight_real

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def robot_config():
    config = yaml.safe_load((ROOT / "configs/real/robot.template.yaml").read_text())
    config["robots"][0]["robot_ip"] = "robot.example.test"
    return config


def cli_args(tmp_path, config):
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(config))
    return ["--robot_config", str(path), "--obstacle_config",
            str(ROOT / "configs/real/obstacles/door_frame.yaml")]


def test_package_selection_matches_robot_gripper_camera(robot_config):
    packages = preflight_real.required_packages(robot_config, {"inference_space": "joint"})
    assert {"nvidia-curobo", "robotiq-gripper", "ur-rtde", "pyserial"} <= packages.keys()
    assert {"mani-skill", "sapien", "zerorpc", "pyrealsense2"}.isdisjoint(packages)
    robot_config["robots"][0]["robot_type"] = "franka"
    robot_config["grippers"][0] = {"gripper_type": "wsg50", "gripper_ip": "wsg.example.test"}
    packages = preflight_real.required_packages(robot_config, {"inference_space": "ee"}, True)
    assert {"zerorpc", "fastcrc", "pyrealsense2", "nvidia-curobo", "pytorch-kinematics"} <= packages.keys()
    assert {"ur-rtde", "robotiq-gripper", "pyserial", "pymodbus"}.isdisjoint(packages)


@pytest.mark.parametrize("input_device", [None, "keyboard", "spacemouse"])
def test_input_device_defaults_and_package_selection(robot_config, input_device):
    if input_device is None:
        robot_config.pop("input_device")
    else:
        robot_config["input_device"] = input_device
    validate_robot_config(robot_config)
    packages = preflight_real.required_packages(robot_config, {"inference_space": "joint"})
    assert ("spnav" in packages) == (input_device == "spacemouse")
    assert "pynput" in packages


@pytest.mark.parametrize("value", ["joystick", "", None, 1, [], {}])
def test_invalid_input_device_is_rejected(robot_config, value):
    robot_config["input_device"] = value
    with pytest.raises(ValueError, match="input_device"):
        validate_robot_config(robot_config)


def test_default_preflight_is_offline(tmp_path, monkeypatch, robot_config, capsys):
    monkeypatch.setattr(preflight_real, "check_packages", lambda packages: [])
    network, devices = Mock(side_effect=AssertionError), Mock(side_effect=AssertionError)
    monkeypatch.setattr(preflight_real, "check_network", network)
    monkeypatch.setattr(preflight_real, "check_devices", devices)
    assert preflight_real.main(cli_args(tmp_path, robot_config)) == 0
    network.assert_not_called()
    devices.assert_not_called()
    assert "not hardware readiness approval" in capsys.readouterr().out


def test_missing_package_fails_without_connecting(tmp_path, monkeypatch, robot_config):
    monkeypatch.setattr(preflight_real, "check_packages", lambda packages: [("FAIL", "robotiq", "missing")])
    network = Mock(side_effect=AssertionError)
    monkeypatch.setattr(preflight_real, "check_network", network)
    assert preflight_real.main(cli_args(tmp_path, robot_config) + ["--connect"]) == 1
    network.assert_not_called()


@pytest.mark.parametrize("policy_config", ["ee.yaml", "embodisteer.yaml"])
@pytest.mark.parametrize("missing_package", ["nvidia-curobo", "pytorch-kinematics"])
def test_shared_policy_dependencies_are_required_before_network_checks(
    tmp_path, monkeypatch, robot_config, capsys, policy_config, missing_package,
):
    def version(name):
        if name == missing_package:
            raise preflight_real.metadata.PackageNotFoundError(name)
        return {"pymodbus": "3.8.6", "pyserial": "3.5", "robotiq-gripper": "0.1.0"}.get(name, "1.0")

    monkeypatch.setattr(preflight_real.metadata, "version", version)
    monkeypatch.setattr(preflight_real.util, "find_spec", lambda name: object())
    network = Mock(side_effect=AssertionError("Must not connect with missing dependencies"))
    monkeypatch.setattr(preflight_real, "check_network", network)
    args = cli_args(tmp_path, robot_config) + [
        "--policy-config", str(ROOT / "configs/policy" / policy_config), "--connect",
    ]
    assert preflight_real.main(args) == 1
    assert f"FAIL {missing_package}" in capsys.readouterr().out
    network.assert_not_called()


def test_preflight_rejects_template_before_package_checks(tmp_path, monkeypatch, robot_config):
    robot_config["robots"][0]["robot_ip"] = "YOUR_ROBOT_IP"
    packages = Mock(side_effect=AssertionError)
    monkeypatch.setattr(preflight_real, "check_packages", packages)
    assert preflight_real.main(cli_args(tmp_path, robot_config)) == 1
    packages.assert_not_called()


def test_explicit_checks_are_called(tmp_path, monkeypatch, robot_config):
    monkeypatch.setattr(preflight_real, "check_packages", lambda packages: [])
    devices, network = Mock(return_value=[]), Mock(return_value=[])
    monkeypatch.setattr(preflight_real, "check_devices", devices)
    monkeypatch.setattr(preflight_real, "check_network", network)
    assert preflight_real.main(cli_args(tmp_path, robot_config) + ["--connect", "--check-devices"]) == 0
    devices.assert_called_once()
    network.assert_called_once_with(robot_config, 2.0)


def test_network_checks_use_correct_ports_and_send_no_commands(monkeypatch, robot_config):
    robot_config["robots"].append({"robot_type": "franka", "robot_ip": "franka.example.test"})
    robot_config["grippers"].append({"gripper_type": "wsg50", "gripper_ip": "wsg.example.test", "gripper_port": 1001})
    connection = MagicMock()
    connect = Mock(return_value=connection)
    monkeypatch.setattr(preflight_real.socket, "create_connection", connect)
    results = preflight_real.check_network(robot_config, 1.5)
    assert all(result[0] == "PASS" for result in results)
    assert [item.args[0] for item in connect.call_args_list] == [
        ("robot.example.test", 30004), ("franka.example.test", 4242), ("wsg.example.test", 1001)]
    connection.__enter__.return_value.send.assert_not_called()
    connection.__enter__.return_value.sendall.assert_not_called()


def test_package_discovery_reports_missing_and_wrong_versions(monkeypatch):
    monkeypatch.setattr(preflight_real.metadata, "version", lambda name: "0.0")
    monkeypatch.setattr(preflight_real.util, "find_spec", lambda name: object())
    assert preflight_real.check_packages({"pymodbus": "pymodbus"})[0][0] == "FAIL"
    monkeypatch.setattr(preflight_real.util, "find_spec", lambda name: None)
    assert preflight_real.check_packages({"torch": "torch"})[0][0] == "FAIL"


def test_device_check_does_not_accept_regular_files(tmp_path):
    fake = tmp_path / "serial"
    fake.touch()
    assert preflight_real.check_device_path(fake, "serial")[0] == "FAIL"


@pytest.mark.parametrize("field,value", [
    ("gripper_obs_latency", -0.1), ("gripper_action_latency", float("nan")),
    ("gripper_slave_id", 0), ("gripper_slave_id", True), ("gripper_slave_id", 248),
    ("gripper_serial_port", 123),
])
def test_invalid_gripper_values_are_rejected(robot_config, field, value):
    robot_config["grippers"][0][field] = value
    with pytest.raises(ValueError):
        validate_robot_config(robot_config)


@pytest.mark.parametrize("field,value", [
    ("center", [float("nan"), 0, 0]), ("half_extent", [0, 1, 1]),
    ("quat_wxyz", [0, 0, 0, 0]), ("quat_wxyz", [float("inf"), 0, 0, 0]),
    ("center", ["0", 0, 0]),
])
def test_invalid_obstacle_values_are_rejected(tmp_path, field, value):
    entry = {"center": [0, 0, 0], "half_extent": [1, 1, 1], "quat_wxyz": [1, 0, 0, 0]}
    entry[field] = value
    path = tmp_path / "obstacles.yaml"
    path.write_text(yaml.safe_dump({"obstacles": [entry]}))
    with pytest.raises(ValueError):
        load_obstacle_geometry(path)


def test_obstacle_quaternions_normalized_without_torch(tmp_path):
    path = tmp_path / "obstacles.yaml"
    path.write_text(yaml.safe_dump({"obstacles": [
        {"center": [0, 0, 0], "half_extent": [1, 1, 1], "quat_wxyz": [2, 0, 0, 0]}]}))
    assert load_obstacle_geometry(path)[0]["quat_wxyz"] == [1, 0, 0, 0]


def test_transform_rejects_reflections_and_nonfinite_values(robot_config):
    robot_config["tx_left_right"][0][0] = -1
    with pytest.raises(ValueError, match="determinant"):
        validate_robot_config(robot_config)
    robot_config["tx_left_right"][0][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_robot_config(robot_config)


def test_joint_wsg_combination_is_rejected(robot_config):
    robot_config["grippers"][0]["gripper_type"] = "wsg50"
    with pytest.raises(ValueError, match="Robotiq geometry"):
        validate_real_policy(robot_config, {"inference_space": "joint"})
    validate_real_policy(robot_config, {"inference_space": "ee"})
