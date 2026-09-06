from pathlib import Path

import pytest

from embodisteer.adapters.real import load_obstacle_config, load_yaml_mapping, validate_robot_config
from umi.real_world import gripper_controller


ROOT = Path(__file__).resolve().parents[1]


def test_robot_template_is_explicitly_incomplete():
    config = load_yaml_mapping(ROOT / "configs" / "real" / "robot.template.yaml")
    validate_robot_config(config, require_addresses=False)
    assert config["input_device"] == "keyboard"
    assert config["grippers"] == [{
        "gripper_type": "robotiq",
        "gripper_serial_port": "/dev/ttyUSB0",
        "gripper_slave_id": 9,
        "gripper_obs_latency": 0.01,
        "gripper_action_latency": 0.1,
    }]
    with pytest.raises(ValueError, match="robot_ip is still a template placeholder"):
        validate_robot_config(config)


@pytest.fixture
def robot_config():
    config = load_yaml_mapping(ROOT / "configs" / "real" / "robot.template.yaml")
    config["robots"][0]["robot_ip"] = "robot.example.test"
    return config


@pytest.mark.parametrize("explicit_type", [True, False])
def test_robotiq_config_accepts_serial_without_ip(robot_config, explicit_type):
    if not explicit_type:
        robot_config["grippers"][0].pop("gripper_type")
    validate_robot_config(robot_config)


def test_robotiq_config_requires_serial_device(robot_config):
    robot_config["grippers"][0].pop("gripper_serial_port")
    robot_config["grippers"][0]["gripper_ip"] = "gripper.example.test"
    with pytest.raises(ValueError, match="missing gripper_serial_port"):
        validate_robot_config(robot_config)


def test_wsg50_config_accepts_tcp_without_serial_device(robot_config):
    robot_config["grippers"][0] = {
        "gripper_type": "wsg50",
        "gripper_ip": "gripper.example.test",
        "gripper_port": 1000,
        "gripper_obs_latency": 0.01,
        "gripper_action_latency": 0.1,
    }
    validate_robot_config(robot_config)


def test_wsg50_config_requires_ip(robot_config):
    robot_config["grippers"][0]["gripper_type"] = "wsg50"
    with pytest.raises(ValueError, match="missing gripper_ip"):
        validate_robot_config(robot_config)


@pytest.mark.parametrize("gripper_type, field, placeholder", [
    ("robotiq", "gripper_serial_port", "CHANGE_ME"),
    ("wsg50", "gripper_ip", "YOUR_GRIPPER_IP"),
])
def test_gripper_address_placeholders(robot_config, gripper_type, field, placeholder):
    robot_config["grippers"][0] = {
        "gripper_type": gripper_type,
        field: placeholder,
        "gripper_obs_latency": 0.01,
        "gripper_action_latency": 0.1,
    }
    validate_robot_config(robot_config, require_addresses=False)
    with pytest.raises(ValueError, match=f"{field} is still a template placeholder"):
        validate_robot_config(robot_config)


@pytest.mark.parametrize("gripper_type", ["unknown", "", None])
def test_unknown_gripper_type_is_rejected(robot_config, gripper_type):
    robot_config["grippers"][0]["gripper_type"] = gripper_type
    with pytest.raises(ValueError, match="gripper_type must be 'robotiq' or 'wsg50'"):
        validate_robot_config(robot_config)


def test_controller_factory_forwards_robotiq_parameters(monkeypatch):
    calls = {}

    class FakeRobotiq:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    import umi.real_world.robotiq_controller as module
    monkeypatch.setattr(module, "RobotiqController", FakeRobotiq)
    shm = object()
    result = gripper_controller.create_gripper_controller(shm, {
        "gripper_type": "robotiq",
        "gripper_serial_port": "/dev/serial/by-id/gripper-left",
        "gripper_slave_id": 7,
        "gripper_obs_latency": 0.025,
    })
    assert isinstance(result, FakeRobotiq)
    assert calls == {
        "shm_manager": shm,
        "port": "/dev/serial/by-id/gripper-left",
        "slave_id": 7,
        "receive_latency": 0.025,
    }


def test_controller_factory_forwards_wsg50_parameters(monkeypatch):
    calls = {}

    class FakeWSG:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    import umi.real_world.wsg_controller as module
    monkeypatch.setattr(module, "WSGController", FakeWSG)
    shm = object()
    result = gripper_controller.create_gripper_controller(shm, {
        "gripper_type": "wsg50",
        "gripper_ip": "wsg.example.test",
        "gripper_port": 1001,
        "gripper_obs_latency": 0.03,
    })
    assert isinstance(result, FakeWSG)
    assert calls == {
        "shm_manager": shm,
        "hostname": "wsg.example.test",
        "port": 1001,
        "receive_latency": 0.03,
        "use_meters": True,
    }


def test_obstacle_layout_is_loadable_and_normalized():
    obstacles = load_obstacle_config(
        ROOT / "configs" / "real" / "obstacles" / "door_frame.yaml"
    )
    assert len(obstacles) == 4
    assert all(tuple(item["quat"].shape) == (1, 4) for item in obstacles)


def test_robotiq_dependencies_are_pinned_in_real_environment_specs():
    environment = load_yaml_mapping(ROOT / "environment" / "environment-real.yaml")
    conda_pip = next(
        entry["pip"] for entry in environment["dependencies"]
        if isinstance(entry, dict) and "pip" in entry
    )
    requirements = (ROOT / "environment" / "real-requirements.txt").read_text().splitlines()
    expected = {
        "robotiq_gripper @ https://github.com/frankWang67/robotiq_modbus_gripper/"
        "archive/582a6c26a2462adb58c60ba229ad9578556cf464.tar.gz",
        "pymodbus==3.8.6",
        "pyserial==3.5",
    }
    assert "-r real-requirements.txt" in set(conda_pip)
    assert expected <= set(requirements)
