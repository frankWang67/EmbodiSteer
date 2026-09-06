"""Controller wiring tests: every hardware constructor is replaced by a mock."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from umi.real_world.gripper_controller import create_gripper_controller


@pytest.fixture
def controllers(monkeypatch):
    import umi.real_world.robotiq_controller as robotiq_module
    import umi.real_world.wsg_controller as wsg_module

    robotiq, wsg = Mock(), Mock()
    monkeypatch.setattr(robotiq_module, "RobotiqController", robotiq)
    monkeypatch.setattr(wsg_module, "WSGController", wsg)
    return robotiq, wsg


def test_factory_defaults_to_robotiq(controllers):
    robotiq, wsg = controllers
    shm = object()
    result = create_gripper_controller(shm, {
        "gripper_serial_port": "/dev/ttyUSB1",
        "gripper_obs_latency": 0.02,
    })
    assert result is robotiq.return_value
    robotiq.assert_called_once_with(
        shm_manager=shm, port="/dev/ttyUSB1", slave_id=9, receive_latency=0.02,
    )
    wsg.assert_not_called()


def test_factory_defaults_wsg_tcp_port(controllers):
    robotiq, wsg = controllers
    shm = object()
    result = create_gripper_controller(shm, {
        "gripper_type": "wsg50",
        "gripper_ip": "wsg.example.test",
        "gripper_obs_latency": 0.03,
    })
    assert result is wsg.return_value
    wsg.assert_called_once_with(
        shm_manager=shm, hostname="wsg.example.test", port=1000,
        receive_latency=0.03, use_meters=True,
    )
    robotiq.assert_not_called()


def test_factory_rejects_unknown_type_without_constructing_hardware(controllers):
    with pytest.raises(ValueError, match="Unsupported gripper_type"):
        create_gripper_controller(object(), {"gripper_type": "unknown"})
    for controller in controllers:
        controller.assert_not_called()


def _mock_environment_devices(monkeypatch, module):
    """No camera enumeration/reset, device creation or shared-memory process."""
    for name in (
        "reset_all_elgato_devices", "MultiUvcCamera", "VideoRecorder",
        "MultiCameraVisualizer", "ReplayBuffer", "SharedMemoryManager",
    ):
        monkeypatch.setattr(module, name, Mock())
    for module_name, class_name in (
        ("rtde_interpolation_controller", "RTDEInterpolationController"),
        ("franka_interpolation_controller", "FrankaInterpolationController"),
    ):
        monkeypatch.setitem(sys.modules, f"umi.real_world.{module_name}",
                            SimpleNamespace(**{class_name: Mock()}))
    monkeypatch.setattr(
        module, "get_sorted_v4l_paths", Mock(return_value=["/dev/video-test"])
    )


@pytest.mark.parametrize("gripper_type", ["robotiq", "wsg50"])
def test_single_arm_environment_routes_connection_parameters(
    tmp_path, monkeypatch, controllers, gripper_type,
):
    from umi.real_world import umi_env

    _mock_environment_devices(monkeypatch, umi_env)
    robotiq, wsg = controllers
    shm = object()
    connection = (
        {"gripper_serial_port": "/dev/ttyUSB2", "gripper_slave_id": 8}
        if gripper_type == "robotiq"
        else {"gripper_ip": "wsg.example.test", "gripper_port": 1002}
    )
    env = umi_env.UmiEnv(
        output_dir=tmp_path / "episodes", robot_ip="robot.example.test",
        gripper_type=gripper_type, gripper_obs_latency=0.04,
        gripper_action_latency=0.15, shm_manager=shm,
        enable_multi_cam_vis=False, **connection,
    )
    if gripper_type == "robotiq":
        robotiq.assert_called_once_with(
            shm_manager=shm, port="/dev/ttyUSB2", slave_id=8, receive_latency=0.04,
        )
        wsg.assert_not_called()
        assert env.gripper is robotiq.return_value
    else:
        wsg.assert_called_once_with(
            shm_manager=shm, hostname="wsg.example.test", port=1002,
            receive_latency=0.04, use_meters=True,
        )
        robotiq.assert_not_called()
        assert env.gripper is wsg.return_value
    assert env.gripper_action_latency == 0.15


def test_bimanual_environment_preserves_per_gripper_connections(
    tmp_path, monkeypatch, controllers,
):
    from umi.real_world import bimanual_umi_env

    _mock_environment_devices(monkeypatch, bimanual_umi_env)
    robotiq, wsg = controllers
    robotiq_instances = [object(), object()]
    robotiq.side_effect = robotiq_instances
    shm = object()
    configs = [
        {"gripper_serial_port": "/dev/ttyUSB0", "gripper_obs_latency": 0.01,
         "gripper_action_latency": 0.1},
        {"gripper_type": "robotiq", "gripper_serial_port": "/dev/ttyUSB1",
         "gripper_slave_id": 8, "gripper_obs_latency": 0.02,
         "gripper_action_latency": 0.2},
        {"gripper_type": "wsg50", "gripper_ip": "wsg.example.test",
         "gripper_port": 1003, "gripper_obs_latency": 0.03,
         "gripper_action_latency": 0.3},
    ]
    env = bimanual_umi_env.BimanualUmiEnv(
        output_dir=tmp_path / "episodes",
        robots_config=[
            {"robot_type": "ur5", "robot_ip": f"robot-{idx}.example.test",
             "tcp_offset": 0.235, "robot_obs_latency": 0.0001}
            for idx in range(len(configs))
        ],
        grippers_config=configs, shm_manager=shm, enable_multi_cam_vis=False,
    )
    assert env.grippers == robotiq_instances + [wsg.return_value]
    assert env.grippers_config == configs
    assert robotiq.call_args_list == [
        call(shm_manager=shm, port="/dev/ttyUSB0", slave_id=9, receive_latency=0.01),
        call(shm_manager=shm, port="/dev/ttyUSB1", slave_id=8, receive_latency=0.02),
    ]
    wsg.assert_called_once_with(
        shm_manager=shm, hostname="wsg.example.test", port=1003,
        receive_latency=0.03, use_meters=True,
    )


def test_robotiq_controller_passes_serial_settings_to_mock_driver(monkeypatch):
    from umi.real_world import robotiq_controller

    # Replace the external driver before construction: activate() is a mock.
    driver = Mock()
    monkeypatch.setitem(sys.modules, "robotiq_gripper", SimpleNamespace(
        RobotiqModBusGripper=driver,
    ))
    monkeypatch.setattr(robotiq_controller, "SharedMemoryQueue", Mock())
    monkeypatch.setattr(robotiq_controller, "SharedMemoryRingBuffer", Mock())
    monkeypatch.setattr(robotiq_controller.mp, "Event", Mock())
    controller = robotiq_controller.RobotiqController(
        shm_manager=object(), port="/dev/ttyUSB3", slave_id=6,
        receive_latency=0.04,
    )
    driver.assert_called_once_with(width=0.085, port="/dev/ttyUSB3", slave_id=6)
    driver.return_value.activate.assert_called_once_with()
    assert controller.receive_latency == 0.04
