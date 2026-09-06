"""Real-runtime CLI checks; never construct hardware or load a checkpoint."""

from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

from click.testing import CliRunner
import numpy as np
import pytest
import yaml

import eval_real

ROOT = Path(__file__).resolve().parents[1]
ROBOT_TEMPLATE = ROOT / "configs/real/robot.template.yaml"


def base_args(robot_config=ROBOT_TEMPLATE):
    return [
        "--robot_config", str(robot_config),
        "--obstacle_config", str(ROOT / "configs/real/obstacles/door_frame.yaml"),
        "--policy-config", str(ROOT / "configs/policy/embodisteer.yaml"),
    ]


@pytest.fixture(autouse=True)
def forbid_runtime_startup(monkeypatch):
    guards = []
    for module, names in [
        (eval_real, ["BimanualUmiEnv", "Spacemouse", "Keyboard", "KeystrokeCounter",
                     "SharedMemoryManager", "load_obstacle_config"]),
        (eval_real.torch, ["load"]),
        (eval_real.hydra.utils, ["get_class"]),
        (eval_real.av, ["open"]),
    ]:
        for name in names:
            guard = Mock(side_effect=AssertionError(f"Unexpected runtime call: {name}"))
            monkeypatch.setattr(module, name, guard)
            guards.append(guard)
    yield
    for guard in guards:
        guard.assert_not_called()


def test_help_describes_installed_runtime_requirement():
    result = CliRunner().invoke(eval_real.main, ["--help"])
    assert result.exit_code == 0, result.output
    assert "real runtime" in result.output
    assert "no runtime dependencies" not in result.output


@pytest.mark.parametrize("flag", ["--dry_run", "--dry-run"])
@pytest.mark.parametrize("with_checkpoint_args", [False, True])
@pytest.mark.parametrize("input_device", [None, "keyboard", "spacemouse"])
def test_dry_run_exits_before_checkpoint_and_device_startup(
    flag, with_checkpoint_args, input_device, tmp_path,
):
    config = yaml.safe_load(ROBOT_TEMPLATE.read_text())
    if input_device is None:
        config.pop("input_device")
    else:
        config["input_device"] = input_device
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(config))
    args = base_args(path) + [flag]
    output = tmp_path / "recording"
    if with_checkpoint_args:
        args += ["--input", str(tmp_path / "missing.ckpt"), "--output", str(output)]
    result = CliRunner().invoke(eval_real.main, args)
    assert result.exit_code == 0, result.output
    assert "Configuration only (template addresses allowed)" in result.output
    assert "Runtime modules were imported" in result.output
    assert f"input_device={input_device or 'keyboard'}" in result.output
    assert "No device connection was attempted" in result.output
    assert not output.exists()


def test_dry_run_rejects_invalid_configuration(tmp_path):
    config = yaml.safe_load(ROBOT_TEMPLATE.read_text())
    config["grippers"][0]["gripper_obs_latency"] = -1
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(config))
    result = CliRunner().invoke(eval_real.main, base_args(path) + ["--dry_run"])
    assert result.exit_code != 0
    assert "gripper_obs_latency" in result.output
    assert "Real-world configuration OK" not in result.output


def test_physical_run_rejects_template_addresses():
    result = CliRunner().invoke(eval_real.main, base_args())
    assert result.exit_code != 0
    assert "placeholder" in result.output


def test_physical_run_still_requires_checkpoint_and_output(tmp_path):
    config = yaml.safe_load(ROBOT_TEMPLATE.read_text())
    config["robots"][0]["robot_ip"] = "robot.example.test"
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(config))
    result = CliRunner().invoke(eval_real.main, base_args(path))
    assert result.exit_code == 2
    assert "--input and --output are required" in result.output


@pytest.mark.parametrize("spnav_error", ["ModuleNotFoundError", "OSError"])
def test_keyboard_dry_run_needs_no_simulation_or_spacemouse_dependencies(spnav_error):
    program = """
import importlib.abc
import runpy
import sys

class BlockOptionalDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mani_skill', 'sapien'}:
            raise AssertionError('Unexpected simulation import: ' + fullname)
        if fullname == 'spnav':
            raise SPNAV_ERROR('SpaceMouse dependency unavailable')

sys.meta_path.insert(0, BlockOptionalDependencies())
sys.argv = ['eval_real.py', *sys.argv[1:]]
runpy.run_path('eval_real.py', run_name='__main__')
""".replace("SPNAV_ERROR", spnav_error)
    result = subprocess.run(
        [sys.executable, "-c", program, *base_args(), "--dry_run"],
        cwd=ROOT, text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "No device connection was attempted" in result.stdout


@pytest.mark.parametrize("input_device", ["keyboard", "spacemouse"])
def test_input_device_factory_selects_only_configured_backend(monkeypatch, input_device):
    keyboard, spacemouse = Mock(), Mock()
    monkeypatch.setattr(eval_real, "Keyboard", keyboard)
    monkeypatch.setattr(eval_real, "Spacemouse", spacemouse)
    shm = object()
    result = eval_real.create_input_device(input_device, shm)
    selected, unused = (keyboard, spacemouse) if input_device == "keyboard" else (spacemouse, keyboard)
    assert result is selected.return_value
    selected.assert_called_once_with(shm_manager=shm)
    unused.assert_not_called()


def test_input_device_factory_rejects_unknown_type():
    with pytest.raises(ValueError, match="input_device"):
        eval_real.create_input_device("joystick", object())


@pytest.mark.parametrize("input_device", ["keyboard", "spacemouse"])
def test_teleoperation_instructions_are_printed(capsys, input_device):
    eval_real.print_teleop_instructions(input_device, num_robots=2)
    output = capsys.readouterr().out
    assert "Human teleoperation controls" in output
    assert f"Input device: {input_device}" in output
    assert "C = start policy execution" in output
    assert "Q = quit" in output
    assert "0 = control all robots" in output
    if input_device == "keyboard":
        assert "W/S = X +/-" in output
        assert "Z = close; X = open" in output
    else:
        assert "six SpaceMouse axes" in output
        assert "button 0 = close" in output
        assert "W/S" not in output
    assert "WARNING: A" not in output
    assert "A = control all robots" not in output
    assert "delete the last recorded episode and its videos" in output
    assert "not an emergency stop" in output


@pytest.mark.parametrize("input_device", ["keyboard", "spacemouse"])
@pytest.mark.parametrize("has_match_dataset", [False, True])
@pytest.mark.parametrize("has_match_episode", [False, True])
def test_teleoperation_instructions_match_available_options(
    capsys, input_device, has_match_dataset, has_match_episode,
):
    eval_real.print_teleop_instructions(
        input_device, num_robots=1,
        has_match_dataset=has_match_dataset, has_match_episode=has_match_episode,
    )
    output = capsys.readouterr().out
    assert "do not press 2" in output
    assert "1/2 = control robot" not in output
    assert "0 = control all robots" not in output
    assert ("E/P = next/previous" in output) == has_match_episode
    assert ("M = move robot 1" in output) == has_match_dataset
    assert "WARNING: W" not in output
    assert "E/W = next/previous" not in output


@pytest.mark.parametrize("input_device", ["keyboard", "spacemouse"])
def test_policy_instructions_describe_return_to_human_control(capsys, input_device):
    eval_real.print_policy_instructions(input_device)
    output = capsys.readouterr().out
    assert "Policy execution controls" in output
    assert "S = end the episode and return to human control" in output
    assert "Q is available only after returning" in output
    assert ("holding it after returning commands -X motion" in output) == (input_device == "keyboard")
    assert "not an emergency stop" in output


def test_collision_helpers_use_module_level_imports():
    pose = np.zeros(6)
    eval_real.solve_table_collision(pose, gripper_width=0.085, height_threshold=0.1)
    assert pose[2] == pytest.approx(0.1)

    poses = np.zeros((2, 6))
    robots = [{"sphere_center": [0, 0, 0], "sphere_radius": 0.5} for _ in poses]
    eval_real.solve_sphere_collision(poses, robots)
    assert poses[0, 1] == pytest.approx(-0.055)
    assert poses[1, 1] == pytest.approx(0.055)


def test_display_helper_uses_module_level_imports():
    main_img = np.zeros((40, 60, 3), dtype=np.uint8)
    env = Mock()
    env.get_aux_realsense_rgb.return_value = None
    image = eval_real.build_display_image({}, main_img, env, 0, "Episode 0")
    assert image.shape == main_img.shape
    env.get_aux_realsense_rgb.assert_called_once_with(camera_idx=0, output_res=(60, 40))
