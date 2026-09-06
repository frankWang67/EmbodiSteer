"""Lightweight deployment boundaries; no CUDA or hardware packages needed."""

import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from scripts import bootstrap_third_party

ROOT = Path(__file__).resolve().parents[1]


def requirement_lines(path):
    result = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("-r "):
            result.extend(requirement_lines(path.parent / line[3:]))
        elif line and not line.startswith(("#", "--")):
            result.append(line)
    return result


def package_names(lines):
    return {line.split("==")[0].split(" @ ")[0].lower().replace("_", "-") for line in lines}


@pytest.mark.parametrize("profile", ["simulation", "real"])
def test_dependency_profiles_are_separate(profile):
    env_dir = ROOT / "environment"
    config = yaml.safe_load((env_dir / f"environment-{profile}.yaml").read_text())
    pip = next(item["pip"] for item in config["dependencies"] if isinstance(item, dict))
    assert pip == [f"-r {profile}-requirements.txt"]
    packages = package_names(requirement_lines(env_dir / f"{profile}-requirements.txt"))
    assert {"torch", "pytorch-kinematics", "warp-lang", "timm"} <= packages
    hardware = {"ur-rtde", "robotiq-gripper", "pymodbus", "pyserial", "pynput", "zerorpc", "v4l2py"}
    simulation = {"sapien", "mplib", "fast-kinematics", "gymnasium"}
    if profile == "simulation":
        assert hardware.isdisjoint(packages)
        assert simulation <= packages
    else:
        assert simulation.isdisjoint(packages)
        assert hardware <= packages
    assert {"robomimic", "free-mujoco-py", "robosuite", "spnav"}.isdisjoint(packages)


def test_shared_pins_and_compatibility_union_have_no_conflicts():
    versions = {}
    for line in requirement_lines(ROOT / "environment/requirements.txt"):
        name = line.split("==")[0].split(" @ ")[0].lower().replace("_", "-")
        assert versions.setdefault(name, line) == line
    assert "robotiq-gripper" in versions and "sapien" in versions


def test_spacemouse_extra_reuses_the_compatibility_pin():
    env_dir = ROOT / "environment"
    extra = requirement_lines(env_dir / "spacemouse-requirements.txt")
    assert package_names(extra) == {"spnav"}
    assert set(extra) <= set(requirement_lines(env_dir / "legacy-requirements.txt"))


def run_blocked(script, args, denied):
    program = """
import importlib.abc
import runpy
import socket
import sys
class BlockImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in DENIED:
            raise AssertionError('forbidden runtime import: ' + fullname)
sys.meta_path.insert(0, BlockImports())
def no_connection(*args, **kwargs):
    raise AssertionError('configuration checks must not connect')
socket.create_connection = no_connection
sys.argv = [SCRIPT, *ARGS]
runpy.run_path(SCRIPT, run_name='__main__')
"""
    header = f"DENIED = {denied!r}\nSCRIPT = {str(ROOT / script)!r}\nARGS = {args!r}\n"
    return subprocess.run(
        [sys.executable, "-c", header + program], cwd=ROOT,
        text=True, capture_output=True, timeout=30,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


def test_real_config_module_has_no_runtime_imports():
    result = run_blocked("embodisteer/real_config.py", [], {
        "torch", "numpy", "scipy", "av", "cv2", "hydra", "omegaconf",
        "umi", "diffusion_policy", "curobo", "mani_skill", "sapien",
        "robotiq_gripper", "pymodbus", "serial", "rtde_control", "pynput",
    })
    assert result.returncode == 0, result.stdout + result.stderr


def test_real_launcher_keeps_imports_at_module_level():
    tree = ast.parse((ROOT / "eval_real.py").read_text())
    nested_imports = [
        node.lineno
        for function in ast.walk(tree)
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
        for node in ast.walk(function)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert not nested_imports, f"Deferred launcher imports at lines {nested_imports}"


def test_manual_shortcuts_do_not_overlap_keyboard_motion_keys():
    # Inspect the actual event handlers without starting listeners or hardware.
    launcher = ast.parse((ROOT / "eval_real.py").read_text())
    constants = {
        node.targets[0].id: node.value.value
        for node in launcher.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
    }
    event_loops = sorted([
        node for node in ast.walk(launcher)
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name)
        and node.target.id == "key_stroke"
    ], key=lambda node: node.lineno)
    assert len(event_loops) == 2  # Human and policy control are separate phases.
    phase_keys = []
    for loop in event_loops:
        keys = set()
        for node in ast.walk(loop):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "KeyCode":
                value = next(item.value for item in node.keywords if item.arg == "char")
                keys.add(value.value if isinstance(value, ast.Constant) else constants[value.id])
        phase_keys.append(keys)
    assert phase_keys[0] == {"q", "c", "e", "p", "m", "0", "1", "2"}
    assert phase_keys[1] == {"s"}

    keyboard = ast.parse((ROOT / "umi/real_world/keyboard_shared_memory.py").read_text())
    motion_keys = {
        node.left.value for node in ast.walk(keyboard)
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Constant)
        and len(node.ops) == 1 and isinstance(node.ops[0], ast.In)
        and isinstance(node.comparators[0], ast.Name)
        and node.comparators[0].id == "active_keys"
    }
    assert motion_keys == set("wasdrfikjluozx")
    assert phase_keys[0].isdisjoint(motion_keys)


def test_simulation_preview_does_not_import_hardware_or_simulator():
    result = run_blocked("run_sim_workflow.py", [
        "--config", "configs/workflows/simulation.yaml", "--stage", "all", "--dry-run",
    ], {"umi", "rtde_control", "rtde_receive", "robotiq_gripper", "serial",
        "pymodbus", "pynput", "zerorpc", "mani_skill", "sapien", "torch", "curobo"})
    assert result.returncode == 0, result.stdout + result.stderr


def test_real_preflight_help_is_dependency_light():
    result = run_blocked("scripts/preflight_real.py", ["--help"], {
        "torch", "umi", "diffusion_policy", "mani_skill", "sapien",
        "curobo", "robotiq_gripper", "serial", "pymodbus",
    })
    assert result.returncode == 0, result.stdout + result.stderr


def test_real_bootstrap_does_not_materialize_maniskill(tmp_path, monkeypatch):
    commands = []
    manifest = bootstrap_third_party._load_manifest()
    monkeypatch.setattr(bootstrap_third_party, "_load_manifest", lambda: manifest)
    monkeypatch.setattr(bootstrap_third_party, "repository_root", lambda: tmp_path)
    monkeypatch.setattr(bootstrap_third_party, "_run",
                        lambda args, **kwargs: commands.append(args) or "")
    bootstrap_third_party.materialize(profile="real", install=True)
    assert commands
    flattened = json.dumps(commands)
    assert "curobo" in flattened.lower()
    assert "maniskill" not in flattened.lower()
    assert "--no-build-isolation" in flattened


def test_real_manifest_check_ignores_unselected_pending_fork(monkeypatch):
    manifest = bootstrap_third_party._load_manifest()
    manifest["repositories"]["maniskill"]["commit"] = None
    monkeypatch.setattr(bootstrap_third_party, "_load_manifest", lambda: manifest)
    assert bootstrap_third_party.validate_manifest(profile="real") == []
    assert bootstrap_third_party.validate_manifest(profile="simulation")
