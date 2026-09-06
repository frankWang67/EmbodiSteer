#!/usr/bin/env python3
"""Check a real installation offline; optional TCP checks send no commands.

Never construct controllers, open serial ports, reset cameras, activate a
gripper, load a checkpoint, or move a robot. Package discovery is not an ABI,
CUDA, camera-stream, or hardware-protocol test.
"""

from __future__ import annotations

import argparse
from importlib import metadata, util
import os
from pathlib import Path
import shutil
import socket
import stat
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from embodisteer.real_config import (
    load_obstacle_geometry, load_yaml_mapping, validate_real_policy,
    validate_robot_config,
)
from embodisteer.runtime_config import load_policy_config


def required_packages(config, policy, record_realsense=False):
    """Return distribution -> top-level module, conditional on the deployment."""
    packages = {
        "torch": "torch", "torchvision": "torchvision", "numpy": "numpy",
        "scipy": "scipy", "timm": "timm", "diffusers": "diffusers",
        "hydra-core": "hydra", "omegaconf": "omegaconf", "dill": "dill",
        "einops": "einops", "opencv-python": "cv2", "av": "av",
        "zarr": "zarr", "numcodecs": "numcodecs", "imagecodecs": "imagecodecs",
        "atomics": "atomics", "pynput": "pynput", "threadpoolctl": "threadpoolctl",
    }
    if policy["inference_space"] == "joint":
        packages.update({"nvidia-curobo": "curobo", "pytorch-kinematics": "pytorch_kinematics"})
    for robot in config["robots"]:
        if robot["robot_type"] in ("ur5", "ur5e"):
            packages["ur-rtde"] = "rtde_control"
        else:
            packages["zerorpc"] = "zerorpc"
    for gripper in config["grippers"]:
        if gripper.get("gripper_type", "robotiq") == "robotiq":
            packages.update({
                "robotiq-gripper": "robotiq_gripper", "pymodbus": "pymodbus",
                "pyserial": "serial",
            })
        else:
            packages["fastcrc"] = "fastcrc"
    if record_realsense:
        packages["pyrealsense2"] = "pyrealsense2"
    if config.get("input_device", "keyboard") == "spacemouse":
        packages["spnav"] = "spnav"
    return packages


def check_packages(packages):
    """Inspect installed metadata/specs without importing runtime packages."""
    results = []
    serial_versions = {"pymodbus": "3.8.6", "pyserial": "3.5", "robotiq-gripper": "0.1.0"}
    for distribution, module in packages.items():
        try:
            version = metadata.version(distribution)
            if util.find_spec(module) is None:
                raise ModuleNotFoundError(module)
            expected = serial_versions.get(distribution)
            if expected is not None and version != expected:
                results.append(("FAIL", distribution, f"installed {version}; expected {expected}"))
            else:
                results.append(("PASS", distribution, f"{version}; module found (not imported)"))
        except (metadata.PackageNotFoundError, ModuleNotFoundError, ValueError) as exc:
            results.append(("FAIL", distribution, f"missing package/module: {exc}"))
    # Both modules come from the RTDE distribution.
    if "ur-rtde" in packages and util.find_spec("rtde_receive") is None:
        results.append(("FAIL", "rtde_receive", "missing receive module from ur-rtde"))
    return results


def check_device_path(path, label):
    """stat/access only: opening serial ports can change device state."""
    path = Path(path)
    try:
        mode = path.stat().st_mode
        if not stat.S_ISCHR(mode):
            return ("FAIL", label, f"{path} is not a character device")
        if not os.access(path, os.R_OK | os.W_OK):
            return ("FAIL", label, f"read/write permission missing for {path}")
    except OSError as exc:
        return ("FAIL", label, str(exc))
    return ("PASS", label, f"{path}: character device with read/write access; not opened")


def check_devices(config):
    results = []
    for idx, gripper in enumerate(config["grippers"]):
        if gripper.get("gripper_type", "robotiq") == "robotiq":
            results.append(check_device_path(gripper["gripper_serial_port"], f"grippers[{idx}].serial"))
    cameras = sorted(Path("/dev/v4l/by-id").glob("*video*index0"))
    if not cameras:
        results.append(("FAIL", "UVC cameras", "no /dev/v4l/by-id/*video*index0 paths"))
    for path in cameras:
        results.append(check_device_path(path, "UVC camera"))
    if not shutil.which("lsusb"):
        results.append(("FAIL", "lsusb", "install usbutils; no USB enumeration/reset was attempted"))
    if not os.environ.get("DISPLAY"):
        results.append(("WARN", "display", "DISPLAY is unset; real keyboard/OpenCV UI needs a graphical session"))
    return results


def network_targets(config):
    for idx, robot in enumerate(config["robots"]):
        # RTDE endpoint and the existing FrankaInterface server default.
        port = 30004 if robot["robot_type"] in ("ur5", "ur5e") else 4242
        yield f"robots[{idx}]", robot["robot_ip"], port
    for idx, gripper in enumerate(config["grippers"]):
        if gripper.get("gripper_type", "robotiq") == "wsg50":
            yield f"grippers[{idx}]", gripper["gripper_ip"], gripper.get("gripper_port", 1000)


def check_network(config, timeout):
    results = []
    for label, host, port in network_targets(config):
        try:
            with socket.create_connection((host, port), timeout=timeout):
                pass  # No application bytes or controller commands.
            results.append(("PASS", label, f"TCP {host}:{port}; protocol/readiness not tested"))
        except OSError as exc:
            results.append(("FAIL", label, f"TCP {host}:{port}: {exc}"))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot_config", "--robot-config", required=True, type=Path)
    parser.add_argument("--obstacle_config", "--obstacle-config", required=True, type=Path)
    parser.add_argument("--policy-config", type=Path, default=ROOT / "configs/policy/embodisteer.yaml")
    parser.add_argument("--record-realsense", action="store_true", help="require the optional RealSense SDK")
    parser.add_argument("--check-devices", action="store_true", help="stat/access configured serial and UVC paths; never open them")
    parser.add_argument("--connect", "--check-network", action="store_true", dest="connect",
                        help="opt in to TCP connection checks only; never send commands or open serial ports")
    parser.add_argument("--timeout", type=float, default=2.0, help="TCP socket timeout in seconds (DNS resolution is OS-managed)")
    args = parser.parse_args(argv)
    if not 0 < args.timeout <= 30:
        parser.error("--timeout must be in (0, 30]")
    try:
        config = load_yaml_mapping(args.robot_config)
        policy = load_policy_config(args.policy_config)
        validate_robot_config(config)  # Site check rejects template addresses.
        validate_real_policy(config, policy)
        load_obstacle_geometry(args.obstacle_config)
    except (OSError, ValueError) as exc:
        print(f"FAIL configuration: {exc}")
        return 1

    results = [("PASS", "configuration", "site addresses and policy combination validated")]
    results.extend(check_packages(required_packages(config, policy, args.record_realsense)))
    if args.check_devices:
        results.extend(check_devices(config))
    else:
        results.append(("SKIP", "devices", "use --check-devices for stat/access checks"))
    if args.record_realsense:
        results.append(("SKIP", "RealSense hardware", "SDK presence only; device identity/stream not checked"))
    # Do not contact hardware if earlier checks already failed.
    if args.connect and not any(status == "FAIL" for status, _, _ in results):
        results.extend(check_network(config, args.timeout))
    else:
        results.append(("SKIP", "network", "--connect not requested or offline checks failed"))
    for status, label, detail in results:
        print(f"{status} {label}: {detail}")
    print("No controller, serial port, camera stream or gripper was activated. "
          "CUDA/ABI, calibration, protocol responses and safe operation remain unverified.")
    if any(status == "FAIL" for status, _, _ in results):
        print("Install the real profile and selected hardware extras; see docs/real_world.md.")
        return 1
    print("Requested preflight checks passed; this is not hardware readiness approval.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
