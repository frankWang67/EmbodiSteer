# Dependency boundary

This document defines the direct runtime dependencies for EmbodiSteer. It is a
curated installation specification, not a dump of every transitive wheel.

## Sources of truth

Dependencies are split across three sources:

1. `environment/environment-simulation.yaml` and
   `environment/environment-real.yaml` define separate single-purpose
   profiles. Both share the policy/CUDA pins and cuRobo; only simulation adds
   ManiSkill/SAPIEN, while only real adds robot/camera/gripper packages.
2. `third_party/manifest.yaml` pins the editable project forks. The selected
   profile's source trees must be checked out at those exact commits.
3. `pyproject.toml` only describes the local Python packages and development
   test extra. It intentionally does not try to choose a CUDA wheel or replace
   the environment and fork manifests.

`base-requirements.txt` holds shared version pins. The simulation and real
requirements files include it, and each Conda YAML includes its sibling
requirements file. This avoids copying shared pins into both profiles. Use a
local checkout for Conda's relative requirements includes.

`environment.yaml` and `requirements.txt` are compatibility unions, including
the optional `legacy-requirements.txt` (Robomimic/MuJoCo/SpaceMouse helpers).
Neither public profile installs these legacy extras by default. New
installations should use the profile-specific files.

Python, pip, PyAV, CMake, LLVM OpenMP and ExifTool intentionally remain in the
conda portion only. CMake and LLVM OpenMP are native build/runtime constraints,
while PyAV requires FFmpeg libraries and PyExifTool invokes the separately
installed ExifTool executable. They therefore do not appear in
`requirements.txt`, whose scope is Python packages that pip can install without
native system headers.

## Runtime versions

The supported runtime versions are:

| package | supported version |
| --- | --- |
| Python | 3.10.14 |
| PyTorch / torchvision | 2.7.1+cu128 / 0.22.1+cu128 |
| NumPy | 1.26.4 |
| SciPy | 1.15.3 |
| Numba / llvmlite | 0.65.0 / 0.47.0 |
| OpenCV | `opencv-python` 4.7.0.72 |
| ManiSkill fork | 3.0.0b21 at the manifest commit |
| pytorch-kinematics | 0.10.0 |
| SAPIEN / mplib | 3.0.3 / 0.1.1 |
| cuRobo fork | the manifest commit, installed editable |
| warp-lang | 1.12.1 |

The pinned ManiSkill source and the shared requirements use
`pytorch-kinematics==0.10.0`. Use `bootstrap_third_party.py --profile simulation`
for ManiSkill + cuRobo, or `--profile real` for cuRobo alone. Real joint-space
inference still uses cuRobo; removing ManiSkill does not remove that requirement.

## Real-world dependency profile

The default gripper is the serial Robotiq 2F-85. The real profile includes
`robotiq_gripper` from
[`frankWang67/robotiq_modbus_gripper`](https://github.com/frankWang67/robotiq_modbus_gripper)
at commit `582a6c26a2462adb58c60ba229ad9578556cf464`, together with
`pymodbus==3.8.6` (the driver's declared dependency) and `pyserial==3.5`
(required for Modbus RTU serial transport). The driver is a normal pip
dependency, not another editable fork in `third_party/manifest.yaml`.

WSG50 uses the in-tree TCP driver and does not require the Robotiq packages.
Neither gripper is contacted during installation. Connection parameters and
the command for adding these packages to an existing environment are in
[`docs/real_world.md`](../docs/real_world.md#gripper-dependencies).

## Deliberately optional dependencies

The real robot YAML defaults to `input_device: keyboard`. To use `spacemouse`,
install `spacemouse-requirements.txt` and the native `libspnav-dev`/`spacenavd`
packages. Neither default deployment profile installs this extra. Both input
modes still need the real profile's keyboard dependencies for episode shortcuts.

The repository includes optional PushT, Block Pushing, Kitchen, Robomimic and
RealSense modules. Some of those modules import optional packages
such as `pymunk`, `pybullet`, `pygame`, `dm_control`, `tf_agents`,
`pyrealsense2` or `r3m`; they are not required by the EmbodiSteer paper path.
Do not add them to the default environment unless the corresponding optional
module is explicitly brought into the supported scope.

Several calibration helpers also need device-specific extras. For example,
the QR-based UVC latency script requires `qrcode`, and RealSense capture
requires the vendor SDK. These remain hardware extras and must be tested on the
target workstation rather than silently pulled into the common environment.

## Profile acceptance check

The dependency specification is accepted only after all of the following have
been performed in a new environment:

1. create the selected profile's Conda environment;
2. materialize and install only the selected profile's manifest commits;
3. install this repository and the development test extra;
4. run `python -m pip check` with no conflicts;
5. record Python, PyTorch, CUDA, cuRobo, NumPy, GPU and selected SDK versions;
6. run static tests plus a short ManiSkill rollout for simulation, or offline
   preflight and the lab's separately authorized bring-up for real hardware.

Lightweight CI installs neither profile's heavy runtime: it checks dependency
lists, CLI import boundaries and mocked preflight behavior. This is not a
claim that fresh CUDA profile installation or physical bring-up has passed.
