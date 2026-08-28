# Dependency boundary

This document defines the direct runtime dependencies for EmbodiSteer. It is a
curated installation specification, not a dump of every transitive wheel.

## Sources of truth

Dependencies are split across three sources:

1. `environment/environment.yaml` contains the Python/CUDA runtime used by the
   repository, including the pinned versions needed by the public entry points.
2. `third_party/manifest.yaml` pins the two editable project forks. The
   ManiSkill and cuRobo source trees must be checked out at those exact commits
   before installation.
3. `pyproject.toml` only describes the local Python packages and development
   test extra. It intentionally does not try to choose a CUDA wheel or replace
   the environment and fork manifests.

`environment/requirements.txt` mirrors the pip portion for users who maintain a
pip-based installation workflow. New installations should use the conda YAML.

Python, pip, CMake, LLVM OpenMP and ExifTool intentionally remain in the conda
portion only. CMake and LLVM OpenMP are native build/runtime constraints, while
PyExifTool invokes the separately installed ExifTool executable. They therefore
do not appear in `requirements.txt`, whose scope is Python packages that
pip can install.

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

The pinned ManiSkill source and this environment both use
`pytorch-kinematics==0.10.0`. A clean installation must install the pinned
ManiSkill and cuRobo revisions from `third_party/manifest.yaml` so their
metadata and source agree.

## Deliberately optional dependencies

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

## Phase-3 acceptance check

The dependency specification is accepted only after all of the following have
been performed in a new environment:

1. create the conda environment from `environment/environment.yaml`;
2. materialize and install both manifest commits;
3. install this repository and the development test extra;
4. run `python -m pip check` with no conflicts;
5. record `python`, PyTorch, CUDA, ManiSkill, cuRobo, NumPy and GPU versions;
6. run the static tests and a short ManiSkill rollout.
