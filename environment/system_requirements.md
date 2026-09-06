# System requirements

This is the phase-2 environment boundary. Exact package locking and complete
paper reproduction are deferred because checkpoints and training data are out
of scope for the current release pass.

## Common

- Ubuntu 22.04 is the currently tested host.
- Python 3.10 is the target interpreter for the EmbodiSteer environment.
- A CUDA-capable GPU is required for diffusion inference and cuRobo guidance.
- Installing cuRobo requires a CUDA toolkit with `bin/nvcc`; pass its root to
  `scripts/bootstrap_third_party.py --install --cuda-home <path>`. The toolkit
  must use the same CUDA major version as the installed PyTorch build.
- `environment-simulation.yaml` and `environment-real.yaml`, with their
  included requirements files, record the selected direct
  runtime, including NumPy 1.26.4, SciPy 1.15.3, Numba 0.65.0 and
  pytorch-kinematics 0.10.0. They are not a solver-generated lockfile; the
  dependency boundary and clean-install procedure are documented in
  `dependencies.md`.
- `pytest` is a development dependency for the phase-2/phase-3 tests.

## Simulation

Install the pinned ManiSkill and cuRobo forks from `third_party/manifest.yaml`.
The manifest pins the reviewed ManiSkill commit; additional local working-tree
changes are excluded. GPU driver, CUDA and PhysX compatibility should be
printed by the simulation launcher and captured in the reproduction report.

## Real robot

Real deployment additionally requires the supported robot controller,
RealSense/GoPro capture devices, `ur-rtde` or the corresponding vendor driver,
SpaceMouse/spacenav where applicable, and an approved emergency-stop procedure.
No network address, credential, device node permission, or robot motion should
be inferred from this file.

Install `environment-real.yaml` and bootstrap with `--profile real`: real
joint-space inference requires cuRobo but not ManiSkill/SAPIEN. Install
`usbutils` for the UVC runtime's `lsusb` helper, configure serial/video device
permissions, and use a graphical session for keyboard/OpenCV interaction.
RealSense is optional; install its vendor SDK only for auxiliary recording.
The robot YAML defaults to `input_device: keyboard`. Selecting `spacemouse`
requires `libspnav-dev`/`spacenavd` and the optional pinned backend in
`spacemouse-requirements.txt`; see
[input-device setup](../docs/real_world.md#teleoperation-input-device).

The real dry-run imports the installed real runtime, but does not load a
checkpoint or start devices; it does not exercise GPU inference. The real
preflight is offline by default; `--check-devices` reads filesystem metadata,
and `--connect` explicitly opts into TCP connection tests without commands.
Neither is a substitute for the lab's independent hardware safety procedure.
