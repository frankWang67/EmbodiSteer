# Installation

Choose one deployment profile. Simulation and real machines do not need each
other's SDKs. Both share policy/runtime pins and cuRobo for joint-space inference.

## Simulation workstation

Run from a local checkout:

```console
conda env create -f environment/environment-simulation.yaml
conda activate embodisteer-sim
python scripts/bootstrap_third_party.py --profile simulation --check
python scripts/bootstrap_third_party.py --profile simulation --install --cuda-home /usr/local/cuda
python -m pip install -e '.[dev]'
python -m pip check
python run_sim_workflow.py --config configs/workflows/simulation.yaml --stage all --dry-run
```

This installs the pinned ManiSkill and cuRobo forks, not RTDE, Robotiq,
Modbus, RealSense or keyboard drivers. The shell wrapper
`run_sim_pipeline.sh` defaults to `embodisteer-sim`; set
`EMBODISTEER_CONDA_ENV` if using another name.

## Real-robot workstation

Use this profile instead, not as an overlay on the simulation environment:

```console
conda env create -f environment/environment-real.yaml
conda activate embodisteer-real
python scripts/bootstrap_third_party.py --profile real --check
python scripts/bootstrap_third_party.py --profile real --install --cuda-home /usr/local/cuda
python -m pip install -e '.[dev]'
python -m pip check
python eval_real.py --dry_run \
  --robot_config configs/real/robot.template.yaml \
  --obstacle_config configs/real/obstacles/door_frame.yaml \
  --policy-config configs/policy/embodisteer.yaml
```

The real profile includes the default Robotiq serial driver, RTDE, Franka
RPC, UVC recording and keyboard dependencies. It installs **cuRobo only** from
the fork manifest, with no ManiSkill/SAPIEN/PhysX or legacy MuJoCo environment.
cuRobo is still required by the real joint-space paper method. RealSense SDK
installation is optional and workstation-specific; add it when using
`--record_realsense`. The robot YAML defaults to `input_device: keyboard`;
selecting `spacemouse` additionally needs the optional packages in
`environment/spacemouse-requirements.txt`, plus `libspnav-dev`/`spacenavd`.
See [input-device setup](real_world.md#teleoperation-input-device) and
[system requirements](../environment/system_requirements.md) for hardware extras.

Populate a private copy of the robot template, then run the separate offline
preflight:

```console
python scripts/preflight_real.py \
  --robot_config /private/configs/robot.yaml \
  --obstacle_config /private/configs/obstacles.yaml \
  --policy-config configs/policy/embodisteer.yaml
```

See [real-world deployment](real_world.md) for explicit device-path/TCP check
options and the safety boundary. A passing dry-run or preflight is not proof
of functioning hardware or permission to move a robot.

## Shared dependency management

The Python 3.10/CUDA runtime remains pinned. Replace `/usr/local/cuda` with
the CUDA toolkit containing `bin/nvcc` and matching the PyTorch CUDA major
version. cuRobo builds against the installed PyTorch using
`--no-build-isolation`. Omit `--cuda-home` if `CUDA_HOME` is already correct.

Both conda YAMLs include their sibling pip requirements file. Use the local
checkout (do not pass a remote YAML URL to conda). Pip users can install
`environment/simulation-requirements.txt` or
`environment/real-requirements.txt` after supplying the native packages
listed in the corresponding conda YAML.

`environment/environment.yaml` and `environment/requirements.txt` remain
compatibility unions for developers using both profiles and retained legacy
helpers. They are not the default single-purpose installation. Existing
environments are not changed automatically. Shared version pins live only in
`base-requirements.txt`; each profile includes it.

The root `pyproject.toml` describes the local package and development extra,
not CUDA or vendor SDK selection. Fixed fork revisions remain in
[the manifest](../third_party/manifest.yaml). Installation never starts a robot
controller. [Dependency details](../environment/dependencies.md) describe the
optional modules.

## Lightweight checks and tests

`eval_real.py --dry_run` assumes the real profile is installed and imports its
runtime modules normally, but does not load a checkpoint or connect to devices.
It does not require the simulation profile. The simulation workflow preview
does not instantiate a simulator or require hardware drivers.

The following static/schema and mocked preflight tests remain lightweight
(Click, PyYAML, NumPy and pytest); they do not import the real launcher or need
a GPU or devices. Run `tests/test_eval_real_cli_validation.py` in the installed
real environment to test the launcher's help, dry-run and early-exit behavior.

```console
python scripts/diagnostics/check_release_layout.py
python -m pytest -q tests/test_deployment_boundaries.py tests/test_real_preflight.py
```

These tests deliberately block cross-profile imports and mock all network and
hardware operations. The full research suite additionally needs the pinned
forks; run `python -m pytest -q` in a prepared development environment.
Clean installations of each CUDA profile still require separate `pip check`
and runtime smoke tests; lightweight CI does not certify GPU/driver readiness.

The release tree does not bundle ManiSkill/cuRobo meshes or local calibration
recordings. Those come from the fixed forks or site configuration. Consult
[the asset ledger](../third_party/assets.yaml) before redistribution.
