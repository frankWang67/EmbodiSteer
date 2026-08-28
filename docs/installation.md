# Installation

EmbodiSteer is distributed as a self-contained repository.

## Environment

Use Python 3.10 and create the environment from
[`environment/environment.yaml`](../environment/environment.yaml). The file is
the supported CUDA runtime specification; the PyTorch index and native packages
should be adjusted only for the host's CUDA driver. The root `pyproject.toml`
provides editable-package metadata and a small development extra, but it does
not replace the environment specification because cuRobo, ManiSkill, PhysX,
camera drivers and robot drivers are system- and fork-dependent.
The dependency tiers and optional modules are documented in
[`environment/dependencies.md`](../environment/dependencies.md).

```console
conda env create -f environment/environment.yaml
conda activate embodisteer
```

## Fixed forks

Install the two external projects described in
[`third_party/manifest.yaml`](../third_party/manifest.yaml). The bootstrap
checker is read-only:

```console
python scripts/bootstrap_third_party.py --check
```

To materialize and install the exact forks into `third_party/src/`, run the
explicit mutating form only after reviewing the manifest:

```console
python scripts/bootstrap_third_party.py --install
```

ManiSkill and cuRobo are pinned to the commits recorded in the manifest. The
bootstrap script installs only those revisions.

## Local package

After the fixed forks are installed, install this repository. Use the
development extra for the checked-in tests, then verify package metadata:

```console
python -m pip install -e '.[dev]'
python -m pip check
```

## Smoke checks

These checks do not need checkpoints, datasets or robot hardware:

```console
python scripts/diagnostics/check_release_layout.py
python -m compileall -q embodisteer diffusion_policy umi scripts_maniskill scripts_slam_pipeline eval_sim_single_robot.py eval_sim_multi_robots.py eval_real.py
python eval_sim_single_robot.py --help
python eval_sim_multi_robots.py --help
python eval_real.py --help
python eval_real.py --dry_run \
  --robot_config configs/real/robot.template.yaml \
  --obstacle_config configs/real/obstacles/door_frame.yaml
```

The last command intentionally rejects template addresses only when a device
run is requested; `--dry_run` validates shape and geometry without connecting.

The release tree does not bundle ManiSkill/cuRobo meshes or local calibration
recordings. Those assets are supplied by the fixed dependency forks or by a
site-specific configuration. The repository includes Block Pushing and
Kitchen/Franka benchmark assets; consult
[`third_party/assets.yaml`](../third_party/assets.yaml) before redistributing
them.
