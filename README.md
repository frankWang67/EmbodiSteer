# EmbodiSteer

EmbodiSteer is an inference-time steering framework for deploying a frozen
Cartesian Diffusion Policy across robot embodiments. During denoising, the
policy state is lifted into the target robot's joint space, mapped through FK
and a damped Jacobian, and optionally corrected with cuRobo whole-body SDF
collision guidance.

This repository is a self-contained release tree. Simulation and physical
deployment use the same public policy package and versioned configuration.

Read the [`method overview`](docs/method.md) and
[`installation guide`](docs/installation.md), then choose
[`docs/simulation.md`](docs/simulation.md) or
[`docs/real_world.md`](docs/real_world.md). The current release pass excludes
checkpoints and training data; their machine-readable status is in
[`artifacts/manifest.yaml`](artifacts/manifest.yaml), with details in
[`docs/data_and_checkpoints.md`](docs/data_and_checkpoints.md).

## Repository layout

```text
EmbodiSteer/
├── embodisteer/              # public policies and reusable method components
├── diffusion_policy/         # Diffusion Policy model, data and training engine
├── umi/                      # real-device and data support
├── run_sim_pipeline.sh       # simple Conda wrapper for simulation workflows
├── run_sim_workflow.py       # primary simulation experiment interface
├── eval_sim_single_robot.py  # single-robot simulation evaluation
├── eval_sim_multi_robots.py  # multi-robot simulation evaluation
├── scripts_maniskill/        # simulation helpers and diagnostics
├── configs/workflows/        # config-driven data/train/evaluation workflow
├── scripts_real/             # real-device boundary and safety notes
├── scripts_slam_pipeline/    # data/SLAM preparation utilities
├── eval_real.py              # real-world evaluation entry point
├── third_party/              # dependency pins and asset provenance ledger
├── artifacts/                # checkpoint/data publication manifest
├── environment/              # environment and system requirements
├── tests/                    # CPU/layout tests and algorithm tests
└── docs/                     # release documentation
```

The two large external projects are not vendored here. Install the project
forks described in [`third_party/manifest.yaml`](third_party/manifest.yaml).
The ManiSkill and cuRobo entries are pinned to the commits recorded in the
manifest. Only those revisions are supported by the release scripts.

`diffusion_policy/` is a local model, data-loading and training engine. It is
kept API-compatible for checkpoint interoperability. All EmbodiSteer policy
implementations and reusable pose/Jacobian, SDF-reduction and CBF components
live under `embodisteer/`; new Hydra targets should use the public
`embodisteer.policies` aliases. See [`docs/architecture.md`](docs/architecture.md).

## Public policy imports

New configs should use the stable aliases below:

```python
from embodisteer.policies import EmbodiSteerJointPolicy
from embodisteer.policies import EmbodiSteerEESpacePolicy
```

Checkpoints trained with the bundled Diffusion Policy model remain
interoperable: the evaluation launchers load their stored configuration and
weights, then select the stable public EmbodiSteer policy target requested by
the policy YAML before constructing the workspace. New EmbodiSteer
checkpoints/configurations should target `embodisteer.policies`.

## Installation boundary

1. Create an environment from [`environment/environment.yaml`](environment/environment.yaml).
2. Install or expose this checkout (`pip install -e .` or run commands from
   this directory).
3. Validate the dependency manifest:

   ```console
   python scripts/bootstrap_third_party.py --check
   ```

4. Materialize the pinned dependencies with the bootstrap script. The script
   never runs implicitly.

The repository does not publish or require project checkpoint and training
data artifacts for installation. The empty artifact inventory is intentional
rather than a missing download link. Complete paper-table reproduction will be
documented only after their release location is decided.

## Simulation entry points

`run_sim_pipeline.sh` is the simplest interface for simulation experiments; it
enters the `embodisteer` Conda environment and forwards all options to
`run_sim_workflow.py`. The Python workflow
orchestrates collection, conversion, validation, training, multi-profile
evaluation and result aggregation from one workflow YAML. The lower-level
`eval_sim_single_robot.py` and the retained convenience
`eval_sim_multi_robots.py` remain available for direct checkpoint probes.
Together with `eval_real.py`, they read algorithm settings from the same policy
YAML. The paper profile is
[`configs/policy/embodisteer.yaml`](configs/policy/embodisteer.yaml); the
Cartesian profile is [`configs/policy/ee.yaml`](configs/policy/ee.yaml).

```console
python eval_sim_single_robot.py --help
python eval_sim_multi_robots.py --help
./run_sim_pipeline.sh --help
python eval_real.py --help
```

Checkpoint, environment, robot and output paths remain command-line arguments;
inference space, guidance/CBF/SDF, kinematic and baseline settings belong in the
policy YAML, including the Cartesian GD collision geometry and schedule. See
[`docs/policy_configuration.md`](docs/policy_configuration.md). Run simulation
only after installing the pinned ManiSkill fork and providing a compatible
checkpoint.

For a newly trained checkpoint, preview the complete simulation pipeline with:

```console
./run_sim_pipeline.sh \
  --config configs/workflows/simulation.yaml --stage all --dry-run
```

The workflow separates collection, conversion, dataset validation, training
and evaluation into resumable stages. Its eval stage isolates every
profile/robot pair, writes `run_manifest.yaml`, `results.json`, `results.md`,
per-robot metrics and episode arrays, and supports `--resume`, `--force`,
`--checkpoint`, `--output-dir`, `--run-id` and `--robots`. Generated data,
checkpoints and results remain in gitignored or explicitly configured external
locations.

For another task, copy a workflow YAML into
`configs/workflows/<task>/`, change its task, artifact and evaluation fields,
then pass that file with `--config`; the shell script itself does not need to
be copied or edited. Set `EMBODISTEER_CONDA_ENV` only when using a Conda
environment with a different name.

## Real-world code

`eval_real.py` and `umi/real_world/` are included so the simulation and physical
deployment paths share the same policy package. `scripts_real/` documents the
intentional exclusion of duplicate, site-bound demos. Real robot execution
requires an approved emergency-stop procedure, controller and camera setup,
and explicit robot/gripper configuration. Never place credentials or
site-specific network addresses in a public config. The real-world path is not
a substitute for hardware safety validation.

## Scope and licensing

Checkpoint files, training data, wandb runs and large experiment outputs are
not part of this release tree. The repository includes Block Pushing and
Kitchen/Franka benchmark assets. Read
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and the machine-readable
[`third_party/assets.yaml`](third_party/assets.yaml) before redistributing
them. ManiSkill assets are CC BY-NC 4.0, and cuRobo is limited to
non-commercial research/evaluation under its NVIDIA License. The root
`LICENSE` does not replace any asset or external dependency terms.

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). Shihefeng Wang
and Kangchen Lv contributed equally; Mingrui Yu and Xiang Li are
co-corresponding authors.
Contribution and release-note conventions are documented in
[`CONTRIBUTING.md`](CONTRIBUTING.md) and [`CHANGELOG.md`](CHANGELOG.md).
