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
├── eval_sim_single_robot.py  # single-robot simulation evaluation
├── eval_sim_multi_robots.py  # multi-robot simulation evaluation
├── scripts_maniskill/        # simulation helpers and diagnostics
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

Existing Diffusion Policy checkpoint targets continue to resolve through the
`diffusion_policy.policy` namespace. EmbodiSteer checkpoints/configurations
should target `embodisteer.policies`; checkpoint-compatibility classes are
available under `embodisteer.policies.legacy` when required.

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

The current phase does not publish or require project checkpoint and training
data artifacts. The empty artifact inventory is intentional rather than a
missing download link. Complete paper-table reproduction will be documented
only after their release location is decided.

## Simulation entry points

The two simulation entry points are `eval_sim_single_robot.py` and
`eval_sim_multi_robots.py`. Together with `eval_real.py`, they read algorithm
settings from the same policy YAML. The paper profile is
[`configs/policy/embodisteer.yaml`](configs/policy/embodisteer.yaml); the
Cartesian profile is [`configs/policy/ee.yaml`](configs/policy/ee.yaml).

```console
python eval_sim_single_robot.py --help
python eval_sim_multi_robots.py --help
python eval_real.py --help
```

Checkpoint, environment, robot and output paths remain command-line arguments;
inference space, guidance/CBF/SDF, IK, baseline and JM2D settings belong in the
policy YAML, including the Cartesian GD collision geometry and schedule. See
[`docs/policy_configuration.md`](docs/policy_configuration.md). Run simulation only after installing the pinned ManiSkill fork
and providing a compatible checkpoint.

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
