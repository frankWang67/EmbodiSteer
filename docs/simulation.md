# Simulation

The primary simulation interface is the thin
[`run_sim_pipeline.sh`](../run_sim_pipeline.sh) wrapper around
[`run_sim_workflow.py`](../run_sim_workflow.py). The direct
[`eval_sim_single_robot.py`](../eval_sim_single_robot.py) and retained
[`eval_sim_multi_robots.py`](../eval_sim_multi_robots.py) launchers are useful
for focused checkpoint probes. They use the same policy classes, router and
algorithm config as the physical path:

```python
from embodisteer.policies import DiffusionUnetTimmPolicyEmbodiSteer
```

Method selection now lives in a policy YAML:

| policy YAML fields | method |
| --- | --- |
| `inference_space: ee`, `guidance.method: ""` | Cartesian EE baseline |
| `inference_space: joint`, `guidance.method: ""` | EE-to-joint no-guidance comparison (`DiffusionUnetTimmPolicyJointSpace`) |
| `inference_space: joint`, `guidance.method: cbf` | EmbodiSteer with joint-space CBF-QP guidance |
| `inference_space: joint`, `guidance.method: gd` | gradient guidance comparison (`DiffusionUnetTimmPolicyJointSpace`) |

The task and robot identifiers used in the paper are listed in
[`configs/experiments/paper.yaml`](../configs/experiments/paper.yaml). The
ManiSkill fork supplies the task definitions and robot assets; cuRobo supplies
the robot descriptions, collision spheres and SDF interface. Install both
before launching an evaluation.

Algorithm defaults are documented in
[`configs/policy/embodisteer.yaml`](../configs/policy/embodisteer.yaml). Pass a
profile with `--policy-config`; use
[`configs/policy/ee.yaml`](../configs/policy/ee.yaml) for the Cartesian
baseline.
The workflow below supports data generation and training. Operational arguments are
listed by `python eval_sim_single_robot.py --help` and
`python eval_sim_multi_robots.py --help`.

The ManiSkill training config lives at `configs/simulation/hydra/`. The
collection/training launcher passes that path explicitly, while shared
Diffusion Policy training configs remain under `diffusion_policy/config/`.

## End-to-end workflow

Data generation, conversion, training and simulation evaluation are exposed as
independent stages of one config-driven launcher:

```bash
./run_sim_pipeline.sh \
  --config configs/workflows/simulation.yaml --stage all --dry-run
```

Remove `--dry-run` to execute the stages in order. The default workflow uses
the pinned ManiSkill checkout, writes raw demos and the converted `*.zarr.zip`
dataset under ignored `data/` paths, trains into ignored `data/outputs/`, and
evaluates the resulting `checkpoints/latest.ckpt`. Run a single stage with
`--stage collect|convert|validate|train|eval` when resuming after an
interruption. Edit the YAML to choose the task, collection count, GPU,
collection robot UIDs, training overrides, policy profiles and evaluation
robot list; the checked-in workflow evaluates both the EE baseline and
EmbodiSteer. Do not put datasets or checkpoints under tracked source
directories.

To add a task, place its workflow under `configs/workflows/<task>/` and select
it with `--config`. Task names, environment IDs, artifact paths, collection
settings, training overrides, policy profiles and robots all belong in that
YAML; `run_sim_pipeline.sh` remains unchanged. It uses the `embodisteer-sim` Conda
environment by default, which can be overridden with
`EMBODISTEER_CONDA_ENV=<name>`.

Complete pipeline and multi-method evaluation configs are provided for all
three paper tasks:

- `configs/workflows/place_toast/`
- `configs/workflows/turn_faucet/`
- `configs/workflows/make_iced_coffee/`

Each directory contains `full_pipeline.yaml` and
`evaluation_all_methods.yaml`. The former collects 200 demonstrations, trains
for 120 epochs and evaluates EmbodiSteer on nine robots. The latter enables six
methods by default: EE, joint no-guidance, joint GD, EmbodiSteer, post-hoc CBF
and batch sampling. EE-GD and JM2D remain as commented opt-in profiles because
they are substantially more expensive. Replace the template checkpoint path
before a real evaluation.

The eval stage accepts a trained workflow checkpoint by default, or an
explicit external checkpoint and output location:

```bash
./run_sim_pipeline.sh \
  --config configs/workflows/make_iced_coffee/evaluation_all_methods.yaml \
  --stage eval \
  --checkpoint /path/to/model.ckpt \
  --output-dir data/outputs/evaluation \
  --run-id coffee_baselines \
  --robots panda_robotiq_wristcam ur5_robotiq_wristcam
```

The compact eval-only template runs EE, joint no-guidance, joint GD,
EmbodiSteer, post-hoc CBF and batch sampling. Uncomment EE-GD or JM2D when
those optional comparisons are needed. A profile can be either a policy YAML
path or a mapping with `policy_config` plus per-profile runtime overrides such
as `robots`, `obstacle`, `control_mode`, render mode and episode count. All
enabled profiles are parsed and validated before the first GPU job.

Results are isolated as
`<output_dir>/<run_id>/profiles/<profile>/<robot>/`. Each robot stores the
legacy `eval_results.txt`, structured `metrics.json`, episode-level
`episode_metrics.npz` and videos. The run directory additionally contains a
configuration manifest (`run_manifest.yaml`), `results.json` and `results.md`
with robot macro averages and correctly pooled episode statistics. JM2D IK
rates are accumulated over all evaluation inference calls when that optional
profile is enabled.

Use `--resume` to skip completed profile/robot jobs or `--force` to rerun them
in place. A failed job is recorded and, by default, the remaining matrix is
still evaluated; set `evaluation.continue_on_error: false` to stop at the first
failure. The paper protocol uses unseeded `env.reset()`, and manifests label
that reset protocol explicitly.

Select physical GPUs through `CUDA_VISIBLE_DEVICES`; the workflow passes the
environment through unchanged to collection, training and evaluation. For
example, prefix a command with `CUDA_VISIBLE_DEVICES=3`. The training config
remains `device: cuda:0` because CUDA exposes the first selected physical GPU
to the process as logical device zero.

The `validate` stage checks the converter's raw 7D action representation
(position, axis-angle, gripper), which `UmiDataset` deterministically lifts to
the 10D position/rotation-6D/gripper training representation.
