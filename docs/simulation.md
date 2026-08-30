# Simulation

The simulation launchers are
[`eval_sim_single_robot.py`](../eval_sim_single_robot.py) and
[`eval_sim_multi_robots.py`](../eval_sim_multi_robots.py). They use the same
public policy aliases and algorithm config as the physical path:

```python
from embodisteer.policies import EmbodiSteerJointPolicy
```

Method selection now lives in a policy YAML:

| policy YAML fields | method |
| --- | --- |
| `inference_space: ee`, `guidance.method: ""` | Cartesian EE baseline |
| `inference_space: joint`, `guidance.method: ""` | unified EE-to-joint EmbodiSteer without collision guidance |
| `inference_space: joint`, `guidance.method: cbf` | EmbodiSteer with joint-space CBF-QP guidance |
| `inference_space: joint`, `guidance.method: gd` | gradient guidance ablation |

The task and robot identifiers used in the paper are listed in
[`configs/experiments/paper.yaml`](../configs/experiments/paper.yaml). The
ManiSkill fork supplies the task definitions and robot assets; cuRobo supplies
the robot descriptions, collision spheres and SDF interface. Install both
before launching an evaluation.

Algorithm defaults are documented in
[`configs/policy/embodisteer.yaml`](../configs/policy/embodisteer.yaml). Pass a
profile with `--policy-config`; use
[`configs/policy/ee.yaml`](../configs/policy/ee.yaml) for the Cartesian
baseline. An evaluation requires a compatible checkpoint, which is
intentionally not included in this repository. Operational arguments are
listed by `python eval_sim_single_robot.py --help` and
`python eval_sim_multi_robots.py --help`.

The ManiSkill training config lives at `configs/simulation/hydra/`. The
collection/training launcher passes that path explicitly, while shared
Diffusion Policy training configs remain under `diffusion_policy/config/`.

## End-to-end workflow

Data generation, conversion, training and simulation evaluation are exposed as
independent stages of one config-driven launcher:

```console
python scripts_maniskill/run_sim_workflow.py \
  --config configs/workflows/simulation.yaml --stage all --dry-run
```

Remove `--dry-run` to execute the stages in order. The default workflow uses
the pinned ManiSkill checkout, writes raw demos and the converted `*.zarr.zip`
dataset under ignored `data/` paths, trains into ignored `data/outputs/`, and
evaluates the resulting `checkpoints/latest.ckpt`. Run a single stage with
`--stage collect|convert|validate|train|eval` when resuming after an
interruption. Edit the YAML to choose the task, collection count, GPU,
training overrides, policy profiles and robot list; the checked-in workflow
evaluates both the EE baseline and EmbodiSteer. Do not put datasets or
checkpoints under tracked source directories.

The `validate` stage checks the converter's raw 7D action representation
(position, axis-angle, gripper), which `UmiDataset` deterministically lifts to
the 10D position/rotation-6D/gripper training representation.
