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
