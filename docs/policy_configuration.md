# Policy configuration

The three public evaluation entry points share one versioned YAML schema:

- `eval_sim_single_robot.py` evaluates one ManiSkill robot;
- `eval_sim_multi_robots.py` invokes the single-robot entry point for each
  selected robot; and
- `eval_real.py` deploys the same policy settings on configured hardware.

Use [`configs/policy/embodisteer.yaml`](../configs/policy/embodisteer.yaml) for
the paper method and [`configs/policy/ee.yaml`](../configs/policy/ee.yaml) for
the Cartesian policy. Relative config paths are resolved from the repository
root by the shared loader.

## Configuration boundary

Algorithm settings belong in the policy YAML:

- Cartesian or joint inference;
- CBF, gradient, or no guidance;
- guidance schedule, safety margin, SDF reduction and task weights;
- standard or reverse CBF settings;
- Jacobian damping, initialization noise and optional IK refinement; and
- post-hoc CBF, batch-sampling and JM2D baseline settings.

Operational values remain command-line arguments because they identify a
particular run or machine: checkpoint and output paths, environment and robot
IDs, simulator backend, episode counts, camera options, device addresses and
obstacle-layout paths.

## Example

```console
python eval_sim_single_robot.py \
  --input /private/checkpoints/experiment \
  --ckpt_filename latest \
  --env_id PickPlaceToasterToCounter-v1 \
  --robot_uids panda_robotiq_wristcam \
  --control_mode pd_joint_pos \
  --obstacle \
  --policy-config configs/policy/embodisteer.yaml
```

The multi-robot launcher accepts the same `--policy-config` and passes it
unchanged to every child evaluation. The physical launcher also consumes the
same file:

```console
python eval_real.py --dry_run \
  --robot_config configs/real/robot.template.yaml \
  --obstacle_config configs/real/obstacles/door_frame.yaml \
  --policy-config configs/policy/embodisteer.yaml
```

The shared loader rejects unknown fields and inconsistent combinations. In
particular, CBF is not accepted for Cartesian inference, reverse CBF requires
joint-space CBF without a baseline, and a baseline cannot be combined with
per-step guidance. Real-world evaluation currently rejects baseline profiles;
baseline evaluation is a simulation-only path.

The checked-in paper profile (joint-space CBF) is fully wired to the unified
policy. Cartesian `gd` guidance remains a compatibility path: its corner-point
geometry, gradient clamp and logistic schedule are still implementation
defaults rather than independently configurable YAML fields. These values are
listed in the internal phase-3 audit and must be either wired into the schema or
explicitly removed from the supported profile before publication.

To create an ablation, copy a checked-in profile, change the relevant YAML
fields, and preserve the exact file with the experiment results. This records
all EmbodiSteer hyperparameters without reconstructing a long shell command.
