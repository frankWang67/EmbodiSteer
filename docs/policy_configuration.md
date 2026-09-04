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

Ready-to-run ablation profiles are also provided for joint no-guidance,
Cartesian and joint GD, post-hoc CBF, batch sampling and JM2D.
[`configs/workflows/make_iced_coffee/evaluation_all_methods.yaml`](../configs/workflows/make_iced_coffee/evaluation_all_methods.yaml)
enables six profiles in one profile-by-robot matrix; EE-GD and JM2D are
included as commented opt-in entries. Reference-evaluation profiles are
grouped by task under `configs/policy/make_iced_coffee/`,
`configs/policy/turn_faucet/`, and `configs/policy/place_toast/`; they preserve
each task's effective guidance and baseline parameters. Files directly under
`configs/policy/` are generic templates and must not be substituted silently
in a paper comparison.

## Configuration boundary

Algorithm settings belong in the policy YAML:

- Cartesian or joint inference;
- the number of diffusion inference steps;
- CBF, gradient, or no guidance;
- guidance scale and schedule shape, safety margin, gradient clamp,
  end-effector collision points, SDF reduction and task weights;
- CBF settings;
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
particular, CBF is not accepted for Cartesian inference, and a baseline cannot
be combined with per-step guidance. Real-world evaluation currently rejects
baseline profiles; baseline evaluation is a simulation-only path.

The checked-in paper profile (joint-space CBF) and Cartesian `gd` guidance use
the same validated fields. The three launchers and the timing/JM2D diagnostics
all read `num_inference_steps` and guidance hyperparameters from the selected
policy YAML; they do not overwrite them with launcher-specific defaults.

To create an ablation, copy a checked-in profile, change the relevant YAML
fields, and preserve the exact file with the experiment results. This records
all EmbodiSteer hyperparameters without reconstructing a long shell command.
