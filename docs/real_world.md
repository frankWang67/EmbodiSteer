# Real-world deployment

The physical and simulation paths share the unified
`EmbodiSteerJointPolicy`. The supported entry point is:

```console
python eval_real.py \
  --input /private/checkpoints/experiment \
  --output /private/episodes \
  --robot_config /private/configs/robot.yaml \
  --obstacle_config /private/configs/obstacles.yaml \
  --policy-config configs/policy/embodisteer.yaml
```

The policy YAML selects Cartesian or joint-space inference, guidance, CBF/SDF
settings and IK refinement. The paper profile selects joint-space CBF
guidance. Compatible checkpoint Hydra targets are normalized to the public
`EmbodiSteerJointPolicy` alias at runtime.

## Configuration boundary

Start from [`configs/real/robot.template.yaml`](../configs/real/robot.template.yaml)
and keep the populated copy outside version control. It must contain one
robot/gripper pair per entry, a 4x4 `tx_left_right` transform, and explicit
addresses. Obstacle geometry is selected separately with
[`configs/real/obstacles/door_frame.yaml`](../configs/real/obstacles/door_frame.yaml)
or a private YAML in the robot base frame.

Validate before connecting to hardware:

```console
python eval_real.py --dry_run \
  --robot_config /private/configs/robot.yaml \
  --obstacle_config /private/configs/obstacles.yaml \
  --policy-config configs/policy/embodisteer.yaml
```

The dry run does not load a checkpoint, import a device connection, move a
robot or start a camera. A physical run still requires an emergency-stop
operator, controller/camera checks, collision-scene review and a low-speed
bring-up. The release code cannot establish those safety guarantees for a
particular lab setup.
