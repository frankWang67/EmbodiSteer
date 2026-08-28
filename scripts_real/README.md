# Real-device code boundary

The supported physical deployment entry point is the repository-level
[`eval_real.py`](../eval_real.py). It loads the same
`embodisteer.policies.EmbodiSteerJointPolicy` used by the ManiSkill launcher,
validates `configs/real/robot.template.yaml`-style configuration, and offers a
`--dry_run` mode that performs no checkpoint or device I/O.

The controller, camera, keyboard and gripper implementations needed by that
entry point live in `umi/real_world/`. This directory intentionally contains
no site-specific executable scripts: the copied UMI checkout had several
overlapping demos/replay prototypes with private network defaults and no
consistent safety boundary. They were excluded from the isolated release
tree; the algorithm and hardware adapters remain available in the public
package.

Before any physical run, copy the template to a private config, replace all
placeholder addresses, verify the robot model and obstacle frame, and review
the emergency-stop procedure. Never commit that private config or credentials.
