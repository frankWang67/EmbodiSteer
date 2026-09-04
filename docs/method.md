# Method

EmbodiSteer deploys a frozen Cartesian visuomotor diffusion policy on a target
robot without robot-specific policy fine-tuning. The learned model consumes and
predicts a relative Cartesian action representation. At inference time,
EmbodiSteer carries the denoising trajectory in the target robot's joint space,
projects every diffusion update through the robot's kinematics, and can add
whole-body collision guidance.

This document describes the public implementation. The main entry point is
`embodisteer.policies.EmbodiSteerJointPolicy`; the local
`diffusion_policy/` namespace supplies the model, normalizer, scheduler, and
data/training components.

## Joint-space denoising

Let `q^k` be the joint trajectory at reverse-diffusion iteration `k`. Each
iteration performs the following operations:

1. Forward kinematics maps every state in `q^k` to an absolute end-effector
   position and orientation.
2. The absolute pose is expressed relative to the current action chunk's
   starting pose. Rotation is represented in 6D, yielding the policy's
   position-plus-rotation pose-9D representation; the gripper channel is
   appended and the action is normalized.
3. The frozen U-Net predicts the Cartesian diffusion output. The configured
   scheduler produces the next Cartesian target.
4. The target is converted back to an absolute pose. EmbodiSteer computes a
   six-dimensional world-frame residual `Delta x` from the current pose to
   that target.
5. A damped Jacobian pseudoinverse maps the task-space residual to a joint
   update:

   ```text
   Delta q = J^T (J J^T + lambda I)^(-1) Delta x.
   ```

6. The update is clipped when configured and applied to the joint trajectory.

The initial trajectory uses Jacobian-projected task noise around the chunk-start
configuration, with a global scale and per-joint update clip.
The final result is converted through FK to the normal Cartesian action API,
while the joint trajectory is exposed for joint-position control.

## Whole-body distance model

cuRobo represents the robot with link collision spheres and queries them
against an SDF world. For the query used here, cuRobo's signed distance
`d_curobo` is positive inside an obstacle and negative outside it. EmbodiSteer
therefore defines the CBF safety function as

```text
h(q) = -d_curobo(q),
```

so larger `h` is safer. Per-sphere distances are reduced with either a strict
maximum or a normalized smooth top-k aggregation. The top-k implementation
preserves an actual penetration exactly instead of averaging it away. The
current public obstacle adapter accepts oriented cuboids; extending it to
other cuRobo world primitives requires an explicit adapter change.

## Collision guidance

The unified policy accepts three `guidance_method` values:

- `""`: joint-space denoising without collision guidance;
- `"cbf"`: a closed-form, batched CBF correction; and
- `"gd"`: gradient descent on a signed-distance safety-margin penalty.

For CBF guidance, let `a = grad h(q)`, let `m` be the requested safety margin,
and let `gamma_k` be the step-dependent guidance scale. The implementation
uses

```text
H = J^T W J + lambda I
r = max(gamma_k (m - h), 0)

minimize    1/2 Delta q^T H Delta q
subject to  a^T Delta q >= r.
```

Because each trajectory state has one aggregated collision inequality, the
minimum-task-disturbance update has a closed form. Position and rotation can
receive different weights through `W`; regularization keeps `H` invertible.

Gradient guidance instead minimizes
the hinge penalty `relu(d_curobo + m)^p` directly in joint space. Both modes
apply one correction per denoising step and clamp the resulting joint update.

With scheduling enabled, the guidance multiplier follows a logistic curve and
becomes strongest near the end of denoising. This leaves early sampling more
model-driven and concentrates collision correction as the action trajectory
becomes cleaner.

## Comparison methods included in the code

The release keeps the following comparison paths under `embodisteer/policies`:

- Cartesian EE sampling, with optional end-effector-only corner guidance;
- post-hoc CBF, which denoises in Cartesian space and applies one joint-space
  correction after IK;
- batch sampling, which realizes multiple Cartesian candidates with IK and
  selects the safest candidate by whole-body distance;
- JM2D conditional generation, which importance-weights clean Cartesian
  candidates using joint-realized collision energy and applies a final
  one-shot CBF correction.

## Package and deployment boundary

The implementation is split by responsibility:

| path | responsibility |
| --- | --- |
| `embodisteer/policies` | unified method and comparison policies |
| `embodisteer/kinematics` | pose conversion, SE(3) residuals, and damped Jacobian mapping |
| `embodisteer/collision` | cuRobo sign conversion, SDF aggregation, and penalties |
| `embodisteer/guidance` | guidance schedules and closed-form CBF solvers |
| `diffusion_policy` | Diffusion Policy model, dataset, and training base |

Simulation and physical deployment import the same public policy package.
ManiSkill-specific environment extraction lives in `scripts_maniskill/`, while
real-device configuration validation and obstacle conversion live in
`embodisteer/adapters/` and `eval_real.py`. Robot drivers and launch safety are
kept outside the method modules.

## Current limitations

- Checkpoints and training data are not included in this phase; their status
  is recorded in `artifacts/manifest.yaml`.
- ManiSkill and cuRobo are installed from fixed external fork revisions. Their
  code and asset terms are not replaced by the repository's MIT license.
- Whole-body guidance depends on the accuracy of robot collision spheres,
  kinematics, obstacle poses, and controller tracking.
- The public obstacle adapter currently builds cuboid world geometry.
- The software does not establish real-robot safety. Hardware deployment
  requires independent limits, workspace review, low-speed commissioning, and
  an approved emergency-stop procedure.
