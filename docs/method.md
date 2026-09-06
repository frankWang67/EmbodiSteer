# Method

EmbodiSteer deploys a frozen Cartesian visuomotor diffusion policy on a target
robot without robot-specific policy fine-tuning. The learned model consumes and
predicts a relative Cartesian action representation. At inference time,
EmbodiSteer carries the denoising trajectory in the target robot's joint space,
projects every diffusion update through the robot's kinematics, and applies
whole-body CBF collision guidance.

This document describes the public implementation. The main entry point is
[`DiffusionUnetTimmPolicyEmbodiSteer`](../embodisteer/policies/embodisteer.py)
(also exported as `embodisteer.policies.EmbodiSteerJointPolicy`); the local
`diffusion_policy/` namespace supplies the model, normalizer, scheduler, and
data/training components.

## Joint-space denoising

Let $q^k$ be the joint trajectory at reverse-diffusion iteration $k$. Each
iteration performs the following operations:

1. Forward kinematics maps every state in $q^k$ to an absolute end-effector
   position and orientation.
2. The absolute pose is expressed relative to the current action chunk's
   starting pose. Rotation is represented in 6D, yielding the policy's
   position-plus-rotation pose-9D representation; the gripper channel is
   appended and the action is normalized.
3. The frozen U-Net predicts the Cartesian diffusion output. The configured
   scheduler produces the next Cartesian target.
4. The target is converted back to an absolute pose. EmbodiSteer computes a
   six-dimensional world-frame residual $\Delta x$ from the current pose to
   that target.
5. A damped Jacobian pseudoinverse maps the task-space residual to a joint
   update:

   $$
   \Delta q = J^{\top}\left(J J^{\top} + \lambda I\right)^{-1}\Delta x.
   $$

6. The update is clipped when configured and applied to the joint trajectory.

The initial trajectory uses Jacobian-projected task noise around the chunk-start
configuration, with a global scale and per-joint update clip.
The final result is converted through FK to the normal Cartesian action API,
while the joint trajectory is exposed for joint-position control.

## Whole-body distance model

cuRobo represents the robot with link collision spheres and queries them
against an SDF world. For the query used here, cuRobo's signed distance
$d_{\mathrm{cuRobo}}$ is positive inside an obstacle and negative outside it. EmbodiSteer
therefore defines the CBF safety function as

$$
h(q) = -d_{\mathrm{cuRobo}}(q),
$$

so larger $h$ is safer. Per-sphere distances are reduced with either a strict
maximum or a normalized smooth top-k aggregation. The top-k implementation
preserves an actual penetration exactly instead of averaging it away. The
current public obstacle adapter accepts oriented cuboids; extending it to
other cuRobo world primitives requires an explicit adapter change.

## Collision guidance

The paper class uses `guidance_method="cbf"` exclusively at construction.
Its reverse-diffusion loop and `_apply_cbf_guidance` are in `embodisteer.py`.
The shared robot runtime supplies collision queries; the closed-form QP
implementation is in [`guidance/cbf.py`](../embodisteer/guidance/cbf.py).

For CBF guidance, let $m$ be the requested safety margin and let $\gamma_k$
be the step-dependent guidance scale. The implementation uses

$$
\begin{aligned}
\underset{\Delta q}{\operatorname{minimize}}\quad
    & \frac{1}{2}\Delta q^{\top} H\Delta q \\
\text{subject to}\quad
    & a^{\top}\Delta q \ge r.
\end{aligned}
$$

with

$$
\begin{aligned}
a &= \nabla h(q), \\
H &= J^{\top} W J + \lambda I, \\
r &= \max\!\left(\gamma_k\left(m - h(q)\right), 0\right).
\end{aligned}
$$

Because each trajectory state has one aggregated collision inequality, the
minimum-task-disturbance update has a closed form. Position and rotation can
receive different weights through $W$; regularization keeps $H$ invertible.

With scheduling enabled, the guidance multiplier follows a logistic curve and
becomes strongest near the end of denoising. This leaves early sampling more
model-driven and concentrates collision correction as the action trajectory
becomes cleaner.

## Comparison methods included in the code

The release keeps the following comparison paths under `embodisteer/policies`:

- `ee2joint.DiffusionUnetTimmPolicyJointSpace`: joint-space denoising with
  `guidance_method=""` (no guidance) or `"gd"` (gradient descent). GD is not
  part of EmbodiSteer. It minimizes

  $$
  \operatorname{ReLU}\!\left(d_{\mathrm{cuRobo}} + m\right)^p,
  $$

  masks and clips the gradient, then subtracts the scheduled scale times
  that clipped gradient;
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
| `embodisteer/policies/embodisteer.py` | paper method: Cartesian denoising, Jacobian realization, CBF correction |
| `embodisteer/policies/ee2joint.py` | no-guidance/GD comparisons and shared robot runtime |
| `embodisteer/policies/baselines.py`, `jm2d.py`, `ee_space.py` | other comparison policies |
| `embodisteer/kinematics` | pose conversion, SE(3) residuals, and damped Jacobian mapping |
| `embodisteer/collision` | cuRobo sign conversion, SDF aggregation, and penalties |
| `embodisteer/guidance` | guidance schedules and closed-form CBF solvers |
| `diffusion_policy` | Diffusion Policy model, dataset, and training base |

Simulation and physical deployment import the same public policy package.
ManiSkill-specific environment extraction lives in `scripts_maniskill/`, while
real-device configuration validation and obstacle conversion live in
`embodisteer/adapters/` and `eval_real.py`. Robot drivers and launch safety are
kept outside the method modules.
