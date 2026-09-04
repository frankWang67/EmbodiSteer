# Architecture and compatibility boundary

The repository contains two Python namespaces with different responsibilities:

| namespace | responsibility |
| --- | --- |
| `embodisteer` | project-owned policies, kinematics helpers, SDF reductions, CBF solvers and real-world configuration adapters |
| `diffusion_policy` | Diffusion Policy training and model-building engine |

`diffusion_policy` is a directory inside this repository and is imported from
the repository itself. EmbodiSteer additions are deliberately not placed in
this namespace, preserving its public model and checkpoint interfaces.

The unified joint policy is implemented at
`embodisteer.policies.ee2joint.EmbodiSteerJointPolicy`; new launchers target
this stable alias. Its implementation calls the public modules below:

- `embodisteer.kinematics`: pose conversion, SE(3) residuals and damped
  Jacobian pseudoinverse;
- `embodisteer.collision`: cuRobo ESDF sign conversion, top-k aggregation and
  safety-margin penalties; and
- `embodisteer.guidance`: guidance schedules and closed-form CBF-QP solvers.

Rotation and pose-representation helpers remain in `diffusion_policy.common`.
EmbodiSteer-specific rotation conversions and guided-diffusion loss helpers are
namespaced under `embodisteer.kinematics` and `embodisteer.guidance`.

Only ManiSkill and cuRobo are external repositories. Their exact URLs and
commits are recorded in `third_party/manifest.yaml`.

Checkpoint interoperability covers the standard Diffusion Policy payload
format: launchers reuse its workspace and state dictionaries while selecting
an explicit public EmbodiSteer policy target from the policy YAML. Serialized
research snapshots and retired paper-era policy classes are not part of the
release compatibility contract.

For the algorithmic loop, sign conventions and comparison methods, see
[`method.md`](method.md).
