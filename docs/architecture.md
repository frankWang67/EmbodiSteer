# Architecture and compatibility boundary

The repository contains two Python namespaces with different responsibilities:

| namespace | responsibility |
| --- | --- |
| `embodisteer` | project-owned policies, kinematics helpers, SDF reductions, CBF solvers and real-world configuration adapters |
| `diffusion_policy` | Diffusion Policy training and model-building engine |

`diffusion_policy` is a directory inside this repository and is imported from
the repository itself. EmbodiSteer additions are deliberately not placed in
this namespace, preserving its public model and checkpoint interfaces.

The paper method is implemented in
[`embodisteer/policies/embodisteer.py`](../embodisteer/policies/embodisteer.py),
as `DiffusionUnetTimmPolicyEmbodiSteer`. Its `conditional_sample` shows the
Cartesian denoising, Jacobian realization and per-step CBF correction;
`_apply_cbf_guidance` shows the correction's linearization, schedule and clip.

`ee2joint.py` retains `DiffusionUnetTimmPolicyJointSpace` for no-guidance and
GD comparisons. The two are sibling classes sharing the private
`_JointSpacePolicyRuntime` in `ee2joint.py`, which owns robot resources,
FK/IK/Jacobian adapters, collision queries, initialization, and observation/action
I/O. Post-hoc baselines also reuse this runtime, not the paper sampler.
No learned modules are nested under a new attribute, so checkpoint state-dict
keys remain unchanged.

`runtime_config.policy_target` selects the concrete class from the policy YAML
for simulation, real deployment and benchmarks. The public
`embodisteer.policies.EmbodiSteerJointPolicy` alias now denotes only the paper's
CBF class. The historical alias in `ee2joint` forwards to the same class;
direct GD/no-guidance callers must use `DiffusionUnetTimmPolicyJointSpace`.
Direct callers of that class with `guidance_method="cbf"` must migrate to the
new class. Launchers perform this selection automatically, including for old
checkpoint configurations.

The policies call these public modules:

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
