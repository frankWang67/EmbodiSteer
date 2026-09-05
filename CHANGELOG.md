# Changelog

This file records user-visible release changes. The project is currently in a
pre-publication phase, so no versioned release has been tagged yet.

## Unreleased

- Moved the paper's denoising loop and CBF correction into
  `policies/embodisteer.py:DiffusionUnetTimmPolicyEmbodiSteer`. Joint-space
  GD/no-guidance remains in `ee2joint.py:DiffusionUnetTimmPolicyJointSpace`;
  sibling policies share robot resources and checkpoint-compatible I/O.
  Evaluation and benchmark targets now use one method-aware config router.
  The `EmbodiSteerJointPolicy` alias denotes the CBF class only; direct
  GD/no-guidance callers must use `DiffusionUnetTimmPolicyJointSpace`.

- Removed the retired policy implementation tree. Checkpoint interoperability
  is provided by loading Diffusion Policy weights into the stable public
  policy selected by the evaluation profile, rather than by preserving
  experimental inference classes.
- Removed inactive experimental inference branches and their configuration
  fields, leaving the paper's Jacobian-projected initialization, geometric
  residual update, and one guidance correction per denoising step as the
  canonical EmbodiSteer path. Evaluation now records the unseeded reset
  protocol without exposing an ineffective environment-seed argument.
- Promoted the single- and multi-robot simulation evaluators to root-level
  entry points and moved shared EmbodiSteer algorithm settings from CLI flags
  into versioned policy YAML files used by simulation and real-world runs.
- Published the supported runtime pins (including NumPy 1.26.4, SciPy 1.15.3,
  Numba 0.65.0, OpenCV 4.7.0.72 and pytorch-kinematics 0.10.0) and documented
  the main/fork/optional dependency boundary.
- Isolated the EmbodiSteer implementation under `embodisteer/` while keeping
  the `diffusion_policy/` API stable for checkpoint compatibility.
- Added shared kinematics, cuRobo SDF reductions, CBF-QP guidance,
  and public simulation/real-world policy entry points.
- Pinned the reviewed ManiSkill and cuRobo fork revisions in
  `third_party/manifest.yaml`.
- Documented the method, publication artifact status, and redistributed asset
  provenance.

Checkpoint files, training data, and benchmark results are intentionally not
part of this phase. Future entries should include a release date, migration
notes, and links to the corresponding artifact manifest entry.
