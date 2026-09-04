# Changelog

This file records user-visible release changes. The project is currently in a
pre-publication phase, so no versioned release has been tagged yet.

## Unreleased

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
