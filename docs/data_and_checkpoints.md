# Data and checkpoints

Checkpoint files, training datasets, camera recordings, wandb runs and large
experiment outputs are deliberately absent from this code release. Their
publication state is recorded in the machine-readable
[`artifacts/manifest.yaml`](../artifacts/manifest.yaml). An empty `artifacts`
list means that no download is currently offered; it is not a placeholder URL
or an implicit requirement to search the authors' machines.

To reproduce the paper, obtain the artifacts from a separately published,
access-controlled location and pass their paths explicitly:

- simulation: `eval_sim_single_robot.py --input ... --ckpt_filename ... --policy-config ...`
- physical deployment: `eval_real.py --input ... --output ...`
- data generation/conversion/training: edit
  [`configs/workflows/simulation.yaml`](../configs/workflows/simulation.yaml)
  and run `python scripts_maniskill/run_sim_workflow.py --stage all`; use
  `--stage collect`, `convert`, `validate`, or `train` to resume individual
  stages. Keep generated data and checkpoints under the ignored `data/` and
  `data/outputs/` paths (or configure equivalent external paths).

Do not commit credentials, private robot addresses, raw recordings or derived
checkpoints. The root `.gitignore` excludes common output directories, but a
release review must still inspect any new artifact before publishing.

## Assets are a separate category

The `diffusion_policy/` package includes benchmark meshes, URDFs and textures
for Block Pushing and Kitchen/Franka environments. Those files are code
dependencies, not training datasets or learned checkpoints. Their provenance
and license evidence are listed in
[`third_party/assets.yaml`](../third_party/assets.yaml).

ManiSkill and cuRobo assets are not copied into this repository. They are
materialized from the fixed dependency forks described in
[`third_party/manifest.yaml`](../third_party/manifest.yaml). Local robot
calibration files and camera recordings remain excluded.
