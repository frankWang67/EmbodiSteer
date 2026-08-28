# Contributing to EmbodiSteer

Contributions are welcome through issues and pull requests once the public
repository URL is assigned. Please describe the motivation, affected policy or
adapter, and the validation you ran.

## Scope and compatibility

- Put new EmbodiSteer implementation in `embodisteer/`.
- Keep the public `diffusion_policy/` model and checkpoint interfaces stable;
  project-specific policies must not be added to that namespace.
- Do not commit checkpoints, training recordings, credentials, private robot
  addresses, or site-specific calibration files.
- Update `THIRD_PARTY_NOTICES.md` and `third_party/assets.yaml` when adding or
  redistributing an asset, mesh, URDF, texture, or external source file.
- Keep dependency revisions and licensing metadata in
  `third_party/manifest.yaml`.

## Before opening a pull request

Run the checks that are available in your environment:

```console
python scripts/diagnostics/check_release_layout.py
python -m compileall -q embodisteer diffusion_policy umi scripts_maniskill scripts_slam_pipeline eval_real.py
python -m pytest -q
python scripts/bootstrap_third_party.py --check
```

If CUDA, ManiSkill, cuRobo, or hardware is unavailable, state which checks
were skipped and include the relevant CPU/layout output. Real-robot tests must
be performed at low speed with an approved emergency-stop procedure and must
not include identifying lab configuration in logs or patches.

## Pull requests

Keep commits focused and include documentation for new public options. A
reviewer should be able to reproduce the claimed behavior from the checked-in
configuration and tests without access to private paths. Changes that alter
the public policy API or checkpoint compatibility boundary require an explicit
design note in the pull request.
