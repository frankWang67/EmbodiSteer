# Third-party dependencies

`manifest.yaml` is the source of truth for the two project forks required by
the simulation and whole-body guidance code. ManiSkill is pinned to the reviewed
commit recorded there; uncommitted edits in a local checkout are not release
inputs.

`assets.yaml` separately records the provenance, license evidence and
redistribution status of assets included in this repository or supplied by
those forks. Dependency code licenses must not be used as a substitute for the
asset-specific terms in that ledger.

Use `python scripts/bootstrap_third_party.py --check` to validate the manifest.
Cloning or installing dependencies is an explicit developer action and is not
performed by the source tree or by the smoke tests.
