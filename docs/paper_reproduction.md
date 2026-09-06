# Paper reproduction

This release provides stable policy imports and an executable simulation
workflow. The experiment manifest in
[`configs/experiments/paper.yaml`](../configs/experiments/paper.yaml) records
the three tasks, nine robot UIDs, method settings and metric semantics:
`Fail@All` is the per-episode failure indicator, while collision count is the
number of colliding simulation steps in an episode.

The reproduction runbook uses the config-driven simulation workflow in
[`configs/workflows/simulation.yaml`](../configs/workflows/simulation.yaml).
After the pinned dependency checkouts are materialized:

1. install and record the exact three-repository revisions;
2. run a CPU/layout smoke test, then a small GPU evaluation;
3. generate demos, convert them to UMI zarr, train a checkpoint, then run the
   EE baseline and EmbodiSteer variants using the paper's unseeded environment
   reset protocol;
4. collect collision and success metrics using the manifest semantics;
5. validate the physical `--dry_run` and then a reviewed low-speed hardware
   trial; and
6. publish a reproduction report with software, CUDA, GPU and asset versions.

No real robot is started by the simulation workflow or its dry-run checks.
