# Paper reproduction

The phase-2 release establishes the code boundary and stable policy imports;
it does not claim a completed paper reproduction. The experiment manifest in
[`configs/experiments/paper.yaml`](../configs/experiments/paper.yaml) records
the three tasks, nine robot UIDs, method settings and metric semantics:
`Fail@All` is the per-episode failure indicator, while collision count is the
number of colliding simulation steps in an episode.

The phase-3 runbook will, after the checkpoint/data location is decided and
the pinned dependency checkouts are materialized:

1. install and record the exact three-repository revisions;
2. run a CPU/layout smoke test, then a small GPU evaluation;
3. run the EE baseline and EmbodiSteer variants with fixed seeds;
4. collect collision and success metrics using the manifest semantics;
5. validate the physical `--dry_run` and then a reviewed low-speed hardware
   trial; and
6. publish a reproduction report with software, CUDA, GPU and asset versions.

No real robot is started by the phase-2 checks.
