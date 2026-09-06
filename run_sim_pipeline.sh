#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# Task settings live in the workflow YAML selected with --config. Every CLI
# argument is handled by run_sim_workflow.py and is forwarded unchanged here.
EMBODISTEER_CONDA_ENV="${EMBODISTEER_CONDA_ENV:-embodisteer-sim}"
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"

exec conda run --no-capture-output -n "${EMBODISTEER_CONDA_ENV}" \
    python -u run_sim_workflow.py "$@"
