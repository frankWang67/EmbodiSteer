#!/usr/bin/env bash

set -euo pipefail

# Convenience wrapper around the versioned, resumable workflow.  All task,
# collection, training and evaluation settings live in YAML rather than in
# positional shell arguments.
CONFIG_PATH="configs/workflows/simulation.yaml"
if [[ $# -gt 0 && ${1} != -* ]]; then
    CONFIG_PATH="${1}"
    shift
fi

exec python scripts_maniskill/run_sim_workflow.py \
    --config "${CONFIG_PATH}" \
    --stage all \
    "$@"
