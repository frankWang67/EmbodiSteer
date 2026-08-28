#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 TASK_NAME TRAJECTORY_COUNT DATE_TAG GPU_INDEX" >&2
    exit 2
fi

task_name=$1
traj_num=$2
date=$3
gpu_idx=$4

if ! [[ ${traj_num} =~ ^[1-9][0-9]*$ ]]; then
    echo "TRAJECTORY_COUNT must be a positive integer" >&2
    exit 2
fi
if (( traj_num % 5 != 0 )); then
    echo "TRAJECTORY_COUNT must be divisible by the five collection robots" >&2
    exit 2
fi

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MANISKILL_ROOT=${MANISKILL_ROOT:?Set MANISKILL_ROOT to the checked-out ManiSkill fork}
DATA_ROOT=${DATA_ROOT:-"${REPO_ROOT}/data"}

traj_num_per_robot=$((traj_num / 5))
num_procs=$((traj_num_per_robot / 4))
if (( num_procs < 1 )); then
    num_procs=1
fi

h5_file_name=merged_data_${traj_num}_${date}.h5
zarr_file_name=ManiSkill_${task_name}_${date}.zarr.zip

export CUDA_VISIBLE_DEVICES="${gpu_idx}"
mkdir -p "${DATA_ROOT}"

cd "${MANISKILL_ROOT}"
python multi_robot_data_collection.py \
    -e "${task_name}-v1" \
    -f "${h5_file_name}" \
    -n "${traj_num_per_robot}" \
    -c pd_ee_pose \
    --save-video \
    --num-procs "${num_procs}"
cd "${REPO_ROOT}"
python convert_hdf5_to_umi_zarr.py -i "${MANISKILL_ROOT}/demos/${task_name}-v1/motionplanning/${h5_file_name}" -o "${DATA_ROOT}/${zarr_file_name}"
python train.py \
    --config-dir="${REPO_ROOT}/configs/simulation/hydra" \
    --config-name=train_diffusion_unet_timm_maniskill_workspace \
    task.dataset_path="${DATA_ROOT}/${zarr_file_name}" \
    env_name="${task_name}"
