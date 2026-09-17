#!/bin/bash
# Run the repository's GLM-5.3 UCCL reproducer on MI300A through the ROCm image.
# Starts with one node/four ranks; pass 2 only after the one-node test passes.
set -euo pipefail

NODES="${1:-1}"
GPUS_PER_NODE=4
NTASKS=$((NODES * GPUS_PER_NODE))
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH_ROOT="${ROCM_SCRATCH:-${SCRATCH:?set SCRATCH or ROCM_SCRATCH}}"
IMAGE="${ROCM_IMAGE:-${SCRATCH_ROOT}/img/vllm-rocm-alps7-uccl.sqsh}"

export HERE IMAGE
export NUM_SMS="${NUM_SMS:-20}"
export NUM_TOKENS="${NUM_TOKENS:-512}"
export ITERS="${ITERS:-1}"
export MASTER_PORT="${MASTER_PORT:-29512}"
export ENROOT_CACHE_PATH="${SCRATCH_ROOT}/tmp/enroot/cache"
export ENROOT_DATA_PATH="${SCRATCH_ROOT}/tmp/enroot/data"
export ENROOT_RUNTIME_PATH="${ENROOT_RUNTIME_PATH:-/tmp/enroot-runtime-${UID}}"
mkdir -p "${ENROOT_CACHE_PATH}" "${ENROOT_DATA_PATH}" "${ENROOT_RUNTIME_PATH}"
chmod 700 "${ENROOT_RUNTIME_PATH}"
export OCI_ANNOTATION_com__hooks__netstack__source=host
export OCI_ANNOTATION_com__hooks__cxi__enabled=true

srun --account="${SLURM_ACCOUNT:-a-csstaff}" --partition="${SLURM_PARTITION:-mi300}" \
  --nodes="${NODES}" --ntasks="${NTASKS}" --ntasks-per-node="${GPUS_PER_NODE}" \
  --gpus-per-node="${GPUS_PER_NODE}" --cpus-per-task="${CPUS_PER_TASK:-16}" \
  --time="${TIME:-00:30:00}" --exclusive \
  --network=disable_rdzv_get --mpi=pmix -l -u bash -c '
    export RANK="$SLURM_PROCID"
    export LOCAL_RANK="$SLURM_LOCALID"
    export WORLD_SIZE="$SLURM_NTASKS"
    export LOCAL_WORLD_SIZE="${SLURM_NTASKS_PER_NODE:-4}"
    export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
    exec enroot start --rw \
      -m /iopsstor:/iopsstor \
      -m "${HOME}:${HOME}" \
      "${IMAGE}" \
      env \
        FI_PROVIDER=cxi \
        UCCL_EP_TRANSPORT=cxi \
        RANK="$RANK" LOCAL_RANK="$LOCAL_RANK" WORLD_SIZE="$WORLD_SIZE" \
        LOCAL_WORLD_SIZE="$LOCAL_WORLD_SIZE" \
        MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$MASTER_PORT" \
        python "${HERE}/tools/repro_vllm_buffer.py" \
          --num-sms "$NUM_SMS" --num-tokens "$NUM_TOKENS" --iters "$ITERS"
  '
