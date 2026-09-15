#!/bin/bash
# ./scripts/run_repro.sh <nodes> <num_sms>
set -euo pipefail
NODES="${1:-4}"; NUM_SMS="${2:-20}"; FP8="${FP8:-}"; ITERS="${ITERS:-1}"; NUM_TOKENS="${NUM_TOKENS:-512}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export NUM_SMS NUM_TOKENS FP8 ITERS STARVE
srun --account=infra01 --partition=normal --reservation=SD-69241-apertus-1-5-0 \
  --nodes="${NODES}" --ntasks=$((NODES*4)) --ntasks-per-node=4 --gpus-per-node=4 \
  --cpus-per-task=72 --time=00:20:00 --exclusive \
  --network=disable_rdzv_get --mpi=pmix -l \
  --environment="${HERE}/edf/vllm.toml" \
  --container-mounts="${SCRATCH}:${SCRATCH},${HOME}:${HOME},/capstor:/capstor,/iopsstor:/iopsstor" \
  --container-env=NUM_SMS,NUM_TOKENS,FP8,ITERS,STARVE \
  -u bash -c '
    export RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID WORLD_SIZE=$SLURM_NTASKS
    export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
    export MASTER_PORT=29512
    exec python3 '"${HERE}"'/tools/repro_vllm_buffer.py --num-sms "$NUM_SMS" --num-tokens "${NUM_TOKENS:-512}" ${FP8:+--fp8} ${STARVE:+--starve-ranks} --iters "${ITERS:-1}"
  '
