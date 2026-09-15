#!/bin/bash
# Launch the UCCL internode test on N nodes, one task per GPU.
#   ./scripts/run_uccl_internode_test.sh [nodes]    (default 2)
set -euo pipefail

NODES="${1:-2}"
GPUS_PER_NODE=4
NTASKS=$((NODES * GPUS_PER_NODE))
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# The launcher resolves the rendezvous host; every rank needs the same value.
# Allocate first so the node list is known, then use the first node.
export MASTER_PORT="${MASTER_PORT:-29500}"

srun --account=infra01 --partition=normal --reservation=SD-69241-apertus-1-5-0 \
  --nodes="${NODES}" --ntasks="${NTASKS}" --ntasks-per-node="${GPUS_PER_NODE}" \
  --gpus-per-node="${GPUS_PER_NODE}" --cpus-per-task=72 \
  --time="${TIME:-00:30:00}" --exclusive \
  --network=disable_rdzv_get --mpi=pmix -l \
  --environment="${HERE}/edf/vllm.toml" \
  --container-mounts="${SCRATCH}:${SCRATCH},${HOME}:${HOME},/capstor:/capstor,/iopsstor:/iopsstor" \
  --container-env=MASTER_ADDR,MASTER_PORT,NUM_TOKENS,CUDA_LAUNCH_BLOCKING \
  -u bash -c '
    # Rank 0'"'"'s node is the rendezvous host; derive it identically on every rank.
    export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
    exec bash '"${HERE}"'/tools/uccl_internode_test.sh
  '
