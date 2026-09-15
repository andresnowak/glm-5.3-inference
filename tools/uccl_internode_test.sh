#!/bin/bash
# Runs inside the container, once per GPU. Drives the image's own copy of
# DeepEP's test_internode.py through UCCL, so it exercises exactly the code
# vLLM will call (deep_ep.Buffer.dispatch/combine + the low-latency kernels).
#
# Slurm provides the rank layout (one task per GPU), so no torchrun. The test
# reads torchrun's variable names, so map them from Slurm's.
set -euo pipefail

export RANK="${SLURM_PROCID}"
export LOCAL_RANK="${SLURM_LOCALID}"
export WORLD_SIZE="${SLURM_NTASKS}"
export LOCAL_WORLD_SIZE="${SLURM_NTASKS_PER_NODE:-4}"
export MASTER_ADDR="${MASTER_ADDR:?set by the launcher}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
# CUDA_LAUNCH_BLOCKING=1 makes launches synchronous so an async kernel fault is
# reported at the launch that caused it, instead of surfacing later as a sticky
# error on an unrelated runtime call (which is how the vLLM run presented: an
# assert on cudaFuncSetAttribute for a smem size the device actually allows).
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"

DEEP_EP_DIR=/usr/local/lib/python3.12/dist-packages/deep_ep

if [[ "${RANK}" == "0" ]]; then
  echo "=== UCCL internode test ==="
  echo "  world=${WORLD_SIZE} local=${LOCAL_WORLD_SIZE} master=${MASTER_ADDR}:${MASTER_PORT}"
  echo "  UCCL_EP_TRANSPORT=${UCCL_EP_TRANSPORT:-<unset>}"
  echo "  deep_ep from ${DEEP_EP_DIR}"
fi

cd "${DEEP_EP_DIR}"
# test_internode.py does a plain `from utils import ...`, so its own directory
# must be importable.
export PYTHONPATH="${DEEP_EP_DIR}:${PYTHONPATH:-}"

# Shapes match GLM-5.3: hidden 6144, 256 routed experts, top-8.
exec python3 test_internode.py \
  --num-tokens="${NUM_TOKENS:-4096}" \
  --hidden=6144 \
  --num-topk=8 \
  --num-experts=256 \
  --test-ll-compatibility
