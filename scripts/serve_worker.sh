#!/bin/bash
# Runs inside the container, once per node (vLLM owns the 4 GPUs on its node).
#
# Topology: 4 nodes x 4 GH200 = 16 GPUs
#   --tensor-parallel-size 2      attention/embeddings sharded by 2
#   --data-parallel-size 8        8 DP ranks total, 2 per node (2 ranks x TP2 = 4 GPUs)
#   --enable-expert-parallel      => EP = DP x TP = 16, so 256/16 = 16 experts per GPU
#
# Node 0 runs the OpenAI API server; nodes 1..3 run --headless workers that join it.
set -uo pipefail

MODEL="${MODEL:?}"
HEAD_IP="${HEAD_IP:?}"
DP_RPC_PORT="${DP_RPC_PORT:-13345}"
API_PORT="${API_PORT:-8000}"
ALL2ALL="${ALL2ALL:-deepep_high_throughput}"
LOGDIR="${LOGDIR:?}"

NODE_RANK="${SLURM_NODEID}"
# Derive the DP layout from the actual allocation so the node count is a real knob:
# 4 GPUs per node / TP=2 => 2 DP ranks per node, and EP = DP x TP = nodes x 4.
TP_SIZE="${TP_SIZE:-2}"
GPUS_PER_NODE=4
DP_LOCAL=$(( GPUS_PER_NODE / TP_SIZE ))
DP_TOTAL=$(( SLURM_JOB_NUM_NODES * DP_LOCAL ))
START_RANK=$(( NODE_RANK * DP_LOCAL ))

# Weights are already in the scratch HF cache; never reach for the network.
# RUNTIME_PATCHES=<dir> applies container/patches/*.py to site-packages at startup.
# They are pure-Python edits and the EDF is writable, so iterating on a patch does not
# need a 20-minute image rebuild. Each script is idempotent.
if [[ -n "${RUNTIME_PATCHES:-}" ]]; then
  SITE=$(python3 -c 'import site; print(site.getsitepackages()[0])')
  echo "[node ${SLURM_NODEID}] applying runtime patches from ${RUNTIME_PATCHES}"
  python3 "${RUNTIME_PATCHES}/add_enable_shrink.py"          "${SITE}/deep_ep/buffer.py" || true
  python3 "${RUNTIME_PATCHES}/fix_ll_bf16_availability.py"   "${SITE}/vllm/model_executor/kernels/linear/cute_dsl/ll_bf16.py" || true
  python3 "${RUNTIME_PATCHES}/fix_deepep_ht_cudagraph_guard.py" "${SITE}/vllm/config/compilation.py" || true
  # Opt-in experiment, not applied by default.
  if [[ -n "${HT_NUM_QPS_24:-}" ]]; then
    python3 "${RUNTIME_PATCHES}/ht_num_qps_24.py" "${SITE}/vllm/distributed/device_communicators/all2all.py" || true
  fi
fi

# DeepEP buffer size; DeepEP's own hint for hidden=6144 at 16 ranks is ~165 MB.
export VLLM_DEEPEP_BUFFER_SIZE_MB="${DEEPEP_BUFFER_MB:-1024}"

# SIGABRT then dumps every thread's Python stack; py-spy cannot work here (the
# container has no capabilities, so ptrace is denied).
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"

export HF_HUB_OFFLINE=1
# 703.74 GiB off Lustre takes ~255 s quiet, ~9 min under contention; the 600 s default
# aborts startup in the latter case.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-2400}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
# CUDA_LAUNCH_BLOCKING=1 makes launches synchronous so an async fault is reported at
# the launch that caused it. Without it, a fault surfaces later as a sticky error on an
# unrelated runtime call -- which is how this presented: an assert on
# cudaFuncSetAttribute for a smem size the device demonstrably allows, in a combine
# that passes standalone with vLLM's own Buffer arguments.
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"

# --numa-bind: vLLM defaults it off, which leaves all four workers -- and UCCL's CPU-side
#   proxy polling threads -- floating across every Grace domain. vLLM's equivalent of the
#   `numactl --cpunodebind` binding spellbook applies to one-task-per-GPU launches.
# --moe-backend: required as deep_gemm for deepep_high_throughput, which otherwise picks
#   FLASHINFER_CUTLASS and deadlocks. See docs/deepep-high-throughput.md.
# --load-format: runai_streamer is 2.4-4.4x faster than the default. results/RESULTS.md.
COMMON=(
  "${MODEL}"
  --served-model-name glm-5.3
  --tensor-parallel-size "${TP_SIZE}"
  --data-parallel-size "${DP_TOTAL}"
  --data-parallel-size-local "${DP_LOCAL}"
  --data-parallel-address "${HEAD_IP}"
  --data-parallel-rpc-port "${DP_RPC_PORT}"
  --enable-expert-parallel
  --all2all-backend "${ALL2ALL}"
  --numa-bind
  --load-format "${LOAD_FORMAT:-runai_streamer}"
  --kv-cache-dtype fp8
  --max-model-len "${MAX_MODEL_LEN:-32768}"
  --max-num-seqs "${MAX_NUM_SEQS:-16}"
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.90}"
  ${MOE_BACKEND:+--moe-backend} ${MOE_BACKEND:+$MOE_BACKEND}
  ${ENFORCE_EAGER:+--enforce-eager}
)

echo "[node ${NODE_RANK}] $(hostname) nodes=${SLURM_JOB_NUM_NODES} TP=${TP_SIZE} DP=${DP_TOTAL} (local ${DP_LOCAL}, start ${START_RANK}) EP=$(( DP_TOTAL * TP_SIZE )) all2all=${ALL2ALL} uccl=${UCCL_EP_TRANSPORT:-<unset>}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed "s/^/[node ${NODE_RANK}] /"

# --data-parallel-start-rank goes ONLY on the headless workers. On a non-headless
# node it trips arg_utils.py:2252 --
#     if self.data_parallel_start_rank is not None and not headless:
#         self.data_parallel_hybrid_lb = True
# which puts the head in hybrid-lb mode, and the head then rejects every headless
# worker with "Remote engine N must not use --headless in external or hybrid dp lb
# mode". The head infers its own rank 0 from the absence of the flag.
if (( NODE_RANK == 0 )); then
  exec vllm serve "${COMMON[@]}" \
    --host 0.0.0.0 --port "${API_PORT}" \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    2>&1 | tee "${LOGDIR}/serve-node0.log"
else
  exec vllm serve "${COMMON[@]}" --headless \
    --data-parallel-start-rank "${START_RANK}" \
    2>&1 | tee "${LOGDIR}/serve-node${NODE_RANK}.log"
fi
