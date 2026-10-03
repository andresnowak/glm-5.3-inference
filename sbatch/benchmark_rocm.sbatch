#!/bin/bash
# Four-node MI300A GLM-5.3 benchmark: AMD recipe kernels, shared CUDA 11-point suite.
#SBATCH --job-name=glm53-rocm-bench
#SBATCH --account=a-csstaff
#SBATCH --partition=mi300
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=/iopsstor/scratch/cscs/anowak/tmp/vllm-rocm-uccl-glm53/glm53-smoke/slurm-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/anowak/tmp/vllm-rocm-uccl-glm53/glm53-smoke/slurm-%j.err
set -uo pipefail

# Submit from the repository root, or export HERE=/path/to/glm-5.3.
HERE="${HERE:-${SLURM_SUBMIT_DIR:-$PWD}}"

IMAGE="${IMAGE:-/iopsstor/scratch/cscs/anowak/img/vllm-rocm-alps7-uccl.sqsh}"
MODEL="${MODEL:-/iopsstor/scratch/cscs/anowak/hf_cache/hub/models--zai-org--GLM-5.3/snapshots/aca966e4e02791568aa6a4ced368624b3d897f42}"
WORKER="${HERE}/scripts/serve_worker_rocm.sh"
LOGDIR="${LOGDIR:-/iopsstor/scratch/cscs/anowak/tmp/vllm-rocm-uccl-glm53/glm53-smoke/run-${SLURM_JOB_ID}}"
mkdir -p "$LOGDIR"
mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort)
HEAD_NODE="${nodes[0]}"
HEAD_IP="$(getent hosts "$HEAD_NODE" | awk 'NR==1{print $1}')"
HEAD_IP="${HEAD_IP:-$HEAD_NODE}"

export MODEL HEAD_IP LOGDIR
export API_PORT="${API_PORT:-8000}" DP_RPC_PORT="${DP_RPC_PORT:-13345}"
export TP_SIZE="${TP_SIZE:-2}"
# Full GLM-5.3 AMD recipe settings; override any value at submission with --export.
export ALL2ALL="${ALL2ALL:-allgather_reducescatter}"
export MOE_BACKEND="${MOE_BACKEND:-aiter}"
export LINEAR_BACKEND="${LINEAR_BACKEND:-aiter}"
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"
# Match the CUDA suite's capacity settings; MI300A needs headroom for shared CPU/GPU HBM.
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.70}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export LOAD_FORMAT="${LOAD_FORMAT:-runai_streamer}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
export VLLM_ROCM_USE_AITER="${VLLM_ROCM_USE_AITER:-1}"
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS="${VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export UCCL_EP_TRANSPORT=cxi FI_PROVIDER=cxi UCCL_EP_CPU_TIMEOUT_SECS=600
export UCCL_ENABLE_SHRINK_PATCH="${UCCL_ENABLE_SHRINK_PATCH:-${HERE}/container/patches/add_enable_shrink.py}"
export HF_HOME="${HF_HOME:-/iopsstor/scratch/cscs/anowak/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH:-/iopsstor/scratch/cscs/anowak/tmp/enroot/cache}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}" NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,ENV}" NCCL_NET="${NCCL_NET:-Socket}"
export OCI_ANNOTATION_com__hooks__netstack__source=host
export OCI_ANNOTATION_com__hooks__cxi__enabled=true

if [[ "$ALL2ALL" == deepep_low_latency && "$MOE_BACKEND" == aiter ]]; then
  echo "ERROR: AITER MoE is incompatible with DeepEP low-latency batched experts; use allgather_reducescatter or select a non-AITER MoE backend." >&2
  exit 2
fi

# Reflect the actual engine configuration in the unchanged suite's TSV label.
LABEL="rocm_moe-${MOE_BACKEND}_linear-${LINEAR_BACKEND}_kv-${KV_CACHE_DTYPE}_all2all-${ALL2ALL}"
RUN_CONFIG="$LOGDIR/run-config.txt"
cat > "$RUN_CONFIG" <<EOF
correctness_status=unverified
image=$IMAGE
model=$MODEL
all2all_backend=$ALL2ALL
moe_backend=$MOE_BACKEND
linear_backend=$LINEAR_BACKEND
kv_cache_dtype=$KV_CACHE_DTYPE
gpu_memory_utilization=$GPU_MEM_UTIL
topology_nodes=${SLURM_JOB_NUM_NODES} gpus_per_node=4 total_gpus=$((SLURM_JOB_NUM_NODES * 4)) tensor_parallel=$TP_SIZE data_parallel=$((SLURM_JOB_NUM_NODES * 4 / TP_SIZE))
max_model_len=$MAX_MODEL_LEN
max_num_seqs=$MAX_NUM_SEQS
enforce_eager=$ENFORCE_EAGER
omp_num_threads=$OMP_NUM_THREADS
aiter_shared_expert_fusion=$VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS
benchmark_matrix=conc-scaling:1024/128,prompts=4xconc,conc=1,4,16,32,64;prefill:1024,4096,16384/32,prompts=16,conc=8;decode:256/128,512,1024,prompts=16,conc=8
EOF

echo "head=$HEAD_NODE ($HEAD_IP) nodes=${nodes[*]} logs=$LOGDIR"
echo "run configuration: $RUN_CONFIG"
srun --overlap --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" \
  --ntasks-per-node=1 --cpus-per-task=192 --network=disable_rdzv_get --mpi=pmix \
  -l -u bash -c '
    NODE_CACHE="/tmp/glm53-rocm-${SLURM_JOB_ID}-${SLURM_NODEID}"
    mkdir -p "$NODE_CACHE/home" "$NODE_CACHE/cache" "$NODE_CACHE/aiter" "$NODE_CACHE/triton" "$NODE_CACHE/inductor" "$NODE_CACHE/vllm"
    exec enroot start --rw \
      -m /iopsstor:/iopsstor -m /users/anowak:/users/anowak \
      "'"$IMAGE"'" env \
        SLURM_JOB_ID="$SLURM_JOB_ID" SLURM_NODEID="$SLURM_NODEID" SLURM_JOB_NUM_NODES="$SLURM_JOB_NUM_NODES" \
        MODEL="'"$MODEL"'" HEAD_IP="'"$HEAD_IP"'" API_PORT="'"$API_PORT"'" \
        DP_RPC_PORT="'"$DP_RPC_PORT"'" LOGDIR="'"$LOGDIR"'" \
        ALL2ALL="'"$ALL2ALL"'" TP_SIZE="'"$TP_SIZE"'" \
        MAX_MODEL_LEN="'"$MAX_MODEL_LEN"'" MAX_NUM_SEQS="'"$MAX_NUM_SEQS"'" \
        GPU_MEM_UTIL="'"$GPU_MEM_UTIL"'" LOAD_FORMAT="'"$LOAD_FORMAT"'" \
        ENFORCE_EAGER="'"$ENFORCE_EAGER"'" OMP_NUM_THREADS="'"$OMP_NUM_THREADS"'" \
        MOE_BACKEND="'"$MOE_BACKEND"'" LINEAR_BACKEND="'"$LINEAR_BACKEND"'" \
        KV_CACHE_DTYPE="'"$KV_CACHE_DTYPE"'" \
        VLLM_ROCM_USE_AITER="'"$VLLM_ROCM_USE_AITER"'" \
        VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS="'"$VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS"'" \
        VLLM_WORKER_MULTIPROC_METHOD="'"$VLLM_WORKER_MULTIPROC_METHOD"'" \
        MALLOC_ARENA_MAX="'"$MALLOC_ARENA_MAX"'" \
        UCCL_EP_TRANSPORT=cxi FI_PROVIDER=cxi UCCL_EP_CPU_TIMEOUT_SECS=600 \
        TRITON_CACHE_DIR="$NODE_CACHE/triton" TORCHINDUCTOR_CACHE_DIR="$NODE_CACHE/inductor" \
        VLLM_CACHE_ROOT="$NODE_CACHE/vllm" AITER_CACHE_DIR="$NODE_CACHE/aiter" AITER_JIT_CACHE_DIR="$NODE_CACHE/aiter" \
        HOME="$NODE_CACHE/home" XDG_CACHE_HOME="$NODE_CACHE/cache" \
        TORCH_HOME="$NODE_CACHE/torch" TORCH_EXTENSIONS_DIR="$NODE_CACHE/extensions" \
        NCCL_DEBUG="'"$NCCL_DEBUG"'" NCCL_DEBUG_SUBSYS="'"$NCCL_DEBUG_SUBSYS"'" NCCL_NET="'"$NCCL_NET"'" \
        UCCL_ENABLE_SHRINK_PATCH="'"$UCCL_ENABLE_SHRINK_PATCH"'" \
        HF_HOME="'"$HF_HOME"'" HF_HUB_CACHE="'"$HF_HUB_CACHE"'" \
        bash -c '\''mkdir -p "$HOME" "$XDG_CACHE_HOME" "$AITER_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT"; exec bash "$1"'\'' _ "'"$WORKER"'"
  ' &
SERVE_PID=$!
cleanup() { kill "$SERVE_PID" 2>/dev/null || true; wait "$SERVE_PID" 2>/dev/null || true; }
trap cleanup EXIT TERM INT

READY=0
for i in $(seq 1 240); do
  if ! kill -0 "$SERVE_PID" 2>/dev/null; then echo "serve step exited early"; break; fi
  if curl -sf -m 5 "http://${HEAD_IP}:${API_PORT}/health" >/dev/null 2>&1; then READY=1; break; fi
  sleep 15
  (( i % 8 == 0 )) && echo "still waiting $((i*15))s"
done
if (( READY != 1 )); then
  echo "SERVER DID NOT BECOME READY"
  tail -120 "$LOGDIR/serve-node0.log" 2>/dev/null || true
  exit 1
fi

echo "SERVER READY after ~$((i*15))s"
OUT_TSV="$LOGDIR/bench_suite_rocm.tsv"
echo "Starting repository 11-point benchmark suite: $OUT_TSV"
enroot start \
  -m /iopsstor:/iopsstor -m /users/anowak:/users/anowak \
  "$IMAGE" env \
    BASE_URL="http://${HEAD_IP}:${API_PORT}" \
    SNAP="$MODEL" OUT_TSV="$OUT_TSV" LABEL="$LABEL" \
    HOME="/tmp/glm53-rocm-bench-home-${SLURM_JOB_ID}" \
    XDG_CACHE_HOME="/tmp/glm53-rocm-bench-cache-${SLURM_JOB_ID}" \
    TORCH_HOME="/tmp/glm53-rocm-bench-torch-${SLURM_JOB_ID}" \
    HF_HOME="$HF_HOME" HF_HUB_CACHE="$HF_HUB_CACHE" \
    bash -c 'mkdir -p "$HOME" "$XDG_CACHE_HOME"; exec bash "$1"' _ "${HERE}/scripts/bench_suite.sh" \
  2>&1 | tee "$LOGDIR/benchmark-suite.log"
bench_status=${PIPESTATUS[0]}
if (( bench_status != 0 )); then
  echo "BENCHMARK SUITE FAILED: exit $bench_status"
  exit "$bench_status"
fi
echo "BENCHMARK REQUESTS PASS (correctness unverified); GLM-5.3 ROCm result: $OUT_TSV"
