#!/bin/bash
# [2026-09-17T00:20] Four-node MI300A GLM-5.3 UCCL low-latency smoke test.
#SBATCH --job-name=glm53-rocm-uccl
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
HERE="${HERE:-${SLURM_SUBMIT_DIR}}"

IMAGE=/iopsstor/scratch/cscs/anowak/img/vllm-rocm-alps7-uccl.sqsh
MODEL=/iopsstor/scratch/cscs/anowak/hf_cache/hub/models--zai-org--GLM-5.3/snapshots/aca966e4e02791568aa6a4ced368624b3d897f42
WORKER="${HERE}/scripts/serve_worker_rocm.sh"
LOGDIR=/iopsstor/scratch/cscs/anowak/tmp/vllm-rocm-uccl-glm53/glm53-smoke/run-${SLURM_JOB_ID}
mkdir -p "$LOGDIR"
mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort)
HEAD_NODE="${nodes[0]}"
HEAD_IP="$(getent hosts "$HEAD_NODE" | awk 'NR==1{print $1}')"
HEAD_IP="${HEAD_IP:-$HEAD_NODE}"
export MODEL HEAD_IP LOGDIR
export API_PORT=8000 DP_RPC_PORT=13345
export ALL2ALL=allgather_reducescatter TP_SIZE=2
export MAX_MODEL_LEN=8192 MAX_NUM_SEQS=1 GPU_MEM_UTIL=0.70
export OMP_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
export LOAD_FORMAT=runai_streamer ENFORCE_EAGER=1 DEEPEP_BUFFER_MB=1024
export UCCL_EP_TRANSPORT=cxi FI_PROVIDER=cxi UCCL_EP_CPU_TIMEOUT_SECS=600
# Full GLM-5.3 AMD recipe kernels; AITER MoE needs standard all-gather activations.
export VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1
export MOE_BACKEND=aiter LINEAR_BACKEND=aiter KV_CACHE_DTYPE=fp8_e4m3
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export UCCL_ENABLE_SHRINK_PATCH="${HERE}/container/patches/add_enable_shrink.py"
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV NCCL_NET=Socket
export HF_HOME=/iopsstor/scratch/cscs/anowak/hf_cache HF_HUB_CACHE=/iopsstor/scratch/cscs/anowak/hf_cache/hub
export OCI_ANNOTATION_com__hooks__netstack__source=host
export OCI_ANNOTATION_com__hooks__cxi__enabled=true
export ENROOT_CACHE_PATH=/iopsstor/scratch/cscs/anowak/tmp/enroot/cache

echo "head=$HEAD_NODE ($HEAD_IP) nodes=${nodes[*]} logs=$LOGDIR"
srun --overlap --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" \
  --ntasks-per-node=1 --cpus-per-task=192 --network=disable_rdzv_get --mpi=pmix \
  -l -u bash -c '
    exec enroot start --rw \
      -m /iopsstor:/iopsstor -m /users/anowak:/users/anowak \
      "'$IMAGE'" env \
        SLURM_NODEID="$SLURM_NODEID" SLURM_JOB_NUM_NODES="$SLURM_JOB_NUM_NODES" \
        MODEL="'$MODEL'" HEAD_IP="'$HEAD_IP'" API_PORT="'$API_PORT'" \
        DP_RPC_PORT="'$DP_RPC_PORT'" LOGDIR="'$LOGDIR'" \
        ALL2ALL="'$ALL2ALL'" TP_SIZE="'$TP_SIZE'" \
        MAX_MODEL_LEN="'$MAX_MODEL_LEN'" MAX_NUM_SEQS="'$MAX_NUM_SEQS'" \
        GPU_MEM_UTIL="'$GPU_MEM_UTIL'" OMP_NUM_THREADS="'$OMP_NUM_THREADS'" LOAD_FORMAT="'$LOAD_FORMAT'" \
        ENFORCE_EAGER="'$ENFORCE_EAGER'" DEEPEP_BUFFER_MB="'$DEEPEP_BUFFER_MB'" \
        MOE_BACKEND="'$MOE_BACKEND'" LINEAR_BACKEND="'$LINEAR_BACKEND'" \
        KV_CACHE_DTYPE="'$KV_CACHE_DTYPE'" \
        UCCL_EP_TRANSPORT=cxi FI_PROVIDER=cxi UCCL_EP_CPU_TIMEOUT_SECS=600 \
        VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1 \
        VLLM_WORKER_MULTIPROC_METHOD=spawn \
        TRITON_CACHE_DIR="/tmp/triton-cache-node-${SLURM_NODEID}" \
        TORCHINDUCTOR_CACHE_DIR="/tmp/inductor-cache-node-${SLURM_NODEID}" \
        VLLM_CACHE_ROOT="/tmp/vllm-cache-node-${SLURM_NODEID}" \
        HOME="/tmp/glm53-home-${SLURM_JOB_ID}-${SLURM_NODEID}" \
        XDG_CACHE_HOME="/tmp/glm53-cache-${SLURM_JOB_ID}-${SLURM_NODEID}" \
        AITER_CACHE_DIR="/tmp/aiter-cache-node-${SLURM_NODEID}" \
        TORCH_HOME="/tmp/torch-home-node-${SLURM_NODEID}" \
        TORCH_EXTENSIONS_DIR="/tmp/torch-extensions-node-${SLURM_NODEID}" \
        NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV NCCL_NET=Socket \
        UCCL_ENABLE_SHRINK_PATCH="'$UCCL_ENABLE_SHRINK_PATCH'" \
        HF_HOME="'$HF_HOME'" HF_HUB_CACHE="'$HF_HUB_CACHE'" \
        bash "'$WORKER'"
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
curl -sf "http://${HEAD_IP}:${API_PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3","messages":[{"role":"user","content":"What is 2 + 2? Reply with only the digit."}],"max_tokens":4096,"temperature":1,"top_p":1,"chat_template_kwargs":{"reasoning_effort":"low","clear_thinking":true}}' \
  | tee "$LOGDIR/completion.json" || exit 1
echo
# Validate a known answer, not just HTTP 200 or the existence of generated tokens.
if ! jq -e '.choices[0] | .finish_reason == "stop" and (.message.content // "" | test("^\\s*4[.!]?\\s*$"))' \
  "$LOGDIR/completion.json" >/dev/null; then
  echo "SMOKE FAILED: response did not finish with the expected answer 4"
  exit 1
fi
echo "GLM-5.3 ROCm UCCL SMOKE PASS"
