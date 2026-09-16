#!/bin/bash
# Serve GLM-5.3 on Alps GH200 and optionally benchmark it.
#
#   sbatch sbatch/serve.sbatch                                    # serve + one smoke request
#   sbatch --export=ALL,BENCH_SUITE=1 sbatch/serve.sbatch         # + 11-point benchmark matrix
#   sbatch --export=ALL,BENCH_PREFILL=1 sbatch/serve.sbatch       # + long-context prefill sweep
#
# Knobs (all via --export=ALL,NAME=value):
#   ALL2ALL         deepep_low_latency (default) | deepep_high_throughput | allgather_reducescatter
#   MOE_BACKEND     expert-GEMM backend; REQUIRED as deep_gemm for deepep_high_throughput
#   MAX_MODEL_LEN   context length            (default 32768)
#   MAX_NUM_SEQS    max concurrent sequences  (default 16)
#   GPU_MEM_UTIL    fraction of HBM to use    (default 0.90; use 0.80 for HT)
#   LOAD_FORMAT     runai_streamer (default)  | auto
#   EDF             edf/vllm.toml (default)   | edf/vllm-base.toml
#   (node count)    sbatch --nodes=N ...      (default 4 -> EP=16; EP = N*4)
#   TP_SIZE         tensor parallel size      (default 2)
#   ENFORCE_EAGER   1 to disable CUDA graphs
#   RUNTIME_PATCHES apply container/patches/*.py at startup instead of rebuilding
#
# Topology: NODES x 4 GH200, --tensor-parallel-size 2, --data-parallel-size (NODES*2),
# --enable-expert-parallel  =>  EP = DP x TP = NODES * 4.
#
#SBATCH --job-name=glm53-serve
#SBATCH --account=infra01
#SBATCH --partition=preemptable
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=/iopsstor/scratch/cscs/anowak/tmp/glm53/logs/serve-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/anowak/tmp/glm53/logs/serve-%j.err

set -uo pipefail

HERE=/users/anowak/developer/glm-5.3
# EDF override, so the same launcher can drive either image:
#   EDF=edf/vllm.toml       the UCCL image (default)
#   EDF=edf/vllm-base.toml  the stock published image, no UCCL
EDF="${EDF:-${HERE}/edf/vllm.toml}"
[[ "${EDF}" = /* ]] || EDF="${HERE}/${EDF}"
SNAP=/iopsstor/scratch/cscs/anowak/hf_cache/hub/models--zai-org--GLM-5.3/snapshots/aca966e4e02791568aa6a4ced368624b3d897f42
LOGDIR="${LOGROOT:-${SCRATCH}/tmp/glm53/logs}/run-${SLURM_JOB_ID}"
mkdir -p "${LOGDIR}"

mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort)
HEAD_NODE="${nodes[0]}"
HEAD_IP="$(getent hosts "$HEAD_NODE" | awk 'NR==1{print $1}')"
HEAD_IP="${HEAD_IP:-$HEAD_NODE}"
API_PORT=8000

export MODEL="$SNAP" HEAD_IP API_PORT LOGDIR
export ALL2ALL="${ALL2ALL:-deepep_low_latency}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-}"
export LOAD_FORMAT="${LOAD_FORMAT:-}"
export BENCH_SUITE="${BENCH_SUITE:-}"
export RUNTIME_PATCHES="${RUNTIME_PATCHES:-}"
export HT_NUM_QPS_24="${HT_NUM_QPS_24:-}"
export DEEPEP_BUFFER_MB="${DEEPEP_BUFFER_MB:-}"
export MOE_BACKEND="${MOE_BACKEND:-}"
export TP_SIZE="${TP_SIZE:-2}"
export BENCH_PREFILL="${BENCH_PREFILL:-}"
export PREFILL_LENS="${PREFILL_LENS:-}"
export PREFILL_CONCS="${PREFILL_CONCS:-}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"

if [[ "${ALL2ALL}" == "deepep_high_throughput" && "${MOE_BACKEND:-}" != "deep_gemm" ]]; then
  echo "ERROR: deepep_high_throughput requires MOE_BACKEND=deep_gemm." >&2
  echo "       Without it vLLM selects FLASHINFER_CUTLASS, whose autotuner does a" >&2
  echo "       shape-dependent collective and deadlocks under EP." >&2
  echo "       See docs/deepep-high-throughput.md" >&2
  exit 1
fi

echo "head=${HEAD_NODE} (${HEAD_IP})  nodes=${nodes[*]}"
echo "all2all=${ALL2ALL}  edf=${EDF}  logs=${LOGDIR}"

MOUNTS="/iopsstor:/iopsstor,/ritom:/ritom,/users/anowak:/users/anowak"
ENVVARS="MODEL,HEAD_IP,API_PORT,LOGDIR,ALL2ALL,MAX_MODEL_LEN,MAX_NUM_SEQS,TP_SIZE,CUDA_LAUNCH_BLOCKING,ENFORCE_EAGER,LOAD_FORMAT,BENCH_SUITE,RUNTIME_PATCHES,HT_NUM_QPS_24,DEEPEP_BUFFER_MB,MOE_BACKEND,BENCH_PREFILL,PREFILL_LENS,PREFILL_CONCS"

# The server steps: one task per node, all four in one step.
srun --overlap --nodes="${SLURM_JOB_NUM_NODES}" --ntasks="${SLURM_JOB_NUM_NODES}" \
  --ntasks-per-node=1 --cpus-per-task=288 \
  --export="ALL,${ENVVARS}" --container-env="${ENVVARS}" \
  --network=disable_rdzv_get --mpi=pmix -l \
  --environment="${EDF}" \
  --container-mounts="${MOUNTS}" \
  -u bash "${HERE}/scripts/serve_worker.sh" &
SERVE_PID=$!

cleanup() { kill "${SERVE_PID}" 2>/dev/null || true; }
trap cleanup EXIT TERM INT

# Wait for the API to come up. 704 GiB off Lustre across 16 GPUs is not quick, so
# allow a long window; bail early if the serve step dies.
echo "waiting for http://${HEAD_IP}:${API_PORT}/health ..."
READY=0
for i in $(seq 1 240); do   # 240 x 15s = 60 min
  if ! kill -0 "${SERVE_PID}" 2>/dev/null; then echo "serve step exited early"; break; fi
  if curl -sf -m 5 "http://${HEAD_IP}:${API_PORT}/health" >/dev/null 2>&1; then READY=1; break; fi
  sleep 15
  (( i % 8 == 0 )) && echo "  ... still waiting ($((i*15))s)"
done

if (( READY != 1 )); then
  echo "SERVER DID NOT BECOME READY" >&2
  echo "--- tail of node0 log ---" >&2
  tail -80 "${LOGDIR}/serve-node0.log" 2>/dev/null >&2
  exit 1
fi

echo "=== SERVER READY after ~$((i*15))s ==="
curl -sf "http://${HEAD_IP}:${API_PORT}/v1/models" | head -c 800; echo

# GLM-5.3 always opens a <think> block; a small max_tokens is spent entirely on reasoning.
echo "=== one real completion ==="
curl -sf "http://${HEAD_IP}:${API_PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3","messages":[{"role":"user","content":"In one sentence: what is expert parallelism?"}],"max_tokens":1024,"temperature":0}' \
  | tee "${LOGDIR}/completion.json" | head -c 1500; echo

# --model is the tokenizer path; --served-model-name is what goes in the request body.
# --ignore-eos keeps output lengths fixed (GLM-5.3 thinks unconditionally).
echo "=== benchmark (vllm bench serve) ==="
srun --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" --cpus-per-task=32 \
  --network=disable_rdzv_get --mpi=pmix \
  --environment="${EDF}" \
  --container-mounts="${MOUNTS}" \
  -u vllm bench serve \
    --backend openai-chat --endpoint /v1/chat/completions \
    --base-url "http://${HEAD_IP}:${API_PORT}" \
    --model "${SNAP}" --served-model-name glm-5.3 --tokenizer "${SNAP}" \
    --ignore-eos \
    --dataset-name random --random-input-len 1024 --random-output-len 128 \
    --num-prompts 64 --max-concurrency 16 \
  2>&1 | tee "${LOGDIR}/bench.log"

echo "=== done; logs in ${LOGDIR} ==="
