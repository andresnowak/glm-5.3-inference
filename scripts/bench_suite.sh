#!/bin/bash
# A battery of `vllm bench serve` runs against ONE already-running server.
#
# Sweeps against one server: bringing GLM-5.3 up costs ~5 min per job.
# One TSV row per point.
set -euo pipefail

BASE_URL="${BASE_URL:?}"
SNAP="${SNAP:?}"
OUT_TSV="${OUT_TSV:?}"
LABEL="${LABEL:-run}"

printf 'label\tscenario\tin_len\tout_len\tprompts\tconc\treq_s\tout_tok_s\ttotal_tok_s\tttft_ms\ttpot_ms\tp99_tpot_ms\n' > "$OUT_TSV"

# Pull one "Metric: value" number out of a bench log.
metric() { grep -E "$2" "$1" 2>/dev/null | tail -1 | grep -oE "[0-9]+\.?[0-9]*" | tail -1; }

bench_point() {
  local scenario="$1" in_len="$2" out_len="$3" prompts="$4" conc="$5"
  echo "=== ${scenario}: in=${in_len} out=${out_len} prompts=${prompts} conc=${conc} ==="
  local log="${OUT_TSV%.tsv}.${scenario}.in${in_len}.out${out_len}.c${conc}.log"
  vllm bench serve \
    --backend openai-chat --endpoint /v1/chat/completions \
    --base-url "${BASE_URL}" \
    --model "${SNAP}" --served-model-name glm-5.3 --tokenizer "${SNAP}" \
    --ignore-eos \
    --dataset-name random \
    --random-input-len "${in_len}" --random-output-len "${out_len}" \
    --num-prompts "${prompts}" --max-concurrency "${conc}" \
    > "${log}" 2>&1

  local successful failed
  successful="$(metric "${log}" 'Successful requests')"
  failed="$(metric "${log}" 'Failed requests')"
  if [[ "${successful}" != "${prompts}" || "${failed}" != "0" ]]; then
    printf 'benchmark point failed: successful=%s/%s failed=%s; see %s\n' \
      "${successful:-unknown}" "${prompts}" "${failed:-unknown}" "${log}" >&2
    return 1
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${LABEL}" "${scenario}" "${in_len}" "${out_len}" "${prompts}" "${conc}" \
    "$(metric "${log}" 'Request throughput')" \
    "$(metric "${log}" 'Output token throughput')" \
    "$(metric "${log}" 'Total token throughput')" \
    "$(metric "${log}" 'Mean TTFT')" \
    "$(metric "${log}" 'Mean TPOT')" \
    "$(metric "${log}" 'P99 TPOT')" \
    >> "${OUT_TSV}"
  tail -3 "${OUT_TSV}" | head -1
}

# 1. Concurrency scaling: where throughput saturates, and what it costs in latency.
for c in 1 4 16 32 64; do
  bench_point conc-scaling 1024 128 $((c * 4)) "${c}"
done

# 2. Prefill: grow the prompt, keep generation short. Isolates TTFT.
for i in 1024 4096 16384; do
  bench_point prefill "${i}" 32 16 8
done

# 3. Decode: short prompt, long generation. Isolates the per-token MoE all2all.
for o in 128 512 1024; do
  bench_point decode 256 "${o}" 16 8
done

echo "=== suite complete: ${OUT_TSV} ==="
column -t -s $'\t' "${OUT_TSV}" 2>/dev/null || cat "${OUT_TSV}"
