#!/bin/bash
# Prefill-focused sweep: grow the prompt, generate almost nothing.
#
# out_len=8 keeps decode out of the measurement, so this is essentially pure TTFT --
# the number that decides whether HT is worth using as a disaggregated prefill tier.
set -uo pipefail

BASE_URL="${BASE_URL:?}"; SNAP="${SNAP:?}"; OUT_TSV="${OUT_TSV:?}"; LABEL="${LABEL:-run}"

printf 'label\tin_len\tout_len\tconc\treq_s\tout_tok_s\ttotal_tok_s\tttft_ms\tp99_ttft_ms\ttpot_ms\n' > "$OUT_TSV"
metric() { grep -E "$2" "$1" 2>/dev/null | tail -1 | grep -oE "[0-9]+\.?[0-9]*" | tail -1; }

for in_len in ${PREFILL_LENS:-1024 4096 16384 32768 65536 131072}; do
  for conc in ${PREFILL_CONCS:-1 4}; do
    prompts=$(( conc * 4 ))
    echo "=== prefill in=${in_len} conc=${conc} ==="
    log="${OUT_TSV%.tsv}.in${in_len}.c${conc}.log"
    vllm bench serve \
      --backend openai-chat --endpoint /v1/chat/completions \
      --base-url "${BASE_URL}" \
      --model "${SNAP}" --served-model-name glm-5.3 --tokenizer "${SNAP}" \
      --ignore-eos --dataset-name random \
      --random-input-len "${in_len}" --random-output-len 8 \
      --num-prompts "${prompts}" --max-concurrency "${conc}" > "${log}" 2>&1
    printf '%s\t%s\t8\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${LABEL}" "${in_len}" "${conc}" \
      "$(metric "${log}" 'Request throughput')" \
      "$(metric "${log}" 'Output token throughput')" \
      "$(metric "${log}" 'Total token throughput')" \
      "$(metric "${log}" 'Mean TTFT')" \
      "$(metric "${log}" 'P99 TTFT')" \
      "$(metric "${log}" 'Mean TPOT')" >> "${OUT_TSV}"
    tail -1 "${OUT_TSV}"
  done
done
echo "=== prefill sweep complete ==="
column -t -s $'\t' "${OUT_TSV}" 2>/dev/null || cat "${OUT_TSV}"
