#!/bin/bash
# Download zai-org/GLM-5.3 (native FP8, ~704 GiB, 141 shards) into the scratch HF cache.
# Resumable: re-run this script and hf skips completed blobs.
set -euo pipefail

export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_XET_HIGH_PERFORMANCE=1
mkdir -p "$HF_HUB_CACHE"

REPO="${REPO:-zai-org/GLM-5.3}"
WORKERS="${WORKERS:-16}"

echo "[$(date +%Y-%m-%dT%H:%M)] downloading ${REPO} -> ${HF_HUB_CACHE}"
uv run --quiet --with 'huggingface_hub[cli,hf_transfer]' \
  hf download "${REPO}" \
  --max-workers "${WORKERS}" \
  --exclude '.eval_results/*'
echo "[$(date +%Y-%m-%dT%H:%M)] done"
du -sh "${HF_HUB_CACHE}/models--${REPO/\//--}"
