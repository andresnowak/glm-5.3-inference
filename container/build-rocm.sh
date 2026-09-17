#!/bin/bash
# Build the ROCm vLLM + UCCL-EP derivative and export it for the CSCS container engine.
# Run on an x86 MI300 compute node without --environment.
set -euo pipefail
trap 'rm -f "${TMP_OUT:-}" 2>/dev/null || true' EXIT

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAG="${TAG:-vllm-uccl-rocm:alps7-dev}"
OUT="${OUT:-${SCRATCH}/img/vllm-rocm-alps7-uccl.sqsh}"

build_args=()
[[ -n "${BASE_IMAGE:-}" ]] && build_args+=(--build-arg "BASE_IMAGE=${BASE_IMAGE}")
[[ -n "${UCCL_REPO:-}" ]] && build_args+=(--build-arg "UCCL_REPO=${UCCL_REPO}")
[[ -n "${UCCL_REF:-}" ]] && build_args+=(--build-arg "UCCL_REF=${UCCL_REF}")
[[ -n "${AITER_REPO:-}" ]] && build_args+=(--build-arg "AITER_REPO=${AITER_REPO}")
[[ -n "${AITER_REF:-}" ]] && build_args+=(--build-arg "AITER_REF=${AITER_REF}")
[[ -n "${AITER_MAX_JOBS:-}" ]] && build_args+=(--build-arg "AITER_MAX_JOBS=${AITER_MAX_JOBS}")

# Persist the large ROCm base and built layers across node allocations. Podman's graphroot
# is node-local /dev/shm and disappears after each allocation; the registry stores blobs
# as ordinary files under scratch.
cache_args=()
tool="${LR_TOOL:-${HOME}/.local/share/local-registry}"
data="${LR_DATA:-${SCRATCH}/tmp/local-registry/registry}"
reachable() { curl -sf -o /dev/null --max-time 5 "http://$1/v2/"; }
wait_registry() { for _ in $(seq 1 30); do reachable "$1" && return 0; sleep 1; done; return 1; }
if [[ -x "${tool}/registry" ]]; then
  mkdir -p "${data}"
  addr="$(${tool}/registry status 2>/dev/null || true)"
  if [[ -n "${addr}" ]] && ! reachable "${addr}"; then
    "${tool}/registry" down >/dev/null 2>&1 || true
    addr=""
  fi
  [[ -n "${addr}" ]] || addr="$(${tool}/registry up "${data}")"
  wait_registry "${addr}"
  cache_args=(--layers --tls-verify=false
    --cache-from="${addr}/cache-vllm-rocm-uccl"
    --cache-to="${addr}/cache-vllm-rocm-uccl")
  if curl -sf --max-time 5 "http://${addr}/v2/base/tags/list"       | grep -q '"vllm-rocm-alps7-dev"'; then
    build_args+=(--build-arg "BASE_IMAGE=${addr}/base:vllm-rocm-alps7-dev")
    echo "using cached ROCm vLLM base: ${addr}/base:vllm-rocm-alps7-dev"
  else
    echo "FATAL: ROCm base is not warmed in the local registry" >&2
    echo "run the warm-cache step first" >&2
    exit 1
  fi
fi

printf '[%s] building %s on %s\n' "$(date +%Y-%m-%dT%H:%M)" "${TAG}" "$(hostname)"
podman build --network=host --format=docker "${cache_args[@]}" \
  -f "${HERE}/Containerfile.rocm" -t "${TAG}" \
  "${build_args[@]}" "${HERE}"

mkdir -p "$(dirname "${OUT}")"
# Bare MI300 nodes cannot access the login-node Capstor scratch defaults.
export ENROOT_CACHE_PATH="${ROCM_ENROOT_CACHE_PATH:-${SCRATCH}/tmp/enroot/cache}"
export ENROOT_DATA_PATH="${ROCM_ENROOT_DATA_PATH:-${SCRATCH}/tmp/enroot/data}"
export ENROOT_RUNTIME_PATH="${ROCM_ENROOT_RUNTIME_PATH:-/tmp/enroot-runtime-${UID}}"
mkdir -p "${ENROOT_CACHE_PATH}" "${ENROOT_DATA_PATH}" "${ENROOT_RUNTIME_PATH}"
chmod 700 "${ENROOT_RUNTIME_PATH}"
TMP_OUT="${OUT}.new.$$"
rm -f "${TMP_OUT}"
set +e
enroot import -o "${TMP_OUT}" "podman://${TAG}"
import_rc=$?
set -e
if ! unsquashfs -stat "${TMP_OUT}" >/dev/null 2>&1; then
  echo "FATAL: enroot import produced no valid squashfs (exit ${import_rc})" >&2
  exit 1
fi
mv -f "${TMP_OUT}" "${OUT}"
printf '[%s] installed %s\n' "$(date +%Y-%m-%dT%H:%M)" "${OUT}"
ls -lh "${OUT}"
