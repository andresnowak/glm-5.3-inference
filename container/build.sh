#!/bin/bash
# Build the vLLM+UCCL image and export it as a squashfs for the container engine.
#
# Must run on a COMPUTE node, not a login node: podman's graphroot is
# /dev/shm/$USER/root (see ~/.config/containers/storage.conf), the base image is
# ~20 GB, and a login node has neither the RAM budget nor the tolerance for it.
#
#   srun --account=infra01 --partition=normal --reservation=SD-69241-apertus-1-5-0 \
#        --nodes=1 --ntasks=1 --cpus-per-task=288 --time=02:00:00 \
#        --pty bash build.sh
#
# Note: NO --environment here. The build needs the node's own podman, not a container.
set -euo pipefail
trap 'rm -f "${TMP_OUT:-}" 2>/dev/null || true' EXIT

TAG="${TAG:-vllm-uccl:alps7-dev}"
OUT="${OUT:-${SCRATCH}/img/vllm029_alps7_uccl_upstream.sqsh}"   # what edf/vllm.toml serves
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Optional pin overrides, e.g. to build against upstream uccl-project/uccl instead of
# the swiss-ai fork:
#   UCCL_REPO=https://github.com/uccl-project/uccl.git UCCL_REF=<sha> ./build.sh
BUILD_ARGS=()
[[ -n "${UCCL_REPO:-}" ]] && BUILD_ARGS+=(--build-arg "UCCL_REPO=${UCCL_REPO}")
[[ -n "${UCCL_REF:-}" ]]  && BUILD_ARGS+=(--build-arg "UCCL_REF=${UCCL_REF}")

# eth-cscs/local-registry caches the base image and the layers we build; graphroot is
# /dev/shm, so nothing survives the allocation otherwise. Not a podman additional image
# store: that needs user.* xattrs, which Lustre lacks. Tool in $HOME, 26 GB of blobs on
# scratch.
CACHE_ARGS=()
LR_TOOL="${LR_TOOL:-${HOME}/.local/share/local-registry}"
LR_DATA="${LR_DATA:-${SCRATCH}/tmp/local-registry/registry}"
lr_reachable() { curl -sf -o /dev/null --max-time 5 "http://$1/v2/"; }
# `registry up` returns before the registry accepts connections.
lr_wait() { for _ in $(seq 1 15); do lr_reachable "$1" && return 0; sleep 1; done; return 1; }
if [[ -x "${LR_TOOL}/registry" ]]; then
  mkdir -p "${LR_DATA}"
  lr_addr="$("${LR_TOOL}/registry" status 2>/dev/null || true)"
  # The container outlives the srun step that started it; its rootless port forwarder
  # does not, so `status` can report an address nothing answers on. Probe, and recycle
  # it here, in the step that builds.
  if [[ -n "${lr_addr}" ]] && ! lr_reachable "${lr_addr}"; then
    "${LR_TOOL}/registry" down >/dev/null 2>&1 || true
    lr_addr=""
  fi
  if [[ -z "${lr_addr}" ]]; then
    lr_addr="$("${LR_TOOL}/registry" up "${LR_DATA}" 2>/dev/null || true)"
  fi
  if [[ -n "${lr_addr}" ]] && lr_wait "${lr_addr}"; then
    CACHE_ARGS=(--layers --tls-verify=false
                --cache-from="${lr_addr}/cache" --cache-to="${lr_addr}/cache")
    echo "  layer cache: ${lr_addr} (data in ${LR_DATA})"
    base_tag="$(sed -n 's/^ARG BASE_IMAGE=.*://p' "${HERE}/Containerfile" | head -1)"
    if curl -sf --max-time 5 "http://${lr_addr}/v2/base/tags/list" \
         | grep -q "\"${base_tag}\""; then
      BUILD_ARGS+=(--build-arg "BASE_IMAGE=${lr_addr}/base:${base_tag}")
      echo "  base image: ${lr_addr}/base:${base_tag} (cached, no jfrog pull)"
    else
      echo "  base image: not cached; pulling from jfrog (build_session.sh warm-cache)"
    fi
  else
    echo "  layer cache: registry unreachable; building without it" >&2
  fi
else
  echo "  layer cache: none; git clone https://github.com/eth-cscs/local-registry.git ${LR_TOOL}"
fi

echo "[$(date +%Y-%m-%dT%H:%M)] building ${TAG} on $(hostname)"
echo "  uccl: ${UCCL_REPO:-<Containerfile default>} @ ${UCCL_REF:-<Containerfile default>}"
# --network=host per the CSCS docs: podman may otherwise misbehave on some clusters.
# A --cache-to failure is fatal to podman build, which would kill a fully cached build.
if ! podman build --network=host "${CACHE_ARGS[@]}" \
       -f "${HERE}/Containerfile" -t "${TAG}" "${BUILD_ARGS[@]}" "${HERE}"; then
  if [[ ${#CACHE_ARGS[@]} -eq 0 ]]; then exit 1; fi
  echo "build failed with the layer cache attached; retrying without it" >&2
  podman build --network=host \
    -f "${HERE}/Containerfile" -t "${TAG}" "${BUILD_ARGS[@]}" "${HERE}"
fi

mkdir -p "$(dirname "${OUT}")"
# Export to a temp file and rename on success: an interrupted build must not destroy a
# working image.
TMP_OUT="${OUT}.new.$$"
rm -f "${TMP_OUT}"
echo "[$(date +%Y-%m-%dT%H:%M)] exporting to ${TMP_OUT}"

# enroot import exits non-zero even on success (it fails cleaning up its temp dir), so
# validate the artifact rather than the exit status.
set +e
enroot import -o "${TMP_OUT}" "podman://${TAG}"
import_rc=$?
set -e

if ! unsquashfs -stat "${TMP_OUT}" >/dev/null 2>&1; then
  echo "FATAL: enroot import produced no valid squashfs (exit ${import_rc})" >&2
  exit 1
fi
if (( import_rc != 0 )); then
  echo "  note: enroot exited ${import_rc}, but the squashfs validates (its own cleanup failed); continuing"
fi

mv -f "${TMP_OUT}" "${OUT}"
echo "[$(date +%Y-%m-%dT%H:%M)] moved into place: ${OUT}"

ls -lh "${OUT}"
echo "[$(date +%Y-%m-%dT%H:%M)] done"
