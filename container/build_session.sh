#!/bin/bash
# Hold one long-lived bare compute node for repeated image builds.
#
# podman's graphroot is /dev/shm: RAM, per-node, wiped between allocations, so a fresh
# allocation re-pulls the ~20 GB base every time. Inside one allocation the layers stay
# cached and a rebuild only redoes what changed.
#
#   JOB=$(./build_session.sh start)          # prints the job id
#   ./build_session.sh warm-cache "$JOB"     # one-time: cache the base image on scratch
#   ./build_session.sh run "$JOB"            # rebuild (starts the layer cache itself)
#   ./build_session.sh registry "$JOB" status|down|delete
#   UCCL_REF=<sha> ./build_session.sh run "$JOB"   # override pins per run
#   ./build_session.sh stop "$JOB"
#
# NB: no --environment anywhere here. Builds need the node's own podman.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACCOUNT="${SLURM_ACCOUNT:-infra01}"
PARTITION="${SLURM_PARTITION:-normal}"
RESERVATION="${SLURM_RESERVATION-SD-69241-apertus-1-5-0}"
TIME_LIMIT="${SLURM_TIME:-05:00:00}"

case "${1:-}" in
  start)
    name="glm53-buildsession-$(date +%s)"
    log="${HERE}/../logs/build-session-${name}.log"
    mkdir -p "$(dirname "$log")"
    res_args=(); [[ -n "$RESERVATION" ]] && res_args=(--reservation="$RESERVATION")
    nohup srun --job-name="$name" --account="$ACCOUNT" --partition="$PARTITION" \
      "${res_args[@]}" --nodes=1 --ntasks=1 --cpus-per-task=288 \
      --time="$TIME_LIMIT" --exclusive \
      sleep infinity >"$log" 2>&1 &
    for _ in $(seq 1 180); do
      jid=$(squeue --noheader --name="$name" --format='%A' | head -n1 | tr -d ' ')
      if [[ -n "$jid" ]] && squeue --noheader --job="$jid" --states=RUNNING | grep -q .; then
        printf '%s\n' "$jid"; exit 0
      fi
      sleep 2
    done
    echo "allocation did not start; see $log" >&2; exit 1
    ;;
  warm-cache)
    # Push the base image into the local registry, once. Everything after that is a
    # scratch read instead of a ~20 GB jfrog pull. Must happen in a single step: the
    # registry's port forwarder dies with the step that started it.
    jid="${2:?usage: $0 warm-cache JOBID [BASE_IMAGE]}"
    base="${3:-$(sed -n 's/^ARG BASE_IMAGE=//p' "${HERE}/Containerfile" | head -1)}"
    tool="${LR_TOOL:-${HOME}/.local/share/local-registry}"
    data="${LR_DATA:-${SCRATCH}/tmp/local-registry/registry}"
    echo "caching ${base}"
    echo "  in ${data}"
    # Fed over stdin rather than `bash -c "..."`, which would swallow the inner quotes.
    exec srun --jobid="$jid" --overlap --nodes=1 --ntasks=1 --cpus-per-task=288 -u \
      bash -s -- "${tool}" "${data}" "${base}" <<'REMOTE'
set -euo pipefail
tool="$1"; data="$2"; base="$3"
mkdir -p "${data}"
addr="$("${tool}/registry" status 2>/dev/null || true)"
if [[ -z "${addr}" ]] || ! curl -sf -o /dev/null --max-time 5 "http://${addr}/v2/"; then
  "${tool}/registry" down >/dev/null 2>&1 || true
  addr="$("${tool}/registry" up "${data}")"
fi
for _ in $(seq 1 15); do
  curl -sf -o /dev/null --max-time 5 "http://${addr}/v2/" && break
  sleep 1
done
podman pull "${base}"
podman tag "${base}" "${addr}/base:${base##*:}"
podman push --tls-verify=false "${addr}/base:${base##*:}"
echo "cached as ${addr}/base:${base##*:}"
du -sh "${data}"
REMOTE
    ;;
  registry)
    # eth-cscs/local-registry, the build-layer cache, for inspecting or resetting it.
    # There is deliberately no `up` here: a registry started in its own step loses its
    # port forwarder when that step exits. build.sh starts it in the step that builds.
    jid="${2:?usage: $0 registry JOBID status|down|delete}"
    cmd="${3:-status}"
    tool="${LR_TOOL:-${HOME}/.local/share/local-registry}"
    if [[ ! -x "${tool}/registry" ]]; then
      echo "no registry tool at ${tool}; run:" >&2
      echo "  git clone https://github.com/eth-cscs/local-registry.git ${tool}" >&2
      exit 1
    fi
    reg_args=("$cmd")
    exec srun --jobid="$jid" --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 -u \
      "${tool}/registry" "${reg_args[@]}"
    ;;
  run)
    jid="${2:?usage: $0 run JOBID}"
    exec srun --jobid="$jid" --overlap --nodes=1 --ntasks=1 --cpus-per-task=288 \
      -u bash "${HERE}/build.sh"
    ;;
  stop)
    jid="${2:?usage: $0 stop JOBID}"
    scancel "$jid"; echo "cancelled $jid"
    ;;
  stop-build)
    # Cancel only the build step, leaving the session alive.
    # NB: step .0 is the `sleep infinity` that HOLDS the allocation -- cancelling that
    # kills the whole session (and any build running in it). The build steps are the
    # higher-numbered ones, so pick the newest non-.0, non-extern, non-batch step.
    jid="${2:?usage: $0 stop-build JOBID}"
    step=$(squeue -s -j "$jid" --noheader -o "%i" 2>/dev/null \
             | grep -vE "extern|batch" | grep -v "\.0$" | sort -t. -k2 -n | tail -1)
    if [[ -z "$step" ]]; then echo "no build step running in $jid"; exit 0; fi
    echo "cancelling build step $step (session $jid stays up)"
    scancel "$step"
    ;;
  *)
    echo "usage: $0 {start|warm-cache JOBID [BASE]|registry JOBID CMD|run JOBID|stop-build JOBID|stop JOBID}" >&2; exit 2
    ;;
esac
