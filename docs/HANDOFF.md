# Handoff — GLM-5.3 + UCCL-EP on Alps GH200

Written 2026-09-16, at the point of moving to an AMD cluster.

Everything in this repo is **working and measured** on Clariden. This document says what
was built, what was learned, what is Alps/CUDA-specific, and what carries over to ROCm.

---

## 1. Status

| | |
|---|---|
| Serving GLM-5.3 (753B/39B MoE) on 4x GH200 nodes | **working**, validated end to end |
| vLLM DeepEP all2all over Slingshot via UCCL-EP | **working**, both LL and HT |
| Benchmarks (concurrency, prefill, memory) | **done**, see `results/RESULTS.md` |
| Prefill/decode disaggregation | **designed, not wired** — `nixl==1.3.0` is in the image |
| MTP speculative decoding | **not tested** |

Last validation: job `3411463`, `COMPLETED` in 6:44, served `POST /v1/chat/completions`
200 OK with a coherent 337-token answer, `deepep_low_latency`, EP=16.

A default build now reproduces exactly that image: `container/Containerfile` pins
**upstream** `uccl-project/uccl @ e22a3e8f` (not the swiss-ai fork, whose pin is months
stale and fails on both DeepEP backends), and `build.sh` writes the path `edf/vllm.toml`
serves.

---

## 2. Where everything lives

| what | path |
|---|---|
| repo | `/users/anowak/developer/glm-5.3/` |
| serving image (validated), and what a default build writes | `$SCRATCH/img/vllm029_alps7_uccl_upstream.sqsh` |
| weights (703.74 GiB) | `$SCRATCH/hf_cache/hub/models--zai-org--GLM-5.3` |
| build cache tool | `~/.local/share/local-registry` (3 bash scripts, 15 KB) |
| build cache data | `$SCRATCH/tmp/local-registry/registry` (~26 GB) |
| podman config | `~/.config/containers/storage.conf` |

Two backups of `storage.conf` exist: `.bak-20260915T1514` (original) and
`.bak-additionalstore` (the `additionalimagestores` experiment, see §6).

---

## 3. Running it

```bash
sbatch sbatch/download.sbatch                      # weights, once, resumable
JOB=$(container/build_session.sh start)            # build session
container/build_session.sh warm-cache "$JOB"       # once: base image -> registry
container/build_session.sh run "$JOB"              # build
container/build_session.sh stop "$JOB"
sbatch sbatch/serve.sbatch                         # serve
```

Knobs are all `--export` on `serve.sbatch`, documented in its header. The ones that matter:

```bash
MAX_MODEL_LEN=256000                               # long context
--nodes=8                                          # EP=32, ~2x KV cache
BENCH_SUITE=1 MAX_NUM_SEQS=64                      # 11-point benchmark matrix
BENCH_PREFILL=1 PREFILL_LENS="1024 16384 65536"    # TTFT sweep
ALL2ALL=deepep_high_throughput MOE_BACKEND=deep_gemm GPU_MEM_UTIL=0.80
```

---

## 4. The results that matter

Full write-up in `results/RESULTS.md`; raw TSVs beside it. Condensed:

**Backends.** `deepep_low_latency` is the default and the only one that fits long context
(20.9 GiB KV/GPU = 415k tokens). `allgather_reducescatter` decodes marginally faster at
32k but its activation buffers leave only 7.4 GiB and it **cannot start at 256k**.
`deepep_high_throughput` prefills 2.4-2.8x faster and decodes 5.7x slower.

**Prefill vs decode, 64k prompt:** HT 12.01 s TTFT vs LL 34.16 s — a 22-second gap, still
widening with context. Decode: LL 42 ms TPOT vs HT ~246 ms, flat.

**Hence disaggregation.** HT prefill → LL decode for a 64k prompt / 500 tokens:
33.2 s vs 51.7 s LL-only = **1.56x**. KV transfer is cheap (~3.3 GB, ~0.1 s at the
26-41 GB/s UCCL measures over Slingshot). This is the single most valuable unbuilt thing.

**Concurrency (LL):** near-linear to 4, conc 16 is the sweet spot (10x single-stream
throughput for 1.5x latency), 3848 tok/s at conc 64.

**Weight loading:** `runai_streamer` 58-105 s vs 255-434 s default. It is the default loader.

---

## 5. Hard-won findings

### The HT hang — root cause, and it is not what it looked like

`deepep_high_throughput` hung at startup. `DeepEP timeout check failed` was a **red
herring** — a device-side barrier in `ep/include/ep_utils.cuh`, downstream of the real
problem and unrelated to `UCCL_EP_CPU_TIMEOUT_SECS`.

`PYTHONFAULTHANDLER=1` + SIGABRT (py-spy is unusable, `CapEff: 0000000000000000`) showed
every worker blocked in:

```
torch/distributed/distributed_c10d.py:3252   all_reduce
  <- flashinfer/autotuner/autotuner.py:2198  _profile_single_kernel
  <- flashinfer/fused_moe/core.py:960        cutlass_fused_moe
```

FlashInfer's MoE autotuner runs a **collective `all_reduce` while profiling**, and its
search space depends on **local tensor shapes**. Under EP every rank has a different token
count, so ranks issue mismatched collectives and deadlock.

**Fix: `--moe-backend deep_gemm`** (plus `GPU_MEM_UTIL=0.80`; HT's ragged layout OOMs at
0.90). This is an upstream FlashInfer/vLLM bug and is **not reported yet**.

Disproved along the way, so you do not repeat them: CUDA graphs are *not* the cause (HT
hangs with them disabled two different ways); the zero-token-rank hypothesis is wrong
(`--starve-ranks` passes).

### The three patches

`container/patches/`, all pure-Python, so `RUNTIME_PATCHES=` applies them without a rebuild.

| patch | status |
|---|---|
| `fix_deepep_ht_cudagraph_guard.py` | **real upstream vLLM bug, present in main, unreported.** The "disable CUDA graphs for DeepEP HT" guard sits behind a `CompilationMode` early-return while the hazard depends on `cudagraph_mode`. GLM-5.3 runs `CompilationMode.NONE` with `CUDAGraphMode.FULL_AND_PIECEWISE`, so the guard never fires. |
| `fix_ll_bf16_availability.py` | vLLM's cuteDSL router GEMM probes only `cutlass` but also needs `quack`, which the Alps base filters out. Without it the MoE router dies on the first small batch. |
| `add_enable_shrink.py` | no-op against upstream UCCL; kept for older pins. |

### Topology

vLLM has **no EP-size flag**. `EP = data_parallel_size x prefill_context_parallel_size x
tensor_parallel_size` (`distributed/parallel_state.py::initialize_model_parallel`).
On 4-GPU nodes with TP=2: **EP = nodes x 4**. Four nodes is the floor — at EP=8 the
expert weights are 105 GiB/GPU and do not fit.

---

## 6. Build caching — read this before rebuilding anywhere

podman's graphroot here is `/dev/shm` (RAM, per-node, **wiped between allocations**), so
a cold build re-pulls ~20 GB and recompiles everything. Two approaches were tried:

**What does not work: `additionalimagestores` on Lustre.** An additional image store is
read through the **overlay driver**, which needs `user.*` xattrs. Lustre does not support
them:

```
lsetxattr /iopsstor/.../podman-imagestore/overlay/tempdirs/...: operation not supported
```

Confirmed directly: `os.setxattr(f, "user.test", b"y")` → `Operation not supported` on
`/iopsstor`. This is unfixable, not a misconfiguration. The block was removed from
`storage.conf`. (`setfattr` is not installed on these nodes — a test using it returns
`command not found`, which looks exactly like "unsupported". Use `os.setxattr`.)

**What works: `eth-cscs/local-registry`**, which CSCS documents, and which stores blobs as
**ordinary files** — hence fine on Lustre. It caches two different things:

- the **base image**, pushed once by `warm-cache`, consumed via
  `--build-arg BASE_IMAGE=localhost:5000/base:<tag>` (the Containerfile already had
  `ARG BASE_IMAGE`)
- the **layers we build**, via `--cache-from/--cache-to localhost:5000/cache`

**The trap, which is not in the docs:** the registry container outlives the srun step that
started it, but its rootless **port forwarder does not**. `podman ps` shows
`127.0.0.1:5000->5000/tcp` while `ss` shows nothing listening and `curl` returns HTTP 000 —
so `registry status` reports an address nothing answers on, and the build silently misses
the cache. `build.sh` therefore **probes `/v2/`** rather than trusting status, and recycles
the registry `down`+`up` *in the step that is about to build*. It also waits, bounded,
because `up` returns a few seconds before the registry accepts connections (`curl` rc=52).
For the same reason `build_session.sh registry` deliberately has **no `up` subcommand`**.

Also applied: Lustre PFL striping on `.enroot` and the checkpoint
(`1 / 4 / 32 stripes` across `4M / 64M / EOF`, 4M stripe size).

---

## 7. Moving to AMD — what carries and what does not

UCCL supports AMD across all three components (Collective, P2P, **EP**), and AMD's TheRock
has integrated UCCL-Tran/EP/P2P into their official build platform.

### Carries over unchanged

- **The whole approach**: UCCL-EP installs as `uccl.ep` plus a drop-in `deep_ep`, so vLLM's
  `DeepEPHTAll2AllManager` / `DeepEPLLAll2AllManager` find `deep_ep.Buffer` with no
  PYTHONPATH overlay and no per-job build. Select with `VLLM_ALL2ALL_BACKEND=`.
- **`fix_deepep_ht_cudagraph_guard.py`** — pure vLLM control flow, nothing CUDA about it.
  You will need it on ROCm too.
- **The EP topology formula**, `bench_suite.sh` / `bench_prefill.sh`, `serve.sbatch`
  structure, `runai_model_streamer`.
- **The `deepep_v2` exclusion**: vLLM 0.29's `DeepEPV2All2AllManager` wants
  `deep_ep.ElasticBuffer`, which the UCCL wrapper does not provide. Still true on AMD.

### Must change

| Alps/CUDA | AMD |
|---|---|
| `bash build.sh cu13 ep` | `bash build.sh roc7 ep` (ROCm 7.1) or `roc6` (6.4) |
| `USE_LIBFABRIC_CXI=1`, `UCCL_EP_TRANSPORT=cxi` | **CXI is Slingshot-only.** Use RDMA (IB/RoCE) or EFA. |
| NCCL | RCCL: `NCCL_NET_PLUGIN=$(python -c "import uccl; print(uccl.rccl_plugin_path())")` |
| `UCCL_CUDA_ARCH=9.0` (sm_90) | the MI-series target |
| `UCCL_NVL_PEERS=4` (GH200: 4 GPUs/node) | 8 on MI300X nodes |
| `nvidia-smi` Python stub (see below) | not needed; ROCm path differs |
| `--moe-backend deep_gemm` | FlashInfer is CUDA-only; the ROCm MoE path is AITER. **The HT deadlock may not reproduce — but re-test before assuming.** |
| GH200 NUMA binding (4 Grace domains x 72 cores) | different socket layout |
| `fix_ll_bf16_availability.py` | cuteDSL/`quack`, CUDA-only; likely moot |

### The one finding worth carrying most

The measured LL-vs-AGRS ordering **inverts** what `uccl-project/uccl#926` reports (it found
LL 1.66x *faster* than AGRS for GLM-5 on 2x8-GPU H200). The difference is **node width**:
EP=16 over 4-GPU nodes spans four hosts, so DeepEP pays twice the inter-node hops it pays
on 8-GPU nodes.

**MI300X nodes are 8-GPU.** So on AMD you should expect the balance to shift back toward
the published numbers — EP=16 fits in two nodes instead of four. Re-run `bench_suite.sh`
before carrying any backend choice across; the 4-node GH200 conclusion is a property of
this cluster's node width, not of the backends.

### Build gotcha that will follow you

`ep/setup.py` picks `-DUSE_GRACE_HOPPER` from `nvidia-smi --query-gpu=name` (and the arch
from `--query-gpu=compute_cap`) via `subprocess` with an argv list, and offers **no env
override**. `podman build` has no GPU even on a GH200 node, so the Containerfile installs a
Python stand-in. Check whether the ROCm path has the equivalent `rocm-smi` assumption
before a build mystifies you — and note the stub must be Python, not shell: the base image
sets `ENV=/etc/shinit_v2`, which itself calls `nvidia-smi`, so a shell stub recurses
infinitely.

---

## 8. Open items

1. **Wire a NIXL P/D connector.** `nixl==1.3.0` is in the image; the 1.56x is estimated
   from measured TTFT/TPOT, not from a running disaggregated deployment.
2. **Report two upstream bugs**: the vLLM cudagraph guard, and the FlashInfer MoE
   autotuner collective under EP. Both diagnosed, neither filed.
3. **MTP speculative decoding** — untested.
4. **8 nodes / EP=32** for the native 1M window — projected (~42 GiB KV/GPU), not measured.

---

## 9. Gotchas that cost time

- `~/.local/lib/python3.12/site-packages` leaks into the container through the home mount;
  a stray `nanobind` there shadows the image's. Both EDFs set `PYTHONNOUSERSITE=1`.
- The container engine merges the submitting shell's `PATH` **ahead** of the image's, so
  both EDFs pin the image's own `PATH`.
- Builds need a compute node and **no** `--environment` — podman needs the node's own runtime.
- `enroot import` **exits non-zero on success** (it fails cleaning up its own temp dir).
  `build.sh` validates the artifact with `unsquashfs -stat` instead of trusting the exit
  code, and exports to a temp name + `mv` so an interrupted build cannot destroy a working
  image. This was learned by destroying a working image.
- In `build_session.sh`, step `.0` is the `sleep infinity` that **holds the allocation** —
  cancelling it kills the session. Use `stop-build`, which picks the newest non-`.0` step.
- Do not edit a shell script while it is executing: bash reads scripts incrementally, and
  the running build will take a phantom syntax error that looks like a podman failure.
- `vllm bench serve --model glm-5.3` hits HF with a 401. Pass the snapshot path and
  `--served-model-name`.
- GLM-5.3 **thinks unconditionally**, so `max_tokens: 128` returns empty `content` with all
  the budget spent on reasoning tokens. Use >= 1024 for smoke tests.
