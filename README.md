# GLM-5.3 on Alps (Clariden GH200)

Serving `zai-org/GLM-5.3` with vLLM + UCCL-EP, so vLLM's DeepEP all2all backends run over
Slingshot/CXI.

**Measured results live in [`results/RESULTS.md`](results/RESULTS.md).** Short version:
`deepep_low_latency` is the default and the only backend that fits long context;
`deepep_high_throughput` prefills 2.8x faster but decodes 5.7x slower, which makes
prefill/decode disaggregation the interesting deployment.

## Quickstart

```bash
# 1. weights (once) -- ~704 GiB into $SCRATCH/hf_cache, resumable
sbatch sbatch/download.sbatch

# 2. image (once)
#    Needs the build cache tool, cloned once:
#      git clone https://github.com/eth-cscs/local-registry.git ~/.local/share/local-registry
JOB=$(container/build_session.sh start)
container/build_session.sh warm-cache "$JOB"     # once; puts the 12 GB base on scratch
container/build_session.sh run "$JOB"
container/build_session.sh stop "$JOB"

# 3. serve
sbatch sbatch/serve.sbatch
```

Everything else is a knob on step 3:

```bash
# long context
sbatch --export=ALL,MAX_MODEL_LEN=256000 sbatch/serve.sbatch

# 8 nodes -> EP=32, ~2x the KV cache
sbatch --nodes=8 sbatch/serve.sbatch

# the 11-point benchmark matrix
sbatch --export=ALL,BENCH_SUITE=1,MAX_NUM_SEQS=64 sbatch/serve.sbatch

# long-context prefill sweep
sbatch --export=ALL,BENCH_PREFILL=1,MAX_MODEL_LEN=131072,PREFILL_LENS="1024 16384 65536" sbatch/serve.sbatch

# high-throughput backend (prefill-oriented; needs both of these)
sbatch --export=ALL,ALL2ALL=deepep_high_throughput,MOE_BACKEND=deep_gemm,GPU_MEM_UTIL=0.80 sbatch/serve.sbatch
```

## Layout

| path | what |
|---|---|
| `sbatch/serve.sbatch` | the one serve entrypoint; all knobs documented in its header |
| `sbatch/download.sbatch` | weight download |
| `scripts/serve_worker.sh` | runs inside the container, one per node; builds the vLLM command |
| `scripts/bench_suite.sh` | 11-point matrix: concurrency / prefill / decode |
| `scripts/bench_prefill.sh` | long-context TTFT sweep |
| `container/` | Containerfile, build scripts, compatibility patches, import smoke test |
| `edf/vllm.toml` | the UCCL image (default) |
| `edf/vllm-base.toml` | the stock published image, no UCCL — for comparison runs |
| `results/` | benchmark TSVs and the write-up |
| `docs/` | investigation notes |
| `tools/` | standalone UCCL/DeepEP reproducers, used for debugging |

## Topology

`NODES x 4 GH200`, TP=2, DP = 2 per node, **EP = DP x TP = NODES x 4**. vLLM has no EP-size
flag; EP is always `data_parallel_size x prefill_context_parallel_size x tensor_parallel_size`
(`distributed/parallel_state.py::initialize_model_parallel`).

| nodes | EP | experts/GPU | weights/GPU | KV/GPU | 256k context |
|---:|---:|---:|---:|---:|---|
| 2 | 8 | 32 | 105 GiB | — | **does not fit** |
| 4 | 16 | 16 | 61 GiB | 20.9 GiB | yes, 1.62x concurrency |
| 8 | 32 | 8 | ~40 GiB | ~42 GiB | yes, ~3x concurrency |

684 GiB of routed experts shard across EP; ~19.6 GiB (attention, shared expert, embeddings)
is replicated per rank at TP=1, halved at TP=2. **4 nodes is the floor** for this model.

## The image

Built from the published Alps vLLM image (`vllm-cuda:alps7-dev-...`, vLLM 0.29.0,
torch 2.13.0a0, CUDA 13.3) plus:

- **UCCL-EP** from `uccl-project/uccl` (upstream, *not* the swiss-ai fork — the fork pin is
  2.5 months stale and both DeepEP backends fail on it), installed as `uccl.ep` + a drop-in
  `deep_ep`, built with `USE_LIBFABRIC_CXI=1` for Slingshot.
- **`runai_model_streamer`** — 2.4-4.4x faster weight loading, the default loader.
- **`nixl==1.3.0`** — for prefill/decode KV transfer (connector wiring not yet written).
- Three compatibility patches in `container/patches/`, applied at build time and also
  applicable at runtime via `RUNTIME_PATCHES=` (they are pure-Python edits, so iterating on
  them does not need a 20-minute rebuild).

### Patches

| patch | why |
|---|---|
| `add_enable_shrink.py` | no-op against upstream UCCL, which already accepts vLLM 0.29's `enable_shrink` kwarg. Kept for older pins. |
| `fix_ll_bf16_availability.py` | vLLM's cuteDSL router GEMM probes only `cutlass`, but also needs `quack`, which the Alps base filters out. Without this the MoE router dies on the first small batch. |
| `fix_deepep_ht_cudagraph_guard.py` | **real upstream bug.** vLLM's "disable CUDA graphs for DeepEP HT" guard sits behind a `CompilationMode` early-return, while the hazard depends on `cudagraph_mode`. GLM-5.3 uses breakable CUDA graphs (auto-enabled for its architecture), which sets `CompilationMode.NONE` — so the guard never runs. Present in vLLM main too. |

## Gotchas worth knowing

- **`deepep_high_throughput` needs `--moe-backend deep_gemm`.** Otherwise it selects
  FLASHINFER_CUTLASS, whose autotuner does a collective `all_reduce` while profiling and
  searches on *local* tensor shapes. Under EP every rank has a different token count, so
  ranks issue mismatched collectives and deadlock. See `docs/deepep-high-throughput.md`.
- **HT also needs `GPU_MEM_UTIL=0.80`** — its ragged layout uses more activation memory and
  OOMs at 0.90.
- `~/.local/lib/python3.12/site-packages` leaks into the container through the home mount.
  Both EDFs set `PYTHONNOUSERSITE=1`; without it a stray `nanobind` there shadows the image's.
- The container engine merges the submitting shell's `PATH` *ahead* of the image's, so both
  EDFs pin the image's own `PATH`.
- Builds must run on a compute node with **no** `--environment` — podman needs the node's own
  runtime, and its graphroot is `/dev/shm` (RAM, wiped between allocations). The scratch
  additional-image-store in `~/.config/containers/storage.conf` is what makes cold builds fast.
