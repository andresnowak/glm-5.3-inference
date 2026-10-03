# GLM-5.3 on Alps (GH200 / MI300A)

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

## ROCm on MI300A

The validated ROCm launchers are now in the repository:

```bash
# Run from the repository root. Check mi300 availability/account access first.
sinfo -a -o "%P %a"
# Slurm opens its output files before the script can create directories.
mkdir -p /iopsstor/scratch/cscs/anowak/tmp/vllm-rocm-uccl-glm53/glm53-smoke

# Full-model load + one chat completion; shuts down afterward
sbatch sbatch/serve_rocm.sbatch

# Run the same 11-point matrix as CUDA; shuts down afterward
sbatch sbatch/benchmark_rocm.sbatch

# Matching CUDA capacity settings for a new comparison run
sbatch --export=ALL,BENCH_SUITE=1,MAX_MODEL_LEN=32768,MAX_NUM_SEQS=64 sbatch/serve.sbatch
```

These are site-specific launchers using account `a-csstaff`, partition `mi300`,
and existing weights under `/iopsstor/scratch/cscs/anowak/hf_cache`.
The image is `/iopsstor/scratch/cscs/anowak/img/vllm-rocm-alps7-uccl.sqsh`,
built by `container/Containerfile.rocm` and `container/build-rocm.sh`.
The launchers use Enroot directly, not `edf/vllm-rocm.toml`. Repository code must
be under the mounted `/users/anowak` or `/iopsstor`; `HERE` defaults to the
submission directory and can be exported to select another checkout.

Both use **4 nodes × 4 MI300A GPUs**, TP=2, DP=8, EP=16, one Slurm task per node,
and vLLM-managed GPU workers (no torchrun). Node 0 hosts one OpenAI API server;
the other nodes join as headless workers. The current defaults follow the [full GLM-5.3 AMD recipe](https://github.com/vllm-project/recipes/blob/main/models/zai-org/GLM-5.3.yaml):
AITER sparse MLA, AITER linear and MoE kernels, shared-expert fusion,
`fp8_e4m3` KV cache, and the `glm47` reasoning parser. This installed vLLM build
cannot combine AITER MoE with DeepEP low-latency batched experts, so these
launchers use `allgather_reducescatter` and RCCL `NCCL_NET=Socket` instead.
Eager execution is retained for this MI300A deployment. The worker selects the
native sampler to avoid AITER sampler JIT races. The benchmark puts caches under
`/tmp`; explicit NUMA binding uses nodes 0–3, requires all 192 logical CPUs,
and is checked before model startup. `OMP_NUM_THREADS=1` avoids oversubscription.

The smoke configuration is 8K context / one sequence / memory utilization 0.70.
The benchmark defaults to 32K / 64 sequences / **0.70**, matching the CUDA
comparison's context and concurrency limits. Both call the same
`scripts/bench_suite.sh` for benchmarking, with the same 11 points and TSV schema.
Benchmark settings can be overridden via `sbatch --export=ALL,NAME=value`.
The TSV label identifies the kernels, KV dtype, and all-to-all backend;
`run-config.txt` records the configuration and `correctness_status=unverified`.
A benchmark pass means requests completed, **not that answers were correct**.
**TODO: fix ROCm output correctness before treating these timings as valid model
performance.** The model currently generates garbled/incorrect text and failed
a simple `2 + 2` check, including runs without DeepEP and with both Triton and
AITER MoE. Benchmark completion measures request throughput only, not answer
quality. The recipe-aligned shared-expert-fusion configuration still needs a
separate correctness test. No claim is made that the router selected the same
experts repeatedly.
The installed ROCm vLLM is 0.27.2-dev, retained intentionally; the website recipe
lists a newer version baseline. This is a recipe-adapted setup, not an exact
validated upstream configuration. Startup can take about 30 minutes.
Logs and outputs go to the Slurm output directory above, under `run-<jobid>/`.

Historical provenance (different defaults: Triton MoE, BF16 KV, UCCL low-latency):
smoke job **640358** returned HTTP 200; benchmark job **640649** passed all 11
points. The original smoke answer is no longer available to verify coherence.
Those timings must not be relabeled as results from the current AITER defaults. The imported smoke launcher now requests
192 CPUs to match the worker's subsequently added NUMA binding.
Latest benchmark: job **662284** completed all **11 points** in **46:46** on
2026-10-03 with the current AITER/shared-fusion defaults. Output throughput at
concurrency 64 was **154.36 tokens/s** versus the retained CUDA run's **423.12**.
Raw TSV, run configuration, and point logs are in
[`results/rocm-662284/`](results/rocm-662284/). Correctness remains unverified.
Historical results remain in [`results/bench_suite_rocm.tsv`](results/bench_suite_rocm.tsv)
and [`results/rocm-640649/`](results/rocm-640649/).
The [CUDA/ROCm HTML comparison](results/results.html) separates both ROCm runs
and prominently records the correctness TODO. CUDA uses different kernels,
vLLM versions, and graphs, so this is not an identical-software comparison.

## Benchmarks

Both benchmark modes bring the server up, sweep against **that one server**, write a TSV,
and exit. Bringing GLM-5.3 up costs ~5 min, so a sweep shares one server rather than
paying that per point. Results land in `$LOGROOT/run-<jobid>/`; promote the ones worth
keeping into `results/`.

```bash
# throughput + latency, 11 points
sbatch --export=ALL,BENCH_SUITE=1,MAX_NUM_SEQS=64 sbatch/serve.sbatch

# long-context TTFT
sbatch --export=ALL,BENCH_PREFILL=1,MAX_MODEL_LEN=131072 sbatch/serve.sbatch
```

`BENCH_SUITE=1` runs three scenarios through `vllm bench serve`:

| scenario | shape | sweeps | answers |
|---|---|---|---|
| `conc-scaling` | 1024 in / 128 out | conc 1, 4, 16, 32, 64 | where throughput stops scaling |
| `prefill` | grows input, 32 out | in 1k → 32k, conc 8 | TTFT vs prompt length |
| `decode` | 256 in, grows output | out 128 → 2048, conc 8 | whether TPOT drifts with length |

`BENCH_PREFILL=1` is the prefill-only sweep: `out_len=8`, so it is essentially pure TTFT
— the number that decides whether `deepep_high_throughput` is worth running as a
disaggregated prefill tier. Override the grid:

```bash
PREFILL_LENS="1024 16384 65536"   # default: 1024 4096 16384 32768 65536 131072
PREFILL_CONCS="1 4"               # default: 1 4
```

Both write a header row plus one row per point, so two runs diff directly:

```
label  scenario      in_len  out_len  prompts  conc  req_s  out_tok_s  total_tok_s  ttft_ms  tpot_ms  p99_tpot_ms
```

To compare backends, run the same mode twice and change one thing:

```bash
sbatch --export=ALL,BENCH_SUITE=1 sbatch/serve.sbatch                        # deepep_low_latency
sbatch --export=ALL,BENCH_SUITE=1,ALL2ALL=allgather_reducescatter sbatch/serve.sbatch
sbatch --export=ALL,BENCH_PREFILL=1,ALL2ALL=deepep_high_throughput,MOE_BACKEND=deep_gemm,GPU_MEM_UTIL=0.80 sbatch/serve.sbatch
```

Two things that will otherwise waste a job: `deepep_high_throughput` **needs** both
`MOE_BACKEND=deep_gemm` and `GPU_MEM_UTIL=0.80` (`serve.sbatch` refuses it without the
first), and the measurements are only comparable at equal `MAX_NUM_SEQS` and
`MAX_MODEL_LEN`, since both change how much memory is left for KV cache.

For a single ad-hoc query instead of a sweep, a plain `sbatch/serve.sbatch` already posts
one chat completion and saves it to `completion.json`.

## Layout

| path | what |
|---|---|
| `sbatch/serve.sbatch` | CUDA serve entrypoint; all knobs documented in its header |
| `sbatch/serve_rocm.sbatch` | MI300A serving smoke test |
| `sbatch/benchmark_rocm.sbatch` | validated MI300A 11-point benchmark |
| `scripts/serve_worker_rocm.sh` | ROCm vLLM command, one worker launcher per node |
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
  runtime, and its graphroot is `/dev/shm` (RAM, wiped between allocations). The base image
  and the layers we build are cached in a local registry instead; see `container/build.sh`.
- **Nothing writes to `$HOME`.** Logs go to `$LOGROOT` (default `$SCRATCH/tmp/glm53/logs`),
  caches to Ritom via the EDF. The home inode quota is small, and vLLM quietly filling
  `~/.cache` breaks every job with `Disk quota exceeded` — including, confusingly, jobs that
  never start, because the Slurm output redirect is what fails first.
- `normal` is often `DOWN` on this cluster; `preemptable` works with the `normal` QoS.
  Run `sinfo -a` before assuming a partition exists.
