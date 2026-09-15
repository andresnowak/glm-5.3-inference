# `deepep_high_throughput` on Alps GH200 — does not work, and why

**Status: not usable. Use `deepep_low_latency`.** Not a blocker for GLM-5.3: upstream
benchmarks recommend LL for this model class anyway (see "Is HT even wanted?" below).

Config under test throughout: 4 nodes x 4 GH200, `--tensor-parallel-size 2
--data-parallel-size 8 --enable-expert-parallel` => **EP=16**, 16 experts/GPU, fp8 KV,
`--numa-bind`, 256k context, upstream UCCL `e22a3e8fcd636f7efbbd90900458087a41ff73ce`.

## Three distinct failures, in order

| # | UCCL | symptom |
|---|---|---|
| 1 | swiss-ai fork `fa4c325c` | kernel crash: `internode.cu:2021 'unspecified launch failure'` in `cached_notify`, during vLLM's memory-profiling pass |
| 2 | upstream `e22a3e8f` | `RuntimeError: DeepEP error: timeout (dispatch CPU)` + `CUDA error: capturing stream has unjoined work` |
| 3 | upstream + `UCCL_EP_CPU_TIMEOUT_SECS=600` | **hang** during CUDA graph capture: GPU 0%, worker CPU 240%, never reaches "Graph capturing finished" |

Failure 1 is fixed by moving to upstream UCCL. Failures 2 and 3 are the same underlying
problem: raising the timeout only changed how long it takes to give up.

## CORRECTION (2026-09-15, later): CUDA graphs are NOT the cause

The section below was written when the evidence pointed at graph capture. **It is wrong
as a root-cause claim.** Two later runs disproved it:

| run | CUDA graphs | result |
|---|---|---|
| 3410366 | off via `--enforce-eager` | `DeepEP error: timeout (dispatch CPU)` -> hang |
| 3410404 | off via the hoisted guard (fired 12x, verified in the log) | `DeepEP timeout check failed: rank = 0, thread = 0, value = 0` -> hang |

HT hangs with graph capture fully disabled. The missing guard is a **real and separate**
vLLM bug (see below) but it is not why HT fails here.

### What HT's failure is NOT (all tested, all eliminated)
smem size; GPU arch; rank count / `num_rdma_ranks`; vLLM's Buffer kwargs; `num_sms=20`;
buffer capacity; token count (512 / 8192); FP8 dispatch; 30 repeated iterations;
CUDA graphs (two ways); **zero-token ranks**; NUMA binding; **`num_qps_per_rank`**.

`num_qps_per_rank` was the best remaining lead and it is dead: vLLM derives 10
(`num_sms // 2`) where Megatron gets UCCL's default of 24, and Megatron's HT works here.
Patched vLLM to 24 (`container/patches/ht_num_qps_24.py`, applied at runtime, verified in
the log on every node) -- identical hang:
```
DeepEP timeout check failed: rank = 0, thread = 0, value = 0
DeepEP timeout check failed: rank = 0, thread = 2, value = 1024
```
Note the mix: some proxy threads receive data, others get nothing, then everything stalls.

The zero-token test is worth recording because the hypothesis was attractive and wrong:
vLLM's own `modular_kernel.py` notes that a rank receiving no tokens "is only relevant for
CUDAGraph incompatible all2all kernels like the DeepEP high-throughput kernels". Forcing
15 of 16 ranks to receive zero tokens (`repro_vllm_buffer.py --starve-ranks`, routing
touching only 8 experts) still **passed** dispatch and combine.

### The one thing that still distinguishes the failing case
`swiss-ai`/upstream UCCL HT works in **Megatron training** (`fused_a2a.py` falls back to
plain `buffer.dispatch()`/`combine()` because the UCCL wrapper has no `HybridEPBuffer`,
and it uses `allocate_on_comm_stream` too -- so API, async and stream handling all match
vLLM). Every standalone reproduction passes. Only vLLM hangs. The remaining untested
differences are vLLM's **gloo `cpu_group`** for Buffer construction (Megatron passes its
EP process group) and vLLM's **flat 1 GB buffer sizing** vs Megatron's config-derived
sizes (`Buffer.get_dispatch_config(16).get_*_buffer_size_hint()` returns ~165 MB / ~161 MB
for hidden=6144 at 16 ranks; vLLM passes 1024 MB for both).

**Recommendation: stop here.** `deepep_low_latency` works, is what uccl#926 recommends for
MLA/DSA models, and is the only backend that fits 256k context (20.9 GiB KV vs 7.4 GiB for
allgather_reducescatter). HT would buy some prefill throughput for a great deal more
debugging in someone else's code.

## Original (incorrect) root-cause section: vLLM captures a CUDA graph it knows it cannot capture

vLLM documents the incompatibility in `fused_moe/modular_kernel.py`:

> "... this is only relevant for CUDAGraph **incompatible** all2all kernels like the DeepEP
> high-throughput kernels. CUDAGraph **compatible** all2all kernels like the DeepEP
> low-latency kernels are always batched ..."

DeepEP HT drives CPU-side proxy threads. CUDA graph capture requires all work to be
enqueued on the capturing stream, so a host thread that must observe GPU state and respond
mid-capture cannot be recorded -- the capture stalls with "unjoined work" while the proxies
spin (hence GPU 0% / CPU 240%).

**But vLLM 0.29.0 never acts on this.** Verified in the installed tree:
- `use_deepep_ht_kernels` is consumed only in `experts/cutlass_moe.py`,
  `experts/trtllm_bf16_moe.py`, `fused_moe/config.py`, `fused_moe/all2all_utils.py` --
  **none of them touch graph capture**;
- every site that forces `cudagraph_mode = NONE` (`platforms/xpu.py`, `platforms/rocm.py`,
  `config/vllm.py:1376,1480,1586`) keys off platform or other config, **never the all2all
  backend**;
- the only `deepep_high_throughput` reference in `config/vllm.py` is an unrelated ROCm DBO
  check.

vLLM PR #46887 ("[WideEP] Remove DeepEP high-throughput backend, redirect to DeepEP v2")
lists "Remove CUDAGraph disable block for deepep_high_throughput" among its deletions --
so such a guard exists in some version, but **not in 0.29.0**. That missing guard is the bug.

## Workaround: tested, does NOT work

`--enforce-eager` (`ENFORCE_EAGER=1`) and the hoisted guard
(`container/patches/fix_deepep_ht_cudagraph_guard.py`) both disable graph capture
successfully and **neither fixes the hang**.

The guard patch is still worth keeping and reporting upstream on its own merits: the
upstream check is gated on `CompilationMode`, while the hazard depends on
`cudagraph_mode`, so any model using breakable CUDA graphs (GLM-5.3 is auto-opted in via
`DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES`) slips past it silently. Present in vLLM main
as well as 0.29.0.

## Is HT even wanted here?

Probably not. uccl-project/uccl#926 benchmarks `deepep_ll` against `allgather_reducescatter`
across MoE families and finds LL is the right backend for MLA/DSA models, which GLM-5.3 is:

| model | arch | deepep_ll TPOT | allgather_rs TPOT |
|---|---|---:|---:|
| DeepSeek-V3-0324 | MLA | 24.3 ms | 43.8 ms |
| **GLM-5-FP8** | **DSA/MLA** | **28.58 ms** | 47.41 ms |
| GLM-4.7-FP8 | GQA | 94.79 ms | 24.74 ms (LL 3.8x worse) |

It also cites "CUDA graphs supported" as a specific LL advantage -- consistent with the
above. vLLM upstream is separately moving to delete HT in favour of DeepEP v2 (#46887),
which the UCCL wrapper does not implement (`deep_ep.ElasticBuffer` is absent).

## Related upstream issues
- uccl#923 - silent deadlock, 256 experts + FP8 + EP=16 (open)
- uccl#926 - deepep_ll dominates for MLA/DSA models (open)
- uccl#1043 - accept vLLM `enable_shrink` argument (merged; why upstream UCCL is required)
- vllm#46887 - remove DeepEP HT, redirect to DeepEP v2 (open)
