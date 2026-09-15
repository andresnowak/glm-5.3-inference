# GLM-5.3 on Alps GH200 — measured results

All numbers below are from this repo's scripts on Clariden, 2026-09-15.

**Topology unless stated otherwise:** 4 nodes x 4 GH200 (95.0 GiB each), `--tensor-parallel-size 2`,
`--data-parallel-size 8`, `--enable-expert-parallel` => **EP=16**, 16 of 256 experts per GPU.
fp8 KV cache, `--numa-bind`, upstream UCCL `e22a3e8f`.

Raw data: `bench_suite.tsv`, `bench_prefill.tsv`.

---

## 1. Which all2all backend?

There is no single winner. The three differ in *what they are good at*, and one of them
cannot run long context at all.

| | `deepep_low_latency` | `deepep_high_throughput` | `allgather_reducescatter` |
|---|---|---|---|
| MoE kernel chosen | BATCHED_DEEPGEMM | DEEPGEMM (forced) | FLASHINFER_CUTLASS |
| TPOT @ conc 1 | **42 ms** | 241 ms | — |
| TPOT @ conc 16 | 64 ms | 248 ms | **45.6 ms** (32k) |
| TTFT, 64k prompt | 34.2 s | **12.0 s** | — |
| KV cache / GPU | **20.9 GiB** | 10.1 GiB | 7.4 GiB |
| 256k context | **yes** | no | **no** (needs 12.86 GiB, hasn't got it) |
| works out of the box | yes | no (see below) | yes |

**Default to `deepep_low_latency`.** It is the only backend that fits a long context, and
its decode is 5.7x faster than HT.

`allgather_reducescatter` decodes slightly faster than LL at short context (45.6 vs 64.9 ms
TPOT at 32k) but allgathers every token to every rank, so its activation buffers leave only
7.4 GiB for KV and it **fails to start at 256k**:
```
ValueError: To serve at least one request with the model's max seq len (256000),
12.86 GiB KV cache is needed, which is larger than the available KV cache memory
```
This inverts uccl-project/uccl#926, which measured LL 1.66x *faster* than AGRS for GLM-5 on
2x8-GPU H200 nodes. The difference is node width: EP=16 over 4-GPU nodes spans four hosts,
so DeepEP pays twice the Slingshot hops it pays on 8-GPU nodes.

## 2. Concurrency scaling (LL, 1024-in / 128-out)

| conc | req/s | out tok/s | total tok/s | TTFT ms | TPOT ms | P99 TPOT |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.15 | 19.5 | 178 | 1215 | 42.0 | 43.6 |
| 4 | 0.56 | 71.9 | 654 | 1341 | 45.4 | 48.8 |
| 16 | 1.58 | 201.6 | 1834 | 2023 | 63.6 | 73.3 |
| 32 | 2.25 | 287.5 | 2615 | 2565 | 90.9 | 117.7 |
| 64 | 3.31 | 423.1 | 3848 | 3575 | 122.3 | 145.6 |

- Single-stream is **~24 tok/s** (1/TPOT); 19.5 tok/s end-to-end including the 1.2 s prefill.
- Scaling is near-linear to 4 (3.7x for 4x), then falls off (16->64 gives 2.1x for 4x).
- **conc 16 is the sweet spot**: 10x the throughput of single-stream for 1.5x the latency.

HT's TPOT is *flat* at ~246 ms across all concurrencies, where LL's degrades 42 -> 122 ms.

## 3. Prefill: HT is 2.4-2.8x faster, and the gap grows

Concurrency 1, 8 output tokens, so this is essentially pure time-to-first-token.

| prompt | HT TTFT | LL TTFT | HT speedup | saving | HT tok/s | LL tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1k | 0.81 s | 1.21 s | 1.49x | 0.4 s | 418 | 689 |
| 4k | 1.38 s | 3.38 s | 2.45x | 2.0 s | 1355 | 1119 |
| 16k | 4.14 s | 10.81 s | 2.61x | 6.7 s | 2828 | 1480 |
| 32k | 8.04 s | 19.23 s | 2.39x | 11.2 s | 3384 | 1682 |
| **64k** | **12.01 s** | **34.16 s** | **2.84x** | **22.2 s** | **4802** | 1906 |

At 1k the difference is 0.4 s and not worth any complexity. At 64k it is **22 seconds**, and
HT's prefill throughput is still climbing (4802 tok/s) where LL's has flattened (1906).

## 4. The case for prefill/decode disaggregation

Neither backend is good at both halves, so the useful deployment splits them.
For a 64k prompt generating 500 tokens:

```
LL only:                    34.16 + 500 x 0.035          =  51.7 s
HT only:                    12.01 + 500 x 0.235          = 129.5 s
HT prefill -> LL decode:    12.01 + 0.15 + 500 x 0.042   =  33.2 s
```

**1.56x faster than the best single configuration.** The 0.15 s is the KV transfer: GLM-5.3's
MLA KV is ~52 KB/token (20.87 GiB / 415k tokens), so 64k of context is ~3.3 GB, about 0.1 s
at the 26-41 GB/s UCCL measured over Slingshot.

`nixl==1.3.0` is in the image for this; the connector wiring is not yet written.

## 5. Memory (per GPU, of 95.0 GiB)

| item | LL | HT (util 0.80) |
|---|---:|---:|
| weights + non-torch | 60.99 GiB | ~61 GiB |
| peak activation | 1.97 GiB | larger (OOMs at util 0.90) |
| CUDA graph pool | 0.69 GiB | n/a |
| **KV cache** | **20.87 GiB** | **10.06 GiB** |
| KV in tokens | 415,296 | ~200k |

At 256k context that is **1.62x concurrency** on LL; at 32k it is 12.68x. The longest single
context that fits on 4 nodes is ~400k tokens — the native 1M window needs 8 nodes (EP=32),
where expert weights halve to 21.4 GiB/GPU and KV roughly doubles.

## 6. Weight loading

| loader | time for 703.74 GiB |
|---|---:|
| `auto` (default safetensors) | 255-434 s |
| **`runai_streamer`** | **58-105 s** |

2.4-4.4x faster; it is the default in `scripts/serve_worker.sh`. The spread is filesystem
contention — the fast end is a quiet system. Time-to-serving fell from ~660 s to ~285 s.

The checkpoint is already PFL-striped on scratch (1 / 4 / 19 stripes across the
4 MiB / 64 MiB / EOF components), so this is on top of striping, not instead of it.
