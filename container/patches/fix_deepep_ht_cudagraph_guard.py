"""Make vLLM's DeepEP high-throughput CUDA-graph guard reachable.

`CompilationConfig.set_splitting_ops_for_v1()` disables CUDA graphs for
deepep_high_throughput, but ~100 lines above that it returns early unless
`self.mode == CompilationMode.VLLM_COMPILE`. The guard is thus keyed on the compilation
mode while the hazard depends on cudagraph_mode -- and breakable CUDA graphs (auto-enabled
for GlmMoeDsaForCausalLM) deliberately set CompilationMode.NONE while still capturing.

Hoists the guard above the early return. The upstream copy below then finds cudagraph_mode
already NONE and does nothing. Present in vLLM main as well as 0.29.0.

NB: this fixes the missing guard, not the HT hang -- see docs/deepep-high-throughput.md.
"""

import sys

EARLY_RETURN = (
    "        # To compatible with OOT hardware plugin platform (for example vllm-ascend)\n"
    "        # which currently only supports sequence parallelism in eager mode.\n"
    "        if self.mode != CompilationMode.VLLM_COMPILE:\n"
)

HOISTED = (
    "        # PATCHED (Alps): hoisted above the CompilationMode early-return below.\n"
    "        # DeepEP high-throughput is not CUDA-graph compatible, but the upstream copy\n"
    "        # of this check sits after `if self.mode != CompilationMode.VLLM_COMPILE:\n"
    "        # return`, so it never runs when graphs are captured without torch.compile\n"
    "        # (the breakable-cudagraph path sets CompilationMode.NONE deliberately).\n"
    "        # The hazard depends on cudagraph_mode, not on the compilation mode.\n"
    "        if (\n"
    '            all2all_backend == "deepep_high_throughput"\n'
    "            and data_parallel_size > 1\n"
    "            and self.cudagraph_mode != CUDAGraphMode.NONE\n"
    "        ):\n"
    "            logger.info(\n"
    '                "DeepEP: Disabling CUDA Graphs since DeepEP high-throughput kernels "\n'
    '                "are optimized for prefill and are incompatible with CUDA Graphs."\n'
    "            )\n"
    "            self.cudagraph_mode = CUDAGraphMode.NONE\n"
    "\n"
)


def main() -> int:
    path = sys.argv[1]
    src = open(path).read()

    if "PATCHED (Alps): hoisted above" in src:
        print("fix_deepep_ht_cudagraph_guard: already present, nothing to do")
        return 0

    if EARLY_RETURN not in src:
        raise SystemExit(
            "fix_deepep_ht_cudagraph_guard: set_splitting_ops_for_v1 early return "
            "not in the expected shape"
        )
    if 'all2all_backend == "deepep_high_throughput"' not in src:
        raise SystemExit(
            "fix_deepep_ht_cudagraph_guard: upstream HT guard not found; refusing to patch"
        )

    src = src.replace(EARLY_RETURN, HOISTED + EARLY_RETURN, 1)
    open(path, "w").write(src)
    print("fix_deepep_ht_cudagraph_guard: patched", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
