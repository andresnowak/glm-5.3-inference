"""Make vLLM's cuteDSL ll_bf16 router GEMM check for `quack` before claiming availability.

`is_available()` probes only cutlass, but the module also imports `quack` lazily in
`_fake_gemm_tensors()`. The Alps base filters quack-kernels out of requirements while
nvidia-cutlass-dsl still arrives from NGC, so the probe returns True and the MoE router
dies on the first small batch (x.shape[0] <= 16). Teaching the probe the truth lets the
router fall back to its Tier-2/3 paths.
"""

import sys

OLD = (
    "        import cutlass  # noqa: F401\n"
    "        import cutlass.cute  # noqa: F401\n"
)
NEW = (
    "        import cutlass  # noqa: F401\n"
    "        import cutlass.cute  # noqa: F401\n"
    "        import quack  # noqa: F401  # imported lazily by _fake_gemm_tensors()\n"
)


def main() -> int:
    path = sys.argv[1]
    src = open(path).read()

    if "import quack" in src:
        print("fix_ll_bf16_availability: already present, nothing to do")
        return 0

    if OLD not in src:
        raise SystemExit("fix_ll_bf16_availability: is_available() not in the expected shape")

    src = src.replace(OLD, NEW, 1)
    open(path, "w").write(src)
    print("fix_ll_bf16_availability: patched", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
