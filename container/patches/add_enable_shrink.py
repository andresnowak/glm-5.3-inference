"""Accept vLLM 0.29's `enable_shrink` kwarg in UCCL's deep_ep wrapper.

vLLM's DeepEPLLAll2AllManager passes `enable_shrink=<fault tolerance flag>`; wrapper
versions without the parameter raise TypeError before any kernel runs. Upstream UCCL has
accepted it since uccl-project/uccl#1043, so this is a no-op there and only matters for
older pins. Refuses True rather than pretending to support shrink.
"""

import sys

SIG_OLD = (
    "        explicitly_destroy: bool = False,\n"
    "        is_intranode: Optional[bool] = None,\n"
)
SIG_NEW = (
    "        explicitly_destroy: bool = False,\n"
    "        enable_shrink: bool = False,\n"
    "        is_intranode: Optional[bool] = None,\n"
)

BODY_ANCHOR = '        if "LOCAL_RANK" in os.environ:'
GUARD = (
    "        if enable_shrink:\n"
    "            raise NotImplementedError(\n"
    "                \"UCCL-EP does not implement DeepEP's shrink/elastic-EP path; \"\n"
    "                \"run without --enable-fault-tolerance.\"\n"
    "            )\n"
)


def main() -> int:
    path = sys.argv[1]
    src = open(path).read()

    if "enable_shrink" in src:
        print("add_enable_shrink: already present, nothing to do")
        return 0

    if SIG_OLD not in src:
        raise SystemExit("add_enable_shrink: Buffer.__init__ signature not in the expected shape")
    src = src.replace(SIG_OLD, SIG_NEW, 1)

    if BODY_ANCHOR not in src:
        raise SystemExit("add_enable_shrink: Buffer.__init__ body not in the expected shape")
    src = src.replace(BODY_ANCHOR, GUARD + BODY_ANCHOR, 1)

    open(path, "w").write(src)
    print("add_enable_shrink: patched", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
