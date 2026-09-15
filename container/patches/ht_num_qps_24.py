"""Experiment: give vLLM's DeepEP HT manager Megatron's QP count (24 instead of num_sms//2).

vLLM derives num_qps_per_rank from its SM count (10); Megatron passes nothing and gets
UCCL's default of 24. Tested and ruled out as a cause of the HT hang -- kept because it is
the only way to vary the QP count without editing the image.
"""

import sys

OLD = "            num_qps_per_rank = self.num_sms // 2\n"
NEW = (
    "            # PATCHED (Alps experiment): match the UCCL/Megatron default of 24\n"
    "            # instead of num_sms // 2 == 10.\n"
    "            num_qps_per_rank = 24\n"
)


def main() -> int:
    path = sys.argv[1]
    src = open(path).read()
    if "PATCHED (Alps experiment)" in src:
        print("ht_num_qps_24: already present, nothing to do")
        return 0
    if OLD not in src:
        raise SystemExit("ht_num_qps_24: num_qps_per_rank assignment not in expected shape")
    src = src.replace(OLD, NEW, 1)
    open(path, "w").write(src)
    print("ht_num_qps_24: patched", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
