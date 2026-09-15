#!/bin/bash
# Import smoke test for the vLLM+UCCL image. Needs a GPU, so it runs on a compute
# node rather than in the build (podman build has no GPU and uccl.ep links the
# CUDA driver).
#
#   srun ... --environment=/users/anowak/developer/glm-5.3/edf/vllm.toml \
#        bash /users/anowak/developer/glm-5.3/container/tests/import_smoke.sh
set -u
rc=0

echo "=== interpreter ==="
python3 -VV
python3 -c "import sys; print('user site enabled:', __import__('site').ENABLE_USER_SITE)"

echo "=== base stack unmoved ==="
python3 - <<'PY' || rc=1
import importlib.metadata as md
for p in ("torch", "vllm", "flashinfer-python"):
    print(f"  {p}={md.version(p)}")
import torch
print("  torch.cuda:", torch.version.cuda, "devices:", torch.cuda.device_count())
assert torch.cuda.is_available(), "no CUDA device"
PY

echo "=== uccl.ep loads ==="
python3 - <<'PY' || rc=1
from uccl import ep
print("  uccl.ep:", ep.__file__)
print("  has Buffer:", hasattr(ep, "Buffer"), "| has UcclProxy:", hasattr(ep, "UcclProxy"))
PY

echo "=== deep_ep presents DeepEP's API ==="
python3 - <<'PY' || rc=1
import deep_ep, inspect
print("  deep_ep:", deep_ep.__file__)
need = ["Buffer", "Config", "EventOverlap", "EventHandle"]
for n in need:
    assert hasattr(deep_ep, n), f"deep_ep missing {n}"
B = deep_ep.Buffer
# the exact surface vLLM 0.29's DeepEP managers touch
for m in ("get_low_latency_rdma_size_hint", "set_num_sms", "low_latency_dispatch",
          "low_latency_combine", "get_dispatch_layout", "dispatch", "combine"):
    assert hasattr(B, m), f"deep_ep.Buffer missing {m}"
sig = inspect.signature(B.__init__)
print("  Buffer.__init__:", list(sig.parameters)[1:])
print("  ElasticBuffer (deepep_v2, expected absent):", hasattr(deep_ep, "ElasticBuffer"))
PY

echo "=== vLLM sees deep_ep ==="
python3 - <<'PY' || rc=1
from vllm.utils.import_utils import has_deep_ep
print("  has_deep_ep():", has_deep_ep())
assert has_deep_ep(), "vLLM does not see deep_ep"
from vllm.utils.deep_gemm import has_deep_gemm
print("  has_deep_gemm():", has_deep_gemm(), "(vendored in the vLLM wheel)")
PY

echo "=== transport selection ==="
echo "  UCCL_EP_TRANSPORT=${UCCL_EP_TRANSPORT:-<unset>}  (must be exactly 'cxi')"
[[ "${UCCL_EP_TRANSPORT:-}" == "cxi" ]] || { echo "  FAIL: not cxi"; rc=1; }

echo
[[ $rc -eq 0 ]] && echo "SMOKE OK" || echo "SMOKE FAILED"
exit $rc
