#!/bin/bash
# One-GPU import and API smoke test for the ROCm vLLM + UCCL-EP image.
set -u
rc=0

echo "=== ROCm stack ==="
python - <<'PY' || rc=1
import importlib.metadata as md
import torch

print("torch", md.version("torch"), "hip", torch.version.hip)
print("vllm", md.version("vllm"))
print("devices", torch.cuda.device_count(), torch.cuda.get_device_name(0))
assert torch.version.hip, "not a ROCm torch build"
assert torch.cuda.is_available(), "no ROCm GPU"
PY

echo "=== UCCL and DeepEP API ==="
python - <<'PY' || rc=1
import inspect
import deep_ep
from uccl import ep

print("uccl.ep", ep.__file__)
print("deep_ep", deep_ep.__file__)
for name in ("Buffer", "Config", "EventOverlap", "EventHandle"):
    assert hasattr(deep_ep, name), f"deep_ep missing {name}"
for name in (
    "get_low_latency_rdma_size_hint",
    "set_num_sms",
    "low_latency_dispatch",
    "low_latency_combine",
    "get_dispatch_layout",
    "dispatch",
    "combine",
):
    assert hasattr(deep_ep.Buffer, name), f"Buffer missing {name}"
print("Buffer.__init__", inspect.signature(deep_ep.Buffer.__init__))
print("ElasticBuffer", hasattr(deep_ep, "ElasticBuffer"), "(expected false)")
PY

echo "=== transport ==="
echo "FI_PROVIDER=${FI_PROVIDER:-<unset>} UCCL_EP_TRANSPORT=${UCCL_EP_TRANSPORT:-<unset>}"
[[ "${FI_PROVIDER:-}" == "cxi" ]] || rc=1
[[ "${UCCL_EP_TRANSPORT:-}" == "cxi" ]] || rc=1

[[ $rc -eq 0 ]] && echo "ROCm SMOKE OK" || echo "ROCm SMOKE FAILED"
exit "$rc"
