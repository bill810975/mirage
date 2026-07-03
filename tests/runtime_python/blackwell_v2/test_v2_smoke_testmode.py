"""Pytest smoke entry for the Runtime-V2 framework (single fast case).

Full matrices go through run_suite.py (subprocess isolation, GPU pinning,
timeouts); this file keeps a pytest-shaped smoke so `pytest
tests/runtime_python/blackwell_v2/` exercises the v2 pipeline end-to-end
(graph build -> JIT -> run -> compare) in one process.

Runtime: one JIT compile (~30-60 s on B200). Requires a free GPU; pin with
CUDA_VISIBLE_DEVICES.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA GPU required", allow_module_level=True)


@pytest.mark.parametrize("runtime", ["v2"])
def test_v2_elementwise_and_linear_smoke(runtime, tmp_path):
    from v2_harness import run_correctness_case

    spec = {
        "runtime": runtime,
        "M": 8,
        "only": ["rms_h4096", "silu_i12288_g48", "lin3_sq"],
        "mode": "correctness",
    }
    result = run_correctness_case(spec, str(tmp_path))
    failures = {
        op: m
        for op, m in result["ops"].items()
        if isinstance(m, dict) and not m.get("pass_vs_torch")
    }
    assert not failures, f"ops failed vs torch reference: {failures}"
