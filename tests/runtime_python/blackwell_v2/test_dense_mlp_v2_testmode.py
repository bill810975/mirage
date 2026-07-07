"""Pytest entry for the FUSED DENSE-MLP megakernel in Runtime-V2
(dsv3_dense_mlp_fused_v2, M5 of the DSv3-decode-on-v2 port).

Builds the v2 fused dense MLP as a real megakernel (graph build -> JIT ->
run), feeds fp8 weights + raw f32 block scales, and compares the bf16 output vs
a PyTorch reference (rmsnorm -> gate_up fp8 GEMV -> silu[384-interleave] ->
down fp8 GEMV). The fused kernel UE8M0-rounds activations, so the gate is
HIGH-COSINE (>=0.999) NOT bit-identical.

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


def test_dense_mlp_v2_vs_torch(tmp_path):
    from dsv3_dense_mlp_v2_check import run_one

    m = run_one(seed=20260707, out_dir=str(tmp_path))
    assert m["PASS"], f"dense_mlp_v2 failed vs torch reference: {m}"
    # explicit cosine floor for a clearer failure message.
    assert m["cos"] >= 0.999, f"cosine {m['cos']} < 0.999: {m}"
