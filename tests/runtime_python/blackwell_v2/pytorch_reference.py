"""Canonical PyTorch references for the Runtime-V2 per-task test framework.

Every reference computes in float32 and casts back to the I/O dtype at the
end, so the reference is strictly more precise than the bf16 kernels under
test. One function per task family; both the correctness harness and any
future kernel-wrapper tests must import from here (single source of truth).
"""

import torch


def ref_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm as implemented by rmsnorm_v2.cuh / rmsnorm_hopper.

    NOTE the kernel adds eps AFTER the mean (rsqrt(mean(x^2) + eps)), matches
    rmsnorm_v2.cuh line ~159: rsqrt(reduce/HIDDEN + eps).
    """
    xf = x.to(torch.float32)
    rms_rcp = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * rms_rcp * w.to(torch.float32)).to(x.dtype)


def ref_silu_mul(x: torch.Tensor, num_groups: int) -> torch.Tensor:
    """SiLU-mul with the MPK grouped column layout.

    The MPK graph partitions the input's 2*I columns across `num_groups`
    tasks; each task sees a contiguous slice [gate_g | up_g] of width
    2*I/num_groups and computes silu(gate_g) * up_g (silu_mul_v2.cuh:
    d_mul = d_input + OUTPUT_SIZE where OUTPUT_SIZE is the PER-TASK width).
    With num_groups == 1 this degenerates to the standard [gate | up] concat.
    """
    M, two_i = x.shape
    assert two_i % (2 * num_groups) == 0
    per = two_i // num_groups  # slice width per group = 2*I/num_groups
    half = per // 2
    xf = x.to(torch.float32).view(M, num_groups, per)
    gate = xf[:, :, :half]
    up = xf[:, :, half:]
    out = torch.nn.functional.silu(gate) * up
    return out.reshape(M, two_i // 2).to(x.dtype)


def ref_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """out = x @ w.T  (weight stored [N, K] row-major, same as HF Linear)."""
    return (x.to(torch.float32) @ w.to(torch.float32).T).to(x.dtype)


def ref_linear_residual(
    x: torch.Tensor, w: torch.Tensor, residual: torch.Tensor
) -> torch.Tensor:
    """out = x @ w.T + residual."""
    acc = x.to(torch.float32) @ w.to(torch.float32).T
    return (acc + residual.to(torch.float32)).to(x.dtype)


def ref_embedding(tokens: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """out[i] = w[tokens[i]]."""
    return w[tokens.view(-1).long()]


def compare_metrics(out: torch.Tensor, ref: torch.Tensor) -> dict:
    """Uniform correctness metrics between a kernel output and a reference.

    Returns cos (fp64 flattened cosine), max_abs, rel_max (max_abs scaled by
    ref max magnitude), rel_l2, bitexact (bf16 bit equality fraction).
    """
    o = out.detach().to(torch.float64).flatten()
    r = ref.detach().to(torch.float64).flatten()
    denom = o.norm() * r.norm()
    cos = float((o @ r) / denom) if denom > 0 else (1.0 if o.equal(r) else 0.0)
    diff = (o - r).abs()
    max_abs = float(diff.max())
    ref_scale = float(r.abs().max())
    rel_max = max_abs / ref_scale if ref_scale > 0 else max_abs
    rel_l2 = float(diff.norm() / (r.norm() + 1e-30))
    same_bits = (
        out.view(torch.int16) == ref.view(torch.int16)
        if out.dtype == torch.bfloat16 and ref.dtype == torch.bfloat16
        else (out == ref)
    )
    bitexact_frac = float(same_bits.float().mean())
    return {
        "cos": cos,
        "max_abs": max_abs,
        "rel_max": rel_max,
        "rel_l2": rel_l2,
        "bitexact_frac": bitexact_frac,
        "nan_in_out": bool(torch.isnan(out).any()),
    }
