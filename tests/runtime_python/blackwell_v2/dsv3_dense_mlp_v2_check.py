"""Correctness harness: the FUSED DENSE-MLP megakernel in Runtime-V2
(dsv3_dense_mlp_fused_v2, M5 of the DSv3-decode-on-v2 port) vs a PyTorch
reference on the SAME fp8 weight bytes + raw f32 block scales.

The v2 kernel is the WHOLE dense MLP for one decode token (bs=1, M=1):
  post-attn RMSNorm(self.x) -> UE8M0-quant activation -> W13(gate_up) fp8 GEMV
  -> silu(gate)*up (384-chunk interleave) -> UE8M0-requant -> W2(down) fp8 GEMV
  -> bf16 (PRE-AllReduce; the RowParallel down_proj AllReduce + residual are
  OUTSIDE the task).

The kernel UE8M0-rounds the ACTIVATION scales internally, so this is a
HIGH-COSINE (>=0.999) equivalence NOT a bit-match. The reference reuses the v1
FFN reference's fp8/UE8M0 helpers (dsv3_ffn_ref) so the quant/dequant math is
the same one the v1 group-GEMV reference uses.

Usage:
  python dsv3_dense_mlp_v2_check.py [--seed S] [--out DIR]
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dsv3_ffn_ref as FR  # noqa: E402  (fp8_decode, ue8m0_group_quant, ...)
from v2_harness import make_pk  # noqa: E402

DEV = "cuda"

# DSv3 TP8 EP2 per-rank DENSE-MLP shapes (== dsv3_dense_mlp_fused_sm100.cuh).
HIDDEN = 7168
W13_N = 4608
W2_K = 2304
SILU_OUT = 2304
CHUNK = 384
GRP = 128
KG1 = HIDDEN // GRP  # 56
KG2 = W2_K // GRP    # 18
NB1 = W13_N // GRP   # 36
NB2 = HIDDEN // GRP  # 56
# scratch = [bar u64[2] @0 | y13 f32[W13_N] @64] = 64 + 4608*4 = 18496 bytes.
SCRATCH_BYTES = 18496


def gen_weight_fp8(name, n_rows, k, seed):
    """Random bf16 weight ~ O(1/sqrt(k)) -> per-128-block UE8M0-style scale, but
    the DENSE path keeps the WEIGHT scale RAW f32 (scale_ue8m0=False). We still
    quantize with a per-block pow2 scale so |q| <= 448, then use that scale RAW
    as the kernel does. Returns (q_u8[n_rows,k], scale_f32[n_rows/128, k/128])."""
    import math

    g = torch.Generator(device=DEV).manual_seed(seed)
    w = torch.randn((n_rows, k), generator=g, device=DEV, dtype=torch.float32)
    w = w * (1.0 / math.sqrt(k))
    # per-128x128 block amax -> pow2 scale (raw f32); satfinite e4m3 cast.
    wb = w.reshape(n_rows // GRP, GRP, k // GRP, GRP)
    amax = wb.abs().amax(dim=(1, 3))  # [n/128, k/128]
    s = FR.quant_scale_ref(amax)      # raw f32 pow2 scale
    inv = 1.0 / s
    wq = wb * inv[:, None, :, None]
    wq = wq.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)
    wq = wq.reshape(n_rows, k).contiguous()
    return wq, s.contiguous()


def ref_dense_mlp(x_bf16, w13_u8, w13_s, w2_u8, w2_s, rms_w_bf16):
    """PyTorch reference of the fused dense MLP. Mirrors the kernel phases:
    rmsnorm -> bf16 -> UE8M0 quant -> W13 GEMV -> silu[384] -> UE8M0 requant ->
    W2 GEMV -> bf16. Weight scales used RAW (dense scale_ue8m0=False)."""
    # Phase A: rmsnorm, then round to bf16 (the kernel writes s_norm as bf16).
    normed = FR.ref_rmsnorm_f32(x_bf16, rms_w_bf16)          # f32 [HIDDEN]
    normed_bf16 = normed.to(torch.bfloat16).float()          # kernel bf16 store
    # Phase 0: UE8M0 per-128 quant of the bf16 normed.
    a_u8, a_s = FR.ue8m0_group_quant(normed_bf16.reshape(HIDDEN))
    # Phase 1: W13 GEMV over all 4608 rows (raw f32 weight scale).
    y13 = FR._group_gemv_ref(a_u8, a_s, w13_u8.reshape(-1), w13_s, W13_N, HIDDEN)
    # Phase 2: silu(gate)*up, 384-chunk interleave.
    y13v = y13.reshape(-1)
    silu_out = torch.empty(SILU_OUT, device=DEV, dtype=torch.float32)
    c = torch.arange(SILU_OUT, device=DEV)
    cp = c // CHUNK
    wc = c % CHUNK
    gate = y13v[cp * 768 + wc]
    up = y13v[cp * 768 + 384 + wc]
    silu_out = (gate * torch.sigmoid(gate)) * up
    # Phase 2b: UE8M0 requant of silu_out.
    i_u8, i_s = FR.ue8m0_group_quant(silu_out)
    # Phase 3: W2 GEMV over all 7168 rows -> bf16.
    out = FR._group_gemv_ref(i_u8, i_s, w2_u8.reshape(-1), w2_s, HIDDEN, W2_K)
    return out.to(torch.bfloat16)


def run_v2(x, w13_u8, w13_s, w2_u8, w2_s, rms_w, out_dir):
    """Build + run the v2 fused dense MLP on one token. Returns out bf16."""
    pk = make_pk("v2", M=1, test_mode=True)
    num_tasks = pk.num_workers
    scratch = torch.zeros((1, SCRATCH_BYTES // 2), device=DEV,
                          dtype=torch.bfloat16)
    out = torch.zeros((1, HIDDEN), device=DEV, dtype=torch.bfloat16)

    at = lambda t, nm: pk.attach_input(torch_tensor=t, name=f"dm_{nm}")
    hidden = at(x, "x")
    w13d = at(w13_u8, "w13")
    w13sd = at(w13_s, "w13s")
    w2d = at(w2_u8, "w2")
    w2sd = at(w2_s, "w2s")
    rmsd = at(rms_w, "rms")
    scr = at(scratch, "scr")
    out_dt = at(out, "out")
    pk.dsv3_dense_mlp_mega_v2_layer(
        hidden=hidden, w13=w13d, w13_scale=w13sd, w2=w2d, w2_scale=w2sd,
        rmsnorm_weight=rmsd, bar=scr, output=out_dt, num_tasks=num_tasks,
        nwarps=4)
    pk.compile(output_dir=os.path.join(out_dir, "v2_compile"))
    pk()
    torch.cuda.synchronize()
    try:
        pk.finalize()
    except Exception:  # noqa: BLE001
        pass
    return out.clone(), num_tasks


def _cmp(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    diff = (a - b).abs()
    denom = (a.norm() * b.norm()).item()
    cos = float((a @ b).item() / denom) if denom > 0 else float("nan")
    rel = float((diff / (b.abs() + 1e-6)).max().item())
    return {
        "n": a.numel(),
        "cos": cos,
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "rel_max": rel,
        "a_absmax": float(a.abs().max().item()),
        "b_absmax": float(b.abs().max().item()),
        "nan_in_out": bool(torch.isnan(a).any().item()),
    }


def run_one(seed, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn((1, HIDDEN), generator=g, device=DEV,
                    dtype=torch.float32).to(torch.bfloat16)
    rms_w = (1.0 + 0.1 * torch.randn((HIDDEN,), generator=g, device=DEV,
                                     dtype=torch.float32)).to(torch.bfloat16)
    w13_u8, w13_s = gen_weight_fp8("w13", W13_N, HIDDEN, seed + 1)
    w2_u8, w2_s = gen_weight_fp8("w2", HIDDEN, W2_K, seed + 2)

    ref = ref_dense_mlp(x, w13_u8, w13_s, w2_u8, w2_s, rms_w)
    got, num_tasks = run_v2(x, w13_u8, w13_s, w2_u8, w2_s, rms_w, out_dir)
    m = _cmp(got, ref)
    m["num_tasks"] = num_tasks
    # HIGH-COSINE gate: the fused kernel UE8M0-rounds activations (NOT
    # bit-identical). cos>=0.999 + no NaN + sane magnitude.
    m["PASS"] = bool(m["cos"] >= 0.999 and not m["nan_in_out"] and
                     m["a_absmax"] > 0.0)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260707)
    ap.add_argument("--out", type=str,
                    default=os.path.join(os.path.dirname(__file__),
                                         "_results", "dsv3_dense_mlp_v2"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    r = run_one(args.seed, args.out)
    result = {"seed": args.seed, "metrics": r, "PASS": r["PASS"]}
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(result, f, indent=1, default=str)
    print(json.dumps(result, indent=1, default=str))
    if not result["PASS"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
