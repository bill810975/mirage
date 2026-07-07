"""BIT-MATCH harness: v2 FUSED attn megakernel (attn_block_megakernel_v2) vs
the v1 FUSED attn megakernel (attn_block_megakernel_sm100) on IDENTICAL input
bytes. This is the M2 exit gate for T-E of the DSv3-decode-on-v2 effort.

Both kernels are the WHOLE decode-attention block for one token; deterministic
in isolation (one CTA per head/split, no cross-CTA FP atomicAdd in the reduce),
so the correct gate is BIT-EXACT out + kv_cache[step]. The only expected
divergence is the SAME FMA-contraction that v1 has vs itself across
compilation units, so we report cos / max-abs-diff / exact-fraction and PASS on
cos>=1-1e-6 AND max_abs_diff<=a few ULP (bf16), treating a genuine bit-exact
result as the strong pass and a sub-ULP-contraction result as acceptable.

Reuses the DSv3-shaped input generators from dsv3_attn_harness so the bytes are
the calibrated production-envelope inputs. Runs each config in-process (v1 then
v2) on the same generated tensors.

Usage:
  python attn_mega_v2_bitmatch.py [--kv-offset N] [--seed S] [--out DIR]
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dsv3_attn_harness as H  # noqa: E402
import dsv3_attn_ref as R  # noqa: E402  (shape constants HIDDEN/QKHEAD/...)
from v2_harness import make_pk  # noqa: E402

DEV = "cuda"

# v1 kernel ATTN_SCRATCH_BYTES (attn_block_megakernel_sm100.cuh) — verified by
# the builder (ATTN_BLOCK_MEGAKERNEL_SCRATCH_BYTES == 434864, 0 slack). The v2
# port needs +16 bytes (the 3-slot u64 GMEM grid barrier at the top shifts the
# AttnScratch arrays up by V2_BAR_EXTRA=16).
ATTN_SCRATCH_BYTES = 434864
V2_ATTN_SCRATCH_BYTES = ATTN_SCRATCH_BYTES + 16  # 434880


def _ln_cat(w):
    return torch.cat(
        [w["input_ln_w"], w["q_a_ln_w"], w["kv_a_ln_w"]]
    ).contiguous()


def _make_pk_step(runtime, step):
    """test-mode PersistentKernel with an INJECTED decode `step` (== the MLA
    position, KV = step+1). make_pk hardwires test-mode step=0/max_seq=1; to
    exercise the multi-split MLA path (nsp>1) + narrow TPR we set step +
    max_seq_length directly. step==0 falls back to the plain make_pk."""
    if step == 0:
        return make_pk(runtime, M=1, test_mode=True)
    import mirage as mi
    from mirage.mpk.persistent_kernel import PersistentKernel

    num_workers, num_schedulers = mi.get_configurations_from_gpu(0)
    p = PersistentKernel.get_default_init_parameters()
    p["num_workers"] = num_workers
    p["num_local_schedulers"] = num_schedulers
    p["use_cutlass_kernel"] = True
    p["use_v2_runtime"] = runtime == "v2"
    p["max_num_batched_tokens"] = 1
    p["max_num_batched_requests"] = 1
    p["profiler_tensor"] = None
    p["trace_name"] = "attn_bm"
    p["test_mode"] = True
    S = step + 8
    p["max_seq_length"] = S
    p["max_num_pages"] = max(1, (S + 4095) // 4096)
    p["page_size"] = 4096
    p["meta_tensors"] = {
        "prompt_lengths": torch.tensor([1], dtype=torch.int32, device=DEV),
        "step": torch.tensor([step], dtype=torch.int32, device=DEV),
    }
    return PersistentKernel(**p)


def run_v1(weights, cos_sin, kv_prefill, kvo, out_dir, step=0):
    """Build + run the v1 FUSED attn mega on one token. Returns (out, kv_row)."""
    pk = _make_pk_step("v1", step)
    scratch = torch.zeros(
        (1, ATTN_SCRATCH_BYTES // 2), device=DEV, dtype=torch.bfloat16
    )
    kv_cache = torch.zeros((H.KV_ROWS, R.QKHEAD), device=DEV,
                           dtype=torch.bfloat16)
    if kvo > 0:
        kv_cache[:kvo] = kv_prefill
    out = torch.zeros((1, R.HIDDEN), device=DEV, dtype=torch.bfloat16)
    ln_cat = _ln_cat(weights)

    at = lambda t, nm: pk.attach_input(torch_tensor=t, name=f"v1_{nm}")
    hidden = at(weights["x"], "x")
    qkv_a_w = at(weights["qkv_a_w"], "qkvaw")
    qkv_a_s = at(weights["qkv_a_s"], "qkvas")
    ln_w = at(ln_cat, "lnw")
    q_b_w = at(weights["q_b_w"], "qbw")
    q_b_s = at(weights["q_b_s"], "qbs")
    cs = at(cos_sin, "cs")
    kv = at(kv_cache, "kv")
    kvbv_w = at(weights["kvbv_w"], "kvbvw")
    kvbv_s = at(weights["kvbv_s"], "kvbvs")
    oproj_w = at(weights["oproj_w"], "opw")
    oproj_s = at(weights["oproj_s"], "ops")
    # v1 fuses residual; at TP8 the builder binds a ZERO residual. For the
    # bit-match we must feed BOTH kernels the SAME residual buffer — use the
    # real x so the o_proj residual add is exercised identically in v1 & v2.
    resid = at(weights["x"], "resid")
    scr = at(scratch, "scr")
    out_dt = at(out, "out")
    pk.attn_block_megakernel_layer(
        hidden=hidden, qkv_a_w=qkv_a_w, qkv_a_s=qkv_a_s, ln_weights=ln_w,
        q_b_w=q_b_w, q_b_s=q_b_s, cos_sin=cs, kv_cache=kv, kvbv_w=kvbv_w,
        kvbv_s=kvbv_s, oproj_w=oproj_w, oproj_s=oproj_s, residual=resid,
        out=out_dt, scratch=scr)
    pk.compile(output_dir=os.path.join(out_dir, "v1_compile"))
    pk()
    torch.cuda.synchronize()
    row = kv_cache[step].clone()
    try:
        pk.finalize()
    except Exception:  # noqa: BLE001
        pass
    return out.clone(), row


def run_v2(weights, cos_sin, kv_prefill, kvo, out_dir, step=0):
    """Build + run the v2 FUSED attn mega on one token. Returns (out, kv_row,
    num_tasks)."""
    pk = _make_pk_step("v2", step)
    num_tasks = pk.num_workers
    scratch = torch.zeros(
        (1, V2_ATTN_SCRATCH_BYTES // 2), device=DEV, dtype=torch.bfloat16
    )
    kv_cache = torch.zeros((H.KV_ROWS, R.QKHEAD), device=DEV,
                           dtype=torch.bfloat16)
    if kvo > 0:
        kv_cache[:kvo] = kv_prefill
    out = torch.zeros((1, R.HIDDEN), device=DEV, dtype=torch.bfloat16)
    ln_cat = _ln_cat(weights)

    at = lambda t, nm: pk.attach_input(torch_tensor=t, name=f"v2_{nm}")
    hidden = at(weights["x"], "x")
    qkv_a_w = at(weights["qkv_a_w"], "qkvaw")
    qkv_a_s = at(weights["qkv_a_s"], "qkvas")
    ln_w = at(ln_cat, "lnw")
    q_b_w = at(weights["q_b_w"], "qbw")
    q_b_s = at(weights["q_b_s"], "qbs")
    cs = at(cos_sin, "cs")
    kv = at(kv_cache, "kv")
    kvbv_w = at(weights["kvbv_w"], "kvbvw")
    kvbv_s = at(weights["kvbv_s"], "kvbvs")
    oproj_w = at(weights["oproj_w"], "opw")
    oproj_s = at(weights["oproj_s"], "ops")
    resid = at(weights["x"], "resid")
    scr = at(scratch, "scr")
    out_dt = at(out, "out")
    pk.dsv3_attn_mega_layer(
        hidden=hidden, qkv_a_w=qkv_a_w, qkv_a_s=qkv_a_s, ln_weights=ln_w,
        q_b_w=q_b_w, q_b_s=q_b_s, cos_sin=cs, kv_cache=kv, kvbv_w=kvbv_w,
        kvbv_s=kvbv_s, oproj_w=oproj_w, oproj_s=oproj_s, residual=resid,
        out=out_dt, scratch=scr, num_tasks=num_tasks)
    pk.compile(output_dir=os.path.join(out_dir, "v2_compile"))
    pk()
    torch.cuda.synchronize()
    row = kv_cache[step].clone()
    try:
        pk.finalize()
    except Exception:  # noqa: BLE001
        pass
    return out.clone(), row, num_tasks


def _cmp(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    exact = int((a == b).sum().item())
    n = a.numel()
    diff = (a - b).abs()
    denom = (a.norm() * b.norm()).item()
    cos = float((a @ b).item() / denom) if denom > 0 else float("nan")
    return {
        "n": n,
        "exact": exact,
        "exact_frac": exact / n,
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "cos": cos,
        "a_absmax": float(a.abs().max().item()),
    }


# PASS: byte-identical (exact_frac==1.0) is the STRONG pass and is
# unconditional — even for an all-zero comparison vector (where cos is NaN by
# 0/0). Otherwise require cos~1 + sub-few-ULP diff (the documented
# FMA-contraction caveat v1 has vs itself).
_BF16_ULP = 2.0 ** -7  # ~0.0078 relative; a few ULP on O(1-10) values


def _ok(m):
    if m["exact_frac"] == 1.0:  # bit-identical -> pass (NaN cos is 0/0)
        return True
    return (m["cos"] >= 1 - 1e-6 and
            m["max_abs_diff"] <= 8 * _BF16_ULP * max(1.0, m["a_absmax"]))


def run_one(step, seed, out_dir):
    """One bit-match config at decode position `step` (KV = step+1). Prefills
    kv history [0,step) so the MLA attends over `step` history rows + the new
    row. Both v1 & v2 see IDENTICAL bytes. Returns the per-config result dict."""
    kvo = step  # prefill [0,step); kernel writes + we read row `step`.
    weights = H.gen_block_inputs(seed)
    cos_sin = H.build_cos_sin()
    kv_prefill = H.gen_kv_prefill(seed, kvo)
    sub = os.path.join(out_dir, f"step{step}")
    os.makedirs(sub, exist_ok=True)

    t0 = time.time()
    v1_out, v1_row = run_v1(weights, cos_sin, kv_prefill, kvo, sub, step=step)
    t1 = time.time()
    v2_out, v2_row, num_tasks = run_v2(weights, cos_sin, kv_prefill, kvo, sub,
                                       step=step)
    t2 = time.time()

    # nsp / TPR this config exercises (mirrors the kernel math) — documents
    # WHICH MLA path was covered (single-split TPR=256 at step 0; multi-split
    # narrow-TPR at large step).
    kv = step + 1
    nsp = min(8, max(1, (kv + 63) // 64))
    tile = (kv + nsp - 1) // nsp
    tpr = 256 // max(1, tile)
    tpr = 8 if tpr >= 8 else (4 if tpr >= 4 else (2 if tpr >= 2 else 1))

    out_m = _cmp(v1_out, v2_out)
    row_m = _cmp(v1_row, v2_row)
    r = {
        "step": step, "kv": kv, "nsp": nsp, "tile": tile, "TPR": tpr,
        "num_tasks": num_tasks, "v1_run_s": t1 - t0, "v2_run_s": t2 - t1,
        "out": out_m, "kv_row": row_m,
        "out_bit_exact": out_m["exact_frac"] == 1.0,
        "kv_row_bit_exact": row_m["exact_frac"] == 1.0,
        "out_pass": bool(_ok(out_m)), "kv_row_pass": bool(_ok(row_m)),
        "PASS": bool(_ok(out_m) and _ok(row_m)),
    }
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=str, default="0,513",
                    help="comma-sep decode positions to bit-match. 0 = "
                         "single-split KV=1 TPR=256; 513 = multi-split nsp=8 "
                         "TPR=2 (exercises the A/B score/lmax/lsum + merge).")
    ap.add_argument("--seed", type=int, default=20260706)
    ap.add_argument("--out", type=str,
                    default=os.path.join(os.path.dirname(__file__),
                                         "_results", "attn_mega_v2_bitmatch"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    steps = [int(s) for s in args.steps.split(",") if s.strip() != ""]
    configs = [run_one(st, args.seed, args.out) for st in steps]
    result = {
        "seed": args.seed,
        "steps": steps,
        "configs": configs,
        "PASS": all(c["PASS"] for c in configs),
    }
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(result, f, indent=1, default=str)
    print(json.dumps(result, indent=1, default=str))
    if not result["PASS"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
