"""Standalone correctness gate for the v2 tail lm_head GEMV
(dsv3_lmhead_gemv_v2) — the M3 decode-blocker fix.

Runs the op through the FULL v2 compile pipeline (Python layer method ->
register_dsv3_lmhead_gemv_v2_task -> codegen -> nvcc JIT -> runtime dispatch)
in test_mode and compares bf16 output vs the float32 torch reference
(logits = rmsnorm_out.float() @ w_lm_head.float().T).

Usage:
    python lmhead_gemv_check.py            # small + mid + production shapes
    python lmhead_gemv_check.py --only prod

Each shape is M=1 (bs=1 decode). The production shape is the real DSv3 tail:
K=7168 (hidden), N=129280 (padded vocab), block_n=128 (1010 tasks).
"""

import argparse
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2_harness import build_op, make_pk  # noqa: E402
from pytorch_reference import compare_metrics  # noqa: E402


# (name, K, N, block_n). All N % block_n == 0; block_n % (4*8) == 0.
SHAPES = {
    "small": ("lmh_small", 768, 512, 128),  # 4 tasks, quick
    "mid": ("lmh_mid", 2048, 4096, 128),  # 32 tasks
    "prod": ("lmh_prod", 7168, 129280, 128),  # 1010 tasks — real DSv3 tail
}


def run_shape(key: str, out_dir: str) -> dict:
    name, K, N, block_n = SHAPES[key]
    M = 1
    spec = {"op": "dsv3_lmhead_gemv", "name": name, "M": M,
            "N": N, "K": K, "block_n": block_n}

    pk = make_pk("v2", M, test_mode=True)
    torch_tensors = {}
    handle = build_op(pk, "v2", spec, torch_tensors)

    pk.compile(output_dir=os.path.join(out_dir, f"compile_{key}"))
    pk()
    torch.cuda.synchronize()

    out = handle["output"]
    ref = handle["ref"]()
    m = compare_metrics(out, ref)
    ok = bool(m["cos"] >= 0.999 and m["rel_max"] <= 3e-2 and not m["nan_in_out"])
    try:
        pk.finalize()
    except Exception as e:  # noqa: BLE001
        m["_finalize_warning"] = str(e)
    print(
        f"[{key}] K={K} N={N} block_n={block_n} num_tasks={N // block_n} "
        f"M={M}: cos={m['cos']:.6f} rel_max={m['rel_max']:.4e} "
        f"max_abs={m['max_abs']:.4e} nan={m['nan_in_out']} -> "
        f"{'PASS' if ok else 'FAIL'}"
    )
    print(f"    out[0,:6]={out[0, :6].float().tolist()}")
    print(f"    ref[0,:6]={ref[0, :6].float().tolist()}")
    m["_ok"] = ok
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(SHAPES.keys()), default=None)
    args = ap.parse_args()
    keys = [args.only] if args.only else ["small", "mid", "prod"]

    all_ok = True
    with tempfile.TemporaryDirectory(prefix="lmhead_gemv_") as td:
        for k in keys:
            m = run_shape(k, td)
            all_ok = all_ok and m["_ok"]
    print("=" * 60)
    print("ALL PASS" if all_ok else "SOME FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
