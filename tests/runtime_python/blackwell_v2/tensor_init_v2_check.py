"""Standalone correctness gate for `tensor_init_v2` (T-A of the DSv3-decode-on-v2
effort).

Why standalone (not folded into run_suite.py's build_op matrix): tensor_init is
not a 1-in-1-out elementwise op — it uses the special tensor_init_layer(target,
dummy, grid, block, input_maps) signature with a dummy dep tensor, and the
shared suite seeds every `_out` buffer to zeros (so a zeros-vs-zeros check would
pass even if the kernel did nothing). This harness instead seeds the TARGET with
a non-zero poison pattern and asserts the kernel zeroed it — proving the write
actually happened — then compares the v2 kernel byte-for-byte against the v1
kernel on the identical seeded buffer.

CASES (dtype-aware after the int64[2] byte-exact fix):
  * bf16 shapes (t_7168 / t_2x4096 / t_small) — the historical widths; gated
    BOTH on all-zero AND bit-exact vs v1 (no regression). v1-compatible
    (cols % 8 == 0), so the v1 arm compiles them.
  * t_attn_scratch — the REAL DSv3 v2 attn-block-megakernel scratch shape
    (bf16, (1,(434864+16)//2)=217440 cols), the other decode-path v2 caller.
  * t_bar_i64x2 — the FAILING case: the FFN-megakernel barrier `_ffn_bar`
    (int64[2] = 16 bytes). dim[1]=2 is an INT64 element count, so the old
    bf16-typed v2 body (OUTPUT_SIZE=2) BOTH tripped `OUTPUT_SIZE % 8 == 0` AND
    would have zeroed only 2 bf16 = 4 of the true 16 bytes. This case is NOT
    v1-compatible (v1's `static_assert(OUTPUT_SIZE % 8 == 0)` fails to compile
    for a 2-wide row), so it runs on the v2 arm ONLY and is gated on all-16-
    bytes-zero (the semantic requirement: two u64 barrier counters == 0 at
    alloc). It is seeded with a distinct 0xAA byte poison so a short-zeroed
    tail (bytes [4,16)) would leave 0xAA and FAIL.

Runs one (runtime, target-shape) case per subprocess invocation; the parent
below spawns v1 and v2 and diffs their outputs.

  # one arm (as spawned by --drive):
  python tensor_init_v2_check.py --arm v1 --out /tmp/ti_v1.pt
  python tensor_init_v2_check.py --arm v2 --out /tmp/ti_v2.pt
  # full gate (spawns both arms + compares), pin GPU 5:
  CUDA_VISIBLE_DEVICES=5 python tensor_init_v2_check.py --drive
"""

import argparse
import os
import subprocess
import sys

import torch

# Real DSv3 v2 attn-block-megakernel scratch width (builder.py:
# (ATTN_BLOCK_MEGAKERNEL_SCRATCH_BYTES + 16) // 2 with the +16 v2 barrier pad).
_ATTN_SCRATCH_COLS = (434864 + 16) // 2  # 217440 bf16 elems (== 434880 bytes)

# Each case: (name, rows, cols, torch_dtype, v1_compatible).
#   v1_compatible == the v1 kernel's `static_assert(OUTPUT_SIZE % 8 == 0)` holds
#   under bf16 accounting (cols % 8 == 0) AND the byte range v1 zeroes equals the
#   full buffer (i.e. sizeof(dtype) == 2). Only such cases are registered on the
#   v1 arm; others (the int64[2] bar) run v2-only and are gated on all-zero.
TARGET_CASES = [
    # name              rows  cols                 dtype            v1_ok
    ("t_7168",          1,    7168,                torch.bfloat16,  True),
    ("t_2x4096",        2,    4096,                torch.bfloat16,  True),
    ("t_small",         1,    256,                 torch.bfloat16,  True),
    ("t_attn_scratch",  1,    _ATTN_SCRATCH_COLS,  torch.bfloat16,  True),
    ("t_bar_i64x2",     1,    2,                   torch.int64,     False),
]

_POISON_BF16 = 0x3C00      # bf16 1.0 bit-pattern — a clearly non-zero seed
_POISON_BYTE = 0xAA        # per-byte poison for the int64 bar (tail-sensitive)


def _seeded_target(rows: int, cols: int, dtype: torch.dtype) -> torch.Tensor:
    """A (rows, cols) tensor of `dtype` pre-filled with a non-zero poison so an
    all-zero result proves the kernel wrote EVERY byte. For bf16 use the 1.0
    bit-pattern; for int64 fill every byte with 0xAA (so a short-zeroed tail is
    caught)."""
    if dtype == torch.bfloat16:
        raw = torch.full((rows, cols), _POISON_BF16, dtype=torch.int16,
                         device="cuda")
        return raw.view(torch.bfloat16)
    if dtype == torch.int64:
        # 0xAAAAAAAAAAAAAAAA as a signed int64.
        val = int.from_bytes(bytes([_POISON_BYTE]) * 8, "little", signed=True)
        return torch.full((rows, cols), val, dtype=torch.int64, device="cuda")
    raise ValueError(f"unhandled dtype {dtype}")


def _cases_for_arm(arm: str):
    """v1 arm registers only v1-compatible cases (a v1-incompatible shape would
    fail the compile-time static_assert and abort the whole megakernel nvcc)."""
    if arm == "v1":
        return [c for c in TARGET_CASES if c[4]]
    return TARGET_CASES


def run_arm(arm: str, out_path: str) -> None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from v2_harness import make_pk  # noqa: E402

    runtime = arm  # "v1" | "v2"
    M = 1
    pk = make_pk(runtime, M, test_mode=True)

    # A single dummy dep tensor shared by all init tasks (a real graph input so
    # the v2 consumer_dep_prefix has an edge to wait on; its data is untouched).
    dummy = torch.zeros((1, 8), dtype=torch.bfloat16, device="cuda")
    dummy_dt = pk.attach_input(torch_tensor=dummy, name="ti_dummy")

    targets = {}
    for nm, rows, cols, dt, _v1ok in _cases_for_arm(arm):
        t = _seeded_target(rows, cols, dt)
        targets[nm] = t
        tgt_dt = pk.attach_input(torch_tensor=t, name=nm)
        pk.tensor_init_layer(
            target=tgt_dt,
            dummy=dummy_dt,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
            dummy_input_map=(-1, -1, -1),
            target_input_map=(-1, -1, -1),
        )

    pk.compile(output_dir=os.path.join(os.path.dirname(out_path), f"compile_{arm}"))
    pk()
    torch.cuda.synchronize()

    # Save each target as raw bytes (uint8) so the comparison is byte-exact and
    # dtype-agnostic (int64 and bf16 alike).
    saved = {
        nm: targets[nm].contiguous().view(torch.uint8).detach().cpu()
        for nm, _, _, _, _ in _cases_for_arm(arm)
    }
    torch.save(saved, out_path)
    try:
        pk.finalize()
    except Exception:  # noqa: BLE001
        pass
    print(f"[{arm}] wrote {out_path}")


def drive() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    wdir = os.path.join(here, "_results", "tensor_init_v2")
    os.makedirs(wdir, exist_ok=True)
    py = sys.executable
    paths = {}
    for arm in ("v1", "v2"):
        out = os.path.join(wdir, f"out_{arm}.pt")
        paths[arm] = out
        print(f"=== running {arm} arm ===", flush=True)
        r = subprocess.run(
            [py, os.path.abspath(__file__), "--arm", arm, "--out", out],
            env={**os.environ},
        )
        if r.returncode != 0:
            print(f"ARM {arm} FAILED (exit {r.returncode})")
            return 1

    v1 = torch.load(paths["v1"])
    v2 = torch.load(paths["v2"])
    ok = True
    print("\n--- tensor_init_v2 correctness gate ---")
    print(f"{'case':<16} {'bytes':<8} {'v2_all_zero':<12} {'v1_all_zero':<12} "
          f"{'bitexact_v2_vs_v1':<18} {'verdict'}")
    for nm, _, _, _, v1ok in TARGET_CASES:
        a2 = v2[nm]
        nbytes = a2.numel()
        v2_zero = bool((a2 == 0).all())
        if v1ok:
            a1 = v1[nm]
            v1_zero = bool((a1 == 0).all())
            bit = bool((a2 == a1).all())
            v1_zero_s, bit_s = str(v1_zero), str(bit)
            passed = v2_zero and v1_zero and bit
        else:
            # v2-only case (v1 cannot compile a 2-wide/int64 row): gate on
            # all-bytes-zero, which IS bit-exact vs the "correct v1" (all-zero).
            v1_zero_s, bit_s = "n/a(v1-incompat)", "n/a"
            passed = v2_zero
        verdict = "PASS" if passed else "FAIL"
        ok = ok and passed
        print(f"{nm:<16} {nbytes:<8} {str(v2_zero):<12} {v1_zero_s:<12} "
              f"{bit_s:<18} {verdict}")
    print("\nRESULT:",
          "PASS — v2 all-zero (bit-exact vs v1 where v1 compiles)"
          if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["v1", "v2"])
    ap.add_argument("--out")
    ap.add_argument("--drive", action="store_true")
    args = ap.parse_args()
    if args.drive:
        sys.exit(drive())
    assert args.arm and args.out, "need --arm and --out (or --drive)"
    run_arm(args.arm, args.out)
