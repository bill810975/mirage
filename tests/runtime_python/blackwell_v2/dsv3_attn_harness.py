"""Case builders/runners for the DSv3 fused-ATTN v2 task chain (Step 3b).

Mirrors dsv3_ffn_harness.py: reuses the calibrated v2 framework machinery
(v2_harness.make_pk, pytorch_reference.compare_metrics, v2_prof_decode);
this module only adds the ATTN-specific graph builders, DSv3-shaped inputs,
torch references (dsv3_attn_ref) and the v1-driver input dumps.

Case kinds (dispatched by run_attn_suite.py in a subprocess per case):
  attn_correctness: test-mode single-pass, ONE attn block at a chosen
      kv_offset (step = kv_offset since iter_num == 0); per-op compare vs
      the torch refs + input dump for the v1 driver same-bytes A/B.
  attn_perf: offline-mode chain of L attn blocks (block i+1's x = block i's
      out; per-block own weights + own kv_cache => cold-L2 by footprint),
      S iterations at a chosen kv_offset, profiled (v2 role profiler
      window) or unprofiled (wall cross-check).
  attn_multistep: offline-mode ONE block, S small (e.g. 4), kv_offset 0 —
      dumps the kv rows + final tensors for the multi-step v1 A/B (proves
      the iter_num/step plumbing).

Graph-shape contract (annotated_graph case-2/3): each op's DECLARED output
is consumed ONLY by the next op; kv_cache is a hidden write of qb_rope_kv
and an ALIAS input of mla_partial; g_mla_m/l are hidden writes with aliases
into mla_merge; residual is an ALIAS of the block input x.
"""

import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dsv3_attn_ref as R
from pytorch_reference import compare_metrics
from v2_harness import make_pk

DEV = "cuda"

KV_ROWS = 4224  # covers kv_offset 4064 + S 32 + slack; multiple of 128


# ---------------------------------------------------------------------------
# Deterministic DSv3-shaped input generation.
# ---------------------------------------------------------------------------
def _gen(seed: int):
    return torch.Generator(device=DEV).manual_seed(seed)


def gen_fp8_bytes(shape, gen) -> torch.Tensor:
    """Random e4m3 bytes with the NaN encodings (x7F/xFF) remapped — same
    convention as the FFN harness / drivers."""
    b = torch.randint(0, 256, shape, generator=gen, device=DEV,
                      dtype=torch.int32)
    nan_mask = (b & 0x7F) == 0x7F
    b = torch.where(nan_mask, (b & 0x80) | 0x3E, b)
    return b.to(torch.uint8).contiguous()


def gen_pow2_scales(shape, gen, lo_exp=-9, hi_exp=-5) -> torch.Tensor:
    e = torch.randint(lo_exp, hi_exp + 1, shape, generator=gen, device=DEV)
    return torch.ldexp(torch.ones(shape, device=DEV, dtype=torch.float64),
                       e).float().contiguous()


def build_cos_sin(rows: int = KV_ROWS) -> torch.Tensor:
    """Real rotary table: inv_freq per PAIR; [cos(64)|sin(64)] per row.
    Entries d and d+1 share the pair's angle (only even d is read)."""
    inv_freq = 10000.0 ** (-torch.arange(0, 32, device=DEV,
                                         dtype=torch.float64) / 32.0)
    pos = torch.arange(rows, device=DEV, dtype=torch.float64)
    ang = pos[:, None] * inv_freq[None, :]  # (rows, 32)
    cs = torch.zeros((rows, 128), dtype=torch.float64, device=DEV)
    cs[:, 0:64:2] = torch.cos(ang)
    cs[:, 1:64:2] = torch.cos(ang)
    cs[:, 64:128:2] = torch.sin(ang)
    cs[:, 65:128:2] = torch.sin(ang)
    return cs.to(torch.bfloat16).contiguous()


def gen_block_inputs(seed: int) -> dict:
    """One attn block's weights + (for block 0) the raw hidden x. Scale
    choices keep activations O(1) and pre-softmax score std O(1-30) (checked
    and reported by the ref)."""
    gen = _gen(seed)

    def uni(shape, lo, hi, dtype=torch.float32):
        t = torch.rand(shape, generator=gen, device=DEV) * (hi - lo) + lo
        return t.to(dtype).contiguous()

    t = {
        "x": uni((1, R.HIDDEN), -1.0, 1.0, torch.bfloat16),
        "input_ln_w": uni((R.HIDDEN,), 0.8, 1.2, torch.bfloat16),
        "q_a_ln_w": uni((R.QLORA,), 0.8, 1.2, torch.bfloat16),
        "kv_a_ln_w": uni((R.KVLORA,), 0.8, 1.2, torch.bfloat16),
        # scale ranges tuned so every GEMV activation stays in the
        # production-like O(1-10) envelope (k_mac_u4's fp16 partial sums
        # overflow past |a|~36 x |w|~448 — the documented v1 caveat; random
        # e4m3 bytes have mean |w| ~ 28 with a 448 tail, so hot scales
        # push synthetic activations far beyond production's).
        "qkv_a_w": gen_fp8_bytes((R.QKVAN, R.HIDDEN), gen),
        "qkv_a_s": gen_pow2_scales((R.QKVAN // 128, R.HIDDEN // 128), gen,
                                   -11, -9),
        "q_b_w": gen_fp8_bytes((R.HLOCAL * R.QKHEAD, R.QLORA), gen),
        "q_b_s": gen_pow2_scales(
            (R.HLOCAL * R.QKHEAD // 128, R.QLORA // 128), gen, -11, -9),
        "kvbv_w": gen_fp8_bytes((R.HLOCAL, R.VHEAD, R.KVLORA), gen),
        "kvbv_s": gen_pow2_scales((R.HLOCAL, 1, R.KVLORA // 128), gen,
                                  -9, -7),
        "oproj_w": gen_fp8_bytes((R.HIDDEN, R.OIN), gen),
        "oproj_s": gen_pow2_scales((R.HIDDEN // 128, R.OIN // 128), gen,
                                   -9, -7),
    }
    return t


def gen_kv_prefill(seed: int, rows: int) -> torch.Tensor:
    """History rows [0, rows): latent part ~N(0,1) (post-rmsnorm scale),
    rope part ~N(0,3) (raw-GEMV-output scale class)."""
    gen = _gen(seed + 777)
    kv = torch.zeros((rows, R.QKHEAD), device=DEV, dtype=torch.float32)
    if rows > 0:
        kv[:, :R.KVLORA] = torch.randn((rows, R.KVLORA), generator=gen,
                                       device=DEV)
        kv[:, R.KVLORA:] = torch.randn((rows, R.QKROPE), generator=gen,
                                       device=DEV)
    return kv.to(torch.bfloat16).contiguous()


def alloc_block_buffers() -> dict:
    z = lambda shape, dt: torch.zeros(shape, device=DEV, dtype=dt)
    return {
        "g_qkva": z((R.QKVAN,), torch.float32),
        "g_qpe": z((R.HLOCAL * R.QKHEAD,), torch.float32),
        "g_mla_acc": z((R.HLOCAL, R.SPLITS, R.KVLORA), torch.float32),
        "g_mla_m": z((R.HLOCAL, R.SPLITS), torch.float32),
        "g_mla_l": z((R.HLOCAL, R.SPLITS), torch.float32),
        "g_attn": z((R.HLOCAL, R.KVLORA), torch.float32),
        "g_attn_deq": z((R.HLOCAL, R.KVLORA), torch.float32),
        "g_red": z((R.OIN,), torch.float32),
        "out": z((1, R.HIDDEN), torch.bfloat16),
        "kv_cache": z((KV_ROWS, R.QKHEAD), torch.bfloat16),
        "ready_ones": torch.ones((R.HLOCAL,), device=DEV, dtype=torch.int32),
        "g_head_done": z((R.HLOCAL,), torch.int32),
    }


# ---------------------------------------------------------------------------
# Graph builder: one attn block = 6 chained ops.
# ---------------------------------------------------------------------------
def build_attn_block(pk, prefix: str, weights: dict, bufs: dict,
                     x_dt, x_torch, cos_sin_dt, cfg: dict):
    """x_dt: DTensor for this block's input hidden (the chain edge in);
    x_torch: its torch tensor (residual = alias attach of the SAME tensor).
    Returns (out DTensor, out torch tensor)."""
    at = lambda t, nm: pk.attach_input(torch_tensor=t, name=f"{prefix}_{nm}")

    in_ln = at(weights["input_ln_w"], "inln")
    qa_ln = at(weights["q_a_ln_w"], "qaln")
    kva_ln = at(weights["kv_a_ln_w"], "kvaln")
    qkv_a_w = at(weights["qkv_a_w"], "qkvaw")
    qkv_a_s = at(weights["qkv_a_s"], "qkvas")
    q_b_w = at(weights["q_b_w"], "qbw")
    q_b_s = at(weights["q_b_s"], "qbs")
    kvbv_w = at(weights["kvbv_w"], "kvbvw")
    kvbv_s = at(weights["kvbv_s"], "kvbvs")
    oproj_w = at(weights["oproj_w"], "opw")
    oproj_s = at(weights["oproj_s"], "ops")

    g_qkva = at(bufs["g_qkva"], "gqkva")
    g_qpe = at(bufs["g_qpe"], "gqpe")
    g_mla_acc = at(bufs["g_mla_acc"], "gmlaacc")
    g_mla_m = at(bufs["g_mla_m"], "gmlam")
    g_mla_l = at(bufs["g_mla_l"], "gmlal")
    g_attn = at(bufs["g_attn"], "gattn")
    g_attn_deq = at(bufs["g_attn_deq"], "gattndeq")
    g_red = at(bufs["g_red"], "gred")
    out = at(bufs["out"], "out")
    kv_cache = at(bufs["kv_cache"], "kv")
    ready_ones = at(bufs["ready_ones"], "ones")

    # ALIASES (fresh attach of the same torch tensor -> fresh guid -> no
    # graph edge; ordering rides the chain events transitively).
    kv_alias = at(bufs["kv_cache"], "kv_alias")
    m_alias = at(bufs["g_mla_m"], "gmlam_alias")
    l_alias = at(bufs["g_mla_l"], "gmlal_alias")
    resid_alias = at(x_torch, "resid_alias")

    kvo = cfg["kv_offset"]
    fold = cfg["fold_merge"]
    # T2 gets the real attach; the fused op gets an ALIAS (a second declared
    # edge T2->T3' would double the pair's event bookkeeping — same rule as
    # kv_cache/residual).
    head_done_t2 = at(bufs["g_head_done"], "ghd") if fold else None
    head_done_alias = at(bufs["g_head_done"], "ghd_alias") if fold else None
    pk.dsv3_attn_p0_qkva_layer(
        x=x_dt, input_ln_w=in_ln, qkv_a_w=qkv_a_w, qkv_a_s=qkv_a_s,
        g_qkva=g_qkva, num_tasks=cfg["n1"], nwarps=cfg["nw_p0"])
    pk.dsv3_attn_qb_rope_kv_layer(
        g_qkva=g_qkva, q_a_ln_w=qa_ln, kv_a_ln_w=kva_ln, q_b_w=q_b_w,
        q_b_s=q_b_s, cos_sin=cos_sin_dt, kv_cache=kv_cache, g_qpe=g_qpe,
        num_tasks=cfg["n2"], kv_offset=kvo, nwarps=cfg["nw_qb"],
        g_head_done=head_done_t2)
    if fold:
        # Round-4 fold: ONE op = partial + last-arriver merge (T4 removed;
        # the T3'->T5 edge is g_attn_deq; acc/m/l/attn are hidden writes).
        pk.dsv3_attn_mla_fused_layer(
            g_qpe=g_qpe, kv_cache=kv_alias, g_mla_m=g_mla_m,
            g_mla_l=g_mla_l, g_mla_acc=g_mla_acc,
            g_head_done=head_done_alias,
            g_attn=g_attn, g_attn_deq=g_attn_deq, kv_offset=kvo,
            nwarps=cfg["nw_mla"])
    else:
        pk.dsv3_attn_mla_partial_layer(
            g_qpe=g_qpe, kv_cache=kv_alias, g_mla_m=g_mla_m, g_mla_l=g_mla_l,
            g_mla_acc=g_mla_acc, kv_offset=kvo, nwarps=cfg["nw_mla"])
        pk.dsv3_attn_mla_merge_layer(
            g_mla_acc=g_mla_acc, g_mla_m=m_alias, g_mla_l=l_alias,
            g_attn=g_attn, g_attn_deq=g_attn_deq, kv_offset=kvo)
    pk.dsv3_attn_wuv_layer(
        g_attn_deq=g_attn_deq, kvbv_w=kvbv_w, kvbv_s=kvbv_s,
        ready_ones=ready_ones, g_red=g_red, num_tasks=cfg["n5"],
        nwarps=cfg["nw_wuv"])
    pk.dsv3_attn_oproj_layer(
        g_red=g_red, oproj_w=oproj_w, oproj_s=oproj_s,
        residual=resid_alias, output=out, num_tasks=cfg["n6"],
        nwarps=cfg["nw_oproj"])
    return out, bufs["out"]


def default_cfg(spec: dict) -> dict:
    nwarps = spec.get("nwarps", 4)  # global default; per-op keys override
    # oproj's per-warp cp.async ring (OP_GEMV_RBT*32*16*STAGES = 32KB/warp)
    # exceeds the ~205KB smem budget at 7 warps — cap the DEFAULT at 4 so
    # `--nwarps 7` without an explicit oproj override can't build an
    # oversized allocation; an explicit spec["nw_oproj"] still wins.
    nw_oproj = spec.get("nw_oproj", min(nwarps, 4))
    # oproj row-blocks = 896: at nwarps=7 n6=128 makes 128*7=896 exactly one
    # block per warp (halves the GEMV wall); at 4 warps 112*4*2=896 (2 each).
    n6_default = 128 if nw_oproj == 7 else 112
    return {
        "n1": spec.get("n1", 136),
        "n2": spec.get("n2", 136),
        "n5": spec.get("n5", 128),
        "n6": spec.get("n6", n6_default),
        "kv_offset": spec.get("kv_offset", 0),
        "nwarps": nwarps,
        "nw_p0": spec.get("nw_p0", nwarps),
        "nw_qb": spec.get("nw_qb", nwarps),
        "nw_mla": spec.get("nw_mla", nwarps),
        "nw_wuv": spec.get("nw_wuv", nwarps),
        "nw_oproj": nw_oproj,
        "fold_merge": bool(spec.get("fold_merge", False)),
    }


def block_instances(i: int, cfg: dict):
    if cfg["fold_merge"]:
        return [
            (i, "attn_p0_qkva", cfg["n1"]),
            (i, "attn_qb_rope_kv", cfg["n2"]),
            (i, "attn_mla_fused", 128),
            (i, "attn_wuv", cfg["n5"]),
            (i, "attn_oproj", cfg["n6"]),
        ]
    return [
        (i, "attn_p0_qkva", cfg["n1"]),
        (i, "attn_qb_rope_kv", cfg["n2"]),
        (i, "attn_mla_partial", 128),
        (i, "attn_mla_merge", 16),
        (i, "attn_wuv", cfg["n5"]),
        (i, "attn_oproj", cfg["n6"]),
    ]


# ---------------------------------------------------------------------------
# Input dump for the v1 driver (--load-dir): raw little-endian binaries.
# ln_weights is dumped as the v1 kernel's 9216-d concat.
# ---------------------------------------------------------------------------
def dump_driver_inputs(weights: dict, cos_sin: torch.Tensor,
                       kv_prefill: torch.Tensor, dump_dir: str):
    os.makedirs(dump_dir, exist_ok=True)

    def wr(t: torch.Tensor, fname: str):
        if t.dtype == torch.bfloat16:
            raw = t.contiguous().view(torch.uint16).cpu().numpy()
        else:
            raw = t.contiguous().cpu().numpy()
        raw.tofile(os.path.join(dump_dir, fname))

    wr(weights["x"], "x.bin")
    ln_cat = torch.cat([weights["input_ln_w"], weights["q_a_ln_w"],
                        weights["kv_a_ln_w"]]).contiguous()
    wr(ln_cat, "ln_weights.bin")
    wr(weights["qkv_a_w"], "qkv_a_w.bin")
    wr(weights["qkv_a_s"], "qkv_a_s.bin")
    wr(weights["q_b_w"], "q_b_w.bin")
    wr(weights["q_b_s"], "q_b_s.bin")
    wr(weights["kvbv_w"], "kvbv_w.bin")
    wr(weights["kvbv_s"], "kvbv_s.bin")
    wr(weights["oproj_w"], "oproj_w.bin")
    wr(weights["oproj_s"], "oproj_s.bin")
    wr(cos_sin, "cos_sin.bin")
    wr(kv_prefill, "kv_prefill.bin")
    meta = {"kv_prefill_rows": int(kv_prefill.shape[0]),
            "cos_sin_rows": int(cos_sin.shape[0]), "kv_rows": KV_ROWS}
    with open(os.path.join(dump_dir, "meta.json"), "w") as f:
        json.dump(meta, f)


# ---------------------------------------------------------------------------
# attn_correctness case (test-mode: ONE iteration, step == kv_offset)
# ---------------------------------------------------------------------------
def run_attn_correctness_case(spec: dict, out_dir: str) -> dict:
    cfg = default_cfg(spec)
    seed = spec.get("seed", 20260702)
    kvo = cfg["kv_offset"]
    weights = gen_block_inputs(seed)
    bufs = alloc_block_buffers()
    cos_sin = build_cos_sin()
    kv_prefill = gen_kv_prefill(seed, kvo)
    if kvo > 0:
        bufs["kv_cache"][:kvo] = kv_prefill

    pk = make_pk("v2", M=1, test_mode=True)
    x_dt = pk.attach_input(torch_tensor=weights["x"], name="b0_x")
    cs_dt = pk.attach_input(torch_tensor=cos_sin, name="attn_cos_sin")
    build_attn_block(pk, "b0", weights, bufs, x_dt, weights["x"], cs_dt, cfg)

    t0 = time.time()
    pk.compile(output_dir=os.path.join(out_dir, "compile"))
    compile_s = time.time() - t0
    t0 = time.time()
    pk()
    torch.cuda.synchronize()
    run_s = time.time() - t0

    results = {}

    def cm(name, out_t, ref_t, extra=None):
        m = compare_metrics(out_t.float(), ref_t.float())
        if extra:
            m.update(extra)
        results[name] = m
        return m

    pos = kvo  # iter_num == 0 in test-mode
    # --- per-op refs ON the kernel's own upstream tensors ---
    a_deq_ref, g_qkva_ref = R.ref_p0_qkva(
        weights["x"], weights["input_ln_w"], weights["qkv_a_w"],
        weights["qkv_a_s"])
    cm("g_qkva", bufs["g_qkva"], g_qkva_ref)

    q_deq_ref = R.ref_q_deq(bufs["g_qkva"], weights["q_a_ln_w"])
    kv_row_ref = R.ref_kv_row(bufs["g_qkva"], weights["kv_a_ln_w"], cos_sin,
                              pos)
    cm("kv_row", bufs["kv_cache"][pos].float(), kv_row_ref)
    g_qpe_ref = R.ref_qpe(q_deq_ref, weights["q_b_w"], weights["q_b_s"],
                          cos_sin, pos)
    cm("g_qpe", bufs["g_qpe"], g_qpe_ref)

    attn_ref = R.ref_mla_attn(bufs["g_qpe"], bufs["kv_cache"].float(),
                              pos + 1)
    cm("g_attn", bufs["g_attn"].reshape(-1),
       R.bf16r(attn_ref.reshape(-1)))
    deq_ref = R.raw448_quant_dequant(bufs["g_attn"])
    cm("g_attn_deq", bufs["g_attn_deq"].reshape(-1), deq_ref.reshape(-1))

    g_red_ref = R.ref_wuv(bufs["g_attn_deq"], weights["kvbv_w"],
                          weights["kvbv_s"])
    cm("g_red", bufs["g_red"], g_red_ref)

    out_ref = R.ref_oproj(bufs["g_red"], weights["oproj_w"],
                          weights["oproj_s"], weights["x"])
    cm("out_vs_kernelinputs_ref", bufs["out"].reshape(-1).float(), out_ref)

    # --- full-torch chain (score health + end-to-end sanity) ---
    chain = R.ref_full_chain(
        weights["x"], weights["input_ln_w"], weights["q_a_ln_w"],
        weights["kv_a_ln_w"], weights["qkv_a_w"], weights["qkv_a_s"],
        weights["q_b_w"], weights["q_b_s"], weights["kvbv_w"],
        weights["kvbv_s"], weights["oproj_w"], weights["oproj_s"],
        cos_sin, bufs["kv_cache"][:pos] if pos > 0 else
        torch.zeros((0, R.QKHEAD), device=DEV), pos)
    cm("out_vs_fullchain_ref", bufs["out"].reshape(-1).float(), chain["out"])
    results["score_std"] = chain["score_std"]

    # dump inputs + kernel tensors for the v1 driver same-bytes A/B
    if spec.get("dump_inputs", True):
        dump_driver_inputs(weights, cos_sin, kv_prefill,
                           os.path.join(out_dir, "driver_inputs"))
        kdump = {k: bufs[k] for k in
                 ("g_qkva", "g_qpe", "g_mla_acc", "g_mla_m", "g_mla_l",
                  "g_attn", "g_attn_deq", "g_red", "out")}
        kdump["kv_rows"] = bufs["kv_cache"][:pos + 1]
        torch.save({k: v.cpu() for k, v in kdump.items()},
                   os.path.join(out_dir, "v2_intermediates.pt"))

    def ok(name, cos_min=0.999, rel_max=3e-2):
        m = results.get(name)
        if not m or "cos" not in m:
            return None
        return bool(m["cos"] >= cos_min and m["rel_max"] <= rel_max)

    results["_pass"] = {
        "g_qkva": ok("g_qkva"),
        "kv_row": ok("kv_row"),
        "g_qpe": ok("g_qpe"),
        "g_attn": ok("g_attn"),
        "g_attn_deq": ok("g_attn_deq"),
        "g_red": ok("g_red"),
        "out": ok("out_vs_kernelinputs_ref"),
        "out_fullchain": ok("out_vs_fullchain_ref"),
    }

    try:
        pk.finalize()
    except Exception as e:  # noqa: BLE001
        results["_finalize_warning"] = str(e)
    return {
        "kind": "attn_correctness",
        "cfg": cfg,
        "seed": seed,
        "compile_s": compile_s,
        "run_s": run_s,
        "ops": results,
    }


# ---------------------------------------------------------------------------
# attn_multistep case: offline-mode ONE block, small S, kv_offset 0 —
# proves the iter_num/step plumbing (kv rows 0..S-1 + final out for the v1
# driver multi-step A/B).
# ---------------------------------------------------------------------------
def run_attn_multistep_case(spec: dict, out_dir: str) -> dict:
    cfg = default_cfg(spec)
    S = spec.get("iters", 4)
    seed = spec.get("seed", 20260702)
    weights = gen_block_inputs(seed)
    bufs = alloc_block_buffers()
    cos_sin = build_cos_sin()
    if spec.get("inplace_x"):
        # DISTINCT-x multistep (review C3 closure): out IS the x buffer, so
        # each iteration's input hidden = the previous iteration's block
        # output (physically the same storage; graph-wise still a pure chain
        # because out/x/residual are three attaches = three guids). The v1
        # driver mirrors with --inplace-x. Makes a step misalignment VISIBLE
        # in out (the constant-x variant was insensitive to it).
        bufs["out"] = weights["x"]

    pk = make_pk("v2", 1, test_mode=False, max_seq_length=S,
                 trace_name=os.path.join(out_dir, "trace_attn_ms"))
    x_dt = pk.attach_input(torch_tensor=weights["x"], name="b0_x")
    cs_dt = pk.attach_input(torch_tensor=cos_sin, name="attn_cos_sin")
    build_attn_block(pk, "b0", weights, bufs, x_dt, weights["x"], cs_dt, cfg)

    # In-place-x runs MUTATE weights["x"] (out == x): the driver dump must
    # carry the PRE-RUN x0, so dump BEFORE pk() (dump-order bug fix).
    if spec.get("dump_inputs", True):
        dump_driver_inputs(weights, cos_sin,
                           gen_kv_prefill(seed, cfg["kv_offset"]),
                           os.path.join(out_dir, "driver_inputs"))

    t0 = time.time()
    pk.compile(output_dir=os.path.join(out_dir, "compile"))
    compile_s = time.time() - t0
    pk()
    torch.cuda.synchronize()

    kdump = {k: bufs[k] for k in
             ("g_qkva", "g_qpe", "g_mla_acc", "g_mla_m", "g_mla_l",
              "g_attn", "g_attn_deq", "g_red", "out")}
    # v2 convention: max_seq_length=S runs S-1 LIVE task iterations
    # (iter_num 0..S-2; the would-be post-done iteration is not executed in
    # unprofiled mode) -> rows [kvo, kvo+S-1) are written; the driver A/B
    # must run --steps S-1.
    kdump["kv_rows"] = bufs["kv_cache"][:cfg["kv_offset"] + S - 1]
    torch.save({k: v.cpu() for k, v in kdump.items()},
               os.path.join(out_dir, "v2_intermediates.pt"))
    return {
        "kind": "attn_multistep",
        "cfg": cfg,
        "iters": S,
        "seed": seed,
        "compile_s": compile_s,
        "final_step": int(pk.meta_tensors["step"][0].item()),
    }


# ---------------------------------------------------------------------------
# attn_perf case
# ---------------------------------------------------------------------------
def run_attn_perf_case(spec: dict, out_dir: str) -> dict:
    cfg = default_cfg(spec)
    L = spec.get("L", 3)
    S = spec.get("iters", 32)
    profiled = spec.get("profiled", True)
    seed = spec.get("seed", 20260702)
    kvo = cfg["kv_offset"]

    result = {"kind": "attn_perf", "cfg": cfg, "L": L, "iters": S,
              "profiled": profiled, "seed": seed}

    prof = None
    if profiled:
        prof = torch.zeros(120000 * 128, dtype=torch.uint64,
                           device="cuda").contiguous()

    pk = make_pk("v2", 1, test_mode=False, max_seq_length=S,
                 profiler_tensor=prof,
                 trace_name=os.path.join(out_dir, "trace_attn_v2"))

    all_bufs, instances = [], []
    cos_sin = build_cos_sin()
    cs_dt = pk.attach_input(torch_tensor=cos_sin, name="attn_cos_sin")
    x0 = gen_block_inputs(seed)["x"]
    x_dt, x_torch = pk.attach_input(torch_tensor=x0, name="chain_x0"), x0
    for i in range(L):
        w = gen_block_inputs(seed + 1000 * (i + 1))
        b = alloc_block_buffers()
        if kvo > 0:
            b["kv_cache"][:kvo] = gen_kv_prefill(seed + 1000 * (i + 1), kvo)
        all_bufs.append(b)
        out_dt, out_torch = build_attn_block(pk, f"b{i}", w, b, x_dt,
                                             x_torch, cs_dt, cfg)
        x_dt, x_torch = out_dt, out_torch
        instances += block_instances(i, cfg)
    result["instances"] = [
        {"instance": i, "op": op, "ntasks": n} for (i, op, n) in instances]

    t0 = time.time()
    pk.compile(output_dir=os.path.join(out_dir, "compile"))
    result["compile_s"] = time.time() - t0

    torch.cuda.synchronize()
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    pk()
    ev1.record()
    torch.cuda.synchronize()
    result["wall_ms"] = ev0.elapsed_time(ev1)
    result["final_step"] = int(pk.meta_tensors["step"][0].item())
    # sanity: outputs finite
    result["out_finite"] = [bool(torch.isfinite(b["out"].float()).all())
                            for b in all_bufs]

    if profiled:
        import numpy as np

        raw_path = os.path.join(out_dir, "prof_attn_v2.npy")
        np.save(raw_path, prof.cpu().numpy())
        result["prof_raw"] = raw_path
        v2map = getattr(pk, "_v2_task_graph_for_prof", None)
        if v2map is not None:
            from v2_prof_decode import decode_window_table, summarize

            table = decode_window_table(prof, v2map["queues"],
                                        v2map["task_types"])
            # Ablation-4 support: persist the raw per-(block,iter,q) consumer
            # rows so begin-skew vs body-straggler decomposition is possible
            # offline (r5 runs only kept the summary).
            with open(os.path.join(out_dir, "v2_rows.json"), "w") as f:
                json.dump({"rows": table["rows"],
                           "n_window_iters": table["n_window_iters"]}, f)
            summary = summarize(
                table, result["instances"],
                os.path.join(out_dir, "compile", "task_graph_rank0.json"),
                total_iters=S)
            with open(os.path.join(out_dir, "table_attn_v2.json"), "w") as f:
                json.dump({"validation": table["validation"],
                           "n_window_iters": table["n_window_iters"],
                           "summary": summary}, f, indent=1)
            result["decode_summary"] = summary
            result["decode_validation"] = {
                "n_errors": len(table["validation"]["errors"]),
                "n_warnings": len(table["validation"]["warnings"]),
                "first_errors": table["validation"]["errors"][:5],
            }
            if not table["validation"].get("decode_ok") or "error" in summary:
                result["status"] = "decode_failed"

    try:
        pk.finalize()
    except Exception as e:  # noqa: BLE001
        result["_finalize_warning"] = str(e)
    return result
