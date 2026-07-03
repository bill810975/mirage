"""Case builders/runners for the DSv3 fused-FFN v2 task chain (Step 3a).

Reuses the calibrated v2 framework machinery (v2_harness.make_pk,
pytorch_reference.compare_metrics, v2_prof_decode) — this module only adds
the FFN-specific graph builders, DSv3-shaped inputs, torch references
(dsv3_ffn_ref) and the v1-driver input dumps.

Case kinds (dispatched by run_ffn_suite.py in a subprocess per case):
  ffn_correctness: test-mode single-pass, ONE FFN block; per-op compare vs
      the torch refs + input dump for the v1 driver same-bytes A/B.
  ffn_perf: offline-mode chain of L FFN blocks (block i+1's hidden = block
      i's out; per-block own weights => cold-L2 by footprint), S iterations,
      profiled (v2 role profiler window) or unprofiled (wall cross-check).

Graph-shape contract (annotated_graph case-2/3): each op's DECLARED output
is consumed ONLY by the next op; a_fp8/a_scale/i_scale/si_scale/logits are
hidden writes (bound as inputs of the writer); meta/a_fp8/... reads by
later ops go through fresh attach_input ALIASES of the same torch tensor.
"""

import json
import os
import struct
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dsv3_ffn_ref as R
from pytorch_reference import compare_metrics
from v2_harness import make_pk

DEV = "cuda"


# ---------------------------------------------------------------------------
# Deterministic DSv3-shaped input generation (driver-style distributions).
# ---------------------------------------------------------------------------
def _gen(seed: int):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return g


def gen_fp8_bytes(shape, gen) -> torch.Tensor:
    """Random e4m3 bytes with the NaN encodings (x7F/xFF) remapped — same
    convention as ffn_ws_driver.gen_fp8_bytes."""
    b = torch.randint(0, 256, shape, generator=gen, device=DEV,
                      dtype=torch.int32)
    nan_mask = (b & 0x7F) == 0x7F
    b = torch.where(nan_mask, (b & 0x80) | 0x3E, b)
    return b.to(torch.uint8).contiguous()


def gen_pow2_scales(shape, gen, lo_exp=-9, hi_exp=-5) -> torch.Tensor:
    e = torch.randint(lo_exp, hi_exp + 1, shape, generator=gen, device=DEV)
    return torch.ldexp(torch.ones(shape, device=DEV, dtype=torch.float64),
                       e).float().contiguous()


def gen_block_inputs(seed: int, force_local8: bool = False) -> dict:
    """One FFN block's weights + (for block 0) the hidden input."""
    gen = _gen(seed)

    def uni(shape, lo, hi, dtype=torch.float32):
        t = torch.rand(shape, generator=gen, device=DEV) * (hi - lo) + lo
        return t.to(dtype).contiguous()

    t = {
        "hidden": uni((1, R.HIDDEN), -1.0, 1.0, torch.bfloat16),
        "rms_w": uni((R.HIDDEN,), 0.8, 1.2, torch.bfloat16),
        "router_w": uni((R.ROUTER_N, R.HIDDEN), -0.04, 0.04, torch.bfloat16),
        "bias": uni((R.ROUTER_N,), -0.05, 0.05, torch.float32),
        "w13": gen_fp8_bytes((R.E_LOCAL, R.W13_N, R.HIDDEN), gen),
        "w13_scale": gen_pow2_scales((R.E_LOCAL, R.NB1, R.KG1), gen),
        "w2": gen_fp8_bytes((R.E_LOCAL, R.W2_N, R.W2_K), gen),
        "w2_scale": gen_pow2_scales((R.E_LOCAL, R.NB2, R.KG2), gen),
        "wgu": gen_fp8_bytes((R.SH_GU_N, R.HIDDEN), gen),
        "wgu_scale": gen_pow2_scales((R.NB_SHGU, R.KG_SHGU), gen),
        "wdn": gen_fp8_bytes((R.W2_N, R.SH_DN_K), gen),
        "wdn_scale": gen_pow2_scales((R.NB_SHDN, R.KG_SHDN), gen),
    }
    if force_local8:
        # forced max-work arm: experts >= 128 can never win the top-8 ->
        # all 8 global winners are EP-local -> active_count == 8.
        t["bias"][128:] -= 1000.0
    return t


def alloc_block_buffers() -> dict:
    """Intermediate + output buffers for one block (all attached)."""
    z = lambda shape, dt: torch.zeros(shape, device=DEV, dtype=dt)
    return {
        "rmsnorm_out": z((1, R.HIDDEN), torch.bfloat16),
        "a_fp8": z((R.HIDDEN,), torch.uint8),
        "a_scale": z((R.KG1,), torch.float32),
        "inter": z((R.ROUTER_N, R.RKSPLIT), torch.float32),
        "logits": z((R.ROUTER_N,), torch.bfloat16),
        "meta": z((R.META_INTS,), torch.int32),
        "y13": z((R.MAX_ACTIVE, R.W13_N), torch.float32),
        "sg": z((R.SH_GU_N,), torch.float32),
        "i_fp8": z((R.MAX_ACTIVE, R.W2_K), torch.uint8),
        "i_scale": z((R.MAX_ACTIVE, R.KG2), torch.float32),
        "si_fp8": z((R.SH_DN_K,), torch.uint8),
        "si_scale": z((R.KG_SHDN,), torch.float32),
        "out": z((1, R.W2_N), torch.bfloat16),
    }


# ---------------------------------------------------------------------------
# Graph builder: one FFN block = 6 chained ops.
# ---------------------------------------------------------------------------
def build_ffn_block(pk, prefix: str, weights: dict, bufs: dict,
                    hidden_dt, cfg: dict):
    """hidden_dt: the DTensor for this block's input hidden state.
    Returns the out DTensor (feed to the next block)."""
    at = lambda t, nm: pk.attach_input(torch_tensor=t, name=f"{prefix}_{nm}")

    rms_w = at(weights["rms_w"], "rmsw")
    router_w = at(weights["router_w"], "routerw")
    bias = at(weights["bias"], "bias")
    w13 = at(weights["w13"], "w13")
    w13_s = at(weights["w13_scale"], "w13s")
    w2 = at(weights["w2"], "w2")
    w2_s = at(weights["w2_scale"], "w2s")
    wgu = at(weights["wgu"], "wgu")
    wgu_s = at(weights["wgu_scale"], "wgus")
    wdn = at(weights["wdn"], "wdn")
    wdn_s = at(weights["wdn_scale"], "wdns")

    rmsnorm_out = at(bufs["rmsnorm_out"], "rmsout")
    a_fp8 = at(bufs["a_fp8"], "afp8")
    a_scale = at(bufs["a_scale"], "ascale")
    inter = at(bufs["inter"], "inter")
    logits = at(bufs["logits"], "logits")
    meta = at(bufs["meta"], "meta")
    y13 = at(bufs["y13"], "y13")
    sg = at(bufs["sg"], "sg")
    i_fp8 = at(bufs["i_fp8"], "ifp8")
    i_scale = at(bufs["i_scale"], "iscale")
    si_fp8 = at(bufs["si_fp8"], "sifp8")
    si_scale = at(bufs["si_scale"], "siscale")
    out = at(bufs["out"], "out")

    # ALIASES: same torch tensor, fresh DTensor/guid -> no graph edge (the
    # chain events provide the ordering transitively). Unique names required
    # (codegen keys all_tensors by name).
    a_fp8_al = at(bufs["a_fp8"], "afp8_alias")
    a_scale_al = at(bufs["a_scale"], "ascale_alias")
    meta_al_silu = at(bufs["meta"], "meta_alias_silu")
    meta_al_w2 = at(bufs["meta"], "meta_alias_w2")
    i_scale_al = at(bufs["i_scale"], "iscale_alias")
    si_scale_al = at(bufs["si_scale"], "siscale_alias")

    # T0 rmsnorm (existing v2 task)
    pk.rmsnorm_layer(input=hidden_dt, weight=rms_w, output=rmsnorm_out,
                     grid_dim=(1, 1, 1), block_dim=(128, 1, 1))
    # T1 router + quant
    pk.dsv3_ffn_router_quant_layer(
        input=rmsnorm_out, gate_weight=router_w, a_fp8=a_fp8, a_scale=a_scale,
        inter=inter, num_tasks=cfg["nr"], nwarps=cfg["nwarps"])
    # T2 topk-sigmoid
    pk.dsv3_ffn_topk_sigmoid_layer(
        inter=inter, bias=bias, logits=logits, meta=meta,
        local_expert_start=cfg["les"], num_local_experts=cfg["nle"],
        routed_scaling_factor=cfg["rsf"])
    # T3 W13 + shared gate_up
    pk.dsv3_ffn_w13_gemv_layer(
        meta=meta, a_fp8=a_fp8_al, a_scale=a_scale_al, w13=w13,
        w13_scale=w13_s, wgu=wgu, wgu_scale=wgu_s, y13=y13, sg=sg,
        num_tasks=cfg["nt13"], nwarps=cfg["nwarps"])
    # T4 silu + requant
    pk.dsv3_ffn_silu_quant_layer(
        y13=y13, sg=sg, meta=meta_al_silu, i_scale=i_scale,
        si_scale=si_scale, i_fp8=i_fp8, si_fp8=si_fp8)
    # T5 W2 + shared down -> bf16 out
    pk.dsv3_ffn_w2_gemv_layer(
        i_fp8=i_fp8, si_fp8=si_fp8, meta=meta_al_w2, i_scale=i_scale_al,
        si_scale=si_scale_al, w2=w2, w2_scale=w2_s, wdn=wdn,
        wdn_scale=wdn_s, output=out, num_tasks=cfg["nt2"],
        nwarps=cfg["nwarps"], rblk=cfg["rblk"])
    return out


def default_cfg(spec: dict) -> dict:
    nwarps = spec.get("nwarps", 4)
    return {
        "nr": spec.get("nr", 136),
        "nt13": spec.get("nt13", 136),
        "nt2": spec.get("nt2", 64 if nwarps == 7 else 112),
        "nwarps": nwarps,
        "rblk": spec.get("rblk", 8 if nwarps == 7 else 16),
        "les": spec.get("les", 0),
        "nle": spec.get("nle", 128),
        "rsf": spec.get("rsf", 2.5),
    }


# per-block op instances (for v2_prof_decode.summarize)
def block_instances(i: int, cfg: dict):
    return [
        (i, "rmsnorm_7168", 1),
        (i, "ffn_router_quant", cfg["nr"]),
        (i, "ffn_topk_sigmoid", 1),
        (i, "ffn_w13_gemv", cfg["nt13"]),
        (i, "ffn_silu_quant", 1),
        (i, "ffn_w2_gemv", cfg["nt2"]),
    ]


# ---------------------------------------------------------------------------
# Input dump for the v1 driver (--load-dir): raw little-endian binaries in
# the driver's expected shapes.
# ---------------------------------------------------------------------------
_DUMP_KEYS = [
    ("hidden", "hidden.bin"), ("rms_w", "rmsw.bin"),
    ("router_w", "router.bin"), ("bias", "bias.bin"),
    ("w13", "w13.bin"), ("w13_scale", "w13s.bin"),
    ("w2", "w2.bin"), ("w2_scale", "w2s.bin"),
    ("wgu", "wgu.bin"), ("wgu_scale", "wgus.bin"),
    ("wdn", "wdn.bin"), ("wdn_scale", "wdns.bin"),
]


def dump_driver_inputs(weights: dict, dump_dir: str):
    os.makedirs(dump_dir, exist_ok=True)
    for key, fname in _DUMP_KEYS:
        t = weights[key]
        if t.dtype == torch.bfloat16:
            raw = t.contiguous().view(torch.uint16).cpu().numpy()
        else:
            raw = t.contiguous().cpu().numpy()
        raw.tofile(os.path.join(dump_dir, fname))


# ---------------------------------------------------------------------------
# ffn_correctness case
# ---------------------------------------------------------------------------
def run_ffn_correctness_case(spec: dict, out_dir: str) -> dict:
    cfg = default_cfg(spec)
    seed = spec.get("seed", 20260702)
    weights = gen_block_inputs(seed, force_local8=spec.get("force_local8",
                                                           False))
    bufs = alloc_block_buffers()

    pk = make_pk("v2", M=1, test_mode=True)
    hidden_dt = pk.attach_input(torch_tensor=weights["hidden"], name="b0_hidden")
    build_ffn_block(pk, "b0", weights, bufs, hidden_dt, cfg)

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

    # --- pure-torch chain reference (REF routing) ---
    ref = R.ref_full_chain(weights, cfg["les"], cfg["nle"], cfg["rsf"])

    # T0 rmsnorm vs fp32 ref (bf16 bit-match fraction reported)
    m_rms = compare_metrics(bufs["rmsnorm_out"].reshape(-1),
                            ref["normed_bf16"])
    results["rmsnorm_out"] = m_rms

    # T1 quant vs EXACT emulation ON THE KERNEL's rmsnorm_out (isolates the
    # quant math from the rmsnorm reduce-tree difference)
    a_ref_q, a_ref_s = R.ue8m0_group_quant(
        bufs["rmsnorm_out"].reshape(-1).float())
    results["a_fp8_bytes"] = {
        "byte_match_frac": float((bufs["a_fp8"] == a_ref_q).float().mean()),
        "n_mismatch": int((bufs["a_fp8"] != a_ref_q).sum()),
    }
    cm("a_scale", bufs["a_scale"], a_ref_s)

    # T1 router: inter + logits vs fp32 matmul ref on the KERNEL's normed
    inter_ref, logits_ref = R.ref_router(bufs["rmsnorm_out"].reshape(-1),
                                         weights["router_w"])
    cm("inter", bufs["inter"], inter_ref)
    cm("logits", bufs["logits"].float(), logits_ref.float())

    # T2 topk vs f64 selection ref ON THE KERNEL's inter (exact gate)
    routing_ref = R.ref_topk_sigmoid(bufs["inter"], weights["bias"],
                                     cfg["les"], cfg["nle"], cfg["rsf"])
    meta_k = bufs["meta"].cpu().tolist()
    ac_k = meta_k[R.META_INTS * 0 + 0]
    experts_k = meta_k[2:2 + R.MAX_ACTIVE]
    weights_k = [struct.unpack("<f", struct.pack("<i", meta_k[10 + s]))[0]
                 for s in range(R.MAX_ACTIVE)]
    results["meta"] = {
        "magic_ok": meta_k[1] == R.META_MAGIC,
        "active_count_kernel": ac_k,
        "active_count_ref": routing_ref["active_count"],
        "experts_kernel": experts_k[:ac_k],
        "experts_ref": routing_ref["experts"],
        "experts_exact": experts_k[:ac_k] == routing_ref["experts"],
        "weights_kernel": weights_k[:ac_k],
        "weights_ref": routing_ref["weights"],
        "weights_max_rel_err": max(
            (abs(a - b) / max(abs(b), 1e-20)
             for a, b in zip(weights_k[:ac_k], routing_ref["weights"])),
            default=0.0),
    }

    # T3 W13 vs ref ON THE KERNEL's a_fp8 + kernel routing (active slots only)
    y13_ref, sg_ref = R.ref_w13(bufs["a_fp8"], bufs["a_scale"],
                                weights["w13"], weights["w13_scale"],
                                weights["wgu"], weights["wgu_scale"],
                                experts_k[:ac_k])
    if ac_k > 0:
        cm("y13_active", bufs["y13"][:ac_k], y13_ref)
    cm("sg", bufs["sg"], sg_ref)

    # T4 silu+requant vs emulation ON THE KERNEL's y13/sg (silu_fast vs
    # torch sigmoid -> compare in the DECODED domain + byte fractions)
    i_ref_q, i_ref_s, si_ref_q, si_ref_s = R.ref_silu_quant(
        bufs["y13"].float(), bufs["sg"].float(), ac_k)
    if ac_k > 0:
        i_k_dec = R.fp8_decode(bufs["i_fp8"][:ac_k]) * \
            bufs["i_scale"][:ac_k].repeat_interleave(R.GRP, dim=1)
        i_r_dec = R.fp8_decode(i_ref_q) * \
            i_ref_s.repeat_interleave(R.GRP, dim=1)
        cm("i_dequant", i_k_dec, i_r_dec, extra={
            "byte_match_frac": float(
                (bufs["i_fp8"][:ac_k] == i_ref_q).float().mean()),
        })
    si_k_dec = R.fp8_decode(bufs["si_fp8"]) * \
        bufs["si_scale"].repeat_interleave(R.GRP)
    si_r_dec = R.fp8_decode(si_ref_q) * si_ref_s.repeat_interleave(R.GRP)
    cm("si_dequant", si_k_dec, si_r_dec, extra={
        "byte_match_frac": float((bufs["si_fp8"] == si_ref_q).float().mean()),
    })

    # T5 out vs ref ON THE KERNEL's i_fp8 (+ vs the full-torch chain)
    out_ref_k = R.ref_w2(bufs["i_fp8"][:ac_k], bufs["i_scale"][:ac_k],
                         bufs["si_fp8"], bufs["si_scale"], weights["w2"],
                         weights["w2_scale"], weights["wdn"],
                         weights["wdn_scale"], experts_k[:ac_k],
                         weights_k[:ac_k])
    cm("out_vs_kernelinputs_ref", bufs["out"].reshape(-1).float(), out_ref_k)
    cm("out_vs_fullchain_ref", bufs["out"].reshape(-1).float(),
       ref["out_f32"])
    results["routing_ref_vs_chain"] = {
        "ref_active_count": ref["routing"]["active_count"],
        "ref_experts": ref["routing"]["experts"],
        "kernel_matches_fullchain_ref":
            experts_k[:ac_k] == ref["routing"]["experts"],
    }

    # dump inputs + kernel intermediates for the v1 driver same-bytes A/B
    if spec.get("dump_inputs", True):
        dump_dir = os.path.join(out_dir, "driver_inputs")
        dump_driver_inputs(weights, dump_dir)
        kdump = {
            "rmsnorm_out": bufs["rmsnorm_out"],
            "inter": bufs["inter"],
            "logits": bufs["logits"],
            "y13": bufs["y13"],
            "sg": bufs["sg"],
            "a_fp8": bufs["a_fp8"],
            "a_scale": bufs["a_scale"],
            "i_fp8": bufs["i_fp8"],
            "i_scale": bufs["i_scale"],
            "si_fp8": bufs["si_fp8"],
            "si_scale": bufs["si_scale"],
            "meta": bufs["meta"],
            "out": bufs["out"],
        }
        torch.save({k: v.cpu() for k, v in kdump.items()},
                   os.path.join(out_dir, "v2_intermediates.pt"))

    # per-op PASS flags (framework thresholds)
    def ok(name, cos_min=0.999, rel_max=3e-2):
        m = results.get(name)
        if not m or "cos" not in m:
            return None
        return bool(m["cos"] >= cos_min and m["rel_max"] <= rel_max)

    results["_pass"] = {
        "rmsnorm": ok("rmsnorm_out"),
        "a_quant_exact": results["a_fp8_bytes"]["n_mismatch"] == 0,
        "inter": ok("inter"),
        "logits": ok("logits"),
        "topk_exact": results["meta"]["experts_exact"]
        and results["meta"]["active_count_kernel"]
        == results["meta"]["active_count_ref"]
        and results["meta"]["weights_max_rel_err"] < 1e-5,
        "y13": ok("y13_active") if ac_k > 0 else True,
        "sg": ok("sg"),
        "silu_quant": ok("i_dequant") if ac_k > 0 else True,
        "out": ok("out_vs_kernelinputs_ref"),
        "out_fullchain": ok("out_vs_fullchain_ref"),
    }

    try:
        pk.finalize()
    except Exception as e:  # noqa: BLE001
        results["_finalize_warning"] = str(e)
    return {
        "kind": "ffn_correctness",
        "cfg": cfg,
        "seed": seed,
        "compile_s": compile_s,
        "run_s": run_s,
        "ops": results,
    }


# ---------------------------------------------------------------------------
# ffn_perf case
# ---------------------------------------------------------------------------
def run_ffn_perf_case(spec: dict, out_dir: str) -> dict:
    cfg = default_cfg(spec)
    L = spec.get("L", 3)
    S = spec.get("iters", 32)
    profiled = spec.get("profiled", True)
    seed = spec.get("seed", 20260702)

    result = {"kind": "ffn_perf", "cfg": cfg, "L": L, "iters": S,
              "profiled": profiled, "seed": seed}

    prof = None
    if profiled:
        prof = torch.zeros(120000 * 128, dtype=torch.uint64,
                           device="cuda").contiguous()

    pk = make_pk("v2", 1, test_mode=False, max_seq_length=S,
                 profiler_tensor=prof,
                 trace_name=os.path.join(out_dir, "trace_ffn_v2"))

    all_weights, all_bufs, instances = [], [], []
    x0 = gen_block_inputs(seed)["hidden"]  # block-0 hidden
    hidden_dt = pk.attach_input(torch_tensor=x0, name="chain_x0")
    for i in range(L):
        w = gen_block_inputs(seed + 1000 * (i + 1),
                             force_local8=spec.get("force_local8", False))
        b = alloc_block_buffers()
        all_weights.append(w)
        all_bufs.append(b)
        out_dt = build_ffn_block(pk, f"b{i}", w, b, hidden_dt, cfg)
        hidden_dt = out_dt  # chain: block i's out = block i+1's hidden
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

    # per-block routing actually used (iteration-invariant by construction)
    result["active_counts"] = [int(b["meta"][0].item()) for b in all_bufs]
    result["meta_magic_ok"] = [int(b["meta"][1].item()) == R.META_MAGIC
                               for b in all_bufs]

    if profiled:
        import numpy as np

        raw_path = os.path.join(out_dir, "prof_ffn_v2.npy")
        np.save(raw_path, prof.cpu().numpy())
        result["prof_raw"] = raw_path
        v2map = getattr(pk, "_v2_task_graph_for_prof", None)
        if v2map is not None:
            from v2_prof_decode import decode_window_table, summarize

            table = decode_window_table(prof, v2map["queues"],
                                        v2map["task_types"])
            summary = summarize(
                table, result["instances"],
                os.path.join(out_dir, "compile", "task_graph_rank0.json"),
                total_iters=S)
            with open(os.path.join(out_dir, "table_ffn_v2.json"), "w") as f:
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
