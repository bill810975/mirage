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


def alloc_block_buffers(rung=None) -> dict:
    """Intermediate + output buffers for one block (all attached)."""
    z = lambda shape, dt: torch.zeros(shape, device=DEV, dtype=dt)
    if rung not in ("mega", "megafg"):
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
    # Rung B (mega): MAX_INPUTS_PER_TASK=14 forces packed GMEM layouts.
    # Offsets MUST stay in lockstep with dsv3_ffn_v2_spec.h (MEGA_* consts).
    # The returned dict exposes the SAME keys as the unpacked layout, as
    # views into the packs — the compare/poison/dump code paths are shared.
    xfer = z((MEGA_XFER_FLOATS,), torch.float32)
    art = z((MEGA_ART_BYTES,), torch.uint8)
    # in-op barrier/counter state; int64 => poison-skipped. Coarse mega uses
    # bar[2]; the fine-grained variant packs FGBAR_COUNT (21) per-slot counters.
    bar = z((FGBAR_COUNT if rung == "megafg" else 2,), torch.int64)
    out = z((1, R.W2_N), torch.bfloat16)

    def av(off, nbytes, dt, shape):
        return art[off:off + nbytes].view(dt).view(shape)

    return {
        "rmsnorm_out": av(MEGA_ART_OFF_RMSNORM, R.HIDDEN * 2,
                          torch.bfloat16, (1, R.HIDDEN)),
        "a_fp8": av(MEGA_ART_OFF_AFP8, R.HIDDEN, torch.uint8, (R.HIDDEN,)),
        "a_scale": av(MEGA_ART_OFF_ASCALE, R.KG1 * 4, torch.float32,
                      (R.KG1,)),
        "logits": av(MEGA_ART_OFF_LOGITS, R.ROUTER_N * 2, torch.bfloat16,
                     (R.ROUTER_N,)),
        "meta": av(MEGA_ART_OFF_META, R.META_INTS * 4, torch.int32,
                   (R.META_INTS,)),
        "i_fp8": av(MEGA_ART_OFF_IFP8, R.MAX_ACTIVE * R.W2_K, torch.uint8,
                    (R.MAX_ACTIVE, R.W2_K)),
        "i_scale": av(MEGA_ART_OFF_ISCALE, R.MAX_ACTIVE * R.KG2 * 4,
                      torch.float32, (R.MAX_ACTIVE, R.KG2)),
        "si_fp8": av(MEGA_ART_OFF_SIFP8, R.SH_DN_K, torch.uint8,
                     (R.SH_DN_K,)),
        "si_scale": av(MEGA_ART_OFF_SISCALE, R.KG_SHDN * 4, torch.float32,
                       (R.KG_SHDN,)),
        "inter": xfer[MEGA_XFER_OFF_INTER_F:
                      MEGA_XFER_OFF_INTER_F + R.ROUTER_N * R.RKSPLIT].view(
            R.ROUTER_N, R.RKSPLIT),
        "y13": xfer[MEGA_XFER_OFF_Y13_F:
                    MEGA_XFER_OFF_Y13_F + R.MAX_ACTIVE * R.W13_N].view(
            R.MAX_ACTIVE, R.W13_N),
        "sg": xfer[MEGA_XFER_OFF_SG_F:MEGA_XFER_OFF_SG_F + R.SH_GU_N],
        "out": out,
        "_xfer": xfer,
        "_art": art,
        "_bar": bar,
    }


# ---- Rung B packed-layout offsets: MUST mirror dsv3_ffn_v2_spec.h ----------
def _a16(n):
    return (n + 15) & ~15


MEGA_XFER_OFF_INTER_F = 0
MEGA_XFER_OFF_Y13_F = R.ROUTER_N * R.RKSPLIT
MEGA_XFER_OFF_SG_F = MEGA_XFER_OFF_Y13_F + R.MAX_ACTIVE * R.W13_N
MEGA_XFER_FLOATS = MEGA_XFER_OFF_SG_F + R.SH_GU_N

MEGA_SC_OFF_W13 = 0
MEGA_SC_OFF_WGU = MEGA_SC_OFF_W13 + R.E_LOCAL * R.NB1 * R.KG1
MEGA_SC_OFF_W2 = MEGA_SC_OFF_WGU + R.NB_SHGU * R.KG_SHGU
MEGA_SC_OFF_WDN = MEGA_SC_OFF_W2 + R.E_LOCAL * R.NB2 * R.KG2
MEGA_SC_FLOATS = MEGA_SC_OFF_WDN + R.NB_SHDN * R.KG_SHDN

MEGA_ART_OFF_RMSNORM = 0
MEGA_ART_OFF_AFP8 = _a16(MEGA_ART_OFF_RMSNORM + R.HIDDEN * 2)
MEGA_ART_OFF_ASCALE = _a16(MEGA_ART_OFF_AFP8 + R.HIDDEN)
MEGA_ART_OFF_LOGITS = _a16(MEGA_ART_OFF_ASCALE + R.KG1 * 4)
MEGA_ART_OFF_META = _a16(MEGA_ART_OFF_LOGITS + R.ROUTER_N * 2)
MEGA_ART_OFF_IFP8 = _a16(MEGA_ART_OFF_META + R.META_INTS * 4)
MEGA_ART_OFF_ISCALE = _a16(MEGA_ART_OFF_IFP8 + R.MAX_ACTIVE * R.W2_K)
MEGA_ART_OFF_SIFP8 = _a16(MEGA_ART_OFF_ISCALE + R.MAX_ACTIVE * R.KG2 * 4)
MEGA_ART_OFF_SISCALE = _a16(MEGA_ART_OFF_SIFP8 + R.SH_DN_K)
MEGA_ART_BYTES = _a16(MEGA_ART_OFF_SISCALE + R.KG_SHDN * 4)

# fine-grained bar layout (mirror dsv3_ffn_v2_spec.h FGBAR_*): [GB1][rsv]
# [y_done*8][sg_done][y_target*8][sg_target][epoch] = 21 u64 elements.
FGBAR_COUNT = 2 + R.MAX_ACTIVE + 1 + R.MAX_ACTIVE + 1 + 1


def assert_mega_coresidency(compile_dir: str, num_workers: int,
                            num_tasks: int,
                            enum_name: str = "TASK_DSV3_FFN_MEGA_V2"):
    """Rung-B deadlock-safety HARD GATE (run after compile, BEFORE launch):
    from the compiled task graph + the exact per-SM plan twin, verify every
    mega-op instance's tasks are (a) one contiguous id run of num_tasks and
    (b) assigned to num_tasks DISTINCT workers. Two same-op tasks serialized
    on one strict-FIFO worker would deadlock the in-op GMEM barrier."""
    import json as _json

    from mirage.mpk.v2_task_schedule import build_v2_worker_task_queues

    tg_path = os.path.join(compile_dir, "task_graph_rank0.json")
    with open(tg_path) as f:
        tg = _json.load(f)
    types = [int(t.get("task_type", -1)) for t in tg.get("all_tasks", [])]
    from mirage.mpk.profiler_persistent import event_name_list
    mega_ids = [tid for tid, name in event_name_list.items()
                if name == enum_name]
    assert len(mega_ids) == 1, f"mega enum resolution failed: {mega_ids}"
    mega_tid = mega_ids[0]
    mega_pos = [i for i, tt in enumerate(types) if tt == mega_tid]
    assert mega_pos, "no mega tasks found in the compiled graph"

    queues = build_v2_worker_task_queues(tg, num_workers)
    worker_of = {}
    for w, q in enumerate(queues):
        for pos in q:
            assert pos not in worker_of, f"task {pos} scheduled twice"
            worker_of[pos] = w

    # Multi-block chains register the L mega ops back-to-back, so all L*NT
    # tasks form ONE contiguous id range in all_tasks (no gaps). Partition it
    # into consecutive num_tasks-sized CHUNKS (one per block instance) and
    # verify each chunk maps to num_tasks DISTINCT workers. Within one block's
    # barrier every worker then holds exactly ONE task; the >1 tasks a worker
    # gets across the whole graph are from DIFFERENT blocks, chain-serialized
    # (block k+1 depends on block k's output), so they never co-spin.
    assert mega_pos == list(range(mega_pos[0], mega_pos[0] + len(mega_pos))), (
        "mega tasks are not one contiguous id range: "
        f"{mega_pos[0]}..{mega_pos[-1]} vs {len(mega_pos)} tasks")
    assert len(mega_pos) % num_tasks == 0, (
        f"{len(mega_pos)} mega tasks not a multiple of num_tasks={num_tasks}")
    n_inst = len(mega_pos) // num_tasks
    for k in range(n_inst):
        chunk = mega_pos[k * num_tasks:(k + 1) * num_tasks]
        ws = [worker_of[p] for p in chunk]
        assert len(set(ws)) == num_tasks, (
            f"mega block {k} (tasks {chunk[0]}..{chunk[-1]}): only "
            f"{len(set(ws))} distinct workers for {num_tasks} tasks — "
            f"WOULD DEADLOCK the in-op barrier")
    return n_inst


# ---------------------------------------------------------------------------
# Graph builder: one FFN block = 6 chained ops.
# ---------------------------------------------------------------------------
def build_ffn_block(pk, prefix: str, weights: dict, bufs: dict,
                    hidden_dt, cfg: dict):
    """hidden_dt: the DTensor for this block's input hidden state.
    Returns the out DTensor (feed to the next block)."""
    at = lambda t, nm: pk.attach_input(torch_tensor=t, name=f"{prefix}_{nm}")

    if cfg.get("rung") in ("mega", "megafg"):
        # Rung B: ONE op per block; packed inputs (MAX_INPUTS_PER_TASK=14).
        if "_scales_pack" not in weights:
            sp = torch.empty(MEGA_SC_FLOATS, device=DEV, dtype=torch.float32)
            sp[MEGA_SC_OFF_W13:MEGA_SC_OFF_WGU] = \
                weights["w13_scale"].reshape(-1)
            sp[MEGA_SC_OFF_WGU:MEGA_SC_OFF_W2] = \
                weights["wgu_scale"].reshape(-1)
            sp[MEGA_SC_OFF_W2:MEGA_SC_OFF_WDN] = \
                weights["w2_scale"].reshape(-1)
            sp[MEGA_SC_OFF_WDN:MEGA_SC_FLOATS] = \
                weights["wdn_scale"].reshape(-1)
            weights["_scales_pack"] = sp
        out = at(bufs["out"], "out")
        common = dict(
            input=hidden_dt,
            rms_weight=at(weights["rms_w"], "rmsw"),
            gate_weight=at(weights["router_w"], "routerw"),
            bias=at(weights["bias"], "bias"),
            w13=at(weights["w13"], "w13"),
            wgu=at(weights["wgu"], "wgu"),
            w2=at(weights["w2"], "w2"),
            wdn=at(weights["wdn"], "wdn"),
            scales=at(weights["_scales_pack"], "scpack"),
            xfer=at(bufs["_xfer"], "xfer"),
            bar=at(bufs["_bar"], "bar"),
            artifacts=at(bufs["_art"], "art"),
            output=out,
            num_tasks=cfg["ntm"],
            local_expert_start=cfg["les"], num_local_experts=cfg["nle"],
            routed_scaling_factor=cfg["rsf"], nwarps=cfg["nwarps_m"],
            rblk=cfg["rblk_m"])
        if cfg.get("rung") == "megafg":
            pk.dsv3_ffn_mega_fg_layer(stream=cfg["stream"], **common)
        else:
            pk.dsv3_ffn_mega_layer(**common)
        return out

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

    if cfg.get("rung") == "a":
        # Rung A 2-op chain: [w13_rqr_topk] -> [w2_silu].
        pk.dsv3_ffn_w13_rqr_topk_layer(
            input=hidden_dt, rms_weight=rms_w, gate_weight=router_w,
            bias=bias, a_fp8=a_fp8, a_scale=a_scale, rmsnorm_out=rmsnorm_out,
            inter=inter, logits=logits, meta=meta, w13=w13, w13_scale=w13_s,
            wgu=wgu, wgu_scale=wgu_s, y13=y13, sg=sg, num_tasks=cfg["nta"],
            local_expert_start=cfg["les"], num_local_experts=cfg["nle"],
            routed_scaling_factor=cfg["rsf"], nwarps=cfg["nwarps_a"])
        pk.dsv3_ffn_w2_silu_layer(
            y13=y13, sg=sg, meta=meta_al_w2, i_fp8=i_fp8, i_scale=i_scale,
            si_fp8=si_fp8, si_scale=si_scale, w2=w2, w2_scale=w2_s,
            wdn=wdn, wdn_scale=wdn_s, output=out, num_tasks=cfg["nt2"],
            nwarps=cfg["nwarps_w2"], rblk=cfg["rblk"])
        return out

    if cfg.get("fold"):
        # FOLDED 3-op chain: rmsnorm/topk/silu recomputed redundantly inside
        # the MAC tasks (v1's per-CTA trick). Same artifacts written (task-0
        # hidden writes) so the correctness comparisons below are unchanged.
        pk.dsv3_ffn_router_quant_rms_layer(
            input=hidden_dt, rms_weight=rms_w, gate_weight=router_w,
            a_fp8=a_fp8, a_scale=a_scale, rmsnorm_out=rmsnorm_out,
            inter=inter, num_tasks=cfg["nr"], nwarps=cfg["nwarps_rq"])
        pk.dsv3_ffn_w13_topk_layer(
            inter=inter, bias=bias, a_fp8=a_fp8_al, a_scale=a_scale_al,
            w13=w13, w13_scale=w13_s, wgu=wgu, wgu_scale=wgu_s, meta=meta,
            logits=logits, y13=y13, sg=sg, num_tasks=cfg["nt13"],
            local_expert_start=cfg["les"], num_local_experts=cfg["nle"],
            routed_scaling_factor=cfg["rsf"], nwarps=cfg["nwarps_w13"])
        pk.dsv3_ffn_w2_silu_layer(
            y13=y13, sg=sg, meta=meta_al_w2, i_fp8=i_fp8, i_scale=i_scale,
            si_fp8=si_fp8, si_scale=si_scale, w2=w2, w2_scale=w2_s,
            wdn=wdn, wdn_scale=wdn_s, output=out, num_tasks=cfg["nt2"],
            nwarps=cfg["nwarps_w2"], rblk=cfg["rblk"])
        return out

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
    # per-op warp counts (fold chain only): default to the global nwarps.
    nw_rq = spec.get("nwarps_rq", nwarps)
    nw_w13 = spec.get("nwarps_w13", nwarps)
    nw_w2 = spec.get("nwarps_w2", nwarps)
    return {
        "nr": spec.get("nr", 136),
        "nt13": spec.get("nt13", 136),
        "nt2": spec.get("nt2", 128 if nw_w2 == 7 else 112),
        "nwarps": nwarps,
        "nwarps_rq": nw_rq,
        "nwarps_w13": nw_w13,
        "nwarps_w2": nw_w2,
        "rblk": spec.get("rblk", 8 if nw_w2 == 7 else 16),
        "les": spec.get("les", 0),
        "nle": spec.get("nle", 128),
        "rsf": spec.get("rsf", 2.5),
        "fold": spec.get("fold", False),
        # fusion-ladder rungs (scratch/v2_ffn_fuse): None | "a" | "mega"
        "rung": spec.get("rung"),
        "nta": spec.get("nta", 136),          # rung A: w13_rqr_topk tasks
        "nwarps_a": spec.get("nwarps_a", 7),
        "ntm": spec.get("ntm", 136),          # rung B: MUST == num_workers
        "nwarps_m": spec.get("nwarps_m", 7),
        "rblk_m": spec.get("rblk_m", 8),
        "stream": spec.get("stream", 1),      # megafg: 1=stream, 0=FG0 control
    }


# per-block op instances (for v2_prof_decode.summarize)
def block_instances(i: int, cfg: dict):
    if cfg.get("rung") == "a":
        return [
            (i, "ffn_w13_rqr_topk", cfg["nta"]),
            (i, "ffn_w2_silu", cfg["nt2"]),
        ]
    if cfg.get("rung") == "mega":
        return [
            (i, "ffn_mega", cfg["ntm"]),
        ]
    if cfg.get("rung") == "megafg":
        return [
            (i, "ffn_mega_fg", cfg["ntm"]),
        ]
    if cfg.get("fold"):
        return [
            (i, "ffn_router_quant_rms", cfg["nr"]),
            (i, "ffn_w13_topk", cfg["nt13"]),
            (i, "ffn_w2_silu", cfg["nt2"]),
        ]
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
    bufs = alloc_block_buffers(cfg.get("rung"))

    pk = make_pk("v2", M=1, test_mode=True)
    hidden_dt = pk.attach_input(torch_tensor=weights["hidden"], name="b0_hidden")
    build_ffn_block(pk, "b0", weights, bufs, hidden_dt, cfg)

    t0 = time.time()
    pk.compile(output_dir=os.path.join(out_dir, "compile"))
    compile_s = time.time() - t0
    if cfg.get("rung") in ("mega", "megafg"):
        # deadlock-safety HARD GATE before any launch
        _en = ("TASK_DSV3_FFN_MEGA_FG_V2" if cfg.get("rung") == "megafg"
               else "TASK_DSV3_FFN_MEGA_V2")
        n_inst = assert_mega_coresidency(os.path.join(out_dir, "compile"),
                                         pk.num_workers, cfg["ntm"],
                                         enum_name=_en)
        print(f"[{cfg.get('rung')}] co-residency gate PASSED "
              f"({n_inst} instances)")
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
        b = alloc_block_buffers(cfg.get("rung"))
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
    if cfg.get("rung") in ("mega", "megafg"):
        # deadlock-safety HARD GATE before any launch
        _en = ("TASK_DSV3_FFN_MEGA_FG_V2" if cfg.get("rung") == "megafg"
               else "TASK_DSV3_FFN_MEGA_V2")
        n_inst = assert_mega_coresidency(os.path.join(out_dir, "compile"),
                                         pk.num_workers, cfg["ntm"],
                                         enum_name=_en)
        result["mega_coresidency_instances"] = n_inst
        print(f"[{cfg.get('rung')}] co-residency gate PASSED "
              f"({n_inst} instances)")

    # Reviewer-mandated skip/race gate: poison every intermediate + output
    # buffer (floats -> NaN, fp8 bytes -> 0xFF = e4m3 NaN, meta ints ->
    # INT32_MAX) BEFORE the run. Any first-iteration protocol race / skipped
    # stage that consumes an unwritten buffer propagates NaN into out (or
    # corrupts meta/magic); caught by the post-run NaN scan + magic check +
    # the bit-compare of outs.pt against a clean run. Never-consumed slots
    # (y13/i_fp8 rows >= active_count) legitimately retain poison.
    if spec.get("poison_fill"):
        for b in all_bufs:
            for _name, t in b.items():
                if t.dtype in (torch.float32, torch.bfloat16):
                    t.fill_(float("nan"))
                elif t.dtype == torch.uint8:
                    t.fill_(0xFF)
                elif t.dtype == torch.int32:
                    t.fill_(0x7FFFFFFF)
        torch.cuda.synchronize()

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

    if spec.get("save_outs"):
        torch.save({i: {"out": all_bufs[i]["out"].cpu(),
                        "meta": all_bufs[i]["meta"].cpu()}
                    for i in range(L)}, os.path.join(out_dir, "outs.pt"))
        result["out_nan_counts"] = [
            int(torch.isnan(b["out"].float()).sum()) for b in all_bufs]

    # Optional: dump each block's ACTUAL chain input (x0 for block 0, block
    # i-1's out for i>0; steady-state deterministic) so the v1 driver can be
    # run on the SAME bytes (--load-dir): writes b{i}/hidden.bin only — the
    # weight files come from the seed dumps (identical generation).
    if spec.get("dump_block_inputs"):
        dd = spec["dump_block_inputs"]
        prev = x0
        for i in range(L):
            d = os.path.join(dd, f"b{i}")
            os.makedirs(d, exist_ok=True)
            prev.contiguous().view(torch.uint16).cpu().numpy().tofile(
                os.path.join(d, "hidden.bin"))
            prev = all_bufs[i]["out"]

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
