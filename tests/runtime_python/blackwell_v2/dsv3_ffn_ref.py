"""PyTorch references for the DSv3 fused-FFN v2 task chain (Step 3a).

Emulates the v1 ffn_full_megakernel math in torch:
  rmsnorm -> UE8M0 group quant -> router GEMV (split-K partials + bf16
  logits) -> topk-sigmoid group routing with EP-local filter -> W13 + shared
  gate_up FP8 group-GEMV -> silu + UE8M0 requant -> W2 + shared down ->
  bf16 out.

Exactness notes (design doc / Codex review):
  - UE8M0 quant emulation is exact (exponent arithmetic via frexp +
    float8_e4m3fn cast); byte-level agreement is EXPECTED and reported.
  - GEMV refs accumulate in fp32 (kernel: fp16 packed inner dot per 16-elem
    group, fp32 across groups) -> tolerance compare (cos / rel_max).
  - Routing selection ref runs in float64 on the bf16 logits (same as the
    ffn_ws_driver host predictor); on non-tie inputs the selected experts
    must match the kernel meta EXACTLY.
  - silu ref uses torch sigmoid (kernel: silu_fast/__expf) -> tolerance.
"""

import torch

HIDDEN = 7168
W13_N = 1024
W2_K = 512
W2_N = 7168
E_LOCAL = 128
GRP = 128
KG1 = HIDDEN // GRP  # 56
KG2 = W2_K // GRP    # 4
NB1 = W13_N // GRP   # 8
NB2 = W2_N // GRP    # 56
MAX_ACTIVE = 8
ROUTER_N = 256
RKSPLIT = 4
NUM_GROUPS = 8
EXPERTS_PER_GROUP = ROUTER_N // NUM_GROUPS  # 32
TOPK_GROUP = 4
TOPK_EXPERTS = 8
SH_GU_N = 512
SH_DN_K = 256
KG_SHGU = HIDDEN // GRP  # 56
KG_SHDN = SH_DN_K // GRP  # 2
NB_SHGU = SH_GU_N // GRP  # 4
NB_SHDN = W2_N // GRP     # 56
RMS_EPS = 1e-6

META_INTS = 24
META_MAGIC = 0xD5F3


def fp8_decode(u8: torch.Tensor) -> torch.Tensor:
    """uint8 bytes -> e4m3 values as float32."""
    return u8.view(torch.float8_e4m3fn).float()


def quant_scale_ref(amax: torch.Tensor) -> torch.Tensor:
    """Exact emulation of v1 quant_scale: decode_ue8m0(encode_ue8m0(amax/448)).
    encode rounds the exponent UP unless the input is an exact power of two.
    The division MUST be fp32 (the kernel divides in fp32; a double division
    could land on the other side of a power-of-two boundary)."""
    s_in = torch.clamp(amax.float() / 448.0, min=1e-30)
    m, e = torch.frexp(s_in)  # s_in = m * 2^e, m in [0.5, 1)
    # IEEE mantissa==0 <=> s_in is a power of two <=> m == 0.5 -> exponent k =
    # e-1 (unbiased); otherwise ceil -> e.
    exp = torch.where(m == 0.5, e - 1, e)
    exp = torch.clamp(exp, min=-127, max=128)
    return torch.ldexp(torch.ones_like(s_in, dtype=torch.float64),
                       exp).float()


def ue8m0_group_quant(x_f32: torch.Tensor, group: int = GRP):
    """v1 quant_group_warp semantics: per-`group` amax -> pow2 scale ->
    e4m3 satfinite cast. x: (..., K). Returns (q_u8, scale_f32[..., K//group])."""
    orig_shape = x_f32.shape
    K = orig_shape[-1]
    assert K % group == 0
    xg = x_f32.reshape(-1, K // group, group)
    amax = xg.abs().amax(dim=-1)
    s = quant_scale_ref(amax)
    inv = 1.0 / s
    v = xg * inv.unsqueeze(-1)
    # satfinite: clamp to the e4m3 finite max (448); the kernel's to_f8 is
    # __NV_SATFINITE. By construction |v| <= 448 (s >= amax/448), so the clamp
    # only guards float-rounding edges.
    v = v.clamp(-448.0, 448.0)
    q = v.to(torch.float8_e4m3fn).view(torch.uint8)
    return (q.reshape(orig_shape),
            s.reshape(*orig_shape[:-1], K // group))


def ref_rmsnorm_f32(x_bf16: torch.Tensor, w_bf16: torch.Tensor):
    x = x_bf16.float()
    w = w_bf16.float()
    rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + RMS_EPS)
    return (x * rms * w)


def ref_router(normed_bf16: torch.Tensor, gate_w_bf16: torch.Tensor):
    """Returns (inter_ref f32[256,4] split-K quarter partials,
    logits_bf16[256])."""
    n = normed_bf16.float().reshape(HIDDEN)
    w = gate_w_bf16.float()  # (256, 7168)
    Kc = HIDDEN // RKSPLIT
    inter = torch.empty(ROUTER_N, RKSPLIT, device=n.device, dtype=torch.float32)
    for sp in range(RKSPLIT):
        inter[:, sp] = w[:, sp * Kc:(sp + 1) * Kc] @ n[sp * Kc:(sp + 1) * Kc]
    logits = inter.sum(dim=1).to(torch.bfloat16)
    return inter, logits


def ref_topk_sigmoid(inter_f32: torch.Tensor, bias_f32: torch.Tensor,
                     les: int, nle: int, rsf: float):
    """v1 Phase C in float64 selection (driver host-predictor semantics).
    Consumes the KERNEL's inter partials for an exact selection gate.
    Returns dict(meta ints, logits_bf16, weights list)."""
    tot = inter_f32.float().sum(dim=1)  # left-fold over 4 == sum, fp32
    logits = tot.to(torch.bfloat16)
    lg = logits.double()
    sig = torch.sigmoid(lg)
    biased = sig + bias_f32.double()
    # group top-2 sums -> top-4 groups
    bg = biased.reshape(NUM_GROUPS, EXPERTS_PER_GROUP)
    top2 = bg.topk(2, dim=1).values.sum(dim=1)
    gsel = torch.zeros(NUM_GROUPS, dtype=torch.bool)
    gs = top2.cpu()
    chosen = []
    for _ in range(TOPK_GROUP):
        best, bi = -1e300, 0
        for g in range(NUM_GROUPS):
            if not gsel[g] and float(gs[g]) > best:
                best, bi = float(gs[g]), g
        gsel[bi] = True
        chosen.append(bi)
    masked = biased.clone()
    for g in range(NUM_GROUPS):
        if not gsel[g]:
            masked[g * EXPERTS_PER_GROUP:(g + 1) * EXPERTS_PER_GROUP] = -10000.0
    # global top-8, index tie-break
    mb = masked.cpu()
    cand = sorted(range(ROUTER_N), key=lambda e: (-float(mb[e]), e))
    top8 = cand[:TOPK_EXPERTS]
    sig_c = sig.cpu()
    wsum = float(sum(float(sig_c[e]) for e in top8))
    inv = 1.0 / (wsum + 1e-20)
    experts, weights = [], []
    for e in top8:
        if les <= e < les + nle and len(experts) < MAX_ACTIVE:
            experts.append(e - les)
            weights.append(float(sig_c[e]) * inv * rsf)
    meta = [len(experts), META_MAGIC]
    meta += experts + [0] * (MAX_ACTIVE - len(experts))
    wbits = weights + [0.0] * (MAX_ACTIVE - len(weights))
    return {
        "logits": logits,
        "global_top8": top8,
        "active_count": len(experts),
        "experts": experts,
        "weights": weights,
        "meta_ints": meta,
        "weights_padded": wbits,
    }


def _group_gemv_ref(a_q_u8, a_scale, w_q_u8, w_scale, n_rows, K):
    """FP8 group GEMV ref: y[n] = sum_g (sum_{k in g} a[k]*w[n,k]) *
    a_scale[g] * w_scale[n//128, g], fp32.
    a_q: (K,) u8; a_scale: (K//128,); w_q: (n_rows, K) u8;
    w_scale: (n_rows//128, K//128)."""
    kg = K // GRP
    a = fp8_decode(a_q_u8).reshape(kg, GRP)
    w = fp8_decode(w_q_u8).reshape(n_rows, kg, GRP)
    part = torch.einsum("nkg,kg->nk", w, a)  # per-group partial dots
    ws = w_scale.reshape(n_rows // GRP, kg).float()
    ws_full = ws.repeat_interleave(GRP, dim=0)  # (n_rows, kg)
    y = (part * a_scale.float().unsqueeze(0) * ws_full).sum(dim=1)
    return y


def ref_w13(a_fp8, a_scale, w13_u8, w13_scale, wgu_u8, wgu_scale, experts):
    """Returns (y13 f32[len(experts),1024], sg f32[512]) from KERNEL a_fp8."""
    ys = []
    for e in experts:
        ys.append(_group_gemv_ref(a_fp8, a_scale,
                                  w13_u8[e].reshape(-1), w13_scale[e],
                                  W13_N, HIDDEN))
    y13 = torch.stack(ys) if ys else torch.zeros(0, W13_N, device=a_fp8.device)
    sg = _group_gemv_ref(a_fp8, a_scale, wgu_u8.reshape(-1), wgu_scale,
                         SH_GU_N, HIDDEN)
    return y13, sg


def ref_silu_quant(y13_f32, sg_f32, active_count):
    """silu(gate)*up + UE8M0 requant. Returns (i_fp8 u8[ac,512],
    i_scale f32[ac,4], si_fp8 u8[256], si_scale f32[2])."""
    if active_count > 0:
        gate = y13_f32[:active_count, :512]
        up = y13_f32[:active_count, 512:]
        inter = torch.sigmoid(gate) * gate * up
        i_fp8, i_scale = ue8m0_group_quant(inter)
    else:
        dev = y13_f32.device
        i_fp8 = torch.zeros(0, W2_K, dtype=torch.uint8, device=dev)
        i_scale = torch.zeros(0, KG2, device=dev)
    sh = torch.sigmoid(sg_f32[:256]) * sg_f32[:256] * sg_f32[256:]
    si_fp8, si_scale = ue8m0_group_quant(sh)
    return i_fp8, i_scale, si_fp8, si_scale


def ref_w2(i_fp8, i_scale, si_fp8, si_scale, w2_u8, w2_scale, wdn_u8,
           wdn_scale, experts, weights):
    """out f32[7168] = sum_slot w_slot * W2[e_slot] @ i[slot] + WDN @ si."""
    dev = si_fp8.device
    out = torch.zeros(W2_N, dtype=torch.float32, device=dev)
    for s, (e, w) in enumerate(zip(experts, weights)):
        y = _group_gemv_ref(i_fp8[s].reshape(-1), i_scale[s],
                            w2_u8[e].reshape(-1), w2_scale[e], W2_N, W2_K)
        out += w * y
    out += _group_gemv_ref(si_fp8, si_scale, wdn_u8.reshape(-1), wdn_scale,
                           W2_N, SH_DN_K)
    return out


def ref_full_chain(t, les, nle, rsf):
    """t: dict of the block's input torch tensors (hidden, rms_w, router_w,
    bias, w13, w13_scale, w2, w2_scale, wgu, wgu_scale, wdn, wdn_scale).
    Pure-torch chain (REF routing). Returns dict of intermediates."""
    normed_f32 = ref_rmsnorm_f32(t["hidden"].reshape(-1), t["rms_w"])
    normed_bf16 = normed_f32.to(torch.bfloat16)
    a_fp8, a_scale = ue8m0_group_quant(normed_bf16.float())
    inter, logits = ref_router(normed_bf16, t["router_w"])
    routing = ref_topk_sigmoid(inter, t["bias"], les, nle, rsf)
    y13, sg = ref_w13(a_fp8, a_scale, t["w13"], t["w13_scale"], t["wgu"],
                      t["wgu_scale"], routing["experts"])
    i_fp8, i_scale, si_fp8, si_scale = ref_silu_quant(
        y13, sg, routing["active_count"])
    out = ref_w2(i_fp8, i_scale, si_fp8, si_scale, t["w2"], t["w2_scale"],
                 t["wdn"], t["wdn_scale"], routing["experts"],
                 routing["weights"])
    return {
        "normed_bf16": normed_bf16,
        "a_fp8": a_fp8,
        "a_scale": a_scale,
        "inter": inter,
        "logits": logits,
        "routing": routing,
        "y13": y13,
        "sg": sg,
        "i_fp8": i_fp8,
        "i_scale": i_scale,
        "si_fp8": si_fp8,
        "si_scale": si_scale,
        "out_f32": out,
        "out_bf16": out.to(torch.bfloat16),
    }
