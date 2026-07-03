"""Pure-torch references for the DSv3 fused-ATTN v2 task chain (Step 3b).

Mirrors the v1 kernel's math (attn_block_megakernel_sm100.cuh) in fp32/fp64:
  - TWO quant emulations (design-review MAJOR fix):
      ue8m0_quant_dequant : ys = max(amax/448, 1e-10) -> yq = 2^ceil (the
          k_enc_ue8m0/k_dec_ue8m0 semantics; P0 quant, q_a requant, o_proj
          quant)
      raw448_quant_dequant: ys used DIRECTLY, no pow2 rounding (the MLA
          merge's g_attn_deq quant — NOT UE8M0)
  - fp8 e4m3 round-trips via torch.float8_e4m3fn (byte-exact vs
    __nv_fp8_e4m3 per the 3a R4 finding)
  - GEMV refs in fp32 (the kernel MACs in paired fp16 -> ~0.1-0.2% class
    tolerance, gated by cos)
  - flash-MLA reference as a straight fp64 softmax-attention

All refs are ON-KERNEL-INPUT capable: each op's ref takes the upstream
tensor(s) the kernel actually consumed so per-op error is isolated.
"""

import math

import torch

HIDDEN = 7168
QLORA = 1536
KVLORA = 512
QKROPE = 64
QKHEAD = 576
VHEAD = 128
QKVAN = 2176
HLOCAL = 16
OIN = 2048
GRP = 128
SPLITS = 8
EPS = 1e-6
FP8MAX = 448.0
COSSIN_STRIDE = 128
COSSIN_SINOFF = 64

# softmax scale: (1/sqrt(192)) * mscale^2, mscale = 0.1*ln(40)+1
MSCALE = 0.1 * math.log(40.0) + 1.0
SM_SCALE = (1.0 / math.sqrt(192.0)) * MSCALE * MSCALE


def bf16r(x: torch.Tensor) -> torch.Tensor:
    """Round fp32 -> bf16 -> fp32 (the kernel's k_bf16)."""
    return x.to(torch.bfloat16).float()


def fp8_decode(b: torch.Tensor) -> torch.Tensor:
    """uint8 bytes -> e4m3 values (fp32)."""
    return b.contiguous().view(torch.float8_e4m3fn).float()


def fp8_roundtrip(v: torch.Tensor) -> torch.Tensor:
    """fp32 -> e4m3 (RN-satfinite via torch cast) -> fp32."""
    return v.to(torch.float8_e4m3fn).float()


def _ceil_pow2(ys: torch.Tensor) -> torch.Tensor:
    """k_dec_ue8m0(k_enc_ue8m0(ys)) for ys > 0: 2^e if mantissa==0 else
    2^(e+1) (exact-pow2 stays, else round UP to the next pow2)."""
    m, e = torch.frexp(ys.double())  # ys = m * 2^e, m in [0.5, 1)
    yq = torch.where(m == 0.5, torch.ldexp(torch.ones_like(m), e - 1),
                     torch.ldexp(torch.ones_like(m), e))
    return yq.float()


def _grouped(v: torch.Tensor, grp: int = GRP):
    assert v.shape[-1] % grp == 0
    return v.reshape(*v.shape[:-1], v.shape[-1] // grp, grp)


def ue8m0_quant_dequant(v: torch.Tensor, grp: int = GRP) -> torch.Tensor:
    """Per-128-group: ys=max(amax/448,1e-10) -> yq=ceil-pow2 -> fp8
    roundtrip of clamp(v/yq) * yq. Mirrors quant_hidden/quant_ue8m0/q_a
    requant paths exactly."""
    g = _grouped(v.float(), grp)
    amax = g.abs().amax(dim=-1, keepdim=True)
    ys = torch.clamp(amax / FP8MAX, min=1e-10)
    yq = _ceil_pow2(ys)
    q = torch.clamp(g / yq, -FP8MAX, FP8MAX)
    return (fp8_roundtrip(q) * yq).reshape(v.shape)


def raw448_quant_dequant(v: torch.Tensor, grp: int = GRP) -> torch.Tensor:
    """MLA-merge quant: ys used RAW (no pow2 rounding)."""
    g = _grouped(v.float(), grp)
    amax = g.abs().amax(dim=-1, keepdim=True)
    ys = torch.clamp(amax / FP8MAX, min=1e-10)
    q = torch.clamp(g / ys, -FP8MAX, FP8MAX)
    return (fp8_roundtrip(q) * ys).reshape(v.shape)


def block_scaled_weight(w_bytes: torch.Tensor, wsc: torch.Tensor):
    """fp8 bytes [N,K] + per-128-block scale [N/128, K/128] -> fp32 [N,K]."""
    N, K = w_bytes.shape
    wf = fp8_decode(w_bytes)
    s = wsc.float().repeat_interleave(GRP, dim=0).repeat_interleave(GRP,
                                                                    dim=1)
    return wf * s[:N, :K]


# ---------------------------------------------------------------------------
# per-op references
# ---------------------------------------------------------------------------
def ref_p0_adeq(x: torch.Tensor, input_ln_w: torch.Tensor) -> torch.Tensor:
    """Phase-0: rmsnorm (fp32, single bf16 round) + UE8M0 quant-dequant.
    NOTE the rms reduction association differs from the kernel tree ->
    yq/normed can flip a knife-edge ULP; gate is cos (the bit gate is the
    v1 driver A/B)."""
    xf = x.reshape(-1).float()
    rcp = torch.rsqrt(xf.square().mean() + EPS)
    normed = bf16r(xf * (rcp * input_ln_w.float()))
    return ue8m0_quant_dequant(normed)


def ref_gemv(a_deq: torch.Tensor, w_bytes: torch.Tensor,
             wsc: torch.Tensor) -> torch.Tensor:
    """out[n] = bf16(sum_k a[k] * W[n,k] * wsc[n//128,k//128]) in fp32."""
    W = block_scaled_weight(w_bytes, wsc)
    return bf16r(W @ a_deq.float())


def ref_p0_qkva(x, input_ln_w, qkv_a_w, qkv_a_s):
    a_deq = ref_p0_adeq(x, input_ln_w)
    return a_deq, ref_gemv(a_deq, qkv_a_w, qkv_a_s)


def ref_q_deq(g_qkva: torch.Tensor, q_a_ln_w: torch.Tensor) -> torch.Tensor:
    """q_a-ln + UE8M0 requant ON the kernel's g_qkva."""
    qa = g_qkva.reshape(-1)[:QLORA].float()
    rcp = torch.rsqrt(qa.square().mean() + EPS)
    # kernel: k_bf16(src*q_rcp*w) with association (src*q_rcp)*w
    normed = bf16r(qa * rcp * q_a_ln_w.float())
    return ue8m0_quant_dequant(normed)


def ref_kv_row(g_qkva: torch.Tensor, kv_a_ln_w: torch.Tensor,
               cos_sin: torch.Tensor, pos: int) -> torch.Tensor:
    """The appended kv_cache row [step]: kv_a-ln (0:512) + roped k_pe
    (512:576), each bf16-rounded like the kernel."""
    v = g_qkva.reshape(-1).float()
    clat = v[QLORA:QLORA + KVLORA]
    rcp = torch.rsqrt(clat.square().mean() + EPS)
    row = torch.empty(QKHEAD, dtype=torch.float32, device=v.device)
    row[:KVLORA] = bf16r(clat * rcp * kv_a_ln_w.float())
    kpe = v[QLORA + KVLORA:QLORA + KVLORA + QKROPE]
    cs = cos_sin[pos].float()
    for i in range(QKROPE // 2):
        d0 = 2 * i
        c = cs[d0]
        s = cs[COSSIN_SINOFF + d0]
        k0, k1 = kpe[d0], kpe[d0 + 1]
        row[KVLORA + d0] = bf16r(k0 * c - k1 * s)
        row[KVLORA + d0 + 1] = bf16r(k1 * c + k0 * s)
    return bf16r(row)  # stored as bf16


def ref_qpe(q_deq: torch.Tensor, q_b_w: torch.Tensor, q_b_s: torch.Tensor,
            cos_sin: torch.Tensor, pos: int) -> torch.Tensor:
    """q_b GEMV (fp32 ref of the fp16-MAC kernel) + fused YaRN rope on the
    pe-part, with the kernel's exact bf16 rounding points."""
    y = ref_gemv(q_deq, q_b_w, q_b_s)  # already bf16-rounded sums
    y = y.reshape(HLOCAL, QKHEAD).clone()
    cs = cos_sin[pos].float()
    for i in range(QKROPE // 2):
        d0 = 2 * i
        c = cs[d0]
        s = cs[COSSIN_SINOFF + d0]
        q0 = y[:, KVLORA + d0].clone()
        q1 = y[:, KVLORA + d0 + 1].clone()
        y[:, KVLORA + d0] = bf16r(q0 * c - q1 * s)
        y[:, KVLORA + d0 + 1] = bf16r(q1 * c + q0 * s)
    return y.reshape(-1)


def ref_mla_attn(g_qpe: torch.Tensor, kv: torch.Tensor,
                 KV: int) -> torch.Tensor:
    """fp64 flash-equivalent: per head h, softmax((kv[:KV] @ q_h) * sm) @
    kv[:KV, :512]. Returns (16, 512) fp32 = the pre-quant merge (compare vs
    g_attn with a bf16 round)."""
    q = g_qpe.reshape(HLOCAL, QKHEAD).double()
    K = kv[:KV].double()  # (KV, 576)
    scores = (q @ K.T) * SM_SCALE  # (16, KV)
    p = torch.softmax(scores, dim=-1)
    out = p @ K[:, :KVLORA]  # (16, 512)
    return out.float()


def ref_wuv(g_attn_deq: torch.Tensor, kvbv_w: torch.Tensor,
            kvbv_s: torch.Tensor) -> torch.Tensor:
    """g_red[h*128+n] = bf16(sum_k deq[h,k] * W[h,n,k] * s[h,k//128])."""
    a = g_attn_deq.reshape(HLOCAL, KVLORA).float()
    wf = fp8_decode(kvbv_w.reshape(HLOCAL, VHEAD, KVLORA))
    s = kvbv_s.reshape(HLOCAL, KVLORA // GRP).float()
    s_full = s.repeat_interleave(GRP, dim=1)  # (16, 512)
    out = torch.einsum("hnk,hk->hn", wf * s_full[:, None, :], a)
    return bf16r(out.reshape(-1))


def ref_oproj(g_red: torch.Tensor, oproj_w: torch.Tensor,
              oproj_s: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """UE8M0 quant of g_red + o_proj GEMV + fused residual add (the kernel's
    out[n] = bf16(k_bf16(dot + resid)))."""
    o_deq = ue8m0_quant_dequant(g_red.reshape(-1).float())
    W = block_scaled_weight(oproj_w, oproj_s)
    dot = W @ o_deq
    return bf16r(dot + residual.reshape(-1).float())


def ref_full_chain(x, input_ln_w, q_a_ln_w, kv_a_ln_w, qkv_a_w, qkv_a_s,
                   q_b_w, q_b_s, kvbv_w, kvbv_s, oproj_w, oproj_s,
                   cos_sin, kv_prefix: torch.Tensor, pos: int):
    """End-to-end fp32/fp64 reference for ONE decode step at position `pos`
    with kv_prefix = the (pos, 576) bf16 history. Returns a dict of every
    stage."""
    dev = x.device
    a_deq, g_qkva = ref_p0_qkva(x, input_ln_w, qkv_a_w, qkv_a_s)
    q_deq = ref_q_deq(g_qkva, q_a_ln_w)
    kv_row = ref_kv_row(g_qkva, kv_a_ln_w, cos_sin, pos)
    g_qpe = ref_qpe(q_deq, q_b_w, q_b_s, cos_sin, pos)
    kv = torch.zeros((pos + 1, QKHEAD), dtype=torch.float32, device=dev)
    if pos > 0:
        kv[:pos] = kv_prefix[:pos].float()
    kv[pos] = kv_row
    attn = ref_mla_attn(g_qpe, kv, pos + 1)          # (16,512) pre-round
    g_attn = bf16r(attn.reshape(-1))                 # kernel rounds to bf16
    g_attn_deq = raw448_quant_dequant(
        g_attn.reshape(HLOCAL, KVLORA)).reshape(-1)  # merge quant (raw 448)
    g_red = ref_wuv(g_attn_deq, kvbv_w, kvbv_s)
    out = ref_oproj(g_red, oproj_w, oproj_s, x)      # residual = x
    return {
        "a_deq": a_deq, "g_qkva": g_qkva, "q_deq": q_deq, "kv_row": kv_row,
        "g_qpe": g_qpe, "g_attn": g_attn, "g_attn_deq": g_attn_deq,
        "g_red": g_red, "out": out,
        "score_std": float(((kv[:pos + 1].double() @ g_qpe.reshape(
            HLOCAL, QKHEAD).double().T) * SM_SCALE).std()),
    }
