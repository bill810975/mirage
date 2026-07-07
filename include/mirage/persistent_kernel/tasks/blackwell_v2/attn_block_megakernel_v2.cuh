/* Copyright 2026 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
#pragma once

// ============================================================================
// FUSED decode-attention megakernel in Runtime-V2 "megakernel-shape" (Form-2)
// format (T-E of the DSv3-decode-on-v2 effort). ONE v2 task runs the WHOLE
// decode attention for one token (bs=1, TP8 EP2), exactly like the v1
// tasks/blackwell/attn_block_megakernel_sm100.cuh, but hosted on the v2
// role-split runtime: 136 co-resident tasks == 136 workers (hard host-assert),
// self-syncing around the 3 in-op grid barriers via monotonic GMEM
// count-barriers (v1's grid-barrier algorithm rebuilt across co-resident v2
// tasks), CONSUMER-ONLY (128 physical threads / 4 warps).
//
// WHY consumer-only 128 threads, and how it stays BIT-EXACT vs v1's 256:
//   The v1 body is uniform-256-thread with block-wide __syncthreads() and NO
//   tcgen05 / CuTe / bar.sync — pure scalar-ILP FP8 GEMV + flash MLA. A v2
//   consumer role runs only W0-3 (128 threads); a __syncthreads() there would
//   wait the helper/controller warps (W4-7, in different loops) -> deadlock.
//   The correctness-first port therefore runs 128 PHYSICAL threads and:
//     * keeps V1_NTHREAD=256 / V1_NWARP=8 as the LOGICAL constants for every
//       VALUE-AFFECTING block collective (RMSNorm ss-tree, rms_rcp, MLA
//       TPR=NTHREAD/nr, MLA lmax/lsum, the red8[8] layout + v1's sequential
//       cross-warp combine order);
//     * uses PHYS_NTHREAD=128 / PHYS_NWARP=4 for PHYSICAL coverage (elementwise
//       grid-strided loops, gtid/gthreads, the GEMV/BMM gwarp/gwarps, the
//       per-group quant warp stride);
//     * "A/B thread emulation" for the 256-thread block reductions: physical
//       thread t plays v1 threads t (role A) and t+128 (role B). Because
//       lane(t+128)==lane(t), warp(t+128)==warp(t)+4, and every MLA TPR in
//       {1,2,4,8} divides 128, A's and B's warp-local shuffles are run
//       SEPARATELY (partials kept apart) then stored red8[warpl]=A,
//       red8[warpl+4]=B, after which v1's cross-warp combine over red8[0..8)
//       is VERBATIM. This reproduces v1's EXACT per-reduction fp result (not
//       just value-equivalent). (Codex thread 019f358e + ablation-logic-
//       reviewer vetted; same emulation the sibling de-fused dsv3_attn_v2.cuh
//       chain uses.)
//     * calls v1's GEMV/BMM device functions (gemv_grid_cpa_t,
//       gemv_grid_cpa_qb_rope_smem_t, gemv_grid_cpa_oproj_smem_t,
//       wuv_bmm_grid) VERBATIM — they have only __syncwarp, so they run
//       correctly at 128 threads; each output ROW is computed by exactly ONE
//       warp with the SAME K-reduction lane order regardless of gwarps (the
//       gwarps=136*4 re-stride only changes WHICH warp owns a row + how many
//       row-blocks per warp -> bit-identical output).
//
// The only NUMERIC caveat is the SAME one v1 has vs itself: FMA-contraction
// across compilation units. It is validated by the bit-match harness (the op
// is deterministic in isolation) with a per-element tolerance covering that
// contraction, exactly like the v1 fused vs its own gate.
//
// STRUCTURE vs v1 (the 3 adaptations):
//   1. __syncthreads() -> attn_v2_sync() = `bar.sync 6, 128` (ids 1-5 are
//      linear/rmsnorm/ffn/attn-v2-chain/AR; 6 is free). __syncwarp() unchanged.
//   2. attn_grid_barrier(136) -> attn_v2_grid_barrier(bar, site, need): a
//      MONOTONIC GMEM count-barrier (one u64 slot per barrier-site) that never
//      resets (need = num_tasks*(iter_num+1)), with v1's ALL-128-consumer-
//      thread __threadfence PRESERVED (a thread0-only fence would not publish
//      tid 1..127's stores). COARSE whole-grid (the per-slot fine-grained
//      variant regressed — memory feedback_ffn_megakernel_fg_counter_regress).
//   3. blockIdx.x -> task_offset (baked per-instance in TaskMetadata via the
//      runtime.cc task_offset=bid.x block for this task type). step (decode
//      position) from runtime_config.step[0], unchanged.
//
// TP8 ROW-PARALLEL o_proj: the kernel writes a residual-FREE partial and the
// downstream AllReduce (#3) sums across ranks + adds the residual ONCE. The v2
// builder MUST bind a ZERO residual buffer (do NOT re-enable a fused-residual
// epilogue at TP8) — this port calls v1's oproj GEMV which adds `residual`, so
// the ZERO binding is what keeps it residual-free. (Preserved via the builder,
// M3 plumb.)
//
// input_ptrs ABI — 14 slots (IDENTICAL to the v1 mega):
//   [0]  hidden(bf16 raw self.x) [1] qkv_a_w(fp8) [2] qkv_a_s(f32)
//   [3]  ln_weights(bf16 [input_ln(7168)|q_a_ln(1536)|kv_a_ln(512)])
//   [4]  q_b_w(fp8) [5] q_b_s(f32) [6] cos_sin(bf16 [cos(64)|sin(64)]/row)
//   [7]  kv_cache(bf16 flat, read history + write row[step], same buffer as
//   out) [8]  kvbv_w(fp8) [9] kvbv_s(f32 [H,1,4]) [10] oproj_w(fp8) [11]
//   oproj_s(f32) [12] residual(bf16 — ZERO at TP8) [13] scratch(u8 AttnScratch
//   base +
//        the 3-slot u64 GMEM barrier at the very top)
//   + out bound as output_ptrs[0].
// ============================================================================

#include "mirage/persistent_kernel/mpk_atoms.cuh"
#include "mirage/persistent_kernel/runtime_header.h"
#include "mirage/persistent_kernel/tasks/blackwell/attn_block_megakernel_sm100.cuh"
#include "mirage/persistent_kernel/tasks/blackwell_v2/attn_block_megakernel_v2_spec.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>
#include <stdint.h>

namespace kernel {
namespace attn_block_megakernel_v2 {

namespace v1a = ::kernel::attn_block_megakernel_sm100;

// Pin the spec-header CPA_RING_U4 to v1's macro so drift is a compile error.
// (V1's CPA_RING_U4 is a function-local constexpr, so re-declare the equality
// against the value; K_* / NTHREAD / NWARP / MLA_SPLITS come from the v1
// header as #defines.)
static_assert(NTHREAD == 256 && NWARP == 8,
              "the 128-thread A/B emulation assumes v1 NTHREAD=256 / NWARP=8");
static_assert(CPA_RING_U4 == 1024, "CPA_RING_U4 must match v1's ring depth");
static_assert(V1_NWARP == NWARP, "V1_NWARP must equal the v1 NWARP macro");

// ---- V2-safe intra-CTA barrier: replaces every block-wide __syncthreads()
// over the 128 consumer threads. Named barrier id 6 (1-5 taken). "memory"
// clobber orders the SMEM/GMEM accesses the barrier separates. ----
__device__ __forceinline__ void attn_v2_sync() {
  asm volatile("bar.sync 6, 128;" ::: "memory");
}

// ---- V2 megakernel-shape GMEM grid barrier: the monotonic replacement for
// v1's attn_grid_barrier(136). ONE u64 slot per barrier-site; never reset;
// need = num_tasks*(iter_num+1). Preserves v1's ALL-thread __threadfence: all
// 128 consumer threads fence their prior global stores BEFORE thread 0 bumps
// the count, so a peer that later passes the barrier sees every thread's
// writes (a thread0-only fence would publish only thread 0's stores). Called
// by ALL 128 consumer threads. ----
__device__ __forceinline__ void attn_v2_grid_barrier(unsigned long long *bar,
                                                     int site,
                                                     unsigned long long need) {
  attn_v2_sync();  // all 128 threads' prior work retired
  __threadfence(); // EVERY consumer thread publishes its global stores
  attn_v2_sync();  // ensure all fences retired before thread 0 bumps
  if (threadIdx.x == 0) {
    atom_add_release_gpu_u64(&bar[site], 1ull);
    while (ld_acquire_sys_u64(&bar[site]) < need) {
      __nanosleep(64);
    }
  }
  attn_v2_sync(); // broadcast: everyone leaves together, acquire peers' data
}

// ===========================================================================
//  A/B-EMULATED block collectives. Physical thread t (0..127) plays v1 thread
//  t (role A) and v1 thread t+128 (role B). warpl = t>>5 in [0,4).
// ===========================================================================

// Phase-0 RMSNorm(self.x) + UE8M0 per-128-group quant into block-local s_deq.
// EXACT-TREE 128-thread port of v1's rmsnorm_quant_hidden_block_smem: the
// sum-of-squares reduction reproduces v1's tree (per-thread partial stride
// V1_NTHREAD=256, per-warp shfl_XOR, warp-0 xor tree over red8[0..8),
// broadcast via red8[0], rsqrtf + K_EPS). The UE8M0 quant body is v1's,
// warp-strided by PHYS_NWARP (group math is warp-local -> value-identical).
__device__ __forceinline__ void
    rmsnorm_quant_hidden_block_smem_ab(__nv_bfloat16 const *__restrict__ self_x,
                                       __nv_bfloat16 const *__restrict__ ln_w,
                                       float *__restrict__ s_deq,
                                       float *__restrict__ red8,
                                       int n,
                                       int warpl,
                                       int lane) {
  int const tid = threadIdx.x;
  // --- ss = Σ(float(self_x[i]))² over [0:n], fp32. A/B: role A visits
  // i = tid, tid+256, ...; role B visits i = tid+128, tid+384, ... (== v1
  // threads t and t+128). Keep the partials SEPARATE.
  float psA = 0.f, psB = 0.f;
  for (int i = tid; i < n; i += NTHREAD) {
    float v = __bfloat162float(self_x[i]);
    psA += v * v;
  }
  for (int i = tid + 128; i < n; i += NTHREAD) {
    float v = __bfloat162float(self_x[i]);
    psB += v * v;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    psA += __shfl_xor_sync(0xffffffffu, psA, o);
    psB += __shfl_xor_sync(0xffffffffu, psB, o);
  }
  if (lane == 0) {
    red8[warpl] = psA;     // v1 warp warpl partial (role A)
    red8[warpl + 4] = psB; // v1 warp warpl+4 partial (role B)
  }
  attn_v2_sync();
  // v1: cross-warp xor-tree of the NWARP=8 partials inside warp 0 (lanes 0-7),
  // broadcast through red8[0].
  float ss = (tid < NWARP) ? red8[tid] : 0.f;
#pragma unroll
  for (int o = NWARP / 2; o > 0; o >>= 1) {
    ss += __shfl_xor_sync(0xffffffffu, ss, o);
  }
  if (tid == 0) {
    red8[0] = ss;
  }
  attn_v2_sync();
  ss = red8[0];   // uniform across the block
  attn_v2_sync(); // re-converge before red8 is reused by a later phase
  float rms_rcp = rsqrtf(ss / (float)n + K_EPS);

  // --- UE8M0 per-128-group quant of the NORMED value (v1 body verbatim; the
  // group->warp map is redistributed 8->PHYS_NWARP, value-identical). ---
  int ng = n / K_GRP;
  for (int gx = warpl; gx < ng; gx += PHYS_NWARP) {
    __nv_bfloat16 const *h = self_x + gx * K_GRP;
    __nv_bfloat16 const *w = ln_w + gx * K_GRP;
    float v[4];
    float mx = 1e-10f;
    __nv_bfloat162 const *h2 =
        reinterpret_cast<__nv_bfloat162 const *>(h) + lane * 2;
    __nv_bfloat162 const *w2 =
        reinterpret_cast<__nv_bfloat162 const *>(w) + lane * 2;
    float2 a0 = __bfloat1622float2(h2[0]);
    float2 a1 = __bfloat1622float2(h2[1]);
    float2 g0 = __bfloat1622float2(w2[0]);
    float2 g1 = __bfloat1622float2(w2[1]);
    v[0] = __bfloat162float(__float2bfloat16(a0.x * (rms_rcp * g0.x)));
    v[1] = __bfloat162float(__float2bfloat16(a0.y * (rms_rcp * g0.y)));
    v[2] = __bfloat162float(__float2bfloat16(a1.x * (rms_rcp * g1.x)));
    v[3] = __bfloat162float(__float2bfloat16(a1.y * (rms_rcp * g1.y)));
#pragma unroll
    for (int t = 0; t < 4; t++) {
      mx = fmaxf(mx, fabsf(v[t]));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    }
    float ys = fmaxf(mx / K_FP8MAX, 1e-10f);
    float yq = v1a::k_dec_ue8m0(v1a::k_enc_ue8m0(ys));
    float *d = s_deq + gx * K_GRP + lane * 4;
#pragma unroll
    for (int t = 0; t < 4; t++) {
      float q = fminf(fmaxf(v[t] / yq, -K_FP8MAX), K_FP8MAX);
      d[t] = (float)__nv_fp8_e4m3(q) * yq;
    }
  }
  attn_v2_sync(); // publish s_deq within the block before qkv_a reads it
}

// fp32 block reduction of Σ src[i]² over [0:n] -> 1/sqrt(mean+eps), uniform
// across the block. EXACT-TREE 128-thread port of v1's rms_rcp_block (which
// uses shfl_DOWN then a SEQUENTIAL Σ over red8[0..8)). A/B: role A partial ps
// over i=tid,+256; role B over i=tid+128,+256; shfl_down each; red8[warpl]=A,
// red8[warpl+4]=B; then the SEQUENTIAL Σ red8[i] i∈[0,8) matches v1 exactly.
__device__ __forceinline__ float rms_rcp_block_ab(float const *__restrict__ src,
                                                  int n,
                                                  float *__restrict__ red8,
                                                  int warpl,
                                                  int lane) {
  int const tid = threadIdx.x;
  float psA = 0.f, psB = 0.f;
  for (int i = tid; i < n; i += NTHREAD) {
    float x = src[i];
    psA += x * x;
  }
  for (int i = tid + 128; i < n; i += NTHREAD) {
    float x = src[i];
    psB += x * x;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    psA += __shfl_down_sync(0xffffffffu, psA, o);
    psB += __shfl_down_sync(0xffffffffu, psB, o);
  }
  if (lane == 0) {
    red8[warpl] = psA;
    red8[warpl + 4] = psB;
  }
  attn_v2_sync();
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < NWARP; i++) {
    ss += red8[i];
  }
  attn_v2_sync();
  return 1.0f / sqrtf(ss / n + K_EPS);
}

// ===========================================================================
//  FLASH MLA partial (v1 S9), A/B-emulated at 128 threads. Block (h,sp)
//  computes the un-normalized softmax over its KV sub-range [r0,r1). TPR is
//  computed from v1's LOGICAL NTHREAD=256 (value-affecting via the score
//  grouping); score / lmax / lsum use A/B emulation; exp + V-accum are
//  elementwise (value-identical). Reads kv_cache + g_qpe, writes g_mla_*.
// ===========================================================================
__device__ __noinline__ void
    mla_partial_ab(__nv_bfloat16 const *__restrict__ kv_cache,
                   float const *__restrict__ g_qpe,
                   float *__restrict__ g_mla_acc,
                   float *__restrict__ g_mla_m,
                   float *__restrict__ g_mla_l,
                   float *__restrict__ s_score,
                   float *__restrict__ red8,
                   int h,
                   int sp,
                   int r0,
                   int r1,
                   float sm) {
  int const tid = threadIdx.x;
  int const lane = tid & 31, warpl = tid >> 5; // for the lmax/lsum red8 stores
  float const *q = &g_qpe[h * K_QKHEAD];
  int const nr = r1 - r0;
  // TPR from v1's LOGICAL NTHREAD (256), collapsed to a power of two — VERBATIM
  // v1 selection. TPR in {1,2,4,8} => 128 % TPR == 0.
  int TPR = NTHREAD / (nr > 0 ? nr : 1);
  if (TPR < 1) {
    TPR = 1;
  }
  if (TPR > 8) {
    TPR = 8;
  }
  if (TPR >= 8) {
    TPR = 8;
  } else if (TPR >= 4) {
    TPR = 4;
  } else if (TPR >= 2) {
    TPR = 2;
  } else {
    TPR = 1;
  }
  int const laneInWarp = tid & 31;
  unsigned const grpmask =
      ((TPR >= 32) ? 0xffffffffu
                   : (((1u << TPR) - 1u) << ((laneInWarp / TPR) * TPR)));
  int const sub = tid % TPR; // == v1 (t+128)%TPR since 128%TPR==0
  // --- SCORE. A/B: run v1's per-row c-loop + shfl_down(width=TPR) reduction
  // for role A (v1 thread t, row = t/TPR) then role B (v1 thread t+128,
  // row = t/TPR + 128/TPR), each stepping rows_per_step = NTHREAD/TPR. Same
  // sub / grpmask; only the row index differs. Union over t∈[0,128) == v1's
  // full [0,256) thread set -> each s_score[rr] is v1's exact per-row
  // reduction.
  int const rows_per_step = NTHREAD / TPR;
  int const rowA = tid / TPR;
  int const rowB = rowA + 128 / TPR;
  for (int rr = rowA; rr < nr; rr += rows_per_step) {
    int r = r0 + rr;
    uint4 const *kvr =
        reinterpret_cast<uint4 const *>(&kv_cache[(size_t)r * K_QKHEAD]);
    float dot = 0.f;
    for (int c = sub; c < K_QKHEAD / 8; c += TPR) {
      uint4 kw = kvr[c];
      __nv_bfloat162 const *k2 = reinterpret_cast<__nv_bfloat162 const *>(&kw);
      float const *qc = &q[c * 8];
#pragma unroll
      for (int p = 0; p < 4; p++) {
        float2 kf = __bfloat1622float2(k2[p]);
        dot += qc[2 * p] * kf.x + qc[2 * p + 1] * kf.y;
      }
    }
#pragma unroll
    for (int o = TPR >> 1; o > 0; o >>= 1) {
      dot += __shfl_down_sync(grpmask, dot, o, TPR);
    }
    if (sub == 0) {
      s_score[rr] = dot * sm;
    }
  }
  for (int rr = rowB; rr < nr; rr += rows_per_step) {
    int r = r0 + rr;
    uint4 const *kvr =
        reinterpret_cast<uint4 const *>(&kv_cache[(size_t)r * K_QKHEAD]);
    float dot = 0.f;
    for (int c = sub; c < K_QKHEAD / 8; c += TPR) {
      uint4 kw = kvr[c];
      __nv_bfloat162 const *k2 = reinterpret_cast<__nv_bfloat162 const *>(&kw);
      float const *qc = &q[c * 8];
#pragma unroll
      for (int p = 0; p < 4; p++) {
        float2 kf = __bfloat1622float2(k2[p]);
        dot += qc[2 * p] * kf.x + qc[2 * p + 1] * kf.y;
      }
    }
#pragma unroll
    for (int o = TPR >> 1; o > 0; o >>= 1) {
      dot += __shfl_down_sync(grpmask, dot, o, TPR);
    }
    if (sub == 0) {
      s_score[rr] = dot * sm;
    }
  }
  attn_v2_sync();
  // --- row-max. A/B: lmaxA over rr=tid,+256; lmaxB over rr=tid+128,+256;
  // shfl_xor each; red8[warpl]=A, red8[warpl+4]=B; then v1's SEQUENTIAL max
  // over red8[0..8).
  float lmaxA = -1e30f, lmaxB = -1e30f;
  for (int rr = tid; rr < nr; rr += NTHREAD) {
    lmaxA = fmaxf(lmaxA, s_score[rr]);
  }
  for (int rr = tid + 128; rr < nr; rr += NTHREAD) {
    lmaxB = fmaxf(lmaxB, s_score[rr]);
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    lmaxA = fmaxf(lmaxA, __shfl_xor_sync(0xffffffffu, lmaxA, o));
    lmaxB = fmaxf(lmaxB, __shfl_xor_sync(0xffffffffu, lmaxB, o));
  }
  if (lane == 0) {
    red8[warpl] = lmaxA;
    red8[warpl + 4] = lmaxB;
  }
  attn_v2_sync();
  float gmax = -1e30f;
#pragma unroll
  for (int i = 0; i < NWARP; i++) {
    gmax = fmaxf(gmax, red8[i]);
  }
  attn_v2_sync();
  // exp in place (elementwise; A and B disjoint rr sets cover all nr).
  for (int rr = tid; rr < nr; rr += NTHREAD) {
    s_score[rr] = __expf(s_score[rr] - gmax);
  }
  for (int rr = tid + 128; rr < nr; rr += NTHREAD) {
    s_score[rr] = __expf(s_score[rr] - gmax);
  }
  attn_v2_sync();
  // --- exp-sum. A/B like row-max but Σ.
  float lsumA = 0.f, lsumB = 0.f;
  for (int rr = tid; rr < nr; rr += NTHREAD) {
    lsumA += s_score[rr];
  }
  for (int rr = tid + 128; rr < nr; rr += NTHREAD) {
    lsumB += s_score[rr];
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    lsumA += __shfl_xor_sync(0xffffffffu, lsumA, o);
    lsumB += __shfl_xor_sync(0xffffffffu, lsumB, o);
  }
  if (lane == 0) {
    red8[warpl] = lsumA;
    red8[warpl + 4] = lsumB;
  }
  attn_v2_sync();
  float gsum = 0.f;
#pragma unroll
  for (int i = 0; i < NWARP; i++) {
    gsum += red8[i];
  }
  int base = h * MLA_SPLITS + sp;
  if (tid == 0) {
    g_mla_m[base] = (nr > 0) ? gmax : -1e30f;
    g_mla_l[base] = gsum;
  }
  // V accumulation: elementwise over d (each d independent; A/B stride to
  // cover all K_KVLORA). accv[d] = Σ_rr s_score[rr]*V[r0+rr][d].
  float *accv = &g_mla_acc[(size_t)base * K_KVLORA];
  for (int d = tid; d < K_KVLORA; d += NTHREAD) {
    float acc = 0.f;
    for (int rr = 0; rr < nr; rr++) {
      acc += s_score[rr] *
             __bfloat162float(kv_cache[(size_t)(r0 + rr) * K_QKHEAD + d]);
    }
    accv[d] = acc;
  }
  for (int d = tid + 128; d < K_KVLORA; d += NTHREAD) {
    float acc = 0.f;
    for (int rr = 0; rr < nr; rr++) {
      acc += s_score[rr] *
             __bfloat162float(kv_cache[(size_t)(r0 + rr) * K_QKHEAD + d]);
    }
    accv[d] = acc;
  }
  attn_v2_sync();
}

// FLASH MLA merge (v1 S10+S11 fused), A/B-emulated at 128 threads. Merges head
// h's nsp partials and quantizes its 512 attn_out inline. The gmax/denom
// reductions are per-thread SEQUENTIAL over the nsp splits (NOT cross-thread —
// value-identical at any thread count). The d-loop is elementwise (A/B stride).
// The warpl<KGv(4) dequant block is already <=4 warps in v1 -> physical warps
// 0..3 do the same 4 groups with the same lane reduction, NO A/B needed.
// Sets g_head_wuv_ready[h]=1 (device release) at the end (lever 5).
__device__ __noinline__ void
    mla_merge_quant_ab(float *__restrict__ g_attn,
                       float *__restrict__ g_attn_deq,
                       float const *__restrict__ g_mla_acc,
                       float const *__restrict__ g_mla_m,
                       float const *__restrict__ g_mla_l,
                       float *__restrict__ s_attn,
                       int h,
                       int nsp,
                       int *__restrict__ g_head_wuv_ready) {
  int const tid = threadIdx.x, lane = tid & 31, warpl = tid >> 5;
  float const *mrow = &g_mla_m[h * MLA_SPLITS];
  float const *lrow = &g_mla_l[h * MLA_SPLITS];
  float gmax = -1e30f;
#pragma unroll
  for (int s = 0; s < MLA_SPLITS; s++) {
    if (s < nsp) {
      gmax = fmaxf(gmax, mrow[s]);
    }
  }
  float denom = 0.f;
#pragma unroll
  for (int s = 0; s < MLA_SPLITS; s++) {
    if (s < nsp) {
      denom += lrow[s] * __expf(mrow[s] - gmax);
    }
  }
  float inv = (denom > 0.f) ? 1.0f / denom : 0.f;
  for (int d = tid; d < K_KVLORA; d += NTHREAD) {
    float acc = 0.f;
#pragma unroll
    for (int s = 0; s < MLA_SPLITS; s++) {
      if (s < nsp) {
        float w = __expf(mrow[s] - gmax);
        acc += g_mla_acc[((size_t)(h * MLA_SPLITS + s)) * K_KVLORA + d] * w;
      }
    }
    float v = v1a::k_bf16(acc * inv);
    s_attn[d] = v;
    g_attn[h * K_KVLORA + d] = v;
  }
  for (int d = tid + 128; d < K_KVLORA; d += NTHREAD) {
    float acc = 0.f;
#pragma unroll
    for (int s = 0; s < MLA_SPLITS; s++) {
      if (s < nsp) {
        float w = __expf(mrow[s] - gmax);
        acc += g_mla_acc[((size_t)(h * MLA_SPLITS + s)) * K_KVLORA + d] * w;
      }
    }
    float v = v1a::k_bf16(acc * inv);
    s_attn[d] = v;
    g_attn[h * K_KVLORA + d] = v;
  }
  attn_v2_sync();
  int KGv = K_KVLORA / K_GRP; // 4
  // v1 uses warps 0..3 (of 8) for the 4 groups; at 128 threads warps 0..3 are
  // ALL the consumer warps -> same layout, same lane reduction, bit-identical.
  if (warpl < KGv) {
    float const *ar = &s_attn[warpl * K_GRP];
    float *dq = &g_attn_deq[h * K_KVLORA + warpl * K_GRP];
    float mx = 1e-10f;
    for (int j = lane; j < K_GRP; j += 32) {
      float a = fabsf(ar[j]);
      mx = fmaxf(mx, a);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      float ot = __shfl_xor_sync(0xffffffffu, mx, o);
      mx = fmaxf(mx, ot);
    }
    float ys = fmaxf(mx / K_FP8MAX, 1e-10f);
    for (int j = lane; j < K_GRP; j += 32) {
      float vq = fminf(fmaxf(ar[j] / ys, -K_FP8MAX), K_FP8MAX);
      dq[j] = (float)__nv_fp8_e4m3(vq) * ys;
    }
  }
  // Lever 5 (WUV_HEAD_SPINWAIT): the dequant loop above is MULTI-WARP (warps
  // 0..3 each wrote a 128-wide slice). This block-wide barrier makes all 4
  // groups' g_attn_deq[h] writes complete + visible to tid0 before tid0
  // publishes the readiness flag; then tid0 does a DEVICE-scope release so
  // head h's deq is visible to other CTAs before the flag flips (device scope
  // is correct — all 16 heads live on this rank).
  attn_v2_sync();
  if (threadIdx.x == 0) {
    __threadfence(); // device release
    asm volatile(
        "st.global.release.gpu.u32 [%0], %1;\n" ::"l"(&g_head_wuv_ready[h]),
        "r"(1)
        : "memory");
  }
}

// ===========================================================================
//  MPK v2 task entry. Consumer-only (128 threads). Mirrors the v1 mega's ABI
//  but reads task_offset (NOT merge_task_offset) as the logical CTA id and
//  self-syncs via the monotonic GMEM barrier. `num_tasks` == num_workers.
// ===========================================================================
__device__ __noinline__ void attn_block_megakernel_v2_task_impl(
    mirage::runtime::TaskDesc const *task_desc,
    int task_offset,
    int num_tasks,
    int iter_num,
    mirage::runtime::RuntimeConfig const &runtime_config) {
  __nv_bfloat16 const *hidden =
      static_cast<__nv_bfloat16 const *>(task_desc->input_ptrs[0]);
  __nv_fp8_e4m3 const *qkv_a_w =
      static_cast<__nv_fp8_e4m3 const *>(task_desc->input_ptrs[1]);
  float const *qkv_a_s = static_cast<float const *>(task_desc->input_ptrs[2]);
  __nv_bfloat16 const *ln_weights =
      static_cast<__nv_bfloat16 const *>(task_desc->input_ptrs[3]);
  // Phase-0 deep-fusion concat: ln_weights = [input_ln(7168) | q_a_ln(1536) |
  // kv_a_ln(512)] = 9216. Same static_assert guard as v1.
  static_assert(K_HIDDEN == 7168 && K_QLORA == 1536 && K_KVLORA == 512,
                "Phase-0 ln_weights layout [input_ln(7168)|q_a_ln(1536)|"
                "kv_a_ln(512)]=9216 — these constants pin the concat offsets.");
  static_assert((K_HIDDEN + K_QLORA + K_KVLORA) == 9216,
                "Phase-0 ln_weights total length must be 9216-d.");
  __nv_bfloat16 const *input_ln_w = ln_weights;          // offset 0
  __nv_bfloat16 const *q_a_ln_w = ln_weights + K_HIDDEN; // offset 7168
  __nv_bfloat16 const *kv_a_ln_w =
      ln_weights + K_HIDDEN + K_QLORA; // offset 8704
  (void)input_ln_w;
  __nv_fp8_e4m3 const *q_b_w =
      static_cast<__nv_fp8_e4m3 const *>(task_desc->input_ptrs[4]);
  float const *q_b_s = static_cast<float const *>(task_desc->input_ptrs[5]);
  __nv_bfloat16 const *cos_sin =
      static_cast<__nv_bfloat16 const *>(task_desc->input_ptrs[6]);
  __nv_bfloat16 *kv_cache =
      static_cast<__nv_bfloat16 *>(task_desc->input_ptrs[7]);
  __nv_fp8_e4m3 const *kvbv_w =
      static_cast<__nv_fp8_e4m3 const *>(task_desc->input_ptrs[8]);
  float const *kvbv_s = static_cast<float const *>(task_desc->input_ptrs[9]);
  __nv_fp8_e4m3 const *oproj_w =
      static_cast<__nv_fp8_e4m3 const *>(task_desc->input_ptrs[10]);
  float const *oproj_s = static_cast<float const *>(task_desc->input_ptrs[11]);
  __nv_bfloat16 const *residual =
      static_cast<__nv_bfloat16 const *>(task_desc->input_ptrs[12]);
  __nv_bfloat16 *out = static_cast<__nv_bfloat16 *>(task_desc->output_ptrs[0]);
  uint8_t *scratch_base = static_cast<uint8_t *>(task_desc->input_ptrs[13]);

  int const step = runtime_config.step[0];

  // The scratch tensor top holds the 3-slot u64 GMEM grid barrier (24 bytes,
  // one monotonic counter per barrier-site), then the v1 AttnScratch
  // activation arrays. v1 put an 8-byte AttnGridBarrier(count,gen) at the top
  // (ATTN_BARRIER_BYTES) and attn_make_scratch lays the arrays out starting at
  // that offset (first array 16-aligned -> byte 16). The v2 barrier needs 24
  // bytes, so the v2 builder MUST allocate ATTN_SCRATCH_BYTES + V2_BAR_EXTRA
  // (=+16) bytes, ALL of bar[0..3) ZEROED at alloc. We pass a base advanced by
  // V2_BAR_EXTRA(=16) to attn_make_scratch so its internal 8-byte skip +
  // 16-align lands the first array at scratch_base + 16 + 16 = byte 32 (>= 24
  // => never aliases the 24-byte barrier region; 8 bytes of harmless slack).
  static constexpr int V2_BAR_BYTES = 3 * (int)sizeof(unsigned long long); // 24
  static constexpr int V2_BAR_EXTRA =
      V2_BAR_BYTES - v1a::ATTN_BARRIER_BYTES; // 24 - 8 = 16
  unsigned long long *bar =
      reinterpret_cast<unsigned long long *>(scratch_base);
  v1a::AttnScratch sc = v1a::attn_make_scratch(scratch_base + V2_BAR_EXTRA);
  // need = num_tasks * (iter_num + 1): the monotonic target for THIS decode
  // iteration. Every one of the num_tasks==num_workers workers bumps each
  // barrier slot exactly once per invocation.
  unsigned long long const need =
      (unsigned long long)num_tasks * (unsigned long long)(iter_num + 1);

  // --- dynamic SMEM region 0 (see attn_block_megakernel_v2_spec.h). ---
  extern __shared__ __align__(1024) uint8_t s_smem_raw[];
  uint8_t *s_smem = s_smem_raw + task_desc->smem_region_offset(SM_REGION_WORK);
  uint4 *s_wbuf = reinterpret_cast<uint4 *>(s_smem + SM_OFF_WBUF);
  float *red8 = reinterpret_cast<float *>(s_smem + SM_OFF_RED8);
  float *s_score = reinterpret_cast<float *>(s_smem + SM_OFF_SCORE);
  float *s_act = reinterpret_cast<float *>(s_smem + SM_OFF_ACT);
  float *s_qbdeq = s_act; // q_b phase reuses the front of s_act
  float *s_odeq = s_act;  // o_proj phase reuses the front of s_act
  __shared__ int s_mla_last;

  int tid = threadIdx.x, lane = tid & 31, warpl = tid >> 5;
  uint4 *my_wbuf = s_wbuf + (size_t)warpl * CPA_RING_U4;
  int worker_idx = task_offset; // logical CTA id (baked in runtime.cc)
  // PHYSICAL global thread/warp coverage (128 threads/worker, 4 warps/worker).
  int gtid = worker_idx * PHYS_NTHREAD + tid;
  int gthreads = num_tasks * PHYS_NTHREAD;
  int gwarp = gtid >> 5;
  int gwarps = gthreads >> 5;
  int KV = step + 1, pos = step;

  // ===================== S2: Phase-0 RMSNorm + quant + qkv_a GEMM ===========
  rmsnorm_quant_hidden_block_smem_ab(
      hidden /*=raw self.x*/, input_ln_w, s_act, red8, K_HIDDEN, warpl, lane);
  v1a::gemv_grid_cpa_t<2, 6>(s_act,
                             qkv_a_w,
                             qkv_a_s,
                             sc.g_qkva,
                             K_QKVAN,
                             K_HIDDEN,
                             gwarp,
                             gwarps,
                             lane,
                             my_wbuf);
  attn_v2_grid_barrier(bar, 0, need); // qkv_a -> layernorm

  // ============ S3 q_a_layernorm + S5 kv_a_layernorm + rope_k + append =====
  {
    float q_rcp = rms_rcp_block_ab(sc.g_qkva, K_QLORA, red8, warpl, lane);
    float kv_rcp =
        rms_rcp_block_ab(sc.g_qkva + K_QLORA, K_KVLORA, red8, warpl, lane);
    int ngq = K_QLORA / K_GRP; // 12
    // q_a_layernorm + UE8M0 requant into block-local s_qbdeq (v1 body; warp
    // stride 8 -> PHYS_NWARP; group math warp-local -> value-identical).
    for (int g = warpl; g < ngq; g += PHYS_NWARP) {
      float const *src = sc.g_qkva + g * K_GRP;
      __nv_bfloat16 const *w = q_a_ln_w + g * K_GRP;
      float nv[4];
      float mx = 1e-10f;
#pragma unroll
      for (int t = 0; t < 4; t++) {
        int j = lane + t * 32;
        float v = v1a::k_bf16(src[j] * q_rcp * __bfloat162float(w[j]));
        nv[t] = v;
        mx = fmaxf(mx, fabsf(v));
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        float ot = __shfl_xor_sync(0xffffffffu, mx, o);
        mx = fmaxf(mx, ot);
      }
      float ys = fmaxf(mx / K_FP8MAX, 1e-10f);
      float yq = v1a::k_dec_ue8m0(v1a::k_enc_ue8m0(ys));
      float *d = s_qbdeq + g * K_GRP;
#pragma unroll
      for (int t = 0; t < 4; t++) {
        int j = lane + t * 32;
        float qv = fminf(fmaxf(nv[t] / yq, -K_FP8MAX), K_FP8MAX);
        d[j] = (float)__nv_fp8_e4m3(qv) * yq;
      }
    }
    // kv_a_layernorm -> kv_cache row [0:512). PHYSICAL grid-stride.
    for (int i = gtid; i < K_KVLORA; i += gthreads) {
      float v = v1a::k_bf16(sc.g_qkva[K_QLORA + i] * kv_rcp *
                            __bfloat162float(kv_a_ln_w[i]));
      kv_cache[(size_t)step * K_QKHEAD + i] = __float2bfloat16(v);
    }
    // rope(k_pe) on g_qkva[2048:2112) + append to kv_cache[step][512:576).
    // W0 tail lighten: worker-0 threads 0..31 each do one independent rope
    // pair (bit-identical to the serial path; parallelized across pairs).
    if (worker_idx == 0 && tid < K_QKROPE / 2) {
      int pr = tid;
      int d0 = pr * 2, d1 = d0 + 1;
      float c = __bfloat162float(cos_sin[pos * K_COSSIN_STRIDE + d0]);
      float s = __bfloat162float(
          cos_sin[pos * K_COSSIN_STRIDE + K_COSSIN_SINOFF + d0]);
      float k0 = sc.g_qkva[2048 + d0], k1 = sc.g_qkva[2048 + d1];
      kv_cache[(size_t)step * K_QKHEAD + 512 + d0] =
          __float2bfloat16(v1a::k_bf16(k0 * c - k1 * s));
      kv_cache[(size_t)step * K_QKHEAD + 512 + d1] =
          __float2bfloat16(v1a::k_bf16(k1 * c + k0 * s));
    }
  }
  // v1's S3->S4 sync (q_b reads block-local s_qbdeq). The kv_a-ln/rope writes
  // to kv_cache are published cross-block by the q_b->MLA grid barrier below.
  attn_v2_sync();

  // ===================== S4+S6 FUSED: q_b GEMM + rope -> g_qpe ==============
  v1a::gemv_grid_cpa_qb_rope_smem_t<8, 4>(s_qbdeq,
                                          q_b_w,
                                          q_b_s,
                                          sc.g_qpe,
                                          K_HLOCAL * K_QKHEAD,
                                          K_QLORA,
                                          cos_sin,
                                          pos,
                                          gwarp,
                                          gwarps,
                                          lane,
                                          my_wbuf);
  // Zero the per-head completion counters + readiness flags BEFORE the
  // q_b->MLA barrier (its __threadfence publishes the zeros to every CTA).
  // gtid<16 (with PHYS_NTHREAD=128 -> worker 0 threads 0..15), same writer set
  // as v1.
  if (gtid < K_HLOCAL) {
    sc.g_head_done[gtid] = 0;
    sc.g_head_wuv_ready[gtid] = 0;
  }
  attn_v2_grid_barrier(bar, 1, need); // q_b -> MLA (publishes g_qpe, kv_cache,
                                      // zeroed flags via the all-thread fence)

  // ===================== S9/S10: FLASH MLA decode (KV-split) ================
  double mscale = 0.1 * log(40.0) + 1.0;
  float sm = (float)((1.0 / sqrt(192.0)) * mscale * mscale);
  int nsp = (KV + 63) / 64;
  if (nsp < 1) {
    nsp = 1;
  }
  if (nsp > MLA_SPLITS) {
    nsp = MLA_SPLITS;
  }
  int tile = (KV + nsp - 1) / nsp;
  int ntask = K_HLOCAL * nsp;
  // HAZARD #2: ntask = 16*nsp <= 128 < num_workers(136) — a WORKER/CTA
  // invariant, UNCHANGED by the 256->128 thread port. One partial per worker;
  // the per-head atomicAdd counts nsp arrivals; all partials run before any
  // W_UV spinner needs that head.
  {
    int idx = worker_idx;
    if (idx < ntask) {
      int h = idx / nsp, sp = idx % nsp;
      int r0 = sp * tile, r1 = r0 + tile;
      if (r1 > KV) {
        r1 = KV;
      }
      mla_partial_ab(kv_cache,
                     sc.g_qpe,
                     sc.g_mla_acc,
                     sc.g_mla_m,
                     sc.g_mla_l,
                     s_score,
                     red8,
                     h,
                     sp,
                     r0,
                     r1,
                     sm); // ends with attn_v2_sync (publishes acc into tid0)
      if (threadIdx.x == 0) {
        __threadfence(); // device release (publish g_mla_acc)
        int old = atomicAdd(&sc.g_head_done[h], 1);
        s_mla_last = (old == nsp - 1) ? 1 : 0;
        if (s_mla_last) {
          __threadfence(); // device acquire (this block sees all splits' acc)
        }
      }
      attn_v2_sync(); // broadcast s_mla_last + hand the acquire to all threads
      if (s_mla_last) {
        mla_merge_quant_ab(sc.g_attn,
                           sc.g_attn_deq,
                           sc.g_mla_acc,
                           sc.g_mla_m,
                           sc.g_mla_l,
                           s_score,
                           h,
                           nsp,
                           sc.g_head_wuv_ready);
      }
    }
  }

  // ===================== S12 W_UV BMM -> g_red ==============================
  // Lever 5: wuv_bmm_grid spin-waits per head on g_head_wuv_ready[h] (no grid
  // barrier here). Called VERBATIM (only __syncwarp inside).
  v1a::wuv_bmm_grid(sc.g_attn_deq,
                    kvbv_w,
                    kvbv_s,
                    sc.g_red,
                    gwarp,
                    gwarps,
                    lane,
                    sc.g_head_wuv_ready);
  attn_v2_grid_barrier(bar, 2, need); // W_UV -> * (publishes g_red)

  // ===================== S13: quantize g_red + o_proj GEMM + residual =======
  // Lever 3: quant g_red[2048] block-locally into s_odeq (A/B not needed — the
  // quant is per-128-group warp-local; warp stride 8 -> PHYS_NWARP is
  // value-identical), then o_proj reads s_odeq. Body is v1's
  // quant_ue8m0_block_smem inlined with PHYS_NWARP stride + attn_v2_sync.
  {
    int ng = K_OIN / K_GRP;
    for (int gx = warpl; gx < ng; gx += PHYS_NWARP) {
      float const *s = sc.g_red + gx * K_GRP + lane * 4;
      float4 a = *reinterpret_cast<float4 const *>(s);
      float v[4] = {a.x, a.y, a.z, a.w};
      float mx = 1e-10f;
#pragma unroll
      for (int t = 0; t < 4; t++) {
        mx = fmaxf(mx, fabsf(v[t]));
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
      }
      float ys = fmaxf(mx / K_FP8MAX, 1e-10f);
      float yq = v1a::k_dec_ue8m0(v1a::k_enc_ue8m0(ys));
      float *d = s_odeq + gx * K_GRP + lane * 4;
#pragma unroll
      for (int t = 0; t < 4; t++) {
        float q = fminf(fmaxf(v[t] / yq, -K_FP8MAX), K_FP8MAX);
        d[t] = (float)__nv_fp8_e4m3(q) * yq;
      }
    }
    attn_v2_sync(); // publish s_odeq before o_proj reads it
  }
  v1a::gemv_grid_cpa_oproj_smem_t<8, 4>(s_odeq,
                                        oproj_w,
                                        oproj_s,
                                        residual, // ZERO at TP8 (AR combines)
                                        out,
                                        K_HIDDEN,
                                        K_OIN,
                                        gwarp,
                                        gwarps,
                                        lane,
                                        my_wbuf);
  // Publish the output stores globally before MPK signals task completion.
  __threadfence();
  attn_v2_sync();
}

} // namespace attn_block_megakernel_v2
} // namespace kernel
