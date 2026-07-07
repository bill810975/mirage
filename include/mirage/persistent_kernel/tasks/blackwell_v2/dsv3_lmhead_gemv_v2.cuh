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
// DSv3 tail lm_head GEMV in Runtime-V2 format (M3 decode-blocker fix).
//
// Replaces the fragile TMA+tcgen05 `linear_sm100_v3` lm_head — which throws a
// deep, sanitizer-unattributable async cudaErrorIllegalAddress in the M3
// decode path (mbar / A-TMA / tcgen05-MMA-descriptor / swizzle internal fault;
// the W-TMA OPERANDS were proven clean) — with a plain scalar/cp.async bf16
// GEMV, the SAME non-TMA pattern the attn+FFN v2 megakernels use successfully.
//
// At bs=1 the lm_head is a memory-bound read-once GEMV
//   logits[v] = sum_k rmsnorm_out[k] * w_lm_head[v][k],  v in [0, N)
// (N = padded vocab, K = hidden = 7168). Streaming the weight (N*K*2 bytes,
// ~1.85 GB for the full DSv3 vocab) dominates; scalar-MAC compute is free, so
// dropping TMA/tcgen05 costs no meaningful perf and removes the fault surface.
//
// SHAPE / DTYPE contract (matches the current builder lm_head + the downstream
// argmax_partial_sm100_v2 consumer, which reads its input as bf16 row-major):
//   input_ptrs[0]  rmsnorm_out  bf16  [1, K]         (final-norm output)
//   input_ptrs[1]  w_lm_head    bf16  [N, K] row-major
//   output_ptrs[0] logits       bf16  [1, N] row-major
//
// TILING (a NORMAL v2 task — NOT a mega co-residency task; no grid barrier):
//   grid = (N / BLOCK_N, 1, 1); task_offset (= blockIdx.x, baked in runtime.cc)
//   selects the tile. Each task owns BLOCK_N *contiguous* output rows and is
//   fully independent; the v2 scheduler round-robins the tiles onto the 136
//   workers. The 4 consumer warps split the tile's BLOCK_N rows; each warp
//   walks its rows in RBX-row chunks, streaming the weight via a cp.async ring
//   and reducing the K-dot across its 32 lanes. All BLOCK_N rows are covered
//   (BLOCK_N must be a multiple of nwarps*RBX; the builder picks a BLOCK_N that
//   divides N so no tail rows are dropped).
//
// Reuses the VERBATIM cp.async + bf16 unpack helpers from the v1 FFN kernel
// (cpasync16 / cpasync_commit / cpasync_wait / bf16x2), already included in
// every generated test.cu.
// ============================================================================

#include "mirage/persistent_kernel/runtime_header.h"
#include "mirage/persistent_kernel/tasks/blackwell/ffn_full_megakernel_sm100.cuh"
#include "mirage/persistent_kernel/tasks/blackwell_v2/dsv3_lmhead_gemv_v2_spec.h"

#include <cuda_bf16.h>
#include <stdint.h>

namespace kernel {
namespace dsv3_lmhead_gemv_v2 {

namespace v1k = ::kernel::ffn_full_megakernel_sm100;

// Consumer-warp named barrier (threads 0-127 only). The v2 worker block is 256
// threads (8 warps): warps 0-3 are the consumer role, warps 4-7 are the
// single-warp loader/launcher/storer/controller roles. A plain __syncthreads()
// (bar 0, all 256) would DEADLOCK — the helper warps never reach this body — so
// consumer-only tasks MUST use a 128-thread NAMED barrier. Barrier 0 is the
// implicit block-wide sync, 1 is used by linear_v2, 2 by rmsnorm_v2, 3 by the
// FFN v2 consumer bodies; reuse 3 here (this task never co-runs with an FFN v2
// task in the SAME worker instruction, so there is no barrier-id collision).
__device__ __forceinline__ void lmh_consumer_sync() {
  asm volatile("bar.sync 3, 128;");
}

// ---------------------------------------------------------------------------
// bf16 uint4 (16B = 8 bf16) cp.async-pipelined GEMV. ONE warp computes RBX
// consecutive output rows; the 32 lanes split the K dimension (16 bf16 per
// lane per K-step) and a warp-shuffle reduces the per-lane partials. Mirrors
// v1's dgemv_cpa16 pipeline structure but plain bf16 (no FP8 unpack, no group
// scales): a straight bf16-dot fp32-accumulate.
//
//   s_act    : shared, the K-length bf16 activation (uint4 view, K/8 elems)
//   w        : global, w_lm_head base (bf16), rows start at n0
//   K        : reduction dim (multiple of 8)
//   n0       : first output row this warp computes
//   lane     : threadIdx.x & 31
//   wbuf_base: this warp's cp.async ring (uint4*, RBX*32*STAGES*16 bytes)
//   y_out    : RBX-length fp32 output (valid on lane 0)
// ---------------------------------------------------------------------------
template <int RBX>
__device__ __forceinline__ void
    lmh_dgemv_bf16(uint4 const *__restrict__ s_act,
                   __nv_bfloat16 const *__restrict__ w,
                   int K,
                   int n0,
                   int lane,
                   uint4 *wbuf_base,
                   float *y_out) {
  constexpr int STAGES = ::kernel::dsv3_lmhead_gemv_v2::STAGES;
  uint4 const *w16 =
      reinterpret_cast<uint4 const *>(w + static_cast<size_t>(n0) * K);
  // uint4 = 16 bytes = 8 bf16, so each row is K/8 uint4 chunks; the K-loop
  // strides 32 lanes at a time => SS = (K/8)/32 K-steps (lane u handles the
  // (step*32 + lane)-th uint4 of the row).
  int const KU8 = K >> 3;  // uint4 (8 bf16) chunks per row = K/8
  int const SS = KU8 >> 5; // K-steps (each lane handles one uint4 per step)

  float y[RBX];
#pragma unroll
  for (int r = 0; r < RBX; r++) {
    y[r] = 0.f;
  }

  uint32_t const sbase = __cvta_generic_to_shared(wbuf_base);
  uint32_t const STRIDE = (uint32_t)(RBX * 32 * 16);

  // Prologue: kick off STAGES-1 stages.
  int const pf = (STAGES - 1 < SS) ? (STAGES - 1) : SS;
#pragma unroll
  for (int s = 0; s < STAGES - 1; s++) {
    if (s < pf) {
      uint32_t b = sbase + (uint32_t)s * STRIDE;
#pragma unroll
      for (int r = 0; r < RBX; r++) {
        v1k::cpasync16(b + (uint32_t)((r * 32 + lane) * 16),
                       &w16[static_cast<size_t>(r) * KU8 +
                            static_cast<size_t>(s) * 32 + lane]);
      }
    }
    v1k::cpasync_commit();
  }

  for (int ss = 0; ss < SS; ss++) {
    int const sp = ss + (STAGES - 1);
    if (sp < SS) {
      uint32_t b = sbase + (uint32_t)(sp % STAGES) * STRIDE;
#pragma unroll
      for (int r = 0; r < RBX; r++) {
        v1k::cpasync16(b + (uint32_t)((r * 32 + lane) * 16),
                       &w16[static_cast<size_t>(r) * KU8 +
                            static_cast<size_t>(sp) * 32 + lane]);
      }
    }
    v1k::cpasync_commit();
    v1k::cpasync_wait<STAGES - 1>();
    __syncwarp();

    // Activation uint4 for this K-step (8 bf16 from shared).
    uint4 const av = s_act[ss * 32 + lane];
    float a0, a1, a2, a3, a4, a5, a6, a7;
    v1k::bf16x2(av.x, a0, a1);
    v1k::bf16x2(av.y, a2, a3);
    v1k::bf16x2(av.z, a4, a5);
    v1k::bf16x2(av.w, a6, a7);

    uint32_t const cur = sbase + (uint32_t)(ss % STAGES) * STRIDE;
#pragma unroll
    for (int r = 0; r < RBX; r++) {
      uint4 wv;
      uint32_t saddr = cur + (uint32_t)((r * 32 + lane) * 16);
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                   : "=r"(wv.x), "=r"(wv.y), "=r"(wv.z), "=r"(wv.w)
                   : "r"(saddr));
      float w0, w1, w2, w3, w4_, w5, w6, w7;
      v1k::bf16x2(wv.x, w0, w1);
      v1k::bf16x2(wv.y, w2, w3);
      v1k::bf16x2(wv.z, w4_, w5);
      v1k::bf16x2(wv.w, w6, w7);
      float acc = a0 * w0;
      acc += a1 * w1;
      acc += a2 * w2;
      acc += a3 * w3;
      acc += a4 * w4_;
      acc += a5 * w5;
      acc += a6 * w6;
      acc += a7 * w7;
      y[r] += acc;
    }
  }

  // Warp-shuffle reduce each row's partial across the 32 lanes.
#pragma unroll
  for (int r = 0; r < RBX; r++) {
    float v = y[r];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      v += __shfl_down_sync(0xffffffffu, v, o);
    }
    if (lane == 0) {
      y_out[r] = v;
    }
  }
}

// ---------------------------------------------------------------------------
// Task body. K, N, BLOCK_N, NWARPS, RBX are compile-time (baked by
// register_dsv3_lmhead_gemv_v2_task). task_offset selects the N-tile.
//   BLOCK_N == NWARPS * RBX * ROWS_PER_WARP_CHUNKS
// Each of the NWARPS warps computes (BLOCK_N / NWARPS) rows in RBX-row chunks.
// ---------------------------------------------------------------------------
template <int K, int N, int BLOCK_N, int NWARPS, int RBX>
__device__ __forceinline__ void
    lmhead_gemv_task_impl(mirage::runtime::TaskDesc const *task_desc,
                          int task_offset) {
  // K%256==0 => (K/8)%32==0, so the lane-strided K-loop has no ragged tail
  // (SS*32 == K/8 exactly) and the 128-thread activation stage (K/8 uint4)
  // divides evenly. DSv3 hidden K=7168 = 256*28. ✓
  static_assert(K % 256 == 0,
                "K must be a multiple of 256 (uint4=8 bf16, 32-lane K-stride)");
  static_assert(BLOCK_N % (NWARPS * RBX) == 0,
                "BLOCK_N must be a multiple of NWARPS*RBX (no dropped rows)");
  constexpr int ROWS_PER_WARP = BLOCK_N / NWARPS;      // rows each warp does
  constexpr int CHUNKS_PER_WARP = ROWS_PER_WARP / RBX; // RBX-row chunks
  static_assert(ROWS_PER_WARP % RBX == 0, "ROWS_PER_WARP must divide RBX");

  __nv_bfloat16 const *act_g =
      static_cast<__nv_bfloat16 const *>(task_desc->input_ptrs[0]);
  __nv_bfloat16 const *w =
      static_cast<__nv_bfloat16 const *>(task_desc->input_ptrs[1]);
  __nv_bfloat16 *out = static_cast<__nv_bfloat16 *>(task_desc->output_ptrs[0]);

  extern __shared__ __align__(1024) char smem[];
  __nv_bfloat16 *s_act = reinterpret_cast<__nv_bfloat16 *>(
      smem + task_desc->smem_region_offset(LMH_REGION_ACT));
  uint4 *s_ring = reinterpret_cast<uint4 *>(
      smem + task_desc->smem_region_offset(LMH_REGION_RING));

  int const lane = threadIdx.x & 31;
  int const ws = threadIdx.x >> 5;

  // Stage the K-length bf16 activation into SMEM once (all 128 threads), reused
  // for every row this task computes. K/8 uint4 loads.
  {
    uint32_t const sb = static_cast<uint32_t>(__cvta_generic_to_shared(s_act));
    uint4 const *g4 = reinterpret_cast<uint4 const *>(act_g);
    constexpr int NU4 = K / 8;
#pragma unroll
    for (int u = threadIdx.x; u < NU4; u += 128) {
      v1k::cpasync16(sb + (uint32_t)u * 16, &g4[u]);
    }
    v1k::cpasync_commit();
    v1k::cpasync_wait<0>();
  }
  lmh_consumer_sync(); // bar.sync 3, 128 — activation visible to all 4 warps

  uint4 const *s_act4 = reinterpret_cast<uint4 const *>(s_act);
  uint4 *my_ring = s_ring + (size_t)ws * (ring_bytes_per_warp(RBX) / 16);

  // This warp's first output row within the whole vocab.
  int const tile_base = task_offset * BLOCK_N;
  int const warp_base = tile_base + ws * ROWS_PER_WARP;

#pragma unroll
  for (int c = 0; c < CHUNKS_PER_WARP; c++) {
    int const n0 = warp_base + c * RBX;
    float yb[RBX];
    lmh_dgemv_bf16<RBX>(s_act4, w, K, n0, lane, my_ring, yb);
    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < RBX; r++) {
        out[n0 + r] = __float2bfloat16_rn(yb[r]);
      }
    }
  }

  // Drain any in-flight cp.async before the framework reuses the ring slot.
  v1k::cpasync_wait<0>();
  __syncwarp();
}

} // namespace dsv3_lmhead_gemv_v2
} // namespace kernel
