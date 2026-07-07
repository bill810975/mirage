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
// FUSED DENSE-MLP decode megakernel in Runtime-V2 "megakernel-shape" (Form-2)
// format (M5 of the DSv3-decode-on-v2 effort). ONE v2 task runs the WHOLE dense
// MLP for one token (bs=1, TP8 EP2 per-rank, M=1), exactly like the v1
// tasks/blackwell/dsv3_dense_mlp_fused_sm100.cuh, but hosted on the v2
// role-split runtime: 136 co-resident tasks == 136 workers (hard host-assert),
// self-syncing around the SINGLE in-op grid barrier (W13 -> W2) via a MONOTONIC
// GMEM count-barrier (v1's self-reset sense/gen barrier rebuilt across
// co-resident v2 tasks), CONSUMER-ONLY (128 physical threads / 4 warps).
//
// ALL GEMV / quant / silu math is the v1 code itself: the device helpers
// (dgemv_cpa16, dgemv_cpa, quant_group_warp, quant_scale, to_f8, silu, f8x4)
// are called from ::kernel::dsv3_dense_mlp (the v1 kernel, already #included in
// every generated test.cu), so per-row / per-group values are bit-identical to
// v1 GIVEN identical input bytes — task/warp scheduling cannot change the
// values because every helper's accumulation is warp-local with a fixed lane
// order.
//
// THE ONE REAL CHANGE vs the v1 body: the self-reset sense/gen grid barrier
// (v1's DenseMlpGridBarrier / dense_mlp_grid_barrier, last-arriver resets
// count=0) is replaced by the MONOTONIC idiom (same as attn_block_megakernel_v2
// / dsv3_ffn_v2): a single u64 slot, NEVER self-reset, target
// need = num_tasks*(iter_num+1). The bar tensor MUST be zeroed ONLY at step 0
// (its tensor_init is registered with skip_after_step0=True by the builder);
// re-zeroing on step>=1 resets the counter while the target keeps growing ->
// iter-1 hang (the exact bug fixed for attn/ffn).
//
// WHY consumer-only 128 threads, and how it stays high-cosine (>=0.999) vs v1's
// 256 (approach (b), correctness-first — NOT A/B emulation):
//   The v1 body is uniform-TPB-thread (TPB = blockDim.x) with block-wide
//   __syncthreads() and NO tcgen05 / CuTe / bar.sync — pure scalar-ILP FP8
//   GEMV + a block-local RMSNorm reduction + a block-local silu. A v2 consumer
//   role runs only W0-3 (128 threads); a __syncthreads() there would wait the
//   helper/controller warps (W4-7, in different loops) -> deadlock. This port
//   therefore runs 128 PHYSICAL threads and:
//     * replaces every block-wide __syncthreads() with dense_v2_sync() =
//       `bar.sync 7, 128` (ids 1-6 taken: linear/rmsnorm/ffn/attn-v2-chain/
//       AR/attn-mega). __syncwarp() unchanged.
//     * runs the GEMV phases GRID-STRIDED over the PHYSICAL grid warps: gwarp =
//       (worker_idx*128 + tid) >> 5, gwarps = (num_tasks*128) >> 5 (== 136*4 =
//       544). This is v1's EXACT grid-stride math with nwl/gwarps 8->4 warps
//       per worker; each output ROW is still computed by exactly ONE warp with
//       the SAME K-reduction lane order regardless of gwarps (the re-stride
//       only changes WHICH warp owns a row + how many row-blocks per warp ->
//       the per-row value is bit-identical to v1).
//     * the block-local RMSNorm sum-of-squares reduces over PHYS_NWARP(4) warp
//       partials instead of nwl(8). The result differs from v1-256 by <1
//       ULP-scale (different tree order) but is IDENTICAL across all 136
//       co-resident blocks (every block runs the same 128-thread reduction), so
//       the redundant per-block rmsnorm+quant stays CONSISTENT across output
//       rows. The fused kernel is high-cosine (>=0.999) NOT bit-identical to
//       the PyTorch ref anyway (it UE8M0-rounds activations), so the 128-vs-256
//       reduction-order delta is within tolerance.
//
// input_ptrs ABI — 7 slots (IDENTICAL shape to the v1 dense mega, except slot
// [6] now holds the MONOTONIC u64[2] barrier at the top instead of v1's 8-byte
// self-reset (count,gen)):
//   [0] hidden(bf16 raw pre-rmsnorm self.x) [1] w13(fp8) [2] w13_scale(RAW f32)
//   [3] w2(fp8) [4] w2_scale(RAW f32) [5] rmsnorm_weight(bf16)
//   [6] scratch(u8: [bar u64[2] @0 | y13 f32[W13_N] @64], bar[0] persists via
//       skip_after_step0, y13 fully written-before-read every step)
//   + out bound as output_ptrs[0] (bf16[1,HIDDEN], PRE-AllReduce/residual —
//     the RowParallel down_proj AllReduce + residual stay OUTSIDE this task).
// ============================================================================

#include "mirage/persistent_kernel/mpk_atoms.cuh"
#include "mirage/persistent_kernel/runtime_header.h"
#include "mirage/persistent_kernel/tasks/blackwell/dsv3_dense_mlp_fused_sm100.cuh"
#include "mirage/persistent_kernel/tasks/blackwell_v2/dsv3_dense_mlp_fused_v2_spec.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>
#include <stdint.h>

namespace kernel {
namespace dsv3_dense_mlp_v2 {

namespace v1d = ::kernel::dsv3_dense_mlp;

// Pin the spec-header shapes to the v1 kernel's constants so any drift is a
// compile error.
static_assert(
    HIDDEN == v1d::HIDDEN && W13_N == v1d::W13_N && W2_K == v1d::W2_K &&
        SILU_OUT == v1d::SILU_OUT && GRP == v1d::GRP && KG1 == v1d::KG1 &&
        KG2 == v1d::KG2,
    "dsv3_dense_mlp_fused_v2_spec.h shapes drifted from the v1 kernel");
static_assert(RBX_W13 == 8 && RBX_W2 == 16 && ST_W13 == 4 && ST_W2 == 3,
              "dense-MLP v2 GEMV row-block/stage constants drifted from v1");

// Consumer-only intra-CTA barrier: replaces every block-wide __syncthreads()
// over the 128 consumer threads. Named barrier id 7 (1-6 taken by
// linear/rmsnorm/ffn/attn-v2-chain/AR/attn-mega). "memory" clobber orders the
// SMEM/GMEM accesses the barrier separates.
__device__ __forceinline__ void dense_v2_sync() {
  asm volatile("bar.sync 7, 128;" ::: "memory");
}

// V2 megakernel-shape GMEM grid barrier: the MONOTONIC replacement for v1's
// self-reset dense_mlp_grid_barrier(136). ONE u64 slot; never reset;
// need = num_tasks*(iter_num+1). Preserves v1's ALL-thread __threadfence: all
// 128 consumer threads fence their prior global stores (the W13 y13 writes)
// BEFORE thread 0 bumps the count, so a peer that later passes the barrier sees
// EVERY thread's writes (a thread0-only fence would publish only thread 0's).
// Called by ALL 128 consumer threads.
__device__ __forceinline__ void dense_v2_grid_barrier(unsigned long long *bar,
                                                      unsigned long long need) {
  dense_v2_sync(); // all 128 threads' prior work retired
  __threadfence(); // EVERY consumer thread publishes its global stores
  dense_v2_sync(); // ensure all fences retired before thread 0 bumps
  if (threadIdx.x == 0) {
    atom_add_release_gpu_u64(&bar[0], 1ull);
    while (ld_acquire_sys_u64(&bar[0]) < need) {
      __nanosleep(64);
    }
  }
  dense_v2_sync(); // broadcast: everyone leaves together, acquire peers' data
}

// ===========================================================================
//  MPK v2 task entry. Consumer-only (128 threads). Mirrors the v1 dense mega's
//  phase structure but reads task_offset (NOT merge_task_offset) as the logical
//  CTA id and self-syncs via the monotonic GMEM barrier. `num_tasks` ==
//  num_workers. sync_tag / nwarps are consumer-only no-ops (there is no
//  cross-role flag protocol — the redundant-per-block rmsnorm keeps s_norm
//  block-local; the only shared state is bar[0] + the global y13 which is
//  fully-written-before-read).
// ===========================================================================
__device__ __noinline__ void
    dense_mlp_v2_task_impl(mirage::runtime::TaskDesc const *task_desc,
                           int task_offset,
                           int num_tasks,
                           int nwarps,
                           unsigned long long sync_tag,
                           int iter_num) {
  (void)nwarps;
  (void)sync_tag;

  __nv_bfloat16 const *hidden =
      static_cast<__nv_bfloat16 const *>(task_desc->input_ptrs[0]);
  uint8_t const *w13 = static_cast<uint8_t const *>(task_desc->input_ptrs[1]);
  float const *w13_scale = static_cast<float const *>(task_desc->input_ptrs[2]);
  uint8_t const *w2 = static_cast<uint8_t const *>(task_desc->input_ptrs[3]);
  float const *w2_scale = static_cast<float const *>(task_desc->input_ptrs[4]);
  __nv_bfloat16 const *rmsnorm_weight =
      static_cast<__nv_bfloat16 const *>(task_desc->input_ptrs[5]);
  uint8_t *scratch_base = static_cast<uint8_t *>(task_desc->input_ptrs[6]);

  __nv_bfloat16 *out = static_cast<__nv_bfloat16 *>(task_desc->output_ptrs[0]);

  // The scratch tensor top holds the MONOTONIC u64[2] grid barrier (16 bytes,
  // one live counter at bar[0]); the global y13 lives at offset 64 (v1's
  // make_scratch layout — first array 16-aligned past the barrier region). The
  // builder zeroes ALL of scratch at step 0 only (skip_after_step0=True) so
  // bar[0] persists monotonically; y13 is fully written by Phase 1 before any
  // Phase 2 read every step (its zero-init is not required).
  unsigned long long *bar =
      reinterpret_cast<unsigned long long *>(scratch_base);
  v1d::Scratch sc = v1d::make_scratch(scratch_base);
  unsigned long long const need =
      (unsigned long long)num_tasks * (unsigned long long)(iter_num + 1);

  int const worker_idx = task_offset; // logical CTA id (baked in runtime.cc)
  int const tid = threadIdx.x;
  int const lane = tid & 31;
  int const wlocal = tid >> 5; // within-worker warp id (0..PHYS_NWARP-1)
  // PHYSICAL global thread/warp coverage (128 threads/worker, 4 warps/worker).
  int const gtid = worker_idx * PHYS_NTHREAD + tid;
  int const gwarp = gtid >> 5;
  int const gwarps = (num_tasks * PHYS_NTHREAD) >> 5;

  // --- dynamic SMEM region 0 (see dsv3_dense_mlp_fused_v2_spec.h). Same
  // 16-byte-aligned carve as v1's s_smem, but sized for PHYS_NWARP warps. ---
  extern __shared__ __align__(1024) uint8_t s_smem_raw[];
  uint8_t *s_smem = s_smem_raw + task_desc->smem_region_offset(SM_REGION_WORK);
  uint4 *s_wbuf = reinterpret_cast<uint4 *>(s_smem + SM_OFF_WBUF);
  uint4 *my_wbuf = s_wbuf + (size_t)wlocal * WBUF_U4;
  uint32_t *my_wbuf4 = reinterpret_cast<uint32_t *>(my_wbuf);
  __nv_bfloat16 *s_norm =
      reinterpret_cast<__nv_bfloat16 *>(s_smem + SM_OFF_NORM);
  uint8_t *s_a = s_smem + SM_OFF_A;
  float *s_as = reinterpret_cast<float *>(s_smem + SM_OFF_AS);
  float *s_silu = reinterpret_cast<float *>(s_smem + SM_OFF_SILU);
  uint8_t *s_ifp8 = s_smem + SM_OFF_IFP8;
  float *s_iscale = reinterpret_cast<float *>(s_smem + SM_OFF_ISCALE);
  float *s_red = reinterpret_cast<float *>(s_smem + SM_OFF_RED);

  // ====================================================================
  //  Phase A — RMSNorm. Every block computes the FULL sum(x^2) over HIDDEN
  //  redundantly (block-local, PHYS_NWARP=4 warp partials), then writes the
  //  bf16 normed into block-local s_norm. normed[i] = bf16(x[i]*rms_rcp*w[i]).
  // ====================================================================
  {
    float ss = 0.f;
    for (int i = tid; i < HIDDEN; i += PHYS_NTHREAD) {
      float v = __bfloat162float(hidden[i]);
      ss += v * v;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      ss += __shfl_xor_sync(0xffffffffu, ss, o);
    }
    if (lane == 0) {
      s_red[wlocal] = ss;
    }
    dense_v2_sync();
    float tot = (tid < PHYS_NWARP) ? s_red[tid] : 0.f;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      tot += __shfl_xor_sync(0xffffffffu, tot, o);
    }
    if (tid == 0) {
      s_red[0] = tot;
    }
    dense_v2_sync();
    float rms_rcp = rsqrtf(s_red[0] / float(HIDDEN) + v1d::RMS_EPS);
    for (int i = tid; i < HIDDEN; i += PHYS_NTHREAD) {
      float v = __bfloat162float(hidden[i]);
      float wt = __bfloat162float(rmsnorm_weight[i]);
      s_norm[i] = __float2bfloat16(v * rms_rcp * wt);
    }
    dense_v2_sync();
  }

  // ====================================================================
  //  Phase 0 — FP8-quantize the BF16 normed (UE8M0 scale) into block-local
  //  s_a / s_as (v1 quant_group_warp verbatim; warp stride PHYS_NWARP).
  // ====================================================================
  for (int g = wlocal; g < KG1; g += PHYS_NWARP) {
    v1d::quant_group_warp<__nv_bfloat16>(s_norm, s_a, s_as, g, lane);
  }
  dense_v2_sync();

  // ====================================================================
  //  Phase 1 — W13 GEMV over ALL 4608 rows -> global sc.y13[n].
  //  W13_N/RBX_W13 = 576 warp-jobs distributed over gwarps grid warps.
  // ====================================================================
  int const n13 = W13_N / RBX_W13; // 576
  for (int idx = gwarp; idx < n13; idx += gwarps) {
    int n0 = idx * RBX_W13;
    float const *ws = w13_scale + (size_t)(n0 / GRP) * KG1;
    float yb[RBX_W13];
    v1d::dgemv_cpa16<RBX_W13, ST_W13>(
        s_a, s_as, w13, ws, HIDDEN, n0, lane, my_wbuf, yb);
    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < RBX_W13; r++) {
        sc.y13[n0 + r] = yb[r];
      }
    }
  }

  // ---- GMEM BARRIER (W13 -> W2): the y13 all-to-all. -----------------------
  dense_v2_grid_barrier(bar, need);

  // ====================================================================
  //  Phase 2 — silu_mul (384-chunk interleave) + UE8M0 requant.
  //  Every block recomputes the full silu_out from the global y13 into
  //  block-local s_silu, then requants -> block-local s_ifp8 / s_iscale, so
  //  Phase 3 reads the W2 input from SMEM (no cold y13 re-read in the GEMV) and
  //  no Phase2->3 grid barrier is needed.
  //
  //  384-interleave: out[c] = silu(y13[cp*768 + wc]) * y13[cp*768 + 384 + wc]
  //  where cp = c/384, wc = c%384.
  // ====================================================================
  for (int c = tid; c < SILU_OUT; c += PHYS_NTHREAD) {
    int cp = c / v1d::CHUNK;
    int wc = c % v1d::CHUNK;
    float gate = sc.y13[cp * 768 + wc];
    float up = sc.y13[cp * 768 + 384 + wc];
    s_silu[c] = v1d::silu(gate) * up;
  }
  dense_v2_sync();
  // requant silu_out (per-128-group UE8M0) into block-local s_ifp8 / s_iscale.
  for (int g = wlocal; g < KG2; g += PHYS_NWARP) {
    v1d::quant_group_warp<float>(s_silu, s_ifp8, s_iscale, g, lane);
  }
  dense_v2_sync();

  // ====================================================================
  //  Phase 3 — W2 GEMV over ALL 7168 rows -> out (bf16). 4B path (K=2304 not
  //  %512). HIDDEN/RBX_W2 = 448 warp-jobs over gwarps grid warps.
  // ====================================================================
  int const n2 = HIDDEN / RBX_W2; // 448
  for (int idx = gwarp; idx < n2; idx += gwarps) {
    int n0 = idx * RBX_W2;
    float const *ws = w2_scale + (size_t)(n0 / GRP) * KG2;
    float yb[RBX_W2];
    v1d::dgemv_cpa<RBX_W2, ST_W2>(
        s_ifp8, s_iscale, w2, ws, W2_K, KG2, n0, lane, my_wbuf4, yb);
    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < RBX_W2; r++) {
        out[n0 + r] = __float2bfloat16(yb[r]);
      }
    }
  }

  // Drain the cp.async ring so the codegen consumer page-release suffix cannot
  // reclaim the SMEM ring while a store is still in flight. (Consumer-only: no
  // helper handshake needed; the redundant-per-block rmsnorm/silu keep all
  // other shared state block-local.)
  v1d::cpasync_wait<0>();
  __syncwarp();
  dense_v2_sync();
}

} // namespace dsv3_dense_mlp_v2
} // namespace kernel
