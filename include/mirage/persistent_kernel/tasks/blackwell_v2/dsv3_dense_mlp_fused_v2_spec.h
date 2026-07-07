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
// SMEM region layout for the FUSED DENSE-MLP decode megakernel in Runtime-V2
// (M5 of the DSv3-decode-on-v2 effort). This is the v2 port of
// tasks/blackwell/dsv3_dense_mlp_fused_sm100.cuh — dense layers 0-2:
//   post-attn RMSNorm + W13(gate_up) GEMV + silu(gate)*up (384-chunk
//   interleave) + W2(down) GEMV -> bf16. NO router / topk / EP filter /
//   shared-expert (that is the FFN mega; the dense MLP is strictly simpler),
//   so the ONLY cross-worker state is the y13 global + a SINGLE grid barrier
//   between the W13 and W2 phases.
//
// The v1 kernel uses ONE dynamic `extern __shared__ __align__(1024) uint8_t
// s_smem[]` pool laid out as (v1, 256 threads / 8 warps):
//   s_wbuf  = nwl * WBUF_U4 uint4   (per-warp cp.async weight ring; nwl=TPB/32)
//   s_norm  = HIDDEN bf16           (14336 B; block-local rmsnorm output)
//   s_a     = HIDDEN u8             (7168 B;  block-local W13 fp8 activation)
//   s_as    = KG1 f32              (224 B;   block-local W13 activation scale)
//   s_silu  = SILU_OUT f32          (9216 B;  block-local silu intermediate)
//   s_ifp8  = W2_K u8               (2304 B;  block-local W2 fp8 activation)
//   s_iscale= KG2 f32               (72 B;    block-local W2 activation scale)
//   s_red   = nwl f32               (per-warp rmsnorm reduce partials)
//
// The v2 CONSUMER-ONLY port runs 128 PHYSICAL threads (4 warps), so the
// per-warp cp.async ring shrinks 8 -> 4 warps (PHYS_NWARP). Unlike the attn
// mega there is NO 256-thread block reduction to A/B-emulate: the ONLY block
// collective here is the RMSNorm sum-of-squares, which is a plain block
// reduction over PHYS_NWARP warp-partials (its tree order changes 8->4 warps,
// changing the fp result by <1 ULP-scale — the fused kernel is high-cosine
// (>=0.999) NOT bit-identical to the PyTorch ref anyway, since it UE8M0-rounds
// activations; and CRUCIALLY it is IDENTICAL across the 136 co-resident blocks
// because every block runs the same 128-thread reduction, so the redundant
// per-block rmsnorm+quant is CONSISTENT across all output rows). See approach
// (b) note in dsv3_dense_mlp_fused_v2.cuh.
//
// ONE region (ordinal 0). alignment=1024 (megakernel convention — a smaller
// align silently misaligns other tasks' 1024-aligned TMA/AR in the shared
// test.cu -> cudaErrorMisalignedAddress, caught only in-MPK). The device .cuh
// carves the sub-buffers out of region 0 with the SAME running 16-byte-aligned
// offsets v1 uses.
//
// Host-safe (no __device__ / no PTX asm) — included by task_register.cc for
// the planner AND by the device .cuh for the region ordinal + shape asserts.
// ============================================================================

#include "mirage/kernel/task_register.h"

namespace kernel {
namespace dsv3_dense_mlp_v2 {

// ---- problem shapes (mirror dsv3_dense_mlp_fused_sm100; the device side
// static_asserts equality against the v1 kernel's ::kernel::dsv3_dense_mlp
// constants) ----------------------------------------------------------------
inline constexpr int HIDDEN = 7168;   // W13 K, W2 N
inline constexpr int W13_N = 4608;    // gate+up output width
inline constexpr int W2_K = 2304;     // silu output width / W2 K
inline constexpr int SILU_OUT = 2304; // = W2_K
inline constexpr int GRP = 128;
inline constexpr int KG1 = HIDDEN / GRP; // 56 (W13 K-groups)
inline constexpr int KG2 = W2_K / GRP;   // 18 (W2 K-groups)

// ---- physical (v2) vs logical (v1) warp counts ----------------------------
// PHYS_NWARP = the number of consumer warps that actually run (4).
inline constexpr int PHYS_NTHREAD = 128;
inline constexpr int PHYS_NWARP = 4;

// ---- GEMV row-block / pipeline stages (mirror the v1 kernel's constexprs) --
// RBX_* must divide GRP=128 (the shared per-N-block weight-scale row is keyed
// by n0/GRP and shared by all RBX rows). The per-warp cp.async ring is sized
// as the worst case across both GEMV paths:
//   W13: dgemv_cpa16<8,4>  -> RBX*32*ST uint4 = 8*32*4 = 1024 uint4 = 16 KB
//   W2:  dgemv_cpa<16,3>   -> RBX*32*ST u32   = 16*32*3 = 1536 u32 = 384 uint4
// so WBUF_U4 = max(1024, 384) = 1024 uint4 = 16 KB/warp (== v1's WBUF_U4).
inline constexpr int RBX_W13 = 8;
inline constexpr int RBX_W2 = 16;
inline constexpr int ST_W13 = 4;
inline constexpr int ST_W2 = 3;
inline constexpr int WBUF_U4 = 1024; // max(8*32*4, ceil(16*32*3/4)) = 1024

inline constexpr int align_up_16(int n) {
  return (n + 15) & ~15;
}

// Running 16-byte-aligned offsets, mirroring the v1 s_smem carve. All in BYTES.
// s_wbuf: PHYS_NWARP per-warp cp.async rings (16-byte uint4 elements).
inline constexpr int SM_OFF_WBUF = 0;
inline constexpr int SM_WBUF_BYTES = PHYS_NWARP * WBUF_U4 * 16; // 65536

// s_norm: HIDDEN bf16 (block-local rmsnorm output).
inline constexpr int SM_OFF_NORM = SM_OFF_WBUF + SM_WBUF_BYTES;
inline constexpr int SM_NORM_BYTES = align_up_16(HIDDEN * 2); // 14336

// s_a: HIDDEN u8 (block-local W13 fp8 activation; 16-aligned for uint4 reads).
inline constexpr int SM_OFF_A = SM_OFF_NORM + SM_NORM_BYTES;
inline constexpr int SM_A_BYTES = align_up_16(HIDDEN); // 7168

// s_as: KG1 f32 (block-local W13 activation scale).
inline constexpr int SM_OFF_AS = SM_OFF_A + SM_A_BYTES;
inline constexpr int SM_AS_BYTES = align_up_16(KG1 * 4); // 224

// s_silu: SILU_OUT f32 (block-local silu intermediate).
inline constexpr int SM_OFF_SILU = SM_OFF_AS + SM_AS_BYTES;
inline constexpr int SM_SILU_BYTES = align_up_16(SILU_OUT * 4); // 9216

// s_ifp8: W2_K u8 (block-local W2 fp8 activation; 16-aligned).
inline constexpr int SM_OFF_IFP8 = SM_OFF_SILU + SM_SILU_BYTES;
inline constexpr int SM_IFP8_BYTES = align_up_16(W2_K); // 2304

// s_iscale: KG2 f32 (block-local W2 activation scale).
inline constexpr int SM_OFF_ISCALE = SM_OFF_IFP8 + SM_IFP8_BYTES;
inline constexpr int SM_ISCALE_BYTES = align_up_16(KG2 * 4); // 80 (18*4=72)

// s_red: PHYS_NWARP f32 (rmsnorm block-reduce partials).
inline constexpr int SM_OFF_RED = SM_OFF_ISCALE + SM_ISCALE_BYTES;
inline constexpr int SM_RED_BYTES = align_up_16(PHYS_NWARP * 4); // 16

inline constexpr int SM_TOTAL_BYTES =
    SM_OFF_RED + SM_RED_BYTES; // ~99 KB — see static footprint check below.

// 65536 + 14336 + 7168 + 224 + 9216 + 2304 + 80 + 16 = 98880 B (~96.6 KB).
static_assert(SM_TOTAL_BYTES <= 205 * 1024,
              "dense-MLP v2 SMEM footprint must fit the ~205 KB per-worker "
              "dynamic budget (B200 SM100a 227 KB minus static overhead).");

inline constexpr int SM_REGION_WORK = 0;

inline ::mirage::runtime::TaskSmemInfo make_dense_mlp_v2_smem_info(int nwarps) {
  // Consumer-only: nwarps is fixed at PHYS_NWARP (4). The ring is sized for
  // PHYS_NWARP regardless (the v2 consumer runs exactly 128 threads).
  (void)nwarps;
  ::mirage::runtime::TaskSmemInfo info{SM_TOTAL_BYTES, /*alignment=*/1024, {}};
  info.regions.push_back({"densemlp_work",
                          SM_TOTAL_BYTES,
                          /*alignment=*/1024,
                          /*page_count=*/-1,
                          /*can_pack=*/false,
                          /*release_step=*/2,
                          /*contiguous=*/true});
  return info;
}

} // namespace dsv3_dense_mlp_v2
} // namespace kernel
