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
// SMEM region layout for the FUSED decode-attention megakernel in Runtime-V2
// (T-E of the DSv3-decode-on-v2 effort).
//
// The v1 kernel (tasks/blackwell/attn_block_megakernel_sm100.cuh) uses ONE
// dynamic `extern __shared__ __align__(1024) uint8_t s_smem[]` pool laid out
// as (v1, 256 threads / 8 warps):
//   s_wbuf  = NWARP * CPA_RING_U4 uint4  (per-warp cp.async weight ring)
//   red8    = NWARP floats                (block-reduce partials)
//   s_score = 512 floats                  (MLA per-head scores; reused s_attn)
//   s_act   = K_HIDDEN floats             (28 KB; aliased s_qbdeq/s_odeq)
//
// The v2 CONSUMER-ONLY port runs 128 PHYSICAL threads (4 warps), so the
// per-warp cp.async ring shrinks 8 -> 4 warps (PHYS_NWARP). The block-reduce
// `red8` KEEPS the v1 LOGICAL size (8 entries) because the 128-thread A/B
// emulation stores role-A partials in red8[0..4) and role-B in red8[4..8) so
// v1's cross-warp combine over red8[0..8) is reproduced bit-exactly.
//
// One region (ordinal 0). alignment=1024 (megakernel convention — a smaller
// align silently misaligns other tasks' 1024-aligned TMA/AR in the shared
// test.cu -> cudaErrorMisalignedAddress, caught only in-MPK). The device .cuh
// carves s_wbuf/red8/s_score/s_act out of region 0 with the SAME running
// 16-byte-aligned offsets v1 uses.
//
// Host-safe (no __device__ / no PTX asm) — included by task_register.cc for
// the planner AND by the device .cuh for the region ordinal.
// ============================================================================

#include "mirage/kernel/task_register.h"

namespace kernel {
namespace attn_block_megakernel_v2 {

// ---- physical (v2) vs logical (v1) warp counts ----------------------------
// PHYS_NWARP = the number of consumer warps that actually run (4). V1_NWARP =
// the v1 logical warp count (8) that fixes the red8 layout + combine order.
inline constexpr int PHYS_NTHREAD = 128;
inline constexpr int PHYS_NWARP = 4;
inline constexpr int V1_NWARP = 8;

// Must match the v1 kernel's CPA_RING_U4 (worst-case GEMV weight ring depth,
// uint4 per warp). The device .cuh static_asserts equality against its own
// constexpr.
inline constexpr int CPA_RING_U4 = 1024;

// Running 16-byte-aligned offsets, mirroring the v1 s_smem carve. All in BYTES.
inline constexpr int align_up_16(int n) {
  return (n + 15) & ~15;
}

// s_wbuf: PHYS_NWARP per-warp cp.async rings (16-byte uint4 elements).
inline constexpr int SM_OFF_WBUF = 0;
inline constexpr int SM_WBUF_BYTES = PHYS_NWARP * CPA_RING_U4 * 16; // 65536

// red8: V1_NWARP floats (LOGICAL size — A/B emulation needs 8 slots).
inline constexpr int SM_OFF_RED8 = SM_OFF_WBUF + SM_WBUF_BYTES;
inline constexpr int SM_RED8_BYTES = align_up_16(V1_NWARP * 4); // 32

// s_score: 512 floats (per-head MLA scores; also reused as s_attn in merge).
inline constexpr int SM_OFF_SCORE = SM_OFF_RED8 + SM_RED8_BYTES;
inline constexpr int SM_SCORE_BYTES = align_up_16(512 * 4); // 2048

// s_act: K_HIDDEN(7168) floats block-local dequant activation (aliased across
// the qkv_a / q_b / o_proj phases exactly as v1 does).
inline constexpr int SM_OFF_ACT = SM_OFF_SCORE + SM_SCORE_BYTES;
inline constexpr int SM_ACT_BYTES = align_up_16(7168 * 4); // 28672

inline constexpr int SM_TOTAL_BYTES =
    SM_OFF_ACT + SM_ACT_BYTES; // 65536+32+2048+28672 = 96288 (~94 KB)

inline constexpr int SM_REGION_WORK = 0;

inline ::mirage::runtime::TaskSmemInfo make_smem_info() {
  ::mirage::runtime::TaskSmemInfo info{SM_TOTAL_BYTES, /*alignment=*/1024, {}};
  info.regions.push_back({"attnmega_work",
                          SM_TOTAL_BYTES,
                          /*alignment=*/1024,
                          /*page_count=*/-1,
                          /*can_pack=*/false,
                          /*release_step=*/2,
                          /*contiguous=*/true});
  return info;
}

} // namespace attn_block_megakernel_v2
} // namespace kernel
