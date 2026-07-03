/* Copyright 2026 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
#pragma once

// Single source of truth for the DSv3 fused-ATTN v2 task family's SMEM region
// layouts (Step 3b of the V2 migration: the v1 attn_block_megakernel expressed
// as a v2 task CHAIN:
//   dsv3_attn_p0_qkva_v2 -> dsv3_attn_qb_rope_kv_v2 -> dsv3_attn_mla_partial_v2
//   -> dsv3_attn_mla_merge_v2 -> dsv3_attn_wuv_v2 -> dsv3_attn_oproj_v2).
//
// Host-safe: included by src/kernel/task_register.cc (regions for the
// planner) AND by the device-side dsv3_attn_v2.cuh (typed views). The region
// ordinals here are the contract for task_desc->smem_region_offset(N).
//
// Shapes are the DSv3 TP8 decode per-rank shapes (16 local heads), kept in
// lock-step with kernel::attn_block_megakernel_sm100's K_* macros
// (static_asserts in dsv3_attn_v2.cuh).

#include "mirage/kernel/task_register.h"

namespace kernel {
namespace dsv3_attn_v2 {

// ---- problem shapes (mirror attn_block_megakernel_sm100's K_* macros;
// device side asserts equality) --------------------------------------------
inline constexpr int HIDDEN = 7168;  // K_HIDDEN
inline constexpr int QLORA = 1536;   // K_QLORA
inline constexpr int KVLORA = 512;   // K_KVLORA
inline constexpr int QKROPE = 64;    // K_QKROPE
inline constexpr int QKHEAD = 576;   // K_QKHEAD
inline constexpr int VHEAD = 128;    // K_VHEAD
inline constexpr int QKVAN = 2176;   // K_QKVAN
inline constexpr int HLOCAL = 16;    // K_HLOCAL
inline constexpr int OIN = 2048;     // K_OIN
inline constexpr int GRP = 128;      // K_GRP
inline constexpr int SPLITS = 8;     // MLA_SPLITS

// ---- per-warp cp.async ring geometry (bytes) -------------------------------
// == the verbatim GEMV templates' STAGES*RBT*32*16 requirement. Round-3
// tuning: deeper rings at 4 MAC warps (in-flight bytes per warp is the
// concurrency lever; warp-count is not — round-1 verdict). STAGES here must
// match the template args in dsv3_attn_v2.cuh (static_asserts there).
// NOTE: at these depths the nwarps=7 ring (7*32KB + work > 14 pages) no
// longer fits for qb/oproj — unused since round 1 reverted them to 4 warps.
inline constexpr int P0_GEMV_RBT = 4;      // 544 blocks = 1/warp exactly:
inline constexpr int P0_GEMV_STAGES = 8;   //   -38% GEMV vs <2,6> (round 3)
inline constexpr int QB_GEMV_RBT = 8;      // 1152 blocks (16 straggled: NULL)
inline constexpr int QB_GEMV_STAGES = 4;   //   original <8,4> retained
inline constexpr int OP_GEMV_RBT = 16;     // 448 blocks = 448 warps = 1/warp
inline constexpr int OP_GEMV_STAGES = 4;   //   K=2048 => 4 super-steps
inline constexpr int P0_RING_BYTES_PER_WARP =
    P0_GEMV_RBT * 32 * 16 * P0_GEMV_STAGES;
inline constexpr int QB_RING_BYTES_PER_WARP =
    QB_GEMV_RBT * 32 * 16 * QB_GEMV_STAGES;
inline constexpr int OP_RING_BYTES_PER_WARP =
    OP_GEMV_RBT * 32 * 16 * OP_GEMV_STAGES;
inline constexpr int NWARPS_V2 = 4; // stage-1 consumer-only

// Stage-2 (nwarps=7) cross-role tag-flag block appended to each WORK region:
// u64[4]: [0] consumer->helper GO (monotonic phase tags), [1..3] per-helper
// DONE. A 16-B scalar slot (rms_rcp / q_rcp handoff) precedes it.
inline constexpr int ATTN_FLAG_BYTES = 4 * 8;

// ---- T1 p0_qkva regions -----------------------------------------------------
// [0] WORK: s_act f32[HIDDEN] (28672 B) | red8 f32[8]+pad (32 B) |
//           scalar f32+pad (16 B) | flags u64[4] (32 B) = 28752 B
// [1] RING: nwarps * 6144 B. The first HIDDEN*2 = 14336 B double as the
//           bf16 x staging buffer during the prologue (dead before the GEMV
//           touches the ring).
inline constexpr int P0_REGION_WORK = 0;
inline constexpr int P0_REGION_RING = 1;
inline constexpr int P0_RED_OFF = HIDDEN * 4; // bytes into WORK
inline constexpr int P0_SCALAR_OFF = P0_RED_OFF + 32;
inline constexpr int P0_FLAGS_OFF = P0_SCALAR_OFF + 16;

inline ::mirage::runtime::TaskSmemInfo make_p0_qkva_smem_info(int nwarps) {
  int const work_bytes = P0_FLAGS_OFF + ATTN_FLAG_BYTES;
  int const ring_bytes = nwarps * P0_RING_BYTES_PER_WARP;
  int const ring_pages = (ring_bytes + 16383) / 16384;
  ::mirage::runtime::TaskSmemInfo info{work_bytes + ring_bytes,
                                       /*alignment=*/1024,
                                       {}};
  info.regions.push_back({"attn_p0_work", work_bytes, 1024, /*page_count=*/2,
                          /*can_pack=*/false, /*release_step=*/2,
                          /*contiguous=*/true});
  info.regions.push_back({"attn_p0_ring", ring_bytes, 1024,
                          /*page_count=*/ring_pages, /*can_pack=*/false,
                          /*release_step=*/2, /*contiguous=*/true});
  return info;
}

// ---- T2 qb_rope_kv regions --------------------------------------------------
// [0] WORK: s_qbdeq f32[QLORA] (6144 B) | red8 f32[8]+pad (32 B) |
//           scalar (16 B) | flags u64[4] (32 B) = 6224 B
// [1] RING: nwarps * 16384 B
inline constexpr int QB_REGION_WORK = 0;
inline constexpr int QB_REGION_RING = 1;
inline constexpr int QB_RED_OFF = QLORA * 4; // bytes into WORK
inline constexpr int QB_SCALAR_OFF = QB_RED_OFF + 32;
inline constexpr int QB_FLAGS_OFF = QB_SCALAR_OFF + 16;

inline ::mirage::runtime::TaskSmemInfo make_qb_rope_kv_smem_info(int nwarps) {
  int const work_bytes = QB_FLAGS_OFF + ATTN_FLAG_BYTES;
  int const ring_bytes = nwarps * QB_RING_BYTES_PER_WARP;
  int const ring_pages = (ring_bytes + 16383) / 16384;
  ::mirage::runtime::TaskSmemInfo info{work_bytes + ring_bytes,
                                       /*alignment=*/1024,
                                       {}};
  info.regions.push_back({"attn_qb_work", work_bytes, 1024, /*page_count=*/1,
                          /*can_pack=*/false, /*release_step=*/2,
                          /*contiguous=*/true});
  info.regions.push_back({"attn_qb_ring", ring_bytes, 1024,
                          /*page_count=*/ring_pages, /*can_pack=*/false,
                          /*release_step=*/2, /*contiguous=*/true});
  return info;
}

// ---- T3 mla_partial regions (shared by the round-4 FUSED partial+merge) ----
// [0] WORK: s_score f32[512] (2048 B) | red8 f32[8]+pad (32 B) |
//           flags u64[4] (32 B) | s_last i32+pad (16 B) = 2128 B
inline constexpr int MP_REGION_WORK = 0;
inline constexpr int MP_RED_OFF = KVLORA * 4; // bytes into WORK
inline constexpr int MP_FLAGS_OFF = MP_RED_OFF + 32;
inline constexpr int MP_LAST_OFF = MP_FLAGS_OFF + ATTN_FLAG_BYTES;

inline ::mirage::runtime::TaskSmemInfo make_mla_partial_smem_info() {
  int const work_bytes = MP_LAST_OFF + 16;
  ::mirage::runtime::TaskSmemInfo info{work_bytes, /*alignment=*/1024, {}};
  info.regions.push_back({"attn_mp_work", work_bytes, 1024, /*page_count=*/1,
                          /*can_pack=*/true, /*release_step=*/2,
                          /*contiguous=*/true});
  return info;
}

// ---- T4 mla_merge regions ---------------------------------------------------
// [0] WORK: s_attn f32[512] (2048 B) — mla_merge_quant never touches red8.
inline constexpr int MM_REGION_WORK = 0;

inline ::mirage::runtime::TaskSmemInfo make_mla_merge_smem_info() {
  int const work_bytes = KVLORA * 4;
  ::mirage::runtime::TaskSmemInfo info{work_bytes, /*alignment=*/1024, {}};
  info.regions.push_back({"attn_mm_work", work_bytes, 1024, /*page_count=*/1,
                          /*can_pack=*/true, /*release_step=*/2,
                          /*contiguous=*/true});
  return info;
}

// ---- T5 wuv: GMEM-only, no SMEM regions (v1's wuv_bmm_grid uses none) -------
inline ::mirage::runtime::TaskSmemInfo make_wuv_smem_info() {
  return ::mirage::runtime::TaskSmemInfo{/*size=*/0, /*alignment=*/1, {}};
}

// ---- T6 oproj regions -------------------------------------------------------
// [0] WORK: s_odeq f32[OIN] (8192 B) | flags u64[4] (32 B) = 8224 B
// [1] RING: nwarps * 16384 B
inline constexpr int OP_REGION_WORK = 0;
inline constexpr int OP_REGION_RING = 1;
inline constexpr int OP_FLAGS_OFF = OIN * 4; // bytes into WORK

inline ::mirage::runtime::TaskSmemInfo make_oproj_smem_info(int nwarps) {
  int const work_bytes = OP_FLAGS_OFF + ATTN_FLAG_BYTES;
  int const ring_bytes = nwarps * OP_RING_BYTES_PER_WARP;
  int const ring_pages = (ring_bytes + 16383) / 16384;
  ::mirage::runtime::TaskSmemInfo info{work_bytes + ring_bytes,
                                       /*alignment=*/1024,
                                       {}};
  info.regions.push_back({"attn_op_work", work_bytes, 1024, /*page_count=*/1,
                          /*can_pack=*/false, /*release_step=*/2,
                          /*contiguous=*/true});
  info.regions.push_back({"attn_op_ring", ring_bytes, 1024,
                          /*page_count=*/ring_pages, /*can_pack=*/false,
                          /*release_step=*/2, /*contiguous=*/true});
  return info;
}

} // namespace dsv3_attn_v2
} // namespace kernel
