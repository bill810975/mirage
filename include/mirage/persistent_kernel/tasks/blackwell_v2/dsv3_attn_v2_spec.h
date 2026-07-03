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
// == the verbatim GEMV templates' STAGES*RBT*32*16 requirement:
//   qkv_a: gemv_grid_cpa_t<2,6>            -> 2*32*16*6 =  6144 B/warp
//   q_b:   gemv_grid_cpa_qb_rope_smem_t<8,4> -> 8*32*16*4 = 16384 B/warp
//   oproj: gemv_grid_cpa_oproj_smem_t<8,4>   -> 8*32*16*4 = 16384 B/warp
inline constexpr int P0_RING_BYTES_PER_WARP = 6144;
inline constexpr int GEMV_RING_BYTES_PER_WARP = 16384;
inline constexpr int NWARPS_V2 = 4; // stage-1 consumer-only

// ---- T1 p0_qkva regions -----------------------------------------------------
// [0] WORK: s_act f32[HIDDEN] (28672 B) | red8 f32[8]+pad (32 B) = 28704 B
// [1] RING: 4 * 6144 = 24576 B
inline constexpr int P0_REGION_WORK = 0;
inline constexpr int P0_REGION_RING = 1;
inline constexpr int P0_RED_OFF = HIDDEN * 4; // bytes into WORK

inline ::mirage::runtime::TaskSmemInfo make_p0_qkva_smem_info() {
  int const work_bytes = HIDDEN * 4 + 32;
  int const ring_bytes = NWARPS_V2 * P0_RING_BYTES_PER_WARP;
  ::mirage::runtime::TaskSmemInfo info{work_bytes + ring_bytes,
                                       /*alignment=*/1024,
                                       {}};
  info.regions.push_back({"attn_p0_work", work_bytes, 1024, /*page_count=*/2,
                          /*can_pack=*/false, /*release_step=*/2,
                          /*contiguous=*/true});
  info.regions.push_back({"attn_p0_ring", ring_bytes, 1024, /*page_count=*/2,
                          /*can_pack=*/false, /*release_step=*/2,
                          /*contiguous=*/true});
  return info;
}

// ---- T2 qb_rope_kv regions --------------------------------------------------
// [0] WORK: s_qbdeq f32[QLORA] (6144 B) | red8 f32[8]+pad (32 B) = 6176 B
// [1] RING: 4 * 16384 = 65536 B
inline constexpr int QB_REGION_WORK = 0;
inline constexpr int QB_REGION_RING = 1;
inline constexpr int QB_RED_OFF = QLORA * 4; // bytes into WORK

inline ::mirage::runtime::TaskSmemInfo make_qb_rope_kv_smem_info() {
  int const work_bytes = QLORA * 4 + 32;
  int const ring_bytes = NWARPS_V2 * GEMV_RING_BYTES_PER_WARP;
  ::mirage::runtime::TaskSmemInfo info{work_bytes + ring_bytes,
                                       /*alignment=*/1024,
                                       {}};
  info.regions.push_back({"attn_qb_work", work_bytes, 1024, /*page_count=*/1,
                          /*can_pack=*/false, /*release_step=*/2,
                          /*contiguous=*/true});
  info.regions.push_back({"attn_qb_ring", ring_bytes, 1024,
                          /*page_count=*/NWARPS_V2, /*can_pack=*/false,
                          /*release_step=*/2, /*contiguous=*/true});
  return info;
}

// ---- T3 mla_partial regions -------------------------------------------------
// [0] WORK: s_score f32[512] (2048 B) | red8 f32[8]+pad (32 B) = 2080 B
inline constexpr int MP_REGION_WORK = 0;
inline constexpr int MP_RED_OFF = KVLORA * 4; // bytes into WORK

inline ::mirage::runtime::TaskSmemInfo make_mla_partial_smem_info() {
  int const work_bytes = KVLORA * 4 + 32;
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
// [0] WORK: s_odeq f32[OIN] (8192 B)
// [1] RING: 4 * 16384 = 65536 B
inline constexpr int OP_REGION_WORK = 0;
inline constexpr int OP_REGION_RING = 1;

inline ::mirage::runtime::TaskSmemInfo make_oproj_smem_info() {
  int const work_bytes = OIN * 4;
  int const ring_bytes = NWARPS_V2 * GEMV_RING_BYTES_PER_WARP;
  ::mirage::runtime::TaskSmemInfo info{work_bytes + ring_bytes,
                                       /*alignment=*/1024,
                                       {}};
  info.regions.push_back({"attn_op_work", work_bytes, 1024, /*page_count=*/1,
                          /*can_pack=*/false, /*release_step=*/2,
                          /*contiguous=*/true});
  info.regions.push_back({"attn_op_ring", ring_bytes, 1024,
                          /*page_count=*/NWARPS_V2, /*can_pack=*/false,
                          /*release_step=*/2, /*contiguous=*/true});
  return info;
}

} // namespace dsv3_attn_v2
} // namespace kernel
