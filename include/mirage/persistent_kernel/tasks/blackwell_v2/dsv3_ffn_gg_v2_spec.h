/* Copyright 2026 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
#pragma once

// Single source of truth for the DSv3 W13/W2 grouped-GEMM PER-TILE PIPELINE
// v2 tasks (ffn item 1, spec: scratch/v2_rewrite/ffn_item1_spec.md):
//   TASK_DSV3_FFN_W13_PIPE_V2  — one (slot, n_tile) [routed, 64 tasks] or one
//                                n_tile [shared gate_up, ALWAYS_ACTIVE=1,
//                                4 tasks] W13 output tile.
//   TASK_DSV3_FFN_W2_PIPE_V2   — one n_tile [56 tasks]; slot segments are
//                                internal (slots-ascending-then-shared, Q6).
//
// This header pins the ABI the ferret-v2 optimizer must keep:
//   * SMEM region table (spec §2): 8 W K-stage regions (16 KB each, 1 page,
//     TMA dst => 1024-aligned) + 1 BSF region (24,592 B, 2 pages) holding the
//     8 per-stage B/SFA/SFB slices + the taddr scratch. 9 regions, 10 pages,
//     155,664 B total. 4 free pages => STAGES can grow to 12 (Class B).
//   * SEM ordinal table (spec §3): 30 op-private ordinals (<= 31).
//   * Stage geometry: STAGES=8 (v1-proven; v1 hung at 4 — never go below 6),
//     ACC_STAGES=2, MMA_N=16, BLOCK_M=128 weight rows/tile, bK=128 (== GRP,
//     one scale block per K-stage).
//   * Tile decomposition (spec §5): routed W13 tile_idx = slot * (N/128) +
//     n_tile (slot-major); shared W13 / W2 tile_idx = n_tile.
//
// Host-safe: included by src/kernel/task_register.cc (planner regions +
// init_semaphores counts) AND by the device-side dsv3_ffn_gg_v2.cuh.
//
// Shape constants are imported from kernel::dsv3_ffn_v2 (dsv3_ffn_v2_spec.h),
// which the device side pins to kernel::ffn_full_megakernel_sm100 via
// static_asserts (dsv3_ffn_v2.cuh:73-79) — one chain of truth.

#include "mirage/kernel/task_register.h"
#include "mirage/persistent_kernel/tasks/blackwell_v2/dsv3_ffn_v2_spec.h"

namespace kernel {
namespace dsv3_ffn_gg_v2 {

namespace F = ::kernel::dsv3_ffn_v2; // DSv3 FFN shape constants (spec'd)

// ---- pipeline stage geometry (spec §2/§4; v1 production constants, W5) ----
inline constexpr int STAGES = 8;     // K-stage ring depth (never < 6)
inline constexpr int ACC_STAGES = 2; // TMEM accumulator stages
inline constexpr int MMA_N = 16;     // padded-token dimension
inline constexpr int BLOCK_M = 128;  // weight rows per output tile
inline constexpr int BK = 128;       // K per stage == GRP (scale-block invar.)
static_assert(BK == F::GRP, "bK must equal the 128-wide scale group");

// ---- task counts per op instance (spec §5) --------------------------------
inline constexpr int W13_TILES_PER_SLOT = F::W13_N / BLOCK_M; // 8
inline constexpr int W13_ROUTED_TILES =
    F::MAX_ACTIVE * W13_TILES_PER_SLOT;                       // 64
inline constexpr int W13_SHARED_TILES = F::SH_GU_N / BLOCK_M; // 4
inline constexpr int W2_TILES = F::W2_N / BLOCK_M;            // 56

// ---- SMEM regions (spec §2) ------------------------------------------------
// | region   | ordinal | size      | align | page_count | contents          |
// | W_0..W_7 | 0..7    | 16,384 B  | 1024  | 1 each     | fp8 W K-stage tile|
// |          |         |           |       |            | [128x128], TMA dst|
// | BSF      | 8       | 24,592 B  | 1024  | 2          | 8 slices @ s*3072:|
// |          |         |           |       |            | B[16x128]u8 @+0,  |
// |          |         |           |       |            | SFA u32[128]@+2048|
// |          |         |           |       |            | SFB u32[128]@+2560|
// |          |         |           |       |            | + taddr 16B@+24576|
// total: 9 regions / 10 pages / 155,664 B  (<= 224,256; <= 14 pages; <= 16)
inline constexpr int W_BYTES = BLOCK_M * BK; // 16384 (fp8 = 1 B/elt)
inline constexpr int BSF_STAGE_STRIDE = 3072;
inline constexpr int BSF_OFF_B = 0;      // B tile u8[MMA_N x BK]   (2048 B)
inline constexpr int BSF_OFF_SFA = 2048; // SFA u32[128] UTCCP splat (512 B)
inline constexpr int BSF_OFF_SFB = 2560; // SFB u32[128] UTCCP splat (512 B)
inline constexpr int BSF_OFF_TADDR = STAGES * BSF_STAGE_STRIDE; // 24576
inline constexpr int BSF_TADDR_BYTES = 16;
inline constexpr int BSF_BYTES = BSF_OFF_TADDR + BSF_TADDR_BYTES; // 24592
static_assert(MMA_N * BK == 2048, "B slice layout drifted");

// Region ordinals: must match the push_back order in make_ffn_pipe_smem_info()
// and the smem_region_offset(...) calls in dsv3_ffn_gg_v2.cuh.
inline constexpr int REGION_W_0 = 0;           // .. REGION_W_0 + STAGES - 1
inline constexpr int REGION_BSF = STAGES;      // 8
inline constexpr int NUM_REGIONS = STAGES + 1; // 9

inline constexpr int total_smem_bytes() {
  return STAGES * W_BYTES + BSF_BYTES; // 155,664
}

// Keep PLANNER_CAPACITY_BYTES in sync with
// python/mirage/mpk/v2_smem_planner.py:CAPACITY_BYTES.
inline constexpr int PLANNER_CAPACITY_BYTES = 225 * 1024 - 6 * 1024; // 224256
static_assert(total_smem_bytes() == 155664,
              "dsv3_ffn_gg_v2 SMEM footprint drifted from the spec §2 table");
static_assert(total_smem_bytes() <= PLANNER_CAPACITY_BYTES,
              "dsv3_ffn_gg_v2 SMEM footprint exceeds the planner capacity");
static_assert(NUM_REGIONS <= 16, "MAX_SMEM_REGIONS_PER_TASK is 16");

// ---- SEM ordinal table (spec §3; relative to dyn_sem_base) -----------------
//   [+0 ..+7 ]  W_tma_mbar   (count=1,    loader->launcher; TMA tx=16384 B,
//                             async-arrived => loader re-inits at task start)
//   [+8 ..+15]  B_sf_mbar    (count=1,    loader->launcher; loader arrives
//                             ONCE per stage AFTER cpasync_wait<0> AND
//                             `fence.proxy.async.shared::cta` — the async-
//                             proxy fence is MANDATORY (Codex 4a): the
//                             fence.proxy.async VARIANT is the chosen one
//                             (not cp.async.mbarrier.arrive.noinc), so this
//                             mbar is thread-arrived; re-init kept anyway.)
//   [+16..+23]  mma_mbar     (count=1,    launcher->loader, "stage K MMA
//                             done, refill ok"; tcgen05.commit = async)
//   [+24..+25]  mainloop_mbar(count=1,    launcher->consumer, "ACC[s] full";
//                             tcgen05.commit = async)
//   [+26..+27]  epilogue_mbar(count=4*32, consumer->launcher, "ACC drained")
//   [+28]       tmem_ready   (count=1,    launcher->consumer, taddr publish)
//   [+29]       consumer_done(count=4*32, consumer->launcher, TMEM released)
inline constexpr int WARP_SIZE = 32;
inline constexpr int SEM_W_TMA_BASE = 0;
inline constexpr int SEM_B_SF_BASE = STAGES;                       // 8
inline constexpr int SEM_MMA_BASE = 2 * STAGES;                    // 16
inline constexpr int SEM_MAINLOOP_BASE = 3 * STAGES;               // 24
inline constexpr int SEM_EPILOGUE_BASE = 3 * STAGES + ACC_STAGES;  // 26
inline constexpr int SEM_TMEM_READY = 3 * STAGES + 2 * ACC_STAGES; // 28
inline constexpr int SEM_CONSUMER_DONE = SEM_TMEM_READY + 1;       // 29
inline constexpr int NUM_OP_SEMS = SEM_CONSUMER_DONE + 1;          // 30
static_assert(NUM_OP_SEMS <= 31,
              "op-private SEM budget is 31 (SEM_OP_BASE..MAX_DYNAMIC-1); "
              "STAGES > 8 would overflow — fold W/B mbars first");

inline ::mirage::runtime::TaskSmemInfo make_ffn_pipe_smem_info() {
  ::mirage::runtime::TaskSmemInfo info{total_smem_bytes(),
                                       /*alignment=*/1024,
                                       {}};
  // 8 W K-stage regions: exactly 1 page each, TMA dst (128B swizzle) =>
  // 1024-aligned, never packed.
  for (int s = 0; s < STAGES; s++) {
    info.regions.push_back({"ffnpipe_W_" + std::to_string(s),
                            W_BYTES,
                            /*alignment=*/1024,
                            /*page_count=*/1,
                            /*can_pack=*/false,
                            /*release_step=*/1,
                            /*contiguous=*/true});
  }
  // BSF: 8 per-stage B/SFA/SFB slices + taddr scratch, 2 contiguous pages.
  info.regions.push_back({"ffnpipe_BSF",
                          BSF_BYTES,
                          /*alignment=*/1024,
                          /*page_count=*/2,
                          /*can_pack=*/false,
                          /*release_step=*/1,
                          /*contiguous=*/true});
  return info;
}

} // namespace dsv3_ffn_gg_v2
} // namespace kernel
