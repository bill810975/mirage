/* Copyright 2026 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
#pragma once

// SMEM region layout for the DSv3 tail lm_head GEMV in Runtime-V2 format
// (M3 fix: replaces the fragile TMA+tcgen05 linear_sm100_v3 lm_head with a
// plain scalar/cp.async bf16 GEMV — the same non-TMA pattern the attn+FFN v2
// megakernels use successfully).
//
// The task is CONSUMER-ONLY (a normal, embarrassingly-parallel GEMV — no
// loader/launcher/producer roles, no cross-task in-op barrier). Two SMEM
// regions per task:
//   [0] ACT : the staged bf16 activation (rmsnorm_out[1, K]) = K*2 bytes.
//             Loaded once per task, reused for every output row the task owns.
//   [1] RING: the per-warp cp.async double-buffer for streaming weight rows.
//             Sized RBX rows * 32 lanes * STAGES * 16 bytes per warp.
//
// Host-safe: included by src/kernel/task_register.cc (regions for the planner)
// AND by the device-side dsv3_lmhead_gemv_v2.cuh (smem_region_offset ordinals).
// No __device__ / no PTX asm.

#include "mirage/kernel/task_register.h"

namespace kernel {
namespace dsv3_lmhead_gemv_v2 {

// The kernel streams the weight in uint4 (16B = 8 bf16) chunks, so the
// reduction dim K must be a multiple of 8 (7168 = 8 * 896 for DSv3).
// STAGES double-buffers the weight ring (2 = ping/pong).
inline constexpr int STAGES = 2;

inline constexpr int LMH_REGION_ACT = 0;
inline constexpr int LMH_REGION_RING = 1;

// The v2 SMEM planner packs regions into 16KB pages; a region's declared
// page_count must satisfy page_count*16KB >= size (else "page_count_too_small")
// and a multi-page region is allocated as contiguous physical pages.
inline constexpr int LMH_PAGE_BYTES = 16 * 1024;
inline constexpr int lmh_ceil_pages(int bytes) {
  return (bytes + LMH_PAGE_BYTES - 1) / LMH_PAGE_BYTES;
}

// Bytes for the per-warp cp.async ring: RBX consecutive rows, each row's K
// split across 32 lanes into 16B (uint4) chunks, STAGES deep.
inline constexpr int ring_bytes_per_warp(int rbx) {
  return rbx * 32 * STAGES * 16;
}

// K = reduction dim (7168 for DSv3), nwarps = consumer warps (4), rbx = rows
// computed per warp per inner chunk (8).
inline ::mirage::runtime::TaskSmemInfo
    make_smem_info(int K, int nwarps, int rbx) {
  int const act_bytes = K * 2; // bf16 activation, staged once
  int const ring_bytes = nwarps * ring_bytes_per_warp(rbx);
  ::mirage::runtime::TaskSmemInfo info{act_bytes + ring_bytes,
                                       /*alignment=*/1024,
                                       {}};
  // ACT: contiguous, packable only when it fits one page (14336 B for K=7168).
  info.regions.push_back({"lmh_act",
                          act_bytes,
                          1024,
                          /*page_count=*/lmh_ceil_pages(act_bytes),
                          /*can_pack=*/false,
                          /*release_step=*/2,
                          /*contiguous=*/true});
  // RING: contiguous multi-page (the warp-offset math needs a single flat
  // buffer). page_count = ceil(ring_bytes / 16KB).
  info.regions.push_back({"lmh_ring",
                          ring_bytes,
                          1024,
                          /*page_count=*/lmh_ceil_pages(ring_bytes),
                          /*can_pack=*/false,
                          /*release_step=*/2,
                          /*contiguous=*/true});
  return info;
}

} // namespace dsv3_lmhead_gemv_v2
} // namespace kernel
