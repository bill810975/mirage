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
// DSv3 W13/W2 grouped GEMM as PER-TILE v2 PIPELINE tasks (ffn item 1).
// Requirement of record: scratch/v2_rewrite/ffn_item1_spec.md.
//
//   TASK_DSV3_FFN_W13_PIPE_V2 (two instances of one type):
//     routed  (ALWAYS_ACTIVE=0): task = (slot in 0..7, n_tile in 0..7), 64
//       tasks; y13[slot, n0:n0+128] = W13[e_slot][n0:n0+128, :] @ a.
//     shared  (ALWAYS_ACTIVE=1): task = n_tile in 0..3, 4 tasks;
//       sg[n0:n0+128] = wgu[n0:n0+128, :] @ a. meta ignored (always live).
//   TASK_DSV3_FFN_W2_PIPE_V2: task = n_tile in 0..55, 56 tasks;
//     out[n0+t] = sum_s ew_s * (W2[e_s][n0+t, :] @ i_s) + wdn[n0+t, :] @ si.
//     Segment order = slots ascending then shared (Q6 determinism pin);
//     f32 register accumulation, ONE bf16 store, no atomics.
//
// Warp/role model (reference pipeline, house-style §1): loader W4 / launcher
// W5 / consumers W0-3 / storer W6. Source engine being re-hosted: the v1
// swapAB block-scaled FP8 UMMA (tasks/blackwell/fp8_group_gemm_sm100.cuh,
// MMA_M=128, MMA_N=16, bK=128, STAGES=8, ACC=2, UE8M0 scales via UTCCP).
//
// >>> PIPELINE STATUS: BOTH W13 (v007-v009) and W2 (v010) are REAL tcgen05
// >>> pipelines. Loader stages W tiles (TMA) + B row 0 (cp.async) + splatted
// >>> UE8M0 scales; launcher issues UTCCP + block-scaled UMMA into a TMEM ACC;
// >>> consumer reads the ACC via tcgen05.ld (col 0). W2 adds a VARIABLE
// per-tile
// >>> segment loop (active+1 segments = routed slots 0..active-1 then one
// >>> shared-down, cycling the 2-stage ACC ring; consumer does the
// cross-segment
// >>> ew-weighted register sum). What is PINNED and must be kept by any
// rewrite:
// >>>   * the spec.h region/SEM tables + stage geometry (the ABI),
// >>>   * role function signatures (registration emits these calls),
// >>>   * stale-arrival re-init ownership (spec §3 table): loader owns
// >>>     W_tma + B_sf + mma; launcher lane 0 owns mainloop + epilogue;
// >>>     re-init ALSO runs before the inactive-slot bail (Codex 4b),
// >>>   * dep-wait placement: loader inline BEFORE reading meta (routed
// >>>     coords depend on meta); launcher/consumer/storer get the codegen
// >>>     consumer_dep_prefix (registration-emitted),
// >>>   * page release ownership (Q3, linear's proven combination): codegen
// >>>     loader page prefix (waits all pages, releases the 4 pages this op
// >>>     does NOT use) + launcher task-end blanket release of the pages the
// >>>     task USES (task_uses_page-gated — this op uses 10 of 14, unlike
// >>>     linear's 14/14) + auto_consumer_finish=false + EMPTY-protocol
// >>>     storer. Ownership may be TRANSFERRED to the storer (per-stage
// >>>     release, spec §4) .cuh-only — move the release, never duplicate it:
// >>>     every page must be arrived EXACTLY once per task,
// >>>   * inactive-slot bail (spec §4): all four roles detect slot >=
// >>>     active_count independently from meta, NO cross-role wait on the
// >>>     bail path, pages still released exactly once,
// >>>   * B_sf publication rule (Codex 4a): the fence.proxy.async variant is
// >>>     the CHOSEN one — cpasync_wait<0> then
// >>>     `fence.proxy.async.shared::cta` then ONE release-arrive per stage
// >>>     (covers both the cp.async B bytes and the plain st.shared SFA/SFB
// >>>     splats). count=1; do NOT switch to cp.async.mbarrier.arrive.noinc
// >>>     without re-ledgering,
// >>>   * B-tile padding rows 1..15 zero-filled ONCE at task start before
// >>>     any B_sf release (cp.async only ever rewrites row 0),
// >>>   * tcgen05 lifecycle (real body): alloc 64 cols -> taddr cached in
// >>>     registers at alloc -> lane-0 tmem_ready -> consumers read taddr
// >>>     once -> all-128 consumer_done -> dealloc with the CACHED taddr;
// >>>     alloc/dealloc same warp, sync.aligned, all 32 lanes,
// >>>   * no __syncthreads / no named barrier in role bodies (mbars +
// >>>     __syncwarp only; named-barrier ids 1/2/3/6 are taken),
// >>>   * real body SMEM only via task_desc->smem_region_offset(REGION_*)
// >>>     with `extern __shared__ __align__(1024)` (never smaller).
// ============================================================================

#include "mirage/persistent_kernel/runtime_header.h"
#include "mirage/persistent_kernel/tasks/blackwell_v2/dsv3_ffn_gg_v2_spec.h"
#include "mirage/persistent_kernel/tasks/blackwell_v2/dsv3_ffn_v2.cuh"
// Block-scaled FP8 UMMA math (VERBATIM re-use — do not hand-roll): the v1
// group-GEMM descriptor/MMA helpers in namespace kernel::sm100 (make_umma_desc,
// make_sf_desc, replace_smem_desc_addr, advance_umma_desc_lo,
// make_runtime_instr_desc_with_sf_id, SM100_MMA_MXF8F6F4_SS). We include the v1
// header (NOT the byte-identical blackwell_v2 copy): the megakernel already
// pulls in blackwell/sm100_utils.cuh transitively, and both define the SAME
// kernel::sm100:: symbols — including the v2 duplicate is a redefinition error.
// #pragma once makes this idempotent with the transitive include.
#include "mirage/persistent_kernel/tasks/blackwell/sm100_utils.cuh"
#include <cute/arch/copy_sm100.hpp> // SM100_UTCCP_4x32dp128bit_1cta (scale->TMEM)
#include <cute/arch/mma_sm100_desc.hpp> // make_instr_desc_block_scaled

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

namespace kernel {
namespace dsv3_ffn_gg_v2 {

namespace v1k = ::kernel::ffn_full_megakernel_sm100;

// Belt-and-braces device-side pin (the spec.h imports its shapes from
// kernel::dsv3_ffn_v2, which dsv3_ffn_v2.cuh pins to the v1 kernel).
static_assert(F::HIDDEN == v1k::HIDDEN && F::W13_N == v1k::W13_N &&
                  F::W2_K == v1k::W2_K && F::W2_N == v1k::W2_N &&
                  F::GRP == v1k::GRP && F::MAX_ACTIVE == v1k::MAX_ACTIVE &&
                  F::SH_GU_N == v1k::SH_GU_N && F::SH_DN_K == v1k::SH_DN_K,
              "dsv3_ffn_gg_v2 shapes drifted from the v1 kernel");

// ── local helpers (own copies, own namespace — no cross-namespace reuse) ──
__device__ __forceinline__ uint32_t ffngg_elect_sync() {
  uint32_t pred = 0;
  asm volatile("{\n\t.reg .pred %%px;\n\t"
               "elect.sync _|%%px, %1;\n\t"
               "@%%px mov.s32 %0, 1;\n\t}"
               : "+r"(pred)
               : "r"(0xFFFFFFFF));
  return pred;
}

__device__ __forceinline__ void ffngg_mbar_init(int mbar_addr, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(mbar_addr),
               "r"(count));
}

// ── v2 raw-address mbarrier idiom (own local copies, matching
// linear_sm100_v2.cuh's own local helpers — the v2 runtime addresses mbars as
// dyn_sem_base + ordinal*8 raw integers; the cutlass ClusterBarrier objects are
// INCOMPATIBLE with that scheme, so we re-express sync in raw PTX). Unique
// asm labels via %= so repeated inlining never collides. ──
__device__ __forceinline__ void ffngg_mbar_wait(int mbar_addr, int phase) {
  asm volatile("{\n\t.reg .pred P1;\n\t"
               "FGG_WAIT_%=:\n\t"
               "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], "
               "%1, 0x989680;\n\t"
               "@P1 bra.uni FGG_DONE_%=;\n\t"
               "bra.uni FGG_WAIT_%=;\n\t"
               "FGG_DONE_%=:\n\t}" ::"r"(mbar_addr),
               "r"(phase));
}

__device__ __forceinline__ void ffngg_mbar_arrive_expect_tx(int mbar_addr,
                                                            int size) {
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::
          "r"(mbar_addr),
      "r"(size)
      : "memory");
}

__device__ __forceinline__ void ffngg_mbar_arrive(int mbar_addr) {
  asm volatile(
      "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(mbar_addr)
      : "memory");
}

// 2D bulk-tensor TMA load (weight tile [128 rows x bK=128] u8, 128B swizzle;
// descriptor built host-side by tma.cuh fill_tma_desc<uint8_t,...,2> — box
// {FP8_BK=128 (K), MMA_M=128 (rows)}; coords {x=K-elt offset, y=row offset}).
// Exact form of every proven 2D fp8 TMA in the tree
// (fp8_group_gemm_sm100_common.cuh:97): .shared::cta.global, no cta_group / no
// L2::cache_hint (linear's .shared::cluster.cta_group::1.L2::cache_hint 3D form
// is ILLEGAL for this 2D u8 descriptor — verified via compute-sanitizer).
__device__ __forceinline__ void ffngg_tma_2d_load(
    int dst, void const *tmap_ptr, int x, int y, int mbar_addr) {
  // fill_tma_desc encodes tensorRank = tma_dim = 5 (tma.cuh:39/243) — the
  // instruction dimensionality must match, so use .5d with the two live coords
  // {x=K, y=row} and 0 for the three size-1 padding dims.
  int const z = 0;
  asm volatile(
      "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::"
      "complete_tx::bytes [%0], [%1, {%2, %3, %4, %4, %4}], [%5];" ::"r"(dst),
      "l"(tmap_ptr),
      "r"(x),
      "r"(y),
      "r"(z),
      "r"(mbar_addr)
      : "memory");
}

// tcgen05 async-completion commit: arrives `mbar_addr` after all outstanding
// tcgen05 ops (UMMA + UTCCP) of this thread complete (v1 umma_arrive analogue).
__device__ __forceinline__ void ffngg_tcgen05_commit(int mbar_addr) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::"
               "cluster.b64 [%0];" ::"r"(mbar_addr)
               : "memory");
}

inline constexpr uint64_t FFNGG_L2_EVICT_FIRST = 0x12F0000000000000ULL;
inline constexpr uint64_t FFNGG_L2_EVICT_LAST = 0x14F0000000000000ULL;

// Exact e4m3 byte -> f32 decode (same value the v1 dgemv helpers see).
__device__ __forceinline__ float ffngg_fp8_to_f32(uint8_t b) {
  __half_raw hr = __nv_cvt_fp8_to_halfraw(b, __NV_E4M3);
  return __half2float(*reinterpret_cast<__half const *>(&hr));
}

// Meta accessors (layout = dsv3_ffn_v2_spec.h META_*).
__device__ __forceinline__ int ffngg_meta_active(int const *meta) {
  return meta[F::META_OFF_COUNT];
}
__device__ __forceinline__ int ffngg_meta_expert(int const *meta, int slot) {
  return meta[F::META_OFF_EXPERTS + slot];
}
__device__ __forceinline__ float ffngg_meta_weight(int const *meta, int slot) {
  return __int_as_float(meta[F::META_OFF_WEIGHTS + slot]);
}

// Stale-arrival re-init block OWNED BY THE LOADER (spec §3 table): W_tma
// (TMA byte-delivery = async-arrived), mma (tcgen05.commit = async-arrived),
// B_sf (thread-arrived under the chosen fence.proxy.async variant; re-init
// kept anyway — harmless, uniform). Runs at task start, BEFORE the first TMA
// and BEFORE the launcher's first commit; ALSO runs before the inactive bail
// (Codex 4b hardening).
__device__ __forceinline__ void ffngg_loader_reinit_mbars(int dyn_sem_base) {
  for (int s = 0; s < STAGES; s++) {
    ffngg_mbar_init(dyn_sem_base + (SEM_W_TMA_BASE + s) * 8, 1);
    ffngg_mbar_init(dyn_sem_base + (SEM_B_SF_BASE + s) * 8, 1);
    ffngg_mbar_init(dyn_sem_base + (SEM_MMA_BASE + s) * 8, 1);
  }
  asm volatile("fence.mbarrier_init.release.cluster;");
}

// Stale-arrival re-init block OWNED BY THE LAUNCHER lane 0 (spec §3 table):
// mainloop (tcgen05.commit = async-arrived) + epilogue (re-init with mainloop
// for prior-occupant residue) + consumer_done. In the real body this runs
// BEFORE arriving tmem_ready; also before the inactive bail.
//
// consumer_done re-init (added round 2, evidence-driven): the spec §3 table
// listed consumer_done as "controller init only (linear precedent)", but the
// per-iteration repro (L=2/iters=4) PROVED a stale-arrival wedge on it — the
// launcher's consumer_done wait hung on iter 1 with the mbar at the OPPOSITE
// parity (opp_ready=1), and a launcher-entry probe showed phase_bit==1 (STALE,
// controller re-init did NOT reset it) on exactly the hung tasks. Mechanism:
// the consumer arrives consumer_done at the very END of its body, immediately
// before its INSTRUCTION_FINISHED arrive that gates cross-task slot reuse — so
// that arrival's visibility races the controller's per-publish re-init on the
// reused ring slot (INSTRUCTION_RING_SIZE=3), leaving the freshly-init'd mbar
// flipped back to phase 1 for the NEXT occupant. epilogue is arrived at the
// same late point with the identical race but did NOT wedge — because the
// launcher already re-inits it here (role-level, late-enough: placed after the
// controller's INSTRUCTION_ARRIVED handshake, so the stray arrival has landed).
// consumer_done was simply missing from this list. Re-initing it here inherits
// epilogue's proven timing. SAFE against the current task's own consumer: that
// arrive is gated after tmem_ready (arrived by the launcher AFTER this
// re-init), so this can never wipe a live arrival. tmem_ready needs no re-init
// — its only arriver (the launcher) fires early in-task, far from the
// slot-reuse boundary, so it has no stale window (empirically: the consumer's
// tmem_ready wait never wedged across every repro).
__device__ __forceinline__ void ffngg_launcher_reinit_mbars(int dyn_sem_base) {
  for (int s = 0; s < ACC_STAGES; s++) {
    ffngg_mbar_init(dyn_sem_base + (SEM_MAINLOOP_BASE + s) * 8, 1);
    ffngg_mbar_init(dyn_sem_base + (SEM_EPILOGUE_BASE + s) * 8, 4 * WARP_SIZE);
  }
  ffngg_mbar_init(dyn_sem_base + SEM_CONSUMER_DONE * 8, 4 * WARP_SIZE);
  asm volatile("fence.mbarrier_init.release.cluster;");
}

// Launcher-owned task-end page release (Q3 = linear's proven combination,
// adapted to a 10-of-14-page op): the codegen loader page prefix has already
// arrived every page this task does NOT use, so the blanket here must be
// task_uses_page-gated or the 4 unused pages would be arrived twice (parity
// double-flip => next occupant deadlocks). Pairs with
// auto_consumer_finish=false in the registration. Runs on EVERY path
// (normal + inactive bail).
__device__ __forceinline__ void ffngg_launcher_release_used_pages(
    mirage::runtime::TaskDesc const *task_desc,
    mirage::runtime_v2::RuntimeSMEM *rt,
    int lane_id) {
  if (lane_id < mirage::runtime::MAX_SMEM_PAGES_PER_TASK &&
      mirage::runtime_v2::task_uses_page(task_desc, lane_id)) {
    mirage::runtime_v2::runtime_finish_page(rt, lane_id, 1);
  }
}

// ════════════════════════════════════════════════════════════════════════════
// W13 pipe (routed ALWAYS_ACTIVE=0 / shared gate_up ALWAYS_ACTIVE=1).
//   inputs : [0] meta i32[24]   [1] a_fp8 u8[HIDDEN]   [2] a_scale f32[56]
//            [3] w u8[E,N,HIDDEN] (routed E=128,N=1024) or u8[N,HIDDEN]
//                (shared N=512)
//            [4] w_scale f32[E,N/128,56] (routed) or f32[N/128,56] (shared)
//   outputs: [0] y f32[MAX_ACTIVE,N] (routed y13) or f32[N] (shared sg)
//   tile identity: tile_idx = task_offset; routed slot = tile_idx/(N/128),
//   n_tile = tile_idx%(N/128) (slot-major); shared n_tile = tile_idx.
// ════════════════════════════════════════════════════════════════════════════

// ── Loader (W4): protocol skeleton. Real body: prefetch tensormap ->
// re-inits -> dep-wait BEFORE meta (routed; shared starts W TMAs first,
// dep inline before the first B cp.async) -> zero-fill B padding rows once
// -> per-stage {wait mma[s]; splat SFA/SFB; W TMA + expect_tx(16384);
// B cp.async row 0; cpasync_wait<0>; fence.proxy.async.shared::cta;
// release-arrive B_sf[s]}.
template <int ALWAYS_ACTIVE>
__device__ __noinline__ void ffn_w13_pipe_loader_task(
    mirage::runtime::TaskDesc const *task_desc,
    mirage::runtime_v2::RuntimeSMEM *runtime_smem,
    mirage::runtime::RuntimeConfig const &runtime_config,
    CUtensorMap const *w_tmap_ptr,
    int const *meta,
    uint8_t const *a_fp8,
    float const *a_scale,
    float const *w_scale,
    int N,
    int tile_idx,
    int instruction_index,
    int iter_num,
    int dyn_sem_base) {
  // COOPERATIVE-LOADER restructure (round-3 primary lever, Class A): the 31
  // non-elected lanes stay ALIVE (no top-level early-return) so the per-K-stage
  // SFA/SFB splat fans out across all 32 loader lanes. Only mbarrier.init /
  // TMA-issue / mbarrier-arrive stay single-elected-lane; the splat + dep-wait
  // go warp-wide. Arrival COUNTS for W_tma/B_sf/mma are UNCHANGED (auditor
  // ledger: W_tma=1 elected, B_sf=1 elected, mma waited — the wait is a
  // non-consuming 32-lane poll now, no count/phase change).
  int const lane = threadIdx.x & 31;
  bool const elected = ffngg_elect_sync();
  // Single-lane prologue: tensormap prefetch + stale-arrival re-inits
  // (mbarrier.init is a single-writer op). The __syncwarp publishes the fresh
  // mbars to the other 31 lanes before ANY lane waits on mma_base below.
  if (elected) {
    asm volatile("prefetch.tensormap [%0];" ::"l"(w_tmap_ptr));
    ffngg_loader_reinit_mbars(dyn_sem_base);
  }
  __syncwarp();
  // Dep-wait BEFORE reading meta: ALL 32 lanes confirm the dep independently
  // (wait_task_dependency is a pure acquire-poll, no arrive —
  // runtime_v2.cuh:586 "safe to call from any thread"); each lane's acquire
  // orders its own meta read below (routed weight TMA coords need the meta
  // expert id, spec §4.3).
  mirage::runtime_v2::wait_task_dependency(runtime_config, task_desc, iter_num);
  int const nt_per_slot = N / BLOCK_M;
  int const n_tile = ALWAYS_ACTIVE ? tile_idx : tile_idx % nt_per_slot;
  int e = 0;
  if (!ALWAYS_ACTIVE) {
    int const slot = tile_idx / nt_per_slot;
    if (slot >= ffngg_meta_active(meta)) {
      // Inactive slot: bail with NO handshake (no TMA was issued, so no mbar
      // is left pending). Pages are handled by the codegen loader prefix
      // (unused) + the launcher blanket (used).
      return;
    }
    e = ffngg_meta_expert(meta, slot);
  }
  int const n0 = n_tile * BLOCK_M;
  int const y_row = e * N + n0; // GMEM_ROW flatten; shared E=1 => y_row = n0
  // Per-128x128-block weight scale row (routed: [E,N/GRP,KG1]; shared:
  // [N/GRP,KG1]) — one constant per K-stage (Q2: exact pow2 => UE8M0 splat).
  float const *wsc =
      w_scale +
      (size_t)(ALWAYS_ACTIVE ? n_tile : (e * (N / F::GRP) + n_tile)) * F::KG1;

  extern __shared__ __align__(1024) char smem_ptr[];
  int const smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));
  int const bsf_reg = task_desc->smem_region_offset(REGION_BSF);
  int const bsf_smem = smem + bsf_reg;
  int const W_tma_base = dyn_sem_base + SEM_W_TMA_BASE * 8;
  int const B_sf_base = dyn_sem_base + SEM_B_SF_BASE * 8;
  int const mma_base = dyn_sem_base + SEM_MMA_BASE * 8;

  // Zero-fill B padding rows 1..15 of every ring slice ONCE (cp.async only ever
  // rewrites row 0, so these stay zero across the ring; zeros x any UE8M0 scale
  // = 0 => padded-column ACC garbage is never read, consumer uses col 0).
  // COOPERATIVE fan-out (Round-A lever, Class A, NCU-evidenced): this loop
  // previously ran with NO lane partition and NO `elected` guard, so all 32
  // lanes issued the SAME 8*120=960 st.shared identically (32x redundant
  // instruction issue to the SAME addresses -- harmless but wasteful; NCU
  // source-correlated pcsamp showed this line mio_throttle+wait dominated,
  // i.e. issue-rate-throttled from the redundant instruction count, not
  // latency-bound). Partition `i` across lanes exactly like the proven SFA/SFB
  // splat (v009): each lane writes disjoint elements to the SAME addresses as
  // before, so the initialized region and its final contents are unchanged.
  for (int s = 0; s < STAGES; s++) {
    uint4 *pad = reinterpret_cast<uint4 *>(
        smem_ptr + bsf_reg + s * BSF_STAGE_STRIDE + BSF_OFF_B + BK);
    for (int i = lane; i < (MMA_N - 1) * BK / 16; i += WARP_SIZE) {
      pad[i] = make_uint4(0, 0, 0, 0);
    }
  }
  // Explicit local convergence: makes the now-partitioned writes visible
  // warp-wide before the first K-stage's `elected`-lane fence/B_sf-arrive
  // publishes them (the existing __syncwarp() inside the K-stage loop, right
  // before that fence, would already cover this via full-warp reconvergence,
  // but an explicit sync immediately after the cooperative write keeps the
  // dependency local and auditable rather than relying on a distant one).
  __syncwarp();

  // K-stage pipeline (KG1=56 stages, one segment; ring depth STAGES).
  int tma_stage = 0;
  int mma_phase = 1;
  // SOFTWARE-PIPELINE (Round-A Lever A2, Class A, NCU-evidenced, Codex-
  // reviewed 019f68db): pending_stage tracks the ring slot of an activation
  // cp.async ISSUED but not yet DRAINED (-1 = none pending). See the
  // drain-before-issue comment below for why this is safe.
  int pending_stage = -1;
  for (int k = 0; k < F::KG1; k++) {
    // Refill gate: launcher signals stage `tma_stage`'s MMA (UMMA + UTCCP) done
    // reading its W/BSF — so this slice is safe to overwrite (v1 ab_empty).
    ffngg_mbar_wait(mma_base + tma_stage * 8, mma_phase);

    // Cooperative SFA/SFB splat (round-3 PRIMARY lever): the UE8M0 byte of
    // wsc[k]/a_scale[k] is a SINGLE broadcast constant per K-stage (exact pow2
    // f32 -> UE8M0 = f32 exponent field, packed 4x; "transpose of a constant is
    // the constant" — v1 warp-6 collapses), so all 32 loader lanes fan out the
    // 2x128 IDENTICAL st.shared (4 SFA + 4 SFB per lane) instead of 256 serial
    // stores on ONE lane. No cross-lane data dependency => no warp-transpose
    // (v1's fp8_utccp_warp_transpose is only for its per-row gather; ours is a
    // constant). All lanes read wsc[k]/a_scale[k] as a broadcast GMEM load.
    uint32_t const sfa_b = (__float_as_uint(wsc[k]) >> 23) & 0xFFu;
    uint32_t const sfb_b = (__float_as_uint(a_scale[k]) >> 23) & 0xFFu;
    uint32_t const sfa_p = sfa_b | (sfa_b << 8) | (sfa_b << 16) | (sfa_b << 24);
    uint32_t const sfb_p = sfb_b | (sfb_b << 8) | (sfb_b << 16) | (sfb_b << 24);
    uint32_t *sfa_ptr = reinterpret_cast<uint32_t *>(
        smem_ptr + bsf_reg + tma_stage * BSF_STAGE_STRIDE + BSF_OFF_SFA);
    uint32_t *sfb_ptr = reinterpret_cast<uint32_t *>(
        smem_ptr + bsf_reg + tma_stage * BSF_STAGE_STRIDE + BSF_OFF_SFB);
#pragma unroll
    for (int i = lane; i < 128; i += WARP_SIZE) {
      sfa_ptr[i] = sfa_p;
      sfb_ptr[i] = sfb_p;
    }
    // Converge the warp so the elected lane's B_sf release-arrive publishes ALL
    // 32 lanes' splat writes (v1 proven ordering: cooperative st.shared ->
    // __syncwarp -> fence.proxy.async -> elect-one release; the __syncwarp
    // makes the other lanes' writes happen-before the elected lane's release,
    // so the launcher's B_sf acquire observes them). Also fences the elected
    // lane's TMA /cp.async issue behind splat completion. OUTSIDE the elected
    // branch (never a __syncwarp under divergent control flow).
    __syncwarp();

    // Single-elected-lane issuance (arrival COUNTS UNCHANGED vs v008): W TMA +
    // expect_tx(16384) [W_tma count=1], B cp.async row 0, async-proxy publish,
    // B_sf release-arrive [count=1]. The other 31 lanes fall straight to the
    // next iteration's mma-wait + splat and rendezvous at the next __syncwarp
    // (skew bounded to <=1 K-stage; each next-slice splat is itself mma-gated).
    if (elected) {
      // Weight tile TMA: coords {x = K-elt offset, y = weight row}, 16 KB tile.
      // Issued FIRST (async/mbarrier-tracked, non-blocking) so it can be in
      // flight while the PREVIOUS stage's activation drain (below) completes.
      int const W_smem =
          smem + task_desc->smem_region_offset(REGION_W_0 + tma_stage);
      ffngg_tma_2d_load(
          W_smem, w_tmap_ptr, k * BK, y_row, W_tma_base + tma_stage * 8);
      ffngg_mbar_arrive_expect_tx(W_tma_base + tma_stage * 8, W_BYTES);

      // SOFTWARE-PIPELINE (Round-A Lever A2, Codex-reviewed 019f68db): drain
      // the PREVIOUS stage's activation cp.async (if any) AFTER issuing THIS
      // stage's weight TMA, instead of a fully-serial per-stage
      // issue-drain-fence-arrive-then-next-TMA chain. NCU source-correlated
      // pcsamp showed this drain (wait_group+fence) as the #1 hotspot in this
      // loader (~33% of its own stalls, ~90% long_scoreboard) and the
      // matching launcher-side W_tma wait as ITS #1 hotspot -- i.e. the
      // loader's per-stage serial drain was rate-limiting how fast it could
      // issue the next stage's weight TMA, starving the launcher.
      // Drain-before-issue (never the reverse) keeps at most ONE cp.async
      // group outstanding at any time: cp.async.wait_group 0 counts TOTAL
      // outstanding groups, not a specific one, so issuing stage k's B before
      // draining stage k-1's B would make wait_group 0 wait for BOTH and
      // break the 1-stage-deferred invariant. Internal order (wait_group ->
      // fence -> arrive) UNCHANGED (frozen: fence AFTER cpasync_wait<0>,
      // BEFORE the release-arrive) -- only WHICH stage's B_sf this fires for
      // (the pending one, not the current one) and WHEN relative to the next
      // stage's W-TMA issue moved. No aliasing risk: each ring slot has its
      // OWN smem slice (indexed by tma_stage), and reuse of a slot is still
      // gated by the SAME mma_base wait above (untouched) -- STAGES=8 gives
      // ample margin for a 1-stage software-pipeline depth.
      if (pending_stage >= 0) {
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        ffngg_mbar_arrive(B_sf_base + pending_stage * 8);
      }

      // Activation row 0 (the single token's K-slice) via cp.async (16B chunks;
      // row 0 swizzle is identity since offset < 128). Issued but NOT drained
      // here -- becomes `pending_stage`, drained at the top of the next
      // elected iteration (or after the loop, for the last stage).
      int const b0 = bsf_smem + tma_stage * BSF_STAGE_STRIDE + BSF_OFF_B;
      uint8_t const *asrc = a_fp8 + (size_t)k * BK;
#pragma unroll
      for (int c = 0; c < BK / 16; c++) {
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(b0 + c * 16),
            "l"(asrc + c * 16));
      }
      asm volatile("cp.async.commit_group;\n" ::: "memory");
      pending_stage = tma_stage;
    }

    tma_stage = (tma_stage + 1) % STAGES;
    if (tma_stage == 0) {
      mma_phase ^= 1;
    }
  }
  // Flush the pipeline: drain the LAST stage's activation cp.async (the loop
  // always exits with exactly one stage pending, since KG1 > 0).
  if (elected && pending_stage >= 0) {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    ffngg_mbar_arrive(B_sf_base + pending_stage * 8);
  }
  (void)instruction_index;
}

// ── Launcher (W5): protocol skeleton + page-release owner. Real body:
// tcgen05.alloc 64 cols -> cache taddr -> lane-0 re-inits + tmem_ready ->
// per segment/stage UTCCP + 4x block-scaled UMMA + commits -> blanket
// release -> wait consumer_done -> dealloc(cached taddr).
template <int ALWAYS_ACTIVE>
__device__ __noinline__ void
    ffn_w13_pipe_launcher_task(mirage::runtime::TaskDesc const *task_desc,
                               mirage::runtime_v2::RuntimeSMEM *runtime_smem,
                               int const *meta,
                               int N,
                               int tile_idx,
                               int dyn_sem_base) {
  int const lane_id = threadIdx.x & 31;
  // Re-init mainloop+epilogue FIRST (also covers the inactive bail; Codex 4b).
  if (lane_id == 0) {
    ffngg_launcher_reinit_mbars(dyn_sem_base);
  }
  if (!ALWAYS_ACTIVE) {
    int const slot = tile_idx / (N / BLOCK_M);
    if (slot >= ffngg_meta_active(meta)) {
      // Inactive: skip alloc/MMA, no cross-role wait; still release the used
      // pages exactly once (bounds-fail rule).
      ffngg_launcher_release_used_pages(task_desc, runtime_smem, lane_id);
      return;
    }
  }

  extern __shared__ __align__(1024) char smem_ptr[];
  int const smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));
  int const bsf_reg = task_desc->smem_region_offset(REGION_BSF);
  int const scratch_addr = smem + bsf_reg + BSF_OFF_TADDR;
  int const W_tma_base = dyn_sem_base + SEM_W_TMA_BASE * 8;
  int const B_sf_base = dyn_sem_base + SEM_B_SF_BASE * 8;
  int const mma_base = dyn_sem_base + SEM_MMA_BASE * 8;
  int const mainloop_base = dyn_sem_base + SEM_MAINLOOP_BASE * 8;
  int const epilogue_base = dyn_sem_base + SEM_EPILOGUE_BASE * 8;
  int const tmem_ready_addr = dyn_sem_base + SEM_TMEM_READY * 8;
  int const consumer_done_addr = dyn_sem_base + SEM_CONSUMER_DONE * 8;

  // tcgen05.alloc 64 cols into the BSF taddr scratch; ALL 32 lanes participate
  // (sync.aligned); cache taddr in a register (scratch page may free before
  // dealloc; alloc/dealloc SAME warp — MLA co-residency invariant).
  asm volatile(
      "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(
          scratch_addr),
      "r"(64));
  int const taddr =
      *reinterpret_cast<int *>(smem_ptr + bsf_reg + BSF_OFF_TADDR);
  if (lane_id == 0) {
    ffngg_mbar_arrive(tmem_ready_addr); // publish taddr to consumers
  }

  if (ffngg_elect_sync()) {
    // Compile-time block-scaled instruction descriptor (v1 :1042-1050).
    auto instr_desc =
        cute::UMMA::make_instr_desc_block_scaled<cutlass::float_e4m3_t,
                                                 cutlass::float_e4m3_t,
                                                 float,
                                                 cutlass::float_ue8m0_t,
                                                 BLOCK_M,
                                                 MMA_N,
                                                 cute::UMMA::Major::K,
                                                 cute::UMMA::Major::K>();
    auto sf_desc = kernel::sm100::make_sf_desc(nullptr);
    using UTCCP_t = cute::SM100::TMEM::UTCCP::SM100_UTCCP_4x32dp128bit_1cta;
    uint32_t const acc_col = static_cast<uint32_t>(taddr); // ACC stage 0
    uint32_t const sfa_tmem =
        static_cast<uint32_t>(taddr) + MMA_N * ACC_STAGES; // +32
    uint32_t const sfb_tmem = sfa_tmem + 4;                // +36

    // One segment (56 K-stages, single ACC tile): wait ACC-stage-0 drained.
    ffngg_mbar_wait(epilogue_base + 0 * 8, 1);

    int tma_stage = 0;
    int tma_phase = 0;
    for (int k = 0; k < F::KG1; k++) {
      ffngg_mbar_wait(W_tma_base + tma_stage * 8, tma_phase);
      ffngg_mbar_wait(B_sf_base + tma_stage * 8, tma_phase);
      asm volatile("tcgen05.fence::after_thread_sync;");

      // UTCCP: splatted UE8M0 scales smem -> TMEM cols (SFA[+32], SFB[+36]).
      kernel::sm100::replace_smem_desc_addr(
          sf_desc,
          smem_ptr + bsf_reg + tma_stage * BSF_STAGE_STRIDE + BSF_OFF_SFA);
      UTCCP_t::copy(static_cast<uint64_t>(sf_desc), sfa_tmem);
      kernel::sm100::replace_smem_desc_addr(
          sf_desc,
          smem_ptr + bsf_reg + tma_stage * BSF_STAGE_STRIDE + BSF_OFF_SFB);
      UTCCP_t::copy(static_cast<uint64_t>(sf_desc), sfb_tmem);

      // 4x block-scaled UMMA (bK/UMMA_K = 128/32). enable_d=0 (overwrite) only
      // on the very first sub-tile of the first K-stage; accumulate otherwise.
      auto a_desc =
          kernel::sm100::make_umma_desc<cute::UMMA::Major::K, BLOCK_M, BK, 128>(
              reinterpret_cast<cutlass::float_e4m3_t *>(
                  smem_ptr +
                  task_desc->smem_region_offset(REGION_W_0 + tma_stage)),
              0,
              0);
      auto b_desc =
          kernel::sm100::make_umma_desc<cute::UMMA::Major::K, MMA_N, BK, 128>(
              reinterpret_cast<cutlass::float_e4m3_t *>(
                  smem_ptr + bsf_reg + tma_stage * BSF_STAGE_STRIDE +
                  BSF_OFF_B),
              0,
              0);
      uint32_t const a_lo = a_desc.lo;
      uint32_t const b_lo = b_desc.lo;
#pragma unroll
      for (int ks = 0; ks < BK / 32; ks++) {
        auto rid = kernel::sm100::make_runtime_instr_desc_with_sf_id(
            instr_desc, ks, ks);
        a_desc.lo = kernel::sm100::advance_umma_desc_lo<cute::UMMA::Major::K,
                                                        BLOCK_M,
                                                        128,
                                                        cutlass::float_e4m3_t>(
            a_lo, 0, ks * 32);
        b_desc.lo = kernel::sm100::advance_umma_desc_lo<cute::UMMA::Major::K,
                                                        MMA_N,
                                                        128,
                                                        cutlass::float_e4m3_t>(
            b_lo, 0, ks * 32);
        kernel::sm100::SM100_MMA_MXF8F6F4_SS::fma(static_cast<uint64_t>(a_desc),
                                                  static_cast<uint64_t>(b_desc),
                                                  acc_col,
                                                  (k == 0 && ks == 0) ? 0u : 1u,
                                                  rid,
                                                  sfa_tmem,
                                                  sfb_tmem);
      }
      ffngg_tcgen05_commit(mma_base + tma_stage * 8); // stage refill-ok
      tma_stage = (tma_stage + 1) % STAGES;
      if (tma_stage == 0) {
        tma_phase ^= 1;
      }
    }
    ffngg_tcgen05_commit(mainloop_base + 0 * 8); // ACC[0] full -> consumer
  }

  // Task-end page release (Q3 ownership; launcher blanket over used pages).
  ffngg_launcher_release_used_pages(task_desc, runtime_smem, lane_id);
  __syncwarp();
  if (lane_id == 0) {
    ffngg_mbar_wait(consumer_done_addr, 0); // TMEM no longer read
  }
  __syncwarp();
  // Dealloc — SAME warp as alloc, sync.aligned, all 32 lanes, cached taddr.
  asm volatile(
      "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr),
      "r"(64));
}

// ── Consumer (W0-3, 128 threads; thread t owns output row n0 + t) ──
// NAIVE stub math (correct, slow): per-128-block scales applied per K-block;
// exact e4m3 decode; f32 accumulate. Real body: wait tmem_ready -> per
// segment wait mainloop[acc] -> tcgen05.ld x16 -> col 0 -> store -> arrive
// epilogue[acc] -> all-128 arrive consumer_done.
template <int ALWAYS_ACTIVE>
__device__ __noinline__ void
    ffn_w13_pipe_consumer_task(mirage::runtime::TaskDesc const *task_desc,
                               int const *meta,
                               uint8_t const *a_fp8,
                               float const *a_scale,
                               uint8_t const *w,
                               float const *w_scale,
                               float *y_out,
                               int N,
                               int tile_idx,
                               int dyn_sem_base) {
  int const nt_per_slot = N / BLOCK_M;
  int const slot = ALWAYS_ACTIVE ? 0 : tile_idx / nt_per_slot;
  int const n_tile = ALWAYS_ACTIVE ? tile_idx : tile_idx % nt_per_slot;
  if (!ALWAYS_ACTIVE) {
    if (slot >= ffngg_meta_active(meta)) {
      return; // inactive slot: y13 rows >= active_count are never read
    }
  }
  int const warp_id = threadIdx.x / WARP_SIZE; // 0..3
  int const lane_id = threadIdx.x & 31;

  extern __shared__ __align__(1024) char smem_ptr[];
  int const bsf_reg = task_desc->smem_region_offset(REGION_BSF);
  int const tmem_ready_addr = dyn_sem_base + SEM_TMEM_READY * 8;
  int const mainloop_base = dyn_sem_base + SEM_MAINLOOP_BASE * 8;
  int const epilogue_base = dyn_sem_base + SEM_EPILOGUE_BASE * 8;
  int const consumer_done_addr = dyn_sem_base + SEM_CONSUMER_DONE * 8;

  // Wait for launcher's taddr publish, then read it (acquire visibility).
  if (lane_id == 0) {
    ffngg_mbar_wait(tmem_ready_addr, 0);
  }
  __syncwarp();
  int const taddr =
      *reinterpret_cast<int *>(smem_ptr + bsf_reg + BSF_OFF_TADDR);

  // One segment: wait ACC[0] full, read [128 rows x 16 cols] from TMEM.
  ffngg_mbar_wait(mainloop_base + 0 * 8, 0);
  asm volatile("tcgen05.fence::after_thread_sync;");

  int const n_local = warp_id * WARP_SIZE + lane_id; // TMEM row = output N-col
  int const t_addr = (warp_id * WARP_SIZE << 16) + taddr; // ACC stage 0, col 0
  float f[16];
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32\n"
      "  {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
      : "=f"(f[0]),
        "=f"(f[1]),
        "=f"(f[2]),
        "=f"(f[3]),
        "=f"(f[4]),
        "=f"(f[5]),
        "=f"(f[6]),
        "=f"(f[7]),
        "=f"(f[8]),
        "=f"(f[9]),
        "=f"(f[10]),
        "=f"(f[11]),
        "=f"(f[12]),
        "=f"(f[13]),
        "=f"(f[14]),
        "=f"(f[15])
      : "r"(t_addr));
  asm volatile("tcgen05.wait::ld.sync.aligned;");

  // Only col 0 is the real token (MMA_N padding cols 1..15 came from zero B).
  int const n0 = n_tile * BLOCK_M;
  y_out[(size_t)slot * N + n0 + n_local] = f[0];

  // Drain ACC[0] then release TMEM. Every consumer thread arrives both.
  ffngg_mbar_arrive(epilogue_base + 0 * 8);
  ffngg_mbar_arrive(consumer_done_addr);

  (void)a_fp8;
  (void)a_scale;
  (void)w;
  (void)w_scale;
  (void)N;
}

// ── Storer (W6): EMPTY protocol under the Q3 (linear-verbatim) ownership —
// the launcher blanket owns page release. The spec §4 per-stage release
// engine is the documented ALTERNATIVE owner: a rewrite may TRANSFER the
// release here (.cuh-only), riding mma[s] parity with
// last_use[s] = (total_iters-1-s)/STAGES + 1, total_iters = 56 (W13) or
// 4*active+2 (W2, from meta — hence the meta arg + the registration-emitted
// dep prefix that orders this body after the dep).
template <int ALWAYS_ACTIVE>
__device__ __noinline__ void
    ffn_w13_pipe_storer_task(mirage::runtime::TaskDesc const *task_desc,
                             mirage::runtime_v2::RuntimeSMEM *runtime_smem,
                             int const *meta,
                             int N,
                             int tile_idx,
                             int dyn_sem_base) {
  if (!ffngg_elect_sync()) {
    return;
  }
  // NAIVE stub: no-op (no release here — exactly-once ownership is the
  // launcher's; see header).
  (void)task_desc;
  (void)runtime_smem;
  (void)meta;
  (void)N;
  (void)tile_idx;
  (void)dyn_sem_base;
}

// ════════════════════════════════════════════════════════════════════════════
// W2 pipe (never inactive: the shared-down segment is live even at
// active_count == 0).
//   inputs : [0] i_fp8 u8[MAX_ACTIVE,512]  [1] si_fp8 u8[256]
//            [2] meta i32[24]  [3] i_scale f32[MAX_ACTIVE,4]
//            [4] si_scale f32[2]
//            [5] w2 u8[E,7168,512]  [6] w2_scale f32[E,56,4]
//            [7] wdn u8[7168,256]   [8] wdn_scale f32[56,2]
//   outputs: [0] out bf16[1,7168]
//   tile identity: tile_idx = task_offset = n_tile in 0..55.
//   K-iteration space (real body): segments = [slot 0..active-1] (4 K-stages
//   each) + [shared-down] (2 K-stages, B = si_fp8, SFB = si_scale, ew = 1.0)
//   -> 4*active+2 K-stages, active+1 ACC segments over the 2 ACC stages.
// ════════════════════════════════════════════════════════════════════════════

__device__ __noinline__ void ffn_w2_pipe_loader_task(
    mirage::runtime::TaskDesc const *task_desc,
    mirage::runtime_v2::RuntimeSMEM *runtime_smem,
    mirage::runtime::RuntimeConfig const &runtime_config,
    CUtensorMap const *w2_tmap_ptr,
    CUtensorMap const *wdn_tmap_ptr,
    int const *meta,
    uint8_t const *i_fp8,
    float const *i_scale,
    uint8_t const *si_fp8,
    float const *si_scale,
    int tile_idx,
    int instruction_index,
    int iter_num,
    int dyn_sem_base) {
  // COOPERATIVE-LOADER (mirror W13 v009): 31 non-elected lanes stay ALIVE so
  // the per-K-stage SFA/SFB splat fans out across all 32 lanes; only TMA-issue
  // / cp.async / mbar-arrive stay single-elected-lane.
  int const lane = threadIdx.x & 31;
  bool const elected = ffngg_elect_sync();
  if (elected) {
    asm volatile("prefetch.tensormap [%0];" ::"l"(w2_tmap_ptr));
    asm volatile("prefetch.tensormap [%0];" ::"l"(wdn_tmap_ptr));
    ffngg_loader_reinit_mbars(dyn_sem_base);
  }
  __syncwarp();
  // Routed segments' weight TMA coords need meta (expert id) -> dep-wait FIRST
  // on ALL 32 lanes (wait_task_dependency is a pure acquire-poll, no arrive).
  mirage::runtime_v2::wait_task_dependency(runtime_config, task_desc, iter_num);
  // Weight scales (SFA source) are NOT named loader args (registration passes
  // them as inputs 6/8); read them via task_desc (all input_ptrs are visible).
  float const *w2_scale = static_cast<float const *>(task_desc->input_ptrs[6]);
  float const *wdn_scale = static_cast<float const *>(task_desc->input_ptrs[8]);
  int const active = ffngg_meta_active(meta);
  int const n0 = tile_idx * BLOCK_M; // this tile's 128-row block (== block idx)

  extern __shared__ __align__(1024) char smem_ptr[];
  int const smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));
  int const bsf_reg = task_desc->smem_region_offset(REGION_BSF);
  int const bsf_smem = smem + bsf_reg;
  int const W_tma_base = dyn_sem_base + SEM_W_TMA_BASE * 8;
  int const B_sf_base = dyn_sem_base + SEM_B_SF_BASE * 8;
  int const mma_base = dyn_sem_base + SEM_MMA_BASE * 8;

  // Zero-fill B padding rows 1..15 of every ring slice ONCE (cp.async only ever
  // rewrites row 0, so these stay zero across the ring; zeros x any UE8M0 scale
  // = 0 => padded-column ACC garbage is never read, consumer uses col 0).
  // COOPERATIVE fan-out (Round-A lever, Class A, NCU-evidenced): this loop
  // previously ran with NO lane partition and NO `elected` guard, so all 32
  // lanes issued the SAME 8*120=960 st.shared identically (32x redundant
  // instruction issue to the SAME addresses -- harmless but wasteful; NCU
  // source-correlated pcsamp showed this line mio_throttle+wait dominated,
  // i.e. issue-rate-throttled from the redundant instruction count, not
  // latency-bound). Partition `i` across lanes exactly like the proven SFA/SFB
  // splat (v009): each lane writes disjoint elements to the SAME addresses as
  // before, so the initialized region and its final contents are unchanged.
  for (int s = 0; s < STAGES; s++) {
    uint4 *pad = reinterpret_cast<uint4 *>(
        smem_ptr + bsf_reg + s * BSF_STAGE_STRIDE + BSF_OFF_B + BK);
    for (int i = lane; i < (MMA_N - 1) * BK / 16; i += WARP_SIZE) {
      pad[i] = make_uint4(0, 0, 0, 0);
    }
  }
  // Explicit local convergence: makes the now-partitioned writes visible
  // warp-wide before the first K-stage's `elected`-lane fence/B_sf-arrive
  // publishes them (the existing __syncwarp() inside the K-stage loop, right
  // before that fence, would already cover this via full-warp reconvergence,
  // but an explicit sync immediately after the cooperative write keeps the
  // dependency local and auditable rather than relying on a distant one).
  __syncwarp();

  // ONE continuous K-stage ring across ALL segments (do NOT reset per segment —
  // exactly W13's single loop, extended over active+1 segments). Segment order
  // = slots ascending (0..active-1) then the shared-down segment (Q6 pin).
  int tma_stage = 0;
  int mma_phase = 1;
  // SOFTWARE-PIPELINE (Round-A Lever A2, mirrors W13; Codex-reviewed
  // 019f68db): pending_stage carries the same 1-stage-deferred activation
  // drain ACROSS segment boundaries (the K-ring is already continuous across
  // segments, so this is a direct extension, not a new concept). -1 = none
  // pending.
  int pending_stage = -1;
  int const total_segs = active + 1;
  for (int seg = 0; seg < total_segs; seg++) {
    bool const is_shared = (seg == active);
    CUtensorMap const *tmap;
    int y_row;
    float const *wsc;        // per-K-stage weight (SFA) scale
    float const *bsc;        // per-K-stage activation (SFB) scale
    uint8_t const *asrc_seg; // activation row-0 base for this segment
    int nk;
    if (!is_shared) {
      int const e = ffngg_meta_expert(meta, seg);
      tmap = w2_tmap_ptr;
      y_row = e * F::W2_N + n0; // GMEM_ROW flatten [E*W2_N, W2_K]
      wsc = w2_scale + ((size_t)e * F::NB2 + tile_idx) * F::KG2;
      bsc = i_scale + (size_t)seg * F::KG2;
      asrc_seg = i_fp8 + (size_t)seg * F::W2_K;
      nk = F::KG2; // 4 K-stages (K=512)
    } else {
      tmap = wdn_tmap_ptr;
      y_row = n0; // shared: E=1 => no e term
      wsc = wdn_scale + (size_t)tile_idx * F::KG_SHDN;
      bsc = si_scale;
      asrc_seg = si_fp8;
      nk = F::KG_SHDN; // 2 K-stages (K=256)
    }
    for (int k = 0; k < nk; k++) {
      // Refill gate: launcher signalled stage tma_stage's MMA done reading it.
      ffngg_mbar_wait(mma_base + tma_stage * 8, mma_phase);

      // Cooperative SFA/SFB splat (all 32 lanes; scale = single broadcast
      // UE8M0 constant per K-stage — exact pow2 f32 exponent field, packed 4x).
      uint32_t const sfa_b = (__float_as_uint(wsc[k]) >> 23) & 0xFFu;
      uint32_t const sfb_b = (__float_as_uint(bsc[k]) >> 23) & 0xFFu;
      uint32_t const sfa_p =
          sfa_b | (sfa_b << 8) | (sfa_b << 16) | (sfa_b << 24);
      uint32_t const sfb_p =
          sfb_b | (sfb_b << 8) | (sfb_b << 16) | (sfb_b << 24);
      uint32_t *sfa_ptr = reinterpret_cast<uint32_t *>(
          smem_ptr + bsf_reg + tma_stage * BSF_STAGE_STRIDE + BSF_OFF_SFA);
      uint32_t *sfb_ptr = reinterpret_cast<uint32_t *>(
          smem_ptr + bsf_reg + tma_stage * BSF_STAGE_STRIDE + BSF_OFF_SFB);
#pragma unroll
      for (int i = lane; i < 128; i += WARP_SIZE) {
        sfa_ptr[i] = sfa_p;
        sfb_ptr[i] = sfb_p;
      }
      // Converge so the elected lane's B_sf release-arrive publishes ALL 32
      // lanes' splat writes (release/acquire cumulativity + __syncwarp edge).
      __syncwarp();

      if (elected) {
        // Weight tile TMA: coords {x = K-elt offset, y = weight row}, 16 KB.
        // Issued FIRST (async/mbarrier-tracked, non-blocking) so it can be in
        // flight while the PREVIOUS stage's activation drain (below)
        // completes.
        int const W_smem =
            smem + task_desc->smem_region_offset(REGION_W_0 + tma_stage);
        ffngg_tma_2d_load(
            W_smem, tmap, k * BK, y_row, W_tma_base + tma_stage * 8);
        ffngg_mbar_arrive_expect_tx(W_tma_base + tma_stage * 8, W_BYTES);

        // SOFTWARE-PIPELINE (Round-A Lever A2, mirrors W13 exactly; see that
        // copy for the full rationale/NCU evidence + Codex review 019f68db).
        // Drain-before-issue keeps at most ONE cp.async group outstanding
        // (wait_group 0 counts TOTAL outstanding groups, not a specific
        // one). Carries continuously across segment boundaries -- the K-ring
        // (tma_stage/mma_phase) is already continuous across segments, so no
        // extra per-segment bookkeeping is needed.
        if (pending_stage >= 0) {
          asm volatile("cp.async.wait_group 0;\n" ::: "memory");
          asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
          ffngg_mbar_arrive(B_sf_base + pending_stage * 8);
        }

        // Activation row 0 (the single token's K-slice) via cp.async (16B).
        // Issued but NOT drained here -- becomes `pending_stage`.
        int const b0 = bsf_smem + tma_stage * BSF_STAGE_STRIDE + BSF_OFF_B;
        uint8_t const *asrc = asrc_seg + (size_t)k * BK;
#pragma unroll
        for (int c = 0; c < BK / 16; c++) {
          asm volatile(
              "cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(b0 + c * 16),
              "l"(asrc + c * 16));
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        pending_stage = tma_stage;
      }

      tma_stage = (tma_stage + 1) % STAGES;
      if (tma_stage == 0) {
        mma_phase ^= 1;
      }
    }
  }
  // Flush the pipeline: drain the LAST stage's activation cp.async (always
  // exits with exactly one stage pending -- the shared segment alone
  // contributes KG_SHDN=2 > 0 stages even at active=0).
  if (elected && pending_stage >= 0) {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    ffngg_mbar_arrive(B_sf_base + pending_stage * 8);
  }
  (void)instruction_index;
}

__device__ __noinline__ void
    ffn_w2_pipe_launcher_task(mirage::runtime::TaskDesc const *task_desc,
                              mirage::runtime_v2::RuntimeSMEM *runtime_smem,
                              int const *meta,
                              int tile_idx,
                              int dyn_sem_base) {
  int const lane_id = threadIdx.x & 31;
  // Re-init mainloop+epilogue+consumer_done FIRST. W2 is NEVER inactive (the
  // shared-down segment is always live, even at active_count == 0) => no bail.
  if (lane_id == 0) {
    ffngg_launcher_reinit_mbars(dyn_sem_base);
  }

  extern __shared__ __align__(1024) char smem_ptr[];
  int const smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));
  int const bsf_reg = task_desc->smem_region_offset(REGION_BSF);
  int const scratch_addr = smem + bsf_reg + BSF_OFF_TADDR;
  int const W_tma_base = dyn_sem_base + SEM_W_TMA_BASE * 8;
  int const B_sf_base = dyn_sem_base + SEM_B_SF_BASE * 8;
  int const mma_base = dyn_sem_base + SEM_MMA_BASE * 8;
  int const mainloop_base = dyn_sem_base + SEM_MAINLOOP_BASE * 8;
  int const epilogue_base = dyn_sem_base + SEM_EPILOGUE_BASE * 8;
  int const tmem_ready_addr = dyn_sem_base + SEM_TMEM_READY * 8;
  int const consumer_done_addr = dyn_sem_base + SEM_CONSUMER_DONE * 8;
  int const active = ffngg_meta_active(meta);

  // tcgen05.alloc 64 cols into the BSF taddr scratch; ALL 32 lanes participate
  // (sync.aligned); cache taddr in a register (scratch page may free before
  // dealloc; alloc/dealloc SAME warp — MLA co-residency invariant).
  asm volatile(
      "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(
          scratch_addr),
      "r"(64));
  int const taddr =
      *reinterpret_cast<int *>(smem_ptr + bsf_reg + BSF_OFF_TADDR);
  if (lane_id == 0) {
    ffngg_mbar_arrive(tmem_ready_addr); // publish taddr to consumers
  }

  if (ffngg_elect_sync()) {
    auto instr_desc =
        cute::UMMA::make_instr_desc_block_scaled<cutlass::float_e4m3_t,
                                                 cutlass::float_e4m3_t,
                                                 float,
                                                 cutlass::float_ue8m0_t,
                                                 BLOCK_M,
                                                 MMA_N,
                                                 cute::UMMA::Major::K,
                                                 cute::UMMA::Major::K>();
    auto sf_desc = kernel::sm100::make_sf_desc(nullptr);
    using UTCCP_t = cute::SM100::TMEM::UTCCP::SM100_UTCCP_4x32dp128bit_1cta;
    // SFA/SFB scratch cols are FIXED (rewritten every K-stage); only the ACC
    // column base moves with the segment's ACC stage.
    uint32_t const sfa_tmem =
        static_cast<uint32_t>(taddr) + MMA_N * ACC_STAGES; // +32
    uint32_t const sfb_tmem = sfa_tmem + 4;                // +36

    // Continuous K-stage ring (tma_stage/tma_phase) advances across ALL
    // segments; INDEPENDENT 2-stage ACC ring (acc = seg % ACC_STAGES) with a
    // per-stage epilogue wait-phase that starts at 1 and flips on each use.
    int tma_stage = 0;
    int tma_phase = 0;
    int epi_phase[ACC_STAGES] = {1, 1};
    int const total_segs = active + 1;
    for (int seg = 0; seg < total_segs; seg++) {
      int const acc = seg % ACC_STAGES;
      uint32_t const acc_col = static_cast<uint32_t>(taddr) + acc * MMA_N;
      bool const is_shared = (seg == active);
      int const nk = is_shared ? F::KG_SHDN : F::KG2;

      // Wait the prior occupant of ACC[acc] fully drained (fresh on 1st use).
      ffngg_mbar_wait(epilogue_base + acc * 8, epi_phase[acc]);
      epi_phase[acc] ^= 1;

      for (int k = 0; k < nk; k++) {
        ffngg_mbar_wait(W_tma_base + tma_stage * 8, tma_phase);
        ffngg_mbar_wait(B_sf_base + tma_stage * 8, tma_phase);
        asm volatile("tcgen05.fence::after_thread_sync;");

        // UTCCP: splatted UE8M0 scales smem -> TMEM cols (SFA[+32], SFB[+36]).
        kernel::sm100::replace_smem_desc_addr(
            sf_desc,
            smem_ptr + bsf_reg + tma_stage * BSF_STAGE_STRIDE + BSF_OFF_SFA);
        UTCCP_t::copy(static_cast<uint64_t>(sf_desc), sfa_tmem);
        kernel::sm100::replace_smem_desc_addr(
            sf_desc,
            smem_ptr + bsf_reg + tma_stage * BSF_STAGE_STRIDE + BSF_OFF_SFB);
        UTCCP_t::copy(static_cast<uint64_t>(sf_desc), sfb_tmem);

        // 4x block-scaled UMMA (bK/UMMA_K = 128/32). enable_d=0 (overwrite)
        // ONLY on the first sub-tile of the first K-stage of THIS segment —
        // each segment's ACC is a fresh, independent accumulation.
        auto a_desc = kernel::sm100::
            make_umma_desc<cute::UMMA::Major::K, BLOCK_M, BK, 128>(
                reinterpret_cast<cutlass::float_e4m3_t *>(
                    smem_ptr +
                    task_desc->smem_region_offset(REGION_W_0 + tma_stage)),
                0,
                0);
        auto b_desc =
            kernel::sm100::make_umma_desc<cute::UMMA::Major::K, MMA_N, BK, 128>(
                reinterpret_cast<cutlass::float_e4m3_t *>(
                    smem_ptr + bsf_reg + tma_stage * BSF_STAGE_STRIDE +
                    BSF_OFF_B),
                0,
                0);
        uint32_t const a_lo = a_desc.lo;
        uint32_t const b_lo = b_desc.lo;
#pragma unroll
        for (int ks = 0; ks < BK / 32; ks++) {
          auto rid = kernel::sm100::make_runtime_instr_desc_with_sf_id(
              instr_desc, ks, ks);
          a_desc.lo =
              kernel::sm100::advance_umma_desc_lo<cute::UMMA::Major::K,
                                                  BLOCK_M,
                                                  128,
                                                  cutlass::float_e4m3_t>(
                  a_lo, 0, ks * 32);
          b_desc.lo =
              kernel::sm100::advance_umma_desc_lo<cute::UMMA::Major::K,
                                                  MMA_N,
                                                  128,
                                                  cutlass::float_e4m3_t>(
                  b_lo, 0, ks * 32);
          kernel::sm100::SM100_MMA_MXF8F6F4_SS::fma(
              static_cast<uint64_t>(a_desc),
              static_cast<uint64_t>(b_desc),
              acc_col,
              (k == 0 && ks == 0) ? 0u : 1u,
              rid,
              sfa_tmem,
              sfb_tmem);
        }
        ffngg_tcgen05_commit(mma_base + tma_stage * 8); // stage refill-ok
        tma_stage = (tma_stage + 1) % STAGES;
        if (tma_stage == 0) {
          tma_phase ^= 1;
        }
      }
      ffngg_tcgen05_commit(mainloop_base +
                           acc * 8); // ACC[acc] full -> consumer
    }
  }

  // Task-end page release (Q3 ownership; launcher blanket over used pages).
  ffngg_launcher_release_used_pages(task_desc, runtime_smem, lane_id);
  __syncwarp();
  if (lane_id == 0) {
    ffngg_mbar_wait(consumer_done_addr, 0); // TMEM no longer read
  }
  __syncwarp();
  // Dealloc — SAME warp as alloc, sync.aligned, all 32 lanes, cached taddr.
  asm volatile(
      "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr),
      "r"(64));
  (void)tile_idx;
}

__device__ __noinline__ void
    ffn_w2_pipe_consumer_task(mirage::runtime::TaskDesc const *task_desc,
                              int const *meta,
                              uint8_t const *i_fp8,
                              float const *i_scale,
                              uint8_t const *si_fp8,
                              float const *si_scale,
                              uint8_t const *w2,
                              float const *w2_scale,
                              uint8_t const *wdn,
                              float const *wdn_scale,
                              __nv_bfloat16 *out,
                              int tile_idx,
                              int dyn_sem_base) {
  int const warp_id = threadIdx.x / WARP_SIZE; // 0..3
  int const lane_id = threadIdx.x & 31;
  int const active = ffngg_meta_active(meta);

  extern __shared__ __align__(1024) char smem_ptr[];
  int const bsf_reg = task_desc->smem_region_offset(REGION_BSF);
  int const tmem_ready_addr = dyn_sem_base + SEM_TMEM_READY * 8;
  int const mainloop_base = dyn_sem_base + SEM_MAINLOOP_BASE * 8;
  int const epilogue_base = dyn_sem_base + SEM_EPILOGUE_BASE * 8;
  int const consumer_done_addr = dyn_sem_base + SEM_CONSUMER_DONE * 8;

  // Wait launcher's taddr publish ONCE (per-warp lane 0), then read taddr.
  if (lane_id == 0) {
    ffngg_mbar_wait(tmem_ready_addr, 0);
  }
  __syncwarp();
  int const taddr =
      *reinterpret_cast<int *>(smem_ptr + bsf_reg + BSF_OFF_TADDR);
  int const n_local = warp_id * WARP_SIZE + lane_id; // 0..127 (TMEM row)

  // Cross-segment weighted sum in a register (Q6 determinism: fixed segment
  // order slots-ascending-then-shared, f32 accum, ONE bf16 store, no atomics).
  // 2-stage ACC ring: mainloop wait-phase per stage starts at 0 and flips on
  // each use. consumer_done is arrived exactly ONCE per thread AFTER the loop.
  int mainloop_phase[ACC_STAGES] = {0, 0};
  float acc_f = 0.f;
  int const total_segs = active + 1;
  for (int seg = 0; seg < total_segs; seg++) {
    int const acc = seg % ACC_STAGES;
    bool const is_shared = (seg == active);
    float const ew = is_shared ? 1.0f : ffngg_meta_weight(meta, seg);

    ffngg_mbar_wait(mainloop_base + acc * 8, mainloop_phase[acc]);
    mainloop_phase[acc] ^= 1;
    asm volatile("tcgen05.fence::after_thread_sync;");

    int const t_addr = (warp_id * WARP_SIZE << 16) + taddr + acc * MMA_N;
    float f[16];
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32\n"
        "  {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
        : "=f"(f[0]),
          "=f"(f[1]),
          "=f"(f[2]),
          "=f"(f[3]),
          "=f"(f[4]),
          "=f"(f[5]),
          "=f"(f[6]),
          "=f"(f[7]),
          "=f"(f[8]),
          "=f"(f[9]),
          "=f"(f[10]),
          "=f"(f[11]),
          "=f"(f[12]),
          "=f"(f[13]),
          "=f"(f[14]),
          "=f"(f[15])
        : "r"(t_addr));
    asm volatile("tcgen05.wait::ld.sync.aligned;");

    // col 0 = the real token; cols 1..15 = zero-B padding. Segment's ACC holds
    // W2[e]@i_s (routed, weight ew) or wdn@si (shared, ew=1.0).
    acc_f += ew * f[0];
    ffngg_mbar_arrive(epilogue_base + acc * 8); // ACC[acc] drained
  }

  // After ALL segments: single bf16 store, then arrive consumer_done ONCE
  // (NOT per-segment — over/under-arrival wedges the next slot occupant).
  int const n0 = tile_idx * BLOCK_M;
  out[(size_t)n0 + n_local] = __float2bfloat16_rn(acc_f);
  ffngg_mbar_arrive(consumer_done_addr);

  (void)i_fp8;
  (void)i_scale;
  (void)si_fp8;
  (void)si_scale;
  (void)w2;
  (void)w2_scale;
  (void)wdn;
  (void)wdn_scale;
}

__device__ __noinline__ void
    ffn_w2_pipe_storer_task(mirage::runtime::TaskDesc const *task_desc,
                            mirage::runtime_v2::RuntimeSMEM *runtime_smem,
                            int const *meta,
                            int tile_idx,
                            int dyn_sem_base) {
  if (!ffngg_elect_sync()) {
    return;
  }
  // NAIVE stub: no-op (see W13 storer note; W2's total_iters = 4*active+2
  // comes from meta, ordered by the registration-emitted dep prefix).
  (void)task_desc;
  (void)runtime_smem;
  (void)meta;
  (void)tile_idx;
  (void)dyn_sem_base;
}

} // namespace dsv3_ffn_gg_v2
} // namespace kernel
