/* Copyright 2026 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License").
 */

// MPK v3 — linear (Qwen3 decode), Channel-based.
//
// Design: synchronization and storage are separate primitives (channel.cuh).
//   * Channel  — mbarriers only (full/empty). Producer/Consumer cursors own
//     the stage index, the single source that keeps the four role functions
//     in sync. Carries no storage.
//   * SmemRing — per-stage SMEM offsets (+ optional page IDs for Phase-E
//     cross-task page release). A role indexes it at the cursor's stage:
//     `Wr.slot_addr(pW.st)`.
// Shape, SMEM/SEM ordinals, and PTX wrappers all come from the shared
// `kernel::linear` namespace (linear_spec.h + linear_device.cuh) — one source
// of truth, no dependency on linear_v2.
//
// W and A share one empty edge (mma_mbar): the launcher's single tcgen05.commit
// per K-iter frees both. So both channels point `empty` at mma_mbar; only pW
// waits it (once/iter), pA just tracks the cursor; on release cW.release_mma
// emits the commit and cA.advance() keeps A's cursor in lockstep.
//
// Correctness: each role re-inits its async edges at task start (loader:
// mma/W_tma/A_tma; launcher: mainloop/epilogue/consumer_done) to clear stray
// arrivals left on a reused ring slot by a prior occupant. This — not a
// task-end drain — is what prevents the cross-task stale-arrival deadlock.
//
// USAGE: tiles_per_task must be 1. tpt>1 produces a partial last task
// (num_tiles % tpt != 0) whose barrier accounting differs from full tasks and
// deadlocks on slot reuse; it's also slower (worse SM occupancy).

#pragma once

#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>

#include "mirage/persistent_kernel/runtime_header.h"
#include "mirage/persistent_kernel/tasks/blackwell_v2/channel.cuh"
#include "mirage/persistent_kernel/tasks/blackwell_v2/linear_device.cuh"

namespace kernel {
namespace linear_v3 {

// ── Single source of truth: kernel::linear (linear_spec.h +
// linear_device.cuh). v3 no longer depends on linear_sm100_v2.cuh — constants,
// ordinals, and PTX wrappers all come from kernel::linear, so v2 can be deleted
// independently.
using ::kernel::linear::A_SIZE;
using ::kernel::linear::BLOCK_K;
using ::kernel::linear::BLOCK_M;
using ::kernel::linear::BLOCK_N;
using ::kernel::linear::CH_A;
using ::kernel::linear::CH_W;
using ::kernel::linear::CHANNELS;
using ::kernel::linear::elect_sync;
using ::kernel::linear::I_DESC;
using ::kernel::linear::L2_EVICT_FIRST;
using ::kernel::linear::L2_EVICT_LAST;
using ::kernel::linear::L2_EVICT_NORMAL;
using ::kernel::linear::mbarrier_arrive;
using ::kernel::linear::mbarrier_arrive_expect_tx;
using ::kernel::linear::mbarrier_wait;
using ::kernel::linear::MMA_K;
using ::kernel::linear::NUM_STAGES;
using ::kernel::linear::SEM_A_TMA_BASE;
using ::kernel::linear::SEM_CONSUMER_DONE;
using ::kernel::linear::SEM_EPILOGUE_BASE;
using ::kernel::linear::SEM_MAINLOOP_BASE;
using ::kernel::linear::SEM_MMA_BASE;
using ::kernel::linear::SEM_TMEM_READY;
using ::kernel::linear::SEM_W_TMA_BASE;
using ::kernel::linear::SMEM_DESC;
using ::kernel::linear::tcgen05_commit;
using ::kernel::linear::tcgen05_mma;
using ::kernel::linear::tma_3d_load_l2;
using ::kernel::linear::W_SIZE;
using ::kernel::linear::WARP_SIZE;
using ::kernel::linear::warp_uniform;

using mpk::ch::By;

// ── SMEM base 1024-byte alignment (128B-swizzle TMA requirement) ────────────
// The W/A tiles are loaded by a 128B-swizzle cp.async.bulk.tensor whose SHARED
// destination must be aligned to the 1024-byte swizzle tile. Declaring the
// extern array `__align__(1024)` is NOT sufficient in the MPK v2 megakernel:
// worker_v2_kernel places a static `__shared__ RuntimeSMEM rt_buf` BEFORE the
// dynamic pool, so the dynamic `extern __shared__` base lands only 128-aligned
// at runtime. The per-stage W/A addresses are `base + smem_region_offset(...)`
// where the region offsets are page-multiples (16 KB, i.e. 1024-multiples), so
// the low bits are inherited from the base — if the base isn't 1024-aligned the
// TMA destination isn't either, and compute-sanitizer reports "Misaligned
// shared or local address" at the first W load (linear_loader_task). Round the
// base up to 1024 (identical workaround to mla_prefill_tp8_sm100.cuh). Costs
// <=1 KB of SMEM and is applied uniformly by loader/launcher/consumer so the
// int `smem` addr and the `char*` scratch reads resolve to the SAME bytes.
__device__ __forceinline__ int aligned_smem_base(char *smem_ptr) {
  int const raw = static_cast<int>(__cvta_generic_to_shared(smem_ptr));
  return (raw + 1023) & ~1023;
}
__device__ __forceinline__ char *aligned_smem_ptr(char *smem_ptr) {
  int const raw = static_cast<int>(__cvta_generic_to_shared(smem_ptr));
  int const aligned = (raw + 1023) & ~1023;
  return smem_ptr + (aligned - raw);
}

// ── Channel + ring type aliases (original design: sync ≠ storage) ───────────
// Channels carry ONLY mbarriers:
//   WChan/AChan: full = per-stream TMA-arrived; empty = SHARED mma_mbar.
//   AccChan: TMEM-backed, SLOTS=2 — DOUBLE-BUFFERED. mainloop_stage cycles %2,
//            alternating TMEM columns taddr+0 / taddr+BLOCK_N so tile t+1's MMA
//            overlaps the consumer's read of tile t.
//            mainloop_mbar/epilogue_mbar each have 2 slots (controller init +
//            launcher re-init touch both). cols_per_slot=BLOCK_N → 2*16=32 cols
//            = the alloc.
// SmemRings carry storage (per-stage SMEM offsets). PAGES_PER_SLOT=0 for now:
//   page release stays a task-end blanket (byte-identical to v2). Flip to the
//   real page counts + call ring.release_pages() in the consumer to enable
//   Phase-E per-stage cross-task overlap.
using WChan = mpk::ch::Channel<NUM_STAGES, By::Tma, By::Mma>;
using AChan = mpk::ch::Channel<NUM_STAGES, By::Tma, By::Mma>;
using AccChan = mpk::ch::TmemChannel<2, By::Mma, By::Warp>;

// W stage = W_SIZE/PAGE = 2 dedicated contiguous pages. When CROSS_TASK_PAGES
// is on, the W ring owns the per-stage cross-task page lifecycle (loader
// acquires / launcher releases page-by-page, so task N+1's loader TMAs into
// pages task N frees while N still computes its later stages). A stages are
// sub-page and packed (multiple A regions per physical page), so per-stage page
// control is unsafe for A — ARing stays 0 (its pages ride the task-end
// blanket).
using ::kernel::linear::CROSS_TASK_PAGES;
// W owns its 2 dedicated pages/stage and runs the cross-task page lifecycle
// (release per stage for overlap, acquire on first touch). A is storage-only:
// its pages — and scratch's, which shares one — are freed at task end by a
// PARALLEL sweep over the pages NO ring owns (each lane its own page). That
// keeps the frees parallel (not serialized on one lane) and needs no per-task
// counter. PAGES_PER_SLOT=0 when cross-task is off → the page methods vanish.
using WRing = mpk::ch::SmemRing<NUM_STAGES, CROSS_TASK_PAGES ? 2 : 0>;
using ARing = mpk::ch::SmemRing<NUM_STAGES>;

// ── Build channels (mbars) + rings (storage) from kernel::linear addresses ──
__device__ __forceinline__ void
    make_wa(int smem,
            int dyn_sem_base,
            mirage::runtime::TaskDesc const *task_desc,
            WChan &Wc,
            AChan &Ac,
            WRing &Wr,
            ARing &Ar) {
  // Phase 3: edge wiring synthesized from CHANNELS (constexpr-folded; the
  // static_asserts in linear_spec.h guarantee these equal the old SEM_*_BASE
  // literals, so this is byte-identical to the hardcoded version).
  constexpr int w_full = CHANNELS[CH_W].full_sem_base;
  constexpr int w_empty = CHANNELS[CH_W].empty_sem_base;
  constexpr int a_full = CHANNELS[CH_A].full_sem_base;
  constexpr int a_sh = CHANNELS[CH_A].shares_empty_with; // 0 -> share W empty
  Wc.full = dyn_sem_base + w_full * 8;
  Wc.empty = dyn_sem_base + w_empty * 8;
  Ac.full = dyn_sem_base + a_full * 8;
  Ac.empty = (a_sh >= 0) ? Wc.empty // SHARED
                         : dyn_sem_base + CHANNELS[CH_A].empty_sem_base * 8;

  // Per-stage SMEM offsets. W also records its 2 physical page ids so the ring
  // owns their cross-task lifecycle. A is storage-only.
  for (int s = 0; s < NUM_STAGES; s++) {
    Wr.slot_offsets[s] =
        smem + task_desc->smem_region_offset(::kernel::linear::REGION_W_0 + s);
    Ar.slot_offsets[s] =
        smem + task_desc->smem_region_offset(::kernel::linear::REGION_A_0 + s);
    if constexpr (CROSS_TASK_PAGES) {
      int const w_page0 =
          task_desc->smem_region_page(::kernel::linear::REGION_W_0 + s);
      Wr.pages[s][0] = w_page0;
      Wr.pages[s][1] = w_page0 + 1; // W_SIZE spans 2 contiguous pages
    }
  }
}

// TmemChannel stores only barrier addresses + cols_per_slot. taddr lives on
// the cursor (set via TmemProducer::set_taddr / TmemConsumer::set_taddr after
// the launcher's tcgen05.alloc publishes it).
__device__ __forceinline__ AccChan make_acc_channel(int dyn_sem_base) {
  return AccChan{
      /*cols_per_slot =*/BLOCK_N,
      /*full          =*/dyn_sem_base + SEM_MAINLOOP_BASE * 8,
      /*empty         =*/dyn_sem_base + SEM_EPILOGUE_BASE * 8,
  };
}

// ── Derived shape, computed once per role from identical inputs ────────────
// Every role used to recompute these — easy to drift if one diverges. One
// function, called the same way from all three roles, makes drift impossible.
struct TaskCtx {
  int num_spatial_tiles; // grid_m (= N_real / BLOCK_M)
  int num_tiles;         // num_spatial_tiles * SPLIT_K
  int tiles;             // tiles_to_process this task (≤ TILES_PER_TASK)
  int iters;             // iters_per_slice (= K / BLOCK_K / SPLIT_K)

  __device__ bool bounds_fail(int tile_idx) const {
    return tile_idx >= num_tiles;
  }
};

template <int SPLIT_K, int TILES_PER_TASK>
__device__ __forceinline__ TaskCtx ctx_from(int N_real, int K, int tile_idx) {
  TaskCtx c;
  c.num_spatial_tiles = N_real / BLOCK_M;
  c.num_tiles = c.num_spatial_tiles * SPLIT_K;
  int left = c.num_tiles - tile_idx;
  c.tiles = (TILES_PER_TASK < left) ? TILES_PER_TASK : left;
  c.iters = (K / BLOCK_K) / SPLIT_K;
  return c;
}

// ── PTX wrappers (same emitted instructions as the inline forms in v2) ─────

// MMA inner loop over BLOCK_K. Emits exactly the same tcgen05.mma sequence as
// v2's hand-rolled k2/k1 nested loops. The `accumulate` flag controls only the
// FIRST tcgen05_mma's enable_d predicate; all subsequent inner MMAs always
// accumulate (matching v2). `i != 0` is equivalent to v2's pass-through of `i`
// because tcgen05_mma's wrapper does `setp.ne.b32 p, %4, 0` — any nonzero value
// produces the same predicate.
__device__ __forceinline__ void
    mma_k_block(int tmem, int W_smem, int A_smem, bool accumulate) {
  uint64_t a_desc = SMEM_DESC | (uint64_t)((uint32_t)W_smem >> 4);
  uint64_t b_desc = SMEM_DESC | (uint64_t)((uint32_t)A_smem >> 4);

  // First 64-byte K chunk, first MMA — enable_d picks zero-vs-accumulate.
  tcgen05_mma(tmem, a_desc, b_desc, I_DESC, accumulate ? 1 : 0);

  // Rest of the first 64-byte K chunk — always accumulate.
  for (int k2 = 1; k2 < 64 / MMA_K; k2++) {
    a_desc += (32 >> 4);
    b_desc += (32 >> 4);
    tcgen05_mma(tmem, a_desc, b_desc, I_DESC, 1);
  }

  // Remaining 64-byte K chunks.
  for (int k1 = 1; k1 < BLOCK_K / 64; k1++) {
    uint64_t a2 =
        SMEM_DESC | (uint64_t)(((uint32_t)W_smem + k1 * BLOCK_M * 128) >> 4);
    uint64_t b2 =
        SMEM_DESC | (uint64_t)(((uint32_t)A_smem + k1 * BLOCK_N * 128) >> 4);
    for (int k2 = 0; k2 < 64 / MMA_K; k2++) {
      tcgen05_mma(tmem, a2, b2, I_DESC, 1);
      a2 += (32 >> 4);
      b2 += (32 >> 4);
    }
  }
}

// 16-register TMEM read (the giant inline asm). The `t_addr` layout is
// `(warp_id * 32 << 16) | t_col` — caller computes this, same as v2.
__device__ __forceinline__ void tcgen05_ld_16(float (&out)[16], int t_addr) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
               : "=f"(out[0]),
                 "=f"(out[1]),
                 "=f"(out[2]),
                 "=f"(out[3]),
                 "=f"(out[4]),
                 "=f"(out[5]),
                 "=f"(out[6]),
                 "=f"(out[7]),
                 "=f"(out[8]),
                 "=f"(out[9]),
                 "=f"(out[10]),
                 "=f"(out[11]),
                 "=f"(out[12]),
                 "=f"(out[13]),
                 "=f"(out[14]),
                 "=f"(out[15])
               : "r"(t_addr));
}

__device__ __forceinline__ void tcgen05_wait_ld() {
  asm volatile("tcgen05.wait::ld.sync.aligned;");
}

// Global stores with L1::no_allocate hint (matches v2's epilogue stores).
__device__ __forceinline__ void st_bf16(nv_bfloat16 *dst, nv_bfloat16 v) {
  asm volatile("st.global.L1::no_allocate.b16 [%0], %1;" ::"l"(dst),
               "h"(*(uint16_t *)&v)
               : "memory");
}

__device__ __forceinline__ void st_f32(float *dst, float v) {
  asm volatile("st.global.L1::no_allocate.b32 [%0], %1;" ::"l"(dst), "f"(v)
               : "memory");
}

// Prefetch a CUtensorMap descriptor into L1 (so the first TMA on it doesn't
// stall on descriptor fetch). Same instruction as v2's inline asm.
__device__ __forceinline__ void prefetch_tensormap(void const *tmap) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(tmap));
}

// Acquire fence following a batch of mbarrier.init writes. Matches v2.
__device__ __forceinline__ void mbar_init_fence() {
  asm volatile("fence.mbarrier_init.release.cluster;");
}

// Per-thread fence required between thread-local register reads and tcgen05
// operations that consume them. Same instruction v2 uses.
__device__ __forceinline__ void tcgen05_fence_after_thread_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;");
}

// UNCONDITIONAL probe-region size (u64 words) + init sentinel, appended after
// the per-worker breadcrumb slots in the pinned buffer. Declared outside the
// diagnostic guards so persistent_kernel_v2.cuh reserves the SAME region size /
// sentinel regardless of whether MPK_V2_LINV3_PROBE is compiled — keeping the
// device/host offset math identical. Kept in lockstep with the guarded
// linv3_probe::* constants below (static_assert'd there).
//
// Layout of the appended probe region (u64 words), base = pinned buffer +
// 2*num_workers*ROLES:
//   [0 .. LINV3_META_WORDS)                       — the legacy single-slot
//       TaskDesc-metadata + per-role PHASE block (worker==TARGET_WORKER-gated).
//   [LINV3_META_WORDS .. +WTMA_PER_WORKER*WTMA_MAX_WORKERS) — the NEW
//   per-worker
//       raw W-TMA argument record (M3 debug): worker w's fields live at
//       META + w*WTMA_PER_WORKER + <field>. Written by EVERY loader linear_v3
//       task (elected lane) on EACH cp.async.bulk.tensor W-load (M3 FAULTING-
//       load extension — the original "first load only, t==0&&i==0" gate MISSED
//       the faulting later load, which is exactly the tail-band N-tile bug). On
//       each load the loader computes the OOB/misalign check ON DEVICE and:
//         * if BAD (dst&1023!=0 / null tmap / row-OOB / kchunk-OOB / src-OOB),
//           it FREEZES that load's args into the slot + sets WT_BAD_FLAG
//           (bitmask of tripped checks) and never overwrites it again;
//         * else it keeps OVERWRITING the slot with the LAST good load so a
//           no-bad-check crash still shows the last-issued operands.
//       WT_COMPLETED is stamped after the TMA issue returns (dumped-but-unset
//       == the faulting loader). WT_LOAD_IDX = the global load counter
//       (t*iters+i) of the recorded load; WT_NUM_LOADS = total loads this task
//       will issue. Per-worker slots mean the faulting worker's record is never
//       overwritten by another; the start-of-loop freeze read of WT_BAD_FLAG
//       protects an already-frozen bad record from a later good load (this or a
//       later task).
constexpr int LINV3_META_WORDS_HOST = 64;
// 24 (was 16): M3 FAULTING-load extension appends WT_LOAD_IDX / WT_BAD_FLAG /
// WT_SRC_LIMIT / WT_NUM_LOADS after the original 16-field record. Kept in
// lockstep with linv3_probe::WTMA_WORDS_PER_WORKER (static_assert'd there).
constexpr int LINV3_WTMA_PER_WORKER_HOST = 24;
// Cover every worker in the TP8 grid (num_workers==136). The pre-fix sanitizer
// showed the faulting set in workers 70..131; capturing [0,135] guarantees the
// faulter's slot exists regardless of which worker it lands on.
constexpr int LINV3_WTMA_MAX_WORKERS_HOST = 136;
constexpr int LINV3_PROBE_WORDS_HOST =
    LINV3_META_WORDS_HOST +
    LINV3_WTMA_PER_WORKER_HOST * LINV3_WTMA_MAX_WORKERS_HOST;
constexpr unsigned long long LINV3_PROBE_SENTINEL_HOST = 0xDEAD5107DEAD5107ull;

// ═══════════════════════════════════════════════════════════════════════════
// DIAGNOSTIC (M3 debug, default-OFF => default build byte-identical) — localize
// a deterministic cudaErrorIllegalAddress breadcrumbed to worker=36 /
// TASK_LINEAR_SM100_V3 / seq_in_iter=4 (an lm_head N-tile). Two env-gated
// tools, both compiling to NOTHING unless their macro is defined:
//   * MPK_V2_LINV3_PROBE   — dump this worker/tile's copied TaskDesc metadata +
//     per-role PHASE markers to the host-mapped PINNED breadcrumb buffer (it
//     survives the context-poisoning fault). Answers "is worker36's TaskDesc
//     metadata valid, and WHICH phase reached the fault?".
//   * MPK_V2_LINV3_SKIP36  — no-op the target task's body (skip data movement +
//     compute + tcgen05.alloc/dealloc) while preserving ALL sync + page parity.
//     Answers "does skipping worker36's tile make the crash VANISH/MOVE (real
//     body/tile/metadata bug) or PERSIST (concurrent poisoner / false
//     attribution)?".
//   * MPK_V2_LINV3_SKIPALL — sibling of SKIP36: the SAME no-op skip stub, but
//     applied to EVERY TASK_LINEAR_SM100_V3 task (drops the worker36/seq==4
//     condition). Reuses the identical op-private + page sync skeleton (loader:
//     reinit; launcher: reinit + arrive SEM_TMEM_READY + task-end page sweep +
//     wait SEM_CONSUMER_DONE; consumer: wait SEM_TMEM_READY + arrive
//     SEM_CONSUMER_DONE) so the megakernel's grid-barrier + page parity stay
//     intact while linear_v3 does NO data movement/compute/tcgen05. Clean
//     causal ablation for "is linear_v3 the faulter?": VANISHES (run completes,
//     garbage lm_head logits) ⇒ linear_v3 IS the faulter; PERSISTS (same
//     illegal-address) ⇒ the faulter is elsewhere. Needs NO attribution and NO
//     breadcrumb buffer.
// MPK_V2_LINV3_PROBE REQUIRES the pinned buffer, which is only allocated when
// MPK_V2_BREADCRUMB is also set (persistent_kernel_v2.cuh). Force the coupling
// explicit rather than silently writing nowhere:
#if defined(MPK_V2_LINV3_PROBE) && !defined(MPK_V2_BREADCRUMB)
#error                                                                         \
    "MPK_V2_LINV3_PROBE needs the pinned breadcrumb buffer: also set MPK_V2_BREADCRUMB (and forward -x MPK_V2_BREADCRUMB -x MPK_V2_LINV3_PROBE)."
#endif

#if defined(MPK_V2_LINV3_PROBE) || defined(MPK_V2_LINV3_SKIP36) ||             \
    defined(MPK_V2_LINV3_SKIPALL)
namespace linv3_probe {

// The breadcrumbed candidate: worker (blockIdx.x) and this-worker's
// seq_in_iter. The tile bound (num_spatial_tiles for lm_head N=129280 / 128)
// is 1010; task_offset >= that is itself the bug.
constexpr int TARGET_WORKER = 36;
constexpr int TARGET_SEQ_IN_ITER = 4;
constexpr int LMHEAD_NUM_TILES = 1010; // 129280 / BLOCK_M(128); validity bound

__device__ __forceinline__ bool
    is_target(int seq_in_iter, mirage::runtime::TaskDesc const *task_desc) {
  return blockIdx.x == TARGET_WORKER && seq_in_iter == TARGET_SEQ_IN_ITER &&
         task_desc->task_type == mirage::runtime::TASK_LINEAR_SM100_V3;
}

// SKIPALL predicate: skip the body of EVERY linear_v3 task. Under SKIP36-only
// it reduces to is_target() (worker36/seq==4) so the two gates can coexist in
// the same build without changing SKIP36's meaning; under SKIPALL it fires for
// any TASK_LINEAR_SM100_V3 task regardless of worker/seq. The three role bodies
// gate their skip stub on THIS (not is_target) so both macros drive the
// identical sync skeleton.
__device__ __forceinline__ bool
    should_skip(int seq_in_iter, mirage::runtime::TaskDesc const *task_desc) {
#if defined(MPK_V2_LINV3_SKIPALL)
  (void)seq_in_iter;
  return task_desc->task_type == mirage::runtime::TASK_LINEAR_SM100_V3;
#else
  return is_target(seq_in_iter, task_desc);
#endif
}
} // namespace linv3_probe
#endif

#ifdef MPK_V2_LINV3_PROBE
namespace linv3_probe {
// ── Probe region layout in the pinned breadcrumb buffer ─────────────────────
// The region is a fixed 64-u64 window APPENDED after the per-worker breadcrumb
// slots (persistent_kernel_v2.cuh reserves 2*num_workers*ROLES u64 first, then
// LINV3_PROBE_WORDS more). Device + host compute the SAME base from
// breadcrumb_num_slots (== num_workers) + the compile constant ROLES.
// Slots are DISJOINT per role (loader / launcher / consumer run concurrently on
// the SAME task+slot) so single-writer stores never race; each field has
// exactly one writer thread (loader elected-lane / launcher lane0 / consumer
// thread0). Every write is followed by __threadfence_system() so the host sees
// it after the launch returns cudaErrorIllegalAddress.
// Metadata/phase block size (worker==TARGET_WORKER-gated single slot).
constexpr int LINV3_META_WORDS = 64;
static_assert(LINV3_META_WORDS == ::kernel::linear_v3::LINV3_META_WORDS_HOST,
              "linv3 metadata block size must match the host reservation");
// Per-worker raw W-TMA record (M3 debug) — WORDS_PER_WORKER u64 for each of
// WTMA_MAX_WORKERS workers, appended after the metadata block.
constexpr int WTMA_WORDS_PER_WORKER =
    ::kernel::linear_v3::LINV3_WTMA_PER_WORKER_HOST;
constexpr int WTMA_MAX_WORKERS =
    ::kernel::linear_v3::LINV3_WTMA_MAX_WORKERS_HOST;
constexpr int LINV3_PROBE_WORDS =
    LINV3_META_WORDS + WTMA_WORDS_PER_WORKER * WTMA_MAX_WORKERS;
static_assert(LINV3_PROBE_WORDS == ::kernel::linear_v3::LINV3_PROBE_WORDS_HOST,
              "linv3 probe region size must match the host reservation");
constexpr unsigned long long SENTINEL = 0xDEAD5107DEAD5107ull; // "unwritten"
static_assert(SENTINEL == ::kernel::linear_v3::LINV3_PROBE_SENTINEL_HOST,
              "linv3 probe sentinel must match the host init");
constexpr unsigned long long MAGIC = 0x4C494E5633000001ull; // "LINV3\0\0\x01"

// Per-worker W-TMA record fields (offsets within a worker's
// WTMA_WORDS_PER_WORKER-word slot). The record is written by the LOADER's
// elected lane at its FIRST W-TMA (t==0,i==0) for EVERY TASK_LINEAR_SM100_V3
// task (NOT worker-gated) so the faulting worker's operands are captured on
// whichever SM it runs. WT_COMPLETED is written AFTER the cp.async.bulk.tensor
// issue returns: dumped-but-COMPLETED-unset == the faulting loader.
enum WtmaField {
  WT_MAGIC = 0,     // WT_SLOT_MAGIC (proves this worker's slot was written)
  WT_DST = 1,       // W_smem = Wr.slot_addr(pW.st) — the EXACT shared addr
                    // the TMA sees (post aligned_smem_base()); low 12 bits
                    // decoded host-side (is it REALLY 1024-aligned?).
  WT_TMAP = 2,      // W_tmap_ptr (input_tma_desc_ptrs[1][0]) — null/garbage?
  WT_COORD_X = 3,   // x coord passed to tma_3d_load_l2 (== 0)
  WT_COORD_Y = 4,   // y coord == cur_off_m (row offset into N)
  WT_COORD_Z = 5,   // z coord == z_coord (== iter_k * BLOCK_K/64)
  WT_BOX_D0 = 6,    // box dim0 (== BK = 64)
  WT_BOX_D1 = 7,    // box dim1 (== BLOCK_M = 128)
  WT_BOX_D2 = 8,    // box dim2 (== BLOCK_K/BK = 2)
  WT_GMEM_BASE = 9, // input_ptrs[1] — the W GMEM base ptr the desc encodes
  WT_SRC_OFF = 10,  // computed source byte-offset from base:
                    //   y*rowstride(=K*2) + z*kchunkstride(=128)
  WT_N_REAL = 11,   // N_real (row bound for the coord-OOB check)
  WT_K = 12,        // K (k-chunk bound: z + box_d2 <= K/BK)
  WT_SEQ_IN_ITER = 13, // this worker's seq_in_iter (context)
  WT_ITER_NUM = 14,    // iter_num (context)
  WT_COMPLETED = 15, // WT_COMPLETED_MAGIC written AFTER the W-TMA issue of the
                     // RECORDED load returns; still SENTINEL => this loader
                     // faulted at/inside that cp.async.bulk.tensor. (For a
                     // FROZEN bad load this stays SENTINEL because the recorded
                     // load is the suspected faulter — its issue never
                     // completes when it is the illegal access.)
  // ── M3 FAULTING-load extension (fields 16..) ──────────────────────────────
  WT_LOAD_IDX = 16,  // global load counter (t*iters + i) of the RECORDED load
                     // — "which load # faulted / was last".
  WT_BAD_FLAG = 17,  // bitmask of tripped checks for the recorded load (see
                     // WtBad); SENTINEL until first written; 0 == a CLEAN load
                     // was recorded (no static check tripped).
  WT_SRC_LIMIT = 18, // weight_buffer_bytes bound (N_real*K*2) used for the
                     // src_off-OOB cross-check (host context).
  WT_NUM_LOADS = 19, // total loads this task issues (c.tiles*c.iters) —
                     // context for WT_LOAD_IDX.
  // 20..23 reserved (record padded to WTMA_WORDS_PER_WORKER=24).
};
// Bitmask values for WT_BAD_FLAG (which static OOB/misalign check tripped).
enum WtBad {
  WTBAD_DST_MISALIGN = 1, // dst & 1023 != 0 (not 1024-aligned for 128B swizzle)
  WTBAD_TMAP_NULL = 2,    // W_tmap_ptr == nullptr
  WTBAD_ROW_OOB = 4,      // coord_y + BLOCK_M > N_real (dim1=N box overrun)
  WTBAD_KCHUNK_OOB = 8,   // coord_z + BLOCK_K/BK > K/BK (dim2=K/64 box overrun)
  WTBAD_SRC_OOB = 16,     // src_off + W_SIZE > N_real*K*2 (buffer byte overrun;
                          // fires only if N_real/K disagree with the descriptor
  // dims baked on the host — a metadata mismatch signal)
};
static_assert(WT_NUM_LOADS < WTMA_WORDS_PER_WORKER,
              "W-TMA record must fit in WTMA_WORDS_PER_WORKER");
constexpr unsigned long long WT_SLOT_MAGIC =
    0x5754414D41300001ull; // "WTAMA0\x01"
constexpr unsigned long long WT_COMPLETED_MAGIC = 0x574F4E45444F4E01ull;

enum Slot {
  // ── loader block [0..24) — has instruction_index/iter_num/runtime_config ──
  S_MAGIC = 0,         // MAGIC (proves loader block written)
  S_LD_PHASE,          // loader phase marker (monotone; see LdPhase)
  S_TASK_TYPE,         // task_desc->task_type
  S_VARIANT_ID,        // task_desc->variant_id
  S_TASK_OFFSET,       // task_metadata.task_offset  (== tile_idx)
  S_TASK_OFFSET_VALID, // 1 iff 0 <= task_offset < num_tiles  (BUG if 0)
  S_NUM_TILES,         // ctx num_tiles (num_spatial_tiles*SPLIT_K)
  S_N_REAL,            // N_real
  S_K,                 // K
  S_MY_COUNT,          // this worker's tasks/iter
  S_INSTR_IDX,         // instruction_index (ring sequence)
  S_SEQ_IN_ITER,       // instruction_index % my_count
  S_ITER_NUM,          // iter_num
  S_DEP_EVENT,         // dependent_event (raw)
  S_DEP_EVENT_IDX,     // decoded event position index (low 32b)
  S_RAW_PAYLOAD,       // task_metadata.raw_payload
  S_IN_PTR0,           // input_ptrs[0]  (A activation)
  S_IN_PTR1,           // input_ptrs[1]  (W weight)
  S_OUT_PTR0,          // output_ptrs[0] (C)
  S_A_DESC,            // input_tma_desc_ptrs[0][0] (A-desc; -1 w/o TMA)
  S_W_DESC,            // input_tma_desc_ptrs[1][0] (W-desc; -1 w/o TMA)
  S_NUM_REGIONS,       // num_smem_regions
  S_SMEM_BAD_MASK,     // bitmask of malformed linear SMEM regions (0 == ok)
  S_LD_RESERVED,       // pad to 24
  // ── launcher block [24..40) ──────────────────────────────────────────────
  S_LC_PHASE = 24,     // launcher phase marker (see LcPhase)
  S_LC_ITER,           // iter_num when the launcher last entered this body
                       // (pairs with S_LC_PHASE; if != loader S_ITER_NUM the
                       // launcher did NOT re-enter this crash iter => its phase
                       // is stale, read it as "died in the launcher prefix").
  S_LC_TADDR,          // tcgen05 alloc addr (SKIP -> not written)
  S_LC_SMEM_BASE_LO12, // aligned_smem_base() low 12 bits
  S_LC_SCRATCH_OFF,    // REGION_SCRATCH byte offset (region_offset)
  S_LC_SCRATCH_PG,     // REGION_SCRATCH physical_page_start
  S_LC_RESERVED,       // pad to 40
  // ── consumer block [40..56) ───────────────────────────────────────────────
  S_CN_PHASE = 40, // consumer phase marker (see CnPhase)
  S_CN_ITER,       // iter_num when the consumer last entered (see S_LC_ITER)
  S_CN_C_PTR,      // C_ptr as the consumer sees it
  S_CN_TADDR,      // taddr the consumer read from SCRATCH
  S_CN_RESERVED,   // pad to 56
  // [56..64) reserved
};

// Loader phase markers — the LAST value written localizes the phase reached.
// Prefix markers matter: the codegen loader page-prefix runs BEFORE this body,
// so a prefix fault would leave S_LD_PHASE unwritten (== SENTINEL) which itself
// is a signal ("died before/in the loader page-prefix").
enum LdPhase {
  LD_BODY_ENTERED = 1,
  LD_AFTER_META_DUMP = 2,
  LD_AFTER_REINIT = 3,
  LD_BEFORE_FIRST_WTMA = 4,
  LD_AFTER_FIRST_WTMA = 5,
  LD_AFTER_DEP_WAIT = 6,
  LD_LOOP_DONE = 7,
  LD_SKIP_RETURN = 100, // SKIP36 path
};
enum LcPhase {
  LC_BODY_ENTERED = 1,
  LC_AFTER_REINIT = 2,
  LC_AFTER_ALLOC = 3,
  LC_TMEM_READY_ARRIVED = 4,
  LC_MMA_LOOP_DONE = 5,
  LC_PAGES_RELEASED = 6,
  LC_CONSUMER_DONE_WAITED = 7,
  LC_AFTER_DEALLOC = 8,
  // SKIP36 path — monotone (100 < 101 < 102 < 103) so the max value present is
  // the furthest skip phase reached.
  LC_SKIP_ENTERED = 100,
  LC_SKIP_TMEM_ARRIVED = 101,   // arrived SEM_TMEM_READY w/o alloc
  LC_SKIP_PAGES_RELEASED = 102, // task-end page sweep done
  LC_SKIP_RETURN = 103,         // waited SEM_CONSUMER_DONE, returning
};
enum CnPhase {
  CN_BODY_ENTERED = 1,
  CN_TMEM_READY_WAITED = 2,
  CN_AFTER_TADDR_READ = 3,
  CN_BEFORE_FIRST_STORE = 4,
  CN_AFTER_FIRST_STORE = 5,
  CN_LOOP_DONE = 6,
  CN_CONSUMER_DONE_ARRIVED = 7,
  CN_SKIP_RETURN = 100,
};

// Pinned probe-region base (or nullptr if no buffer). SAME expression as the
// host dump in persistent_kernel_v2.cuh.
__device__ __forceinline__ unsigned long long *
    base(mirage::runtime::RuntimeConfig const &config) {
  if (config.breadcrumb_device == nullptr || config.breadcrumb_num_slots <= 0) {
    return nullptr;
  }
  return static_cast<unsigned long long *>(config.breadcrumb_device) +
         2ull * static_cast<size_t>(config.breadcrumb_num_slots) *
             MPK_V2_BREADCRUMB_ROLES;
}

// Single-writer field store + system fence (host must see it post-crash).
__device__ __forceinline__ void
    put(unsigned long long *p, int slot, unsigned long long v) {
  if (p == nullptr) {
    return;
  }
  p[slot] = v;
  __threadfence_system();
}

// Monotone phase marker (never rewinds — later phases overwrite; the max value
// present is the furthest phase reached).
__device__ __forceinline__ void mark(unsigned long long *p, int slot, int ph) {
  put(p, slot, static_cast<unsigned long long>(ph));
}

// Per-worker W-TMA record base for `worker` (nullptr if no buffer, or if the
// worker index is outside the reserved [0, WTMA_MAX_WORKERS) range — a worker
// beyond the reservation is simply NOT recorded rather than clobbering another
// slot). SAME base expression as the metadata block, then + META offset +
// worker stride, so the host dump derives identical addresses.
__device__ __forceinline__ unsigned long long *
    wtma_slot(mirage::runtime::RuntimeConfig const &config, int worker) {
  unsigned long long *b = base(config);
  if (b == nullptr || worker < 0 || worker >= WTMA_MAX_WORKERS) {
    return nullptr;
  }
  return b + LINV3_META_WORDS +
         static_cast<size_t>(worker) * WTMA_WORDS_PER_WORKER;
}
} // namespace linv3_probe
#endif // MPK_V2_LINV3_PROBE

// ═══════════════════════════════════════════════════════════════════════════
// Loader role (warp 4, elected lane only) — TMA loop + start-of-task re-init.
// ═══════════════════════════════════════════════════════════════════════════
template <int SPLIT_K = 1, int W_L2_HINT = 0, int TILES_PER_TASK = 1>
__device__ __noinline__ void
    linear_loader_task(mirage::runtime::TaskDesc const *task_desc,
                       mirage::runtime_v2::RuntimeSMEM *runtime_smem,
                       mirage::runtime::RuntimeConfig const &runtime_config,
                       CUtensorMap const *W_tmap_ptr,
                       CUtensorMap const *A_tmap_ptr,
                       int N_real,
                       int K,
                       int tile_idx,
                       int instruction_index,
                       int iter_num,
                       int dyn_sem_base
#if defined(MPK_V2_LINV3_PROBE) || defined(MPK_V2_LINV3_SKIP36) ||             \
    defined(MPK_V2_LINV3_SKIPALL)
                       ,
                       int _linv3_seq_in_iter
#endif
    ) {
  if (!elect_sync()) {
    return;
  }

#if defined(MPK_V2_LINV3_PROBE) || defined(MPK_V2_LINV3_SKIP36) ||             \
    defined(MPK_V2_LINV3_SKIPALL)
  bool const _linv3_hit = linv3_probe::is_target(_linv3_seq_in_iter, task_desc);
  bool const _linv3_skip =
      linv3_probe::should_skip(_linv3_seq_in_iter, task_desc);
  (void)_linv3_hit;
  (void)_linv3_skip;
#endif
#ifdef MPK_V2_LINV3_PROBE
  // Metadata dump (loader block) — the loader is the only role with
  // instruction_index / iter_num / runtime_config, so it dumps the full block.
  // Written unconditionally of SKIP so a NON-skipped PROBE run still records
  // it.
  if (_linv3_hit) {
    unsigned long long *_bp = linv3_probe::base(runtime_config);
    linv3_probe::mark(
        _bp, linv3_probe::S_LD_PHASE, linv3_probe::LD_BODY_ENTERED);
    const TaskCtx _pc = ctx_from<SPLIT_K, TILES_PER_TASK>(N_real, K, tile_idx);
    long long _to =
        static_cast<long long>(task_desc->task_metadata.task_offset);
    // Malformed-region mask over the linear SMEM regions the planner assigns:
    // NUM_REGIONS regions must each land inside [0, MAX_SMEM_PAGES_PER_TASK)
    // and have page_count > 0. A bad start/count/span is a copied-TaskDesc bug.
    unsigned long long _bad = 0ull;
    int const _nreg = task_desc->num_smem_regions;
    for (int r = 0; r < ::kernel::linear::NUM_REGIONS && r < 63; r++) {
      if (r >= _nreg) {
        _bad |= (1ull << r);
        continue;
      }
      mirage::runtime::SmemPageRegionDesc const &_rg =
          task_desc->smem_regions[r];
      int const _st = _rg.physical_page_start;
      int const _pc2 = _rg.page_count;
      if (_st < 0 || _pc2 <= 0 ||
          _st + _pc2 > mirage::runtime::MAX_SMEM_PAGES_PER_TASK) {
        _bad |= (1ull << r);
      }
    }
    linv3_probe::put(_bp, linv3_probe::S_MAGIC, linv3_probe::MAGIC);
    linv3_probe::put(_bp, linv3_probe::S_TASK_TYPE, task_desc->task_type);
    linv3_probe::put(_bp, linv3_probe::S_VARIANT_ID, task_desc->variant_id);
    linv3_probe::put(
        _bp, linv3_probe::S_TASK_OFFSET, static_cast<unsigned long long>(_to));
    linv3_probe::put(_bp,
                     linv3_probe::S_TASK_OFFSET_VALID,
                     (_to >= 0 && _to < static_cast<long long>(_pc.num_tiles))
                         ? 1ull
                         : 0ull);
    linv3_probe::put(_bp, linv3_probe::S_NUM_TILES, _pc.num_tiles);
    linv3_probe::put(_bp, linv3_probe::S_N_REAL, static_cast<unsigned>(N_real));
    linv3_probe::put(_bp, linv3_probe::S_K, static_cast<unsigned>(K));
    // my_count: this worker's tasks/iter (only reconstructable here).
    int const _mc =
        static_cast<int>(runtime_config.v2_per_sm_task_offsets[blockIdx.x + 1] -
                         runtime_config.v2_per_sm_task_offsets[blockIdx.x]);
    linv3_probe::put(_bp, linv3_probe::S_MY_COUNT, static_cast<unsigned>(_mc));
    linv3_probe::put(_bp,
                     linv3_probe::S_INSTR_IDX,
                     static_cast<unsigned>(instruction_index));
    linv3_probe::put(_bp,
                     linv3_probe::S_SEQ_IN_ITER,
                     static_cast<unsigned>(_linv3_seq_in_iter));
    linv3_probe::put(
        _bp, linv3_probe::S_ITER_NUM, static_cast<unsigned>(iter_num));
    unsigned long long const _dep = task_desc->dependent_event;
    linv3_probe::put(_bp, linv3_probe::S_DEP_EVENT, _dep);
    linv3_probe::put(_bp,
                     linv3_probe::S_DEP_EVENT_IDX,
                     (_dep & 0xFFFFFFFFull)); // low-32b position index (safe)
    linv3_probe::put(
        _bp, linv3_probe::S_RAW_PAYLOAD, task_desc->task_metadata.raw_payload);
    linv3_probe::put(
        _bp,
        linv3_probe::S_IN_PTR0,
        reinterpret_cast<unsigned long long>(task_desc->input_ptrs[0]));
    linv3_probe::put(
        _bp,
        linv3_probe::S_IN_PTR1,
        reinterpret_cast<unsigned long long>(task_desc->input_ptrs[1]));
    linv3_probe::put(
        _bp,
        linv3_probe::S_OUT_PTR0,
        reinterpret_cast<unsigned long long>(task_desc->output_ptrs[0]));
#ifdef MPK_ENABLE_TMA
    linv3_probe::put(_bp,
                     linv3_probe::S_A_DESC,
                     reinterpret_cast<unsigned long long>(
                         task_desc->input_tma_desc_ptrs[0][0]));
    linv3_probe::put(_bp,
                     linv3_probe::S_W_DESC,
                     reinterpret_cast<unsigned long long>(
                         task_desc->input_tma_desc_ptrs[1][0]));
#else
    linv3_probe::put(_bp, linv3_probe::S_A_DESC, ~0ull); // sentinel: no TMA
    linv3_probe::put(_bp, linv3_probe::S_W_DESC, ~0ull);
#endif
    linv3_probe::put(
        _bp, linv3_probe::S_NUM_REGIONS, static_cast<unsigned>(_nreg));
    linv3_probe::put(_bp, linv3_probe::S_SMEM_BAD_MASK, _bad);
    linv3_probe::mark(
        _bp, linv3_probe::S_LD_PHASE, linv3_probe::LD_AFTER_META_DUMP);
  }
#endif // MPK_V2_LINV3_PROBE

#if defined(MPK_V2_LINV3_SKIP36) || defined(MPK_V2_LINV3_SKIPALL)
  // NO-OP the target task's loader: preserve only the sync the loader owns.
  // The codegen loader page-prefix has ALREADY run before this body; at
  // CROSS_TASK_PAGES=false the loader body owns NO page arrivals (the launcher
  // task-end sweep frees all pages). So the ONLY thing to preserve here is the
  // start-of-task re-init (clears stray async arrivals on the reused slot).
  // Gate on _linv3_skip: worker36/seq==4 under SKIP36, EVERY linear_v3 task
  // under SKIPALL (identical stub, broader condition).
  if (_linv3_skip) {
    ::kernel::linear::reinit_for_role(::kernel::linear::Role::Loader,
                                      dyn_sem_base);
#ifdef MPK_V2_LINV3_PROBE
    linv3_probe::mark(linv3_probe::base(runtime_config),
                      linv3_probe::S_LD_PHASE,
                      linv3_probe::LD_SKIP_RETURN);
#endif
    return;
  }
#endif // MPK_V2_LINV3_SKIP36 || MPK_V2_LINV3_SKIPALL

  prefetch_tensormap(W_tmap_ptr);
  prefetch_tensormap(A_tmap_ptr);

  extern __shared__ __align__(1024) char smem_ptr[];
  // 1024-align the dynamic SMEM base so 128B-swizzle W/A TMA destinations land
  // on the swizzle tile (see aligned_smem_base note above).
  int const smem = aligned_smem_base(smem_ptr);

  // Shape — one place, used by all three roles.
  const TaskCtx c = ctx_from<SPLIT_K, TILES_PER_TASK>(N_real, K, tile_idx);
  if (c.bounds_fail(tile_idx)) {
    return;
  }

  MPK_V2_PROF_SNAPSHOT()

  constexpr uint64_t W_HINT =
      (W_L2_HINT == 0) ? L2_EVICT_FIRST : L2_EVICT_NORMAL;

  // ── Channels (sync) + rings (storage) + cursors ─────────────────────────
  WChan Wc;
  AChan Ac;
  WRing Wr;
  ARing Ar;
  make_wa(smem, dyn_sem_base, task_desc, Wc, Ac, Wr, Ar);

  // Loader re-init from CHANNELS reinit_*_by policy (table-driven, Phase 2b).
  ::kernel::linear::reinit_for_role(::kernel::linear::Role::Loader,
                                    dyn_sem_base);
#ifdef MPK_V2_LINV3_PROBE
  if (_linv3_hit) {
    linv3_probe::mark(linv3_probe::base(runtime_config),
                      linv3_probe::S_LD_PHASE,
                      linv3_probe::LD_AFTER_REINIT);
  }
#endif

  mpk::ch::Producer<WChan> pW{Wc};
  mpk::ch::Producer<AChan> pA{Ac};
  // Both ph start at 1 (pre-empty). pW.ph mirrors v2's `mma_phase`.

  bool dep_done = false;

#ifdef MPK_V2_LINV3_PROBE
  // M3 FAULTING-load capture (per-worker, EVERY load). Read the per-worker
  // WT_BAD_FLAG ONCE before the loop: if a prior task on this worker already
  // FROZE a bad load, we must not overwrite that record with a later good load,
  // so start "frozen". Otherwise start unfrozen — the loop then overwrites the
  // slot with each load (keeping the LAST good load) until it hits a bad one,
  // which it freezes. Only the elected loader lane runs (elect_sync at entry),
  // so this is single-writer per worker slot.
  unsigned long long *const _wt_slot =
      (task_desc->task_type == mirage::runtime::TASK_LINEAR_SM100_V3)
          ? linv3_probe::wtma_slot(runtime_config, blockIdx.x)
          : nullptr;
  bool _wt_frozen =
      (_wt_slot != nullptr) &&
      (_wt_slot[linv3_probe::WT_BAD_FLAG] != linv3_probe::SENTINEL) &&
      (_wt_slot[linv3_probe::WT_BAD_FLAG] != 0ull);
  int const _wt_num_loads = c.tiles * c.iters;
#endif

  for (int t = 0; t < c.tiles; t++) {
    int const cur_tile_idx = tile_idx + t;
    int const cur_spatial_idx = cur_tile_idx % c.num_spatial_tiles;
    int const cur_k_slice = cur_tile_idx / c.num_spatial_tiles;
    int const cur_k_start = cur_k_slice * c.iters;
    int const cur_off_m = cur_spatial_idx * BLOCK_M;

    for (int i = 0; i < c.iters; i++) {
      int const iter_k = cur_k_start + i;
      int const z_coord = iter_k * (BLOCK_K / 64);

      // Cross-task page acquire on FIRST touch of each W stage (first
      // NUM_STAGES global iters each visit a distinct stage once): wait the
      // prior task's release of this stage's W pages before TMA-ing into them —
      // this overlaps THIS task's W loads with the PRIOR task's compute.
      // Steady-state reuse after that rides the in-task empty edge below.
      if constexpr (CROSS_TASK_PAGES) {
        if (t * c.iters + i < NUM_STAGES) {
          Wr.acquire(pW.st,
                     runtime_smem,
                     instruction_index,
                     mirage::runtime_v2::runtime_wait_page_ready);
        }
      }

      // Wait shared empty (mma_mbar[stage]) — one wait per iter, covers both.
      // (timed-wait on the FIRST ring lap only: that is where the cold-start
      // exposure lives; timing every K-iter measurably slowed the kernel.)
      MPK_V2_TIMED_WAIT_IF(t * c.iters + i < NUM_STAGES,
                           V2_PROF_GROUP_LOADER_PHASE,
                           V2_PROF_MMA_EMPTY_WAIT,
                           pW.wait_free());
      int const W_smem = Wr.slot_addr(pW.st); // storage addr from the ring

#ifdef MPK_V2_LINV3_PROBE
      if (_linv3_hit && t == 0 && i == 0) {
        linv3_probe::mark(linv3_probe::base(runtime_config),
                          linv3_probe::S_LD_PHASE,
                          linv3_probe::LD_BEFORE_FIRST_WTMA);
      }
      // ── RAW W-TMA argument capture (M3 FAULTING-load), PER-WORKER, EVERY ──
      // load. The original probe captured only t==0&&i==0 (the FIRST load) —
      // which is in-bounds — so it MISSED the faulting later (tail-band N-tile)
      // load. Now, on EACH cp.async.bulk.tensor W-load, compute the
      // OOB/misalign check ON DEVICE for THIS load's exact operands and either
      // FREEZE the first bad one (never overwrite it) or overwrite the slot
      // with the last good load. Single writer: this loader is the elected lane
      // (elect_sync() at entry); each put() carries __threadfence_system() so
      // the host sees it post-fault. WT_COMPLETED is stamped after the RECORDED
      // load's TMA issue returns (below) — a frozen bad load leaves it SENTINEL
      // (its issue is the suspected illegal access), pinning the faulter.
      bool _wt_this_load_recorded = false;
      unsigned long long _wt_this_bad = 0ull;
      if (_wt_slot != nullptr && !_wt_frozen) {
        // Box dims: compile-time constants matching the W descriptor in tma.cuh
        // (param_id==1): bd = {BK=64, BLOCK_M=128, BLOCK_K/BK=2}.
        constexpr int _BK = 64;
        // Source byte-offset the descriptor resolves for (x=0, y=cur_off_m,
        // z=z_coord): y*rowstride + z*kchunkstride, with gs={K*2, 128}. 64-bit
        // to avoid overflow for the far tail (cur_off_m*K*2 ~ 129152*14336).
        long long const _src_off = static_cast<long long>(cur_off_m) *
                                       (static_cast<long long>(K) * 2) +
                                   static_cast<long long>(z_coord) * 128;
        long long const _buf_bytes =
            static_cast<long long>(N_real) * static_cast<long long>(K) * 2;
        // The 4 static checks. A box whose TOP coordinate EQUALS the global dim
        // is fully in-bounds (last valid index = dim-1; box [s, s+bd) with
        // s+bd==dim covers [dim-bd, dim-1]) — so `> dim` (NOT `>= dim`) is the
        // correct OOB test. W_SIZE (32768) == the per-load tile bytes.
        bool const _dst_mis =
            ((static_cast<unsigned int>(W_smem)) & 1023u) != 0u;
        bool const _tmap_null = (W_tmap_ptr == nullptr);
        bool const _row_oob = (cur_off_m + BLOCK_M) > N_real;
        bool const _kchunk_oob = (z_coord + (BLOCK_K / _BK)) > (K / _BK);
        bool const _src_oob = (_src_off + W_SIZE) > _buf_bytes;
        _wt_this_bad =
            (_dst_mis ? (unsigned long long)linv3_probe::WTBAD_DST_MISALIGN
                      : 0ull) |
            (_tmap_null ? (unsigned long long)linv3_probe::WTBAD_TMAP_NULL
                        : 0ull) |
            (_row_oob ? (unsigned long long)linv3_probe::WTBAD_ROW_OOB : 0ull) |
            (_kchunk_oob ? (unsigned long long)linv3_probe::WTBAD_KCHUNK_OOB
                         : 0ull) |
            (_src_oob ? (unsigned long long)linv3_probe::WTBAD_SRC_OOB : 0ull);
        unsigned long long *_wt = _wt_slot;
        // WT_MAGIC written LAST so "WT_MAGIC present => record complete" holds.
        linv3_probe::put(
            _wt,
            linv3_probe::WT_DST,
            static_cast<unsigned long long>(static_cast<unsigned int>(W_smem)));
        linv3_probe::put(_wt,
                         linv3_probe::WT_TMAP,
                         reinterpret_cast<unsigned long long>(W_tmap_ptr));
        linv3_probe::put(_wt, linv3_probe::WT_COORD_X, 0ull);
        linv3_probe::put(_wt,
                         linv3_probe::WT_COORD_Y,
                         static_cast<unsigned long long>(
                             static_cast<unsigned int>(cur_off_m)));
        linv3_probe::put(_wt,
                         linv3_probe::WT_COORD_Z,
                         static_cast<unsigned long long>(
                             static_cast<unsigned int>(z_coord)));
        linv3_probe::put(
            _wt, linv3_probe::WT_BOX_D0, static_cast<unsigned long long>(_BK));
        linv3_probe::put(_wt,
                         linv3_probe::WT_BOX_D1,
                         static_cast<unsigned long long>(BLOCK_M));
        linv3_probe::put(_wt,
                         linv3_probe::WT_BOX_D2,
                         static_cast<unsigned long long>(BLOCK_K / _BK));
        linv3_probe::put(
            _wt,
            linv3_probe::WT_GMEM_BASE,
            reinterpret_cast<unsigned long long>(task_desc->input_ptrs[1]));
        linv3_probe::put(_wt,
                         linv3_probe::WT_SRC_OFF,
                         static_cast<unsigned long long>(_src_off));
        linv3_probe::put(
            _wt,
            linv3_probe::WT_N_REAL,
            static_cast<unsigned long long>(static_cast<unsigned int>(N_real)));
        linv3_probe::put(
            _wt,
            linv3_probe::WT_K,
            static_cast<unsigned long long>(static_cast<unsigned int>(K)));
        linv3_probe::put(_wt,
                         linv3_probe::WT_SEQ_IN_ITER,
                         static_cast<unsigned long long>(
                             static_cast<unsigned int>(_linv3_seq_in_iter)));
        linv3_probe::put(_wt,
                         linv3_probe::WT_ITER_NUM,
                         static_cast<unsigned long long>(
                             static_cast<unsigned int>(iter_num)));
        // M3 extension fields.
        linv3_probe::put(_wt,
                         linv3_probe::WT_LOAD_IDX,
                         static_cast<unsigned long long>(
                             static_cast<unsigned int>(t * c.iters + i)));
        linv3_probe::put(_wt, linv3_probe::WT_BAD_FLAG, _wt_this_bad);
        linv3_probe::put(_wt,
                         linv3_probe::WT_SRC_LIMIT,
                         static_cast<unsigned long long>(_buf_bytes));
        linv3_probe::put(_wt,
                         linv3_probe::WT_NUM_LOADS,
                         static_cast<unsigned long long>(
                             static_cast<unsigned int>(_wt_num_loads)));
        // Reset WT_COMPLETED to SENTINEL for THIS newly-recorded load (a prior
        // good load may have stamped it COMPLETED); the post-issue stamp below
        // re-sets it iff THIS load's issue returns. Written BEFORE WT_MAGIC.
        linv3_probe::put(_wt, linv3_probe::WT_COMPLETED, linv3_probe::SENTINEL);
        linv3_probe::put(
            _wt, linv3_probe::WT_MAGIC, linv3_probe::WT_SLOT_MAGIC);
        _wt_this_load_recorded = true;
        // Freeze on the FIRST bad load so later (good) loads in this task — and
        // later tasks (via the start-of-loop WT_BAD_FLAG read) — don't clobber
        // the suspected faulter's operands.
        if (_wt_this_bad != 0ull) {
          _wt_frozen = true;
        }
      }
#endif
      // W TMA — v2 order: cp.async.bulk first, expect_tx after.
      tma_3d_load_l2(
          W_smem, W_tmap_ptr, 0, cur_off_m, z_coord, pW.full_mbar(), W_HINT);
      mbarrier_arrive_expect_tx(pW.full_mbar(), W_SIZE);
#ifdef MPK_V2_LINV3_PROBE
      // The RECORDED load's W-TMA issue returned without a SYNCHRONOUS fault
      // from this thread: stamp WT_COMPLETED so the host can tell a survived
      // load from the faulting one (recorded-but-not-completed). Stamp ONLY
      // when THIS load is the one just written to the slot (else a good load
      // after a frozen bad one would wrongly mark the bad record completed).
      // NOTE: cp.async.bulk.tensor is async — a bad-operand fault can surface
      // later; the FROZEN operands still pin the bad argument. We do NOT stamp
      // COMPLETED for a frozen-bad load (it is the suspected illegal access).
      if (_wt_slot != nullptr && _wt_this_load_recorded &&
          _wt_this_bad == 0ull) {
        linv3_probe::put(_wt_slot,
                         linv3_probe::WT_COMPLETED,
                         linv3_probe::WT_COMPLETED_MAGIC);
      }
      if (_linv3_hit && t == 0 && i == 0) {
        linv3_probe::mark(linv3_probe::base(runtime_config),
                          linv3_probe::S_LD_PHASE,
                          linv3_probe::LD_AFTER_FIRST_WTMA);
      }
#endif

      // Cross-SM dep wait once (gates A — matches v2's prefetch pattern).
      if (!dep_done) {
        mirage::runtime_v2::wait_task_dependency(
            runtime_config, task_desc, iter_num);
        dep_done = true;
#ifdef MPK_V2_LINV3_PROBE
        if (_linv3_hit) {
          linv3_probe::mark(linv3_probe::base(runtime_config),
                            linv3_probe::S_LD_PHASE,
                            linv3_probe::LD_AFTER_DEP_WAIT);
        }
#endif
      }

      // A TMA — A shares the empty edge (already waited via pW), so no separate
      // wait. Storage addr comes from A's ring at A's cursor stage.
      int const A_smem = Ar.slot_addr(pA.st);
      tma_3d_load_l2(
          A_smem, A_tmap_ptr, 0, 0, z_coord, pA.full_mbar(), L2_EVICT_LAST);
      mbarrier_arrive_expect_tx(pA.full_mbar(), A_SIZE);

      // Advance both cursors in lockstep.
      pW.commit_tma();
      pA.commit_tma();
    }
  }
#ifdef MPK_V2_LINV3_PROBE
  if (_linv3_hit) {
    linv3_probe::mark(linv3_probe::base(runtime_config),
                      linv3_probe::S_LD_PHASE,
                      linv3_probe::LD_LOOP_DONE);
  }
#endif
  // No end-of-loader drain: blocking on the launcher's final mma_mbar (the
  // shared W/A empty edge) at task end deadlocks against cross-task slot reuse.
  // Stale arrivals are handled by the start-of-task re-init instead.
}

// ═══════════════════════════════════════════════════════════════════════════
// Launcher role (warp 5, all 32 lanes — alloc/dealloc are sync.aligned).
// ═══════════════════════════════════════════════════════════════════════════
template <int SPLIT_K = 1, int TILES_PER_TASK = 1>
__device__ __noinline__ void
    linear_launcher_task(mirage::runtime::TaskDesc const *task_desc,
                         mirage::runtime_v2::RuntimeSMEM *runtime_smem,
                         int N_real,
                         int K,
                         int tile_idx,
                         int dyn_sem_base
#if defined(MPK_V2_LINV3_PROBE) || defined(MPK_V2_LINV3_SKIP36) ||             \
    defined(MPK_V2_LINV3_SKIPALL)
                         ,
                         int _linv3_seq_in_iter,
                         int _linv3_iter_num,
                         mirage::runtime::RuntimeConfig const &runtime_config
#endif
    ) {
  int const lane_id = threadIdx.x & 31;

  extern __shared__ __align__(1024) char smem_ptr[];
  // 1024-align the dynamic SMEM base (see aligned_smem_base note above). Both
  // the int `smem` addr (used for the W/A ring + tcgen05.alloc scratch) and the
  // `char*` taddr readback below derive from this SAME rounded base.
  int const smem = aligned_smem_base(smem_ptr);
  char *const smem_aligned_ptr = aligned_smem_ptr(smem_ptr);

#if defined(MPK_V2_LINV3_PROBE) || defined(MPK_V2_LINV3_SKIP36) ||             \
    defined(MPK_V2_LINV3_SKIPALL)
  // Compute the gate + the SKIP stub BEFORE the bounds-fail early return: if
  // the copied-metadata bug is exactly task_offset>=num_tiles, the real body
  // would take the bounds-fail return (skipping its page sweep + CONSUMER_DONE
  // arrive = a slot wedge). The SKIP diagnostic must instead run its full sync
  // skeleton regardless, so it produces a clean vanish/persist verdict, not a
  // wedge.
  bool const _linv3_hit = linv3_probe::is_target(_linv3_seq_in_iter, task_desc);
  bool const _linv3_skip =
      linv3_probe::should_skip(_linv3_seq_in_iter, task_desc);
  (void)_linv3_hit;
  (void)_linv3_skip;
#endif
#ifdef MPK_V2_LINV3_PROBE
  if (_linv3_hit && lane_id == 0) {
    // Stamp iter FIRST so the phase word is always paired with a fresh iter.
    linv3_probe::put(linv3_probe::base(runtime_config),
                     linv3_probe::S_LC_ITER,
                     static_cast<unsigned>(_linv3_iter_num));
    linv3_probe::mark(linv3_probe::base(runtime_config),
                      linv3_probe::S_LC_PHASE,
                      linv3_probe::LC_BODY_ENTERED);
  }
#endif

#if defined(MPK_V2_LINV3_SKIP36) || defined(MPK_V2_LINV3_SKIPALL)
  // NO-OP the target task's launcher: skip tcgen05.alloc/dealloc + the MMA
  // loop, but preserve the EXACT op-private + page sync the real body owns so
  // the consumer + the next slot occupant proceed normally:
  //   1. lane0 reinit_for_role(Launcher)         — clears strays (LOAD-BEARING:
  //                                                 re-inits
  //                                                 SEM_CONSUMER_DONE).
  //   2. lane0 arrive SEM_TMEM_READY             — the consumer waits it.
  //   3. lane-parallel release ALL pages (== real body's !Wr.owns(lane) sweep,
  //      which frees all 14 at PAGES_PER_SLOT=0).
  //   4. lane0 wait SEM_CONSUMER_DONE            — the consumer arrives it
  //   x128.
  //   5. return (no dealloc — no alloc happened).
  // Gate on _linv3_skip: worker36/seq==4 under SKIP36, EVERY linear_v3 task
  // under SKIPALL (identical stub, broader condition).
  if (_linv3_skip) {
    if (lane_id == 0) {
      ::kernel::linear::reinit_for_role(::kernel::linear::Role::Launcher,
                                        dyn_sem_base);
    }
    __syncwarp();
    if (lane_id == 0) {
      mbarrier_arrive(dyn_sem_base + SEM_TMEM_READY * 8);
    }
    __syncwarp();
#ifdef MPK_V2_LINV3_PROBE
    if (_linv3_hit && lane_id == 0) {
      linv3_probe::mark(linv3_probe::base(runtime_config),
                        linv3_probe::S_LC_PHASE,
                        linv3_probe::LC_SKIP_TMEM_ARRIVED);
    }
#endif
    // Same page-release condition as the real body's task-end sweep.
    WChan _Wc;
    AChan _Ac;
    WRing _Wr;
    ARing _Ar;
    make_wa(smem, dyn_sem_base, task_desc, _Wc, _Ac, _Wr, _Ar);
    if (lane_id < MAX_SMEM_PAGES_PER_TASK && !_Wr.owns(lane_id)) {
      mirage::runtime_v2::runtime_finish_page(runtime_smem, lane_id, 1);
    }
    __syncwarp();
#ifdef MPK_V2_LINV3_PROBE
    if (_linv3_hit && lane_id == 0) {
      linv3_probe::mark(linv3_probe::base(runtime_config),
                        linv3_probe::S_LC_PHASE,
                        linv3_probe::LC_SKIP_PAGES_RELEASED);
    }
#endif
    if (lane_id == 0) {
      mbarrier_wait(dyn_sem_base + SEM_CONSUMER_DONE * 8, 0);
    }
    __syncwarp();
#ifdef MPK_V2_LINV3_PROBE
    if (_linv3_hit && lane_id == 0) {
      linv3_probe::mark(linv3_probe::base(runtime_config),
                        linv3_probe::S_LC_PHASE,
                        linv3_probe::LC_SKIP_RETURN);
    }
#endif
    return;
  }
#endif // MPK_V2_LINV3_SKIP36 || MPK_V2_LINV3_SKIPALL

  const TaskCtx c = ctx_from<SPLIT_K, TILES_PER_TASK>(N_real, K, tile_idx);
  // NOTE page protocol: this early return skips the blanket page-free below,
  // so a task that actually bounds-fails would desync page_finished parity
  // and deadlock the next task on this slot. Unreachable at tiles_per_task=1
  // (task count == tile count, see header USAGE note); must be revisited if
  // tiles_per_task>1 is ever fixed.
  if (c.bounds_fail(tile_idx)) {
    return;
  }

  MPK_V2_PROF_SNAPSHOT()

#ifdef MPK_V2_LINV3_PROBE
  if (_linv3_hit && lane_id == 0) {
    unsigned long long *_bp = linv3_probe::base(runtime_config);
    // SCRATCH region — read WITHOUT smem_region_offset() unless present
    // (guard).
    if (task_desc->num_smem_regions > ::kernel::linear::REGION_SCRATCH) {
      mirage::runtime::SmemPageRegionDesc const &_sr =
          task_desc->smem_regions[::kernel::linear::REGION_SCRATCH];
      linv3_probe::put(
          _bp,
          linv3_probe::S_LC_SCRATCH_OFF,
          static_cast<unsigned>(_sr.physical_page_start *
                                    mirage::runtime::TASK_SMEM_PAGE_SIZE +
                                _sr.byte_offset));
      linv3_probe::put(_bp,
                       linv3_probe::S_LC_SCRATCH_PG,
                       static_cast<unsigned>(_sr.physical_page_start));
    } else {
      linv3_probe::put(_bp, linv3_probe::S_LC_SCRATCH_OFF, ~0ull);
      linv3_probe::put(_bp, linv3_probe::S_LC_SCRATCH_PG, ~0ull);
    }
    linv3_probe::put(_bp,
                     linv3_probe::S_LC_SMEM_BASE_LO12,
                     static_cast<unsigned>(smem & 0xFFF));
  }
#endif

  // Launcher re-init from CHANNELS/ONESHOT reinit_*_by policy (table-driven,
  // Phase 2b).
  if (lane_id == 0) {
    ::kernel::linear::reinit_for_role(::kernel::linear::Role::Launcher,
                                      dyn_sem_base);
  }
  __syncwarp();
#ifdef MPK_V2_LINV3_PROBE
  if (_linv3_hit && lane_id == 0) {
    linv3_probe::mark(linv3_probe::base(runtime_config),
                      linv3_probe::S_LC_PHASE,
                      linv3_probe::LC_AFTER_REINIT);
  }
#endif

  // ── TMEM alloc + publish (v2-identical) ─────────────────────────────────
  int const scratch_smem_addr =
      smem + task_desc->smem_region_offset(::kernel::linear::REGION_SCRATCH);
  asm volatile(
      "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(
          scratch_smem_addr),
      "r"(BLOCK_N * 2));
  int const taddr = *reinterpret_cast<int *>(
      smem_aligned_ptr +
      task_desc->smem_region_offset(::kernel::linear::REGION_SCRATCH));
#ifdef MPK_V2_LINV3_PROBE
  if (_linv3_hit && lane_id == 0) {
    unsigned long long *_bp = linv3_probe::base(runtime_config);
    linv3_probe::put(
        _bp, linv3_probe::S_LC_TADDR, static_cast<unsigned>(taddr));
    linv3_probe::mark(
        _bp, linv3_probe::S_LC_PHASE, linv3_probe::LC_AFTER_ALLOC);
  }
#endif
  if (lane_id == 0) {
    mbarrier_arrive(dyn_sem_base + SEM_TMEM_READY * 8);
  }
#ifdef MPK_V2_LINV3_PROBE
  if (_linv3_hit && lane_id == 0) {
    linv3_probe::mark(linv3_probe::base(runtime_config),
                      linv3_probe::S_LC_PHASE,
                      linv3_probe::LC_TMEM_READY_ARRIVED);
  }
#endif

  // ── Channels (sync) + rings (storage) + cursors ─────────────────────────
  WChan Wc;
  AChan Ac;
  WRing Wr;
  ARing Ar;
  make_wa(smem, dyn_sem_base, task_desc, Wc, Ac, Wr, Ar);
  mpk::ch::Consumer<WChan> cW{Wc};
  mpk::ch::Consumer<AChan> cA{Ac};
  // Both ph start at 0 — mirrors v2's `tma_phase = 0`.

  AccChan Acc = make_acc_channel(dyn_sem_base);
  mpk::ch::TmemProducer<AccChan> pAcc{Acc};
  pAcc.set_taddr(taddr);
  // pAcc.ph starts at 1 — mirrors v2's `epilogue_phase = 1` (pre-empty).

  int const total_g = c.tiles * c.iters;
  if (elect_sync()) {
    for (int t = 0; t < c.tiles; t++) {
      // Wait epilogue_mbar (TMEM column free); returns column for tile t.
      // (timed-waits below: elected lane == the launcher-phase track writer.)
      int tmem;
      MPK_V2_TIMED_WAIT_IF(true,
                           V2_PROF_GROUP_LAUNCHER_PHASE,
                           V2_PROF_EPILOGUE_WAIT,
                           tmem = pAcc.wait_free());

      for (int i = 0; i < c.iters; i++) {
        // Time only the W wait (representative; A shares the edge). Plain cA
        // wait stays a separate statement so the unprofiled expansion is
        // exactly `cW.wait_full(); cA.wait_full();` — textually identical to
        // baseline (sm100 is sensitive to branches around tcgen05 waits;
        // see sm100_branch_ima).
        MPK_V2_TIMED_WAIT_IF(t * c.iters + i < NUM_STAGES,
                             V2_PROF_GROUP_LAUNCHER_PHASE,
                             V2_PROF_W_TMA_WAIT,
                             cW.wait_full()); // W_tma_mbar[stage]
        cA.wait_full();                       // A_tma_mbar[stage]
        int const W_smem = Wr.slot_addr(cW.st);
        int const A_smem = Ar.slot_addr(cA.st);

        tcgen05_fence_after_thread_sync();

        // Same descriptor math + tcgen05_mma sequence as v2. `i != 0` produces
        // identical PTX to v2's pass-through of `i` (setp.ne treats any nonzero
        // as accumulate).
        mma_k_block(tmem, W_smem, A_smem, /*accumulate=*/i != 0);

        int const stg = cW.st; // stage consumed this iter
        // Release SHARED mma_mbar ONCE per iter — matches v2's single commit.
        cW.release_mma();
        cA.advance();

        // Free this W stage's (dedicated) pages at its LAST use → the next
        // task's loader can TMA its weights into them while we finish later
        // stages. The last NUM_STAGES global iters visit each stage once.
        if constexpr (CROSS_TASK_PAGES) {
          if (t * c.iters + i >= total_g - NUM_STAGES) {
            Wr.release(
                stg, runtime_smem, mirage::runtime_v2::runtime_finish_page);
          }
        }
      }

      // Signal mainloop_mbar — async tcgen05.commit; consumer waits this.
      pAcc.commit_mma();
    }
    // W stages never visited (only when total_g < NUM_STAGES) are freed here so
    // every W page is released exactly once. Empty (zero cost) when
    // total_g >= NUM_STAGES, which holds for every real linear; disjoint from
    // the in-loop releases above, so no double-free.
    if constexpr (CROSS_TASK_PAGES) {
      for (int s = total_g; s < NUM_STAGES; s++) {
        Wr.release(s, runtime_smem, mirage::runtime_v2::runtime_finish_page);
      }
    }
  }
#ifdef MPK_V2_LINV3_PROBE
  if (_linv3_hit && lane_id == 0) {
    linv3_probe::mark(linv3_probe::base(runtime_config),
                      linv3_probe::S_LC_PHASE,
                      linv3_probe::LC_MMA_LOOP_DONE);
  }
#endif

  // Reconverge before freeing pages. The MMA loop ran only on the elected
  // lane; under Volta+ ITS the other lanes are not rejoined at the if-block
  // exit, so without this they could arrive page_finished while the elected
  // lane's MMA is still reading those pages — letting the next task's loader
  // TMA into them mid-MMA.
  __syncwarp();

  // Task-end page free, PARALLEL across lanes: each lane frees its own page
  // unless the W ring already freed it per-stage. When cross-task is off, the W
  // ring owns nothing (owns()==false) → this frees all 14 = baseline. A's pages
  // and scratch's (which shares one) are freed here at task end, which is safe
  // for scratch's whole-task lifetime.
  if (lane_id < MAX_SMEM_PAGES_PER_TASK && !Wr.owns(lane_id)) {
    mirage::runtime_v2::runtime_finish_page(runtime_smem, lane_id, 1);
  }
  __syncwarp();
#ifdef MPK_V2_LINV3_PROBE
  if (_linv3_hit && lane_id == 0) {
    linv3_probe::mark(linv3_probe::base(runtime_config),
                      linv3_probe::S_LC_PHASE,
                      linv3_probe::LC_PAGES_RELEASED);
  }
#endif

  // Wait consumer_done — one-shot, kept raw (matches v2).
  if (lane_id == 0) {
    MPK_V2_TIMED_WAIT_IF(
        true,
        V2_PROF_GROUP_LAUNCHER_PHASE,
        V2_PROF_CONSUMER_DONE_WAIT,
        mbarrier_wait(dyn_sem_base + SEM_CONSUMER_DONE * 8, 0));
  }
  __syncwarp();
#ifdef MPK_V2_LINV3_PROBE
  if (_linv3_hit && lane_id == 0) {
    linv3_probe::mark(linv3_probe::base(runtime_config),
                      linv3_probe::S_LC_PHASE,
                      linv3_probe::LC_CONSUMER_DONE_WAITED);
  }
#endif

  asm volatile(
      "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr),
      "r"(BLOCK_N * 2));
#ifdef MPK_V2_LINV3_PROBE
  if (_linv3_hit && lane_id == 0) {
    linv3_probe::mark(linv3_probe::base(runtime_config),
                      linv3_probe::S_LC_PHASE,
                      linv3_probe::LC_AFTER_DEALLOC);
  }
#endif
}

// ═══════════════════════════════════════════════════════════════════════════
// Consumer role (warps 0–3, 128 threads).
// ═══════════════════════════════════════════════════════════════════════════
template <bool HAS_RESIDUAL,
          int M_REAL = 16,
          int SPLIT_K = 1,
          int TILES_PER_TASK = 1>
__device__ __noinline__ void
    linear_consumer_task(mirage::runtime::TaskDesc const *task_desc,
                         nv_bfloat16 *C_ptr,
                         nv_bfloat16 const *res_ptr,
                         int N_real,
                         int K,
                         int tile_idx,
                         float *workspace,
                         int dyn_sem_base
#if defined(MPK_V2_LINV3_PROBE) || defined(MPK_V2_LINV3_SKIP36) ||             \
    defined(MPK_V2_LINV3_SKIPALL)
                         ,
                         int _linv3_seq_in_iter,
                         int _linv3_iter_num,
                         mirage::runtime::RuntimeConfig const &runtime_config
#endif
    ) {
  int const warp_id = warp_uniform(threadIdx.x / WARP_SIZE);
  int const lane_id = threadIdx.x & 31;

  extern __shared__ __align__(1024) char smem_ptr[];

#if defined(MPK_V2_LINV3_PROBE) || defined(MPK_V2_LINV3_SKIP36) ||             \
    defined(MPK_V2_LINV3_SKIPALL)
  // Gate + SKIP stub BEFORE the bounds-fail early return (see the launcher
  // note): a task_offset>=num_tiles metadata bug must still run the full skip
  // sync skeleton, not take the wedge-prone bounds-fail return.
  bool const _linv3_hit = linv3_probe::is_target(_linv3_seq_in_iter, task_desc);
  bool const _linv3_skip =
      linv3_probe::should_skip(_linv3_seq_in_iter, task_desc);
  (void)_linv3_hit;
  (void)_linv3_skip;
#endif
#ifdef MPK_V2_LINV3_PROBE
  if (_linv3_hit && threadIdx.x == 0) {
    unsigned long long *_bp = linv3_probe::base(runtime_config);
    // Stamp iter FIRST so the phase word is always paired with a fresh iter.
    linv3_probe::put(
        _bp, linv3_probe::S_CN_ITER, static_cast<unsigned>(_linv3_iter_num));
    linv3_probe::mark(
        _bp, linv3_probe::S_CN_PHASE, linv3_probe::CN_BODY_ENTERED);
    linv3_probe::put(_bp,
                     linv3_probe::S_CN_C_PTR,
                     reinterpret_cast<unsigned long long>(C_ptr));
  }
#endif

#if defined(MPK_V2_LINV3_SKIP36) || defined(MPK_V2_LINV3_SKIPALL)
  // NO-OP the target task's consumer: the codegen consumer_dep_prefix already
  // arrived/waited SEM_DEP_READY (do NOT touch it). Preserve only the
  // op-private handshake the body owns: lane0-of-each-warp waits SEM_TMEM_READY
  // (the launcher arrives it), then ALL 128 threads arrive SEM_CONSUMER_DONE
  // (the launcher waits it x128). Skip the taddr read + tcgen05.ld + the
  // stores. Gate on _linv3_skip: worker36/seq==4 under SKIP36, EVERY linear_v3
  // task under SKIPALL (identical stub, broader condition).
  if (_linv3_skip) {
    if (lane_id == 0) {
      mbarrier_wait(dyn_sem_base + SEM_TMEM_READY * 8, 0);
    }
    __syncwarp();
    mbarrier_arrive(dyn_sem_base + SEM_CONSUMER_DONE * 8);
#ifdef MPK_V2_LINV3_PROBE
    if (_linv3_hit && threadIdx.x == 0) {
      linv3_probe::mark(linv3_probe::base(runtime_config),
                        linv3_probe::S_CN_PHASE,
                        linv3_probe::CN_SKIP_RETURN);
    }
#endif
    return;
  }
#endif // MPK_V2_LINV3_SKIP36 || MPK_V2_LINV3_SKIPALL

  const TaskCtx c = ctx_from<SPLIT_K, TILES_PER_TASK>(N_real, K, tile_idx);
  if (c.bounds_fail(tile_idx)) {
    return;
  }

  MPK_V2_PROF_SNAPSHOT()

  // Wait launcher's TMEM-addr publish — one-shot, kept raw (matches v2).
  // (timed-wait on thread 0 only — all four consumer warps' lane 0 wait,
  // but the consumer-phase track has a single designated writer.)
  if (lane_id == 0) {
    MPK_V2_TIMED_WAIT_IF(threadIdx.x == 0,
                         V2_PROF_GROUP_CONSUMER_PHASE,
                         V2_PROF_TMEM_READY_WAIT,
                         mbarrier_wait(dyn_sem_base + SEM_TMEM_READY * 8, 0));
  }
  __syncwarp();
#ifdef MPK_V2_LINV3_PROBE
  if (_linv3_hit && threadIdx.x == 0) {
    linv3_probe::mark(linv3_probe::base(runtime_config),
                      linv3_probe::S_CN_PHASE,
                      linv3_probe::CN_TMEM_READY_WAITED);
  }
#endif
  // Read the TMEM addr the launcher published into the SCRATCH region, using
  // the SAME 1024-rounded base the launcher wrote through (see
  // aligned_smem_base).
  int const taddr = *reinterpret_cast<int *>(
      aligned_smem_ptr(smem_ptr) +
      task_desc->smem_region_offset(::kernel::linear::REGION_SCRATCH));
#ifdef MPK_V2_LINV3_PROBE
  if (_linv3_hit && threadIdx.x == 0) {
    unsigned long long *_bp = linv3_probe::base(runtime_config);
    linv3_probe::put(
        _bp, linv3_probe::S_CN_TADDR, static_cast<unsigned>(taddr));
    linv3_probe::mark(
        _bp, linv3_probe::S_CN_PHASE, linv3_probe::CN_AFTER_TADDR_READ);
  }
#endif

  AccChan Acc = make_acc_channel(dyn_sem_base);
  mpk::ch::TmemConsumer<AccChan> cAcc{Acc};
  cAcc.set_taddr(taddr);
  // cAcc.ph starts at 0 — mirrors v2's `mainloop_phase = 0`.

  for (int t = 0; t < c.tiles; t++) {
    int const cur_tile_idx = tile_idx + t;
    int const cur_spatial_idx = cur_tile_idx % c.num_spatial_tiles;
    int const cur_k_slice = cur_tile_idx / c.num_spatial_tiles;
    int const bid_m = cur_spatial_idx;

    // Wait MMA done for this tile; cursor gives the TMEM column. All 128
    // threads wait; only thread 0 (the consumer-phase writer) times it.
    int t_col;
    MPK_V2_TIMED_WAIT_IF(threadIdx.x == 0,
                         V2_PROF_GROUP_CONSUMER_PHASE,
                         V2_PROF_MAINLOOP_WAIT,
                         t_col = cAcc.wait_full());
#ifdef MPK_V2_LINV3_PROBE
    if (_linv3_hit && threadIdx.x == 0 && t == 0) {
      linv3_probe::mark(linv3_probe::base(runtime_config),
                        linv3_probe::S_CN_PHASE,
                        linv3_probe::CN_BEFORE_FIRST_STORE);
    }
#endif
#ifdef MPK_ENABLE_PROFILING
    // RECONVERGE: in profiling builds the thread-0 timing branch diverges
    // warp 0 (Volta+ ITS doesn't rejoin at the merge) and the tcgen05.ld
    // below needs a converged warp. Unprofiled builds have no branch (the
    // macro is a bare wait), so no reconverge is needed.
    __syncwarp();
#endif

    tcgen05_fence_after_thread_sync();

    int const n_real = bid_m * BLOCK_M + warp_id * 32 + lane_id;
    if (n_real < N_real) {
      int const t_addr = (warp_id * 32 << 16) + t_col;

      float f[16];
      tcgen05_ld_16(f, t_addr);
      tcgen05_wait_ld();

      if constexpr (SPLIT_K == 1) {
        // Precision-clamp round-trip: bf16-quantize GEMM output before adding
        // residual (in float), then bf16-quantize again at store. v2 does this
        // exactly; do NOT collapse the round-trip — semantics change.
        if constexpr (HAS_RESIDUAL) {
#pragma unroll
          for (int m = 0; m < M_REAL; m++) {
            nv_bfloat16 gemm_bf16 = __float2bfloat16(f[m]);
            f[m] = __bfloat162float(gemm_bf16) +
                   __bfloat162float(res_ptr[m * N_real + n_real]);
          }
        }
#pragma unroll
        for (int m = 0; m < M_REAL; m++) {
          st_bf16(C_ptr + m * N_real + n_real, __float2bfloat16(f[m]));
        }
      } else {
        float *ws_base = workspace + cur_k_slice * M_REAL * N_real;
#pragma unroll
        for (int m = 0; m < M_REAL; m++) {
          st_f32(ws_base + m * N_real + n_real, f[m]);
        }
      }
    }

#ifdef MPK_V2_LINV3_PROBE
    if (_linv3_hit && threadIdx.x == 0 && t == 0) {
      linv3_probe::mark(linv3_probe::base(runtime_config),
                        linv3_probe::S_CN_PHASE,
                        linv3_probe::CN_AFTER_FIRST_STORE);
    }
#endif
    // Release epilogue_mbar (128-thread sync arrival).
    cAcc.release_warp();
  }
#ifdef MPK_V2_LINV3_PROBE
  if (_linv3_hit && threadIdx.x == 0) {
    linv3_probe::mark(linv3_probe::base(runtime_config),
                      linv3_probe::S_CN_PHASE,
                      linv3_probe::CN_LOOP_DONE);
  }
#endif

  // Signal consumer_done (128 threads, sync) — one-shot, kept raw.
  mbarrier_arrive(dyn_sem_base + SEM_CONSUMER_DONE * 8);
#ifdef MPK_V2_LINV3_PROBE
  if (_linv3_hit && threadIdx.x == 0) {
    linv3_probe::mark(linv3_probe::base(runtime_config),
                      linv3_probe::S_CN_PHASE,
                      linv3_probe::CN_CONSUMER_DONE_ARRIVED);
  }
#endif
}

} // namespace linear_v3
} // namespace kernel
