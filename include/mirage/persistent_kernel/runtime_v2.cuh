#pragma once

#include "mirage/persistent_kernel/mpk_atoms.cuh"
#include "mirage/persistent_kernel/profiler.h"
#include "mirage/persistent_kernel/runtime_header.h"
#include "mirage/persistent_kernel/tasks/blackwell_v2/task_interface.cuh"
#include "mirage/persistent_kernel/tasks/common/copy_sm80.cuh"
#include <cuda_runtime.h>
#include <stdint.h>

// ── Optional per-task profiling (MPK_ENABLE_PROFILING builds only) ──────────
// Reuses v1's FlashInfer-style profiler format so profiler_persistent.py
// renders v2 traces unchanged. FIVE tracks per SM (num_groups = 5) — the SM
// is a pipeline in v2, so each warp role gets its own Perfetto row and the
// task-level overlap between roles is directly visible:
//   group 0 consumer  (warp 0 lane 0): BEGIN/END per task body — includes
//           the dep-wait spin; gaps = waiting on instruction publish
//   group 1 loader    (lane 0): page-prefix + loader body per task
//   group 2 launcher  (lane 0): launcher body (linear: TMEM/MMA driving)
//   group 3 storer    (lane 0): storer body (idle for Qwen3 tasks)
//   group 4 controller(lane 0): V2_PROF_PREPARE_BATCH (worker 0),
//           V2_PROF_ITER_SYNC (end-of-iter barrier), V2_PROF_GO_WAIT
// Only the LAST V2_PROF_WINDOW_ITERS decode steps are recorded: that's the
// interesting regime (full KV length), and it bounds the event count so the
// fixed-size profiler buffer can't overflow on long runs.
// Non-profiling builds: all macros expand to nothing — zero impact.
static constexpr int V2_PROF_NUM_GROUPS = 8;
static constexpr int V2_PROF_GROUP_CONSUMER = 0;
static constexpr int V2_PROF_GROUP_LOADER = 1;
static constexpr int V2_PROF_GROUP_LAUNCHER = 2;
static constexpr int V2_PROF_GROUP_STORER = 3;
static constexpr int V2_PROF_GROUP_CONTROLLER = 4;
// Phase tracks: sub-slices WITHIN a role's task window, so a bar
// self-explains (wait vs work). Written by the closure-free emitter
// (v2_prof_emit*) because their call sites can't see the role-loop closure.
static constexpr int V2_PROF_GROUP_CONSUMER_PHASE = 5;
static constexpr int V2_PROF_GROUP_LOADER_PHASE = 6;
static constexpr int V2_PROF_GROUP_LAUNCHER_PHASE = 7;
static constexpr int V2_PROF_PREPARE_BATCH = 204;
static constexpr int V2_PROF_ITER_SYNC = 205;
static constexpr int V2_PROF_GO_WAIT = 206;
static constexpr int V2_PROF_DEP_WAIT = 207;  // consumer-phase: dep spin
static constexpr int V2_PROF_PAGE_WAIT = 208; // loader-phase: page prefix
// Timed-wait ids (emitted by MPK_V2_TIMED_WAIT at synchronization points;
// only waits > V2_PROF_WAIT_THRESHOLD_NS produce slices):
static constexpr int V2_PROF_W_TMA_WAIT = 209;         // launcher: TMA landed?
static constexpr int V2_PROF_MMA_EMPTY_WAIT = 210;     // loader: stage free?
static constexpr int V2_PROF_TMEM_READY_WAIT = 211;    // consumer: tmem addr
static constexpr int V2_PROF_MAINLOOP_WAIT = 212;      // consumer: MMA result
static constexpr int V2_PROF_EPILOGUE_WAIT = 213;      // launcher: tmem slot
static constexpr int V2_PROF_CONSUMER_DONE_WAIT = 214; // launcher tail
static constexpr unsigned long long V2_PROF_WAIT_THRESHOLD_NS = 2000;
// Total profiler-buffer entries — MUST match demo.py's profiler_tensor size.
// Sized for 8 tracks x up to V2_PROF_SM_SLOTS SMs x 25 windowed iters; the
// busiest track (consumer-phase: dep + tmem + mainloop slices) can write
// ~12-17k entries, so per-track capacity = (ENTRIES - tail) / (nblocks*8)
// ≈ 13-15k. The emitter counts (never silently drops) overflow in MISC.
static constexpr size_t V2_PROF_BUF_ENTRIES = 120000ull * 128;
// Per-SM slot count for every per-SM tail array below. MUST be >= the
// launch's worker count: B200 production runs 136 workers, and the original
// 128-slot arrays made blocks 128-135 alias the spin accumulators through
// their emitter cursors (junk cursor values -> capacity-guard drops) and
// write their page-suffix counts PAST the buffer end. 256 covers any
// current/near-future part.
static constexpr int V2_PROF_SM_SLOTS = 256;
// Tail of the profiler buffer reserved for accumulators (all in NANOSECONDS
// via %globaltimer — same timebase as the trace events, no clock-rate
// conversion). CONVENTION with demo.py's profiler_tensor; debug-only.
// Layout, growing back from the end (SLOTS = V2_PROF_SM_SLOTS):
//   [SPIN_BASE + bucket*2*SLOTS + sm]         dep-wait ns, per task-type bucket
//   [SPIN_BASE + bucket*2*SLOTS + SLOTS + sm] dep-wait count
//   [SUFFIX_BASE + sm]                        page-suffix ns, per SM
//   [SUFFIX_BASE + SLOTS + sm]                page-suffix count
// Type buckets: 0=linear(v3+res) 1=attn 2=rmsnorm 3=silu
//               4=argmax(partial/reduce) 5=embed 6=other
// (Bucketed by TASK_*_V2 enum NAME below — ids were renumbered in the v2
// merge, so never hardcode the numeric values here.)
static constexpr int V2_PROF_NUM_BUCKETS = 7;
static constexpr size_t V2_PROF_SUFFIX_BASE =
    V2_PROF_BUF_ENTRIES - 2ull * V2_PROF_SM_SLOTS;
static constexpr size_t V2_PROF_SPIN_BASE =
    V2_PROF_SUFFIX_BASE - 2ull * V2_PROF_SM_SLOTS * V2_PROF_NUM_BUCKETS;
// Per-track write cursors for the closure-free emitter (8 groups x
// V2_PROF_SM_SLOTS), then a misc region: [MISC_BASE + sm] counts events
// DROPPED by the capacity guard (must be 0 in a healthy run — the checker
// reports it). Everything from MISC_BASE on is reserved tail — the exporter
// skips it (V2_PROF_TAIL_ENTRIES in profiler_persistent.py must match).
static constexpr size_t V2_PROF_CURSOR_BASE =
    V2_PROF_SPIN_BASE - 8ull * V2_PROF_SM_SLOTS;
static constexpr size_t V2_PROF_MISC_BASE =
    V2_PROF_CURSOR_BASE - V2_PROF_SM_SLOTS;
// The tag's block_group field is 11 bits (2048 tracks); the per-SM cursor
// region must stay addressable through it.
static_assert(V2_PROF_SM_SLOTS * 8 <= 2048,
              "V2_PROF_SM_SLOTS x 8 role/phase tracks must fit the 11-bit "
              "block_group tag field");
// Event-trigger log: a single global ring recording every
// trigger_task_event() fired inside the profiling window, packed as
// [63:32]=globaltimer_lo  [31:8]=event_index  [7:0]=sm. One atomic cursor.
// Diagnoses trigger latency (when did event E actually fire, and from
// which SM's controller).
static constexpr size_t V2_PROF_TRIG_RING_LEN = 1048576;
static constexpr size_t V2_PROF_TRIG_BASE =
    V2_PROF_MISC_BASE - V2_PROF_TRIG_RING_LEN;
static constexpr size_t V2_PROF_TRIG_CURSOR = V2_PROF_TRIG_BASE - 1;
static constexpr size_t V2_PROF_TAIL_ENTRIES =
    V2_PROF_BUF_ENTRIES - V2_PROF_TRIG_CURSOR;
static constexpr int V2_PROF_WINDOW_ITERS = 25;

__device__ __forceinline__ unsigned long long v2_prof_now_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ int v2_prof_bucket(int task_type) {
  using namespace mirage::runtime;
  switch (task_type) {
    case TASK_LINEAR_SM100_V3:
    case TASK_LINEAR_WITH_RESIDUAL_SM100_V3:
      return 0; // linear v3 (+residual)
    case TASK_ATTN_SM100_V2:
      return 1; // attention
    case TASK_RMS_NORM_HOPPER_V2:
      return 2; // rmsnorm
    case TASK_SILU_MUL_V2:
      return 3; // silu_mul
    case TASK_ARGMAX_PARTIAL_SM100_V2:
    case TASK_ARGMAX_REDUCE_SM100_V2:
      return 4; // argmax
    case TASK_EMBEDDING_V2:
      return 5; // embedding
    default:
      return 6;
  }
}

#ifdef MPK_ENABLE_PROFILING
// Closure-free trace-event emitter for sites that cannot see the role-loop
// profiler closure (runtime helpers like consumer_dep_prefix, codegen-emitted
// prefixes). Maintains per-track cursors in the reserved tail; the entry
// layout matches PROFILER_INIT's interleaving exactly, so the exporter needs
// no changes. Single writer per phase track (a designated lane), so the
// cursor bump needs no atomics. NEVER write a role group (0-4) through this
// — those tracks are owned by the role-loop closures.
__device__ __forceinline__ void v2_prof_emit(void *prof_buf,
                                             int group,
                                             uint32_t event_idx,
                                             uint32_t event_type) {
  uint64_t *buf = static_cast<uint64_t *>(prof_buf);
  unsigned int const track = blockIdx.x * V2_PROF_NUM_GROUPS + group;
  unsigned long long const k = buf[V2_PROF_CURSOR_BASE + track]++;
  unsigned int const stride = gridDim.x * V2_PROF_NUM_GROUPS;
  size_t const slot = 1 + (size_t)k * stride + track;
  if (slot >= V2_PROF_MISC_BASE) {
    // capacity guard: never bleed into the reserved tail — but COUNT the
    // drop so the checker can flag a truncated trace.
    atomicAdd(reinterpret_cast<unsigned long long *>(
                  &buf[V2_PROF_MISC_BASE + blockIdx.x]),
              1ULL);
    return;
  }
  uint32_t const event_no = (uint32_t)(k >> 1) & 0x3FF; // pair counter
  tb::ProfilerEntry e;
  e.tag = tb::encode_tag(track, 0, 0) | (event_idx << tb::EVENT_IDX_SHIFT) |
          (event_no << tb::EVENT_NO_SHIFT) | event_type;
  e.delta_time = tb::get_timestamp();
  buf[slot] = e.raw;
}

// Retro-emit a BEGIN/END pair with explicit timestamps — used by
// MPK_V2_TIMED_WAIT after a wait completes, so a slice is written only when
// the wait was long enough to matter and pairs can never dangle.
__device__ __forceinline__ void v2_prof_emit_pair(void *prof_buf,
                                                  int group,
                                                  uint32_t event_idx,
                                                  unsigned long long t0_ns,
                                                  unsigned long long t1_ns) {
  uint64_t *buf = static_cast<uint64_t *>(prof_buf);
  unsigned int const track = blockIdx.x * V2_PROF_NUM_GROUPS + group;
  unsigned long long const k = buf[V2_PROF_CURSOR_BASE + track];
  buf[V2_PROF_CURSOR_BASE + track] = k + 2;
  unsigned int const stride = gridDim.x * V2_PROF_NUM_GROUPS;
  size_t const slot0 = 1 + (size_t)k * stride + track;
  size_t const slot1 = slot0 + stride;
  if (slot1 >= V2_PROF_MISC_BASE) {
    atomicAdd(reinterpret_cast<unsigned long long *>(
                  &buf[V2_PROF_MISC_BASE + blockIdx.x]),
              2ULL);
    return;
  }
  uint32_t const event_no = (uint32_t)(k >> 1) & 0x3FF;
  uint32_t const base = tb::encode_tag(track, 0, 0) |
                        (event_idx << tb::EVENT_IDX_SHIFT) |
                        (event_no << tb::EVENT_NO_SHIFT);
  tb::ProfilerEntry e;
  e.tag = base | tb::EVENT_BEGIN;
  e.delta_time = (uint32_t)t0_ns;
  buf[slot0] = e.raw;
  e.tag = base | tb::EVENT_END;
  e.delta_time = (uint32_t)t1_ns;
  buf[slot1] = e.raw;
}

// Ambient profiling context, so synchronization sites inside task bodies can
// emit without threading (config, iter_num) through every signature:
//   g_v2_prof_buf    — the profiler buffer (same for all SMs); set once in
//                      the kernel prologue.
//   g_v2_prof_window — 1 while the current decode step is inside the traced
//                      window; flipped by worker 0's controller at the
//                      iteration boundary (all SMs are barrier-synced per
//                      iter, so at most one task of fuzz at the edges).
__device__ void *g_v2_prof_buf = nullptr;
__device__ int volatile g_v2_prof_window = 0;

// Snapshot the ambient window flag ONCE per task body (a volatile global
// read per wait would put L2 traffic in hot loops for the whole run).
// Required in scope before any MPK_V2_TIMED_WAIT.
#define MPK_V2_PROF_SNAPSHOT()                                                 \
  bool const _mpk_prof_on =                                                    \
      (g_v2_prof_window != 0) && (g_v2_prof_buf != nullptr);

// Time `expr` (a wait) and emit a phase slice if it exceeded the threshold.
// Call from a SINGLE thread per (SM, group) — the designated writer of that
// phase track.
#define MPK_V2_TIMED_WAIT(group, ev, expr)                                     \
  do {                                                                         \
    if (_mpk_prof_on) {                                                        \
      unsigned long long const _w0 = v2_prof_now_ns();                         \
      expr;                                                                    \
      unsigned long long const _w1 = v2_prof_now_ns();                         \
      if (_w1 - _w0 > V2_PROF_WAIT_THRESHOLD_NS) {                             \
        v2_prof_emit_pair(g_v2_prof_buf, (group), (ev), _w0, _w1);             \
      }                                                                        \
    } else {                                                                   \
      expr;                                                                    \
    }                                                                          \
  } while (0)
#endif
#ifdef MPK_ENABLE_PROFILING
#define MPK_V2_PROF_DECL(grp, pred)                                            \
  PROFILER_CLOSURE_PARAMS_DECL;                                                \
  uint32_t _prof_ctr = 0;                                                      \
  PROFILER_INIT(static_cast<uint64_t *>(config.profiler_buffer),               \
                (grp),                                                         \
                V2_PROF_NUM_GROUPS,                                            \
                (pred));                                                       \
  bool const _prof_pred_base = profiler_write_thread_predicate;
#define MPK_V2_PROF_IN_WINDOW(it)                                              \
  ((it) + V2_PROF_WINDOW_ITERS >= config.v2_max_iters)
// BRANCHLESS wrap (2026-07-15): the window test used to be an if/else wrapped
// around each task body's START/END events. sm100 codegen is sensitive to
// if/else around tcgen05 waits (see the sm100_branch_ima note below at
// MPK_V2_TIMED_WAIT_IF), and the window-gated branch shape wedged profiled
// runs at gate density on the candidate-free reference linear chain
// (all-role convoy jam, in-window iterations only; unprofiled passed at
// every scale). Fold the window test into the profiler's store predicate
// (profiler_write_thread_predicate) instead: control flow around
// execute_task is now IDENTICAL in- and out-of-window — only the predicate
// VALUE changes. Emitted events are unchanged (stores still fire only
// in-window on the designated writer thread; tags/timestamps/event_no
// counter identical: _prof_ctr advances only in-window, branch-free).
#define MPK_V2_PROF_START(ev)                                                  \
  do {                                                                         \
    profiler_write_thread_predicate =                                          \
        _prof_pred_base && MPK_V2_PROF_IN_WINDOW(iter_num);                    \
    PROFILER_EVENT_START((ev), _prof_ctr);                                     \
  } while (0)
#define MPK_V2_PROF_END(ev)                                                    \
  do {                                                                         \
    profiler_write_thread_predicate =                                          \
        _prof_pred_base && MPK_V2_PROF_IN_WINDOW(iter_num);                    \
    PROFILER_EVENT_END((ev), _prof_ctr);                                       \
    _prof_ctr += (MPK_V2_PROF_IN_WINDOW(iter_num) ? 1u : 0u);                  \
  } while (0)
// Conditionally-timed wait: time `expr` only when `cond` (e.g. cold-start lap),
// else run it plain. The cond/branch exists ONLY in profiling builds; the
// non-profiling form (below) is a bare `expr`, textually identical to baseline
// (sm100 codegen is sensitive to if/else around tcgen05 waits — see
// sm100_branch_ima). Keeps the production hot loop free of #ifdef blocks.
#define MPK_V2_TIMED_WAIT_IF(cond, group, ev, expr)                            \
  do {                                                                         \
    if (cond) {                                                                \
      MPK_V2_TIMED_WAIT(group, ev, expr);                                      \
    } else {                                                                   \
      expr;                                                                    \
    }                                                                          \
  } while (0)
#else
#define MPK_V2_PROF_DECL(grp, pred)
#define MPK_V2_PROF_START(ev)
#define MPK_V2_PROF_END(ev)
#define MPK_V2_PROF_IN_WINDOW(it) (false)
#define MPK_V2_PROF_SNAPSHOT()
#define MPK_V2_TIMED_WAIT_IF(cond, group, ev, expr) expr
#define MPK_V2_TIMED_WAIT(group, ev, expr) expr
#endif

namespace mirage {
namespace runtime_v2 {

using namespace mirage::runtime;

// Clean v2 role runtime.
//
// Warp layout:
//   W0-W3: consumer
//   W4:    loader
//   W5:    launcher
//   W6:    storer
//   W7:    controller
//
// The runtime owns instruction-slot scheduling, graph dependency waiting,
// event triggering, and generic SMEM page semaphores. Task-specific behavior
// must live behind the generated role dispatcher.
static constexpr int NUM_CONSUMER_WARPS = 4;
static constexpr int LOADER_WARP = 4;
static constexpr int LAUNCHER_WARP = 5;
static constexpr int STORER_WARP = 6;
static constexpr int CONTROLLER_WARP = 7;
static constexpr int NUM_ROLE_WARPS = 7;
static constexpr int NUM_WARPS = 8;
static constexpr int NUM_THREADS = NUM_WARPS * 32;

// --- setmaxnreg experiment (TODO #18 / MegaKernels register-split) ---------
// 8 warps = 2 warpgroups: WG0 = warps 0-3 (consumers), WG1 = warps 4-7
// (loader/launcher/storer/controller). setmaxnreg.sync.aligned operates at
// warpgroup granularity, so the inc/dec must be issued by every warp of a
// warpgroup uniformly, at the top of its branch (setmaxnreg_findings.md).
// MPK_SETMAXNREG=0 disables (byte-identical to baseline). Values: multiples
// of 8, consumer >= launch baseline (inc), helper <= baseline (dec).
#ifndef MPK_SETMAXNREG
#define MPK_SETMAXNREG 0
#endif
#ifndef MPK_CONSUMER_REGS
#define MPK_CONSUMER_REGS 256
#endif
#ifndef MPK_HELPER_REGS
#define MPK_HELPER_REGS 64
#endif
// MPK_CONSUMER_REGS 0 => skip the consumer inc entirely (dec-only test: helper
// warps release registers, consumers keep the launch baseline). Otherwise the
// launch baseline must be < the inc target, which requires raising the
// launch_bounds thread count (MPK_LB_THREADS) to lower the per-thread baseline.
#ifndef MPK_LB_THREADS
#define MPK_LB_THREADS NUM_THREADS
#endif
// Pad the launched block with extra (idle) warps to force the kernel past the
// register-file ceiling, so setmaxnreg becomes REQUIRED to fit (the MegaKernels
// regime). Extra warps (warp_id >= NUM_WARPS) just dec their registers and
// return. Default = no padding.
#ifndef MPK_LAUNCH_WARPS
#define MPK_LAUNCH_WARPS NUM_WARPS
#endif
static constexpr int MPK_LAUNCH_THREADS = MPK_LAUNCH_WARPS * 32;
#if MPK_SETMAXNREG && MPK_CONSUMER_REGS > 0
#define MPK_SETMAXNREG_INC(N)                                                  \
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(N))
#else
#define MPK_SETMAXNREG_INC(N)
#endif
#if MPK_SETMAXNREG
#define MPK_SETMAXNREG_DEC(N)                                                  \
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(N))
#else
#define MPK_SETMAXNREG_DEC(N)
#endif
static constexpr int INSTRUCTION_RING_SIZE = 3;
// Match Megakernel's page protocol: page availability is tracked by parity
// semaphores keyed by the instruction index, not by runtime page ownership.
static constexpr int PAGE_SEMAPHORE_BITS = 1;

static constexpr int MBAR_INSTRUCTION_ARRIVED = 0;
static constexpr int MBAR_INSTRUCTION_FINISHED = 1;
static constexpr int NUM_INSTRUCTION_MBARS = 2;

// Per-instruction dynamic semaphores. Each task type may declare an
// op-specific init body that is run by the controller (single thread) once
// per published instruction; the body is free to mbar_init any of these
// slots and any role body can mbar_arrive/mbar_wait on them. Drained
// implicitly when the controller waits on instruction_finished[slot] before
// recycling the slot.
static constexpr int MAX_DYNAMIC_SEMAPHORES = 32;

// Slot conventions for dynamic_semaphores[slot][i]:
//   SEM_DEP_READY — consumer warp 0 lane 0 spins on the cross-SM event
//   counter, then arrives this semaphore. Other consumer warps wait on it
//   so they enter the compute body in lockstep with the dep being cleared.
//   SEM_OP_BASE..MAX_DYNAMIC_SEMAPHORES-1 — op-private slots. Any task
//   type that needs intra-task cross-warp coordination (e.g. linear's
//   per-stage TMA→MMA→epilogue handshakes after Phase 3) uses these.
static constexpr int SEM_DEP_READY = 0;
static constexpr int SEM_OP_BASE = 1;

__device__ __forceinline__ int smem_addr(void const *ptr) {
  return static_cast<int>(__cvta_generic_to_shared(ptr));
}

// Fast, near-non-suspending poll (minimal suspend-time hint). Used by the
// controller's eager trigger sweep, which polls many slots per loop and must
// not suspend ~10M cycles on each not-ready slot.
__device__ __forceinline__ bool mbar_poll(int addr, int phase) {
  int ok;
  asm volatile(
      "{\n\t.reg .pred P;\n\t"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%1], %2, 1;\n\t"
      "selp.b32 %0, 1, 0, P;\n\t}"
      : "=r"(ok)
      : "r"(addr), "r"(phase));
  return ok != 0;
}

__device__ __forceinline__ void mbar_init(uint64_t *mbar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_addr(mbar)),
               "r"(count));
}

__device__ __forceinline__ void mbar_arrive(uint64_t *mbar) {
  // NOTE: plain mbarrier.arrive defaults to .release.cta (verified: ptxas emits
  // byte-identical SASS for the explicit .release form on sm_100a/CUDA 12.8),
  // so role warps' .acquire waits already synchronize-with this arrive.
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_addr(mbar))
               : "memory");
}

__device__ __forceinline__ void mbar_wait(uint64_t *mbar, int phase) {
  int addr = smem_addr(mbar);
  asm volatile("{\n\t.reg .pred P;\n\t"
               "WAIT: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, "
               "[%0], %1, 0x989680;\n\t"
               "@P bra DONE;\n\t"
               "bra WAIT;\n\t"
               "DONE:\n\t}" ::"r"(addr),
               "r"(phase));
}

__device__ __forceinline__ size_t get_task_iteration_num(TaskId task_id) {
  return task_id >> 32;
}

__device__ __forceinline__ size_t get_event_position_index(EventId event_id) {
  return event_id & 0xffffffff;
}

__device__ __forceinline__ bool is_nvshmem_event(EventId event_id) {
  return (event_id & EVENT_NVSHMEM_TAG) != 0;
}

struct RuntimeSMEM {
  uint64_t instruction_mbarriers[NUM_INSTRUCTION_MBARS][INSTRUCTION_RING_SIZE];
  uint64_t page_finished[MAX_SMEM_PAGES_PER_TASK][PAGE_SEMAPHORE_BITS];
  uint64_t dynamic_semaphores[INSTRUCTION_RING_SIZE][MAX_DYNAMIC_SEMAPHORES];
  __align__(16) char task_buf[INSTRUCTION_RING_SIZE][sizeof(TaskDesc)];

  __device__ __forceinline__ TaskDesc *task_slot(int slot) {
    return reinterpret_cast<TaskDesc *>(task_buf[slot]);
  }
};

__device__ __forceinline__ int ring_slot(int sequence) {
  return sequence % INSTRUCTION_RING_SIZE;
}

__device__ __forceinline__ int ring_phase(int sequence) {
  return (sequence / INSTRUCTION_RING_SIZE) & 1;
}

// ── Debug-only wedge state dump (MPK_V2_STATE_DUMP builds only) ─────────────
// Diagnoses a pure HANG (not a fault): each role writes a WAIT-SITE word to a
// host-mapped pinned buffer immediately BEFORE entering a potentially-blocking
// mbarrier wait and clears it after; the controller (which stays alive during
// a role wedge, spinning in its slot-reuse / drain / iter-sync polls)
// periodically snapshots the worker's raw mbarrier words (instruction ring,
// page_finished, dynamic_semaphores) into the same buffer. The hang watchdog
// then prints, per wedged worker, WHICH wait each role is blocked in plus the
// raw phase/count state of every mbar — pinning the desynced semaphore in one
// wedge run. Reading an mbarrier via a plain ld.shared is outside the PTX
// contract for mbarrier objects; it is a best-effort diagnostic snapshot
// (worst case: a torn/stale value), never used for synchronization.
// Default build (flag unset): compiles to nothing => byte-identical.
#if defined(MPK_V2_STATE_DUMP)
#ifndef MPK_V2_BREADCRUMB
#error "MPK_V2_STATE_DUMP requires MPK_V2_BREADCRUMB=1 (shares its dump path)"
#endif
namespace v2sd {
static constexpr int WORDS_PER_WORKER = 192;
static constexpr int OFF_WS_ROLE = 0;   // [0..4]  wait-site word per role
static constexpr int OFF_CTRL_SEQ = 5;  // controller absolute sequence
static constexpr int OFF_HEARTBEAT = 6; // janitor snapshot counter
static constexpr int OFF_CTRL_LOC = 7;  // controller loop location code
static constexpr int OFF_PAGE_WS = 8;   // [8..21] per-page wait flag
static constexpr int OFF_ARRIVED = 22;  // [22..24] raw ARRIVED mbar words
static constexpr int OFF_FINISHED = 25; // [25..27] raw FINISHED mbar words
static constexpr int OFF_PAGE_RAW = 28; // [28..41] raw page_finished words
static constexpr int OFF_DYN = 48;      // [48..143] raw dyn sems [slot][32]
static constexpr int OFF_EVENTS = 144;  // [144..191] first 48 GMEM event ctrs
// wait-site codes (word = code<<32 | arg)
static constexpr unsigned long long WS_SEMDEP = 1;   // SEM_DEP_READY wait
static constexpr unsigned long long WS_DEPSPIN = 2;  // cross-SM event spin
static constexpr unsigned long long WS_MMA = 3;      // loader mma_mbar wait
static constexpr unsigned long long WS_EPI = 4;      // launcher epilogue wait
static constexpr unsigned long long WS_WTMA = 5;     // launcher W_tma wait
static constexpr unsigned long long WS_ATMA = 6;     // launcher A_tma wait
static constexpr unsigned long long WS_CDONE = 7;    // launcher consumer_done
static constexpr unsigned long long WS_TMEM = 8;     // consumer tmem_ready
static constexpr unsigned long long WS_MAINLOOP = 9; // consumer mainloop wait
} // namespace v2sd
__device__ unsigned long long *g_v2_sd_buf = nullptr;

__device__ __forceinline__ unsigned long long *v2sd_worker_base() {
  return g_v2_sd_buf + static_cast<size_t>(blockIdx.x) * v2sd::WORDS_PER_WORKER;
}
// Single-writer contract: each (worker, role) wait-site slot is written only
// by that role's designated thread (consumer: threadIdx.x==0; single-warp
// roles: their elected/lane-0 thread) — mirrors the breadcrumb rule.
//
// The role-path markers perturb the exact timing window the M1 race needs
// (0/4 wedges with markers vs 3/3 without, 2026-07-16) — so they are gated
// behind an EXTRA define MPK_V2_SD_MARKERS. The default MPK_V2_STATE_DUMP
// build is janitor-only: zero writes from role warps, role timing untouched.
#if defined(MPK_V2_SD_MARKERS)
#define MPK_V2_SD_WS_SET(role, code, arg)                                      \
  do {                                                                         \
    if (mirage::runtime_v2::g_v2_sd_buf != nullptr) {                          \
      mirage::runtime_v2::v2sd_worker_base()                                   \
          [mirage::runtime_v2::v2sd::OFF_WS_ROLE + (role)] =                   \
              (((unsigned long long)(code)) << 32) |                           \
              (unsigned long long)(unsigned)(arg);                             \
      __threadfence_system();                                                  \
    }                                                                          \
  } while (0)
#define MPK_V2_SD_WS_CLR(role)                                                 \
  do {                                                                         \
    if (mirage::runtime_v2::g_v2_sd_buf != nullptr) {                          \
      mirage::runtime_v2::v2sd_worker_base()                                   \
          [mirage::runtime_v2::v2sd::OFF_WS_ROLE + (role)] = 0ull;             \
    }                                                                          \
  } while (0)
#define MPK_V2_SD_PAGE_SET(page, instr)                                        \
  do {                                                                         \
    if (mirage::runtime_v2::g_v2_sd_buf != nullptr) {                          \
      mirage::runtime_v2::v2sd_worker_base()                                   \
          [mirage::runtime_v2::v2sd::OFF_PAGE_WS + (page)] =                   \
              1ull | (((unsigned long long)(unsigned)(instr)) << 32);          \
      __threadfence_system();                                                  \
    }                                                                          \
  } while (0)
#define MPK_V2_SD_PAGE_CLR(page)                                               \
  do {                                                                         \
    if (mirage::runtime_v2::g_v2_sd_buf != nullptr) {                          \
      mirage::runtime_v2::v2sd_worker_base()                                   \
          [mirage::runtime_v2::v2sd::OFF_PAGE_WS + (page)] = 0ull;             \
    }                                                                          \
  } while (0)
#else // MPK_V2_STATE_DUMP without MPK_V2_SD_MARKERS: janitor-only
#define MPK_V2_SD_WS_SET(role, code, arg)
#define MPK_V2_SD_WS_CLR(role)
#define MPK_V2_SD_PAGE_SET(page, instr)
#define MPK_V2_SD_PAGE_CLR(page)
#endif
#else
#define MPK_V2_SD_WS_SET(role, code, arg)
#define MPK_V2_SD_WS_CLR(role)
#define MPK_V2_SD_PAGE_SET(page, instr)
#define MPK_V2_SD_PAGE_CLR(page)
#endif

#if defined(MPK_V2_STATE_DUMP)
// Controller-side janitor: snapshot this worker's raw mbarrier words into the
// pinned buffer. Called (decimated) from the controller's spin loops — the
// controller warp stays alive during a role wedge, so the last snapshot
// before the watchdog fires reflects the wedged state. The dump requires the
// RuntimeConfig for the cross-SM event counters, threaded via an ambient
// device pointer set in the kernel prologue (diagnostic-only).
__device__ void *g_v2_sd_event_counters = nullptr;
__device__ int g_v2_sd_num_events = 0;
__device__ __noinline__ void v2sd_dump(RuntimeSMEM *rt, int seq, int loc) {
  if (g_v2_sd_buf == nullptr) {
    return;
  }
  unsigned long long *b = v2sd_worker_base();
  // Cross-SM event counters (GMEM, same values every worker sees) — the
  // discriminator for dep-spin wedges: which event is short, and by how much.
  if (g_v2_sd_event_counters != nullptr) {
    unsigned long long const *ec =
        static_cast<unsigned long long const *>(g_v2_sd_event_counters);
    int const n = (g_v2_sd_num_events < 48) ? g_v2_sd_num_events : 48;
    for (int e = 0; e < n; e++) {
      b[v2sd::OFF_EVENTS + e] = ec[e];
    }
  }
  b[v2sd::OFF_CTRL_SEQ] = static_cast<unsigned long long>(seq);
  b[v2sd::OFF_CTRL_LOC] = static_cast<unsigned long long>(loc);
  for (int s = 0; s < INSTRUCTION_RING_SIZE; s++) {
    b[v2sd::OFF_ARRIVED + s] = *reinterpret_cast<unsigned long long volatile *>(
        &rt->instruction_mbarriers[MBAR_INSTRUCTION_ARRIVED][s]);
    b[v2sd::OFF_FINISHED + s] =
        *reinterpret_cast<unsigned long long volatile *>(
            &rt->instruction_mbarriers[MBAR_INSTRUCTION_FINISHED][s]);
  }
  for (int p = 0; p < MAX_SMEM_PAGES_PER_TASK; p++) {
    b[v2sd::OFF_PAGE_RAW + p] =
        *reinterpret_cast<unsigned long long volatile *>(
            &rt->page_finished[p][0]);
  }
  for (int s = 0; s < INSTRUCTION_RING_SIZE; s++) {
    for (int i = 0; i < MAX_DYNAMIC_SEMAPHORES; i++) {
      b[v2sd::OFF_DYN + s * MAX_DYNAMIC_SEMAPHORES + i] =
          *reinterpret_cast<unsigned long long volatile *>(
              &rt->dynamic_semaphores[s][i]);
    }
  }
  b[v2sd::OFF_HEARTBEAT] += 1;
  __threadfence_system();
}
#define MPK_V2_SD_JANITOR(rt, seq, loc, ctr)                                   \
  do {                                                                         \
    if ((((ctr)++) & 0x3FF) == 0) {                                            \
      mirage::runtime_v2::v2sd_dump((rt), (seq), (loc));                       \
    }                                                                          \
  } while (0)
#else
#define MPK_V2_SD_JANITOR(rt, seq, loc, ctr)
#endif

// SMEM address of the first op-private dynamic semaphore for the slot
// owned by `instruction_index`. Tasks that need intra-task cross-warp
// mbarriers (e.g. linear's per-stage TMA→MMA→epilogue handshakes)
// receive this base from codegen and address mbars as base + i*8.
__device__ __forceinline__ int op_sem_base_addr(RuntimeSMEM *rt,
                                                int instruction_index) {
  return smem_addr(
      &rt->dynamic_semaphores[ring_slot(instruction_index)][SEM_OP_BASE]);
}

__device__ __forceinline__ void init_page_state(RuntimeSMEM *rt) {
  if (threadIdx.x == 0) {
    for (int page = 0; page < MAX_SMEM_PAGES_PER_TASK; page++) {
      for (int bit = 0; bit < PAGE_SEMAPHORE_BITS; bit++) {
        mbar_init(&rt->page_finished[page][bit], 1);
        mbar_arrive(&rt->page_finished[page][bit]);
      }
    }
  }
}

__device__ __forceinline__ void runtime_wait_page_ready(RuntimeSMEM *rt,
                                                        int physical_page,
                                                        int instruction_index) {
  MPK_V2_SD_PAGE_SET(physical_page, instruction_index);
#pragma unroll
  for (int bit = 0; bit < PAGE_SEMAPHORE_BITS; bit++) {
    int const phase = (instruction_index >> bit) & 1;
    mbar_wait(&rt->page_finished[physical_page][bit], phase);
  }
  MPK_V2_SD_PAGE_CLR(physical_page);
}

__device__ __forceinline__ void runtime_finish_page(RuntimeSMEM *rt,
                                                    int physical_page,
                                                    int arrive_count = 1) {
#pragma unroll
  for (int bit = 0; bit < PAGE_SEMAPHORE_BITS; bit++) {
    for (int i = 0; i < arrive_count; i++) {
      mbar_arrive(&rt->page_finished[physical_page][bit]);
    }
  }
}

// Phase 3.5: returns true if `physical_page` falls inside any of the task's
// declared SMEM regions. Used by the codegen-emitted loader prefix to decide
// which pages to "claim+release ASAP" vs which to leave for the consumer/last
// user to release after they're done with the data.
__device__ __forceinline__ bool task_uses_page(TaskDesc const *task_desc,
                                               int physical_page) {
  for (int r = 0; r < task_desc->num_smem_regions; r++) {
    SmemPageRegionDesc const &region = task_desc->smem_regions[r];
    int const start = region.physical_page_start;
    int const end = start + region.page_count;
    if (physical_page >= start && physical_page < end) {
      return true;
    }
  }
  return false;
}

__device__ __forceinline__ int runtime_region_physical_page(
    TaskDesc const *task_desc, int region_idx, int page_offset) {
  SmemPageRegionDesc const &region = task_desc->smem_regions[region_idx];
  int const physical_page = region.physical_page_start + page_offset;
  if (physical_page < 0 || physical_page >= MAX_SMEM_PAGES_PER_TASK) {
    return -1;
  }
  return physical_page;
}

__device__ __forceinline__ void
    runtime_wait_region_pages(RuntimeSMEM *rt,
                              TaskDesc const *task_desc,
                              int region_idx,
                              int instruction_index) {
  SmemPageRegionDesc const &region = task_desc->smem_regions[region_idx];
  for (int p = 0; p < region.page_count; p++) {
    int const physical_page =
        runtime_region_physical_page(task_desc, region_idx, p);
    if (physical_page >= 0) {
      runtime_wait_page_ready(rt, physical_page, instruction_index);
    }
  }
}

__device__ __forceinline__ void
    runtime_wait_region_range_pages(RuntimeSMEM *rt,
                                    TaskDesc const *task_desc,
                                    int first_region,
                                    int num_regions,
                                    int instruction_index) {
  for (int r = first_region; r < first_region + num_regions; r++) {
    runtime_wait_region_pages(rt, task_desc, r, instruction_index);
  }
}

__device__ __forceinline__ void runtime_finish_region_pages(
    RuntimeSMEM *rt, TaskDesc const *task_desc, int region_idx) {
  SmemPageRegionDesc const &region = task_desc->smem_regions[region_idx];
  for (int p = 0; p < region.page_count; p++) {
    int const physical_page =
        runtime_region_physical_page(task_desc, region_idx, p);
    if (physical_page >= 0) {
      runtime_finish_page(rt, physical_page, 1);
    }
  }
}

__device__ __forceinline__ void
    runtime_finish_region_range_pages(RuntimeSMEM *rt,
                                      TaskDesc const *task_desc,
                                      int first_region,
                                      int num_regions) {
  for (int r = first_region; r < first_region + num_regions; r++) {
    runtime_finish_region_pages(rt, task_desc, r);
  }
}

__device__ __forceinline__ void wait_task_dependency(
    RuntimeConfig const &config, TaskDesc const *task, int iter_num) {
  EventId dep = task->dependent_event;
  if (dep == EVENT_INVALID_ID || is_nvshmem_event(dep)) {
    return;
  }

  size_t const event_index = get_event_position_index(dep);
  EventCounter const needed =
      static_cast<EventCounter>(config.all_event_num_triggers[event_index]) *
      static_cast<EventCounter>(iter_num + 1);
  while (ld_acquire_sys_u64(&config.all_event_counters[event_index]) < needed) {
    __nanosleep(10);
  }
}

// Same dep-wait as wait_task_dependency, but __noinline__ so when called
// from a consumer task body it doesn't inflate the caller's register count.
// Safe to call from any thread; each thread independently confirms the dep.
__device__ __noinline__ void wait_task_dependency_noinline(
    RuntimeConfig const &config, TaskDesc const *task, int iter_num) {
  wait_task_dependency(config, task, iter_num);
}

// Phase 2 consumer-side dep-wait prefix, run by every task's consumer body
// (and by linear's loader/launcher bodies too, since they share the body
// string). Single-thread spin + per-slot SEM_DEP_READY mbarrier sync.
//   - thread 0 globally spins on the cross-SM event counter, then arrives
//     dynamic_semaphores[slot][SEM_DEP_READY].
//   - Lane 0 of every warp running this body waits on the same semaphore.
//   - __syncwarp() reconverges each warp's 32 lanes after its lane-0 wait.
//
// Wrapped __noinline__ so the multi-line body and its locals don't inflate
// consumer_warp_loop's register frame past the launch_bounds(256) ceiling
// (same constraint that forced wait_task_dependency_noinline above).
//
// Phase: ring_phase(instruction_index) — same parity scheme as
// instruction_arrived. SEM_DEP_READY is init-once at kernel start, then
// arrived exactly once per slot use (either by the consumer prefix here,
// or by the controller for tasks that skip the consumer body — see
// BEGIN_TASK_GRAPH special case in controller_warp_loop).
__device__ __noinline__ void consumer_dep_prefix(RuntimeConfig const &config,
                                                 TaskDesc const *task_desc,
                                                 RuntimeSMEM *rt,
                                                 int instruction_index,
                                                 int iter_num) {
  int const slot = ring_slot(instruction_index);
  int const phase = ring_phase(instruction_index);
#if defined(MPK_V2_STATE_DUMP) && defined(MPK_V2_SD_MARKERS)
  // Wait-site markers. Prefix runs on the consumer warps AND (for linear) the
  // loader/launcher warps; derive the breadcrumb role from the warp id and
  // write only from that role's designated thread.
  int const _sd_warp = threadIdx.x / 32;
  int const _sd_role = (_sd_warp < NUM_CONSUMER_WARPS)
                           ? 0
                           : ((_sd_warp == LOADER_WARP)     ? 1
                              : (_sd_warp == LAUNCHER_WARP) ? 2
                                                            : 3);
  bool const _sd_writer =
      (threadIdx.x == 0) ||
      (_sd_warp >= NUM_CONSUMER_WARPS && (threadIdx.x & 31) == 0);
#endif
  if (threadIdx.x == 0) {
#ifdef MPK_ENABLE_PROFILING
    bool const _in_win =
        config.profiler_buffer != nullptr && MPK_V2_PROF_IN_WINDOW(iter_num);
    unsigned long long const _t0 = v2_prof_now_ns();
    if (_in_win) {
      v2_prof_emit(config.profiler_buffer,
                   V2_PROF_GROUP_CONSUMER_PHASE,
                   V2_PROF_DEP_WAIT,
                   tb::EVENT_BEGIN);
    }
#endif
    MPK_V2_SD_WS_SET(0, mirage::runtime_v2::v2sd::WS_DEPSPIN, 0);
    wait_task_dependency(config, task_desc, iter_num);
    MPK_V2_SD_WS_CLR(0);
#ifdef MPK_ENABLE_PROFILING
    if (_in_win) {
      v2_prof_emit(config.profiler_buffer,
                   V2_PROF_GROUP_CONSUMER_PHASE,
                   V2_PROF_DEP_WAIT,
                   tb::EVENT_END);
      // aggregate accumulators (ns, bucketed by task type) — kept alongside
      // the trace events for table-style analysis without trace parsing.
      unsigned long long *_spin =
          static_cast<unsigned long long *>(config.profiler_buffer);
      size_t const _b =
          V2_PROF_SPIN_BASE +
          2ull * V2_PROF_SM_SLOTS * v2_prof_bucket(task_desc->task_type);
      _spin[_b + blockIdx.x] += v2_prof_now_ns() - _t0;
      _spin[_b + V2_PROF_SM_SLOTS + blockIdx.x] += 1;
    }
#endif
    mbar_arrive(&rt->dynamic_semaphores[slot][SEM_DEP_READY]);
  }
  // All lanes wait the dependency semaphore directly. Do NOT gate this on
  // lane 0 with a trailing __syncwarp(): on sm_100a that construct compiles
  // to a WARPSYNC.COLLECTIVE region around the try-wait whose wake crawls
  // ~5us per templated token at the quiet iteration head, delaying the
  // FINISHED arrival of warps 1-3 and the task's event with it
  // (~300us/step at bs16 after cascade; V2_TODO.md #17 has the full
  // evidence chain).
#if defined(MPK_V2_STATE_DUMP) && defined(MPK_V2_SD_MARKERS)
  if (_sd_writer) {
    MPK_V2_SD_WS_SET(
        _sd_role, v2sd::WS_SEMDEP, (unsigned)slot | ((unsigned)phase << 8));
  }
#endif
  mbar_wait(&rt->dynamic_semaphores[slot][SEM_DEP_READY], phase);
#if defined(MPK_V2_STATE_DUMP) && defined(MPK_V2_SD_MARKERS)
  if (_sd_writer) {
    MPK_V2_SD_WS_CLR(_sd_role);
  }
#endif
}

__device__ __forceinline__ bool task_dependency_ready(
    RuntimeConfig const &config, TaskDesc const *task, int iter_num) {
  EventId dep = task->dependent_event;
  if (dep == EVENT_INVALID_ID || is_nvshmem_event(dep)) {
    return true;
  }

  size_t const event_index = get_event_position_index(dep);
  EventCounter const needed =
      static_cast<EventCounter>(config.all_event_num_triggers[event_index]) *
      static_cast<EventCounter>(iter_num + 1);
  return ld_acquire_sys_u64(&config.all_event_counters[event_index]) >= needed;
}

__device__ __forceinline__ void trigger_task_event(RuntimeConfig const &config,
                                                   TaskDesc const *task) {
  EventId event_id = task->trigger_event;
  if (event_id == EVENT_INVALID_ID || is_nvshmem_event(event_id)) {
    return;
  }

  size_t const event_index = get_event_position_index(event_id);
  atom_add_release_gpu_u64(&config.all_event_counters[event_index], 1);
#ifdef MPK_ENABLE_PROFILING
  if (g_v2_prof_window != 0 && g_v2_prof_buf != nullptr) {
    uint64_t *buf = static_cast<uint64_t *>(g_v2_prof_buf);
    unsigned long long const k = atomicAdd(
        reinterpret_cast<unsigned long long *>(&buf[V2_PROF_TRIG_CURSOR]),
        1ULL);
    if (k < V2_PROF_TRIG_RING_LEN) {
      buf[V2_PROF_TRIG_BASE + k] =
          (v2_prof_now_ns() << 32) |
          ((unsigned long long)(event_index & 0xFFFFFF) << 8) |
          (blockIdx.x & 0xFF);
    }
  }
#endif
}

// Implemented by the generated v2 role dispatch code after this runtime header
// is included.
__device__ __forceinline__ void
    _execute_init_semaphores_v2(TaskDesc const *task_desc,
                                RuntimeConfig const &config,
                                RuntimeSMEM *runtime_smem,
                                int instruction_index,
                                int iter_num);

__device__ __forceinline__ void
    _execute_loader_task_v2(TaskDesc const *task_desc,
                            RuntimeConfig const &config,
                            RuntimeSMEM *runtime_smem,
                            int instruction_index,
                            int iter_num);

__device__ __forceinline__ void
    _execute_launcher_task_v2(TaskDesc const *task_desc,
                              RuntimeConfig const &config,
                              RuntimeSMEM *runtime_smem,
                              int instruction_index,
                              int iter_num);

__device__ __forceinline__ void
    _execute_consumer_task_v2(TaskDesc const *task_desc,
                              RuntimeConfig const &config,
                              RuntimeSMEM *runtime_smem,
                              int instruction_index,
                              int iter_num);

__device__ __forceinline__ void
    _execute_storer_task_v2(TaskDesc const *task_desc,
                            RuntimeConfig const &config,
                            RuntimeSMEM *runtime_smem,
                            int instruction_index,
                            int iter_num);

// Host-side per-(task_type, variant) page-lifecycle mode, defined by the
// generated role-dispatch code (v2_role_codegen.cc emit_page_mode_fn):
// 0 = none/unregistered, 1 = wait-all loader prefix (dense observation),
// 2 = consumer-owned + SkipUsed loader (loader misses the USED pages),
// 3 = consumer-total (loader misses EVERY page; waits none). Consumed by
// build_v2_plan's mixed-chain window assertion (persistent_kernel_v2.cuh)
// — see the mixed-chain boundary note in v2_role_codegen.cc.
int _v2_variant_page_mode(int task_type, int variant_id);

// ── Debug-only per-worker task breadcrumb (MPK_V2_BREADCRUMB builds only) ────
// Localizes a context-poisoning cudaErrorIllegalAddress that only reproduces
// in the full multi-rank megakernel (memcheck can't attribute it under
// -rdc=true + NVSHMEM cooperative). Writes a STARTED word to host-mapped
// PINNED memory BEFORE execute_task and mirrors it to a COMPLETED word AFTER.
// A (worker, role) with STARTED != COMPLETED post-crash was IN FLIGHT when the
// fault hit — a candidate set, not a proven-unique faulter: an illegal address
// poisons the whole context, so other concurrently-running tasks also show up.
// The crash is deterministic, so the CUDA error + a re-run narrow the set.
// FALSE-NEGATIVE window: the consumer role is 4 warps but only threadIdx.x==0
// writes the crumb; a fault confined to consumer warps 1-3 AFTER the body's
// final 128-thread internal barrier reads clean. We deliberately do NOT add a
// cross-warp barrier in this macro — the single-warp roles (loader/launcher/
// storer) would deadlock on a 128-thread bar.sync (only 32 threads present),
// and the consumer bodies already 128-thread-sync internally
// (SEM_CONSUMER_DONE) so the window is a handful of trailing instructions.
// Default build (flag unset): all of this compiles to nothing =>
// byte-identical.
//
// Role ids MUST match the readback helper
// (demo/deepseek_v3/v2_breadcrumb_readback.py) and MPK_V2_BREADCRUMB_ROLES in
// persistent_kernel_v2.cuh.
#define MPK_V2_BC_ROLE_CONSUMER 0
#define MPK_V2_BC_ROLE_LOADER 1
#define MPK_V2_BC_ROLE_LAUNCHER 2
#define MPK_V2_BC_ROLE_STORER 3
#define MPK_V2_BC_ROLE_CONTROLLER 4
#ifndef MPK_V2_BREADCRUMB_ROLES
#define MPK_V2_BREADCRUMB_ROLES 5
#endif

// MPK_MEASURE_ROLE (a separate register-measurement diagnostic) routes ALL
// warps through ONE role loop; that breaks the single-writer guarantee for the
// single-warp roles (every warp's lane_id==0 would write the same slot). The
// two diagnostics are mutually exclusive, so breadcrumbs are inert under it.
#if defined(MPK_V2_BREADCRUMB) && !defined(MPK_MEASURE_ROLE)
// STARTED word: [63:40]=iter_num [39:24]=sequence_in_iter [23:8]=task_type
//               [7:0]=role_id. Written by the single-writer thread selected by
// writer_pred (the role loop's profiler predicate) with a system-scope fence so
// the host sees it after the launch errors. writer_pred is passed in — it is
// the OUTER role-loop macro's parameter, not an identifier in scope here.
#define MPK_V2_BC_STARTED(role_id, writer_pred)                                \
  do {                                                                         \
    if ((writer_pred) && config.breadcrumb_device != nullptr &&                \
        worker_id < config.breadcrumb_num_slots) {                             \
      unsigned long long *_bc =                                                \
          static_cast<unsigned long long *>(config.breadcrumb_device);         \
      size_t const _idx =                                                      \
          (static_cast<size_t>(worker_id) * MPK_V2_BREADCRUMB_ROLES +          \
           (role_id)) *                                                        \
          2ull;                                                                \
      unsigned long long const _w =                                            \
          ((unsigned long long)(iter_num & 0xFFFFFF) << 40) |                  \
          ((unsigned long long)(sequence_in_iter & 0xFFFF) << 24) |            \
          ((unsigned long long)(task->task_type & 0xFFFF) << 8) |              \
          ((unsigned long long)((role_id)&0xFF));                              \
      _bc[_idx + 0] = _w;                                                      \
      __threadfence_system();                                                  \
    }                                                                          \
  } while (0)
// COMPLETED word: mirror STARTED so STARTED == COMPLETED means "this task
// finished". A mid-flight fault leaves COMPLETED holding the PRIOR task's word.
#define MPK_V2_BC_COMPLETED(role_id, writer_pred)                              \
  do {                                                                         \
    if ((writer_pred) && config.breadcrumb_device != nullptr &&                \
        worker_id < config.breadcrumb_num_slots) {                             \
      unsigned long long *_bc =                                                \
          static_cast<unsigned long long *>(config.breadcrumb_device);         \
      size_t const _idx =                                                      \
          (static_cast<size_t>(worker_id) * MPK_V2_BREADCRUMB_ROLES +          \
           (role_id)) *                                                        \
          2ull;                                                                \
      _bc[_idx + 1] = _bc[_idx + 0];                                           \
      __threadfence_system();                                                  \
    }                                                                          \
  } while (0)
#else
#define MPK_V2_BC_STARTED(role_id, writer_pred)
#define MPK_V2_BC_COMPLETED(role_id, writer_pred)
#endif

#define MIRAGE_V2_DEFINE_ROLE_WARP_LOOP(                                       \
    loop_name, execute_task, prof_group, prof_pred, role_id)                   \
  __device__ __noinline__ void loop_name(                                      \
      RuntimeSMEM *rt, RuntimeConfig const &config, int lane_id) {             \
    int const worker_id = blockIdx.x;                                          \
    int const my_count =                                                       \
        static_cast<int>(config.v2_per_sm_task_offsets[worker_id + 1] -        \
                         config.v2_per_sm_task_offsets[worker_id]);            \
    int sequence = 0;                                                          \
    int iter_num = 0;                                                          \
    int sequence_in_iter = 0;                                                  \
    /* profiling: one track per role. The consumer loop runs on 4 warps,  */   \
    /* so its predicate must select warp 0 lane 0 (threadIdx.x == 0);     */   \
    /* single-warp roles use lane_id == 0. The breadcrumb reuses the SAME */   \
    /* prof_pred as its single-writer guard (exactly one thread/worker).  */   \
    MPK_V2_PROF_DECL(prof_group, prof_pred)                                    \
    while (true) {                                                             \
      int const slot = ring_slot(sequence);                                    \
      int const phase = ring_phase(sequence);                                  \
      if (lane_id == 0) {                                                      \
        mbar_wait(&rt->instruction_mbarriers[MBAR_INSTRUCTION_ARRIVED][slot],  \
                  phase);                                                      \
      }                                                                        \
      __syncwarp();                                                            \
      TaskDesc *task = rt->task_slot(slot);                                    \
      if (task->task_type == TASK_TERMINATE) {                                 \
        return;                                                                \
      }                                                                        \
      if (task->task_type != TASK_BEGIN_TASK_GRAPH) {                          \
        MPK_V2_BC_STARTED(role_id, prof_pred);                                 \
        MPK_V2_PROF_START(task->task_type);                                    \
        execute_task(task, config, rt, sequence, iter_num);                    \
        MPK_V2_PROF_END(task->task_type);                                      \
        MPK_V2_BC_COMPLETED(role_id, prof_pred);                               \
      }                                                                        \
      if (lane_id == 0) {                                                      \
        mbar_arrive(                                                           \
            &rt->instruction_mbarriers[MBAR_INSTRUCTION_FINISHED][slot]);      \
      }                                                                        \
      sequence++;                                                              \
      sequence_in_iter++;                                                      \
      if (sequence_in_iter == my_count) {                                      \
        sequence_in_iter = 0;                                                  \
        iter_num++;                                                            \
      }                                                                        \
    }                                                                          \
  }

MIRAGE_V2_DEFINE_ROLE_WARP_LOOP(loader_warp_loop,
                                _execute_loader_task_v2,
                                V2_PROF_GROUP_LOADER,
                                (lane_id == 0),
                                MPK_V2_BC_ROLE_LOADER)
MIRAGE_V2_DEFINE_ROLE_WARP_LOOP(launcher_warp_loop,
                                _execute_launcher_task_v2,
                                V2_PROF_GROUP_LAUNCHER,
                                (lane_id == 0),
                                MPK_V2_BC_ROLE_LAUNCHER)
MIRAGE_V2_DEFINE_ROLE_WARP_LOOP(consumer_warp_loop,
                                _execute_consumer_task_v2,
                                V2_PROF_GROUP_CONSUMER,
                                (threadIdx.x == 0),
                                MPK_V2_BC_ROLE_CONSUMER)
MIRAGE_V2_DEFINE_ROLE_WARP_LOOP(storer_warp_loop,
                                _execute_storer_task_v2,
                                V2_PROF_GROUP_STORER,
                                (lane_id == 0),
                                MPK_V2_BC_ROLE_STORER)

#undef MIRAGE_V2_DEFINE_ROLE_WARP_LOOP

} // namespace runtime_v2
} // namespace mirage

#if defined(MODE_OFFLINE)
__device__ __forceinline__ bool
    prepare_next_batch(mirage::runtime::RuntimeConfig const &config);
#elif defined(MODE_ONLINE_NOTOKEN)
__device__ __forceinline__ bool
    prepare_next_batch(mirage::runtime::RuntimeConfig const &config,
                       size_t iteration_num);
#endif

namespace mirage {
namespace runtime_v2 {

// Set by worker 0 each iteration to prepare_next_batch's "generation done"
// signal (EOS reached or step >= max_seq_length). v1 stops on this return;
// v2 previously ignored it and ran the full max_seq_length iterations, so its
// per-token latency scaled with max_seq_length instead of actual output length.
__device__ unsigned int g_v2_gen_done = 0;

__device__ __noinline__ void controller_warp_loop(RuntimeSMEM *rt,
                                                  RuntimeConfig const &config,
                                                  int lane_id) {
  int const worker_id = blockIdx.x;
  int const num_workers = config.num_workers;
  size_t const my_offset = config.v2_per_sm_task_offsets[worker_id];
  size_t const my_end = config.v2_per_sm_task_offsets[worker_id + 1];
  size_t const my_count = my_end - my_offset;
  int sequence = 0;

#if defined(MPK_V2_STATE_DUMP)
  unsigned _sd_ctr = 0; // janitor decimation counter (controller lane 0)
#endif

  // profiling track (controller group): prepare/iter-barrier timing.
  MPK_V2_PROF_DECL(V2_PROF_GROUP_CONTROLLER, (lane_id == 0))

  // Per-slot dedup: the last absolute sequence whose graph event we already
  // triggered for this ring slot. This lets the controller trigger events
  // OUT OF ORDER — eagerly, as soon as a task's role warps finish — without
  // ever double-counting. Out-of-order triggering is REQUIRED to avoid a
  // deferred-trigger deadlock: an earlier consumer task (next in this SM's
  // ring) can block on a graph event whose producer is a LATER, already
  // finished task on the same ring. The old in-order
  // wait_finished_and_trigger_through could never reach that later producer,
  // so its event never fired and the whole pipeline froze.
  int triggered_seq[INSTRUCTION_RING_SIZE];
#pragma unroll
  for (int s = 0; s < INSTRUCTION_RING_SIZE; s++) {
    triggered_seq[s] = -1;
  }

  // Lane 0 only. Non-blocking: trigger every in-flight (published, not yet
  // slot-reused) task whose role warps have finished and which we have not
  // triggered yet. task_slot(slot) still holds sequence s's TaskDesc until it
  // is reused at s + INSTRUCTION_RING_SIZE (> sequence), so this is safe.
  auto eager_trigger_inflight = [&]() {
    int lo = sequence - INSTRUCTION_RING_SIZE;
    if (lo < 0) {
      lo = 0;
    }
    for (int s = lo; s < sequence; s++) {
      int const slot = ring_slot(s);
      if (triggered_seq[slot] == s) {
        continue;
      }
      if (mbar_poll(
              smem_addr(
                  &rt->instruction_mbarriers[MBAR_INSTRUCTION_FINISHED][slot]),
              ring_phase(s))) {
        trigger_task_event(config, rt->task_slot(slot));
        triggered_seq[slot] = s;
      }
    }
  };

  // Lane 0 spins until done_sequence's slot has finished (so its slot can be
  // reused), eagerly triggering any other finished in-flight tasks while it
  // waits — this is what breaks the cycle.
  auto wait_slot_finished_eager = [&](int done_sequence) {
    if (lane_id == 0) {
      int const done_slot = ring_slot(done_sequence);
      int const done_phase = ring_phase(done_sequence);
      while (!mbar_poll(
          smem_addr(
              &rt->instruction_mbarriers[MBAR_INSTRUCTION_FINISHED][done_slot]),
          done_phase)) {
        eager_trigger_inflight();
        MPK_V2_SD_JANITOR(rt, sequence, 1, _sd_ctr);
      }
      eager_trigger_inflight();
    }
    __syncwarp();
  };

  // Phase 2: cross-SM dependency wait moved out of the controller. Each
  // task's consumer prefix now calls wait_task_dependency_noinline before
  // running its body. Controller becomes pure fetch+publish; only
  // wait_slot_finished_eager (above) is kept, both for slot reuse
  // and for promptly publishing intra-stream producer events that the
  // consumer's spin needs to terminate.
  for (int iter_num = 0; iter_num < config.v2_max_iters; iter_num++) {
    // prepare_next_batch mutates per-iteration decode state used by every
    // worker. Worker 0 does the mutation once; all other workers wait for the
    // system-scope counter before issuing this iteration's task stream.
    if (worker_id == 0) {
      if (lane_id == 0) {
#ifdef MPK_ENABLE_PROFILING
        // arm/disarm the ambient timed-wait window for ALL SMs; they read it
        // only after their go-counter acquire below, so it is coherent.
        g_v2_prof_window = MPK_V2_PROF_IN_WINDOW(iter_num) ? 1 : 0;
#endif
        bool _cont = true;
        MPK_V2_PROF_START(V2_PROF_PREPARE_BATCH);
#if defined(MODE_OFFLINE)
        _cont = ::prepare_next_batch(config);
#elif defined(MODE_ONLINE_NOTOKEN)
        _cont = ::prepare_next_batch(config, iter_num);
#endif
        MPK_V2_PROF_END(V2_PROF_PREPARE_BATCH);
#ifdef MPK_ENABLE_PROFILING
        _cont = true; // profiling: run all iters, don't early-exit
#endif
        // Mirror v1: prepare_next_batch returns false when generation is done
        // (EOS or step >= max_seq_length). Publish it BEFORE the go-counter
        // increment so other workers see it after their acquire-load below.
        g_v2_gen_done = _cont ? 0u : 1u;
        __threadfence_system();
        atomicAdd_system(config.v2_iter_go_counter, 1ULL);
      }
    } else {
      if (lane_id == 0) {
        MPK_V2_PROF_START(V2_PROF_GO_WAIT);
        unsigned long long const needed =
            static_cast<unsigned long long>(iter_num + 1);
        while (ld_acquire_sys_u64(config.v2_iter_go_counter) < needed) {
          __nanosleep(50);
          MPK_V2_SD_JANITOR(rt, sequence, 4, _sd_ctr);
        }
        MPK_V2_PROF_END(V2_PROF_GO_WAIT);
      }
    }
    __syncwarp();

    // THE FIX: stop as soon as generation is actually finished, instead of
    // running all v2_max_iters (= max_seq_length) iterations. worker 0 set
    // g_v2_gen_done above (ordered before its go-counter increment); every
    // other worker has passed the acquire-load of that counter, so this read
    // observes it. Mirrors v1, which terminates on prepare_next_batch's return.
    {
      unsigned int _done = 0;
      if (lane_id == 0) {
        _done = *reinterpret_cast<unsigned int volatile *>(&g_v2_gen_done);
      }
      _done = __shfl_sync(0xffffffff, _done, 0);
      if (_done) {
        break;
      }
    }

    for (size_t i = 0; i < my_count; i++) {
      int const slot = ring_slot(sequence);
      int const phase = ring_phase(sequence);

      // The instruction ring is finite. Before writing a new TaskDesc into a
      // reused slot, wait until the role warps have finished the previous
      // sequence that occupied the same slot.
      if (sequence >= INSTRUCTION_RING_SIZE) {
        wait_slot_finished_eager(sequence - INSTRUCTION_RING_SIZE);
      }

      size_t const task_pos = config.v2_per_sm_task_positions[my_offset + i];
#if defined(MPK_V2_BREADCRUMB) && !defined(MPK_MEASURE_ROLE)
      // Controller breadcrumb (role 4). Fault sites here: an OOB task_pos
      // faulting the TaskDesc copy below, or a codegen'd _execute_init_
      // semaphores_v2. STARTED encodes task_pos (not task_type — not yet
      // known) so an OOB index is directly visible. Single-writer: lane 0.
      if (lane_id == 0 && config.breadcrumb_device != nullptr &&
          worker_id < config.breadcrumb_num_slots) {
        unsigned long long *_bc =
            static_cast<unsigned long long *>(config.breadcrumb_device);
        size_t const _idx =
            (static_cast<size_t>(worker_id) * MPK_V2_BREADCRUMB_ROLES +
             MPK_V2_BC_ROLE_CONTROLLER) *
            2ull;
        _bc[_idx + 0] =
            ((unsigned long long)(iter_num & 0xFFFFFF) << 40) |
            ((unsigned long long)(i & 0xFFFF) << 24) |
            ((unsigned long long)(task_pos & 0xFFFF) << 8) |
            ((unsigned long long)(MPK_V2_BC_ROLE_CONTROLLER & 0xFF));
        __threadfence_system();
      }
#endif
      {
        // The controller warp cooperatively copies one TaskDesc from the
        // compiled global task table into the shared-memory ring slot. The
        // following __syncwarp makes lane 0 wait for every lane's copy chunks
        // before it publishes instruction_arrived to the role warps.
        char *dst = rt->task_buf[slot];
        char const *src =
            reinterpret_cast<char const *>(&config.all_tasks[task_pos]);
        constexpr int CHUNKS = (sizeof(TaskDesc) + 15) / 16;
        for (int c = lane_id; c < CHUNKS; c += 32) {
          ::kernel::load_smem(dst + c * 16, src + c * 16);
        }
        ::kernel::cp_async_fence();
        ::kernel::cp_async_wait<0>();
        // The TaskDesc was copied via cp.async; role warps read it with normal
        // loads after the INSTRUCTION_ARRIVED mbar. Publish the cp.async
        // writes to the generic proxy before the arrive (v1 got this implicitly
        // from its __syncthreads after cp_async_wait; v2's warp-specialized
        // handshake does not). Defensive hardening: isolation testing of the
        // 2026-05 page-parity hang showed this fence ALONE does not fix it
        // (the launcher __syncwarp in linear_v3 does), but the proxy-ordering
        // gap is real per the PTX memory model, so it stays.
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      }
      __syncwarp();

      // Op-declared per-instruction semaphore initialization. The body is
      // emitted by codegen and runs single-threaded (lane 0) once per
      // published instruction, before role warps wake. Empty for ops that
      // don't declare any dynamic semaphores; Phase 3+ will populate it.
      //
      // ALSO: BEGIN_TASK_GRAPH skips the role-warp consumer body entirely
      // (the role-warp-loop macro early-returns from execute_task), so its
      // slot's SEM_DEP_READY would never be arrived. To keep the
      // ring_phase parity in sync for the next task at this slot,
      // controller arrives SEM_DEP_READY here on its behalf. Phase 3.5:
      // the per-page parity needs the same protection — every task in the
      // pipeline must arrive each page exactly once, otherwise consecutive
      // tasks deadlock on page_finished. Controller arrives all pages on
      // BEGIN_TASK_GRAPH's behalf.
      if (lane_id == 0) {
        _execute_init_semaphores_v2(
            rt->task_slot(slot), config, rt, sequence, iter_num);
        if (rt->task_slot(slot)->task_type == TASK_BEGIN_TASK_GRAPH) {
          mbar_arrive(&rt->dynamic_semaphores[slot][SEM_DEP_READY]);
          for (int p = 0; p < MAX_SMEM_PAGES_PER_TASK; p++) {
            runtime_finish_page(rt, p, 1);
          }
        }
      }
      __syncwarp();

      // No more controller-side dep wait — consumers handle it themselves
      // via wait_task_dependency_noinline in their consumer prefix. The
      // controller is now pure fetch+publish.
      //
      // Intra-stream producer events (sequence S-1, S-2 in this same ring
      // stream feeding this same SM's consumer) get triggered at slot reuse
      // via wait_slot_finished_eager(sequence - INSTRUCTION_RING_SIZE)
      // above. That introduces up to RING-1 instructions of latency for
      // intra-stream consumer-producer chains; in Qwen3 these are rare
      // because the worker queues round-robin tasks across SMs. If
      // measurement shows this is hot, Phase 4 can move event triggering
      // into the storer warp and cut this latency.
      if (lane_id == 0) {
        // Do not release role warps until the copied TaskDesc is visible in
        // shared memory. The block fence must happen before mbar_arrive
        // because role warps wake on that mbarrier and immediately read
        // rt->task_buf[slot].
        __threadfence_block();
        mbar_arrive(&rt->instruction_mbarriers[MBAR_INSTRUCTION_ARRIVED][slot]);
      }
      __syncwarp();
#if defined(MPK_V2_BREADCRUMB) && !defined(MPK_MEASURE_ROLE)
      // Controller breadcrumb COMPLETED: the per-task fetch+publish (copy +
      // init_semaphores + arrive) for this slot finished without faulting.
      if (lane_id == 0 && config.breadcrumb_device != nullptr &&
          worker_id < config.breadcrumb_num_slots) {
        unsigned long long *_bc =
            static_cast<unsigned long long *>(config.breadcrumb_device);
        size_t const _idx =
            (static_cast<size_t>(worker_id) * MPK_V2_BREADCRUMB_ROLES +
             MPK_V2_BC_ROLE_CONTROLLER) *
            2ull;
        _bc[_idx + 1] = _bc[_idx + 0];
        __threadfence_system();
      }
#endif
      sequence++;
    }

    // Drain all live ring slots for this worker before ending the decode
    // iteration: block until each of the last RING tasks has finished (pumping
    // eager triggers so cross-dependencies among them resolve), then a final
    // eager sweep guarantees every published task's event has fired. This
    // preserves the iteration boundary expected by prepare_next_batch and the
    // global step/token state.
    if (lane_id == 0) {
      int lo = sequence - INSTRUCTION_RING_SIZE;
      if (lo < 0) {
        lo = 0;
      }
      for (int s = lo; s < sequence; s++) {
        int const slot = ring_slot(s);
        int const ph = ring_phase(s);
        while (!mbar_poll(
            smem_addr(
                &rt->instruction_mbarriers[MBAR_INSTRUCTION_FINISHED][slot]),
            ph)) {
          eager_trigger_inflight();
          MPK_V2_SD_JANITOR(rt, sequence, 2, _sd_ctr);
        }
      }
      eager_trigger_inflight();
    }
    __syncwarp();

    // RACE-3 FIX (2026-07-16): snapshot the loop-exit step value BEFORE this
    // worker's iter-sync arrival — and BEFORE the fence below, so the load
    // is ordered ahead of the (relaxed) arrival atomic on weakly-ordered
    // hardware. config.step[0] is mutated only by worker 0's
    // prepare_next_batch at the NEXT iteration's head, which cannot run
    // until EVERY worker has arrived this barrier — so a pre-arrival
    // snapshot reads THIS iteration's value on every worker and the break
    // decision below is UNIFORM. The old post-barrier read raced
    // prepare(M+1): a straggler waking late from the barrier spin read the
    // already-advanced step and broke one iteration early, publishing
    // TERMINATE while the other workers entered the next iteration and
    // waited forever on its never-published tasks' events (observed:
    // 6/136 workers exited at iter 30 of 32; the gateup event counter came
    // up short by exactly their iteration-31 task population; the split can
    // only fire when step == max_seq_length-2, i.e. at the final boundary —
    // matching every observed end-of-run wedge; profiled builds pin
    // g_v2_gen_done to 0, making this check the sole loop exit there).
    int step0 = 0;
    if (lane_id == 0) {
      step0 = *reinterpret_cast<int volatile *>(&config.step[0]);
    }

    // All workers must finish the current iteration before any worker starts
    // the next prepare_next_batch. The system fence orders this worker's event
    // updates before it increments the cross-worker iteration counter.
    __threadfence_system();
    if (lane_id == 0) {
      MPK_V2_PROF_START(V2_PROF_ITER_SYNC);
      atomicAdd_system(config.v2_iter_sync_counter, 1ULL);
      unsigned long long const needed =
          static_cast<unsigned long long>(num_workers) *
          static_cast<unsigned long long>(iter_num + 1);
      while (ld_acquire_sys_u64(config.v2_iter_sync_counter) < needed) {
        __nanosleep(50);
        MPK_V2_SD_JANITOR(rt, sequence, 3, _sd_ctr);
      }
      MPK_V2_PROF_END(V2_PROF_ITER_SYNC);
    }
    __syncwarp();

    // RACE-3 AMPLIFIER (debug, env-gated default-OFF): widen the
    // barrier-exit straggle window on a worker subset so the half-exit race
    // below (now fixed — see the snapshot note at the barrier) can be
    // reproduced on demand against the OLD read placement, and proven
    // closed against the new one. ~5 ms of post-barrier delay.
#ifdef MPK_V2_RACE3_AMPLIFY
    if ((worker_id & 15) == 3) {
      for (int _d = 0; _d < 5000; _d++) {
        __nanosleep(1000);
      }
    }
#endif
#ifdef MPK_V2_RACE3_OLD_READ
    // A/B arm (debug, default-OFF): reproduce the PRE-FIX racy post-barrier
    // read. Combined with MPK_V2_RACE3_AMPLIFY this wedges the half-exit
    // race on demand (the delayed workers read step AFTER worker 0's
    // next-iteration prepare advanced it); the fixed build under the same
    // amplifier must pass.
    if (lane_id == 0) {
      step0 = *reinterpret_cast<int volatile *>(&config.step[0]);
    }
#endif

    // Step is updated by prepare_next_batch. Broadcast lane 0's PRE-ARRIVAL
    // snapshot (taken above, before the iter-sync fence+arrival) so the
    // controller warp exits the loop uniformly. Do NOT re-read step here:
    // the post-barrier value races worker 0's NEXT-iteration
    // prepare_next_batch (see the RACE-3 note at the snapshot site).
    step0 = __shfl_sync(0xffffffff, step0, 0);
    if (step0 >= config.max_seq_length - 1) {
      break;
    }
  }

  // Publish a terminate instruction through the same instruction_arrived path
  // so all role warps leave their role_warp_loop cleanly.
  int const term_slot = ring_slot(sequence);
  if (sequence >= INSTRUCTION_RING_SIZE) {
    wait_slot_finished_eager(sequence - INSTRUCTION_RING_SIZE);
  }
  if (lane_id == 0) {
    rt->task_slot(term_slot)->task_type = TASK_TERMINATE;
    __threadfence_block();
    mbar_arrive(
        &rt->instruction_mbarriers[MBAR_INSTRUCTION_ARRIVED][term_slot]);
  }
}

__global__ __launch_bounds__(MPK_LAUNCH_THREADS,
                             1) void worker_v2_kernel(RuntimeConfig config) {
  __shared__ __align__(16) char rt_buf[sizeof(RuntimeSMEM)];
  RuntimeSMEM *rt = reinterpret_cast<RuntimeSMEM *>(rt_buf);

  int const warp_id = threadIdx.x / 32;
  int const lane_id = threadIdx.x % 32;

#if defined(MPK_SETMAXNREG_EARLY) && MPK_SETMAXNREG
  // CUTLASS convention: redistribute registers as the FIRST thing in the
  // kernel, before any shared-memory ops or __syncthreads. Warpgroup-aligned:
  // WG0 (consumers) inc, all other warpgroups dec.
  if (warp_id < NUM_CONSUMER_WARPS) {
    MPK_SETMAXNREG_INC(MPK_CONSUMER_REGS);
  } else {
    MPK_SETMAXNREG_DEC(MPK_HELPER_REGS);
  }
#endif

#ifdef MPK_ENABLE_PROFILING
  if (threadIdx.x == 0) {
    // same value from every block — benign race, long before first use.
    g_v2_prof_buf = config.profiler_buffer;
  }
#endif
#if defined(MPK_V2_STATE_DUMP)
  if (threadIdx.x == 0) {
    // same value from every block — benign race, diagnostic-only.
    g_v2_sd_event_counters = config.all_event_counters;
    g_v2_sd_num_events = config.num_events;
  }
#endif

  if (threadIdx.x == 0) {
    for (int slot = 0; slot < INSTRUCTION_RING_SIZE; slot++) {
      mbar_init(&rt->instruction_mbarriers[MBAR_INSTRUCTION_ARRIVED][slot], 1);
      mbar_init(&rt->instruction_mbarriers[MBAR_INSTRUCTION_FINISHED][slot],
                NUM_ROLE_WARPS);
      // SEM_DEP_READY: per-slot semaphore signaled by consumer thread 0
      // and waited on by lane 0 of every warp running the consumer body.
      // Init-once + ring_phase parity (matches instruction_arrived). Each
      // slot must be arrived once per use to keep parity in sync — see the
      // BEGIN_TASK_GRAPH special case in controller_warp_loop.
      mbar_init(&rt->dynamic_semaphores[slot][SEM_DEP_READY], 1);
    }
  }
  init_page_state(rt);
  if (threadIdx.x == 0) {
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  __syncthreads();

#if defined(MPK_MEASURE_ROLE)
  // Diagnostic: route ALL warps through one role so worker_v2_kernel's
  // reported register count == that single role's footprint (per-role
  // register measurement; the dispatch is otherwise fused into one blob).
  if (MPK_MEASURE_ROLE == 0) {
    consumer_warp_loop(rt, config, lane_id);
  } else if (MPK_MEASURE_ROLE == 1) {
    loader_warp_loop(rt, config, lane_id);
  } else if (MPK_MEASURE_ROLE == 2) {
    launcher_warp_loop(rt, config, lane_id);
  } else if (MPK_MEASURE_ROLE == 3) {
    storer_warp_loop(rt, config, lane_id);
  } else if (MPK_MEASURE_ROLE == 4) {
    controller_warp_loop(rt, config, lane_id);
  }
  return;
#endif
  if (warp_id < NUM_CONSUMER_WARPS) {
    // WG0 (warps 0-3): all consumers. Claim registers for the task bodies.
#ifndef MPK_SETMAXNREG_EARLY
    MPK_SETMAXNREG_INC(MPK_CONSUMER_REGS);
#endif
    consumer_warp_loop(rt, config, lane_id);
  } else {
    // WG1 (warps 4-7): loader/launcher/storer/controller. Release registers
    // back to the pool BEFORE splitting into per-role branches, so the
    // setmaxnreg.dec is warpgroup-aligned across all four helper warps.
#ifndef MPK_SETMAXNREG_EARLY
    MPK_SETMAXNREG_DEC(MPK_HELPER_REGS);
#endif
    if (warp_id == LOADER_WARP) {
      loader_warp_loop(rt, config, lane_id);
    } else if (warp_id == LAUNCHER_WARP) {
      launcher_warp_loop(rt, config, lane_id);
    } else if (warp_id == STORER_WARP) {
      storer_warp_loop(rt, config, lane_id);
    } else if (warp_id == CONTROLLER_WARP) {
      controller_warp_loop(rt, config, lane_id);
    }
  }
}

inline void launch_worker_v2(RuntimeConfig const &config,
                             int num_workers,
                             cudaStream_t stream) {
#ifdef MPK_ENABLE_PROFILING
  // Per-SM profiler tail arrays (cursors/spin/suffix/misc) are sized for
  // V2_PROF_SM_SLOTS workers; a larger grid would silently corrupt them
  // (this is exactly what happened at 136 workers with the old 128-slot
  // arrays). Fail loudly instead.
  if (num_workers > V2_PROF_SM_SLOTS) {
    printf("[v2] FATAL: num_workers=%d exceeds V2_PROF_SM_SLOTS=%d — "
           "profiler tail arrays would be corrupted. Increase "
           "V2_PROF_SM_SLOTS in runtime_v2.cuh.\n",
           num_workers,
           V2_PROF_SM_SLOTS);
    abort();
  }
#endif
  int smem = MAX_DYNAMIC_SHARED_MEMORY_SIZE;
  cudaFuncSetAttribute(
      worker_v2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  worker_v2_kernel<<<dim3(num_workers, 1, 1),
                     dim3(MPK_LAUNCH_THREADS, 1, 1),
                     smem,
                     stream>>>(config);
}

} // namespace runtime_v2
} // namespace mirage
