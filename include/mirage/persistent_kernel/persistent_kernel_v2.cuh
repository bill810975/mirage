// v2 persistent kernel driver.
//
// Replaces scheduler_kernel + worker_kernel with a scheduler-less path:
//   - Host pre-computes per-SM task list via round-robin
//   - worker_v2_kernel walks its SM's list directly (runtime_v2.cuh)
//   - Host loops iterations, calling prepare_next_batch between
//
// Leaves persistent_kernel.cuh (v1) untouched. Depends on persistent_kernel.cuh
// for init_persistent_kernel/global_runtime_config/init_kernel/prepare_kernel.

#pragma once

// NOTE: this file assumes the including translation unit has ALREADY included
// "persistent_kernel.cuh" before this file — it depends on
// global_runtime_config, prepare_kernel, and prepare_next_batch being defined
// there. We do NOT re-include persistent_kernel.cuh because it lacks include
// guards.
#include "mirage/persistent_kernel/runtime_v2.cuh"
#include "mirage/persistent_kernel/tasks/blackwell_v2/task_header.cuh"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <string>
#include <thread>
#include <vector>

namespace mirage {
namespace runtime_v2 {

using ::mirage::runtime::EventDesc;
using ::mirage::runtime::RuntimeConfig;
using ::mirage::runtime::TaskId;

#if defined(MPK_V2_BREADCRUMB) && defined(MPK_V2_STATE_DUMP)
// Host-side view of the wedge state-dump buffer (see runtime_v2.cuh v2sd::).
inline unsigned long long *g_v2_sd_host = nullptr;
inline int g_v2_sd_num_workers = 0;

// Decode + print the per-worker wedge state. `only_wedged`: print only
// workers with a nonzero wait-site or page-wait flag (plus one healthy
// reference worker).
inline void dump_v2_state() {
  if (g_v2_sd_host == nullptr || g_v2_sd_num_workers <= 0) {
    printf("[v2][state_dump] no buffer — nothing to dump\n");
    return;
  }
  char const *site_name[] = {"-",
                             "SEM_DEP_READY",
                             "DEP_SPIN",
                             "LDR_MMA",
                             "LNCH_EPILOGUE",
                             "LNCH_W_TMA",
                             "LNCH_A_TMA",
                             "LNCH_CONSUMER_DONE",
                             "CONS_TMEM_READY",
                             "CONS_MAINLOOP"};
  char const *role_name[5] = {
      "consumer", "loader", "launcher", "storer", "controller"};
  char const *loc_name[6] = {
      "-", "slot-reuse-wait", "drain", "iter-sync", "go-wait", "?"};
  printf("[v2][state_dump] ==== per-worker wedge state (wait-site words + "
         "controller mbar snapshots) ====\n");
  for (int w = 0; w < g_v2_sd_num_workers; w++) {
    unsigned long long const *b =
        g_v2_sd_host + static_cast<size_t>(w) * v2sd::WORDS_PER_WORKER;
    // In janitor-only builds (no MPK_V2_SD_MARKERS) there are no wait-site
    // flags — a wedged worker is identified by the BREADCRUMB, so print every
    // worker's snapshot (grep by worker id afterwards).
    bool wedged = false;
    for (int r = 0; r < 5; r++) {
      if (b[v2sd::OFF_WS_ROLE + r] != 0ull) {
        wedged = true;
      }
    }
    for (int p = 0; p < 14; p++) {
      if (b[v2sd::OFF_PAGE_WS + p] != 0ull) {
        wedged = true;
      }
    }
    unsigned long long const loc = b[v2sd::OFF_CTRL_LOC];
    printf("[v2][state_dump] worker=%d %s ctrl_seq=%llu ctrl_loc=%s "
           "heartbeat=%llu\n",
           w,
           wedged ? "[WEDGED]" : "[-]",
           b[v2sd::OFF_CTRL_SEQ],
           loc_name[(loc < 5) ? loc : 5],
           b[v2sd::OFF_HEARTBEAT]);
    for (int r = 0; r < 5; r++) {
      unsigned long long const ws = b[v2sd::OFF_WS_ROLE + r];
      if (ws != 0ull) {
        unsigned long long const code = ws >> 32;
        unsigned const arg = static_cast<unsigned>(ws & 0xFFFFFFFFull);
        printf("[v2][state_dump]   role=%s BLOCKED at %s arg=0x%x "
               "(stage=%u phase=%u k=%u)\n",
               role_name[r],
               (code < 10) ? site_name[code] : "?",
               arg,
               arg & 0xFF,
               (arg >> 8) & 0xFF,
               (arg >> 16) & 0xFFFF);
      }
    }
    for (int p = 0; p < 14; p++) {
      unsigned long long const pw = b[v2sd::OFF_PAGE_WS + p];
      if (pw != 0ull) {
        printf("[v2][state_dump]   loader BLOCKED at PAGE_WAIT page=%d "
               "instr=%llu (expects parity %llu)\n",
               p,
               pw >> 32,
               (pw >> 32) & 1ull);
      }
    }
    printf("[v2][state_dump]   ARRIVED raw:  %016llx %016llx %016llx\n",
           b[v2sd::OFF_ARRIVED + 0],
           b[v2sd::OFF_ARRIVED + 1],
           b[v2sd::OFF_ARRIVED + 2]);
    printf("[v2][state_dump]   FINISHED raw: %016llx %016llx %016llx\n",
           b[v2sd::OFF_FINISHED + 0],
           b[v2sd::OFF_FINISHED + 1],
           b[v2sd::OFF_FINISHED + 2]);
    printf("[v2][state_dump]   pages raw:");
    for (int p = 0; p < 14; p++) {
      printf(" %llx", b[v2sd::OFF_PAGE_RAW + p]);
    }
    printf("\n");
    for (int s = 0; s < 3; s++) {
      printf("[v2][state_dump]   dyn[slot%d]:", s);
      for (int i = 0; i < 32; i++) {
        printf(" %llx", b[v2sd::OFF_DYN + s * 32 + i]);
      }
      printf("\n");
    }
  }
  printf("[v2][state_dump] ==== end per-worker wedge state ====\n");
}
#endif

// ── Host-side: build per-SM static task plan ────────────────────────────────
// Algorithm: walk all events in order, round-robin assign each event's task
// range to workers. Matches v1 scheduler semantics closely enough to preserve
// task ordering within events.
//
// MUST stay in lockstep with the Python twin build_v2_worker_task_queues
// (python/mirage/mpk/v2_task_schedule.py): the kernel executes THIS plan,
// while the Python queues drive the SMEM page planner. Same algorithm on
// both sides (task-pushing event types, continuous round-robin cursor,
// task 1 prepended to worker 0) — divergence silently desyncs the page plan
// from the actual execution order.
//
// Reads all_events (device) by cudaMemcpying to host scratch.
// Allocates v2_per_sm_task_positions / v2_per_sm_task_offsets on device and
// fills config.v2_* fields.
inline void build_v2_plan(RuntimeConfig &config) {
  int const num_workers = config.num_workers;
  int const num_events = config.num_events;

  // Pull all_events to host
  std::vector<EventDesc> h_events(num_events);
  cudaMemcpy(h_events.data(),
             config.all_events,
             num_events * sizeof(EventDesc),
             cudaMemcpyDeviceToHost);

  // Pull first_tasks (the begin-of-graph seed tasks) to host
  // They're pushed to a specific worker when EVENT_END_OF_TASK_GRAPH fires.
  // For v2, we include task_pos=1 (begin_task_graph) in SM 0's list.

  // Per-SM task position lists (one iteration's worth)
  std::vector<std::vector<size_t>> per_sm(num_workers);

  // Round-robin assign each task-pushing event's [first, last) range.
  // EVENT_LAUNCH_TASKS / _MASSIVE_TASKS / _DEPENDENT_TASKS push tasks;
  // EVENT_END_OF_TASK_GRAPH / EVENT_TERMINATION / EVENT_EMPTY do not.
  size_t next_worker = 0;
  int pushed_events = 0;
  for (int e = 0; e < num_events; e++) {
    EventDesc const &ev = h_events[e];
    if (ev.event_type != ::mirage::runtime::EVENT_LAUNCH_TASKS &&
        ev.event_type != ::mirage::runtime::EVENT_LAUNCH_MASSIVE_TASKS &&
        ev.event_type != ::mirage::runtime::EVENT_LAUNCH_DEPENDENT_TASKS) {
      continue;
    }
    if (ev.first_task_id >= ev.last_task_id) {
      continue;
    }
    pushed_events++;
    for (size_t t = ev.first_task_id; t < ev.last_task_id; t++) {
      per_sm[next_worker % num_workers].push_back(t);
      next_worker++;
    }
  }
  // SM 0 also runs the "begin_task_graph" task (task_pos=1) that v1's
  // scheduler pushes at iter boundary via END_OF_TASK_GRAPH. We prepend
  // it so SM 0 runs it first each iter.
  per_sm[0].insert(per_sm[0].begin(), 1);

  // Flatten into offsets + positions
  std::vector<size_t> h_offsets(num_workers + 1);
  size_t total = 0;
  for (int s = 0; s < num_workers; s++) {
    h_offsets[s] = total;
    total += per_sm[s].size();
  }
  h_offsets[num_workers] = total;

  std::vector<size_t> h_positions(total);
  for (int s = 0; s < num_workers; s++) {
    std::copy(
        per_sm[s].begin(), per_sm[s].end(), h_positions.begin() + h_offsets[s]);
  }

  // Allocate on device
  size_t *d_offsets = nullptr, *d_positions = nullptr;
  cudaMalloc(&d_offsets, (num_workers + 1) * sizeof(size_t));
  cudaMalloc(&d_positions, total * sizeof(size_t));
  cudaMemcpy(d_offsets,
             h_offsets.data(),
             (num_workers + 1) * sizeof(size_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_positions,
             h_positions.data(),
             total * sizeof(size_t),
             cudaMemcpyHostToDevice);

  config.v2_per_sm_task_offsets = d_offsets;
  config.v2_per_sm_task_positions = d_positions;

  // Allocate device-side iter barrier counters (zeroed once at init)
  unsigned long long *d_sync = nullptr, *d_go = nullptr;
  cudaMalloc(&d_sync, sizeof(unsigned long long));
  cudaMalloc(&d_go, sizeof(unsigned long long));
  cudaMemset(d_sync, 0, sizeof(unsigned long long));
  cudaMemset(d_go, 0, sizeof(unsigned long long));
  config.v2_iter_sync_counter = d_sync;
  config.v2_iter_go_counter = d_go;
  config.v2_max_iters = config.max_seq_length;
  config.v2_enabled = true;

#ifdef MPK_V2_BREADCRUMB
  // Debug-only per-worker task breadcrumb. Allocate a HOST-MAPPED PINNED buffer
  // (2 u64 per worker) so the crumbs survive a context-poisoning
  // cudaErrorIllegalAddress and stay readable from the host afterwards. The
  // device sees it via cudaHostGetDevicePointer; worker_v2_kernel writes it
  // with
  // __threadfence_system(). Default build (flag unset) never reaches this — the
  // pointers stay nullptr and every device write compiles out.
  {
    // 5 role tracks per worker (consumer/loader/launcher/storer/controller),
    // 2 u64 each (STARTED, COMPLETED). All 5 role-loops run concurrently in the
    // SAME block (blockIdx.x == worker_id), so each (worker, role) pair needs a
    // private slot pair — otherwise their writes would clobber and hide which
    // role is mid-fault. Index = (worker_id * MPK_V2_BREADCRUMB_ROLES +
    // role_id) * 2 + {0=STARTED, 1=COMPLETED}. Keep MPK_V2_BREADCRUMB_ROLES in
    // lockstep with the role count in runtime_v2.cuh.
    size_t const bc_slots =
        static_cast<size_t>(num_workers) * MPK_V2_BREADCRUMB_ROLES;
    // Reserve the per-worker breadcrumb span (2 u64 per (worker,role)) FIRST,
    // then APPEND a fixed linear_v3 metadata-probe region (MPK_V2_LINV3_PROBE;
    // see linear_sm100_v3.cuh linv3_probe). Device + host derive the probe base
    // from the SAME expression: base + 2*num_workers*ROLES. The extra region is
    // tiny (LINV3_PROBE_WORDS u64) and always reserved so the offset math is
    // identical whether or not the probe is compiled.
    size_t const bc_bytes =
        2ull * bc_slots * sizeof(unsigned long long) +
        static_cast<size_t>(::kernel::linear_v3::LINV3_PROBE_WORDS_HOST) *
            sizeof(unsigned long long);
    void *bc_host = nullptr;
    cudaError_t bc_err = cudaHostAlloc(&bc_host, bc_bytes, cudaHostAllocMapped);
    if (bc_err != cudaSuccess || bc_host == nullptr) {
      printf("[v2][breadcrumb] FATAL: cudaHostAlloc(%zu bytes) failed: %s\n",
             bc_bytes,
             cudaGetErrorString(bc_err));
      abort();
    }
    // Zero the per-worker breadcrumb span so a worker that never even STARTED
    // reads (0,0) == equal (not a false fault).
    memset(bc_host, 0, bc_bytes);
    // Fill the APPENDED linear_v3 probe region with a SENTINEL so an
    // "unwritten" probe field is distinguishable from a valid 0 (e.g.
    // task_offset==0). Same base expression the device + dump use.
    {
      unsigned long long *bc_u64 = static_cast<unsigned long long *>(bc_host);
      unsigned long long *probe =
          bc_u64 +
          2ull * static_cast<size_t>(num_workers) * MPK_V2_BREADCRUMB_ROLES;
      for (int i = 0; i < ::kernel::linear_v3::LINV3_PROBE_WORDS_HOST; i++) {
        probe[i] = ::kernel::linear_v3::LINV3_PROBE_SENTINEL_HOST;
      }
    }
    void *bc_dev = nullptr;
    bc_err = cudaHostGetDevicePointer(&bc_dev, bc_host, 0);
    if (bc_err != cudaSuccess || bc_dev == nullptr) {
      printf("[v2][breadcrumb] FATAL: cudaHostGetDevicePointer failed: %s\n",
             cudaGetErrorString(bc_err));
      abort();
    }
    config.breadcrumb_host = bc_host;
    config.breadcrumb_device = bc_dev;
    config.breadcrumb_num_slots = num_workers;
    printf("[v2][breadcrumb] ENABLED: %d workers x %d roles, host=%p dev=%p "
           "(%zu bytes host-mapped pinned)\n",
           num_workers,
           (int)MPK_V2_BREADCRUMB_ROLES,
           bc_host,
           bc_dev,
           bc_bytes);
  }
#endif

#if defined(MPK_V2_BREADCRUMB) && defined(MPK_V2_STATE_DUMP)
  // Debug-only wedge state-dump buffer (see runtime_v2.cuh v2sd:: block):
  // per-worker wait-site words + controller mbar snapshots, host-mapped pinned
  // so the watchdog can decode a pure hang. Separate allocation (not appended
  // to the breadcrumb span) so no other probe's offset math changes.
  {
    size_t const sd_bytes = static_cast<size_t>(num_workers) *
                            v2sd::WORDS_PER_WORKER * sizeof(unsigned long long);
    void *sd_host = nullptr;
    cudaError_t sd_err = cudaHostAlloc(&sd_host, sd_bytes, cudaHostAllocMapped);
    if (sd_err != cudaSuccess || sd_host == nullptr) {
      printf("[v2][state_dump] FATAL: cudaHostAlloc(%zu) failed: %s\n",
             sd_bytes,
             cudaGetErrorString(sd_err));
      abort();
    }
    memset(sd_host, 0, sd_bytes);
    void *sd_dev = nullptr;
    sd_err = cudaHostGetDevicePointer(&sd_dev, sd_host, 0);
    if (sd_err != cudaSuccess || sd_dev == nullptr) {
      printf("[v2][state_dump] FATAL: cudaHostGetDevicePointer failed: %s\n",
             cudaGetErrorString(sd_err));
      abort();
    }
    sd_err = cudaMemcpyToSymbol(
        mirage::runtime_v2::g_v2_sd_buf, &sd_dev, sizeof(sd_dev));
    if (sd_err != cudaSuccess) {
      printf("[v2][state_dump] FATAL: cudaMemcpyToSymbol failed: %s\n",
             cudaGetErrorString(sd_err));
      abort();
    }
    g_v2_sd_host = static_cast<unsigned long long *>(sd_host);
    g_v2_sd_num_workers = num_workers;
    printf("[v2][state_dump] ENABLED: %d workers x %d words, host=%p dev=%p\n",
           num_workers,
           v2sd::WORDS_PER_WORKER,
           sd_host,
           sd_dev);
  }
#endif

  size_t max_per_sm = 0;
  for (int s = 0; s < num_workers; s++) {
    size_t n = h_offsets[s + 1] - h_offsets[s];
    if (n > max_per_sm) {
      max_per_sm = n;
    }
  }
  printf("[v2] static plan built: %d workers, %zu total tasks/iter "
         "(avg %zu/SM, max %zu/SM, pushed_events=%d)\n",
         num_workers,
         total,
         total / num_workers,
         max_per_sm,
         pushed_events);
}

#ifdef MPK_V2_BREADCRUMB
// Host-side breadcrumb dump. Called right after the launch returns an error
// (the host-mapped pinned buffer survives the context poisoning). Prints every
// (worker, role) whose STARTED word != COMPLETED word — the task(s) that were
// executing when the fault hit. Task-type is printed NUMERICALLY here; the
// Python helper (scratch/v2_breadcrumb_readback.py) maps it to a name via
// profiler_persistent.event_name_list. Role ids match runtime_v2.cuh.
inline void dump_breadcrumb(RuntimeConfig const &config) {
  if (config.breadcrumb_host == nullptr || config.breadcrumb_num_slots <= 0) {
    printf("[v2][breadcrumb] no buffer (not enabled?) — nothing to dump\n");
    return;
  }
  unsigned long long const *bc =
      static_cast<unsigned long long const *>(config.breadcrumb_host);
  char const *role_name[MPK_V2_BREADCRUMB_ROLES] = {
      "consumer", "loader", "launcher", "storer", "controller"};
  int n_fault = 0;
  printf("[v2][breadcrumb] ==== IN-FLIGHT TASK(S) at crash (STARTED != "
         "COMPLETED) — fault candidate set ====\n");
  printf(
      "[v2][breadcrumb] NOTE: an illegal address poisons the WHOLE context, "
      "so every task that happened to be in flight (across all workers/roles) "
      "shows here — the true faulter is IN this set, not necessarily unique. "
      "Cross-check with the CUDA error + re-run (the crash is "
      "deterministic).\n");
  for (int w = 0; w < config.breadcrumb_num_slots; w++) {
    for (int r = 0; r < MPK_V2_BREADCRUMB_ROLES; r++) {
      size_t const idx =
          (static_cast<size_t>(w) * MPK_V2_BREADCRUMB_ROLES + r) * 2ull;
      unsigned long long const started = bc[idx + 0];
      unsigned long long const completed = bc[idx + 1];
      if (started != completed) {
        // Decode STARTED word: [63:40]=iter [39:24]=seq [23:8]=task_type
        //                      [7:0]=role. (Controller uses task_pos in the
        //                      task_type field — see runtime_v2.cuh.)
        unsigned long long const iter = (started >> 40) & 0xFFFFFF;
        unsigned long long const seq = (started >> 24) & 0xFFFF;
        unsigned long long const ttype_or_pos = (started >> 8) & 0xFFFF;
        unsigned long long const role = started & 0xFF;
        char const *rn =
            (role < (unsigned)MPK_V2_BREADCRUMB_ROLES) ? role_name[role] : "?";
        if (role == 4) { // controller: field is task_pos
          printf("[v2][breadcrumb] IN-FLIGHT: worker=%d role=%s(%llu) "
                 "iter=%llu seq_in_iter=%llu task_pos=%llu  "
                 "(STARTED=0x%016llx COMPLETED=0x%016llx)\n",
                 w,
                 rn,
                 role,
                 iter,
                 seq,
                 ttype_or_pos,
                 started,
                 completed);
        } else {
          printf("[v2][breadcrumb] IN-FLIGHT: worker=%d role=%s(%llu) "
                 "iter=%llu seq_in_iter=%llu task_type=%llu  "
                 "(STARTED=0x%016llx COMPLETED=0x%016llx)\n",
                 w,
                 rn,
                 role,
                 iter,
                 seq,
                 ttype_or_pos,
                 started,
                 completed);
        }
        n_fault++;
      }
    }
  }
  if (n_fault == 0) {
    printf("[v2][breadcrumb] no in-flight task found (all STARTED==COMPLETED). "
           "The fault may be OUTSIDE execute_task (e.g. controller drain, "
           "iter-barrier, or a role with no crumb), or a non-instrumented "
           "path. Raw buffer follows for manual inspection:\n");
    for (int w = 0; w < config.breadcrumb_num_slots; w++) {
      for (int r = 0; r < MPK_V2_BREADCRUMB_ROLES; r++) {
        size_t const idx =
            (static_cast<size_t>(w) * MPK_V2_BREADCRUMB_ROLES + r) * 2ull;
        if (bc[idx + 0] != 0ull || bc[idx + 1] != 0ull) {
          printf("[v2][breadcrumb]   w=%d %s: STARTED=0x%016llx "
                 "COMPLETED=0x%016llx\n",
                 w,
                 role_name[r],
                 bc[idx + 0],
                 bc[idx + 1]);
        }
      }
    }
  }
  printf("[v2][breadcrumb] ==== %d in-flight (worker,role) slot(s) at crash — "
         "the illegal access is INSIDE one of these in-flight slots/roles "
         "(consumer/loader/launcher/storer = a task body; controller = "
         "fetch/publish) ====\n",
         n_fault);
}
#endif

#ifdef MPK_V2_LINV3_PROBE
// Host-side decode of the linear_v3 metadata/phase probe region (appended to
// the pinned breadcrumb buffer; survives the context-poisoning fault). Reads
// the SAME base expression the device writes. Prints each field + a validity
// flag so ONE box session decides: metadata bad (task_offset>=num_tiles / null
// or garbage ptr / bad SMEM region) vs which phase reached the fault (the LAST
// marker value per role) vs concurrent poisoner (all markers completed AND
// metadata valid).
inline void dump_linv3_probe(RuntimeConfig const &config) {
  namespace P = ::kernel::linear_v3::linv3_probe;
  if (config.breadcrumb_host == nullptr || config.breadcrumb_num_slots <= 0) {
    printf("[v2][linv3_probe] no buffer (MPK_V2_BREADCRUMB not enabled?) — "
           "nothing to dump\n");
    return;
  }
  unsigned long long const *bc =
      static_cast<unsigned long long const *>(config.breadcrumb_host);
  unsigned long long const *p =
      bc + 2ull * static_cast<size_t>(config.breadcrumb_num_slots) *
               MPK_V2_BREADCRUMB_ROLES;

  auto w = [&](int slot) -> unsigned long long { return p[slot]; };
  auto is_unwritten = [&](int slot) -> bool { return p[slot] == P::SENTINEL; };

  printf("[v2][linv3_probe] ==== linear_v3 worker=%d seq_in_iter=%d probe "
         "(SENTINEL=0x%016llx means UNWRITTEN) ====\n",
         P::TARGET_WORKER,
         P::TARGET_SEQ_IN_ITER,
         (unsigned long long)P::SENTINEL);

  if (is_unwritten(P::S_MAGIC)) {
    printf(
        "[v2][linv3_probe] LOADER block UNWRITTEN (S_MAGIC==SENTINEL): the "
        "loader body was NEVER entered for worker=%d seq_in_iter=%d. Either "
        "this worker/seq is not the lm_head tile (retarget), or the fault "
        "hit BEFORE the loader body (codegen loader page-prefix / dispatch). "
        "Check the phase markers below.\n",
        P::TARGET_WORKER,
        P::TARGET_SEQ_IN_ITER);
  } else {
    unsigned long long const to = w(P::S_TASK_OFFSET);
    unsigned long long const nt = w(P::S_NUM_TILES);
    unsigned long long const valid = w(P::S_TASK_OFFSET_VALID);
    printf("[v2][linv3_probe] --- LOADER metadata block ---\n");
    printf("[v2][linv3_probe]   task_type      = %llu (expect 244 "
           "TASK_LINEAR_SM100_V3)\n",
           w(P::S_TASK_TYPE));
    printf("[v2][linv3_probe]   variant_id     = %llu\n", w(P::S_VARIANT_ID));
    printf("[v2][linv3_probe]   task_offset    = %llu   %s (num_tiles=%llu; "
           "valid_flag=%llu)  <<< task_offset>=num_tiles IS THE BUG\n",
           to,
           (valid == 1ull) ? "[VALID]" : "[*** OUT-OF-RANGE / BUG ***]",
           nt,
           valid);
    printf("[v2][linv3_probe]   N_real         = %llu (expect 129280)\n",
           w(P::S_N_REAL));
    printf("[v2][linv3_probe]   K              = %llu\n", w(P::S_K));
    printf("[v2][linv3_probe]   my_count       = %llu\n", w(P::S_MY_COUNT));
    printf("[v2][linv3_probe]   instruction_idx= %llu\n", w(P::S_INSTR_IDX));
    printf("[v2][linv3_probe]   seq_in_iter    = %llu (expect %d)\n",
           w(P::S_SEQ_IN_ITER),
           P::TARGET_SEQ_IN_ITER);
    printf("[v2][linv3_probe]   iter_num       = %llu\n", w(P::S_ITER_NUM));
    printf("[v2][linv3_probe]   dependent_event= 0x%016llx (idx=%llu)\n",
           w(P::S_DEP_EVENT),
           w(P::S_DEP_EVENT_IDX));
    printf("[v2][linv3_probe]   raw_payload    = 0x%016llx\n",
           w(P::S_RAW_PAYLOAD));
    printf("[v2][linv3_probe]   input_ptrs[0] A= 0x%016llx  %s\n",
           w(P::S_IN_PTR0),
           (w(P::S_IN_PTR0) == 0ull) ? "[*** NULL ***]" : "");
    printf("[v2][linv3_probe]   input_ptrs[1] W= 0x%016llx  %s\n",
           w(P::S_IN_PTR1),
           (w(P::S_IN_PTR1) == 0ull) ? "[*** NULL ***]" : "");
    printf("[v2][linv3_probe]   output_ptrs[0]C= 0x%016llx  %s\n",
           w(P::S_OUT_PTR0),
           (w(P::S_OUT_PTR0) == 0ull) ? "[*** NULL ***]" : "");
    printf("[v2][linv3_probe]   A_tma_desc     = 0x%016llx\n", w(P::S_A_DESC));
    printf("[v2][linv3_probe]   W_tma_desc     = 0x%016llx\n", w(P::S_W_DESC));
    printf("[v2][linv3_probe]   num_smem_regs  = %llu\n", w(P::S_NUM_REGIONS));
    printf("[v2][linv3_probe]   smem_bad_mask  = 0x%016llx  %s <<< nonzero = a "
           "malformed SMEM region (copied-TaskDesc bug)\n",
           w(P::S_SMEM_BAD_MASK),
           (w(P::S_SMEM_BAD_MASK) == 0ull) ? "[ok]" : "[*** BAD REGION ***]");
  }

  // Phase markers: the LAST value reached localizes where the fault hit.
  auto phase_str = [&](int slot) {
    if (is_unwritten(slot)) {
      return std::string("UNWRITTEN(prefix or never-entered)");
    }
    return std::to_string((unsigned long long)p[slot]);
  };
  printf("[v2][linv3_probe] --- PHASE markers (last-reached localizes the "
         "fault) ---\n");
  printf(
      "[v2][linv3_probe]   loader   phase = %s   (1 entered, 2 meta-dumped, "
      "3 reinit, 4 before-Wtma, 5 after-Wtma, 6 after-dep-wait, 7 loop-done, "
      "100 SKIP-return)\n",
      phase_str(P::S_LD_PHASE).c_str());
  printf(
      "[v2][linv3_probe]   launcher phase = %s (iter=%s)   (1 entered, "
      "2 reinit, 3 after-alloc, 4 tmem-ready-arrived, 5 mma-loop-done, "
      "6 pages-released, 7 consumer-done-waited, 8 after-dealloc; SKIP path: "
      "101 skip-tmem-arrived, 102 skip-pages-released, 103 skip-return)\n",
      phase_str(P::S_LC_PHASE).c_str(),
      phase_str(P::S_LC_ITER).c_str());
  printf("[v2][linv3_probe]   consumer phase = %s (iter=%s)   (1 entered, "
         "2 tmem-ready-waited, 3 after-taddr-read, 4 before-first-store, "
         "5 after-first-store, 6 loop-done, 7 consumer-done-arrived, "
         "100 SKIP-return)\n",
         phase_str(P::S_CN_PHASE).c_str(),
         phase_str(P::S_CN_ITER).c_str());
  printf(
      "[v2][linv3_probe]   (NOTE: if launcher/consumer iter != loader "
      "iter_num above, that role did NOT re-enter the crash iteration => its "
      "phase is STALE (died in that role's codegen prefix before the "
      "body)).\n");
  if (!is_unwritten(P::S_LC_TADDR)) {
    printf(
        "[v2][linv3_probe]   launcher taddr = %llu; smem_base_lo12 = 0x%llx; "
        "scratch_off = %llu; scratch_pg = %llu\n",
        w(P::S_LC_TADDR),
        w(P::S_LC_SMEM_BASE_LO12),
        w(P::S_LC_SCRATCH_OFF),
        w(P::S_LC_SCRATCH_PG));
  }
  if (!is_unwritten(P::S_CN_C_PTR)) {
    printf(
        "[v2][linv3_probe]   consumer C_ptr = 0x%016llx; taddr(read) = %llu\n",
        w(P::S_CN_C_PTR),
        w(P::S_CN_TADDR));
  }
  printf(
      "[v2][linv3_probe] VERDICT GUIDE: (a) task_offset OUT-OF-RANGE or a "
      "NULL/garbage ptr or nonzero smem_bad_mask => COPIED-TaskDesc metadata "
      "bug for worker36's tile. (b) a phase marker STUCK mid-body (loader "
      "5/6, launcher 3/4, consumer 3/4) => the fault is in THAT phase's "
      "access. (c) ALL markers completed (loader 7, launcher 7/8, consumer "
      "7) AND metadata valid => worker36's body finished cleanly => the "
      "faulter is CONCURRENT (a poisoner), NOT this tile => re-run with "
      "SKIP36 to confirm (crash persists => concurrent).\n");

  // ── PER-WORKER RAW W-TMA argument records (M3 debug) ──────────────────────
  // The primary M3 deliverable: for EVERY loader TASK_LINEAR_SM100_V3 the exact
  // operands its FIRST cp.async.bulk.tensor W-load saw, in a private per-worker
  // slot (so the faulting worker's record is never overwritten). The faulting
  // loader = the slot with WT_DST/WT_TMAP dumped (WT_MAGIC written) but
  // WT_COMPLETED still SENTINEL. For each such worker (and any with a bad
  // operand) we print the dst (+ dst&1023 alignment), tmap (null check),
  // coords, box dims, gmem base + source byte-offset, and the coord/source
  // bounds so the bad argument is pinned. Region base = p + LINV3_META_WORDS,
  // worker w at + w*WTMA_WORDS_PER_WORKER (same expr as the device wtma_slot).
  unsigned long long const *wt_base = p + P::LINV3_META_WORDS;
  printf("[v2][linv3_probe] --- PER-WORKER W-TMA records (workers 0..%d; the "
         "FAULTING loader = WT_MAGIC written but WT_COMPLETED still SENTINEL) "
         "---\n",
         P::WTMA_MAX_WORKERS - 1);
  int wt_written = 0;
  int wt_faulting = 0;
  int wt_bad_operand = 0;
  for (int wkr = 0; wkr < P::WTMA_MAX_WORKERS; wkr++) {
    unsigned long long const *s =
        wt_base + static_cast<size_t>(wkr) * P::WTMA_WORDS_PER_WORKER;
    if (s[P::WT_MAGIC] == P::SENTINEL) {
      continue; // this worker never ran a linear_v3 loader first-W-TMA
    }
    wt_written++;
    bool const completed = (s[P::WT_COMPLETED] == P::WT_COMPLETED_MAGIC);
    unsigned long long const dst = s[P::WT_DST];
    unsigned long long const tmap = s[P::WT_TMAP];
    unsigned long long const coord_y = s[P::WT_COORD_Y];
    unsigned long long const coord_z = s[P::WT_COORD_Z];
    unsigned long long const box_d1 = s[P::WT_BOX_D1];
    unsigned long long const box_d2 = s[P::WT_BOX_D2];
    unsigned long long const gmem = s[P::WT_GMEM_BASE];
    unsigned long long const src_off = s[P::WT_SRC_OFF];
    unsigned long long const n_real = s[P::WT_N_REAL];
    unsigned long long const k = s[P::WT_K];
    // M3 FAULTING-load extension fields (the recorded load is the FROZEN first
    // bad load, or the last good load if none was bad).
    unsigned long long const load_idx = s[P::WT_LOAD_IDX];
    unsigned long long const num_loads = s[P::WT_NUM_LOADS];
    unsigned long long const src_limit = s[P::WT_SRC_LIMIT];
    unsigned long long const bad_flag = s[P::WT_BAD_FLAG];
    // Re-derive the bad-operand checks host-side (cross-check the device
    // WT_BAD_FLAG bitmask): dst must be 1024-aligned (128B-swizzle tile), tmap
    // non-null, coords in bounds (row: y+box_d1<=N_real; k-chunk:
    // z+box_d2<=K/64 — a box top EQUAL to the dim is in-bounds, so `>` is the
    // OOB test), and the source byte-range must fit the buffer
    // (src_off+W_SIZE<=N_real*K*2; a hit here with in-bounds coords means the
    // descriptor N/K disagrees with the kernel's N_real/K = a metadata
    // mismatch).
    bool const dst_misaligned = (dst & 1023ull) != 0ull;
    bool const tmap_null = (tmap == 0ull);
    bool const row_oob = (coord_y + box_d1 > n_real);
    bool const kchunk_oob =
        (k > 0ull) ? (coord_z + box_d2 > (k / 64ull)) : false;
    // W_SIZE = 32768 (BLOCK_M*BLOCK_K*2) = per-load tile bytes.
    bool const src_oob =
        (src_limit > 0ull)
            ? (src_off + (unsigned long long)::kernel::linear::W_SIZE >
               src_limit)
            : false;
    bool const bad =
        dst_misaligned || tmap_null || row_oob || kchunk_oob || src_oob;
    if (!completed) {
      wt_faulting++;
    }
    if (bad) {
      wt_bad_operand++;
    }
    // Only print the interesting rows (faulting or bad-operand) plus a couple
    // of clean examples would be noisy across 136 workers — print faulting /
    // bad-operand rows in full, and summarize the rest.
    if (!completed || bad) {
      printf(
          "[v2][linv3_probe]   worker=%d  %s%s  seq_in_iter=%llu iter=%llu\n"
          "[v2][linv3_probe]       load# = %llu of %llu   (RECORDED load = "
          "frozen-first-bad, or last-good if no bad)  device WT_BAD_FLAG = "
          "0x%llx %s\n"
          "[v2][linv3_probe]       W_smem(dst) = %llu (0x%llx)  dst&1023 = "
          "%llu  %s\n"
          "[v2][linv3_probe]       W_tmap      = 0x%016llx  %s\n"
          "[v2][linv3_probe]       coords x,y,z= %llu, %llu, %llu   box "
          "d0,d1,d2 = %llu, %llu, %llu\n"
          "[v2][linv3_probe]       gmem_base   = 0x%016llx  %s\n"
          "[v2][linv3_probe]       src_off(B)  = %llu   (row y+box_d1=%llu vs "
          "N_real=%llu %s; kchunk z+box_d2=%llu vs K/64=%llu %s; "
          "src_off+W_SIZE="
          "%llu vs buf_bytes=%llu %s)\n",
          wkr,
          completed ? "[COMPLETED]" : "[*** FAULTING: WT_COMPLETED UNSET ***]",
          bad ? " [*** BAD OPERAND ***]" : "",
          s[P::WT_SEQ_IN_ITER],
          s[P::WT_ITER_NUM],
          load_idx,
          num_loads,
          bad_flag,
          (bad_flag == 0ull)
              ? "[clean per device]"
              : "[bits: 1=dst-misalign 2=tmap-null 4=row-OOB 8=kchunk-OOB "
                "16=src-OOB]",
          dst,
          dst,
          (dst & 1023ull),
          dst_misaligned ? "[*** NOT 1024-ALIGNED ***]" : "[1024-aligned ok]",
          tmap,
          tmap_null ? "[*** NULL TENSOR-MAP ***]" : "",
          s[P::WT_COORD_X],
          coord_y,
          coord_z,
          s[P::WT_BOX_D0],
          box_d1,
          box_d2,
          gmem,
          (gmem == 0ull) ? "[*** NULL GMEM BASE ***]" : "",
          src_off,
          coord_y + box_d1,
          n_real,
          row_oob ? "[*** ROW OOB ***]" : "ok",
          coord_z + box_d2,
          (k / 64ull),
          kchunk_oob ? "[*** KCHUNK OOB ***]" : "ok",
          src_off + (unsigned long long)::kernel::linear::W_SIZE,
          src_limit,
          src_oob ? "[*** SRC OOB (N_real/K vs descriptor MISMATCH) ***]"
                  : "ok");
    }
  }
  printf(
      "[v2][linv3_probe]   W-TMA summary (EVERY-LOAD capture): %d worker "
      "slot(s) written, %d FAULTING (recorded load's WT_COMPLETED unset), %d "
      "with a BAD OPERAND (frozen first-bad load).\n",
      wt_written,
      wt_faulting,
      wt_bad_operand);
  if (wt_written == 0) {
    printf("[v2][linv3_probe]   (no per-worker W-TMA slot written — no "
           "TASK_LINEAR_SM100_V3 loader reached its first W-TMA, or the probe "
           "region base math is off. If the fault is at the lm_head W-TMA this "
           "MUST be non-empty; if empty, the loader died BEFORE the W-TMA "
           "capture point — read the phase markers above.)\n");
  } else if (wt_bad_operand == 0) {
    printf(
        "[v2][linv3_probe]   VERDICT: NO load (across ALL t*iters loads of "
        "every worker) tripped a static OOB/misalign check — every recorded "
        "operand is 1024-aligned dst, non-null tmap, in-bounds coord AND "
        "in-buffer source. The illegal address is therefore NOT a statically-"
        "detectable W-TMA operand. Remaining causes: (1) N_real/K the kernel "
        "uses AGREE with the descriptor here, so the descriptor's baked dims "
        "(tensor_desc.dim[0]/dim[1] in tma.cuh param_id==1) are the only bound "
        "the probe cannot see — but src-OOB would have flagged a coarse "
        "mismatch, so a subtle one remains possible; (2) a 128B-swizzle / "
        "descriptor-internal issue (swizzle vs "
        "1024-aligned-but-not-swizzle-tile "
        "dst, stride/interleave); (3) a concurrent poisoner (confirm SKIPALL "
        "makes it vanish). ESCALATE to a swizzle/descriptor audit — do NOT "
        "attribute to a coord/source operand.%s\n",
        (wt_faulting > 0)
            ? " NOTE: some records are FAULTING (WT_COMPLETED unset) with "
              "clean "
              "operands — that is the async-TMA signature: the recorded load's "
              "issue returned but the fault surfaced later; inspect those rows."
            : "");
  } else {
    printf("[v2][linv3_probe]   VERDICT: inspect the BAD-OPERAND worker row(s) "
           "above — the RECORDED load is the FROZEN first bad load (load# "
           "shown), and its device WT_BAD_FLAG bitmask names the exact tripped "
           "check (1=dst-not-1024-aligned / 2=null-tmap / 4=row-OOB / "
           "8=kchunk-OOB / 16=src-OOB=N_real-vs-descriptor mismatch). That is "
           "the illegal-address cause at linear_device.cuh:88 — the targeted "
           "fix follows from WHICH bit is set.\n");
  }

  printf("[v2][linv3_probe] ==== end linv3 probe ====\n");
}
#endif // MPK_V2_LINV3_PROBE

// ── Device kernel: advance per-iter state ──────────────────────────────────
// Runs prepare_next_batch (paged KV bookkeeping + token append) once per iter
// between worker_v2_kernel launches. Single block, single warp is enough —
// prepare_next_batch is serial inner loops, ~tens of μs.
#if defined(MODE_OFFLINE) || defined(MODE_ONLINE) ||                           \
    defined(MODE_ONLINE_NOTOKEN)
__global__ void v2_iter_advance_kernel(RuntimeConfig config,
                                       int end_of_task_graph_event_pos) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
#ifdef MODE_ONLINE_NOTOKEN
  (void)::prepare_next_batch(config, 0);
#else
  (void)::prepare_next_batch(config);
#endif
}
#endif

} // namespace runtime_v2
} // namespace mirage

// ── C entry points (global scope so HARD_CODE can call them unqualified) ───
// Must be called after init_persistent_kernel + build_v2_plan (one-time setup).
// Each launch call runs one decode step.
// Reset iter barrier counters before each launch.
__global__ inline void
    v2_reset_counters_kernel(unsigned long long *sync_counter,
                             unsigned long long *go_counter) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *sync_counter = 0;
    *go_counter = 0;
  }
}

extern "C" inline void
    launch_persistent_kernel_v2(cudaStream_t default_stream) {
  // 1. Reset task-queue state (zeroes event counters) — v1's prepare_kernel.
  int end_of_task_graph_event_pos = global_runtime_config.num_events - 1;
  ::prepare_kernel<<<dim3(global_runtime_config.num_workers, 1, 1),
                     dim3(128, 1, 1),
                     0,
                     default_stream>>>(global_runtime_config,
                                       end_of_task_graph_event_pos);

  // 2. Reset v2 iter-barrier counters.
  v2_reset_counters_kernel<<<1, 1, 0, default_stream>>>(
      global_runtime_config.v2_iter_sync_counter,
      global_runtime_config.v2_iter_go_counter);

  // 3. Single persistent kernel launch — device loops through decode steps
  //    via the per-SM static plan, calls prepare_next_batch on SM 0 at iter
  //    boundaries, and terminates on step >= max_seq_length.
  mirage::runtime_v2::launch_worker_v2(
      global_runtime_config, global_runtime_config.num_workers, default_stream);

#ifdef MPK_V2_BREADCRUMB
  // ── Host-side HANG WATCHDOG (default-OFF) ──────────────────────────────────
  // On a pure HANG (a device spin-wait, no CUDA error), the host blocks forever
  // in cudaStreamSynchronize below and the breadcrumb is never dumped (it is
  // only dumped on a launch *error*). This watchdog closes that gap: if the env
  // var MPK_V2_HANG_WATCHDOG_S is set to a positive integer N (and the run was
  // compiled with MPK_V2_BREADCRUMB so the pinned buffer exists), spawn a host
  // std::thread that starts its timer HERE (kernel-launch time — so N seconds
  // is N seconds into the actual kernel run/hang, NOT wall time since process
  // start; the ~10-15min JIT happened earlier at mpk.compile()). If the launch
  // hasn't returned after N seconds, the thread dumps the breadcrumb DIRECTLY
  // from the host-mapped pinned buffer (plain host memory — readable during a
  // live device hang, NO CUDA call needed), flushes, and _Exit()s to break the
  // hang so the captured stdout contains the STARTED-not-COMPLETED fault set.
  //
  // Data-race note: the pinned buffer is written monotonically by the live
  // device workers (STARTED then COMPLETED, __threadfence_system). A concurrent
  // host read during the hang can at worst observe an in-progress marker (a
  // STARTED with COMPLETED not yet advanced) — which is EXACTLY the in-flight
  // signal we want. A torn 64-bit read is possible in principle but the words
  // are only ever written 0 -> valued once per iter/task and we only need to
  // distinguish "STARTED != COMPLETED"; an in-progress marker is acceptable for
  // a diagnostic. _Exit() (not abort()) avoids running atexit/global dtors that
  // could themselves block on the poisoned/hung CUDA context.
  std::atomic<bool> v2_launch_done{false};
  std::thread v2_watchdog;
  int v2_watchdog_secs = 0;
  {
    char const *wd = std::getenv("MPK_V2_HANG_WATCHDOG_S");
    if (wd != nullptr && wd[0] != '\0') {
      v2_watchdog_secs = std::atoi(wd);
    }
  }
  if (v2_watchdog_secs > 0) {
    printf("[v2][watchdog] ARMED: will dump breadcrumb + _Exit if the kernel "
           "launch does not return within %d s (timer starts NOW, at "
           "kernel-launch)\n",
           v2_watchdog_secs);
    fflush(stdout);
    int const secs = v2_watchdog_secs;
    std::atomic<bool> *done_ptr = &v2_launch_done;
    v2_watchdog = std::thread([secs, done_ptr]() {
      // Poll in 100 ms slices so a normal (fast) launch cancels the watchdog
      // promptly and we do not linger a whole timeout window per decode step.
      long const total_ms = static_cast<long>(secs) * 1000L;
      long waited_ms = 0;
      while (waited_ms < total_ms) {
        if (done_ptr->load(std::memory_order_acquire)) {
          return; // launch returned — stand down
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        waited_ms += 100;
      }
      if (done_ptr->load(std::memory_order_acquire)) {
        return; // returned right at the deadline
      }
      // Timeout with the launch still outstanding => HANG. Dump directly from
      // the host-mapped pinned breadcrumb (host memory; no CUDA call).
      printf("\n[v2][watchdog] *** HANG DETECTED: kernel launch did not return "
             "within %d s. Dumping the per-(worker,role) breadcrumb from the "
             "host-mapped pinned buffer (STARTED != COMPLETED = the task(s) "
             "in flight when the hang began), then _Exit(134) to break the "
             "hang. ***\n",
             secs);
      fflush(stdout);
      mirage::runtime_v2::dump_breadcrumb(global_runtime_config);
      fflush(stdout);
#ifdef MPK_V2_LINV3_PROBE
      mirage::runtime_v2::dump_linv3_probe(global_runtime_config);
      fflush(stdout);
#endif
#if defined(MPK_V2_STATE_DUMP)
      mirage::runtime_v2::dump_v2_state();
      fflush(stdout);
#endif
      printf("[v2][watchdog] *** breadcrumb dumped; forcing _Exit(134) to "
             "escape the hang ***\n");
      fflush(stdout);
      std::fflush(nullptr);
      std::_Exit(134);
    });
  }
#endif // MPK_V2_BREADCRUMB

  cudaError_t err = cudaStreamSynchronize(default_stream);

#ifdef MPK_V2_BREADCRUMB
  // Launch returned (normally or with an error) BEFORE the watchdog fired.
  // Signal the watchdog to stand down and reap it so its timer does not carry
  // into a subsequent launch.
  if (v2_watchdog.joinable()) {
    v2_launch_done.store(true, std::memory_order_release);
    v2_watchdog.join();
  }
#endif
  if (err != cudaSuccess) {
    printf("[v2] worker_v2_kernel error: %s\n", cudaGetErrorString(err));
#ifdef MPK_V2_BREADCRUMB
    // The host-mapped pinned breadcrumb survives the context poisoning; decode
    // which task was in flight when the fault hit.
    mirage::runtime_v2::dump_breadcrumb(global_runtime_config);
    fflush(stdout);
#endif
#if defined(MPK_V2_STATE_DUMP)
    mirage::runtime_v2::dump_v2_state();
    fflush(stdout);
#endif
#ifdef MPK_V2_LINV3_PROBE
    // Decode the linear_v3 metadata/phase probe (worker36 lm_head tile).
    mirage::runtime_v2::dump_linv3_probe(global_runtime_config);
    fflush(stdout);
#endif
  }
#ifdef MPK_V2_LINV3_PROBE
  else {
    // No fault: the probe still records "did worker36's target tile run and
    // complete cleanly?" — a useful verdict when PROBE is combined with SKIP36
    // (target completed, no fault => the skipped body was the culprit).
    mirage::runtime_v2::dump_linv3_probe(global_runtime_config);
    fflush(stdout);
  }
#endif
}

// Must be called once, AFTER init_persistent_kernel, BEFORE first launch.
extern "C" inline void init_persistent_kernel_v2() {
  mirage::runtime_v2::build_v2_plan(global_runtime_config);
}
