/* Copyright 2026 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
#include "mirage/kernel/v2_role_codegen.h"

#include <cassert>

namespace mirage {
namespace kernel {

namespace {

namespace rt = mirage::runtime;
namespace tr = mirage::transpiler;

enum class V2Role {
  InitSemaphores,
  Loader,
  Launcher,
  Consumer,
  Storer,
};

std::string const &role_body(rt::TaskRoleVariantCode const &code, V2Role role) {
  switch (role) {
    case V2Role::InitSemaphores:
      return code.init_semaphores;
    case V2Role::Loader:
      return code.loader;
    case V2Role::Launcher:
      return code.launcher;
    case V2Role::Consumer:
      return code.consumer;
    case V2Role::Storer:
      return code.storer;
  }
  return code.consumer;
}

// Phase 3.5: page-lifecycle prefix at start of every loader body
// (MegaKernels NoOp/matvec pattern, lane-parallel). For each physical page:
//   - lane K waits for the previous task's release of page K
//   - if THIS task does not use page K, lane K arrives page K right away
//     (the "claim+release ASAP" pattern — frees pages the task doesn't
//     touch so the next task's loader can re-TMA into them sooner)
// Pages this task uses are released later by the consumer suffix instead.
// Net: every page gets exactly one arrive per task.
char const *kLoaderPagePrefix =
    "{\n"
    "  int const _lane = threadIdx.x & 31;\n"
    "#ifdef MPK_ENABLE_PROFILING\n"
    "  bool const _pg_prof = _lane == 0 &&\n"
    "      runtime_config.profiler_buffer != nullptr &&\n"
    "      iter_num + V2_PROF_WINDOW_ITERS >= runtime_config.v2_max_iters;\n"
    "  if (_pg_prof) {\n"
    "    v2_prof_emit(runtime_config.profiler_buffer,\n"
    "                 V2_PROF_GROUP_LOADER_PHASE, V2_PROF_PAGE_WAIT,\n"
    "                 tb::EVENT_BEGIN);\n"
    "  }\n"
    "#endif\n"
    "  if (_lane < MAX_SMEM_PAGES_PER_TASK) {\n"
    "    runtime_wait_page_ready(runtime_smem, _lane, instruction_index);\n"
    "    if (!task_uses_page(task_desc, _lane)) {\n"
    "      runtime_finish_page(runtime_smem, _lane, 1);\n"
    "    }\n"
    "  }\n"
    "  __syncwarp();\n"
    "#ifdef MPK_ENABLE_PROFILING\n"
    "  if (_pg_prof) {\n"
    "    v2_prof_emit(runtime_config.profiler_buffer,\n"
    "                 V2_PROF_GROUP_LOADER_PHASE, V2_PROF_PAGE_WAIT,\n"
    "                 tb::EVENT_END);\n"
    "  }\n"
    "#endif\n"
    "}\n";

// Design E (race-2 fix, 2026-07-16; hardened 2026-07-17): the consumer
// CLAIMS its used pages (this block, emitted BEFORE the body) for BOTH
// opt-in modes — consumer-owned (SkipUsed loader, used-only suffix;
// rmsnorm/silu) and consumer-TOTAL (no loader page code, suffix releases
// ALL pages; the DSv3 FFN GEMV chains). Both require the structural
// predicate (auto_consumer_finish AND all helper-role bodies empty — the
// consumer is provably the sole toucher of the used pages).
//
// Why the claim moved onto the consumer (race 2, fixed in 7d271a01): the
// shared W4 loader lags structurally, so a fast consumer could complete the
// whole task INCLUDING the release suffix before the lagging loader's
// wait-all prefix waited the used pages — parity advanced one use ahead and
// the loader wedged forever. Program-ordering claim -> body -> release on
// one warp-group closes that window.
//
// Why the FFN GEMV chains need the loader FULLY out (the 7d271a01 SkipUsed
// half was broken THERE — root-caused 2026-07-17 from the nwarps=4 FFN
// GEMV chain wedge): SkipUsed
// made the loader's page observations SPARSE — it never waited pages the
// current task uses. Page-parity waits are mod-2 phase waits (waiter at
// sequence s passes iff releases(p) ≡ s (mod 2)); their correctness relies
// on DENSE observation (the wait-all prefix pins releases(p) == s exactly
// at every step via the loader's own program order). With SkipUsed, a page
// used by >= 2 consecutive tasks gets NO loader wait at those sequences, so
// the loader's next wait on it (e.g. a zero-region task's all-unused
// prefix) can execute 2 releases early — indistinguishable mod 2 from
// on-time (boot phase 1 != parity 0) — and RELEASE out of order, advancing
// the page one occurrence ahead and permanently wedging the intermediate
// consumer claim. Reproduced 3/3 (nwarps=4 FFN local8 chain under SM
// contention): router{0,1} -> w13{0..4} -> silu{} — the loader blazed to
// silu's prefix while the consumer sat in router's dep-spin, released {0,1}
// at 0 post-boot releases, and the w13 claim wedged at parity one-ahead
// (state-dump raw words + wait-site markers, workers 2/3 reproduced with
// zero free parameters). With NO loader page ops at consumer-owned tasks
// there is nothing sparse to alias: on an all-consumer-owned worker stream
// (the FFN GEMV chains) every page op is warp-0 program-ordered — claims
// pass exactly on time, alias- and deadlock-free by construction.
//
// Phase accounting: every task arrives every page exactly once per sequence
// (consumer-owned: suffix-all = 14; others: loader-prefix(unused) +
// consumer-suffix(used) or launcher blanket = 14; controller arrives all on
// BEGIN_TASK_GRAPH's behalf; the iteration drain publishes nothing).
// page_finished[p] is worker-global (never slot-indexed), parity is keyed
// to the ABSOLUTE sequence s (wait-arg s&1). Deadlock-freedom for
// consumer-owned tasks: the claim at sequence s is satisfied by s-1's
// release, program-ordered earlier on the same warp (or by a cross-warp
// owner whose progress is independent), a strictly decreasing chain
// grounded at the boot pre-arm.
//
// MIXED-CHAIN BOUNDARY (load-bearing invariant, ASSERTED at plan build —
// see build_v2_plan's per-page window check): a page-waiting warp's mod-2
// wait aliases two-early iff some page carries TWO pending releases when
// it checks — i.e. iff a page p goes UNOBSERVED by the waiting side for
// the two sequences before the wait: p in miss(a) AND miss(b) AND
// waits(c) for a consecutive queue window [a, b, c], where miss = pages
// the task's loader does not wait (consumer-total: ALL pages;
// consumer-owned/SkipUsed: the task's USED pages; wait-all: none) and
// waits = pages the task's loader waits (wait-all: ALL; SkipUsed: the
// UNUSED pages; consumer-total: none — its consumer claim only waits
// pages whose prior releases are warp-0 program-ordered in all-total
// chains). The window is exactly 3 wide because the ring bounds pending
// releases to the last 2 sequences. This predicate is why the modes are
// scoped as they are: qwen3's plan REALLY packs [silu, rmsnorm, linear]
// on one worker queue (the plan assertion caught it live during
// validation) — under consumer-total for rmsnorm/silu that window would
// intersect on every rmsnorm page, while under the shipped SkipUsed form
// miss(silu) is EMPTY (zero regions => its prefix waits all 14 pages,
// dense) and the window is safe. The FFN chains are all consumer-total
// with no page-waiting tasks at all => no window can exist there.
//
// bar.sync id 8 (ids 2..7 are in use across blackwell_v2) orders consumer
// threads 14..127's smem-region writes after lanes 0..13's claims; all 128
// consumer threads (warps 0-3) participate.
char const *kConsumerPageClaim =
    "{\n"
    "  if (threadIdx.x < MAX_SMEM_PAGES_PER_TASK &&\n"
    "      task_uses_page(task_desc, threadIdx.x)) {\n"
    "    runtime_wait_page_ready(runtime_smem, threadIdx.x,\n"
    "                            instruction_index);\n"
    "  }\n"
    "  asm volatile(\"bar.sync 8, 128;\" ::: \"memory\");\n"
    "}\n";

// Loader page-lifecycle prefix for consumer-owned (SHIPPED Design E) tasks:
// claim and release ONLY the pages this task does NOT use; used pages are
// claimed by the consumer (kConsumerPageClaim) and released by the consumer
// suffix. NOTE this makes the loader's observation of the task's USED pages
// SPARSE — safe only under the plan-time window assertion (a page must
// never go unobserved by page-waiting warps for two consecutive sequences
// before a page wait; see build_v2_plan). Kept for rmsnorm_v2/silu_mul_v2,
// whose mixed (qwen3 mlp) queues pack them adjacent to each other and to
// wait-all linears — where the consumer-TOTAL form below would widen the
// alias window (assertion-caught during validation) but this shipped form
// stays within pending<=1 per page.
char const *kLoaderPagePrefixSkipUsed =
    "{\n"
    "  int const _lane = threadIdx.x & 31;\n"
    "#ifdef MPK_ENABLE_PROFILING\n"
    "  bool const _pg_prof = _lane == 0 &&\n"
    "      runtime_config.profiler_buffer != nullptr &&\n"
    "      iter_num + V2_PROF_WINDOW_ITERS >= runtime_config.v2_max_iters;\n"
    "  if (_pg_prof) {\n"
    "    v2_prof_emit(runtime_config.profiler_buffer,\n"
    "                 V2_PROF_GROUP_LOADER_PHASE, V2_PROF_PAGE_WAIT,\n"
    "                 tb::EVENT_BEGIN);\n"
    "  }\n"
    "#endif\n"
    "  if (_lane < MAX_SMEM_PAGES_PER_TASK &&\n"
    "      !task_uses_page(task_desc, _lane)) {\n"
    "    runtime_wait_page_ready(runtime_smem, _lane, instruction_index);\n"
    "    runtime_finish_page(runtime_smem, _lane, 1);\n"
    "  }\n"
    "  __syncwarp();\n"
    "#ifdef MPK_ENABLE_PROFILING\n"
    "  if (_pg_prof) {\n"
    "    v2_prof_emit(runtime_config.profiler_buffer,\n"
    "                 V2_PROF_GROUP_LOADER_PHASE, V2_PROF_PAGE_WAIT,\n"
    "                 tb::EVENT_END);\n"
    "  }\n"
    "#endif\n"
    "}\n";

// Phase 3.5: page-lifecycle suffix at the end of every consumer body.
// Releases the pages this task uses (the ones the loader prefix did NOT
// release). Tasks that do their own release (e.g. linear's launcher
// blanket) opt out via auto_consumer_finish=false.
//
// Iterates physical pages — NOT regions — because the planner packs
// multiple sub-page regions into the same physical page (e.g. linear's
// six 4-KB A regions land on two pages, four+two). A region-based loop
// would arrive page X once per packed region, multi-flipping parity.
//
// Lane-parallel match for the loader prefix: lane K of consumer warp 0 arrives
// page K iff this task uses page K (the loader prefix already arrived
// pages this task doesn't use). Together they guarantee one arrive per
// page per task without the single-thread serialization that blocked
// consumer warp 0 in the original implementation.
char const *kConsumerPageSuffix =
    "{\n"
    "#ifdef MPK_ENABLE_PROFILING\n"
    "  unsigned long long _sfx_t0 = 0;\n"
    "  if (threadIdx.x == 0 &&\n"
    "      iter_num + V2_PROF_WINDOW_ITERS >= runtime_config.v2_max_iters) {\n"
    "    _sfx_t0 = v2_prof_now_ns();\n"
    "  }\n"
    "#endif\n"
    // RACE-2 STATUS (scope closure, 2026-07-17). The cross-warp
    // release-before-claim race is FIXED for every opted-in task by the
    // consumer-hosted claim (rmsnorm_v2 + silu_mul_v2 via consumer-owned/
    // SkipUsed — 7d271a01; the DSv3 FFN GEMV harness chain router_quant /
    // topk_sigmoid / w13_gemv / silu_quant / w2_gemv + the folded rqr /
    // w13_topk / w2_silu at nwarps=4 via consumer-TOTAL, which also closes
    // the SkipUsed mod-2 sparse-observation alias those chains hit — see
    // the kConsumerPageClaim comment for mechanism + fingerprint; mode
    // scoping is enforced by the build_v2_plan per-page window assertion).
    // The nwarps=7 multi-role forms of the FFN tasks are
    // STRUCTURALLY EXCLUDED from race 2 (not merely unfixed): their
    // consumer epilogue (mac_task_epilogue, mbar HELPERS_DONE or tag-flag
    // variant) holds warp-0 lanes 0..13 — the release lanes — until every
    // helper arrives, and the loader-role arrive is program-ordered after
    // its own wait-all page prefix, so the release can never precede the
    // claim; all six bodies are single-exit (audited 2026-07-17).
    // HISTORY of failed attempt 1 (kept as a warning): a "suffix waits a
    // loader-arrived per-slot pages_claimed mbarrier at ring_phase parity"
    // repair deadlocked profiled AND unprofiled L=6 chains at iter 0 (4/4)
    // and was reverted — arriver-set/waiter-set cardinality mismatch; any
    // mbarrier-handshake re-attempt must make the arriver set exactly equal
    // the waiter set with a ring-wraparound phase-accounting proof.
    "  if (threadIdx.x < MAX_SMEM_PAGES_PER_TASK &&\n"
    "      task_uses_page(task_desc, threadIdx.x)) {\n"
    "    runtime_finish_page(runtime_smem, threadIdx.x, 1);\n"
    "  }\n"
    "  __syncwarp();\n"
    "#ifdef MPK_ENABLE_PROFILING\n"
    "  if (_sfx_t0 != 0 && runtime_config.profiler_buffer != nullptr) {\n"
    "    unsigned long long *_sfx =\n"
    "        static_cast<unsigned long long "
    "*>(runtime_config.profiler_buffer);\n"
    "    _sfx[V2_PROF_SUFFIX_BASE + blockIdx.x] += v2_prof_now_ns() - "
    "_sfx_t0;\n"
    "    _sfx[V2_PROF_SUFFIX_BASE + V2_PROF_SM_SLOTS + blockIdx.x] += 1;\n"
    "  }\n"
    "#endif\n"
    "}\n";

// Suffix for CONSUMER-OWNED tasks (consumer-total lifecycle): release EVERY
// page, used and unused — the loader executes no page code at these tasks
// (see the kConsumerPageClaim comment). Lane-parallel on warp-0 lanes
// 0..13; exactly one arrive per page per sequence, all program-ordered
// after the consumer's own claim + body.
char const *kConsumerPageSuffixAll =
    "{\n"
    "#ifdef MPK_ENABLE_PROFILING\n"
    "  unsigned long long _sfx_t0 = 0;\n"
    "  if (threadIdx.x == 0 &&\n"
    "      iter_num + V2_PROF_WINDOW_ITERS >= runtime_config.v2_max_iters) {\n"
    "    _sfx_t0 = v2_prof_now_ns();\n"
    "  }\n"
    "#endif\n"
    "  if (threadIdx.x < MAX_SMEM_PAGES_PER_TASK) {\n"
    "    runtime_finish_page(runtime_smem, threadIdx.x, 1);\n"
    "  }\n"
    "  __syncwarp();\n"
    "#ifdef MPK_ENABLE_PROFILING\n"
    "  if (_sfx_t0 != 0 && runtime_config.profiler_buffer != nullptr) {\n"
    "    unsigned long long *_sfx =\n"
    "        static_cast<unsigned long long "
    "*>(runtime_config.profiler_buffer);\n"
    "    _sfx[V2_PROF_SUFFIX_BASE + blockIdx.x] += v2_prof_now_ns() - "
    "_sfx_t0;\n"
    "    _sfx[V2_PROF_SUFFIX_BASE + V2_PROF_SM_SLOTS + blockIdx.x] += 1;\n"
    "  }\n"
    "#endif\n"
    "}\n";

bool has_role_body(std::vector<rt::TaskRoleVariantCode> const &variants,
                   V2Role role) {
  for (rt::TaskRoleVariantCode const &variant : variants) {
    if (!role_body(variant, role).empty()) {
      return true;
    }
    // Phase 3.5: even an empty user body means we will emit a synthetic
    // loader (just the page-lifecycle prefix) for any task that opted in.
    if (role == V2Role::Loader && variant.auto_loader_page_lifecycle) {
      return true;
    }
  }
  return false;
}

void emit_role_cases(
    tr::CodeKeeper &code,
    std::map<rt::TaskType, std::string> const &task_type_to_name,
    rt::TaskRegister const &task_register,
    V2Role role) {
  for (auto const &task : task_register.all_v2_task_role_variants) {
    if (!has_role_body(task.second, role)) {
      continue;
    }
    auto name_it = task_type_to_name.find(task.first);
    assert(name_it != task_type_to_name.end());
    code.e("case $:", name_it->second);
    bool first_variant = true;
    for (size_t variant_id = 0; variant_id < task.second.size(); variant_id++) {
      rt::TaskRoleVariantCode const &variant = task.second[variant_id];
      std::string const &body = role_body(variant, role);
      // Phase 3.5: the loader case may need to emit a body even when the
      // user-provided body is empty, to carry the auto page-lifecycle
      // prefix. Other roles only emit if they have user content (or, for
      // consumer, if they have user content; the auto suffix piggybacks
      // on the user body, it does not synthesize one on its own).
      bool const auto_loader_prefix =
          (role == V2Role::Loader) && variant.auto_loader_page_lifecycle;
      bool const auto_consumer_suffix = (role == V2Role::Consumer) &&
                                        variant.auto_consumer_finish &&
                                        !body.empty();
      // Per-variant page-lifecycle mode (both gated on the structural check:
      // auto_consumer_finish with every helper-role body empty — the
      // consumer is provably the sole toucher of the used pages):
      //   - consumer-TOTAL (consumer_total_page_lifecycle: the DSv3 FFN
      //     GEMV chain + folded chain): consumer claims used pages, suffix
      //     releases ALL pages, loader emits NO page code.
      //   - consumer-owned (consumer_owned_page_claim: rmsnorm_v2 +
      //     silu_mul_v2): consumer claims used pages, SkipUsed loader
      //     prefix, suffix releases used pages (the shipped 7d271a01 form).
      // Everything else — the multi-role nwarps=7 FFN GEMV forms
      // (structurally race-2-excluded via their helper-done epilogue, see
      // kConsumerPageSuffix comment), the pipeline linears — keeps the
      // shipped wait-all prefix + suffix, byte-identical emission.
      bool const helpers_empty =
          variant.auto_consumer_finish && variant.loader.empty() &&
          variant.launcher.empty() && variant.storer.empty();
      bool const consumer_total =
          variant.consumer_total_page_lifecycle && helpers_empty;
      bool const consumer_owned_pages =
          !consumer_total && variant.consumer_owned_page_claim && helpers_empty;
      // Consumer-total tasks emit NO loader page code.
      bool const emit_loader_prefix = auto_loader_prefix && !consumer_total;
      if (body.empty() && !emit_loader_prefix) {
        continue;
      }
      std::string const cond = first_variant ? "if" : "else if";
      code.e("  $ (task_desc->variant_id == $) {", cond, variant_id);
      if (emit_loader_prefix) {
        code.e("$",
               consumer_owned_pages ? kLoaderPagePrefixSkipUsed
                                    : kLoaderPagePrefix);
      }
      if (auto_consumer_suffix && (consumer_total || consumer_owned_pages)) {
        code.e("$", kConsumerPageClaim);
      }
      if (!body.empty()) {
        code.e("$", body);
      }
      if (auto_consumer_suffix) {
        code.e("$",
               consumer_total ? kConsumerPageSuffixAll : kConsumerPageSuffix);
      }
      code.e("}");
      first_variant = false;
    }
    code.e("  break;");
  }
}

// Host-side page-mode table for build_v2_plan's mixed-chain window
// assertion (see the kConsumerPageClaim boundary note): returns
//   0 = no auto page lifecycle known for (task_type, variant)
//       (unregistered types, e.g. BEGIN/TERMINATE),
//   1 = wait-all loader prefix (dense observation),
//   2 = consumer-owned + SkipUsed loader (loader misses the USED pages),
//   3 = consumer-total (loader misses EVERY page; waits none).
// Declared in runtime_v2.cuh next to the _execute_*_v2 prototypes.
void emit_page_mode_fn(
    tr::CodeKeeper &code,
    std::map<rt::TaskType, std::string> const &task_type_to_name,
    rt::TaskRegister const &task_register) {
  code.e("int _v2_variant_page_mode(int task_type, int variant_id) {");
  code.e("(void)variant_id;");
  code.e("switch (task_type) {");
  for (auto const &task : task_register.all_v2_task_role_variants) {
    auto name_it = task_type_to_name.find(task.first);
    assert(name_it != task_type_to_name.end());
    code.e("case $:", name_it->second);
    for (size_t variant_id = 0; variant_id < task.second.size(); variant_id++) {
      rt::TaskRoleVariantCode const &variant = task.second[variant_id];
      bool const helpers_empty =
          variant.auto_consumer_finish && variant.loader.empty() &&
          variant.launcher.empty() && variant.storer.empty();
      int mode = 0;
      if (variant.consumer_total_page_lifecycle && helpers_empty) {
        mode = 3;
      } else if (variant.consumer_owned_page_claim && helpers_empty) {
        mode = 2;
      } else if (variant.auto_loader_page_lifecycle) {
        mode = 1;
      }
      code.e("  if (variant_id == $) { return $; }", variant_id, mode);
    }
    code.e("  return 0;");
  }
  code.e("default:");
  code.e("  return 0;");
  code.e("}");
  code.e("}");
}

void emit_role_dispatcher(
    tr::CodeKeeper &code,
    std::map<rt::TaskType, std::string> const &task_type_to_name,
    rt::TaskRegister const &task_register,
    char const *function_name,
    V2Role role) {
  code.e("__device__ __forceinline__ void");
  code.e("$(TaskDesc const *task_desc,", function_name);
  code.e("  RuntimeConfig const &runtime_config,");
  code.e("  RuntimeSMEM *runtime_smem,");
  code.e("  int instruction_index,");
  code.e("  int iter_num) {");
  code.e("(void)runtime_config;");
  code.e("(void)runtime_smem;");
  code.e("(void)instruction_index;");
  code.e("(void)iter_num;");
  code.e("switch (task_desc->task_type) {");
  emit_role_cases(code, task_type_to_name, task_register, role);
  code.e("default:");
  code.e("  break;");
  code.e("}");
  code.e("}");
}

} // namespace

void generate_v2_role_dispatch_code(
    tr::CodeKeeper &code,
    std::map<rt::TaskType, std::string> const &task_type_to_name,
    rt::TaskRegister const &task_register) {
  // The dispatchers reference v2-only types (RuntimeSMEM, etc.) and helpers.
  // Skip them entirely in v1 builds — v1 has its own dispatch path.
  code.e("#ifdef USE_RUNTIME_V2");
  code.e("namespace mirage {");
  code.e("namespace runtime_v2 {");
  code.e("using namespace mirage::runtime;");
  emit_role_dispatcher(code,
                       task_type_to_name,
                       task_register,
                       "_execute_init_semaphores_v2",
                       V2Role::InitSemaphores);
  emit_role_dispatcher(code,
                       task_type_to_name,
                       task_register,
                       "_execute_loader_task_v2",
                       V2Role::Loader);
  emit_role_dispatcher(code,
                       task_type_to_name,
                       task_register,
                       "_execute_launcher_task_v2",
                       V2Role::Launcher);
  emit_role_dispatcher(code,
                       task_type_to_name,
                       task_register,
                       "_execute_consumer_task_v2",
                       V2Role::Consumer);
  emit_role_dispatcher(code,
                       task_type_to_name,
                       task_register,
                       "_execute_storer_task_v2",
                       V2Role::Storer);
  emit_page_mode_fn(code, task_type_to_name, task_register);
  code.e("} // namespace runtime_v2");
  code.e("} // namespace mirage");
  code.e("#endif // USE_RUNTIME_V2");
}

} // namespace kernel
} // namespace mirage
