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

// Design E (race-2 fix, 2026-07-16): CONSUMER-OWNED page lifecycle.
// For tasks where the consumer is the SOLE toucher of the task's used pages
// (auto_consumer_finish AND all helper-role bodies empty — the structural
// predicate `consumer_owned_pages` below), the USED-page CLAIM is hosted on
// the consumer itself (this block, emitted BEFORE the body), and the loader
// prefix skips used pages entirely (kLoaderPagePrefixSkipUsed). This closes
// the cross-warp release-before-claim race: the shared W4 loader lags
// structurally (it may still be draining the previous linear's ~96
// mma-gated TMA issues), so a fast consumer could complete the whole task
// INCLUDING the release suffix before the lagging loader's prefix waited
// the used pages — their parity advanced one use ahead and the loader
// wedged forever (state-dump fingerprint: loader in-flight at an rmsnorm
// task with exactly the task's used pages at claim-count+1; all-worker
// convoy behind its unfired event). Latent PRODUCTION race, previously
// timing-masked unprofiled.
//
// Phase-accounting proof (ring-wraparound included): this change relocates
// a NON-MUTATING wait and re-partitions which role performs the SAME single
// release per page per sequence — the per-page arrival timeline is
// byte-identical to the shipped protocol. page_finished[p] is worker-global
// (never slot-indexed, so INSTRUCTION_RING_SIZE=3 and the 14-seqs/iter
// non-divisibility never enter), parity is keyed to the ABSOLUTE sequence s
// (wait-arg s&1), and advances exactly once per sequence by exactly one
// owner: loader-prefix (pages the task does not use), consumer-suffix
// (used pages, consumer-owned tasks), launcher blanket (the four pipeline
// linears, auto_consumer_finish=false), or the controller on
// BEGIN_TASK_GRAPH's behalf; the iteration drain publishes nothing and
// TERMINATE precedes no further waits. Deadlock-freedom: the claim at
// sequence s is satisfied by s-1's release, whose owner warp necessarily
// reaches s-1 before s (per-warp monotonic sequence) — a strictly
// decreasing chain grounded at the boot pre-arm — and the loader<->consumer
// page coupling is severed in BOTH directions (the loader no longer waits
// used pages; the consumer never waits on the loader).
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

// Loader page-lifecycle prefix for CONSUMER-OWNED tasks (see above): claim
// and release ONLY the pages this task does NOT use; used pages are claimed
// by the consumer (kConsumerPageClaim) and released by the consumer suffix.
// The loader therefore can never wedge on a used page regardless of lag.
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
    // Race 2 (cross-warp release-before-claim) is FIXED for the OPTED-IN
    // consumer-owned tasks (consumer_owned_page_claim: rmsnorm_v2 +
    // silu_mul_v2 — the tasks whose suffix race produced the observed
    // production wedge) by kConsumerPageClaim + kLoaderPagePrefixSkipUsed
    // above (Design E, 2026-07-16): their release is now program-ordered
    // after the consumer's OWN claim of the same pages, so it can no longer
    // overtake a lagging loader. RESIDUAL SCOPE (documented-open, all
    // harness-only; production DSv3 decode uses the pipe/mega forms with
    // launcher-blanket release): (a) the nwarps=7 multi-role FFN GEMV trio
    // (non-empty helper bodies) keeps wait-all prefix + this suffix; (b) the
    // nwarps=4 consumer-only FFN GEMV chain deliberately does NOT opt in —
    // an automatic transform wedged its claim wait there (worker-random,
    // marker-robust; parity one-off on the claimed pages, mechanism
    // unresolved within budget), so it keeps the shipped behavior and its
    // race-2 window. HISTORY of failed attempt 1: a "suffix waits a
    // loader-arrived per-slot pages_claimed mbarrier at ring_phase parity"
    // repair deadlocked profiled AND unprofiled L=6 chains at iter 0 (4/4)
    // and was reverted — arriver-set/waiter-set cardinality mismatch
    // hypothesis; any mbarrier-handshake re-attempt must make the arriver
    // set exactly equal the waiter set with a ring-wraparound
    // phase-accounting proof. Design E needs neither (no new sync object;
    // the per-page arrival timeline is unchanged).
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
      // Design E predicate (per-variant): the used-page claim moves onto the
      // consumer ONLY for registrations that explicitly opted in
      // (consumer_owned_page_claim — currently rmsnorm_v2 + silu_mul_v2, the
      // tasks whose suffix race produced the observed wedge) AND that pass
      // the structural check (auto_consumer_finish with every helper-role
      // body empty — the consumer is provably the sole toucher of the used
      // pages). Everything else — the multi-role nwarps=7 FFN GEMV trio, the
      // consumer-only nwarps=4 GEMV chain (which wedged its claim wait under
      // an automatic transform in harness validation; mechanism unresolved),
      // the pipeline linears — keeps the shipped wait-all prefix + suffix,
      // byte-identical to the pre-fix emission.
      bool const consumer_owned_pages =
          variant.consumer_owned_page_claim && variant.auto_consumer_finish &&
          variant.loader.empty() && variant.launcher.empty() &&
          variant.storer.empty();
      if (body.empty() && !auto_loader_prefix) {
        continue;
      }
      std::string const cond = first_variant ? "if" : "else if";
      code.e("  $ (task_desc->variant_id == $) {", cond, variant_id);
      if (auto_loader_prefix) {
        code.e("$",
               consumer_owned_pages ? kLoaderPagePrefixSkipUsed
                                    : kLoaderPagePrefix);
      }
      if (auto_consumer_suffix && consumer_owned_pages) {
        code.e("$", kConsumerPageClaim);
      }
      if (!body.empty()) {
        code.e("$", body);
      }
      if (auto_consumer_suffix) {
        code.e("$", kConsumerPageSuffix);
      }
      code.e("}");
      first_variant = false;
    }
    code.e("  break;");
  }
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
  code.e("} // namespace runtime_v2");
  code.e("} // namespace mirage");
  code.e("#endif // USE_RUNTIME_V2");
}

} // namespace kernel
} // namespace mirage
