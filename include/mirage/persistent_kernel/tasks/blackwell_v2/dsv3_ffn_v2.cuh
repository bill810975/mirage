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
// DSv3 fused-FFN block in Runtime-V2 format (Step 3a of the V2 migration).
//
// The v1 whole-grid task (tasks/blackwell/ffn_full_megakernel_sm100.cuh,
// 136-CTA lockstep, 3 internal grid barriers) is re-expressed as a CHAIN of
// v2 per-SM tasks; the v1 grid barriers become v2 cross-task event
// dependencies, the v1 grid-strided phases become N-task ops:
//
//   rmsnorm_v2 (existing)          Phase A
//   dsv3_ffn_router_quant_v2 (NR)  Phase 0 (UE8M0 quant) + Phase B (router
//                                  split-K GEMV -> inter partials)
//   dsv3_ffn_topk_sigmoid_v2 (1)   Phase C (logits reduce + sigmoid + group
//                                  top-8 + EP-local filter -> meta)
//   dsv3_ffn_w13_gemv_v2 (NT13)    Phase 1 (routed W13 + shared gate_up)
//   dsv3_ffn_silu_quant_v2 (1)     Phase 2 (silu_fast + UE8M0 requant)
//   dsv3_ffn_w2_gemv_v2 (NT2)      Phase 3 + final (W2 + shared down,
//                                  OUTPUT-STATIONARY: each task owns disjoint
//                                  out rows, fp32 accum in registers, direct
//                                  bf16 store — no atomics, no zero-init op;
//                                  deliberate structural change vs v1's
//                                  atomicAdd, per design review)
//
// ALL GEMV / quant / topk math is the v1 code itself: the helpers
// (dgemv_cpa16_h2, dgemv_cpa, router_partial_cpa, quant_group_warp,
// quant_scale, to_f8, silu_fast) are called from
// kernel::ffn_full_megakernel_sm100 (already included in every generated
// test.cu), so per-row / per-group / per-partial values are bit-identical to
// v1 given identical input bytes — task/warp scheduling cannot change values
// because every helper's accumulation is warp-local with a fixed lane order.
//
// Warp model: STAGE 1 (v2-idiomatic) runs all MAC bodies on the 4 consumer
// warps (threads 0-127). STAGE 2 (attribution ablation, param-gated at
// registration) additionally emits the same body into the loader/launcher/
// storer role cases: warp-slot ws = threadIdx.x>>5 in 0..6 becomes the
// stride lane, giving 7 MAC warps. Cross-role safety (per design review):
//   - op-private mbar[0] HELPERS_DONE (arrive count 3): each helper arrives
//     after MAC + cp.async drain, on EVERY path; consumer warp 0 waits it
//     (+__syncwarp) after the 128-thread barrier and BEFORE returning, so
//     the codegen auto consumer page-release suffix (lanes 0-13) cannot
//     release SMEM pages while helper warps still use them.
//   - op-private mbar[1] ACT_READY (arrive count 1): consumers stage the
//     activation into SMEM then arrive; helpers wait it before MAC.
//   - parity is always 0: both mbars are re-initialized by the controller's
//     init_semaphores body on every instruction publish, and the slot cannot
//     be republished until all roles arrived INSTRUCTION_FINISHED.
// ============================================================================

#include "mirage/persistent_kernel/runtime_header.h"
#include "mirage/persistent_kernel/tasks/blackwell/ffn_full_megakernel_sm100.cuh"
#include "mirage/persistent_kernel/tasks/blackwell_v2/dsv3_ffn_v2_spec.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

namespace kernel {
namespace dsv3_ffn_v2 {

namespace v1k = ::kernel::ffn_full_megakernel_sm100;

// The spec header re-declares the shapes for the host-side planner; pin them
// to the v1 kernel's constants so drift is a compile error.
static_assert(HIDDEN == v1k::HIDDEN && W13_N == v1k::W13_N &&
                  W2_K == v1k::W2_K && W2_N == v1k::W2_N && GRP == v1k::GRP &&
                  KG1 == v1k::KG1 && KG2 == v1k::KG2 &&
                  MAX_ACTIVE == v1k::MAX_ACTIVE &&
                  ROUTER_N == v1k::ROUTER_N && RKSPLIT == v1k::RKSPLIT &&
                  SH_GU_N == v1k::SH_GU_N && SH_DN_K == v1k::SH_DN_K &&
                  KG_SHDN == v1k::KG_SHDN,
              "dsv3_ffn_v2_spec.h shapes drifted from the v1 kernel");

// Consumer-warp named barrier (threads 0-127; consumer role only). Barrier 0
// is the implicit block-wide sync, 1 is used by linear_v2, 2 by rmsnorm_v2;
// use 3 here.
__device__ __forceinline__ void consumer_sync() {
  asm volatile("bar.sync 3, 128;");
}

// mbarrier ops on a raw shared-memory address (the op-private dynamic
// semaphore base handed over by codegen via op_sem_base_addr). Same PTX as
// mirage::runtime_v2::mbar_arrive/mbar_wait, kept here so this header does
// not depend on runtime_v2.cuh types. %= uniquifies labels across inlining.
__device__ __forceinline__ void ffnv2_mbar_arrive_addr(int addr) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(addr)
               : "memory");
}
__device__ __forceinline__ void ffnv2_mbar_wait_addr(int addr, int phase) {
  asm volatile("{\n\t.reg .pred P%=;\n\t"
               "W%=: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 "
               "P%=, [%0], %1, 0x989680;\n\t"
               "@P%= bra D%=;\n\t"
               "bra W%=;\n\t"
               "D%=:\n\t}" ::"r"(addr),
               "r"(phase));
}

// Byte offsets of the two op-private mbars from op_sem_base_addr.
static constexpr int SEMOFF_HELPERS_DONE = 0;
static constexpr int SEMOFF_ACT_READY = 8;

// ---------------------------------------------------------------------------
// nwarps=7 cross-role handshake for the FOLDED tasks: monotonic TAG-FLAGS in
// task SMEM (u64[4]: [0] ACT_READY, [1..3] HELPER_DONE per helper warp-slot).
// Replaces the op-private mbarriers + controller init_semaphores (whose
// re-init/parity machinery is implicated in a deterministic multi-iteration
// wedge): no initialization, no parity, no controller involvement. The tag is
// a salted bijection of the per-SM monotonic instruction sequence — unique
// for the kernel lifetime — so stale bytes from any previous task can never
// satisfy a wait (64-bit exact-match). sync_tag == 0 => stage-1 (4-warp)
// variant: all handshakes compile out.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void ffnv2_flag_store_release(uint64_t *f,
                                                         uint64_t v) {
  asm volatile("st.release.cta.shared::cta.u64 [%0], %1;" ::"r"(
                   static_cast<int>(__cvta_generic_to_shared(f))),
               "l"(v)
               : "memory");
}
__device__ __forceinline__ uint64_t ffnv2_flag_load_acquire(uint64_t *f) {
  uint64_t v;
  asm volatile("ld.acquire.cta.shared::cta.u64 %0, [%1];"
               : "=l"(v)
               : "r"(static_cast<int>(__cvta_generic_to_shared(f)))
               : "memory");
  return v;
}
// Single-thread poll (no warp sync — safe under divergence, e.g. the
// consumer-thread0 epilogue wait). Backoff keeps the spin off the LSU pipes.
__device__ __forceinline__ void ffnv2_flag_poll(uint64_t *f, uint64_t tag) {
  while (ffnv2_flag_load_acquire(f) != tag) {
    __nanosleep(64);
  }
}
// Whole-warp wait: ALL 32 lanes must call this together. Lane 0 polls (an
// all-lane tight acquire spin from 3 helper warps saturates the SM's LSU
// pipes and starves the consumer warps' cp.async/compute); __syncwarp
// releases the other lanes and the lane-0 acquire orders the protected SMEM
// reads for the whole warp.
__device__ __forceinline__ void ffnv2_flag_wait(uint64_t *f, uint64_t tag) {
  if ((threadIdx.x & 31) == 0) {
    ffnv2_flag_poll(f, tag);
  }
  __syncwarp();
}

// LEGACY multi-role epilogue for the UNFOLDED chain's mbar protocol
// (op_sem_addr < 0 => stage-1: plain consumer barrier only). The folded
// tasks use the tag-flag epilogue below instead.
__device__ __forceinline__ void
    mac_task_epilogue(bool is_consumer, int op_sem_addr) {
  v1k::cpasync_wait<0>();
  __syncwarp();
  if (is_consumer) {
    consumer_sync();
    if (op_sem_addr >= 0 && threadIdx.x < 32) {
      if (threadIdx.x == 0) {
        ffnv2_mbar_wait_addr(op_sem_addr + SEMOFF_HELPERS_DONE, 0);
      }
      __syncwarp();
    }
  } else {
    if ((threadIdx.x & 31) == 0) {
      ffnv2_mbar_arrive_addr(op_sem_addr + SEMOFF_HELPERS_DONE);
    }
  }
}

// Multi-role epilogue. sync_tag == 0 => stage-1 (consumer-only variant): all
// handshakes compile to the plain consumer barrier.
__device__ __forceinline__ void
    mac_task_epilogue(bool is_consumer, uint64_t *s_flags, uint64_t sync_tag) {
  // Every MAC warp drains its own cp.async ring before any page can be
  // released (empty groups complete immediately; this is cheap).
  v1k::cpasync_wait<0>();
  __syncwarp();
  if (is_consumer) {
    consumer_sync(); // warps 1-3 done before warp 0 releases pages
    if (sync_tag != 0 && threadIdx.x < 32) {
      if (threadIdx.x == 0) {
        ffnv2_flag_poll(&s_flags[1], sync_tag);
        ffnv2_flag_poll(&s_flags[2], sync_tag);
        ffnv2_flag_poll(&s_flags[3], sync_tag);
      }
      __syncwarp(); // hold lanes 0-13 (the page-release lanes) until helpers
                    // are done
    }
  } else {
    int const ws = threadIdx.x >> 5; // 4/5/6
    if ((threadIdx.x & 31) == 0) {
      ffnv2_flag_store_release(&s_flags[1 + (ws - 4)], sync_tag);
    }
  }
}

// ============================================================================
// T1 — router_quant. v1 Phase 0 (FP8/UE8M0 quant of the bf16 normed) + v1
// Phase B (router gate GEMV, split-K=4, fp32 partials).
//   inputs : [0] rmsnorm_out bf16[1,HIDDEN]
//            [1] router_gate_w bf16[ROUTER_N,HIDDEN]
//            [2] a_fp8 u8[HIDDEN]      (hidden write — no graph edge)
//            [3] a_scale f32[KG1]      (hidden write)
//   outputs: [0] inter f32[ROUTER_N*RKSPLIT]  (the T1->T2 chain edge)
//   work map: quant group g and router pair t are strided by the GLOBAL warp
//   id (task_offset*nwarps + ws), matching v1's gwarp map shape.
// ============================================================================
__device__ __noinline__ void
    router_quant_task_impl(mirage::runtime::TaskDesc const *task_desc,
                           int task_offset,
                           int num_tasks,
                           int nwarps,
                           int op_sem_addr) {
  __nv_bfloat16 const *x =
      static_cast<__nv_bfloat16 const *>(task_desc->input_ptrs[0]);
  __nv_bfloat16 const *wr =
      static_cast<__nv_bfloat16 const *>(task_desc->input_ptrs[1]);
  uint8_t *a_fp8 = static_cast<uint8_t *>(task_desc->input_ptrs[2]);
  float *a_scale = static_cast<float *>(task_desc->input_ptrs[3]);
  float *inter = static_cast<float *>(task_desc->output_ptrs[0]);

  extern __shared__ char smem[];
  __nv_bfloat16 *s_norm = reinterpret_cast<__nv_bfloat16 *>(
      smem + task_desc->smem_region_offset(RQ_REGION_NORM));
  uint4 *s_ring = reinterpret_cast<uint4 *>(
      smem + task_desc->smem_region_offset(RQ_REGION_RING));

  int const lane = threadIdx.x & 31;
  int const ws = threadIdx.x >> 5; // 0-3 consumer / 4 loader / 5 launcher /
                                   // 6 storer (role dispatch guarantees this)
  bool const is_consumer = threadIdx.x < 128;

  // Stage the activation into SMEM (consumers), or wait for it (helpers).
  if (is_consumer) {
    uint32_t const sb =
        static_cast<uint32_t>(__cvta_generic_to_shared(s_norm));
    uint4 const *g4 = reinterpret_cast<uint4 const *>(x);
    constexpr int NU4 = HIDDEN * 2 / 16; // 896
    for (int u = threadIdx.x; u < NU4; u += 128) {
      v1k::cpasync16(sb + (uint32_t)u * 16, &g4[u]);
    }
    v1k::cpasync_commit();
    v1k::cpasync_wait<0>();
    consumer_sync();
    if (op_sem_addr >= 0 && threadIdx.x == 0) {
      ffnv2_mbar_arrive_addr(op_sem_addr + SEMOFF_ACT_READY);
    }
  } else {
    ffnv2_mbar_wait_addr(op_sem_addr + SEMOFF_ACT_READY, 0);
  }

  // v1 Phase 0: per-128-group UE8M0 quant (one warp per group), directly to
  // the (hidden) global a_fp8/a_scale. Group-local math == v1 regardless of
  // which warp/task runs the group.
  for (int g = task_offset * nwarps + ws; g < KG1; g += num_tasks * nwarps) {
    v1k::quant_group_warp<__nv_bfloat16>(s_norm, a_fp8, a_scale, g, lane);
  }

  // v1 Phase B: (expert, split) pairs strided by global warp id. The partial
  // math (router_partial_cpa) is verbatim v1 -> inter is bit-exact vs v1
  // given identical s_norm bytes.
  uint4 *my_ring = s_ring + (size_t)ws * (RQ_RING_BYTES_PER_WARP / 16);
  int const total_pairs = ROUTER_N * RKSPLIT;
  for (int t = task_offset * nwarps + ws; t < total_pairs;
       t += num_tasks * nwarps) {
    int const e = t / RKSPLIT, sp = t % RKSPLIT;
    float const acc = v1k::router_partial_cpa<RKSPLIT, 4>(
        s_norm, wr + (size_t)e * v1k::ROUTER_K, sp, lane, my_ring);
    if (lane == 0) {
      inter[e * RKSPLIT + sp] = acc;
    }
  }

  mac_task_epilogue(is_consumer, op_sem_addr);
}

// ============================================================================
// T2 — topk_sigmoid core. v1 Phase C ported to the 4 consumer warps (128
// threads): warp w handles groups {w, w+4} where v1 used one warp per group —
// the per-group shfl trees are warp-local, so per-group values are bit-exact.
// Shared by the standalone 1-task op AND the folded w13_topk (each task
// recomputes the routing redundantly, v1's per-CTA trick). Publishes the
// META_INTS meta into SMEM (wk + TK_OFF_META, all callers) and optionally to
// GMEM (meta_gmem/logits_out non-null: the standalone op always, the folded
// op only on task_offset 0). Ends with a consumer_sync (s_meta visible).
// ============================================================================
__device__ __forceinline__ void
    topk_compute(char *wk,                    // TK work region base
                 float const *inter,          // f32[ROUTER_N*RKSPLIT]
                 float const *bias,           // f32[ROUTER_N]
                 __nv_bfloat16 *logits_out,   // bf16[ROUTER_N] or nullptr
                 int *meta_gmem,              // i32[META_INTS] or nullptr
                 int local_expert_start,
                 int num_local_experts,
                 float routed_scaling_factor) {
  float *s_sig = reinterpret_cast<float *>(wk + TK_OFF_SIG);
  float *s_biased = reinterpret_cast<float *>(wk + TK_OFF_BIASED);
  float *s_gscore = reinterpret_cast<float *>(wk + TK_OFF_GSCORE);
  float *s_gactw = reinterpret_cast<float *>(wk + TK_OFF_GACTW);
  float *s_top8v = reinterpret_cast<float *>(wk + TK_OFF_TOP8V);
  int *s_gsel = reinterpret_cast<int *>(wk + TK_OFF_GSEL);
  int *s_gacte = reinterpret_cast<int *>(wk + TK_OFF_GACTE);
  int *s_top8i = reinterpret_cast<int *>(wk + TK_OFF_TOP8I);

  int const lane = threadIdx.x & 31;
  int const wid = threadIdx.x >> 5; // 0..3

  using namespace v1k; // NUM_EXPERTS/NUM_GROUPS/EXPERTS_PER_GROUP/
                       // TOPK_GROUP/TOPK_EXPERTS constants

  // (i) reduce the RKSPLIT partials -> bf16 logits (production boundary),
  // sigmoid + bias. Elementwise per e — stride change vs v1 is value-neutral.
  // VECTORIZED: one float4 load = one expert's 4 partials (16B, coalesced);
  // fixed x+y+z+w order == the v1 sp-ascending sum order (bit-identical).
  static_assert(RKSPLIT == 4, "float4 pass assumes RKSPLIT == 4");
  for (int e = threadIdx.x; e < ROUTER_N; e += 128) {
    float4 const p = reinterpret_cast<float4 const *>(inter)[e];
    float const tot = ((p.x + p.y) + p.z) + p.w;
    __nv_bfloat16 const lgb = __float2bfloat16(tot);
    if (logits_out != nullptr) {
      logits_out[e] = lgb;
    }
    float const lg = __bfloat162float(lgb);
    float const s = 1.0f / (1.0f + expf(-lg));
    s_sig[e] = s;
    s_biased[e] = s + bias[e];
  }
  consumer_sync();

  // (ii) group score = top-2 biased per group of 32 (warp-local, verbatim).
  for (int g = wid; g < NUM_GROUPS; g += 4) {
    float const v = s_biased[g * EXPERTS_PER_GROUP + lane];
    float t1 = v;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      t1 = fmaxf(t1, __shfl_xor_sync(0xffffffffu, t1, o));
    }
    unsigned const ismax = __ballot_sync(0xffffffffu, v == t1);
    int const firstmax = __ffs(ismax) - 1;
    float v2;
    if (lane == firstmax) {
      v2 = -1e30f;
    } else {
      v2 = v;
    }
    float t2 = v2;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      t2 = fmaxf(t2, __shfl_xor_sync(0xffffffffu, t2, o));
    }
    if (lane == 0) {
      s_gscore[g] = t1 + t2;
    }
  }
  consumer_sync();

  // (iii) top-4 group selection (thread 0, verbatim).
  if (threadIdx.x == 0) {
    float gsc[NUM_GROUPS];
#pragma unroll
    for (int g = 0; g < NUM_GROUPS; g++) {
      gsc[g] = s_gscore[g];
      s_gsel[g] = 0;
    }
#pragma unroll
    for (int ki = 0; ki < TOPK_GROUP; ki++) {
      int bg = 0;
      float bs = -1e30f;
#pragma unroll
      for (int g = 0; g < NUM_GROUPS; g++) {
        if (!s_gsel[g] && gsc[g] > bs) {
          bs = gsc[g];
          bg = g;
        }
      }
      s_gsel[bg] = 1;
    }
  }
  consumer_sync();
  // mask non-selected groups (verbatim; stride value-neutral).
  for (int n = threadIdx.x; n < NUM_EXPERTS; n += 128) {
    if (!s_gsel[n / EXPERTS_PER_GROUP]) {
      s_biased[n] = -10000.f;
    }
  }
  consumer_sync();

  // per-group local top-8 (warp-local rounds, verbatim; warp w does groups
  // {w, w+4}).
  for (int g = wid; g < NUM_GROUPS; g += 4) {
    float lvv = s_biased[g * EXPERTS_PER_GROUP + lane];
    int lvi = g * EXPERTS_PER_GROUP + lane;
#pragma unroll
    for (int k = 0; k < TOPK_EXPERTS; k++) {
      float bv = lvv;
      int bi = lvi;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        float const ov = __shfl_xor_sync(0xffffffffu, bv, o);
        int const oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) {
          bv = ov;
          bi = oi;
        }
      }
      if (lane == 0) {
        s_top8v[g * TOPK_EXPERTS + k] = bv;
        s_top8i[g * TOPK_EXPERTS + k] = bi;
      }
      if (lvi == bi) {
        lvv = -10000.f;
      }
    }
  }
  consumer_sync();

  // warp 0 merges the 64 candidates -> global top-8 + normalized weights
  // (verbatim, incl. wsum over the GLOBAL 8 and the rsf multiply).
  if (wid == 0) {
    constexpr int CPL = (NUM_GROUPS * TOPK_EXPERTS) / 32; // 2
    float cv[CPL];
    int ci[CPL];
#pragma unroll
    for (int j = 0; j < CPL; j++) {
      cv[j] = s_top8v[lane * CPL + j];
      ci[j] = s_top8i[lane * CPL + j];
    }
    float wsum = 0.f;
#pragma unroll
    for (int k = 0; k < TOPK_EXPERTS; k++) {
      float bv = -1e30f;
      int bi = NUM_EXPERTS;
#pragma unroll
      for (int j = 0; j < CPL; j++) {
        if (cv[j] > bv || (cv[j] == bv && ci[j] < bi)) {
          bv = cv[j];
          bi = ci[j];
        }
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        float const ov = __shfl_xor_sync(0xffffffffu, bv, o);
        int const oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) {
          bv = ov;
          bi = oi;
        }
      }
      if (lane == 0) {
        s_gacte[k] = bi;
        s_gactw[k] = s_sig[bi];
      }
      wsum += s_sig[bi];
#pragma unroll
      for (int j = 0; j < CPL; j++) {
        if (ci[j] == bi) {
          cv[j] = -10000.f;
        }
      }
    }
    if (lane == 0) {
      float const inv = 1.0f / (wsum + 1e-20f);
#pragma unroll
      for (int k = 0; k < TOPK_EXPERTS; k++) {
        s_gactw[k] = s_gactw[k] * inv * routed_scaling_factor;
      }
    }
  }
  consumer_sync();

  // EP-local filter (verbatim) + publish meta (SMEM always, GMEM optional).
  if (threadIdx.x == 0) {
    int *s_meta = reinterpret_cast<int *>(wk + TK_OFF_META);
    int const local_expert_end = local_expert_start + num_local_experts;
    int me[MAX_ACTIVE];
    float mw[MAX_ACTIVE];
#pragma unroll
    for (int s = 0; s < MAX_ACTIVE; s++) {
      me[s] = 0;
      mw[s] = 0.f;
    }
    int active_count = 0;
#pragma unroll
    for (int k = 0; k < TOPK_EXPERTS; k++) {
      int const e = s_gacte[k];
      if (e >= local_expert_start && e < local_expert_end &&
          active_count < MAX_ACTIVE) {
        me[active_count] = e - local_expert_start;
        mw[active_count] = s_gactw[k];
        active_count++;
      }
    }
    int mvals[META_INTS];
    mvals[META_OFF_COUNT] = active_count;
    mvals[META_OFF_MAGIC] = META_MAGIC;
#pragma unroll
    for (int s = 0; s < MAX_ACTIVE; s++) {
      mvals[META_OFF_EXPERTS + s] = me[s];
      mvals[META_OFF_WEIGHTS + s] = __float_as_int(mw[s]);
    }
#pragma unroll
    for (int i = META_OFF_WEIGHTS + MAX_ACTIVE; i < META_INTS; i++) {
      mvals[i] = 0;
    }
#pragma unroll
    for (int i = 0; i < META_INTS; i++) {
      s_meta[i] = mvals[i];
      if (meta_gmem != nullptr) {
        meta_gmem[i] = mvals[i];
      }
    }
  }
  consumer_sync();
}

// The standalone 1-task topk op (the un-folded chain).
__device__ __noinline__ void
    topk_sigmoid_task_impl(mirage::runtime::TaskDesc const *task_desc,
                           int local_expert_start,
                           int num_local_experts,
                           float routed_scaling_factor) {
  float const *inter = static_cast<float const *>(task_desc->input_ptrs[0]);
  float const *bias = static_cast<float const *>(task_desc->input_ptrs[1]);
  __nv_bfloat16 *logits_out =
      static_cast<__nv_bfloat16 *>(task_desc->input_ptrs[2]);
  int *meta = static_cast<int *>(task_desc->output_ptrs[0]);
  extern __shared__ char smem[];
  char *wk = smem + task_desc->smem_region_offset(TK_REGION_WORK);
  topk_compute(wk, inter, bias, logits_out, meta, local_expert_start,
               num_local_experts, routed_scaling_factor);
}

// Small helper: load the routing meta into registers (uniform loads).
struct RoutingMeta {
  int active_count;
  int experts[MAX_ACTIVE];
  float weights[MAX_ACTIVE];
};
__device__ __forceinline__ RoutingMeta load_meta(int const *meta) {
  RoutingMeta m;
  m.active_count = meta[META_OFF_COUNT];
#pragma unroll
  for (int s = 0; s < MAX_ACTIVE; s++) {
    m.experts[s] = meta[META_OFF_EXPERTS + s];
    m.weights[s] = __int_as_float(meta[META_OFF_WEIGHTS + s]);
  }
  return m;
}

// ============================================================================
// T3 — w13_gemv. v1 Phase 1: routed W13 (8-row blocks, dgemv_cpa16_h2<8,4>)
// + shared gate_up (4-row blocks, dgemv_cpa16_h2<4,2>), warp-item map
// IDENTICAL to v1 (idx over active_count*(1024/8) + 512/4), strided by the
// global warp id. The item space scales with runtime active_count while the
// task count stays static — dynamic balance in a static plan.
//   inputs : [0] meta i32[META_INTS]      (the T2->T3 chain edge)
//            [1] a_fp8 u8[HIDDEN]  [2] a_scale f32[KG1]   (aliases, hidden)
//            [3] w13 u8[E,1024,7168]   [4] w13_scale f32[E,8,56]
//            [5] wgu u8[512,7168]      [6] wgu_scale f32[4,56]
//   outputs: [0] y13 f32[MAX_ACTIVE,1024]   [1] sg f32[512]
// ============================================================================
__device__ __noinline__ void
    w13_gemv_task_impl(mirage::runtime::TaskDesc const *task_desc,
                       int task_offset,
                       int num_tasks,
                       int nwarps,
                       int op_sem_addr) {
  int const *meta = static_cast<int const *>(task_desc->input_ptrs[0]);
  uint8_t const *a_fp8_g =
      static_cast<uint8_t const *>(task_desc->input_ptrs[1]);
  float const *a_scale_g = static_cast<float const *>(task_desc->input_ptrs[2]);
  uint8_t const *w13 = static_cast<uint8_t const *>(task_desc->input_ptrs[3]);
  float const *w13_scale =
      static_cast<float const *>(task_desc->input_ptrs[4]);
  uint8_t const *wgu = static_cast<uint8_t const *>(task_desc->input_ptrs[5]);
  float const *wgu_s = static_cast<float const *>(task_desc->input_ptrs[6]);
  float *y13 = static_cast<float *>(task_desc->output_ptrs[0]);
  float *sg = static_cast<float *>(task_desc->output_ptrs[1]);

  extern __shared__ char smem[];
  uint8_t *s_a = reinterpret_cast<uint8_t *>(
      smem + task_desc->smem_region_offset(W13_REGION_ACT));
  float *s_as = reinterpret_cast<float *>(s_a + W13_ACT_SCALE_OFF);
  uint4 *s_ring = reinterpret_cast<uint4 *>(
      smem + task_desc->smem_region_offset(W13_REGION_RING));

  int const lane = threadIdx.x & 31;
  int const ws = threadIdx.x >> 5;
  bool const is_consumer = threadIdx.x < 128;

  if (is_consumer) {
    uint32_t const sb = static_cast<uint32_t>(__cvta_generic_to_shared(s_a));
    uint4 const *ga = reinterpret_cast<uint4 const *>(a_fp8_g);
    constexpr int NU4_A = HIDDEN / 16; // 448
    for (int u = threadIdx.x; u < NU4_A; u += 128) {
      v1k::cpasync16(sb + (uint32_t)u * 16, &ga[u]);
    }
    uint4 const *gs = reinterpret_cast<uint4 const *>(a_scale_g);
    constexpr int NU4_S = KG1 * 4 / 16; // 14
    uint32_t const sbs =
        static_cast<uint32_t>(__cvta_generic_to_shared(s_as));
    for (int u = threadIdx.x; u < NU4_S; u += 128) {
      v1k::cpasync16(sbs + (uint32_t)u * 16, &gs[u]);
    }
    v1k::cpasync_commit();
    v1k::cpasync_wait<0>();
    consumer_sync();
    if (op_sem_addr >= 0 && threadIdx.x == 0) {
      ffnv2_mbar_arrive_addr(op_sem_addr + SEMOFF_ACT_READY);
    }
  } else {
    ffnv2_mbar_wait_addr(op_sem_addr + SEMOFF_ACT_READY, 0);
  }

  RoutingMeta const m = load_meta(meta);

  constexpr int RBX_W13 = 8;
  constexpr int ST_W13 = 4;
  constexpr int RBX_SH = 4;
  constexpr int ST_SH13 = 2;
  int const n13 = m.active_count * (W13_N / RBX_W13);
  int const nsh1 = SH_GU_N / RBX_SH;
  int const ntot1 = n13 + nsh1;

  uint4 *my_ring = s_ring + (size_t)ws * (GEMV_RING_BYTES_PER_WARP / 16);
  for (int idx = task_offset * nwarps + ws; idx < ntot1;
       idx += num_tasks * nwarps) {
    if (idx < n13) {
      int const slot = idx / (W13_N / RBX_W13);
      int const n0 = (idx % (W13_N / RBX_W13)) * RBX_W13;
      int const e = m.experts[slot];
      uint8_t const *wb = w13 + (size_t)e * W13_N * HIDDEN;
      float const *wsc = w13_scale + (size_t)e * v1k::NB1 * KG1 +
                         (size_t)(n0 / GRP) * KG1;
      float yb[RBX_W13];
      v1k::dgemv_cpa16_h2<RBX_W13, ST_W13>(
          s_a, s_as, wb, wsc, HIDDEN, KG1, n0, lane, my_ring, yb);
      if (lane == 0) {
#pragma unroll
        for (int r = 0; r < RBX_W13; r++) {
          y13[(size_t)slot * W13_N + n0 + r] = yb[r];
        }
      }
    } else {
      int const n0 = (idx - n13) * RBX_SH;
      float const *wsc = wgu_s + (size_t)(n0 / GRP) * v1k::KG_SHGU;
      float yb[RBX_SH];
      v1k::dgemv_cpa16_h2<RBX_SH, ST_SH13>(
          s_a, s_as, wgu, wsc, v1k::SH_GU_K, v1k::KG_SHGU, n0, lane, my_ring,
          yb);
      if (lane == 0) {
#pragma unroll
        for (int r = 0; r < RBX_SH; r++) {
          sg[n0 + r] = yb[r];
        }
      }
    }
  }

  mac_task_epilogue(is_consumer, op_sem_addr);
}

// ============================================================================
// T4 — silu_quant. v1 Phase 2: routed silu_fast(gate)*up + per-128-group
// UE8M0 requant; shared ditto. Group-local math (verbatim); reads y13/sg
// from GMEM (float4 loads, 16B-aligned) instead of v1's block-local staging
// — value-neutral. Single task, consumer-only (tiny op).
//   inputs : [0] y13 f32[MAX_ACTIVE,1024]   [1] sg f32[512]
//            [2] meta i32 (alias)
//            [3] i_scale f32[MAX_ACTIVE,KG2] (hidden write)
//            [4] si_scale f32[KG_SHDN]       (hidden write)
//   outputs: [0] i_fp8 u8[MAX_ACTIVE,512]    [1] si_fp8 u8[256]
// ============================================================================
__device__ __noinline__ void
    silu_quant_task_impl(mirage::runtime::TaskDesc const *task_desc) {
  float const *y13 = static_cast<float const *>(task_desc->input_ptrs[0]);
  float const *sg = static_cast<float const *>(task_desc->input_ptrs[1]);
  int const *meta = static_cast<int const *>(task_desc->input_ptrs[2]);
  float *i_scale = static_cast<float *>(task_desc->input_ptrs[3]);
  float *si_scale = static_cast<float *>(task_desc->input_ptrs[4]);
  uint8_t *i_fp8 = static_cast<uint8_t *>(task_desc->output_ptrs[0]);
  uint8_t *si_fp8 = static_cast<uint8_t *>(task_desc->output_ptrs[1]);

  int const lane = threadIdx.x & 31;
  int const wid = threadIdx.x >> 5; // 0..3

  int const active_count = meta[META_OFF_COUNT];
  int const ng = active_count * KG2;
  for (int gg = wid; gg < ng; gg += 4) {
    int const slot = gg / KG2;
    int const g = gg % KG2;
    float const *y = y13 + (size_t)slot * W13_N;
    int const i0 = g * GRP + lane * 4;
    float4 const gpart = *reinterpret_cast<float4 const *>(&y[i0]);
    float4 const upart = *reinterpret_cast<float4 const *>(&y[512 + i0]);
    float v[4], amax = 0.f;
    v[0] = v1k::silu_fast(gpart.x) * upart.x;
    v[1] = v1k::silu_fast(gpart.y) * upart.y;
    v[2] = v1k::silu_fast(gpart.z) * upart.z;
    v[3] = v1k::silu_fast(gpart.w) * upart.w;
#pragma unroll
    for (int t = 0; t < 4; t++) {
      amax = fmaxf(amax, fabsf(v[t]));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    }
    float const s = v1k::quant_scale(amax);
    float const inv = 1.f / s;
    if (lane == 0) {
      i_scale[slot * KG2 + g] = s;
    }
#pragma unroll
    for (int t = 0; t < 4; t++) {
      i_fp8[(size_t)slot * W2_K + i0 + t] = v1k::to_f8(v[t] * inv);
    }
  }
  // shared expert: gate = sg[0:256], up = sg[256:512] (verbatim).
  for (int g = wid; g < KG_SHDN; g += 4) {
    float v[4], amax = 0.f;
#pragma unroll
    for (int t = 0; t < 4; t++) {
      int const i = g * GRP + lane * 4 + t;
      float const val = v1k::silu_fast(sg[i]) * sg[256 + i];
      v[t] = val;
      amax = fmaxf(amax, fabsf(val));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    }
    float const s = v1k::quant_scale(amax);
    float const inv = 1.f / s;
    if (lane == 0) {
      si_scale[g] = s;
    }
#pragma unroll
    for (int t = 0; t < 4; t++) {
      int const i = g * GRP + lane * 4 + t;
      si_fp8[i] = v1k::to_f8(v[t] * inv);
    }
  }
  // No SMEM regions -> no page-release ordering concern; GMEM stores are
  // published by the role-loop FINISHED arrive + event release chain.
}

// ============================================================================
// T5 — w2_gemv. v1 Phase 3 + final, OUTPUT-STATIONARY: each warp-item owns
// one RBLK-row block of out; it accumulates all active routed slots
// (dgemv_cpa16_h2<RBLK,2> per slot — per-contribution values bit-exact vs
// v1) plus the covered shared-down 4-row blocks (dgemv_cpa<4,3>, v1's exact
// shared path) in fp32 registers, then stores bf16 once. No atomicAdd, no
// zero-init, deterministic sum order (v1's atomicAdd order was
// nondeterministic).
//   inputs : [0] i_fp8 u8[MAX_ACTIVE,512]   [1] si_fp8 u8[256]
//            [2] meta i32 (alias)   [3] i_scale f32 (alias)
//            [4] si_scale f32 (alias)
//            [5] w2 u8[E,7168,512]   [6] w2_scale f32[E,56,4]
//            [7] wdn u8[7168,256]    [8] wdn_scale f32[56,2]
//   outputs: [0] out bf16[1,7168]
// ============================================================================
template <int RBLK>
__device__ __noinline__ void
    w2_gemv_task_impl(mirage::runtime::TaskDesc const *task_desc,
                      int task_offset,
                      int num_tasks,
                      int nwarps,
                      int op_sem_addr) {
  static_assert(RBLK == 16 || RBLK == 8, "RBLK must divide GRP and be >=4");
  uint8_t const *i_fp8_g =
      static_cast<uint8_t const *>(task_desc->input_ptrs[0]);
  uint8_t const *si_fp8_g =
      static_cast<uint8_t const *>(task_desc->input_ptrs[1]);
  int const *meta = static_cast<int const *>(task_desc->input_ptrs[2]);
  float const *i_scale_g = static_cast<float const *>(task_desc->input_ptrs[3]);
  float const *si_scale_g =
      static_cast<float const *>(task_desc->input_ptrs[4]);
  uint8_t const *w2 = static_cast<uint8_t const *>(task_desc->input_ptrs[5]);
  float const *w2s = static_cast<float const *>(task_desc->input_ptrs[6]);
  uint8_t const *wdn = static_cast<uint8_t const *>(task_desc->input_ptrs[7]);
  float const *wdns = static_cast<float const *>(task_desc->input_ptrs[8]);
  __nv_bfloat16 *out = static_cast<__nv_bfloat16 *>(task_desc->output_ptrs[0]);

  extern __shared__ char smem[];
  uint8_t *s_act = reinterpret_cast<uint8_t *>(
      smem + task_desc->smem_region_offset(W2_REGION_ACT));
  uint8_t *s_ifp8 = s_act;
  float *s_iscale = reinterpret_cast<float *>(s_act + W2_ACT_ISCALE_OFF);
  uint8_t *s_sifp8 = s_act + W2_ACT_SIFP8_OFF;
  float *s_siscale = reinterpret_cast<float *>(s_act + W2_ACT_SISCALE_OFF);
  uint4 *s_ring = reinterpret_cast<uint4 *>(
      smem + task_desc->smem_region_offset(W2_REGION_RING));

  int const lane = threadIdx.x & 31;
  int const ws = threadIdx.x >> 5;
  bool const is_consumer = threadIdx.x < 128;

  if (is_consumer) {
    uint32_t const sb =
        static_cast<uint32_t>(__cvta_generic_to_shared(s_act));
    uint4 const *g4 = reinterpret_cast<uint4 const *>(i_fp8_g);
    constexpr int NU4_I = MAX_ACTIVE * W2_K / 16; // 256
    for (int u = threadIdx.x; u < NU4_I; u += 128) {
      v1k::cpasync16(sb + (uint32_t)u * 16, &g4[u]);
    }
    uint4 const *gsc = reinterpret_cast<uint4 const *>(i_scale_g);
    constexpr int NU4_IS = MAX_ACTIVE * KG2 * 4 / 16; // 8
    for (int u = threadIdx.x; u < NU4_IS; u += 128) {
      v1k::cpasync16(sb + W2_ACT_ISCALE_OFF + (uint32_t)u * 16, &gsc[u]);
    }
    uint4 const *gsi = reinterpret_cast<uint4 const *>(si_fp8_g);
    constexpr int NU4_SI = SH_DN_K / 16; // 16
    for (int u = threadIdx.x; u < NU4_SI; u += 128) {
      v1k::cpasync16(sb + W2_ACT_SIFP8_OFF + (uint32_t)u * 16, &gsi[u]);
    }
    // si_scale is 8 B (not a 16B multiple): plain copy.
    if (threadIdx.x < KG_SHDN) {
      s_siscale[threadIdx.x] = si_scale_g[threadIdx.x];
    }
    v1k::cpasync_commit();
    v1k::cpasync_wait<0>();
    consumer_sync();
    if (op_sem_addr >= 0 && threadIdx.x == 0) {
      ffnv2_mbar_arrive_addr(op_sem_addr + SEMOFF_ACT_READY);
    }
  } else {
    ffnv2_mbar_wait_addr(op_sem_addr + SEMOFF_ACT_READY, 0);
  }

  RoutingMeta const m = load_meta(meta);

  constexpr int ST_W2 = 2;
  constexpr int RBX_SH = 4;
  constexpr int ST_SH2 = 3;
  int const nblk = W2_N / RBLK;
  uint4 *my_ring = s_ring + (size_t)ws * (GEMV_RING_BYTES_PER_WARP / 16);

  for (int item = task_offset * nwarps + ws; item < nblk;
       item += num_tasks * nwarps) {
    int const n0 = item * RBLK;
    float acc[RBLK];
#pragma unroll
    for (int r = 0; r < RBLK; r++) {
      acc[r] = 0.f;
    }
    for (int slot = 0; slot < m.active_count; slot++) {
      int const e = m.experts[slot];
      float const ew = m.weights[slot];
      float yb[RBLK];
      v1k::dgemv_cpa16_h2<RBLK, ST_W2>(
          s_ifp8 + (size_t)slot * W2_K,
          s_iscale + slot * KG2,
          w2 + (size_t)e * W2_N * W2_K,
          w2s + (size_t)e * v1k::NB2 * KG2 + (size_t)(n0 / GRP) * KG2,
          W2_K, KG2, n0, lane, my_ring, yb);
      if (lane == 0) {
#pragma unroll
        for (int r = 0; r < RBLK; r++) {
          acc[r] += ew * yb[r];
        }
      }
    }
    // shared down: the RBLK/4 covered 4-row blocks (v1's exact shared path,
    // 4-byte dgemv_cpa staging into a uint32 view of the same warp ring).
#pragma unroll
    for (int sb4 = 0; sb4 < RBLK / RBX_SH; sb4++) {
      int const mm0 = n0 + sb4 * RBX_SH;
      float yb4[RBX_SH];
      v1k::dgemv_cpa<RBX_SH, ST_SH2>(
          s_sifp8, s_siscale, wdn,
          wdns + (size_t)(mm0 / GRP) * KG_SHDN,
          SH_DN_K, KG_SHDN, mm0, lane,
          reinterpret_cast<uint32_t *>(my_ring), yb4);
      if (lane == 0) {
#pragma unroll
        for (int r = 0; r < RBX_SH; r++) {
          acc[sb4 * RBX_SH + r] += yb4[r];
        }
      }
    }
    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < RBLK; r++) {
        out[n0 + r] = __float2bfloat16_rn(acc[r]);
      }
    }
  }

  mac_task_epilogue(is_consumer, op_sem_addr);
}

// ============================================================================
// FOLDED 3-op chain: router_quant_rms -> w13_topk -> w2_silu. Replicates
// v1's redundant-per-CTA rmsnorm / topk / silu INSIDE the MAC tasks, removing
// the three 1-task serial ops and their dep boundaries. All redundant math is
// task-invariant (identical thread mapping + reduce order per task => every
// task computes identical bytes), so the work partition stays value-neutral.
// ============================================================================

// ----------------------------------------------------------------------------
// T1' — router_quant_rms. v1 Phase A (redundant rmsnorm) + Phase 0 (quant) +
// Phase B (router split-K GEMV).
//   inputs : [0] hidden bf16[1,HIDDEN]          (chain edge in)
//            [1] rms_w bf16[HIDDEN]
//            [2] router_gate_w bf16[ROUTER_N,HIDDEN]
//            [3] a_fp8 u8[HIDDEN]      (hidden write)
//            [4] a_scale f32[KG1]      (hidden write)
//            [5] rmsnorm_out bf16[1,HIDDEN] (hidden write, task 0 — artifact)
//   outputs: [0] inter f32[ROUTER_N*RKSPLIT]
// ----------------------------------------------------------------------------
__device__ __noinline__ void
    router_quant_rms_task_impl(mirage::runtime::TaskDesc const *task_desc,
                               int task_offset,
                               int num_tasks,
                               int nwarps,
                               unsigned long long sync_tag) {
  __nv_bfloat16 const *x =
      static_cast<__nv_bfloat16 const *>(task_desc->input_ptrs[0]);
  __nv_bfloat16 const *rms_w =
      static_cast<__nv_bfloat16 const *>(task_desc->input_ptrs[1]);
  __nv_bfloat16 const *wr =
      static_cast<__nv_bfloat16 const *>(task_desc->input_ptrs[2]);
  uint8_t *a_fp8 = static_cast<uint8_t *>(task_desc->input_ptrs[3]);
  float *a_scale = static_cast<float *>(task_desc->input_ptrs[4]);
  __nv_bfloat16 *rmsnorm_out =
      static_cast<__nv_bfloat16 *>(task_desc->input_ptrs[5]);
  float *inter = static_cast<float *>(task_desc->output_ptrs[0]);

  extern __shared__ char smem[];
  char *nb = smem + task_desc->smem_region_offset(RQ_REGION_NORM);
  __nv_bfloat16 *s_norm = reinterpret_cast<__nv_bfloat16 *>(nb);
  float *s_red = reinterpret_cast<float *>(nb + RQR_OFF_RED);
  uint64_t *s_flags = reinterpret_cast<uint64_t *>(nb + RQR_OFF_FLAGS);
  uint4 *s_ring = reinterpret_cast<uint4 *>(
      smem + task_desc->smem_region_offset(RQ_REGION_RING));

  int const lane = threadIdx.x & 31;
  int const ws = threadIdx.x >> 5;
  bool const is_consumer = threadIdx.x < 128;

  if (is_consumer) {
    // Stage hidden into s_norm, then rmsnorm IN PLACE. VECTORIZED (uint4 = 8
    // bf16 per access): the naive scalar-bf16 loops are latency-bound at
    // ~15-20us per CTA (measured in v1's Phase A probe AND the first fold
    // attempt); the same 28KB of reads + 7K FMAs is ~3-4us vectorized.
    // Identical thread mapping + fixed 4-partial sum order per task =>
    // every task still produces identical normed bytes.
    uint32_t const sb =
        static_cast<uint32_t>(__cvta_generic_to_shared(s_norm));
    uint4 const *g4 = reinterpret_cast<uint4 const *>(x);
    constexpr int NU4 = HIDDEN * 2 / 16; // 896
    for (int u = threadIdx.x; u < NU4; u += 128) {
      v1k::cpasync16(sb + (uint32_t)u * 16, &g4[u]);
    }
    v1k::cpasync_commit();
    v1k::cpasync_wait<0>();
    consumer_sync();
    uint4 *s_norm4 = reinterpret_cast<uint4 *>(s_norm);
    float ss = 0.f;
#pragma unroll
    for (int r = 0; r < NU4 / 128; r++) { // 7 rounds, independent
      uint4 const q = s_norm4[threadIdx.x + r * 128];
      __nv_bfloat162 const *h2 = reinterpret_cast<__nv_bfloat162 const *>(&q);
#pragma unroll
      for (int j = 0; j < 4; j++) {
        float2 const f = __bfloat1622float2(h2[j]);
        ss += f.x * f.x + f.y * f.y;
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      ss += __shfl_xor_sync(0xffffffffu, ss, o);
    }
    if (lane == 0) {
      s_red[ws] = ss;
    }
    consumer_sync();
    // fixed-order 4-partial sum (deterministic, identical across threads and
    // tasks).
    float const tot = s_red[0] + s_red[1] + s_red[2] + s_red[3];
    float const rms_rcp = rsqrtf(tot / float(HIDDEN) + v1k::RMS_EPS);
    uint4 const *w4 = reinterpret_cast<uint4 const *>(rms_w);
#pragma unroll
    for (int r = 0; r < NU4 / 128; r++) {
      int const u = threadIdx.x + r * 128;
      uint4 const qx = s_norm4[u];
      uint4 const qw = w4[u]; // vectorized GMEM read (L2-shared across tasks)
      __nv_bfloat162 const *x2 = reinterpret_cast<__nv_bfloat162 const *>(&qx);
      __nv_bfloat162 const *w2 = reinterpret_cast<__nv_bfloat162 const *>(&qw);
      uint4 qo;
      __nv_bfloat162 *o2 = reinterpret_cast<__nv_bfloat162 *>(&qo);
#pragma unroll
      for (int j = 0; j < 4; j++) {
        float2 const fx = __bfloat1622float2(x2[j]);
        float2 const fw = __bfloat1622float2(w2[j]);
        o2[j] = __floats2bfloat162_rn(fx.x * rms_rcp * fw.x,
                                      fx.y * rms_rcp * fw.y);
      }
      s_norm4[u] = qo;
    }
    consumer_sync();
    // helpers may only see NORMED bytes (design-review hard requirement).
    if (sync_tag != 0 && threadIdx.x == 0) {
      ffnv2_flag_store_release(&s_flags[0], sync_tag);
    }
    if (task_offset == 0) {
      uint4 *ro4 = reinterpret_cast<uint4 *>(rmsnorm_out);
#pragma unroll
      for (int r = 0; r < NU4 / 128; r++) {
        int const u = threadIdx.x + r * 128;
        ro4[u] = s_norm4[u];
      }
    }
  } else {
    ffnv2_flag_wait(&s_flags[0], sync_tag);
    __syncwarp();
  }

  // v1 Phase 0 quant + Phase B router GEMV (identical to router_quant_task).
  for (int g = task_offset * nwarps + ws; g < KG1; g += num_tasks * nwarps) {
    v1k::quant_group_warp<__nv_bfloat16>(s_norm, a_fp8, a_scale, g, lane);
  }
  uint4 *my_ring = s_ring + (size_t)ws * (RQ_RING_BYTES_PER_WARP / 16);
  int const total_pairs = ROUTER_N * RKSPLIT;
  for (int t = task_offset * nwarps + ws; t < total_pairs;
       t += num_tasks * nwarps) {
    int const e = t / RKSPLIT, sp = t % RKSPLIT;
    float const acc = v1k::router_partial_cpa<RKSPLIT, 4>(
        s_norm, wr + (size_t)e * v1k::ROUTER_K, sp, lane, my_ring);
    if (lane == 0) {
      inter[e * RKSPLIT + sp] = acc;
    }
  }

  mac_task_epilogue(is_consumer, s_flags, sync_tag);
}

// ----------------------------------------------------------------------------
// T2' — w13_topk. Redundant per-task topk (v1 Phase C trick) + the W13 GEMV.
// The a_fp8/a_scale cp.async staging is ISSUED first and drains while the
// topk computes (the fold both removes the serial topk op and hides the
// activation load latency under it).
//   inputs : [0] inter f32 (chain edge)   [1] bias f32[ROUTER_N]
//            [2] a_fp8 u8 (alias)         [3] a_scale f32 (alias)
//            [4] w13   [5] w13_scale      [6] wgu   [7] wgu_scale
//            [8] meta i32[META_INTS]  (hidden write, task 0 — artifact +
//                the w2_silu routing source)
//            [9] logits bf16[ROUTER_N] (hidden write, task 0 — artifact)
//   outputs: [0] y13 f32[MAX_ACTIVE,1024]   [1] sg f32[512]
// ----------------------------------------------------------------------------
__device__ __noinline__ void
    w13_topk_task_impl(mirage::runtime::TaskDesc const *task_desc,
                       int task_offset,
                       int num_tasks,
                       int nwarps,
                       unsigned long long sync_tag,
                       int local_expert_start,
                       int num_local_experts,
                       float routed_scaling_factor) {
  float const *inter = static_cast<float const *>(task_desc->input_ptrs[0]);
  float const *bias = static_cast<float const *>(task_desc->input_ptrs[1]);
  uint8_t const *a_fp8_g =
      static_cast<uint8_t const *>(task_desc->input_ptrs[2]);
  float const *a_scale_g = static_cast<float const *>(task_desc->input_ptrs[3]);
  uint8_t const *w13 = static_cast<uint8_t const *>(task_desc->input_ptrs[4]);
  float const *w13_scale =
      static_cast<float const *>(task_desc->input_ptrs[5]);
  uint8_t const *wgu = static_cast<uint8_t const *>(task_desc->input_ptrs[6]);
  float const *wgu_s = static_cast<float const *>(task_desc->input_ptrs[7]);
  int *meta_gmem = static_cast<int *>(task_desc->input_ptrs[8]);
  __nv_bfloat16 *logits_out =
      static_cast<__nv_bfloat16 *>(task_desc->input_ptrs[9]);
  float *y13 = static_cast<float *>(task_desc->output_ptrs[0]);
  float *sg = static_cast<float *>(task_desc->output_ptrs[1]);

  extern __shared__ char smem[];
  uint8_t *s_a = reinterpret_cast<uint8_t *>(
      smem + task_desc->smem_region_offset(W13TK_REGION_ACT));
  float *s_as = reinterpret_cast<float *>(s_a + W13_ACT_SCALE_OFF);
  uint4 *s_ring = reinterpret_cast<uint4 *>(
      smem + task_desc->smem_region_offset(W13TK_REGION_RING));
  char *tk = smem + task_desc->smem_region_offset(W13TK_REGION_TK);
  int const *s_meta = reinterpret_cast<int const *>(tk + TK_OFF_META);
  uint64_t *s_flags = reinterpret_cast<uint64_t *>(tk + TK_OFF_FLAGS);

  int const lane = threadIdx.x & 31;
  int const ws = threadIdx.x >> 5;
  bool const is_consumer = threadIdx.x < 128;

  if (is_consumer) {
    // (1) issue the activation staging; do NOT wait — it lands during topk.
    uint32_t const sb = static_cast<uint32_t>(__cvta_generic_to_shared(s_a));
    uint4 const *ga = reinterpret_cast<uint4 const *>(a_fp8_g);
    constexpr int NU4_A = HIDDEN / 16; // 448
    for (int u = threadIdx.x; u < NU4_A; u += 128) {
      v1k::cpasync16(sb + (uint32_t)u * 16, &ga[u]);
    }
    uint4 const *gs = reinterpret_cast<uint4 const *>(a_scale_g);
    constexpr int NU4_S = KG1 * 4 / 16; // 14
    uint32_t const sbs =
        static_cast<uint32_t>(__cvta_generic_to_shared(s_as));
    for (int u = threadIdx.x; u < NU4_S; u += 128) {
      v1k::cpasync16(sbs + (uint32_t)u * 16, &gs[u]);
    }
    v1k::cpasync_commit();
    // (2) redundant per-task topk (task 0 also publishes the GMEM artifacts).
    topk_compute(tk, inter, bias,
                 task_offset == 0 ? logits_out : nullptr,
                 task_offset == 0 ? meta_gmem : nullptr,
                 local_expert_start, num_local_experts,
                 routed_scaling_factor);
    // (3) activation staged + s_meta published -> release the helpers.
    v1k::cpasync_wait<0>();
    consumer_sync();
    if (sync_tag != 0 && threadIdx.x == 0) {
      ffnv2_flag_store_release(&s_flags[0], sync_tag);
    }
  } else {
    ffnv2_flag_wait(&s_flags[0], sync_tag);
    __syncwarp();
  }

  RoutingMeta const m = load_meta(s_meta);

  constexpr int RBX_W13 = 8;
  constexpr int ST_W13 = 4;
  constexpr int RBX_SH = 4;
  constexpr int ST_SH13 = 2;
  int const n13 = m.active_count * (W13_N / RBX_W13);
  int const nsh1 = SH_GU_N / RBX_SH;
  int const ntot1 = n13 + nsh1;

  uint4 *my_ring = s_ring + (size_t)ws * (GEMV_RING_BYTES_PER_WARP / 16);
  for (int idx = task_offset * nwarps + ws; idx < ntot1;
       idx += num_tasks * nwarps) {
    if (idx < n13) {
      int const slot = idx / (W13_N / RBX_W13);
      int const n0 = (idx % (W13_N / RBX_W13)) * RBX_W13;
      int const e = m.experts[slot];
      uint8_t const *wb = w13 + (size_t)e * W13_N * HIDDEN;
      float const *wsc = w13_scale + (size_t)e * v1k::NB1 * KG1 +
                         (size_t)(n0 / GRP) * KG1;
      float yb[RBX_W13];
      v1k::dgemv_cpa16_h2<RBX_W13, ST_W13>(
          s_a, s_as, wb, wsc, HIDDEN, KG1, n0, lane, my_ring, yb);
      if (lane == 0) {
#pragma unroll
        for (int r = 0; r < RBX_W13; r++) {
          y13[(size_t)slot * W13_N + n0 + r] = yb[r];
        }
      }
    } else {
      int const n0 = (idx - n13) * RBX_SH;
      float const *wsc = wgu_s + (size_t)(n0 / GRP) * v1k::KG_SHGU;
      float yb[RBX_SH];
      v1k::dgemv_cpa16_h2<RBX_SH, ST_SH13>(
          s_a, s_as, wgu, wsc, v1k::SH_GU_K, v1k::KG_SHGU, n0, lane, my_ring,
          yb);
      if (lane == 0) {
#pragma unroll
        for (int r = 0; r < RBX_SH; r++) {
          sg[n0 + r] = yb[r];
        }
      }
    }
  }

  mac_task_epilogue(is_consumer, s_flags, sync_tag);
}

// ----------------------------------------------------------------------------
// T3' — w2_silu. Redundant per-task silu+requant (v1 Phase 2 trick, incl. the
// v1 cp.async y13-staging fast path) + the output-stationary W2 GEMV. The
// y13/sg staging borrows the RING region (<= 34 KB, spans consumer slices
// only); the post-silu consumer_sync separates all silu reads from the GEMV's
// ring reuse.
//   inputs : [0] y13 f32 (chain edge)   [1] sg f32 (chain edge)
//            [2] meta i32 (alias — read from GMEM, transitively ordered)
//            [3] i_fp8 u8 (hidden write, task 0)  [4] i_scale f32 (hidden, t0)
//            [5] si_fp8 u8 (hidden, t0)           [6] si_scale f32 (hidden, t0)
//            [7] w2   [8] w2_scale   [9] wdn   [10] wdn_scale
//   outputs: [0] out bf16[1,W2_N]
// ----------------------------------------------------------------------------
template <int RBLK>
__device__ __noinline__ void
    w2_silu_task_impl(mirage::runtime::TaskDesc const *task_desc,
                      int task_offset,
                      int num_tasks,
                      int nwarps,
                      unsigned long long sync_tag) {
  static_assert(RBLK == 16 || RBLK == 8, "RBLK must divide GRP and be >=4");
  float const *y13_g = static_cast<float const *>(task_desc->input_ptrs[0]);
  float const *sg_g = static_cast<float const *>(task_desc->input_ptrs[1]);
  int const *meta = static_cast<int const *>(task_desc->input_ptrs[2]);
  uint8_t *i_fp8_gmem = static_cast<uint8_t *>(task_desc->input_ptrs[3]);
  float *i_scale_gmem = static_cast<float *>(task_desc->input_ptrs[4]);
  uint8_t *si_fp8_gmem = static_cast<uint8_t *>(task_desc->input_ptrs[5]);
  float *si_scale_gmem = static_cast<float *>(task_desc->input_ptrs[6]);
  uint8_t const *w2 = static_cast<uint8_t const *>(task_desc->input_ptrs[7]);
  float const *w2s = static_cast<float const *>(task_desc->input_ptrs[8]);
  uint8_t const *wdn = static_cast<uint8_t const *>(task_desc->input_ptrs[9]);
  float const *wdns = static_cast<float const *>(task_desc->input_ptrs[10]);
  __nv_bfloat16 *out = static_cast<__nv_bfloat16 *>(task_desc->output_ptrs[0]);

  extern __shared__ char smem[];
  uint8_t *s_act = reinterpret_cast<uint8_t *>(
      smem + task_desc->smem_region_offset(W2_REGION_ACT));
  uint8_t *s_ifp8 = s_act;
  float *s_iscale = reinterpret_cast<float *>(s_act + W2_ACT_ISCALE_OFF);
  uint8_t *s_sifp8 = s_act + W2_ACT_SIFP8_OFF;
  float *s_siscale = reinterpret_cast<float *>(s_act + W2_ACT_SISCALE_OFF);
  uint4 *s_ring = reinterpret_cast<uint4 *>(
      smem + task_desc->smem_region_offset(W2_REGION_RING));
  // y13/sg staging views over the ring (silu phase only).
  float *s_y13 = reinterpret_cast<float *>(
      reinterpret_cast<char *>(s_ring) + W2S_RING_Y13_OFF);
  float *s_sg = reinterpret_cast<float *>(
      reinterpret_cast<char *>(s_ring) + W2S_RING_SG_OFF);
  uint64_t *s_flags = reinterpret_cast<uint64_t *>(s_act + W2_ACT_FLAGS_OFF);

  int const lane = threadIdx.x & 31;
  int const ws = threadIdx.x >> 5;
  bool const is_consumer = threadIdx.x < 128;

  // Consumers may read meta immediately (their dep-prefix acquire orders it
  // vs the producing w13_topk's task-0 write). HELPERS run NO dep-prefix —
  // their ordering rides the ACT_READY flag acquire, so they must load meta
  // only after the flag wait (below).
  RoutingMeta m;
  if (is_consumer) {
    m = load_meta(meta);
  }

  if (is_consumer) {
    // (1) stage y13[0..active*W13_N) + sg into the ring (v1 Phase-2 fast
    // path: one coalesced cp.async run + one wait).
    uint32_t const sb =
        static_cast<uint32_t>(__cvta_generic_to_shared(s_y13));
    uint4 const *y4 = reinterpret_cast<uint4 const *>(y13_g);
    int const nu4_y = (m.active_count * W13_N) >> 2; // uint4 count
    for (int u = threadIdx.x; u < nu4_y; u += 128) {
      v1k::cpasync16(sb + (uint32_t)u * 16, &y4[u]);
    }
    uint32_t const sbs =
        static_cast<uint32_t>(__cvta_generic_to_shared(s_sg));
    uint4 const *g4 = reinterpret_cast<uint4 const *>(sg_g);
    constexpr int NU4_SG = SH_GU_N / 4; // 128
    for (int u = threadIdx.x; u < NU4_SG; u += 128) {
      v1k::cpasync16(sbs + (uint32_t)u * 16, &g4[u]);
    }
    v1k::cpasync_commit();
    v1k::cpasync_wait<0>();
    consumer_sync();

    // (2) silu + requant (v1 Phase 2, verbatim group math) -> ACT region.
    int const wid = ws; // 0..3
    int const ng = m.active_count * KG2;
    for (int gg = wid; gg < ng; gg += 4) {
      int const slot = gg / KG2;
      int const g = gg % KG2;
      float const *y = s_y13 + (size_t)slot * W13_N;
      int const i0 = g * GRP + lane * 4;
      float4 const gpart = *reinterpret_cast<float4 const *>(&y[i0]);
      float4 const upart = *reinterpret_cast<float4 const *>(&y[512 + i0]);
      float v[4], amax = 0.f;
      v[0] = v1k::silu_fast(gpart.x) * upart.x;
      v[1] = v1k::silu_fast(gpart.y) * upart.y;
      v[2] = v1k::silu_fast(gpart.z) * upart.z;
      v[3] = v1k::silu_fast(gpart.w) * upart.w;
#pragma unroll
      for (int t = 0; t < 4; t++) {
        amax = fmaxf(amax, fabsf(v[t]));
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
      }
      float const s = v1k::quant_scale(amax);
      float const inv = 1.f / s;
      if (lane == 0) {
        s_iscale[slot * KG2 + g] = s;
      }
#pragma unroll
      for (int t = 0; t < 4; t++) {
        s_ifp8[(size_t)slot * W2_K + i0 + t] = v1k::to_f8(v[t] * inv);
      }
    }
    for (int g = wid; g < KG_SHDN; g += 4) {
      float v[4], amax = 0.f;
#pragma unroll
      for (int t = 0; t < 4; t++) {
        int const i = g * GRP + lane * 4 + t;
        float const val = v1k::silu_fast(s_sg[i]) * s_sg[256 + i];
        v[t] = val;
        amax = fmaxf(amax, fabsf(val));
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
      }
      float const s = v1k::quant_scale(amax);
      float const inv = 1.f / s;
      if (lane == 0) {
        s_siscale[g] = s;
      }
#pragma unroll
      for (int t = 0; t < 4; t++) {
        int const i = g * GRP + lane * 4 + t;
        s_sifp8[i] = v1k::to_f8(v[t] * inv);
      }
    }
    // ALL silu reads of the ring-staged y13/sg complete before this barrier;
    // the GEMV below may then reuse the ring for weight staging.
    consumer_sync();
    if (sync_tag != 0 && threadIdx.x == 0) {
      ffnv2_flag_store_release(&s_flags[0], sync_tag);
    }
    // (3) task 0 publishes the quant artifacts (compare + downstream debug).
    if (task_offset == 0) {
      for (int i = threadIdx.x; i < m.active_count * W2_K; i += 128) {
        i_fp8_gmem[i] = s_ifp8[i];
      }
      for (int i = threadIdx.x; i < m.active_count * KG2; i += 128) {
        i_scale_gmem[i] = s_iscale[i];
      }
      for (int i = threadIdx.x; i < SH_DN_K; i += 128) {
        si_fp8_gmem[i] = s_sifp8[i];
      }
      for (int i = threadIdx.x; i < KG_SHDN; i += 128) {
        si_scale_gmem[i] = s_siscale[i];
      }
    }
  } else {
    ffnv2_flag_wait(&s_flags[0], sync_tag);
    m = load_meta(meta); // ordered by the flag acquire (no helper dep-prefix)
  }

  constexpr int ST_W2 = 2;
  constexpr int RBX_SH = 4;
  constexpr int ST_SH2 = 3;
  int const nblk = W2_N / RBLK;
  uint4 *my_ring = s_ring + (size_t)ws * (GEMV_RING_BYTES_PER_WARP / 16);

  for (int item = task_offset * nwarps + ws; item < nblk;
       item += num_tasks * nwarps) {
    int const n0 = item * RBLK;
    float acc[RBLK];
#pragma unroll
    for (int r = 0; r < RBLK; r++) {
      acc[r] = 0.f;
    }
    for (int slot = 0; slot < m.active_count; slot++) {
      int const e = m.experts[slot];
      float const ew = m.weights[slot];
      float yb[RBLK];
      v1k::dgemv_cpa16_h2<RBLK, ST_W2>(
          s_ifp8 + (size_t)slot * W2_K,
          s_iscale + slot * KG2,
          w2 + (size_t)e * W2_N * W2_K,
          w2s + (size_t)e * v1k::NB2 * KG2 + (size_t)(n0 / GRP) * KG2,
          W2_K, KG2, n0, lane, my_ring, yb);
      if (lane == 0) {
#pragma unroll
        for (int r = 0; r < RBLK; r++) {
          acc[r] += ew * yb[r];
        }
      }
    }
#pragma unroll
    for (int sb4 = 0; sb4 < RBLK / RBX_SH; sb4++) {
      int const mm0 = n0 + sb4 * RBX_SH;
      float yb4[RBX_SH];
      v1k::dgemv_cpa<RBX_SH, ST_SH2>(
          s_sifp8, s_siscale, wdn,
          wdns + (size_t)(mm0 / GRP) * KG_SHDN,
          SH_DN_K, KG_SHDN, mm0, lane,
          reinterpret_cast<uint32_t *>(my_ring), yb4);
      if (lane == 0) {
#pragma unroll
        for (int r = 0; r < RBX_SH; r++) {
          acc[sb4 * RBX_SH + r] += yb4[r];
        }
      }
    }
    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < RBLK; r++) {
        out[n0 + r] = __float2bfloat16_rn(acc[r]);
      }
    }
  }

  mac_task_epilogue(is_consumer, s_flags, sync_tag);
}

// ============================================================================
// FUSION-LADDER EXPERIMENT (scratch/v2_ffn_fuse): Rung A 2-op chain
// (w13_rqr_topk -> w2_silu) and Rung B 1-op "megakernel shape" (ffn_mega).
// All math is the exact committed/v1 code: the rms / quant / router / topk /
// W13 / silu / W2 blocks below are verbatim copies of the tuned 3-op task
// bodies (which are themselves verbatim v1 helper calls); only the
// synchronization scaffolding differs.
// ============================================================================

static_assert(E_LOCAL == v1k::E_LOCAL && NB1 == v1k::NB1 && NB2 == v1k::NB2 &&
                  KG_SHGU == v1k::KG_SHGU && NB_SHGU == v1k::NB_SHGU &&
                  NB_SHDN == v1k::NB_SHDN,
              "fusion-ladder spec shapes drifted from the v1 kernel");

// Consumer-only: stage hidden into s_norm and rmsnorm IN PLACE. VERBATIM copy
// of the router_quant_rms_task_impl block (identical instruction sequence =>
// identical normed bytes). Caller handles flag release + artifact publish.
__device__ __forceinline__ void
    ffnv2_rms_stage_and_norm(__nv_bfloat16 const *x,
                             __nv_bfloat16 const *rms_w,
                             __nv_bfloat16 *s_norm,
                             float *s_red) {
  int const lane = threadIdx.x & 31;
  int const ws = threadIdx.x >> 5;
  uint32_t const sb = static_cast<uint32_t>(__cvta_generic_to_shared(s_norm));
  uint4 const *g4 = reinterpret_cast<uint4 const *>(x);
  constexpr int NU4 = HIDDEN * 2 / 16; // 896
  for (int u = threadIdx.x; u < NU4; u += 128) {
    v1k::cpasync16(sb + (uint32_t)u * 16, &g4[u]);
  }
  v1k::cpasync_commit();
  v1k::cpasync_wait<0>();
  consumer_sync();
  uint4 *s_norm4 = reinterpret_cast<uint4 *>(s_norm);
  float ss = 0.f;
#pragma unroll
  for (int r = 0; r < NU4 / 128; r++) { // 7 rounds, independent
    uint4 const q = s_norm4[threadIdx.x + r * 128];
    __nv_bfloat162 const *h2 = reinterpret_cast<__nv_bfloat162 const *>(&q);
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float2 const f = __bfloat1622float2(h2[j]);
      ss += f.x * f.x + f.y * f.y;
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    ss += __shfl_xor_sync(0xffffffffu, ss, o);
  }
  if (lane == 0) {
    s_red[ws] = ss;
  }
  consumer_sync();
  float const tot = s_red[0] + s_red[1] + s_red[2] + s_red[3];
  float const rms_rcp = rsqrtf(tot / float(HIDDEN) + v1k::RMS_EPS);
  uint4 const *w4 = reinterpret_cast<uint4 const *>(rms_w);
#pragma unroll
  for (int r = 0; r < NU4 / 128; r++) {
    int const u = threadIdx.x + r * 128;
    uint4 const qx = s_norm4[u];
    uint4 const qw = w4[u];
    __nv_bfloat162 const *x2 = reinterpret_cast<__nv_bfloat162 const *>(&qx);
    __nv_bfloat162 const *w2 = reinterpret_cast<__nv_bfloat162 const *>(&qw);
    uint4 qo;
    __nv_bfloat162 *o2 = reinterpret_cast<__nv_bfloat162 *>(&qo);
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float2 const fx = __bfloat1622float2(x2[j]);
      float2 const fw = __bfloat1622float2(w2[j]);
      o2[j] = __floats2bfloat162_rn(fx.x * rms_rcp * fw.x,
                                    fx.y * rms_rcp * fw.y);
    }
    s_norm4[u] = qo;
  }
  consumer_sync();
}

// Consumer-cooperative bf16[HIDDEN] SMEM -> GMEM publish (task-0 artifact).
__device__ __forceinline__ void
    ffnv2_publish_norm(__nv_bfloat16 const *s_norm, __nv_bfloat16 *dst) {
  uint4 const *s4 = reinterpret_cast<uint4 const *>(s_norm);
  uint4 *d4 = reinterpret_cast<uint4 *>(dst);
  constexpr int NU4 = HIDDEN * 2 / 16;
#pragma unroll
  for (int r = 0; r < NU4 / 128; r++) {
    int const u = threadIdx.x + r * 128;
    d4[u] = s4[u];
  }
}

// ----------------------------------------------------------------------------
// Rung A — w13_rqr_topk. The ENTIRE router_quant_rms op folded into the W13
// task prologue: rmsnorm (redundant, vectorized) + quant of ALL 56 groups
// (task-local, into SMEM — a_fp8 never round-trips GMEM) + the FULL router
// GEMV (all 1024 pairs task-locally strided — the deliberately-checked 136x
// redundancy: 3.67 MB bf16 router weight PER TASK) + redundant topk, then the
// W13+sharedGU GEMV (global warp stride, verbatim).
//   inputs : [0] hidden bf16[1,H]   [1] rms_w bf16[H]   [2] router_w bf16
//            [3] bias f32[256]
//            [4] a_fp8 u8[H] (hidden, t0 artifact)  [5] a_scale f32[56] (t0)
//            [6] rmsnorm_out bf16[1,H] (t0)         [7] inter f32[256,4] (t0)
//            [8] logits bf16[256] (t0)              [9] meta i32[24] (t0 —
//                REAL consumer: the downstream w2_silu reads it)
//            [10] w13   [11] w13_scale   [12] wgu   [13] wgu_scale
//   outputs: [0] y13 f32[8,1024]   [1] sg f32[512]
// Flags (A_TK region tail): [0] NORM_READY  [1..3] RDONE  [4] META_READY
//   [5..7] epilogue (mac_task_epilogue base &flags[4]).
// ----------------------------------------------------------------------------
__device__ __noinline__ void
    w13_rqr_topk_task_impl(mirage::runtime::TaskDesc const *task_desc,
                           int task_offset,
                           int num_tasks,
                           int nwarps,
                           unsigned long long sync_tag,
                           int local_expert_start,
                           int num_local_experts,
                           float routed_scaling_factor) {
  __nv_bfloat16 const *x =
      static_cast<__nv_bfloat16 const *>(task_desc->input_ptrs[0]);
  __nv_bfloat16 const *rms_w =
      static_cast<__nv_bfloat16 const *>(task_desc->input_ptrs[1]);
  __nv_bfloat16 const *wr =
      static_cast<__nv_bfloat16 const *>(task_desc->input_ptrs[2]);
  float const *bias = static_cast<float const *>(task_desc->input_ptrs[3]);
  uint8_t *a_fp8_gmem = static_cast<uint8_t *>(task_desc->input_ptrs[4]);
  float *a_scale_gmem = static_cast<float *>(task_desc->input_ptrs[5]);
  __nv_bfloat16 *rmsnorm_out =
      static_cast<__nv_bfloat16 *>(task_desc->input_ptrs[6]);
  float *inter_gmem = static_cast<float *>(task_desc->input_ptrs[7]);
  __nv_bfloat16 *logits_out =
      static_cast<__nv_bfloat16 *>(task_desc->input_ptrs[8]);
  int *meta_gmem = static_cast<int *>(task_desc->input_ptrs[9]);
  uint8_t const *w13 = static_cast<uint8_t const *>(task_desc->input_ptrs[10]);
  float const *w13_scale =
      static_cast<float const *>(task_desc->input_ptrs[11]);
  uint8_t const *wgu = static_cast<uint8_t const *>(task_desc->input_ptrs[12]);
  float const *wgu_s = static_cast<float const *>(task_desc->input_ptrs[13]);
  float *y13 = static_cast<float *>(task_desc->output_ptrs[0]);
  float *sg = static_cast<float *>(task_desc->output_ptrs[1]);

  extern __shared__ char smem[];
  char *nb = smem + task_desc->smem_region_offset(A_REGION_NORM);
  __nv_bfloat16 *s_norm = reinterpret_cast<__nv_bfloat16 *>(nb);
  float *s_red = reinterpret_cast<float *>(nb + RQR_OFF_RED);
  uint8_t *s_a = reinterpret_cast<uint8_t *>(
      smem + task_desc->smem_region_offset(A_REGION_ACT));
  float *s_as = reinterpret_cast<float *>(s_a + W13_ACT_SCALE_OFF);
  uint4 *s_ring = reinterpret_cast<uint4 *>(
      smem + task_desc->smem_region_offset(A_REGION_RING));
  char *tk_base = smem + task_desc->smem_region_offset(A_REGION_TK);
  float *s_inter = reinterpret_cast<float *>(tk_base + A_TK_OFF_INTER);
  char *wk = tk_base + A_TK_OFF_WK;
  int const *s_meta = reinterpret_cast<int const *>(wk + TK_OFF_META);
  uint64_t *s_flags = reinterpret_cast<uint64_t *>(tk_base + A_TK_OFF_FLAGS);

  int const lane = threadIdx.x & 31;
  int const ws = threadIdx.x >> 5;
  bool const is_consumer = threadIdx.x < 128;

  if (is_consumer) {
    ffnv2_rms_stage_and_norm(x, rms_w, s_norm, s_red);
    // helpers may only see NORMED bytes.
    if (sync_tag != 0 && threadIdx.x == 0) {
      ffnv2_flag_store_release(&s_flags[0], sync_tag);
    }
    if (task_offset == 0) {
      ffnv2_publish_norm(s_norm, rmsnorm_out);
    }
  } else {
    ffnv2_flag_wait(&s_flags[0], sync_tag);
    __syncwarp();
  }

  // quant: ALL 56 groups, TASK-LOCAL warp stride, into SMEM (group-local math
  // verbatim; every task computes identical bytes).
  for (int g = ws; g < KG1; g += nwarps) {
    v1k::quant_group_warp<__nv_bfloat16>(s_norm, s_a, s_as, g, lane);
  }
  // router: ALL 1024 (e,sp) pairs, TASK-LOCAL warp stride, into SMEM s_inter.
  // (This is the checked redundancy: the full 3.67 MB router weight per task.)
  uint4 *my_ring = s_ring + (size_t)ws * (GEMV_RING_BYTES_PER_WARP / 16);
  int const total_pairs = ROUTER_N * RKSPLIT;
  for (int t = ws; t < total_pairs; t += nwarps) {
    int const e = t / RKSPLIT, sp = t % RKSPLIT;
    float const acc = v1k::router_partial_cpa<RKSPLIT, 4>(
        s_norm, wr + (size_t)e * v1k::ROUTER_K, sp, lane, my_ring);
    if (lane == 0) {
      s_inter[e * RKSPLIT + sp] = acc;
    }
  }
  v1k::cpasync_wait<0>();
  __syncwarp();

  if (is_consumer) {
    // wait for the helpers' quant+router SMEM writes (whole-warp acquire).
    if (sync_tag != 0) {
      ffnv2_flag_wait(&s_flags[1], sync_tag);
      ffnv2_flag_wait(&s_flags[2], sync_tag);
      ffnv2_flag_wait(&s_flags[3], sync_tag);
    }
    consumer_sync(); // converge consumers; orders all quant/router writes
    // task-0 artifacts: a_fp8 / a_scale / inter (compare + debug surface).
    if (task_offset == 0) {
      uint4 const *sa4 = reinterpret_cast<uint4 const *>(s_a);
      uint4 *ga4 = reinterpret_cast<uint4 *>(a_fp8_gmem);
      constexpr int NU4_A = HIDDEN / 16; // 448
      for (int u = threadIdx.x; u < NU4_A; u += 128) {
        ga4[u] = sa4[u];
      }
      for (int i = threadIdx.x; i < KG1; i += 128) {
        a_scale_gmem[i] = s_as[i];
      }
      float4 const *si4 = reinterpret_cast<float4 const *>(s_inter);
      float4 *gi4 = reinterpret_cast<float4 *>(inter_gmem);
      for (int u = threadIdx.x; u < ROUTER_N; u += 128) { // 1024 f32 = 256 f4
        if (u < (ROUTER_N * RKSPLIT) / 4) {
          gi4[u] = si4[u];
        }
      }
    }
    // redundant per-task topk on the task's own SMEM logits partials.
    topk_compute(wk, s_inter, bias,
                 task_offset == 0 ? logits_out : nullptr,
                 task_offset == 0 ? meta_gmem : nullptr,
                 local_expert_start, num_local_experts,
                 routed_scaling_factor);
    if (sync_tag != 0 && threadIdx.x == 0) {
      ffnv2_flag_store_release(&s_flags[4], sync_tag); // META_READY
    }
  } else {
    if (lane == 0) {
      ffnv2_flag_store_release(&s_flags[1 + (ws - 4)], sync_tag); // RDONE
    }
    ffnv2_flag_wait(&s_flags[4], sync_tag); // META_READY
    __syncwarp();
  }

  RoutingMeta const m = load_meta(s_meta);

  // W13 + sharedGU GEMV — verbatim w13_topk (global warp stride); the
  // activation is ALREADY in SMEM (computed by the in-task quant).
  constexpr int RBX_W13 = 8;
  constexpr int ST_W13 = 4;
  constexpr int RBX_SH = 4;
  constexpr int ST_SH13 = 2;
  int const n13 = m.active_count * (W13_N / RBX_W13);
  int const nsh1 = SH_GU_N / RBX_SH;
  int const ntot1 = n13 + nsh1;

  for (int idx = task_offset * nwarps + ws; idx < ntot1;
       idx += num_tasks * nwarps) {
    if (idx < n13) {
      int const slot = idx / (W13_N / RBX_W13);
      int const n0 = (idx % (W13_N / RBX_W13)) * RBX_W13;
      int const e = m.experts[slot];
      uint8_t const *wb = w13 + (size_t)e * W13_N * HIDDEN;
      float const *wsc = w13_scale + (size_t)e * v1k::NB1 * KG1 +
                         (size_t)(n0 / GRP) * KG1;
      float yb[RBX_W13];
      v1k::dgemv_cpa16_h2<RBX_W13, ST_W13>(
          s_a, s_as, wb, wsc, HIDDEN, KG1, n0, lane, my_ring, yb);
      if (lane == 0) {
#pragma unroll
        for (int r = 0; r < RBX_W13; r++) {
          y13[(size_t)slot * W13_N + n0 + r] = yb[r];
        }
      }
    } else {
      int const n0 = (idx - n13) * RBX_SH;
      float const *wsc = wgu_s + (size_t)(n0 / GRP) * v1k::KG_SHGU;
      float yb[RBX_SH];
      v1k::dgemv_cpa16_h2<RBX_SH, ST_SH13>(
          s_a, s_as, wgu, wsc, v1k::SH_GU_K, v1k::KG_SHGU, n0, lane, my_ring,
          yb);
      if (lane == 0) {
#pragma unroll
        for (int r = 0; r < RBX_SH; r++) {
          sg[n0 + r] = yb[r];
        }
      }
    }
  }

  mac_task_epilogue(is_consumer, &s_flags[4], sync_tag); // uses [5..7]
}

// ----------------------------------------------------------------------------
// Rung B — ffn_mega: the whole FFN slice as ONE op whose 136 tasks (MUST equal
// num_workers — asserted host-side; 2 same-op tasks serialized on one worker
// would deadlock) self-synchronize around the two in-op all-to-alls (router
// inter, W13 y13/sg) via monotonic-count GMEM barriers — v1's grid-barrier
// algorithm rebuilt across co-resident v2 tasks. The router stays
// GRID-strided (no Rung-A redundancy).
//   inputs : [0] hidden   [1] rms_w   [2] router_w   [3] bias
//            [4] w13   [5] wgu   [6] w2   [7] wdn
//            [8] scales pack f32 (MEGA_SC_* offsets)
//            [9] xfer pack f32 (MEGA_XFER_*: inter | y13 | sg)
//            [10] bar u64[2] (zeroed at alloc; target NT*(iter_num+1))
//            [11] artifacts pack u8 (MEGA_ART_*: task-0 compare surface)
//   outputs: [0] out bf16[1,W2_N]
// Flags (M_TK tail, u64[16]): [0] NORM_READY  [1..3] PH1  [4] GO1
//   [5] META_READY  [6..8] PH2  [9] GO2  [10] SILU_READY  [11+] epilogue base.
// ----------------------------------------------------------------------------
template <int RBLK>
__device__ __noinline__ void
    ffn_mega_task_impl(mirage::runtime::TaskDesc const *task_desc,
                       int task_offset,
                       int num_tasks,
                       int nwarps,
                       unsigned long long sync_tag,
                       int local_expert_start,
                       int num_local_experts,
                       float routed_scaling_factor,
                       int iter_num) {
  static_assert(RBLK == 16 || RBLK == 8, "RBLK must divide GRP and be >=4");
  __nv_bfloat16 const *x =
      static_cast<__nv_bfloat16 const *>(task_desc->input_ptrs[0]);
  __nv_bfloat16 const *rms_w =
      static_cast<__nv_bfloat16 const *>(task_desc->input_ptrs[1]);
  __nv_bfloat16 const *wr =
      static_cast<__nv_bfloat16 const *>(task_desc->input_ptrs[2]);
  float const *bias = static_cast<float const *>(task_desc->input_ptrs[3]);
  uint8_t const *w13 = static_cast<uint8_t const *>(task_desc->input_ptrs[4]);
  uint8_t const *wgu = static_cast<uint8_t const *>(task_desc->input_ptrs[5]);
  uint8_t const *w2 = static_cast<uint8_t const *>(task_desc->input_ptrs[6]);
  uint8_t const *wdn = static_cast<uint8_t const *>(task_desc->input_ptrs[7]);
  float const *sc = static_cast<float const *>(task_desc->input_ptrs[8]);
  float const *w13_scale = sc + MEGA_SC_OFF_W13;
  float const *wgu_s = sc + MEGA_SC_OFF_WGU;
  float const *w2s = sc + MEGA_SC_OFF_W2;
  float const *wdns = sc + MEGA_SC_OFF_WDN;
  float *xfer = static_cast<float *>(task_desc->input_ptrs[9]);
  float *g_inter = xfer + MEGA_XFER_OFF_INTER_F;
  float *g_y13 = xfer + MEGA_XFER_OFF_Y13_F;
  float *g_sg = xfer + MEGA_XFER_OFF_SG_F;
  unsigned long long *bar =
      static_cast<unsigned long long *>(task_desc->input_ptrs[10]);
  uint8_t *art = static_cast<uint8_t *>(task_desc->input_ptrs[11]);
  __nv_bfloat16 *out = static_cast<__nv_bfloat16 *>(task_desc->output_ptrs[0]);

  extern __shared__ char smem[];
  char *nb = smem + task_desc->smem_region_offset(M_REGION_NORM);
  __nv_bfloat16 *s_norm = reinterpret_cast<__nv_bfloat16 *>(nb);
  float *s_red = reinterpret_cast<float *>(nb + RQR_OFF_RED);
  uint8_t *s_a = reinterpret_cast<uint8_t *>(
      smem + task_desc->smem_region_offset(M_REGION_ACT));
  float *s_as = reinterpret_cast<float *>(s_a + W13_ACT_SCALE_OFF);
  uint4 *s_ring = reinterpret_cast<uint4 *>(
      smem + task_desc->smem_region_offset(M_REGION_RING));
  char *tk_base = smem + task_desc->smem_region_offset(M_REGION_TK);
  char *wk = tk_base + M_TK_OFF_WK;
  int const *s_meta = reinterpret_cast<int const *>(wk + TK_OFF_META);
  uint64_t *s_flags = reinterpret_cast<uint64_t *>(tk_base + M_TK_OFF_FLAGS);
  uint8_t *s_act = reinterpret_cast<uint8_t *>(
      smem + task_desc->smem_region_offset(M_REGION_W2ACT));
  uint8_t *s_ifp8 = s_act;
  float *s_iscale = reinterpret_cast<float *>(s_act + W2_ACT_ISCALE_OFF);
  uint8_t *s_sifp8 = s_act + W2_ACT_SIFP8_OFF;
  float *s_siscale = reinterpret_cast<float *>(s_act + W2_ACT_SISCALE_OFF);
  // y13/sg staging views over the ring (silu phase only; consumer slices).
  float *s_y13 = reinterpret_cast<float *>(
      reinterpret_cast<char *>(s_ring) + W2S_RING_Y13_OFF);
  float *s_sg = reinterpret_cast<float *>(
      reinterpret_cast<char *>(s_ring) + W2S_RING_SG_OFF);

  int const lane = threadIdx.x & 31;
  int const ws = threadIdx.x >> 5;
  bool const is_consumer = threadIdx.x < 128;
  bool const has_helpers = nwarps > 4;
  unsigned long long const bar_need =
      (unsigned long long)num_tasks * (unsigned long long)(iter_num + 1);

  // ---- P0: rmsnorm (redundant, consumers) ---------------------------------
  if (is_consumer) {
    ffnv2_rms_stage_and_norm(x, rms_w, s_norm, s_red);
    if (has_helpers && threadIdx.x == 0) {
      ffnv2_flag_store_release(&s_flags[0], sync_tag); // NORM_READY
    }
    if (task_offset == 0) {
      ffnv2_publish_norm(
          s_norm,
          reinterpret_cast<__nv_bfloat16 *>(art + MEGA_ART_OFF_RMSNORM));
    }
  } else {
    ffnv2_flag_wait(&s_flags[0], sync_tag);
    __syncwarp();
  }

  // ---- P1: quant (task-local, ALL warps) + router slice (GRID stride) -----
  for (int g = ws; g < KG1; g += nwarps) {
    v1k::quant_group_warp<__nv_bfloat16>(s_norm, s_a, s_as, g, lane);
  }
  uint4 *my_ring = s_ring + (size_t)ws * (GEMV_RING_BYTES_PER_WARP / 16);
  int const total_pairs = ROUTER_N * RKSPLIT;
  for (int t = task_offset * nwarps + ws; t < total_pairs;
       t += num_tasks * nwarps) {
    int const e = t / RKSPLIT, sp = t % RKSPLIT;
    float const acc = v1k::router_partial_cpa<RKSPLIT, 4>(
        s_norm, wr + (size_t)e * v1k::ROUTER_K, sp, lane, my_ring);
    if (lane == 0) {
      g_inter[e * RKSPLIT + sp] = acc;
    }
  }
  v1k::cpasync_wait<0>();
  __syncwarp();

  // ---- GMEM BARRIER 1 (the inter all-to-all) ------------------------------
  if (is_consumer) {
    consumer_sync(); // all consumer router/quant work done + visible cta-scope
    if (threadIdx.x == 0) {
      if (has_helpers) {
        ffnv2_flag_poll(&s_flags[1], sync_tag); // PH1: helper stores done
        ffnv2_flag_poll(&s_flags[2], sync_tag);
        ffnv2_flag_poll(&s_flags[3], sync_tag);
      }
      __threadfence(); // make the whole CTA's inter stores gpu-visible
      atom_add_release_gpu_u64(&bar[0], 1ull);
      while (ld_acquire_sys_u64(&bar[0]) < bar_need) {
        __nanosleep(64);
      }
      ffnv2_flag_store_release(&s_flags[4], sync_tag); // GO1
    }
    if (ws == 0) {
      __syncwarp();
    } else {
      ffnv2_flag_wait(&s_flags[4], sync_tag);
    }
  } else {
    if (lane == 0) {
      ffnv2_flag_store_release(&s_flags[1 + (ws - 4)], sync_tag); // PH1
    }
    ffnv2_flag_wait(&s_flags[4], sync_tag); // GO1
  }

  // ---- P2: redundant topk (consumers) + W13 slice (GRID stride) -----------
  if (is_consumer) {
    if (task_offset == 0) { // a_fp8/a_scale artifacts (quant done since GO1)
      uint4 const *sa4 = reinterpret_cast<uint4 const *>(s_a);
      uint4 *ga4 = reinterpret_cast<uint4 *>(art + MEGA_ART_OFF_AFP8);
      constexpr int NU4_A = HIDDEN / 16;
      for (int u = threadIdx.x; u < NU4_A; u += 128) {
        ga4[u] = sa4[u];
      }
      float *gas = reinterpret_cast<float *>(art + MEGA_ART_OFF_ASCALE);
      for (int i = threadIdx.x; i < KG1; i += 128) {
        gas[i] = s_as[i];
      }
    }
    topk_compute(
        wk, g_inter, bias,
        task_offset == 0
            ? reinterpret_cast<__nv_bfloat16 *>(art + MEGA_ART_OFF_LOGITS)
            : nullptr,
        task_offset == 0 ? reinterpret_cast<int *>(art + MEGA_ART_OFF_META)
                         : nullptr,
        local_expert_start, num_local_experts, routed_scaling_factor);
    if (has_helpers && threadIdx.x == 0) {
      ffnv2_flag_store_release(&s_flags[5], sync_tag); // META_READY
    }
  } else {
    ffnv2_flag_wait(&s_flags[5], sync_tag); // META_READY
    __syncwarp();
  }

  RoutingMeta const m = load_meta(s_meta);

  {
    constexpr int RBX_W13 = 8;
    constexpr int ST_W13 = 4;
    constexpr int RBX_SH = 4;
    constexpr int ST_SH13 = 2;
    int const n13 = m.active_count * (W13_N / RBX_W13);
    int const nsh1 = SH_GU_N / RBX_SH;
    int const ntot1 = n13 + nsh1;
    for (int idx = task_offset * nwarps + ws; idx < ntot1;
         idx += num_tasks * nwarps) {
      if (idx < n13) {
        int const slot = idx / (W13_N / RBX_W13);
        int const n0 = (idx % (W13_N / RBX_W13)) * RBX_W13;
        int const e = m.experts[slot];
        uint8_t const *wb = w13 + (size_t)e * W13_N * HIDDEN;
        float const *wsc = w13_scale + (size_t)e * v1k::NB1 * KG1 +
                           (size_t)(n0 / GRP) * KG1;
        float yb[RBX_W13];
        v1k::dgemv_cpa16_h2<RBX_W13, ST_W13>(
            s_a, s_as, wb, wsc, HIDDEN, KG1, n0, lane, my_ring, yb);
        if (lane == 0) {
#pragma unroll
          for (int r = 0; r < RBX_W13; r++) {
            g_y13[(size_t)slot * W13_N + n0 + r] = yb[r];
          }
        }
      } else {
        int const n0 = (idx - n13) * RBX_SH;
        float const *wsc = wgu_s + (size_t)(n0 / GRP) * v1k::KG_SHGU;
        float yb[RBX_SH];
        v1k::dgemv_cpa16_h2<RBX_SH, ST_SH13>(
            s_a, s_as, wgu, wsc, v1k::SH_GU_K, v1k::KG_SHGU, n0, lane,
            my_ring, yb);
        if (lane == 0) {
#pragma unroll
          for (int r = 0; r < RBX_SH; r++) {
            g_sg[n0 + r] = yb[r];
          }
        }
      }
    }
  }
  v1k::cpasync_wait<0>();
  __syncwarp();

  // ---- GMEM BARRIER 2 (the y13/sg all-to-all == the i_fp8 boundary) -------
  if (is_consumer) {
    consumer_sync();
    if (threadIdx.x == 0) {
      if (has_helpers) {
        ffnv2_flag_poll(&s_flags[6], sync_tag); // PH2
        ffnv2_flag_poll(&s_flags[7], sync_tag);
        ffnv2_flag_poll(&s_flags[8], sync_tag);
      }
      __threadfence();
      atom_add_release_gpu_u64(&bar[1], 1ull);
      while (ld_acquire_sys_u64(&bar[1]) < bar_need) {
        __nanosleep(64);
      }
      ffnv2_flag_store_release(&s_flags[9], sync_tag); // GO2
    }
    if (ws == 0) {
      __syncwarp();
    } else {
      ffnv2_flag_wait(&s_flags[9], sync_tag);
    }
  } else {
    if (lane == 0) {
      ffnv2_flag_store_release(&s_flags[6 + (ws - 4)], sync_tag); // PH2
    }
    ffnv2_flag_wait(&s_flags[9], sync_tag); // GO2
  }

  // ---- P3: redundant silu+requant (consumers, verbatim w2_silu) -----------
  if (is_consumer) {
    uint32_t const sb =
        static_cast<uint32_t>(__cvta_generic_to_shared(s_y13));
    uint4 const *y4 = reinterpret_cast<uint4 const *>(g_y13);
    int const nu4_y = (m.active_count * W13_N) >> 2;
    for (int u = threadIdx.x; u < nu4_y; u += 128) {
      v1k::cpasync16(sb + (uint32_t)u * 16, &y4[u]);
    }
    uint32_t const sbs =
        static_cast<uint32_t>(__cvta_generic_to_shared(s_sg));
    uint4 const *g4 = reinterpret_cast<uint4 const *>(g_sg);
    constexpr int NU4_SG = SH_GU_N / 4;
    for (int u = threadIdx.x; u < NU4_SG; u += 128) {
      v1k::cpasync16(sbs + (uint32_t)u * 16, &g4[u]);
    }
    v1k::cpasync_commit();
    v1k::cpasync_wait<0>();
    consumer_sync();

    int const wid = ws; // 0..3
    int const ng = m.active_count * KG2;
    for (int gg = wid; gg < ng; gg += 4) {
      int const slot = gg / KG2;
      int const g = gg % KG2;
      float const *y = s_y13 + (size_t)slot * W13_N;
      int const i0 = g * GRP + lane * 4;
      float4 const gpart = *reinterpret_cast<float4 const *>(&y[i0]);
      float4 const upart = *reinterpret_cast<float4 const *>(&y[512 + i0]);
      float v[4], amax = 0.f;
      v[0] = v1k::silu_fast(gpart.x) * upart.x;
      v[1] = v1k::silu_fast(gpart.y) * upart.y;
      v[2] = v1k::silu_fast(gpart.z) * upart.z;
      v[3] = v1k::silu_fast(gpart.w) * upart.w;
#pragma unroll
      for (int t = 0; t < 4; t++) {
        amax = fmaxf(amax, fabsf(v[t]));
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
      }
      float const s = v1k::quant_scale(amax);
      float const inv = 1.f / s;
      if (lane == 0) {
        s_iscale[slot * KG2 + g] = s;
      }
#pragma unroll
      for (int t = 0; t < 4; t++) {
        s_ifp8[(size_t)slot * W2_K + i0 + t] = v1k::to_f8(v[t] * inv);
      }
    }
    for (int g = wid; g < KG_SHDN; g += 4) {
      float v[4], amax = 0.f;
#pragma unroll
      for (int t = 0; t < 4; t++) {
        int const i = g * GRP + lane * 4 + t;
        float const val = v1k::silu_fast(s_sg[i]) * s_sg[256 + i];
        v[t] = val;
        amax = fmaxf(amax, fabsf(val));
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
      }
      float const s = v1k::quant_scale(amax);
      float const inv = 1.f / s;
      if (lane == 0) {
        s_siscale[g] = s;
      }
#pragma unroll
      for (int t = 0; t < 4; t++) {
        int const i = g * GRP + lane * 4 + t;
        s_sifp8[i] = v1k::to_f8(v[t] * inv);
      }
    }
    // ALL silu reads of the ring-staged y13/sg complete before this barrier;
    // the GEMV below may then reuse the ring for weight staging.
    consumer_sync();
    if (has_helpers && threadIdx.x == 0) {
      ffnv2_flag_store_release(&s_flags[10], sync_tag); // SILU_READY
    }
    if (task_offset == 0) {
      uint8_t *gi = art + MEGA_ART_OFF_IFP8;
      for (int i = threadIdx.x; i < m.active_count * W2_K; i += 128) {
        gi[i] = s_ifp8[i];
      }
      float *gis = reinterpret_cast<float *>(art + MEGA_ART_OFF_ISCALE);
      for (int i = threadIdx.x; i < m.active_count * KG2; i += 128) {
        gis[i] = s_iscale[i];
      }
      uint8_t *gsi = art + MEGA_ART_OFF_SIFP8;
      for (int i = threadIdx.x; i < SH_DN_K; i += 128) {
        gsi[i] = s_sifp8[i];
      }
      float *gss = reinterpret_cast<float *>(art + MEGA_ART_OFF_SISCALE);
      for (int i = threadIdx.x; i < KG_SHDN; i += 128) {
        gss[i] = s_siscale[i];
      }
    }
  } else {
    ffnv2_flag_wait(&s_flags[10], sync_tag); // SILU_READY
    __syncwarp();
  }

  // ---- W2 + sharedDN, output-stationary (verbatim w2_silu GEMV) -----------
  {
    constexpr int ST_W2 = 2;
    constexpr int RBX_SH = 4;
    constexpr int ST_SH2 = 3;
    int const nblk = W2_N / RBLK;
    for (int item = task_offset * nwarps + ws; item < nblk;
         item += num_tasks * nwarps) {
      int const n0 = item * RBLK;
      float acc[RBLK];
#pragma unroll
      for (int r = 0; r < RBLK; r++) {
        acc[r] = 0.f;
      }
      for (int slot = 0; slot < m.active_count; slot++) {
        int const e = m.experts[slot];
        float const ew = m.weights[slot];
        float yb[RBLK];
        v1k::dgemv_cpa16_h2<RBLK, ST_W2>(
            s_ifp8 + (size_t)slot * W2_K,
            s_iscale + slot * KG2,
            w2 + (size_t)e * W2_N * W2_K,
            w2s + (size_t)e * v1k::NB2 * KG2 + (size_t)(n0 / GRP) * KG2,
            W2_K, KG2, n0, lane, my_ring, yb);
        if (lane == 0) {
#pragma unroll
          for (int r = 0; r < RBLK; r++) {
            acc[r] += ew * yb[r];
          }
        }
      }
#pragma unroll
      for (int sb4 = 0; sb4 < RBLK / RBX_SH; sb4++) {
        int const mm0 = n0 + sb4 * RBX_SH;
        float yb4[RBX_SH];
        v1k::dgemv_cpa<RBX_SH, ST_SH2>(
            s_sifp8, s_siscale, wdn,
            wdns + (size_t)(mm0 / GRP) * KG_SHDN,
            SH_DN_K, KG_SHDN, mm0, lane,
            reinterpret_cast<uint32_t *>(my_ring), yb4);
        if (lane == 0) {
#pragma unroll
          for (int r = 0; r < RBX_SH; r++) {
            acc[sb4 * RBX_SH + r] += yb4[r];
          }
        }
      }
      if (lane == 0) {
#pragma unroll
        for (int r = 0; r < RBLK; r++) {
          out[n0 + r] = __float2bfloat16_rn(acc[r]);
        }
      }
    }
  }

  mac_task_epilogue(is_consumer, &s_flags[11],
                    has_helpers ? sync_tag : 0ull); // uses [12..14]
}

} // namespace dsv3_ffn_v2
} // namespace kernel
