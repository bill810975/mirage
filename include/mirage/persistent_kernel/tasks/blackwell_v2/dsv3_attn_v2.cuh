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
// DSv3 fused decode-ATTENTION block in Runtime-V2 format (Step 3b of the V2
// migration; follows the Step-3a FFN recipe).
//
// The v1 whole-grid task (tasks/blackwell/attn_block_megakernel_sm100.cuh,
// 136-CTA lockstep, 3 grid barriers + 2 device-scope handoffs) is re-expressed
// as a CHAIN of v2 per-SM tasks; every v1 cross-CTA sync point becomes a v2
// cross-task event dependency, every v1 grid-strided phase becomes an N-task
// op, and every v1 "block-local redundant to avoid a barrier" phase (levers
// 1-3) becomes a task-local redundant prologue:
//
//   dsv3_attn_p0_qkva_v2   (N1)  P0 rmsnorm+UE8M0 quant (redundant/task)
//                                + qkv_a GEMV -> g_qkva f32[2176]
//        v1 barrier B1 (qkv_a -> layernorm)      => T1->T2 event
//   dsv3_attn_qb_rope_kv_v2 (N2) q_a-ln+requant (redundant/task) + q_b GEMV
//                                + fused q-rope -> g_qpe f32[16*576];
//                                task 0: kv_a-ln + rope(k_pe) -> kv_cache[step]
//        v1 barrier B2 (q_b -> MLA)              => T2->T3 event
//   dsv3_attn_mla_partial_v2 (128) (h,sp) flash partial -> g_mla_acc/m/l
//        v1 lever-4 atomic-merge handoff         => T3->T4 event
//   dsv3_attn_mla_merge_v2 (16)  per-head merge + NON-UE8M0 448-quant
//                                -> g_attn (hidden) + g_attn_deq f32[16*512]
//        v1 lever-5 wuv spin-wait handoff        => T4->T5 event
//   dsv3_attn_wuv_v2       (N5)  W_UV per-head BMM -> g_red f32[2048]
//        v1 barrier B3 (W_UV -> o_proj)          => T5->T6 event
//   dsv3_attn_oproj_v2     (N6)  UE8M0 quant g_red (redundant/task) +
//                                o_proj GEMV + residual -> out bf16[1,7168]
//
// VALUE FIDELITY (bit-exact vs v1 given identical input bytes):
//   - The GEMV/BMM bodies are the v1 device functions CALLED VERBATIM
//     (gemv_grid_cpa_t<2,6>, gemv_grid_cpa_qb_rope_smem_t<8,4>,
//     gemv_grid_cpa_oproj_smem_t<8,4>, wuv_bmm_grid): they contain no
//     __syncthreads (only __syncwarp) and their per-row accumulation order
//     depends only on (n, K) — not on which warp/task runs the row.
//   - Per-group quant / layernorm-requant / rope / kv-ln loops are v1's
//     bodies with only the outer warp stride remapped 8 -> 4 warps
//     (group-local warp math => values identical).
//   - v1's 256-thread block reductions are EXACT-TREE emulated at 128
//     threads: v2 thread t plays v1 threads t (role A) and t+128 (role B).
//     Since lane(t+128)==lane(t) and warp(t+128)==warp(t)+4, every v1
//     warp-local shuffle is reproduced by doing A's and B's shuffle
//     separately in the same lanes, then storing red8[w]=A-warp-w and
//     red8[w+4]=B-warp-w — after which v1's cross-warp combines (sequential
//     sum/max over red8[0..8) or the warp-0 xor tree) are verbatim.
//   - mla_partial keeps v1's TPR computed from NTHREAD=256 (the v1 constant);
//     TPR in {1,2,4,8} divides 128, so role B has subB==subA and
//     rowB=rowA+128/TPR — the masked group shfl_down patterns are preserved.
//   - wuv_bmm_grid's per-head acquire-spin is satisfied immediately by a
//     constant ones_i32[16] buffer bound as an input (the REAL g_attn_deq
//     handoff is the T4->T5 event dep) so the v1 function needs no edit.
//
// Every __syncthreads in ported block-collective code becomes the consumer
// named barrier `bar.sync 4, 128` (ids 1/2/3 are linear/rmsnorm/ffn). Every
// SMEM-using task ends with an UNCONDITIONAL epilogue (cp.async drain +
// consumer barrier) so warps 1-3 finish all SMEM before warp 0 runs the
// codegen auto page-release suffix (design-review BLOCKER fix; 3a
// mac_task_epilogue pattern).
// ============================================================================

#include "mirage/persistent_kernel/runtime_header.h"
#include "mirage/persistent_kernel/tasks/blackwell/attn_block_megakernel_sm100.cuh"
#include "mirage/persistent_kernel/tasks/blackwell_v2/dsv3_attn_v2_spec.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>
#include <stdint.h>

namespace kernel {
namespace dsv3_attn_v2 {

namespace v1a = ::kernel::attn_block_megakernel_sm100;

// Pin the spec-header shapes to the v1 kernel's macros so drift is a compile
// error. (K_* / NTHREAD / NWARP / MLA_SPLITS are #defines from the v1 header.)
static_assert(HIDDEN == K_HIDDEN && QLORA == K_QLORA && KVLORA == K_KVLORA &&
                  QKROPE == K_QKROPE && QKHEAD == K_QKHEAD &&
                  VHEAD == K_VHEAD && QKVAN == K_QKVAN &&
                  HLOCAL == K_HLOCAL && OIN == K_OIN && GRP == K_GRP &&
                  SPLITS == MLA_SPLITS,
              "dsv3_attn_v2_spec.h shapes drifted from the v1 kernel");
static_assert(NTHREAD == 256 && NWARP == 8,
              "the 128-thread exact-tree emulation assumes v1 NTHREAD=256/"
              "NWARP=8");

// Consumer-warp named barrier (threads 0-127). 1=linear_v2, 2=rmsnorm_v2,
// 3=dsv3_ffn_v2; use 4 here.
__device__ __forceinline__ void attn_consumer_sync() {
  asm volatile("bar.sync 4, 128;");
}

// Unconditional task epilogue (design-review BLOCKER fix): drain this warp's
// cp.async ring (no-op when the task issued none), then hold warp 0 (the
// page-release lanes) until warps 1-3 arrived. Must run on EVERY path.
__device__ __forceinline__ void attn_task_epilogue() {
  v1a::k_cpa_wait<0>();
  __syncwarp();
  attn_consumer_sync();
}

// ============================================================================
// EXACT-TREE 128-thread emulation of v1's rms_rcp_block
// (attn_block_megakernel_sm100.cuh: per-thread partial with stride
// NTHREAD=256, per-warp shfl_DOWN tree, red8[w] at lane 0, then a REDUNDANT
// per-thread sequential sum over red8[0..8), 1.0f/sqrtf).
// v2 thread t plays v1 threads t (A) and t+128 (B); red8[w]=A-warp-w,
// red8[w+4]=B-warp-w reproduces v1's 8-slot layout exactly.
// ============================================================================
__device__ __forceinline__ float
    rms_rcp_block_128emu(float const *__restrict__ src,
                         int n,
                         float *__restrict__ red8) {
  int tid = threadIdx.x, lane = tid & 31, warpl = tid >> 5;
  float psA = 0.f, psB = 0.f;
  for (int i = tid; i < n; i += NTHREAD) {
    float x = src[i];
    psA += x * x;
  }
  for (int i = tid + 128; i < n; i += NTHREAD) {
    float x = src[i];
    psB += x * x;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    psA += __shfl_down_sync(0xffffffffu, psA, o);
    psB += __shfl_down_sync(0xffffffffu, psB, o);
  }
  if (lane == 0) {
    red8[warpl] = psA;
    red8[warpl + 4] = psB;
  }
  attn_consumer_sync();
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < NWARP; i++) {
    ss += red8[i];
  }
  attn_consumer_sync();
  return 1.0f / sqrtf(ss / n + K_EPS);
}

// ============================================================================
// T1 — p0_qkva. v1 Phase-0 (rmsnorm_quant_hidden_block_smem: fused RMSNorm +
// per-128-group UE8M0 quant of the normed value, redundant per CTA) + v1 S2
// (qkv_a GEMV, grid-strided).
//   inputs : [0] x bf16[1,HIDDEN] (RAW residual stream — the chain edge in)
//            [1] input_ln_w bf16[HIDDEN]
//            [2] qkv_a_w fp8[QKVAN,HIDDEN]  [3] qkv_a_s f32[17,56]
//   outputs: [0] g_qkva f32[QKVAN]  (the T1->T2 chain edge)
// ============================================================================
__device__ __noinline__ void
    p0_qkva_task_impl(mirage::runtime::TaskDesc const *task_desc,
                      int task_offset,
                      int num_tasks) {
  __nv_bfloat16 const *x =
      static_cast<__nv_bfloat16 const *>(task_desc->input_ptrs[0]);
  __nv_bfloat16 const *input_ln_w =
      static_cast<__nv_bfloat16 const *>(task_desc->input_ptrs[1]);
  __nv_fp8_e4m3 const *qkv_a_w =
      static_cast<__nv_fp8_e4m3 const *>(task_desc->input_ptrs[2]);
  float const *qkv_a_s = static_cast<float const *>(task_desc->input_ptrs[3]);
  float *g_qkva = static_cast<float *>(task_desc->output_ptrs[0]);

  extern __shared__ char smem[];
  float *s_act = reinterpret_cast<float *>(
      smem + task_desc->smem_region_offset(P0_REGION_WORK));
  float *red8 = reinterpret_cast<float *>(
      reinterpret_cast<char *>(s_act) + P0_RED_OFF);
  uint4 *s_ring = reinterpret_cast<uint4 *>(
      smem + task_desc->smem_region_offset(P0_REGION_RING));

  int tid = threadIdx.x, lane = tid & 31, warpl = tid >> 5;

  // ---- Phase-0 RMSNorm reduction: EXACT-TREE port of
  // rmsnorm_quant_hidden_block_smem's xor-tree variant (per-thread partial
  // stride 256, per-warp shfl_XOR, warp-0 xor tree over red8[0..8),
  // broadcast via red8[0], rsqrtf). A/B emulation as described above.
  float psA = 0.f, psB = 0.f;
  for (int i = tid; i < K_HIDDEN; i += NTHREAD) {
    float v = __bfloat162float(x[i]);
    psA += v * v;
  }
  for (int i = tid + 128; i < K_HIDDEN; i += NTHREAD) {
    float v = __bfloat162float(x[i]);
    psB += v * v;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    psA += __shfl_xor_sync(0xffffffffu, psA, o);
    psB += __shfl_xor_sync(0xffffffffu, psB, o);
  }
  if (lane == 0) {
    red8[warpl] = psA;
    red8[warpl + 4] = psB;
  }
  attn_consumer_sync();
  // v1: warp-0 xor tree over the NWARP partials (lanes 0-7 hold red8[0..8);
  // xor with o=4,2,1 stays inside the 8-aligned lane group), broadcast
  // through red8[0].
  float ss = (tid < NWARP) ? red8[tid] : 0.f;
#pragma unroll
  for (int o = NWARP / 2; o > 0; o >>= 1) {
    ss += __shfl_xor_sync(0xffffffffu, ss, o);
  }
  if (tid == 0) {
    red8[0] = ss;
  }
  attn_consumer_sync();
  ss = red8[0];
  attn_consumer_sync(); // re-converge before red8 is reused later
  float rms_rcp = rsqrtf(ss / (float)K_HIDDEN + K_EPS);

  // ---- UE8M0 per-128-group quant of the NORMED value into s_act (v1 body
  // verbatim; outer warp stride 8 -> 4; group math is warp-local).
  {
    int ng = K_HIDDEN / K_GRP;
    for (int gx = warpl; gx < ng; gx += 4) {
      __nv_bfloat16 const *h = x + gx * K_GRP;
      __nv_bfloat16 const *w = input_ln_w + gx * K_GRP;
      float v[4];
      float mx = 1e-10f;
      __nv_bfloat162 const *h2 =
          reinterpret_cast<__nv_bfloat162 const *>(h) + lane * 2;
      __nv_bfloat162 const *w2 =
          reinterpret_cast<__nv_bfloat162 const *>(w) + lane * 2;
      float2 a0 = __bfloat1622float2(h2[0]);
      float2 a1 = __bfloat1622float2(h2[1]);
      float2 g0 = __bfloat1622float2(w2[0]);
      float2 g1 = __bfloat1622float2(w2[1]);
      v[0] = __bfloat162float(__float2bfloat16(a0.x * (rms_rcp * g0.x)));
      v[1] = __bfloat162float(__float2bfloat16(a0.y * (rms_rcp * g0.y)));
      v[2] = __bfloat162float(__float2bfloat16(a1.x * (rms_rcp * g1.x)));
      v[3] = __bfloat162float(__float2bfloat16(a1.y * (rms_rcp * g1.y)));
#pragma unroll
      for (int t = 0; t < 4; t++) {
        mx = fmaxf(mx, fabsf(v[t]));
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
      }
      float ys = fmaxf(mx / K_FP8MAX, 1e-10f);
      float yq = v1a::k_dec_ue8m0(v1a::k_enc_ue8m0(ys));
      float *d = s_act + gx * K_GRP + lane * 4;
#pragma unroll
      for (int t = 0; t < 4; t++) {
        float q = fminf(fmaxf(v[t] / yq, -K_FP8MAX), K_FP8MAX);
        d[t] = (float)__nv_fp8_e4m3(q) * yq;
      }
    }
  }
  attn_consumer_sync(); // publishes s_act (v1's trailing __syncthreads)

  // ---- qkv_a GEMV (VERBATIM v1 template; activation from task-local SMEM,
  // exactly as v1 lever 1 passes s_act).
  uint4 *my_ring = s_ring + (size_t)warpl * (P0_RING_BYTES_PER_WARP / 16);
  v1a::gemv_grid_cpa_t<2, 6>(s_act,
                             qkv_a_w,
                             qkv_a_s,
                             g_qkva,
                             K_QKVAN,
                             K_HIDDEN,
                             task_offset * 4 + warpl,
                             num_tasks * 4,
                             lane,
                             my_ring);
  attn_task_epilogue();
}

// ============================================================================
// T2 — qb_rope_kv. v1 S3 (q_a-ln + requant, redundant per CTA into SMEM) +
// v1 S4+S6 (q_b GEMV + fused q-rope, grid-strided) + — task 0 only — v1 S5
// (kv_a-ln restriped from v1's grid-strided 512-elem loop + the literal
// worker-0/tid<32 rope(k_pe) tail), writing kv_cache[step] (hidden write).
//   inputs : [0] g_qkva f32[QKVAN] (chain edge in)
//            [1] q_a_ln_w bf16[QLORA]   [2] kv_a_ln_w bf16[KVLORA]
//            [3] q_b_w fp8[HLOCAL*QKHEAD,QLORA]  [4] q_b_s f32[72,12]
//            [5] cos_sin bf16[max_pos,128] ([cos(64)|sin(64)] per row)
//            [6] kv_cache bf16[rows,QKHEAD] (hidden read/write, row [step])
//   outputs: [0] g_qpe f32[HLOCAL*QKHEAD]  (the T2->T3 chain edge)
//   params : kv_offset (step = iter_num + kv_offset)
// ============================================================================
__device__ __noinline__ void
    qb_rope_kv_task_impl(mirage::runtime::TaskDesc const *task_desc,
                         int task_offset,
                         int num_tasks,
                         int kv_offset,
                         int iter_num) {
  float const *g_qkva = static_cast<float const *>(task_desc->input_ptrs[0]);
  __nv_bfloat16 const *q_a_ln_w =
      static_cast<__nv_bfloat16 const *>(task_desc->input_ptrs[1]);
  __nv_bfloat16 const *kv_a_ln_w =
      static_cast<__nv_bfloat16 const *>(task_desc->input_ptrs[2]);
  __nv_fp8_e4m3 const *q_b_w =
      static_cast<__nv_fp8_e4m3 const *>(task_desc->input_ptrs[3]);
  float const *q_b_s = static_cast<float const *>(task_desc->input_ptrs[4]);
  __nv_bfloat16 const *cos_sin =
      static_cast<__nv_bfloat16 const *>(task_desc->input_ptrs[5]);
  __nv_bfloat16 *kv_cache =
      static_cast<__nv_bfloat16 *>(task_desc->input_ptrs[6]);
  float *g_qpe = static_cast<float *>(task_desc->output_ptrs[0]);

  extern __shared__ char smem[];
  float *s_qbdeq = reinterpret_cast<float *>(
      smem + task_desc->smem_region_offset(QB_REGION_WORK));
  float *red8 = reinterpret_cast<float *>(
      reinterpret_cast<char *>(s_qbdeq) + QB_RED_OFF);
  uint4 *s_ring = reinterpret_cast<uint4 *>(
      smem + task_desc->smem_region_offset(QB_REGION_RING));

  int tid = threadIdx.x, lane = tid & 31, warpl = tid >> 5;
  int const step = iter_num + kv_offset;
  int const pos = step;

  // q_rcp: EXACT-TREE emulation of v1's rms_rcp_block over g_qkva[0:QLORA)
  // (global reads, exactly as v1 reads its scratch).
  float q_rcp = rms_rcp_block_128emu(g_qkva, K_QLORA, red8);

  // Task 0 only: kv_rcp (exact tree) + kv_a-ln (v1's grid-strided 512-elem
  // elementwise loop RESTRIPED into this task — value-exact) + rope(k_pe)
  // (v1's literal worker-0/tid<32 pair loop). task_offset is task-uniform,
  // so the barriers inside rms_rcp_block_128emu are non-divergent.
  if (task_offset == 0) {
    float kv_rcp = rms_rcp_block_128emu(g_qkva + K_QLORA, K_KVLORA, red8);
    for (int i = tid; i < K_KVLORA; i += 128) {
      float v = v1a::k_bf16(g_qkva[K_QLORA + i] * kv_rcp *
                            __bfloat162float(kv_a_ln_w[i]));
      kv_cache[(size_t)step * K_QKHEAD + i] = __float2bfloat16(v);
    }
    if (tid < K_QKROPE / 2) {
      int pr = tid;
      int d0 = pr * 2, d1 = d0 + 1;
      float c = __bfloat162float(cos_sin[pos * K_COSSIN_STRIDE + d0]);
      float s = __bfloat162float(
          cos_sin[pos * K_COSSIN_STRIDE + K_COSSIN_SINOFF + d0]);
      float k0 = g_qkva[2048 + d0], k1 = g_qkva[2048 + d1];
      kv_cache[(size_t)step * K_QKHEAD + 512 + d0] =
          __float2bfloat16(v1a::k_bf16(k0 * c - k1 * s));
      kv_cache[(size_t)step * K_QKHEAD + 512 + d1] =
          __float2bfloat16(v1a::k_bf16(k1 * c + k0 * s));
    }
  }

  // q_a-ln + UE8M0 requant into task-local SMEM s_qbdeq (v1 lever-2 body
  // verbatim; outer warp stride 8 -> 4).
  {
    int ngq = K_QLORA / K_GRP; // 12
    for (int g = warpl; g < ngq; g += 4) {
      float const *src = g_qkva + g * K_GRP;
      __nv_bfloat16 const *w = q_a_ln_w + g * K_GRP;
      float nv[4];
      float mx = 1e-10f;
#pragma unroll
      for (int t = 0; t < 4; t++) {
        int j = lane + t * 32;
        float v = v1a::k_bf16(src[j] * q_rcp * __bfloat162float(w[j]));
        nv[t] = v;
        mx = fmaxf(mx, fabsf(v));
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        float ot = __shfl_xor_sync(0xffffffffu, mx, o);
        mx = fmaxf(mx, ot);
      }
      float ys = fmaxf(mx / K_FP8MAX, 1e-10f);
      float yq = v1a::k_dec_ue8m0(v1a::k_enc_ue8m0(ys));
      float *d = s_qbdeq + g * K_GRP;
#pragma unroll
      for (int t = 0; t < 4; t++) {
        int j = lane + t * 32;
        float qv = fminf(fmaxf(nv[t] / yq, -K_FP8MAX), K_FP8MAX);
        d[j] = (float)__nv_fp8_e4m3(qv) * yq;
      }
    }
  }
  attn_consumer_sync(); // publishes s_qbdeq (v1's __syncthreads at S3->S4)

  // q_b GEMV + fused YaRN rope (VERBATIM v1 template, SMEM activation).
  uint4 *my_ring = s_ring + (size_t)warpl * (GEMV_RING_BYTES_PER_WARP / 16);
  v1a::gemv_grid_cpa_qb_rope_smem_t<8, 4>(s_qbdeq,
                                          q_b_w,
                                          q_b_s,
                                          g_qpe,
                                          K_HLOCAL * K_QKHEAD,
                                          K_QLORA,
                                          cos_sin,
                                          pos,
                                          task_offset * 4 + warpl,
                                          num_tasks * 4,
                                          lane,
                                          my_ring);
  attn_task_epilogue();
}

// ============================================================================
// T3 — mla_partial. v1 S9: block (h,sp) computes the un-normalized softmax
// over its KV sub-range. 128 static tasks (h = t>>3, sp = t&7); tasks with
// sp >= nsp(KV) are runtime no-ops (v1's idle blocks). The body is an
// EXACT-TREE 128-thread port of v1's mla_partial: TPR is v1's (from
// NTHREAD=256); score/lmax/lsum phases use the A/B emulation; the exp and
// V-accumulation restripes are elementwise (value-exact).
//   inputs : [0] g_qpe (chain edge in)  [1] kv_cache (ALIAS — read history)
//            [2] g_mla_m f32[16*8] (hidden write) [3] g_mla_l (hidden write)
//   outputs: [0] g_mla_acc f32[16*8*512] (the T3->T4 chain edge)
//   params : kv_offset
// ============================================================================
__device__ __noinline__ void
    mla_partial_task_impl(mirage::runtime::TaskDesc const *task_desc,
                          int task_offset,
                          int kv_offset,
                          int iter_num) {
  float const *g_qpe = static_cast<float const *>(task_desc->input_ptrs[0]);
  __nv_bfloat16 const *kv_cache =
      static_cast<__nv_bfloat16 const *>(task_desc->input_ptrs[1]);
  float *g_mla_m = static_cast<float *>(task_desc->input_ptrs[2]);
  float *g_mla_l = static_cast<float *>(task_desc->input_ptrs[3]);
  float *g_mla_acc = static_cast<float *>(task_desc->output_ptrs[0]);

  extern __shared__ char smem[];
  float *s_score = reinterpret_cast<float *>(
      smem + task_desc->smem_region_offset(MP_REGION_WORK));
  float *red8 = reinterpret_cast<float *>(
      reinterpret_cast<char *>(s_score) + MP_RED_OFF);

  int tid = threadIdx.x;
  int const step = iter_num + kv_offset;
  int const KV = step + 1;
  // v1 split math VERBATIM.
  int nsp = (KV + 63) / 64;
  if (nsp < 1) {
    nsp = 1;
  }
  if (nsp > MLA_SPLITS) {
    nsp = MLA_SPLITS;
  }
  int const tile = (KV + nsp - 1) / nsp;
  int const h = task_offset >> 3;
  int const sp = task_offset & 7;

  if (sp < nsp) {
    int r0 = sp * tile, r1 = r0 + tile;
    if (r1 > KV) {
      r1 = KV;
    }
    double mscale = 0.1 * log(40.0) + 1.0;
    float sm = (float)((1.0 / sqrt(192.0)) * mscale * mscale);
    float const *q = &g_qpe[h * K_QKHEAD];
    int const nr = r1 - r0;
    // ---- TPR selection VERBATIM (NTHREAD is v1's 256, NOT the v2 thread
    // count — required for the exact grouping).
    int TPR = NTHREAD / (nr > 0 ? nr : 1);
    if (TPR < 1) {
      TPR = 1;
    }
    if (TPR > 8) {
      TPR = 8;
    }
    if (TPR >= 8) {
      TPR = 8;
    } else if (TPR >= 4) {
      TPR = 4;
    } else if (TPR >= 2) {
      TPR = 2;
    } else {
      TPR = 1;
    }
    {
      int laneInWarp = tid & 31;
      unsigned grpmask =
          ((TPR >= 32) ? 0xffffffffu
                       : (((1u << TPR) - 1u) << ((laneInWarp / TPR) * TPR)));
      int rows_per_step = NTHREAD / TPR;
      // Role A (v1 tid = tid) then role B (v1 tid = tid+128), sequentially.
      // TPR | 128 => subB == subA and the grpmask is identical; only the row
      // base shifts by 128/TPR. Each masked shuffle converges because all
      // lanes of a group share the same row sequence within a role pass.
#pragma unroll
      for (int role = 0; role < 2; role++) {
        int const v1tid = tid + role * 128;
        int const sub = v1tid % TPR;
        int const row = v1tid / TPR;
        for (int rr = row; rr < nr; rr += rows_per_step) {
          int r = r0 + rr;
          uint4 const *kvr =
              reinterpret_cast<uint4 const *>(&kv_cache[(size_t)r * K_QKHEAD]);
          float dot = 0.f;
          for (int c = sub; c < K_QKHEAD / 8; c += TPR) {
            uint4 kw = kvr[c];
            __nv_bfloat162 const *k2 =
                reinterpret_cast<__nv_bfloat162 const *>(&kw);
            float const *qc = &q[c * 8];
#pragma unroll
            for (int p = 0; p < 4; p++) {
              float2 kf = __bfloat1622float2(k2[p]);
              dot += qc[2 * p] * kf.x + qc[2 * p + 1] * kf.y;
            }
          }
#pragma unroll
          for (int o = TPR >> 1; o > 0; o >>= 1) {
            dot += __shfl_down_sync(grpmask, dot, o, TPR);
          }
          if (sub == 0) {
            s_score[rr] = dot * sm;
          }
        }
      }
    }
    attn_consumer_sync(); // v1's post-score __syncthreads
    // ---- lmax (A/B emulation of v1's xor tree + red8 + redundant max).
    float lmA = -1e30f, lmB = -1e30f;
    for (int rr = tid; rr < nr; rr += NTHREAD) {
      lmA = fmaxf(lmA, s_score[rr]);
    }
    for (int rr = tid + 128; rr < nr; rr += NTHREAD) {
      lmB = fmaxf(lmB, s_score[rr]);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      lmA = fmaxf(lmA, __shfl_xor_sync(0xffffffffu, lmA, o));
      lmB = fmaxf(lmB, __shfl_xor_sync(0xffffffffu, lmB, o));
    }
    if ((tid & 31) == 0) {
      red8[tid >> 5] = lmA;
      red8[(tid >> 5) + 4] = lmB;
    }
    attn_consumer_sync();
    float gmax = -1e30f;
#pragma unroll
    for (int i = 0; i < NWARP; i++) {
      gmax = fmaxf(gmax, red8[i]);
    }
    attn_consumer_sync();
    // exp (elementwise restripe — value-exact).
    for (int rr = tid; rr < nr; rr += 128) {
      s_score[rr] = __expf(s_score[rr] - gmax);
    }
    attn_consumer_sync();
    // ---- lsum (A/B emulation; v1 xor tree + red8 + redundant sequential Σ).
    float lsA = 0.f, lsB = 0.f;
    for (int rr = tid; rr < nr; rr += NTHREAD) {
      lsA += s_score[rr];
    }
    for (int rr = tid + 128; rr < nr; rr += NTHREAD) {
      lsB += s_score[rr];
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      lsA += __shfl_xor_sync(0xffffffffu, lsA, o);
      lsB += __shfl_xor_sync(0xffffffffu, lsB, o);
    }
    if ((tid & 31) == 0) {
      red8[tid >> 5] = lsA;
      red8[(tid >> 5) + 4] = lsB;
    }
    attn_consumer_sync();
    float gsum = 0.f;
#pragma unroll
    for (int i = 0; i < NWARP; i++) {
      gsum += red8[i];
    }
    int const base = h * MLA_SPLITS + sp;
    if (tid == 0) {
      g_mla_m[base] = (nr > 0) ? gmax : -1e30f;
      g_mla_l[base] = gsum;
    }
    // V accumulation (per-d independent, inner rr loop sequential — the
    // restripe is value-exact).
    float *accv = &g_mla_acc[(size_t)base * K_KVLORA];
    for (int d = tid; d < K_KVLORA; d += 128) {
      float acc = 0.f;
      for (int rr = 0; rr < nr; rr++) {
        acc += s_score[rr] *
               __bfloat162float(kv_cache[(size_t)(r0 + rr) * K_QKHEAD + d]);
      }
      accv[d] = acc;
    }
  }
  attn_task_epilogue(); // UNCONDITIONAL (also on the sp>=nsp no-op path)
}

// ============================================================================
// T4 — mla_merge. v1 S10+S11 (mla_merge_quant): per-head merge of the nsp
// partials + inline NON-UE8M0 448-quant (ys used raw — v1 line ~1575). One
// task per head (16). Port changes vs v1: the d-loop stride 256 -> 128
// (elementwise, value-exact); the per-128-group quant loop is UNCHANGED (v1
// already restricts it to warps 0-3); the lever-5 readiness-flag store +
// trailing fence are DROPPED (the T4->T5 event dep is the handoff).
//   inputs : [0] g_mla_acc (chain edge in)  [1] g_mla_m (alias)
//            [2] g_mla_l (alias)            [3] g_attn (hidden write —
//            the pre-quant bf16-rounded merge, kept for the v1 A/B compare)
//   outputs: [0] g_attn_deq f32[16*512] (the T4->T5 chain edge)
//   params : kv_offset (for nsp)
// ============================================================================
__device__ __noinline__ void
    mla_merge_task_impl(mirage::runtime::TaskDesc const *task_desc,
                        int task_offset,
                        int kv_offset,
                        int iter_num) {
  float const *g_mla_acc =
      static_cast<float const *>(task_desc->input_ptrs[0]);
  float const *g_mla_m = static_cast<float const *>(task_desc->input_ptrs[1]);
  float const *g_mla_l = static_cast<float const *>(task_desc->input_ptrs[2]);
  float *g_attn = static_cast<float *>(task_desc->input_ptrs[3]);
  float *g_attn_deq = static_cast<float *>(task_desc->output_ptrs[0]);

  extern __shared__ char smem[];
  float *s_attn = reinterpret_cast<float *>(
      smem + task_desc->smem_region_offset(MM_REGION_WORK));

  int tid = threadIdx.x, lane = tid & 31, warpl = tid >> 5;
  int const step = iter_num + kv_offset;
  int const KV = step + 1;
  int nsp = (KV + 63) / 64;
  if (nsp < 1) {
    nsp = 1;
  }
  if (nsp > MLA_SPLITS) {
    nsp = MLA_SPLITS;
  }
  int const h = task_offset;

  float const *mrow = &g_mla_m[h * MLA_SPLITS];
  float const *lrow = &g_mla_l[h * MLA_SPLITS];
  float gmax = -1e30f;
#pragma unroll
  for (int s = 0; s < MLA_SPLITS; s++) {
    if (s < nsp) {
      gmax = fmaxf(gmax, mrow[s]);
    }
  }
  float denom = 0.f;
#pragma unroll
  for (int s = 0; s < MLA_SPLITS; s++) {
    if (s < nsp) {
      denom += lrow[s] * __expf(mrow[s] - gmax);
    }
  }
  float inv = (denom > 0.f) ? 1.0f / denom : 0.f;
  for (int d = tid; d < K_KVLORA; d += 128) { // v1 stride 256 -> 128
    float acc = 0.f;
#pragma unroll
    for (int s = 0; s < MLA_SPLITS; s++) {
      if (s < nsp) {
        float w = __expf(mrow[s] - gmax);
        acc += g_mla_acc[((size_t)(h * MLA_SPLITS + s)) * K_KVLORA + d] * w;
      }
    }
    float v = v1a::k_bf16(acc * inv);
    s_attn[d] = v;
    g_attn[h * K_KVLORA + d] = v;
  }
  attn_consumer_sync(); // v1's __syncthreads before the quant phase
  int const KGv = K_KVLORA / K_GRP; // 4
  if (warpl < KGv) {                // v2 warps 0-3 == v1's active subset
    float const *ar = &s_attn[warpl * K_GRP];
    float *dq = &g_attn_deq[h * K_KVLORA + warpl * K_GRP];
    float mx = 1e-10f;
    for (int j = lane; j < K_GRP; j += 32) {
      float a = fabsf(ar[j]);
      mx = fmaxf(mx, a);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      float ot = __shfl_xor_sync(0xffffffffu, mx, o);
      mx = fmaxf(mx, ot);
    }
    float ys = fmaxf(mx / K_FP8MAX, 1e-10f); // RAW 448 scale — NOT UE8M0
    for (int j = lane; j < K_GRP; j += 32) {
      float vq = fminf(fmaxf(ar[j] / ys, -K_FP8MAX), K_FP8MAX);
      dq[j] = (float)__nv_fp8_e4m3(vq) * ys;
    }
  }
  attn_task_epilogue();
}

// ============================================================================
// T5 — wuv. v1 S12: W_UV per-head BMM, VERBATIM v1a::wuv_bmm_grid call.
// The verbatim body's per-head acquire-spin reads ready_ones (a constant
// int32[16] == 1 buffer) and exits immediately — the REAL g_attn_deq handoff
// is the T4->T5 event dependency. No SMEM regions (v1 uses none here).
//   inputs : [0] g_attn_deq (chain edge in)  [1] kvbv_w fp8[16,128,512]
//            [2] kvbv_s f32[16,1,4]          [3] ready_ones i32[16] (=1)
//   outputs: [0] g_red f32[OIN]  (the T5->T6 chain edge)
// ============================================================================
__device__ __noinline__ void
    wuv_task_impl(mirage::runtime::TaskDesc const *task_desc,
                  int task_offset,
                  int num_tasks) {
  float const *g_attn_deq =
      static_cast<float const *>(task_desc->input_ptrs[0]);
  __nv_fp8_e4m3 const *kvbv_w =
      static_cast<__nv_fp8_e4m3 const *>(task_desc->input_ptrs[1]);
  float const *kvbv_s = static_cast<float const *>(task_desc->input_ptrs[2]);
  int *ready_ones = static_cast<int *>(task_desc->input_ptrs[3]);
  float *g_red = static_cast<float *>(task_desc->output_ptrs[0]);

  int lane = threadIdx.x & 31, warpl = threadIdx.x >> 5;
  v1a::wuv_bmm_grid(g_attn_deq,
                    kvbv_w,
                    kvbv_s,
                    g_red,
                    task_offset * 4 + warpl,
                    num_tasks * 4,
                    lane,
                    ready_ones);
  // No SMEM regions -> plain return (page suffix no-ops); GMEM stores are
  // published by the role-loop FINISHED arrive + event release chain.
}

// ============================================================================
// T6 — oproj. v1 S13: UE8M0 quant of g_red (v1 lever-3
// quant_ue8m0_block_smem body, redundant per task, outer stride 8 -> 4) +
// o_proj GEMV with fused residual add (VERBATIM v1 template, SMEM act).
//   inputs : [0] g_red (chain edge in)  [1] oproj_w fp8[HIDDEN,OIN]
//            [2] oproj_s f32[56,16]     [3] residual bf16[1,HIDDEN] (ALIAS
//            of the block input x — a declared edge would fork x)
//   outputs: [0] out bf16[1,HIDDEN]  (the block output / next block's x)
// ============================================================================
__device__ __noinline__ void
    oproj_task_impl(mirage::runtime::TaskDesc const *task_desc,
                    int task_offset,
                    int num_tasks) {
  float const *g_red = static_cast<float const *>(task_desc->input_ptrs[0]);
  __nv_fp8_e4m3 const *oproj_w =
      static_cast<__nv_fp8_e4m3 const *>(task_desc->input_ptrs[1]);
  float const *oproj_s = static_cast<float const *>(task_desc->input_ptrs[2]);
  __nv_bfloat16 const *residual =
      static_cast<__nv_bfloat16 const *>(task_desc->input_ptrs[3]);
  __nv_bfloat16 *out = static_cast<__nv_bfloat16 *>(task_desc->output_ptrs[0]);

  extern __shared__ char smem[];
  float *s_odeq = reinterpret_cast<float *>(
      smem + task_desc->smem_region_offset(OP_REGION_WORK));
  uint4 *s_ring = reinterpret_cast<uint4 *>(
      smem + task_desc->smem_region_offset(OP_REGION_RING));

  int lane = threadIdx.x & 31, warpl = threadIdx.x >> 5;

  // v1 quant_ue8m0_block_smem body verbatim (outer warp stride 8 -> 4).
  {
    int ng = K_OIN / K_GRP; // 16
    for (int gx = warpl; gx < ng; gx += 4) {
      float const *s = g_red + gx * K_GRP + lane * 4;
      float4 a = *reinterpret_cast<float4 const *>(s);
      float v[4] = {a.x, a.y, a.z, a.w};
      float mx = 1e-10f;
#pragma unroll
      for (int t = 0; t < 4; t++) {
        mx = fmaxf(mx, fabsf(v[t]));
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
      }
      float ys = fmaxf(mx / K_FP8MAX, 1e-10f);
      float yq = v1a::k_dec_ue8m0(v1a::k_enc_ue8m0(ys));
      float *d = s_odeq + gx * K_GRP + lane * 4;
#pragma unroll
      for (int t = 0; t < 4; t++) {
        float q = fminf(fmaxf(v[t] / yq, -K_FP8MAX), K_FP8MAX);
        d[t] = (float)__nv_fp8_e4m3(q) * yq;
      }
    }
  }
  attn_consumer_sync(); // publishes s_odeq (v1's trailing __syncthreads)

  uint4 *my_ring = s_ring + (size_t)warpl * (GEMV_RING_BYTES_PER_WARP / 16);
  v1a::gemv_grid_cpa_oproj_smem_t<8, 4>(s_odeq,
                                        oproj_w,
                                        oproj_s,
                                        residual,
                                        out,
                                        K_HIDDEN,
                                        K_OIN,
                                        task_offset * 4 + warpl,
                                        num_tasks * 4,
                                        lane,
                                        my_ring);
  attn_task_epilogue();
}

} // namespace dsv3_attn_v2
} // namespace kernel
