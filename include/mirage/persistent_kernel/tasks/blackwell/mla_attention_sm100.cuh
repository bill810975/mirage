/* Copyright 2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// ===========================================================================
// MLA Paged Attention for DeepSeek V3 — MPK Device Function
//
// ARCHITECTURE NOTE:
//
// The ideal implementation would directly reuse the NVIDIA/FlashInfer MLA
// kernel (mla_sm100_2sm.cuh), which uses 2-SM clusters, TMA, TMEM, and
// CUTLASS CollectiveBuilder for maximum performance. However, it cannot be
// called as a device function because:
//
// 1. CollectiveMmaQK::to_underlying_arguments() is a HOST function that
//    creates TMA descriptors. These descriptors need actual GPU memory
//    addresses which are only available at runtime on the host.
//
// 2. The MPK task system passes raw pointers via TaskDesc, with no mechanism
//    for host-side CUTLASS Params preparation.
//
// TO ENABLE FLASHINFER MLA INTEGRATION, the MPK framework needs:
// - A host-side "prepare_params" callback per task type that runs before
//   kernel launch, converting TaskDesc pointers into CUTLASS Params
// - The Params struct passed to the device function via a pre-allocated
//   device buffer (similar to how CUTLASS kernel launches work)
// This is a framework-level change tracked separately.
//
// CURRENT IMPLEMENTATION:
// Uses the same MMA approach as the existing attention_sm100.cuh (which
// powers Qwen3), adapted for MLA's asymmetric dimensions:
// - QK_HEAD_DIM=576 (512 latent + 64 rope)
// - V_HEAD_DIM=512 (latent only)
// - Single KV head (MQA after weight absorption)
// - 4 independent warps, each handling 1 Q head at a time
// - MMA m16n16k16 for both QK and PV matmuls
// - Online softmax with per-warp reductions (no cross-warp S_O_BUFFER)
//
// Performance: Same MMA throughput as Qwen3's attention kernel (m16n16k16).
// The per-Q-head independent warp approach avoids the S_O_BUFFER limitation
// that prevents the original attention_sm100 architecture from handling
// MLA's 512-dim output.
// ===========================================================================

#pragma once
#include "tasks/ampere/mma.cuh"
#include "tasks/ampere/smem_layout.cuh"
#include "tasks/common/common_header.cuh"

#include <cutlass/arch/barrier.h>

namespace kernel {

template <typename T,
          int NUM_Q_HEADS,
          int QK_HEAD_DIM,
          int V_HEAD_DIM,
          int MAX_SEQ_LEN,
          int PAGE_SIZE,
          int MAX_TOKENS = 8,
          int KV_TILE_SIZE = 32>
__device__ __forceinline__ void mla_paged_attention_sm100_task_impl(
    void const *q_nope_pe_ptr,
    void *ckv_kpe_cache_ptr,
    void const *c_latent_new_ptr,
    void const *k_pe_new_ptr,
    void *output_ptr,
    int const *qo_indptr_buffer_ptr,
    int const *paged_kv_indptr_buffer_ptr,
    int const *paged_kv_indices_buffer_ptr,
    int const *paged_kv_last_page_len_buffer_ptr,
    int16_t request_id) {

  constexpr int BARRIER_ID = 6;
  cutlass::arch::NamedBarrier wg_barrier(NUM_THREADS, BARRIER_ID);

  if (threadIdx.x >= NUM_THREADS) {
    return;
  }

  // Constants
  constexpr int ROPE_DIM = QK_HEAD_DIM - V_HEAD_DIM;
  constexpr int Q_STRIDE = NUM_Q_HEADS * QK_HEAD_DIM;
  constexpr int O_STRIDE = NUM_Q_HEADS * V_HEAD_DIM;
  constexpr int MAX_PAGES = (MAX_SEQ_LEN + PAGE_SIZE - 1) / PAGE_SIZE;
  constexpr int CP_CHUNK_SIZE = 16 / sizeof(T);
  // Each warp handles 1 Q head independently (4 warps = 4 heads per pass)
  constexpr int QH_PER_PASS = NUM_WARPS;
  constexpr int NUM_QH_PASSES = (NUM_Q_HEADS + QH_PER_PASS - 1) / QH_PER_PASS;
  // MMA: each warp handles MAX_TOKENS rows
  constexpr int MMA_ITERS_M = (MAX_TOKENS + 15) / 16;
  constexpr int MMA_ITERS_N_QK = (KV_TILE_SIZE + 15) / 16;
  constexpr int MMA_ITERS_K_QK = (QK_HEAD_DIM + 15) / 16;
  constexpr int MMA_ITERS_N_PV = (V_HEAD_DIM + 15) / 16;
  constexpr int MMA_ITERS_K_PV = (KV_TILE_SIZE + 15) / 16;

  float const sm_scale = 1.0f / sqrtf(static_cast<float>(QK_HEAD_DIM));

  int warp_idx = warp_id();
  int lane_idx = lane_id();

  // Request metadata
  int const first_token = qo_indptr_buffer_ptr[request_id];
  int const last_token = qo_indptr_buffer_ptr[request_id + 1];
  int const num_tokens = last_token - first_token;
  if (num_tokens == 0) return;

  int const first_page = paged_kv_indptr_buffer_ptr[request_id];
  int const num_pages =
      paged_kv_indptr_buffer_ptr[request_id + 1] - first_page;
  int const seq_len =
      (num_pages - 1) * PAGE_SIZE +
      paged_kv_last_page_len_buffer_ptr[request_id];
  int const num_kv_iters = (seq_len + KV_TILE_SIZE - 1) / KV_TILE_SIZE;

  // Page indices in shared memory
  __shared__ int s_page_indices[MAX_PAGES];
  for (int i = threadIdx.x; i < num_pages; i += NUM_THREADS) {
    s_page_indices[i] = paged_kv_indices_buffer_ptr[first_page + i];
  }
  wg_barrier.arrive_and_wait();

  // Pointers
  T const *__restrict__ d_q =
      reinterpret_cast<T const *>(q_nope_pe_ptr) + first_token * Q_STRIDE;
  T const *__restrict__ d_c_new =
      reinterpret_cast<T const *>(c_latent_new_ptr) + first_token * V_HEAD_DIM;
  T const *__restrict__ d_k_pe_new =
      reinterpret_cast<T const *>(k_pe_new_ptr) + first_token * ROPE_DIM;
  T *__restrict__ d_cache = reinterpret_cast<T *>(ckv_kpe_cache_ptr);
  T *__restrict__ d_output =
      reinterpret_cast<T *>(output_ptr) + first_token * O_STRIDE;

  // Shared memory: Q + K double-buffer + V double-buffer
  constexpr size_t S_Q_SIZE =
      sizeof(T) * QH_PER_PASS * MAX_TOKENS * QK_HEAD_DIM;
  constexpr size_t S_K_SIZE = sizeof(T) * KV_TILE_SIZE * QK_HEAD_DIM;
  constexpr size_t S_V_SIZE = sizeof(T) * KV_TILE_SIZE * V_HEAD_DIM;
  constexpr size_t S_TOTAL = S_Q_SIZE + 2 * S_K_SIZE + 2 * S_V_SIZE;
  static_assert(S_TOTAL <= mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE,
                "MLA smem exceeds limit");

  extern __shared__ char smem[];
  T *s_q = reinterpret_cast<T *>(smem);
  T *s_k[2] = {reinterpret_cast<T *>(smem + S_Q_SIZE),
                reinterpret_cast<T *>(smem + S_Q_SIZE + S_K_SIZE)};
  T *s_v[2] = {reinterpret_cast<T *>(smem + S_Q_SIZE + 2 * S_K_SIZE),
                reinterpret_cast<T *>(smem + S_Q_SIZE + 2 * S_K_SIZE + S_V_SIZE)};

  using QSmem = smem_row<T, 3, 3, 3, MAX_TOKENS, QK_HEAD_DIM, QK_HEAD_DIM>;
  using KSmem = smem_row<T, 3, 3, 3, KV_TILE_SIZE, QK_HEAD_DIM, QK_HEAD_DIM>;
  using VSmem = smem_row<T, 3, 3, 3, KV_TILE_SIZE, V_HEAD_DIM, V_HEAD_DIM>;
  T zero_val = T(0);

  // Phase 1: Write new KV to cache
  for (int idx = threadIdx.x; idx < num_tokens * QK_HEAD_DIM;
       idx += NUM_THREADS) {
    int t = idx / QK_HEAD_DIM;
    int d = idx % QK_HEAD_DIM;
    int cache_pos = seq_len - num_tokens + t;
    int pi = s_page_indices[cache_pos / PAGE_SIZE];
    int po = cache_pos % PAGE_SIZE;
    T val = (d < V_HEAD_DIM)
                ? d_c_new[t * V_HEAD_DIM + d]
                : d_k_pe_new[t * ROPE_DIM + (d - V_HEAD_DIM)];
    d_cache[(pi * PAGE_SIZE + po) * QK_HEAD_DIM + d] = val;
  }
  wg_barrier.arrive_and_wait();

  // Phase 2: Attention — outer loop over Q heads
  for (int qh_pass = 0; qh_pass < NUM_QH_PASSES; qh_pass++) {
    int my_qh = qh_pass * QH_PER_PASS + warp_idx;
    bool active = (my_qh < NUM_Q_HEADS);

    // Load Q for this warp's head
    if (active) {
      T *my_sq = s_q + warp_idx * MAX_TOKENS * QK_HEAD_DIM;
      for (int idx = lane_idx; idx < num_tokens * QK_HEAD_DIM; idx += 32) {
        int t = idx / QK_HEAD_DIM;
        int d = idx % QK_HEAD_DIM;
        my_sq[t * QK_HEAD_DIM + d] =
            d_q[t * Q_STRIDE + my_qh * QK_HEAD_DIM + d];
      }
    }
    wg_barrier.arrive_and_wait();

    // Per-warp accumulators
    float m_local[MMA_ITERS_M][2];
    float d_acc[MMA_ITERS_M][2];
    float o_acc[MMA_ITERS_M][MMA_ITERS_N_PV][8];
#pragma unroll
    for (int m = 0; m < MMA_ITERS_M; m++) {
      m_local[m][0] = -inf;
      m_local[m][1] = -inf;
      d_acc[m][0] = 1.f;
      d_acc[m][1] = 1.f;
#pragma unroll
      for (int n = 0; n < MMA_ITERS_N_PV; n++) {
        clear_8_floats(o_acc[m][n]);
      }
    }

    // Prefetch first KV tile
    int first_kv_len = min(seq_len, KV_TILE_SIZE);
    for (int idx = threadIdx.x; idx < first_kv_len * QK_HEAD_DIM;
         idx += NUM_THREADS) {
      int row = idx / QK_HEAD_DIM, col = idx % QK_HEAD_DIM;
      int cp = row;
      int pi = s_page_indices[cp / PAGE_SIZE];
      int po = cp % PAGE_SIZE;
      s_k[0][row * QK_HEAD_DIM + col] =
          d_cache[(pi * PAGE_SIZE + po) * QK_HEAD_DIM + col];
    }
    for (int idx = threadIdx.x; idx < first_kv_len * V_HEAD_DIM;
         idx += NUM_THREADS) {
      int row = idx / V_HEAD_DIM, col = idx % V_HEAD_DIM;
      int cp = row;
      int pi = s_page_indices[cp / PAGE_SIZE];
      int po = cp % PAGE_SIZE;
      s_v[0][row * V_HEAD_DIM + col] =
          d_cache[(pi * PAGE_SIZE + po) * QK_HEAD_DIM + col];
    }
    wg_barrier.arrive_and_wait();

    int curr_kv_len = first_kv_len;
    int curr_buf = 0;
    int kv_loaded = first_kv_len;

    // Inner loop over KV tiles
    for (int kv_iter = 0; kv_iter < num_kv_iters; kv_iter++) {
      int next_buf = 1 - curr_buf;
      int next_kv_len = 0;

      // Prefetch next tile
      if (kv_iter + 1 < num_kv_iters) {
        next_kv_len = min(seq_len - kv_loaded, KV_TILE_SIZE);
        for (int idx = threadIdx.x; idx < next_kv_len * QK_HEAD_DIM;
             idx += NUM_THREADS) {
          int row = idx / QK_HEAD_DIM, col = idx % QK_HEAD_DIM;
          int cp = kv_loaded + row;
          int pi = s_page_indices[cp / PAGE_SIZE];
          int po = cp % PAGE_SIZE;
          s_k[next_buf][row * QK_HEAD_DIM + col] =
              d_cache[(pi * PAGE_SIZE + po) * QK_HEAD_DIM + col];
        }
        for (int idx = threadIdx.x; idx < next_kv_len * V_HEAD_DIM;
             idx += NUM_THREADS) {
          int row = idx / V_HEAD_DIM, col = idx % V_HEAD_DIM;
          int cp = kv_loaded + row;
          int pi = s_page_indices[cp / PAGE_SIZE];
          int po = cp % PAGE_SIZE;
          s_v[next_buf][row * V_HEAD_DIM + col] =
              d_cache[(pi * PAGE_SIZE + po) * QK_HEAD_DIM + col];
        }
        kv_loaded += next_kv_len;
      }
      wg_barrier.arrive_and_wait();

      if (!active) {
        curr_buf = next_buf;
        curr_kv_len = next_kv_len;
        wg_barrier.arrive_and_wait();
        continue;
      }

      QSmem q_smem(s_q + warp_idx * MAX_TOKENS * QK_HEAD_DIM);
      KSmem k_smem(s_k[curr_buf]);
      VSmem v_smem(s_v[curr_buf]);

      // ---- QK^T via MMA ----
      float x_frag_f[MMA_ITERS_M][MMA_ITERS_N_QK][8];
#pragma unroll
      for (int m = 0; m < MMA_ITERS_M; m++)
#pragma unroll
        for (int n = 0; n < MMA_ITERS_N_QK; n++)
          clear_8_floats(x_frag_f[m][n]);

      uint32_t q_frag[4], kt_frag[4];
#pragma unroll
      for (int m = 0; m < MMA_ITERS_M; m++) {
#pragma unroll
        for (int n = 0; n < MMA_ITERS_N_QK; n++) {
#pragma unroll
          for (int k = 0; k < MMA_ITERS_K_QK; k++) {
            int q_row = m * 16 + (lane_idx & 0xF);
            int q_col = k * 16 + ((lane_idx >> 4) << 3);
            T *src_q =
                q_row < num_tokens ? q_smem(q_row, q_col) : &zero_val;

            int kt_col = n * 16 + ((lane_idx >> 4) << 3) + (lane_idx & 0x7);
            int kt_row = k * 16 + (((lane_idx & 0xF) >> 3) << 3);
            T *src_kt =
                kt_col < curr_kv_len ? k_smem(kt_col, kt_row) : &zero_val;

            ldsm(src_q, q_frag);
            ldsm(src_kt, kt_frag);
            mma_m16n16k16_bf16bf16bf32(
                x_frag_f[m][n], q_frag, kt_frag, x_frag_f[m][n]);
          }
        }
      }

      // ---- Online softmax ----
      float m_prev[MMA_ITERS_M][2];
#pragma unroll
      for (int m = 0; m < MMA_ITERS_M; m++) {
        m_prev[m][0] = m_local[m][0];
        m_prev[m][1] = m_local[m][1];
#pragma unroll
        for (int n = 0; n < MMA_ITERS_N_QK; n++) {
#pragma unroll
          for (int fi = 0; fi < 8; fi++) {
            int row = (m << 4) + (lane_idx >> 2) + (((fi & 3) >> 1) << 3);
            int col = n * 16 + ((lane_idx & 3) << 1) + ((fi >> 2) << 3) +
                      (fi & 1);
            bool valid = (row < num_tokens) &&
                         (col + kv_iter * KV_TILE_SIZE <=
                          row + seq_len - num_tokens);
            x_frag_f[m][n][fi] = valid ? x_frag_f[m][n][fi] : -inf;
            m_local[m][(fi & 3) >> 1] =
                max(m_local[m][(fi & 3) >> 1], x_frag_f[m][n][fi]);
          }
        }
        m_local[m][0] =
            max(m_local[m][0], shfl_xor_sync(m_local[m][0], 0x1));
        m_local[m][0] =
            max(m_local[m][0], shfl_xor_sync(m_local[m][0], 0x2));
        m_local[m][1] =
            max(m_local[m][1], shfl_xor_sync(m_local[m][1], 0x1));
        m_local[m][1] =
            max(m_local[m][1], shfl_xor_sync(m_local[m][1], 0x2));
      }

      float rescale[MMA_ITERS_M][2];
#pragma unroll
      for (int m = 0; m < MMA_ITERS_M; m++) {
        rescale[m][0] =
            expf(m_prev[m][0] * sm_scale - m_local[m][0] * sm_scale);
        rescale[m][1] =
            expf(m_prev[m][1] * sm_scale - m_local[m][1] * sm_scale);
      }

      float d_partial[MMA_ITERS_M][2];
#pragma unroll
      for (int m = 0; m < MMA_ITERS_M; m++) {
        d_partial[m][0] = 0.f;
        d_partial[m][1] = 0.f;
#pragma unroll
        for (int n = 0; n < MMA_ITERS_N_QK; n++) {
#pragma unroll
          for (int fi = 0; fi < 8; fi++) {
            x_frag_f[m][n][fi] =
                x_frag_f[m][n][fi] != -inf
                    ? expf(x_frag_f[m][n][fi] * sm_scale -
                           m_local[m][(fi & 3) >> 1] * sm_scale)
                    : 0.f;
            d_partial[m][(fi & 3) >> 1] += x_frag_f[m][n][fi];
          }
        }
        d_partial[m][0] += shfl_xor_sync(d_partial[m][0], 0x1);
        d_partial[m][0] += shfl_xor_sync(d_partial[m][0], 0x2);
        d_partial[m][1] += shfl_xor_sync(d_partial[m][1], 0x1);
        d_partial[m][1] += shfl_xor_sync(d_partial[m][1], 0x2);
        d_acc[m][0] = d_acc[m][0] * rescale[m][0] + d_partial[m][0];
        d_acc[m][1] = d_acc[m][1] * rescale[m][1] + d_partial[m][1];
      }

      // Rescale output
#pragma unroll
      for (int m = 0; m < MMA_ITERS_M; m++)
#pragma unroll
        for (int n = 0; n < MMA_ITERS_N_PV; n++)
#pragma unroll
          for (int fi = 0; fi < 8; fi++)
            o_acc[m][n][fi] *= rescale[m][(fi & 3) >> 1];

      // ---- PV via MMA ----
#pragma unroll
      for (int m = 0; m < MMA_ITERS_M; m++) {
#pragma unroll
        for (int kk = 0; kk < MMA_ITERS_K_PV; kk++) {
          uint32_t p_frag[4];
          convert_f32_to_bf16_uint32(x_frag_f[m][kk], p_frag);
#pragma unroll
          for (int nn = 0; nn < MMA_ITERS_N_PV; nn++) {
            uint32_t v_frag[4];
            int v_row = kk * 16 + (lane_idx & 0xF);
            int v_col = nn * 16 + ((lane_idx >> 4) << 3);
            T *src_v = v_row < curr_kv_len
                           ? &s_v[curr_buf][v_row * V_HEAD_DIM + v_col]
                           : &zero_val;
            ldsm_t(src_v, v_frag);
            mma_m16n16k16_bf16bf16bf32(
                o_acc[m][nn], p_frag, v_frag, o_acc[m][nn]);
          }
        }
      }

      wg_barrier.arrive_and_wait();
      curr_buf = next_buf;
      curr_kv_len = next_kv_len;
    } // KV loop

    // Write output for this Q-head pass
    if (active) {
      for (int idx = lane_idx; idx < num_tokens * V_HEAD_DIM; idx += 32) {
        int row = idx / V_HEAD_DIM;
        int col = idx % V_HEAD_DIM;
        int mma_m = row / 16;
        int mma_n = col / 16;
        int t_idx = (row % 8) * 4 + (col % 8) / 2;
        int fi = ((col % 16) / 8) * 4 + ((row % 16) / 8) * 2 + (col % 2);

        if (t_idx == lane_idx && mma_m < MMA_ITERS_M &&
            mma_n < MMA_ITERS_N_PV) {
          float o_val = o_acc[mma_m][mma_n][fi];
          float d_val = d_acc[mma_m][(fi & 3) >> 1];
          o_val = (d_val > 0.f) ? (o_val / d_val) : 0.f;
          d_output[row * O_STRIDE + my_qh * V_HEAD_DIM + col] = T(o_val);
        }
      }
    }
    wg_barrier.arrive_and_wait();
  } // QH pass loop
}

} // namespace kernel
