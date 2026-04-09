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
// MLA Paged Attention v1 for DeepSeek V3 -- MPK Device Function
//
// Architecture:
// - 1 block per (request, Q-head), qh_idx passed as parameter
// - All 128 threads cooperate on vectorized KV loading (uint4)
// - Warp 0 does all MMA (QK + softmax + PV)
// - Combined QK_HEAD_DIM=576 K buffer (NOT split nope/pe)
// - Separate V buffer (V_HEAD_DIM=512 wide)
// - KV_TILE_SIZE=32 with double-buffer
// - expf with sm_scale (NOT exp2f)
// - Online softmax with shfl_xor reduction
// - PV from float registers via convert_f32_to_bf16_uint32 (NOT via smem)
//
// Grid: dim3(num_requests, NUM_Q_HEADS)
// Block: 128 threads (4 warps)
// ===========================================================================

#pragma once
#include "tasks/ampere/mma.cuh"
#include "tasks/ampere/smem_layout.cuh"
#include "tasks/common/common_header.cuh"

#include <cutlass/arch/barrier.h>
#include <cuda_pipeline.h>

namespace kernel {

// T_to_float / float_to_T: work even with __CUDA_NO_BFLOAT16_CONVERSIONS__
template <typename T>
__device__ __forceinline__ float T_to_float(T v) {
  return static_cast<float>(v);
}
template <typename T>
__device__ __forceinline__ T float_to_T(float v) {
  return static_cast<T>(v);
}
#if defined(__CUDA_NO_BFLOAT16_CONVERSIONS__)
template <>
__device__ __forceinline__ float T_to_float<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}
template <>
__device__ __forceinline__ __nv_bfloat16 float_to_T<__nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}
#endif

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
    void const *cos_ptr,    // [max_seq_len, ROPE_DIM] cos position embeddings (nullptr=skip)
    void const *sin_ptr,    // [max_seq_len, ROPE_DIM] sin position embeddings (nullptr=skip)
    int16_t request_id,
    int qh_idx = -1) {  // -1 = loop over all Q-heads (persistent kernel mode)

  // Determine head range: qh_idx=-1 means loop all, qh_idx>=0 means single head
  int qh_start = (qh_idx < 0) ? 0 : qh_idx;
  int qh_end = (qh_idx < 0) ? NUM_Q_HEADS : (qh_idx + 1);

  for (int _qh = qh_start; _qh < qh_end; _qh++) {
  // _qh is the current Q-head index for this iteration

  constexpr int BARRIER_ID = 6;
  cutlass::arch::NamedBarrier wg_barrier(NUM_THREADS, BARRIER_ID);

  if (threadIdx.x >= NUM_THREADS) return;

  // ---- Constants ----
  constexpr int Q_STRIDE = NUM_Q_HEADS * QK_HEAD_DIM;
  constexpr int O_STRIDE = NUM_Q_HEADS * V_HEAD_DIM;
  constexpr int MAX_PAGES = (MAX_SEQ_LEN + PAGE_SIZE - 1) / PAGE_SIZE;
  constexpr int PIPE_STAGES = 2;

  // Vectorized load: 8 bf16 = 128 bits = uint4
  constexpr int VEC = 8;
  constexpr int QK_VEC = QK_HEAD_DIM / VEC;   // 72
  constexpr int V_VEC = V_HEAD_DIM / VEC;     // 64

  // MMA iteration counts
  constexpr int MMA_M = (MAX_TOKENS + 15) / 16;              // 1
  constexpr int MMA_N_QK = (KV_TILE_SIZE + 15) / 16;         // 2
  constexpr int MMA_K_QK = (QK_HEAD_DIM + 15) / 16;          // 36
  constexpr int MMA_N_PV = (V_HEAD_DIM + 15) / 16;           // 32
  constexpr int MMA_K_PV = (KV_TILE_SIZE + 15) / 16;         // 2

  float const sm_scale = 1.0f / sqrtf(static_cast<float>(QK_HEAD_DIM));

  int warp_idx = warp_id();
  int lane_idx = lane_id();

  // ---- Request metadata ----
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

  // ---- Shared memory layout ----
  // Q:  [MAX_TOKENS * QK_HEAD_DIM]       (combined nope+pe, loaded once)
  // K:  [2 * KV_TILE_SIZE * QK_HEAD_DIM] (double-buffer, combined 576-wide)
  // V:  [2 * KV_TILE_SIZE * V_HEAD_DIM]  (double-buffer, 512-wide)
  __shared__ int s_page_indices[MAX_PAGES];
  __shared__ __align__(16) T s_ldmatrix_zeros[16];

  constexpr int SQ = MAX_TOKENS * QK_HEAD_DIM;
  constexpr int SK = KV_TILE_SIZE * QK_HEAD_DIM;
  constexpr int SV = KV_TILE_SIZE * V_HEAD_DIM;

  extern __shared__ char smem[];
  T *s_q = reinterpret_cast<T *>(smem);                          // [MAX_TOKENS][QK_HEAD_DIM]
  T *s_k = s_q + SQ;                                            // [2][KV_TILE_SIZE][QK_HEAD_DIM]
  T *s_v = s_k + PIPE_STAGES * SK;                              // [2][KV_TILE_SIZE][V_HEAD_DIM]

  // ---- Load page indices + zero buffer ----
  for (int i = threadIdx.x; i < num_pages; i += NUM_THREADS) {
    s_page_indices[i] = paged_kv_indices_buffer_ptr[first_page + i];
  }
  if (threadIdx.x < 16) {
    s_ldmatrix_zeros[threadIdx.x] = float_to_T<T>(0.f);
  }
  wg_barrier.arrive_and_wait();

  // ---- Pointers ----
  T const *__restrict__ d_q =
      reinterpret_cast<T const *>(q_nope_pe_ptr) + first_token * Q_STRIDE;
  T const *__restrict__ d_c_new =
      reinterpret_cast<T const *>(c_latent_new_ptr) + first_token * V_HEAD_DIM;
  T const *__restrict__ d_k_pe_new =
      reinterpret_cast<T const *>(k_pe_new_ptr) +
      first_token * (QK_HEAD_DIM - V_HEAD_DIM);
  T *__restrict__ d_cache = reinterpret_cast<T *>(ckv_kpe_cache_ptr);
  T *__restrict__ d_output =
      reinterpret_cast<T *>(output_ptr) + first_token * O_STRIDE;

  // ---- RoPE cos/sin pointers ----
  constexpr int ROPE_DIM = QK_HEAD_DIM - V_HEAD_DIM;
  constexpr int ROPE_HALF = ROPE_DIM / 2;
  T const *__restrict__ d_cos = (cos_ptr != nullptr)
      ? reinterpret_cast<T const *>(cos_ptr) : nullptr;
  T const *__restrict__ d_sin = (sin_ptr != nullptr)
      ? reinterpret_cast<T const *>(sin_ptr) : nullptr;

  // ---- Phase 0: Write new KV to cache with RoPE on k_pe (head-0 block only) ----
  if (_qh == 0) {
    for (int idx = threadIdx.x; idx < num_tokens * QK_HEAD_DIM;
         idx += NUM_THREADS) {
      int t = idx / QK_HEAD_DIM;
      int d = idx % QK_HEAD_DIM;
      int cache_pos = seq_len - num_tokens + t;
      int pi = s_page_indices[cache_pos / PAGE_SIZE];
      int po = cache_pos % PAGE_SIZE;
      T val;
      if (d < V_HEAD_DIM) {
        val = d_c_new[t * V_HEAD_DIM + d];
      } else {
        // Apply RoPE to k_pe before writing to cache
        int pe_idx = d - V_HEAD_DIM;  // 0..ROPE_DIM-1
        float k_val = T_to_float<T>(d_k_pe_new[t * ROPE_DIM + pe_idx]);
        if (d_cos != nullptr) {
          int pair_idx = pe_idx % ROPE_HALF;
          float cos_v = T_to_float<T>(d_cos[cache_pos * ROPE_DIM + pe_idx]);
          float sin_v = T_to_float<T>(d_sin[cache_pos * ROPE_DIM + pe_idx]);
          float k_pair;
          if (pe_idx < ROPE_HALF) {
            k_pair = T_to_float<T>(d_k_pe_new[t * ROPE_DIM + pe_idx + ROPE_HALF]);
            k_val = k_val * cos_v - k_pair * sin_v;
          } else {
            k_pair = T_to_float<T>(d_k_pe_new[t * ROPE_DIM + pe_idx - ROPE_HALF]);
            k_val = k_pair * sin_v + k_val * cos_v;
          }
        }
        val = float_to_T<T>(k_val);
      }
      d_cache[(pi * PAGE_SIZE + po) * QK_HEAD_DIM + d] = val;
    }
  }
  wg_barrier.arrive_and_wait();

  // ---- Phase 1: Load Q (combined 576-wide, all threads, vectorized) ----
  for (int idx = threadIdx.x; idx < num_tokens * QK_VEC; idx += NUM_THREADS) {
    int t = idx / QK_VEC;
    int vc = idx % QK_VEC;
    reinterpret_cast<uint4 *>(s_q + t * QK_HEAD_DIM)[vc] =
        reinterpret_cast<const uint4 *>(
            d_q + t * Q_STRIDE + _qh * QK_HEAD_DIM)[vc];
  }
  wg_barrier.arrive_and_wait();

  // ---- Phase 1b: Apply RoPE to q_pe (last ROPE_DIM dims in smem) ----
  if (d_cos != nullptr) {
    for (int idx = threadIdx.x; idx < num_tokens * ROPE_DIM;
         idx += NUM_THREADS) {
      int t = idx / ROPE_DIM;
      int pe_idx = idx % ROPE_DIM;
      int smem_offset = t * QK_HEAD_DIM + V_HEAD_DIM + pe_idx;
      float q_val = T_to_float<T>(s_q[smem_offset]);
      int seq_pos = seq_len - num_tokens + t;
      float cos_v = T_to_float<T>(d_cos[seq_pos * ROPE_DIM + pe_idx]);
      float sin_v = T_to_float<T>(d_sin[seq_pos * ROPE_DIM + pe_idx]);
      float q_pair;
      if (pe_idx < ROPE_HALF) {
        q_pair = T_to_float<T>(s_q[smem_offset + ROPE_HALF]);
        s_q[smem_offset] = float_to_T<T>(q_val * cos_v - q_pair * sin_v);
      } else {
        q_pair = T_to_float<T>(s_q[smem_offset - ROPE_HALF]);
        s_q[smem_offset] = float_to_T<T>(q_pair * sin_v + q_val * cos_v);
      }
    }
    wg_barrier.arrive_and_wait();
  }

  // ---- Accumulators (warp 0 only, but declared for all to avoid divergence) ----
  float m_local[MMA_M][2];
  float d_acc[MMA_M][2];
  float o_acc[MMA_M][MMA_N_PV][8];
#pragma unroll
  for (int m = 0; m < MMA_M; m++) {
    m_local[m][0] = -INFINITY;
    m_local[m][1] = -INFINITY;
    d_acc[m][0] = 1.f;
    d_acc[m][1] = 1.f;
#pragma unroll
    for (int n = 0; n < MMA_N_PV; n++) {
      clear_8_floats(o_acc[m][n]);
    }
  }

  // ---- KV loading lambda (all 128 threads cooperate) ----
  // Loads combined K (576-wide) and separate V (512-wide) for a tile
  auto load_kv_tile = [&](int kv_start, int tile_len, int buf) {
    T *k_dst = s_k + buf * SK;
    T *v_dst = s_v + buf * SV;
    // Load K: tile_len rows x QK_HEAD_DIM cols
    for (int idx = threadIdx.x; idx < tile_len * QK_VEC; idx += NUM_THREADS) {
      int row = idx / QK_VEC, vc = idx % QK_VEC;
      int cp = kv_start + row;
      int pi = s_page_indices[cp / PAGE_SIZE];
      int po = cp % PAGE_SIZE;
      reinterpret_cast<uint4 *>(k_dst + row * QK_HEAD_DIM)[vc] =
          reinterpret_cast<const uint4 *>(
              d_cache + (pi * PAGE_SIZE + po) * QK_HEAD_DIM)[vc];
    }
    // Load V: tile_len rows x V_HEAD_DIM cols (first 512 of each cache row)
    for (int idx = threadIdx.x; idx < tile_len * V_VEC; idx += NUM_THREADS) {
      int row = idx / V_VEC, vc = idx % V_VEC;
      int cp = kv_start + row;
      int pi = s_page_indices[cp / PAGE_SIZE];
      int po = cp % PAGE_SIZE;
      reinterpret_cast<uint4 *>(v_dst + row * V_HEAD_DIM)[vc] =
          reinterpret_cast<const uint4 *>(
              d_cache + (pi * PAGE_SIZE + po) * QK_HEAD_DIM)[vc];
    }
  };

  // Load first tile
  int first_kv_len = min(seq_len, KV_TILE_SIZE);
  load_kv_tile(0, first_kv_len, 0);
  int kv_loaded = first_kv_len;
  int stage = 0;

  // ---- Phase 2: KV tile loop ----
  for (int kv_iter = 0; kv_iter < num_kv_iters; kv_iter++) {
    int curr_kv_len = min(seq_len - kv_iter * KV_TILE_SIZE, KV_TILE_SIZE);
    int next_buf = 1 - stage;

    // Prefetch next tile into other buffer (all threads)
    if (kv_loaded < seq_len) {
      int next_len = min(seq_len - kv_loaded, KV_TILE_SIZE);
      load_kv_tile(kv_loaded, next_len, next_buf);
      kv_loaded += next_len;
    }
    wg_barrier.arrive_and_wait();

    // ---- Warps 1-3: help load, then skip MMA ----
    if (warp_idx != 0) {
      wg_barrier.arrive_and_wait();
      stage = next_buf;
      continue;
    }

    // ==== Warp 0: QK^T MMA ====
    float x_frag_f[MMA_M][MMA_N_QK][8];
#pragma unroll
    for (int m = 0; m < MMA_M; m++)
#pragma unroll
      for (int n = 0; n < MMA_N_QK; n++)
        clear_8_floats(x_frag_f[m][n]);

    T *my_k = s_k + stage * SK;
    T *my_v = s_v + stage * SV;

    // Q * K^T  (combined 576-wide)
    uint32_t a_frag[4], b_frag[4];
#pragma unroll
    for (int m = 0; m < MMA_M; m++) {
#pragma unroll
      for (int n = 0; n < MMA_N_QK; n++) {
#pragma unroll
        for (int k = 0; k < MMA_K_QK; k++) {
          int q_row = m * 16 + (lane_idx & 0xF);
          int q_col = k * 16 + ((lane_idx >> 4) << 3);
          T *src_a = q_row < num_tokens
                         ? s_q + q_row * QK_HEAD_DIM + q_col
                         : s_ldmatrix_zeros;

          int kt_col = n * 16 + ((lane_idx >> 4) << 3) + (lane_idx & 0x7);
          int kt_row = k * 16 + (((lane_idx & 0xF) >> 3) << 3);
          T *src_b = kt_col < curr_kv_len
                         ? my_k + kt_col * QK_HEAD_DIM + kt_row
                         : s_ldmatrix_zeros;

          ldsm(src_a, a_frag);
          ldsm(src_b, b_frag);
          mma_m16n16k16_bf16bf16bf32(
              x_frag_f[m][n], a_frag, b_frag, x_frag_f[m][n]);
        }
      }
    }

    // ==== Online softmax (warp 0) ====
    float m_prev[MMA_M][2];
#pragma unroll
    for (int m = 0; m < MMA_M; m++) {
      m_prev[m][0] = m_local[m][0];
      m_prev[m][1] = m_local[m][1];
#pragma unroll
      for (int n = 0; n < MMA_N_QK; n++) {
#pragma unroll
        for (int fi = 0; fi < 8; fi++) {
          int row = (m << 4) + (lane_idx >> 2) + (((fi & 3) >> 1) << 3);
          int col =
              n * 16 + ((lane_idx & 3) << 1) + ((fi >> 2) << 3) + (fi & 1);
          bool valid =
              (row < num_tokens) &&
              (col + kv_iter * KV_TILE_SIZE <=
               row + seq_len - num_tokens);
          x_frag_f[m][n][fi] = valid ? x_frag_f[m][n][fi] : -INFINITY;
          m_local[m][(fi & 3) >> 1] =
              max(m_local[m][(fi & 3) >> 1], x_frag_f[m][n][fi]);
        }
      }
      // Warp-level max reduction via shfl_xor
      m_local[m][0] =
          max(m_local[m][0],
              __shfl_xor_sync(0xFFFFFFFF, m_local[m][0], 0x1));
      m_local[m][0] =
          max(m_local[m][0],
              __shfl_xor_sync(0xFFFFFFFF, m_local[m][0], 0x2));
      m_local[m][1] =
          max(m_local[m][1],
              __shfl_xor_sync(0xFFFFFFFF, m_local[m][1], 0x1));
      m_local[m][1] =
          max(m_local[m][1],
              __shfl_xor_sync(0xFFFFFFFF, m_local[m][1], 0x2));
    }

    // Rescale previous output accumulator
    float rescale[MMA_M][2];
#pragma unroll
    for (int m = 0; m < MMA_M; m++) {
      rescale[m][0] =
          expf(m_prev[m][0] * sm_scale - m_local[m][0] * sm_scale);
      rescale[m][1] =
          expf(m_prev[m][1] * sm_scale - m_local[m][1] * sm_scale);
    }

    // Compute exp and sum for denominator
    float d_partial[MMA_M][2];
#pragma unroll
    for (int m = 0; m < MMA_M; m++) {
      d_partial[m][0] = 0.f;
      d_partial[m][1] = 0.f;
#pragma unroll
      for (int n = 0; n < MMA_N_QK; n++) {
#pragma unroll
        for (int fi = 0; fi < 8; fi++) {
          x_frag_f[m][n][fi] =
              x_frag_f[m][n][fi] != -INFINITY
                  ? expf(x_frag_f[m][n][fi] * sm_scale -
                         m_local[m][(fi & 3) >> 1] * sm_scale)
                  : 0.f;
          d_partial[m][(fi & 3) >> 1] += x_frag_f[m][n][fi];
        }
      }
      d_partial[m][0] +=
          __shfl_xor_sync(0xFFFFFFFF, d_partial[m][0], 0x1);
      d_partial[m][0] +=
          __shfl_xor_sync(0xFFFFFFFF, d_partial[m][0], 0x2);
      d_partial[m][1] +=
          __shfl_xor_sync(0xFFFFFFFF, d_partial[m][1], 0x1);
      d_partial[m][1] +=
          __shfl_xor_sync(0xFFFFFFFF, d_partial[m][1], 0x2);
      d_acc[m][0] = d_acc[m][0] * rescale[m][0] + d_partial[m][0];
      d_acc[m][1] = d_acc[m][1] * rescale[m][1] + d_partial[m][1];
    }

    // Rescale previous o_acc
#pragma unroll
    for (int m = 0; m < MMA_M; m++)
#pragma unroll
      for (int n = 0; n < MMA_N_PV; n++)
#pragma unroll
        for (int fi = 0; fi < 8; fi++)
          o_acc[m][n][fi] *= rescale[m][(fi & 3) >> 1];

    // ==== PV MMA: attention probs (in registers) x V (from smem) ====
    // P is [MAX_TOKENS x KV_TILE_SIZE] in x_frag_f registers
    // V is [KV_TILE_SIZE x V_HEAD_DIM] in s_v
    // We convert P fragments to bf16 via convert_f32_to_bf16_uint32
#pragma unroll
    for (int m = 0; m < MMA_M; m++) {
#pragma unroll
      for (int kk = 0; kk < MMA_K_PV; kk++) {
        // Convert the P fragment from float to bf16 uint32 for MMA A-operand
        uint32_t p_frag[4];
        convert_f32_to_bf16_uint32(x_frag_f[m][kk], p_frag);

#pragma unroll
        for (int nn = 0; nn < MMA_N_PV; nn++) {
          // Load V fragment from smem via ldsm_t (B-operand, transposed)
          uint32_t v_frag[4];
          int v_row = kk * 16 + (lane_idx & 0xF);
          int v_col = nn * 16 + ((lane_idx >> 4) << 3);
          T *src_v = v_row < curr_kv_len
                         ? &my_v[v_row * V_HEAD_DIM + v_col]
                         : s_ldmatrix_zeros;
          ldsm_t(src_v, v_frag);
          mma_m16n16k16_bf16bf16bf32(
              o_acc[m][nn], p_frag, v_frag, o_acc[m][nn]);
        }
      }
    }

    wg_barrier.arrive_and_wait();
    stage = next_buf;
  } // end KV tile loop

  // ---- Phase 3: Output normalization + write (warp 0 only) ----
  if (warp_idx != 0) return;

#pragma unroll
  for (int mma_m = 0; mma_m < MMA_M; mma_m++) {
#pragma unroll
    for (int mma_n = 0; mma_n < MMA_N_PV; mma_n++) {
#pragma unroll
      for (int fi = 0; fi < 8; fi++) {
        int row_local = (lane_idx / 4) + (((fi >> 1) & 1) * 8);
        int col_local =
            ((fi >> 2) & 1) * 8 + (lane_idx % 4) * 2 + (fi & 1);
        int row = mma_m * 16 + row_local;
        int col = mma_n * 16 + col_local;
        if (row >= num_tokens || col >= V_HEAD_DIM) continue;
        float d_val = d_acc[mma_m][(fi & 3) >> 1];
        float o_val = o_acc[mma_m][mma_n][fi];
        o_val = (d_val > 0.f) ? (o_val / d_val) : 0.f;
        d_output[row * O_STRIDE + _qh * V_HEAD_DIM + col] =
            float_to_T<T>(o_val);
      }
    }
  }
  } // end Q-head loop
}

} // namespace kernel
