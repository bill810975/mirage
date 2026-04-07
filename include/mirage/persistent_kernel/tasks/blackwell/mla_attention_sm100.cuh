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

#pragma once
#include "tasks/ampere/mma.cuh"
#include "tasks/ampere/smem_layout.cuh"
#include "tasks/common/common_header.cuh"

#include <cutlass/arch/barrier.h>

namespace kernel {

// MLA (Multi-head Latent Attention) paged attention for DeepSeek V3.
//
// MMA-based implementation using independent per-warp computation.
// Each warp handles one Q head independently — no cross-warp reduction
// needed, eliminating the S_O_BUFFER bottleneck.
//
// Design:
// - 4 warps, each processes 1 Q head at a time (4 heads in parallel)
// - Outer loop tiles over Q heads: NUM_Q_HEADS / 4 passes
// - Inner loop tiles over KV sequence: seq_len / KV_TILE_SIZE passes
// - QK matmul: m16n16k16 MMA with online softmax
// - PV matmul: m16n16k16 MMA accumulating output
// - Q in shared memory, K/V double-buffered in shared memory
// - Output in registers, written to global memory per Q-head pass
//
// Shared memory: ~177KB for KV_TILE=32 (fits in SM100's 228KB)
//   S_Q:  4 × MAX_TOKENS × QK_HEAD_DIM × 2 = 37KB
//   S_K:  2 × KV_TILE × QK_HEAD_DIM × 2    = 74KB
//   S_V:  2 × KV_TILE × V_HEAD_DIM × 2     = 66KB
//
// Key properties:
// - QK uses QK_HEAD_DIM (576 = 512 latent + 64 rope)
// - V/output uses V_HEAD_DIM (512, latent only)
// - Single KV head (MQA after weight absorption)
// - Separate c_latent and k_pe inputs (combined when writing to cache)
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
  constexpr int NUM_WARPS = NUM_THREADS / 32; // 4
  constexpr int ROPE_DIM = QK_HEAD_DIM - V_HEAD_DIM;
  constexpr int Q_STRIDE = NUM_Q_HEADS * QK_HEAD_DIM;
  constexpr int O_STRIDE = NUM_Q_HEADS * V_HEAD_DIM;
  constexpr int MAX_PAGES = (MAX_SEQ_LEN + PAGE_SIZE - 1) / PAGE_SIZE;
  constexpr int CP_CHUNK_SIZE = 16 / sizeof(T);
  // Heads processed per pass: one per warp
  constexpr int QH_PER_PASS = NUM_WARPS; // 4
  constexpr int NUM_QH_PASSES = (NUM_Q_HEADS + QH_PER_PASS - 1) / QH_PER_PASS;
  // MMA dimensions: each warp handles MAX_TOKENS Q rows
  constexpr int MMA_ITERS_M = (MAX_TOKENS + 15) / 16; // 1 for MAX_TOKENS=8
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
  if (num_tokens == 0) {
    return;
  }

  int const first_page = paged_kv_indptr_buffer_ptr[request_id];
  int const last_page = paged_kv_indptr_buffer_ptr[request_id + 1];
  int const num_pages = last_page - first_page;
  int const seq_len =
      (num_pages - 1) * PAGE_SIZE +
      paged_kv_last_page_len_buffer_ptr[request_id];
  int const num_kv_iters = (seq_len + KV_TILE_SIZE - 1) / KV_TILE_SIZE;

  // Load page indices into shared memory
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

  // ===== Shared memory layout =====
  // Q: [QH_PER_PASS, MAX_TOKENS, QK_HEAD_DIM] — each warp's Q head
  constexpr size_t S_Q_OFFSET = 0;
  constexpr size_t S_Q_SIZE =
      sizeof(T) * QH_PER_PASS * MAX_TOKENS * QK_HEAD_DIM;

  // K double buffer: [KV_TILE_SIZE, QK_HEAD_DIM]
  constexpr size_t S_K0_OFFSET = S_Q_OFFSET + S_Q_SIZE;
  constexpr size_t S_K_SIZE = sizeof(T) * KV_TILE_SIZE * QK_HEAD_DIM;
  constexpr size_t S_K1_OFFSET = S_K0_OFFSET + S_K_SIZE;

  // V double buffer: [KV_TILE_SIZE, V_HEAD_DIM]
  constexpr size_t S_V0_OFFSET = S_K1_OFFSET + S_K_SIZE;
  constexpr size_t S_V_SIZE = sizeof(T) * KV_TILE_SIZE * V_HEAD_DIM;
  constexpr size_t S_V1_OFFSET = S_V0_OFFSET + S_V_SIZE;

  constexpr size_t S_TOTAL = S_V1_OFFSET + S_V_SIZE;
  static_assert(S_TOTAL <= mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE,
                "MLA attention exceeds shared memory limit");

  extern __shared__ char smem[];
  T *s_q = reinterpret_cast<T *>(smem + S_Q_OFFSET);
  T *s_k[2] = {reinterpret_cast<T *>(smem + S_K0_OFFSET),
                reinterpret_cast<T *>(smem + S_K1_OFFSET)};
  T *s_v[2] = {reinterpret_cast<T *>(smem + S_V0_OFFSET),
                reinterpret_cast<T *>(smem + S_V1_OFFSET)};

  // Smem row layouts for MMA
  using QSmem = smem_row<T, 3, 3, 3, MAX_TOKENS, QK_HEAD_DIM, QK_HEAD_DIM>;
  using KSmem = smem_row<T, 3, 3, 3, KV_TILE_SIZE, QK_HEAD_DIM, QK_HEAD_DIM>;
  using VSmem = smem_row<T, 3, 3, 3, KV_TILE_SIZE, V_HEAD_DIM, V_HEAD_DIM>;

  T zero_val = T(0);

  // ===== Phase 1: Write new KV entries to cache =====
  for (int idx = threadIdx.x; idx < num_tokens * QK_HEAD_DIM;
       idx += NUM_THREADS) {
    int t = idx / QK_HEAD_DIM;
    int d = idx % QK_HEAD_DIM;
    int cache_pos = seq_len - num_tokens + t;
    int page_idx = s_page_indices[cache_pos / PAGE_SIZE];
    int page_off = cache_pos % PAGE_SIZE;
    T val;
    if (d < V_HEAD_DIM) {
      val = d_c_new[t * V_HEAD_DIM + d];
    } else {
      val = d_k_pe_new[t * ROPE_DIM + (d - V_HEAD_DIM)];
    }
    d_cache[(page_idx * PAGE_SIZE + page_off) * QK_HEAD_DIM + d] = val;
  }
  wg_barrier.arrive_and_wait();

  // Helper: load a KV tile from paged cache into shared memory
  auto load_kv_tile = [&](int kv_start, int kv_len, int buf_idx) {
    T *sk = s_k[buf_idx];
    T *sv = s_v[buf_idx];
    // Load K (full QK_HEAD_DIM)
    for (int idx = threadIdx.x; idx < kv_len * QK_HEAD_DIM / CP_CHUNK_SIZE;
         idx += NUM_THREADS) {
      int row = (idx * CP_CHUNK_SIZE) / QK_HEAD_DIM;
      int col = (idx * CP_CHUNK_SIZE) % QK_HEAD_DIM;
      int cache_pos = kv_start + row;
      int page_idx = s_page_indices[cache_pos / PAGE_SIZE];
      int page_off = cache_pos % PAGE_SIZE;
      T const *src =
          d_cache + (page_idx * PAGE_SIZE + page_off) * QK_HEAD_DIM + col;
      T *dst = sk + row * QK_HEAD_DIM + col;
#pragma unroll
      for (int c = 0; c < CP_CHUNK_SIZE; c++) {
        dst[c] = src[c];
      }
    }
    // Load V (first V_HEAD_DIM dims only)
    for (int idx = threadIdx.x; idx < kv_len * V_HEAD_DIM / CP_CHUNK_SIZE;
         idx += NUM_THREADS) {
      int row = (idx * CP_CHUNK_SIZE) / V_HEAD_DIM;
      int col = (idx * CP_CHUNK_SIZE) % V_HEAD_DIM;
      int cache_pos = kv_start + row;
      int page_idx = s_page_indices[cache_pos / PAGE_SIZE];
      int page_off = cache_pos % PAGE_SIZE;
      T const *src =
          d_cache + (page_idx * PAGE_SIZE + page_off) * QK_HEAD_DIM + col;
      T *dst = sv + row * V_HEAD_DIM + col;
#pragma unroll
      for (int c = 0; c < CP_CHUNK_SIZE; c++) {
        dst[c] = src[c];
      }
    }
  };

  // ===== Phase 2: Attention computation =====
  // Outer loop over Q-head groups
  for (int qh_pass = 0; qh_pass < NUM_QH_PASSES; qh_pass++) {
    int qh_base = qh_pass * QH_PER_PASS;
    int my_qh = qh_base + warp_idx; // This warp's Q head index
    bool warp_active = (my_qh < NUM_Q_HEADS);

    // Load this warp's Q data into shared memory
    if (warp_active) {
      T *my_sq = s_q + warp_idx * MAX_TOKENS * QK_HEAD_DIM;
      for (int idx = lane_idx; idx < num_tokens * QK_HEAD_DIM; idx += 32) {
        int t = idx / QK_HEAD_DIM;
        int d = idx % QK_HEAD_DIM;
        my_sq[t * QK_HEAD_DIM + d] =
            d_q[t * Q_STRIDE + my_qh * QK_HEAD_DIM + d];
      }
    }
    wg_barrier.arrive_and_wait();

    // Per-warp online softmax state (in registers)
    float m_local[MMA_ITERS_M][2];
    float d_acc[MMA_ITERS_M][2];
    float o_acc[MMA_ITERS_M][MMA_ITERS_N_PV][8];

#pragma unroll
    for (int m = 0; m < MMA_ITERS_M; m++) {
      m_local[m][0] = -INFINITY;
      m_local[m][1] = -INFINITY;
      d_acc[m][0] = 1.f;
      d_acc[m][1] = 1.f;
#pragma unroll
      for (int n = 0; n < MMA_ITERS_N_PV; n++) {
        clear_8_floats(o_acc[m][n]);
      }
    }

    // Prefetch first KV tile
    int kv_loaded = 0;
    int first_kv_len = min(seq_len, KV_TILE_SIZE);
    load_kv_tile(0, first_kv_len, 0);
    cp_async_fence();
    kv_loaded = first_kv_len;

    int curr_kv_len = first_kv_len;
    int curr_buf = 0;

    // Inner loop over KV tiles
    for (int kv_iter = 0; kv_iter < num_kv_iters; kv_iter++) {
      // Prefetch next KV tile
      int next_buf = 1 - curr_buf;
      int next_kv_len = 0;
      if (kv_iter + 1 < num_kv_iters) {
        next_kv_len = min(seq_len - kv_loaded, KV_TILE_SIZE);
        load_kv_tile(kv_loaded, next_kv_len, next_buf);
        cp_async_fence();
        cp_async_wait<1>();
        kv_loaded += next_kv_len;
      } else {
        cp_async_wait<0>();
      }
      wg_barrier.arrive_and_wait();

      if (!warp_active) {
        curr_buf = next_buf;
        curr_kv_len = next_kv_len;
        wg_barrier.arrive_and_wait();
        continue;
      }

      // Setup smem pointers for current warp
      QSmem q_smem(s_q + warp_idx * MAX_TOKENS * QK_HEAD_DIM);
      KSmem k_smem(s_k[curr_buf]);
      VSmem v_smem(s_v[curr_buf]);

      // ---- QK^T computation ----
      // Each warp independently computes: Q[MAX_TOKENS, QK_HEAD_DIM] @ K[KV_TILE, QK_HEAD_DIM]^T
      // MMA layout per warp: iterate over M (Q rows), N (KV cols), K (head dim)
      // Since we have 1 warp, the warp tiles across N in m16n16k16 blocks

      float x_frag_f[MMA_ITERS_M][MMA_ITERS_N_QK][8];
#pragma unroll
      for (int m = 0; m < MMA_ITERS_M; m++) {
#pragma unroll
        for (int n = 0; n < MMA_ITERS_N_QK; n++) {
          clear_8_floats(x_frag_f[m][n]);
        }
      }

      uint32_t q_frag[4], kt_frag[4];

#pragma unroll
      for (int m = 0; m < MMA_ITERS_M; m++) {
#pragma unroll
        for (int n = 0; n < MMA_ITERS_N_QK; n++) {
          int q_row_base = m * 16;
          int kt_col_base = n * 16;

#pragma unroll
          for (int k = 0; k < MMA_ITERS_K_QK; k++) {
            int q_row = q_row_base + (lane_idx & 0xF);
            int q_col = k * 16 + ((lane_idx >> 4) << 3);
            T *src_q = q_row < num_tokens ? q_smem(q_row, q_col) : &zero_val;

            int kt_col = kt_col_base + ((lane_idx >> 4) << 3) + (lane_idx & 0x7);
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
      // Flatten N dimension for softmax across all KV positions in this tile
      // Each m16n16k16 produces an 8-element fragment; we need max/sum across
      // all N tiles for each M row.

      float m_prev[MMA_ITERS_M][2];
#pragma unroll
      for (int m = 0; m < MMA_ITERS_M; m++) {
        m_prev[m][0] = m_local[m][0];
        m_prev[m][1] = m_local[m][1];

        // Update max across all N tiles
#pragma unroll
        for (int n = 0; n < MMA_ITERS_N_QK; n++) {
#pragma unroll
          for (int fi = 0; fi < 8; fi++) {
            int row = (m << 4) + (lane_idx >> 2) + (((fi & 0x3) >> 1) << 3);
            int col = n * 16 + ((lane_idx & 0x3) << 1) +
                      ((fi >> 2) << 3) + (fi & 0x1);
            int token_idx = row;
            bool is_valid =
                (row < num_tokens) &&
                (col + kv_iter * KV_TILE_SIZE <=
                 token_idx + seq_len - num_tokens);
            x_frag_f[m][n][fi] = is_valid ? x_frag_f[m][n][fi] : -INFINITY;
            m_local[m][(fi & 0x3) >> 1] =
                max(m_local[m][(fi & 0x3) >> 1], x_frag_f[m][n][fi]);
          }
        }
        // Reduce max across 4 lanes within the warp (for the 4 columns each
        // lane handles)
        m_local[m][0] =
            max(m_local[m][0], __shfl_xor_sync(0xffffffff, m_local[m][0], 0x1));
        m_local[m][0] =
            max(m_local[m][0], __shfl_xor_sync(0xffffffff, m_local[m][0], 0x2));
        m_local[m][1] =
            max(m_local[m][1], __shfl_xor_sync(0xffffffff, m_local[m][1], 0x1));
        m_local[m][1] =
            max(m_local[m][1], __shfl_xor_sync(0xffffffff, m_local[m][1], 0x2));
      }

      // Rescale factors
      float rescale[MMA_ITERS_M][2];
#pragma unroll
      for (int m = 0; m < MMA_ITERS_M; m++) {
        rescale[m][0] =
            expf(m_prev[m][0] * sm_scale - m_local[m][0] * sm_scale);
        rescale[m][1] =
            expf(m_prev[m][1] * sm_scale - m_local[m][1] * sm_scale);
      }

      // Compute exp and partial sum
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
                x_frag_f[m][n][fi] != -INFINITY
                    ? expf(x_frag_f[m][n][fi] * sm_scale -
                           m_local[m][(fi & 0x3) >> 1] * sm_scale)
                    : 0.f;
            d_partial[m][(fi & 0x3) >> 1] += x_frag_f[m][n][fi];
          }
        }
        d_partial[m][0] +=
            __shfl_xor_sync(0xffffffff, d_partial[m][0], 0x1);
        d_partial[m][0] +=
            __shfl_xor_sync(0xffffffff, d_partial[m][0], 0x2);
        d_partial[m][1] +=
            __shfl_xor_sync(0xffffffff, d_partial[m][1], 0x1);
        d_partial[m][1] +=
            __shfl_xor_sync(0xffffffff, d_partial[m][1], 0x2);
        d_acc[m][0] *= rescale[m][0];
        d_acc[m][1] *= rescale[m][1];
        d_acc[m][0] += d_partial[m][0];
        d_acc[m][1] += d_partial[m][1];
      }

      // Rescale accumulated output
#pragma unroll
      for (int m = 0; m < MMA_ITERS_M; m++) {
#pragma unroll
        for (int n = 0; n < MMA_ITERS_N_PV; n++) {
#pragma unroll
          for (int fi = 0; fi < 8; fi++) {
            o_acc[m][n][fi] *= rescale[m][(fi & 0x3) >> 1];
          }
        }
      }

      // ---- PV computation ----
      // P[MAX_TOKENS, KV_TILE] @ V[KV_TILE, V_HEAD_DIM]
      // We need to convert x_frag_f (the softmax output) to bf16 for MMA input.
      // The P matrix is spread across MMA_ITERS_N_QK n-tiles.
      // For PV MMA: M = MAX_TOKENS, N = V_HEAD_DIM, K = KV_TILE_SIZE
      // MMA iterates: M iters × N iters × K iters

      // For each KV_TILE n-tile in P (which becomes a k-tile in PV):
#pragma unroll
      for (int m = 0; m < MMA_ITERS_M; m++) {
#pragma unroll
        for (int kk = 0; kk < MMA_ITERS_K_PV; kk++) {
          // kk-th K tile of PV = kk-th N tile of QK
          uint32_t p_frag[4];
          convert_f32_to_bf16_uint32(x_frag_f[m][kk], p_frag);

#pragma unroll
          for (int nn = 0; nn < MMA_ITERS_N_PV; nn++) {
            uint32_t v_frag[4];
            // V row = KV position within tile, V col = output dimension
            // For MMA K dimension: V is transposed
            int v_row = kk * 16 + (lane_idx & 0xF);
            int v_col = nn * 16 + ((lane_idx >> 4) << 3);
            T *src_v =
                v_row < curr_kv_len ? &s_v[curr_buf][v_row * V_HEAD_DIM + v_col]
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
    } // end KV loop

    // ---- Write output for this Q-head pass ----
    if (warp_active) {
      // Each warp writes its Q head's output to global memory
      // Output layout: [num_tokens, NUM_Q_HEADS * V_HEAD_DIM]
      // This warp's output goes to columns [my_qh * V_HEAD_DIM, (my_qh+1) *
      // V_HEAD_DIM)

      // The output is in o_acc[MMA_ITERS_M][MMA_ITERS_N_PV][8] fragments.
      // We need to normalize by d_acc and write each element.
      // Fragment layout for m16n16k16 output (same as attention_sm100.cuh):
      for (int elem_idx = lane_idx;
           elem_idx < num_tokens * V_HEAD_DIM; elem_idx += 32) {
        int row = elem_idx / V_HEAD_DIM;
        int col = elem_idx % V_HEAD_DIM;

        int mma_m = row / 16;
        int mma_n = col / 16;
        int t_idx = (row % 8) * 4 + (col % 8) / 2;
        int frag_idx =
            ((col % 16) / 8) * 4 + ((row % 16) / 8) * 2 + (col % 2);

        // Since we have 1 warp (no cross-warp reduction),
        // we need to check if this element belongs to our warp's lane.
        // In a single-warp MMA, the fragment mapping is direct.

        // Actually, the fragment is owned by the thread that computed it.
        // We need to extract the right value from o_acc.
        // For a single warp, t_idx maps to the thread within the warp.
        // Only write if this lane owns this fragment element.
        if (t_idx == lane_idx && mma_m < MMA_ITERS_M &&
            mma_n < MMA_ITERS_N_PV) {
          float o_val = o_acc[mma_m][mma_n][frag_idx];
          float d_val = d_acc[mma_m][(frag_idx & 0x3) >> 1];
          if (d_val > 0.f) {
            o_val /= d_val;
          }
          d_output[row * O_STRIDE + my_qh * V_HEAD_DIM + col] = T(o_val);
        }
      }
    }
    wg_barrier.arrive_and_wait();

  } // end QH pass loop
}

} // namespace kernel
