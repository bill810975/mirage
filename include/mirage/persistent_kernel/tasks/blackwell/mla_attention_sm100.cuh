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

  // ---- Accumulators ----
  // Split PV output into PV_CHUNKS to reduce register pressure.
  // o_acc holds only PV_CHUNK_SIZE columns at a time (64 floats vs 256).
  constexpr int PV_CHUNK_SIZE = 8;  // process 8 of 32 MMA_N_PV blocks at a time
  constexpr int PV_CHUNKS = (MMA_N_PV + PV_CHUNK_SIZE - 1) / PV_CHUNK_SIZE;  // 4

  float m_local[MMA_M][2];
  float d_acc[MMA_M][2];
  float o_acc[MMA_M][PV_CHUNK_SIZE][8];  // only 64 floats instead of 256!
#pragma unroll
  for (int m = 0; m < MMA_M; m++) {
    m_local[m][0] = -INFINITY;
    m_local[m][1] = -INFINITY;
    d_acc[m][0] = 1.f;
    d_acc[m][1] = 1.f;
#pragma unroll
    for (int n = 0; n < PV_CHUNK_SIZE; n++) {
      clear_8_floats(o_acc[m][n]);
    }
  }

  // ---- KV loading lambda ----
  // stride parameter: NUM_THREADS (128) when all warps load, 32 when warp 0 only
  auto load_kv_tile = [&](int kv_start, int tile_len, int buf, int stride = NUM_THREADS) {
    T *k_dst = s_k + buf * SK;
    T *v_dst = s_v + buf * SV;
    int tid = threadIdx.x % stride;  // local thread index within active threads
    for (int idx = tid; idx < tile_len * QK_VEC; idx += stride) {
      int row = idx / QK_VEC, vc = idx % QK_VEC;
      int cp = kv_start + row;
      int pi = s_page_indices[cp / PAGE_SIZE];
      int po = cp % PAGE_SIZE;
      reinterpret_cast<uint4 *>(k_dst + row * QK_HEAD_DIM)[vc] =
          reinterpret_cast<const uint4 *>(
              d_cache + (pi * PAGE_SIZE + po) * QK_HEAD_DIM)[vc];
    }
    // Load V: tile_len rows x V_HEAD_DIM cols (first 512 of each cache row)
    for (int idx = tid; idx < tile_len * V_VEC; idx += stride) {
      int row = idx / V_VEC, vc = idx % V_VEC;
      int cp = kv_start + row;
      int pi = s_page_indices[cp / PAGE_SIZE];
      int po = cp % PAGE_SIZE;
      reinterpret_cast<uint4 *>(v_dst + row * V_HEAD_DIM)[vc] =
          reinterpret_cast<const uint4 *>(
              d_cache + (pi * PAGE_SIZE + po) * QK_HEAD_DIM)[vc];
    }
  };

  // Warps 1-3: skip MMA, jump to end of Q-head iteration
  if (warp_idx != 0) goto end_qh_iter;

  // ---- Phase 2+3 combined: PV chunked, each chunk re-does QK+softmax ----
  // Warp 0 only. Process PV_CHUNK_SIZE (8) output columns at a time.
  // Each chunk re-runs the full KV tile loop (QK + softmax + PV).
  // This trades 4x compute for 4x less register pressure (no spill).

  for (int pv_chunk = 0; pv_chunk < PV_CHUNKS; pv_chunk++) {
    int pv_col_base = pv_chunk * PV_CHUNK_SIZE;

    // Reset accumulators for this chunk
    float chunk_m[MMA_M][2];
    float chunk_d[MMA_M][2];
#pragma unroll
    for (int m = 0; m < MMA_M; m++) {
      chunk_m[m][0] = -INFINITY; chunk_m[m][1] = -INFINITY;
      chunk_d[m][0] = 1.f; chunk_d[m][1] = 1.f;
#pragma unroll
      for (int n = 0; n < PV_CHUNK_SIZE; n++) clear_8_floats(o_acc[m][n]);
    }

    // Re-load first KV tile
    load_kv_tile(0, min(seq_len, KV_TILE_SIZE), 0, 32);
    int pv_kv_loaded = min(seq_len, KV_TILE_SIZE);
    int pv_stage = 0;

    for (int kv_iter = 0; kv_iter < num_kv_iters; kv_iter++) {
      int pv_kv_len = min(seq_len - kv_iter * KV_TILE_SIZE, KV_TILE_SIZE);
      int pv_next_buf = 1 - pv_stage;
      if (pv_kv_loaded < seq_len) {
        int nl = min(seq_len - pv_kv_loaded, KV_TILE_SIZE);
        load_kv_tile(pv_kv_loaded, nl, pv_next_buf, 32);
        pv_kv_loaded += nl;
      }
      __syncwarp();

      T *my_k2 = s_k + pv_stage * SK;
      T *my_v2 = s_v + pv_stage * SV;

      // Recompute QK^T for this tile
      float xf[MMA_M][MMA_N_QK][8];
      uint32_t af[4], bf[4];
#pragma unroll
      for (int m = 0; m < MMA_M; m++)
#pragma unroll
        for (int n = 0; n < MMA_N_QK; n++) clear_8_floats(xf[m][n]);
#pragma unroll
      for (int m = 0; m < MMA_M; m++)
#pragma unroll
        for (int n = 0; n < MMA_N_QK; n++)
#pragma unroll
          for (int k = 0; k < MMA_K_QK; k++) {
            int qr = m*16+(lane_idx&0xF), qc = k*16+((lane_idx>>4)<<3);
            T *sa = qr < num_tokens ? s_q+qr*QK_HEAD_DIM+qc : s_ldmatrix_zeros;
            int kc = n*16+((lane_idx>>4)<<3)+(lane_idx&0x7);
            int kr = k*16+(((lane_idx&0xF)>>3)<<3);
            T *sb = kc < pv_kv_len ? my_k2+kc*QK_HEAD_DIM+kr : s_ldmatrix_zeros;
            ldsm(sa, af); ldsm(sb, bf);
            mma_m16n16k16_bf16bf16bf32(xf[m][n], af, bf, xf[m][n]);
          }

      // Online softmax
      float mp[MMA_M][2];
#pragma unroll
      for (int m = 0; m < MMA_M; m++) {
        mp[m][0] = chunk_m[m][0]; mp[m][1] = chunk_m[m][1];
#pragma unroll
        for (int n = 0; n < MMA_N_QK; n++)
#pragma unroll
          for (int fi = 0; fi < 8; fi++) {
            int row = (m<<4)+(lane_idx>>2)+(((fi&3)>>1)<<3);
            int col = n*16+((lane_idx&3)<<1)+((fi>>2)<<3)+(fi&1);
            bool valid = (row < num_tokens) &&
                         (col+kv_iter*KV_TILE_SIZE <= row+seq_len-num_tokens);
            xf[m][n][fi] = valid ? xf[m][n][fi] : -INFINITY;
            chunk_m[m][(fi&3)>>1] = max(chunk_m[m][(fi&3)>>1], xf[m][n][fi]);
          }
        chunk_m[m][0] = max(chunk_m[m][0], __shfl_xor_sync(0xFFFFFFFF, chunk_m[m][0], 0x1));
        chunk_m[m][0] = max(chunk_m[m][0], __shfl_xor_sync(0xFFFFFFFF, chunk_m[m][0], 0x2));
        chunk_m[m][1] = max(chunk_m[m][1], __shfl_xor_sync(0xFFFFFFFF, chunk_m[m][1], 0x1));
        chunk_m[m][1] = max(chunk_m[m][1], __shfl_xor_sync(0xFFFFFFFF, chunk_m[m][1], 0x2));
      }
      float rsc[MMA_M][2];
#pragma unroll
      for (int m = 0; m < MMA_M; m++) {
        rsc[m][0] = expf(mp[m][0]*sm_scale - chunk_m[m][0]*sm_scale);
        rsc[m][1] = expf(mp[m][1]*sm_scale - chunk_m[m][1]*sm_scale);
      }
      float dp[MMA_M][2];
#pragma unroll
      for (int m = 0; m < MMA_M; m++) {
        dp[m][0] = 0.f; dp[m][1] = 0.f;
#pragma unroll
        for (int n = 0; n < MMA_N_QK; n++)
#pragma unroll
          for (int fi = 0; fi < 8; fi++) {
            xf[m][n][fi] = xf[m][n][fi] != -INFINITY
                ? expf(xf[m][n][fi]*sm_scale - chunk_m[m][(fi&3)>>1]*sm_scale) : 0.f;
            dp[m][(fi&3)>>1] += xf[m][n][fi];
          }
        dp[m][0] += __shfl_xor_sync(0xFFFFFFFF, dp[m][0], 0x1);
        dp[m][0] += __shfl_xor_sync(0xFFFFFFFF, dp[m][0], 0x2);
        dp[m][1] += __shfl_xor_sync(0xFFFFFFFF, dp[m][1], 0x1);
        dp[m][1] += __shfl_xor_sync(0xFFFFFFFF, dp[m][1], 0x2);
        chunk_d[m][0] = chunk_d[m][0]*rsc[m][0] + dp[m][0];
        chunk_d[m][1] = chunk_d[m][1]*rsc[m][1] + dp[m][1];
      }

      // Rescale + PV for this chunk only
#pragma unroll
      for (int m = 0; m < MMA_M; m++)
#pragma unroll
        for (int n = 0; n < PV_CHUNK_SIZE; n++)
#pragma unroll
          for (int fi = 0; fi < 8; fi++)
            o_acc[m][n][fi] *= rsc[m][(fi&3)>>1];

#pragma unroll
      for (int m = 0; m < MMA_M; m++)
#pragma unroll
        for (int kk = 0; kk < MMA_K_PV; kk++) {
          uint32_t pf[4];
          convert_f32_to_bf16_uint32(xf[m][kk], pf);
#pragma unroll
          for (int nn = 0; nn < PV_CHUNK_SIZE; nn++) {
            uint32_t vf[4];
            int vr = kk*16+(lane_idx&0xF);
            int vc = (pv_col_base+nn)*16+((lane_idx>>4)<<3);
            T *sv = vr < pv_kv_len ? &my_v2[vr*V_HEAD_DIM+vc] : s_ldmatrix_zeros;
            ldsm_t(sv, vf);
            mma_m16n16k16_bf16bf16bf32(o_acc[m][nn], pf, vf, o_acc[m][nn]);
          }
        }

      __syncwarp();
      pv_stage = pv_next_buf;
    } // end KV tile loop for this PV chunk

    // Write this chunk's output
#pragma unroll
    for (int mma_m = 0; mma_m < MMA_M; mma_m++)
#pragma unroll
      for (int mma_n = 0; mma_n < PV_CHUNK_SIZE; mma_n++)
#pragma unroll
        for (int fi = 0; fi < 8; fi++) {
          int row_local = (lane_idx/4) + (((fi>>1)&1)*8);
          int col_local = ((fi>>2)&1)*8 + (lane_idx%4)*2 + (fi&1);
          int row = mma_m*16 + row_local;
          int col = (pv_col_base+mma_n)*16 + col_local;
          if (row >= num_tokens || col >= V_HEAD_DIM) continue;
          float dv = chunk_d[mma_m][(fi&3)>>1];
          float ov = o_acc[mma_m][mma_n][fi];
          ov = (dv > 0.f) ? (ov / dv) : 0.f;
          d_output[row*O_STRIDE + _qh*V_HEAD_DIM + col] = float_to_T<T>(ov);
        }
  } // end PV chunk loop
  end_qh_iter:; // warps 1-3 jump here, warp 0 falls through
  wg_barrier.arrive_and_wait(); // sync all warps before next Q-head iteration
  } // end Q-head loop
}

} // namespace kernel
