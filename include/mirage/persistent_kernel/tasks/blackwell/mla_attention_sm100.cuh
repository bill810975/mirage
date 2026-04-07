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
#include "tasks/common/common_header.cuh"

#include <cutlass/arch/barrier.h>

namespace kernel {

// MLA (Multi-head Latent Attention) paged attention for DeepSeek V3.
//
// Correctness-first implementation. Each thread independently computes
// attention for one (token, q_head) pair using online softmax over the
// full KV sequence. No MMA, no cross-warp reduction needed.
//
// Key properties:
// - QK dot product uses QK_HEAD_DIM (576 = 512 latent + 64 rope)
// - V/output uses V_HEAD_DIM (512, latent only)
// - Single KV head (MQA after weight absorption)
// - Combined KV cache: [c_latent(512), k_pe(64)]
// - Paged KV cache with per-request page tables
//
// Performance: ~O(num_tokens * num_q_heads * seq_len * head_dim / NUM_THREADS)
// This is a baseline; will be replaced by the optimized 2-SM MLA kernel.
template <typename T,
          int NUM_Q_HEADS,
          int QK_HEAD_DIM,
          int V_HEAD_DIM,
          int MAX_SEQ_LEN,
          int PAGE_SIZE,
          int MAX_TOKENS = 8>
__device__ __forceinline__ void mla_paged_attention_sm100_task_impl(
    void const *q_nope_pe_ptr,
    void *ckv_kpe_cache_ptr,
    void const *kv_new_ptr,
    void *output_ptr,
    int const *qo_indptr_buffer_ptr,
    int const *paged_kv_indptr_buffer_ptr,
    int const *paged_kv_indices_buffer_ptr,
    int const *paged_kv_last_page_len_buffer_ptr,
    int16_t request_id) {

  constexpr int BARRIER_ID = 6;
  cutlass::arch::NamedBarrier barrier(NUM_THREADS, BARRIER_ID);

  if (threadIdx.x >= NUM_THREADS) {
    return;
  }

  constexpr int Q_STRIDE = NUM_Q_HEADS * QK_HEAD_DIM;
  constexpr int O_STRIDE = NUM_Q_HEADS * V_HEAD_DIM;
  constexpr int MAX_PAGES = (MAX_SEQ_LEN + PAGE_SIZE - 1) / PAGE_SIZE;

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

  // Load page indices into shared memory
  __shared__ int s_page_indices[MAX_PAGES];
  for (int i = threadIdx.x; i < num_pages; i += NUM_THREADS) {
    s_page_indices[i] = paged_kv_indices_buffer_ptr[first_page + i];
  }
  barrier.arrive_and_wait();

  // Pointers
  T const *__restrict__ d_q =
      reinterpret_cast<T const *>(q_nope_pe_ptr) + first_token * Q_STRIDE;
  T const *__restrict__ d_kv_new =
      reinterpret_cast<T const *>(kv_new_ptr) + first_token * QK_HEAD_DIM;
  T *__restrict__ d_cache = reinterpret_cast<T *>(ckv_kpe_cache_ptr);
  T *__restrict__ d_output =
      reinterpret_cast<T *>(output_ptr) + first_token * O_STRIDE;

  // Phase 1: Write new KV entries to cache
  // New tokens go to positions [seq_len - num_tokens, seq_len)
  for (int idx = threadIdx.x; idx < num_tokens * QK_HEAD_DIM;
       idx += NUM_THREADS) {
    int t = idx / QK_HEAD_DIM;
    int d = idx % QK_HEAD_DIM;
    int cache_pos = seq_len - num_tokens + t;
    int page_idx = s_page_indices[cache_pos / PAGE_SIZE];
    int page_off = cache_pos % PAGE_SIZE;
    d_cache[(page_idx * PAGE_SIZE + page_off) * QK_HEAD_DIM + d] =
        d_kv_new[t * QK_HEAD_DIM + d];
  }
  barrier.arrive_and_wait();

  // Phase 2: Compute attention
  // Total work items: num_tokens * NUM_Q_HEADS
  // Each thread handles one or more (token, head) pairs
  int total_work = num_tokens * NUM_Q_HEADS;

  for (int work_idx = threadIdx.x; work_idx < total_work;
       work_idx += NUM_THREADS) {
    int token_idx = work_idx / NUM_Q_HEADS;
    int head_idx = work_idx % NUM_Q_HEADS;

    // This token can see KV up to: seq_len - num_tokens + token_idx + 1
    int valid_kv_len = seq_len - num_tokens + token_idx + 1;

    // Q pointer for this (token, head)
    T const *q_ptr = d_q + token_idx * Q_STRIDE + head_idx * QK_HEAD_DIM;

    // Online softmax over KV sequence
    float m = -INFINITY; // running max
    float d_sum = 0.f;   // running exp sum
    // Accumulate output in fp32
    float o_acc[V_HEAD_DIM];
    // Can't have V_HEAD_DIM=512 floats on stack for all threads.
    // Use a tiled approach: process V_HEAD_DIM in chunks.

    // Actually, 512 floats = 2KB per thread. With 128 threads = 256KB total
    // register pressure. This exceeds register file. Need to tile the output.

    // Tile output dimension: process V_TILE elements at a time
    constexpr int V_TILE = 32; // Process 32 output dims at a time
    static_assert(V_HEAD_DIM % V_TILE == 0);

    // For each output tile, we need to re-scan the full KV sequence
    // (since we need the attention weights which depend on the full QK).
    //
    // Optimization: compute attention weights once, then apply to V tiles.
    // But storing attention weights for full seq_len is too much memory.
    //
    // Two-pass approach:
    // Pass 1: Compute softmax normalization constants (m, d_sum)
    // Pass 2: For each V tile, compute weighted sum

    // Pass 1: Compute m and d_sum via online softmax over KV
    for (int kv_pos = 0; kv_pos < valid_kv_len; kv_pos++) {
      // Get K vector from cache
      int page_idx = s_page_indices[kv_pos / PAGE_SIZE];
      int page_off = kv_pos % PAGE_SIZE;
      T const *k_ptr =
          d_cache + (page_idx * PAGE_SIZE + page_off) * QK_HEAD_DIM;

      // QK dot product (all QK_HEAD_DIM dims)
      float score = 0.f;
      for (int dd = 0; dd < QK_HEAD_DIM; dd++) {
        score += float(q_ptr[dd]) * float(k_ptr[dd]);
      }
      score /= sqrtf(float(QK_HEAD_DIM));

      // Online softmax update
      float m_new = max(m, score);
      float exp_diff = expf(m - m_new);
      float exp_score = expf(score - m_new);
      d_sum = d_sum * exp_diff + exp_score;
      m = m_new;
    }

    // Pass 2: Compute output for each V tile
    float inv_d = (d_sum > 0.f) ? (1.f / d_sum) : 0.f;

    for (int v_start = 0; v_start < V_HEAD_DIM; v_start += V_TILE) {
      float o_tile[V_TILE];
      for (int i = 0; i < V_TILE; i++) {
        o_tile[i] = 0.f;
      }

      for (int kv_pos = 0; kv_pos < valid_kv_len; kv_pos++) {
        int page_idx = s_page_indices[kv_pos / PAGE_SIZE];
        int page_off = kv_pos % PAGE_SIZE;
        T const *k_ptr =
            d_cache + (page_idx * PAGE_SIZE + page_off) * QK_HEAD_DIM;
        // V = first V_HEAD_DIM dims of cache entry
        T const *v_ptr = k_ptr; // same base, just first V_HEAD_DIM dims

        // Recompute attention weight for this KV position
        float score = 0.f;
        for (int dd = 0; dd < QK_HEAD_DIM; dd++) {
          score += float(q_ptr[dd]) * float(k_ptr[dd]);
        }
        score /= sqrtf(float(QK_HEAD_DIM));
        float weight = expf(score - m) * inv_d;

        // Accumulate V tile
        for (int i = 0; i < V_TILE; i++) {
          o_tile[i] += weight * float(v_ptr[v_start + i]);
        }
      }

      // Write output tile
      T *out_ptr =
          d_output + token_idx * O_STRIDE + head_idx * V_HEAD_DIM + v_start;
      for (int i = 0; i < V_TILE; i++) {
        out_ptr[i] = T(o_tile[i]);
      }
    }
  }
}

} // namespace kernel
