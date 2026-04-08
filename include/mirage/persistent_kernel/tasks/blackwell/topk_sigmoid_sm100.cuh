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

// TopK Sigmoid routing for DeepSeek V3 MoE.
//
// DeepSeek V3 uses sigmoid scoring (not softmax) for expert routing:
//   score[i] = sigmoid(logit[i])
//   routing_score[i] = score[i] + e_score_correction_bias[i]  (for TopK selection)
//   weight[i] = score[i]  (for final gating weight, no bias)
//
// After TopK selection, the k weights are renormalized to sum to 1.
//
// Based on topk_softmax_sm100.cuh — same thread/warp partitioning, same TopK
// selection logic, only the scoring function changes.

#pragma once
#include "topk_softmax_sm100.cuh"

namespace kernel {

template <typename T,
          int VPT,
          int NUM_EXPERTS,
          int WARPS_PER_CTA,
          int BYTES_PER_LDG>
__device__ __forceinline__ void topk_sigmoid_task_impl(
    void *__restrict__ input_ptr,         // [num_rows, NUM_EXPERTS] router logits
    bool const *__restrict__ finished,
    void *__restrict__ output_ptr,        // [num_rows, k] output weights (float)
    void const *__restrict__ bias_ptr,    // [NUM_EXPERTS] e_score_correction_bias (float)
    int const num_rows,
    int const k,
    void *__restrict__ mpk_routing_indices_ptr,
    void *__restrict__ mpk_active_expert_ids_ptr,
    int const start_expert,
    int const end_expert,
    bool const renormalize) {

  T *input = static_cast<T *>(input_ptr);
  float *output = static_cast<float *>(output_ptr);
  float const *bias = static_cast<float const *>(bias_ptr);
  int *mpk_routing_indices = static_cast<int *>(mpk_routing_indices_ptr);
  int *mpk_active_expert_ids = static_cast<int *>(mpk_active_expert_ids_ptr);

  // Initialize routing indices and active expert marks (same as softmax)
  for (int expert = start_expert + threadIdx.x; expert < end_expert;
       expert += blockDim.x) {
    if (mpk_routing_indices != nullptr) {
      for (int row = 0; row < num_rows; ++row) {
        mpk_routing_indices[expert * num_rows + row] = 0;
      }
    }
    if (mpk_active_expert_ids != nullptr) {
      mpk_active_expert_ids[expert - start_expert] = -1;
    }
  }
  if (threadIdx.x == NUM_EXPERTS && mpk_active_expert_ids != nullptr) {
    mpk_active_expert_ids[NUM_EXPERTS] = 0;
  }
  __syncthreads();

  // Compile-time constants (identical to softmax version)
  static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
  static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS),
                "NUM_EXPERTS must be power of 2");
  static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG),
                "BYTES_PER_LDG must be power of 2");
  static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
  static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
  static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
  static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

  static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT;
  static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;

  int const warp_idx = threadIdx.x / WARP_SIZE;
  int const lane_idx = threadIdx.x % WARP_SIZE;
  int const warp_base_row = warp_idx * ROWS_PER_WARP;

  int const thread_row_in_warp = lane_idx / THREADS_PER_ROW;
  int const thread_row = warp_base_row + thread_row_in_warp;
  uint32_t const warp_mask = (num_rows % 2 == 1 && thread_row == num_rows - 1)
                                 ? 0x0000ffff
                                 : 0xffffffff;

  if (thread_row < num_rows) {
    bool const row_is_active = finished ? !finished[thread_row] : true;

    // Load logits (vectorized, same as softmax)
    T *thread_row_ptr = input + thread_row * ELTS_PER_ROW;
    int const thread_group_idx = lane_idx % THREADS_PER_ROW;
    int const first_elt_read_by_thread =
        thread_group_idx * (BYTES_PER_LDG / sizeof(T));
    T *thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

    using AccessType = cutlass::AlignedArray<T, ELTS_PER_LDG>;
    T row_chunk_temp[VPT];
    AccessType *row_chunk_vec_ptr =
        reinterpret_cast<AccessType *>(&row_chunk_temp);
    AccessType *vec_thread_read_ptr =
        reinterpret_cast<AccessType *>(thread_read_ptr);

    for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
      row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
    }

    cutlass::NumericConverter<float, T> converter;

    float row_chunk[VPT];       // sigmoid scores (no bias) — used as final weights
    float row_chunk_biased[VPT]; // sigmoid + bias — used for TopK selection
    for (int ii = 0; ii < VPT; ++ii) {
      float logit = converter(row_chunk_temp[ii]);
      row_chunk_temp[ii] = static_cast<T>(0); // reset for split-k gate linear

      // Sigmoid: score = 1 / (1 + exp(-logit))
      float score = 1.0f / (1.0f + expf(-logit));
      row_chunk[ii] = score;

      // Routing score = score + bias (for TopK selection only)
      int expert_idx = first_elt_read_by_thread + ii;
      // Handle vectorized layout: expert_idx for multi-LDG threads
      int ldg_idx = ii / ELTS_PER_LDG;
      int elt_in_ldg = ii % ELTS_PER_LDG;
      expert_idx = first_elt_read_by_thread +
                   ldg_idx * (ELTS_PER_LDG * THREADS_PER_ROW) + elt_in_ldg;
      float b = (bias != nullptr && expert_idx < NUM_EXPERTS)
                    ? bias[expert_idx]
                    : 0.f;
      row_chunk_biased[ii] = score + b;
    }

    // Reset input buffer to 0 for split-k gate linear
    for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
      vec_thread_read_ptr[ii * THREADS_PER_ROW] = row_chunk_vec_ptr[ii];
    }

    // ---- TopK selection using BIASED scores (same structure as softmax) ----
    int start_col = first_elt_read_by_thread;
    static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;
    float row_sum_for_renormalize = 0.f;

    for (int k_idx = 0; k_idx < k; ++k_idx) {
      // Find local max using BIASED scores (for routing selection)
      float max_val = row_chunk_biased[0];
      int expert = start_col;
      for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD;
           ++ldg, col += COLS_PER_GROUP_LDG) {
        for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
          float val = row_chunk_biased[ldg * ELTS_PER_LDG + ii];
          if (val > max_val) {
            max_val = val;
            expert = col + ii;
          }
        }
      }

      // Argmax reduce across subgroup
      for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
        float other_max =
            __shfl_xor_sync(warp_mask, max_val, mask, THREADS_PER_ROW);
        int other_expert =
            __shfl_xor_sync(warp_mask, expert, mask, THREADS_PER_ROW);
        if (other_max > max_val ||
            (other_max == max_val && other_expert < expert)) {
          max_val = other_max;
          expert = other_expert;
        }
      }

      // Write output using UNBIASED score (for gating weight)
      // Broadcast the winning expert's unbiased score from the thread that owns it
      int const ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
      int const thread_to_own =
          (expert / ELTS_PER_LDG) % THREADS_PER_ROW;
      int const offset_in_ldg = expert % ELTS_PER_LDG;
      float unbiased_score = row_chunk[ldg_group_for_expert * ELTS_PER_LDG +
                                       offset_in_ldg];
      // Broadcast from the owning thread to all threads in the subgroup
      unbiased_score =
          __shfl_sync(warp_mask, unbiased_score, thread_to_own, THREADS_PER_ROW);

      if (thread_group_idx == 0) {
        bool const node_uses_expert =
            expert >= start_expert && expert < end_expert;
        bool const should_process_row = row_is_active && node_uses_expert;
        int const out_idx = k * thread_row + k_idx;
        output[out_idx] = unbiased_score;
        row_sum_for_renormalize += unbiased_score;
        if (should_process_row && mpk_routing_indices != nullptr) {
          int const local_expert = expert - start_expert;
          mpk_routing_indices[local_expert * num_rows + thread_row] = k_idx + 1;
          if (mpk_active_expert_ids != nullptr) {
            mpk_active_expert_ids[local_expert] = local_expert;
          }
        }
      }

      // Blank out the winning value for the next iteration
      if (k_idx + 1 < k) {
        if (thread_group_idx == thread_to_own) {
          row_chunk_biased[ldg_group_for_expert * ELTS_PER_LDG +
                           offset_in_ldg] = -10000.f;
          row_chunk[ldg_group_for_expert * ELTS_PER_LDG + offset_in_ldg] =
              -10000.f;
        }
      }
    }

    // Renormalize: weights sum to 1 (DeepSeek V3: norm_topk_prob=true)
    if (renormalize && thread_group_idx == 0) {
      float inv = 1.f / row_sum_for_renormalize;
      for (int k_idx = 0; k_idx < k; ++k_idx) {
        int const out_idx = k * thread_row + k_idx;
        output[out_idx] = output[out_idx] * inv;
      }
    }
  }
  __syncthreads();

  // Compact active expert marks into dense list
  if (mpk_active_expert_ids != nullptr) {
    for (int expert = start_expert + threadIdx.x; expert < end_expert;
         expert += blockDim.x) {
      int const local_expert = expert - start_expert;
      int const mark = mpk_active_expert_ids[local_expert];
      if (mark >= 0) {
        int const pos = atomicAdd(mpk_active_expert_ids + NUM_EXPERTS, 1);
        mpk_active_expert_ids[pos] = expert;
      }
    }
  }
}

} // namespace kernel
