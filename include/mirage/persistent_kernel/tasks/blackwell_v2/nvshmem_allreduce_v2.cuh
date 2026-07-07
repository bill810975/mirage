/* Copyright 2026 CMU
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

// ---------------------------------------------------------------------------
// v2 (role-split runtime) port of the NVSHMEM tile all-reduce.
//
// PROBLEM (master plan risk #1): the v1 body (blackwell/allreduce.cuh) runs on
// a whole 256-thread block — it uses block-wide __syncthreads() and
// blockDim.x-strided loops. Under the v2 runtime the block is ALWAYS 256
// threads (8 warps), but a task body dispatched on the CONSUMER role runs on
// only W0-3 == the 128 contiguous threads threadIdx.x 0..127, while W4-7 go on
// to run OTHER tasks. Pasting the v1 body into the consumer role therefore
//   (a) DEADLOCKS on __syncthreads() (W4-7 never reach it), and
//   (b) mis-strides the reduce loop (stride 256 over 128 live threads).
//
// FIX (consumer-only 128-thread rewrite, Codex thread 019f354a + ablation
// review): reuse the v1 cross-rank machinery VERBATIM (mpkar_signal_for_barrier
// / mpkar_wait_until_ge / mpkar_mc_ptr / the NVLS + P2P reduce bodies — all
// rank-to-rank or per-element, thread-count-agnostic) but
//   * replace every __syncthreads() with ar_v2_consumer_sync() == a named
//     barrier `bar.sync 5, 128` over exactly the 128 consumer threads (ids
//     1/2/3/4 are linear/rmsnorm/ffn/attn_v2; 5 is free), with a "memory"
//     clobber so it is a real compiler barrier (Codex guard #2), and
//   * stride every intra-CTA loop by AR_V2_ATC == 128 (the active-thread count)
//     instead of blockDim.x.
// tid 0 is in the consumer set, so `if (threadIdx.x==0) sync_counter[0]+=1`
// stays exactly-once-per-call and the per-team pSync/sync_counter double-buffer
// keeps cross-rank epochs aligned identically to the 128-thread v1 launch
// (both AR call sites already launch block_dim=(128,1,1); v2 just moves those
// 128 onto W0-3).
//
// This file is GMEM-only (no shared memory) — same as the v1 AR.
// ---------------------------------------------------------------------------

#include "tasks/blackwell/allreduce.cuh" // v1 mpkar_* helpers (reused verbatim)

#ifdef USE_NVSHMEM

namespace kernel {
namespace nvshmem_allreduce_v2 {

// Active-thread count of the v2 consumer role.
static constexpr int AR_V2_ATC = 128;

// Named-barrier over exactly the 128 consumer threads. Replaces the v1
// __syncthreads(). The "memory" clobber forbids the compiler from moving
// memory ops across the barrier (so it is equivalent to __syncthreads(), not a
// bare fence-free sync). Barrier id 5 is dedicated to the v2 AR (1/2/3/4 taken
// by linear/rmsnorm/ffn/attn v2 consumer bodies).
__device__ __forceinline__ void ar_v2_consumer_sync() {
  asm volatile("bar.sync 5, 128;" ::: "memory");
}

// ========================= dissemination barrier (128-thread) ==============
// 1:1 copy of ::kernel::mpkar_sync_dissem_pow2_block<k,logk> with the two
// changes: __syncthreads() -> ar_v2_consumer_sync(), blockDim.x -> AR_V2_ATC.
// The cross-rank signal/wait bodies are byte-identical
// (mpkar_signal_for_barrier / mpkar_wait_until_ge are rank-to-rank P2P,
// thread-count-agnostic).
template <int k, int logk>
static __device__ __forceinline__ void
    ar_v2_sync_dissem_pow2_block(nvshmem_team_t team) {
  nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
  int size = teami->size;
  long volatile *sync_counter =
      (long volatile *)::kernel::mpkar_team_get_sync_counter(teami);
  long volatile *pSync =
      (long volatile *)::kernel::mpkar_team_get_psync_sync(teami) +
      MPKAR_NVSHMEMI_SYNC_SIZE * (sync_counter[0] % 2);

  int shift;
  int to_nbr_idx, to_nbr;
  int from_nbr_idx, from_nbr;
  int temp = size - 1;
  int phase_num = 0;
  long volatile *counter = sync_counter;

  while (temp) {
    // notify neighbors
    for (int j = threadIdx.x + 1; j <= k - 1; j += AR_V2_ATC) {
      shift = j << phase_num;
      if (shift >= size) {
        break;
      }
      to_nbr_idx = teami->my_pe + shift;
      to_nbr = ::kernel::mpkar_team_translate_pe(teami, to_nbr_idx);
      ::kernel::mpkar_signal_for_barrier(
          (long *)pSync + nvshmemi_device_state_d.mype, counter[0], to_nbr);
    }
    // wait for neighbors
    for (int j = threadIdx.x + 1; j <= k - 1; j += AR_V2_ATC) {
      shift = j << phase_num;
      if (shift >= size) {
        break;
      }
      from_nbr_idx = teami->my_pe - shift;
      if (from_nbr_idx < 0) {
        from_nbr_idx = size + from_nbr_idx;
      }
      from_nbr = ::kernel::mpkar_team_translate_pe(teami, from_nbr_idx);
      ::kernel::mpkar_wait_until_ge(pSync + from_nbr, counter[0]);
    }
    temp >>= logk;
    phase_num++;
    ar_v2_consumer_sync();
  }
  if (threadIdx.x == 0) {
    sync_counter[0] += 1;
  }
  ar_v2_consumer_sync();
}

// Strided-team dissemination barrier (128-thread). Mirror of
// ::kernel::mpkar_sync_dissem_strided_block. TP8 EP2 uses strided teams, but
// the P2P path picks full-radix pow2 (k==size) for p2p-connected teams; keep
// this for parity / non-pow2 completeness.
static __device__ __forceinline__ void ar_v2_sync_dissem_strided_block(
    nvshmemi_team_t *teami, long volatile *pSync, long volatile *sync_counter) {
  int start = teami->start;
  int stride = teami->stride;
  int size = teami->size;
  int k = min(
      nvshmemi_device_state_d.gpu_coll_env_params_var.barrier_tg_dissem_kval,
      size);
  int my_idx = (nvshmemi_device_state_d.mype - start) / stride;
  int temp = size - 1;
  int num_phases = 0;
  while (temp) {
    num_phases++;
    temp /= k;
  }

  long volatile *counter = sync_counter;
  int pow_k = 1;
  for (int i = 0; i < num_phases; i++) {
    for (int j = threadIdx.x + 1; j <= k - 1; j += AR_V2_ATC) {
      int shift = j * pow_k;
      if (shift >= size) {
        break;
      }
      int to_nbr_idx = (my_idx + shift) % size;
      int to_nbr = start + to_nbr_idx * stride;
      ::kernel::mpkar_signal_for_barrier(
          (long *)pSync + nvshmemi_device_state_d.mype, counter[0], to_nbr);
    }
    for (int j = threadIdx.x + 1; j <= k - 1; j += AR_V2_ATC) {
      int shift = j * pow_k;
      if (shift >= size) {
        break;
      }
      int from_nbr_idx = my_idx - shift;
      if (from_nbr_idx < 0) {
        from_nbr_idx = size + from_nbr_idx;
      }
      int from_nbr = start + from_nbr_idx * stride;
      ::kernel::mpkar_wait_until_ge(pSync + from_nbr, counter[0]);
    }
    pow_k *= k;
    ar_v2_consumer_sync();
  }
  if (threadIdx.x == 0) {
    sync_counter[0] += 1;
  }
  ar_v2_consumer_sync();
}

// Generic (non-pow2) dissemination barrier (128-thread). Mirror of
// ::kernel::mpkar_sync_dissem_generic_block.
static __device__ __forceinline__ void
    ar_v2_sync_dissem_generic_block(nvshmem_team_t team) {
  nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
  int size = teami->size;
  long volatile *sync_counter =
      (long volatile *)::kernel::mpkar_team_get_sync_counter(teami);
  long volatile *pSync =
      (long volatile *)::kernel::mpkar_team_get_psync_sync(teami) +
      MPKAR_NVSHMEMI_SYNC_SIZE * (sync_counter[0] % 2);

  int k = min(
      nvshmemi_device_state_d.gpu_coll_env_params_var.barrier_tg_dissem_kval,
      size);
  int my_idx = teami->my_pe;
  int temp = size - 1;
  int num_phases = 0;
  while (temp) {
    num_phases++;
    temp /= k;
  }

  long volatile *counter = sync_counter;
  int pow_k = 1;
  for (int i = 0; i < num_phases; i++) {
    for (int j = threadIdx.x + 1; j <= k - 1; j += AR_V2_ATC) {
      int shift = j * pow_k;
      if (shift >= size) {
        break;
      }
      int to_nbr_idx = (my_idx + shift) % size;
      int to_nbr = teami->pe_mapping[to_nbr_idx];
      ::kernel::mpkar_signal_for_barrier(
          (long *)pSync + nvshmemi_device_state_d.mype, counter[0], to_nbr);
    }
    for (int j = threadIdx.x + 1; j <= k - 1; j += AR_V2_ATC) {
      int shift = j * pow_k;
      if (shift >= size) {
        break;
      }
      int from_nbr_idx = my_idx - shift;
      if (from_nbr_idx < 0) {
        from_nbr_idx = size + from_nbr_idx;
      }
      int from_nbr = teami->pe_mapping[from_nbr_idx];
      ::kernel::mpkar_wait_until_ge(pSync + from_nbr, counter[0]);
    }
    pow_k *= k;
    ar_v2_consumer_sync();
  }
  if (threadIdx.x == 0) {
    sync_counter[0] += 1;
  }
  ar_v2_consumer_sync();
}

// Top-level 128-thread cross-rank barrier dispatch. Mirror of
// ::kernel::mpkar_sync_block.
static __device__ __forceinline__ void ar_v2_sync_block(nvshmem_team_t team) {
  nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
  int size = teami->size;
  int k = min(
      nvshmemi_device_state_d.gpu_coll_env_params_var.barrier_tg_dissem_kval,
      size);
  k = max(k, 2);
  if (teami->are_gpus_p2p_connected) {
    k = size;
  }
  switch (k) {
    case 2:
      ar_v2_sync_dissem_pow2_block<2, 1>(team);
      break;
    case 4:
      ar_v2_sync_dissem_pow2_block<4, 2>(team);
      break;
    case 8:
      ar_v2_sync_dissem_pow2_block<8, 3>(team);
      break;
    case 16:
      ar_v2_sync_dissem_pow2_block<16, 4>(team);
      break;
    case 32:
      ar_v2_sync_dissem_pow2_block<32, 5>(team);
      break;
    default: {
      if (teami->stride > 0) {
        long volatile *sync_counter =
            (long volatile *)::kernel::mpkar_team_get_sync_counter(teami);
        long volatile *pSync =
            (long volatile *)::kernel::mpkar_team_get_psync_sync(teami) +
            MPKAR_NVSHMEMI_SYNC_SIZE * (sync_counter[0] % 2);
        ar_v2_sync_dissem_strided_block(teami, pSync, sync_counter);
      } else {
        ar_v2_sync_dissem_generic_block(team);
      }
      break;
    }
  }
}

// ========================= 128-thread reduce bodies =========================
// Same math as the v1 mpkar_nvls_*/mpkar_p2p_* blocks, restrided to AR_V2_ATC.
template <typename T>
static __device__ __forceinline__ void ar_v2_nvls_reduce_v4_block(
    int4 *__restrict__ dst, int4 const *__restrict__ mc_src, int nelems_v4) {
  for (int j = threadIdx.x; j < nelems_v4; j += AR_V2_ATC) {
    uint32_t u4[4];
    if constexpr (sizeof(T) == 2) {
      if constexpr (cuda::std::is_same<T, __nv_bfloat16>::value) {
        ::kernel::mpkar_nvls_ld_reduce_bf16_v4(
            u4[0], u4[1], u4[2], u4[3], mc_src + j);
      } else {
        ::kernel::mpkar_nvls_ld_reduce_f16_v4(
            u4[0], u4[1], u4[2], u4[3], mc_src + j);
      }
      asm("st.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(dst + j),
          "r"(u4[0]),
          "r"(u4[1]),
          "r"(u4[2]),
          "r"(u4[3]));
    } else {
      float f4[4];
      ::kernel::mpkar_nvls_ld_reduce_f32_v4(
          f4[0], f4[1], f4[2], f4[3], mc_src + j);
      asm("st.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(dst + j),
          "r"(__float_as_uint(f4[0])),
          "r"(__float_as_uint(f4[1])),
          "r"(__float_as_uint(f4[2])),
          "r"(__float_as_uint(f4[3])));
    }
  }
}

template <typename T>
static __device__ __forceinline__ void
    ar_v2_nvls_reduce_add_residual_v4_block(int4 *__restrict__ dst,
                                            int4 const *__restrict__ mc_src,
                                            int4 const *__restrict__ residual,
                                            int nelems_v4) {
  for (int j = threadIdx.x; j < nelems_v4; j += AR_V2_ATC) {
    if constexpr (sizeof(T) == 2) {
      if constexpr (cuda::std::is_same<T, __nv_bfloat16>::value) {
        uint32_t u4[4];
        ::kernel::mpkar_nvls_ld_reduce_bf16_v4(
            u4[0], u4[1], u4[2], u4[3], mc_src + j);
        int4 const r4 = residual[j];
        u4[0] = ::kernel::mpkar_add_bf16x2(u4[0], static_cast<uint32_t>(r4.x));
        u4[1] = ::kernel::mpkar_add_bf16x2(u4[1], static_cast<uint32_t>(r4.y));
        u4[2] = ::kernel::mpkar_add_bf16x2(u4[2], static_cast<uint32_t>(r4.z));
        u4[3] = ::kernel::mpkar_add_bf16x2(u4[3], static_cast<uint32_t>(r4.w));
        asm("st.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(dst + j),
            "r"(u4[0]),
            "r"(u4[1]),
            "r"(u4[2]),
            "r"(u4[3]));
      } else {
        uint32_t u4[4];
        ::kernel::mpkar_nvls_ld_reduce_f16_v4(
            u4[0], u4[1], u4[2], u4[3], mc_src + j);
        asm("st.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(dst + j),
            "r"(u4[0]),
            "r"(u4[1]),
            "r"(u4[2]),
            "r"(u4[3]));
      }
    } else {
      float f4[4];
      ::kernel::mpkar_nvls_ld_reduce_f32_v4(
          f4[0], f4[1], f4[2], f4[3], mc_src + j);
      int4 const r4 = residual[j];
      f4[0] += __uint_as_float(static_cast<uint32_t>(r4.x));
      f4[1] += __uint_as_float(static_cast<uint32_t>(r4.y));
      f4[2] += __uint_as_float(static_cast<uint32_t>(r4.z));
      f4[3] += __uint_as_float(static_cast<uint32_t>(r4.w));
      asm("st.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(dst + j),
          "r"(__float_as_uint(f4[0])),
          "r"(__float_as_uint(f4[1])),
          "r"(__float_as_uint(f4[2])),
          "r"(__float_as_uint(f4[3])));
    }
  }
}

template <typename T>
static __device__ __forceinline__ void
    ar_v2_p2p_reduce_v4_block(int4 *__restrict__ dst,
                              void const *local_input_ptr,
                              nvshmemi_team_t *teami,
                              int nelems_v4) {
  static_assert(cuda::std::is_same<T, __nv_bfloat16>::value,
                "P2P AR fallback currently bf16 only.");
  int4 const *local_v4 = reinterpret_cast<int4 const *>(local_input_ptr);
  int const npes = teami->size;
  for (int j = threadIdx.x; j < nelems_v4; j += AR_V2_ATC) {
    int4 acc = local_v4[j];
    for (int p = 0; p < npes; p++) {
      int peer_world_pe = teami->pe_mapping[p];
      if (peer_world_pe == nvshmemi_device_state_d.mype) {
        continue;
      }
      int4 *peer_v4 = reinterpret_cast<int4 *>(
          ::kernel::mpkar_peer_ptr(local_input_ptr, peer_world_pe));
      int4 const peer_val = peer_v4[j];
      acc.x = ::kernel::mpkar_add_bf16x2(static_cast<uint32_t>(acc.x),
                                         static_cast<uint32_t>(peer_val.x));
      acc.y = ::kernel::mpkar_add_bf16x2(static_cast<uint32_t>(acc.y),
                                         static_cast<uint32_t>(peer_val.y));
      acc.z = ::kernel::mpkar_add_bf16x2(static_cast<uint32_t>(acc.z),
                                         static_cast<uint32_t>(peer_val.z));
      acc.w = ::kernel::mpkar_add_bf16x2(static_cast<uint32_t>(acc.w),
                                         static_cast<uint32_t>(peer_val.w));
    }
    dst[j] = acc;
  }
}

template <typename T>
static __device__ __forceinline__ void
    ar_v2_p2p_reduce_add_residual_v4_block(int4 *__restrict__ dst,
                                           void const *local_input_ptr,
                                           int4 const *__restrict__ residual,
                                           nvshmemi_team_t *teami,
                                           int nelems_v4) {
  static_assert(cuda::std::is_same<T, __nv_bfloat16>::value,
                "P2P AR fallback currently bf16 only.");
  int4 const *local_v4 = reinterpret_cast<int4 const *>(local_input_ptr);
  int const npes = teami->size;
  for (int j = threadIdx.x; j < nelems_v4; j += AR_V2_ATC) {
    int4 acc = local_v4[j];
    for (int p = 0; p < npes; p++) {
      int peer_world_pe = teami->pe_mapping[p];
      if (peer_world_pe == nvshmemi_device_state_d.mype) {
        continue;
      }
      int4 *peer_v4 = reinterpret_cast<int4 *>(
          ::kernel::mpkar_peer_ptr(local_input_ptr, peer_world_pe));
      int4 const peer_val = peer_v4[j];
      acc.x = ::kernel::mpkar_add_bf16x2(static_cast<uint32_t>(acc.x),
                                         static_cast<uint32_t>(peer_val.x));
      acc.y = ::kernel::mpkar_add_bf16x2(static_cast<uint32_t>(acc.y),
                                         static_cast<uint32_t>(peer_val.y));
      acc.z = ::kernel::mpkar_add_bf16x2(static_cast<uint32_t>(acc.z),
                                         static_cast<uint32_t>(peer_val.z));
      acc.w = ::kernel::mpkar_add_bf16x2(static_cast<uint32_t>(acc.w),
                                         static_cast<uint32_t>(peer_val.w));
    }
    int4 const r4 = residual[j];
    acc.x = ::kernel::mpkar_add_bf16x2(static_cast<uint32_t>(acc.x),
                                       static_cast<uint32_t>(r4.x));
    acc.y = ::kernel::mpkar_add_bf16x2(static_cast<uint32_t>(acc.y),
                                       static_cast<uint32_t>(r4.y));
    acc.z = ::kernel::mpkar_add_bf16x2(static_cast<uint32_t>(acc.z),
                                       static_cast<uint32_t>(r4.z));
    acc.w = ::kernel::mpkar_add_bf16x2(static_cast<uint32_t>(acc.w),
                                       static_cast<uint32_t>(r4.w));
    dst[j] = acc;
  }
}

// ========================= public v2 impl ==================================
// Signature-identical to ::kernel::nvshmem_tile_allreduce_impl. Consumer-only
// (128 threads). This runs AFTER the consumer dep-prefix, i.e. only on
// threadIdx.x 0..127.
template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int OUTPUT_STRIDE,
          bool ADD_RESIDUAL>
__device__ __forceinline__ void
    nvshmem_tile_allreduce_v2_impl(void *input_ptr,
                                   void *residual_ptr,
                                   void *output_ptr,
                                   void *_teams,
                                   int task_offset,
                                   int active_tokens) {
  nvshmem_team_t *teams = reinterpret_cast<nvshmem_team_t *>(_teams);
  nvshmem_team_t team = teams[task_offset];
  int const num_active_rows = max(0, min(active_tokens, BATCH_SIZE));

  // --- Phase 1: ensure local data is visible, then cross-GPU barrier ---
  __threadfence();
  ar_v2_sync_block(team);

  // --- Phase 2: NVLS multicast ld_reduce -> local store ---
  nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
  void *mc_src = ::kernel::mpkar_mc_ptr(teami, input_ptr);

  static_assert(OUTPUT_SIZE % (16 / sizeof(T)) == 0,
                "OUTPUT_SIZE must be a multiple of 16/sizeof(T) for v4 NVLS");

  constexpr int ELEMS_PER_V4 = 16 / sizeof(T);
  constexpr int V4_PER_ROW = OUTPUT_SIZE / ELEMS_PER_V4;
  constexpr int STRIDE_V4 = OUTPUT_STRIDE / ELEMS_PER_V4;

  int4 *dst_v4 = reinterpret_cast<int4 *>(output_ptr);
  int4 const *src_mc_v4 = reinterpret_cast<int4 const *>(mc_src);
  int4 const *residual_v4 = reinterpret_cast<int4 const *>(residual_ptr);

  bool const use_nvls = (mc_src != nullptr);
  if constexpr (OUTPUT_SIZE == OUTPUT_STRIDE) {
    int total_v4 = V4_PER_ROW * num_active_rows;
    if constexpr (ADD_RESIDUAL) {
      if (use_nvls) {
        ar_v2_nvls_reduce_add_residual_v4_block<T>(
            dst_v4, src_mc_v4, residual_v4, total_v4);
      } else {
        ar_v2_p2p_reduce_add_residual_v4_block<T>(
            dst_v4, input_ptr, residual_v4, teami, total_v4);
      }
    } else {
      if (use_nvls) {
        ar_v2_nvls_reduce_v4_block<T>(dst_v4, src_mc_v4, total_v4);
      } else {
        ar_v2_p2p_reduce_v4_block<T>(dst_v4, input_ptr, teami, total_v4);
      }
    }
  } else {
    for (int row = 0; row < num_active_rows; row++) {
      void const *row_input = static_cast<char const *>(input_ptr) +
                              row * STRIDE_V4 * (int)sizeof(int4);
      if constexpr (ADD_RESIDUAL) {
        if (use_nvls) {
          ar_v2_nvls_reduce_add_residual_v4_block<T>(
              dst_v4 + row * STRIDE_V4,
              src_mc_v4 + row * STRIDE_V4,
              residual_v4 + row * STRIDE_V4,
              V4_PER_ROW);
        } else {
          ar_v2_p2p_reduce_add_residual_v4_block<T>(dst_v4 + row * STRIDE_V4,
                                                    row_input,
                                                    residual_v4 +
                                                        row * STRIDE_V4,
                                                    teami,
                                                    V4_PER_ROW);
        }
      } else {
        if (use_nvls) {
          ar_v2_nvls_reduce_v4_block<T>(dst_v4 + row * STRIDE_V4,
                                        src_mc_v4 + row * STRIDE_V4,
                                        V4_PER_ROW);
        } else {
          ar_v2_p2p_reduce_v4_block<T>(
              dst_v4 + row * STRIDE_V4, row_input, teami, V4_PER_ROW);
        }
      }
    }
  }

  // --- Phase 3: ensure PULL stores are visible locally ---
  __threadfence();
  ar_v2_consumer_sync();
}

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int OUTPUT_STRIDE>
__device__ __forceinline__ void nvshmem_tile_allreduce_v2(void *input_ptr,
                                                          void *output_ptr,
                                                          void *_teams,
                                                          int task_offset,
                                                          int active_tokens) {
  nvshmem_tile_allreduce_v2_impl<T,
                                 BATCH_SIZE,
                                 OUTPUT_SIZE,
                                 OUTPUT_STRIDE,
                                 false>(
      input_ptr, nullptr, output_ptr, _teams, task_offset, active_tokens);
}

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int OUTPUT_STRIDE>
__device__ __forceinline__ void
    nvshmem_tile_allreduce_v2_with_residual(void *input_ptr,
                                            void *residual_ptr,
                                            void *output_ptr,
                                            void *_teams,
                                            int task_offset,
                                            int active_tokens) {
  nvshmem_tile_allreduce_v2_impl<T,
                                 BATCH_SIZE,
                                 OUTPUT_SIZE,
                                 OUTPUT_STRIDE,
                                 true>(
      input_ptr, residual_ptr, output_ptr, _teams, task_offset, active_tokens);
}

} // namespace nvshmem_allreduce_v2
} // namespace kernel

#endif // USE_NVSHMEM
