/* Copyright 2025 Mirage Team
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
// v2 (role-split runtime) port of the tensor_init zero-fill (T-A of the
// DSv3-decode-on-v2 effort).
//
// The v1 body (blackwell/tensor_init.cuh::tensor_init_zero_sm100_task_impl)
// 16B-vectorizes the zero-fill and strides by blockDim.x. Under the v2 runtime
// the block is ALWAYS 256 threads (8 warps), but a task body dispatched on the
// CONSUMER role runs on only W0-3 == the 128 contiguous threads threadIdx.x
// 0..127, while W4-7 go on to run OTHER tasks. Reusing the v1 body verbatim on
// the consumer role would mis-stride (stride 256 over 128 live threads leaves
// indices [128,256) of each 256-wide window unwritten). So this v2 body
//   * guards `if (threadIdx.x >= CONSUMER_NUM_THREADS) return;` (only W0-3
//     participate; there is no cross-warp sync so no deadlock hazard), and
//   * strides every intra-CTA loop by CONSUMER_NUM_THREADS == 128 (the active
//     thread count) instead of blockDim.x.
//
// BYTE-EXACT, DTYPE-AGNOSTIC (fix for the DSv3-decode-on-v2 nvcc failure):
// the v1 body casts target_ptr to bf16* and treats OUTPUT_SIZE / OUTPUT_STRIDE
// as bf16 element counts. That is only byte-correct when sizeof(dtype)==2.
// The DSv3 v2 FFN-megakernel barrier tensor `_ffn_bar` is int64[2] (16 bytes):
// its dtensor dim/stride are element counts of ITS OWN dtype (dim[1]=2 int64s),
// so the codegen passed OUTPUT_SIZE=2 and the old bf16-typed body both (a)
// tripped `OUTPUT_SIZE % 8 == 0` and (b) would have addressed only 2 bf16 = 4
// bytes of the true 16-byte buffer. So this body is parameterized in BYTES:
//   * ROW_BYTES       = OUTPUT_SIZE   * dtype_size (bytes to zero per row)
//   * ROW_STRIDE_BYTES= OUTPUT_STRIDE * dtype_size (byte stride between rows)
// and the codegen multiplies the element counts by get_datatype_size(dtype).
// It fast-paths the 16B-aligned bulk with int4{0,0,0,0} stores (identical to
// v1 for the bf16 attn/ffn scratch, all of whose ROW_BYTES are 16B-multiples)
// and adds a scalar BYTE tail for the 0..15-byte remainder so ANY size/dtype
// the decode path uses is zeroed EXACTLY — bit-identical (all-zero) over the
// same byte span v1 covers for a bf16 target, and correct for int64[2].
//
// GMEM-only (no shared memory) — same as the v1 tensor_init.
// ---------------------------------------------------------------------------

#include "../common/worker_config.h"
#include "tasks/common/common_header.cuh"

namespace kernel {
namespace v2 {

// Byte-exact zero-fill for a 2-D tile, v2-safe (128-thread consumer role).
// ROW_BYTES / ROW_STRIDE_BYTES are BYTE counts (the codegen has already
// multiplied the dtensor element counts by the target's dtype size), so this
// works for ANY dtype (bf16 scratch AND int64[2] barrier). 16B (int4) bulk
// store for the aligned head + scalar byte tail for the 0..15-byte remainder.
//
// Alignment safety: the int4 bulk store requires every `row_base` to be
// 16B-aligned. `target_ptr` (a fresh cudaMalloc'd cuda_tensor / a P1-P2 view
// base) is 16B-aligned by construction, so row 0 is safe; row R is aligned iff
// ROW_STRIDE_BYTES % 16 == 0. That holds for EVERY decode-path caller (bf16
// scratch: stride is a bf16 elem count and the buffer bytes are %16; int64
// bar: single row) — but to keep the kernel correct for a hypothetical
// odd-stride multi-row target, fall back to a pure per-byte scalar fill when
// ROW_STRIDE_BYTES is not a 16B multiple (this whole branch is resolved at
// compile time because ROW_STRIDE_BYTES is a template constant, so the fast
// path is unaffected).
template <int BATCH_SIZE, int ROW_BYTES, int ROW_STRIDE_BYTES>
__device__ __forceinline__ void
    tensor_init_zero_v2_task_impl(void *target_ptr) {
  if (threadIdx.x >= CONSUMER_NUM_THREADS) {
    return;
  }
  uint8_t *base = static_cast<uint8_t *>(target_ptr);
  constexpr int VEC_BYTES = 16; // int4 = 16 bytes
  constexpr bool ROWS_16B_ALIGNED =
      (BATCH_SIZE <= 1) || (ROW_STRIDE_BYTES % VEC_BYTES == 0);
  if (ROWS_16B_ALIGNED) {
    constexpr int VEC_PER_ROW = ROW_BYTES / VEC_BYTES;
    constexpr int TAIL_START = VEC_PER_ROW * VEC_BYTES; // first non-vec byte
    int4 const zero = {0, 0, 0, 0};
#pragma unroll
    for (int row = 0; row < BATCH_SIZE; ++row) {
      uint8_t *row_base = base + row * ROW_STRIDE_BYTES;
      // 16B-aligned bulk (matches v1's int4 stores for the bf16 scratch).
      int4 *row_vec = reinterpret_cast<int4 *>(row_base);
      for (int i = threadIdx.x; i < VEC_PER_ROW; i += CONSUMER_NUM_THREADS) {
        row_vec[i] = zero;
      }
      // Scalar byte tail (0..15 bytes) — only iterates when ROW_BYTES is not a
      // multiple of 16 (empty loop otherwise).
      for (int b = TAIL_START + threadIdx.x; b < ROW_BYTES;
           b += CONSUMER_NUM_THREADS) {
        row_base[b] = 0;
      }
    }
  } else {
    // Odd-stride fallback (never taken by the decode path): pure per-byte fill,
    // no int4 alignment assumption on rows > 0.
#pragma unroll
    for (int row = 0; row < BATCH_SIZE; ++row) {
      uint8_t *row_base = base + row * ROW_STRIDE_BYTES;
      for (int b = threadIdx.x; b < ROW_BYTES; b += CONSUMER_NUM_THREADS) {
        row_base[b] = 0;
      }
    }
  }
} // tensor_init_zero_v2_task_impl

} // namespace v2
} // namespace kernel
