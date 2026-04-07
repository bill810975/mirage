/* Test wrapper for MLA paged attention kernel.
 * Compiles the MLA device function into a launchable kernel for testing.
 */

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// Must be defined before kernel headers (runtime_header.h needs MPK_TARGET_CC)
#ifndef MPK_TARGET_CC
#define MPK_TARGET_CC 1000
#endif

// Include order: common_header first (defines NUM_THREADS, MAX_DYNAMIC_SHARED_MEMORY_SIZE, etc.)
// cutlass stub is resolved via cutlass_stub/ dir placed first in include path (setup.py)
#include "mirage/persistent_kernel/tasks/ampere/smem_layout.cuh"
#include "mirage/persistent_kernel/tasks/common/common_header.cuh"

// Include our MLA kernel (pulls in cutlass/arch/barrier.h -> resolved to stub)
#include "mirage/persistent_kernel/tasks/blackwell/mla_attention_sm100.cuh"

// Test configuration matching DeepSeek V3 with 8-GPU TP
constexpr int NUM_Q_HEADS = 16;
constexpr int QK_HEAD_DIM = 576;
constexpr int V_HEAD_DIM = 512;
constexpr int MAX_SEQ_LEN = 4096;
constexpr int PAGE_SIZE = 16;
constexpr int MAX_TOKENS = 8;
constexpr int KV_TILE_SIZE = 32;

using Element = __nv_bfloat16;

// Wrapper kernel
template <typename T, int NQH, int QKD, int VD, int MSL, int PS, int MT>
__global__ void mla_attention_test_kernel(
    void const *q_ptr, void *cache_ptr, void const *c_new_ptr,
    void const *k_pe_new_ptr, void *output_ptr, int const *qo_indptr,
    int const *kv_indptr, int const *kv_indices, int const *kv_last_page_len) {
  int16_t req_id = static_cast<int16_t>(blockIdx.x);
  int qh_idx = static_cast<int>(blockIdx.y);
  kernel::mla_paged_attention_sm100_task_impl<T, NQH, QKD, VD, MSL, PS, MT>(
      q_ptr, cache_ptr, c_new_ptr, k_pe_new_ptr, output_ptr, qo_indptr,
      kv_indptr, kv_indices, kv_last_page_len, req_id, qh_idx);
}

void mla_attention(torch::Tensor q_nope_pe, torch::Tensor ckv_kpe_cache,
                   torch::Tensor c_latent_new, torch::Tensor k_pe_new,
                   torch::Tensor output, torch::Tensor qo_indptr,
                   torch::Tensor kv_indptr, torch::Tensor kv_indices,
                   torch::Tensor kv_last_page_len, int num_requests) {
  TORCH_CHECK(q_nope_pe.dtype() == torch::kBFloat16);
  TORCH_CHECK(ckv_kpe_cache.dtype() == torch::kBFloat16);

  // Dynamic shared memory size:
  //   s_q:  MAX_TOKENS * QK_HEAD_DIM
  //   s_k:  2 * KV_TILE_SIZE * QK_HEAD_DIM   (double-buffer, combined 576-wide)
  //   s_v:  2 * KV_TILE_SIZE * V_HEAD_DIM    (double-buffer, 512-wide)
  constexpr size_t smem_size =
      sizeof(Element) * (MAX_TOKENS * QK_HEAD_DIM +
                         2 * KV_TILE_SIZE * QK_HEAD_DIM +
                         2 * KV_TILE_SIZE * V_HEAD_DIM);

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  // Set max dynamic shared memory
  cudaFuncSetAttribute(
      mla_attention_test_kernel<Element, NUM_Q_HEADS, QK_HEAD_DIM, V_HEAD_DIM,
                                 MAX_SEQ_LEN, PAGE_SIZE, MAX_TOKENS>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  dim3 grid(num_requests, NUM_Q_HEADS);
  mla_attention_test_kernel<Element, NUM_Q_HEADS, QK_HEAD_DIM, V_HEAD_DIM,
                             MAX_SEQ_LEN, PAGE_SIZE, MAX_TOKENS>
      <<<grid, NUM_THREADS, smem_size, stream>>>(
          q_nope_pe.data_ptr(), ckv_kpe_cache.data_ptr(),
          c_latent_new.data_ptr(), k_pe_new.data_ptr(), output.data_ptr(),
          qo_indptr.data_ptr<int>(), kv_indptr.data_ptr<int>(),
          kv_indices.data_ptr<int>(), kv_last_page_len.data_ptr<int>());

  cudaError_t err = cudaDeviceSynchronize();
  TORCH_CHECK(err == cudaSuccess, "MLA kernel failed: ",
              cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mla_attention", &mla_attention, "MLA Paged Attention (test)");
}
