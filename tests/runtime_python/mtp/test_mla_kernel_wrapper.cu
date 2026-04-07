/* Test wrapper for MLA paged attention kernel.
 * Compiles the MLA device function into a launchable kernel for testing.
 */

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// Minimal stubs to satisfy the kernel's dependencies
#ifndef NUM_THREADS
#define NUM_THREADS 128
#endif

namespace mirage {
namespace runtime {
constexpr size_t MAX_DYNAMIC_SHARED_MEMORY_SIZE = 228 * 1024;
}
} // namespace mirage

// Minimal barrier stub
namespace cutlass {
namespace arch {
struct NamedBarrier {
  __device__ NamedBarrier(int, int) {}
  __device__ void arrive_and_wait() { __syncthreads(); }
};
} // namespace arch
} // namespace cutlass

// Minimal MMA stubs for non-SM100 compilation
// These will be replaced by real implementations on SM100
#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 800
#endif

// Stub helper functions that the kernel depends on
__device__ inline int warp_id() {
  return threadIdx.x / 32;
}
__device__ inline int lane_id() {
  return threadIdx.x & 0x1f;
}
__device__ inline void cp_async_fence() {}
__device__ inline void clear_8_floats(float *f) {
  for (int i = 0; i < 8; i++) f[i] = 0.f;
}

// For testing on non-SM100, we use a simplified scalar fallback
// The real MMA-based kernel requires SM100 hardware

// Include the smem layout helpers
#include "mirage/persistent_kernel/tasks/ampere/smem_layout.cuh"
#include "mirage/persistent_kernel/tasks/common/common_header.cuh"

// Include our MLA kernel
#include "mirage/persistent_kernel/tasks/blackwell/mla_attention_sm100.cuh"

// Test configuration matching DeepSeek V3 with 8-GPU TP
constexpr int NUM_Q_HEADS = 16;
constexpr int QK_HEAD_DIM = 576;
constexpr int V_HEAD_DIM = 512;
constexpr int MAX_SEQ_LEN = 4096;
constexpr int PAGE_SIZE = 16;
constexpr int MAX_TOKENS = 8;

using Element = __nv_bfloat16;

// Wrapper kernel
template <typename T, int NQH, int QKD, int VD, int MSL, int PS, int MT>
__global__ void mla_attention_test_kernel(
    void const *q_ptr, void *cache_ptr, void const *c_new_ptr,
    void const *k_pe_new_ptr, void *output_ptr, int const *qo_indptr,
    int const *kv_indptr, int const *kv_indices, int const *kv_last_page_len) {
  int16_t req_id = static_cast<int16_t>(blockIdx.x);
  kernel::mla_paged_attention_sm100_task_impl<T, NQH, QKD, VD, MSL, PS, MT>(
      q_ptr, cache_ptr, c_new_ptr, k_pe_new_ptr, output_ptr, qo_indptr,
      kv_indptr, kv_indices, kv_last_page_len, req_id);
}

void mla_attention(torch::Tensor q_nope_pe, torch::Tensor ckv_kpe_cache,
                   torch::Tensor c_latent_new, torch::Tensor k_pe_new,
                   torch::Tensor output, torch::Tensor qo_indptr,
                   torch::Tensor kv_indptr, torch::Tensor kv_indices,
                   torch::Tensor kv_last_page_len, int num_requests) {
  TORCH_CHECK(q_nope_pe.dtype() == torch::kBFloat16);
  TORCH_CHECK(ckv_kpe_cache.dtype() == torch::kBFloat16);

  // Dynamic shared memory size
  constexpr size_t smem_size =
      sizeof(Element) * (4 * MAX_TOKENS * QK_HEAD_DIM +
                         2 * 32 * QK_HEAD_DIM + 2 * 32 * V_HEAD_DIM);

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  // Set max dynamic shared memory
  cudaFuncSetAttribute(
      mla_attention_test_kernel<Element, NUM_Q_HEADS, QK_HEAD_DIM, V_HEAD_DIM,
                                 MAX_SEQ_LEN, PAGE_SIZE, MAX_TOKENS>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  mla_attention_test_kernel<Element, NUM_Q_HEADS, QK_HEAD_DIM, V_HEAD_DIM,
                             MAX_SEQ_LEN, PAGE_SIZE, MAX_TOKENS>
      <<<num_requests, NUM_THREADS, smem_size, stream>>>(
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
