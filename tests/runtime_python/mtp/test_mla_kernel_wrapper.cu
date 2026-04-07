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

// Minimal barrier stub (the test launches a proper thread block)
namespace cutlass {
namespace arch {
struct NamedBarrier {
  __device__ NamedBarrier(int, int) {}
  __device__ void arrive_and_wait() { __syncthreads(); }
};
} // namespace arch
} // namespace cutlass

// Include our MLA kernel
#include "mirage/persistent_kernel/tasks/blackwell/mla_attention_sm100.cuh"

// Wrapper kernel that calls the device function
template <typename T, int NUM_Q_HEADS, int QK_HEAD_DIM, int V_HEAD_DIM,
          int MAX_SEQ_LEN, int PAGE_SIZE, int MAX_TOKENS>
__global__ void mla_attention_test_kernel(
    void const *q_nope_pe_ptr, void *ckv_kpe_cache_ptr,
    void const *kv_new_ptr, void *output_ptr,
    int const *qo_indptr_buffer_ptr, int const *paged_kv_indptr_buffer_ptr,
    int const *paged_kv_indices_buffer_ptr,
    int const *paged_kv_last_page_len_buffer_ptr) {
  // blockIdx.x = request_id
  int16_t request_id = static_cast<int16_t>(blockIdx.x);
  kernel::mla_paged_attention_sm100_task_impl<T, NUM_Q_HEADS, QK_HEAD_DIM,
                                               V_HEAD_DIM, MAX_SEQ_LEN,
                                               PAGE_SIZE, MAX_TOKENS>(
      q_nope_pe_ptr, ckv_kpe_cache_ptr, kv_new_ptr, output_ptr,
      qo_indptr_buffer_ptr, paged_kv_indptr_buffer_ptr,
      paged_kv_indices_buffer_ptr, paged_kv_last_page_len_buffer_ptr,
      request_id);
}

// Test configuration matching DeepSeek V3 with 8-GPU TP
constexpr int NUM_Q_HEADS = 16;  // 128 / 8 GPUs
constexpr int QK_HEAD_DIM = 576; // 512 latent + 64 rope
constexpr int V_HEAD_DIM = 512;
constexpr int MAX_SEQ_LEN = 4096;
constexpr int PAGE_SIZE = 16;
constexpr int MAX_TOKENS = 8;

using Element = __nv_bfloat16;

void mla_attention(torch::Tensor q_nope_pe,     // [num_tokens, NUM_Q_HEADS * QK_HEAD_DIM]
                   torch::Tensor ckv_kpe_cache,  // [num_pages, PAGE_SIZE, QK_HEAD_DIM]
                   torch::Tensor kv_new,         // [num_tokens, QK_HEAD_DIM]
                   torch::Tensor output,         // [num_tokens, NUM_Q_HEADS * V_HEAD_DIM]
                   torch::Tensor qo_indptr,      // [num_requests + 1]
                   torch::Tensor kv_indptr,      // [num_requests + 1]
                   torch::Tensor kv_indices,     // [total_pages]
                   torch::Tensor kv_last_page_len, // [num_requests]
                   int num_requests) {

  TORCH_CHECK(q_nope_pe.dtype() == torch::kBFloat16);
  TORCH_CHECK(ckv_kpe_cache.dtype() == torch::kBFloat16);

  int num_blocks = num_requests;
  int num_threads = NUM_THREADS;

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  mla_attention_test_kernel<Element, NUM_Q_HEADS, QK_HEAD_DIM, V_HEAD_DIM,
                             MAX_SEQ_LEN, PAGE_SIZE, MAX_TOKENS>
      <<<num_blocks, num_threads, 0, stream>>>(
          q_nope_pe.data_ptr(), ckv_kpe_cache.data_ptr(), kv_new.data_ptr(),
          output.data_ptr(), qo_indptr.data_ptr<int>(),
          kv_indptr.data_ptr<int>(), kv_indices.data_ptr<int>(),
          kv_last_page_len.data_ptr<int>());

  cudaError_t err = cudaDeviceSynchronize();
  TORCH_CHECK(err == cudaSuccess, "MLA kernel failed: ", cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mla_attention", &mla_attention, "MLA Paged Attention (test)");
}
