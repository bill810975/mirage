// Minimal FP8 linear test using MPK's fill_tma_desc (not tma_2d::create_tma_desc)
// This isolates whether MPK's TMA desc creation is correct for FP8.
#include <cstdio>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>

#include "mirage/persistent_kernel/tma.cuh"
#include "mirage/persistent_kernel/tasks/blackwell/storage.cuh"
#include "mirage/persistent_kernel/tasks/blackwell/linear_fp8_1d2d_sm100.cuh"

using bfloat16 = nv_bfloat16;

// Wrapper kernel — 256 threads like MPK worker
__global__ __launch_bounds__(256) void fp8_linear_wrapper(
    kernel::tma::tma_2d<cutlass::float_e4m3_t, 3, 3, 3,
        1536, 7168, 128, 128, 7168, 1, 1, 1, 16384, true> tma_a,
    kernel::tma::tma_2d<cutlass::float_e4m3_t, 3, 3, 3,
        1, 7168, 16, 128, 7168, 1, 1, 1, 2048, true> tma_b,
    uint32_t const *weight_scale,
    uint32_t const *input_scale,
    kernel::tma::tma_2d<bfloat16, 0, 3, 3,
        1, 1536, 16, 128, 1536, 1, 1, 1, 2048, true> tma_out) {
  cute::Layout layout_Bias = cute::make_layout(cute::make_shape(1, 1536),
      cute::make_stride(1536, cute::Int<1>{}));
  cute::Tensor mBias = cute::make_tensor(
      cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>(nullptr)), layout_Bias);
  kernel::linear_fp8_1d2d_sm100_task_impl<cutlass::float_e4m3_t,
      decltype(tma_a), decltype(tma_b), decltype(mBias), decltype(tma_out),
      128, 16, 1, 1536, 7168, true, false, 4, 2, 2>(
      tma_a, tma_b, weight_scale, input_scale, mBias, tma_out);
}

int main() {
  cuInit(0);
  printf("FP8 linear test using MPK fill_tma_desc\n");

  // Allocate tensors
  int batch = 1, output = 1536, reduction = 7168;
  cutlass::float_e4m3_t *d_input, *d_weight;
  float *d_input_scale, *d_weight_scale;
  bfloat16 *d_output;
  cudaMalloc(&d_input, batch * reduction);
  cudaMalloc(&d_weight, output * reduction);
  cudaMalloc(&d_input_scale, batch * (reduction / 128) * sizeof(float));
  cudaMalloc(&d_weight_scale, (output / 128 + 1) * (reduction / 128) * sizeof(float));
  cudaMalloc(&d_output, batch * output * 2);
  cudaMemset(d_input, 0, batch * reduction);
  cudaMemset(d_weight, 0, output * reduction);
  cudaMemset(d_input_scale, 0, batch * (reduction / 128) * sizeof(float));
  cudaMemset(d_weight_scale, 0, (output / 128 + 1) * (reduction / 128) * sizeof(float));
  cudaMemset(d_output, 0, batch * output * 2);

  // Create TMA descs using MPK's fill_tma_desc (same as persistent kernel)
  CUtensorMap h_input_desc, h_weight_desc, h_output_desc;
  CUtensorMap *d_input_desc, *d_weight_desc, *d_output_desc;

  // Input TMA: [batch=1, reduction=7168], smem=[16, 128]
  {
    uint64_t gs[2] = {1, 7168};
    uint64_t gst[2] = {1, 7168};
    uint32_t ss[2] = {16, 128};
    mirage::runtime::fill_tma_desc<cutlass::float_e4m3_t, 3, 3, 3, 2>(
        &h_input_desc, d_input, gs, gst, ss, 1, 1);
  }
  // Weight TMA: [output=1536, reduction=7168], smem=[128, 128]
  {
    uint64_t gs[2] = {1536, 7168};
    uint64_t gst[2] = {1, 7168};
    uint32_t ss[2] = {128, 128};
    mirage::runtime::fill_tma_desc<cutlass::float_e4m3_t, 3, 3, 3, 2>(
        &h_weight_desc, d_weight, gs, gst, ss, 1, 1);
  }
  // Output TMA: [batch=1, output=1536], smem=[16, 128]
  {
    uint64_t gs[2] = {1, 1536};
    uint64_t gst[2] = {1, 1536};
    uint32_t ss[2] = {16, 128};
    mirage::runtime::fill_tma_desc<bfloat16, 0, 3, 3, 2>(
        &h_output_desc, d_output, gs, gst, ss, 1, 1);
  }

  cudaMalloc(&d_input_desc, sizeof(CUtensorMap));
  cudaMalloc(&d_weight_desc, sizeof(CUtensorMap));
  cudaMalloc(&d_output_desc, sizeof(CUtensorMap));
  cudaMemcpy(d_input_desc, &h_input_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
  cudaMemcpy(d_weight_desc, &h_weight_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
  cudaMemcpy(d_output_desc, &h_output_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);

  // Create tma_2d objects from device desc pointers (same as MPK worker)
  int smem = 224 * 1024;
  cudaFuncSetAttribute(fp8_linear_wrapper,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

  printf("Launching: grid=(1,1), block=256, smem=%d\n", smem);

  using TMA_A = kernel::tma::tma_2d<cutlass::float_e4m3_t, 3, 3, 3,
      1536, 7168, 128, 128, 7168, 1, 1, 1, 16384, true>;
  using TMA_B = kernel::tma::tma_2d<cutlass::float_e4m3_t, 3, 3, 3,
      1, 7168, 16, 128, 7168, 1, 1, 1, 2048, true>;
  using TMA_OUT = kernel::tma::tma_2d<bfloat16, 0, 3, 3,
      1, 1536, 16, 128, 1536, 1, 1, 1, 2048, true>;

  TMA_A tma_a(d_weight_desc);
  TMA_B tma_b(d_input_desc);
  TMA_OUT tma_out(d_output_desc);

  fp8_linear_wrapper<<<1, 256, smem>>>(
      tma_a, tma_b,
      (uint32_t const *)d_weight_scale,
      (uint32_t const *)d_input_scale,
      tma_out);

  auto err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("FAILED: %s\n", cudaGetErrorString(err));
    return 1;
  }
  printf("PASSED\n");

  cudaFree(d_input); cudaFree(d_weight); cudaFree(d_output);
  cudaFree(d_input_scale); cudaFree(d_weight_scale);
  cudaFree(d_input_desc); cudaFree(d_weight_desc); cudaFree(d_output_desc);
  return 0;
}
