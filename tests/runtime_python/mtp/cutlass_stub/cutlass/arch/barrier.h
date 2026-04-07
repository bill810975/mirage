#pragma once
// Minimal stub for cutlass::arch::NamedBarrier used by mla_attention_sm100.cuh.
// Replaces the full CUTLASS barrier.h to avoid SM90 TMA cascade includes.

namespace cutlass {
namespace arch {

struct NamedBarrier {
  __device__ __forceinline__ NamedBarrier(int /* thread_count */, int /* id */) {}
  __device__ __forceinline__ void arrive_and_wait() { __syncthreads(); }
  __device__ __forceinline__ void arrive() {}
  __device__ __forceinline__ void wait() { __syncthreads(); }
};

} // namespace arch
} // namespace cutlass
