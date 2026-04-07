---
name: port-kernel
description: Port a high-performance CUDA kernel from vLLM/SGLang/FlashInfer into the Mirage MPK persistent kernel framework. Use this when you need to implement or replace a kernel for DeepSeek V3 MTP support.
argument-hint: <kernel-name> [--source vllm|sglang|flashinfer]
allowed-tools: Bash, Read, Write, Edit, Grep, Glob, Agent
---

# Port Kernel to MPK

This skill ports a high-performance CUDA kernel from reference implementations (vLLM, SGLang, FlashInfer) into the Mirage MPK persistent kernel framework.

## Three-Step Process

### Step 1: Find Reference Kernel

Search the reference repos for the best available implementation of the target kernel.

**Search locations:**
- FlashInfer: `~/claude/flashinfer/`
  - Attention: `include/flashinfer/attention/decode.cuh`, `prefill.cuh`
  - MLA: `flashinfer/gdn_kernels/`, the MLA kernel is already in our repo at `include/mirage/persistent_kernel/tasks/blackwell/mla_sm100_2sm.cuh`
  - Sampling: `include/flashinfer/sampling/`
  - Mamba/SSU MTP: `include/flashinfer/mamba/`
- vLLM: `~/claude/vllm/`
  - Attention: `csrc/attention/`, `vllm/attention/backends/`
  - Speculative: `vllm/v1/spec_decode/utils.py` (Triton kernels), `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py`
  - MTP model: `vllm/model_executor/models/deepseek_mtp.py`
- SGLang: `~/claude/sglang/`
  - Speculative: `sgl-kernel/csrc/speculative/`
  - Attention: `python/sglang/srt/layers/attention/`

**Evaluation criteria for choosing the source:**
1. Does it target SM100 (Blackwell)? SM90 (Hopper)? Or generic?
2. Does it use hardware-specific features (TMA, UMMA, TMEM, cluster)?
3. Is it a `__global__` kernel or can it be adapted to a `__device__` function?
4. What's its performance relative to alternatives?

### Step 2: Adapt to MPK Format

MPK kernels must be `__device__ __forceinline__` functions with this pattern:

```cuda
// File: include/mirage/persistent_kernel/tasks/blackwell/<kernel_name>_sm100.cuh
#pragma once
#include "tasks/common/common_header.cuh"
#include <cutlass/arch/barrier.h>

namespace kernel {

template <typename T, int PARAM1, int PARAM2, ...>
__device__ __forceinline__ void <kernel_name>_sm100_task_impl(
    void const *input_ptr_0,
    void *output_ptr_0,
    // ... additional pointers
    // For attention-like kernels, also:
    int const *qo_indptr_buffer_ptr,
    int const *paged_kv_indptr_buffer_ptr,
    int const *paged_kv_indices_buffer_ptr,
    int const *paged_kv_last_page_len_buffer_ptr,
    int16_t request_id) {

  // Guard: only use NUM_THREADS threads (128)
  constexpr int BARRIER_ID = 6;
  cutlass::arch::NamedBarrier barrier(NUM_THREADS, BARRIER_ID);
  if (threadIdx.x >= NUM_THREADS) return;

  // ... kernel body ...
}

} // namespace kernel
```

**Integration checklist:**
1. [ ] CUDA kernel file created in `include/mirage/persistent_kernel/tasks/blackwell/`
2. [ ] TaskType enum added to `include/mirage/persistent_kernel/runtime_header.h`
3. [ ] Task name mapping added to `src/kernel/graph.cc`
4. [ ] Registration function added to `src/kernel/task_register.cc` and `.h`
5. [ ] Task name string added to `src/kernel/runtime.cc`
6. [ ] Include added to `include/mirage/persistent_kernel/tasks/blackwell/task_header.cuh`
7. [ ] Python method added to `python/mirage/mpk/persistent_kernel.py`

**Key constraints:**
- Must be a `__device__` function (no `__global__`, no kernel launch)
- NUM_THREADS = 128 (4 warps of 32 threads)
- Shared memory: up to ~228KB dynamic on SM100
- Synchronization: `cutlass::arch::NamedBarrier` only (no cluster sync, no TMA descriptors from host)
- Cannot use features requiring host-side setup: TMA descriptors, TMEM allocation, cluster launch
- CAN use: MMA (m16n16k16, UMMA), shared memory, warp shuffles, cp_async

**Adaptation strategies when source kernel uses incompatible features:**
- **Source uses TMA**: Replace with `cp_async` or direct global memory loads
- **Source uses TMEM**: Replace with shared memory or register accumulation
- **Source uses cluster sync**: Restructure to per-warp independent computation
- **Source uses kernel launch**: Extract core computation loop as device function
- **Source is Triton**: Translate to equivalent CUDA with MMA

### Step 3: Port and Validate

1. Create the adapted CUDA kernel file
2. Register it in the task system (all 7 files)
3. Write a Python reference implementation for correctness testing
4. Write a CUDA test wrapper (`test_<kernel>_wrapper.cu`) + Python test
5. Verify output matches the reference within tolerance (bf16: rtol=1e-2, atol=1e-2)

## Example Usage

```
/port-kernel mla_attention --source flashinfer
/port-kernel rejection_sampler --source vllm
/port-kernel moe_gate_routing --source sglang
```

## Reference: Existing MPK Kernels as Templates

Study these for the correct integration pattern:
- `attention_sm100.cuh`: Paged attention with MMA, online softmax, multi-token
- `linear_sm100_mpk.cuh`: CuTe GEMM with TMA and UMMA
- `moe_linear_sm100.cuh`: MoE expert linear with routing
- `target_verify.cuh`: Simple speculative verification kernel

## Reference: Kernel Registration Pattern

In `task_register.cc`, the code generation follows this pattern:
```cpp
int TaskRegister::register_<kernel>_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // Extract params
  int param1 = params[0];
  // ...
  
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::<kernel>_task_impl<bfloat16, $, $>(", param1, param2);
  code.e("    task_desc->input_ptrs[0],");
  // ...
  code.e("    task_desc->output_ptrs[0]);");
  return register_task_variant(TASK_<KERNEL>, code.to_string());
}
```
