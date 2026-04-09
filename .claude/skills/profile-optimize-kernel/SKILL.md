---
name: profile-optimize-kernel
description: Profile a CUDA kernel with NCU, identify bottlenecks, apply targeted optimizations, verify correctness, and repeat until performance target is reached.
argument-hint: <kernel-name> [--target-us <latency_us>] [--reps <N>]
allowed-tools: Bash, Read, Write, Edit, Grep, Glob, Agent
---

# Profile → Analyze → Optimize → Verify Loop

This skill drives an iterative profiling and optimization cycle for a CUDA kernel.
It uses NVIDIA Nsight Compute (NCU) for hardware-level analysis, identifies the
dominant bottleneck, applies one targeted optimization per iteration, verifies
correctness, and repeats until the performance target is met or no more wins are found.

## Machine Setup (B200 / SM100)

```bash
PYTHON=/raid/user_data/muhengl/.venv/bin/python
NCU=/usr/local/cuda-12.8/bin/ncu
CUDA_VISIBLE_DEVICES=0   # use a free GPU
```

## Workflow

### Step 0: Establish Baseline

Run the benchmark to record baseline latency before any changes:

```bash
cd ~/mirage/tests/runtime_python/mtp
CUDA_VISIBLE_DEVICES=0 $PYTHON test_mla_kernel.py 2>&1 | grep -A 20 "Benchmark"
```

Record numbers in a table: `Batch | KV len | Latency(us) | TFLOPS`.

### Step 1: Quick NCU Overview

Use `--set roofline` for a fast overview showing whether the kernel is compute-bound
or memory-bound, and the achieved SM throughput:

```bash
cd ~/mirage/tests/runtime_python/mtp
CUDA_VISIBLE_DEVICES=0 $NCU \
  --set roofline \
  --target-processes all \
  --kernel-name "mla_attention_test_kernel" \
  --launch-count 1 \
  --clock-control none \
  -o /tmp/mla_roofline \
  $PYTHON -c "
import torch, test_mla_kernel
q = torch.randn(1, 16*576, device='cuda', dtype=torch.bfloat16)
cache = torch.randn(32, 16, 576, device='cuda', dtype=torch.bfloat16)
c_new = torch.randn(1, 512, device='cuda', dtype=torch.bfloat16)
k_pe = torch.randn(1, 64, device='cuda', dtype=torch.bfloat16)
out = torch.zeros(1, 16*512, device='cuda', dtype=torch.bfloat16)
qi = torch.tensor([0,1], dtype=torch.int32, device='cuda')
ki = torch.tensor([0,2], dtype=torch.int32, device='cuda')
idx = torch.arange(2, dtype=torch.int32, device='cuda')
lpl = torch.tensor([16], dtype=torch.int32, device='cuda')
test_mla_kernel.mla_attention(q, cache, c_new, k_pe, out, qi, ki, idx, lpl, 1)
" 2>&1 | grep -E "Section|sm_|Memory|Compute|DRAM|Warp"
```

### Step 2: Full NCU Metrics (Stall Analysis)

When the bottleneck isn't clear from the overview, collect stall reasons and
instruction mix:

```bash
CUDA_VISIBLE_DEVICES=0 $NCU \
  --metrics \
sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_bytes_pipe_lsu_mem_shared_op_ld.sum,\
dram__bytes_read.sum,\
smsp__warp_issue_stalled_wait_mio_throttle_per_warp_active.pct,\
smsp__warp_issue_stalled_barrier_per_warp_active.pct,\
smsp__warp_issue_stalled_dispatch_stall_per_warp_active.pct,\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,\
smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_tex_throttle_per_warp_active.pct \
  --target-processes all \
  --kernel-name "mla_attention_test_kernel" \
  --launch-count 1 \
  --clock-control none \
  $PYTHON -c "..." 2>&1
```

### Step 3: Bottleneck Decision Tree

Based on NCU output, identify the dominant bottleneck:

```
sm__throughput < 10%?
  → Latency-bound / launch overhead
    • Kernel is too small or has too many sequential operations
    • Fix: increase parallelism (more blocks, larger tiles)

sm__throughput > 10%, tensor_op < 50%?
  → Memory-bound (smem or DRAM)
    Look at stall reasons:
    • stalled_long_scoreboard high → DRAM latency → add async prefetch
    • stalled_mio_throttle high → shared memory bank conflicts → fix smem layout
    • stalled_barrier high → sync overhead → reduce __syncthreads()
    Fix: improve memory access patterns, prefetch, vectorize loads

tensor_op > 50%, sm__throughput < 50%?
  → Compute-bound but low occupancy
    • warp_active low → few warps in flight
    • Fix: increase warps per block or blocks per SM

tensor_op > 50%, sm__throughput > 80%?
  → Near roofline, diminishing returns
    → Consider algorithmic changes (e.g. split-K, Flash2 vs Flash3 style)
```

### Step 4: Apply One Optimization

Apply **one** targeted change per iteration. Common optimizations for decode attention:

#### 4A. Increase Block Parallelism
If only 1 block/request with few threads → launch one block per Q-head:
```cpp
// Old: 1 block handles all NUM_Q_HEADS sequentially
// New: 1 block per Q-head, each block does its own QK^T + PV
// Change grid from (num_requests) to (num_requests * NUM_Q_HEADS)
// Remove qh_pass outer loop, use blockIdx.y for Q-head index
```

#### 4B. Add cp.async / Async Smem Prefetch
Replace blocking loads with cp.async to hide DRAM latency:
```cpp
// Replace: s_k[0][i] = d_cache[i];
// With:    __pipeline_memcpy_async(&s_k[0][i], &d_cache[i], 16);
//          __pipeline_commit(); ... __pipeline_wait_prior(1);
```

#### 4C. Increase Tile Size (KV_TILE_SIZE)
Larger tiles amortize load overhead, improve MMA utilization.
Current: KV_TILE_SIZE=32. Try 64 or 128 (check smem budget).

#### 4D. Vectorized Loads (16-byte)
Replace element-wise loads with 128-bit loads:
```cpp
// Replace loop with float4/uint4 vectorized loads
```

#### 4E. Fuse Q-head Passes (if warps underutilized)
If warps in a block are idle while waiting, use warp specialization.

### Step 5: Rebuild and Verify

After each change:

```bash
cd ~/mirage/tests/runtime_python/mtp
rm -rf build/
PATH=/usr/local/cuda-12.8/bin:$PATH CUDA_HOME=/usr/local/cuda-12.8 \
  TORCH_CUDA_ARCH_LIST="10.0" $PYTHON setup.py build_ext --inplace 2>&1 | grep -E "error:|building"

CUDA_VISIBLE_DEVICES=0 $PYTHON test_mla_kernel.py 2>&1
```

**Correctness gate**: max diff ≤ 0.05 on all 3 tests before accepting any optimization.

### Step 6: Record and Repeat

After each iteration, record the new latency table. If improvement > 5%, go to Step 1.
If improvement < 5% or 3 iterations find no wins, escalate to architectural change.

## Performance Targets

**2SM SM100a reference kernel** (mla_sm100_2sm.cuh, tested 2026-04-07):
| Config | Latency | TFLOPS |
|--------|---------|--------|
| bs=1 kv=512 | 59 μs | 2.4 |
| bs=1 kv=1024 | 68 μs | 4.2 |
| bs=1 kv=4096 | 131 μs | 8.7 |
| bs=8 kv=1024 | 68 μs | 33.5 |

**Our MMA kernel** (after per-Q-head blocks + vectorized loads):
| Config | Latency | vs 2SM |
|--------|---------|--------|
| bs=1 kv=512 | 294 μs | 5.0× slower |
| bs=1 kv=1024 | 570 μs | 8.4× slower |
| bs=1 kv=4096 | 2306 μs | 17.6× slower |

Remaining gap is fundamental: ldmatrix+mma.sync vs TMA+TMEM+UMMA.
Possible further optimizations: larger KV tile (48 fits with 213KB smem), cp.async, split-KV.
Long-term: FlashInfer integration (requires MPK framework host-side CUTLASS Params support).

## File Locations

| File | Role |
|------|------|
| `include/.../blackwell/mla_attention_sm100.cuh` | Our MLA kernel |
| `tests/runtime_python/mtp/test_mla_kernel.py` | Benchmark + correctness |
| `tests/runtime_python/mtp/test_mla_kernel_wrapper.cu` | CUDA extension wrapper |
| `tests/runtime_python/mtp/setup.py` | Build config |
| `tests/runtime_python/blackwell/sm100_mla/` | FlashInfer reference |

## Quick Reference: Key NCU Metrics

| Metric | Good range | If bad |
|--------|-----------|--------|
| `sm__throughput` | > 60% | Low parallelism or mem stall |
| `tensor_op_hmma` | > 40% | MMA underutilized |
| `warp_active` | > 50% | Low occupancy |
| `stalled_barrier` | < 10% | Too many __syncthreads |
| `stalled_long_scoreboard` | < 20% | DRAM latency |
| `stalled_mio_throttle` | < 15% | Smem bank conflicts |
