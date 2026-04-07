# MTP Testing Guide

This document describes all components that need testing for the DeepSeek V3 MTP implementation, and how to run each test.

## Overview

| Component | Test File | Requires GPU | Status |
|-----------|-----------|-------------|--------|
| Verification logic (strict/probabilistic/synthetic) | `test_mtp_verification.py` | No | Reference impl tested, 23/23 pass |
| MLA attention reference | `test_mtp_verification.py` | No | Reference impl tested |
| MLA CUDA kernel vs reference | `test_mla_kernel.py` | Yes (any GPU) | Needs GPU |
| MTP config validation | `test_mtp_verification.py` (add) | No | TODO |
| DeepSeek V3 builder smoke test | `test_builder_smoke.py` | Yes (SM100) | TODO |
| End-to-end MTP decode | `test_e2e_mtp.py` | Yes (SM100) | TODO |

---

## Phase 1: Local Tests (No GPU Required)

### Verification Kernels + MLA Reference

These test the Python reference implementations that the CUDA kernels must match.

```bash
# From repo root, using a Python env with torch + pytest:
cd /path/to/mirage
python -m pytest tests/runtime_python/mtp/test_mtp_verification.py -v
```

**Expected: 23/23 tests pass** (~1 second)

Tests cover:
- **Strict verification** (7 tests): all-accepted, none-accepted, partial, single-draft, 7-drafts, last-position-mismatch
- **Probabilistic verification** (4 tests): greedy mode, sampling mode, deterministic seeds
- **Synthetic verification** (5 tests): high/zero rate, mismatch rejection, decay statistics, determinism
- **MLA attention reference** (5 tests): single-token, multi-head, causal property, output bounds, bf16 precision
- **Paged MLA reference** (2 tests): single-request, multi-token causal

---

## Phase 2: GPU Kernel Tests (Requires CUDA GPU)

### MLA Paged Attention Kernel

Compares the CUDA kernel implementation against the PyTorch reference.

#### Build the test extension:
```bash
cd tests/runtime_python/mtp
pip install -e .
```

If building on B200 (SM100), the setup.py may need architecture flags added:
```bash
TORCH_CUDA_ARCH_LIST="10.0" pip install -e .
```

#### Run the test:
```bash
python test_mla_kernel.py
```

**Expected output:**
```
MLA Paged Attention Kernel Tests
  Test: single request, single token — PASSED
  Test: multi-token causal — PASSED  
  Test: small heads — PASSED
```

**Tolerance: rtol=5e-2, atol=5e-2** (bf16 precision with scalar kernel)

Tests cover:
- Single request, 1 token, 32 KV entries
- Multi-token (4 tokens) with causal masking (MTP verify scenario)
- Shape validation for 16 Q heads

---

## Phase 3: Integration Tests (Requires SM100 / B200)

### Verification CUDA Kernels

The verification kernels (`target_verify_mtp.cuh`) are `__device__` functions called from within the persistent megakernel. To test them standalone, a wrapper kernel needs to be written (similar to `test_mla_kernel_wrapper.cu`).

**TODO**: Create `test_verify_kernel_wrapper.cu` and compare against Python references.

Quick validation approach (without wrapper):
```python
# The Python references in test_mtp_verification.py exactly match
# the CUDA kernel logic. If the Python tests pass, the CUDA kernels
# should produce identical results (they use the same algorithm and RNG).
# Full validation requires compiling and running the CUDA wrappers.
```

### DeepSeek V3 Builder Smoke Test

Verifies the computation graph can be constructed without errors.

```bash
# Requires DeepSeek V3 weights (converted to MPK format)
python -c "
from mirage.mpk.models.deepseek_v3.builder import DeepSeekV3Builder
# ... smoke test with dummy weights
"
```

**TODO**: Create `test_builder_smoke.py` with dummy weight tensors.

---

## Phase 4: End-to-End Tests (Requires SM100 + Weights)

### Full MTP Decode

```bash
python demo/deepseek_v3/demo.py --use-mirage --mtp --num-speculative-tokens 4
```

**Expected:**
- Produces coherent text output
- MTP acceptance rate > 70% on typical prompts
- Latency improvement vs non-MTP baseline

---

## Files Changed by MTP Implementation

### New CUDA Kernels
| File | Purpose | Test |
|------|---------|------|
| `include/.../blackwell/mla_attention_sm100.cuh` | MLA paged attention | `test_mla_kernel.py` |
| `include/.../speculative_decoding/target_verify_mtp.cuh` | 3 verification modes | `test_mtp_verification.py` (ref) |

### New Python Files
| File | Purpose |
|------|---------|
| `python/mirage/mpk/models/deepseek_v3/builder.py` | Model builder |
| `python/mirage/mpk/speculative.py` (modified) | MTPConfig |

### Modified C++ Registration
| File | Changes |
|------|---------|
| `include/.../runtime_header.h` | 5 new TaskTypes (266-270) |
| `src/kernel/graph.cc` | 5 new task name → type mappings |
| `src/kernel/task_register.cc` | 5 new registration functions |
| `src/kernel/runtime.cc` | 5 new task type names |
| `include/mirage/kernel/task_register.h` | 5 new declarations |

---

## Quick Test Script

Run all local tests:
```bash
#!/bin/bash
set -e
echo "=== Running MTP verification + MLA reference tests ==="
python -m pytest tests/runtime_python/mtp/test_mtp_verification.py -v
echo ""
echo "=== All local tests passed ==="
echo ""
echo "To run GPU tests on B200:"
echo "  cd tests/runtime_python/mtp && pip install -e . && python test_mla_kernel.py"
```

---

## Known Limitations

1. **MLA kernel performance**: Current scalar implementation is ~10-100x slower than the optimized 2-SM MLA kernel. Correctness-first; performance optimization is Phase 2+ work.

2. **MTP draft loop**: Currently a skeleton in the builder. Full autoregressive draft with KV cache management needs implementation.

3. **FP8 weights**: Dequantized to BF16 at load time. Fused FP8 GEMM would save memory.

4. **Cross-warp reduction**: The MLA kernel uses per-thread scalar computation (no MMA). This is correct but slow.

5. **kv_a_layernorm**: Should apply only to first 512 dims of the 576-dim KV vector. Currently skipped in the builder.
