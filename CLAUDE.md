# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Mirage?

**Mirage Persistent Kernel (MPK)** is a compiler and runtime that automatically transforms LLM inference into a single fused GPU megakernel. It reduces LLM inference latency by 1.2×–6.7× by keeping all computation and communication inside a single persistent kernel launch. The active development branch is `mpk`.

## Build & Install

```bash
# Full build (C++/CUDA + Rust + Cython)
pip install -e . -v
export MIRAGE_HOME=$(pwd)

# Skip native compilation (Python-only changes)
MIRAGE_SKIP_NATIVE_BUILD=1 pip install -e . -v
```

The build pipeline (`setup.py`) automatically:
1. Builds two Rust libraries via cargo: `abstract_subexpr` (e-graph rewriting) and `formal_verifier_equiv` (Z3 equivalence)
2. Compiles the C++/CUDA runtime via CMake
3. Cythonizes `python/mirage/_cython/core.pyx`

GPU targets and backends are controlled in `config.cmake` (`USE_CUDA`, `USE_NKI`).

## Running Tests

### Python unit tests
```bash
pytest tests/python/test_tensor_program.py -v
pytest tests/python/ -m "not impure"   # skip network/env-dependent tests
```

### Runtime CUDA extension tests (each subdirectory has its own `setup.py`)
```bash
# Build a runtime test extension, e.g. MTP kernels
cd tests/runtime_python/mtp
python setup.py build_ext --inplace
pytest test_mla_kernel.py -v
pytest test_mtp_verification.py -v

# Hopper attention, norm-linear, MoE tests follow the same pattern
cd tests/runtime_python/hopper
python setup.py build_ext --inplace
pytest test_*.py -v
```

### Integration / CI tests
```bash
# Qwen3 end-to-end correctness check (requires GPU + model weights)
bash tests/ci-tests/run_ci_tests_qwen3.sh

# Qwen2.5 latency regression test
bash tests/ci-tests/run_python_tests.sh before-installation   # baseline
bash tests/ci-tests/run_python_tests.sh after-installation    # optimized + compare
```

### Demo
```bash
python demo/qwen3/demo.py                         # PyTorch/Triton/FlashInfer baseline
python demo/qwen3/demo.py --use-mirage            # MPK megakernel
python demo/qwen3/demo.py --use-mirage --profiling  # + Perfetto timeline
```

## Architecture

### Layer overview

```
Python API  (python/mirage/)
  └── MPK subsystem  (python/mirage/mpk/)
Cython FFI  (python/mirage/_cython/core.pyx)
C++ Runtime (src/ + include/mirage/)
  ├── base/      – tensor IR, graph, layouts, types
  ├── kernel/    – operator specs, CUDA codegen
  ├── threadblock/ – block-level ops (matmul, RMSNorm, concat…)
  ├── transpiler/  – PTX codegen
  ├── triton_transpiler/ – Triton dialect
  └── search/    – optimization engine
        ├── abstract_expr/abstract_subexpr/  (Rust + egg)
        └── verification/formal_verifier_equiv/ (Rust + Z3)
CUDA headers (include/mirage/persistent_kernel/)
  └── tasks/{ampere,hopper,blackwell,cute,common,speculative_decoding}/
```

### Key Python modules

| Module | Role |
|---|---|
| `python/mirage/kernel.py` | `KNGraph` – kernel-level operator graph; operator fusion and CUDA compilation |
| `python/mirage/threadblock.py` | `TBGraph` – thread-block-level graph API |
| `python/mirage/mpk/persistent_kernel.py` | `PersistentKernel` – main MPK API: attach tensors, define layers, `compile()`, `__call__()` |
| `python/mirage/mpk/mpk.py` | `MPK` wrapper + `MPKMetadata` config dataclass |
| `python/mirage/mpk/models/` | Model builders: `qwen3/builder.py`, `deepseek_v3/builder.py` |
| `python/mirage/mpk/base_dynamic_shard_loader.py` | Multi-GPU weight sharding (COL_PARALLEL / ROW_PARALLEL) |
| `python/mirage/mpk/speculative.py` | Speculative decoding (lookahead, prompt lookup) |
| `python/mirage/mpk/multigpu.py` | NVSHMEM allreduce coordination |
| `python/mirage/_cython/core.pyx` | Cython bindings to C++ runtime; dtype constants; graph abstractions |
| `python/mirage/__init__.py` | Preloads native `.so` files (libz3, libabstract_subexpr, libformal_verifier) |

### CUDA task headers

Architecture-specific kernel implementations live under `include/mirage/persistent_kernel/tasks/`:
- `common/` – dtype helpers, shared primitives
- `hopper/` – SM90a warp-specialized persistent tasks (attention, norm-linear, MoE)
- `blackwell/` – SM100 tasks
- `ampere/` – SM80 tasks
- `cute/` – CuTe-based matmul variants
- `speculative_decoding/` – MTP verification, MLA attention

### Adding a new model

Follow the pattern in `python/mirage/mpk/models/qwen3/builder.py`:
1. Create `python/mirage/mpk/models/<model>/builder.py` subclassing `MirageModelConfig`
2. Map HuggingFace weight names to MPK tensors via `attach_input` / `new_tensor`
3. Register the builder in `python/mirage/mpk/model_registry.py`

### Adding a new task/kernel

1. Add CUDA `.cuh` header under `include/mirage/persistent_kernel/tasks/<arch>/`
2. Expose it through `PersistentKernel` in `python/mirage/mpk/persistent_kernel.py`
3. Add a unit test in `tests/runtime_python/` with its own `setup.py` (see `mtp/` for reference)

## Code style

C++ code uses Chromium style (`.clang-format`). Format is checked by CI (`code-format.yml`). Run `clang-format -i` on changed files before committing.
