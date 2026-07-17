# CLAUDE.md

Guidance for Claude Code when working in this repository.

**No standing performance goal is encoded here — the user sets the goal per session.** The
2026-05→07 DeepSeek-V3 decode campaign is CONCLUDED (record: `experiment_history/`, the
anti-loop archive; methodology: the `.claude/skills/v2-*` suites). Do not resume it unasked.

## Project

Mirage Persistent Kernel (MPK): a compiler + runtime that fuses an entire LLM inference
pass (compute + TP communication) into a single persistent CUDA megakernel. `mpk` is the
long-lived integration branch (PRs land there); `main` is the legacy superoptimizer
(kernel search — not part of the MPK runtime path). This worktree: `dsv3-decode-clean`.

Two runtimes coexist:

- **Runtime v1 (default)** — dynamic scheduler kernel pushes `TaskDesc`s into per-worker
  (per-SM) queues; task kernels in `include/mirage/persistent_kernel/tasks/{blackwell,hopper}/`.
- **Runtime V2 (opt-in, demo flag `--use-v2`)** — static per-SM task plan (no dynamic
  scheduler), role-split warps per worker (loader W4=TMA, launcher W5=tcgen05/TMEM,
  consumers W0-3=math/epilogue, storer W6=page release); kernels in `tasks/blackwell_v2/`,
  runtime in `runtime_v2.cuh`/`persistent_kernel_v2.cuh`. Without `--use-v2` the built
  megakernel is byte-identical to a v1-only tree.

## Current state

- **Runtime V2 runs DeepSeek-V3 decode end-to-end** at TP8 EP2 (opt-in `--use-v2`), and
  the Qwen3 demo runs `--use-v2` single-GPU. v1 remains the default path for both.
- **DSv3 FFN W13/W2 per-tile TMA+tcgen05 pipeline kernels** are landed env-gated
  (`MPK_DSV3_V2_FFN_PIPE=1`, default OFF, only meaningful under `--use-v2`).
- **Three skill suites are the entry points for substantive work — load the relevant one
  at task start:**
  - `.claude/skills/v2-kernel-writing` — write / port / optimize any Runtime-V2 task kernel.
  - `.claude/skills/v2-model-support` — bring a model up on Runtime-V2 end-to-end.
  - `.claude/skills/v2-perf-iteration` — run a perf campaign (measure→plan→implement→re-measure→record).
- **v2 runtime races: ALL KNOWN RACES FIXED (2026-07-16).** Three distinct mechanisms
  were root-caused and fixed: race-1 launcher-ITS early page release (`689dadc5`),
  race-2 consumer-suffix release-before-loader-claim (`7d271a01` Design E + `7b6ae2bb`
  consumer-TOTAL lifecycle for the FFN GEMV chain; nwarps=7 structurally excluded with
  proof), race-3 iteration-barrier half-exit (`025029a1`). A plan-time assertion now
  guards the mixed-chain page-window invariant (it caught a real qwen3 plan shape).
  One honest boundary (documented in `7b6ae2bb`, never observed): launcher-owned
  releases (pipeline linears) retain the baseline protocol's theoretical two-pending
  property vs a lagging launcher — pre-existing, unchanged, out of scope. If a NEW
  wedge appears: read the commit messages of these four commits first (each carries
  the fingerprint method: role positions + page-parity arithmetic + state dump).

## Build

```bash
pip install -e . -v          # editable install; CMake + Cython
```

- After editing `src/**/*.cc` or `include/mirage/**/*.h`, re-run the install; if Python still
  lags, the Cython `.so` is stale: `rm python/mirage/core.*.so && touch python/mirage/_cython/*.pyx`, re-install.
- Editing `.cuh` task kernels needs **no rebuild** — they are JIT-compiled by `nvcc`
  when the megakernel is assembled at launch (~10-15 min; recompiled every launch, no
  .so cache exists).

## Test / lint

```bash
bash scripts/format.sh                     # clang-format-15; CI enforces this
pytest tests/python/                       # legacy search/transpiler tests
pytest tests/runtime_python/               # MPK runtime tests (test_mode/ needs no GPU)
pytest tests/runtime_python/blackwell_v2/  # Runtime-V2 kernel harness
```

The `test-mode` skill covers unit-testing one task end-to-end. Run `format.sh` before pushing.

## Running demos

```bash
# Qwen3 (single GPU; add --use-v2 for the v2 runtime)
python demo/qwen3/demo.py --use-mirage [--use-v2]

# DeepSeek-V3 — minimal single-GPU sanity run
python demo/deepseek_v3/demo.py --model-path /path/to/DeepSeek-V3 \
  --use-mirage --correctness --layers 3 --max-num-batched-tokens 1 \
  --max-seq-length 512 --max-new-tokens 1
```

Multi-GPU needs `mpirun -np $TP` + NVSHMEM/MPI env — **read `demo/deepseek_v3/readme.md`
first**; the CLI surface is large and flag names are non-obvious. DSv3 weight conversion
is slow cold: set `MPK_DEEPSEEK_WEIGHT_CACHE_DIR`.

## Architecture: the compile path in one pass

1. **Builder** (`python/mirage/mpk/models/<model>/builder.py`) composes the task graph by
   calling fused-op methods on `PersistentKernel` (`python/mirage/mpk/persistent_kernel.py`).
   It also owns tensor lifetimes/aliasing (`new_tensor` vs `attach_input`) and TP shard rules.
2. **Graph registration** (`src/kernel/graph.cc::register_task`) maps task-type name →
   `TASK_*` enum + `(num_inputs, num_outputs, enum, variant)` tuple + register fn.
3. **Task codegen** (`src/kernel/task_register.cc`) emits each task's C++ snippet into the
   megakernel dispatch switch; snippets call device functions in the `tasks/**/*.cuh` headers.
4. **Runtime JIT** (`src/kernel/runtime.cc::print_task_graph`) writes `test.cu` (the whole
   megakernel) and invokes `nvcc -arch=sm_100a`. Config constants
   (`MPK_MAX_NUM_BATCHED_TOKENS`, page/seq sizes, …) bake in here.

V2 differences: under `use_v2_runtime=True` the builder lowers to a static per-SM plan
(task slots, events, rings) instead of scheduler queues, and registration goes through
`register_v2_task_role_variant` with per-role code strings. Deep dives: the
`mpk-internals` skill (v1 pipeline) and `v2-kernel-writing/references/` (v2 protocol).

## Key invariants / footguns (timeless)

- **`grid_dim` must match the kernel's real parallelism**, and the `dim_maps` (3rd arg of
  `tb_graph.new_input`) must line up with it — wrong maps silently produce wrong per-task
  pointer offsets.
- **Outputs are often passed as `new_input(store_in_dmem=True)`** (the MPK convention):
  the graph.cc tuple must then be `(N+1, 0, ...)` — not `(N, 1, ...)` — and codegen reads
  `input_ptrs[N]` for the output.
- **SMEM budgets:** v1 dynamic SMEM ≈ 205 KB/worker (B200's 227 KB minus static overhead;
  over-tiling ⇒ runtime `Invalid __shared__ write`). v2 region plan: 16 KB pages, ≤ 14
  pages, total ≤ 224256 B (`v2-kernel-writing/references/house-style.md` §4).
- **Every task's `extern __shared__` must be `__align__(1024)`** — a smaller alignment
  misaligns *other* tasks' TMA/AR in the shared TU (`cudaErrorMisalignedAddress`); only an
  in-MPK run catches it, so probe `--layers 0-3` in-MPK before trusting any port.
- **v2 §1.1 dep-prefix (lethal):** every v2 consumer body MUST begin with the emitted
  `consumer_dep_prefix(...)` (cross-SM event spin + per-slot `SEM_DEP_READY` arrive) —
  skipping it silently deadlocks the *next* occupant of that ring slot.
  `v2-kernel-writing/references/wiring-recipe.md` §1.1.
- **`skip_after_step0` for monotonic barriers:** any v2 task that zeroes
  monotonic-grid-barrier scratch must guard the memset with `step == 0`. Re-zeroing every
  step gives the iter-0-fine / iter-1-hang signature.
- **Task-type IDs 231..256 are the "TMA range":** weight `CUtensorMap`s are auto-created
  for enums inside it. A TMA-consuming task with an enum outside it needs the explicit
  task-type list in `runtime.cc` (~:1458) + a `tma.cuh` case (tasks 356/357 = the pattern).
- **TP shard rules live in TWO places:** the builder's shard-rule regex list AND
  `demo.py`'s weight-conversion pass — update both. Bump the weight-cache version string
  whenever conversion logic changes (the key can't see code; stale weights are silent).
- **DSv3 dual-dispatch Q_LEN gates:** prefill tasks early-return at `Q_LEN <= 8`, decode
  at `Q_LEN > 8`; the decode chain is also the live prefill path — never delete one side
  unconditionally.
- **GPU safety:** never crash-loop the megakernel — each crash can leave an unkillable
  D-state zombie (reboot-only). Validate in test-mode first; timeout-guard every
  demo/mpirun run (a hung megakernel emits no signal).

## Tunable env vars (current)

- `MPK_DSV3_V2_FFN_PIPE=1` — (v2-only, default OFF) route DSv3 FFN W13/W2 through the
  per-tile TMA+tcgen05 pipeline tasks (enums 356/357) instead of the FFN mega path.
- `--profiling` demo flag (compiles `MPK_ENABLE_PROFILING`) — per-task profiler; v2 traces
  export via `scripts/v2_perfetto_export.py`. The former profiled-only wedges were the v2
  races, all fixed (see Current state); profiled runs pass the former wedge windows
  (L6-profiled 2/2 PASS post-fix). Instrumentation still skews timing — never quote an
  instrumented run as a perf baseline.
- `MPK_CONVERT_SEMAPHORE=K` — cap concurrent per-rank DSv3 weight conversion (host-OOM
  guard for TP8 cold starts; default off).
- `MPK_DISABLE_DIRECT_PAGED_DECODE_KV=1` — force dense KV gather (MLA decode regression
  isolation).
- `MPK_RDC_FALSE=1` — (Blackwell) legacy self-contained allreduce path, for
  NVSHMEM/`-rdc=true` bisection. `-rdc=true` is the default and works on SM100a.
- `MPK_MLA_TP4_V_SPLITS`, `MPK_MLA_TP4_HEAD_GROUPS` — v1 TP4 MLA decode tuning knobs.
- `MPK_DEEPSEEK_WEIGHT_CACHE_DIR=/path` — DSv3 weight-conversion cache dir.

## Adding a task / model (pointers)

- **v1 task:** enum in `runtime_header.h` → `task_register.cc` → `graph.cc` →
  `runtime.cc` (name + metadata handler) → `.cuh` in `tasks/blackwell/` → wrapper in
  `persistent_kernel.py` → builder call site. Skill: `add-mpk-task`.
- **v2 task:** follow the `v2-kernel-writing` skill; the 8-file wiring checklist is its
  `references/wiring-recipe.md`.
- **Model:** `python/mirage/mpk/models/<name>/builder.py` + `model_registry.py` +
  `demo/<name>/`. Skills: `add-mpk-model` (v1), `v2-model-support` (v2).
- **Task unit test:** `tests/runtime_python/blackwell*/sm100_<task>/`. Skill: `test-mode`.
- **Kernel-optimizer dispatch:** `.claude/skills/v2-kernel-writing/references/ferret-v2-dispatch.md`.

## In-tree references

- `demo/deepseek_v3/readme.md` — full DSv3 run reference, env, known limitations
- `README.md` / `INSTALL.md` — MPK overview + install
- `NCU_Usage_Manual.md` — profiling MPK kernels with Nsight Compute
- `FUSED_KERNEL_DEBUG_METHODOLOGY.md` — **read before debugging a fused megakernel**
  (order-of-operations + the 6 gate-fidelity classes; written after ~9 wasted debug rounds)
- `experiment_history/` — **historical archive** of the concluded DSv3 campaign; read
  `INDEX.md` for anti-loop when a new proposal maps to an old lever, else skip it.
- `WORKFLOW.md` — superseded stub; the live loop doc is the `v2-perf-iteration` skill.
- Remote-box playbook: `.claude/skills/v2-model-support/references/box-orchestration.md`
  (box availability changes — verify with the user before assuming any host exists).

## Security / hygiene

- **Mask infra IPs, hostnames, and key paths before pushing any doc to a shared/public
  remote.** Never commit credentials. (`CLAUDE.md` IS tracked — keep it mask-clean.)
- Never commit `scratch/`, `outputs/`, `experiment_history/`, weight caches, generated
  `test.cu`/`.so` artifacts, or local helper scripts unless the user asks.
- Land perf levers env-gated default-OFF; keep the default build byte-identical unless a
  default-flip is explicitly justified and measured.
