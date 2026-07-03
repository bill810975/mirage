# Runtime-V2 per-task correctness + performance framework

Test framework for MPK Runtime V2 (`PersistentKernel(use_v2_runtime=True)`,
`-DUSE_RUNTIME_V2`, static per-SM task plan, warp-specialized roles).
Everything runs through the REAL v2 pipeline: Python layer API → task
registration → v2 queue/smem planning → JIT nvcc → `launch_v2_func`.

```
pytorch_reference.py   fp32 references + uniform compare_metrics()
v2_harness.py          case builders/runners (correctness + perf)
v2_prof_decode.py      v2 8-track profiler buffer → per-(task,iter) table → verdict metrics
case_runner.py         subprocess entry (one case per process; hang/crash isolated)
run_suite.py           orchestrator: case matrix, GPU pinning, timeouts, tables
e2e_qwen3_check.sh     qwen3 demo v1-vs-v2 e2e calibration (tokens + ms/tok + trace)
_results/<tag>/...     artifacts (specs, logs, outputs .pt, raw prof buffers, tables)
```

## Correctness harness

One test-mode (single-pass) graph per (runtime, M) containing INDEPENDENT
ops, each with deterministically-seeded inputs (`gen_tensor(name, ...)`:
seed = sha1(name), so v1 and v2 subprocesses see bit-identical inputs).
`prompt_lengths = mbt = M` so runtime-M kernels (silu_mul, rmsnorm) and
static-M kernels (linear v2/v3: `m_real` = output dim0, a compile-time
template arg) agree on the row count.

Checks per op:
- vs torch fp32 reference: `cos`, `rel_max` (max|Δ| / max|ref|), `rel_l2`,
  NaN scan. PASS = cos ≥ 0.999 ∧ rel_max ≤ 3e-2 ∧ no NaN.
- vs v1 counterpart (same inputs through the v1 kernel, separate process):
  cos / rel_max / bf16 bit-exact fraction. Mapping:
  rmsnorm_v2↔rmsnorm_hopper, silu_mul_v2↔silu_mul,
  linear v2 & v3↔linear_sm100 (cutlass), linear_with_residual v2/v3↔
  linear_with_residual. Bit-exactness is NOT required for the linears
  (different accumulation order); it IS expected for the elementwise ops.

M contract (framework-discovered, 2026-07-02): the v2 linear family
processes ONE 16-row activation tile per task (`BLOCK_N=16` in
linear_spec.h, consumer `M_REAL<=16`); at M>16 rows 16+ were silently left
uncomputed (caught by this suite at M=128: cos=√(1/8), bitexact=16/128 —
two independent metrics both giving 16/128). The v2-variant kernel has the
same constants, so this is NOT a v3 regression; whether 16 rows is the
intended contract or a legacy limitation is not established by the tile
constant — either way it permanently bars the v2 linear family from
mbt>16 prefill until the kernel grows multi-A-tile support. The layer
methods now assert `output.dim(0) <= 16` (all 4 registration paths;
task_register.cc asserts backstop direct kn_graph use), and the
correctness matrix runs linears at M ≤ 16 only (elementwise ops cover
M=128).

## Perf harness — verdict metric definition

Graph: a serialized CHAIN of L identical blocks (block output feeds the next
block, enforcing dataflow serialization at the op level; every block has its
OWN weight copies). Chains (real Qwen3-8B decode shapes):
- `mlp`: rmsnorm(4096) → linear[24576,4096] (gateup) → silu_mul(G=48) →
  linear+res[4096,12288] (down)
- `qkv`: rmsnorm(4096) → linear[6144,4096] (qkv) → linear+res[4096,6144]
- `sq` : linear[4096,4096] × L

Cold-L2 policy: per iteration the chain streams L×(all block weights)
(mlp: ≈1.2 GB/iter at L=4) ≫ B200 L2 (126 MB), so every weight read is a
cold stream — the same regime as production decode, where 16 GB of weights
pass through L2 every token. No explicit flush kernel can be injected into
a megakernel iteration; the chain's aggregate footprint IS the flush.
The `sq` chain is forced to L≥8 (at L=4 its per-weight reuse distance
≈96 MB would fit L2 and warm-contaminate the numbers).
(History: warm-L2 per-task gates over-stated wins ~2.5×; see
`feedback_gate_warm_isolated_vs_mpk_cold_barrier`.)

Drive: non-test offline mode, `prompt_len=1`, `max_seq_length=S` (=v2
iteration count). With a profiler tensor attached, `MPK_ENABLE_PROFILING`
makes the v2 loop run all S iterations and record the LAST
`V2_PROF_WINDOW_ITERS=25` in the 8-track role profiler. Iterations `0..S-2`
are live M=1 decode steps; iteration `S-1` is the single post-done zero-M
iteration (prepare_next_batch zeroes qo_indptr) — the decoder DROPS it, and
also drops the first window iteration (window-flip edge fuzz). S=32 ⇒ 23
clean iterations × L instances of samples per op.

M>1 perf is meaningful for the LINEAR family only (static m_real = mbt ⇒
full M rows computed every iteration regardless of the runtime batch);
runtime-M ops (rmsnorm/silu) cannot be held at M>1 across a decode window.

Metrics (from the consumer role track, group 0; ns from %globaltimer):
- `body_span` p50/p90 — **the primary per-op verdict**: per-sample
  consumer span MINUS that task's V2_DEP_WAIT. V2_DEP_WAIT is exact
  (emitted around every `wait_task_dependency`, NOT thresholded — only the
  MPK_V2_TIMED_WAIT phase slices have a 2 µs threshold), so this isolates
  the op's own cost from chain-position wait.
- `task_span` p50/p90: consumer BEGIN→END per tile-task, INCLUDING dep-wait
  — the production-visible scheduled span at the production grid (136
  workers). In a serialized chain this is structurally wait-dominated for
  downstream ops; report it alongside body, never alone.
- `op_wall` p50/p90: per op instance per iteration,
  max(consumer END) − min(consumer BEGIN) over the instance's tasks — the
  op's critical-path contribution (includes the dependency ramp).
- `dep_wait` p50/p90: V2_DEP_WAIT inside the consumer window.
- `loader_ahead`: loader BEGIN of task k vs consumer END of task k−1 on the
  same SM, same iteration only (across the iteration barrier it would
  measure go-wait, not pipelining). Positive ⇒ the loader is prefetching
  the next task while the consumer still executes the previous one — the
  v2 pipeline-overlap signature (must be visible for linear v3, else the
  plumbing or the decode is wrong).
- `load_hidden`: consumer END − loader END for the same task (>0 ⇒ the load
  finished under the consume).
- `iter_wall` p50: per-iteration max−min consumer timestamps — cross-checked
  against the UNPROFILED sibling case (`*_nowall`, same graph compiled
  without MPK_ENABLE_PROFILING): `wall_ms / final_step` must bracket it
  (profiled iter_wall excludes prepare/go barriers; the unprofiled wall
  includes them plus launch overhead).

Trust gates (framework self-validation, all must hold before numbers are
believed — `_decode_ok` in the summary aggregates 1):
1. Decode integrity (hard failures): BEGIN/END alternation, per-(SM,role)
   pair count a whole multiple of the SM's filtered queue, task_type of
   every pair equals the queue's expected type, profiler overflow drop
   counters (MISC region) == 0.
2. Reproducibility: 3 independent runs (fresh process) ⇒ body_span_p50
   spread < 5%.
3. Wall cross-check: Σ op_wall_p50 over the chain ≤ iter_wall_p50 ≤
   unprofiled wall_ms/iter, with iter_wall/wall-per-iter ≥ ~0.7 (barriers
   and prepare are the remainder).
4. Overlap visibility: linear v3 loader_ahead_pos_frac > 0.5 and
   dep/tmem/mainloop phase slices present on the phase tracks.

## Running

```bash
# correctness (v1+v2, M=1/8/128), GPU 6:
.venv/bin/python tests/runtime_python/blackwell_v2/run_suite.py --what correctness --devices 6
# perf (chains × {v2-v3, v2-v2, v1}), 3 repeats:
.venv/bin/python tests/runtime_python/blackwell_v2/run_suite.py --what perf --devices 6 --repeats 3
```

Each case is a subprocess with a hard timeout (default 2400 s) and
`CUDA_VISIBLE_DEVICES` pinning; a hang kills only that case. Compiled
launchers (`mpk_launcher_rank0*.so`) are saved per case dir and reloadable
via `load_mpk_kernel` (v2 binding fixed in persistent_kernel.py).

## Observed liveness anomaly (UNRESOLVED; profiled-small-chain wedge-prone)

Two related phenomena, both isolated by the framework's gates:

STALLS: some profiled runs show sporadic iterations (irregular spacing,
0.5–2.5 ms) with a GLOBAL stall — every task of the iteration's final op
waits in V2_MAINLOOP_WAIT simultaneously on all SMs (loader+launcher
stalled the same span ⇒ the data path, not one straggler; the serialized
chain makes it land on the last op because nothing downstream hides it).
Census (`stall_census.py`): 4/25 iterations in one run concurrent with
heavy box co-activity; 0/25 in quiet runs on both GPU 5 and GPU 6 with
body_p50s matching ≤1.5%. p50 verdicts are robust (affected ≈3% of
samples, beyond p90); `iter_wall_p90` exposes the tail honestly.

WEDGES: PROFILED small-chain perf runs wedge NONDETERMINISTICALLY
(megakernel spins, no forward progress; reaped by the case timeout —
wedged runs produce NO numbers, never wrong ones). Tally at framework
calibration: mlp_v2v3 5 pass/1 wedge; qkv_v2v3 1 pass/1 wedge; sq_v2v3
0/1 (serial); linvar=v2 arms 0/2 (matches the documented linear_sm100_v2
"gate_up MAINLOOP deadlock, mma_mbar stale" comment and the
demo/qwen3/demo.py:701 "GateUp v2 hangs in integration" note). Unprofiled
chains 6/6 pass; correctness (test-mode) 14/14 pass; the FULL qwen3 v2
PROFILED e2e (512 forced iterations, ~140 tasks/SM/iter) passed — the
wedge-prone regime is "profiled + sparse chain (1–8 tasks/SM/iter)".
Wedges occurred both concurrent AND serial: attribution is NOT
environmental.

Leading hypothesis (low confidence, recorded not claimed): the profiled
build's per-task role instrumentation (role START/END fences, timed-wait
%globaltimer reads inside the mbar/TMA/MMA handshake paths) perturbs
role/channel timing enough in SPARSE chains to expose a latent v2
mbar/channel liveness race of the documented MAINLOOP stale-phase class;
dense full-model streams never open the window. Discriminator ladder (not
yet run): (1) unprofiled sibling [done: immune]; (2) MPK_ENABLE_PROFILING
compiled but window/writes forced off; (3) role START/END fences only, no
buffer writes; (4) full profiling — plus the decisive V2_MAINLOOP_WAIT
watchdog dumping the stuck wait-object {sm, slot, mbar phase-vs-expected}.

REPORTING RULE: perf numbers are conditional on non-wedged runs; report
the pass-rate alongside them (a wedge is a liveness outcome, not a
discarded sample). Run verdict-grade perf serially on a quiet box and
check the census.

## Scope of the fidelity claim

What this framework establishes: it faithfully and REPRODUCIBLY MEASURES
per-task v2 body_span/op_wall at the production grid (136 workers) in a
cold-L2 serialized chain — internally consistent (decode gates, wall
cross-check) and calibrated against the v1 e2e baseline. What it does NOT
establish: that a per-task improvement measured here transfers 1:1 to
production e2e (external validity needs a lever-transfer experiment:
change something, predict the e2e delta from the harness, verify).
Per-task-win ≠ e2e-win is a documented failure mode of this project.

## Known gaps / caveats

- v1 profiled runs trace exactly ONE iteration (v1 profiler semantics force
  request_done after the first step when profiling): v1 per-task numbers
  come from that single live M=1 iteration × L instances — much lower
  sample depth than the v2 window (23 iters), so v1 comparisons are a
  sanity band, not a matched-precision baseline.
- Perf verdict numbers are per-task spans at the production grid inside a
  chain of REAL ops; they are NOT isolated-kernel microbenchmarks (that
  regime historically mis-ranks — this is intentional).
- 32-bit timestamp wrap (~4.3 s) inside a window is detected and flagged;
  phase attribution is skipped in that case (durations stay valid).
- attention_sm100 / embedding_v2 / argmax_v2 / rotary / mul_sum_add are NOT
  yet in the per-task matrix (exercised only via the qwen3 e2e).
- The v2 PRECOMPILED-load path (load_mpk_kernel) remains broken beyond the
  v2 function-binding fix: reload of a saved .so IMAs on any v2 task
  (rmsnorm too, so not TMA-specific) — the saved/reloaded task-graph JSON
  path does not reproduce the state the compile path bakes. JIT compile is
  the supported path.
- MAX_DYNAMIC_SHARED_MEMORY_SIZE (runtime_header.h) and the planner
  CAPACITY_BYTES (v2_smem_planner.py) are still two hardcoded constants
  with only a python-side canary (`_check_v2_smem_capacity`) tying them;
  a build-time cross-assert would be stronger.
