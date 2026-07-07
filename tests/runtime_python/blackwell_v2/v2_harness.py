"""Shared harness for the Runtime-V2 per-task correctness + performance
framework.

Two case families, both executed in a SUBPROCESS per case (see case_runner.py
/ run_suite.py) so a hang or crash in one case cannot take down the suite:

CORRECTNESS ("mode": "correctness")
    A test_mode (single-pass) PersistentKernel graph containing one or more
    INDEPENDENT ops, each with its own deterministically-seeded inputs and its
    own output tensor. The same op list is buildable against runtime "v1" and
    "v2"; outputs are saved to .pt so run_suite can do the v1-vs-v2
    equivalence comparison, and each output is compared here against the
    float32 PyTorch reference (pytorch_reference.py).

    Runtime row count: test-mode's first prepare_next_batch schedules a
    prefill chunk of min(prompt_lengths[0], mbt) rows. The harness always
    sets prompt_lengths = mbt = M so runtime-M kernels (silu_mul, rmsnorm)
    and static-M kernels (linear v2/v3, m_real = output dim0) agree on M.

PERF ("mode": "perf")
    A NON-test offline-mode graph: a serialized CHAIN of L identical
    "blocks" (dims cycle, output of block i feeds block i+1, every block has
    its OWN weight copies so per-iteration weight traffic is cold-L2 exactly
    like production layers). Driven for S = max_seq_length iterations with
    prompt_len = 1: iterations 0..S-2 are live M=1 decode steps, iteration
    S-1 is the single zero-M post-done iteration (dropped by the decoder).
    With profiler_tensor attached, MPK_ENABLE_PROFILING forces all S
    iterations and records the LAST V2_PROF_WINDOW_ITERS=25 in the 8-track
    role profiler; v2_prof_decode.py turns that into per-(task, iteration)
    consumer spans.

    M>1 perf is supported for the LINEAR family only (m_real is a
    compile-time template arg = mbt), because prepare_next_batch zeroes
    qo_indptr after generation is done, so runtime-M kernels do no work in
    post-done iterations and cannot be measured at M>1 in a decode loop.

Verdict metrics are defined in README.md and computed in v2_prof_decode.py.
"""

import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_reference import (
    compare_metrics,
    ref_linear,
    ref_linear_residual,
    ref_rmsnorm,
    ref_silu_mul,
)

DTYPE = torch.bfloat16


# --------------------------------------------------------------------------
# Deterministic tensor generation: same name -> same tensor, independent of
# call order, so the v1 and v2 subprocesses see identical inputs.
# --------------------------------------------------------------------------
def _seed_from(name: str) -> int:
    import hashlib

    return int(hashlib.sha1(name.encode()).hexdigest()[:8], 16)


def gen_tensor(name: str, shape, kind: str) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(_seed_from(name))
    if kind == "act":  # activations ~ O(1)
        t = torch.randn(shape, generator=g, device="cuda", dtype=torch.float32)
    elif kind == "w_linear":  # linear weight scaled so out ~ O(1)
        k = shape[-1]
        t = torch.randn(shape, generator=g, device="cuda", dtype=torch.float32)
        t = t * (1.0 / math.sqrt(k))
    elif kind == "w_norm":  # rmsnorm gamma ~ 1
        t = 1.0 + 0.1 * torch.randn(
            shape, generator=g, device="cuda", dtype=torch.float32
        )
    elif kind == "zeros":
        t = torch.zeros(shape, device="cuda", dtype=torch.float32)
    else:
        raise ValueError(kind)
    return t.to(DTYPE).contiguous()


# --------------------------------------------------------------------------
# v1 grid rules (mirror demo/qwen3/demo.py)
# --------------------------------------------------------------------------
def v1_linear_grid(n: int, use_cutlass: bool = True) -> int:
    if n % 64 == 0 and not use_cutlass:
        return n // 64
    if n / 96 > 400:
        assert n % 256 == 0, f"unsupported v1 linear N={n}"
        return n // 256
    if n % 96 == 0:
        return 96
    if n % 64 == 0:
        return 64
    raise AssertionError(f"unsupported v1 linear N={n}")


# --------------------------------------------------------------------------
# Op registry. Each op spec:
#   {"op": "rmsnorm"|"silu_mul"|"linear"|"linear_v3"|"linear_residual"|
#          "linear_residual_v3",
#    "name": unique id, "M": rows, and op-specific dims}
# build_op() registers it on the graph for the requested runtime and returns
# {"inputs": {...}, "output": tensor, "ref_fn": callable}.
# NOTE for runtime="v1" the *_v3 ops fall back to the same v1 counterpart as
# their v2 siblings (there is only one v1 linear kernel family).
# --------------------------------------------------------------------------
def build_op(pk, runtime: str, spec: dict, torch_tensors: dict):
    op = spec["op"]
    name = spec["name"]
    M = spec["M"]

    def attach(t, suffix):
        return pk.attach_input(torch_tensor=t, name=f"{name}_{suffix}")

    if op == "rmsnorm":
        H = spec["H"]
        x = torch_tensors.setdefault(f"{name}_x", gen_tensor(f"{name}_x", (M, H), "act"))
        w = torch_tensors.setdefault(f"{name}_w", gen_tensor(f"{name}_w", (H,), "w_norm"))
        out = torch_tensors.setdefault(f"{name}_out", gen_tensor(f"{name}_out", (M, H), "zeros"))
        pk.rmsnorm_layer(
            input=attach(x, "x"),
            weight=attach(w, "w"),
            output=attach(out, "out"),
            grid_dim=(M, 1, 1),
            block_dim=(128, 1, 1),
        )
        return {"output": out, "ref": lambda: ref_rmsnorm(x, w)}

    if op == "silu_mul":
        I = spec["I"]
        G = spec["G"]
        assert I % G == 0
        x = torch_tensors.setdefault(f"{name}_x", gen_tensor(f"{name}_x", (M, 2 * I), "act"))
        out = torch_tensors.setdefault(f"{name}_out", gen_tensor(f"{name}_out", (M, I), "zeros"))
        pk.silu_mul_layer(
            input=attach(x, "x"),
            output=attach(out, "out"),
            grid_dim=(G, 1, 1),
            block_dim=(128, 1, 1),
        )
        return {"output": out, "ref": lambda: ref_silu_mul(x, G)}

    if op in ("linear", "linear_v3"):
        N, K = spec["N"], spec["K"]
        x = torch_tensors.setdefault(f"{name}_x", gen_tensor(f"{name}_x", (M, K), "act"))
        w = torch_tensors.setdefault(f"{name}_w", gen_tensor(f"{name}_w", (N, K), "w_linear"))
        out = torch_tensors.setdefault(f"{name}_out", gen_tensor(f"{name}_out", (M, N), "zeros"))
        x_dt, w_dt, out_dt = attach(x, "x"), attach(w, "w"), attach(out, "out")
        if runtime == "v1":
            pk.linear_layer(
                input=x_dt,
                weight=w_dt,
                output=out_dt,
                grid_dim=(v1_linear_grid(N), 1, 1),
                block_dim=(128, 1, 1),
            )
        elif op == "linear":
            pk.linear_layer_v2(input=x_dt, weight=w_dt, output=out_dt, tiles_per_task=1)
        else:
            pk.linear_layer_v3(input=x_dt, weight=w_dt, output=out_dt, tiles_per_task=1)
        return {"output": out, "ref": lambda: ref_linear(x, w)}

    if op == "dsv3_lmhead_gemv":
        # v2 tail lm_head GEMV (M3 fix). Same math as `linear` (out = x @ w.T,
        # weight [N,K] row-major) but the dedicated non-TMA scalar/cp.async
        # bf16 GEMV. M is fixed at 1 (bs=1) — the layer computes all M rows the
        # output has, and at M=1 the reference matches exactly.
        N, K = spec["N"], spec["K"]
        block_n = spec.get("block_n", 128)
        x = torch_tensors.setdefault(f"{name}_x", gen_tensor(f"{name}_x", (M, K), "act"))
        w = torch_tensors.setdefault(f"{name}_w", gen_tensor(f"{name}_w", (N, K), "w_linear"))
        out = torch_tensors.setdefault(f"{name}_out", gen_tensor(f"{name}_out", (M, N), "zeros"))
        x_dt, w_dt, out_dt = attach(x, "x"), attach(w, "w"), attach(out, "out")
        assert runtime == "v2", "dsv3_lmhead_gemv is a v2-only op"
        pk.dsv3_lmhead_gemv_layer(
            input=x_dt, weight=w_dt, output=out_dt, block_n=block_n
        )
        return {"output": out, "ref": lambda: ref_linear(x, w)}

    if op in ("linear_residual", "linear_residual_v3"):
        N, K = spec["N"], spec["K"]
        x = torch_tensors.setdefault(f"{name}_x", gen_tensor(f"{name}_x", (M, K), "act"))
        w = torch_tensors.setdefault(f"{name}_w", gen_tensor(f"{name}_w", (N, K), "w_linear"))
        r = torch_tensors.setdefault(f"{name}_r", gen_tensor(f"{name}_r", (M, N), "act"))
        out = torch_tensors.setdefault(f"{name}_out", gen_tensor(f"{name}_out", (M, N), "zeros"))
        x_dt, w_dt = attach(x, "x"), attach(w, "w")
        r_dt, out_dt = attach(r, "r"), attach(out, "out")
        if runtime == "v1":
            pk.linear_with_residual_layer(
                input=x_dt,
                weight=w_dt,
                residual=r_dt,
                output=out_dt,
                grid_dim=(N // 64, 1, 1),
                block_dim=(128, 1, 1),
            )
        elif op == "linear_residual":
            pk.linear_with_residual_layer_v2(
                input=x_dt, weight=w_dt, residual=r_dt, output=out_dt
            )
        else:
            pk.linear_with_residual_layer_v3(
                input=x_dt, weight=w_dt, residual=r_dt, output=out_dt, tiles_per_task=1
            )
        return {"output": out, "ref": lambda: ref_linear_residual(x, w, r)}

    raise ValueError(f"unknown op {op}")


# --------------------------------------------------------------------------
# Standard correctness op matrix for a given M (qwen3-8B real shapes + one
# small generic shape per family).
# --------------------------------------------------------------------------
def correctness_ops(M: int, runtime: str):
    ops = [
        {"op": "rmsnorm", "name": "rms_h4096", "M": M, "H": 4096},
        {"op": "rmsnorm", "name": "rms_h1024", "M": M, "H": 1024},
        {"op": "silu_mul", "name": "silu_i12288_g48", "M": M, "I": 12288, "G": 48},
        {"op": "silu_mul", "name": "silu_i1024_g1", "M": M, "I": 1024, "G": 1},
    ]
    # v2 linear family contract: one 16-row activation tile per task
    # (BLOCK_N=16 in linear_spec.h; consumer template M_REAL<=16). M>16
    # silently computed only the first 16 rows before the layer-method
    # asserts were added — keep linears out of the M>16 matrix.
    if M > 16:
        return ops
    ops += [
        {"op": "linear", "name": "lin2_qkv", "M": M, "N": 6144, "K": 4096},
        {"op": "linear", "name": "lin2_gateup", "M": M, "N": 24576, "K": 4096},
        {"op": "linear", "name": "lin2_sq", "M": M, "N": 4096, "K": 4096},
        {"op": "linear_residual", "name": "linres2_o", "M": M, "N": 4096, "K": 4096},
        {"op": "linear_residual", "name": "linres2_down", "M": M, "N": 4096, "K": 12288},
    ]
    if runtime != "v1":
        # v3 variants only exist on the v2 runtime; on v1 they'd duplicate
        # the exact same v1 counterpart op (same inputs -> same task) and
        # the graph would contain two identical writers of one buffer.
        ops += [
            {"op": "linear_v3", "name": "lin3_qkv", "M": M, "N": 6144, "K": 4096},
            {"op": "linear_v3", "name": "lin3_gateup", "M": M, "N": 24576, "K": 4096},
            {"op": "linear_v3", "name": "lin3_sq", "M": M, "N": 4096, "K": 4096},
            {"op": "linear_residual_v3", "name": "linres3_o", "M": M, "N": 4096, "K": 4096},
            {"op": "linear_residual_v3", "name": "linres3_down", "M": M, "N": 4096, "K": 12288},
        ]
    else:
        # v1 arm still needs outputs under the v3 case names so run_suite can
        # compare v2's v3 ops against the v1 counterpart: reuse identical
        # inputs (same seeds as the v3 cases) through the v1 kernel.
        ops += [
            {"op": "linear", "name": "lin3_qkv", "M": M, "N": 6144, "K": 4096},
            {"op": "linear", "name": "lin3_gateup", "M": M, "N": 24576, "K": 4096},
            {"op": "linear", "name": "lin3_sq", "M": M, "N": 4096, "K": 4096},
            {"op": "linear_residual", "name": "linres3_o", "M": M, "N": 4096, "K": 4096},
            {"op": "linear_residual", "name": "linres3_down", "M": M, "N": 4096, "K": 12288},
        ]
    return ops


# --------------------------------------------------------------------------
# PersistentKernel construction
# --------------------------------------------------------------------------
def _check_v2_smem_capacity():
    """Canary for the planner-vs-runtime smem geometry drift that produced
    the Invalid __shared__ write at linear_scratch (dyn byte 221184): the
    Python planner's CAPACITY_BYTES must not exceed the v2 runtime's
    MAX_DYNAMIC_SHARED_MEMORY_SIZE (USE_RUNTIME_V2 branch, cc>=90 offline).
    Parses both sources textually — cheap, and fails loudly on future edits."""
    import re

    import mirage.mpk.v2_smem_planner as planner

    hdr = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "include", "mirage", "persistent_kernel",
        "runtime_header.h",
    )
    with open(hdr) as f:
        src = f.read()
    m = re.search(
        r"#ifdef USE_RUNTIME_V2.*?(\d+)\s*\*\s*1024\s*-\s*"
        r"WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE",
        src,
        re.S,
    )
    assert m, "USE_RUNTIME_V2 smem branch not found in runtime_header.h"
    v2_dyn = int(m.group(1)) * 1024 - 6 * 1024  # cc>=90 => reserved 6KB
    cap = planner.CAPACITY_BYTES
    assert cap <= v2_dyn, (
        f"v2 smem planner CAPACITY_BYTES={cap} exceeds the v2 runtime dynamic "
        f"segment {v2_dyn} — region offsets would be out of bounds "
        f"(see runtime_header.h USE_RUNTIME_V2 branch)"
    )


def make_pk(
    runtime: str,
    M: int,
    test_mode: bool,
    max_seq_length: int = None,
    profiler_tensor=None,
    trace_name: str = "v2fw",
):
    import mirage as mi
    from mirage.mpk.persistent_kernel import PersistentKernel

    if runtime == "v2":
        _check_v2_smem_capacity()

    num_workers, num_schedulers = mi.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["use_cutlass_kernel"] = True  # production v1 config on B200
    params["use_v2_runtime"] = runtime == "v2"
    params["max_num_batched_tokens"] = M
    params["max_num_batched_requests"] = 1
    params["profiler_tensor"] = profiler_tensor
    params["trace_name"] = trace_name

    if test_mode:
        params["test_mode"] = True
        params["max_seq_length"] = max(M, 1)
        params["max_num_pages"] = max(1, (M + 4095) // 4096)
        params["page_size"] = 4096
        params["meta_tensors"] = {
            "prompt_lengths": torch.tensor([M], dtype=torch.int32, device="cuda"),
        }
        return PersistentKernel(**params)

    # non-test offline mode: full meta tensor set (mirrors demo/qwen3/demo.py)
    S = max_seq_length
    assert S is not None
    params["max_seq_length"] = S
    params["page_size"] = 4096
    params["max_num_pages"] = max(2, (S + 4095) // 4096 + 1)
    tokens = torch.zeros((1, S), dtype=torch.long, device="cuda")
    params["meta_tensors"] = {
        "step": torch.zeros((1,), dtype=torch.int32, device="cuda"),
        "tokens": tokens,
        "input_tokens": torch.zeros((M, 1), dtype=torch.long, device="cuda"),
        "output_tokens": torch.zeros((M, 1), dtype=torch.long, device="cuda"),
        "num_new_tokens": torch.ones((1,), dtype=torch.int32, device="cuda"),
        "prompt_lengths": torch.tensor([1], dtype=torch.int32, device="cuda"),
        "qo_indptr_buffer": torch.empty((2,), dtype=torch.int32, device="cuda"),
        "paged_kv_indptr_buffer": torch.empty((2,), dtype=torch.int32, device="cuda"),
        "paged_kv_indices_buffer": torch.empty(
            (params["max_num_pages"],), dtype=torch.int32, device="cuda"
        ),
        "paged_kv_last_page_len_buffer": torch.empty(
            (1,), dtype=torch.int32, device="cuda"
        ),
    }
    params["eos_token_id"] = -1
    return PersistentKernel(**params)


# --------------------------------------------------------------------------
# Correctness case
# --------------------------------------------------------------------------
def run_correctness_case(spec: dict, out_dir: str) -> dict:
    runtime = spec["runtime"]
    M = spec["M"]
    only = spec.get("only")  # optional list of op names (fallback isolation)

    ops = correctness_ops(M, runtime)
    if only:
        ops = [o for o in ops if o["name"] in only]

    pk = make_pk(runtime, M, test_mode=True)
    torch_tensors = {}
    built = []
    for op_spec in ops:
        handle = build_op(pk, runtime, op_spec, torch_tensors)
        built.append((op_spec, handle))

    t0 = time.time()
    pk.compile(output_dir=os.path.join(out_dir, "compile"))
    compile_s = time.time() - t0

    t0 = time.time()
    pk()
    torch.cuda.synchronize()
    run_s = time.time() - t0

    results = {}
    saved = {}
    for op_spec, handle in built:
        name = op_spec["name"]
        out = handle["output"]
        ref = handle["ref"]()
        m = compare_metrics(out, ref)
        m["pass_vs_torch"] = bool(
            m["cos"] >= 0.999 and m["rel_max"] <= 3e-2 and not m["nan_in_out"]
        )
        results[name] = m
        saved[name] = out.cpu()

    torch.save(saved, os.path.join(out_dir, f"outputs_{runtime}_M{M}.pt"))
    try:
        pk.finalize()
    except Exception as e:  # noqa: BLE001 — finalize glitches shouldn't fail the case
        results["_finalize_warning"] = str(e)
    return {
        "kind": "correctness",
        "runtime": runtime,
        "M": M,
        "compile_s": compile_s,
        "run_s": run_s,
        "ops": results,
        "outputs_file": os.path.join(out_dir, f"outputs_{runtime}_M{M}.pt"),
    }


# --------------------------------------------------------------------------
# Perf chains. A chain is L copies of a block whose dims cycle; every block
# instance gets its own weights (cold-L2, production-like). Blocks:
#   mlp : rmsnorm(4096) -> linear gateup [24576,4096] -> silu_mul(G=48)
#         -> linear_residual down [4096,12288]  (real qwen3-8B decode block)
#   qkv : rmsnorm(4096) -> linear qkv [6144,4096]
#         -> linear_residual o-like [4096,6144]
#   sq  : linear [4096,4096] (pure GEMV chain)
# "linvar": "v2"|"v3" picks linear_layer_v2 / _v3 on the v2 runtime.
# --------------------------------------------------------------------------
def build_perf_chain(pk, runtime: str, chain: str, L: int, M: int, linvar: str,
                     torch_tensors: dict):
    lin_op = "linear" if (runtime == "v1" or linvar == "v2") else "linear_v3"
    linres_op = (
        "linear_residual" if (runtime == "v1" or linvar == "v2") else "linear_residual_v3"
    )
    instances = []  # (instance_idx, op_kind, opname)

    x = torch_tensors.setdefault("chain_x0", gen_tensor("chain_x0", (M, 4096), "act"))
    x_dt = pk.attach_input(torch_tensor=x, name="chain_x0")

    for i in range(L):
        if chain == "mlp":
            t1 = gen_tensor(f"c{i}_t1", (M, 4096), "zeros")
            t2 = gen_tensor(f"c{i}_t2", (M, 24576), "zeros")
            t3 = gen_tensor(f"c{i}_t3", (M, 12288), "zeros")
            xo = gen_tensor(f"c{i}_xo", (M, 4096), "zeros")
            wn = gen_tensor(f"c{i}_wn", (4096,), "w_norm")
            wg = gen_tensor(f"c{i}_wg", (24576, 4096), "w_linear")
            wd = gen_tensor(f"c{i}_wd", (4096, 12288), "w_linear")
            for nm, t in [(f"c{i}_t1", t1), (f"c{i}_t2", t2), (f"c{i}_t3", t3),
                          (f"c{i}_xo", xo), (f"c{i}_wn", wn), (f"c{i}_wg", wg),
                          (f"c{i}_wd", wd)]:
                torch_tensors[nm] = t
            t1_dt = pk.attach_input(torch_tensor=t1, name=f"c{i}_t1")
            t2_dt = pk.attach_input(torch_tensor=t2, name=f"c{i}_t2")
            t3_dt = pk.attach_input(torch_tensor=t3, name=f"c{i}_t3")
            xo_dt = pk.attach_input(torch_tensor=xo, name=f"c{i}_xo")
            wn_dt = pk.attach_input(torch_tensor=wn, name=f"c{i}_wn")
            wg_dt = pk.attach_input(torch_tensor=wg, name=f"c{i}_wg")
            wd_dt = pk.attach_input(torch_tensor=wd, name=f"c{i}_wd")

            pk.rmsnorm_layer(input=x_dt, weight=wn_dt, output=t1_dt,
                             grid_dim=(M, 1, 1), block_dim=(128, 1, 1))
            instances.append((i, "rmsnorm_4096", 1))
            if runtime == "v1":
                pk.linear_layer(input=t1_dt, weight=wg_dt, output=t2_dt,
                                grid_dim=(v1_linear_grid(24576), 1, 1),
                                block_dim=(128, 1, 1))
                n_gu = v1_linear_grid(24576)
            elif linvar == "v2":
                pk.linear_layer_v2(input=t1_dt, weight=wg_dt, output=t2_dt)
                n_gu = 24576 // 128
            else:
                pk.linear_layer_v3(input=t1_dt, weight=wg_dt, output=t2_dt)
                n_gu = 24576 // 128
            instances.append((i, "linear_gateup_24576x4096", n_gu))
            pk.silu_mul_layer(input=t2_dt, output=t3_dt,
                              grid_dim=(48, 1, 1), block_dim=(128, 1, 1))
            instances.append((i, "silu_mul_12288_g48", 48))
            if runtime == "v1":
                pk.linear_with_residual_layer(
                    input=t3_dt, weight=wd_dt, residual=x_dt, output=xo_dt,
                    grid_dim=(4096 // 64, 1, 1), block_dim=(128, 1, 1))
                n_dn = 4096 // 64
            elif linvar == "v2":
                pk.linear_with_residual_layer_v2(
                    input=t3_dt, weight=wd_dt, residual=x_dt, output=xo_dt)
                n_dn = 4096 // 128
            else:
                pk.linear_with_residual_layer_v3(
                    input=t3_dt, weight=wd_dt, residual=x_dt, output=xo_dt)
                n_dn = 4096 // 128
            instances.append((i, "linear_res_down_4096x12288", n_dn))
            x, x_dt = xo, xo_dt

        elif chain == "qkv":
            t1 = gen_tensor(f"c{i}_t1", (M, 4096), "zeros")
            t2 = gen_tensor(f"c{i}_t2", (M, 6144), "zeros")
            xo = gen_tensor(f"c{i}_xo", (M, 4096), "zeros")
            wn = gen_tensor(f"c{i}_wn", (4096,), "w_norm")
            wq = gen_tensor(f"c{i}_wq", (6144, 4096), "w_linear")
            wo = gen_tensor(f"c{i}_wo", (4096, 6144), "w_linear")
            for nm, t in [(f"c{i}_t1", t1), (f"c{i}_t2", t2), (f"c{i}_xo", xo),
                          (f"c{i}_wn", wn), (f"c{i}_wq", wq), (f"c{i}_wo", wo)]:
                torch_tensors[nm] = t
            t1_dt = pk.attach_input(torch_tensor=t1, name=f"c{i}_t1")
            t2_dt = pk.attach_input(torch_tensor=t2, name=f"c{i}_t2")
            xo_dt = pk.attach_input(torch_tensor=xo, name=f"c{i}_xo")
            wn_dt = pk.attach_input(torch_tensor=wn, name=f"c{i}_wn")
            wq_dt = pk.attach_input(torch_tensor=wq, name=f"c{i}_wq")
            wo_dt = pk.attach_input(torch_tensor=wo, name=f"c{i}_wo")

            pk.rmsnorm_layer(input=x_dt, weight=wn_dt, output=t1_dt,
                             grid_dim=(M, 1, 1), block_dim=(128, 1, 1))
            instances.append((i, "rmsnorm_4096", 1))
            if runtime == "v1":
                pk.linear_layer(input=t1_dt, weight=wq_dt, output=t2_dt,
                                grid_dim=(v1_linear_grid(6144), 1, 1),
                                block_dim=(128, 1, 1))
                n_q = v1_linear_grid(6144)
            elif linvar == "v2":
                pk.linear_layer_v2(input=t1_dt, weight=wq_dt, output=t2_dt)
                n_q = 6144 // 128
            else:
                pk.linear_layer_v3(input=t1_dt, weight=wq_dt, output=t2_dt)
                n_q = 6144 // 128
            instances.append((i, "linear_qkv_6144x4096", n_q))
            if runtime == "v1":
                pk.linear_with_residual_layer(
                    input=t2_dt, weight=wo_dt, residual=x_dt, output=xo_dt,
                    grid_dim=(4096 // 64, 1, 1), block_dim=(128, 1, 1))
                n_o = 4096 // 64
            elif linvar == "v2":
                pk.linear_with_residual_layer_v2(
                    input=t2_dt, weight=wo_dt, residual=x_dt, output=xo_dt)
                n_o = 4096 // 128
            else:
                pk.linear_with_residual_layer_v3(
                    input=t2_dt, weight=wo_dt, residual=x_dt, output=xo_dt)
                n_o = 4096 // 128
            instances.append((i, "linear_res_o_4096x6144", n_o))
            x, x_dt = xo, xo_dt

        elif chain == "sq":
            xo = gen_tensor(f"c{i}_xo", (M, 4096), "zeros")
            ws = gen_tensor(f"c{i}_ws", (4096, 4096), "w_linear")
            torch_tensors[f"c{i}_xo"] = xo
            torch_tensors[f"c{i}_ws"] = ws
            xo_dt = pk.attach_input(torch_tensor=xo, name=f"c{i}_xo")
            ws_dt = pk.attach_input(torch_tensor=ws, name=f"c{i}_ws")
            if runtime == "v1":
                pk.linear_layer(input=x_dt, weight=ws_dt, output=xo_dt,
                                grid_dim=(v1_linear_grid(4096), 1, 1),
                                block_dim=(128, 1, 1))
                n_s = v1_linear_grid(4096)
            elif linvar == "v2":
                pk.linear_layer_v2(input=x_dt, weight=ws_dt, output=xo_dt)
                n_s = 4096 // 128
            else:
                pk.linear_layer_v3(input=x_dt, weight=ws_dt, output=xo_dt)
                n_s = 4096 // 128
            instances.append((i, "linear_sq_4096x4096", n_s))
            x, x_dt = xo, xo_dt
        else:
            raise ValueError(chain)

    return instances


def run_perf_case(spec: dict, out_dir: str) -> dict:
    """One perf measurement: profiled run -> per-task window table; plus an
    UNPROFILED wall-clock run of the same graph for the cross-check."""
    runtime = spec["runtime"]
    chain = spec["chain"]
    L = spec.get("L", 4)
    M = spec.get("M", 1)
    S = spec.get("iters", 32)  # = max_seq_length = v2 iteration count
    linvar = spec.get("linvar", "v3")
    profiled = spec.get("profiled", True)
    if chain == "sq" and L < 8:
        # cold-L2 gate: the sq chain streams only ~32MB/instance; at L=4 the
        # per-weight reuse distance (~96MB) fits B200's 126MB L2 and the
        # measurement would be warm-contaminated. L=8 pushes it to ~256MB.
        L = 8

    result = {
        "kind": "perf",
        "runtime": runtime,
        "chain": chain,
        "L": L,
        "M": M,
        "iters": S,
        "linvar": linvar,
    }

    def _v1_csv_summary(csv_path):
        """v1 fallback numbers: per-task-type p50/p90 of duration_ns from the
        standard exporter CSV (v1 profiled runs trace exactly ONE iteration,
        so samples = tasks x L instances x 1 iter)."""
        import csv as _csv

        by_type = {}
        with open(csv_path) as f:
            for row in _csv.DictReader(f):
                by_type.setdefault(row["task_type_name"], []).append(
                    int(row["duration_ns"])
                )
        out = {}
        for t, vals in by_type.items():
            vals.sort()

            def pc(p):
                return vals[min(len(vals) - 1, int(round(p / 100 * (len(vals) - 1))))]

            out[t] = {
                "n_samples": len(vals),
                "task_span_us": {"p50": pc(50) / 1e3, "p90": pc(90) / 1e3},
            }
        return out

    prof = None
    if profiled:
        prof = torch.zeros(120000 * 128, dtype=torch.uint64, device="cuda").contiguous()

    pk = make_pk(
        runtime,
        M,
        test_mode=False,
        max_seq_length=S,
        profiler_tensor=prof,
        trace_name=os.path.join(out_dir, f"trace_{runtime}_{chain}_M{M}_{linvar}"),
    )
    torch_tensors = {}
    instances = build_perf_chain(pk, runtime, chain, L, M, linvar, torch_tensors)
    result["instances"] = [
        {"instance": i, "op": op, "ntasks": n} for (i, op, n) in instances
    ]

    t0 = time.time()
    pk.compile(output_dir=os.path.join(out_dir, "compile"))
    result["compile_s"] = time.time() - t0

    torch.cuda.synchronize()
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    pk()
    ev1.record()
    torch.cuda.synchronize()
    result["wall_ms"] = ev0.elapsed_time(ev1)
    if profiled:
        # pk.__call__ runs the CPU-side trace export before returning AND the
        # profiled build forces all S iterations — this wall is diagnostic
        # only. The *_nowall sibling case is the wall-clock source of truth.
        result["wall_ms_note"] = "profiled: includes trace export + forced iters"
    result["final_step"] = int(pk.meta_tensors["step"][0].item())
    result["final_qo_last"] = int(pk.meta_tensors["qo_indptr_buffer"][-1].item())

    if profiled:
        # persist raw buffer + the queue/type mapping for offline decoding
        import numpy as np

        raw_path = os.path.join(out_dir, f"prof_{runtime}_{chain}_M{M}_{linvar}.npy")
        np.save(raw_path, prof.cpu().numpy())
        result["prof_raw"] = raw_path
        v2map = getattr(pk, "_v2_task_graph_for_prof", None)
        if v2map is not None:
            with open(os.path.join(out_dir, "v2_map.json"), "w") as f:
                json.dump(v2map, f)
            result["v2_map"] = os.path.join(out_dir, "v2_map.json")

        # decode in-process (v2 only; v1 numbers come from the standard CSV
        # that pk() already exported via profiler_persistent.export_to_csv)
        if runtime == "v2" and v2map is not None:
            from v2_prof_decode import decode_window_table, summarize

            table = decode_window_table(prof, v2map["queues"], v2map["task_types"])
            summary = summarize(
                table,
                result["instances"],
                os.path.join(out_dir, "compile", "task_graph_rank0.json"),
                total_iters=S,
            )
            table_path = os.path.join(
                out_dir, f"table_{runtime}_{chain}_M{M}_{linvar}.json")
            with open(table_path, "w") as f:
                json.dump(
                    {"validation": table["validation"],
                     "n_window_iters": table["n_window_iters"],
                     "summary": summary},
                    f, indent=1,
                )
            result["window_table"] = table_path
            result["decode_summary"] = summary
            result["decode_validation"] = {
                "n_errors": len(table["validation"]["errors"]),
                "n_warnings": len(table["validation"]["warnings"]),
                "first_errors": table["validation"]["errors"][:5],
                "first_warnings": table["validation"]["warnings"][:5],
            }
            # hard gate: decode integrity failure (or a summarize mapping
            # error) means the numbers are NOT trustworthy — fail the case.
            if not table["validation"].get("decode_ok") or "error" in summary:
                result["status"] = "decode_failed"
        elif runtime == "v1":
            csv_path = pk.trace_name + ".csv" if hasattr(pk, "trace_name") else None
            csv_path = os.path.join(
                out_dir, f"trace_{runtime}_{chain}_M{M}_{linvar}.csv")
            if os.path.exists(csv_path):
                result["v1_csv_summary"] = _v1_csv_summary(csv_path)

    try:
        pk.finalize()
    except Exception as e:  # noqa: BLE001
        result["_finalize_warning"] = str(e)
    return result
