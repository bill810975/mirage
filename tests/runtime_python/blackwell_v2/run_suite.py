"""Runtime-V2 test-framework runner.

Spawns one subprocess per case (hang-proof: hard timeout + kill), pins each
case to a GPU via CUDA_VISIBLE_DEVICES, collects JSON results, performs the
v1-vs-v2 output equivalence comparison, and prints the calibration tables.

Usage examples:
  # correctness matrix (v1 + v2, M in 1/8/128) on GPU 6
  python run_suite.py --what correctness --devices 6

  # perf matrix on GPU 6, 3 repeats for run-to-run variance
  python run_suite.py --what perf --devices 6 --repeats 3

  # everything, two GPUs in parallel
  python run_suite.py --what all --devices 5,6

Results land under _results/<tag>/<case>/ next to this file.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from queue import Queue

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def build_cases(what, Ms, repeats, only=None, iters=32, L=4,
                chains=("mlp", "qkv", "sq"),
                arms=(("v2", "v3"), ("v2", "v2"), ("v1", "-")),
                rep_arms=None):
    """rep_arms: arms that get `repeats` profiled runs (others get 1)."""
    cases = []
    if what in ("correctness", "all"):
        for M in Ms:
            for runtime in ("v2", "v1"):
                cases.append(
                    {
                        "mode": "correctness",
                        "runtime": runtime,
                        "M": M,
                        **({"only": only} if only else {}),
                        "case_id": f"corr_{runtime}_M{M}",
                        "timeout_s": 900,
                    }
                )
    if what in ("perf", "all"):
        for chain in chains:
            for runtime, linvar in arms:
                tag = f"{runtime}{'' if runtime == 'v1' else '_' + linvar}"
                n_reps = repeats if (
                    rep_arms is None or (runtime, linvar) in rep_arms
                ) else 1
                for rep in range(n_reps):
                    cases.append(
                        {
                            "mode": "perf",
                            "runtime": runtime,
                            "chain": chain,
                            "linvar": linvar if linvar != "-" else "v3",
                            "M": 1,
                            "L": L,
                            "iters": iters,
                            "profiled": True,
                            "case_id": f"perf_{chain}_{tag}_M1_r{rep}",
                            "timeout_s": 900,
                        }
                    )
                # unprofiled sibling: true wall-clock (no MPK_ENABLE_PROFILING
                # forced iterations, no instrumentation) for the cross-check
                cases.append(
                    {
                        "mode": "perf",
                        "runtime": runtime,
                        "chain": chain,
                        "linvar": linvar if linvar != "-" else "v3",
                        "M": 1,
                        "L": L,
                        "iters": iters,
                        "profiled": False,
                        "case_id": f"perf_{chain}_{tag}_M1_nowall",
                        "timeout_s": 900,
                    }
                )
    return cases


def run_case(case, device, tag_dir, reuse_compile_from=None):
    case_dir = os.path.join(tag_dir, case["case_id"])
    os.makedirs(case_dir, exist_ok=True)
    spec_path = os.path.join(case_dir, "spec.json")
    result_path = os.path.join(case_dir, "result.json")
    with open(spec_path, "w") as f:
        json.dump(case, f, indent=1)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    env.setdefault("PYTHONUNBUFFERED", "1")
    log_path = os.path.join(case_dir, "log.txt")
    # provenance: record co-tenancy on the target GPU (perf numbers taken on
    # a shared card are suspect; the report should be able to prove isolation)
    try:
        cotenants = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        with open(os.path.join(case_dir, "gpu_procs_at_start.txt"), "w") as f:
            f.write(cotenants)
    except Exception:  # noqa: BLE001
        pass
    t0 = time.time()
    with open(log_path, "w") as log:
        p = subprocess.Popen(
            [PY, os.path.join(HERE, "case_runner.py"), spec_path, result_path],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=case_dir,
        )
        try:
            rc = p.wait(timeout=case.get("timeout_s", 2400))
        except subprocess.TimeoutExpired:
            p.kill()
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                pass
            rc = -9
    wall = time.time() - t0
    if os.path.exists(result_path):
        with open(result_path) as f:
            result = json.load(f)
    else:
        result = {"spec": case, "status": "timeout" if rc == -9 else "crashed"}
    result["_rc"] = rc
    result["_wall_s"] = wall
    result["_dir"] = case_dir
    return result


def compare_v1_v2_outputs(tag_dir, Ms):
    """Load saved outputs_{runtime}_M{M}.pt pairs and compute equivalence."""
    import torch

    sys.path.insert(0, HERE)
    from pytorch_reference import compare_metrics

    rows = []
    for M in Ms:
        f1 = os.path.join(tag_dir, f"corr_v1_M{M}", f"outputs_v1_M{M}.pt")
        f2 = os.path.join(tag_dir, f"corr_v2_M{M}", f"outputs_v2_M{M}.pt")
        if not (os.path.exists(f1) and os.path.exists(f2)):
            continue
        o1 = torch.load(f1)
        o2 = torch.load(f2)
        for name in sorted(set(o1) & set(o2)):
            m = compare_metrics(o2[name], o1[name])
            rows.append({"M": M, "op": name, **m})
    return rows


def fmt_table(rows, cols):
    if not rows:
        return "(no rows)"
    widths = [max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols]
    out = ["  ".join(c.ljust(w) for c, w in zip(cols, widths))]
    for r in rows:
        out.append("  ".join(str(r.get(c, "")).ljust(w) for c, w in zip(cols, widths)))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["correctness", "perf", "all"], default="all")
    ap.add_argument("--devices", default="6")
    ap.add_argument("--Ms", default="1,8,128")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--iters", type=int, default=32)
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--only", default=None, help="comma list of op names")
    ap.add_argument("--chains", default="mlp,qkv,sq")
    ap.add_argument("--arms", default="v2_v3,v2_v2,v1",
                    help="comma list of runtime[_linvar] arms")
    ap.add_argument("--rep-arms", default=None,
                    help="arms getting `repeats` runs (default: all)")
    args = ap.parse_args()

    devices = [d.strip() for d in args.devices.split(",")]
    Ms = [int(m) for m in args.Ms.split(",")]
    tag = args.tag or time.strftime("%Y%m%d_%H%M%S")
    tag_dir = os.path.join(HERE, "_results", tag)
    os.makedirs(tag_dir, exist_ok=True)

    def parse_arm(s):
        parts = s.split("_")
        return (parts[0], parts[1] if len(parts) > 1 else "-")

    arms = tuple(parse_arm(a) for a in args.arms.split(","))
    rep_arms = (
        tuple(parse_arm(a) for a in args.rep_arms.split(","))
        if args.rep_arms
        else None
    )
    only = args.only.split(",") if args.only else None
    cases = build_cases(
        args.what, Ms, args.repeats, only, args.iters, args.L,
        chains=tuple(args.chains.split(",")), arms=arms, rep_arms=rep_arms,
    )
    print(f"[suite] {len(cases)} cases -> {tag_dir} on devices {devices}")

    q = Queue()
    for c in cases:
        q.put(c)
    results = []
    lock = threading.Lock()

    def worker(dev):
        while True:
            try:
                c = q.get_nowait()
            except Exception:  # noqa: BLE001
                return
            print(f"[dev{dev}] START {c['case_id']}")
            r = run_case(c, dev, tag_dir)
            with lock:
                results.append(r)
            print(
                f"[dev{dev}] DONE  {c['case_id']}: {r.get('status')} "
                f"({r.get('_wall_s', 0):.0f}s)"
            )

    threads = [threading.Thread(target=worker, args=(d,)) for d in devices]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    with open(os.path.join(tag_dir, "all_results.json"), "w") as f:
        json.dump(results, f, indent=1, default=str)

    # ---------------- report ----------------
    print("\n================ SUITE REPORT ================")
    corr_rows = []
    for r in results:
        if r.get("kind") != "correctness":
            continue
        for op, m in (r.get("ops") or {}).items():
            if not isinstance(m, dict):
                continue
            corr_rows.append(
                {
                    "case": f"{r['runtime']}_M{r['M']}",
                    "op": op,
                    "cos": f"{m['cos']:.6f}",
                    "rel_max": f"{m['rel_max']:.2e}",
                    "bitexact": f"{m['bitexact_frac']:.3f}",
                    "pass": m["pass_vs_torch"],
                }
            )
    if corr_rows:
        print("\n--- correctness vs torch(fp32) ---")
        print(fmt_table(corr_rows, ["case", "op", "cos", "rel_max", "bitexact", "pass"]))

    if args.what in ("correctness", "all"):
        eq_rows = compare_v1_v2_outputs(tag_dir, Ms)
        if eq_rows:
            print("\n--- v2 output vs v1 counterpart (same inputs) ---")
            print(
                fmt_table(
                    [
                        {
                            "M": e["M"],
                            "op": e["op"],
                            "cos": f"{e['cos']:.6f}",
                            "rel_max": f"{e['rel_max']:.2e}",
                            "bitexact": f"{e['bitexact_frac']:.3f}",
                        }
                        for e in eq_rows
                    ],
                    ["M", "op", "cos", "rel_max", "bitexact"],
                )
            )
            with open(os.path.join(tag_dir, "v1_v2_equivalence.json"), "w") as f:
                json.dump(eq_rows, f, indent=1)

    perf_rows = []
    wall_rows = []
    for r in results:
        if r.get("kind") != "perf":
            continue
        case_id = r["spec"]["case_id"] if "spec" in r else "?"
        if not (r.get("spec") or {}).get("profiled", True):
            wall_rows.append(
                {
                    "case": case_id,
                    "wall_ms": round(r.get("wall_ms", -1), 2),
                    "final_step": r.get("final_step"),
                    "ms_per_live_iter": (
                        round(r["wall_ms"] / max(r.get("final_step", 1), 1), 3)
                        if r.get("wall_ms") is not None
                        else "-"
                    ),
                }
            )
            continue
        summ = r.get("decode_summary") or {}
        if not summ and r.get("v1_csv_summary"):
            # v1 arm: standard-CSV fallback (span only, single traced iter)
            for t, s in r["v1_csv_summary"].items():
                perf_rows.append(
                    {
                        "case": case_id,
                        "op": t,
                        "n": s.get("n_samples"),
                        "body_p50_us": "-",
                        "body_p90_us": "-",
                        "span_p50_us": _f(s, "task_span_us", "p50"),
                        "wall_p50_us": "-",
                        "dep_p50_us": "-",
                        "ahead_frac": "-",
                        "ok": "v1csv",
                    }
                )
            continue
        ok = summ.get("_decode_ok")
        for op, s in summ.items():
            if op.startswith("_") or not isinstance(s, dict):
                continue
            perf_rows.append(
                {
                    "case": case_id,
                    "op": op,
                    "n": s.get("n_samples"),
                    "body_p50_us": _f(s, "body_span_us", "p50"),
                    "body_p90_us": _f(s, "body_span_us", "p90"),
                    "span_p50_us": _f(s, "task_span_us", "p50"),
                    "wall_p50_us": _f(s, "op_wall_us", "p50"),
                    "dep_p50_us": round(s.get("dep_wait_us_p50", 0), 2),
                    "ahead_frac": (
                        round(s["loader_ahead_pos_frac"], 2)
                        if s.get("loader_ahead_pos_frac") is not None
                        else "-"
                    ),
                    "ok": ok,
                }
            )
        it = summ.get("_iteration") or {}
        if it:
            perf_rows.append(
                {
                    "case": case_id,
                    "op": "_iteration",
                    "n": it.get("n_iters_used"),
                    "body_p50_us": "-",
                    "body_p90_us": "-",
                    "span_p50_us": "-",
                    "wall_p50_us": (
                        round(it["iter_wall_us_p50"], 1)
                        if it.get("iter_wall_us_p50")
                        else "-"
                    ),
                    "dep_p50_us": "-",
                    "ahead_frac": "-",
                    "ok": ok,
                }
            )
    if perf_rows:
        print("\n--- perf (v2 window decode; body=span-dep_wait is the verdict) ---")
        print(
            fmt_table(
                perf_rows,
                ["case", "op", "n", "body_p50_us", "body_p90_us", "span_p50_us",
                 "wall_p50_us", "dep_p50_us", "ahead_frac", "ok"],
            )
        )
    if wall_rows:
        print("\n--- unprofiled wall-clock cross-check ---")
        print(fmt_table(wall_rows, ["case", "wall_ms", "final_step", "ms_per_live_iter"]))

    # reproducibility gate: body_span_p50 spread across _r<N> repeats
    import re as _re

    groups = {}
    for row in perf_rows:
        if row["op"].startswith("_") or row["body_p50_us"] in ("-", None):
            continue
        m = _re.match(r"(.+)_r(\d+)$", row["case"])
        if not m:
            continue
        groups.setdefault((m.group(1), row["op"]), []).append(row["body_p50_us"])
    rep_rows = []
    for (case_base, op), vals in sorted(groups.items()):
        if len(vals) < 2:
            continue
        lo, hi = min(vals), max(vals)
        mid = sum(vals) / len(vals)
        spread = (hi - lo) / mid if mid else 0
        rep_rows.append(
            {
                "case": case_base,
                "op": op,
                "n_runs": len(vals),
                "body_p50s": "/".join(f"{v:.2f}" for v in vals),
                "spread_pct": round(100 * spread, 1),
                "pass<5%": spread < 0.05,
            }
        )
    if rep_rows:
        print("\n--- reproducibility (body_span_p50 across repeats) ---")
        print(fmt_table(rep_rows, ["case", "op", "n_runs", "body_p50s",
                                   "spread_pct", "pass<5%"]))

    fails = [
        r for r in results if r.get("status") != "ok"
    ]
    print(f"\n{len(results) - len(fails)}/{len(results)} cases ok; "
          f"failures: {[r.get('spec', {}).get('case_id') for r in fails]}")
    print(f"artifacts: {tag_dir}")


def _f(s, k1, k2):
    v = (s.get(k1) or {}).get(k2)
    return round(v, 2) if isinstance(v, (int, float)) else "-"


if __name__ == "__main__":
    main()
