"""Orchestrator for the DSv3 fused-FFN v2 chain cases (Step 3a).

Mirrors run_suite.py's hang-proof pattern: one subprocess per case
(ffn_case_runner.py), CUDA_VISIBLE_DEVICES pinning, hard timeout, JSON
results under _results/<tag>/.

Usage:
  .venv/bin/python tests/runtime_python/blackwell_v2/run_ffn_suite.py \
      --what correctness --devices 6 --tag ffn_corr_r1
  .venv/bin/python tests/runtime_python/blackwell_v2/run_ffn_suite.py \
      --what perf --devices 6 --repeats 3 --tag ffn_perf_r1
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def build_cases(what: str, repeats: int, nwarps: int, L: int, iters: int,
                fold: bool = False, extra: dict | None = None):
    extra = extra or {}
    cases = []
    if what == "pipe":
        # ffn item 1 gate matrix (ffn_item1_spec.md §7): per-tile pipeline
        # W13/W2 correctness — active in {0,1,4,8} (forced routing), the
        # EP-local filter case (les=128 second half), and the typical seed.
        # n_tile edges are inherently covered: every per-tile task (first +
        # last tile of each op instance) runs in every case.
        base = {"mode": "ffn_correctness", "pipe": True,
                "dump_inputs": False, "timeout_s": 2400}
        cases.append({"name": "pipe_corr_typical", **base})
        for k in (0, 1, 4, 8):
            cases.append({"name": f"pipe_corr_active{k}",
                          "force_active": k, **base})
        cases.append({"name": "pipe_corr_filter_les128", "les": 128, **base})
        for c in cases:
            c.update(extra)
        return cases
    if what in ("correctness", "all"):
        cases.append({"name": "ffn_corr_typical", "mode": "ffn_correctness",
                      "nwarps": nwarps, "fold": fold, "timeout_s": 2400})
        cases.append({"name": "ffn_corr_local8", "mode": "ffn_correctness",
                      "force_local8": True, "dump_inputs": False,
                      "nwarps": nwarps, "fold": fold, "timeout_s": 2400})
    if what in ("perf", "all"):
        for r in range(repeats):
            cases.append({"name": f"ffn_perf_prof_r{r}", "mode": "ffn_perf",
                          "profiled": True, "L": L, "iters": iters,
                          "nwarps": nwarps, "fold": fold, "timeout_s": 3000})
        cases.append({"name": "ffn_perf_nowall", "mode": "ffn_perf",
                      "profiled": False, "L": L, "iters": iters,
                      "nwarps": nwarps, "fold": fold, "timeout_s": 3000})
        cases.append({"name": "ffn_perf_prof_local8", "mode": "ffn_perf",
                      "profiled": True, "L": L, "iters": iters,
                      "force_local8": True, "nwarps": nwarps, "fold": fold,
                      "timeout_s": 3000})
    for c in cases:
        c.update(extra)
    return cases


def run_case(case, device, tag_dir):
    case_dir = os.path.join(tag_dir, case["name"])
    os.makedirs(case_dir, exist_ok=True)
    spec_path = os.path.join(case_dir, "spec.json")
    result_path = os.path.join(case_dir, "result.json")
    with open(spec_path, "w") as f:
        json.dump(case, f, indent=1)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    log_path = os.path.join(case_dir, "log.txt")
    with open(log_path, "w") as logf:
        p = subprocess.Popen(
            [PY, os.path.join(HERE, "ffn_case_runner.py"), spec_path,
             result_path],
            stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=HERE)
        try:
            rc = p.wait(timeout=case.get("timeout_s", 2400))
        except subprocess.TimeoutExpired:
            p.kill()
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                pass
            return {"name": case["name"], "status": "timeout(wedge?)"}
    if not os.path.exists(result_path):
        return {"name": case["name"], "status": f"crashed rc={rc}"}
    with open(result_path) as f:
        res = json.load(f)
    res["name"] = case["name"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="all",
                    choices=["correctness", "perf", "all", "pipe"])
    ap.add_argument("--devices", default="6")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--nwarps", type=int, default=4, choices=[4, 7])
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--iters", type=int, default=32)
    ap.add_argument("--fold", action="store_true",
                    help="folded 3-op chain (router_quant_rms/w13_topk/"
                         "w2_silu) instead of the 6-op chain")
    ap.add_argument("--tag", default="ffn_r0")
    ap.add_argument("--extra", default=None,
                    help="JSON dict merged into every case spec (e.g. "
                         "'{\"rung\":\"mega\"}' for the fusion-ladder rungs)")
    args = ap.parse_args()

    device = args.devices.split(",")[0]
    tag_dir = os.path.join(HERE, "_results", args.tag)
    os.makedirs(tag_dir, exist_ok=True)

    cases = build_cases(args.what, args.repeats, args.nwarps, args.L,
                        args.iters, fold=args.fold,
                        extra=json.loads(args.extra) if args.extra else None)
    all_results = []
    for case in cases:  # serial (verdict-grade: quiet, one at a time)
        print(f"[ffn_suite] running {case['name']} on cuda:{device} ...",
              flush=True)
        res = run_case(case, device, tag_dir)
        status = res.get("status", "ok")
        print(f"[ffn_suite]   -> {status}", flush=True)
        all_results.append(res)

    with open(os.path.join(tag_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=1, default=str)

    # compact console summary
    for res in all_results:
        name, status = res.get("name"), res.get("status", "ok")
        line = f"  {name}: {status}"
        if res.get("kind") == "ffn_correctness" and "ops" in res:
            p = res["ops"].get("_pass", {})
            line += "  pass=" + ",".join(
                f"{k}:{v}" for k, v in p.items())
        if res.get("kind") == "ffn_perf":
            line += (f"  wall_ms={res.get('wall_ms', -1):.1f}"
                     f" steps={res.get('final_step')}"
                     f" active={res.get('active_counts')}")
        print(line)


if __name__ == "__main__":
    main()
