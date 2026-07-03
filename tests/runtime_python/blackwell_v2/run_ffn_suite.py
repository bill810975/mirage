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


def build_cases(what: str, repeats: int, nwarps: int, L: int, iters: int):
    cases = []
    if what in ("correctness", "all"):
        cases.append({"name": "ffn_corr_typical", "mode": "ffn_correctness",
                      "nwarps": nwarps, "timeout_s": 2400})
        cases.append({"name": "ffn_corr_local8", "mode": "ffn_correctness",
                      "force_local8": True, "dump_inputs": False,
                      "nwarps": nwarps, "timeout_s": 2400})
    if what in ("perf", "all"):
        for r in range(repeats):
            cases.append({"name": f"ffn_perf_prof_r{r}", "mode": "ffn_perf",
                          "profiled": True, "L": L, "iters": iters,
                          "nwarps": nwarps, "timeout_s": 3000})
        cases.append({"name": "ffn_perf_nowall", "mode": "ffn_perf",
                      "profiled": False, "L": L, "iters": iters,
                      "nwarps": nwarps, "timeout_s": 3000})
        cases.append({"name": "ffn_perf_prof_local8", "mode": "ffn_perf",
                      "profiled": True, "L": L, "iters": iters,
                      "force_local8": True, "nwarps": nwarps,
                      "timeout_s": 3000})
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
                    choices=["correctness", "perf", "all"])
    ap.add_argument("--devices", default="6")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--nwarps", type=int, default=4, choices=[4, 7])
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--iters", type=int, default=32)
    ap.add_argument("--tag", default="ffn_r0")
    args = ap.parse_args()

    device = args.devices.split(",")[0]
    tag_dir = os.path.join(HERE, "_results", args.tag)
    os.makedirs(tag_dir, exist_ok=True)

    cases = build_cases(args.what, args.repeats, args.nwarps, args.L,
                        args.iters)
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
