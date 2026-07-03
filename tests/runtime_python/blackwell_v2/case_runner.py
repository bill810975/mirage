"""Subprocess entry point: run ONE framework case, write a JSON result.

Usage:  python case_runner.py <spec.json> <result.json>

The parent (run_suite.py) sets CUDA_VISIBLE_DEVICES and enforces a timeout;
this process only ever sees one GPU as device 0. Exit code 0 = result
written (which may still carry op-level failures); nonzero = infrastructure
failure (traceback in the result file when possible).
"""

import json
import os
import sys
import traceback


def main():
    spec_path, result_path = sys.argv[1], sys.argv[2]
    with open(spec_path) as f:
        spec = json.load(f)
    out_dir = os.path.dirname(os.path.abspath(result_path))
    os.makedirs(out_dir, exist_ok=True)

    result = {"spec": spec, "status": "crashed"}
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import torch  # noqa: F401 — fail fast if env broken

        from v2_harness import run_correctness_case, run_perf_case

        if spec["mode"] == "correctness":
            case_result = run_correctness_case(spec, out_dir)
        elif spec["mode"] == "perf":
            case_result = run_perf_case(spec, out_dir)
        else:
            raise ValueError(spec["mode"])
        result.update(case_result)
        # a case function may downgrade its own status (e.g. decode_failed);
        # only default to ok when it didn't set one
        result["status"] = case_result.get("status", "ok")
    except Exception:  # noqa: BLE001
        result["traceback"] = traceback.format_exc()
        with open(result_path, "w") as f:
            json.dump(result, f, indent=1, default=str)
        sys.exit(1)

    with open(result_path, "w") as f:
        json.dump(result, f, indent=1, default=str)


if __name__ == "__main__":
    main()
