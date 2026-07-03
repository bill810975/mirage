"""Subprocess entry for ONE DSv3-FFN v2 framework case (mirrors
case_runner.py). Usage: python ffn_case_runner.py <spec.json> <result.json>"""

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
        import torch  # noqa: F401

        from dsv3_ffn_harness import (run_ffn_correctness_case,
                                      run_ffn_perf_case)

        if spec["mode"] == "ffn_correctness":
            case_result = run_ffn_correctness_case(spec, out_dir)
        elif spec["mode"] == "ffn_perf":
            case_result = run_ffn_perf_case(spec, out_dir)
        else:
            raise ValueError(spec["mode"])
        result.update(case_result)
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
