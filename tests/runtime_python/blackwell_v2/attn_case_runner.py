"""Subprocess entry for ONE DSv3-ATTN v2 framework case (mirrors
ffn_case_runner.py). Usage: python attn_case_runner.py <spec.json> <result.json>"""

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

        from dsv3_attn_harness import (run_attn_correctness_case,
                                       run_attn_multistep_case,
                                       run_attn_perf_case)

        if spec["mode"] == "attn_correctness":
            case_result = run_attn_correctness_case(spec, out_dir)
        elif spec["mode"] == "attn_multistep":
            case_result = run_attn_multistep_case(spec, out_dir)
        elif spec["mode"] == "attn_perf":
            case_result = run_attn_perf_case(spec, out_dir)
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
