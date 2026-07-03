"""Offline re-decode of a saved perf case (prof_*.npy + v2_map.json +
result.json instances) — lets decoder fixes be validated without GPU time.

Usage: python redecode.py <case_dir>
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2_prof_decode import decode_window_table, summarize


def main(case_dir):
    r = json.load(open(os.path.join(case_dir, "result.json")))
    v2map = json.load(open(r["v2_map"]))
    buf = np.load(r["prof_raw"])
    table = decode_window_table(buf, v2map["queues"], v2map["task_types"])
    v = table["validation"]
    print(f"decode_ok={v['decode_ok']} errors={len(v['errors'])} "
          f"warnings={len(v['warnings'])} n_window_iters={table['n_window_iters']}")
    for e in v["errors"][:5]:
        print("  ERR:", e)
    summ = summarize(
        table,
        r["instances"],
        os.path.join(case_dir, "compile", "task_graph_rank0.json"),
        total_iters=r["iters"],
    )
    if "error" in summ:
        print("SUMMARIZE ERROR:", summ["error"])
        return 1
    for op, s in summ.items():
        if op.startswith("_") or not isinstance(s, dict):
            continue
        print(
            f"{op}: n={s['n_samples']} inst={s['n_instances']} "
            f"body_p50={s['body_span_us']['p50']:.2f}us "
            f"body_p90={s['body_span_us']['p90']:.2f}us "
            f"span_p50={s['task_span_us']['p50']:.2f}us "
            f"dep_p50={s['dep_wait_us_p50']:.2f}us "
            f"opwall_p50={s['op_wall_us']['p50']:.2f}us "
            f"ahead_frac={s['loader_ahead_pos_frac']} "
            f"hidden_p50={s['load_hidden_us_p50']}"
        )
    print("_iteration:", summ["_iteration"])
    print("_decode_ok:", summ["_decode_ok"])
    # persist refreshed summary next to the original
    out = os.path.join(case_dir, "summary_redecoded.json")
    with open(out, "w") as f:
        json.dump(summ, f, indent=1)
    print("saved:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
