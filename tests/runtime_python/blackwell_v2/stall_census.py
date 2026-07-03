"""Census of stall iterations across saved profiled perf cases.

For every case dir with a raw profiler buffer: recompute per-iteration
consumer walls and count outliers (> 1.5x the case median). Prints one row
per case. Usage: python stall_census.py <case_dir> [<case_dir> ...]
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2_prof_decode import TWO32, decode_window_table


def census(case_dir):
    rp = os.path.join(case_dir, "result.json")
    if not os.path.exists(rp):
        return None
    r = json.load(open(rp))
    if "prof_raw" not in r or "v2_map" not in r:
        return None
    if not os.path.exists(r["prof_raw"]):
        return None
    v2map = json.load(open(r["v2_map"]))
    buf = np.load(r["prof_raw"])
    table = decode_window_table(buf, v2map["queues"], v2map["task_types"])
    per_iter = {}
    for row in table["rows"]:
        if "consumer" not in row:
            continue
        b, e, _ = row["consumer"]
        pi = per_iter.setdefault(row["iter_w"], [None, None])
        pi[0] = b if pi[0] is None else min(pi[0], b)
        pi[1] = e if pi[1] is None else max(pi[1], e)
    walls = sorted(
        ((e - b) % TWO32) / 1e3 for b, e in per_iter.values() if b is not None
    )
    if not walls:
        return None
    med = walls[len(walls) // 2]
    stalls = [w for w in walls if w > 1.5 * med]
    return {
        "case": os.path.basename(case_dir),
        "n_iters": len(walls),
        "median_us": round(med, 1),
        "n_stall": len(stalls),
        "max_us": round(walls[-1], 1),
    }


if __name__ == "__main__":
    dirs = []
    for a in sys.argv[1:]:
        if os.path.isdir(a) and os.path.exists(os.path.join(a, "result.json")):
            dirs.append(a)
        elif os.path.isdir(a):
            dirs += [os.path.join(a, d) for d in sorted(os.listdir(a))]
    for d in dirs:
        c = census(d)
        if c:
            print(
                f"{c['case']:32s} iters={c['n_iters']:3d} "
                f"median={c['median_us']:8.1f}us n_stall={c['n_stall']} "
                f"max={c['max_us']:9.1f}us"
            )
