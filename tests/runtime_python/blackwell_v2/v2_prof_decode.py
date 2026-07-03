"""Decode a Runtime-V2 8-track profiler buffer into a per-(task, iteration)
table with role spans, and summarize into the framework's verdict metrics.

Model (verified against runtime_v2.cuh + v2_role_codegen.cc @ a9f03e9c):
  - 8 tracks per SM: 0 consumer / 1 loader / 2 launcher / 3 storer /
    4 controller / 5 consumer-phase / 6 loader-phase / 7 launcher-phase.
  - Role tracks 0-3: the role warp loop wraps EVERY task (empty role bodies
    included) with BEGIN/END tagged event_idx = task_type — except
    TASK_BEGIN_TASK_GRAPH which is skipped entirely (runtime_v2.cuh ~745).
  - Window gating is per-iteration (g_v2_prof_window flipped by worker 0 at
    the iteration boundary), so per (SM, role-track) the in-window pairs are
    whole multiples of that SM's filtered queue, in queue x iteration order.
  - Phase tracks 5-7 carry sparse wait slices (only waits above a threshold
    emit) with V2_* pseudo event ids; they are paired per (track, event_idx)
    in stream order and attributed to the enclosing consumer task window by
    timestamp containment.
  - Timestamps are %globaltimer_lo (32-bit ns, wraps ~4.29 s). Pairing uses
    STREAM ORDER (wrap-safe); durations are mod-2^32. If a wrap is detected
    inside the window the table flags it and skips containment attribution.

Verdict metrics (see README.md):
  task_span   consumer BEGIN->END per tile-task (includes dep-wait)
  op_wall     per op instance per iteration: max consumer END - min BEGIN
  dep_wait    V2_DEP_WAIT phase time inside the consumer window
  loader_lead consumer_begin - loader_begin for the same (task, iteration)
  iter_wall   per iteration: max consumer END - min consumer BEGIN (all ops)
"""

import json

TWO32 = 1 << 32

# tag layout (profiler.h): [31:22] event_no [21:11] block_group [10:2]
# event_idx [1:0] type
EVENT_BEGIN = 0
EVENT_END = 1
EVENT_INSTANT = 2

# Tail layout mirrors runtime_v2.cuh (V2_PROF_SM_SLOTS=256, 136-worker fix):
V2_PROF_SM_SLOTS = 256
V2_PROF_BUF_ENTRIES = 120000 * 128
V2_PROF_SUFFIX_BASE = V2_PROF_BUF_ENTRIES - 2 * V2_PROF_SM_SLOTS
V2_PROF_SPIN_BASE = V2_PROF_SUFFIX_BASE - 2 * V2_PROF_SM_SLOTS * 7
V2_PROF_CURSOR_BASE = V2_PROF_SPIN_BASE - 8 * V2_PROF_SM_SLOTS
V2_PROF_MISC_BASE = V2_PROF_CURSOR_BASE - V2_PROF_SM_SLOTS  # drop counters
V2_PROF_TAIL_ENTRIES = ((1048576 + 1) + V2_PROF_SM_SLOTS + 8 * V2_PROF_SM_SLOTS
                        + 2 * V2_PROF_SM_SLOTS * 7 + 2 * V2_PROF_SM_SLOTS)
V2_NUM_GROUPS = 8

TRACK_NAMES = [
    "consumer",
    "loader",
    "launcher",
    "storer",
    "controller",
    "consumer-phase",
    "loader-phase",
    "launcher-phase",
]

# from python/mirage/mpk/profiler_persistent.py event_name_list
V2_PHASE_IDS = {
    205: "V2_ITER_SYNC",
    206: "V2_GO_WAIT",
    207: "V2_DEP_WAIT",
    208: "V2_PAGE_WAIT",
    209: "V2_W_TMA_WAIT",
    210: "V2_MMA_EMPTY_WAIT",
    211: "V2_TMEM_READY_WAIT",
    212: "V2_MAINLOOP_WAIT",
    213: "V2_EPILOGUE_WAIT",
    214: "V2_CONSUMER_DONE_WAIT",
}


def _task_type_names():
    try:
        from mirage.mpk.profiler_persistent import event_name_list

        return dict(event_name_list)
    except Exception:  # noqa: BLE001
        return {}


def decode_raw_events(buf_np):
    """buf -> {(block, group): [(event_idx, event_no, event_type, ts), ...]}
    in stream (cursor) order. Accepts a 1-D uint64 numpy array."""
    import numpy as np

    header = int(buf_np[0])
    nblocks = header & 0xFFFFFFFF
    ngroups = header >> 32
    assert ngroups >= 5, f"not a v2 buffer (ngroups={ngroups})"

    main = buf_np[: len(buf_np) - V2_PROF_TAIL_ENTRIES]
    nz = np.nonzero(main)[0]
    nz = nz[nz >= 1]  # skip header
    stride = nblocks * ngroups

    per_track = {}
    for slot in nz.tolist():
        e = int(main[slot])
        tag = e & 0xFFFFFFFF
        ts = e >> 32
        track = (slot - 1) % stride
        block = track // ngroups
        group = track % ngroups
        event_no = tag >> 22
        # block_group in tag [21:11] should match slot-derived track
        event_idx = (tag >> 2) & 0x1FF
        event_type = tag & 0x3
        per_track.setdefault((block, group), []).append(
            (event_idx, event_no, event_type, ts)
        )
    return nblocks, ngroups, per_track


def pair_stream(events, per_event_idx=False):
    """Pair BEGIN/END in stream order. Returns (pairs, errors) where each
    pair is (event_idx, begin_ts, end_ts, dur_ns)."""
    pairs = []
    errors = []
    if per_event_idx:
        open_by_idx = {}
        for ev in events:
            idx, _no, typ, ts = ev
            if typ == EVENT_BEGIN:
                if idx in open_by_idx:
                    errors.append(f"double BEGIN idx={idx}")
                open_by_idx[idx] = ts
            elif typ == EVENT_END:
                if idx not in open_by_idx:
                    errors.append(f"END without BEGIN idx={idx}")
                    continue
                b = open_by_idx.pop(idx)
                pairs.append((idx, b, ts, (ts - b) % TWO32))
        for idx in open_by_idx:
            errors.append(f"dangling BEGIN idx={idx}")
    else:
        open_ev = None
        for ev in events:
            idx, _no, typ, ts = ev
            if typ == EVENT_BEGIN:
                if open_ev is not None:
                    errors.append(f"double BEGIN idx={idx}")
                open_ev = (idx, ts)
            elif typ == EVENT_END:
                if open_ev is None:
                    errors.append(f"END without BEGIN idx={idx}")
                    continue
                if open_ev[0] != idx:
                    errors.append(f"BEGIN idx={open_ev[0]} closed by END idx={idx}")
                pairs.append((idx, open_ev[1], ts, (ts - open_ev[1]) % TWO32))
                open_ev = None
        if open_ev is not None:
            errors.append(f"dangling BEGIN idx={open_ev[0]}")
    return pairs, errors


def instance_map_from_task_graph(task_graph_json_path, instances):
    """Map task positions -> (instance_idx, op_name).

    Tasks are appended to all_tasks in REGISTRATION order; the task-pushing
    events (901/902/903) cover them in contiguous ranges (a fully-chained
    graph has a single event spanning every op). The mapping is therefore:
    concatenate the pushed ranges in task-id order and slice them by each
    registered instance's ntasks, in registration order. Validated by (a)
    total pushed count == sum(ntasks) and (b) per-slice task_type
    homogeneity. Returns (pos_to_inst, types, errors)."""
    with open(task_graph_json_path) as f:
        tg = json.load(f)
    types = [t.get("task_type", -1) for t in tg.get("all_tasks", [])]
    spans = sorted(
        (ev["first_task_id"], ev["last_task_id"])
        for ev in tg.get("all_events", [])
        if ev.get("event_type") in (901, 902, 903)
    )
    pushed = [p for (f0, l0) in spans for p in range(f0, l0)]
    errors = []
    total_expected = sum(i["ntasks"] for i in instances)
    if len(pushed) != total_expected:
        errors.append(
            f"pushed task count {len(pushed)} != sum of registered op "
            f"ntasks {total_expected}"
        )
        return {}, types, errors
    pos_to_inst = {}
    cur = 0
    for inst in instances:
        seg = pushed[cur : cur + inst["ntasks"]]
        tset = {types[p] for p in seg}
        if len(tset) != 1:
            errors.append(
                f"instance {inst['instance']}/{inst['op']}: mixed task types "
                f"{sorted(tset)} in slice — mapping misaligned"
            )
        for p in seg:
            pos_to_inst[p] = (inst["instance"], inst["op"])
        cur += inst["ntasks"]
    return pos_to_inst, types, errors


def decode_window_table(prof_tensor, queues, task_types):
    """Main entry. prof_tensor: torch uint64 tensor (or numpy array).
    queues: per-SM ordered task positions; task_types: per-position type id.
    Returns a JSON-serializable dict with rows + validation."""
    import numpy as np

    if hasattr(prof_tensor, "cpu"):
        buf = prof_tensor.cpu().numpy()
    else:
        buf = np.asarray(prof_tensor)
    nblocks, ngroups, per_track = decode_raw_events(buf)
    names = _task_type_names()

    # task types that never emit on role tracks
    begin_graph_ids = {
        tid for tid, nm in names.items() if nm == "TASK_BEGIN_TASK_GRAPH"
    }

    validation = {
        "nblocks": nblocks,
        "ngroups": ngroups,
        "track_event_counts": {
            f"b{b}g{g}": len(v) for (b, g), v in sorted(per_track.items())
        },
        "errors": [],
        "warnings": [],
    }

    # overflow drop counters (must be zero, else per-track streams have holes
    # and the positional queue mapping is invalid -> hard decode failure)
    if len(buf) >= V2_PROF_BUF_ENTRIES:
        drops = buf[V2_PROF_MISC_BASE : V2_PROF_MISC_BASE + V2_PROF_SM_SLOTS]
        total_drops = int(drops.sum())
        validation["dropped_events"] = total_drops
        if total_drops:
            validation["errors"].append(
                f"profiler overflow: {total_drops} dropped events (MISC region)"
            )

    # ---- role tracks: queue-aligned pairing --------------------------------
    # rows[(block, iter_w, qslot)] = {"task_pos", "type", "consumer": (b,e,d),
    #                                 "loader": ..., ...}
    rows = {}
    n_iters_by_block = {}
    wrap_flag = False
    for b in range(nblocks):
        fq = [p for p in queues[b] if task_types[p] not in begin_graph_ids] \
            if b < len(queues) else []
        if not fq:
            continue
        for g, role in ((0, "consumer"), (1, "loader"), (2, "launcher"), (3, "storer")):
            evs = per_track.get((b, g), [])
            if not evs:
                continue
            pairs, errs = pair_stream(evs)
            for e in errs:
                validation["errors"].append(f"b{b} {role}: {e}")
            # wrap detection within the stream
            last = None
            for (_i, bt, _e, _d) in pairs:
                if last is not None and (last - bt) % TWO32 > (1 << 31):
                    pass  # bt >= last modularly; fine
                if last is not None and bt < last and (last - bt) > (1 << 31):
                    wrap_flag = True
                last = bt
            if len(pairs) % len(fq) != 0:
                # Per-iteration window gating means in-window pairs must be a
                # whole multiple of the queue; anything else = dropped events
                # or a model violation -> positional mapping untrustworthy.
                extra = len(pairs) % len(fq)
                validation["errors"].append(
                    f"b{b} {role}: {len(pairs)} pairs not a multiple of "
                    f"queue len {len(fq)} (hard decode failure; trimmed "
                    f"{extra} from front for diagnosis only)"
                )
                pairs = pairs[extra:]
            n_it = len(pairs) // len(fq)
            n_iters_by_block.setdefault(b, {})[role] = n_it
            for k, (idx, bts, ets, dur) in enumerate(pairs):
                it = k // len(fq)
                q = k % len(fq)
                pos = fq[q]
                if idx != task_types[pos]:
                    validation["errors"].append(
                        f"b{b} {role} iter{it} q{q}: type {idx} != queue type "
                        f"{task_types[pos]} (pos {pos})"
                    )
                    continue
                key = (b, it, q)
                r = rows.setdefault(
                    key,
                    {"block": b, "iter_w": it, "q": q, "task_pos": pos, "type": idx},
                )
                r[role] = (bts, ets, dur)

    if wrap_flag:
        validation["warnings"].append(
            "32-bit timestamp wrap inside window; phase attribution skipped"
        )

    # consistent window-iteration count across blocks/roles
    iter_counts = {
        v for d in n_iters_by_block.values() for v in d.values()
    }
    validation["window_iters_per_block_role"] = sorted(iter_counts)
    n_window_iters = max(iter_counts) if iter_counts else 0

    # ---- phase tracks -------------------------------------------------------
    phase_rows = []  # (block, phase_name, begin, end, dur)
    for b in range(nblocks):
        for g in (5, 6, 7):
            evs = per_track.get((b, g), [])
            if not evs:
                continue
            pairs, errs = pair_stream(evs, per_event_idx=True)
            for e in errs:
                validation["warnings"].append(f"b{b} track{g}: {e}")
            for idx, bts, ets, dur in pairs:
                phase_rows.append(
                    (b, V2_PHASE_IDS.get(idx, f"phase_{idx}"), bts, ets, dur)
                )

    # attribute dep-wait (and other consumer-phase slices) to consumer windows
    if not wrap_flag:
        cons_by_block = {}
        for (b, it, q), r in rows.items():
            if "consumer" in r:
                cons_by_block.setdefault(b, []).append((r["consumer"][0], r["consumer"][1], (it, q)))
        for b in cons_by_block:
            cons_by_block[b].sort()
        for (b, pname, bts, ets, dur) in phase_rows:
            lst = cons_by_block.get(b)
            if not lst:
                continue
            import bisect

            i = bisect.bisect_right(lst, (bts, TWO32, (1 << 30, 0))) - 1
            if i >= 0:
                cb, ce, key = lst[i]
                if bts >= cb and bts <= ce:
                    r = rows[(b, key[0], key[1])]
                    r.setdefault("phases", {}).setdefault(pname, 0)
                    r["phases"][pname] += dur

    validation["decode_ok"] = len(validation["errors"]) == 0
    return {
        "n_window_iters": n_window_iters,
        "rows": [
            {
                **{k: v for k, v in r.items()},
            }
            for r in rows.values()
        ],
        "validation": validation,
        "names": {str(k): v for k, v in names.items()},
    }


# ---------------------------------------------------------------------------
# Summarization: window table + instance map -> verdict metrics per op
# ---------------------------------------------------------------------------
def _pctl(sorted_vals, p):
    if not sorted_vals:
        return None
    k = min(len(sorted_vals) - 1, int(round(p / 100.0 * (len(sorted_vals) - 1))))
    return sorted_vals[k]


def summarize(table, instances, task_graph_json, total_iters,
              drop_first_window_iter=True, drop_last_iter=True):
    """instances: [{"instance": i, "op": name, "ntasks": n}, ...] in
    registration order — matched 1:1 against task-pushing events in the task
    graph. total_iters = S (max_seq_length): absolute iter of window slot w is
    S - n_window + w; the LAST absolute iteration (S-1) is the zero-M
    post-done iteration and is dropped from stats."""
    pos_to_inst, types, map_errors = instance_map_from_task_graph(
        task_graph_json, instances
    )
    if map_errors:
        return {"error": "; ".join(map_errors)}

    n_window = table["n_window_iters"]
    w0_abs = total_iters - n_window
    drop_abs = set()
    if drop_last_iter:
        drop_abs.add(total_iters - 1)
    if drop_first_window_iter:
        drop_abs.add(w0_abs)

    # collect. Pipeline overlap needs the per-SM stream order: sort each
    # block's rows by (iter_w, q-slot) — that IS execution order on that SM.
    per_op = {}
    per_iter_all = {}
    rows_by_block = {}
    for r in table["rows"]:
        if "consumer" not in r:
            continue
        rows_by_block.setdefault(r["block"], []).append(r)

    for b, rl in rows_by_block.items():
        rl.sort(key=lambda r: (r["iter_w"], r["q"]))

    for b, rl in rows_by_block.items():
        prev = None  # previous task's consumer end, SAME iteration only
        prev_iter = None
        for r in rl:
            it_abs = w0_abs + r["iter_w"]
            inst_op = pos_to_inst.get(r["task_pos"])
            cb, ce, cd = r["consumer"]
            # pipeline overlap: loader BEGIN of THIS task vs consumer END of
            # the PREVIOUS task on the same SM (ring depth allows the loader
            # to run ahead). Only within one iteration — across the iteration
            # barrier "ahead" would measure the go-wait, not pipelining.
            if (
                prev is not None
                and prev_iter == r["iter_w"]
                and "loader" in r
                and it_abs not in drop_abs
                and inst_op is not None
            ):
                lb = r["loader"][0]
                ahead = (prev - lb) % TWO32  # >0 (small) => loader started
                # before previous consumer finished
                d = per_op.setdefault(
                    inst_op[1],
                    {"spans": [], "dep": [], "body": [], "ahead": [],
                     "walls": {}, "insts": set(), "hidden": []},
                )
                if ahead < (1 << 31):
                    d["ahead"].append(ahead)
                else:
                    d["ahead"].append(-((TWO32 - ahead) % TWO32))
            prev = ce
            prev_iter = r["iter_w"]
            if it_abs in drop_abs or inst_op is None:
                continue
            inst, op = inst_op
            d = per_op.setdefault(
                op,
                {"spans": [], "dep": [], "body": [], "ahead": [],
                 "walls": {}, "insts": set(), "hidden": []},
            )
            d["insts"].add(inst)
            d["spans"].append(cd)
            dep = r.get("phases", {}).get("V2_DEP_WAIT", 0)
            d["dep"].append(dep)
            # per-sample body = span - dep-wait (V2_DEP_WAIT is exact /
            # unthresholded: emitted around every wait_task_dependency)
            d["body"].append(max(cd - dep, 0))
            if "loader" in r:
                # load hidden under consume: loader END before consumer END
                le = r["loader"][1]
                hid = (ce - le) % TWO32
                d["hidden"].append(hid if hid < (1 << 31) else -((TWO32 - hid) % TWO32))
            w = d["walls"].setdefault((inst, it_abs), [None, None])
            w[0] = cb if w[0] is None else min(w[0], cb)
            w[1] = ce if w[1] is None else max(w[1], ce)
            pi = per_iter_all.setdefault(it_abs, [None, None])
            pi[0] = cb if pi[0] is None else min(pi[0], cb)
            pi[1] = ce if pi[1] is None else max(pi[1], ce)

    summary = {}
    for op, d in per_op.items():
        spans = sorted(d["spans"])
        deps = sorted(d["dep"])
        bodies = sorted(d["body"])
        walls = sorted((e - b) % TWO32 for b, e in d["walls"].values())
        aheads = sorted(d["ahead"])
        summary[op] = {
            "n_samples": len(spans),
            "n_instances": len(d["insts"]),
            # inclusive scheduled span (production-visible, incl. dep-wait)
            "task_span_us": {
                "p50": _pctl(spans, 50) / 1e3 if spans else None,
                "p90": _pctl(spans, 90) / 1e3 if spans else None,
                "max": spans[-1] / 1e3 if spans else None,
            },
            # PRIMARY per-op verdict: span minus dep-wait, per sample
            "body_span_us": {
                "p50": _pctl(bodies, 50) / 1e3 if bodies else None,
                "p90": _pctl(bodies, 90) / 1e3 if bodies else None,
            },
            "dep_wait_us_p50": (_pctl(deps, 50) or 0) / 1e3,
            "dep_wait_us_p90": (_pctl(deps, 90) or 0) / 1e3,
            "op_wall_us": {
                "p50": _pctl(walls, 50) / 1e3 if walls else None,
                "p90": _pctl(walls, 90) / 1e3 if walls else None,
            },
            # >0: this task's loader began before the previous task's
            # consumer finished on the same SM => pipelining across tasks
            "loader_ahead_us_p50": (_pctl(aheads, 50) / 1e3) if aheads else None,
            "loader_ahead_pos_frac": (
                sum(1 for x in d["ahead"] if x > 0) / len(aheads) if aheads else None
            ),
            "load_hidden_us_p50": (
                _pctl(sorted(d["hidden"]), 50) / 1e3 if d["hidden"] else None
            ),
        }
    iter_walls = sorted(
        (e - b) % TWO32 for b, e in per_iter_all.values() if b is not None
    )
    summary["_iteration"] = {
        "n_iters_used": len(iter_walls),
        "iter_wall_us_p50": _pctl(iter_walls, 50) / 1e3 if iter_walls else None,
        "iter_wall_us_p90": _pctl(iter_walls, 90) / 1e3 if iter_walls else None,
    }
    summary["_decode_ok"] = bool(table.get("validation", {}).get("decode_ok", False))
    return summary
