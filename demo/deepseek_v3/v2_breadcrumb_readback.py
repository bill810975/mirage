#!/usr/bin/env python3
"""Readback / name-annotation helper for the Runtime-V2 per-worker breadcrumb.

The breadcrumb (MPK_V2_BREADCRUMB=1) writes, for every (worker, role), a STARTED
word to host-mapped PINNED memory BEFORE each task's execute_task and a COMPLETED
word AFTER. When the megakernel dies with cudaErrorIllegalAddress, the pinned
buffer survives the context poisoning and the C++ launch path
(persistent_kernel_v2.cuh::dump_breadcrumb) prints one line per IN-FLIGHT
(worker, role) — the ones where STARTED != COMPLETED — to STDOUT, with the
task_type as a NUMBER. That is a fault-CANDIDATE set (an illegal address poisons
the whole context, so every task in flight at the crash shows up); the true
faulter is IN the set, and the deterministic crash + the CUDA error narrow it.

This helper turns those numeric task_type ids into human-readable task NAMES
(via mirage.mpk.profiler_persistent.event_name_list, which self-syncs from the
C++ enum in runtime_header.h). Two ways to use it on the box:

  1. Pipe / pass the captured demo stdout (recommended — zero coupling):
       python demo/deepseek_v3/v2_breadcrumb_readback.py < demo_stdout.log
       python demo/deepseek_v3/v2_breadcrumb_readback.py demo_stdout.log
     It scans for the "[v2][breadcrumb] IN-FLIGHT: ..." lines and reprints
     them with task_type=<NAME> appended.

  2. Decode a single raw STARTED word (hex or dec) by hand:
       python demo/deepseek_v3/v2_breadcrumb_readback.py --word 0x0000000101<...>

STARTED word layout (MSB->LSB), see runtime_v2.cuh MPK_V2_BC_STARTED:
    [63:40] iter_num (24b)  [39:24] sequence_in_iter (16b)
    [23:8]  task_type (16b) [7:0]   role_id (8b)
  (Controller role=4 stores task_pos in the task_type field, not a task_type.)
"""

import argparse
import re
import sys

ROLE_NAME = {0: "consumer", 1: "loader", 2: "launcher", 3: "storer",
             4: "controller"}


def _load_name_map():
    """task_type id -> name, from the profiler's self-syncing table.

    Falls back to an empty map if mirage isn't importable (then names show as
    TASK_<id>)."""
    try:
        from mirage.mpk.profiler_persistent import event_name_list
        return dict(event_name_list)
    except Exception as e:  # pragma: no cover - best effort
        sys.stderr.write(
            f"[readback] WARN: could not import event_name_list ({e}); "
            "task names will be numeric.\n")
        return {}


def task_name(name_map, tid):
    return name_map.get(int(tid), f"TASK_#{int(tid)}")


def decode_word(word):
    """Return (iter_num, seq_in_iter, task_type_or_pos, role_id)."""
    word = int(word)
    iter_num = (word >> 40) & 0xFFFFFF
    seq = (word >> 24) & 0xFFFF
    ttype_or_pos = (word >> 8) & 0xFFFF
    role = word & 0xFF
    return iter_num, seq, ttype_or_pos, role


def annotate_stream(lines, name_map):
    """Scan text lines for the C++ breadcrumb IN-FLIGHT lines and reprint them
    with the task name resolved. Also surfaces a terse final summary."""
    # e.g. "[v2][breadcrumb] IN-FLIGHT: worker=17 role=consumer(0)
    #       iter=0 seq_in_iter=42 task_type=340  (STARTED=0x... COMPLETED=0x...)"
    # (also matches the older "FAULTING TASK:" label for saved logs.)
    fault_re = re.compile(
        r"(?:IN-FLIGHT|FAULTING TASK): worker=(\d+) role=(\w+)\((\d+)\) "
        r"iter=(\d+) seq_in_iter=(\d+) "
        r"(task_type|task_pos)=(\d+)")
    faults = []
    saw_marker = False
    for line in lines:
        if "[v2][breadcrumb]" in line:
            saw_marker = True
        m = fault_re.search(line)
        if not m:
            continue
        worker, role_s, role_id, it, seq, field, val = m.groups()
        role_id = int(role_id)
        if field == "task_type":
            nm = task_name(name_map, val)
            annotated = (f"IN-FLIGHT: worker={worker} "
                         f"role={ROLE_NAME.get(role_id, role_s)}({role_id}) "
                         f"iter={it} seq_in_iter={seq} "
                         f"task_type={val} ({nm})")
        else:  # controller: task_pos
            annotated = (f"IN-FLIGHT: worker={worker} "
                         f"role={ROLE_NAME.get(role_id, role_s)}({role_id}) "
                         f"iter={it} seq_in_iter={seq} task_pos={val} "
                         f"(controller fetch/publish — task_pos is an index "
                         f"into config.v2_per_sm_task_positions)")
        faults.append(annotated)
        print("[readback] " + annotated)

    print("")
    if not saw_marker:
        print("[readback] No '[v2][breadcrumb]' lines found in the input. "
              "Was MPK_V2_BREADCRUMB=1 set AND forwarded (-x MPK_V2_BREADCRUMB) "
              "AND did the kernel actually fault?")
    elif not faults:
        print("[readback] Breadcrumb ran but reported NO faulting (worker,role) "
              "(all STARTED==COMPLETED). The fault is likely OUTSIDE "
              "execute_task (controller drain / iter-barrier / a non-crumbed "
              "path). Inspect the raw buffer lines the C++ dump printed.")
    else:
        print(f"[readback] ==== {len(faults)} IN-FLIGHT (worker,role) slot(s) "
              f"at the crash — the illegal access is INSIDE one of these task "
              f"bodies (fault-candidate set; re-run to narrow) ====")
    return faults


def annotate_linv3(lines, wtma_only=False):
    """Surface the C++ dump_linv3_probe output (MPK_V2_LINV3_PROBE). The pinned
    probe region is decoded in C++ (Python has no direct access to it), so this
    just re-prints the '[v2][linv3_probe] ...' block and flags the decisive
    lines: an OUT-OF-RANGE task_offset / NULL ptr / BAD SMEM region (=> copied-
    TaskDesc metadata bug), a stuck phase marker (=> fault in that phase), all
    markers completed + metadata valid (=> concurrent poisoner), OR — the M3
    W-TMA goal — a FAULTING per-worker W-TMA record (dst/tmap dumped but
    WT_COMPLETED unset) and/or a BAD OPERAND (dst not 1024-aligned / NULL
    tensor-map / coord/source OOB / NULL gmem base) at linear_device.cuh:88.

    wtma_only=True (the --wtma flag) restricts the reprint to the per-worker
    W-TMA record section (workers + summary + verdict), for a focused read of
    the raw W-TMA operands on the box."""
    probe = [ln.rstrip("\n") for ln in lines if "[v2][linv3_probe]" in ln]
    if not probe:
        print("[readback] No '[v2][linv3_probe]' lines found. Was "
              "MPK_V2_LINV3_PROBE=1 set AND forwarded (-x MPK_V2_LINV3_PROBE) "
              "AND MPK_V2_BREADCRUMB=1 set (-x MPK_V2_BREADCRUMB)?")
        return
    # Any '*** ... ***' marker the C++ dump emits is a red flag. This single
    # substring test subsumes the old explicit list (NULL / BAD / OUT-OF-RANGE /
    # etc.) AND the new W-TMA flags (FAULTING / NOT 1024-ALIGNED / NULL
    # TENSOR-MAP / ROW OOB / KCHUNK OOB / NULL GMEM BASE) without per-string
    # coupling, plus UNWRITTEN (which has no '***').
    def is_flag(ln):
        u = ln.upper()
        return "***" in ln or "OUT-OF-RANGE" in u or "UNWRITTEN" in u

    # For --wtma, keep only the per-worker W-TMA section: from the
    # "PER-WORKER W-TMA records" header through "end linv3 probe".
    if wtma_only:
        start = next((i for i, ln in enumerate(probe)
                      if "PER-WORKER W-TMA records" in ln), None)
        if start is None:
            print("[readback] --wtma: no 'PER-WORKER W-TMA records' section in "
                  "the input. Is this an OLD probe build (before the W-TMA "
                  "dump), or did the loader die before the W-TMA capture? Fall "
                  "back to the full probe (drop --wtma) and read the phase "
                  "markers.")
            return
        probe = probe[start:]
        print("[readback] ==== linear_v3 PER-WORKER W-TMA records (decoded in "
              "C++) ====")
    else:
        print("[readback] ==== linear_v3 metadata/phase + per-worker W-TMA "
              "probe (decoded in C++) ====")
    flagged = []
    for ln in probe:
        print(ln)
        if is_flag(ln):
            flagged.append(ln.strip())
    print("")
    if flagged:
        print(f"[readback] ==== {len(flagged)} FLAGGED line(s) — inspect these "
              f"first (a FAULTING/BAD-OPERAND W-TMA row pins the illegal "
              f"address; a metadata/region flag is a copied-TaskDesc bug) ====")
        for f in flagged:
            print("[readback]   " + f)
    else:
        print("[readback] No red-flags in the probe. If a W-TMA summary line is "
              "present with 0 FAULTING + 0 BAD OPERAND, the illegal address is "
              "NOT a W-TMA operand (=> concurrent poisoner: confirm with "
              "MPK_V2_LINV3_SKIP36, or a fault downstream of the first W-TMA). "
              "Otherwise read the PHASE markers: a marker STUCK mid-body "
              "localizes the faulting phase.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logfile", nargs="?",
                    help="captured demo stdout (default: read stdin)")
    ap.add_argument("--word", help="decode a single raw STARTED word "
                                    "(hex 0x.. or decimal) and exit")
    ap.add_argument("--linv3", action="store_true",
                    help="surface the MPK_V2_LINV3_PROBE metadata/phase + "
                         "per-worker W-TMA probe block from the captured stdout "
                         "(instead of the per-worker breadcrumb IN-FLIGHT "
                         "lines)")
    ap.add_argument("--wtma", action="store_true",
                    help="like --linv3 but restrict to the per-worker RAW "
                         "W-TMA argument records (the M3 dump: dst+alignment, "
                         "tensor-map, coords, box, gmem base, source offset, "
                         "COMPLETED marker) — flags the faulting loader + any "
                         "bad operand at linear_device.cuh:88")
    args = ap.parse_args()

    name_map = _load_name_map()

    if args.word is not None:
        w = int(args.word, 0)
        it, seq, tt, role = decode_word(w)
        rn = ROLE_NAME.get(role, f"role#{role}")
        if role == 4:
            print(f"iter={it} seq_in_iter={seq} task_pos={tt} role={rn}({role})")
        else:
            print(f"iter={it} seq_in_iter={seq} "
                  f"task_type={tt} ({task_name(name_map, tt)}) "
                  f"role={rn}({role})")
        return

    if args.logfile:
        with open(args.logfile, "r", errors="replace") as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()
    if args.wtma:
        annotate_linv3(lines, wtma_only=True)
    elif args.linv3:
        annotate_linv3(lines)
    else:
        annotate_stream(lines, name_map)


if __name__ == "__main__":
    main()
