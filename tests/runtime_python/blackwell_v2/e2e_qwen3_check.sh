#!/usr/bin/env bash
# Qwen3-8B e2e calibration: demo v1 vs --use-v2 on one local GPU.
#   stage 1: v1 run   -> tokens + ms/tok        (reference: ~4.03 ms/tok)
#   stage 2: v2 run   -> tokens + ms/tok        (the number under test)
#   stage 3: v2 run --profiling -> role-track perfetto trace + raw buffer
# Usage: e2e_qwen3_check.sh <gpu_index> <out_dir> [max_new_tokens]
set -u
GPU="${1:?gpu index}"
OUT="${2:?output dir}"
MNT="${3:-64}"
PY=/home/muhengl/mirage/.venv/bin/python
DEMO_DIR=/home/muhengl/mirage/demo/qwen3
mkdir -p "$OUT"
# mpi4py needs libmpi on the loader path (the demo's try/except only catches
# ImportError, not the shared-lib load failure)
export LD_LIBRARY_PATH="/usr/mpi/gcc/openmpi-4.1.9a1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

run_demo () {
  local name="$1"; shift
  echo "=== [$name] $(date +%H:%M:%S) starting"
  ( cd "$DEMO_DIR" && CUDA_VISIBLE_DEVICES="$GPU" timeout 3600 \
      "$PY" demo.py --use-mirage --max-new-tokens "$MNT" \
        --save-tokens "$OUT/tokens_${name}.json" \
        --output-dir "$OUT/compile_${name}" \
        "$@" ) > "$OUT/log_${name}.txt" 2>&1
  local rc=$?
  echo "=== [$name] rc=$rc; latency line:"
  grep -E "per-token latency" "$OUT/log_${name}.txt" || tail -3 "$OUT/log_${name}.txt"
}

mkdir -p "$OUT/compile_v1" "$OUT/compile_v2" "$OUT/compile_v2prof"
run_demo v1
run_demo v2 --use-v2
run_demo v2prof --use-v2 --profiling --trace-name "$OUT/qwen3_v2_roles" \
                --prof-dump "$OUT/qwen3_v2_prof.npy"

echo "=== token comparison:"
"$PY" - "$OUT" <<'EOF'
import json, sys, os
out = sys.argv[1]
def load(n):
    p = os.path.join(out, f"tokens_{n}.json")
    return json.load(open(p)) if os.path.exists(p) else None
v1, v2 = load("v1"), load("v2")
if v1 and v2:
    t1, t2 = v1["token_ids"], v2["token_ids"]
    n = next((i for i, (a, b) in enumerate(zip(t1, t2)) if a != b), min(len(t1), len(t2)))
    print(f"v1 ms/tok={v1['latency_ms_per_token']:.3f}  v2 ms/tok={v2['latency_ms_per_token']:.3f}")
    print(f"prefix match: {n}/{min(len(t1),len(t2))} tokens")
    print("v1 text head:", v1["text"][:160].replace("\n", " "))
    print("v2 text head:", v2["text"][:160].replace("\n", " "))
else:
    print("missing token dumps:", v1 is None, v2 is None)
EOF
