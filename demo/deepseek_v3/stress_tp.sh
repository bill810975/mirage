#!/usr/bin/env bash
# DeepSeek V3 MPK stress/regression launcher.
#
# Default case mirrors the local stress/profile workflow.  Set
# CASE=tiny_nomtp_decode to reproduce the small-QLen non-MTP decode regression:
#   CASE=tiny_nomtp_decode TP=4 GPUS=0,1,4,6 bash demo/deepseek_v3/stress_tp.sh
set -euo pipefail

CASE="${CASE:-stress}"
MODEL_PATH="${MODEL_PATH:-/raid/catalyst/models/DeepSeek-V3}"
TP="${TP:-4}"
GPUS="${GPUS:-0,1,2,3}"
OUT="${OUT:-/tmp/deepseek_v3_${CASE}_tp${TP}_$(date +%H%M%S).log}"

export CUDA_VISIBLE_DEVICES="$GPUS"
source /raid/user_data/muhengl/.venv/bin/activate
export PATH=/usr/mpi/gcc/openmpi-4.1.9a1/bin:$PATH
export MPI_HOME=/usr/mpi/gcc/openmpi-4.1.9a1
export MPI_INC_PATH=$MPI_HOME/include
export MPI_LIB_PATH=$MPI_HOME/lib
export NVSHMEM_HOME=/home/muhengl/local/nvshmem-3.6.5-dev/usr
export NVSHMEM_INC_PATH=$NVSHMEM_HOME/include/nvshmem_13
export NVSHMEM_LIB_PATH=$NVSHMEM_HOME/lib/x86_64-linux-gnu/nvshmem/13
export LD_LIBRARY_PATH=$NVSHMEM_LIB_PATH:$MPI_HOME/lib:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=/home/muhengl/local/nvshmem-3.6.5-extract/usr/lib/x86_64-linux-gnu/nvshmem/13/libnvshmem_host.so.3.6.5
export NVSHMEM_SYMMETRIC_SIZE="${NVSHMEM_SYMMETRIC_SIZE:-4294967296}"

cd "$(dirname "$0")/../.."

COMMON_ARGS=(
  --model-path "$MODEL_PATH"
  --use-mirage
)

case "$CASE" in
  stress)
    LAYERS="${LAYERS:-0-10}"
    PROMPT_LEN="${PROMPT_LEN:-1024}"
    DECODE="${DECODE:-8}"
    MBT="${MBT:-128}"
    BATCH="${BATCH:-1}"
    MTP="${MTP:-2}"
    MAX_SEQ="${MAX_SEQ:-$((PROMPT_LEN + DECODE + 128))}"
    PAGES_PER_REQ=$(((MAX_SEQ + 127) / 128))
    MAX_PAGES="${MAX_PAGES:-$((PAGES_PER_REQ * BATCH + 16))}"
    CASE_ARGS=(
      --layers "$LAYERS"
      --mtp "$MTP"
      --max-num-batched-tokens "$MBT"
      --max-num-batched-requests "$BATCH"
      --prompt-length "$PROMPT_LEN"
      --max-new-tokens "$DECODE"
      --max-seq-length "$MAX_SEQ"
      --max-num-pages "$MAX_PAGES"
      --page-size 128
      --ignore-eos
    )
    ;;
  tiny_nomtp_decode)
    LAYERS="${LAYERS:-0-0}"
    MBT="${MBT:-1}"
    MTP="${MTP:-0}"
    MAX_SEQ="${MAX_SEQ:-16}"
    CASE_ARGS=(
      --layers "$LAYERS"
      --mtp "$MTP"
      --max-num-batched-tokens "$MBT"
      --max-seq-length "$MAX_SEQ"
    )
    ;;
  *)
    echo "Unknown CASE=$CASE" >&2
    exit 2
    ;;
esac

echo "[deepseek_v3_stress] CASE=$CASE TP=$TP GPUS=$CUDA_VISIBLE_DEVICES OUT=$OUT"
START=$(date +%s)
set +e
mpirun --allow-run-as-root -np "$TP" \
  -x CUDA_VISIBLE_DEVICES -x MASTER_PORT -x LD_LIBRARY_PATH -x LD_PRELOAD \
  -x PATH -x MPI_INC_PATH -x MPI_LIB_PATH -x NVSHMEM_INC_PATH \
  -x NVSHMEM_LIB_PATH -x NVSHMEM_SYMMETRIC_SIZE \
  python demo/deepseek_v3/demo.py "${COMMON_ARGS[@]}" "${CASE_ARGS[@]}" \
  > "$OUT" 2>&1
RC=$?
set -e
END=$(date +%s)
echo "[deepseek_v3_stress] elapsed=$((END - START))s rc=$RC"
grep -E "Prompt length|per-token latency|Traceback|illegal|FAILED" "$OUT" || true
exit "$RC"
