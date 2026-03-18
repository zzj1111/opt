#!/bin/bash
# Main evaluation script: iterate over all checkpoints × all benchmarks.
#
# Server (multi-GPU):
#   bash scripts/run_eval.sh --parallel 4
#   bash scripts/run_eval.sh --benchmarks math500,gsm8k --checkpoints layer_0,layer_1
#
# Local Mac (MPS):
#   bash scripts/run_eval.sh --device mps --batch-size 4 --lite
#
# Local CPU:
#   bash scripts/run_eval.sh --device cpu --batch-size 1 --lite --benchmarks gsm8k
#
# Limited VRAM (4-bit):
#   bash scripts/run_eval.sh --load-in-4bit --batch-size 8
#
# Prerequisites:
#   pip install "lm-eval[math,ifeval,sentencepiece]>=0.4.0" pyyaml
#
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
EVAL_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

# ── Defaults ────────────────────────────────────────────────────────────────
DEVICE="auto"       # auto | cuda | mps | cpu
GPU=0
BATCH_SIZE="auto"
PARALLEL=1
REGISTRY="$EVAL_DIR/checkpoints/registry.yaml"
CONFIG_DIR="$EVAL_DIR/configs"
OUTPUT_DIR="$EVAL_DIR/results/raw"
FORCE=false
LITE=false
LOAD_IN_4BIT=false
LOAD_IN_8BIT=false
MAX_GEN_TOKS=""

ALL_BENCHMARKS="math500 gsm8k mbpp ifeval mmlu_pro bbh mgsm ceval"
BENCHMARKS=""
CHECKPOINTS=""

# ── Arg parsing ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --device)         DEVICE="$2";      shift 2 ;;
        --gpu)            GPU="$2";         shift 2 ;;
        --batch-size)     BATCH_SIZE="$2";  shift 2 ;;
        --parallel)       PARALLEL="$2";    shift 2 ;;
        --registry)       REGISTRY="$2";    shift 2 ;;
        --benchmarks)     BENCHMARKS="$2";  shift 2 ;;
        --checkpoints)    CHECKPOINTS="$2"; shift 2 ;;
        --max-gen-toks)   MAX_GEN_TOKS="$2"; shift 2 ;;
        --load-in-4bit)   LOAD_IN_4BIT=true; shift  ;;
        --load-in-8bit)   LOAD_IN_8BIT=true; shift  ;;
        --lite)           LITE=true;        shift   ;;
        --force)          FORCE=true;       shift   ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

BENCHMARKS="${BENCHMARKS:-$ALL_BENCHMARKS}"

# ── Verify setup ────────────────────────────────────────────────────────────
python3 -c "import lm_eval" 2>/dev/null || {
    echo "lm-eval not found. Installing..."
    pip install "lm-eval[math,ifeval,sentencepiece]>=0.4.0"
}

python3 -c "import yaml" 2>/dev/null || pip install pyyaml

# ── Generate subsample indices (idempotent) ──────────────────────────────────
if [[ ! -f "$EVAL_DIR/results/indices/mmlu_pro.json" ]]; then
    echo "[setup] Generating subsample indices..."
    python3 "$SCRIPT_DIR/subsample.py" --out-dir "$EVAL_DIR/results/indices"
fi

# ── Load checkpoint registry ────────────────────────────────────────────────
CKPT_LABELS=$(python3 - <<EOF
import yaml
with open("$REGISTRY") as f:
    reg = yaml.safe_load(f)
# Filter out FILL_IN_PATH entries
valid = [k for k, v in reg.items() if "FILL_IN" not in str(v)]
print(" ".join(valid))
EOF
)

if [[ -n "$CHECKPOINTS" ]]; then
    CKPT_LABELS=$(echo "$CHECKPOINTS" | tr ',' ' ')
fi

LITE_TAG=""
if $LITE; then LITE_TAG=" [LITE]"; fi

echo "======================================================"
echo "Device:      $DEVICE${LITE_TAG}"
echo "Checkpoints: $CKPT_LABELS"
echo "Benchmarks:  $BENCHMARKS"
echo "Batch size:  $BATCH_SIZE"
echo "Output:      $OUTPUT_DIR"
echo "======================================================"

# ── Build job list ───────────────────────────────────────────────────────────
JOBS=()
for LABEL in $CKPT_LABELS; do
    CKPT_PATH=$(python3 - <<EOF
import yaml
with open("$REGISTRY") as f:
    reg = yaml.safe_load(f)
print(reg.get("$LABEL", ""))
EOF
)
    for BENCH in $BENCHMARKS; do
        JOBS+=("$LABEL|$CKPT_PATH|$BENCH")
    done
done

echo "Total jobs: ${#JOBS[@]}"

# ── Run jobs (sequential or parallel) ───────────────────────────────────────
EXTRA_FLAGS=""
if $FORCE;        then EXTRA_FLAGS="$EXTRA_FLAGS --force"; fi
if $LITE;         then EXTRA_FLAGS="$EXTRA_FLAGS --lite"; fi
if $LOAD_IN_4BIT; then EXTRA_FLAGS="$EXTRA_FLAGS --load-in-4bit"; fi
if $LOAD_IN_8BIT; then EXTRA_FLAGS="$EXTRA_FLAGS --load-in-8bit"; fi
if [[ -n "$MAX_GEN_TOKS" ]]; then EXTRA_FLAGS="$EXTRA_FLAGS --max-gen-toks $MAX_GEN_TOKS"; fi

run_job() {
    local LABEL="$1" CKPT_PATH="$2" BENCH="$3" GPU_ID="$4"
    python3 "$SCRIPT_DIR/run_single.py" \
        --ckpt-label "$LABEL" \
        --ckpt-path  "$CKPT_PATH" \
        --benchmark  "$BENCH" \
        --config-dir "$CONFIG_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --device     "$DEVICE" \
        --gpu        "$GPU_ID" \
        --batch-size "$BATCH_SIZE" \
        $EXTRA_FLAGS
}

if [[ "$PARALLEL" -le 1 ]]; then
    for JOB in "${JOBS[@]}"; do
        IFS='|' read -r LABEL CKPT_PATH BENCH <<< "$JOB"
        run_job "$LABEL" "$CKPT_PATH" "$BENCH" "$GPU"
    done
else
    # Parallel mode: distribute across GPUs 0..(PARALLEL-1)
    declare -A PIDS
    JOB_IDX=0
    for JOB in "${JOBS[@]}"; do
        IFS='|' read -r LABEL CKPT_PATH BENCH <<< "$JOB"
        GPU_ID=$(( JOB_IDX % PARALLEL ))
        run_job "$LABEL" "$CKPT_PATH" "$BENCH" "$GPU_ID" &
        PIDS[$!]="$LABEL×$BENCH"
        JOB_IDX=$(( JOB_IDX + 1 ))

        if (( JOB_IDX % PARALLEL == 0 )); then
            for PID in "${!PIDS[@]}"; do
                wait "$PID" && echo "[done] ${PIDS[$PID]}" || echo "[FAIL] ${PIDS[$PID]}"
            done
            unset PIDS; declare -A PIDS
        fi
    done
    for PID in "${!PIDS[@]}"; do
        wait "$PID" && echo "[done] ${PIDS[$PID]}" || echo "[FAIL] ${PIDS[$PID]}"
    done
fi

echo ""
echo "======================================================"
echo "All jobs complete. Run aggregate.py to build CSV:"
echo "  python scripts/aggregate.py"
echo "======================================================"
