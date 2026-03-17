#!/bin/bash
# Main evaluation script: iterate over all checkpoints × all benchmarks.
#
# Usage:
#   cd eval_experiment/
#   bash scripts/run_eval.sh                          # run everything
#   bash scripts/run_eval.sh --benchmarks math500,gsm8k   # specific benchmarks
#   bash scripts/run_eval.sh --checkpoints layer_0,layer_1  # specific checkpoints
#   bash scripts/run_eval.sh --gpu 0 --batch-size 32
#   bash scripts/run_eval.sh --parallel 2             # use 2 GPUs in parallel
#
# Prerequisites:
#   pip install lm-eval[math,ifeval,sentencepiece] pyyaml
#
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
EVAL_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

# ── Defaults ────────────────────────────────────────────────────────────────
GPU=0
BATCH_SIZE=16
PARALLEL=1         # number of parallel GPU workers
REGISTRY="$EVAL_DIR/checkpoints/registry.yaml"
CONFIG_DIR="$EVAL_DIR/configs"
OUTPUT_DIR="$EVAL_DIR/results/raw"
FORCE=false

ALL_BENCHMARKS="math500 gsm8k mbpp ifeval mmlu_pro bbh mgsm ceval"
BENCHMARKS=""
CHECKPOINTS=""

# ── Arg parsing ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)          GPU="$2";         shift 2 ;;
        --batch-size)   BATCH_SIZE="$2";  shift 2 ;;
        --parallel)     PARALLEL="$2";    shift 2 ;;
        --registry)     REGISTRY="$2";    shift 2 ;;
        --benchmarks)   BENCHMARKS="$2";  shift 2 ;;
        --checkpoints)  CHECKPOINTS="$2"; shift 2 ;;
        --force)        FORCE=true;       shift   ;;
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

echo "======================================================"
echo "Checkpoints: $CKPT_LABELS"
echo "Benchmarks:  $BENCHMARKS"
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
FORCE_FLAG=""
if $FORCE; then FORCE_FLAG="--force"; fi

if [[ "$PARALLEL" -le 1 ]]; then
    for JOB in "${JOBS[@]}"; do
        IFS='|' read -r LABEL PATH BENCH <<< "$JOB"
        python3 "$SCRIPT_DIR/run_single.py" \
            --ckpt-label "$LABEL" \
            --ckpt-path  "$PATH" \
            --benchmark  "$BENCH" \
            --config-dir "$CONFIG_DIR" \
            --output-dir "$OUTPUT_DIR" \
            --gpu        "$GPU" \
            --batch-size "$BATCH_SIZE" \
            $FORCE_FLAG
    done
else
    # Parallel mode: distribute jobs across GPUs 0..(PARALLEL-1)
    declare -A PIDS
    JOB_IDX=0
    for JOB in "${JOBS[@]}"; do
        IFS='|' read -r LABEL PATH BENCH <<< "$JOB"
        GPU_ID=$(( JOB_IDX % PARALLEL ))

        python3 "$SCRIPT_DIR/run_single.py" \
            --ckpt-label "$LABEL" \
            --ckpt-path  "$PATH" \
            --benchmark  "$BENCH" \
            --config-dir "$CONFIG_DIR" \
            --output-dir "$OUTPUT_DIR" \
            --gpu        "$GPU_ID" \
            --batch-size "$BATCH_SIZE" \
            $FORCE_FLAG &

        PIDS[$!]="$LABEL×$BENCH"
        JOB_IDX=$(( JOB_IDX + 1 ))

        # Wait when we hit PARALLEL concurrent jobs
        if (( JOB_IDX % PARALLEL == 0 )); then
            for PID in "${!PIDS[@]}"; do
                wait "$PID" && echo "[done] ${PIDS[$PID]}" \
                            || echo "[FAIL] ${PIDS[$PID]}"
            done
            unset PIDS
            declare -A PIDS
        fi
    done
    # Wait for remaining jobs
    for PID in "${!PIDS[@]}"; do
        wait "$PID" && echo "[done] ${PIDS[$PID]}" || echo "[FAIL] ${PIDS[$PID]}"
    done
fi

echo ""
echo "======================================================"
echo "All jobs complete. Run aggregate.py to build CSV:"
echo "  python scripts/aggregate.py"
echo "======================================================"
