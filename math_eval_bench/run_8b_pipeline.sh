#!/bin/bash
# ==============================================================================
# Qwen3-8B-Base end-to-end pipeline:
#   1. EVAL all *Qwen3-8B-Base* checkpoints (and the un-trained baseline).
#   2. ANALYSE results -> derive top-N / worst-N layers from math_avg ranking.
#   3. TRAIN boost+only sweep using the auto-derived layer sets.
#   4. Loops the boost ckpts back into eval after the sweep finishes (because
#      run_8b_base_simple_boost_only_tmux.sh has its own dummy LR loop, the GPUs
#      stay busy until you kill the tmux session).
#
# Each step skips itself if the work is already done (matches existing dirs +
# wandb runs), so re-running this script is safe.
#
# Usage:
#   bash run_8b_pipeline.sh                  # run the whole pipeline
#   bash run_8b_pipeline.sh --no-eval        # skip step 1 (assume eval done)
#   bash run_8b_pipeline.sh --no-train       # skip step 3 (just analyze)
#   bash run_8b_pipeline.sh --eval-args "--no-base --max-tokens 3072"
#   bash run_8b_pipeline.sh --train-args "--no-dummy --only 1,2,3"
#   bash run_8b_pipeline.sh --no-tmux        # already inside a tmux session

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TRAIN_DIR="$PROJ_DIR/examples/grpo_trainer"

EVAL_SCRIPT="$SCRIPT_DIR/run_eval_8b_math.sh"
TRAIN_SCRIPT="$TRAIN_DIR/run_8b_base_simple_boost_only_tmux.sh"
ANALYZE_SCRIPT="$SCRIPT_DIR/analyze_8b_layer_ranking.py"

CONDA_INIT="${CONDA_INIT:-/code/hongpaul-sandbox/cuda/miniconda3/bin/activate}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/code/hongpaul-sandbox/cuda/miniconda3/envs/cuda}"

NO_EVAL=false; NO_TRAIN=false; NO_TMUX=false
EVAL_EXTRA=""; TRAIN_EXTRA=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-eval)    NO_EVAL=true; shift ;;
        --no-train)   NO_TRAIN=true; shift ;;
        --no-tmux)    NO_TMUX=true; shift ;;
        --eval-args)  EVAL_EXTRA="$2"; shift 2 ;;
        --train-args) TRAIN_EXTRA="$2"; shift 2 ;;
        *)            echo "unknown arg: $1"; exit 1 ;;
    esac
done

# ---------------- tmux auto-launch ----------------
if [[ -z "${TMUX:-}" ]] && [[ "$NO_TMUX" == "false" ]]; then
    SESSION="8b_pipeline_$(date +%m%d_%H%M)"
    FULL_ARGS="--no-tmux"
    $NO_EVAL  && FULL_ARGS="$FULL_ARGS --no-eval"
    $NO_TRAIN && FULL_ARGS="$FULL_ARGS --no-train"
    [[ -n "$EVAL_EXTRA"  ]] && FULL_ARGS="$FULL_ARGS --eval-args $(printf '%q' "$EVAL_EXTRA")"
    [[ -n "$TRAIN_EXTRA" ]] && FULL_ARGS="$FULL_ARGS --train-args $(printf '%q' "$TRAIN_EXTRA")"
    tmux new-session -d -s "$SESSION" \
        "source $CONDA_INIT && conda activate $CONDA_ENV_PATH && \
         cd $SCRIPT_DIR && \
         bash $SCRIPT_DIR/$(basename "$0") $FULL_ARGS; exec bash"
    echo "Tmux '$SESSION' launched. Attach: tmux attach -t $SESSION"
    exit 0
fi

# Conda is already active inside tmux; activate manually if not (--no-tmux from outside).
if ! command -v python3 >/dev/null 2>&1 || ! python3 -c "import vllm" >/dev/null 2>&1; then
    [[ -f "$CONDA_INIT" ]] && source "$CONDA_INIT" && conda activate "$CONDA_ENV_PATH"
fi

# ============================ STEP 1: EVAL ============================
if $NO_EVAL; then
    echo ""
    echo "[1/3] EVAL skipped (--no-eval)."
else
    echo ""
    echo "============================================================"
    echo "[1/3] EVAL — running $EVAL_SCRIPT --no-tmux $EVAL_EXTRA"
    echo "============================================================"
    # Run synchronously; eval script's own worker pool keeps GPUs full.
    bash "$EVAL_SCRIPT" --no-tmux $EVAL_EXTRA
    eval_rc=$?
    if [[ $eval_rc -ne 0 ]]; then
        echo "[1/3] EVAL exited with status $eval_rc, continuing to analysis anyway."
    fi
fi

# ============================ STEP 2: ANALYSE ============================
echo ""
echo "============================================================"
echo "[2/3] ANALYSE — ranking layers from local eval results"
echo "============================================================"
RANKING_OUT=$(python3 "$ANALYZE_SCRIPT" --csv /tmp/8b_layer_ranking.csv 2>&1 1>/tmp/8b_layer_ranking.sh)
ANALYZE_RC=$?
# Print the human-readable summary (stderr) to console
echo "$RANKING_OUT"

if [[ $ANALYZE_RC -ne 0 ]]; then
    echo "[2/3] ANALYSE failed (no usable eval results). Aborting before training."
    exit 1
fi
# Source the assignments produced on stdout
source /tmp/8b_layer_ranking.sh
echo ""
echo "  Derived layer sets:"
echo "    TOP1=$TOP1   TOP5=$TOP5   TOP10=$TOP10"
echo "    WORST1=$WORST1   WORST5=$WORST5   WORST10=$WORST10"

# ============================ STEP 3: TRAIN ============================
if $NO_TRAIN; then
    echo ""
    echo "[3/3] TRAIN skipped (--no-train). Pipeline complete after analysis."
    exit 0
fi

if [[ -z "$TOP5" || -z "$TOP10" || -z "$WORST5" || -z "$WORST10" ]]; then
    echo "[3/3] TRAIN aborted — analysis didn't produce all required layer sets."
    echo "  Make sure layer-trained 8B ckpts exist and have eval results."
    exit 1
fi

echo ""
echo "============================================================"
echo "[3/3] TRAIN — boost+only sweep with auto-derived layer sets"
echo "  Will also kick off dummy LR loop after the 8 main exps."
echo "  Pass --train-args '--no-dummy' to skip the loop."
echo "============================================================"
bash "$TRAIN_SCRIPT" --no-tmux \
    --top5    "$TOP5" \
    --top10   "$TOP10" \
    --worst5  "$WORST5" \
    --worst10 "$WORST10" \
    $TRAIN_EXTRA
echo ""
echo "Pipeline finished."
