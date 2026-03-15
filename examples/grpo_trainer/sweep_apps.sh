#!/usr/bin/env bash
# Sweep: run APPS training with multiple LRs on layer 13, then dummy GPU hold.
#
# Usage:
#   bash sweep_apps.sh [--lrs <lr1,lr2,...>] [--skip N] [--train-layer-ids IDS] -- [extra args]
#
# Examples:
#   bash sweep_apps.sh -- --model Qwen/Qwen3-1.7B
#   bash sweep_apps.sh --lrs 1e-5,2e-5 --train-layer-ids 13 -- --model Qwen/Qwen3-1.7B

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/run_apps.sh"

LRS="5e-6,1e-5,2e-5"
SKIP=0
TRAIN_LAYER_IDS="13"
PASSTHROUGH_ARGS=()
SEEN_SEP=false
i=0
args=("$@")
while [[ $i -lt ${#args[@]} ]]; do
    arg="${args[$i]}"
    if [[ "$arg" == "--" ]]; then
        SEEN_SEP=true
        i=$((i + 1))
        continue
    fi
    if $SEEN_SEP; then
        PASSTHROUGH_ARGS+=("$arg")
    elif [[ "$arg" == "--lrs" ]]; then
        i=$((i + 1)); LRS="${args[$i]}"
    elif [[ "$arg" == "--skip" ]]; then
        i=$((i + 1)); SKIP="${args[$i]}"
    elif [[ "$arg" == "--train-layer-ids" ]]; then
        i=$((i + 1)); TRAIN_LAYER_IDS="${args[$i]}"
    else
        PASSTHROUGH_ARGS+=("$arg")
    fi
    i=$((i + 1))
done

IFS=',' read -ra LR_LIST <<< "$LRS"
TOTAL=${#LR_LIST[@]}

run_sweep() {
    local n=1
    for LR in "${LR_LIST[@]}"; do
        if [[ $n -le $SKIP ]]; then
            echo "  [$n/$TOTAL] Skipping lr=$LR"
            n=$((n + 1))
            continue
        fi

        echo "========================================"
        echo "  [$n/$TOTAL] APPS layer=$TRAIN_LAYER_IDS lr=$LR"
        echo "========================================"

        bash "$TRAIN_SCRIPT" \
            --train-layer-ids "$TRAIN_LAYER_IDS" \
            --lr "$LR" \
            --no-tmux \
            "${PASSTHROUGH_ARGS[@]}"

        echo "  [$n/$TOTAL] lr=$LR finished."
        echo ""
        n=$((n + 1))
    done

    echo "All $TOTAL APPS sweep experiments completed."
    echo "========================================"
    echo "  Starting dummy GPU hold job..."
    echo "========================================"
    DUMMY_RUN_NAME="dummy_$(hostname)_$(date +%m%d_%H%M)" \
        python "$SCRIPT_DIR/dummy_gpu_hold.py"
}

# If not inside tmux, launch a tmux session
if [[ -z "${TMUX:-}" ]]; then
    TMUX_SESSION="sweep_apps_$(date +%m%d_%H%M)"
    FULL_ARGS=""
    [[ "$LRS" != "5e-6,1e-5,2e-5" ]] && FULL_ARGS="$FULL_ARGS --lrs $(printf '%q' "$LRS")"
    [[ $SKIP -gt 0 ]] && FULL_ARGS="$FULL_ARGS --skip $SKIP"
    [[ "$TRAIN_LAYER_IDS" != "13" ]] && FULL_ARGS="$FULL_ARGS --train-layer-ids $(printf '%q' "$TRAIN_LAYER_IDS")"
    FULL_ARGS="$FULL_ARGS --"
    for arg in "${PASSTHROUGH_ARGS[@]}"; do FULL_ARGS="$FULL_ARGS $(printf '%q' "$arg")"; done

    tmux new-session -d -s "$TMUX_SESSION" \
        "source /code/hongpaul-sandbox/cuda/miniconda3/bin/activate && \
         conda activate /code/hongpaul-sandbox/cuda/miniconda3/envs/cuda && \
         cd $(pwd) && \
         bash $SCRIPT_DIR/sweep_apps.sh $FULL_ARGS; \
         exec bash"
    echo "Tmux session '$TMUX_SESSION' started. Attach with: tmux attach -t $TMUX_SESSION"
    exit 0
fi

run_sweep
