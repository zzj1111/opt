#!/usr/bin/env bash
# Sweep freeze_largest with multiple learning rates.
#
# Usage:
#   bash sweep_freeze_largest_lr.sh [--lrs <lr1,lr2,...>] [--skip N] -- [extra args]
#
# Examples:
#   bash sweep_freeze_largest_lr.sh --lrs 3e-6,5e-6 -- --model Qwen/Qwen3-1.7B
#   bash sweep_freeze_largest_lr.sh -- --model Qwen/Qwen3-1.7B   # default: 1e-6,3e-6,5e-6

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/ath_run_qwen3-8b.sh"

LRS="1e-6,3e-6,5e-6"
SKIP=0
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
    else
        echo "Unknown arg: $arg"
        exit 1
    fi
    i=$((i + 1))
done

IFS=',' read -ra LR_LIST <<< "$LRS"
TOTAL=${#LR_LIST[@]}

run_sweep() {
    local n=1
    for LR in "${LR_LIST[@]}"; do
        if [[ $n -le $SKIP ]]; then
            echo "  [$n/$TOTAL] Skipping freeze_largest lr=$LR"
            n=$((n + 1))
            continue
        fi

        echo "========================================"
        echo "  [$n/$TOTAL] freeze_largest  lr=$LR"
        echo "========================================"

        bash "$TRAIN_SCRIPT" \
            --freeze-layers 0 \
            --lr "$LR" \
            --note freezeL97 \
            --no-tmux \
            "${PASSTHROUGH_ARGS[@]}" \
            '+actor_rollout_ref.actor.freeze_largest=True'

        echo "  [$n/$TOTAL] freeze_largest lr=$LR finished."
        echo ""
        n=$((n + 1))
    done

    echo "All $TOTAL sweep experiments completed."
}

# If not inside tmux, launch a tmux session
if [[ -z "${TMUX:-}" ]]; then
    TMUX_SESSION="sweep_freezeL_$(date +%m%d_%H%M)"
    FULL_ARGS=""
    [[ "$LRS" != "1e-6,3e-6,5e-6" ]] && FULL_ARGS="$FULL_ARGS --lrs $(printf '%q' "$LRS")"
    [[ $SKIP -gt 0 ]] && FULL_ARGS="$FULL_ARGS --skip $SKIP"
    FULL_ARGS="$FULL_ARGS --"
    for arg in "${PASSTHROUGH_ARGS[@]}"; do FULL_ARGS="$FULL_ARGS $(printf '%q' "$arg")"; done

    tmux new-session -d -s "$TMUX_SESSION" \
        "source /code/hongpaul-sandbox/cuda/miniconda3/bin/activate && \
         conda activate /code/hongpaul-sandbox/cuda/miniconda3/envs/cuda && \
         cd $(pwd) && \
         bash $SCRIPT_DIR/sweep_freeze_largest_lr.sh $FULL_ARGS; \
         exec bash"
    echo "Tmux session '$TMUX_SESSION' started. Attach with: tmux attach -t $TMUX_SESSION"
    exit 0
fi

run_sweep
