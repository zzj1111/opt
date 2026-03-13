#!/usr/bin/env bash
# Sweep: sparse_train_k experiments on layer 14 with fixed LR.
#
# Usage:
#   bash sweep_sparse_k.sh [--skip N] -- [extra args for ath_run_qwen3-8b.sh]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/ath_run_qwen3-8b.sh"

SPARSE_KS=(20000 50000 100000)
LR="1e-4"
LAYER="14"
SKIP=0

# Parse args
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
    elif [[ "$arg" == "--skip" ]]; then
        i=$((i + 1)); SKIP="${args[$i]}"
    fi
    i=$((i + 1))
done

TOTAL=${#SPARSE_KS[@]}

run_sweep() {
    local n=1
    for K in "${SPARSE_KS[@]}"; do
        if [[ $n -le $SKIP ]]; then
            echo "  [$n/$TOTAL] Skipping sparse_k=$K"
            n=$((n + 1))
            continue
        fi

        echo "========================================"
        echo "  [$n/$TOTAL] layer=$LAYER  sparse_k=$K  lr=$LR"
        echo "========================================"

        bash "$TRAIN_SCRIPT" \
            --train-layer-ids "$LAYER" \
            --freeze-layers 0 \
            --sparse-k "$K" \
            --lr "$LR" \
            --no-tmux \
            "${PASSTHROUGH_ARGS[@]}"

        echo "  [$n/$TOTAL] sparse_k=$K finished."
        echo ""
        n=$((n + 1))
    done

    echo "All $TOTAL sparse_k sweep experiments completed."
}

# If not inside tmux, launch a tmux session
if [[ -z "${TMUX:-}" ]]; then
    TMUX_SESSION="sweep_sparse_$(date +%m%d_%H%M)"
    FULL_ARGS=""
    [[ $SKIP -gt 0 ]] && FULL_ARGS="$FULL_ARGS --skip $SKIP"
    if [[ ${#PASSTHROUGH_ARGS[@]} -gt 0 ]]; then
        FULL_ARGS="$FULL_ARGS --"
        for arg in "${PASSTHROUGH_ARGS[@]}"; do FULL_ARGS="$FULL_ARGS $(printf '%q' "$arg")"; done
    fi

    tmux new-session -d -s "$TMUX_SESSION" \
        "source /code/hongpaul-sandbox/cuda/miniconda3/bin/activate && \
         conda activate /code/hongpaul-sandbox/cuda/miniconda3/envs/cuda && \
         cd $(pwd) && \
         bash $SCRIPT_DIR/sweep_sparse_k.sh $FULL_ARGS; \
         exec bash"
    echo "Tmux session '$TMUX_SESSION' started. Attach with: tmux attach -t $TMUX_SESSION"
    exit 0
fi

run_sweep
