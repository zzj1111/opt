#!/usr/bin/env bash
# Sweep: sparse_train_k × lr experiments on layer 14.
#
# Usage:
#   bash sweep_sparse_k.sh [--skip N] [--ks 20000,50000] [--lrs 2e-4,3e-4,5e-4] -- [extra args]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/ath_run_qwen3-8b.sh"

KS="20000,50000"
LRS="2e-4,3e-4,5e-4"
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
    elif [[ "$arg" == "--ks" ]]; then
        i=$((i + 1)); KS="${args[$i]}"
    elif [[ "$arg" == "--lrs" ]]; then
        i=$((i + 1)); LRS="${args[$i]}"
    elif [[ "$arg" == "--layer" ]]; then
        i=$((i + 1)); LAYER="${args[$i]}"
    fi
    i=$((i + 1))
done

IFS=',' read -ra K_LIST <<< "$KS"
IFS=',' read -ra LR_LIST <<< "$LRS"
TOTAL=$(( ${#K_LIST[@]} * ${#LR_LIST[@]} ))

run_sweep() {
    local n=1
    for K in "${K_LIST[@]}"; do
        for LR in "${LR_LIST[@]}"; do
            if [[ $n -le $SKIP ]]; then
                echo "  [$n/$TOTAL] Skipping sparse_k=$K lr=$LR"
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

            echo "  [$n/$TOTAL] sparse_k=$K lr=$LR finished."
            echo ""
            n=$((n + 1))
        done
    done

    echo "All $TOTAL sparse_k×lr sweep experiments completed."
    echo "========================================"
    echo "  Starting dummy GPU hold job..."
    echo "========================================"
    DUMMY_RUN_NAME="dummy_$(hostname)_$(date +%m%d_%H%M)" \
        python "$SCRIPT_DIR/dummy_gpu_hold.py"
}

# If not inside tmux, launch a tmux session
if [[ -z "${TMUX:-}" ]]; then
    TMUX_SESSION="sweep_sparse_$(date +%m%d_%H%M)"
    FULL_ARGS=""
    [[ "$KS" != "20000,50000" ]] && FULL_ARGS="$FULL_ARGS --ks $(printf '%q' "$KS")"
    [[ "$LRS" != "2e-4,3e-4,5e-4" ]] && FULL_ARGS="$FULL_ARGS --lrs $(printf '%q' "$LRS")"
    [[ $SKIP -gt 0 ]] && FULL_ARGS="$FULL_ARGS --skip $SKIP"
    [[ "$LAYER" != "14" ]] && FULL_ARGS="$FULL_ARGS --layer $LAYER"
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
