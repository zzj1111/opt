#!/usr/bin/env bash
# Sweep: run a list of (sparse_k, lr) experiments on layer 14.
#
# Each experiment is a "K:LR" pair. Runs sequentially.
#
# Usage:
#   bash sweep_sparse_k_v2.sh [--skip N] [--layer L] <K1:LR1> <K2:LR2> ... -- [extra args]
#
# Examples:
#   bash sweep_sparse_k_v2.sh 20000:1e-3 20000:3e-3 10000:2e-3 10000:6e-3 -- --model Qwen/Qwen3-1.7B

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/ath_run_qwen3-8b.sh"

LAYER="14"
SKIP=0
EXPERIMENTS=()
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
    elif [[ "$arg" == "--layer" ]]; then
        i=$((i + 1)); LAYER="${args[$i]}"
    else
        EXPERIMENTS+=("$arg")
    fi
    i=$((i + 1))
done

if [[ ${#EXPERIMENTS[@]} -eq 0 ]]; then
    echo "Usage: bash sweep_sparse_k_v2.sh [--skip N] [--layer L] <K:LR> [<K:LR> ...] -- [extra args]"
    echo "  e.g. bash sweep_sparse_k_v2.sh 20000:1e-3 20000:3e-3 10000:2e-3 10000:6e-3 -- --model Qwen/Qwen3-1.7B"
    exit 1
fi

TOTAL=${#EXPERIMENTS[@]}

run_sweep() {
    local n=1
    for EXP in "${EXPERIMENTS[@]}"; do
        K="${EXP%%:*}"
        LR="${EXP##*:}"

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

    echo "All $TOTAL experiments completed."
    echo "========================================"
    echo "  Starting dummy GPU hold job..."
    echo "========================================"
    DUMMY_RUN_NAME="dummy_$(hostname)_$(date +%m%d_%H%M)" \
        python "$SCRIPT_DIR/dummy_gpu_hold.py"
}

# If not inside tmux, launch a tmux session
if [[ -z "${TMUX:-}" ]]; then
    TMUX_SESSION="sweep_spk2_$(date +%m%d_%H%M)"
    FULL_ARGS=""
    [[ $SKIP -gt 0 ]] && FULL_ARGS="$FULL_ARGS --skip $SKIP"
    [[ "$LAYER" != "14" ]] && FULL_ARGS="$FULL_ARGS --layer $LAYER"
    for exp in "${EXPERIMENTS[@]}"; do FULL_ARGS="$FULL_ARGS $(printf '%q' "$exp")"; done
    if [[ ${#PASSTHROUGH_ARGS[@]} -gt 0 ]]; then
        FULL_ARGS="$FULL_ARGS --"
        for arg in "${PASSTHROUGH_ARGS[@]}"; do FULL_ARGS="$FULL_ARGS $(printf '%q' "$arg")"; done
    fi

    tmux new-session -d -s "$TMUX_SESSION" \
        "source /code/hongpaul-sandbox/cuda/miniconda3/bin/activate && \
         conda activate /code/hongpaul-sandbox/cuda/miniconda3/envs/cuda && \
         cd $(pwd) && \
         bash $SCRIPT_DIR/sweep_sparse_k_v2.sh $FULL_ARGS; \
         exec bash"
    echo "Tmux session '$TMUX_SESSION' started. Attach with: tmux attach -t $TMUX_SESSION"
    exit 0
fi

run_sweep
