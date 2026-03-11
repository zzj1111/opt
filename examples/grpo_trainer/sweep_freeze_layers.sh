#!/usr/bin/env bash
# Sweep: sequentially run training experiments, each training specified layers.
# Each positional argument defines one experiment's layer IDs (comma-separated for multi-layer).
#
# Usage:
#   bash sweep_freeze_layers.sh <layer_spec1> <layer_spec2> ... -- [extra args for ath_run_qwen3-8b.sh]
#
# Examples:
#   # Train layer 0, then layer 14, then layer 27
#   bash sweep_freeze_layers.sh 0 14 27 -- --model Qwen/Qwen3-1.7B --lr 1e-6
#
#   # Train layers 0,1 together, then layers 26,27 together
#   bash sweep_freeze_layers.sh 0,1 26,27 -- --model Qwen/Qwen3-1.7B --lr 1e-6

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/ath_run_qwen3-8b.sh"

# Split args by "--": before is layer specs, after is passthrough args
LAYER_SPECS=()
PASSTHROUGH_ARGS=()
SEEN_SEP=false
for arg in "$@"; do
    if [[ "$arg" == "--" ]]; then
        SEEN_SEP=true
        continue
    fi
    if $SEEN_SEP; then
        PASSTHROUGH_ARGS+=("$arg")
    else
        LAYER_SPECS+=("$arg")
    fi
done

if [[ ${#LAYER_SPECS[@]} -eq 0 ]]; then
    echo "Error: specify at least one layer spec."
    echo "Usage: bash sweep_freeze_layers.sh <layers1> [<layers2> ...] -- [training args]"
    echo "  e.g. bash sweep_freeze_layers.sh 0 14 27 -- --model Qwen/Qwen3-1.7B"
    exit 1
fi

run_sweep() {
    local i=1
    local total=${#LAYER_SPECS[@]}
    for LAYERS in "${LAYER_SPECS[@]}"; do
        echo "========================================"
        echo "  [$i/$total] Training layer(s): $LAYERS"
        echo "========================================"

        bash "$TRAIN_SCRIPT" \
            --train-layer-ids "$LAYERS" \
            --freeze-layers 0 \
            --no-tmux \
            "${PASSTHROUGH_ARGS[@]}"

        echo ""
        echo "  [$i/$total] Layer(s) $LAYERS finished."
        echo ""
        i=$((i + 1))
    done

    echo "All $total sweep experiments completed."
}

# If not inside tmux, launch a tmux session for the whole sweep
if [[ -z "${TMUX:-}" ]]; then
    TMUX_SESSION="sweep_freeze_$(date +%m%d_%H%M)"
    # Rebuild full command for tmux
    FULL_ARGS=""
    for spec in "${LAYER_SPECS[@]}"; do FULL_ARGS="$FULL_ARGS $(printf '%q' "$spec")"; done
    FULL_ARGS="$FULL_ARGS --"
    for arg in "${PASSTHROUGH_ARGS[@]}"; do FULL_ARGS="$FULL_ARGS $(printf '%q' "$arg")"; done

    tmux new-session -d -s "$TMUX_SESSION" \
        "source /code/hongpaul-sandbox/cuda/miniconda3/bin/activate && \
         conda activate /code/hongpaul-sandbox/cuda/miniconda3/envs/cuda && \
         cd $(pwd) && \
         bash $SCRIPT_DIR/sweep_freeze_layers.sh $FULL_ARGS; \
         exec bash"
    echo "Tmux session '$TMUX_SESSION' started. Attach with: tmux attach -t $TMUX_SESSION"
    exit 0
fi

run_sweep
