#!/usr/bin/env bash
# Sweep: sequentially run training experiments over layer × lr combinations.
#
# Usage:
#   bash sweep_freeze_layers.sh [--lrs <lr1,lr2,...>] [--skip N] [--freeze-largest] \
#       <layer_spec1> <layer_spec2> ... -- [extra args for ath_run_qwen3-8b.sh]
#
# Examples:
#   # Layer 0, 14, 27 each with 3 LRs (default), then freeze_largest
#   bash sweep_freeze_layers.sh --freeze-largest 27 -- --model Qwen/Qwen3-1.7B
#
#   # Custom LR list
#   bash sweep_freeze_layers.sh --lrs 1e-5,3e-5 0 14 27 -- --model Qwen/Qwen3-1.7B

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/ath_run_qwen3-8b.sh"

LRS="1e-6,3e-6,5e-6"
SKIP=0
FREEZE_LARGEST=false
LAYER_SPECS=()
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
    elif [[ "$arg" == "--freeze-largest" ]]; then
        FREEZE_LARGEST=true
    else
        LAYER_SPECS+=("$arg")
    fi
    i=$((i + 1))
done

if [[ ${#LAYER_SPECS[@]} -eq 0 ]] && ! $FREEZE_LARGEST; then
    echo "Usage: bash sweep_freeze_layers.sh [--lrs 1e-6,3e-6,5e-6] [--skip N] [--freeze-largest] <layers1> [<layers2> ...] -- [training args]"
    echo "  e.g. bash sweep_freeze_layers.sh 0 14 27 -- --model Qwen/Qwen3-1.7B"
    echo "  e.g. bash sweep_freeze_layers.sh --freeze-largest 27 -- --model Qwen/Qwen3-1.7B"
    exit 1
fi

IFS=',' read -ra LR_LIST <<< "$LRS"
LAYER_TOTAL=$(( ${#LAYER_SPECS[@]} * ${#LR_LIST[@]} ))
TOTAL=$(( LAYER_TOTAL + ( $FREEZE_LARGEST && echo 1 || echo 0 ) ))
if $FREEZE_LARGEST; then TOTAL=$(( LAYER_TOTAL + 1 )); else TOTAL=$LAYER_TOTAL; fi

run_sweep() {
    local n=1
    for LR in "${LR_LIST[@]}"; do
        for LAYERS in "${LAYER_SPECS[@]}"; do
            if [[ $n -le $SKIP ]]; then
                echo "  [$n/$TOTAL] Skipping layer(s)=$LAYERS lr=$LR"
                n=$((n + 1))
                continue
            fi

            echo "========================================"
            echo "  [$n/$TOTAL] layer(s)=$LAYERS  lr=$LR"
            echo "========================================"

            bash "$TRAIN_SCRIPT" \
                --train-layer-ids "$LAYERS" \
                --freeze-layers 0 \
                --lr "$LR" \
                --no-tmux \
                "${PASSTHROUGH_ARGS[@]}"

            echo "  [$n/$TOTAL] layer(s)=$LAYERS lr=$LR finished."
            echo ""
            n=$((n + 1))
        done
    done

    if $FREEZE_LARGEST; then
        if [[ $n -le $SKIP ]]; then
            echo "  [$n/$TOTAL] Skipping freeze_largest"
        else
            echo "========================================"
            echo "  [$n/$TOTAL] freeze_largest=True"
            echo "========================================"

            bash "$TRAIN_SCRIPT" \
                --no-tmux \
                --note freezeL95 \
                "${PASSTHROUGH_ARGS[@]}" \
                '+actor_rollout_ref.actor.freeze_largest=True'

            echo "  [$n/$TOTAL] freeze_largest finished."
        fi
    fi

    echo "All $TOTAL sweep experiments completed."
}

# If not inside tmux, launch a tmux session for the whole sweep
if [[ -z "${TMUX:-}" ]]; then
    TMUX_SESSION="sweep_freeze_$(date +%m%d_%H%M)"
    FULL_ARGS=""
    [[ "$LRS" != "1e-6,3e-6,5e-6" ]] && FULL_ARGS="$FULL_ARGS --lrs $(printf '%q' "$LRS")"
    [[ $SKIP -gt 0 ]] && FULL_ARGS="$FULL_ARGS --skip $SKIP"
    $FREEZE_LARGEST && FULL_ARGS="$FULL_ARGS --freeze-largest"
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
