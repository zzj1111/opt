#!/bin/bash
# =============================================================================
# Hyperparameter Sweep Script
# Runs experiments sequentially in a single tmux session.
# Designed for servers that reclaim idle nodes after 3 hours.
#
# Usage:
#   bash examples/grpo_trainer/sweep.sh
#
# The script will:
#   1. Launch a tmux session "sweep"
#   2. Run each parameter combination one after another
#   3. Keep the GPU busy so the node won't be reclaimed
# =============================================================================

set -x

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJ_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)

# ===================== EDIT YOUR SWEEP GRID HERE =====================
GPUS="0,1,2,3"
MODEL="Qwen/Qwen3-1.7B"
NOTE="sweep"

# Parameter grid (space-separated values)
LR_LIST="1e-7 1e-6 1e-5"
BETA1_LIST="0.9"
BETA2_LIST="0.95 0.999"
# =====================================================================

# Build the list of all experiments
EXPERIMENTS=()
for LR in $LR_LIST; do
    for BETA1 in $BETA1_LIST; do
        for BETA2 in $BETA2_LIST; do
            EXPERIMENTS+=("$LR $BETA1 $BETA2")
        done
    done
done

TOTAL=${#EXPERIMENTS[@]}
echo "Total experiments: $TOTAL"
for i in "${!EXPERIMENTS[@]}"; do
    read -r LR BETA1 BETA2 <<< "${EXPERIMENTS[$i]}"
    echo "  [$((i+1))/$TOTAL] lr=$LR beta1=$BETA1 beta2=$BETA2"
done

# If not inside tmux, launch a tmux session and re-run inside it
if [[ -z "$TMUX" ]]; then
    tmux new-session -d -s "sweep" \
        "source /code/hongpaul-sandbox/cuda/miniconda3/bin/activate && \
         conda activate /code/hongpaul-sandbox/cuda/miniconda3/envs/cuda && \
         cd $PROJ_DIR && \
         bash $SCRIPT_DIR/sweep.sh; \
         exec bash"
    echo "Sweep started in tmux session 'sweep'. Attach with: tmux attach -t sweep"
    exit 0
fi

# Run experiments sequentially
for i in "${!EXPERIMENTS[@]}"; do
    read -r LR BETA1 BETA2 <<< "${EXPERIMENTS[$i]}"
    echo ""
    echo "=============================================="
    echo " Experiment [$((i+1))/$TOTAL]"
    echo " lr=$LR  beta1=$BETA1  beta2=$BETA2"
    echo "=============================================="

    # Run directly (already inside tmux with conda activated)
    bash "$SCRIPT_DIR/run_qwen3-8b.sh" \
        --gpus "$GPUS" \
        --model "$MODEL" \
        --lr "$LR" \
        --beta1 "$BETA1" \
        --beta2 "$BETA2" \
        --note "$NOTE"

    echo ""
    echo "[Experiment $((i+1))/$TOTAL finished with exit code $?]"
    echo ""
done

echo "=============================================="
echo " All $TOTAL experiments completed!"
echo "=============================================="
