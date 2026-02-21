#!/bin/bash
# =============================================================================
# Hyperparameter Sweep Script
# Runs experiments sequentially in a single tmux session.
# All experiments are placed under a single sweep directory for easy download.
#
# Usage:
#   bash examples/grpo_trainer/sweep.sh
#
# The script will:
#   1. Create a sweep directory: checkpoints/sweep_MMDD/
#   2. Launch a tmux session "sweep"
#   3. Run each parameter combination, saving to sweep_MMDD/<exp>/
# =============================================================================

set -x

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJ_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)

# ===================== EDIT YOUR SWEEP GRID HERE =====================
GPUS="0,1,2,3,4,5,6,7"
MODEL="Qwen/Qwen3-1.7B"
NOTE="sweep"

# --- AdamW grid ---
ADAMW_LR_LIST="1e-7 1e-6 1e-5"
ADAMW_BETA1_LIST="0.9"
ADAMW_BETA2_LIST="0.95 0.999"

# --- SGD grid ---
SGD_LR_LIST="0.01 0.1 1"
SGD_MOMENTUM_LIST="0"
# =====================================================================

# Create sweep root directory
DATE=$(date +%m%d)
SWEEP_NUM=1
while true; do
    SWEEP_DIR="checkpoints/sweep_${DATE}_${SWEEP_NUM}"
    if mkdir "$SWEEP_DIR" 2>/dev/null; then
        break
    fi
    SWEEP_NUM=$((SWEEP_NUM + 1))
done
echo "Sweep directory: $SWEEP_DIR"

# Build the list of all experiments: "OPTIM LR PARAM1 PARAM2"
EXPERIMENTS=()

# SGD experiments (run first)
for LR in $SGD_LR_LIST; do
    for MU in $SGD_MOMENTUM_LIST; do
        EXPERIMENTS+=("sgd $LR $MU 0")
    done
done

# AdamW experiments
for LR in $ADAMW_LR_LIST; do
    for BETA1 in $ADAMW_BETA1_LIST; do
        for BETA2 in $ADAMW_BETA2_LIST; do
            EXPERIMENTS+=("adamw $LR $BETA1 $BETA2")
        done
    done
done

TOTAL=${#EXPERIMENTS[@]}
echo "Total experiments: $TOTAL"
for i in "${!EXPERIMENTS[@]}"; do
    read -r OPTIM LR P1 P2 <<< "${EXPERIMENTS[$i]}"
    if [[ "$OPTIM" == "adamw" ]]; then
        echo "  [$((i+1))/$TOTAL] adamw lr=$LR beta1=$P1 beta2=$P2"
    else
        echo "  [$((i+1))/$TOTAL] sgd   lr=$LR momentum=$P1"
    fi
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
    read -r OPTIM LR P1 P2 <<< "${EXPERIMENTS[$i]}"
    echo ""
    echo "=============================================="
    echo " Experiment [$((i+1))/$TOTAL]"

    if [[ "$OPTIM" == "adamw" ]]; then
        echo " adamw  lr=$LR  beta1=$P1  beta2=$P2"
        echo "=============================================="
        bash "$SCRIPT_DIR/run_qwen3-8b.sh" \
            --gpus "$GPUS" \
            --model "$MODEL" \
            --optim adamw \
            --lr "$LR" \
            --beta1 "$P1" \
            --beta2 "$P2" \
            --ckpt-root "$SWEEP_DIR" \
            --note "$NOTE"
    else
        echo " sgd    lr=$LR  momentum=$P1"
        echo "=============================================="
        bash "$SCRIPT_DIR/run_qwen3-8b.sh" \
            --gpus "$GPUS" \
            --model "$MODEL" \
            --optim sgd \
            --lr "$LR" \
            --momentum "$P1" \
            --ckpt-root "$SWEEP_DIR" \
            --note "$NOTE"
    fi

    echo ""
    echo "[Experiment $((i+1))/$TOTAL finished with exit code $?]"
    echo ""
done

echo "=============================================="
echo " All $TOTAL experiments completed!"
echo " Results in: $SWEEP_DIR"
echo "=============================================="
