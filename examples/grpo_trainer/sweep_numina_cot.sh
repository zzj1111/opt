#!/bin/bash
# =============================================================================
# Hyperparameter Sweep Script for NuminaMath-CoT GRPO Training
# Runs experiments sequentially in a single tmux session.
# All experiments are placed under a single sweep directory for easy download.
#
# Usage:
#   bash examples/grpo_trainer/sweep_numina_cot.sh [--tmux SESSION_NAME] [--note NOTE] [--model-dtype fp32|bf16]
#
# The script will:
#   1. Create a sweep directory: checkpoints/sweep_numina_MMDD/
#   2. Launch a tmux session "sweep_numina"
#   3. Run each parameter combination, saving to sweep_numina_MMDD/<exp>/
# =============================================================================

set -x

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJ_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)

# Parse CLI args
TMUX_NAME="sweep_numina"
NOTE="sweep"
MODEL_DTYPE="fp32"
while [[ $# -gt 0 ]]; do
    case $1 in
        --tmux) TMUX_NAME="$2"; shift 2 ;;
        --note) NOTE="$2"; shift 2 ;;
        --model-dtype) MODEL_DTYPE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ===================== EDIT YOUR SWEEP GRID HERE =====================
GPUS="0,1,2,3,4,5,6,7"
MODEL="Qwen/Qwen3-1.7B"
DATA_DIR="$PROJ_DIR/data/numina_math_cot_author"

# --- AdamW grid ---
ADAMW_LR_LIST="1e-6"
ADAMW_BETA1_LIST="0.9"
ADAMW_BETA2_LIST="0.999"

# --- SGD grid ---
SGD_LR_LIST="0.1 0.01"
SGD_MOMENTUM_LIST="0"

# --- Rollout grid (applied to all optimizers) ---
ROLLOUT_N_LIST="5"
# =====================================================================

# Create sweep root directory
DATE=$(date +%m%d)
SWEEP_NUM=1
while true; do
    SWEEP_DIR="checkpoints/sweep_numina_${DATE}_${SWEEP_NUM}"
    if mkdir "$SWEEP_DIR" 2>/dev/null; then
        break
    fi
    SWEEP_NUM=$((SWEEP_NUM + 1))
done
echo "Sweep directory: $SWEEP_DIR"

# Build the list of all experiments: "OPTIM LR PARAM1 PARAM2 ROLLOUT_N"
EXPERIMENTS=()

# AdamW experiments
for LR in $ADAMW_LR_LIST; do
    for BETA1 in $ADAMW_BETA1_LIST; do
        for BETA2 in $ADAMW_BETA2_LIST; do
            for RN in $ROLLOUT_N_LIST; do
                EXPERIMENTS+=("adamw $LR $BETA1 $BETA2 $RN")
            done
        done
    done
done

# SGD experiments
for LR in $SGD_LR_LIST; do
    for MU in $SGD_MOMENTUM_LIST; do
        for RN in $ROLLOUT_N_LIST; do
            EXPERIMENTS+=("sgd $LR $MU 0 $RN")
        done
    done
done

TOTAL=${#EXPERIMENTS[@]}
echo "Total experiments: $TOTAL"
for i in "${!EXPERIMENTS[@]}"; do
    read -r OPTIM LR P1 P2 RN <<< "${EXPERIMENTS[$i]}"
    if [[ "$OPTIM" == "adamw" ]]; then
        echo "  [$((i+1))/$TOTAL] adamw lr=$LR beta1=$P1 beta2=$P2 rollout.n=$RN"
    else
        echo "  [$((i+1))/$TOTAL] sgd   lr=$LR momentum=$P1 rollout.n=$RN"
    fi
done

# If not inside tmux, launch a tmux session and re-run inside it
if [[ -z "$TMUX" ]]; then
    tmux new-session -d -s "$TMUX_NAME" \
        "source /code/hongpaul-sandbox/cuda/miniconda3/bin/activate && \
         conda activate /code/hongpaul-sandbox/cuda/miniconda3/envs/cuda && \
         cd $PROJ_DIR && \
         bash $SCRIPT_DIR/sweep_numina_cot.sh --tmux $TMUX_NAME --note $NOTE --model-dtype $MODEL_DTYPE; \
         exec bash"
    echo "Sweep started in tmux session '$TMUX_NAME'. Attach with: tmux attach -t $TMUX_NAME"
    exit 0
fi

# Run experiments sequentially
for i in "${!EXPERIMENTS[@]}"; do
    read -r OPTIM LR P1 P2 RN <<< "${EXPERIMENTS[$i]}"
    echo ""
    echo "=============================================="
    echo " Experiment [$((i+1))/$TOTAL]"

    if [[ "$OPTIM" == "adamw" ]]; then
        echo " adamw  lr=$LR  beta1=$P1  beta2=$P2  rollout.n=$RN"
        echo "=============================================="
        bash "$SCRIPT_DIR/run_qwen3-1.7b_numina_cot.sh" \
            --gpus "$GPUS" \
            --model "$MODEL" \
            --optim adamw \
            --lr "$LR" \
            --beta1 "$P1" \
            --beta2 "$P2" \
            --rollout-n "$RN" \
            --ckpt-root "$SWEEP_DIR" \
            --data-dir "$DATA_DIR" \
            --model-dtype "$MODEL_DTYPE" \
            --note "$NOTE"
    else
        echo " sgd    lr=$LR  momentum=$P1  rollout.n=$RN"
        echo "=============================================="
        bash "$SCRIPT_DIR/run_qwen3-1.7b_numina_cot.sh" \
            --gpus "$GPUS" \
            --model "$MODEL" \
            --optim sgd \
            --lr "$LR" \
            --momentum "$P1" \
            --rollout-n "$RN" \
            --ckpt-root "$SWEEP_DIR" \
            --data-dir "$DATA_DIR" \
            --model-dtype "$MODEL_DTYPE" \
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
