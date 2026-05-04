#!/bin/bash
# ==============================================================================
# Qwen3-1.7B-Base middle-5 single-block RL training on MATH (with tmux)
# ==============================================================================
#
# Trains the 5 middle transformer layers (layers 12-16) of Qwen3-1.7B-Base
# (28 total layers, so 12..16 spans the center). Comparison to a full RL
# baseline at the same LR is provided as Exp 2.
#
# Experiments:
#   1. Layers 12-16 only,   LR=5e-6   (middle-5 layer training)
#   2. Full RL,             LR=1e-6   (baseline, all layers)
#
# All: batch=512, minibatch=128, microbatch=8, epochs=10 (override with EPOCHS=N), max_response_length=3072
# Model: Qwen3-1.7B-Base, 8 GPUs, saves only last-step checkpoint in HF format.
#
# Usage:
#   bash run_math_mid5_qwen3_1.7b_base_tmux.sh                          # Run all
#   bash run_math_mid5_qwen3_1.7b_base_tmux.sh --gpus 0,1,2,3,4,5,6,7  # Specific GPUs
#   bash run_math_mid5_qwen3_1.7b_base_tmux.sh --skip 1                 # Skip exp 1
#   bash run_math_mid5_qwen3_1.7b_base_tmux.sh --only 1                 # Only exp 1
#   bash run_math_mid5_qwen3_1.7b_base_tmux.sh --no-tmux                # No tmux

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJ_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)

# ========== Configuration ==========
MODEL="Qwen/Qwen3-1.7B-Base"
GPUS="0,1,2,3,4,5,6,7"
CKPT_ROOT="$PROJ_DIR/checkpoints"
DATA_DIR="$PROJ_DIR/data/math"
CONDA_INIT="${CONDA_INIT:-/code/hongpaul-sandbox/cuda/miniconda3/bin/activate}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/code/hongpaul-sandbox/cuda/miniconda3/envs/cuda}"
SKIP=0
ONLY=""
NO_TMUX=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)            GPUS="$2"; shift 2 ;;
        --model)           MODEL="$2"; shift 2 ;;
        --ckpt-root)       CKPT_ROOT="$2"; shift 2 ;;
        --data-dir)        DATA_DIR="$2"; shift 2 ;;
        --skip)            SKIP="$2"; shift 2 ;;
        --only)            ONLY="$2"; shift 2 ;;
        --no-tmux)         NO_TMUX=true; shift ;;
        *)                 EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# If not inside tmux, launch a tmux session and re-run inside it
if [[ -z "${TMUX:-}" ]] && [[ "$NO_TMUX" == "false" ]]; then
    TMUX_SESSION="math_mid5_$(date +%m%d_%H%M)"

    FULL_ARGS="--no-tmux"
    FULL_ARGS="$FULL_ARGS --gpus $(printf '%q' "$GPUS")"
    FULL_ARGS="$FULL_ARGS --model $(printf '%q' "$MODEL")"
    FULL_ARGS="$FULL_ARGS --ckpt-root $(printf '%q' "$CKPT_ROOT")"
    FULL_ARGS="$FULL_ARGS --data-dir $(printf '%q' "$DATA_DIR")"
    [[ $SKIP -gt 0 ]] && FULL_ARGS="$FULL_ARGS --skip $SKIP"
    [[ -n "$ONLY" ]] && FULL_ARGS="$FULL_ARGS --only $(printf '%q' "$ONLY")"
    for arg in "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; do FULL_ARGS="$FULL_ARGS $(printf '%q' "$arg")"; done

    tmux new-session -d -s "$TMUX_SESSION" \
        "source $CONDA_INIT && \
         conda activate $CONDA_ENV_PATH && \
         cd $PROJ_DIR && \
         bash $SCRIPT_DIR/run_math_mid5_qwen3_1.7b_base_tmux.sh $FULL_ARGS; \
         exec bash"
    echo "Tmux session '$TMUX_SESSION' started."
    echo "  Attach with:  tmux attach -t $TMUX_SESSION"
    exit 0
fi

NGPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
MODEL_SHORT=$(basename "$MODEL")
DATE=$(date +%m%d_%H%M)

# Validate data
if [[ ! -f "$DATA_DIR/train.parquet" ]]; then
    echo "ERROR: Data not found: $DATA_DIR/train.parquet"
    exit 1
fi

# ========== Shared training function ==========
run_train() {
    local EXP_NAME="$1"
    local DATA_DIR="$2"
    local LR="$3"
    local FREEZE_ARGS="$4"
    local MAX_RESP_LEN="${5:-3072}"
    local BATCH_SIZE="${6:-512}"
    local MINI_BATCH="${7:-128}"
    local MICRO_BATCH="${8:-8}"
    local ROLLOUT_N="${9:-5}"
    local EPOCHS="${10:-${EPOCHS:-10}}"
    local SAVE_FREQ="${11:--1}"

    local STEPS_PER_EPOCH
    STEPS_PER_EPOCH=$(python3 -c "
import pandas as pd
n = len(pd.read_parquet('$DATA_DIR/train.parquet'))
print(n // $BATCH_SIZE)
")
    local TOTAL_STEPS=$((STEPS_PER_EPOCH * EPOCHS))
    if [[ "$SAVE_FREQ" == "-1" ]]; then
        SAVE_FREQ=$TOTAL_STEPS
    fi

    local ROLLOUT_LOG_PROB_MICRO=16
    local REF_LOG_PROB_MICRO=16
    local TP=1

    mkdir -p "$CKPT_ROOT/$EXP_NAME"
    local LOG_FILE="$CKPT_ROOT/$EXP_NAME/train.log"

    echo "  Experiment:  $EXP_NAME"
    echo "  Model:       $MODEL"
    echo "  Data:        $DATA_DIR"
    echo "  LR:          $LR"
    echo "  Freeze:      ${FREEZE_ARGS:-none (full)}"
    echo "  MaxRespLen:  $MAX_RESP_LEN"
    echo "  Batch:       $BATCH_SIZE, MiniBatch: $MINI_BATCH, MicroBatch: $MICRO_BATCH"
    echo "  Epochs:      $EPOCHS, Steps/epoch: $STEPS_PER_EPOCH, Total: $TOTAL_STEPS"
    echo "  SaveFreq:    $SAVE_FREQ"
    echo "  GPUs:        $NGPUS x TP=$TP"
    echo "  Log:         $LOG_FILE"

    export CUDA_VISIBLE_DEVICES=$GPUS
    export WANDB_API_KEY="${WANDB_API_KEY:-b8f38344ec7231ee89baa74ef7209dd5a43df6b2}"
    export WANDB_ENTITY="${WANDB_ENTITY:-mhong-university-of-minnesota}"
    export VERL_DEFAULT_LOCAL_DIR="$CKPT_ROOT/$EXP_NAME"

    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        "data.train_files='$DATA_DIR/train.parquet'" \
        "data.val_files='$DATA_DIR/test.parquet'" \
        data.train_batch_size=$BATCH_SIZE \
        data.max_prompt_length=1024 \
        data.max_response_length=$MAX_RESP_LEN \
        data.filter_overlong_prompts=True \
        "data.truncation='error'" \
        actor_rollout_ref.model.path=$MODEL \
        actor_rollout_ref.actor.optim.lr=$LR \
        "actor_rollout_ref.actor.optim.betas=[0.9,0.999]" \
        "actor_rollout_ref.actor.checkpoint.save_contents='[\"hf_model\"]'" \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$ROLLOUT_LOG_PROB_MICRO \
        actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.n=$ROLLOUT_N \
        actor_rollout_ref.rollout.temperature=0.9 \
        actor_rollout_ref.rollout.top_k=20 \
        actor_rollout_ref.rollout.top_p=0.95 \
        actor_rollout_ref.actor.clip_ratio_low=0.2 \
        actor_rollout_ref.actor.clip_ratio_high=0.28 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$REF_LOG_PROB_MICRO \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        trainer.critic_warmup=0 \
        trainer.logger='["console","wandb"]' \
        trainer.project_name=verl_grpo_math \
        "trainer.experiment_name='$EXP_NAME'" \
        "trainer.default_local_dir='$CKPT_ROOT/$EXP_NAME'" \
        trainer.n_gpus_per_node=$NGPUS \
        trainer.nnodes=1 \
        trainer.save_freq=$SAVE_FREQ \
        trainer.test_freq=5 \
        trainer.total_epochs=$EPOCHS \
        $FREEZE_ARGS \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} 2>&1 | tee "$LOG_FILE"
}

should_run() {
    local exp_num=$1
    if [[ -n "$ONLY" ]]; then
        echo "$ONLY" | tr ',' '\n' | grep -qx "$exp_num" && return 0 || return 1
    fi
    [[ $exp_num -gt $SKIP ]] && return 0 || return 1
}

# ========== Run Experiments ==========
echo "============================================================"
echo "  Qwen3-1.7B-Base middle-5 layer RL on MATH"
echo "  Exp 1: Layers 12-16 only, LR=5e-6"
echo "  Exp 2: Full RL,           LR=1e-6 (baseline)"
echo "  Model: $MODEL | GPUs: $GPUS ($NGPUS)"
echo "  Data:  $DATA_DIR"
echo "  Ckpt:  $CKPT_ROOT"
echo "============================================================"
echo ""

# --- Exp 1: Middle 5 layers (12,13,14,15,16) ---
EXP1_NAME="${DATE}_exp1_mid5_layers12-16_Qwen3-1.7B-Base_math_lr5e-6"
if should_run 1; then
    echo "=========================================="
    echo "  [1/2] Layers 12-16 (middle-5), LR=5e-6"
    echo "=========================================="
    run_train "$EXP1_NAME" "$DATA_DIR" "5e-6" \
        "+actor_rollout_ref.actor.train_layer_ids='12,13,14,15,16'"
    echo "  [1/2] Done."
    echo ""
fi

# --- Exp 2: Full RL baseline ---
EXP2_NAME="${DATE}_exp2_full_Qwen3-1.7B-Base_math_lr1e-6"
if should_run 2; then
    echo "=========================================="
    echo "  [2/2] Full RL Qwen3-1.7B-Base, LR=1e-6"
    echo "=========================================="
    run_train "$EXP2_NAME" "$DATA_DIR" "1e-6" ""
    echo "  [2/2] Done."
    echo ""
fi

echo ""
echo "============================================================"
echo "  Initial experiments complete. Entering infinite 5-layer sweep."
echo "============================================================"

# ========== Infinite 5-layer sweep ==========
# Cycles through these contiguous 5-layer windows forever (or until killed).
# Edit this array to change/extend the layer sets covered.
SWEEP_LAYERS=(
    "0,1,2,3,4"
    "3,4,5,6,7"
    "6,7,8,9,10"
    "9,10,11,12,13"
    "15,16,17,18,19"
    "18,19,20,21,22"
    "21,22,23,24,25"
    "23,24,25,26,27"
)

SWEEP_LR="${SWEEP_LR:-5e-6}"
ITER=0
while true; do
    for LAYERS in "${SWEEP_LAYERS[@]}"; do
        ITER=$((ITER + 1))
        # Build a layer-tag suitable for filenames: "0-4" instead of "0,1,2,3,4"
        FIRST_L=${LAYERS%%,*}
        LAST_L=${LAYERS##*,}
        TAG="layers${FIRST_L}-${LAST_L}"
        EXP_NAME="$(date +%m%d_%H%M)_loop${ITER}_${TAG}_Qwen3-1.7B-Base_math_lr${SWEEP_LR}"
        echo "=========================================="
        echo "  [LOOP iter=$ITER] $TAG (layers [$LAYERS]), LR=$SWEEP_LR"
        echo "=========================================="
        run_train "$EXP_NAME" "$DATA_DIR" "$SWEEP_LR" \
            "+actor_rollout_ref.actor.train_layer_ids='$LAYERS'"
        echo "  [LOOP iter=$ITER] $TAG done."
        echo ""
    done
done
