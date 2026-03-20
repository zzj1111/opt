#!/bin/bash
# ==============================================================================
# 6 Experiments: Qwen3-1.7B RL training with MATH and MBPP
# ==============================================================================
#
# Experiments:
#   1. Full RL MATH,           LR=1e-6
#   2. Layer 14 only MATH,     LR=5e-6
#   3. Full RL MBPP,           LR=1e-6
#   4. Layer 15 only MBPP,     LR=2e-5
#   5. Full RL MATH -> Full RL MBPP,   LR=1e-6
#   6. Layer 14 MATH -> Layer 15 MBPP, LR=5e-6 -> 2e-5
#
# All use max_response_length=8192 (8K), Qwen3-1.7B base model, 8 GPUs.
# Saves only last-step checkpoint in HuggingFace format.
# After all training, uploads all checkpoints to HuggingFace Hub.
#
# Usage:
#   bash run_6experiments.sh                          # Run all 6
#   bash run_6experiments.sh --gpus 0,1,2,3           # Use specific GPUs
#   bash run_6experiments.sh --skip 3                 # Skip first 3
#   bash run_6experiments.sh --only 1,2               # Run only 1 and 2
#   bash run_6experiments.sh --no-upload              # Skip HF upload
#
# Environment variables (set before running or pass via --hf-token etc.):
#   WANDB_API_KEY       - Weights & Biases API key
#   WANDB_ENTITY        - W&B entity (team/user)
#   HF_TOKEN            - HuggingFace Hub token for upload

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJ_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)

# ========== Configuration ==========
MODEL="Qwen/Qwen3-1.7B"
GPUS="0,1,2,3,4,5,6,7"
CKPT_ROOT="$PROJ_DIR/checkpoints"
MATH_DATA="$PROJ_DIR/data/math"
MBPP_DATA="$PROJ_DIR/data/mbpp"
HF_TOKEN="${HF_TOKEN:-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}"
HF_REPO_PREFIX=""
SKIP=0
ONLY=""
DO_UPLOAD=true
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)            GPUS="$2"; shift 2 ;;
        --model)           MODEL="$2"; shift 2 ;;
        --ckpt-root)       CKPT_ROOT="$2"; shift 2 ;;
        --math-data)       MATH_DATA="$2"; shift 2 ;;
        --mbpp-data)       MBPP_DATA="$2"; shift 2 ;;
        --skip)            SKIP="$2"; shift 2 ;;
        --only)            ONLY="$2"; shift 2 ;;
        --no-upload)       DO_UPLOAD=false; shift ;;
        --hf-token)        HF_TOKEN="$2"; shift 2 ;;
        --hf-repo-prefix)  HF_REPO_PREFIX="$2"; shift 2 ;;
        *)                 EXTRA_ARGS+=("$1"); shift ;;
    esac
done

NGPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
MODEL_SHORT=$(basename "$MODEL")
DATE=$(date +%m%d)

# Validate data
for DF in "$MATH_DATA/train.parquet" "$MBPP_DATA/train.parquet"; do
    if [[ ! -f "$DF" ]]; then
        echo "ERROR: Data not found: $DF"
        exit 1
    fi
done

# ========== Shared training function ==========
#
# Args: EXP_NAME DATA_DIR LR FREEZE_ARGS [MAX_RESP_LEN] [BATCH_SIZE] [EPOCHS] [SAVE_FREQ]
run_train() {
    local EXP_NAME="$1"
    local DATA_DIR="$2"
    local LR="$3"
    local FREEZE_ARGS="$4"
    local MAX_RESP_LEN="${5:-8192}"
    local BATCH_SIZE="${6:-256}"
    local EPOCHS="${7:-2}"
    local SAVE_FREQ="${8:--1}"

    # Calculate steps for save_freq = last step only
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

    # Derive per-GPU batch sizes from NGPUS
    local MINI_BATCH=$((BATCH_SIZE / 2))
    local MICRO_BATCH=8
    local ROLLOUT_LOG_PROB_MICRO=8
    local REF_LOG_PROB_MICRO=8
    # TP=1 is enough for 1.7B; keeps all 8 GPUs as data-parallel
    local TP=1

    mkdir -p "$CKPT_ROOT/$EXP_NAME"
    local LOG_FILE="$CKPT_ROOT/$EXP_NAME/train.log"

    echo "  Experiment:  $EXP_NAME"
    echo "  Data:        $DATA_DIR"
    echo "  LR:          $LR"
    echo "  Freeze:      ${FREEZE_ARGS:-none (full)}"
    echo "  MaxRespLen:  $MAX_RESP_LEN"
    echo "  Batch:       $BATCH_SIZE, MiniBatch: $MINI_BATCH, MicroBatch: $MICRO_BATCH"
    echo "  Epochs:      $EPOCHS, Steps/epoch: $STEPS_PER_EPOCH, Total: $TOTAL_STEPS"
    echo "  SaveFreq:    $SAVE_FREQ (last step only)"
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
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.rollout.temperature=0.9 \
        actor_rollout_ref.rollout.top_k=20 \
        actor_rollout_ref.rollout.top_p=0.95 \
        actor_rollout_ref.actor.clip_ratio_low=0.2 \
        actor_rollout_ref.actor.clip_ratio_high=0.28 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$REF_LOG_PROB_MICRO \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        trainer.critic_warmup=0 \
        "trainer.logger='[\"console\",\"wandb\"]'" \
        trainer.project_name=rl_6experiments \
        "trainer.experiment_name='$EXP_NAME'" \
        "trainer.default_local_dir='$CKPT_ROOT/$EXP_NAME'" \
        trainer.n_gpus_per_node=$NGPUS \
        trainer.nnodes=1 \
        trainer.save_freq=$SAVE_FREQ \
        trainer.test_freq=10 \
        trainer.total_epochs=$EPOCHS \
        $FREEZE_ARGS \
        "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
}

# Find the last global_step_*/actor/huggingface checkpoint
find_last_ckpt() {
    local EXP_DIR="$1"
    local LAST_STEP=""
    for d in "$EXP_DIR"/global_step_*/actor/huggingface; do
        if [[ -d "$d" ]]; then
            LAST_STEP="$d"
        fi
    done
    echo "$LAST_STEP"
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
echo "  6 Experiments: Qwen3-1.7B RL Training"
echo "  Model: $MODEL | GPUs: $GPUS ($NGPUS)"
echo "  MATH data: $MATH_DATA"
echo "  MBPP data: $MBPP_DATA"
echo "  Checkpoint root: $CKPT_ROOT"
echo "============================================================"
echo ""

ORIG_MODEL="$MODEL"  # Save for restore after stage-2 experiments

# --- Exp 1: Full RL MATH, LR=1e-6 ---
EXP1_NAME="${DATE}_exp1_full_math_lr1e-6"
if should_run 1; then
    echo "=========================================="
    echo "  [1/6] Full RL MATH, LR=1e-6"
    echo "=========================================="
    MODEL="$ORIG_MODEL"
    run_train "$EXP1_NAME" "$MATH_DATA" "1e-6" "" "8192" "256" "2"
    echo "  [1/6] Done."
    echo ""
fi

# --- Exp 2: Layer 14 only MATH, LR=5e-6 ---
EXP2_NAME="${DATE}_exp2_layer14_math_lr5e-6"
if should_run 2; then
    echo "=========================================="
    echo "  [2/6] Layer 14 MATH, LR=5e-6"
    echo "=========================================="
    MODEL="$ORIG_MODEL"
    run_train "$EXP2_NAME" "$MATH_DATA" "5e-6" \
        "+actor_rollout_ref.actor.train_layer_ids=14" \
        "8192" "256" "2"
    echo "  [2/6] Done."
    echo ""
fi

# --- Exp 3: Full RL MBPP, LR=1e-6 ---
EXP3_NAME="${DATE}_exp3_full_mbpp_lr1e-6"
if should_run 3; then
    echo "=========================================="
    echo "  [3/6] Full RL MBPP, LR=1e-6"
    echo "=========================================="
    MODEL="$ORIG_MODEL"
    run_train "$EXP3_NAME" "$MBPP_DATA" "1e-6" "" "8192" "128" "30"
    echo "  [3/6] Done."
    echo ""
fi

# --- Exp 4: Layer 15 only MBPP, LR=2e-5 ---
EXP4_NAME="${DATE}_exp4_layer15_mbpp_lr2e-5"
if should_run 4; then
    echo "=========================================="
    echo "  [4/6] Layer 15 MBPP, LR=2e-5"
    echo "=========================================="
    MODEL="$ORIG_MODEL"
    run_train "$EXP4_NAME" "$MBPP_DATA" "2e-5" \
        "+actor_rollout_ref.actor.train_layer_ids=15" \
        "8192" "128" "30"
    echo "  [4/6] Done."
    echo ""
fi

# --- Exp 5: Full RL MATH -> Full RL MBPP, LR=1e-6 ---
EXP5_MATH_NAME="${DATE}_exp5_stage1_full_math_lr1e-6"
EXP5_MBPP_NAME="${DATE}_exp5_stage2_full_mbpp_lr1e-6"
if should_run 5; then
    echo "=========================================="
    echo "  [5/6] Full RL MATH -> Full RL MBPP"
    echo "=========================================="

    echo "  [5/6] Stage 1: Full RL MATH, LR=1e-6"
    MODEL="$ORIG_MODEL"
    run_train "$EXP5_MATH_NAME" "$MATH_DATA" "1e-6" "" "8192" "256" "2"

    STAGE1_CKPT=$(find_last_ckpt "$CKPT_ROOT/$EXP5_MATH_NAME")
    if [[ -z "$STAGE1_CKPT" ]]; then
        echo "ERROR: No checkpoint found from stage 1. Skipping stage 2."
    else
        echo "  [5/6] Stage 1 ckpt: $STAGE1_CKPT"
        echo "  [5/6] Stage 2: Full RL MBPP, LR=1e-6"
        MODEL="$STAGE1_CKPT"
        run_train "$EXP5_MBPP_NAME" "$MBPP_DATA" "1e-6" "" "8192" "128" "30"
    fi
    echo "  [5/6] Done."
    echo ""
fi

# --- Exp 6: Layer 14 MATH -> Layer 15 MBPP ---
EXP6_MATH_NAME="${DATE}_exp6_stage1_layer14_math_lr5e-6"
EXP6_MBPP_NAME="${DATE}_exp6_stage2_layer15_mbpp_lr2e-5"
if should_run 6; then
    echo "=========================================="
    echo "  [6/6] Layer 14 MATH -> Layer 15 MBPP"
    echo "=========================================="

    echo "  [6/6] Stage 1: Layer 14 MATH, LR=5e-6"
    MODEL="$ORIG_MODEL"
    run_train "$EXP6_MATH_NAME" "$MATH_DATA" "5e-6" \
        "+actor_rollout_ref.actor.train_layer_ids=14" \
        "8192" "256" "2"

    STAGE1_CKPT=$(find_last_ckpt "$CKPT_ROOT/$EXP6_MATH_NAME")
    if [[ -z "$STAGE1_CKPT" ]]; then
        echo "ERROR: No checkpoint found from stage 1. Skipping stage 2."
    else
        echo "  [6/6] Stage 1 ckpt: $STAGE1_CKPT"
        echo "  [6/6] Stage 2: Layer 15 MBPP, LR=2e-5"
        MODEL="$STAGE1_CKPT"
        run_train "$EXP6_MBPP_NAME" "$MBPP_DATA" "2e-5" \
            "+actor_rollout_ref.actor.train_layer_ids=15" \
            "8192" "128" "30"
    fi
    echo "  [6/6] Done."
    echo ""
fi

# ========== Upload to HuggingFace ==========
if [[ "$DO_UPLOAD" == "true" ]]; then
    if [[ -z "$HF_TOKEN" ]]; then
        echo "WARNING: HF_TOKEN not set, skipping upload."
        echo "  Set HF_TOKEN env var or pass --hf-token <token>"
    else
        echo "============================================================"
        echo "  Uploading checkpoints to HuggingFace Hub"
        echo "============================================================"
        python3 "$SCRIPT_DIR/upload_checkpoints.py" \
            --ckpt-root "$CKPT_ROOT" \
            --token "$HF_TOKEN" \
            --prefix "${DATE}_exp" \
            ${HF_REPO_PREFIX:+--repo-prefix "$HF_REPO_PREFIX"}
    fi
fi

echo ""
echo "============================================================"
echo "  All experiments complete!"
echo "============================================================"
