#!/bin/bash
# ==============================================================================
# 6 Experiments (Part 2): Qwen3-1.7B-Base Layer 12-27 on NuminaMath-CoT (with tmux)
# ==============================================================================
#
# Experiments:
#   1. Layer 12 Qwen3-1.7B-Base,     LR=5e-6
#   2. Layer 15 Qwen3-1.7B-Base,     LR=5e-6
#   3. Layer 18 Qwen3-1.7B-Base,     LR=1e-5
#   4. Layer 21 Qwen3-1.7B-Base,     LR=1e-5
#   5. Layer 24 Qwen3-1.7B-Base,     LR=1e-5
#   6. Layer 27 Qwen3-1.7B-Base,     LR=1e-5
#
# All: batch=512, minibatch=128, microbatch=8, epochs=2, max_response_length=3072
# 8 GPUs, saves only last-step checkpoint in HuggingFace format.
#
# Usage:
#   bash run_numina_base_part2_tmux.sh                          # Run all 6
#   bash run_numina_base_part2_tmux.sh --gpus 0,1,2,3,4,5,6,7  # Use specific GPUs
#   bash run_numina_base_part2_tmux.sh --skip 2                 # Skip first 2
#   bash run_numina_base_part2_tmux.sh --only 1,3               # Run only 1 and 3
#   bash run_numina_base_part2_tmux.sh --no-tmux                # Skip tmux auto-launch

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJ_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)

# ========== Configuration ==========
MODEL="Qwen/Qwen3-1.7B-Base"
GPUS="0,1,2,3,4,5,6,7"
CKPT_ROOT="$PROJ_DIR/checkpoints"
DATA_DIR="$PROJ_DIR/data/numina_math_cot_author"
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
    TMUX_SESSION="numina_base_p2_$(date +%m%d_%H%M)"

    FULL_ARGS="--no-tmux"
    FULL_ARGS="$FULL_ARGS --gpus $(printf '%q' "$GPUS")"
    FULL_ARGS="$FULL_ARGS --model $(printf '%q' "$MODEL")"
    FULL_ARGS="$FULL_ARGS --ckpt-root $(printf '%q' "$CKPT_ROOT")"
    FULL_ARGS="$FULL_ARGS --data-dir $(printf '%q' "$DATA_DIR")"
    [[ $SKIP -gt 0 ]] && FULL_ARGS="$FULL_ARGS --skip $SKIP"
    [[ -n "$ONLY" ]] && FULL_ARGS="$FULL_ARGS --only $(printf '%q' "$ONLY")"
    for arg in "${EXTRA_ARGS[@]}"; do FULL_ARGS="$FULL_ARGS $(printf '%q' "$arg")"; done

    tmux new-session -d -s "$TMUX_SESSION" \
        "source $CONDA_INIT && \
         conda activate $CONDA_ENV_PATH && \
         cd $PROJ_DIR && \
         bash $SCRIPT_DIR/run_numina_base_part2_tmux.sh $FULL_ARGS; \
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
    local EPOCHS="${10:-2}"
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
        trainer.project_name=verl_grpo_numina_cot \
        "trainer.experiment_name='$EXP_NAME'" \
        "trainer.default_local_dir='$CKPT_ROOT/$EXP_NAME'" \
        trainer.n_gpus_per_node=$NGPUS \
        trainer.nnodes=1 \
        trainer.save_freq=$SAVE_FREQ \
        trainer.test_freq=5 \
        trainer.total_epochs=$EPOCHS \
        $FREEZE_ARGS \
        "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
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
echo "  6 Experiments (Part 2): Qwen3-1.7B-Base Layers 12-27"
echo "  Layers: 12,15 (5e-6), 18,21,24,27 (1e-5) | epochs=2"
echo "  Model: $MODEL | GPUs: $GPUS ($NGPUS)"
echo "  Data: $DATA_DIR"
echo "  Checkpoint root: $CKPT_ROOT"
echo "============================================================"
echo ""

# --- Exp 1: Layer 12, LR=5e-6 ---
EXP1_NAME="${DATE}_exp1_layer12_Qwen3-1.7B-Base_numina_cot_lr5e-6"
if should_run 1; then
    echo "=========================================="
    echo "  [1/6] Layer 12 Qwen3-1.7B-Base, LR=5e-6"
    echo "=========================================="
    run_train "$EXP1_NAME" "$DATA_DIR" "5e-6" \
        "+actor_rollout_ref.actor.train_layer_ids=12"
    echo "  [1/6] Done."
    echo ""
fi

# --- Exp 2: Layer 15, LR=5e-6 ---
EXP2_NAME="${DATE}_exp2_layer15_Qwen3-1.7B-Base_numina_cot_lr5e-6"
if should_run 2; then
    echo "=========================================="
    echo "  [2/6] Layer 15 Qwen3-1.7B-Base, LR=5e-6"
    echo "=========================================="
    run_train "$EXP2_NAME" "$DATA_DIR" "5e-6" \
        "+actor_rollout_ref.actor.train_layer_ids=15"
    echo "  [2/6] Done."
    echo ""
fi

# --- Exp 3: Layer 18, LR=1e-5 ---
EXP3_NAME="${DATE}_exp3_layer18_Qwen3-1.7B-Base_numina_cot_lr1e-5"
if should_run 3; then
    echo "=========================================="
    echo "  [3/6] Layer 18 Qwen3-1.7B-Base, LR=1e-5"
    echo "=========================================="
    run_train "$EXP3_NAME" "$DATA_DIR" "1e-5" \
        "+actor_rollout_ref.actor.train_layer_ids=18"
    echo "  [3/6] Done."
    echo ""
fi

# --- Exp 4: Layer 21, LR=1e-5 ---
EXP4_NAME="${DATE}_exp4_layer21_Qwen3-1.7B-Base_numina_cot_lr1e-5"
if should_run 4; then
    echo "=========================================="
    echo "  [4/6] Layer 21 Qwen3-1.7B-Base, LR=1e-5"
    echo "=========================================="
    run_train "$EXP4_NAME" "$DATA_DIR" "1e-5" \
        "+actor_rollout_ref.actor.train_layer_ids=21"
    echo "  [4/6] Done."
    echo ""
fi

# --- Exp 5: Layer 24, LR=1e-5 ---
EXP5_NAME="${DATE}_exp5_layer24_Qwen3-1.7B-Base_numina_cot_lr1e-5"
if should_run 5; then
    echo "=========================================="
    echo "  [5/6] Layer 24 Qwen3-1.7B-Base, LR=1e-5"
    echo "=========================================="
    run_train "$EXP5_NAME" "$DATA_DIR" "1e-5" \
        "+actor_rollout_ref.actor.train_layer_ids=24"
    echo "  [5/6] Done."
    echo ""
fi

# --- Exp 6: Layer 27, LR=1e-5 ---
EXP6_NAME="${DATE}_exp6_layer27_Qwen3-1.7B-Base_numina_cot_lr1e-5"
if should_run 6; then
    echo "=========================================="
    echo "  [6/6] Layer 27 Qwen3-1.7B-Base, LR=1e-5"
    echo "=========================================="
    run_train "$EXP6_NAME" "$DATA_DIR" "1e-5" \
        "+actor_rollout_ref.actor.train_layer_ids=27"
    echo "  [6/6] Done."
    echo ""
fi

echo ""
echo "============================================================"
echo "  Part 2 complete!"
echo "  Starting dummy GPU hold job..."
echo "============================================================"
DUMMY_RUN_NAME="dummy_numina_base_p2_$(hostname)_$(date +%m%d_%H%M)" \
    python3 "$SCRIPT_DIR/dummy_gpu_hold.py"
