#!/bin/bash
# ==============================================================================
# 15 Experiments (Part 2): Qwen3-1.7B-Base Layers 12-27 on NuminaMath-CoT (with tmux)
# ==============================================================================
#
# Experiments (priority layers first, then fill remaining):
#   1.  Layer 12 Qwen3-1.7B-Base,     LR=5e-6
#   2.  Layer 15 Qwen3-1.7B-Base,     LR=5e-6
#   3.  Layer 18 Qwen3-1.7B-Base,     LR=5e-6
#   4.  Layer 21 Qwen3-1.7B-Base,     LR=5e-6
#   5.  Layer 24 Qwen3-1.7B-Base,     LR=5e-6
#   6.  Layer 27 Qwen3-1.7B-Base,     LR=5e-6
#   --- remaining layers to keep node busy ---
#   7.  Layer 14 Qwen3-1.7B-Base,     LR=5e-6
#   8.  Layer 16 Qwen3-1.7B-Base,     LR=5e-6
#   9.  Layer 17 Qwen3-1.7B-Base,     LR=5e-6
#   10. Layer 19 Qwen3-1.7B-Base,     LR=5e-6
#   11. Layer 20 Qwen3-1.7B-Base,     LR=5e-6
#   12. Layer 22 Qwen3-1.7B-Base,     LR=5e-6
#   13. Layer 23 Qwen3-1.7B-Base,     LR=5e-6
#   14. Layer 25 Qwen3-1.7B-Base,     LR=5e-6
#   15. Layer 26 Qwen3-1.7B-Base,     LR=5e-6
#
# All: batch=512, minibatch=128, microbatch=8, epochs=2, max_response_length=3072
# 8 GPUs, saves only last-step checkpoint in HuggingFace format.
#
# Usage:
#   bash run_numina_base_part2_tmux.sh                          # Run all 15
#   bash run_numina_base_part2_tmux.sh --gpus 0,1,2,3,4,5,6,7  # Use specific GPUs
#   bash run_numina_base_part2_tmux.sh --skip 2                 # Skip first 2
#   bash run_numina_base_part2_tmux.sh --only 1,3               # Run only 1 and 3
#   bash run_numina_base_part2_tmux.sh --no-tmux                # Skip tmux auto-launch

set -uo pipefail

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
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} 2>&1 | tee "$LOG_FILE"
}

should_run() {
    local exp_num=$1
    if [[ -n "$ONLY" ]]; then
        echo "$ONLY" | tr ',' '\n' | grep -qx "$exp_num" && return 0 || return 1
    fi
    [[ $exp_num -gt $SKIP ]] && return 0 || return 1
}

# Priority layers first, then remaining layers
LAYERS=(12 15 18 21 24 27 14 16 17 19 20 22 23 25 26)
TOTAL=${#LAYERS[@]}

# ========== Run Experiments ==========
echo "============================================================"
echo "  $TOTAL Experiments (Part 2): Qwen3-1.7B-Base Layers 12-27"
echo "  Priority: 12,15,18,21,24,27 then 14,16,17,19,20,22,23,25,26"
echo "  All layers LR=5e-6 | epochs=2"
echo "  Model: $MODEL | GPUs: $GPUS ($NGPUS)"
echo "  Data: $DATA_DIR"
echo "  Checkpoint root: $CKPT_ROOT"
echo "============================================================"
echo ""

for i in "${!LAYERS[@]}"; do
    LAYER=${LAYERS[$i]}
    EXP_NUM=$((i + 1))
    EXP_NAME="${DATE}_exp${EXP_NUM}_layer${LAYER}_Qwen3-1.7B-Base_numina_cot_lr5e-6"
    if should_run $EXP_NUM; then
        echo "=========================================="
        echo "  [$EXP_NUM/$TOTAL] Layer $LAYER Qwen3-1.7B-Base, LR=5e-6"
        echo "=========================================="
        run_train "$EXP_NAME" "$DATA_DIR" "5e-6" \
            "+actor_rollout_ref.actor.train_layer_ids=$LAYER"
        echo "  [$EXP_NUM/$TOTAL] Done."
        echo ""
    fi
done

echo ""
echo "============================================================"
echo "  Part 2 complete!"
echo "  Starting dummy GPU hold job..."
echo "============================================================"
DUMMY_RUN_NAME="dummy_numina_base_p2_$(hostname)_$(date +%m%d_%H%M)" \
    python3 "$SCRIPT_DIR/dummy_gpu_hold.py"
