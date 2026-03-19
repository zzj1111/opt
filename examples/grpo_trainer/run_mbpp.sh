#!/bin/bash
# Train on MBPP dataset with layer-selective freezing.
#
# Usage:
#   bash examples/grpo_trainer/run_mbpp.sh
#   bash examples/grpo_trainer/run_mbpp.sh --train-layer-ids 13 --lr 1e-5
#   bash examples/grpo_trainer/run_mbpp.sh --model Qwen/Qwen3-8B --gpus 0,1,2,3,4,5,6,7

set -x

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJ_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)

# Defaults
LR=5e-6
ROUND=""
NOTE=""
GPUS="0,1,2,3,4,5,6,7"
MODEL="/code/hongpaul-sandbox/temp/OPT-RL/opt/checkpoints/0312_r1_Qwen3-1.7B_b10.9_b20.999_lr5e-6_layer_14/global_step_70/actor/huggingface"
DATA_DIR="$PROJ_DIR/data/mbpp"
CKPT_ROOT="checkpoints"
FREEZE_LAYERS=0
TRAIN_LAYER_IDS=""
NO_TMUX=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --lr) LR="$2"; shift 2 ;;
        --round) ROUND="$2"; shift 2 ;;
        --note) NOTE="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --ckpt-root) CKPT_ROOT="$2"; shift 2 ;;
        --freeze-layers) FREEZE_LAYERS="$2"; shift 2 ;;
        --train-layer-ids) TRAIN_LAYER_IDS="$2"; shift 2 ;;
        --no-tmux) NO_TMUX=true; shift ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

MODEL_SHORT=$(basename "$MODEL")
NGPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
DATE=$(date +%m%d_%H%M)

# Layer tag for experiment name
LAYER_TAG=""
if [[ -n "$TRAIN_LAYER_IDS" ]]; then
    LAYER_TAG="layer_$(echo "$TRAIN_LAYER_IDS" | tr ',' '_')"
fi

EXP_NAME="mbpp_${DATE}_${MODEL_SHORT}_lr${LR}"
if [[ -n "$ROUND" ]]; then EXP_NAME="mbpp_${DATE}_r${ROUND}_${MODEL_SHORT}_lr${LR}"; fi
if [[ -n "$LAYER_TAG" ]]; then EXP_NAME="${EXP_NAME}_${LAYER_TAG}"; fi
if [[ -n "$NOTE" ]]; then EXP_NAME="${EXP_NAME}_${NOTE}"; fi
mkdir -p "$CKPT_ROOT/$EXP_NAME"

# Hydra overrides for layer freezing
FREEZE_ARGS=""
if [[ -n "$TRAIN_LAYER_IDS" ]]; then
    FREEZE_ARGS="+actor_rollout_ref.actor.train_layer_ids=$TRAIN_LAYER_IDS"
fi
if [[ "$FREEZE_LAYERS" -gt 0 ]]; then
    FREEZE_ARGS="$FREEZE_ARGS +actor_rollout_ref.actor.freeze_except_last_n_layers=$FREEZE_LAYERS"
fi

# tmux auto-launch
if [[ -z "$TMUX" ]] && [[ "$NO_TMUX" == "false" ]]; then
    TMUX_SESSION="train_${EXP_NAME}"
    ARGS="--lr $LR --gpus $GPUS --model $MODEL --ckpt-root $CKPT_ROOT --freeze-layers $FREEZE_LAYERS"
    if [[ -n "$ROUND" ]]; then ARGS="$ARGS --round $ROUND"; fi
    if [[ -n "$NOTE" ]]; then ARGS="$ARGS --note $NOTE"; fi
    if [[ -n "$TRAIN_LAYER_IDS" ]]; then ARGS="$ARGS --train-layer-ids $TRAIN_LAYER_IDS"; fi
    for arg in "${EXTRA_ARGS[@]}"; do ARGS="$ARGS $arg"; done

    tmux new-session -d -s "$TMUX_SESSION" \
        "source /code/hongpaul-sandbox/cuda/miniconda3/bin/activate && \
         conda activate /code/hongpaul-sandbox/cuda/miniconda3/envs/cuda && \
         cd $PROJ_DIR && \
         bash $SCRIPT_DIR/run_mbpp.sh --no-tmux $ARGS; \
         exec bash"
    echo "Tmux session '$TMUX_SESSION' started. Attach with: tmux attach -t $TMUX_SESSION"
    exit 0
fi

export CUDA_VISIBLE_DEVICES=$GPUS
export WANDB_API_KEY="b8f38344ec7231ee89baa74ef7209dd5a43df6b2"
export WANDB_ENTITY="mhong-university-of-minnesota"
export VERL_DEFAULT_LOCAL_DIR="$CKPT_ROOT/$EXP_NAME"

LOG_FILE="$CKPT_ROOT/$EXP_NAME/train.log"
echo "Logging to: $LOG_FILE"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.optim.betas=[0.9,0.999] \
    actor_rollout_ref.actor.checkpoint.save_contents='["hf_model"]' \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.rollout.top_k=20 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=verl_grpo_mbpp \
    trainer.experiment_name="$EXP_NAME" \
    trainer.default_local_dir="$CKPT_ROOT/$EXP_NAME" \
    trainer.n_gpus_per_node=$NGPUS \
    trainer.nnodes=1 \
    $FREEZE_ARGS \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.total_epochs=30 "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
