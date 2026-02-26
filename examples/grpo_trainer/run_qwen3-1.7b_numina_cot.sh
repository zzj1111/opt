#!/bin/bash
# GRPO training on NuminaMath-CoT with Qwen3-1.7B
# Prerequisite: python examples/data_preprocess/numina_math_cot.py --max_samples 50000

set -x

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJ_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)

# Parse command-line arguments
BETA1=0.9
BETA2=0.999
LR=1e-6
ROUND=""
NOTE=""
GPUS="0,1,2,3"
MODEL="Qwen/Qwen3-1.7B"
OPTIM="adamw"
MOMENTUM=0.9
ROLLOUT_N=5
CKPT_ROOT="checkpoints"
DATA_DIR="$PROJ_DIR/data/numina_math_cot"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --beta1) BETA1="$2"; shift 2 ;;
        --beta2) BETA2="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --round) ROUND="$2"; shift 2 ;;
        --note) NOTE="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --optim) OPTIM="$2"; shift 2 ;;
        --momentum) MOMENTUM="$2"; shift 2 ;;
        --rollout-n) ROLLOUT_N="$2"; shift 2 ;;
        --ckpt-root) CKPT_ROOT="$2"; shift 2 ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# Verify data exists
if [[ ! -f "$DATA_DIR/train.parquet" ]]; then
    echo "ERROR: $DATA_DIR/train.parquet not found."
    echo "Run: python examples/data_preprocess/numina_math_cot.py --max_samples 50000"
    exit 1
fi

MODEL_SHORT=$(basename "$MODEL")
NGPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
DATE=$(date +%m%d)

if [[ "$OPTIM" == "sgd" ]]; then
    OPTIM_TAG="sgd_mu${MOMENTUM}"
else
    OPTIM_TAG="b1${BETA1}_b2${BETA2}"
fi
ROLLOUT_TAG="n${ROLLOUT_N}"

# Auto-increment round
if [[ -z "$ROUND" ]]; then
    ROUND=1
    while true; do
        EXP_NAME="${DATE}_r${ROUND}_${MODEL_SHORT}_numina_cot_${OPTIM_TAG}_lr${LR}_${ROLLOUT_TAG}"
        if [[ -n "$NOTE" ]]; then EXP_NAME="${EXP_NAME}_${NOTE}"; fi
        if mkdir -p "$CKPT_ROOT" && mkdir "$CKPT_ROOT/$EXP_NAME" 2>/dev/null; then
            break
        fi
        ROUND=$((ROUND + 1))
    done
else
    EXP_NAME="${DATE}_r${ROUND}_${MODEL_SHORT}_numina_cot_${OPTIM_TAG}_lr${LR}_${ROLLOUT_TAG}"
    if [[ -n "$NOTE" ]]; then EXP_NAME="${EXP_NAME}_${NOTE}"; fi
    mkdir -p "$CKPT_ROOT/$EXP_NAME"
fi

# If not inside tmux, launch a tmux session and re-run this script inside it
if [[ -z "$TMUX" ]]; then
    TMUX_SESSION="train_${EXP_NAME}"
    # Build the full command to re-run inside tmux
    ARGS="--beta1 $BETA1 --beta2 $BETA2 --lr $LR --round $ROUND --gpus $GPUS --model $MODEL --optim $OPTIM --momentum $MOMENTUM --rollout-n $ROLLOUT_N --ckpt-root $CKPT_ROOT --data-dir $DATA_DIR"
    if [[ -n "$NOTE" ]]; then ARGS="$ARGS --note $NOTE"; fi
    for arg in "${EXTRA_ARGS[@]}"; do ARGS="$ARGS $arg"; done

    tmux new-session -d -s "$TMUX_SESSION" \
        "source /code/hongpaul-sandbox/cuda/miniconda3/bin/activate && \
         conda activate /code/hongpaul-sandbox/cuda/miniconda3/envs/cuda && \
         cd $PROJ_DIR && \
         bash $SCRIPT_DIR/run_qwen3-1.7b_numina_cot.sh $ARGS; \
         exec bash"
    echo "Tmux session '$TMUX_SESSION' started. Attach with: tmux attach -t $TMUX_SESSION"
    exit 0
fi

export CUDA_VISIBLE_DEVICES=$GPUS
export WANDB_API_KEY="b8f38344ec7231ee89baa74ef7209dd5a43df6b2"
export WANDB_ENTITY="mhong-university-of-minnesota"
export VERL_DEFAULT_LOCAL_DIR="$CKPT_ROOT/$EXP_NAME"

LOG_FILE="$CKPT_ROOT/$EXP_NAME/train.log"
echo "Experiment: $EXP_NAME"
echo "Logging to: $LOG_FILE"

# Build optimizer args
if [[ "$OPTIM" == "sgd" ]]; then
    OPTIM_ARGS="actor_rollout_ref.actor.optim.optimizer=SGD actor_rollout_ref.actor.optim.optimizer_impl=torch.optim actor_rollout_ref.actor.optim.weight_decay=0.0 actor_rollout_ref.actor.optim.override_optimizer_config={momentum:$MOMENTUM}"
else
    OPTIM_ARGS="actor_rollout_ref.actor.optim.betas=[$BETA1,$BETA2]"
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=$LR \
    ${OPTIM_ARGS} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.rollout.top_k=20 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_numina_cot' \
    trainer.experiment_name="$EXP_NAME" \
    trainer.default_local_dir="$CKPT_ROOT/$EXP_NAME" \
    trainer.n_gpus_per_node=$NGPUS \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.test_freq=10 \
    trainer.total_epochs=2 "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
