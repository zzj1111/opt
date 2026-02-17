# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.

set -x

# Parse command-line arguments
BETA1=0.9
BETA2=0.999
LR=1e-6
ROUND=""
NOTE=""
GPUS="0,1,2,3"
MODEL="Qwen/Qwen3-4B"
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
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# Extract short model name: "Qwen/Qwen3-4B" -> "Qwen3-4B"
MODEL_SHORT=$(basename "$MODEL")

# Count number of GPUs from comma-separated list
NGPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)

# Auto-generate experiment name: date_round_beta1_beta2_lr[_note]
DATE=$(date +%m%d)

# Auto-increment round if not explicitly specified
if [[ -z "$ROUND" ]]; then
    MAX_ROUND=0
    for dir in checkpoints/${DATE}_r*; do
        if [[ -d "$dir" ]]; then
            R=$(basename "$dir" | grep -oP '(?<=_r)\d+')
            if [[ -n "$R" && "$R" -gt "$MAX_ROUND" ]]; then
                MAX_ROUND=$R
            fi
        fi
    done
    ROUND=$((MAX_ROUND + 1))
fi

EXP_NAME="${DATE}_r${ROUND}_${MODEL_SHORT}_b1${BETA1}_b2${BETA2}_lr${LR}"
if [[ -n "$NOTE" ]]; then
    EXP_NAME="${EXP_NAME}_${NOTE}"
fi

export CUDA_VISIBLE_DEVICES=$GPUS
export WANDB_API_KEY="b8f38344ec7231ee89baa74ef7209dd5a43df6b2"
export WANDB_ENTITY="mhong-university-of-minnesota"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/gsm8k/train.parquet \
    data.val_files=data/gsm8k/test.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.optim.betas="[$BETA1,$BETA2]" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.rollout.top_k=20 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name="$EXP_NAME" \
    trainer.default_local_dir="checkpoints/$EXP_NAME" \
    trainer.n_gpus_per_node=$NGPUS \
    trainer.nnodes=1 \
    trainer.save_freq=1000 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 "${EXTRA_ARGS[@]}"