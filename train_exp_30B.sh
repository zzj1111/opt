export CUDA_VISIBLE_DEVICES=7
export REWARD_CUDA_VISIBLE_DEVICES=6
export WANDB_API_KEY="b8f38344ec7231ee89baa74ef7209dd5a43df6b2"
export WANDB_ENTITY="mhong-university-of-minnesota"
#export Model_path="/code/hongpaul-sandbox/temp/CudaForge_plus/verl/data/Qwen3_8b"
export Model_path="Qwen/Qwen3-8B"
#export Model_path="Qwen/Qwen3-30B-A3B-Instruct-2507"


export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export RAY_BACKEND_LOG_LEVEL=-1  # 有些版本支持

max_response_length=16384

# loss_mode=gspo
# loss_agg_mode="seq-mean-token-mean"

project_name=CudaForge_RL
exp_name="d0107r2_GSPO_l123_32B"

export RUBRIC_VLLM_URL="http://localhost:8093/v1/chat/completions"
export RUBRIC_MODEL_NAME="Qwen/Qwen3-8B"
export RUBRIC_VLLM_TIMEOUT_SEC=60


CKPTS_DIR=/code/hongpaul-sandbox/temp/CudaForge_plus/${project_name}/${exp_name}

source /mnt/data1/zha00175/miniconda/bin/activate
conda activate verl

mkdir -p logs
#path=dataset/CudaForge/Level1
path=/home/zha00175/verl/dataset/CudaForge
#path=/home/zha00175/verl/dataset/CudaForge/new/l1/
# export VLLM_ATTENTION_BACKEND=XFORMERS

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 trainer.project_name=CudaForge_RL \
 algorithm.adv_estimator=grpo \
 data.train_files=/home/zha00175/cuda/CudaForge/run/train.parquet \
 data.val_files=${path}/test.parquet \
 actor_rollout_ref.model.use_shm=True \
 actor_rollout_ref.model.lora_rank=128 \
 actor_rollout_ref.model.lora_alpha=128 \
 data.train_batch_size=16 \
 data.val_batch_size=8 \
 actor_rollout_ref.model.target_modules=all-linear \
 data.max_prompt_length=8192 \
 data.max_response_length=16384 \
 data.shuffle=True \
 actor_rollout_ref.rollout.load_format=safetensors \
 actor_rollout_ref.model.path=$Model_path \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=4 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.n=5 \
 actor_rollout_ref.rollout.layered_summon=True \
 actor_rollout_ref.actor.use_kl_loss=True \
 actor_rollout_ref.model.enable_gradient_checkpointing=True \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
 actor_rollout_ref.actor.fsdp_config.param_offload=True \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
 actor_rollout_ref.rollout.temperature=0.9 \
 actor_rollout_ref.rollout.top_k=20 \
 actor_rollout_ref.rollout.top_p=0.95 \
 actor_rollout_ref.actor.clip_ratio_low=0.2 \
 actor_rollout_ref.actor.clip_ratio_high=0.28 \
 reward_model.enable=False \
 reward_model.reward_manager=dapo \
 custom_reward_function.path=./verl/utils/reward_score/CudaForge.py \
 actor_rollout_ref.actor.kl_loss_coef=0.01 \
 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
 trainer.val_before_train=False \
 trainer.n_gpus_per_node=1 \
 trainer.nnodes=1 \
 trainer.default_local_dir="${CKPTS_DIR}" \
 trainer.save_freq=20 \
 trainer.test_freq=20 \
 trainer.logger='["console","wandb"]' \
 trainer.project_name="DAPO" \
 trainer.experiment_name="d1222r1_GSPO_level1_32B" \
 trainer.total_epochs=5 \
