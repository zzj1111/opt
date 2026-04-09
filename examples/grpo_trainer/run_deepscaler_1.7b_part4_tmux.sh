#!/bin/bash
# ==============================================================================
# Part 4/4: Qwen3-1.7B-Base Layers 21-27 on DeepScaleR (8K)
# ==============================================================================
# 7 experiments, epochs=2, max_response_length=8192, 8 GPUs
#
# Usage:
#   bash run_deepscaler_1.7b_part4_tmux.sh
#   bash run_deepscaler_1.7b_part4_tmux.sh --skip N
#   bash run_deepscaler_1.7b_part4_tmux.sh --only 1,3
#   bash run_deepscaler_1.7b_part4_tmux.sh --no-tmux

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJ_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
SCRIPT_NAME=$(basename "$0")

MODEL="Qwen/Qwen3-1.7B-Base"
GPUS="0,1,2,3,4,5,6,7"
CKPT_ROOT="$PROJ_DIR/checkpoints"
DATA_DIR="$PROJ_DIR/data/deepscaler"
CONDA_INIT="${CONDA_INIT:-/code/hongpaul-sandbox/cuda/miniconda3/bin/activate}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/code/hongpaul-sandbox/cuda/miniconda3/envs/cuda}"
SKIP=0; ONLY=""; NO_TMUX=false; EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)       GPUS="$2"; shift 2 ;;
        --model)      MODEL="$2"; shift 2 ;;
        --ckpt-root)  CKPT_ROOT="$2"; shift 2 ;;
        --data-dir)   DATA_DIR="$2"; shift 2 ;;
        --skip)       SKIP="$2"; shift 2 ;;
        --only)       ONLY="$2"; shift 2 ;;
        --no-tmux)    NO_TMUX=true; shift ;;
        *)            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "${TMUX:-}" ]] && [[ "$NO_TMUX" == "false" ]]; then
    TMUX_SESSION="dsr_p4_$(date +%m%d_%H%M)"
    FULL_ARGS="--no-tmux --gpus $(printf '%q' "$GPUS") --model $(printf '%q' "$MODEL") --ckpt-root $(printf '%q' "$CKPT_ROOT") --data-dir $(printf '%q' "$DATA_DIR")"
    [[ $SKIP -gt 0 ]] && FULL_ARGS="$FULL_ARGS --skip $SKIP"
    [[ -n "$ONLY" ]] && FULL_ARGS="$FULL_ARGS --only $(printf '%q' "$ONLY")"
    for arg in ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}; do FULL_ARGS="$FULL_ARGS $(printf '%q' "$arg")"; done
    tmux new-session -d -s "$TMUX_SESSION" \
        "source $CONDA_INIT && conda activate $CONDA_ENV_PATH && cd $PROJ_DIR && bash $SCRIPT_DIR/$SCRIPT_NAME $FULL_ARGS; exec bash"
    echo "Tmux '$TMUX_SESSION' started.  Attach: tmux attach -t $TMUX_SESSION"; exit 0
fi

NGPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
MODEL_SHORT=$(basename "$MODEL")
DATE=$(date +%m%d_%H%M)
[[ ! -f "$DATA_DIR/train.parquet" ]] && echo "ERROR: $DATA_DIR/train.parquet not found" && exit 1

run_train() {
    local EXP_NAME="$1" DATA_DIR="$2" LR="$3" FREEZE_ARGS="$4"
    local BATCH_SIZE="${5:-512}" MINI_BATCH="${6:-128}" MICRO_BATCH="${7:-8}"
    local ROLLOUT_N="${8:-5}" EPOCHS="${9:-2}" SAVE_FREQ="${10:--1}"
    local STEPS_PER_EPOCH
    STEPS_PER_EPOCH=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('$DATA_DIR/train.parquet')) // $BATCH_SIZE)")
    local TOTAL_STEPS=$((STEPS_PER_EPOCH * EPOCHS))
    [[ "$SAVE_FREQ" == "-1" ]] && SAVE_FREQ=$TOTAL_STEPS
    mkdir -p "$CKPT_ROOT/$EXP_NAME"
    local LOG_FILE="$CKPT_ROOT/$EXP_NAME/train.log"
    echo "  ---- $EXP_NAME ----"
    echo "  Model=$MODEL LR=$LR Freeze=${FREEZE_ARGS:-full} Epochs=$EPOCHS Steps=$TOTAL_STEPS"
    export CUDA_VISIBLE_DEVICES=$GPUS
    export WANDB_API_KEY="${WANDB_API_KEY:-b8f38344ec7231ee89baa74ef7209dd5a43df6b2}"
    export WANDB_ENTITY="${WANDB_ENTITY:-mhong-university-of-minnesota}"
    export VERL_DEFAULT_LOCAL_DIR="$CKPT_ROOT/$EXP_NAME"
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        "data.train_files='$DATA_DIR/train.parquet'" \
        "data.val_files='$DATA_DIR/test.parquet'" \
        data.train_batch_size=$BATCH_SIZE data.max_prompt_length=1024 data.max_response_length=8192 \
        data.filter_overlong_prompts=True "data.truncation='error'" \
        actor_rollout_ref.model.path=$MODEL actor_rollout_ref.actor.optim.lr=$LR \
        "actor_rollout_ref.actor.optim.betas=[0.9,0.999]" \
        "actor_rollout_ref.actor.checkpoint.save_contents='[\"hf_model\"]'" \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH \
        actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.n=$ROLLOUT_N actor_rollout_ref.rollout.temperature=0.9 \
        actor_rollout_ref.rollout.top_k=20 actor_rollout_ref.rollout.top_p=0.95 \
        actor_rollout_ref.actor.clip_ratio_low=0.2 actor_rollout_ref.actor.clip_ratio_high=0.28 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False trainer.critic_warmup=0 \
        trainer.logger='["console","wandb"]' trainer.project_name=verl_grpo_deepscaler \
        "trainer.experiment_name='$EXP_NAME'" "trainer.default_local_dir='$CKPT_ROOT/$EXP_NAME'" \
        trainer.n_gpus_per_node=$NGPUS trainer.nnodes=1 \
        trainer.save_freq=$SAVE_FREQ trainer.test_freq=5 trainer.total_epochs=$EPOCHS \
        $FREEZE_ARGS ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} 2>&1 | tee "$LOG_FILE"
}

should_run() {
    local n=$1
    if [[ -n "$ONLY" ]]; then echo "$ONLY" | tr ',' '\n' | grep -qx "$n"; else [[ $n -gt $SKIP ]]; fi
}

TOTAL=7
echo "============================================================"
echo "  Part 4/4: $MODEL_SHORT — Layers 21-27 ($TOTAL exp)"
echo "  Data: $DATA_DIR | GPUs: $GPUS ($NGPUS) | epochs=2 | 8K"
echo "============================================================"

should_run 1 && \
    run_train "${DATE}_exp1_layer21_${MODEL_SHORT}_deepscaler_lr5e-6" "$DATA_DIR" "5e-6" "+actor_rollout_ref.actor.train_layer_ids=21" && \
    echo "  [1/$TOTAL] Done." && echo ""

should_run 2 && \
    run_train "${DATE}_exp2_layer22_${MODEL_SHORT}_deepscaler_lr5e-6" "$DATA_DIR" "5e-6" "+actor_rollout_ref.actor.train_layer_ids=22" && \
    echo "  [2/$TOTAL] Done." && echo ""

should_run 3 && \
    run_train "${DATE}_exp3_layer23_${MODEL_SHORT}_deepscaler_lr5e-6" "$DATA_DIR" "5e-6" "+actor_rollout_ref.actor.train_layer_ids=23" && \
    echo "  [3/$TOTAL] Done." && echo ""

should_run 4 && \
    run_train "${DATE}_exp4_layer24_${MODEL_SHORT}_deepscaler_lr5e-6" "$DATA_DIR" "5e-6" "+actor_rollout_ref.actor.train_layer_ids=24" && \
    echo "  [4/$TOTAL] Done." && echo ""

should_run 5 && \
    run_train "${DATE}_exp5_layer25_${MODEL_SHORT}_deepscaler_lr5e-6" "$DATA_DIR" "5e-6" "+actor_rollout_ref.actor.train_layer_ids=25" && \
    echo "  [5/$TOTAL] Done." && echo ""

should_run 6 && \
    run_train "${DATE}_exp6_layer26_${MODEL_SHORT}_deepscaler_lr5e-6" "$DATA_DIR" "5e-6" "+actor_rollout_ref.actor.train_layer_ids=26" && \
    echo "  [6/$TOTAL] Done." && echo ""

should_run 7 && \
    run_train "${DATE}_exp7_layer27_${MODEL_SHORT}_deepscaler_lr5e-6" "$DATA_DIR" "5e-6" "+actor_rollout_ref.actor.train_layer_ids=27" && \
    echo "  [7/$TOTAL] Done." && echo ""

echo ""; echo "  Part 4/4 complete!"
DUMMY_RUN_NAME="dummy_dsr_p4_$(hostname)_$(date +%m%d_%H%M)" python3 "$SCRIPT_DIR/dummy_gpu_hold.py"
