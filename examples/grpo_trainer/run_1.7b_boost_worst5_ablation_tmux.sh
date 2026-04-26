#!/bin/bash
# ==============================================================================
# Ablation: Qwen3-1.7B-Base boost the WORST N layers (expect degradation)
# ==============================================================================
#
# Ablation design:
#   - Main sweep boosts TOP N layers (best math_avg) — shows gain over full RL.
#   - This script boosts WORST N layers (worst math_avg) — should show
#     degradation, proving that LAYER SELECTION matters (not just "more LR
#     on any subset").
#
# Experiments (3 total — single boost_lr=2e-6 to match main sweep):
#   1. worst5  boost_lr=2e-6   (paired with main sweep exp1: top5 @ 2e-6)
#   2. worst1  boost_lr=2e-6   (paired with main sweep exp2: top1 @ 2e-6)
#   3. worst10 boost_lr=2e-6   (paired with main sweep exp3: top10 @ 2e-6)
#
# Worst-N selection (single-layer math_avg, no AIME, ascending):
#   worst1  = [25]                                        — L25 is the single
#                                                           worst (0.4714)
#   worst5  = [25, 24, 26, 22, 21]                        — math_avg range
#                                                           0.4714-0.4797
#   worst10 = [25, 24, 26, 22, 21, 23, 27, 18, 3, 17]     — math_avg range
#                                                           0.4714-0.4937
#
# Reference: full RL baseline math_avg = 0.5122 (much higher than worst-N
# layers' single-layer scores).
#
# base_lr=1e-6, batch=512, mini=128, micro=8, epochs=2, max_response=3072.
# 8 GPUs sequential. Order: worst5 first (main baseline pair), then worst1,
# then worst10.

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJ_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
SCRIPT_NAME=$(basename "$0")

MODEL="Qwen/Qwen3-1.7B-Base"
GPUS="0,1,2,3,4,5,6,7"
CKPT_ROOT="$PROJ_DIR/checkpoints"
DATA_DIR="$PROJ_DIR/data/numina_math_cot_author"
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
    TMUX_SESSION="boost_worst_abl_$(date +%m%d_%H%M)"
    FULL_ARGS="--no-tmux --gpus $(printf '%q' "$GPUS") --model $(printf '%q' "$MODEL") --ckpt-root $(printf '%q' "$CKPT_ROOT") --data-dir $(printf '%q' "$DATA_DIR")"
    [[ $SKIP -gt 0 ]] && FULL_ARGS="$FULL_ARGS --skip $SKIP"
    [[ -n "$ONLY" ]] && FULL_ARGS="$FULL_ARGS --only $(printf '%q' "$ONLY")"
    for arg in "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; do FULL_ARGS="$FULL_ARGS $(printf '%q' "$arg")"; done
    tmux new-session -d -s "$TMUX_SESSION" \
        "source $CONDA_INIT && conda activate $CONDA_ENV_PATH && cd $PROJ_DIR && bash $SCRIPT_DIR/$SCRIPT_NAME $FULL_ARGS; exec bash"
    echo "Tmux '$TMUX_SESSION' started.  Attach: tmux attach -t $TMUX_SESSION"; exit 0
fi

NGPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
MODEL_SHORT=$(basename "$MODEL")
DATE=$(date +%m%d_%H%M)
[[ ! -f "$DATA_DIR/train.parquet" ]] && echo "ERROR: $DATA_DIR/train.parquet not found" && exit 1

run_train() {
    local EXP_NAME="$1" BOOST_IDS="$2" BOOST_LR="$3"
    local BASE_LR="${4:-1e-6}"
    local BATCH_SIZE="${5:-512}" MINI_BATCH="${6:-128}" MICRO_BATCH="${7:-8}"
    local ROLLOUT_N="${8:-5}" EPOCHS="${9:-2}" SAVE_FREQ="${10:--1}"
    local STEPS_PER_EPOCH
    STEPS_PER_EPOCH=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('$DATA_DIR/train.parquet')) // $BATCH_SIZE)")
    local TOTAL_STEPS=$((STEPS_PER_EPOCH * EPOCHS))
    [[ "$SAVE_FREQ" == "-1" ]] && SAVE_FREQ=$TOTAL_STEPS
    mkdir -p "$CKPT_ROOT/$EXP_NAME"
    local LOG_FILE="$CKPT_ROOT/$EXP_NAME/train.log"
    echo "  ---- $EXP_NAME ----"
    echo "  Model=$MODEL  base_lr=$BASE_LR  boost_lr=$BOOST_LR  boost_ids=[$BOOST_IDS]"
    echo "  Epochs=$EPOCHS  Steps=$TOTAL_STEPS  GPUs=$NGPUS"
    export CUDA_VISIBLE_DEVICES=$GPUS
    export WANDB_API_KEY="${WANDB_API_KEY:-b8f38344ec7231ee89baa74ef7209dd5a43df6b2}"
    export WANDB_ENTITY="${WANDB_ENTITY:-mhong-university-of-minnesota}"
    export VERL_DEFAULT_LOCAL_DIR="$CKPT_ROOT/$EXP_NAME"
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        "data.train_files='$DATA_DIR/train.parquet'" \
        "data.val_files='$DATA_DIR/test.parquet'" \
        data.train_batch_size=$BATCH_SIZE data.max_prompt_length=1024 data.max_response_length=3072 \
        data.filter_overlong_prompts=True "data.truncation='error'" \
        actor_rollout_ref.model.path=$MODEL actor_rollout_ref.actor.optim.lr=$BASE_LR \
        "+actor_rollout_ref.actor.boost_layer_ids='$BOOST_IDS'" \
        +actor_rollout_ref.actor.boost_lr=$BOOST_LR \
        actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
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
        trainer.logger='["console","wandb"]' trainer.project_name=verl_grpo_numina_cot_boost \
        "trainer.experiment_name='$EXP_NAME'" "trainer.default_local_dir='$CKPT_ROOT/$EXP_NAME'" \
        trainer.n_gpus_per_node=$NGPUS trainer.nnodes=1 \
        trainer.save_freq=$SAVE_FREQ trainer.test_freq=5 trainer.total_epochs=$EPOCHS \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} 2>&1 | tee "$LOG_FILE"
}

should_run() {
    local n=$1
    if [[ -n "$ONLY" ]]; then echo "$ONLY" | tr ',' '\n' | grep -qx "$n"; else [[ $n -gt $SKIP ]]; fi
}

# Worst-N layer sets (by single-layer math_avg, no AIME, ascending order)
WORST1="25"
WORST5="25,24,26,22,21"
WORST10="25,24,26,22,21,23,27,18,3,17"

#   exp_num | top_N | layer_ids | boost_lr | base_lr
declare -a EXPS=(
    "1|5|$WORST5|2e-6|1e-6"
    "2|1|$WORST1|2e-6|1e-6"
    "3|10|$WORST10|2e-6|1e-6"
)

TOTAL=${#EXPS[@]}
echo "============================================================"
echo "  Ablation: boost WORST N layers  ($TOTAL exp sequential)"
echo "  Paired with main sweep's top{5,1,10} exp1-3 for comparison."
echo "  Expected: degradation vs main sweep + degradation vs full RL."
echo ""
echo "  worst1  = [25]"
echo "  worst5  = [25, 24, 26, 22, 21]"
echo "  worst10 = [25, 24, 26, 22, 21, 23, 27, 18, 3, 17]"
echo ""
echo "  GPUs: $GPUS ($NGPUS) | base_lr=1e-6 | boost_lr=2e-6 | epochs=2"
echo "============================================================"

for row in "${EXPS[@]}"; do
    IFS='|' read -r EXP_NUM TOP_N LAYER_IDS BOOST_LR BASE_LR <<< "$row"
    EXP_NAME="${DATE}_abl_exp${EXP_NUM}_boost_worst${TOP_N}_${MODEL_SHORT}_numina_cot_bst${BOOST_LR}_base${BASE_LR}"
    if should_run $EXP_NUM; then
        run_train "$EXP_NAME" "$LAYER_IDS" "$BOOST_LR" "$BASE_LR"
        echo "  [$EXP_NUM/$TOTAL] Done.  Cleaning up..."
        ray stop --force 2>/dev/null || true
        pkill -9 -f main_ppo 2>/dev/null || true
        pkill -9 -f vllm 2>/dev/null || true
        sleep 25
        echo ""
    fi
done

echo ""; echo "  Worst-N boost ablation complete!"
