#!/bin/bash
# ==============================================================================
# Qwen3-4B-Base — simple boost / only / worst sweep (8 experiments, 8 GPUs)
# ==============================================================================
#
# Mirrors top with worst (top5/top10 × boost/only, then worst5/worst10 × boost/only).
#
#   1. top5    boost   (16/14/19/22/17)                base_lr=1e-6  boost_lr=2e-6
#   2. top10   boost   (16/14/19/22/17/15/12/18/20/10) base_lr=1e-6  boost_lr=2e-6
#   3. top5    only    (same TOP5 set, freeze rest)    lr=2e-6
#   4. top10   only    (same TOP10 set, freeze rest)   lr=2e-6
#   5. worst5  boost   (2/3/1/30/32)                   base_lr=1e-6  boost_lr=2e-6
#   6. worst10 boost   (2/3/1/30/32/29/27/33/4/28)     base_lr=1e-6  boost_lr=2e-6
#   7. worst5  only    (WORST5 set, freeze rest)       lr=2e-6
#   8. worst10 only    (WORST10 set, freeze rest)      lr=2e-6
#
# Layer ranking source — Qwen3-4B-Base 8K eval, math_avg (no AIME):
#   /home/zha00175/csvdoc/wandb_opt_rl_eval_analysis_final.xlsx
#   Full RL math_avg = 0.6365  (highest single layer L16 = 0.6432)
#   Sweet spot L10-L22 (mid-band).  Worst tails: L1-L4 + L27-L33.
#
# batch=512, mini=128, micro=8, epochs=2, max_response=3072.
# 8 GPUs (default 0-7), runs one exp at a time.
#
# Usage:
#   bash run_4b_base_simple_boost_only_tmux.sh
#   bash run_4b_base_simple_boost_only_tmux.sh --skip 4     # skip top phases
#   bash run_4b_base_simple_boost_only_tmux.sh --only 1,5   # 1 top + 1 worst
#   bash run_4b_base_simple_boost_only_tmux.sh --no-tmux

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJ_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
SCRIPT_NAME=$(basename "$0")

MODEL="Qwen/Qwen3-4B-Base"
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
    TMUX_SESSION="simple_boost_4b_$(date +%m%d_%H%M)"
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
    # MODE = "boost"  -> base_lr everywhere + boost_lr on LAYER_IDS (current behaviour)
    # MODE = "only"   -> train ONLY LAYER_IDS at LR=BOOST_LR, freeze the rest
    local EXP_NAME="$1" LAYER_IDS="$2" BOOST_LR="$3"
    local BASE_LR="${4:-1e-6}" MODE="${5:-boost}"
    local BATCH_SIZE="${6:-512}" MINI_BATCH="${7:-128}" MICRO_BATCH="${8:-8}"
    local ROLLOUT_N="${9:-5}" EPOCHS="${10:-2}" SAVE_FREQ="${11:--1}"
    local STEPS_PER_EPOCH
    STEPS_PER_EPOCH=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('$DATA_DIR/train.parquet')) // $BATCH_SIZE)")
    local TOTAL_STEPS=$((STEPS_PER_EPOCH * EPOCHS))
    [[ "$SAVE_FREQ" == "-1" ]] && SAVE_FREQ=$TOTAL_STEPS
    mkdir -p "$CKPT_ROOT/$EXP_NAME"
    local LOG_FILE="$CKPT_ROOT/$EXP_NAME/train.log"

    # Build mode-specific overrides
    local LAYER_OVERRIDES
    local LR_FOR_OPTIM
    if [[ "$MODE" == "only" ]]; then
        LAYER_OVERRIDES=("+actor_rollout_ref.actor.train_layer_ids='$LAYER_IDS'")
        LR_FOR_OPTIM="$BOOST_LR"
        echo "  ---- $EXP_NAME ----"
        echo "  Model=$MODEL  mode=only  lr=$BOOST_LR  train_layer_ids=[$LAYER_IDS]"
    else
        LAYER_OVERRIDES=(
            "+actor_rollout_ref.actor.boost_layer_ids='$LAYER_IDS'"
            "+actor_rollout_ref.actor.boost_lr=$BOOST_LR"
        )
        LR_FOR_OPTIM="$BASE_LR"
        echo "  ---- $EXP_NAME ----"
        echo "  Model=$MODEL  mode=boost  base_lr=$BASE_LR  boost_lr=$BOOST_LR  boost_ids=[$LAYER_IDS]"
    fi
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
        actor_rollout_ref.model.path=$MODEL actor_rollout_ref.actor.optim.lr=$LR_FOR_OPTIM \
        "${LAYER_OVERRIDES[@]}" \
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

# Layer rankings for Qwen3-4B-Base (math_avg from 8K eval, full RL = 0.6365)
TOP5="16,14,19,22,17"
TOP10="16,14,19,22,17,15,12,18,20,10"
WORST5="2,3,1,30,32"
WORST10="2,3,1,30,32,29,27,33,4,28"

#   exp_num | tag | N | layer_ids | boost_lr | base_lr | mode | layer_kind
declare -a EXPS=(
    "1|top|5|$TOP5|2e-6|1e-6|boost|top"
    "2|top|10|$TOP10|2e-6|1e-6|boost|top"
    "3|top|5|$TOP5|2e-6|1e-6|only|top"
    "4|top|10|$TOP10|2e-6|1e-6|only|top"
    "5|wst|5|$WORST5|2e-6|1e-6|boost|worst"
    "6|wst|10|$WORST10|2e-6|1e-6|boost|worst"
    "7|wst|5|$WORST5|2e-6|1e-6|only|worst"
    "8|wst|10|$WORST10|2e-6|1e-6|only|worst"
)

TOTAL=${#EXPS[@]}
echo "============================================================"
echo "  Simple sweep: $MODEL_SHORT on NuminaMath-CoT  ($TOTAL exp sequential)"
echo "  1. top5    boost   base_lr=1e-6 + boost_lr=2e-6"
echo "  2. top10   boost   base_lr=1e-6 + boost_lr=2e-6"
echo "  3. top5    only    lr=2e-6"
echo "  4. top10   only    lr=2e-6"
echo "  5. worst5  boost   base_lr=1e-6 + boost_lr=2e-6"
echo "  6. worst10 boost   base_lr=1e-6 + boost_lr=2e-6"
echo "  7. worst5  only    lr=2e-6"
echo "  8. worst10 only    lr=2e-6"
echo "  GPUs: $GPUS ($NGPUS) | epochs=2"
echo "  Use --skip N or --only 1,3,5 to control which exps run."
echo "============================================================"

for row in "${EXPS[@]}"; do
    IFS='|' read -r EXP_NUM TAG TOP_N LAYER_IDS BOOST_LR BASE_LR MODE LAYER_KIND <<< "$row"
    if [[ "$MODE" == "only" ]]; then
        EXP_NAME="${DATE}_exp${EXP_NUM}_only_${LAYER_KIND}${TOP_N}_${MODEL_SHORT}_numina_cot_lr${BOOST_LR}"
    else
        EXP_NAME="${DATE}_exp${EXP_NUM}_boost_${LAYER_KIND}${TOP_N}_${MODEL_SHORT}_numina_cot_bst${BOOST_LR}_base${BASE_LR}"
    fi
    if should_run $EXP_NUM; then
        run_train "$EXP_NAME" "$LAYER_IDS" "$BOOST_LR" "$BASE_LR" "$MODE"
        echo "  [$EXP_NUM/$TOTAL] Done.  Cleaning up ray/vllm before next exp..."
        # Safe to use node-wide since we're the only job running on this node
        ray stop --force 2>/dev/null || true
        pkill -9 -f main_ppo 2>/dev/null || true
        pkill -9 -f vllm 2>/dev/null || true
        sleep 25
        echo ""
    fi
done

echo ""; echo "  Sweep complete!"
