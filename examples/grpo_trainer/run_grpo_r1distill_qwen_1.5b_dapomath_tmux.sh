#!/bin/bash
# ==============================================================================
# GRPO training: DeepSeek-R1-Distill-Qwen-1.5B on open-r1/DAPO-Math-17k-Processed
# ==============================================================================
# Built on the stable run_4b_base_part1_tmux.sh recipe (verl GRPO with KL
# loss, no DAPO dynamic sampling, no token-level dynamic batching) — that
# script trains reliably and we want the same behaviour here.
#
# Per-user spec:
#   - Model           : DeepSeek-R1-Distill-Qwen-1.5B
#   - Data            : open-r1/DAPO-Math-17k-Processed (all config, 17,398 rows)
#   - max_response    : 8192 tokens
#   - Group size (n)  : 5
#
# Layer training:
#   - --layers "10"               single layer
#   - --layers "0 6 12"           sequential sweep
#   - --layers "full 10 12"       include full RL + per-layer
#
# Multi-node parallel:
#   - --part K/N    stride slicing; same --layers everywhere; tmux name has part
#   e.g. 4 nodes × 29 exps (full + layer 0-27) = 8 + 7 + 7 + 7 per node.
#
# Usage:
#   bash run_grpo_r1distill_qwen_1.5b_dapomath_tmux.sh                          # full RL
#   bash run_grpo_r1distill_qwen_1.5b_dapomath_tmux.sh --layers "full 0 6 12"
#   bash run_grpo_r1distill_qwen_1.5b_dapomath_tmux.sh --layers "full 0..27" --part 1/4
#   bash run_grpo_r1distill_qwen_1.5b_dapomath_tmux.sh --no-tmux
#   bash run_grpo_r1distill_qwen_1.5b_dapomath_tmux.sh --model /local/path/to/model

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJ_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
SCRIPT_NAME=$(basename "$0")

# ===== defaults =====
MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
CKPT_ROOT="${CKPT_ROOT:-$PROJ_DIR/checkpoints}"
DATA_DIR="${DATA_DIR:-$PROJ_DIR/data}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/dapo-math-17k-processed.parquet}"
TEST_FILE="${TEST_FILE:-$DATA_DIR/aime-2024.parquet}"
CONDA_INIT="${CONDA_INIT:-/code/hongpaul-sandbox/cuda/miniconda3/bin/activate}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/code/hongpaul-sandbox/cuda/miniconda3/envs/cuda}"
WANDB_API_KEY="${WANDB_API_KEY:-b8f38344ec7231ee89baa74ef7209dd5a43df6b2}"
WANDB_ENTITY="${WANDB_ENTITY:-mhong-university-of-minnesota}"
WANDB_PROJECT="${WANDB_PROJECT:-LayerRLGRPO}"
NO_TMUX=false; EXTRA_ARGS=()
LAYERS="${LAYERS:-}"
PART_K=1; PART_N=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)       GPUS="$2"; shift 2 ;;
        --model)      MODEL="$2"; shift 2 ;;
        --ckpt-root)  CKPT_ROOT="$2"; shift 2 ;;
        --data-dir)   DATA_DIR="$2"; TRAIN_FILE="$DATA_DIR/dapo-math-17k-processed.parquet"; TEST_FILE="$DATA_DIR/aime-2024.parquet"; shift 2 ;;
        --layers)     LAYERS="$2"; shift 2 ;;
        --part)       IFS='/' read -r PART_K PART_N <<< "$2"; shift 2 ;;
        --no-tmux)    NO_TMUX=true; shift ;;
        *)            EXTRA_ARGS+=("$1"); shift ;;
    esac
done
if [[ ! "$PART_K" =~ ^[0-9]+$ ]] || [[ ! "$PART_N" =~ ^[0-9]+$ ]] || (( PART_K < 1 )) || (( PART_N < 1 )) || (( PART_K > PART_N )); then
    echo "ERROR: --part must be 'K/N' with 1 ≤ K ≤ N (got '$PART_K/$PART_N')"
    exit 1
fi

# ===== tmux auto-launch =====
if [[ -z "${TMUX:-}" ]] && [[ "$NO_TMUX" == "false" ]]; then
    PART_TAG=""
    (( PART_N > 1 )) && PART_TAG="_p${PART_K}of${PART_N}"
    TMUX_SESSION="grpo_r1d1.5b${PART_TAG}_$(date +%m%d_%H%M)"
    FULL_ARGS="--no-tmux --gpus $(printf '%q' "$GPUS") --model $(printf '%q' "$MODEL") --ckpt-root $(printf '%q' "$CKPT_ROOT") --data-dir $(printf '%q' "$DATA_DIR")"
    [[ -n "$LAYERS" ]] && FULL_ARGS="$FULL_ARGS --layers $(printf '%q' "$LAYERS")"
    (( PART_N > 1 )) && FULL_ARGS="$FULL_ARGS --part ${PART_K}/${PART_N}"
    for arg in "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; do FULL_ARGS="$FULL_ARGS $(printf '%q' "$arg")"; done
    tmux new-session -d -s "$TMUX_SESSION" \
        "source $CONDA_INIT && conda activate $CONDA_ENV_PATH && cd $PROJ_DIR && bash $SCRIPT_DIR/$SCRIPT_NAME $FULL_ARGS; exec bash"
    echo "Tmux '$TMUX_SESSION' started.  Attach: tmux attach -t $TMUX_SESSION"; exit 0
fi

# ===== preflight: auto-download data if missing =====
mkdir -p "$DATA_DIR" "$CKPT_ROOT"
if [[ ! -f "$TRAIN_FILE" ]]; then
    echo "[data] downloading open-r1/DAPO-Math-17k-Processed (all, 17,398 rows)..."
    wget -O "$TRAIN_FILE" "https://huggingface.co/datasets/open-r1/DAPO-Math-17k-Processed/resolve/main/all/train-00000-of-00001.parquet?download=true"
fi
if [[ ! -f "$TEST_FILE" ]]; then
    echo "[data] downloading AIME-2024 (val)..."
    wget -O "$TEST_FILE" "https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024/resolve/main/data/aime-2024.parquet?download=true"
fi

# ===== hyper-params (matched to run_4b_base_part1_tmux.sh) =====
NGPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
MODEL_SHORT=$(basename "$MODEL")
DATE=$(date +%m%d_%H%M)

# Halved from run_4b_base defaults (was 512/128/8) — R1-Distill produces long
# CoT (~4-8K tokens) so smaller batches reduce KV cache pressure during the
# rollout phase and ease NCCL load.
BATCH_SIZE=256
MINI_BATCH=64
MICRO_BATCH=4
ROLLOUT_N=5
EPOCHS="${EPOCHS:-5}"

# Save twice: mid + end. STEPS_PER_EPOCH = data_size / BATCH_SIZE (matches verl
# when gen_batch_size is NOT set, which is our case — GRPO uses train_batch_size).
STEPS_PER_EPOCH=$(python3 -c "import pandas as pd; print(max(1, len(pd.read_parquet('$TRAIN_FILE')) // $BATCH_SIZE))")
TOTAL_STEPS=$((STEPS_PER_EPOCH * EPOCHS))
SAVE_FREQ=$(( (TOTAL_STEPS + 1) / 2 ))

export CUDA_VISIBLE_DEVICES=$GPUS
export WANDB_API_KEY WANDB_ENTITY

# Run one GRPO experiment. $1 = layer index/csv (e.g. "10", "0,14") or "full".
run_one_grpo() {
    local LAYER_TAG="${1:-full}"
    local layer_args=()
    local exp_suffix="full"
    local lr="1e-6"
    if [[ -n "$LAYER_TAG" && "$LAYER_TAG" != "full" ]]; then
        layer_args=(+actor_rollout_ref.actor.train_layer_ids="$LAYER_TAG")
        exp_suffix="layer${LAYER_TAG}"
        lr="5e-6"   # higher LR for layer-only (matches run_4b_base_part1)
    fi
    local exp_name="${DATE}_GRPO_${MODEL_SHORT}_${exp_suffix}_n${ROLLOUT_N}_r8k_lr${lr}"
    local CKPTS_DIR="${CKPT_ROOT}/${WANDB_PROJECT}/${exp_name}"
    mkdir -p "$CKPTS_DIR"
    local LOG_FILE="$CKPTS_DIR/train.log"
    export VERL_DEFAULT_LOCAL_DIR="$CKPTS_DIR"

    echo "============================================================"
    echo "  GRPO  —  $exp_name"
    echo "  Model        : $MODEL"
    echo "  Train        : $TRAIN_FILE"
    echo "  Val          : $TEST_FILE"
    echo "  GPUs         : $GPUS ($NGPUS)"
    echo "  batch/mini/μ : $BATCH_SIZE / $MINI_BATCH / $MICRO_BATCH"
    echo "  rollout n    : $ROLLOUT_N    max_resp: 8192"
    echo "  freeze       : ${LAYER_TAG} ${layer_args[*]:-}   lr=${lr}"
    echo "  total_steps  : $TOTAL_STEPS  (= $STEPS_PER_EPOCH steps/epoch × $EPOCHS epoch)"
    echo "  save_freq    : $SAVE_FREQ   → saves at step $SAVE_FREQ and step $TOTAL_STEPS"
    echo "  ckpts        : $CKPTS_DIR"
    echo "============================================================"

    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        "data.train_files='$TRAIN_FILE'" \
        "data.val_files='$TEST_FILE'" \
        data.train_batch_size=$BATCH_SIZE \
        data.max_prompt_length=1024 \
        data.max_response_length=8192 \
        data.filter_overlong_prompts=True \
        "data.truncation='error'" \
        actor_rollout_ref.model.path=$MODEL \
        actor_rollout_ref.actor.optim.lr=$lr \
        "actor_rollout_ref.actor.optim.betas=[0.9,0.999]" \
        "actor_rollout_ref.actor.checkpoint.save_contents='[\"hf_model\"]'" \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.clip_ratio_low=0.2 \
        actor_rollout_ref.actor.clip_ratio_high=0.28 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
        actor_rollout_ref.rollout.n=$ROLLOUT_N \
        actor_rollout_ref.rollout.temperature=0.9 \
        actor_rollout_ref.rollout.top_k=20 \
        actor_rollout_ref.rollout.top_p=0.95 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        trainer.critic_warmup=0 \
        trainer.logger='["console","wandb"]' \
        trainer.project_name="$WANDB_PROJECT" \
        "trainer.experiment_name='$exp_name'" \
        "trainer.default_local_dir='$CKPTS_DIR'" \
        trainer.n_gpus_per_node=$NGPUS \
        trainer.nnodes=1 \
        trainer.save_freq=$SAVE_FREQ \
        trainer.test_freq=5 \
        trainer.total_epochs=$EPOCHS \
        trainer.resume_mode=auto \
        "${layer_args[@]}" \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} 2>&1 | tee "$LOG_FILE"
}

# ===== dispatch =====
if [[ -z "$LAYERS" ]]; then
    run_one_grpo "full"
else
    read -ra LAYER_ARR <<< "$LAYERS"
    MY_LAYERS=()
    for ((i = PART_K - 1; i < ${#LAYER_ARR[@]}; i += PART_N)); do
        MY_LAYERS+=("${LAYER_ARR[$i]}")
    done
    echo ""
    echo "============================================================"
    echo "  Sweep total ${#LAYER_ARR[@]} setting(s): ${LAYER_ARR[*]}"
    if (( PART_N > 1 )); then
        echo "  --part $PART_K/$PART_N → this node runs ${#MY_LAYERS[@]} of them: ${MY_LAYERS[*]}"
    else
        echo "  Single-node run: all ${#MY_LAYERS[@]} on this box"
    fi
    echo "============================================================"
    if (( ${#MY_LAYERS[@]} == 0 )); then
        echo "  [skip] no items assigned to part $PART_K/$PART_N (LAYERS too short)"
        exit 0
    fi
    for L in "${MY_LAYERS[@]}"; do
        run_one_grpo "$L"
        echo ""
        echo "  [done] $L  (part $PART_K/$PART_N)"
        echo ""
    done
fi
