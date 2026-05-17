#!/bin/bash
# ==============================================================================
# DAPO training: DeepSeek-R1-Distill-Qwen-1.5B on DAPO-Math-17K
# ==============================================================================
# Faithful to recipe/dapo/run_dapo_qwen2.5_32b.sh (the canonical "full DAPO"
# with dynamic sampling / group filtering), scaled down to a single 8-GPU node
# for the 1.5B model.
#
# User overrides vs DAPO defaults:
#   - Model         : DeepSeek-R1-Distill-Qwen-1.5B (was Qwen2.5-32B)
#   - n_resp_per_prompt (= group size) : 5  (DAPO default was 16)
#   - max_response_length              : 8192  (DAPO 32B used 20480; 7B used 8192)
#
# Everything else inherits canonical DAPO settings:
#   - GRPO + decoupled clip (low=0.2, high=0.28)
#   - No KL term (in reward or loss)
#   - Token-level loss aggregation
#   - Overlong buffer (4k buffer, penalty=1.0)
#   - Dynamic sampling (filter all-correct / all-wrong groups, gen_bsz = 3x train_bsz)
#   - lr=1e-6, warmup=10, wd=0.1
#   - temperature=1.0, top_p=1.0 for rollout; top_p=0.7 for val
#   - DAPO reward manager + dapo-math-17k parquet + AIME-2024 val
#
# Usage:
#   bash run_dapo_r1distill_qwen_1.5b_tmux.sh                        # full RL
#   bash run_dapo_r1distill_qwen_1.5b_tmux.sh --layers "10"          # single layer
#   bash run_dapo_r1distill_qwen_1.5b_tmux.sh --layers "0 6 12 18"   # sweep, sequential
#   bash run_dapo_r1distill_qwen_1.5b_tmux.sh --layers "full 10 12"  # full + per-layer sweep
#   bash run_dapo_r1distill_qwen_1.5b_tmux.sh --no-tmux
#   bash run_dapo_r1distill_qwen_1.5b_tmux.sh --model /local/path/to/R1-Distill-Qwen-1.5B
#   bash run_dapo_r1distill_qwen_1.5b_tmux.sh --ckpt-root /custom/dir
#
# Multi-node parallel sweep (--part k/N stride pattern, run one per node):
#   # Same --layers list on every node; each node gets a 1/N stride slice.
#   # Node 1 (does items 1, 1+N, 1+2N, ...):
#   bash run_dapo_..._tmux.sh --layers "full 0 4 8 12 16 20 24" --part 1/4
#   # Node 2:
#   bash run_dapo_..._tmux.sh --layers "full 0 4 8 12 16 20 24" --part 2/4
#   # ... and so on, up to --part 4/4 on node 4.
#   # 8 experiments / 4 nodes = 2 per node, runs sequential within each.

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJ_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
SCRIPT_NAME=$(basename "$0")

# ===== defaults =====
MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
CKPT_ROOT="${CKPT_ROOT:-$PROJ_DIR/checkpoints}"
DATA_DIR="${DATA_DIR:-$PROJ_DIR/data}"           # holds dapo-math-17k.parquet + aime-2024.parquet
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/dapo-math-17k.parquet}"
TEST_FILE="${TEST_FILE:-$DATA_DIR/aime-2024.parquet}"
CONDA_INIT="${CONDA_INIT:-/code/hongpaul-sandbox/cuda/miniconda3/bin/activate}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/code/hongpaul-sandbox/cuda/miniconda3/envs/cuda}"
WANDB_API_KEY="${WANDB_API_KEY:-b8f38344ec7231ee89baa74ef7209dd5a43df6b2}"
WANDB_ENTITY="${WANDB_ENTITY:-mhong-university-of-minnesota}"
NO_TMUX=false; EXTRA_ARGS=()
# Layer training: empty = single full-RL run. Use space-separated indices for
# a sweep. The literal token "full" runs the un-layered full-RL variant.
# e.g. --layers "full 0 6 12"  → 4 sequential experiments.
LAYERS="${LAYERS:-}"
# Multi-node stride slicing: --part k/N → this node runs the k-th 1/N slice
# of LAYERS (1-indexed). Default 1/1 = single node, run all.
PART_K=1; PART_N=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)       GPUS="$2"; shift 2 ;;
        --model)      MODEL="$2"; shift 2 ;;
        --ckpt-root)  CKPT_ROOT="$2"; shift 2 ;;
        --data-dir)   DATA_DIR="$2"; TRAIN_FILE="$DATA_DIR/dapo-math-17k.parquet"; TEST_FILE="$DATA_DIR/aime-2024.parquet"; shift 2 ;;
        --layers)     LAYERS="$2"; shift 2 ;;
        --part)       IFS='/' read -r PART_K PART_N <<< "$2"; shift 2 ;;
        --no-tmux)    NO_TMUX=true; shift ;;
        *)            EXTRA_ARGS+=("$1"); shift ;;
    esac
done
# Validate --part
if [[ ! "$PART_K" =~ ^[0-9]+$ ]] || [[ ! "$PART_N" =~ ^[0-9]+$ ]] || (( PART_K < 1 )) || (( PART_N < 1 )) || (( PART_K > PART_N )); then
    echo "ERROR: --part must be 'K/N' with 1 ≤ K ≤ N (got '$PART_K/$PART_N')"
    exit 1
fi

# ===== tmux auto-launch =====
if [[ -z "${TMUX:-}" ]] && [[ "$NO_TMUX" == "false" ]]; then
    PART_TAG=""
    (( PART_N > 1 )) && PART_TAG="_p${PART_K}of${PART_N}"
    TMUX_SESSION="dapo_r1d1.5b${PART_TAG}_$(date +%m%d_%H%M)"
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
    echo "[data] $TRAIN_FILE missing, downloading DAPO-Math-17k from HF..."
    wget -O "$TRAIN_FILE" "https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/resolve/main/data/dapo-math-17k.parquet?download=true"
fi
if [[ ! -f "$TEST_FILE" ]]; then
    echo "[data] $TEST_FILE missing, downloading AIME-2024 from HF..."
    wget -O "$TEST_FILE" "https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024/resolve/main/data/aime-2024.parquet?download=true"
fi

# ===== DAPO canonical hyper-params =====
project_name='LayerRLDAPO'
DATE=$(date +%m%d_%H%M)
MODEL_SHORT=$(basename "$MODEL")

adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024 * 2))      # 2048
max_response_length=$((1024 * 8))    # 8192 — user spec
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))    # 4096
overlong_penalty_factor=1.0
loss_agg_mode="token-mean"

# Dynamic sampling (DAPO "Full")
enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10
train_prompt_bsz=512
gen_prompt_bsz=$((train_prompt_bsz * 3))
n_resp_per_prompt=5                  # user spec — group size
train_prompt_mini_bsz=32

# Rollout decoding
temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.7

# Hardware (1.5B fits trivially on 1 GPU; keep sp=1, tp=1)
NGPUS_PER_NODE=$(echo "$GPUS" | tr ',' '\n' | wc -l)
sp_size=1
gen_tp=1
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
# Offloading is for large models (e.g. 32B). For 1.5B keep params/optim on GPU,
# otherwise UpdateAnalysisTracker's FSDP.summon_full_params asserts (it requires
# the FlatParameter to be on the compute device, not CPU-offloaded).
offload=False

# Save twice total — middle + end. Compute total step count from dataset size,
# then save_freq = ceil(total_steps / 2). With 1 epoch and train_prompt_bsz
# prompts/step, total_steps ≈ |train_set| / train_prompt_bsz.
TOTAL_EPOCHS=1
STEPS_PER_EPOCH=$(python3 -c "import pandas as pd; print(max(1, len(pd.read_parquet('$TRAIN_FILE')) // $train_prompt_bsz))")
TOTAL_STEPS=$((STEPS_PER_EPOCH * TOTAL_EPOCHS))
SAVE_FREQ=$(( (TOTAL_STEPS + 1) / 2 ))      # ceil(N/2) → saves at step ceil(N/2) and step N

export CUDA_VISIBLE_DEVICES=$GPUS
export WANDB_API_KEY WANDB_ENTITY

# Run one DAPO experiment. $1 = layer index (e.g. "10") or "full" / empty for
# no-layer-restriction (full RL).
run_one_dapo() {
    local LAYER_TAG="${1:-full}"
    local layer_args=()
    local exp_suffix="full"
    if [[ -n "$LAYER_TAG" && "$LAYER_TAG" != "full" ]]; then
        layer_args=(+actor_rollout_ref.actor.train_layer_ids="$LAYER_TAG")
        exp_suffix="layer${LAYER_TAG}"
    fi
    local exp_name="${DATE}_DAPO_${MODEL_SHORT}_${exp_suffix}_n${n_resp_per_prompt}_r8k"
    local CKPTS_DIR="${CKPT_ROOT}/${project_name}/${exp_name}"
    mkdir -p "$CKPTS_DIR"
    local LOG_FILE="$CKPTS_DIR/train.log"

    echo "============================================================"
    echo "  DAPO  —  $exp_name"
    echo "  Model       : $MODEL"
    echo "  Train       : $TRAIN_FILE"
    echo "  Val         : $TEST_FILE"
    echo "  GPUs        : $GPUS ($NGPUS_PER_NODE)"
    echo "  group_n     : $n_resp_per_prompt   max_resp: $max_response_length"
    echo "  freeze      : ${LAYER_TAG} ${layer_args[*]:-}"
    echo "  total_steps : $TOTAL_STEPS  (= $STEPS_PER_EPOCH steps/epoch × $TOTAL_EPOCHS epoch)"
    echo "  save_freq   : $SAVE_FREQ  → saves at step $SAVE_FREQ and step $TOTAL_STEPS"
    echo "  ckpts ->    : $CKPTS_DIR"
    echo "============================================================"

    python3 -m recipe.dapo.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.test_freq=5 \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    "${layer_args[@]}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} 2>&1 | tee "$LOG_FILE"
}

# ===== dispatch =====
if [[ -z "$LAYERS" ]]; then
    # No --layers: single full-RL run (--part ignored)
    run_one_dapo "full"
else
    # Sweep: split LAYERS via --part k/N stride. Node K gets items K-1, K-1+N, ...
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
        echo "  Single-node run (--part 1/1): all ${#MY_LAYERS[@]} on this box"
    fi
    echo "  (runs sequentially; each writes its own ckpt + wandb run)"
    echo "============================================================"
    if (( ${#MY_LAYERS[@]} == 0 )); then
        echo "  [skip] no items assigned to part $PART_K/$PART_N (LAYERS too short)"
        exit 0
    fi
    for L in "${MY_LAYERS[@]}"; do
        run_one_dapo "$L"
        echo ""
        echo "  [done] $L  (part $PART_K/$PART_N)"
        echo ""
    done
fi
