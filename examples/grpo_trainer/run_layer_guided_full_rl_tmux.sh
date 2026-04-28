#!/bin/bash
# ==============================================================================
# Qwen3-1.7B-Base — layer-ranking-guided FULL RL  (8 GPUs, sequential)
# ==============================================================================
#
# Uses the per-layer math_avg ranking we already know to drive 3 strategies:
#
#   Phase A — Freeze the worst-N layers, train everything else at full LR.
#       1. lg1  freeze WORST5  + train rest @ 1e-6
#       2. lg2  freeze WORST10 + train rest @ 1e-6
#
#   Phase C — Warm-start from only-top-N ckpt, then continue full RL at low LR.
#       3. lg3  init = only_top5  ckpt,  full RL @ 5e-7
#       4. lg4  init = only_top10 ckpt,  full RL @ 5e-7
#
#   Phase B — Combined: freeze worst-5  +  boost top-5  +  base elsewhere.
#       5. lg5  freeze WORST5  + boost TOP5 @ 2e-6  + base @ 1e-6
#
# Then enters a DUMMY LR-sweep loop on the lg1 design (freeze worst-5 full RL)
# cycling LR ∈ {3e-7, 5e-7, 7e-7, 1.5e-6, 3e-6, 5e-6}. Pass --no-dummy to skip.
#
# Layer rankings (Qwen3-1.7B-Base, 8K eval, math_avg):
#   TOP5    = 10,9,16,12,13
#   TOP10   = 10,9,16,12,13,2,7,14,15,11
#   WORST5  = 25,24,26,22,21
#   WORST10 = 25,24,26,22,21,23,27,18,3,17
#
# Usage:
#   bash run_layer_guided_full_rl_tmux.sh
#   bash run_layer_guided_full_rl_tmux.sh --dry-run
#   bash run_layer_guided_full_rl_tmux.sh --no-dummy
#   bash run_layer_guided_full_rl_tmux.sh --skip 2          # skip first N
#   bash run_layer_guided_full_rl_tmux.sh --only 1,3,5      # run only listed
#   bash run_layer_guided_full_rl_tmux.sh --no-tmux         # already inside tmux
#   bash run_layer_guided_full_rl_tmux.sh --gpus 0,1,2,3
#   bash run_layer_guided_full_rl_tmux.sh --ckpt-root /path
#   bash run_layer_guided_full_rl_tmux.sh \
#        --only-top5-ckpt /abs/path/to/only_top5/global_step_134/actor/huggingface \
#        --only-top10-ckpt /abs/path/to/only_top10/global_step_134/actor/huggingface

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJ_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
SCRIPT_NAME=$(basename "$0")

MODEL="Qwen/Qwen3-1.7B-Base"
GPUS="0,1,2,3,4,5,6,7"
CKPT_ROOT="${CKPT_ROOT:-$PROJ_DIR/checkpoints}"
DATA_DIR="$PROJ_DIR/data/numina_math_cot_author"
CONDA_INIT="${CONDA_INIT:-/code/hongpaul-sandbox/cuda/miniconda3/bin/activate}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/code/hongpaul-sandbox/cuda/miniconda3/envs/cuda}"
NUM_LAYERS=28

# Layer rankings — Qwen3-1.7B-Base 8K eval, math_avg (no AIME)
TOP5="10,9,16,12,13"
TOP10="10,9,16,12,13,2,7,14,15,11"
WORST5="25,24,26,22,21"
WORST10="25,24,26,22,21,23,27,18,3,17"

# Stage-1 ckpt paths used by Phase C (warm-start). If empty, auto-detected by
# globbing CKPT_ROOT for *only_top5_*Qwen3-1.7B-Base* / *only_top10_*.
ONLY_TOP5_CKPT=""
ONLY_TOP10_CKPT=""

DRY_RUN=false
NO_TMUX=false
NO_DUMMY=false
SKIP=0
ONLY=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)              GPUS="$2"; shift 2 ;;
        --model)             MODEL="$2"; shift 2 ;;
        --ckpt-root)         CKPT_ROOT="$2"; shift 2 ;;
        --data-dir)          DATA_DIR="$2"; shift 2 ;;
        --skip)              SKIP="$2"; shift 2 ;;
        --only)              ONLY="$2"; shift 2 ;;
        --dry-run)           DRY_RUN=true; shift ;;
        --no-tmux)           NO_TMUX=true; shift ;;
        --no-dummy)          NO_DUMMY=true; shift ;;
        --only-top5-ckpt)    ONLY_TOP5_CKPT="$2"; shift 2 ;;
        --only-top10-ckpt)   ONLY_TOP10_CKPT="$2"; shift 2 ;;
        *)                   EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ---------------- Tmux auto-launch ----------------
if [[ -z "${TMUX:-}" ]] && [[ "$NO_TMUX" == "false" ]]; then
    TMUX_SESSION="lg_full_rl_$(date +%m%d_%H%M)"
    FULL_ARGS="--no-tmux --gpus $(printf '%q' "$GPUS") --model $(printf '%q' "$MODEL") --ckpt-root $(printf '%q' "$CKPT_ROOT") --data-dir $(printf '%q' "$DATA_DIR")"
    [[ $SKIP -gt 0 ]]                && FULL_ARGS="$FULL_ARGS --skip $SKIP"
    [[ -n "$ONLY" ]]                 && FULL_ARGS="$FULL_ARGS --only $(printf '%q' "$ONLY")"
    [[ -n "$ONLY_TOP5_CKPT" ]]       && FULL_ARGS="$FULL_ARGS --only-top5-ckpt $(printf '%q' "$ONLY_TOP5_CKPT")"
    [[ -n "$ONLY_TOP10_CKPT" ]]      && FULL_ARGS="$FULL_ARGS --only-top10-ckpt $(printf '%q' "$ONLY_TOP10_CKPT")"
    $DRY_RUN  && FULL_ARGS="$FULL_ARGS --dry-run"
    $NO_DUMMY && FULL_ARGS="$FULL_ARGS --no-dummy"
    for arg in "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; do FULL_ARGS="$FULL_ARGS $(printf '%q' "$arg")"; done
    tmux new-session -d -s "$TMUX_SESSION" \
        "source $CONDA_INIT && conda activate $CONDA_ENV_PATH && cd $PROJ_DIR && bash $SCRIPT_DIR/$SCRIPT_NAME $FULL_ARGS; exec bash"
    echo "Tmux '$TMUX_SESSION' started.  Attach: tmux attach -t $TMUX_SESSION"
    exit 0
fi

NGPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
MODEL_SHORT=$(basename "$MODEL")
DATE=$(date +%m%d_%H%M)
[[ ! -f "$DATA_DIR/train.parquet" ]] && echo "ERROR: $DATA_DIR/train.parquet not found" && exit 1

# ---------------- Helpers ----------------

# all_layers_except "25,24,26,22,21" 28
#   -> "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,23,27,embed,norm,lm_head"
all_layers_except() {
    local exclude_csv="$1" n_layers="$2"
    declare -A exclude_set
    IFS=',' read -ra exclude_arr <<< "$exclude_csv"
    for x in "${exclude_arr[@]}"; do exclude_set[$x]=1; done
    local out=()
    for ((i=0; i<n_layers; i++)); do
        [[ -z "${exclude_set[$i]:-}" ]] && out+=("$i")
    done
    out+=("embed" "norm" "lm_head")
    (IFS=,; echo "${out[*]}")
}

# resolve_ckpt <pattern> -> echoes most recently modified ckpt path or "" if none.
resolve_ckpt() {
    local pat="$1"
    local best=""
    local best_mtime=0
    shopt -s nullglob
    for d in "$CKPT_ROOT"/$pat; do
        [[ -d "$d" ]] || continue
        # find max global_step_N/actor/huggingface/config.json
        local cand="" cand_num=-1
        for step_dir in "$d"/global_step_*; do
            [[ -d "$step_dir" ]] || continue
            local num="${step_dir##*global_step_}"
            [[ "$num" =~ ^[0-9]+$ ]] || continue
            if [[ -f "$step_dir/actor/huggingface/config.json" ]] && (( num > cand_num )); then
                cand="$step_dir/actor/huggingface"
                cand_num=$num
            fi
        done
        [[ -z "$cand" ]] && continue
        local mtime
        mtime=$(stat -c %Y "$cand" 2>/dev/null || echo 0)
        if (( mtime > best_mtime )); then
            best="$cand"
            best_mtime=$mtime
        fi
    done
    shopt -u nullglob
    echo "$best"
}

# Auto-detect only-top5 / only-top10 ckpts unless user supplied them.
if [[ -z "$ONLY_TOP5_CKPT" ]]; then
    ONLY_TOP5_CKPT=$(resolve_ckpt "*only_top5_${MODEL_SHORT}*")
fi
if [[ -z "$ONLY_TOP10_CKPT" ]]; then
    ONLY_TOP10_CKPT=$(resolve_ckpt "*only_top10_${MODEL_SHORT}*")
fi

# ---------------- run_train ----------------
# Args: EXP_NAME MODEL_PATH LR TRAIN_LAYER_IDS BOOST_LAYER_IDS BOOST_LR
#  - MODEL_PATH    "" -> use $MODEL
#  - TRAIN_LAYER_IDS "" -> train every layer
#  - BOOST_LAYER_IDS "" -> no boost group
run_train() {
    local EXP_NAME="$1" MODEL_PATH="${2:-}" LR="$3"
    local TRAIN_IDS="${4:-}" BOOST_IDS="${5:-}" BOOST_LR="${6:-}"
    local BATCH_SIZE="${7:-512}" MINI_BATCH="${8:-128}" MICRO_BATCH="${9:-8}"
    local ROLLOUT_N="${10:-5}" EPOCHS="${11:-2}" SAVE_FREQ="${12:--1}"
    [[ -z "$MODEL_PATH" ]] && MODEL_PATH="$MODEL"
    local STEPS_PER_EPOCH
    STEPS_PER_EPOCH=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('$DATA_DIR/train.parquet')) // $BATCH_SIZE)")
    local TOTAL_STEPS=$((STEPS_PER_EPOCH * EPOCHS))
    [[ "$SAVE_FREQ" == "-1" ]] && SAVE_FREQ=$TOTAL_STEPS
    mkdir -p "$CKPT_ROOT/$EXP_NAME"
    local LOG_FILE="$CKPT_ROOT/$EXP_NAME/train.log"

    local LAYER_OVERRIDES=()
    [[ -n "$TRAIN_IDS" ]] && LAYER_OVERRIDES+=("+actor_rollout_ref.actor.train_layer_ids='$TRAIN_IDS'")
    if [[ -n "$BOOST_IDS" && -n "$BOOST_LR" ]]; then
        LAYER_OVERRIDES+=("+actor_rollout_ref.actor.boost_layer_ids='$BOOST_IDS'")
        LAYER_OVERRIDES+=("+actor_rollout_ref.actor.boost_lr=$BOOST_LR")
    fi

    echo "  ---- $EXP_NAME ----"
    echo "  MODEL_PATH=$MODEL_PATH"
    echo "  LR=$LR  TRAIN_IDS=${TRAIN_IDS:-<all>}  BOOST_IDS=${BOOST_IDS:-<none>}  BOOST_LR=${BOOST_LR:-<none>}"
    echo "  Epochs=$EPOCHS  Steps=$TOTAL_STEPS  GPUs=$NGPUS"

    export CUDA_VISIBLE_DEVICES=$GPUS
    export WANDB_API_KEY="${WANDB_API_KEY:-b8f38344ec7231ee89baa74ef7209dd5a43df6b2}"
    export WANDB_ENTITY="${WANDB_ENTITY:-mhong-university-of-minnesota}"
    export VERL_DEFAULT_LOCAL_DIR="$CKPT_ROOT/$EXP_NAME"
    # Avoid flashinfer JIT (broken ninja build on this box) — fall back to torch sampler.
    export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        "data.train_files='$DATA_DIR/train.parquet'" \
        "data.val_files='$DATA_DIR/test.parquet'" \
        data.train_batch_size=$BATCH_SIZE data.max_prompt_length=1024 data.max_response_length=3072 \
        data.filter_overlong_prompts=True "data.truncation='error'" \
        actor_rollout_ref.model.path="$MODEL_PATH" actor_rollout_ref.actor.optim.lr=$LR \
        "${LAYER_OVERRIDES[@]+"${LAYER_OVERRIDES[@]}"}" \
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
        trainer.logger='["console","wandb"]' trainer.project_name=verl_grpo_layer_guided \
        "trainer.experiment_name='$EXP_NAME'" "trainer.default_local_dir='$CKPT_ROOT/$EXP_NAME'" \
        trainer.n_gpus_per_node=$NGPUS trainer.nnodes=1 \
        trainer.save_freq=$SAVE_FREQ trainer.test_freq=5 trainer.total_epochs=$EPOCHS \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} 2>&1 | tee "$LOG_FILE"
}

cleanup_between_runs() {
    ray stop --force 2>/dev/null || true
    pkill -9 -f main_ppo 2>/dev/null || true
    pkill -9 -f vllm 2>/dev/null || true
    sleep 25
}

should_run() {
    local n=$1
    if [[ -n "$ONLY" ]]; then echo "$ONLY" | tr ',' '\n' | grep -qx "$n"; else [[ $n -gt $SKIP ]]; fi
}

# Pre-compute layer-id strings.
TRAIN_NOT_WORST5=$(all_layers_except  "$WORST5"  "$NUM_LAYERS")
TRAIN_NOT_WORST10=$(all_layers_except "$WORST10" "$NUM_LAYERS")

# ---------------- Plan ----------------
echo "============================================================"
echo "  Layer-guided full RL — $MODEL_SHORT"
echo "  Ckpt root:  $CKPT_ROOT"
echo "  GPUs:       $GPUS ($NGPUS)"
echo "  Layer sets: TOP5=$TOP5  WORST5=$WORST5"
echo "  trainable(¬worst5)=$TRAIN_NOT_WORST5"
echo "  Phase C ckpts (auto):"
echo "    only_top5  -> ${ONLY_TOP5_CKPT:-<not found>}"
echo "    only_top10 -> ${ONLY_TOP10_CKPT:-<not found>}"
echo "============================================================"

# ---------------- Run main experiments ----------------
TOTAL=5

# A1: freeze WORST5, full RL @ 1e-6
EXP_NAME="${DATE}_lg1_frzWst5_full_${MODEL_SHORT}_numina_cot_lr1e-6"
should_run 1 && {
    echo ""
    echo "============================================================"
    echo "  [1/$TOTAL] A1  freeze WORST5  +  full LR=1e-6  →  $EXP_NAME"
    echo "============================================================"
    run_train "$EXP_NAME" "" "1e-6" "$TRAIN_NOT_WORST5" "" ""
    cleanup_between_runs
}

# A2: freeze WORST10, full RL @ 1e-6
EXP_NAME="${DATE}_lg2_frzWst10_full_${MODEL_SHORT}_numina_cot_lr1e-6"
should_run 2 && {
    echo ""
    echo "============================================================"
    echo "  [2/$TOTAL] A2  freeze WORST10 +  full LR=1e-6  →  $EXP_NAME"
    echo "============================================================"
    run_train "$EXP_NAME" "" "1e-6" "$TRAIN_NOT_WORST10" "" ""
    cleanup_between_runs
}

# C1: warm-start from only-top5 -> full RL @ 5e-7
EXP_NAME="${DATE}_lg3_ws_top5_full_${MODEL_SHORT}_numina_cot_lr5e-7"
should_run 3 && {
    if [[ -z "$ONLY_TOP5_CKPT" ]]; then
        echo "  [3/$TOTAL] SKIP — only_top5 ckpt not found (use --only-top5-ckpt /abs/path)"
    else
        echo ""
        echo "============================================================"
        echo "  [3/$TOTAL] C1  warm-start only_top5  +  full LR=5e-7  →  $EXP_NAME"
        echo "============================================================"
        run_train "$EXP_NAME" "$ONLY_TOP5_CKPT" "5e-7" "" "" ""
        cleanup_between_runs
    fi
}

# C2: warm-start from only-top10 -> full RL @ 5e-7
EXP_NAME="${DATE}_lg4_ws_top10_full_${MODEL_SHORT}_numina_cot_lr5e-7"
should_run 4 && {
    if [[ -z "$ONLY_TOP10_CKPT" ]]; then
        echo "  [4/$TOTAL] SKIP — only_top10 ckpt not found (use --only-top10-ckpt /abs/path)"
    else
        echo ""
        echo "============================================================"
        echo "  [4/$TOTAL] C2  warm-start only_top10  +  full LR=5e-7  →  $EXP_NAME"
        echo "============================================================"
        run_train "$EXP_NAME" "$ONLY_TOP10_CKPT" "5e-7" "" "" ""
        cleanup_between_runs
    fi
}

# B: freeze WORST5 + boost TOP5 @ 2e-6 + base @ 1e-6
EXP_NAME="${DATE}_lg5_frzWst5_bstTop5_${MODEL_SHORT}_numina_cot_bst2e-6_base1e-6"
should_run 5 && {
    echo ""
    echo "============================================================"
    echo "  [5/$TOTAL] B  freeze WORST5 + boost TOP5 @ 2e-6, base @ 1e-6  →  $EXP_NAME"
    echo "============================================================"
    run_train "$EXP_NAME" "" "1e-6" "$TRAIN_NOT_WORST5" "$TOP5" "2e-6"
    cleanup_between_runs
}

if $DRY_RUN; then
    echo ""
    echo "[dry-run mode — no training launched. Remove --dry-run to execute.]"
    exit 0
fi

echo ""
echo "============================================================"
echo "  All planned experiments done ($TOTAL)."
echo "============================================================"

# ---------------- Dummy LR sweep loop ----------------
if $NO_DUMMY; then
    echo "  --no-dummy set, exiting."
    exit 0
fi

# Cycle these LRs forever, using the lg1 design (freeze worst5 full RL).
DUMMY_LRS=(3e-7 5e-7 7e-7 1.5e-6 3e-6 5e-6)
echo ""
echo "============================================================"
echo "  Dummy LR-sweep loop (freeze WORST5, full RL):"
echo "  Cycling LR ∈ {${DUMMY_LRS[*]}} forever."
echo "  Stop: tmux kill-session, pkill main_ppo, or Ctrl+C."
echo "============================================================"

DUMMY_ITER=0
while true; do
    for LR in "${DUMMY_LRS[@]}"; do
        DUMMY_ITER=$((DUMMY_ITER + 1))
        DUMMY_TAG=$(echo "$LR" | tr -d '.')   # 1.5e-6 -> 15e-6
        DUMMY_NAME="$(date +%m%d_%H%M)_dummy${DUMMY_ITER}_frzWst5_${MODEL_SHORT}_numina_cot_lr${DUMMY_TAG}"
        echo ""
        echo "============================================================"
        echo "  Dummy iter $DUMMY_ITER  LR=$LR  →  $DUMMY_NAME"
        echo "============================================================"
        run_train "$DUMMY_NAME" "" "$LR" "$TRAIN_NOT_WORST5" "" "" || true
        cleanup_between_runs
    done
done
