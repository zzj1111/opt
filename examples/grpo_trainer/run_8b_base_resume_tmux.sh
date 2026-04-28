#!/bin/bash
# ==============================================================================
# Qwen3-8B-Base — auto-resume layer training (single 8-GPU node, sequential)
# ==============================================================================
#
# 1. Scans CKPT_ROOT for completed Qwen3-8B-Base layer-wise / full-RL runs.
#    A run counts as "done" iff some `global_step_*/actor/huggingface/config.json`
#    exists under its experiment directory.
# 2. Prints which runs are done vs missing for layers 0-35 + full RL.
# 3. Runs every missing experiment back-to-back on all 8 GPUs.
# 4. After all missing experiments finish, enters a DUMMY loop that keeps
#    training layer 0 forever (so GPUs never sit idle).
#
# Existing-checkpoint detection only looks at experiment-name *suffix*, so any
# date / exp_num prefix used by the original launchers (e.g. `0422_0025_exp10_`)
# is fine.
#
# Usage:
#   bash run_8b_base_resume_tmux.sh                     # full pipeline
#   bash run_8b_base_resume_tmux.sh --dry-run           # only print plan
#   bash run_8b_base_resume_tmux.sh --no-dummy          # skip the loop at end
#   bash run_8b_base_resume_tmux.sh --layers 7,12,18    # only train these
#   bash run_8b_base_resume_tmux.sh --no-tmux           # already inside tmux
#   bash run_8b_base_resume_tmux.sh --ckpt-root /path   # override scan dir
#   bash run_8b_base_resume_tmux.sh --gpus 0,1,2,3      # use fewer GPUs

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJ_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
SCRIPT_NAME=$(basename "$0")

MODEL="Qwen/Qwen3-8B-Base"
GPUS="0,1,2,3,4,5,6,7"
CKPT_ROOT="${CKPT_ROOT:-/code/hongpaul-sandbox/temp/OPT-RL/opt/checkpoints}"
DATA_DIR="$PROJ_DIR/data/numina_math_cot_author"
CONDA_INIT="${CONDA_INIT:-/code/hongpaul-sandbox/cuda/miniconda3/bin/activate}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/code/hongpaul-sandbox/cuda/miniconda3/envs/cuda}"
NUM_LAYERS=36
LAYER_LR="5e-6"
FULL_LR="1e-6"

DRY_RUN=false
NO_TMUX=false
NO_DUMMY=false
LAYER_FILTER=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)       GPUS="$2"; shift 2 ;;
        --model)      MODEL="$2"; shift 2 ;;
        --ckpt-root)  CKPT_ROOT="$2"; shift 2 ;;
        --data-dir)   DATA_DIR="$2"; shift 2 ;;
        --layers)     LAYER_FILTER="$2"; shift 2 ;;
        --dry-run)    DRY_RUN=true; shift ;;
        --no-tmux)    NO_TMUX=true; shift ;;
        --no-dummy)   NO_DUMMY=true; shift ;;
        *)            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ---------------- Tmux auto-launch ----------------
if [[ -z "${TMUX:-}" ]] && [[ "$NO_TMUX" == "false" ]]; then
    TMUX_SESSION="8b_resume_$(date +%m%d_%H%M)"
    FULL_ARGS="--no-tmux --gpus $(printf '%q' "$GPUS") --model $(printf '%q' "$MODEL") --ckpt-root $(printf '%q' "$CKPT_ROOT") --data-dir $(printf '%q' "$DATA_DIR")"
    [[ -n "$LAYER_FILTER" ]] && FULL_ARGS="$FULL_ARGS --layers $(printf '%q' "$LAYER_FILTER")"
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

# ---------------- Detection helpers ----------------

# Returns 0 (true) if the given exp dir has at least one global_step_*/actor/huggingface/config.json
exp_is_done() {
    local d="$1"
    [[ -d "$d" ]] || return 1
    compgen -G "$d/global_step_*/actor/huggingface/config.json" > /dev/null
}

# Find which 8B-Base exps are already done.
DONE_LAYERS=()
DONE_LAYER_DIRS=()
FULL_DONE_DIR=""

shopt -s nullglob
for d in "$CKPT_ROOT"/*_layer*_${MODEL_SHORT}_numina_cot_lr${LAYER_LR}; do
    name=$(basename "$d")
    layer_n=$(echo "$name" | sed -nE "s/.*_layer([0-9]+)_${MODEL_SHORT}_.*/\1/p")
    [[ -z "$layer_n" ]] && continue
    if exp_is_done "$d"; then
        DONE_LAYERS+=("$layer_n")
        DONE_LAYER_DIRS+=("$name")
    fi
done
for d in "$CKPT_ROOT"/*_full_${MODEL_SHORT}_numina_cot_lr${FULL_LR}; do
    if exp_is_done "$d"; then
        FULL_DONE_DIR=$(basename "$d")
        break
    fi
done
shopt -u nullglob

# Layers requested.
if [[ -n "$LAYER_FILTER" ]]; then
    IFS=',' read -ra REQUESTED <<< "$LAYER_FILTER"
else
    REQUESTED=()
    for ((i=0; i<NUM_LAYERS; i++)); do REQUESTED+=("$i"); done
fi

# Compute missing layers (subset of REQUESTED that isn't in DONE_LAYERS).
MISSING_LAYERS=()
for L in "${REQUESTED[@]}"; do
    seen=false
    for D in "${DONE_LAYERS[@]+"${DONE_LAYERS[@]}"}"; do
        [[ "$D" == "$L" ]] && { seen=true; break; }
    done
    $seen || MISSING_LAYERS+=("$L")
done

# Full RL: include only if the user didn't pass --layers.
RUN_FULL=false
if [[ -z "$LAYER_FILTER" ]] && [[ -z "$FULL_DONE_DIR" ]]; then
    RUN_FULL=true
fi

# ---------------- Print plan ----------------
echo "============================================================"
echo "  Qwen3-8B-Base resume scan"
echo "  Ckpt root:    $CKPT_ROOT"
echo "  Done layers:  ${#DONE_LAYERS[@]} / $NUM_LAYERS"
if (( ${#DONE_LAYERS[@]} > 0 )); then
    sorted_done=$(echo "${DONE_LAYERS[@]}" | tr ' ' '\n' | sort -n | uniq | paste -sd ',' -)
    echo "                $sorted_done"
fi
echo "  Full RL done: ${FULL_DONE_DIR:-no}"
echo "  Layers requested: ${REQUESTED[*]}"
echo "  Missing (will run): ${MISSING_LAYERS[*]:-(none)}  ($((${#MISSING_LAYERS[@]})) experiments)"
$RUN_FULL && echo "  Full RL also queued (1 experiment)"
echo "  GPUs: $GPUS ($NGPUS)  | model: $MODEL  | data: $DATA_DIR"
echo "  After missing experiments → dummy loop training layer 0 forever (--no-dummy to skip)"
echo "============================================================"
echo ""

if (( ${#MISSING_LAYERS[@]} == 0 )) && ! $RUN_FULL && ! $NO_DUMMY; then
    echo "Nothing missing — going straight to dummy loop."
elif (( ${#MISSING_LAYERS[@]} == 0 )) && ! $RUN_FULL; then
    echo "Nothing missing.  --no-dummy was set, exiting."
    exit 0
fi

if $DRY_RUN; then
    echo "[DRY RUN] Would run (in order):"
    idx=1
    if $RUN_FULL; then echo "  [$idx]  Full RL  LR=$FULL_LR"; idx=$((idx+1)); fi
    for L in "${MISSING_LAYERS[@]+"${MISSING_LAYERS[@]}"}"; do
        echo "  [$idx]  Layer $L  LR=$LAYER_LR"
        idx=$((idx+1))
    done
    $NO_DUMMY || echo "  [..] Dummy loop: layer 0 forever"
    exit 0
fi

# ---------------- run_train (lifted from run_8b_base_part1_tmux.sh) ----------------
run_train() {
    local EXP_NAME="$1" DATA_DIR="$2" LR="$3" FREEZE_ARGS="$4"
    local BATCH_SIZE="${5:-512}" MINI_BATCH="${6:-128}" MICRO_BATCH="${7:-16}"
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
    # Avoid flashinfer JIT (broken ninja build on this box) — fall back to torch sampler.
    export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
    export VLLM_USE_V1="${VLLM_USE_V1:-1}"
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        "data.train_files='$DATA_DIR/train.parquet'" \
        "data.val_files='$DATA_DIR/test.parquet'" \
        data.train_batch_size=$BATCH_SIZE data.max_prompt_length=1024 data.max_response_length=3072 \
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
        trainer.logger='["console","wandb"]' trainer.project_name=verl_grpo_numina_cot \
        "trainer.experiment_name='$EXP_NAME'" "trainer.default_local_dir='$CKPT_ROOT/$EXP_NAME'" \
        trainer.n_gpus_per_node=$NGPUS trainer.nnodes=1 \
        trainer.save_freq=$SAVE_FREQ trainer.test_freq=5 trainer.total_epochs=$EPOCHS \
        $FREEZE_ARGS ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} 2>&1 | tee "$LOG_FILE"
}

cleanup_between_runs() {
    ray stop --force 2>/dev/null || true
    pkill -9 -f main_ppo 2>/dev/null || true
    pkill -9 -f vllm 2>/dev/null || true
    sleep 25
}

# ---------------- Run missing experiments sequentially ----------------
TOTAL=$(( ${#MISSING_LAYERS[@]} ))
$RUN_FULL && TOTAL=$((TOTAL + 1))
COUNT=0

if $RUN_FULL; then
    COUNT=$((COUNT + 1))
    EXP_NAME="${DATE}_resume${COUNT}_full_${MODEL_SHORT}_numina_cot_lr${FULL_LR}"
    echo "============================================================"
    echo "  [$COUNT/$TOTAL] Full RL  LR=$FULL_LR  →  $EXP_NAME"
    echo "============================================================"
    run_train "$EXP_NAME" "$DATA_DIR" "$FULL_LR" ""
    cleanup_between_runs
fi

for L in "${MISSING_LAYERS[@]+"${MISSING_LAYERS[@]}"}"; do
    COUNT=$((COUNT + 1))
    EXP_NAME="${DATE}_resume${COUNT}_layer${L}_${MODEL_SHORT}_numina_cot_lr${LAYER_LR}"
    echo "============================================================"
    echo "  [$COUNT/$TOTAL] Layer $L  LR=$LAYER_LR  →  $EXP_NAME"
    echo "============================================================"
    run_train "$EXP_NAME" "$DATA_DIR" "$LAYER_LR" "+actor_rollout_ref.actor.train_layer_ids=$L"
    cleanup_between_runs
done

echo ""
echo "============================================================"
echo "  All missing experiments done ($TOTAL)."
echo "============================================================"

# ---------------- Dummy loop: keep training L0 forever ----------------
if $NO_DUMMY; then
    echo "  --no-dummy set, exiting."
    exit 0
fi

echo "  Entering dummy loop: keeps training layer 0 to keep GPUs busy."
echo "  To stop: tmux kill-session, or pkill main_ppo, or Ctrl+C."

DUMMY_ITER=0
while true; do
    DUMMY_ITER=$((DUMMY_ITER + 1))
    DUMMY_NAME="$(date +%m%d_%H%M)_dummy${DUMMY_ITER}_layer0_${MODEL_SHORT}_numina_cot_lr${LAYER_LR}"
    echo ""
    echo "============================================================"
    echo "  Dummy iter $DUMMY_ITER  →  $DUMMY_NAME"
    echo "============================================================"
    run_train "$DUMMY_NAME" "$DATA_DIR" "$LAYER_LR" "+actor_rollout_ref.actor.train_layer_ids=0" || true
    cleanup_between_runs
done
