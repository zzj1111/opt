#!/bin/bash
# ==============================================================================
# Qwen3-8B-Base — simple boost / only / worst sweep (8 experiments, 8 GPUs)
# ==============================================================================
#
# Same 4-phase layout as 1.7B / 4B simple sweep, but layer sets are CLI args
# (the orchestrator script run_8b_pipeline.sh fills them in from analysis).
#
#   1. top5    boost   ($TOP5)                base_lr=1e-6  boost_lr=2e-6
#   2. top10   boost   ($TOP10)               base_lr=1e-6  boost_lr=2e-6
#   3. top5    only    ($TOP5, freeze rest)   lr=2e-6
#   4. top10   only    ($TOP10, freeze rest)  lr=2e-6
#   5. worst5  boost   ($WORST5)              base_lr=1e-6  boost_lr=2e-6
#   6. worst10 boost   ($WORST10)             base_lr=1e-6  boost_lr=2e-6
#   7. worst5  only    ($WORST5, freeze rest) lr=2e-6
#   8. worst10 only    ($WORST10, freeze rest) lr=2e-6
#
# Layer sets must be supplied via --top5 / --top10 / --worst5 / --worst10
# (or the matching env vars TOP5 / TOP10 / WORST5 / WORST10).
#
# batch=512, mini=128, micro=8, epochs=2, max_response=3072.
# 8 GPUs (default 0-7), runs one exp at a time.
#
# Usage:
#   bash run_8b_base_simple_boost_only_tmux.sh \
#        --top5  16,14,19,22,17 \
#        --top10 16,14,19,22,17,15,12,18,20,10 \
#        --worst5  2,3,1,30,32 \
#        --worst10 2,3,1,30,32,29,27,33,4,28
#   bash run_8b_base_simple_boost_only_tmux.sh --skip 4 --only 5,7   # ablation only

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJ_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
SCRIPT_NAME=$(basename "$0")

MODEL="Qwen/Qwen3-8B-Base"
GPUS="0,1,2,3,4,5,6,7"
CKPT_ROOT="$PROJ_DIR/checkpoints"
DATA_DIR="$PROJ_DIR/data/numina_math_cot_author"
EVAL_DIR="$PROJ_DIR/math_eval_bench"
EVAL_RESULTS_BASE="$EVAL_DIR/results"
EVAL_WANDB_PROJECT="${EVAL_WANDB_PROJECT:-opt_rl_eval_8b_math}"
EVAL_TP="${EVAL_TP:-2}"             # tensor-parallel size per benchmark; 2 -> uses all 8 GPUs (4 benches x 2 GPU)
CONDA_INIT="${CONDA_INIT:-/code/hongpaul-sandbox/cuda/miniconda3/bin/activate}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/code/hongpaul-sandbox/cuda/miniconda3/envs/cuda}"
TOP5="${TOP5:-}"; TOP10="${TOP10:-}"; WORST5="${WORST5:-}"; WORST10="${WORST10:-}"
RUN_EVAL_AFTER_TRAIN=true
SKIP=0; ONLY=""; NO_TMUX=false; NO_DUMMY=false; EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)       GPUS="$2"; shift 2 ;;
        --model)      MODEL="$2"; shift 2 ;;
        --ckpt-root)  CKPT_ROOT="$2"; shift 2 ;;
        --data-dir)   DATA_DIR="$2"; shift 2 ;;
        --top5)       TOP5="$2"; shift 2 ;;
        --top10)      TOP10="$2"; shift 2 ;;
        --worst5)     WORST5="$2"; shift 2 ;;
        --worst10)    WORST10="$2"; shift 2 ;;
        --skip)       SKIP="$2"; shift 2 ;;
        --only)       ONLY="$2"; shift 2 ;;
        --no-tmux)    NO_TMUX=true; shift ;;
        --no-dummy)   NO_DUMMY=true; shift ;;
        --no-eval)    RUN_EVAL_AFTER_TRAIN=false; shift ;;
        --eval-wandb-project) EVAL_WANDB_PROJECT="$2"; shift 2 ;;
        --eval-tp)    EVAL_TP="$2"; shift 2 ;;
        *)            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# Validate layer sets are supplied (unless using --only / --skip to bypass).
if [[ -z "$TOP5" || -z "$TOP10" || -z "$WORST5" || -z "$WORST10" ]]; then
    echo "ERROR: must pass --top5 / --top10 / --worst5 / --worst10 layer sets"
    echo "  TOP5='$TOP5'  TOP10='$TOP10'  WORST5='$WORST5'  WORST10='$WORST10'"
    echo "  hint: run_8b_pipeline.sh derives these from eval results"
    exit 1
fi

if [[ -z "${TMUX:-}" ]] && [[ "$NO_TMUX" == "false" ]]; then
    TMUX_SESSION="simple_boost_8b_$(date +%m%d_%H%M)"
    FULL_ARGS="--no-tmux --gpus $(printf '%q' "$GPUS") --model $(printf '%q' "$MODEL") --ckpt-root $(printf '%q' "$CKPT_ROOT") --data-dir $(printf '%q' "$DATA_DIR")"
    FULL_ARGS="$FULL_ARGS --top5 $(printf '%q' "$TOP5") --top10 $(printf '%q' "$TOP10")"
    FULL_ARGS="$FULL_ARGS --worst5 $(printf '%q' "$WORST5") --worst10 $(printf '%q' "$WORST10")"
    [[ $SKIP -gt 0 ]] && FULL_ARGS="$FULL_ARGS --skip $SKIP"
    [[ -n "$ONLY" ]] && FULL_ARGS="$FULL_ARGS --only $(printf '%q' "$ONLY")"
    $NO_DUMMY && FULL_ARGS="$FULL_ARGS --no-dummy"
    $RUN_EVAL_AFTER_TRAIN || FULL_ARGS="$FULL_ARGS --no-eval"
    FULL_ARGS="$FULL_ARGS --eval-wandb-project $(printf '%q' "$EVAL_WANDB_PROJECT")"
    FULL_ARGS="$FULL_ARGS --eval-tp $(printf '%q' "$EVAL_TP")"
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

# eval_ckpt EXP_NAME — runs math500/gsm8k/olympiadbench/amc on the just-trained
# checkpoint, parallelised across 4 GPUs (one bench per GPU). Output dir
# $EVAL_RESULTS_BASE/${EXP_NAME}_8k_t06/ + WandB project $EVAL_WANDB_PROJECT.
eval_ckpt() {
    $RUN_EVAL_AFTER_TRAIN || return 0
    local exp_name="$1"
    [[ -z "$exp_name" ]] && return 0

    local exp_dir="$CKPT_ROOT/$exp_name"
    [[ -d "$exp_dir" ]] || { echo "  [eval] skipped — exp dir missing: $exp_dir"; return 0; }

    local model_path="" best_num=-1
    for step_dir in "$exp_dir"/global_step_*; do
        [[ -d "$step_dir" ]] || continue
        local num="${step_dir##*global_step_}"
        [[ "$num" =~ ^[0-9]+$ ]] || continue
        if [[ -f "$step_dir/actor/huggingface/config.json" ]] && (( num > best_num )); then
            model_path="$step_dir/actor/huggingface"
            best_num=$num
        fi
    done
    if [[ -z "$model_path" ]]; then
        echo "  [eval] skipped — no global_step_*/actor/huggingface under $exp_dir"
        return 0
    fi

    local out_root="$EVAL_RESULTS_BASE/${exp_name}_8k_t06"
    if [[ -f "$out_root/overall_summary.json" ]]; then
        echo "  [eval] already done: $out_root"
        return 0
    fi
    mkdir -p "$out_root"

    IFS=',' read -ra GPU_LIST_LOCAL <<< "$GPUS"
    local n_gpus=${#GPU_LIST_LOCAL[@]}
    local benches=(math500 gsm8k olympiadbench amc)
    local avgs=("" "" "" "amc:32")
    local n_benches=${#benches[@]}

    # Pick a feasible TP: must divide $n_gpus and $n_gpus / TP must be >= n_benches
    # so each of the 4 benches gets its own contiguous GPU slice.
    local tp="$EVAL_TP"
    while (( tp > 1 )) && (( n_gpus / tp < n_benches )); do
        tp=$((tp / 2))
    done
    while (( tp > 1 )) && (( n_gpus % tp != 0 )); do
        tp=$((tp - 1))
    done
    (( tp < 1 )) && tp=1
    local workers_per_round=$(( n_gpus / tp ))

    echo ""
    echo "  ---- eval $exp_name  (T=0.6, 8K) ----"
    echo "  model_path:  $model_path"
    echo "  out_root :   $out_root"
    echo "  wandb    :   $EVAL_WANDB_PROJECT"
    echo "  TP=$tp    GPUs=$n_gpus    workers_per_round=$workers_per_round    benches=$n_benches"

    local pids=()
    local gpu_idx=0
    for i in 0 1 2 3; do
        local b="${benches[$i]}"
        local avg_n="${avgs[$i]}"
        # Slice $tp consecutive GPUs for this bench
        local slice=()
        for ((j=0; j<tp; j++)); do
            slice+=("${GPU_LIST_LOCAL[$(((gpu_idx + j) % n_gpus))]}")
        done
        gpu_idx=$(( gpu_idx + tp ))
        local g
        g=$(IFS=,; echo "${slice[*]}")
        local extra=()
        [[ -n "$avg_n" ]] && extra=(--avg-at-map "$avg_n")
        local log_file="$out_root/${b}.log"
        (
            CUDA_VISIBLE_DEVICES="$g" \
            VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}" \
            python3 "$EVAL_DIR/eval.py" \
                --backend vllm \
                --model "$model_path" \
                --benchmarks "$b" \
                --tensor-parallel-size "$tp" \
                --dtype auto \
                --gpu-memory-utilization 0.85 \
                --max-tokens 8192 \
                --temperature 0.6 \
                --top-p 0.95 \
                --top-k 20 \
                --seed 42 \
                "${extra[@]}" \
                --wandb-project "$EVAL_WANDB_PROJECT" \
                --wandb-entity "${WANDB_ENTITY:-mhong-university-of-minnesota}" \
                --wandb-run-name "${exp_name}_${b}_t06" \
                --output-dir "$out_root" \
                > "$log_file" 2>&1
        ) &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || true; done

    # Merge per-benchmark summaries -> overall_summary.json so re-runs honour skip.
    python3 - "$out_root" "${benches[@]}" <<'PY'
import json, os, sys
root, *benches = sys.argv[1], *sys.argv[2:]
combined = {"benchmarks": []}
for b in benches:
    s = os.path.join(root, b, "summary.json")
    if not os.path.exists(s):
        print(f"  [eval merge] missing {s}", flush=True)
        continue
    d = json.load(open(s))
    if isinstance(d, dict) and ("name" in d or "accuracy" in d):
        combined["benchmarks"].append(d)
    elif isinstance(d, dict) and "benchmarks" in d:
        combined["benchmarks"].extend(d["benchmarks"])
out = os.path.join(root, "overall_summary.json")
with open(out, "w") as f:
    json.dump(combined, f, indent=2)
print(f"  [eval] merged {len(combined['benchmarks'])} benches -> {out}")
PY

    python3 - "$out_root" <<'PY'
import json, os, sys
root = sys.argv[1]
p = os.path.join(root, "overall_summary.json")
try:
    d = json.load(open(p))
except Exception as e:
    print(f"  [eval] cannot read {p}: {e}")
    raise SystemExit
items = d.get("benchmarks", [])
parts = []
total = 0; n = 0
for it in items:
    name = it.get("name") or it.get("benchmark") or "?"
    acc = it.get("accuracy")
    if isinstance(acc, (int, float)):
        parts.append(f"{name}={acc:.4f}")
        total += acc; n += 1
ma = total / n if n else 0
print(f"  [eval] {' '.join(parts)}  | math_avg={ma:.4f}")
PY
    echo ""
}

# Layer sets come from CLI (TOP5/TOP10/WORST5/WORST10 already validated).
echo "  layer sets:"
echo "    TOP5    = $TOP5"
echo "    TOP10   = $TOP10"
echo "    WORST5  = $WORST5"
echo "    WORST10 = $WORST10"

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
        echo "  [$EXP_NUM/$TOTAL] Done.  Cleaning up ray/vllm before eval..."
        # Safe to use node-wide since we're the only job running on this node
        ray stop --force 2>/dev/null || true
        pkill -9 -f main_ppo 2>/dev/null || true
        pkill -9 -f vllm 2>/dev/null || true
        sleep 25
        eval_ckpt "$EXP_NAME"
        echo ""
    fi
done

echo ""; echo "  Sweep complete!"

# ---------------- Dummy LR-sweep loop (keep GPUs busy) ----------------
if $NO_DUMMY; then
    echo "  --no-dummy set, exiting."
    exit 0
fi

# Cycles top5 boost (the strongest single config) at varying boost_lr.
# Other layers stay at base_lr=1e-6. Free LR-sensitivity sweep on top5 boost.
DUMMY_LRS=(1e-6 1.5e-6 2.5e-6 3e-6 5e-6)
echo ""
echo "============================================================"
echo "  Dummy loop: cycle top5 boost at boost_lr ∈ {${DUMMY_LRS[*]}} forever."
echo "  TOP5 layers: $TOP5"
echo "  Stop: tmux kill-session, pkill main_ppo, or Ctrl+C."
echo "============================================================"

DUMMY_ITER=0
while true; do
    for BLR in "${DUMMY_LRS[@]}"; do
        DUMMY_ITER=$((DUMMY_ITER + 1))
        DUMMY_NAME="$(date +%m%d_%H%M)_dummy${DUMMY_ITER}_boost_top5_${MODEL_SHORT}_numina_cot_bst${BLR}_base1e-6"
        echo ""
        echo "============================================================"
        echo "  Dummy iter $DUMMY_ITER  boost_lr=$BLR  →  $DUMMY_NAME"
        echo "============================================================"
        run_train "$DUMMY_NAME" "$TOP5" "$BLR" "1e-6" "boost" || true
        ray stop --force 2>/dev/null || true
        pkill -9 -f main_ppo 2>/dev/null || true
        pkill -9 -f vllm 2>/dev/null || true
        sleep 25
        eval_ckpt "$DUMMY_NAME"
    done
done
