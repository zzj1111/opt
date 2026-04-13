#!/bin/bash
# ==============================================================================
# One-click evaluation of all 04* checkpoints
# ==============================================================================
#
# Evaluates all models under CKPT_ROOT that start with "04" on 15 benchmarks.
# AMC/AIME use average@32. Results logged to WandB.
#
# Auto-launches tmux session and activates conda environment.
#
# Usage:
#   bash run_eval_04_tmux.sh                       # Run all 04* models
#   bash run_eval_04_tmux.sh --dry-run             # Preview what would run
#   bash run_eval_04_tmux.sh --gpus 0,1,2,3        # Use specific GPUs
#   bash run_eval_04_tmux.sh --ckpt-root /path     # Override checkpoint dir
#   bash run_eval_04_tmux.sh --no-tmux             # Skip tmux auto-launch
#   bash run_eval_04_tmux.sh --only amc,aime2024   # Run only specific benchmarks

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ========== Configuration ==========
CKPT_ROOT="${CKPT_ROOT:-/code/hongpaul-sandbox/temp/OPT-RL/opt/checkpoints}"
GPUS="0,1,2,3,4,5,6,7"
RESULTS_BASE="$SCRIPT_DIR/results"
LOG_DIR="$SCRIPT_DIR/logs/eval_04"

# Conda
CONDA_INIT="${CONDA_INIT:-/code/hongpaul-sandbox/cuda/miniconda3/bin/activate}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/code/hongpaul-sandbox/cuda/miniconda3/envs/cuda}"

# WandB
WANDB_API_KEY="${WANDB_API_KEY:-b8f38344ec7231ee89baa74ef7209dd5a43df6b2}"
WANDB_ENTITY="${WANDB_ENTITY:-mhong-university-of-minnesota}"
WANDB_PROJECT="${WANDB_PROJECT:-opt_rl_eval}"

# Benchmarks
BENCHMARKS="math500 gsm8k mbpp humaneval arc_challenge mmlu_pro bbh mgsm ceval amc aime2024 aime2025 olympiadbench gpqa_diamond ifeval"

# Generation params
TEMPERATURE=0.6
TOP_P=0.95
TOP_K=20
MAX_TOKENS_LIST="3072 8192"   # Run each model at both 3k and 8k
SEED=42

# average@N for competition benchmarks
AVG_AT_MAP="amc:32,aime2024:32,aime2025:32"

# Parsing
DRY_RUN=false
NO_TMUX=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)            GPUS="$2"; shift 2 ;;
        --ckpt-root)       CKPT_ROOT="$2"; shift 2 ;;
        --dry-run)         DRY_RUN=true; shift ;;
        --no-tmux)         NO_TMUX=true; shift ;;
        --benchmarks)      BENCHMARKS="$2"; shift 2 ;;
        --wandb-project)   WANDB_PROJECT="$2"; shift 2 ;;
        --avg-at-map)      AVG_AT_MAP="$2"; shift 2 ;;
        *)                 EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ========== Tmux auto-launch ==========
if [[ -z "${TMUX:-}" ]] && [[ "$NO_TMUX" == "false" ]]; then
    TMUX_SESSION="eval_04_$(date +%m%d_%H%M)"

    FULL_ARGS="--no-tmux"
    FULL_ARGS="$FULL_ARGS --gpus $(printf '%q' "$GPUS")"
    FULL_ARGS="$FULL_ARGS --ckpt-root $(printf '%q' "$CKPT_ROOT")"
    FULL_ARGS="$FULL_ARGS --wandb-project $(printf '%q' "$WANDB_PROJECT")"
    FULL_ARGS="$FULL_ARGS --avg-at-map $(printf '%q' "$AVG_AT_MAP")"
    $DRY_RUN && FULL_ARGS="$FULL_ARGS --dry-run"
    [[ -n "$BENCHMARKS" ]] && FULL_ARGS="$FULL_ARGS --benchmarks $(printf '%q' "$BENCHMARKS")"
    for arg in "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; do FULL_ARGS="$FULL_ARGS $(printf '%q' "$arg")"; done

    tmux new-session -d -s "$TMUX_SESSION" \
        "source $CONDA_INIT && \
         conda activate $CONDA_ENV_PATH && \
         export WANDB_API_KEY=$WANDB_API_KEY && \
         export WANDB_ENTITY=$WANDB_ENTITY && \
         cd $SCRIPT_DIR && \
         bash $SCRIPT_DIR/run_eval_04_tmux.sh $FULL_ARGS; \
         exec bash"
    echo "Tmux session '$TMUX_SESSION' started."
    echo "  Attach with:  tmux attach -t $TMUX_SESSION"
    exit 0
fi

# ========== Setup ==========
export WANDB_API_KEY
export WANDB_ENTITY

NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
mkdir -p "$LOG_DIR"

PYTHON="${PYTHON:-python3}"

# ========== Resolve model path ==========
# Checkpoint structure: CKPT_ROOT/04xxx_exp_name/global_step_NNN/
# Pick the largest global_step_* directory under each 04* experiment.
resolve_model_path() {
    local exp_dir="$1"
    local best_step=""
    local best_num=-1
    for step_dir in "$exp_dir"/global_step_*; do
        [[ -d "$step_dir" ]] || continue
        local num="${step_dir##*global_step_}"
        if [[ "$num" =~ ^[0-9]+$ ]] && (( num > best_num )); then
            best_num=$num
            best_step="$step_dir"
        fi
    done
    if [[ -n "$best_step" ]]; then
        echo "$best_step"
    else
        # No global_step_* subdirs — treat exp_dir itself as the model path
        echo "$exp_dir"
    fi
}

# ========== Collect models ==========
if [[ ! -d "$CKPT_ROOT" ]]; then
    echo "ERROR: Checkpoint root not found: $CKPT_ROOT"
    exit 1
fi

MODELS=()
MODEL_NAMES=()
for d in "$CKPT_ROOT"/04*; do
    [[ -d "$d" ]] || continue
    model_path=$(resolve_model_path "$d")
    MODELS+=("$model_path")
    MODEL_NAMES+=("$(basename "$d")")
done

TOTAL=${#MODELS[@]}
if [[ $TOTAL -eq 0 ]]; then
    echo "ERROR: No 04* directories found in $CKPT_ROOT"
    exit 1
fi

NUM_LENGTHS=$(echo $MAX_TOKENS_LIST | wc -w)
TOTAL_RUNS=$((TOTAL * NUM_LENGTHS))

echo "============================================================"
echo "  Checkpoint Evaluation (04*)"
echo "  Models:      $TOTAL"
echo "  Max tokens:  $MAX_TOKENS_LIST ($NUM_LENGTHS configs per model)"
echo "  Total runs:  $TOTAL_RUNS"
echo "  Benchmarks:  $BENCHMARKS"
echo "  Avg@N:       $AVG_AT_MAP"
echo "  GPUs:        $NUM_GPUS (TP=1 per model)"
echo "  Params:      T=$TEMPERATURE P=$TOP_P K=$TOP_K"
echo "  WandB:       $WANDB_ENTITY / $WANDB_PROJECT"
echo "  Results:     $RESULTS_BASE"
echo "============================================================"
echo ""

if $DRY_RUN; then
    echo "[DRY RUN] Would evaluate:"
    for i in "${!MODELS[@]}"; do
        for mt in $MAX_TOKENS_LIST; do
            local_tag="$mt"; [[ "$mt" == "3072" ]] && local_tag="3k"; [[ "$mt" == "8192" ]] && local_tag="8k"
            echo "  ${MODEL_NAMES[$i]}  ->  ${MODELS[$i]}  (max_tokens=$local_tag)"
        done
    done
    echo ""
    echo "Total: $TOTAL_RUNS runs ($TOTAL models x $NUM_LENGTHS token lengths)"
    exit 0
fi

# ========== Run function ==========
run_model() {
    local model_path="$1"
    local gpu_id="$2"
    local max_tokens="$3"
    local exp_name="$4"   # experiment name (04xxx_exp_name), NOT global_step_*

    # Tag: 3k or 8k
    local tok_tag="${max_tokens}"
    [[ "$max_tokens" == "3072" ]] && tok_tag="3k"
    [[ "$max_tokens" == "8192" ]] && tok_tag="8k"

    local output_dir="$RESULTS_BASE/${exp_name}_${tok_tag}"
    local log_file="$LOG_DIR/${exp_name}_${tok_tag}.log"
    local run_label="${exp_name}_${tok_tag}"

    # Skip if already completed
    if [[ -f "$output_dir/overall_summary.json" ]]; then
        echo "  [SKIP] $run_label — already completed"
        return 0
    fi

    echo "  [GPU $gpu_id] $run_label"

    CUDA_VISIBLE_DEVICES="$gpu_id" $PYTHON "$SCRIPT_DIR/eval.py" \
        --backend vllm \
        --model "$model_path" \
        --benchmarks $BENCHMARKS \
        --tensor-parallel-size 1 \
        --dtype auto \
        --gpu-memory-utilization 0.90 \
        --max-tokens "$max_tokens" \
        --temperature $TEMPERATURE \
        --top-p $TOP_P \
        --top-k $TOP_K \
        --seed $SEED \
        --avg-at-map "$AVG_AT_MAP" \
        --wandb-project "$WANDB_PROJECT" \
        --wandb-entity "$WANDB_ENTITY" \
        --wandb-run-name "$run_label" \
        --output-dir "$output_dir" \
        > "$log_file" 2>&1

    return $?
}

# ========== Build run queue: (model_path, exp_name, max_tokens) triples ==========
RUN_MODELS=()
RUN_NAMES=()
RUN_TOKENS=()
for i in "${!MODELS[@]}"; do
    for mt in $MAX_TOKENS_LIST; do
        RUN_MODELS+=("${MODELS[$i]}")
        RUN_NAMES+=("${MODEL_NAMES[$i]}")
        RUN_TOKENS+=("$mt")
    done
done
TOTAL_RUNS=${#RUN_MODELS[@]}

# ========== Parallel execution ==========
COMPLETED=0
FAILED_RUNS=()

for ((i=0; i<TOTAL_RUNS; i+=NUM_GPUS)); do
    batch_end=$((i + NUM_GPUS))
    [[ $batch_end -gt $TOTAL_RUNS ]] && batch_end=$TOTAL_RUNS
    batch_size=$((batch_end - i))

    echo ""
    echo "--- Batch: runs $((i+1))-$batch_end / $TOTAL_RUNS ---"

    PIDS=()
    for ((j=0; j<batch_size; j++)); do
        run_idx=$((i + j))
        gpu_id=$(echo "$GPUS" | cut -d',' -f$((j+1)))
        run_model "${RUN_MODELS[$run_idx]}" "$gpu_id" "${RUN_TOKENS[$run_idx]}" "${RUN_NAMES[$run_idx]}" &
        PIDS+=($!)
    done

    # Wait for batch
    for pid_idx in "${!PIDS[@]}"; do
        run_idx=$((i + pid_idx))
        tok_tag="${RUN_TOKENS[$run_idx]}"
        [[ "$tok_tag" == "3072" ]] && tok_tag="3k"
        [[ "$tok_tag" == "8192" ]] && tok_tag="8k"
        local_name="${RUN_NAMES[$run_idx]}_${tok_tag}"
        if wait "${PIDS[$pid_idx]}"; then
            COMPLETED=$((COMPLETED + 1))
            echo "  [DONE] $local_name ($COMPLETED/$TOTAL_RUNS)"
        else
            FAILED_RUNS+=("$local_name")
            echo "  [FAIL] $local_name — check $LOG_DIR/${local_name}.log"
        fi
    done
done

echo ""
echo "============================================================"
echo "  Evaluation Complete"
echo "  Succeeded: $COMPLETED / $TOTAL_RUNS"
if [[ ${#FAILED_RUNS[@]} -gt 0 ]]; then
    echo "  Failed: ${FAILED_RUNS[*]}"
fi
echo "  Results:  $RESULTS_BASE"
echo "  WandB:    https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"
echo "============================================================"

# Generate comparison CSV
if [[ $COMPLETED -ge 2 ]]; then
    echo ""
    echo "Generating comparison table..."
    $PYTHON "$SCRIPT_DIR/compare_results.py" "$RESULTS_BASE/" --format csv \
        > "$RESULTS_BASE/summary_04_eval.csv" 2>/dev/null || true
    echo "  Saved to: $RESULTS_BASE/summary_04_eval.csv"
fi
