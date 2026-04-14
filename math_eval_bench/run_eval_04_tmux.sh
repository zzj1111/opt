#!/bin/bash
# ==============================================================================
# One-click evaluation of all 04* checkpoints (Worker Mode)
# ==============================================================================
#
# Evaluates all models under CKPT_ROOT that start with "04" on 15 benchmarks.
# AMC/AIME use average@32. Results logged to WandB.
#
# Worker mode: each GPU independently pulls tasks from a shared queue.
# No GPU ever sits idle waiting for another.
#
# Auto-launches tmux session and activates conda environment.
#
# Usage:
#   bash run_eval_04_tmux.sh                       # Run all 04* models
#   bash run_eval_04_tmux.sh --dry-run             # Preview what would run
#   bash run_eval_04_tmux.sh --gpus 0,1,2,3        # Use specific GPUs
#   bash run_eval_04_tmux.sh --ckpt-root /path     # Override checkpoint dir
#   bash run_eval_04_tmux.sh --no-tmux             # Skip tmux auto-launch
#   bash run_eval_04_tmux.sh --max-tokens "3072"   # Only 3k (default: "3072 8192")

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
MAX_TOKENS_LIST="3072 8192"
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
        --max-tokens)      MAX_TOKENS_LIST="$2"; shift 2 ;;
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
    FULL_ARGS="$FULL_ARGS --max-tokens $(printf '%q' "$MAX_TOKENS_LIST")"
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

IFS=',' read -ra GPU_LIST <<< "$GPUS"
NUM_GPUS=${#GPU_LIST[@]}

PYTHON="${PYTHON:-python3}"
LOCK_DIR="$LOG_DIR/locks"
QUEUE_FILE="$LOG_DIR/task_queue.txt"
mkdir -p "$LOG_DIR" "$LOCK_DIR"

# ========== Resolve model path ==========
# Checkpoint structure: CKPT_ROOT/04xxx_exp_name/global_step_NNN/actor/huggingface/
# Pick the largest global_step_* and append actor/huggingface/ if it exists.
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
        # Check for actor/huggingface/ subdirectory
        if [[ -d "$best_step/actor/huggingface" ]]; then
            echo "$best_step/actor/huggingface"
        elif [[ -d "$best_step/actor" ]]; then
            echo "$best_step/actor"
        else
            echo "$best_step"
        fi
    else
        echo "$exp_dir"
    fi
}

# ========== Build task queue (full models first) ==========
if [[ ! -d "$CKPT_ROOT" ]]; then
    echo "ERROR: Checkpoint root not found: $CKPT_ROOT"
    exit 1
fi

FULL_QUEUE=$(mktemp)
LAYER_QUEUE=$(mktemp)
for d in "$CKPT_ROOT"/04*; do
    [[ -d "$d" ]] || continue
    exp_name=$(basename "$d")
    model_path=$(resolve_model_path "$d")
    for mt in $MAX_TOKENS_LIST; do
        tok_tag="$mt"
        [[ "$mt" == "3072" ]] && tok_tag="3k"
        [[ "$mt" == "8192" ]] && tok_tag="8k"
        output_dir="$RESULTS_BASE/${exp_name}_${tok_tag}"
        if [[ -f "$output_dir/overall_summary.json" ]]; then
            continue
        fi
        line="$model_path|$exp_name|$mt|$tok_tag"
        if [[ "$exp_name" == *full* ]]; then
            echo "$line" >> "$FULL_QUEUE"
        else
            echo "$line" >> "$LAYER_QUEUE"
        fi
    done
done
cat "$FULL_QUEUE" "$LAYER_QUEUE" > "$QUEUE_FILE" 2>/dev/null
rm -f "$FULL_QUEUE" "$LAYER_QUEUE"

TOTAL_TASKS=$(wc -l < "$QUEUE_FILE")

echo "============================================================"
echo "  Checkpoint Evaluation — Worker Mode (04*)"
echo "  Pending tasks: $TOTAL_TASKS"
echo "  Max tokens:    $MAX_TOKENS_LIST"
echo "  Benchmarks:    $BENCHMARKS"
echo "  Avg@N:         $AVG_AT_MAP"
echo "  GPUs:          ${GPU_LIST[*]} ($NUM_GPUS workers, TP=1)"
echo "  Params:        T=$TEMPERATURE P=$TOP_P K=$TOP_K"
echo "  WandB:         $WANDB_ENTITY / $WANDB_PROJECT"
echo "  Results:       $RESULTS_BASE"
echo "============================================================"
echo ""

if $DRY_RUN; then
    echo "[DRY RUN] Would evaluate:"
    while IFS='|' read -r path name mt tok; do
        echo "  $name ($tok)  ->  $path"
    done < "$QUEUE_FILE"
    echo ""
    echo "Total: $TOTAL_TASKS tasks"
    exit 0
fi

if [[ $TOTAL_TASKS -eq 0 ]]; then
    echo "All tasks already completed!"
    exit 0
fi

# ========== Worker function ==========
worker() {
    local gpu_id="$1"
    local completed=0
    local failed=0

    while true; do
        # Atomically grab next task from queue
        local task=""
        task=$( flock -x 200 bash -c '
            t=$(head -1 "'"$QUEUE_FILE"'" 2>/dev/null)
            if [ -n "$t" ]; then
                sed -i "1d" "'"$QUEUE_FILE"'"
            fi
            echo "$t"
        ' 200>"$LOCK_DIR/queue.lock" )

        [[ -z "$task" ]] && break

        IFS='|' read -r model_path exp_name max_tokens tok_tag <<< "$task"
        local run_label="${exp_name}_${tok_tag}"
        local output_dir="$RESULTS_BASE/${run_label}"
        local log_file="$LOG_DIR/${run_label}.log"

        echo "[GPU $gpu_id] START $run_label"

        if CUDA_VISIBLE_DEVICES="$gpu_id" $PYTHON "$SCRIPT_DIR/eval.py" \
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
            > "$log_file" 2>&1; then
            completed=$((completed + 1))
            echo "[GPU $gpu_id] DONE  $run_label (worker total: $completed)"
        else
            failed=$((failed + 1))
            echo "[GPU $gpu_id] FAIL  $run_label — see $log_file"
        fi
    done

    echo "[GPU $gpu_id] Worker finished. Completed: $completed, Failed: $failed"
}

# ========== Launch workers ==========
WORKER_PIDS=()
for gpu_id in "${GPU_LIST[@]}"; do
    worker "$gpu_id" &
    WORKER_PIDS+=($!)
    echo "Worker launched on GPU $gpu_id (PID $!)"
done

# Wait for all workers
for pid in "${WORKER_PIDS[@]}"; do
    wait "$pid" || true
done

echo ""
echo "============================================================"
echo "  All workers finished."
echo "  Results:  $RESULTS_BASE"
echo "  WandB:    https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"
echo "============================================================"

# Generate comparison CSV
COMPLETED_COUNT=$(ls "$RESULTS_BASE"/04*/overall_summary.json 2>/dev/null | wc -l)
if [[ $COMPLETED_COUNT -ge 2 ]]; then
    echo ""
    echo "Generating comparison table ($COMPLETED_COUNT results)..."
    $PYTHON "$SCRIPT_DIR/compare_results.py" "$RESULTS_BASE/" --format csv \
        > "$RESULTS_BASE/summary_04_eval.csv" 2>/dev/null || true
    echo "  Saved to: $RESULTS_BASE/summary_04_eval.csv"
fi
