#!/bin/bash
# ==============================================================================
# One-click MATH evaluation of Qwen3-8B-Base checkpoints (Worker Mode)
# ==============================================================================
#
# Auto-discovers any checkpoint under CKPT_ROOT whose name matches the PATTERN
# (default: anything containing "Qwen3-8B-Base"), resolves its model path, and
# runs the math benchmarks (AIME/MGSM excluded by default).
#
# Worker mode: each GPU independently pulls tasks from a shared queue. One
# 8B model fits on a single H100/H200 (≈22 GB weights + KV cache), so TP=1
# and 8 workers run 8 ckpts in parallel.
#
# Auto-launches tmux + activates conda + logs to WandB.
#
# Usage:
#   bash run_eval_8b_math.sh                          # All *Qwen3-8B-Base* ckpts
#   bash run_eval_8b_math.sh --dry-run                # Preview what would run
#   bash run_eval_8b_math.sh --gpus 0,1,2,3           # Use specific GPUs
#   bash run_eval_8b_math.sh --ckpt-root /path        # Override checkpoint dir
#   bash run_eval_8b_math.sh --no-tmux                # Skip tmux auto-launch
#   bash run_eval_8b_math.sh --max-tokens "3072"      # Only 3k (default: "8192")
#   bash run_eval_8b_math.sh --temperature 0.7        # Override temperature (default: 0.6)
#   bash run_eval_8b_math.sh --pattern "*layer*8B*"   # Override match glob
#   bash run_eval_8b_math.sh --pattern "*foo*,*bar*"  # Multiple globs (OR)
#   bash run_eval_8b_math.sh --force                  # Re-run even if outputs exist
#   bash run_eval_8b_math.sh --tp 2                   # tensor-parallel=2 (default 1)
#   bash run_eval_8b_math.sh --no-base                # skip the un-trained base model
#   bash run_eval_8b_math.sh --base-model /local/path # use a local Qwen3-8B-Base copy
#
# By default the un-trained Qwen/Qwen3-8B-Base is queued FIRST as a reference;
# vLLM downloads it from HF on first run unless you pass --base-model with a
# local path or pre-populate the HF cache.
#
# Output dir / wandb run-name include a temperature tag (t07, t06, ...) so
# repeated runs at different T do not overwrite each other.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ========== Configuration ==========
CKPT_ROOT="${CKPT_ROOT:-/code/hongpaul-sandbox/temp/OPT-RL/opt/checkpoints}"
# Comma-separated globs. A directory matches if it satisfies ANY of them.
# Default: every Qwen3-8B-Base experiment (layer-trained, full RL, boost, only, dummy).
PATTERN="*Qwen3-8B-Base*"
GPUS="0,1,2,3,4,5,6,7"
RESULTS_BASE="$SCRIPT_DIR/results"
LOG_DIR="$SCRIPT_DIR/logs/eval_8b_math"

# Conda
CONDA_INIT="${CONDA_INIT:-/code/hongpaul-sandbox/cuda/miniconda3/bin/activate}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/code/hongpaul-sandbox/cuda/miniconda3/envs/cuda}"

# WandB
WANDB_API_KEY="${WANDB_API_KEY:-b8f38344ec7231ee89baa74ef7209dd5a43df6b2}"
WANDB_ENTITY="${WANDB_ENTITY:-mhong-university-of-minnesota}"
WANDB_PROJECT="${WANDB_PROJECT:-opt_rl_eval_8b_math}"

# Math-only benchmarks (AIME and MGSM excluded)
BENCHMARKS="math500 gsm8k amc olympiadbench"

# Generation params
TEMPERATURE=0.6
TOP_P=0.95
TOP_K=20
MAX_TOKENS_LIST="8192"
SEED=42

# vLLM sizing for 8B
TP_SIZE=1            # 1 GPU per worker; one 8B fits comfortably on H100/H200
GPU_MEM_UTIL=0.85    # 0.85 leaves a bit more headroom than the boost script's 0.90
# 8B reports max_model_len=32768. KV-cache profiling at 32K causes vLLM v1
# engine_core spawn to time out after 5 min ("Engine core initialization
# failed. Failed core proc(s): {}"). Cap to 1024 prompt + 8192 response + margin.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"

# average@N for competition benchmarks (AMC only since AIME is dropped)
AVG_AT_MAP="amc:32"

# Whether to also evaluate the un-trained Qwen3-8B-Base reference.
# DEFAULT OFF: vLLM trying to fetch 'Qwen/Qwen3-8B-Base' from HuggingFace
# was the root cause of repeated "Engine core initialization failed" crashes
# on the advisor's server (slow / no internet / cache miss).
# Enable with --include-base AND a local --base-model /path; the HF id is
# only used as a last resort and will likely hang.
INCLUDE_BASE=false
BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen3-8B-Base}"
BASE_EXP_NAME="Qwen3-8B-Base_baseline"

# Parsing
DRY_RUN=false
NO_TMUX=false
FORCE=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)            GPUS="$2"; shift 2 ;;
        --ckpt-root)       CKPT_ROOT="$2"; shift 2 ;;
        --pattern)         PATTERN="$2"; shift 2 ;;
        --dry-run)         DRY_RUN=true; shift ;;
        --no-tmux)         NO_TMUX=true; shift ;;
        --force)           FORCE=true; shift ;;
        --benchmarks)      BENCHMARKS="$2"; shift 2 ;;
        --wandb-project)   WANDB_PROJECT="$2"; shift 2 ;;
        --avg-at-map)      AVG_AT_MAP="$2"; shift 2 ;;
        --max-tokens)      MAX_TOKENS_LIST="$2"; shift 2 ;;
        --temperature)     TEMPERATURE="$2"; shift 2 ;;
        --max-model-len)   MAX_MODEL_LEN="$2"; shift 2 ;;
        --tp)              TP_SIZE="$2"; shift 2 ;;
        --gpu-mem-util)    GPU_MEM_UTIL="$2"; shift 2 ;;
        --no-base)         INCLUDE_BASE=false; shift ;;
        --include-base)    INCLUDE_BASE=true; shift ;;
        --base-model)      BASE_MODEL_PATH="$2"; INCLUDE_BASE=true; shift 2 ;;
        *)                 EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# Tag temperature so T=0.7 results don't overwrite T=0.6 etc.
T_TAG="t$(echo "$TEMPERATURE" | tr -d '.')"

# ========== Tmux auto-launch ==========
if [[ -z "${TMUX:-}" ]] && [[ "$NO_TMUX" == "false" ]]; then
    TMUX_SESSION="eval_8b_$(date +%m%d_%H%M)"

    FULL_ARGS="--no-tmux"
    FULL_ARGS="$FULL_ARGS --gpus $(printf '%q' "$GPUS")"
    FULL_ARGS="$FULL_ARGS --ckpt-root $(printf '%q' "$CKPT_ROOT")"
    FULL_ARGS="$FULL_ARGS --pattern $(printf '%q' "$PATTERN")"
    FULL_ARGS="$FULL_ARGS --wandb-project $(printf '%q' "$WANDB_PROJECT")"
    FULL_ARGS="$FULL_ARGS --avg-at-map $(printf '%q' "$AVG_AT_MAP")"
    FULL_ARGS="$FULL_ARGS --max-tokens $(printf '%q' "$MAX_TOKENS_LIST")"
    FULL_ARGS="$FULL_ARGS --temperature $(printf '%q' "$TEMPERATURE")"
    FULL_ARGS="$FULL_ARGS --max-model-len $(printf '%q' "$MAX_MODEL_LEN")"
    FULL_ARGS="$FULL_ARGS --tp $(printf '%q' "$TP_SIZE")"
    FULL_ARGS="$FULL_ARGS --gpu-mem-util $(printf '%q' "$GPU_MEM_UTIL")"
    FULL_ARGS="$FULL_ARGS --base-model $(printf '%q' "$BASE_MODEL_PATH")"
    $INCLUDE_BASE || FULL_ARGS="$FULL_ARGS --no-base"
    $DRY_RUN && FULL_ARGS="$FULL_ARGS --dry-run"
    $FORCE   && FULL_ARGS="$FULL_ARGS --force"
    [[ -n "$BENCHMARKS" ]] && FULL_ARGS="$FULL_ARGS --benchmarks $(printf '%q' "$BENCHMARKS")"
    for arg in "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; do FULL_ARGS="$FULL_ARGS $(printf '%q' "$arg")"; done

    tmux new-session -d -s "$TMUX_SESSION" \
        "source $CONDA_INIT && \
         conda activate $CONDA_ENV_PATH && \
         export WANDB_API_KEY=$WANDB_API_KEY && \
         export WANDB_ENTITY=$WANDB_ENTITY && \
         cd $SCRIPT_DIR && \
         bash $SCRIPT_DIR/run_eval_8b_math.sh $FULL_ARGS; \
         exec bash"
    echo "Tmux session '$TMUX_SESSION' started."
    echo "  Attach with:  tmux attach -t $TMUX_SESSION"
    exit 0
fi

# ========== Setup ==========
export WANDB_API_KEY
export WANDB_ENTITY

# Activate conda if vllm isn't available (covers running inside an existing tmux)
if ! python -c "import vllm" >/dev/null 2>&1; then
    if [[ -f "$CONDA_INIT" ]]; then
        echo "Activating conda env: $CONDA_ENV_PATH"
        # shellcheck disable=SC1090
        source "$CONDA_INIT"
        conda activate "$CONDA_ENV_PATH"
    fi
fi

IFS=',' read -ra GPU_LIST <<< "$GPUS"
NUM_GPUS=${#GPU_LIST[@]}

PYTHON="${PYTHON:-$(command -v python)}"
[[ -z "$PYTHON" ]] && PYTHON="python3"

LOCK_DIR="$LOG_DIR/locks"
QUEUE_FILE="$LOG_DIR/task_queue.txt"
mkdir -p "$LOG_DIR" "$LOCK_DIR"

# ========== Resolve model path ==========
# Supports flat HF layout and VeRL global_step_*/actor/huggingface/ layout.
resolve_model_path() {
    local exp_dir="$1"
    if [[ -f "$exp_dir/config.json" ]]; then
        echo "$exp_dir"
        return
    fi
    local best_step=""
    local best_num=-1
    for step_dir in "$exp_dir"/global_step_*; do
        [[ -d "$step_dir" ]] || continue
        local num="${step_dir##*global_step_}"
        if ! [[ "$num" =~ ^[0-9]+$ ]]; then continue; fi
        local candidate=""
        if [[ -f "$step_dir/actor/huggingface/config.json" ]]; then
            candidate="$step_dir/actor/huggingface"
        elif [[ -f "$step_dir/actor/config.json" ]]; then
            candidate="$step_dir/actor"
        elif [[ -f "$step_dir/config.json" ]]; then
            candidate="$step_dir"
        fi
        if [[ -n "$candidate" ]] && (( num > best_num )); then
            best_num=$num
            best_step="$candidate"
        fi
    done
    if [[ -n "$best_step" ]]; then
        echo "$best_step"
    else
        echo ""
    fi
}

# ========== Build task queue ==========
if [[ ! -d "$CKPT_ROOT" ]]; then
    echo "ERROR: Checkpoint root not found: $CKPT_ROOT"
    exit 1
fi

shopt -s nullglob
SKIPPED_NO_MODEL=()
SKIPPED_ALREADY_DONE=()
MATCHED_DIRS=0
declare -A SEEN_DIRS
> "$QUEUE_FILE"

# ---- Always queue the un-trained base model first (unless --no-base) ----
if $INCLUDE_BASE; then
    for mt in $MAX_TOKENS_LIST; do
        tok_tag="$mt"
        [[ "$mt" == "3072" ]] && tok_tag="3k"
        [[ "$mt" == "8192" ]] && tok_tag="8k"
        output_dir="$RESULTS_BASE/${BASE_EXP_NAME}_${tok_tag}_${T_TAG}"
        if [[ -f "$output_dir/overall_summary.json" ]] && ! $FORCE; then
            SKIPPED_ALREADY_DONE+=("${BASE_EXP_NAME}_${tok_tag}_${T_TAG} [base]")
            continue
        fi
        echo "$BASE_MODEL_PATH|$BASE_EXP_NAME|$mt|$tok_tag" >> "$QUEUE_FILE"
    done
fi

# Iterate over each comma-separated glob and union the matched dirs.
IFS=',' read -ra PATTERN_LIST <<< "$PATTERN"
for pat in "${PATTERN_LIST[@]}"; do
    pat=$(echo "$pat" | xargs)   # trim whitespace
    [[ -z "$pat" ]] && continue
    for d in "$CKPT_ROOT"/$pat; do
        [[ -d "$d" ]] || continue
        exp_name=$(basename "$d")
        [[ "$exp_name" == .* ]] && continue
        # Dedupe: a name like *boost*worst* would match multiple globs
        [[ -n "${SEEN_DIRS[$exp_name]:-}" ]] && continue
        SEEN_DIRS[$exp_name]=1
        MATCHED_DIRS=$((MATCHED_DIRS + 1))

        model_path=$(resolve_model_path "$d")
        if [[ -z "$model_path" ]]; then
            SKIPPED_NO_MODEL+=("$exp_name")
            continue
        fi

        for mt in $MAX_TOKENS_LIST; do
            tok_tag="$mt"
            [[ "$mt" == "3072" ]] && tok_tag="3k"
            [[ "$mt" == "8192" ]] && tok_tag="8k"
            output_dir="$RESULTS_BASE/${exp_name}_${tok_tag}_${T_TAG}"
            if [[ -f "$output_dir/overall_summary.json" ]] && ! $FORCE; then
                SKIPPED_ALREADY_DONE+=("${exp_name}_${tok_tag}_${T_TAG}")
                continue
            fi
            echo "$model_path|$exp_name|$mt|$tok_tag" >> "$QUEUE_FILE"
        done
    done
done
shopt -u nullglob

TOTAL_TASKS=$(wc -l < "$QUEUE_FILE")

echo "============================================================"
echo "  Boost Math Eval — Worker Mode"
echo "  Ckpt root:                 $CKPT_ROOT"
echo "  Pattern:                   $PATTERN"
echo "  Matched directories:       $MATCHED_DIRS"
echo "  Skipped (no valid model):  ${#SKIPPED_NO_MODEL[@]}"
echo "  Skipped (already done):    ${#SKIPPED_ALREADY_DONE[@]}  (use --force to re-run)"
echo "  Pending tasks:             $TOTAL_TASKS"
echo "  Max tokens:                $MAX_TOKENS_LIST"
echo "  Benchmarks:                $BENCHMARKS"
echo "  Avg@N:                     $AVG_AT_MAP"
echo "  GPUs:                      ${GPU_LIST[*]} ($NUM_GPUS workers, TP=$TP_SIZE)"
echo "  vLLM:                      gpu_memory_utilization=$GPU_MEM_UTIL"
if $INCLUDE_BASE; then
    echo "  Base model in queue:       $BASE_MODEL_PATH  (as exp '$BASE_EXP_NAME')"
else
    echo "  Base model:                skipped (--no-base)"
fi
echo "  Params:                    T=$TEMPERATURE ($T_TAG) P=$TOP_P K=$TOP_K"
echo "  WandB:                     $WANDB_ENTITY / $WANDB_PROJECT"
echo "  Results:                   $RESULTS_BASE"
echo "============================================================"
echo ""

if (( ${#SKIPPED_NO_MODEL[@]} > 0 )); then
    echo "Skipped (no config.json / global_step_* found):"
    for name in "${SKIPPED_NO_MODEL[@]}"; do
        echo "  - $name"
    done
    echo ""
fi

if (( ${#SKIPPED_ALREADY_DONE[@]} > 0 )); then
    echo "Skipped (overall_summary.json already exists at this T tag):"
    for name in "${SKIPPED_ALREADY_DONE[@]}"; do
        echo "  - $name"
    done
    echo ""
fi

if [[ $MATCHED_DIRS -eq 0 ]]; then
    echo "WARNING: pattern '$PATTERN' matched zero directories under $CKPT_ROOT."
    echo "  -> Either the ckpt root is wrong, or no checkpoints have been trained yet."
    echo "  -> Sanity check: ls -d $CKPT_ROOT/$PATTERN"
    exit 0
fi

if $DRY_RUN; then
    echo "[DRY RUN] Would evaluate:"
    while IFS='|' read -r path name mt tok; do
        echo "  ${name}_${tok}_${T_TAG}  ->  $path"
    done < "$QUEUE_FILE"
    echo ""
    echo "Total: $TOTAL_TASKS tasks"
    exit 0
fi

if [[ $TOTAL_TASKS -eq 0 ]]; then
    echo "All matched checkpoints already evaluated at T=$TEMPERATURE."
    echo "  -> Re-run with --force to overwrite existing _${T_TAG} results, OR"
    echo "  -> change --temperature to evaluate at a different T."
    exit 0
fi

# ========== Worker function ==========
worker() {
    local gpu_id="$1"
    local completed=0
    local failed=0

    while true; do
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
        local run_label="${exp_name}_${tok_tag}_${T_TAG}"
        local output_dir="$RESULTS_BASE/${run_label}"
        local log_file="$LOG_DIR/${run_label}.log"

        echo "[GPU $gpu_id] START $run_label"

        local mml_args=()
        [[ -n "$MAX_MODEL_LEN" ]] && mml_args=(--max-model-len "$MAX_MODEL_LEN")
        if CUDA_VISIBLE_DEVICES="$gpu_id" $PYTHON "$SCRIPT_DIR/eval.py" \
            --backend vllm \
            --model "$model_path" \
            --benchmarks $BENCHMARKS \
            --tensor-parallel-size $TP_SIZE \
            --dtype auto \
            --gpu-memory-utilization $GPU_MEM_UTIL \
            --max-tokens "$max_tokens" \
            --temperature $TEMPERATURE \
            --top-p $TOP_P \
            --top-k $TOP_K \
            --seed $SEED \
            --avg-at-map "$AVG_AT_MAP" \
            "${mml_args[@]}" \
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

for pid in "${WORKER_PIDS[@]}"; do
    wait "$pid" || true
done

echo ""
echo "============================================================"
echo "  All workers finished."
echo "  Results:  $RESULTS_BASE"
echo "  WandB:    https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"
echo "============================================================"
