#!/bin/bash
# ==============================================================================
# Supplementary / resume evaluation
# ==============================================================================
#
# Per-benchmark resume: scans <output_dir>/<bench>/summary.json for every
# requested bench × ckpt, queues ONLY the missing ones. Useful when:
#   - some runs failed mid-eval and left holes
#   - you added new benchmarks and want to backfill
#   - you want to re-run a single bench without re-running the others
#
# After all workers finish, rebuilds overall_summary.json so downstream
# tools see the merged result.
#
# Use --force to ignore existing summaries and re-run everything from scratch.
#
# Usage:
#   bash run_eval_04_supplement.sh                                  # fill missing benches for *Qwen3-8B-Base* ckpts
#   bash run_eval_04_supplement.sh --pattern '04*'                  # original 04-prefix behaviour
#   bash run_eval_04_supplement.sh --pattern '*Qwen3-1.7B*'         # any other glob
#   bash run_eval_04_supplement.sh --benchmarks "math500 amc"       # only check these benches
#   bash run_eval_04_supplement.sh --force                          # re-run everything (delete cached summaries)
#   bash run_eval_04_supplement.sh --include-base                   # also evaluate Qwen/Qwen3-8B-Base (un-trained)
#   bash run_eval_04_supplement.sh --include-base --base-model /local/path/to/base
#   bash run_eval_04_supplement.sh --layers "5,7,11,13"             # only ckpts matching *_layer<N>_* for given N
#   bash run_eval_04_supplement.sh --dry-run                        # preview which benches will run
#   bash run_eval_04_supplement.sh --gpus 0,1,2,3 --no-tmux

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ========== Configuration ==========
CKPT_ROOT="${CKPT_ROOT:-/code/hongpaul-sandbox/temp/OPT-RL/opt/checkpoints}"
PATTERN="${PATTERN:-*Qwen3-8B-Base*}"
# Optionally also evaluate the un-trained base model (--include-base flag).
# Treated as a "ckpt" with name $BASE_NAME pointing at $BASE_MODEL (HF or local).
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B-Base}"
BASE_NAME="${BASE_NAME:-Qwen3-8B-Base_baseline}"
GPUS="0,1,2,3,4,5,6,7"
# Each worker uses TP GPUs. Default: 4 workers × TP=2 on 8 GPUs.
# (TP=1 with 8 workers OOM'd on 8B with heavy benches.)
TP="${TP:-2}"
RESULTS_BASE="$SCRIPT_DIR/results"
LOG_DIR="$SCRIPT_DIR/logs/eval_04_supplement"

# Conda
CONDA_INIT="${CONDA_INIT:-/code/hongpaul-sandbox/cuda/miniconda3/bin/activate}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/code/hongpaul-sandbox/cuda/miniconda3/envs/cuda}"

# WandB
WANDB_API_KEY="${WANDB_API_KEY:-b8f38344ec7231ee89baa74ef7209dd5a43df6b2}"
WANDB_ENTITY="${WANDB_ENTITY:-mhong-university-of-minnesota}"
WANDB_PROJECT="${WANDB_PROJECT:-opt_rl_eval_8b_math}"

# Default: same 12 benches as run_eval_04_tmux.sh (math + code + reasoning + language)
BENCHMARKS="math500 gsm8k olympiadbench amc humaneval_plus mbpp livecodebench gpqa_diamond mmlu_pro ceval mgsm ifeval"

# Generation params
TEMPERATURE=0.6
TOP_P=0.95
TOP_K=20
MAX_TOKENS_LIST="8192"
SEED=42

# average@N for competition benchmarks
AVG_AT_MAP="amc:32"

DRY_RUN=false
NO_TMUX=false
FORCE=false
INCLUDE_BASE=false
LAYERS=""               # comma-list, e.g. "5,7,11,13"; empty = no filter
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)            GPUS="$2"; shift 2 ;;
        --tp)              TP="$2"; shift 2 ;;
        --ckpt-root)       CKPT_ROOT="$2"; shift 2 ;;
        --pattern)         PATTERN="$2"; shift 2 ;;
        --layers)          LAYERS="$2"; shift 2 ;;
        --dry-run)         DRY_RUN=true; shift ;;
        --no-tmux)         NO_TMUX=true; shift ;;
        --force)           FORCE=true; shift ;;
        --include-base)    INCLUDE_BASE=true; shift ;;
        --base-model)      BASE_MODEL="$2"; shift 2 ;;
        --base-name)       BASE_NAME="$2"; shift 2 ;;
        --benchmarks)      BENCHMARKS="$2"; shift 2 ;;
        --wandb-project)   WANDB_PROJECT="$2"; shift 2 ;;
        --avg-at-map)      AVG_AT_MAP="$2"; shift 2 ;;
        --max-tokens)      MAX_TOKENS_LIST="$2"; shift 2 ;;
        *)                 EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ========== Tmux auto-launch ==========
if [[ -z "${TMUX:-}" ]] && [[ "$NO_TMUX" == "false" ]]; then
    TMUX_SESSION="eval_04_supp_$(date +%m%d_%H%M)"

    FULL_ARGS="--no-tmux"
    FULL_ARGS="$FULL_ARGS --gpus $(printf '%q' "$GPUS")"
    FULL_ARGS="$FULL_ARGS --tp $TP"
    FULL_ARGS="$FULL_ARGS --ckpt-root $(printf '%q' "$CKPT_ROOT")"
    FULL_ARGS="$FULL_ARGS --pattern $(printf '%q' "$PATTERN")"
    [[ -n "$LAYERS" ]] && FULL_ARGS="$FULL_ARGS --layers $(printf '%q' "$LAYERS")"
    FULL_ARGS="$FULL_ARGS --wandb-project $(printf '%q' "$WANDB_PROJECT")"
    FULL_ARGS="$FULL_ARGS --avg-at-map $(printf '%q' "$AVG_AT_MAP")"
    FULL_ARGS="$FULL_ARGS --max-tokens $(printf '%q' "$MAX_TOKENS_LIST")"
    FULL_ARGS="$FULL_ARGS --benchmarks $(printf '%q' "$BENCHMARKS")"
    $DRY_RUN      && FULL_ARGS="$FULL_ARGS --dry-run"
    $FORCE        && FULL_ARGS="$FULL_ARGS --force"
    $INCLUDE_BASE && FULL_ARGS="$FULL_ARGS --include-base --base-model $(printf '%q' "$BASE_MODEL") --base-name $(printf '%q' "$BASE_NAME")"
    for arg in "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; do FULL_ARGS="$FULL_ARGS $(printf '%q' "$arg")"; done

    tmux new-session -d -s "$TMUX_SESSION" \
        "source $CONDA_INIT && \
         conda activate $CONDA_ENV_PATH && \
         export WANDB_API_KEY=$WANDB_API_KEY && \
         export WANDB_ENTITY=$WANDB_ENTITY && \
         cd $SCRIPT_DIR && \
         bash $SCRIPT_DIR/run_eval_04_supplement.sh $FULL_ARGS; \
         exec bash"
    echo "Tmux session '$TMUX_SESSION' started."
    echo "  Attach with:  tmux attach -t $TMUX_SESSION"
    exit 0
fi

# ========== Setup ==========
export WANDB_API_KEY
export WANDB_ENTITY

# Activate conda env if not already active (e.g. when running inside an
# existing tmux session that skipped the auto-launch path above).
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

# Group GPUs into TP-sized slices. Each worker owns one slice and runs
# vllm with --tensor-parallel-size $TP on that slice.
if (( NUM_GPUS % TP != 0 )); then
    echo "ERROR: --gpus has $NUM_GPUS GPUs, not divisible by --tp=$TP"
    exit 1
fi
NUM_WORKERS=$(( NUM_GPUS / TP ))
GPU_SLICES=()
for ((w=0; w<NUM_WORKERS; w++)); do
    slice=""
    for ((j=0; j<TP; j++)); do
        idx=$((w * TP + j))
        slice+="${GPU_LIST[$idx]}"
        (( j < TP - 1 )) && slice+=","
    done
    GPU_SLICES+=("$slice")
done

PYTHON="${PYTHON:-$(command -v python)}"
[[ -z "$PYTHON" ]] && PYTHON="python3"
LOCK_DIR="$LOG_DIR/locks"
QUEUE_FILE="$LOG_DIR/task_queue.txt"
mkdir -p "$LOG_DIR" "$LOCK_DIR"

# ========== Resolve model path ==========
# Returns the largest global_step_<N> whose subdir actually contains a
# recognized HF model (config.json present). Returns empty string if no
# valid ckpt exists — caller MUST check and skip rather than fallback to
# exp_dir (which would feed an invalid path to vllm and crash).
resolve_model_path() {
    local exp_dir="$1"
    local best_step=""
    local best_num=-1
    for step_dir in "$exp_dir"/global_step_*; do
        [[ -d "$step_dir" ]] || continue
        local num="${step_dir##*global_step_}"
        [[ "$num" =~ ^[0-9]+$ ]] || continue
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
    echo "$best_step"
}

# ========== Build task queue ==========
# Only queue a task if at least one of the requested benchmarks is missing
# (based on per-benchmark summary.json files)
if [[ ! -d "$CKPT_ROOT" ]]; then
    echo "ERROR: Checkpoint root not found: $CKPT_ROOT"
    exit 1
fi

FULL_QUEUE=$(mktemp)
LAYER_QUEUE=$(mktemp)
shopt -s nullglob

# Optional: queue the un-trained base model first (--include-base).
# Skipped if BASE_MODEL is empty.
if $INCLUDE_BASE && [[ -n "$BASE_MODEL" ]]; then
    for mt in $MAX_TOKENS_LIST; do
        tok_tag="$mt"
        [[ "$mt" == "3072" ]] && tok_tag="3k"
        [[ "$mt" == "8192" ]] && tok_tag="8k"
        base_out="$RESULTS_BASE/${BASE_NAME}_${tok_tag}"
        if $FORCE; then
            base_missing="$BENCHMARKS"
        else
            base_missing=""
            for bm in $BENCHMARKS; do
                [[ ! -f "$base_out/$bm/summary.json" ]] && base_missing="$base_missing $bm"
            done
            base_missing=$(echo $base_missing | xargs)
        fi
        [[ -z "$base_missing" ]] && continue
        echo "$BASE_MODEL|$BASE_NAME|$mt|$tok_tag|$base_missing" >> "$FULL_QUEUE"
    done
fi

SKIPPED_NO_CKPT=()
SKIPPED_LAYER_FILTER=0
# Pre-parse layer filter list (empty = no filter, accept all)
LAYER_LIST=()
if [[ -n "$LAYERS" ]]; then
    IFS=',' read -ra LAYER_LIST <<< "$LAYERS"
fi

for d in "$CKPT_ROOT"/$PATTERN; do
    [[ -d "$d" ]] || continue
    exp_name=$(basename "$d")

    # Optional layer filter: keep only ckpts whose name matches *_layer<N>_*
    # for some N in --layers. Substring match with anchoring underscores so
    # layer5 doesn't accidentally match layer50 etc.
    if (( ${#LAYER_LIST[@]} > 0 )); then
        keep=false
        for L in "${LAYER_LIST[@]}"; do
            if [[ "$exp_name" == *_layer${L}_* ]]; then
                keep=true; break
            fi
        done
        if ! $keep; then
            SKIPPED_LAYER_FILTER=$((SKIPPED_LAYER_FILTER + 1))
            continue
        fi
    fi

    model_path=$(resolve_model_path "$d")
    if [[ -z "$model_path" ]]; then
        SKIPPED_NO_CKPT+=("$exp_name")
        continue
    fi
    for mt in $MAX_TOKENS_LIST; do
        tok_tag="$mt"
        [[ "$mt" == "3072" ]] && tok_tag="3k"
        [[ "$mt" == "8192" ]] && tok_tag="8k"
        output_dir="$RESULTS_BASE/${exp_name}_${tok_tag}"

        # Figure out which of the requested benchmarks are missing
        # (--force ignores existing summaries and re-runs everything)
        if $FORCE; then
            missing="$BENCHMARKS"
        else
            missing=""
            for bm in $BENCHMARKS; do
                if [[ ! -f "$output_dir/$bm/summary.json" ]]; then
                    missing="$missing $bm"
                fi
            done
            missing=$(echo $missing | xargs)
        fi
        [[ -z "$missing" ]] && continue

        line="$model_path|$exp_name|$mt|$tok_tag|$missing"
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
echo "  Supplementary / Resume Evaluation"
echo "  Ckpt root:     $CKPT_ROOT"
echo "  Match pattern: $PATTERN"
[[ -n "$LAYERS" ]] && echo "  Layer filter:  $LAYERS  (skipped by filter: $SKIPPED_LAYER_FILTER)"
$INCLUDE_BASE && echo "  Base model:    $BASE_MODEL  →  $BASE_NAME (queued first)"
echo "  Benchmarks:    $BENCHMARKS"
echo "  Mode:          $($FORCE && echo 'FORCE — re-run everything' || echo 'resume — only fill missing benches')"
echo "  Pending tasks: $TOTAL_TASKS"
echo "  Skipped (no valid ckpt): ${#SKIPPED_NO_CKPT[@]}"
echo "  Max tokens:    $MAX_TOKENS_LIST"
echo "  GPUs:          ${GPU_LIST[*]} → $NUM_WORKERS workers × TP=$TP"
echo "  Params:        T=$TEMPERATURE P=$TOP_P K=$TOP_K"
echo "  WandB:         $WANDB_ENTITY / $WANDB_PROJECT"
echo "  Results:       $RESULTS_BASE (existing dirs will gain new subdirs)"
echo "============================================================"
if (( ${#SKIPPED_NO_CKPT[@]} > 0 )); then
    echo "Skipped (no global_step_*/.../config.json found):"
    for n in "${SKIPPED_NO_CKPT[@]}"; do echo "  - $n"; done
    echo ""
fi
echo ""

if $DRY_RUN; then
    echo "[DRY RUN] Would evaluate:"
    while IFS='|' read -r path name mt tok missing; do
        echo "  $name ($tok) — missing: $missing"
    done < "$QUEUE_FILE"
    echo ""
    echo "Total: $TOTAL_TASKS tasks"
    exit 0
fi

if [[ $TOTAL_TASKS -eq 0 ]]; then
    echo "All requested benchmarks already completed!"
    exit 0
fi

# ========== Worker function ==========
worker() {
    local gpu_slice="$1"   # comma-list of GPU ids, e.g. "0,1"
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

        IFS='|' read -r model_path exp_name max_tokens tok_tag missing_bms <<< "$task"
        local run_label="${exp_name}_${tok_tag}_supp"
        local output_dir="$RESULTS_BASE/${exp_name}_${tok_tag}"
        local log_file="$LOG_DIR/${run_label}.log"

        echo "[GPUs $gpu_slice] START $run_label — bms: $missing_bms"

        if CUDA_VISIBLE_DEVICES="$gpu_slice" $PYTHON "$SCRIPT_DIR/eval.py" \
            --backend vllm \
            --model "$model_path" \
            --benchmarks $missing_bms \
            --tensor-parallel-size $TP \
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
            echo "[GPUs $gpu_slice] DONE  $run_label (worker total: $completed)"
        else
            failed=$((failed + 1))
            echo "[GPUs $gpu_slice] FAIL  $run_label — see $log_file"
        fi
    done

    echo "[GPUs $gpu_slice] Worker finished. Completed: $completed, Failed: $failed"
}

# ========== Launch workers ==========
WORKER_PIDS=()
for slice in "${GPU_SLICES[@]}"; do
    worker "$slice" &
    WORKER_PIDS+=($!)
    echo "Worker launched on GPUs $slice (PID $!, TP=$TP)"
done

for pid in "${WORKER_PIDS[@]}"; do
    wait "$pid" || true
done

echo ""
echo "============================================================"
echo "  All workers finished. Rebuilding overall summaries..."
echo "============================================================"

# ========== Rebuild overall_summary.json from all per-benchmark files ==========
# The original overall_summary.json was written without the new benchmarks;
# rebuild so downstream tools see the complete picture.
RESULTS_BASE_ESC="$RESULTS_BASE" PATTERN_ESC="$PATTERN" $PYTHON << 'PYEOF'
import json, os
from pathlib import Path

results_base = Path(os.environ["RESULTS_BASE_ESC"])
pattern = os.environ["PATTERN_ESC"]
count = 0
# Match ckpt dirs by pattern. Output dirs have a trailing _3k/_8k suffix,
# so we glob "<pattern>*" to catch both.
for d in sorted(results_base.glob(f"{pattern}*")):
    summaries = sorted(d.glob("*/summary.json"))
    if not summaries:
        continue
    overall = d / "overall_summary.json"
    existing = {}
    if overall.exists():
        try:
            existing = json.loads(overall.read_text())
        except Exception:
            existing = {}
    results = {}
    details = []
    for sf in summaries:
        s = json.loads(sf.read_text())
        results[s["benchmark"]] = s["accuracy"]
        details.append(s)
    existing["results"] = results
    existing["details"] = details
    existing.setdefault("model", d.name)
    overall.write_text(json.dumps(existing, indent=2))
    count += 1
print(f"Rebuilt {count} overall_summary.json files")
PYEOF

echo ""
echo "  Results:  $RESULTS_BASE"
echo "  WandB:    https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"
echo "============================================================"
