#!/bin/bash
# Parallel evaluation of all 0407 layer-trained Qwen3-1.7B-Base models.
# Runs 8 models in parallel (one per H200 GPU), TP=1.
#
# Benchmarks: all 15 (math, code, reasoning, knowledge, instruction-following)
#
# Usage:
#   ./run_0407_eval.sh              # Run all models
#   ./run_0407_eval.sh --dry-run    # Print what would run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/mnt/data1/zha00175/miniconda/envs/gsm8k-eval/bin/python"
MODEL_BASE="/local1/vllm_models/hub/Zijian"
RESULTS_BASE="$SCRIPT_DIR/results"
LOG_DIR="$SCRIPT_DIR/logs/0407_eval"

# All benchmarks
BENCHMARKS="math500 gsm8k mbpp humaneval arc_challenge mmlu_pro bbh mgsm ceval amc aime2024 aime2025 olympiadbench gpqa_diamond ifeval"

# Generation params
TEMPERATURE=0.6
TOP_P=0.95
TOP_K=20
MAX_TOKENS=4096
SEED=42

# average@N sampling for competition benchmarks
AVG_AT_MAP="amc:32,aime2024:32,aime2025:32"

# Number of GPUs
NUM_GPUS=8

DRY_RUN=false
[ "${1:-}" = "--dry-run" ] && DRY_RUN=true

mkdir -p "$LOG_DIR"

# Collect all 0407 model directories
MODELS=()
for d in "$MODEL_BASE"/0407_*; do
    [ -d "$d" ] && MODELS+=("$d")
done

TOTAL=${#MODELS[@]}
echo "========================================"
echo "  0407 Layer Training Evaluation"
echo "  Models: $TOTAL"
echo "  Benchmarks: $BENCHMARKS"
echo "  GPUs: $NUM_GPUS x H200 (TP=1)"
echo "  Params: T=$TEMPERATURE P=$TOP_P K=$TOP_K max_tokens=$MAX_TOKENS"
echo "  Avg@N:  $AVG_AT_MAP"
echo "========================================"
echo ""

if $DRY_RUN; then
    echo "[DRY RUN] Would evaluate:"
    for m in "${MODELS[@]}"; do
        echo "  $(basename "$m")"
    done
    exit 0
fi

# Track overall progress
COMPLETED=0
FAILED_MODELS=()

run_model() {
    local model_path="$1"
    local gpu_id="$2"
    local model_name
    model_name=$(basename "$model_path")
    local output_dir="$RESULTS_BASE/${model_name}_t${TEMPERATURE}_p${TOP_P}_k${TOP_K}"
    local log_file="$LOG_DIR/${model_name}.log"

    # Skip if already completed
    if [ -f "$output_dir/overall_summary.json" ]; then
        echo "  [SKIP] $model_name — already completed"
        return 0
    fi

    echo "  [GPU $gpu_id] $model_name"

    CUDA_VISIBLE_DEVICES="$gpu_id" $PYTHON "$SCRIPT_DIR/eval.py" \
        --backend vllm \
        --model "$model_path" \
        --benchmarks $BENCHMARKS \
        --tensor-parallel-size 1 \
        --dtype auto \
        --gpu-memory-utilization 0.90 \
        --max-tokens $MAX_TOKENS \
        --temperature $TEMPERATURE \
        --top-p $TOP_P \
        --top-k $TOP_K \
        --seed $SEED \
        --avg-at-map "$AVG_AT_MAP" \
        --output-dir "$output_dir" \
        > "$log_file" 2>&1

    return $?
}

# Process models in batches of NUM_GPUS
batch_idx=0
for ((i=0; i<TOTAL; i+=NUM_GPUS)); do
    batch_idx=$((batch_idx + 1))
    batch_end=$((i + NUM_GPUS))
    [ $batch_end -gt $TOTAL ] && batch_end=$TOTAL
    batch_size=$((batch_end - i))

    echo ""
    echo "--- Batch $batch_idx: models $((i+1))-$batch_end / $TOTAL ---"

    PIDS=()
    for ((j=0; j<batch_size; j++)); do
        model_idx=$((i + j))
        gpu_id=$j
        run_model "${MODELS[$model_idx]}" "$gpu_id" &
        PIDS+=($!)
    done

    # Wait for all in this batch
    for pid_idx in "${!PIDS[@]}"; do
        model_idx=$((i + pid_idx))
        model_name=$(basename "${MODELS[$model_idx]}")
        if wait "${PIDS[$pid_idx]}"; then
            COMPLETED=$((COMPLETED + 1))
            echo "  [DONE] $model_name ($COMPLETED/$TOTAL)"
        else
            FAILED_MODELS+=("$model_name")
            echo "  [FAIL] $model_name — see $LOG_DIR/${model_name}.log"
        fi
    done
done

echo ""
echo "========================================"
echo "  Evaluation Complete"
echo "  Succeeded: $COMPLETED / $TOTAL"
if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo "  Failed: ${FAILED_MODELS[*]}"
fi
echo "========================================"

# Generate comparison table
if [ $COMPLETED -ge 2 ]; then
    echo ""
    echo "Generating comparison table..."
    $PYTHON "$SCRIPT_DIR/compare_results.py" "$RESULTS_BASE/" --format csv \
        > "$RESULTS_BASE/summary_0407_eval.csv" 2>/dev/null || true
    echo "  Saved to: $RESULTS_BASE/summary_0407_eval.csv"
fi
