#!/bin/bash
# Batch evaluation: run multiple models sequentially.
#
# Usage:
#   ./run_batch.sh                       # Edit MODELS array below
#   ./run_batch.sh models.txt            # One model path per line
#
# models.txt format (lines starting with # are skipped):
#   /path/to/model1
#   /path/to/model2  --enable-thinking
#   # /path/to/skipped_model

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ========== Shared Config ==========
BENCHMARKS="math500 amc aime2024 aime2025 olympiadbench gpqa_diamond"
COMMON_ARGS="--temperature 0.6 --top-p 0.95 --top-k 20 --max-tokens 4096"

# ========== Model List ==========
# Option 1: inline (edit here)
MODELS=(
    # "/path/to/baseline_model"
    # "/path/to/rl_model  --enable-thinking"
)

# Option 2: from file
if [ "${1:-}" != "" ] && [ -f "$1" ]; then
    MODELS=()
    while IFS= read -r line || [ -n "$line" ]; do
        line=$(echo "$line" | sed 's/#.*//' | xargs)
        [ -n "$line" ] && MODELS+=("$line")
    done < "$1"
fi

if [ ${#MODELS[@]} -eq 0 ]; then
    echo "No models configured."
    echo "Either edit MODELS array in this script or run: $0 models.txt"
    exit 1
fi

# ========== Run ==========
TOTAL=${#MODELS[@]}
echo "Batch evaluation: $TOTAL models"
echo ""

FAILED=()
for i in "${!MODELS[@]}"; do
    ENTRY="${MODELS[$i]}"
    # First token is model path, rest are extra args
    MODEL_PATH=$(echo "$ENTRY" | awk '{print $1}')
    EXTRA_ARGS=$(echo "$ENTRY" | awk '{$1=""; print $0}' | xargs)
    MODEL_NAME=$(basename "$MODEL_PATH")

    echo "[$((i+1))/$TOTAL] $MODEL_NAME"
    if "$SCRIPT_DIR/run.sh" "$MODEL_PATH" --benchmarks $BENCHMARKS $COMMON_ARGS $EXTRA_ARGS; then
        echo "  -> Done"
    else
        echo "  -> FAILED"
        FAILED+=("$MODEL_NAME")
    fi
    echo ""
done

# ========== Summary ==========
echo "=================================================="
echo "  Batch complete: $((TOTAL - ${#FAILED[@]}))/$TOTAL succeeded"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "  Failed: ${FAILED[*]}"
fi
echo "=================================================="

# Auto-generate comparison if multiple models ran
if [ $((TOTAL - ${#FAILED[@]})) -ge 2 ]; then
    echo ""
    echo "Generating comparison table..."
    python3 "$SCRIPT_DIR/compare_results.py" results/
fi
