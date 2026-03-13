#!/usr/bin/env bash
# Run layer transplant evaluation on MATH dataset
#
# Usage:
#   bash examples/grpo_trainer/run_transplant_eval.sh
#   CUDA_VISIBLE_DEVICES=6,7 bash examples/grpo_trainer/run_transplant_eval.sh
#
# Prerequisites:
#   1. Download RL checkpoint:
#      huggingface-cli download Mingyi-Hong/opt_RL_ckpt --include "actor/huggingface/*" --local-dir ./opt_RL_ckpt --token YOUR_TOKEN
#   2. Eval data is already in data/math/test.parquet

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

BASE_MODEL="Qwen/Qwen3-1.7B"
RL_MODEL="${RL_MODEL:-$PROJ_DIR/opt_RL_ckpt/actor/huggingface}"
EVAL_DATASET="$PROJ_DIR/data/math/test.parquet"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJ_DIR/transplant_results/window5_$(date +%m%d_%H%M)}"
NUM_SAMPLES=500
BATCH_SIZE=8
MODE="window"
WINDOW_SIZE=5

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"

echo "Running transplant eval: mode=$MODE window=$WINDOW_SIZE samples=$NUM_SAMPLES"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "RL model: $RL_MODEL"
echo "Output: $OUTPUT_DIR"

python "$SCRIPT_DIR/layer_transplant.py" \
    --base_model "$BASE_MODEL" \
    --rl_model "$RL_MODEL" \
    --eval_dataset "$EVAL_DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --mode "$MODE" \
    --window_size "$WINDOW_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --num_samples "$NUM_SAMPLES"
