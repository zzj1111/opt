#!/bin/bash
# Math Benchmark Evaluation Script
# Evaluate a single model on specified benchmarks.
#
# Usage:
#   ./run.sh <model_path> [options]
#
# Examples:
#   ./run.sh /path/to/model
#   ./run.sh /path/to/model --benchmarks all
#   ./run.sh /path/to/model --benchmarks math500 gsm8k --enable-thinking
#   ./run.sh /path/to/model --max-tokens 8192 --temperature 0.6

set -euo pipefail

# ========== Defaults ==========
TP_SIZE=4
DTYPE="auto"
GPU_MEM_UTIL=0.90
MAX_TOKENS=4096
TEMPERATURE=0.6
TOP_P=0.95
TOP_K=20
SEED=42
BENCHMARKS="math500 amc aime2024 aime2025 olympiadbench gpqa_diamond"
EXTRA_ARGS=""

# ========== Parse Arguments ==========
if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_path> [options]"
    echo ""
    echo "Options:"
    echo "  --benchmarks <list>      Benchmarks to run (default: math competition set)"
    echo "  --max-tokens <int>       Max generation tokens (default: $MAX_TOKENS)"
    echo "  --temperature <float>    Sampling temperature (default: $TEMPERATURE)"
    echo "  --top-p <float>          Top-p sampling (default: $TOP_P)"
    echo "  --top-k <int>            Top-k sampling (default: $TOP_K)"
    echo "  --tp <int>               Tensor parallel size (default: $TP_SIZE)"
    echo "  --enable-thinking        Enable Qwen3 thinking mode"
    echo "  --output-dir <path>      Override output directory"
    echo "  --gpus <ids>             CUDA_VISIBLE_DEVICES (default: 0,1,2,3)"
    exit 1
fi

MODEL="$1"
shift

GPUS="0,1,2,3"
OUTPUT_DIR=""
ENABLE_THINKING=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --benchmarks)   shift; BENCHMARKS="$1" ;;
        --max-tokens)   shift; MAX_TOKENS="$1" ;;
        --temperature)  shift; TEMPERATURE="$1" ;;
        --top-p)        shift; TOP_P="$1" ;;
        --top-k)        shift; TOP_K="$1" ;;
        --tp)           shift; TP_SIZE="$1" ;;
        --enable-thinking) ENABLE_THINKING="--enable-thinking" ;;
        --output-dir)   shift; OUTPUT_DIR="$1" ;;
        --gpus)         shift; GPUS="$1" ;;
        *)              EXTRA_ARGS="$EXTRA_ARGS $1" ;;
    esac
    shift
done

# ========== Derived Config ==========
export CUDA_VISIBLE_DEVICES="$GPUS"

MODEL_NAME=$(basename "$MODEL")
if [ -z "$OUTPUT_DIR" ]; then
    SUFFIX="t${TEMPERATURE}_p${TOP_P}_k${TOP_K}"
    [ -n "$ENABLE_THINKING" ] && SUFFIX="${SUFFIX}_thinking"
    OUTPUT_DIR="results/${MODEL_NAME}_${SUFFIX}"
fi

# ========== Run ==========
CONDA_ENV="gsm8k-eval"
PYTHON="/mnt/data1/zha00175/miniconda/envs/${CONDA_ENV}/bin/python"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR"

echo "=================================================="
echo "  Model:       $MODEL"
echo "  Benchmarks:  $BENCHMARKS"
echo "  Output:      $OUTPUT_DIR"
echo "  GPUs:        $CUDA_VISIBLE_DEVICES"
echo "  Params:      T=$TEMPERATURE P=$TOP_P K=$TOP_K max_tokens=$MAX_TOKENS"
echo "=================================================="

exec $PYTHON eval.py \
    --backend vllm \
    --model "$MODEL" \
    --benchmarks $BENCHMARKS \
    --tensor-parallel-size $TP_SIZE \
    --dtype $DTYPE \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --max-tokens $MAX_TOKENS \
    --temperature $TEMPERATURE \
    --top-p $TOP_P \
    --top-k $TOP_K \
    --seed $SEED \
    --output-dir "$OUTPUT_DIR" \
    $ENABLE_THINKING \
    $EXTRA_ARGS
