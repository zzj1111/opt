#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="/mnt/data1/zha00175/miniconda/envs/gsm8k-eval/bin/python"
BENCHMARKS="math500 gsm8k aime2024 aime2025 amc olympiadbench gpqa_diamond mbpp bbh mmlu_pro mgsm ceval ifeval"
COMMON="--backend vllm --tensor-parallel-size 1 --temperature 0.6 --top-p 0.95 --top-k 20 --seed 42 --enable-thinking --trust-remote-code"

mkdir -p logs

declare -a NAMES=("layer0" "layer14" "layer27")
declare -a PATHS=(
    "/home/zha00175/MathCodeCKPT/layer0/huggingface"
    "/home/zha00175/MathCodeCKPT/layer14/huggingface"
    "/home/zha00175/MathCodeCKPT/layer27/huggingface"
)

# 3 rounds: each runs 3k on GPU6 + 8k on GPU7 in parallel
for i in 0 1 2; do
    name="${NAMES[$i]}"
    path="${PATHS[$i]}"

    echo ""
    echo "=== Round $((i+1))/3: $name ==="
    echo "  [GPU 6] ${name}_3k (max_tokens=3072)"
    echo "  [GPU 7] ${name}_8k (max_tokens=8192)"

    CUDA_VISIBLE_DEVICES=6 $PYTHON eval.py \
        --model "$path" --benchmarks $BENCHMARKS \
        --max-tokens 3072 --output-dir "results/${name}_3k" \
        $COMMON > "logs/${name}_3k.log" 2>&1 &
    PID_3K=$!

    CUDA_VISIBLE_DEVICES=7 $PYTHON eval.py \
        --model "$path" --benchmarks $BENCHMARKS \
        --max-tokens 8192 --output-dir "results/${name}_8k" \
        $COMMON > "logs/${name}_8k.log" 2>&1 &
    PID_8K=$!

    wait $PID_3K && echo "  [OK] ${name}_3k" || echo "  [FAIL] ${name}_3k"
    wait $PID_8K && echo "  [OK] ${name}_8k" || echo "  [FAIL] ${name}_8k"
done

echo ""
echo "All done!"
