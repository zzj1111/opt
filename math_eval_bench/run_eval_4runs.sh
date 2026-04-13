#!/bin/bash
# 4 runs: 2 models x 2 token lengths (3K, 8K), all benchmarks, thinking enabled
# Uses GPU 6-7 only, TP=1, runs 2 in parallel per pair

set -euo pipefail
cd "$(dirname "$0")"

CONDA_ENV="gsm8k-eval"
PYTHON="/mnt/data1/zha00175/miniconda/envs/${CONDA_ENV}/bin/python"
BENCHMARKS="math500 gsm8k aime2024 aime2025 amc olympiadbench gpqa_diamond mbpp bbh mmlu_pro mgsm ceval ifeval"
COMMON="--backend vllm --tensor-parallel-size 1 --temperature 0.6 --top-p 0.95 --top-k 20 --seed 42 --enable-thinking --trust-remote-code"

MODEL_BASE="Qwen/Qwen3-1.7B"
MODEL_RL="/home/zha00175/opt_RL_ckpt/actor/huggingface"

echo "=========================================="
echo "  Starting 4 evaluation runs"
echo "  Models: Qwen3-1.7B, opt_RL_ckpt"
echo "  Token lengths: 3072, 8192"
echo "  GPUs: 6, 7 (TP=1, 2 parallel)"
echo "=========================================="

# --- Pair 1: Qwen3-1.7B 3K (GPU6) + 8K (GPU7) ---
echo "[Pair 1/2] Qwen3-1.7B: 3K + 8K in parallel..."

CUDA_VISIBLE_DEVICES=6 $PYTHON eval.py \
    --model "$MODEL_BASE" --benchmarks $BENCHMARKS \
    --max-tokens 3072 --output-dir results/Qwen3-1.7B_3k \
    $COMMON \
    > logs/Qwen3-1.7B_3k.log 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=7 $PYTHON eval.py \
    --model "$MODEL_BASE" --benchmarks $BENCHMARKS \
    --max-tokens 8192 --output-dir results/Qwen3-1.7B_8k \
    $COMMON \
    > logs/Qwen3-1.7B_8k.log 2>&1 &
PID2=$!

echo "  Qwen3-1.7B 3K PID=$PID1 (GPU 6)"
echo "  Qwen3-1.7B 8K PID=$PID2 (GPU 7)"
wait $PID1 && echo "  [OK] Qwen3-1.7B 3K done" || echo "  [FAIL] Qwen3-1.7B 3K failed"
wait $PID2 && echo "  [OK] Qwen3-1.7B 8K done" || echo "  [FAIL] Qwen3-1.7B 8K failed"

# --- Pair 2: opt_RL_ckpt 3K (GPU6) + 8K (GPU7) ---
echo ""
echo "[Pair 2/2] opt_RL_ckpt: 3K + 8K in parallel..."

CUDA_VISIBLE_DEVICES=6 $PYTHON eval.py \
    --model "$MODEL_RL" --benchmarks $BENCHMARKS \
    --max-tokens 3072 --output-dir results/opt_RL_ckpt_3k \
    $COMMON \
    > logs/opt_RL_ckpt_3k.log 2>&1 &
PID3=$!

CUDA_VISIBLE_DEVICES=7 $PYTHON eval.py \
    --model "$MODEL_RL" --benchmarks $BENCHMARKS \
    --max-tokens 8192 --output-dir results/opt_RL_ckpt_8k \
    $COMMON \
    > logs/opt_RL_ckpt_8k.log 2>&1 &
PID4=$!

echo "  opt_RL_ckpt 3K PID=$PID3 (GPU 6)"
echo "  opt_RL_ckpt 8K PID=$PID4 (GPU 7)"
wait $PID3 && echo "  [OK] opt_RL_ckpt 3K done" || echo "  [FAIL] opt_RL_ckpt 3K failed"
wait $PID4 && echo "  [OK] opt_RL_ckpt 8K done" || echo "  [FAIL] opt_RL_ckpt 8K failed"

# --- Comparison ---
echo ""
echo "=========================================="
echo "  All runs complete. Generating comparison..."
echo "=========================================="
$PYTHON compare_results.py \
    results/Qwen3-1.7B_3k \
    results/Qwen3-1.7B_8k \
    results/opt_RL_ckpt_3k \
    results/opt_RL_ckpt_8k

echo ""
$PYTHON compare_results.py \
    results/Qwen3-1.7B_3k \
    results/Qwen3-1.7B_8k \
    results/opt_RL_ckpt_3k \
    results/opt_RL_ckpt_8k \
    --format markdown
