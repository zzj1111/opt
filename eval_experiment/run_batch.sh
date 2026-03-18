#!/usr/bin/env bash
set -euo pipefail

cd eval_experiment/

HF_TOKEN=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 \
bash scripts/batch_eval.sh \
    --device cuda --parallel 8 --max-gen-toks 3072 --force \
    --hf-upload Mingyi-Hong/opt_RL_eval \
    full_rlvr_code:/code/hongpaul-sandbox/temp/OPT-RL/opt/checkpoints/mbpp_full_0318_1543_Qwen3-1.7B_lr1e-6/global_step_90/actor/huggingface \
    layer13_code:/code/hongpaul-sandbox/temp/OPT-RL/opt/checkpoints/checkpoints/mbpp_0318_0058_Qwen3-1.7B_lr2e-5_layer_15/global_step_90/actor/huggingface

echo "========================================================"
echo "  Eval done. Starting dummy GPU hold job..."
echo "========================================================"
DUMMY_RUN_NAME="dummy_eval_$(hostname)_$(date +%m%d_%H%M)" \
    python /code/hongpaul-sandbox/temp/OPT-RL/opt/examples/grpo_trainer/dummy_gpu_hold.py
