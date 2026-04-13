#!/bin/bash
# Re-run MBPP (fixed prompt) for all 12 models +补 mathcodefull_3k bbh
# 6 GPUs (0-3, 6-7), TP=1, 2 rounds of 6

set -euo pipefail
cd "$(dirname "$0")"

PYTHON="/mnt/data1/zha00175/miniconda/envs/gsm8k-eval/bin/python"
COMMON="--backend vllm --tensor-parallel-size 1 --temperature 0.6 --top-p 0.95 --top-k 20 --seed 42 --enable-thinking --trust-remote-code"

# Model definitions: "result_dir|model_path|max_tokens"
MODELS=(
    "Qwen3-1.7B_3k|Qwen/Qwen3-1.7B|3072"
    "Qwen3-1.7B_8k|Qwen/Qwen3-1.7B|8192"
    "opt_RL_ckpt_3k|/home/zha00175/opt_RL_ckpt/actor/huggingface|3072"
    "opt_RL_ckpt_8k|/home/zha00175/opt_RL_ckpt/actor/huggingface|8192"
    "codefull_3k|/home/zha00175/MathCodeCKPT/codefull/huggingface|3072"
    "codefull_8k|/home/zha00175/MathCodeCKPT/codefull/huggingface|8192"
    "coderlayer_3k|/home/zha00175/MathCodeCKPT/coderlayer/actor/huggingface|3072"
    "coderlayer_8k|/home/zha00175/MathCodeCKPT/coderlayer/actor/huggingface|8192"
    "mathcodefull_3k|/home/zha00175/MathCodeCKPT/mathcodefull/huggingface|3072"
    "mathcodefull_8k|/home/zha00175/MathCodeCKPT/mathcodefull/huggingface|8192"
    "mathcodelayer_3k|/home/zha00175/MathCodeCKPT/mathcodelayer/actor/huggingface|3072"
    "mathcodelayer_8k|/home/zha00175/MathCodeCKPT/mathcodelayer/actor/huggingface|8192"
)

GPUS=(0 1 2 3 6 7)
NGPUS=${#GPUS[@]}

run_batch() {
    local start=$1
    local end=$2
    local pids=()

    for ((i=start; i<end && i<${#MODELS[@]}; i++)); do
        IFS='|' read -r name model maxtok <<< "${MODELS[$i]}"
        gpu_idx=$(( (i - start) % NGPUS ))
        gpu=${GPUS[$gpu_idx]}

        # Determine benchmarks: mbpp for all, + bbh for mathcodefull_3k
        benchmarks="mbpp"
        if [[ "$name" == "mathcodefull_3k" ]]; then
            benchmarks="mbpp bbh"
        fi

        echo "[GPU $gpu] $name: $benchmarks (max_tokens=$maxtok)"
        CUDA_VISIBLE_DEVICES=$gpu $PYTHON eval.py \
            --model "$model" \
            --benchmarks $benchmarks \
            --max-tokens $maxtok \
            --output-dir "results/$name" \
            $COMMON \
            > "logs/rerun_mbpp_${name}.log" 2>&1 &
        pids+=($!)
    done

    # Wait for all
    for pid in "${pids[@]}"; do
        wait $pid && echo "  [OK] PID $pid done" || echo "  [FAIL] PID $pid"
    done
}

mkdir -p logs

echo "============================================"
echo "  Re-running MBPP (fixed prompt) - 12 models"
echo "  GPUs: ${GPUS[*]}"
echo "============================================"

echo ""
echo "--- Round 1/2 (models 1-6) ---"
run_batch 0 6

echo ""
echo "--- Round 2/2 (models 7-12) ---"
run_batch 6 12

echo ""
echo "============================================"
echo "  All done! Generating updated summary..."
echo "============================================"

$PYTHON << 'PYEOF'
import json, csv, os

models = [
    'Qwen3-1.7B_3k', 'Qwen3-1.7B_8k',
    'opt_RL_ckpt_3k', 'opt_RL_ckpt_8k',
    'codefull_3k', 'codefull_8k',
    'coderlayer_3k', 'coderlayer_8k',
    'mathcodefull_3k', 'mathcodefull_8k',
    'mathcodelayer_3k', 'mathcodelayer_8k',
]
benchmarks = ['math500', 'gsm8k', 'aime2024', 'aime2025', 'amc', 'olympiadbench',
              'gpqa_diamond', 'mbpp', 'bbh', 'mmlu_pro', 'mgsm', 'ceval', 'ifeval']

rows = []
for model in models:
    row = {'model': model}
    for bench in benchmarks:
        sf = f'results/{model}/{bench}/summary.json'
        if os.path.exists(sf):
            with open(sf) as f:
                s = json.load(f)
            row[bench] = round(s['accuracy'] * 100, 2)
        else:
            row[bench] = '-'
    rows.append(row)

csv_path = 'results/summary_all_models.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['model'] + benchmarks)
    writer.writeheader()
    writer.writerows(rows)

for row in rows:
    vals = [f"{row[b]:.1f}" if isinstance(row[b], float) else row[b] for b in benchmarks]
    print(f"{row['model']:<22} " + " ".join(f"{v:>8}" for v in vals))
print(f"\nCSV: {csv_path}")
PYEOF
