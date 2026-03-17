# Single-Layer RLVR Off-Target Capability Evaluation

Evaluates 30 checkpoints (28 single-layer + 1 full RLVR + 1 baseline) across 8 benchmarks.

## Directory Structure

```
eval_experiment/
├── configs/               # Per-benchmark eval settings
│   ├── math500.yaml       # MATH-500 (0-shot, on-target)
│   ├── gsm8k.yaml         # GSM8K (8-shot)
│   ├── mbpp.yaml          # MBPP (3-shot, execution-based)
│   ├── ifeval.yaml        # IFEval strict (0-shot)
│   ├── mmlu_pro.yaml      # MMLU-Pro (5-shot CoT, ~2K subsample)
│   ├── bbh.yaml           # BBH (3-shot CoT, ~2K subsample)
│   ├── mgsm.yaml          # MGSM (8-shot CoT, 10 languages)
│   └── ceval.yaml         # C-Eval (5-shot)
├── checkpoints/
│   └── registry.yaml      # label → HF model path mapping
├── scripts/
│   ├── run_eval.sh        # Main entry point
│   ├── run_single.py      # One (checkpoint × benchmark) job
│   ├── aggregate.py       # Raw results → scores.csv + deltas.csv
│   ├── visualize.py       # Heatmap + Pareto plot
│   └── subsample.py       # Generate fixed subsample indices (run once)
└── results/
    ├── raw/               # lm-eval JSON output per (ckpt, benchmark)
    ├── indices/           # Fixed subsample indices (commit to git!)
    ├── scores.csv         # 30 × 8 raw scores
    ├── deltas.csv         # 30 × 8 Δ vs baseline
    ├── heatmap.png        # Main visualization
    └── pareto.png         # MATH gain vs off-target trade-off
```

## Quick Start

### 1. Install dependencies

```bash
pip install "lm-eval[math,ifeval,sentencepiece]>=0.4.0" pyyaml seaborn matplotlib pandas
```

### 2. Fill in checkpoint paths

Edit `checkpoints/registry.yaml` to point each label to its HF model directory:

```yaml
baseline:   Qwen/Qwen3-1.7B-Instruct
full_rlvr:  /path/to/full_rlvr/global_step_XXXX/actor/huggingface
layer_0:    /path/to/layer_0/global_step_XXXX/actor/huggingface
...
```

### 3. Generate subsample indices (once)

```bash
cd eval_experiment/
python scripts/subsample.py --seed 42
# Commit the generated results/indices/ files
git add results/indices/ && git commit -m "fix: subsample indices for reproducibility"
```

### 4. Run evaluation

```bash
cd eval_experiment/

# All checkpoints × all benchmarks (sequential, GPU 0)
bash scripts/run_eval.sh

# Specific subset
bash scripts/run_eval.sh --benchmarks math500,gsm8k --checkpoints baseline,layer_13

# 2 GPUs in parallel (GPUs 0 and 1)
bash scripts/run_eval.sh --parallel 2

# Resume (skips already-completed jobs automatically)
bash scripts/run_eval.sh
```

### 5. Aggregate and visualize

```bash
cd eval_experiment/
python scripts/aggregate.py     # → results/scores.csv, results/deltas.csv
python scripts/visualize.py     # → results/heatmap.png, results/pareto.png
```

## Key Notes

- **Thinking mode**: All evaluations use `/no_think` system prompt to disable Qwen3 thinking
- **Reproducibility**: MMLU-Pro and BBH use fixed subsample indices (`results/indices/`)
- **Resume**: `run_eval.sh` skips jobs with existing valid JSON results; use `--force` to rerun
- **LR calibration**: Ensure each single-layer checkpoint used `1/√k` LR scaling before evaluating
- **Metrics**:
  - MATH-500: `exact_match`
  - GSM8K: `exact_match,flexible-extract`
  - MBPP: `pass@1` (execution-based)
  - IFEval: `prompt_level_strict_acc`
  - MMLU-Pro / C-Eval: `acc`
  - BBH: `acc_norm`
  - MGSM: `acc` (averaged over 10 languages)
