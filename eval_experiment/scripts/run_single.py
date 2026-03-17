"""Run lm-evaluation-harness for one (checkpoint, benchmark) pair.

Usage:
    python scripts/run_single.py \\
        --ckpt-label layer_13 \\
        --ckpt-path /path/to/hf_model \\
        --benchmark mbpp \\
        --config-dir configs/ \\
        --output-dir results/raw/ \\
        [--gpu 0] [--batch-size 16] [--no-think]
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


BENCHMARK_ORDER = [
    "math500", "gsm8k", "mbpp", "ifeval", "mmlu_pro", "bbh", "mgsm", "ceval"
]

# Map config filename → lm-eval task name (handles group tasks)
TASK_OVERRIDES = {
    # BBH in lm-eval is run as individual subtask group
    "bbh": "bbh",
    # MGSM runs all 10 language subtasks as a group
    "mgsm": "mgsm_direct",
}

# Metric key to extract from lm-eval output per benchmark
METRIC_KEYS = {
    "math500":  "exact_match,none",
    "gsm8k":    "exact_match,flexible-extract",
    "mbpp":     "pass@1,none",
    "ifeval":   "prompt_level_strict_acc,none",
    "mmlu_pro": "acc,none",
    "bbh":      "acc_norm,none",
    "mgsm":     "acc,none",
    "ceval":    "acc,none",
}

# Qwen3 system prompt to disable thinking mode
NO_THINK_SYSTEM_PROMPT = "You are a helpful assistant."
NO_THINK_SUFFIX = " /no_think"


def load_config(config_dir: str, benchmark: str) -> dict:
    path = Path(config_dir) / f"{benchmark}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def result_path(output_dir: str, ckpt_label: str, benchmark: str) -> Path:
    return Path(output_dir) / f"{ckpt_label}__{benchmark}.json"


def already_done(output_dir: str, ckpt_label: str, benchmark: str) -> bool:
    p = result_path(output_dir, ckpt_label, benchmark)
    if not p.exists():
        return False
    try:
        with open(p) as f:
            data = json.load(f)
        return "results" in data and "error" not in data
    except Exception:
        return False


def extract_score(raw_result: dict, benchmark: str) -> float | None:
    """Extract the primary metric score from lm-eval JSON output."""
    metric_key = METRIC_KEYS.get(benchmark)
    if metric_key is None:
        return None

    results = raw_result.get("results", {})

    # For group tasks (bbh, mgsm), average over subtasks
    scores = []
    for task_name, task_results in results.items():
        if metric_key in task_results:
            val = task_results[metric_key]
            if isinstance(val, (int, float)):
                scores.append(val)

    if not scores:
        # Try direct lookup
        flat = {k: v for d in results.values() for k, v in d.items()}
        if metric_key in flat:
            return flat[metric_key]
        return None

    return sum(scores) / len(scores)


def run_lm_eval(
    ckpt_label: str,
    ckpt_path: str,
    benchmark: str,
    cfg: dict,
    output_dir: str,
    gpu: int = 0,
    batch_size: int = 16,
    no_think: bool = True,
) -> dict:
    out_path = result_path(output_dir, ckpt_label, benchmark)
    os.makedirs(output_dir, exist_ok=True)

    task_name = TASK_OVERRIDES.get(benchmark, cfg["task"])
    num_fewshot = cfg.get("num_fewshot", 0)
    gen_kwargs = cfg.get("gen_kwargs", {})
    limit = cfg.get("limit")
    subsample_file = cfg.get("subsample_indices_file")

    # Build gen_kwargs string
    gk_parts = [f"{k}={v}" for k, v in gen_kwargs.items()]
    gen_kwargs_str = ",".join(gk_parts)

    # System prompt (with /no_think for Qwen3)
    system_prompt = NO_THINK_SYSTEM_PROMPT + (NO_THINK_SUFFIX if no_think else "")

    # lm-eval command
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={ckpt_path},dtype=bfloat16,device_map=auto",
        "--tasks", task_name,
        "--num_fewshot", str(num_fewshot),
        "--batch_size", str(batch_size),
        "--apply_chat_template",
        "--system_instruction", system_prompt,
        "--output_path", str(out_path),
        "--log_samples",
    ]

    if gen_kwargs_str:
        cmd += ["--gen_kwargs", gen_kwargs_str]

    # For subsampled benchmarks, use sample indices file if available
    if subsample_file and Path(subsample_file).exists():
        with open(subsample_file) as f:
            indices = json.load(f)
        # lm-eval supports --limit N; for exact index control use --sample_indices
        # Fall back to --limit if exact indices not supported
        cmd += ["--limit", str(len(indices))]
    elif limit is not None:
        cmd += ["--limit", str(limit)]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    print(f"\n{'='*60}")
    print(f"[run_single] {ckpt_label} × {benchmark}")
    print(f"  task={task_name}, fewshot={num_fewshot}")
    print(f"  ckpt={ckpt_path}")
    print(f"  output={out_path}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, env=env, capture_output=False)

    if result.returncode != 0:
        error = {"error": f"lm_eval exited with code {result.returncode}"}
        with open(out_path, "w") as f:
            json.dump(error, f)
        return error

    # Load and return result
    with open(out_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-label", required=True, help="e.g. layer_13, full_rlvr, baseline")
    parser.add_argument("--ckpt-path", required=True, help="HF model path or hub name")
    parser.add_argument("--benchmark", required=True, choices=BENCHMARK_ORDER)
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--output-dir", default="results/raw")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-think", action="store_true", default=True,
                        help="Disable Qwen3 thinking mode via /no_think system prompt")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if result already exists")
    args = parser.parse_args()

    if not args.force and already_done(args.output_dir, args.ckpt_label, args.benchmark):
        print(f"[skip] {args.ckpt_label} × {args.benchmark} already done")
        return

    cfg = load_config(args.config_dir, args.benchmark)
    run_lm_eval(
        ckpt_label=args.ckpt_label,
        ckpt_path=args.ckpt_path,
        benchmark=args.benchmark,
        cfg=cfg,
        output_dir=args.output_dir,
        gpu=args.gpu,
        batch_size=args.batch_size,
        no_think=args.no_think,
    )


if __name__ == "__main__":
    main()
