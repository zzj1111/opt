"""Run lm-evaluation-harness for one (checkpoint, benchmark) pair.

Usage (server, multi-GPU):
    python scripts/run_single.py \\
        --ckpt-label layer_13 --ckpt-path /path/to/hf_model \\
        --benchmark mbpp --gpu 0

Usage (local Mac/MPS):
    python scripts/run_single.py \\
        --ckpt-label baseline --ckpt-path Qwen/Qwen3-1.7B-Instruct \\
        --benchmark gsm8k --device mps --batch-size 4

Usage (local CPU):
    python scripts/run_single.py \\
        --ckpt-label baseline --ckpt-path Qwen/Qwen3-1.7B-Instruct \\
        --benchmark gsm8k --device cpu --batch-size 1 --lite

Usage (limited VRAM, 4-bit):
    python scripts/run_single.py \\
        --ckpt-label layer_13 --ckpt-path /path/to/hf_model \\
        --benchmark math500 --load-in-4bit
"""

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import yaml


BENCHMARK_ORDER = [
    "math500", "gsm8k", "mbpp", "ifeval", "mmlu_pro", "bbh", "mgsm", "ceval"
]

TASK_OVERRIDES = {
    "bbh":  "bbh",
    "mgsm": "mgsm_direct",
}

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

# Reduced sample counts for --lite mode (quick local smoke test)
LITE_LIMITS = {
    "math500":  50,
    "gsm8k":    100,
    "mbpp":     50,
    "ifeval":   50,
    "mmlu_pro": 100,
    "bbh":      100,
    "mgsm":     50,
    "ceval":    100,
}

NO_THINK_SYSTEM_PROMPT = "You are a helpful assistant."
NO_THINK_SUFFIX = " /no_think"


def detect_device() -> str:
    """Auto-detect the best available device."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def dtype_for_device(device: str) -> str:
    """Choose dtype based on device. MPS doesn't support bfloat16 well."""
    if device == "mps":
        return "float16"
    if device == "cpu":
        return "float32"
    return "bfloat16"


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
    metric_key = METRIC_KEYS.get(benchmark)
    if metric_key is None:
        return None
    results = raw_result.get("results", {})
    scores = []
    for task_results in results.values():
        if metric_key in task_results:
            val = task_results[metric_key]
            if isinstance(val, (int, float)):
                scores.append(val)
    if not scores:
        flat = {k: v for d in results.values() for k, v in d.items()}
        return flat.get(metric_key)
    return sum(scores) / len(scores)


def build_model_args(
    ckpt_path: str,
    device: str,
    load_in_4bit: bool,
    load_in_8bit: bool,
) -> str:
    dtype = dtype_for_device(device)

    parts = [f"pretrained={ckpt_path}", f"dtype={dtype}"]

    if device == "cpu":
        # CPU: no device_map, explicit device
        parts.append("device=cpu")
    elif device == "mps":
        parts.append("device=mps")
    else:
        # CUDA: use device_map=auto so it handles multi-GPU or single-GPU
        parts.append("device_map=auto")

    if load_in_4bit:
        parts.append("load_in_4bit=True")
    elif load_in_8bit:
        parts.append("load_in_8bit=True")

    return ",".join(parts)


def run_lm_eval(
    ckpt_label: str,
    ckpt_path: str,
    benchmark: str,
    cfg: dict,
    output_dir: str,
    device: str = "auto",
    gpu: int = 0,
    batch_size: str = "auto",
    no_think: bool = True,
    lite: bool = False,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
) -> dict:
    if device == "auto":
        device = detect_device()

    out_path = result_path(output_dir, ckpt_label, benchmark)
    os.makedirs(output_dir, exist_ok=True)

    task_name = TASK_OVERRIDES.get(benchmark, cfg["task"])
    num_fewshot = cfg.get("num_fewshot", 0)
    gen_kwargs = cfg.get("gen_kwargs", {}).copy()
    limit = cfg.get("limit")
    subsample_file = cfg.get("subsample_indices_file")

    # --lite: override limit for quick local testing
    if lite:
        limit = LITE_LIMITS.get(benchmark, 50)

    # CPU/MPS: reduce max_gen_toks to speed things up
    if device in ("cpu", "mps") and not lite:
        gen_kwargs["max_gen_toks"] = min(gen_kwargs.get("max_gen_toks", 2048), 1024)

    gen_kwargs_str = ",".join(f"{k}={v}" for k, v in gen_kwargs.items())
    system_prompt = NO_THINK_SYSTEM_PROMPT + (NO_THINK_SUFFIX if no_think else "")
    model_args = build_model_args(ckpt_path, device, load_in_4bit, load_in_8bit)

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
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

    # Sample limit
    if subsample_file and Path(subsample_file).exists() and not lite:
        with open(subsample_file) as f:
            indices = json.load(f)
        cmd += ["--limit", str(len(indices))]
    elif limit is not None:
        cmd += ["--limit", str(limit)]

    env = os.environ.copy()
    # Only set CUDA_VISIBLE_DEVICES on CUDA; leave unset for cpu/mps
    if device == "cuda":
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    lite_tag = " [LITE]" if lite else ""
    print(f"\n{'='*60}")
    print(f"[run_single]{lite_tag} {ckpt_label} × {benchmark}")
    print(f"  device={device}, dtype={dtype_for_device(device)}", end="")
    if load_in_4bit:  print(", 4-bit", end="")
    if load_in_8bit:  print(", 8-bit", end="")
    print(f"\n  task={task_name}, fewshot={num_fewshot}, batch={batch_size}")
    print(f"  ckpt={ckpt_path}")
    print(f"  output={out_path}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        error = {"error": f"lm_eval exited with code {result.returncode}"}
        with open(out_path, "w") as f:
            json.dump(error, f)
        return error

    with open(out_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-label", required=True)
    parser.add_argument("--ckpt-path",  required=True)
    parser.add_argument("--benchmark",  required=True, choices=BENCHMARK_ORDER)
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--output-dir", default="results/raw")
    # Device selection
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "mps", "cpu"],
                        help="Compute device. 'auto' picks cuda>mps>cpu.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index (CUDA only, ignored for mps/cpu)")
    parser.add_argument("--batch-size", default="auto",
                        help="Batch size or 'auto' (lm-eval auto-tunes)")
    # Quantization (for limited VRAM)
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load model in 4-bit (bitsandbytes, requires CUDA)")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Load model in 8-bit (bitsandbytes, requires CUDA)")
    # Misc
    parser.add_argument("--no-think", action="store_true", default=True)
    parser.add_argument("--lite", action="store_true",
                        help="Quick smoke test: use small sample limits locally")
    parser.add_argument("--force", action="store_true")
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
        device=args.device,
        gpu=args.gpu,
        batch_size=args.batch_size,
        no_think=args.no_think,
        lite=args.lite,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )


if __name__ == "__main__":
    main()
