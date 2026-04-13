#!/usr/bin/env python3
"""Multi-Benchmark Math Evaluation CLI.

Evaluate language models on MATH-500, AMC, AIME 2024/2025, OlympiadBench, GPQA Diamond.

Examples:
    # Single benchmark
    python eval.py --backend vllm --model /path/to/model \\
        --benchmarks math500 --tensor-parallel-size 4 \\
        --output-dir results/my_model

    # Multiple benchmarks
    python eval.py --backend vllm --model /path/to/model \\
        --benchmarks math500 aime2024 aime2025 amc olympiadbench gpqa_diamond \\
        --tensor-parallel-size 4 --output-dir results/my_model

    # All benchmarks
    python eval.py --backend vllm --model /path/to/model \\
        --benchmarks all --tensor-parallel-size 4 \\
        --output-dir results/my_model
"""

import argparse
import json
import re
import time
from pathlib import Path

from tqdm import tqdm

from math_eval.answer import strip_thinking
from math_eval.backends import create_backend
from math_eval.benchmarks import BENCHMARKS, get_benchmark
from math_eval.io import write_json, write_jsonl


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a language model on math benchmarks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Benchmarks
    all_names = list(BENCHMARKS.keys())
    p.add_argument(
        "--benchmarks", type=str, nargs="+", required=True,
        help=f"Benchmarks to evaluate. Options: {all_names} or 'all'",
    )

    # Backend
    p.add_argument("--backend", type=str, default="vllm", choices=["vllm", "api"])
    p.add_argument("--model", type=str, required=True,
                   help="Model name/path (HF repo or local path)")

    # Generation
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument("--max-tokens", type=int, default=4096,
                   help="Max new tokens (default: 4096 for chain-of-thought)")
    p.add_argument("--stop", type=str, nargs="*", default=None)

    # vLLM-specific
    vllm_g = p.add_argument_group("vLLM options")
    vllm_g.add_argument("--tensor-parallel-size", type=int, default=1)
    vllm_g.add_argument("--dtype", type=str, default="auto",
                        choices=["auto", "float16", "bfloat16", "float32"])
    vllm_g.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    vllm_g.add_argument("--max-model-len", type=int, default=None)
    vllm_g.add_argument("--trust-remote-code", action="store_true", default=False)

    # API-specific
    api_g = p.add_argument_group("API options")
    api_g.add_argument("--base-url", type=str, default=None)
    api_g.add_argument("--api-key", type=str, default=None)

    # Output
    p.add_argument("--output-dir", type=str, required=True)

    # Thinking
    p.add_argument("--enable-thinking", action="store_true", default=False,
                   help="Enable thinking mode (Qwen3 <think> tags)")

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subset", type=int, default=None,
                   help="Evaluate only first N examples per benchmark (for debugging)")
    p.add_argument("--subset-map", type=str, default=None,
                   help="Per-benchmark subset limits, e.g. 'mmlu_pro:2000,bbh:2000'")
    p.add_argument("--avg-at-map", type=str, default=None,
                   help="Per-benchmark average@N sampling, e.g. 'amc:32,aime2024:32,aime2025:32'")

    # WandB
    wandb_g = p.add_argument_group("WandB options")
    wandb_g.add_argument("--wandb-project", type=str, default=None,
                         help="WandB project name (enables WandB logging)")
    wandb_g.add_argument("--wandb-entity", type=str, default=None,
                         help="WandB entity/team name")
    wandb_g.add_argument("--wandb-run-name", type=str, default=None,
                         help="WandB run name (default: model basename)")

    return p.parse_args()


def _get_avg_at_n(args, benchmark_name: str) -> int:
    """Get average@N sample count for a benchmark. Returns 1 if not configured."""
    if not args.avg_at_map:
        return 1
    for entry in args.avg_at_map.split(","):
        k, v = entry.split(":")
        if k.strip() == benchmark_name:
            return int(v.strip())
    return 1


def run_benchmark(benchmark_name, backend, args, output_dir):
    """Run a single benchmark and return summary dict."""
    bench = get_benchmark(benchmark_name)
    avg_n = _get_avg_at_n(args, benchmark_name)
    print(f"\n{'='*60}")
    print(f"  Benchmark: {bench.name}" + (f"  (average@{avg_n})" if avg_n > 1 else ""))
    print(f"{'='*60}")

    # Load (per-benchmark subset overrides global --subset)
    subset_limit = args.subset
    if args.subset_map:
        for entry in args.subset_map.split(","):
            k, v = entry.split(":")
            if k.strip() == benchmark_name:
                subset_limit = int(v.strip())
    print(f"Loading {bench.name}...")
    items = bench.load(subset=subset_limit)
    print(f"  {len(items)} items")

    # Build prompts
    all_messages = [bench.build_prompt(item) for item in items]

    # Run inference
    if avg_n > 1:
        print(f"Running inference (average@{avg_n}, {len(items)} x {avg_n} = {len(items) * avg_n} generations)...")
        t0 = time.time()
        multi_results = backend.generate_chat_n(all_messages, n=avg_n)
        wall_time = time.time() - t0
        print(f"  Done in {wall_time:.1f}s")
        return _score_avg_at_n(bench, items, multi_results, avg_n, wall_time, output_dir)
    else:
        print("Running inference...")
        t0 = time.time()
        results = backend.generate_chat(all_messages)
        wall_time = time.time() - t0
        print(f"  Done in {wall_time:.1f}s")
        return _score_single(bench, items, results, wall_time, output_dir)


def _score_single(bench, items, results, wall_time, output_dir):
    """Score single-sample benchmark run."""
    records = []
    correct_count = 0
    truncated_count = 0
    extraction_failed = 0

    for item, result in tqdm(zip(items, results), total=len(items), desc="Scoring"):
        pred = bench.extract_answer(result.text)
        ok = bench.is_correct(pred, item.gold_answer)

        correct_count += int(ok)
        if result.finish_reason == "length":
            truncated_count += 1
        if pred is None:
            extraction_failed += 1

        # Separate thinking from response for readability
        think_match = re.search(r"<think>(.*?)</think>", result.text, re.DOTALL)
        thinking = think_match.group(1).strip() if think_match else None
        response = strip_thinking(result.text)

        records.append({
            "id": item.id,
            "question": item.question[:500],
            "gold_answer": item.gold_answer,
            "thinking": thinking,
            "response": response,
            "raw_output": result.text,
            "extracted_answer": pred,
            "correct": ok,
            "finish_reason": result.finish_reason,
        })

    # Write outputs
    bench_dir = output_dir / bench.name
    bench_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(records, str(bench_dir / "predictions.jsonl"))

    accuracy = correct_count / len(records) if records else 0.0
    summary = {
        "benchmark": bench.name,
        "accuracy": round(accuracy, 6),
        "correct": correct_count,
        "total": len(records),
        "truncated": truncated_count,
        "extraction_failed": extraction_failed,
        "wall_time_sec": round(wall_time, 2),
    }
    write_json(summary, str(bench_dir / "summary.json"))

    print(f"\n  {bench.name}: {accuracy:.4f} ({correct_count}/{len(records)})")
    print(f"  Truncated: {truncated_count}, Extract failed: {extraction_failed}")

    return summary


def _score_avg_at_n(bench, items, multi_results, avg_n, wall_time, output_dir):
    """Score average@N benchmark run.

    For each item, score all N samples independently.
    Per-item score = (num correct) / N.
    Overall accuracy = mean of per-item scores.
    """
    records = []
    total_correct_sum = 0.0
    truncated_count = 0
    extraction_failed = 0

    for item, sample_results in tqdm(
        zip(items, multi_results), total=len(items), desc=f"Scoring avg@{avg_n}"
    ):
        sample_scores = []
        sample_details = []
        for result in sample_results:
            pred = bench.extract_answer(result.text)
            ok = bench.is_correct(pred, item.gold_answer)
            sample_scores.append(int(ok))
            if result.finish_reason == "length":
                truncated_count += 1
            if pred is None:
                extraction_failed += 1
            sample_details.append({
                "extracted_answer": pred,
                "correct": ok,
                "finish_reason": result.finish_reason,
                "response": strip_thinking(result.text)[:500],
            })

        item_avg = sum(sample_scores) / len(sample_scores)
        total_correct_sum += item_avg

        # Use first sample for the main record display
        first = sample_results[0]
        think_match = re.search(r"<think>(.*?)</think>", first.text, re.DOTALL)
        thinking = think_match.group(1).strip() if think_match else None

        records.append({
            "id": item.id,
            "question": item.question[:500],
            "gold_answer": item.gold_answer,
            "thinking": thinking,
            "avg_score": round(item_avg, 6),
            "num_correct": sum(sample_scores),
            "num_samples": len(sample_scores),
            "samples": sample_details,
        })

    # Write outputs
    bench_dir = output_dir / bench.name
    bench_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(records, str(bench_dir / "predictions.jsonl"))

    accuracy = total_correct_sum / len(items) if items else 0.0
    summary = {
        "benchmark": bench.name,
        "accuracy": round(accuracy, 6),
        "avg_at_n": avg_n,
        "total": len(items),
        "truncated": truncated_count,
        "extraction_failed": extraction_failed,
        "wall_time_sec": round(wall_time, 2),
    }
    write_json(summary, str(bench_dir / "summary.json"))

    print(f"\n  {bench.name} (avg@{avg_n}): {accuracy:.4f} (over {len(items)} items)")
    print(f"  Truncated: {truncated_count}, Extract failed: {extraction_failed}")

    return summary


def _init_wandb(args):
    """Initialize WandB if --wandb-project is set. Returns the run or None."""
    if not args.wandb_project:
        return None
    try:
        import wandb
    except ImportError:
        print("WARNING: wandb not installed, skipping WandB logging")
        return None
    model_name = Path(args.model).name
    run_name = args.wandb_run_name or model_name
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config={
            "model": args.model,
            "model_name": model_name,
            "backend": args.backend,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "benchmarks": args.benchmarks,
            "avg_at_map": args.avg_at_map,
        },
    )
    return run


def _log_wandb(wandb_run, all_summaries, args):
    """Log benchmark results to WandB."""
    if wandb_run is None:
        return
    import wandb

    # Log per-benchmark accuracy as summary metrics
    metrics = {}
    for s in all_summaries:
        bench = s["benchmark"]
        suffix = f"_avg@{s['avg_at_n']}" if "avg_at_n" in s else ""
        metrics[f"eval/{bench}{suffix}"] = s["accuracy"]
        metrics[f"eval/{bench}_total"] = s["total"]
        if "correct" in s:
            metrics[f"eval/{bench}_correct"] = s["correct"]
        metrics[f"eval/{bench}_wall_time"] = s["wall_time_sec"]

    # Compute category averages
    math_benches = ["math500", "gsm8k", "amc", "aime2024", "aime2025", "olympiadbench", "mgsm"]
    code_benches = ["mbpp", "humaneval"]
    reasoning_benches = ["bbh", "gpqa_diamond", "arc_challenge"]

    for cat_name, cat_list in [("math", math_benches), ("code", code_benches), ("reasoning", reasoning_benches)]:
        scores = [s["accuracy"] for s in all_summaries if s["benchmark"] in cat_list]
        if scores:
            metrics[f"eval/{cat_name}_avg"] = sum(scores) / len(scores)

    all_scores = [s["accuracy"] for s in all_summaries]
    if all_scores:
        metrics["eval/overall_avg"] = sum(all_scores) / len(all_scores)

    wandb.log(metrics)

    # Log results table
    columns = ["benchmark", "accuracy", "total", "wall_time_sec"]
    table = wandb.Table(columns=columns)
    for s in all_summaries:
        table.add_data(s["benchmark"], s["accuracy"], s["total"], s["wall_time_sec"])
    wandb.log({"eval/results_table": table})

    wandb_run.finish()
    print(f"  WandB run finished: {wandb_run.url}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Init WandB
    wandb_run = _init_wandb(args)

    # Resolve benchmark list
    if "all" in args.benchmarks:
        bench_names = list(BENCHMARKS.keys())
    else:
        bench_names = args.benchmarks

    # Create backend (shared across benchmarks)
    print(f"Initializing {args.backend} backend with model={args.model}")
    if args.backend == "vllm":
        backend = create_backend(
            "vllm",
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            trust_remote_code=args.trust_remote_code,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            stop=args.stop,
            enable_thinking=args.enable_thinking,
        )
    else:
        backend = create_backend(
            "api",
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            stop=args.stop,
        )

    # Run benchmarks
    all_summaries = []
    for name in bench_names:
        summary = run_benchmark(name, backend, args, output_dir)
        all_summaries.append(summary)

    # Write overall summary
    overall = {
        "model": args.model,
        "backend": args.backend,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": {s["benchmark"]: s["accuracy"] for s in all_summaries},
        "details": all_summaries,
    }
    write_json(overall, str(output_dir / "overall_summary.json"))

    # Print final table
    print(f"\n{'='*60}")
    print(f"  OVERALL RESULTS - {args.model}")
    print(f"{'='*60}")
    print(f"  {'Benchmark':<20} {'Accuracy':>10} {'Correct':>10}")
    print(f"  {'-'*40}")
    for s in all_summaries:
        correct_str = str(s.get('correct', '-'))
        suffix = f"_avg@{s['avg_at_n']}" if 'avg_at_n' in s else ""
        print(f"  {s['benchmark'] + suffix:<20} {s['accuracy']:>10.4f} {correct_str:>5}/{s['total']:<4}")
    print(f"{'='*60}")
    print(f"  Results saved to: {output_dir}")

    # Log to WandB
    _log_wandb(wandb_run, all_summaries, args)


if __name__ == "__main__":
    main()
