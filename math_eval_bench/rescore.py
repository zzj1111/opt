#!/usr/bin/env python3
"""Re-score existing predictions with updated scoring logic.

Use this after fixing scoring bugs to re-evaluate without re-running inference.

Usage:
    python rescore.py results/baseline_full_eval
    python rescore.py results/baseline_full_eval --benchmarks bbh mgsm
    python rescore.py results/  # re-score all result dirs
"""

import argparse
import json
import sys
from pathlib import Path

from math_eval.benchmarks import BENCHMARKS, get_benchmark
from math_eval.io import write_json, write_jsonl


def rescore_benchmark(bench_dir: Path, benchmark_name: str) -> dict:
    """Re-score a single benchmark's predictions."""
    pred_path = bench_dir / "predictions.jsonl"
    if not pred_path.exists():
        return {}

    bench = get_benchmark(benchmark_name)
    records = []
    with open(pred_path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    old_correct = sum(1 for r in records if r.get("correct"))
    new_correct = 0
    changed = 0

    for rec in records:
        raw = rec.get("raw_output", rec.get("response", ""))
        new_pred = bench.extract_answer(raw)
        new_ok = bench.is_correct(new_pred, rec["gold_answer"])

        if rec.get("extracted_answer") != new_pred or rec.get("correct") != new_ok:
            changed += 1

        rec["extracted_answer"] = new_pred
        rec["correct"] = new_ok
        new_correct += int(new_ok)

    total = len(records)
    accuracy = new_correct / total if total else 0.0

    # Read old summary for wall_time
    old_summary_path = bench_dir / "summary.json"
    wall_time = 0.0
    if old_summary_path.exists():
        with open(old_summary_path, "r") as f:
            old_summary = json.load(f)
            wall_time = old_summary.get("wall_time_sec", 0.0)

    summary = {
        "benchmark": benchmark_name,
        "accuracy": round(accuracy, 6),
        "correct": new_correct,
        "total": total,
        "truncated": sum(1 for r in records if r.get("finish_reason") == "length"),
        "extraction_failed": sum(1 for r in records if r.get("extracted_answer") is None),
        "wall_time_sec": wall_time,
    }

    # Write updated files
    write_jsonl(records, str(pred_path))
    write_json(summary, str(old_summary_path))

    print(f"  {benchmark_name}: {old_correct}/{total} -> {new_correct}/{total} "
          f"(accuracy: {accuracy:.4f}, {changed} records changed)")

    return summary


def rescore_result_dir(result_dir: Path, benchmark_filter: list[str] | None = None):
    """Re-score all benchmarks in a result directory."""
    overall_path = result_dir / "overall_summary.json"
    if not overall_path.exists():
        return

    with open(overall_path, "r") as f:
        overall = json.load(f)

    print(f"\nRe-scoring: {result_dir.name}")

    new_summaries = []
    for bench_dir in sorted(result_dir.iterdir()):
        if not bench_dir.is_dir():
            continue
        bench_name = bench_dir.name
        if bench_name not in BENCHMARKS:
            continue
        if benchmark_filter and bench_name not in benchmark_filter:
            new_summaries.append(
                next((d for d in overall.get("details", []) if d["benchmark"] == bench_name), {})
            )
            continue

        summary = rescore_benchmark(bench_dir, bench_name)
        if summary:
            new_summaries.append(summary)

    # Update overall summary
    overall["results"] = {s["benchmark"]: s["accuracy"] for s in new_summaries if s}
    overall["details"] = [s for s in new_summaries if s]
    write_json(overall, str(overall_path))


def main():
    parser = argparse.ArgumentParser(description="Re-score predictions with updated scoring logic.")
    parser.add_argument("paths", nargs="+", help="Result directory or parent directory")
    parser.add_argument("--benchmarks", nargs="*", default=None,
                        help="Only re-score these benchmarks (default: all)")
    args = parser.parse_args()

    for p in args.paths:
        path = Path(p)
        if (path / "overall_summary.json").exists():
            rescore_result_dir(path, args.benchmarks)
        else:
            for d in sorted(path.iterdir()):
                if d.is_dir() and (d / "overall_summary.json").exists():
                    rescore_result_dir(d, args.benchmarks)


if __name__ == "__main__":
    main()
