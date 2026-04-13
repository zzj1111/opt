#!/usr/bin/env python3
"""Compare evaluation results across multiple models.

Usage:
    python compare_results.py results/
    python compare_results.py results/model_a results/model_b results/model_c
    python compare_results.py results/ --format markdown
    python compare_results.py results/ --format csv --output comparison.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def load_summary(result_dir: Path) -> dict:
    """Load overall_summary.json from a result directory."""
    summary_path = result_dir / "overall_summary.json"
    if not summary_path.exists():
        return {}
    with open(summary_path, "r") as f:
        return json.load(f)


def find_result_dirs(base_path: Path) -> list[Path]:
    """Find all result directories containing overall_summary.json."""
    dirs = []
    if (base_path / "overall_summary.json").exists():
        dirs.append(base_path)
    else:
        for d in sorted(base_path.iterdir()):
            if d.is_dir() and (d / "overall_summary.json").exists():
                dirs.append(d)
    return dirs


def build_comparison(result_dirs: list[Path]) -> tuple[list[str], list[str], dict]:
    """Build comparison data: (model_names, benchmark_names, {model: {bench: accuracy}})."""
    all_benchmarks = set()
    data = {}

    for d in result_dirs:
        summary = load_summary(d)
        if not summary:
            continue
        model_label = d.name
        results = summary.get("results", {})
        data[model_label] = {
            "results": results,
            "meta": {
                "model": summary.get("model", ""),
                "temperature": summary.get("temperature"),
                "top_p": summary.get("top_p"),
                "max_tokens": summary.get("max_tokens"),
                "timestamp": summary.get("timestamp", ""),
            },
            "details": {
                det["benchmark"]: det
                for det in summary.get("details", [])
            },
        }
        all_benchmarks.update(results.keys())

    # Sort benchmarks in a logical order
    bench_order = [
        "math500", "gsm8k", "aime2024", "aime2025", "amc",
        "olympiadbench", "gpqa_diamond", "mbpp", "bbh",
        "mmlu_pro", "mgsm", "ceval", "ifeval",
    ]
    benchmarks = [b for b in bench_order if b in all_benchmarks]
    benchmarks += sorted(all_benchmarks - set(benchmarks))

    models = list(data.keys())
    return models, benchmarks, data


def print_table(models, benchmarks, data):
    """Print a formatted comparison table to stdout."""
    # Header
    bench_width = max(16, max(len(b) for b in benchmarks) + 2)
    model_width = max(20, max(len(m) for m in models) + 2)

    header = f"{'Benchmark':<{bench_width}}" + "".join(
        f"{m:>{model_width}}" for m in models
    )
    print(header)
    print("-" * len(header))

    for bench in benchmarks:
        row = f"{bench:<{bench_width}}"
        best_acc = -1
        for m in models:
            acc = data[m]["results"].get(bench)
            if acc is not None and acc > best_acc:
                best_acc = acc

        for m in models:
            acc = data[m]["results"].get(bench)
            if acc is None:
                row += f"{'--':>{model_width}}"
            else:
                pct = f"{acc*100:.1f}%"
                if acc == best_acc and len(models) > 1:
                    pct = f"*{pct}"
                row += f"{pct:>{model_width}}"
        print(row)

    # Average
    print("-" * len(header))
    row = f"{'Average':<{bench_width}}"
    for m in models:
        accs = [v for v in data[m]["results"].values() if v is not None]
        avg = sum(accs) / len(accs) if accs else 0
        row += f"{avg*100:.1f}%".rjust(model_width)
    print(row)

    # Detail rows
    print()
    detail_keys = ["truncated", "extraction_failed"]
    for key in detail_keys:
        row = f"{key:<{bench_width}}"
        for m in models:
            total = sum(
                data[m]["details"].get(b, {}).get(key, 0)
                for b in benchmarks
            )
            row += f"{total:>{model_width}}"
        print(row)


def print_markdown(models, benchmarks, data):
    """Print a Markdown table."""
    header = "| Benchmark | " + " | ".join(models) + " |"
    sep = "|---" * (len(models) + 1) + "|"
    print(header)
    print(sep)

    for bench in benchmarks:
        row = f"| {bench} |"
        for m in models:
            acc = data[m]["results"].get(bench)
            if acc is None:
                row += " -- |"
            else:
                row += f" {acc*100:.1f}% |"
        print(row)

    # Average
    row = "| **Average** |"
    for m in models:
        accs = [v for v in data[m]["results"].values() if v is not None]
        avg = sum(accs) / len(accs) if accs else 0
        row += f" **{avg*100:.1f}%** |"
    print(row)


def write_csv(models, benchmarks, data, output_path: str):
    """Write comparison to CSV."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["benchmark"] + models)
        for bench in benchmarks:
            row = [bench]
            for m in models:
                acc = data[m]["results"].get(bench)
                row.append(f"{acc:.6f}" if acc is not None else "")
            writer.writerow(row)
    print(f"CSV written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare model evaluation results.")
    parser.add_argument(
        "paths", nargs="+",
        help="Result directories or parent directory containing them",
    )
    parser.add_argument(
        "--format", choices=["table", "markdown", "csv"], default="table",
        help="Output format (default: table)",
    )
    parser.add_argument("--output", "-o", type=str, default=None, help="Output file (for csv)")
    args = parser.parse_args()

    # Collect result directories
    result_dirs = []
    for p in args.paths:
        path = Path(p)
        if not path.exists():
            print(f"Warning: {p} does not exist, skipping", file=sys.stderr)
            continue
        result_dirs.extend(find_result_dirs(path))

    if not result_dirs:
        print("No result directories found.", file=sys.stderr)
        sys.exit(1)

    models, benchmarks, data = build_comparison(result_dirs)

    if not models:
        print("No valid results found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(models)} models, {len(benchmarks)} benchmarks\n")

    if args.format == "table":
        print_table(models, benchmarks, data)
    elif args.format == "markdown":
        print_markdown(models, benchmarks, data)
    elif args.format == "csv":
        output = args.output or "comparison.csv"
        write_csv(models, benchmarks, data, output)


if __name__ == "__main__":
    main()
