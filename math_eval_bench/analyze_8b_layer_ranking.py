"""Rank Qwen3-8B-Base layers by math_avg from local eval result dirs.

Reads $RESULTS_BASE/<exp>_<tok>_<t_tag>/overall_summary.json for every directory
matching the layer-training naming convention, computes math_avg, and prints
shell-source-friendly variable assignments for downstream training scripts:

    TOP1=16
    TOP5="16,14,19,22,17"
    TOP10="16,14,19,22,17,15,12,18,20,10"
    WORST5="2,3,1,30,32"
    WORST10="2,3,1,30,32,29,27,33,4,28"
    BASELINE_FULL=0.6365
    BASELINE_BASE=0.5232

The orchestrator script does:  eval $(python analyze_8b_layer_ranking.py)

Usage:
    python analyze_8b_layer_ranking.py
    python analyze_8b_layer_ranking.py --results-base /abs/path/to/results
    python analyze_8b_layer_ranking.py --tok-tag 8k_t06
    python analyze_8b_layer_ranking.py --csv /tmp/8b_ranking.csv  (also dump human-readable)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

LAYER_RE = re.compile(r"_layer(\d+)_Qwen3-8B-Base_")
FULL_RE = re.compile(r"_full_Qwen3-8B-Base_")
BASE_RE = re.compile(r"^Qwen3-8B-Base_baseline")
MATH_BENCHES_DEFAULT = ("math500", "gsm8k", "olympiadbench", "amc")


def safe_acc(entry: dict) -> float | None:
    a = entry.get("accuracy")
    return float(a) if isinstance(a, (int, float)) else None


def math_avg_from_summary(path: Path, math_benches: tuple[str, ...]) -> float | None:
    try:
        d = json.loads(path.read_text())
    except Exception:
        return None
    rows = d.get("benchmarks", [])
    by_name: dict[str, float] = {}
    for r in rows:
        name = r.get("name") or r.get("benchmark") or ""
        # eval.py writes "amc" with avg_at_n; normalize to bare benchmark name
        for bn in math_benches:
            if name == bn or name.startswith(bn):
                acc = safe_acc(r)
                if acc is not None:
                    by_name[bn] = acc
    if not by_name:
        return None
    return sum(by_name.values()) / len(by_name)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-base",
                   default=str(Path(__file__).parent / "results"))
    p.add_argument("--tok-tag", default="8k_t06",
                   help="suffix that disambiguates which eval to read")
    p.add_argument("--csv", default=None,
                   help="optional human-readable CSV dump (full ranking)")
    p.add_argument("--n-layers", type=int, default=36,
                   help="layer count for the model (Qwen3-8B-Base = 36)")
    args = p.parse_args()

    base = Path(args.results_base)
    if not base.exists():
        print(f"# ERROR: results base not found: {base}", file=sys.stderr)
        return 1

    # Collect (layer_id, math_avg) and the full / baseline references.
    by_layer: dict[int, float] = {}
    full_avg: float | None = None
    baseline_avg: float | None = None

    suffix = f"_{args.tok_tag}"
    for d in sorted(base.iterdir()):
        if not d.is_dir() or not d.name.endswith(suffix):
            continue
        summary = d / "overall_summary.json"
        if not summary.exists():
            continue
        ma = math_avg_from_summary(summary, MATH_BENCHES_DEFAULT)
        if ma is None:
            continue
        m = LAYER_RE.search(d.name)
        if m:
            layer_id = int(m.group(1))
            # Keep the highest math_avg if duplicates (re-evals)
            if layer_id not in by_layer or ma > by_layer[layer_id]:
                by_layer[layer_id] = ma
        elif FULL_RE.search(d.name):
            full_avg = max(full_avg or 0.0, ma)
        elif BASE_RE.match(d.name):
            baseline_avg = max(baseline_avg or 0.0, ma)

    if not by_layer:
        print(f"# ERROR: no Qwen3-8B-Base layer eval results under {base} "
              f"(suffix='{suffix}'). Run eval first.", file=sys.stderr)
        return 1

    # Sort layers by math_avg desc (top) and asc (worst)
    sorted_desc = sorted(by_layer.items(), key=lambda x: x[1], reverse=True)
    sorted_asc = sorted(by_layer.items(), key=lambda x: x[1])

    def to_csv(pairs: list[tuple[int, float]], n: int) -> str:
        return ",".join(str(l) for l, _ in pairs[:n])

    top1 = to_csv(sorted_desc, 1)
    top5 = to_csv(sorted_desc, 5)
    top10 = to_csv(sorted_desc, 10)
    worst1 = to_csv(sorted_asc, 1)
    worst5 = to_csv(sorted_asc, 5)
    worst10 = to_csv(sorted_asc, 10)

    # Stderr: human-readable ranking summary
    print(f"# Qwen3-8B-Base layer ranking (math_avg, {args.tok_tag}, "
          f"{len(by_layer)} layers evaluated)", file=sys.stderr)
    print(f"# baseline (un-trained):  {baseline_avg if baseline_avg else '?'}",
          file=sys.stderr)
    print(f"# full RL:                {full_avg if full_avg else '?'}",
          file=sys.stderr)
    print(f"# top 10:", file=sys.stderr)
    for l, a in sorted_desc[:10]:
        print(f"#   L{l:>2d}  {a:.4f}", file=sys.stderr)
    print(f"# worst 10:", file=sys.stderr)
    for l, a in sorted_asc[:10]:
        print(f"#   L{l:>2d}  {a:.4f}", file=sys.stderr)

    # Stdout: shell-source-friendly assignments
    print(f'TOP1="{top1}"')
    print(f'TOP5="{top5}"')
    print(f'TOP10="{top10}"')
    print(f'WORST1="{worst1}"')
    print(f'WORST5="{worst5}"')
    print(f'WORST10="{worst10}"')
    print(f'N_LAYERS_EVALUATED={len(by_layer)}')
    if full_avg is not None:
        print(f'BASELINE_FULL={full_avg:.6f}')
    if baseline_avg is not None:
        print(f'BASELINE_BASE={baseline_avg:.6f}')

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["layer", "math_avg", "rank_top", "rank_worst"])
            rank_top = {l: i + 1 for i, (l, _) in enumerate(sorted_desc)}
            rank_worst = {l: i + 1 for i, (l, _) in enumerate(sorted_asc)}
            for l, a in sorted(by_layer.items()):
                w.writerow([l, f"{a:.6f}", rank_top[l], rank_worst[l]])
        print(f"# CSV: {out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
