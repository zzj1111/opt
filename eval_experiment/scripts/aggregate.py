"""Aggregate raw lm-eval results into scores.csv and deltas.csv.

Usage:
    cd eval_experiment/
    python scripts/aggregate.py
    python scripts/aggregate.py --raw-dir results/raw --out-dir results
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import yaml


BENCHMARKS = ["math500", "gsm8k", "mbpp", "ifeval", "mmlu_pro", "bbh", "mgsm", "ceval"]

CHECKPOINT_ORDER = (
    ["baseline", "full_rlvr"] +
    [f"layer_{i}" for i in range(28)]
)

# Primary metric key per benchmark (matches run_single.py METRIC_KEYS)
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

# Human-readable column names
BENCH_DISPLAY = {
    "math500":  "MATH-500",
    "gsm8k":    "GSM8K",
    "mbpp":     "MBPP",
    "ifeval":   "IFEval",
    "mmlu_pro": "MMLU-Pro",
    "bbh":      "BBH",
    "mgsm":     "MGSM",
    "ceval":    "C-Eval",
}


def load_registry(registry_path: str) -> dict[str, str]:
    with open(registry_path) as f:
        return yaml.safe_load(f)


def extract_score(raw_result: dict, benchmark: str) -> float | None:
    """Extract primary metric from lm-eval JSON output."""
    if "error" in raw_result:
        return None

    metric_key = METRIC_KEYS[benchmark]
    results = raw_result.get("results", {})

    scores = []
    for task_results in results.values():
        if not isinstance(task_results, dict):
            continue
        # Try exact key
        if metric_key in task_results:
            val = task_results[metric_key]
            if isinstance(val, (int, float)) and val == val:  # not NaN
                scores.append(float(val))
        else:
            # Fallback: try any acc/exact_match key
            for k, v in task_results.items():
                if metric_key.split(",")[0] in k and isinstance(v, (int, float)):
                    scores.append(float(v))
                    break

    return sum(scores) / len(scores) if scores else None


def load_all_scores(raw_dir: str) -> dict[str, dict[str, float | None]]:
    """Returns {ckpt_label: {benchmark: score}}"""
    raw_dir = Path(raw_dir)
    scores: dict[str, dict[str, float | None]] = {}

    for result_file in sorted(raw_dir.glob("*.json")):
        name = result_file.stem  # e.g. "layer_13__mbpp"
        if "__" not in name:
            continue
        ckpt_label, benchmark = name.split("__", 1)
        if benchmark not in BENCHMARKS:
            continue

        try:
            with open(result_file) as f:
                raw = json.load(f)
        except Exception as e:
            print(f"  [warn] Failed to load {result_file}: {e}")
            continue

        score = extract_score(raw, benchmark)
        scores.setdefault(ckpt_label, {})[benchmark] = score

    return scores


def build_scores_df(scores: dict[str, dict[str, float | None]]) -> pd.DataFrame:
    # Determine row order
    labels_found = set(scores.keys())
    ordered = [l for l in CHECKPOINT_ORDER if l in labels_found]
    extra = sorted(labels_found - set(ordered))
    rows = ordered + extra

    data = []
    for label in rows:
        row = {"checkpoint": label}
        for bench in BENCHMARKS:
            row[BENCH_DISPLAY[bench]] = scores.get(label, {}).get(bench)
        data.append(row)

    return pd.DataFrame(data).set_index("checkpoint")


def build_deltas_df(scores_df: pd.DataFrame) -> pd.DataFrame:
    if "baseline" not in scores_df.index:
        print("[warn] 'baseline' not found in scores, delta table will be empty")
        return pd.DataFrame()

    baseline = scores_df.loc["baseline"]
    return scores_df.subtract(baseline)


def print_summary(scores_df: pd.DataFrame, deltas_df: pd.DataFrame):
    print("\n=== Scores (%) ===")
    pct = (scores_df * 100).round(1)
    print(pct.to_string())

    if not deltas_df.empty:
        print("\n=== Deltas vs baseline (pp) ===")
        pct_d = (deltas_df * 100).round(2)
        # Highlight rows where MATH-500 improved
        print(pct_d.to_string())

        # Summary: Pareto-optimal layers
        math_col = BENCH_DISPLAY["math500"]
        off_target = [BENCH_DISPLAY[b] for b in BENCHMARKS if b != "math500"]
        if math_col in deltas_df.columns:
            layer_rows = deltas_df.loc[
                [l for l in deltas_df.index if l.startswith("layer_")]
            ]
            if not layer_rows.empty:
                layer_rows = layer_rows.copy()
                layer_rows["math_gain"] = layer_rows[math_col]
                layer_rows["off_target_mean"] = layer_rows[off_target].mean(axis=1)
                layer_rows["efficiency"] = (
                    layer_rows["math_gain"] /
                    (-layer_rows["off_target_mean"].clip(upper=0) + 1e-6)
                )
                top = layer_rows.sort_values("efficiency", ascending=False).head(5)
                print("\n=== Top-5 Pareto-efficient layers ===")
                print(top[["math_gain", "off_target_mean", "efficiency"]].round(4).to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="results/raw")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--registry", default="checkpoints/registry.yaml")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading results from {args.raw_dir} ...")
    scores = load_all_scores(args.raw_dir)
    print(f"Found {len(scores)} checkpoints with results")

    scores_df = build_scores_df(scores)
    deltas_df = build_deltas_df(scores_df)

    scores_path = os.path.join(args.out_dir, "scores.csv")
    deltas_path = os.path.join(args.out_dir, "deltas.csv")

    scores_df.to_csv(scores_path, float_format="%.6f")
    print(f"Saved: {scores_path}")

    if not deltas_df.empty:
        deltas_df.to_csv(deltas_path, float_format="%.6f")
        print(f"Saved: {deltas_path}")

    print_summary(scores_df, deltas_df)


if __name__ == "__main__":
    main()
