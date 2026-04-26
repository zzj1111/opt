"""Preprocess the agentica-org/DeepCoder-Preview-Dataset (~24K problems) to
parquet format for verl GRPO training.

Each problem is converted to verl format with:
    data_source     = "taco"  (routes through prime_code reward in verl)
    prompt          = chat-format problem statement
    ground_truth    = JSON-encoded {"inputs": [...], "outputs": [...]} test cases
    ability         = "code"

The reward function (verl/utils/reward_score/prime_code) executes the model's
generated solution against the inputs and compares stdout to expected outputs.

Usage:
    # Default — download from HuggingFace
    python3 examples/data_preprocess/deepcoder.py

    # Subsample for quick smoke test
    python3 examples/data_preprocess/deepcoder.py --max_samples 1000

    # Use a local JSON / parquet copy
    python3 examples/data_preprocess/deepcoder.py --local_path /path/to/deepcoder.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import pandas as pd


INSTRUCTION = (
    "Solve the following programming problem. Write your final answer as a single "
    "Python solution inside a ```python``` code block that reads from standard input "
    "and writes the answer to standard output."
)


def _normalize_tests(item: dict[str, Any]) -> dict[str, list[str]] | None:
    """Extract test cases into the {"inputs": [...], "outputs": [...]} schema
    that verl/utils/reward_score/prime_code expects.

    DeepCoder-style records can carry tests under several keys depending on
    upstream source (TACO / APPS / CodeContests / LCB). We try them in order.
    Returns None if no usable tests are found.
    """
    # Direct inputs/outputs at the top level
    if "inputs" in item and "outputs" in item:
        ins, outs = item["inputs"], item["outputs"]
        if isinstance(ins, list) and isinstance(outs, list) and len(ins) == len(outs) > 0:
            return {"inputs": [str(x) for x in ins], "outputs": [str(x) for x in outs]}

    # `tests` key — may be a JSON string or a dict already
    tests = item.get("tests") or item.get("test_cases") or item.get("public_tests")
    if isinstance(tests, str):
        try:
            tests = json.loads(tests)
        except Exception:
            tests = None
    if isinstance(tests, dict):
        ins = tests.get("inputs") or tests.get("input")
        outs = tests.get("outputs") or tests.get("output")
        if isinstance(ins, list) and isinstance(outs, list) and len(ins) == len(outs) > 0:
            return {"inputs": [str(x) for x in ins], "outputs": [str(x) for x in outs]}
    if isinstance(tests, list) and tests:
        # List of {input, output} pairs
        ins, outs = [], []
        for t in tests:
            if isinstance(t, dict) and "input" in t and "output" in t:
                ins.append(str(t["input"]))
                outs.append(str(t["output"]))
        if ins:
            return {"inputs": ins, "outputs": outs}

    return None


def _get_problem_text(item: dict[str, Any]) -> str | None:
    for key in ("problem", "question", "prompt", "description", "statement"):
        v = item.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_repo", default="agentica-org/DeepCoder-Preview-Dataset",
        help="HuggingFace dataset repo (default: %(default)s)",
    )
    parser.add_argument(
        "--hf_configs", default="lcbv5,primeintellect,taco",
        help="Comma-separated config names to merge (default: 3 configs that "
             "have train splits, totalling ~24K problems).",
    )
    parser.add_argument(
        "--hf_split", default="train",
        help="HuggingFace split to load from each config (default: train).",
    )
    parser.add_argument(
        "--local_path", default=None,
        help="If set, load from this local .json or .parquet file instead of HF.",
    )
    parser.add_argument(
        "--local_save_dir", default="data/deepcoder",
        help="Output directory for train/test parquet files.",
    )
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max training samples (0 = use all).")
    parser.add_argument("--train_ratio", type=float, default=0.98,
                        help="Fraction for training (rest = test set).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ---- Load raw data ----
    if args.local_path:
        path = args.local_path
        if path.endswith(".parquet"):
            raw = pd.read_parquet(path).to_dict(orient="records")
        else:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
        print(f"Loaded {len(raw)} problems from {path}")
    else:
        from datasets import load_dataset
        configs = [c.strip() for c in args.hf_configs.split(",") if c.strip()]
        raw = []
        for cfg in configs:
            ds = load_dataset(args.hf_repo, cfg, split=args.hf_split)
            for ex in ds:
                ex["_source_config"] = cfg
                raw.append(ex)
            print(f"  loaded {args.hf_repo}/{cfg} split={args.hf_split}: {len(ds)} examples")
        print(f"Total loaded: {len(raw)} problems across {len(configs)} configs")

    # ---- Convert to verl rows ----
    rows = []
    skipped_no_problem = 0
    skipped_no_tests = 0
    for idx, item in enumerate(raw):
        problem_text = _get_problem_text(item)
        if problem_text is None:
            skipped_no_problem += 1
            continue
        tests = _normalize_tests(item)
        if tests is None:
            skipped_no_tests += 1
            continue

        question = f"{INSTRUCTION}\n\n{problem_text}"
        rows.append({
            # data_source = "taco" routes to verl/utils/reward_score/prime_code
            # which executes the solution against inputs and grades the output.
            "data_source": "taco",
            "prompt": [{"role": "user", "content": question}],
            "ability": "code",
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps(tests),
            },
            "extra_info": {
                "split": "train",
                "index": idx,
                "source": item.get("_source_config",
                                   item.get("source",
                                            item.get("data_source", "deepcoder"))),
                "difficulty": item.get("difficulty"),
            },
        })

    print(f"Kept {len(rows)} / {len(raw)} problems  "
          f"(skipped {skipped_no_problem} without problem text, "
          f"{skipped_no_tests} without parseable tests)")

    df = pd.DataFrame(rows)

    # ---- Optional subsample ----
    if args.max_samples > 0 and len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=args.seed).reset_index(drop=True)
        print(f"Sampled {args.max_samples} examples (seed={args.seed})")

    # ---- Train/test split ----
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    split_idx = max(1, int(len(df) * args.train_ratio))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    # ---- Save ----
    out_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(out_dir, "train.parquet")
    test_path = os.path.join(out_dir, "test.parquet")
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    # Tiny example dump for sanity check
    example = train_df.iloc[0].to_dict()
    example_serialisable = {
        k: (v if isinstance(v, (str, int, float, bool, list, dict)) else str(v))
        for k, v in example.items()
    }
    with open(os.path.join(out_dir, "train_example.json"), "w") as f:
        json.dump(example_serialisable, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(train_df)} train, {len(test_df)} test to {out_dir}")
    print(f"  {train_path}")
    print(f"  {test_path}")


if __name__ == "__main__":
    main()
