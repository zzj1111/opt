"""
Prepare TACO dataset filtered by difficulty for verl GRPO training.

Usage:
    python prepare_taco.py
    python prepare_taco.py --difficulties EASY MEDIUM --max-train 6000
"""

import argparse
import json
import re
import pandas as pd
from datasets import load_dataset

SYSTEM_PROMPT = "You are an expert Python programmer. Solve the following competitive programming problem."

DIFFICULTIES = ["EASY", "MEDIUM", "MEDIUM_HARD", "HARD", "VERY_HARD"]


def build_prompt(problem: dict) -> list[dict]:
    """Build chat-format prompt from a TACO problem."""
    question = problem.get("question", "").strip()

    # Include starter code if present
    starter = problem.get("starter_code", "").strip()
    if starter:
        user_content = (
            f"{question}\n\n"
            f"Use the following starter code:\n```python\n{starter}\n```\n\n"
            "Write a complete Python solution. Output ONLY the code."
        )
    else:
        user_content = (
            f"{question}\n\n"
            "Write a complete Python solution. Output ONLY the code."
        )

    return [{"role": "user", "content": user_content}]


def build_ground_truth(problem: dict) -> str:
    """Build ground_truth JSON for the TACO reward function."""
    # TACO stores test cases as a JSON string in "input_output"
    raw = problem.get("input_output", "")
    try:
        io = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        io = {}

    inputs = io.get("inputs", [])
    outputs = io.get("outputs", [])

    # fn_name is set when it's a function-call style problem
    fn_name = io.get("fn_name", None)

    return json.dumps({
        "inputs": inputs,
        "outputs": outputs,
        "fn_name": fn_name,
        "starter_code": problem.get("starter_code", ""),
    })


def process_split(dataset, difficulties: set[str], max_samples: int | None) -> pd.DataFrame:
    rows = []
    for item in dataset:
        diff = (item.get("difficulty") or "").upper()
        if difficulties and diff not in difficulties:
            continue

        prompt = build_prompt(item)
        ground_truth = build_ground_truth(item)

        # Skip problems with no test cases
        try:
            gt = json.loads(ground_truth)
            if not gt.get("inputs") and not gt.get("outputs"):
                continue
        except Exception:
            continue

        rows.append({
            "data_source": "taco",
            "prompt": prompt,
            "ability": "code",
            "reward_model": {
                "ground_truth": ground_truth,
                "style": "rule",
            },
            "extra_info": {
                "difficulty": item.get("difficulty", ""),
                "tags": item.get("tags", []),
            },
        })

        if max_samples and len(rows) >= max_samples:
            break

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulties", nargs="+", default=["EASY", "MEDIUM"],
                        choices=DIFFICULTIES,
                        help="Difficulty levels to include")
    parser.add_argument("--max-train", type=int, default=6000,
                        help="Max training samples")
    parser.add_argument("--max-test", type=int, default=500,
                        help="Max test samples")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Output directory for parquet files")
    args = parser.parse_args()

    difficulties = set(d.upper() for d in args.difficulties)
    print(f"Loading TACO dataset (difficulties: {difficulties})...")

    ds = load_dataset("BAAI/TACO", trust_remote_code=True)

    print("Processing train split...")
    train_df = process_split(ds["train"], difficulties, args.max_train)
    print(f"Train samples: {len(train_df)}")

    print("Processing test split...")
    test_df = process_split(ds["test"], difficulties, args.max_test)
    print(f"Test samples: {len(test_df)}")

    # Difficulty breakdown
    print("\nDifficulty breakdown (train):")
    for diff, count in train_df["extra_info"].apply(lambda x: x["difficulty"]).value_counts().items():
        print(f"  {diff}: {count}")

    import os
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.parquet")
    test_path = os.path.join(args.output_dir, "test.parquet")

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    print(f"\nSaved: {train_path} ({len(train_df)} rows)")
    print(f"Saved: {test_path} ({len(test_df)} rows)")


if __name__ == "__main__":
    main()
