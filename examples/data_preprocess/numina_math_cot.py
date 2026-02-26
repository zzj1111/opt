"""
Preprocess the AI-MO/NuminaMath-CoT dataset to parquet format for verl GRPO training.

Uses huggingface_hub + pandas (no heavy dependencies like scipy/torch).

Usage:
    python3 examples/data_preprocess/numina_math_cot.py --max_samples 50000
    python3 examples/data_preprocess/numina_math_cot.py --max_samples 0  # use all data
"""

import argparse
import json
import os
from collections import Counter

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files


# ---------------------------------------------------------------------------
# Inline \boxed{} extraction (from verl.utils.reward_score.math_reward)
# ---------------------------------------------------------------------------

def _remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left):]
    left = "\\boxed{"
    assert s[: len(left)] == left
    assert s[-1] == "}"
    return s[len(left): -1]


def _last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    return None if right_brace_idx is None else string[idx: right_brace_idx + 1]


def extract_solution(solution_str):
    """Extract the final answer from a CoT solution string via \\boxed{}."""
    boxed = _last_boxed_only_string(solution_str)
    if boxed is None:
        return None
    try:
        return _remove_boxed(boxed)
    except Exception:
        return None


def download_split(repo_id, split):
    """Download all parquet shards for a split and concatenate into one DataFrame."""
    files = list_repo_files(repo_id, repo_type="dataset")
    parquet_files = sorted(f for f in files if f.startswith(f"data/{split}-") and f.endswith(".parquet"))
    if not parquet_files:
        # Try alternative layout: {split}/0000.parquet
        parquet_files = sorted(f for f in files if f.startswith(f"{split}/") and f.endswith(".parquet"))
    if not parquet_files:
        # Single file layout
        parquet_files = [f for f in files if split in f and f.endswith(".parquet")]

    print(f"  Downloading {len(parquet_files)} parquet file(s) for split '{split}'...")
    dfs = []
    for pf in parquet_files:
        local_path = hf_hub_download(repo_id, pf, repo_type="dataset")
        dfs.append(pd.read_parquet(local_path))
    return pd.concat(dfs, ignore_index=True)


def process_df(df, split, instruction):
    """Convert raw DataFrame to verl format."""
    records = []
    skipped = 0
    for idx, row in df.iterrows():
        source = row.get("source", "unknown")
        problem = row["problem"]
        solution_text = row["solution"]

        ground_truth = extract_solution(solution_text)
        if ground_truth is None:
            skipped += 1
            continue

        question = problem + " " + instruction

        records.append({
            "data_source": f"numina_{source}",
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {"split": split, "index": int(idx), "source": source},
        })
    return pd.DataFrame(records), skipped


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dataset_path", default=None,
                        help="Local path to a directory containing train.parquet/test.parquet.")
    parser.add_argument("--local_save_dir", default="data/numina_math_cot",
                        help="Save directory for processed parquet files.")
    parser.add_argument("--max_samples", type=int, default=50000,
                        help="Max training samples to keep (0 = use all). Sampled uniformly.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling.")

    args = parser.parse_args()

    repo_id = "AI-MO/NuminaMath-CoT"

    # Load data
    if args.local_dataset_path is not None:
        print(f"Loading from local path: {args.local_dataset_path}")
        train_df = pd.read_parquet(os.path.join(args.local_dataset_path, "train.parquet"))
        test_df = pd.read_parquet(os.path.join(args.local_dataset_path, "test.parquet"))
    else:
        print(f"Downloading {repo_id} from HuggingFace...")
        train_df = download_split(repo_id, "train")
        test_df = download_split(repo_id, "test")

    print(f"Raw train: {len(train_df)}, test: {len(test_df)}")

    # Sample training data if requested
    if args.max_samples > 0 and len(train_df) > args.max_samples:
        train_df = train_df.sample(n=args.max_samples, random_state=args.seed).reset_index(drop=True)
        print(f"Sampled {args.max_samples} training examples (seed={args.seed})")

    instruction = "Let's think step by step and output the final answer within \\boxed{}."

    # Process
    train_out, train_skipped = process_df(train_df, "train", instruction)
    print(f"Train: {len(train_df)} -> {len(train_out)} (skipped {train_skipped} without \\boxed{{}})")

    test_out, test_skipped = process_df(test_df, "test", instruction)
    print(f"Test: {len(test_df)} -> {len(test_out)} (skipped {test_skipped} without \\boxed{{}})")

    # Print source distribution
    sources = Counter(train_out["data_source"])
    print("\nTraining data source distribution:")
    for src, cnt in sources.most_common():
        print(f"  {src}: {cnt}")

    # Save
    local_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_dir, exist_ok=True)

    train_out.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_out.to_parquet(os.path.join(local_dir, "test.parquet"))

    # Save one example as JSON for reference
    example = train_out.iloc[0].to_dict()
    with open(os.path.join(local_dir, "train_example.json"), "w") as f:
        json.dump(example, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nSaved to {local_dir}/")
    print(f"  train.parquet: {len(train_out)} examples")
    print(f"  test.parquet:  {len(test_out)} examples")
