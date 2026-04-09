"""
Preprocess the agentica-org/DeepScaleR-Preview-Dataset to parquet format for verl GRPO training.

Usage:
    python3 examples/data_preprocess/deepscaler.py
    python3 examples/data_preprocess/deepscaler.py --max_samples 40000
    python3 examples/data_preprocess/deepscaler.py --local_json /path/to/deepscaler.json
"""

import argparse
import json
import os

import pandas as pd
from huggingface_hub import hf_hub_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_json", default=None,
                        help="Path to local deepscaler.json. If not set, downloads from HuggingFace.")
    parser.add_argument("--local_save_dir", default="data/deepscaler",
                        help="Output directory for train/test parquet files.")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max training samples (0 = use all).")
    parser.add_argument("--train_ratio", type=float, default=0.95,
                        help="Fraction of data for training (rest for test).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # --- Load raw data ---
    if args.local_json:
        json_path = args.local_json
    else:
        print("Downloading deepscaler.json from HuggingFace...")
        json_path = hf_hub_download(
            repo_id="agentica-org/DeepScaleR-Preview-Dataset",
            filename="deepscaler.json",
            repo_type="dataset",
        )

    with open(json_path, encoding="utf-8") as f:
        raw_data = json.load(f)
    print(f"Loaded {len(raw_data)} problems from {json_path}")

    # --- Convert to verl format ---
    instruction = "Let's think step by step and output the final answer within \\boxed{}."
    records = []
    for item in raw_data:
        question = item["problem"] + " " + instruction
        records.append({
            "data_source": "math",
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": item["answer"]},
            "extra_info": {"solution": item.get("solution", "")},
        })

    df = pd.DataFrame(records)

    # --- Sample if requested ---
    if args.max_samples > 0 and len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=args.seed).reset_index(drop=True)
        print(f"Sampled {args.max_samples} examples (seed={args.seed})")

    # --- Split train/test ---
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    split_idx = int(len(df) * args.train_ratio)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    # --- Save ---
    out_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(out_dir, exist_ok=True)

    train_df.to_parquet(os.path.join(out_dir, "train.parquet"), index=False)
    test_df.to_parquet(os.path.join(out_dir, "test.parquet"), index=False)

    example = train_df.iloc[0].to_dict()
    with open(os.path.join(out_dir, "train_example.json"), "w") as f:
        json.dump(example, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nSaved to {out_dir}/")
    print(f"  train.parquet: {len(train_df)} examples")
    print(f"  test.parquet:  {len(test_df)} examples")
    print(f"  data_source:   'math' (uses math_dapo reward)")


if __name__ == "__main__":
    main()
