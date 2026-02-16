# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""
Preprocess KernelBench (CudaForge) dataset to verl parquet format.

Input (raw) supports:
1) A directory containing train.parquet and test.parquet (recommended), OR
2) A HuggingFace save_to_disk DatasetDict directory (contains train/ test/)

Raw example must contain:
- prompt: str
- reference: str

Output:
- train.parquet
- test.parquet
with verl-style schema:
{
  data_source: str,
  prompt: [{"role":"user","content":...}],
  ability: "CUDA",
  reward_model: {"style":"rule","ground_truth":0},
  extra_info: {...}
}
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import datasets

from verl.utils.hdfs_io import copy, makedirs


def load_raw_splits(local_dataset_path: str) -> Tuple[datasets.Dataset, datasets.Dataset]:
    """
    Load raw train/test splits from either:
    - Parquet directory: contains train.parquet + test.parquet
    - HF dataset dir: load_from_disk -> DatasetDict with keys train/test
    """
    p = Path(local_dataset_path)
    if not p.exists():
        raise FileNotFoundError(f"--local_dataset_path not found: {local_dataset_path}")

    train_parquet = p / "train.parquet"
    test_parquet = p / "test.parquet"

    # Preferred: read parquet directly (works for your "two parquet" raw format)
    if train_parquet.exists() and test_parquet.exists():
        train_ds = datasets.Dataset.from_parquet(str(train_parquet))
        test_ds = datasets.Dataset.from_parquet(str(test_parquet))
        return train_ds, test_ds

    # Fallback: load HF save_to_disk directory (DatasetDict)
    ds = datasets.load_from_disk(str(p))
    if not isinstance(ds, datasets.DatasetDict):
        raise ValueError(
            f"load_from_disk returned {type(ds)} but DatasetDict was expected. "
            f"Also train.parquet/test.parquet not found under: {local_dataset_path}"
        )
    if "train" not in ds or "test" not in ds:
        raise ValueError(f"DatasetDict missing train/test keys. Keys={list(ds.keys())}")

    return ds["train"], ds["test"]


def make_map_fn(split: str, data_source: str):
    """
    Map raw {prompt:str, reference:str} into verl expected schema.
    """
    def process_fn(example, idx):
        # raw
        question_raw = example.pop("prompt")
        answer_raw = example.pop("reference")

        # Build verl sample
        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": question_raw,
                }
            ],
            "ability": "CUDA",
            "reward_model": {"style": "rule", "ground_truth": 0},
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": answer_raw,
                "question": question_raw,
            },
        }

        # Optional: if raw example contains any extra columns, keep them for traceability
        # (You can delete this block if you want strict minimal extra_info.)
        if example:
            data["extra_info"]["raw_fields"] = example

        return data

    return process_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="DEPRECATED. Use --local_save_dir instead.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--local_dataset_path",
        default="/home/zha00175/CudaForge/KernelBench/hf_parquet_level1",
        help="Path to raw dataset. Either a dir containing train.parquet/test.parquet, or a HF save_to_disk dir.",
    )
    parser.add_argument(
        "--local_save_dir",
        default="~/verl/dataset/CudaForge/new/l1",
        help="Save directory for the preprocessed dataset (verl format parquet).",
    )
    parser.add_argument("--data_source", default="CudaForge")
    args = parser.parse_args()

    local_dataset_path = args.local_dataset_path
    data_source = args.data_source

    # Load raw train/test
    train_dataset, test_dataset = load_raw_splits(local_dataset_path)

    # Basic validation: ensure required columns exist
    for split_name, ds in [("train", train_dataset), ("test", test_dataset)]:
        cols = set(ds.column_names)
        if "prompt" not in cols or "reference" not in cols:
            raise ValueError(
                f"Raw {split_name} dataset must contain columns prompt/reference. "
                f"Got columns={ds.column_names}"
            )

    # Convert to verl schema
    train_dataset = train_dataset.map(function=make_map_fn("train", data_source), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test", data_source), with_indices=True)

    # Handle save dir args (保持与你给的脚本一致)
    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    local_save_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    # Save to parquet
    train_path = os.path.join(local_save_dir, "train.parquet")
    test_path = os.path.join(local_save_dir, "test.parquet")
    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)

    print(f"[OK] Wrote: {train_path}")
    print(f"[OK] Wrote: {test_path}")

    # Optional: copy to HDFS
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)
        print(f"[OK] Copied to HDFS: {hdfs_dir}")


if __name__ == "__main__":
    main()
