"""Preprocess the MBPP dataset to parquet format for verl training."""

import argparse
import json
import os

import datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir", default="data/mbpp")
    parser.add_argument("--local_dataset_path", default=None)
    args = parser.parse_args()

    data_source = "mbpp"

    if args.local_dataset_path:
        dataset = datasets.load_dataset(args.local_dataset_path)
    else:
        dataset = datasets.load_dataset("google-research-datasets/mbpp", "sanitized")

    # sanitized split: train/test/validation/prompt
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction = (
        "Write a Python function to solve the following problem.\n"
        "Put your solution inside a ```python``` code block.\n"
        "Only provide the function implementation, no test code."
    )

    def make_map_fn(split):
        def process_fn(example, idx):
            prompt_text = example["prompt"]
            test_list = example["test_list"]  # list of assert strings

            question = f"{instruction}\n\n{prompt_text}"

            # ground_truth stores the test cases as JSON string for the reward function
            ground_truth = json.dumps(test_list)

            return {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "code",
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "task_id": example.get("task_id", idx),
                },
            }
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    os.makedirs(args.local_save_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(args.local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(args.local_save_dir, "test.parquet"))
    print(f"Saved {len(train_dataset)} train, {len(test_dataset)} test to {args.local_save_dir}")
