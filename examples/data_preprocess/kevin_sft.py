"""Convert the Kevin-32B SFT JSONL dataset into verl SFT parquet format.

Schema of input JSONL (one record per line):
  - module_id    : str   - identifier (e.g. "fa2/fwd")
  - family       : str   - kernel family
  - split        : str   - "train" or "val"
  - description  : str   - markdown description of the kernel
  - primary_files: dict  - {filename: file_content} reference files
  - completion   : str   - target output (the kernel implementation)

Output parquet has two columns:
  - prompt   : str  - description + reference files framed as a prompt
  - response : str  - the completion (target)

Run:
    python kevin_sft.py \
        --input /home/zha00175/CudaForge_plus/verl/sft_train.jsonl \
        --output_dir /home/zha00175/CudaForge_plus/verl/data/kevin_sft
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def build_prompt(description: str, primary_files: Dict[str, str]) -> str:
    """Combine description + reference files into a single prompt string."""
    parts: List[str] = [description.rstrip(), ""]
    if primary_files:
        parts.append("## Reference Files")
        for filename, content in primary_files.items():
            parts.append("")
            parts.append(f"### {filename}")
            parts.append("```")
            parts.append(content.rstrip())
            parts.append("```")
    parts.append("")
    parts.append("## Implementation")
    parts.append("")
    return "\n".join(parts)


def convert(input_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    by_split: Dict[str, List[dict]] = {}
    with input_path.open() as f:
        for line in f:
            row = json.loads(line)
            split = row.get("split", "train")
            entry = {
                "module_id": row.get("module_id", ""),
                "family": row.get("family", ""),
                "prompt": build_prompt(row["description"], row.get("primary_files", {})),
                "response": row["completion"],
            }
            by_split.setdefault(split, []).append(entry)

    # Always emit a train.parquet; emit val.parquet if present, otherwise a tiny
    # holdout from train (verl expects val_files even if we don't really care).
    if "train" not in by_split:
        raise ValueError(f"No 'train' split rows found in {input_path}")

    train_rows = by_split["train"]
    val_rows = by_split.get("val") or by_split.get("test") or train_rows[:1]

    pd.DataFrame(train_rows).to_parquet(output_dir / "train.parquet", index=False)
    pd.DataFrame(val_rows).to_parquet(output_dir / "val.parquet", index=False)

    print(f"Wrote {len(train_rows)} train rows -> {output_dir / 'train.parquet'}")
    print(f"Wrote {len(val_rows)} val rows   -> {output_dir / 'val.parquet'}")
    # Quick stats
    train_p = sum(len(r["prompt"]) for r in train_rows) / len(train_rows)
    train_r = sum(len(r["response"]) for r in train_rows) / len(train_rows)
    print(f"avg prompt chars   : {train_p:.0f}")
    print(f"avg response chars : {train_r:.0f}")
    max_total = max(len(r["prompt"]) + len(r["response"]) for r in train_rows)
    print(f"max prompt+response chars : {max_total} (~{max_total // 4} tokens at 4 chars/tok)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="/home/zha00175/CudaForge_plus/verl/sft_train.jsonl",
        type=Path,
    )
    ap.add_argument(
        "--output_dir",
        default="/home/zha00175/CudaForge_plus/verl/data/kevin_sft",
        type=Path,
    )
    args = ap.parse_args()
    convert(args.input, args.output_dir)


if __name__ == "__main__":
    main()
