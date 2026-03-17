"""Generate fixed subsample indices for MMLU-Pro and BBH.

Run once before any evaluation to ensure all 30 checkpoints use identical subsets.
Usage:
    python scripts/subsample.py --seed 42
"""

import argparse
import json
import os
import random


def save_indices(indices: list[int], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(sorted(indices), f, indent=2)
    print(f"Saved {len(indices)} indices → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mmlu-pro-total", type=int, default=12032,
                        help="Total MMLU-Pro validation samples")
    parser.add_argument("--mmlu-pro-n", type=int, default=2000)
    parser.add_argument("--bbh-total", type=int, default=6511,
                        help="Total BBH samples across all subtasks")
    parser.add_argument("--bbh-n", type=int, default=2000)
    parser.add_argument("--out-dir", type=str, default="results/indices")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # MMLU-Pro: stratified by subject (27 subjects, sample proportionally)
    # We use simple random sampling here; the --fewshot_as_multiturn flag in
    # lm-eval will handle the actual subsetting via --limit with fixed seed.
    mmlu_indices = sorted(rng.sample(range(args.mmlu_pro_total), args.mmlu_pro_n))
    save_indices(mmlu_indices, os.path.join(args.out_dir, "mmlu_pro.json"))

    # BBH: 27 subtasks × ~241 samples each
    bbh_indices = sorted(rng.sample(range(args.bbh_total), args.bbh_n))
    save_indices(bbh_indices, os.path.join(args.out_dir, "bbh.json"))

    print("\nSubsample indices generated. All checkpoints must use these same files.")
    print(f"MMLU-Pro: {len(mmlu_indices)} / {args.mmlu_pro_total}")
    print(f"BBH:      {len(bbh_indices)} / {args.bbh_total}")
    print("\nIMPORTANT: Commit these index files to git for reproducibility.")


if __name__ == "__main__":
    main()
