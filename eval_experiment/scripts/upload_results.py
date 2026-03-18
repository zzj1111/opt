#!/usr/bin/env python3
"""Upload eval results to Hugging Face Hub.

Usage:
    python scripts/upload_results.py --repo-id Mingyi-Hong/opt_RL_eval --results-dir results/
    python scripts/upload_results.py --repo-id Mingyi-Hong/opt_RL_eval --results-dir results/ --token hf_xxx
"""

import argparse
import os
from pathlib import Path


def upload(repo_id: str, results_dir: str, token: str | None = None, private: bool = False):
    from huggingface_hub import HfApi

    api = HfApi(token=token)

    # Create repo if not exists
    try:
        api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    except Exception as e:
        print(f"[warn] create_repo: {e}")

    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"[error] Results dir not found: {results_dir}")
        return

    # Collect all files to upload
    files = []
    for f in sorted(results_path.rglob("*")):
        if f.is_file():
            rel = f.relative_to(results_path)
            files.append((str(f), f"results/{rel}"))

    if not files:
        print("[warn] No result files found, nothing to upload.")
        return

    print(f"Uploading {len(files)} files to {repo_id}...")
    for local_path, hf_path in files:
        print(f"  {hf_path}")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=hf_path,
            repo_id=repo_id,
            repo_type="dataset",
        )

    print(f"Done. View at: https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload eval results to HF Hub")
    parser.add_argument("--repo-id", required=True, help="HF repo, e.g. Mingyi-Hong/opt_RL_eval")
    parser.add_argument("--results-dir", default="results", help="Local results directory")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF token (or set HF_TOKEN env)")
    parser.add_argument("--private", action="store_true", help="Create private dataset repo")
    args = parser.parse_args()

    if not args.token:
        print("[error] No HF token. Pass --token or set HF_TOKEN env var.")
        return

    upload(args.repo_id, args.results_dir, args.token, args.private)


if __name__ == "__main__":
    main()
