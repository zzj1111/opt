#!/usr/bin/env python3
"""Upload training checkpoints to HuggingFace Hub.

Finds all experiment directories matching a prefix, locates their
last global_step_*/actor/huggingface checkpoint, and uploads each
as a separate HF model repo.

Usage:
    python upload_checkpoints.py --ckpt-root checkpoints --token hf_xxx --prefix 0319_exp
    python upload_checkpoints.py --ckpt-root checkpoints --token hf_xxx --prefix 0319_exp --repo-prefix my-org
"""

import argparse
import os
import re
from pathlib import Path


def find_last_hf_ckpt(exp_dir: Path) -> Path | None:
    """Find the last global_step checkpoint with HF model."""
    candidates = []
    for d in exp_dir.iterdir():
        m = re.match(r"global_step_(\d+)", d.name)
        if m and d.is_dir():
            hf_path = d / "actor" / "huggingface"
            if hf_path.exists() and any(hf_path.iterdir()):
                candidates.append((int(m.group(1)), hf_path))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def upload_to_hub(local_path: Path, repo_id: str, token: str):
    """Upload a HuggingFace model directory to the Hub."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)

    # Create repo if not exists
    try:
        api.create_repo(repo_id, exist_ok=True, private=False)
    except Exception as e:
        print(f"  Warning creating repo {repo_id}: {e}")

    # Upload all files
    api.upload_folder(
        folder_path=str(local_path),
        repo_id=repo_id,
        commit_message=f"Upload checkpoint from {local_path.parent.parent.parent.name}",
    )
    print(f"  Uploaded: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload checkpoints to HuggingFace Hub")
    parser.add_argument("--ckpt-root", required=True, help="Root checkpoint directory")
    parser.add_argument("--token", required=True, help="HuggingFace API token")
    parser.add_argument("--prefix", default="", help="Only process experiment dirs matching this prefix")
    parser.add_argument("--repo-prefix", default="", help="HF org/user prefix for repo names (e.g. 'my-org')")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be uploaded without doing it")
    args = parser.parse_args()

    ckpt_root = Path(args.ckpt_root)
    if not ckpt_root.exists():
        print(f"Checkpoint root not found: {ckpt_root}")
        return

    # Find experiment directories
    exp_dirs = sorted([
        d for d in ckpt_root.iterdir()
        if d.is_dir() and (not args.prefix or d.name.startswith(args.prefix))
    ])

    if not exp_dirs:
        print(f"No experiment directories found in {ckpt_root} with prefix '{args.prefix}'")
        return

    print(f"Found {len(exp_dirs)} experiment(s) to upload:\n")

    uploaded = []
    failed = []

    for exp_dir in exp_dirs:
        hf_path = find_last_hf_ckpt(exp_dir)
        if hf_path is None:
            print(f"  [{exp_dir.name}] No HF checkpoint found, skipping.")
            failed.append(exp_dir.name)
            continue

        # Build repo name
        repo_name = exp_dir.name
        if args.repo_prefix:
            repo_id = f"{args.repo_prefix}/{repo_name}"
        else:
            repo_id = repo_name

        step_dir = hf_path.parent.parent.name  # global_step_XXX
        print(f"  [{exp_dir.name}] {step_dir} -> {repo_id}")

        if args.dry_run:
            print(f"    (dry run, skipping upload)")
            uploaded.append(repo_id)
            continue

        try:
            upload_to_hub(hf_path, repo_id, args.token)
            uploaded.append(repo_id)
        except Exception as e:
            print(f"    FAILED: {e}")
            failed.append(exp_dir.name)

    # Summary
    print(f"\n{'='*50}")
    print(f"  Upload complete: {len(uploaded)}/{len(exp_dirs)} succeeded")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
