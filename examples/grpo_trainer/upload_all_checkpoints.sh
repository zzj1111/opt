#!/bin/bash
# ==============================================================================
# Upload all checkpoints to a SINGLE HuggingFace repo
# ==============================================================================
#
# Each experiment's last checkpoint is uploaded as a subdirectory within
# one shared HF repo. E.g.:
#   repo/0409_exp1_full_xxx/model.safetensors
#   repo/0409_exp2_layer0_xxx/model.safetensors
#
# Usage:
#   bash upload_all_checkpoints.sh
#   bash upload_all_checkpoints.sh --repo mhong-university-of-minnesota/opt-checkpoints
#   bash upload_all_checkpoints.sh --prefix "0409_" --dry-run

CKPT_ROOT="/code/hongpaul-sandbox/temp/OPT-RL/opt/checkpoints"
HF_TOKEN="${HF_TOKEN:-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}"
HF_REPO="mhong-university-of-minnesota/opt-checkpoints"
PREFIX="04"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt-root) CKPT_ROOT="$2"; shift 2 ;;
        --hf-token)  HF_TOKEN="$2"; shift 2 ;;
        --repo)      HF_REPO="$2"; shift 2 ;;
        --prefix)    PREFIX="$2"; shift 2 ;;
        --dry-run)   DRY_RUN=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ ! -d "$CKPT_ROOT" ]]; then
    echo "ERROR: Checkpoint directory not found: $CKPT_ROOT"
    exit 1
fi

if [[ -z "$HF_TOKEN" ]]; then
    echo "ERROR: HF_TOKEN not set. Pass --hf-token or export HF_TOKEN."
    exit 1
fi

echo "============================================================"
echo "  Upload All Checkpoints to Single HF Repo"
echo "  Root:    $CKPT_ROOT"
echo "  Repo:    $HF_REPO"
echo "  Prefix:  ${PREFIX:-<all>}"
echo "  DryRun:  $DRY_RUN"
echo "============================================================"
echo ""

python3 - "$CKPT_ROOT" "$HF_TOKEN" "$HF_REPO" "$PREFIX" "$DRY_RUN" << 'PYEOF'
import os, re, sys
from pathlib import Path

ckpt_root = Path(sys.argv[1])
hf_token = sys.argv[2]
hf_repo = sys.argv[3]
prefix = sys.argv[4]
dry_run = sys.argv[5] == "true"

def find_last_hf_ckpt(exp_dir):
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

# Find all experiment dirs
exp_dirs = sorted([
    d for d in ckpt_root.iterdir()
    if d.is_dir() and (not prefix or d.name.startswith(prefix))
])

if not exp_dirs:
    print(f"No experiment directories found in {ckpt_root}")
    sys.exit(0)

print(f"Found {len(exp_dirs)} experiment(s)\n")

from huggingface_hub import HfApi
api = HfApi(token=hf_token)

# Create repo once
if not dry_run:
    try:
        api.create_repo(hf_repo, exist_ok=True, private=False, repo_type="model")
        print(f"Repo: https://huggingface.co/{hf_repo}\n")
    except Exception as e:
        print(f"Warning creating repo: {e}")

uploaded = []
failed = []

for exp_dir in exp_dirs:
    hf_path = find_last_hf_ckpt(exp_dir)
    if hf_path is None:
        print(f"  [{exp_dir.name}] No HF checkpoint found, skipping.")
        failed.append(exp_dir.name)
        continue

    step_name = hf_path.parent.parent.name
    # Upload into subdirectory named after the experiment
    path_in_repo = exp_dir.name
    print(f"  [{exp_dir.name}] {step_name} -> {hf_repo}/{path_in_repo}/")

    if dry_run:
        uploaded.append(exp_dir.name)
        continue

    try:
        api.upload_folder(
            folder_path=str(hf_path),
            repo_id=hf_repo,
            path_in_repo=path_in_repo,
            commit_message=f"Upload {exp_dir.name} ({step_name})",
        )
        uploaded.append(exp_dir.name)
        print(f"    OK")
    except Exception as e:
        print(f"    FAILED: {e}")
        failed.append(exp_dir.name)

print(f"\n{'='*50}")
print(f"  Uploaded: {len(uploaded)}/{len(exp_dirs)}")
if failed:
    print(f"  Failed: {', '.join(failed)}")
print(f"  Repo: https://huggingface.co/{hf_repo}")
print(f"{'='*50}")
PYEOF
