#!/bin/bash
# ==============================================================================
# Upload all checkpoints under a directory to HuggingFace Hub
# ==============================================================================
#
# Each subdirectory with global_step_*/actor/huggingface is uploaded as a
# separate HuggingFace model repo.
#
# Usage:
#   bash upload_all_checkpoints.sh
#   bash upload_all_checkpoints.sh --ckpt-root /path/to/checkpoints
#   bash upload_all_checkpoints.sh --hf-org my-org --dry-run
#   bash upload_all_checkpoints.sh --prefix "0409_"   # only matching dirs

CKPT_ROOT="/code/hongpaul-sandbox/temp/OPT-RL/opt/checkpoints"
HF_TOKEN="${HF_TOKEN:-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}"
HF_ORG=""
PREFIX=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt-root) CKPT_ROOT="$2"; shift 2 ;;
        --hf-token)  HF_TOKEN="$2"; shift 2 ;;
        --hf-org)    HF_ORG="$2"; shift 2 ;;
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

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

echo "============================================================"
echo "  Upload Checkpoints to HuggingFace"
echo "  Root:   $CKPT_ROOT"
echo "  Org:    ${HF_ORG:-<user default>}"
echo "  Prefix: ${PREFIX:-<all>}"
echo "  DryRun: $DRY_RUN"
echo "============================================================"
echo ""

python3 "$SCRIPT_DIR/upload_checkpoints.py" \
    --ckpt-root "$CKPT_ROOT" \
    --token "$HF_TOKEN" \
    ${PREFIX:+--prefix "$PREFIX"} \
    ${HF_ORG:+--repo-prefix "$HF_ORG"} \
    $(if $DRY_RUN; then echo "--dry-run"; fi)
