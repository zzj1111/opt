#!/usr/bin/env bash
# Batch eval: evaluate multiple checkpoints sequentially, then aggregate + upload.
#
# Usage:
#   bash scripts/batch_eval.sh [options] <label1>:<path1> <label2>:<path2> ...
#
# Examples:
#   # Eval two checkpoints on all benchmarks
#   bash scripts/batch_eval.sh \
#       baseline:Qwen/Qwen3-1.7B \
#       full_rlvr:/home/zha00175/opt_RL_ckpt/actor/huggingface
#
#   # Specific benchmarks + upload
#   bash scripts/batch_eval.sh \
#       --benchmarks math500,gsm8k \
#       --hf-upload Mingyi-Hong/opt_RL_eval \
#       --parallel 2 \
#       baseline:Qwen/Qwen3-1.7B \
#       layer0:/path/to/layer0/ckpt \
#       layer14:/path/to/layer14/ckpt
#
# Options (passed through to run_eval.sh):
#   --benchmarks   Comma-separated benchmark list (default: all 8)
#   --device       cuda | cpu | auto (default: auto)
#   --parallel     Number of GPUs for parallel eval (default: 1)
#   --batch-size   Batch size or 'auto' (default: auto)
#   --max-gen-toks Max generation tokens (default: from config)
#   --hf-upload    HF repo to upload results (e.g. Mingyi-Hong/opt_RL_eval)
#   --force        Re-run even if results exist
#   --lite         Quick test with small sample limits

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_EVAL="$SCRIPT_DIR/run_eval.sh"

# Separate options from label:path specs
OPTS=()
SPECS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --force|--lite|--load-in-4bit|--load-in-8bit)
            OPTS+=("$1"); shift ;;
        --*) OPTS+=("$1" "$2"); shift 2 ;;
        *)   SPECS+=("$1"); shift ;;
    esac
done

if [[ ${#SPECS[@]} -eq 0 ]]; then
    echo "Usage: bash scripts/batch_eval.sh [options] <label1>:<path1> <label2>:<path2> ..."
    echo ""
    echo "Example:"
    echo "  bash scripts/batch_eval.sh \\"
    echo "      --device cuda --parallel 2 --max-gen-toks 3072 \\"
    echo "      --hf-upload Mingyi-Hong/opt_RL_eval \\"
    echo "      baseline:Qwen/Qwen3-1.7B \\"
    echo "      full_rlvr:/home/user/ckpt/actor/huggingface"
    exit 1
fi

TOTAL=${#SPECS[@]}
N=1

for SPEC in "${SPECS[@]}"; do
    # Parse label:path
    LABEL="${SPEC%%:*}"
    CKPT_PATH="${SPEC#*:}"

    if [[ "$LABEL" == "$CKPT_PATH" ]]; then
        echo "[error] Invalid spec '$SPEC'. Use format label:/path/to/model"
        exit 1
    fi

    echo ""
    echo "========================================================"
    echo "  [$N/$TOTAL] Evaluating: $LABEL"
    echo "  Path: $CKPT_PATH"
    echo "========================================================"

    # Strip --hf-upload from per-checkpoint runs (upload once at end)
    RUN_OPTS=()
    SKIP_NEXT=false
    for OPT in "${OPTS[@]}"; do
        if $SKIP_NEXT; then SKIP_NEXT=false; continue; fi
        if [[ "$OPT" == "--hf-upload" ]]; then SKIP_NEXT=true; continue; fi
        RUN_OPTS+=("$OPT")
    done

    bash "$RUN_EVAL" \
        --ckpt-path "$CKPT_PATH" \
        --checkpoints "$LABEL" \
        "${RUN_OPTS[@]}"

    echo "  [$N/$TOTAL] $LABEL done."
    N=$((N + 1))
done

echo ""
echo "========================================================"
echo "  All $TOTAL checkpoints evaluated. Aggregating..."
echo "========================================================"

cd "$EVAL_DIR"
python3 "$SCRIPT_DIR/aggregate.py" && echo "[ok] Aggregation done." || echo "[warn] Aggregation failed."

# Upload if --hf-upload was specified
HF_REPO=""
for i in "${!OPTS[@]}"; do
    if [[ "${OPTS[$i]}" == "--hf-upload" ]]; then
        HF_REPO="${OPTS[$((i+1))]}"
        break
    fi
done

if [[ -n "$HF_REPO" ]]; then
    echo ""
    echo "========================================================"
    echo "  Uploading results to HF: $HF_REPO"
    echo "========================================================"
    python3 "$SCRIPT_DIR/upload_results.py" \
        --repo-id "$HF_REPO" \
        --results-dir "$EVAL_DIR/results" \
    && echo "[ok] Upload complete: https://huggingface.co/datasets/$HF_REPO" \
    || echo "[FAIL] Upload failed."
fi

echo ""
echo "Done. Results in: $EVAL_DIR/results/"
