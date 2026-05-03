#!/bin/bash
# ==============================================================================
# Minimal 8B math eval — sequential, no parallelism, no worker pool.
# ==============================================================================
# For every directory under CKPT_ROOT matching *Qwen3-8B-Base*, runs the 4 math
# benchmarks (math500 / gsm8k / olympiadbench / amc avg@32) via a single
# `python eval.py` call. Skips ckpts that already have an overall_summary.json.
#
# Usage:
#   bash eval_8b_simple.sh              # auto-launches tmux session
#   bash eval_8b_simple.sh --no-tmux    # run inline (no tmux wrap)
#   bash eval_8b_simple.sh --tp 2       # tensor-parallel size (default 2)
#   bash eval_8b_simple.sh --pattern "*foo*"   # narrower glob

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CKPT_ROOT="${CKPT_ROOT:-/code/hongpaul-sandbox/temp/OPT-RL/opt/checkpoints}"
PATTERN="*Qwen3-8B-Base*"
RESULTS="$SCRIPT_DIR/results"
EVAL_PY="$SCRIPT_DIR/eval.py"

# vllm settings
TP=2
GPU_MEM_UTIL=0.85
MAX_MODEL_LEN=16384
MAX_TOKENS=8192
TEMPERATURE=0.6

# wandb
WANDB_API_KEY="${WANDB_API_KEY:-b8f38344ec7231ee89baa74ef7209dd5a43df6b2}"
WANDB_ENTITY="mhong-university-of-minnesota"
WANDB_PROJECT="opt_rl_eval_8b_math"

# conda
CONDA_INIT="/code/hongpaul-sandbox/cuda/miniconda3/bin/activate"
CONDA_ENV_PATH="/code/hongpaul-sandbox/cuda/miniconda3/envs/cuda"

NO_TMUX=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-tmux) NO_TMUX=true; shift ;;
        --tp) TP="$2"; shift 2 ;;
        --pattern) PATTERN="$2"; shift 2 ;;
        --ckpt-root) CKPT_ROOT="$2"; shift 2 ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

# ------ tmux wrap (so user can detach, conda gets activated) ------
if [[ -z "${TMUX:-}" ]] && ! $NO_TMUX; then
    SESSION="eval_8b_$(date +%m%d_%H%M)"
    tmux new-session -d -s "$SESSION" \
        "source $CONDA_INIT && \
         conda activate $CONDA_ENV_PATH && \
         export WANDB_API_KEY=$WANDB_API_KEY && \
         bash $0 --no-tmux --tp $TP --pattern '$PATTERN' --ckpt-root '$CKPT_ROOT'; \
         exec bash"
    echo "tmux '$SESSION' started.  Attach: tmux attach -t $SESSION"
    exit 0
fi

# ------ defensive: activate conda if vllm not importable ------
if ! python -c "import vllm" >/dev/null 2>&1; then
    [[ -f "$CONDA_INIT" ]] && source "$CONDA_INIT" && conda activate "$CONDA_ENV_PATH"
fi
export WANDB_API_KEY
export WANDB_ENTITY
export VLLM_USE_FLASHINFER_SAMPLER=0

echo "=========================================================="
echo "  python:      $(command -v python3)"
echo "  vllm:        $(python3 -c 'import vllm; print(vllm.__version__)' 2>&1 | head -1)"
echo "  ckpt root:   $CKPT_ROOT"
echo "  pattern:     $PATTERN"
echo "  results:     $RESULTS"
echo "  TP:          $TP   gpu_mem=$GPU_MEM_UTIL   max_model_len=$MAX_MODEL_LEN"
echo "  wandb:       $WANDB_ENTITY / $WANDB_PROJECT"
echo "=========================================================="

# ------ main loop: one ckpt at a time, one python per ckpt, all 4 benches ------
shopt -s nullglob
for d in "$CKPT_ROOT"/$PATTERN; do
    [[ -d "$d" ]] || continue
    name=$(basename "$d")

    # find latest global_step_*/actor/huggingface
    model_path=""
    best=-1
    for step_dir in "$d"/global_step_*; do
        [[ -d "$step_dir" ]] || continue
        n="${step_dir##*global_step_}"
        [[ "$n" =~ ^[0-9]+$ ]] || continue
        if [[ -f "$step_dir/actor/huggingface/config.json" ]] && (( n > best )); then
            best=$n
            model_path="$step_dir/actor/huggingface"
        fi
    done
    if [[ -z "$model_path" ]]; then
        echo "[skip] $name — no completed global_step_*/actor/huggingface"
        continue
    fi

    out="$RESULTS/${name}_8k_t06"
    if [[ -f "$out/overall_summary.json" ]]; then
        echo "[skip] $name — already evaluated ($out)"
        continue
    fi
    mkdir -p "$out"

    echo ""
    echo "=========================================================="
    echo "[run]  $name"
    echo "  model: $model_path"
    echo "  out:   $out"
    echo "=========================================================="
    python3 "$EVAL_PY" \
        --backend vllm \
        --model "$model_path" \
        --benchmarks math500 gsm8k amc olympiadbench \
        --tensor-parallel-size "$TP" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-tokens "$MAX_TOKENS" \
        --temperature "$TEMPERATURE" \
        --top-p 0.95 \
        --top-k 20 \
        --seed 42 \
        --avg-at-map "amc:32" \
        --wandb-project "$WANDB_PROJECT" \
        --wandb-entity "$WANDB_ENTITY" \
        --wandb-run-name "${name}_8k_t06" \
        --output-dir "$out" \
    && echo "[done] $name" \
    || echo "[FAIL] $name (continuing to next)"
done

echo ""
echo "=========================================================="
echo "  All ckpts processed."
echo "  Results: $RESULTS"
echo "  WandB:   https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"
echo "=========================================================="
