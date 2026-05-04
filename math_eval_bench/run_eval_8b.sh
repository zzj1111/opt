#!/bin/bash
# Thin tmux wrapper for eval_8b.py — only handles conda activation + tmux detach.
# Everything else (ckpt scan, vllm, scoring, wandb) is in eval_8b.py.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONDA_INIT="/code/hongpaul-sandbox/cuda/miniconda3/bin/activate"
CONDA_ENV_PATH="/code/hongpaul-sandbox/cuda/miniconda3/envs/cuda"
WANDB_API_KEY="${WANDB_API_KEY:-b8f38344ec7231ee89baa74ef7209dd5a43df6b2}"

NO_TMUX=false
PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-tmux) NO_TMUX=true; shift ;;
        *) PASSTHROUGH+=("$1"); shift ;;
    esac
done

if [[ -z "${TMUX:-}" ]] && ! $NO_TMUX; then
    SESSION="eval_8b_py_$(date +%m%d_%H%M)"
    ARGS=$(printf '%q ' "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}")
    tmux new-session -d -s "$SESSION" \
        "source $CONDA_INIT && \
         conda activate $CONDA_ENV_PATH && \
         export WANDB_API_KEY=$WANDB_API_KEY && \
         export VLLM_USE_FLASHINFER_SAMPLER=0 && \
         python3 $SCRIPT_DIR/eval_8b.py $ARGS; exec bash"
    echo "tmux '$SESSION' started.  Attach: tmux attach -t $SESSION"
    exit 0
fi

# Inside-tmux / --no-tmux path: defensive conda + run python.
if ! python -c "import vllm" >/dev/null 2>&1; then
    [[ -f "$CONDA_INIT" ]] && source "$CONDA_INIT" && conda activate "$CONDA_ENV_PATH"
fi
export WANDB_API_KEY
export VLLM_USE_FLASHINFER_SAMPLER=0

exec python3 "$SCRIPT_DIR/eval_8b.py" "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}"
