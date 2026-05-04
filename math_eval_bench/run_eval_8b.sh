#!/bin/bash
# Thin tmux wrapper for eval_8b.py — only handles conda activation + tmux detach.
# Everything else (ckpt scan, vllm, scoring, wandb) is in eval_8b.py.
#
# After eval exits, by default hands off to ../examples/grpo_trainer/run_dummy.sh
# to hold the GPUs (prevents the cluster scheduler from reclaiming the node as
# idle). Pass --no-dummy to skip.
#
#   bash run_eval_8b.sh                # eval, then hand off to run_dummy.sh
#   bash run_eval_8b.sh --no-dummy     # eval, then exit
#   bash run_eval_8b.sh --no-tmux      # run inline (still applies dummy)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DUMMY_SCRIPT="$SCRIPT_DIR/../examples/grpo_trainer/run_dummy.sh"
CONDA_INIT="/code/hongpaul-sandbox/cuda/miniconda3/bin/activate"
CONDA_ENV_PATH="/code/hongpaul-sandbox/cuda/miniconda3/envs/cuda"
WANDB_API_KEY="${WANDB_API_KEY:-b8f38344ec7231ee89baa74ef7209dd5a43df6b2}"

NO_TMUX=false
NO_DUMMY=false
PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-tmux)  NO_TMUX=true; shift ;;
        --no-dummy) NO_DUMMY=true; shift ;;
        *) PASSTHROUGH+=("$1"); shift ;;
    esac
done

if [[ -z "${TMUX:-}" ]] && ! $NO_TMUX; then
    SESSION="eval_8b_py_$(date +%m%d_%H%M)"
    if [[ ${#PASSTHROUGH[@]} -gt 0 ]]; then
        ARGS=$(printf '%q ' "${PASSTHROUGH[@]}")
    else
        ARGS=""
    fi
    DUMMY_FLAG=""
    $NO_DUMMY && DUMMY_FLAG="--no-dummy"
    tmux new-session -d -s "$SESSION" \
        "source $CONDA_INIT && \
         conda activate $CONDA_ENV_PATH && \
         export WANDB_API_KEY=$WANDB_API_KEY && \
         export VLLM_USE_FLASHINFER_SAMPLER=0 && \
         bash $SCRIPT_DIR/$(basename "$0") --no-tmux $DUMMY_FLAG $ARGS; exec bash"
    echo "tmux '$SESSION' started.  Attach: tmux attach -t $SESSION"
    $NO_DUMMY || echo "  (run_dummy.sh will hold GPUs after eval — kill the tmux session to release)"
    exit 0
fi

# Inside-tmux / --no-tmux path: defensive conda + run python.
if ! python -c "import vllm" >/dev/null 2>&1; then
    [[ -f "$CONDA_INIT" ]] && source "$CONDA_INIT" && conda activate "$CONDA_ENV_PATH"
fi
export WANDB_API_KEY
export VLLM_USE_FLASHINFER_SAMPLER=0

if [[ ${#PASSTHROUGH[@]} -gt 0 ]]; then
    python3 "$SCRIPT_DIR/eval_8b.py" "${PASSTHROUGH[@]}"
else
    python3 "$SCRIPT_DIR/eval_8b.py"
fi
EVAL_RC=$?
echo ""
echo "=========================================================="
echo "[run_eval_8b.sh] eval_8b.py exited with rc=$EVAL_RC"
echo "=========================================================="

if $NO_DUMMY; then
    echo "[dummy] skipped (--no-dummy)"
    exit $EVAL_RC
fi

if [[ ! -f "$DUMMY_SCRIPT" ]]; then
    echo "[dummy] WARNING: $DUMMY_SCRIPT not found — exiting without hold"
    exit $EVAL_RC
fi

echo "[dummy] handing off to $DUMMY_SCRIPT (--no-tmux, in this same tmux)"
exec bash "$DUMMY_SCRIPT" --no-tmux
