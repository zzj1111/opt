#!/bin/bash
# Launch dummy GPU hold job in a tmux session.
#
# Usage:
#   bash run_dummy.sh
#   bash run_dummy.sh --gpus 0,1,2,3

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
GPUS="0,1,2,3,4,5,6,7"
NO_TMUX=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)     GPUS="$2"; shift 2 ;;
        --no-tmux)  NO_TMUX=true; shift ;;
        *)          shift ;;
    esac
done

if [[ -z "${TMUX:-}" ]] && [[ "$NO_TMUX" == "false" ]]; then
    TMUX_SESSION="dummy_$(date +%m%d_%H%M)"
    tmux new-session -d -s "$TMUX_SESSION" \
        "cd $(pwd) && bash $SCRIPT_DIR/run_dummy.sh --no-tmux --gpus $GPUS; exec bash"
    echo "Tmux session '$TMUX_SESSION' started."
    echo "  Attach with:  tmux attach -t $TMUX_SESSION"
    exit 0
fi

export CUDA_VISIBLE_DEVICES=$GPUS
DUMMY_RUN_NAME="dummy_$(hostname)_$(date +%m%d_%H%M)" \
    python3 "$SCRIPT_DIR/dummy_gpu_hold.py"
