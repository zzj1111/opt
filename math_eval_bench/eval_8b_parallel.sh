#!/bin/bash
# ==============================================================================
# 8B math eval with N-way parallelism — stride pattern, no flock, no queue.
# ==============================================================================
# - Builds a task list of all *Qwen3-8B-Base* ckpts (one task per ckpt).
# - Spawns N workers, each owning a fixed slice of GPUs (TP=K each).
# - Worker w handles task indices w, w+N, w+2N, ... (stride pattern).
# - Each task = one `python eval.py` call with all 4 benchmarks inside.
# - Skips ckpts whose overall_summary.json already exists.
#
# Default layout on 8 GPUs:  N=4 workers × TP=2  (4 ckpts evaluated at once)
#
# Usage:
#   bash eval_8b_parallel.sh                 # 4 workers × TP=2
#   bash eval_8b_parallel.sh --tp 4          # 2 workers × TP=4
#   bash eval_8b_parallel.sh --tp 1          # 8 workers × TP=1
#   bash eval_8b_parallel.sh --pattern '*resume*'
#   bash eval_8b_parallel.sh --no-tmux

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CKPT_ROOT="${CKPT_ROOT:-/code/hongpaul-sandbox/temp/OPT-RL/opt/checkpoints}"
PATTERN="*Qwen3-8B-Base*"
RESULTS="$SCRIPT_DIR/results"
EVAL_PY="$SCRIPT_DIR/eval.py"
LOG_DIR="$SCRIPT_DIR/logs/eval_8b_parallel"

# vllm settings
TP=2
GPU_MEM_UTIL=0.85
MAX_MODEL_LEN=16384
MAX_TOKENS=8192
TEMPERATURE=0.6
GPUS="0,1,2,3,4,5,6,7"

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
        --no-tmux)   NO_TMUX=true; shift ;;
        --tp)        TP="$2"; shift 2 ;;
        --pattern)   PATTERN="$2"; shift 2 ;;
        --ckpt-root) CKPT_ROOT="$2"; shift 2 ;;
        --gpus)      GPUS="$2"; shift 2 ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"

# ------ tmux wrap ------
if [[ -z "${TMUX:-}" ]] && ! $NO_TMUX; then
    SESSION="eval_8b_par_$(date +%m%d_%H%M)"
    tmux new-session -d -s "$SESSION" \
        "source $CONDA_INIT && \
         conda activate $CONDA_ENV_PATH && \
         export WANDB_API_KEY=$WANDB_API_KEY && \
         bash $0 --no-tmux --tp $TP --pattern '$PATTERN' --ckpt-root '$CKPT_ROOT' --gpus '$GPUS'; \
         exec bash"
    echo "tmux '$SESSION' started.  Attach: tmux attach -t $SESSION"
    exit 0
fi

# ------ defensive conda activate ------
if ! python -c "import vllm" >/dev/null 2>&1; then
    [[ -f "$CONDA_INIT" ]] && source "$CONDA_INIT" && conda activate "$CONDA_ENV_PATH"
fi
export WANDB_API_KEY
export WANDB_ENTITY
export VLLM_USE_FLASHINFER_SAMPLER=0

# ------ derive parallelism ------
IFS=',' read -ra GPU_LIST <<< "$GPUS"
NUM_GPUS=${#GPU_LIST[@]}
if (( NUM_GPUS % TP != 0 )); then
    echo "ERROR: NUM_GPUS=$NUM_GPUS is not divisible by TP=$TP"
    exit 1
fi
N_WORKERS=$((NUM_GPUS / TP))

echo "=========================================================="
echo "  python:        $(command -v python3)"
echo "  vllm:          $(python3 -c 'import vllm; print(vllm.__version__)' 2>&1 | head -1)"
echo "  ckpt root:     $CKPT_ROOT"
echo "  pattern:       $PATTERN"
echo "  results:       $RESULTS"
echo "  GPUs:          $GPUS  ($NUM_GPUS total)"
echo "  layout:        $N_WORKERS workers × TP=$TP"
echo "  vllm:          gpu_mem=$GPU_MEM_UTIL  max_model_len=$MAX_MODEL_LEN"
echo "  wandb:         $WANDB_ENTITY / $WANDB_PROJECT"
echo "=========================================================="

# ------ build task list ------
TASKS=()
shopt -s nullglob
for d in "$CKPT_ROOT"/$PATTERN; do
    [[ -d "$d" ]] || continue
    name=$(basename "$d")

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
    [[ -z "$model_path" ]] && { echo "[skip] $name — no completed ckpt"; continue; }

    out="$RESULTS/${name}_8k_t06"
    if [[ -f "$out/overall_summary.json" ]]; then
        echo "[skip] $name — already evaluated"
        continue
    fi

    TASKS+=("$model_path|$name|$out")
done

N_TASKS=${#TASKS[@]}
echo ""
echo "  $N_TASKS tasks queued, $N_WORKERS workers will share them (stride pattern)"
echo "=========================================================="

if (( N_TASKS == 0 )); then
    echo "Nothing to do."
    exit 0
fi

# ------ worker function: handle task indices w, w+N, w+2N, ... ------
worker() {
    local w="$1"           # worker index 0..N_WORKERS-1
    local gpu_slice="$2"   # e.g. "0,1"
    local idx=$w
    local done_count=0
    while (( idx < N_TASKS )); do
        IFS='|' read -r model_path name out <<< "${TASKS[$idx]}"
        local log_file="$LOG_DIR/${name}_8k_t06.log"
        mkdir -p "$out"
        echo "[W$w gpus=$gpu_slice] START [$idx/$N_TASKS] $name"
        if CUDA_VISIBLE_DEVICES="$gpu_slice" \
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
                > "$log_file" 2>&1
        then
            done_count=$((done_count + 1))
            echo "[W$w gpus=$gpu_slice] DONE  [$idx/$N_TASKS] $name (worker total: $done_count)"
        else
            echo "[W$w gpus=$gpu_slice] FAIL  [$idx/$N_TASKS] $name — see $log_file"
        fi
        idx=$((idx + N_WORKERS))
    done
    echo "[W$w gpus=$gpu_slice] worker finished ($done_count completed)"
}

# ------ launch workers ------
PIDS=()
for ((w=0; w<N_WORKERS; w++)); do
    slice=""
    for ((j=0; j<TP; j++)); do
        idx=$((w * TP + j))
        slice+="${GPU_LIST[$idx]}"
        (( j < TP - 1 )) && slice+=","
    done
    worker "$w" "$slice" &
    PIDS+=($!)
    echo "launched W$w on GPUs $slice (PID $!)"
done

for pid in "${PIDS[@]}"; do wait "$pid" || true; done

echo ""
echo "=========================================================="
echo "  All workers finished."
echo "  Results: $RESULTS"
echo "  WandB:   https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"
echo "=========================================================="
