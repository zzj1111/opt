#!/bin/bash
# ==============================================================================
# 8B math eval — fully sequential, one vLLM at a time, 4 benches as 4 calls.
# ==============================================================================
# For every *Qwen3-8B-Base* ckpt:
#   for bench in math500 gsm8k amc olympiadbench:
#       python eval.py --benchmarks <bench> --output-dir <ckpt>_<bench>_t06/
#       (vLLM loads, evaluates 1 bench, exits)
#
# Total: N_ckpts × 4 vLLM invocations, AT MOST ONE vLLM alive at a time.
# Each (ckpt, bench) pair gets its own wandb run + its own results subdir.
# Per-bench skip: re-running picks up where it left off.
#
# Usage:
#   bash eval_8b_seqbench.sh
#   bash eval_8b_seqbench.sh --tp 8        # one vllm uses all 8 GPUs
#   bash eval_8b_seqbench.sh --tp 1        # one vllm uses 1 GPU (other 7 idle)
#   bash eval_8b_seqbench.sh --pattern '*resume*'
#   bash eval_8b_seqbench.sh --no-tmux

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CKPT_ROOT="${CKPT_ROOT:-/code/hongpaul-sandbox/temp/OPT-RL/opt/checkpoints}"
PATTERN="*Qwen3-8B-Base*"
RESULTS="$SCRIPT_DIR/results"
EVAL_PY="$SCRIPT_DIR/eval.py"

# vllm settings — single instance at a time
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

# benchmarks (sequential within each ckpt)
BENCHES=(math500 gsm8k amc olympiadbench)
declare -A AVG_AT=( [amc]="amc:32" )

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

# ------ tmux wrap ------
if [[ -z "${TMUX:-}" ]] && ! $NO_TMUX; then
    SESSION="eval_8b_seq_$(date +%m%d_%H%M)"
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

# ------ pick GPU slice for the one vllm ------
IFS=',' read -ra GPU_LIST <<< "$GPUS"
NUM_GPUS=${#GPU_LIST[@]}
if (( NUM_GPUS < TP )); then
    echo "ERROR: NUM_GPUS=$NUM_GPUS < TP=$TP"
    exit 1
fi
GPU_SLICE=$(IFS=,; echo "${GPU_LIST[*]:0:$TP}")    # take first TP GPUs

echo "=========================================================="
echo "  python:        $(command -v python3)"
echo "  vllm:          $(python3 -c 'import vllm; print(vllm.__version__)' 2>&1 | head -1)"
echo "  ckpt root:     $CKPT_ROOT"
echo "  pattern:       $PATTERN"
echo "  benches:       ${BENCHES[*]}"
echo "  gpu slice:     $GPU_SLICE  (TP=$TP, other $((NUM_GPUS-TP)) GPUs idle)"
echo "  vllm:          gpu_mem=$GPU_MEM_UTIL  max_model_len=$MAX_MODEL_LEN"
echo "  results:       $RESULTS"
echo "  wandb:         $WANDB_ENTITY / $WANDB_PROJECT"
echo "  policy:        ckpt sequential, bench sequential, ONE vllm at a time"
echo "=========================================================="

# ------ main loop ------
shopt -s nullglob
for d in "$CKPT_ROOT"/$PATTERN; do
    [[ -d "$d" ]] || continue
    ckpt_name=$(basename "$d")

    # find latest global_step_*/actor/huggingface
    model_path=""; best=-1
    for step_dir in "$d"/global_step_*; do
        [[ -d "$step_dir" ]] || continue
        n="${step_dir##*global_step_}"
        [[ "$n" =~ ^[0-9]+$ ]] || continue
        if [[ -f "$step_dir/actor/huggingface/config.json" ]] && (( n > best )); then
            best=$n
            model_path="$step_dir/actor/huggingface"
        fi
    done
    [[ -z "$model_path" ]] && { echo "[skip ckpt] $ckpt_name — no completed ckpt"; continue; }

    echo ""
    echo "=========================================================="
    echo "[ckpt] $ckpt_name"
    echo "  model: $model_path"
    echo "=========================================================="

    for bench in "${BENCHES[@]}"; do
        out="$RESULTS/${ckpt_name}_${bench}_t06"
        if [[ -f "$out/overall_summary.json" ]]; then
            echo "  [skip bench]  $bench  (already done: $out)"
            continue
        fi
        mkdir -p "$out"

        avg_args=()
        [[ -n "${AVG_AT[$bench]:-}" ]] && avg_args=(--avg-at-map "${AVG_AT[$bench]}")

        echo "  [run bench]   $bench  ->  $out"
        if CUDA_VISIBLE_DEVICES="$GPU_SLICE" \
           python3 "$EVAL_PY" \
                --backend vllm \
                --model "$model_path" \
                --benchmarks "$bench" \
                --tensor-parallel-size "$TP" \
                --gpu-memory-utilization "$GPU_MEM_UTIL" \
                --max-model-len "$MAX_MODEL_LEN" \
                --max-tokens "$MAX_TOKENS" \
                --temperature "$TEMPERATURE" \
                --top-p 0.95 \
                --top-k 20 \
                --seed 42 \
                "${avg_args[@]}" \
                --wandb-project "$WANDB_PROJECT" \
                --wandb-entity "$WANDB_ENTITY" \
                --wandb-run-name "${ckpt_name}_${bench}_t06" \
                --output-dir "$out" \
        ; then
            echo "  [done bench]  $bench"
        else
            echo "  [FAIL bench]  $bench (continuing to next bench)"
        fi
    done
done

echo ""
echo "=========================================================="
echo "  All ckpts processed."
echo "  Results: $RESULTS"
echo "  WandB:   https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"
echo "=========================================================="
