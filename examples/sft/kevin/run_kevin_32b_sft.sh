#!/bin/bash
# ==============================================================================
# Kevin-32B SFT on the kernel-generation JSONL (with tmux)
# ==============================================================================
#
# Inputs : /home/zha00175/CudaForge_plus/verl/sft_train.jsonl
#          (preprocessed into data/kevin_sft/{train,val}.parquet by
#           examples/data_preprocess/kevin_sft.py)
#
# Defaults: full fine-tune on 8 GPUs, ulysses sequence-parallel=4,
#           max_length=65536, micro_batch_size=1.  LoRA available by passing
#           MODE=lora LORA_RANK=64 if you want a lighter run.
#
# Override knobs (env vars):
#   MODEL_PATH    Path or HF id of the model      (default: cognition-ai/Kevin-32B)
#   INPUT_JSONL   Source JSONL                    (default: $PROJ_DIR/sft_train.jsonl)
#   DATA_DIR      Where the parquet lives         (default: $PROJ_DIR/data/kevin_sft)
#   CKPT_ROOT     Where to save                   (default: $PROJ_DIR/checkpoints)
#   GPUS          CUDA_VISIBLE_DEVICES list       (default: 0,1,2,3,4,5,6,7)
#   EPOCHS        Training epochs                 (default: 5)
#   MAX_LENGTH    Sequence length cap             (default: 65536)
#   MODE          'full' (default) or 'lora'
#   LORA_RANK     LoRA rank (only when MODE=lora) (default: 64)
#   LR            Optim LR                        (default: 1e-4 for lora, 1e-5 for full)
#   ULYSSES_SP    Sequence-parallel degree        (default: 4)
#   PROJECT       wandb project                   (default: kevin_sft)
#   EXP_NAME      wandb run name                  (default: kevin32b_<MODE>_<DATE>)
#   SKIP_PREP     Set to 1 to skip data prep      (default: unset = run prep)
#   NO_TMUX       Set to 1 to skip tmux           (default: unset = wrap in tmux)

set -uo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJ_DIR=$(cd "$SCRIPT_DIR/../../.." && pwd)

MODEL_PATH="${MODEL_PATH:-cognition-ai/Kevin-32B}"
INPUT_JSONL="${INPUT_JSONL:-$PROJ_DIR/sft_train.jsonl}"
DATA_DIR="${DATA_DIR:-$PROJ_DIR/data/kevin_sft}"
CKPT_ROOT="${CKPT_ROOT:-$PROJ_DIR/checkpoints}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
EPOCHS="${EPOCHS:-5}"
MAX_LENGTH="${MAX_LENGTH:-65536}"
MODE="${MODE:-full}"          # 'full' or 'lora'
LORA_RANK="${LORA_RANK:-64}"
ULYSSES_SP="${ULYSSES_SP:-4}"
PROJECT="${PROJECT:-kevin_sft}"
DATE=$(date +%m%d_%H%M)
EXP_NAME="${EXP_NAME:-kevin32b_${MODE}_${DATE}}"
SKIP_PREP="${SKIP_PREP:-}"
NO_TMUX="${NO_TMUX:-}"

CONDA_INIT="${CONDA_INIT:-/code/hongpaul-sandbox/cuda/miniconda3/bin/activate}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/code/hongpaul-sandbox/cuda/miniconda3/envs/cuda}"

# ---------- Default LR per mode ----------
if [[ -z "${LR:-}" ]]; then
    if [[ "$MODE" == "lora" ]]; then LR="1e-4"; else LR="1e-5"; fi
fi

# ---------- Tmux auto-launch ----------
if [[ -z "${TMUX:-}" ]] && [[ -z "$NO_TMUX" ]]; then
    TMUX_SESSION="kevin_sft_${DATE}"
    tmux new-session -d -s "$TMUX_SESSION" \
        "source $CONDA_INIT && \
         conda activate $CONDA_ENV_PATH && \
         export WANDB_API_KEY='${WANDB_API_KEY:-b8f38344ec7231ee89baa74ef7209dd5a43df6b2}' && \
         export WANDB_ENTITY='${WANDB_ENTITY:-mhong-university-of-minnesota}' && \
         cd $PROJ_DIR && \
         NO_TMUX=1 \
         MODEL_PATH=$(printf '%q' "$MODEL_PATH") \
         INPUT_JSONL=$(printf '%q' "$INPUT_JSONL") \
         DATA_DIR=$(printf '%q' "$DATA_DIR") \
         CKPT_ROOT=$(printf '%q' "$CKPT_ROOT") \
         GPUS=$(printf '%q' "$GPUS") \
         EPOCHS=$(printf '%q' "$EPOCHS") \
         MAX_LENGTH=$(printf '%q' "$MAX_LENGTH") \
         MODE=$(printf '%q' "$MODE") \
         LORA_RANK=$(printf '%q' "$LORA_RANK") \
         ULYSSES_SP=$(printf '%q' "$ULYSSES_SP") \
         PROJECT=$(printf '%q' "$PROJECT") \
         EXP_NAME=$(printf '%q' "$EXP_NAME") \
         LR=$(printf '%q' "$LR") \
         SKIP_PREP=$(printf '%q' "$SKIP_PREP") \
         bash $SCRIPT_DIR/run_kevin_32b_sft.sh; \
         exec bash"
    echo "Tmux session '$TMUX_SESSION' started."
    echo "  Attach with:  tmux attach -t $TMUX_SESSION"
    exit 0
fi

# ---------- Data prep ----------
if [[ -z "$SKIP_PREP" ]]; then
    if [[ ! -f "$INPUT_JSONL" ]]; then
        echo "ERROR: source JSONL not found: $INPUT_JSONL"
        echo "       set INPUT_JSONL=/path/to/sft_train.jsonl or SKIP_PREP=1"
        exit 1
    fi
    echo "[prep] Building $DATA_DIR/{train,val}.parquet from $INPUT_JSONL ..."
    python3 "$PROJ_DIR/examples/data_preprocess/kevin_sft.py" \
        --input "$INPUT_JSONL" \
        --output_dir "$DATA_DIR"
fi

if [[ ! -f "$DATA_DIR/train.parquet" ]]; then
    echo "ERROR: $DATA_DIR/train.parquet not found. Run data prep or set DATA_DIR."
    exit 1
fi

NGPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
SAVE_DIR="$CKPT_ROOT/$EXP_NAME"
mkdir -p "$SAVE_DIR"
LOG_FILE="$SAVE_DIR/train.log"

# ---------- LoRA / full overrides ----------
LORA_ARGS=""
if [[ "$MODE" == "lora" ]]; then
    LORA_ARGS="model.lora_rank=$LORA_RANK model.lora_alpha=$((LORA_RANK*2)) model.target_modules=all-linear"
fi

# ---------- Memory / offload defaults ----------
# 32B + 65K seq is heavy.  Always offload params; under full fine-tune also
# offload optimizer.
OFFLOAD_PARAMS=True
OFFLOAD_OPT=False
if [[ "$MODE" == "full" ]]; then OFFLOAD_OPT=True; fi

echo "============================================================"
echo "  Kevin-32B SFT"
echo "  Mode:        $MODE     ${LORA_ARGS:+(lora_rank=$LORA_RANK)}"
echo "  Model:       $MODEL_PATH"
echo "  Data:        $DATA_DIR"
echo "  Save:        $SAVE_DIR"
echo "  GPUs:        $GPUS  ($NGPUS)   ulysses_sp=$ULYSSES_SP"
echo "  Epochs:      $EPOCHS    LR: $LR"
echo "  MaxLength:   $MAX_LENGTH    OffloadParams=$OFFLOAD_PARAMS  OffloadOpt=$OFFLOAD_OPT"
echo "  Project:     $PROJECT  /  $EXP_NAME"
echo "  Log:         $LOG_FILE"
echo "============================================================"

export CUDA_VISIBLE_DEVICES="$GPUS"

torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/val.parquet" \
    data.prompt_key=prompt \
    data.response_key=response \
    data.train_batch_size=8 \
    data.micro_batch_size_per_gpu=1 \
    data.max_length=$MAX_LENGTH \
    data.truncation=right \
    optim.lr=$LR \
    model.partial_pretrain="$MODEL_PATH" \
    model.enable_gradient_checkpointing=True \
    model.trust_remote_code=True \
    model.fsdp_config.offload_params=$OFFLOAD_PARAMS \
    model.fsdp_config.cpu_offload=$OFFLOAD_OPT \
    $LORA_ARGS \
    use_remove_padding=True \
    ulysses_sequence_parallel_size=$ULYSSES_SP \
    trainer.project_name=$PROJECT \
    trainer.experiment_name="$EXP_NAME" \
    trainer.default_local_dir="$SAVE_DIR" \
    trainer.total_epochs=$EPOCHS \
    trainer.save_freq=-1 \
    trainer.logger='["console","wandb"]' \
    trainer.n_gpus_per_node=$NGPUS \
    trainer.nnodes=1 \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================================"
echo "  Kevin-32B SFT done."
echo "  Checkpoint:  $SAVE_DIR"
echo "============================================================"
