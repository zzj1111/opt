#!/bin/bash
# ==============================================================================
# Best-of-5 MATH evaluation of all *boost* checkpoints (Worker Mode, T=0.6)
# ==============================================================================
#
# For each *boost* checkpoint, run the eval 5 times at T=0.6 with different
# seeds (42, 142, 242, 342, 442). After all seeds finish, pick the seed with
# the highest math_avg and report it as the headline number per checkpoint.
#
# Worker mode: each GPU independently pulls (ckpt, seed) tasks from a shared
# queue. Auto-launches tmux + conda + logs to WandB. Every (ckpt, seed) pair
# is a separate task → 5x parallelism opportunity.
#
# Usage:
#   bash run_eval_boost_math_best5.sh                       # Run all *boost* x 5 seeds
#   bash run_eval_boost_math_best5.sh --dry-run             # Preview what would run
#   bash run_eval_boost_math_best5.sh --gpus 0,1,2,3        # Use specific GPUs
#   bash run_eval_boost_math_best5.sh --ckpt-root /path     # Override checkpoint dir
#   bash run_eval_boost_math_best5.sh --no-tmux             # Skip tmux auto-launch
#   bash run_eval_boost_math_best5.sh --pattern "*boost*"   # Override match glob
#   bash run_eval_boost_math_best5.sh --force               # Re-run even if outputs exist
#   bash run_eval_boost_math_best5.sh --report-only         # Skip eval, just aggregate
#
# Output naming:
#   results/${exp_name}_${tok_tag}_t06_s${seed}/   per-seed eval output
#   results/best5_summary_t06.csv                  best math_avg per checkpoint
#   results/best5_summary_t06.md                   the same in markdown

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ========== Configuration ==========
CKPT_ROOT="${CKPT_ROOT:-/code/hongpaul-sandbox/temp/OPT-RL/opt/checkpoints}"
PATTERN="*boost*"
GPUS="0,1,2,3,4,5,6,7"
RESULTS_BASE="$SCRIPT_DIR/results"
LOG_DIR="$SCRIPT_DIR/logs/eval_boost_math_best5"

# Conda
CONDA_INIT="${CONDA_INIT:-/code/hongpaul-sandbox/cuda/miniconda3/bin/activate}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/code/hongpaul-sandbox/cuda/miniconda3/envs/cuda}"

# WandB
WANDB_API_KEY="${WANDB_API_KEY:-b8f38344ec7231ee89baa74ef7209dd5a43df6b2}"
WANDB_ENTITY="${WANDB_ENTITY:-mhong-university-of-minnesota}"
WANDB_PROJECT="${WANDB_PROJECT:-opt_rl_eval_boost_math_best5}"

# Math benchmarks (AIME excluded)
BENCHMARKS="math500 gsm8k amc olympiadbench mgsm"

# Generation params — T=0.6 fixed by request
TEMPERATURE=0.6
TOP_P=0.95
TOP_K=20
MAX_TOKENS_LIST="8192"
SEEDS=(42 142 242 342 442)            # 5 seeds → 5 evaluation runs per ckpt

# vLLM safety knobs (avoid 8-worker init contention / OOM)
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"  # was 0.90 — too aggressive on shared boxes
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"  # cap context to (prompt 1024 + response 8192) + margin
WORKER_START_STAGGER_SEC="${WORKER_START_STAGGER_SEC:-15}"  # avoid 8 simultaneous CUDA inits

# average@N for competition benchmarks
AVG_AT_MAP="amc:32"

# Parsing
DRY_RUN=false
NO_TMUX=false
FORCE=false
REPORT_ONLY=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)            GPUS="$2"; shift 2 ;;
        --ckpt-root)       CKPT_ROOT="$2"; shift 2 ;;
        --pattern)         PATTERN="$2"; shift 2 ;;
        --dry-run)         DRY_RUN=true; shift ;;
        --no-tmux)         NO_TMUX=true; shift ;;
        --force)           FORCE=true; shift ;;
        --report-only)     REPORT_ONLY=true; shift ;;
        --benchmarks)      BENCHMARKS="$2"; shift 2 ;;
        --wandb-project)   WANDB_PROJECT="$2"; shift 2 ;;
        --avg-at-map)      AVG_AT_MAP="$2"; shift 2 ;;
        --max-tokens)      MAX_TOKENS_LIST="$2"; shift 2 ;;
        --seeds)           IFS=',' read -ra SEEDS <<< "$2"; shift 2 ;;
        *)                 EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# Tag: T fixed at 0.6 here, derive consistently with the sibling script.
T_TAG="t$(echo "$TEMPERATURE" | tr -d '.')"

# ========== Tmux auto-launch ==========
if [[ -z "${TMUX:-}" ]] && [[ "$NO_TMUX" == "false" ]]; then
    TMUX_SESSION="eval_boost_best5_$(date +%m%d_%H%M)"

    FULL_ARGS="--no-tmux"
    FULL_ARGS="$FULL_ARGS --gpus $(printf '%q' "$GPUS")"
    FULL_ARGS="$FULL_ARGS --ckpt-root $(printf '%q' "$CKPT_ROOT")"
    FULL_ARGS="$FULL_ARGS --pattern $(printf '%q' "$PATTERN")"
    FULL_ARGS="$FULL_ARGS --wandb-project $(printf '%q' "$WANDB_PROJECT")"
    FULL_ARGS="$FULL_ARGS --avg-at-map $(printf '%q' "$AVG_AT_MAP")"
    FULL_ARGS="$FULL_ARGS --max-tokens $(printf '%q' "$MAX_TOKENS_LIST")"
    FULL_ARGS="$FULL_ARGS --seeds $(printf '%q' "$(IFS=,; echo "${SEEDS[*]}")")"
    $DRY_RUN     && FULL_ARGS="$FULL_ARGS --dry-run"
    $FORCE       && FULL_ARGS="$FULL_ARGS --force"
    $REPORT_ONLY && FULL_ARGS="$FULL_ARGS --report-only"
    [[ -n "$BENCHMARKS" ]] && FULL_ARGS="$FULL_ARGS --benchmarks $(printf '%q' "$BENCHMARKS")"
    for arg in "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; do FULL_ARGS="$FULL_ARGS $(printf '%q' "$arg")"; done

    tmux new-session -d -s "$TMUX_SESSION" \
        "source $CONDA_INIT && \
         conda activate $CONDA_ENV_PATH && \
         export WANDB_API_KEY=$WANDB_API_KEY && \
         export WANDB_ENTITY=$WANDB_ENTITY && \
         cd $SCRIPT_DIR && \
         bash $SCRIPT_DIR/run_eval_boost_math_best5.sh $FULL_ARGS; \
         exec bash"
    echo "Tmux session '$TMUX_SESSION' started."
    echo "  Attach with:  tmux attach -t $TMUX_SESSION"
    exit 0
fi

# ========== Setup ==========
export WANDB_API_KEY
export WANDB_ENTITY

if ! python -c "import vllm" >/dev/null 2>&1; then
    if [[ -f "$CONDA_INIT" ]]; then
        echo "Activating conda env: $CONDA_ENV_PATH"
        # shellcheck disable=SC1090
        source "$CONDA_INIT"
        conda activate "$CONDA_ENV_PATH"
    fi
fi

IFS=',' read -ra GPU_LIST <<< "$GPUS"
NUM_GPUS=${#GPU_LIST[@]}

PYTHON="${PYTHON:-$(command -v python)}"
[[ -z "$PYTHON" ]] && PYTHON="python3"

LOCK_DIR="$LOG_DIR/locks"
QUEUE_FILE="$LOG_DIR/task_queue.txt"
mkdir -p "$LOG_DIR" "$LOCK_DIR"

# ========== Resolve model path ==========
resolve_model_path() {
    local exp_dir="$1"
    if [[ -f "$exp_dir/config.json" ]]; then
        echo "$exp_dir"
        return
    fi
    local best_step=""
    local best_num=-1
    for step_dir in "$exp_dir"/global_step_*; do
        [[ -d "$step_dir" ]] || continue
        local num="${step_dir##*global_step_}"
        if ! [[ "$num" =~ ^[0-9]+$ ]]; then continue; fi
        local candidate=""
        if [[ -f "$step_dir/actor/huggingface/config.json" ]]; then
            candidate="$step_dir/actor/huggingface"
        elif [[ -f "$step_dir/actor/config.json" ]]; then
            candidate="$step_dir/actor"
        elif [[ -f "$step_dir/config.json" ]]; then
            candidate="$step_dir"
        fi
        if [[ -n "$candidate" ]] && (( num > best_num )); then
            best_num=$num
            best_step="$candidate"
        fi
    done
    [[ -n "$best_step" ]] && echo "$best_step" || echo ""
}

# ========== Build task queue ==========
[[ ! -d "$CKPT_ROOT" ]] && { echo "ERROR: Checkpoint root not found: $CKPT_ROOT"; exit 1; }

shopt -s nullglob
SKIPPED_NO_MODEL=()
SKIPPED_ALREADY_DONE=()
MATCHED_DIRS=0
> "$QUEUE_FILE"
for d in "$CKPT_ROOT"/$PATTERN; do
    [[ -d "$d" ]] || continue
    exp_name=$(basename "$d")
    [[ "$exp_name" == .* ]] && continue
    MATCHED_DIRS=$((MATCHED_DIRS + 1))

    model_path=$(resolve_model_path "$d")
    if [[ -z "$model_path" ]]; then
        SKIPPED_NO_MODEL+=("$exp_name")
        continue
    fi

    for mt in $MAX_TOKENS_LIST; do
        tok_tag="$mt"
        [[ "$mt" == "3072" ]] && tok_tag="3k"
        [[ "$mt" == "8192" ]] && tok_tag="8k"
        for seed in "${SEEDS[@]}"; do
            output_dir="$RESULTS_BASE/${exp_name}_${tok_tag}_${T_TAG}_s${seed}"
            if [[ -f "$output_dir/overall_summary.json" ]] && ! $FORCE; then
                SKIPPED_ALREADY_DONE+=("${exp_name}_${tok_tag}_${T_TAG}_s${seed}")
                continue
            fi
            echo "$model_path|$exp_name|$mt|$tok_tag|$seed" >> "$QUEUE_FILE"
        done
    done
done
shopt -u nullglob

TOTAL_TASKS=$(wc -l < "$QUEUE_FILE")

# ========== Header / diagnostics ==========
echo "============================================================"
echo "  Boost Math Best-of-5 Eval — Worker Mode"
echo "  Ckpt root:                 $CKPT_ROOT"
echo "  Pattern:                   $PATTERN"
echo "  Matched directories:       $MATCHED_DIRS"
echo "  Skipped (no valid model):  ${#SKIPPED_NO_MODEL[@]}"
echo "  Skipped (already done):    ${#SKIPPED_ALREADY_DONE[@]}  (use --force to re-run)"
echo "  Pending tasks:             $TOTAL_TASKS  ($MATCHED_DIRS ckpts × ${#SEEDS[@]} seeds × $(echo $MAX_TOKENS_LIST | wc -w) max-tokens)"
echo "  Seeds:                     ${SEEDS[*]}"
echo "  Max tokens:                $MAX_TOKENS_LIST"
echo "  Benchmarks:                $BENCHMARKS"
echo "  Avg@N:                     $AVG_AT_MAP"
echo "  GPUs:                      ${GPU_LIST[*]} ($NUM_GPUS workers, TP=1)"
echo "  Params:                    T=$TEMPERATURE ($T_TAG) P=$TOP_P K=$TOP_K"
echo "  WandB:                     $WANDB_ENTITY / $WANDB_PROJECT"
echo "  Results:                   $RESULTS_BASE"
echo "============================================================"
echo ""

if (( ${#SKIPPED_NO_MODEL[@]} > 0 )); then
    echo "Skipped (no config.json / global_step_* found):"
    for name in "${SKIPPED_NO_MODEL[@]}"; do echo "  - $name"; done
    echo ""
fi

if [[ $MATCHED_DIRS -eq 0 ]]; then
    echo "WARNING: pattern '$PATTERN' matched zero directories under $CKPT_ROOT."
    echo "  -> Sanity check: ls -d $CKPT_ROOT/$PATTERN"
    exit 0
fi

if $DRY_RUN; then
    echo "[DRY RUN] Would evaluate (model | exp | mt | seed):"
    cat "$QUEUE_FILE" | head -50
    echo "..."
    echo "Total: $TOTAL_TASKS tasks"
    exit 0
fi

# ========== Worker function ==========
worker() {
    local gpu_id="$1"
    local completed=0 failed=0

    while true; do
        local task=""
        task=$( flock -x 200 bash -c '
            t=$(head -1 "'"$QUEUE_FILE"'" 2>/dev/null)
            if [ -n "$t" ]; then
                sed -i "1d" "'"$QUEUE_FILE"'"
            fi
            echo "$t"
        ' 200>"$LOCK_DIR/queue.lock" )

        [[ -z "$task" ]] && break

        IFS='|' read -r model_path exp_name max_tokens tok_tag seed <<< "$task"
        local run_label="${exp_name}_${tok_tag}_${T_TAG}_s${seed}"
        local output_dir="$RESULTS_BASE/${run_label}"
        local log_file="$LOG_DIR/${run_label}.log"

        echo "[GPU $gpu_id] START $run_label"

        if CUDA_VISIBLE_DEVICES="$gpu_id" $PYTHON "$SCRIPT_DIR/eval.py" \
            --backend vllm \
            --model "$model_path" \
            --benchmarks $BENCHMARKS \
            --tensor-parallel-size 1 \
            --dtype auto \
            --gpu-memory-utilization "$GPU_MEM_UTIL" \
            --max-model-len "$MAX_MODEL_LEN" \
            --max-tokens "$max_tokens" \
            --temperature $TEMPERATURE \
            --top-p $TOP_P \
            --top-k $TOP_K \
            --seed "$seed" \
            --avg-at-map "$AVG_AT_MAP" \
            --wandb-project "$WANDB_PROJECT" \
            --wandb-entity "$WANDB_ENTITY" \
            --wandb-run-name "$run_label" \
            --output-dir "$output_dir" \
            > "$log_file" 2>&1; then
            completed=$((completed + 1))
            echo "[GPU $gpu_id] DONE  $run_label (worker total: $completed)"
        else
            failed=$((failed + 1))
            echo "[GPU $gpu_id] FAIL  $run_label — see $log_file"
        fi
    done

    echo "[GPU $gpu_id] Worker finished. Completed: $completed, Failed: $failed"
}

# ========== Launch workers ==========
if ! $REPORT_ONLY && [[ $TOTAL_TASKS -gt 0 ]]; then
    WORKER_PIDS=()
    idx=0
    for gpu_id in "${GPU_LIST[@]}"; do
        if (( idx > 0 && WORKER_START_STAGGER_SEC > 0 )); then
            sleep "$WORKER_START_STAGGER_SEC"      # stagger CUDA inits
        fi
        worker "$gpu_id" &
        WORKER_PIDS+=($!)
        echo "Worker launched on GPU $gpu_id (PID $!)  [stagger: ${WORKER_START_STAGGER_SEC}s]"
        idx=$((idx + 1))
    done
    for pid in "${WORKER_PIDS[@]}"; do wait "$pid" || true; done
    echo ""
fi

if $REPORT_ONLY; then
    echo "[--report-only] Skipping eval, going straight to aggregation."
elif [[ $TOTAL_TASKS -eq 0 ]]; then
    echo "All matched checkpoints already evaluated at T=$TEMPERATURE for all seeds."
    echo "  -> Re-run with --force to overwrite, OR"
    echo "  -> Continuing to aggregation step."
fi

# ========== Aggregate best-of-5 per checkpoint ==========
echo ""
echo "============================================================"
echo "  Aggregating best-of-${#SEEDS[@]} math_avg per checkpoint"
echo "============================================================"

SUMMARY_CSV="$RESULTS_BASE/best5_summary_${T_TAG}.csv"
SUMMARY_MD="$RESULTS_BASE/best5_summary_${T_TAG}.md"

SEEDS_PY="$(IFS=,; echo "${SEEDS[*]}")" \
RESULTS_BASE="$RESULTS_BASE" \
PATTERN="$PATTERN" \
T_TAG="$T_TAG" \
SUMMARY_CSV="$SUMMARY_CSV" \
SUMMARY_MD="$SUMMARY_MD" \
BENCHMARKS_PY="$BENCHMARKS" \
$PYTHON - <<'PY'
import json, os, re
from pathlib import Path

results_base = Path(os.environ["RESULTS_BASE"])
seeds = [int(s) for s in os.environ["SEEDS_PY"].split(",") if s]
pattern = os.environ["PATTERN"]                             # informational
t_tag = os.environ["T_TAG"]
benchmarks = os.environ["BENCHMARKS_PY"].split()
out_csv = Path(os.environ["SUMMARY_CSV"])
out_md = Path(os.environ["SUMMARY_MD"])

# math_avg = mean over the 4 'core' math benchmarks for which we have data.
# We compute math_avg dynamically using whichever of the listed benchmarks
# actually have an accuracy entry in the summary (so missing ones don't
# silently zero things out).
EXCLUDE_FROM_AVG = {"mgsm"}                                  # exclude per group convention
AVG_BENCHES = [b for b in benchmarks if b not in EXCLUDE_FROM_AVG]

# Collect: ckpt_name -> {seed -> per-bench dict}
ckpt_results: dict[str, dict[int, dict]] = {}
suffix_re = re.compile(rf"_(\dk|\d+)_({re.escape(t_tag)})_s(\d+)$")

for d in sorted(results_base.glob("*")):
    if not d.is_dir():
        continue
    summary_path = d / "overall_summary.json"
    if not summary_path.exists():
        continue
    m = suffix_re.search(d.name)
    if not m:
        continue
    seed = int(m.group(3))
    if seed not in seeds:
        continue
    base_name = d.name[: m.start()]                           # strip "_8k_t06_s42"

    try:
        data = json.loads(summary_path.read_text())
    except Exception:
        continue
    bench_acc = {}
    for entry in data.get("benchmarks", []):
        # eval.py writes either {"name", "accuracy"} or
        # {"name", "accuracy", "avg_at_n"} per benchmark
        name = entry.get("name") or entry.get("benchmark")
        acc = entry.get("accuracy")
        if name is not None and acc is not None:
            bench_acc[name] = float(acc)
    ckpt_results.setdefault(base_name, {})[seed] = bench_acc

if not ckpt_results:
    print(f"No completed runs found under {results_base} for T tag '{t_tag}'.")
    raise SystemExit(0)

rows = []
for ckpt, by_seed in sorted(ckpt_results.items()):
    seed_avgs = {}
    for s, accs in by_seed.items():
        present = [accs[b] for b in AVG_BENCHES if b in accs]
        if not present:
            continue
        seed_avgs[s] = sum(present) / len(present)
    if not seed_avgs:
        continue
    best_seed = max(seed_avgs, key=seed_avgs.get)
    best_avg = seed_avgs[best_seed]
    worst_avg = min(seed_avgs.values())
    spread = best_avg - worst_avg
    row = {
        "checkpoint": ckpt,
        "n_seeds": len(by_seed),
        "best_seed": best_seed,
        "best_math_avg": best_avg,
        "mean_math_avg": sum(seed_avgs.values()) / len(seed_avgs),
        "worst_math_avg": worst_avg,
        "spread": spread,
    }
    # Per-benchmark best-seed accuracy
    best_accs = by_seed[best_seed]
    for b in benchmarks:
        row[f"best_{b}"] = best_accs.get(b)
    rows.append(row)

rows.sort(key=lambda r: r["best_math_avg"], reverse=True)

# ---- CSV ----
import csv
fieldnames = list(rows[0].keys())
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"CSV  -> {out_csv}")

# ---- Markdown ----
md_lines = []
md_lines.append(f"# Best-of-{len(seeds)} math_avg @ T=0.6 — boost checkpoints\n")
md_lines.append(f"Seeds: {seeds}\n")
md_lines.append(f"math_avg = mean({', '.join(AVG_BENCHES)})\n")
md_lines.append("")
md_lines.append("| Checkpoint | n_seeds | best_seed | best math_avg | mean | worst | spread | "
                + " | ".join(f"best_{b}" for b in benchmarks) + " |")
md_lines.append("|" + "|".join("---" for _ in range(7 + len(benchmarks))) + "|")
for r in rows:
    fmt = lambda x: f"{x:.4f}" if isinstance(x, float) else ("" if x is None else str(x))
    md_lines.append(
        "| " + " | ".join([
            r["checkpoint"],
            str(r["n_seeds"]),
            str(r["best_seed"]),
            fmt(r["best_math_avg"]),
            fmt(r["mean_math_avg"]),
            fmt(r["worst_math_avg"]),
            fmt(r["spread"]),
            *(fmt(r.get(f"best_{b}")) for b in benchmarks),
        ]) + " |"
    )
out_md.write_text("\n".join(md_lines) + "\n")
print(f"MD   -> {out_md}")

# ---- Console preview ----
print("")
print(f"{'checkpoint':<60s}  {'best':>7s}  {'mean':>7s}  {'worst':>7s}  {'best_seed':>9s}")
for r in rows[:30]:
    print(f"{r['checkpoint'][:60]:<60s}  {r['best_math_avg']:.4f}  "
          f"{r['mean_math_avg']:.4f}  {r['worst_math_avg']:.4f}  {r['best_seed']:>9d}")
PY

echo ""
echo "============================================================"
echo "  Done."
echo "  Summary CSV: $SUMMARY_CSV"
echo "  Summary MD:  $SUMMARY_MD"
echo "  WandB:       https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"
echo "============================================================"
