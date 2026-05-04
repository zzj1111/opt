"""8B math evaluator — process-isolated per ckpt.

Outer loop is a thin orchestrator that for every ckpt:
  1. Spawns a FRESH python subprocess (this script with --single-ckpt mode)
  2. The subprocess loads vLLM, evaluates 4 benches, writes summary.json, EXITS
  3. After exit, the orchestrator pkills any leftover vllm/EngineCore process
  4. Waits for GPU memory to actually free up (polls nvidia-smi)
  5. Only then proceeds to the next ckpt

Why subprocess per ckpt: vLLM spawns engine_core as a subprocess. `del llm`
in the parent does NOT kill that subprocess reliably. The cleanest way to
guarantee CUDA cleanup is to let the OS reap the entire python process —
when this process exits, the kernel forcibly tears down all child processes,
all CUDA contexts, all NCCL state. Nothing leaks.

Usage:
  python eval_8b.py                        # outer loop over all ckpts
  python eval_8b.py --single-ckpt /path/to/ckpt   # inner: eval ONE ckpt and exit
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

_THIS = Path(__file__).resolve()
_THIS_DIR = _THIS.parent
sys.path.insert(0, str(_THIS_DIR))


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # mode
    p.add_argument("--single-ckpt", default=None,
                   help="if set, evaluate ONLY this one ckpt dir and exit "
                        "(used internally by the outer-loop spawn)")

    # outer-loop only
    p.add_argument("--ckpt-root", default="/code/hongpaul-sandbox/temp/OPT-RL/opt/checkpoints")
    p.add_argument("--pattern", default="*Qwen3-8B-Base*")
    p.add_argument("--gpu-free-gib", type=float, default=120.0,
                   help="orchestrator: each GPU must have at least this much free "
                        "memory (GiB) before the next ckpt is launched")
    p.add_argument("--gpu-free-timeout", type=int, default=120,
                   help="orchestrator: max seconds to wait for GPU to free up")

    # both modes
    p.add_argument("--results-dir", default=str(_THIS_DIR / "results"))
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amc-n-samples", type=int, default=32)
    p.add_argument("--benches", default="math500,gsm8k,amc,olympiadbench")
    p.add_argument("--subset", type=int, default=None)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--wandb-project", default="opt_rl_eval_8b_math")
    p.add_argument("--wandb-entity", default="mhong-university-of-minnesota")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


# =============================================================================
# Helpers
# =============================================================================

def find_model_path(ckpt_dir: Path) -> Path | None:
    best, best_n = None, -1
    for step in ckpt_dir.glob("global_step_*"):
        if not step.is_dir():
            continue
        try:
            n = int(step.name.split("_")[-1])
        except ValueError:
            continue
        cand = step / "actor" / "huggingface"
        if (cand / "config.json").is_file() and n > best_n:
            best, best_n = cand, n
    return best


def kill_leftover_vllm() -> None:
    """Nuclear cleanup of any vllm / EngineCore process the previous eval left behind."""
    for pat in ("vllm.entrypoints", "vllm.engine", "vllm.worker", "EngineCore", "engine_core"):
        try:
            subprocess.run(["pkill", "-9", "-f", pat], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass


def gpu_free_gib(visible_gpus: list[int] | None = None) -> list[float]:
    """Return free memory (GiB) per GPU. Empty list if nvidia-smi not available."""
    cmd = ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
    if visible_gpus is not None and len(visible_gpus) > 0:
        cmd.extend(["-i", ",".join(str(i) for i in visible_gpus)])
    try:
        out = subprocess.check_output(cmd, text=True, timeout=10)
    except Exception:
        return []
    return [float(x) / 1024 for x in out.strip().splitlines() if x.strip()]


def wait_for_gpus_free(min_free_gib: float, timeout_sec: int) -> None:
    """Block until every GPU has >= min_free_gib free, or timeout expires."""
    elapsed = 0
    while elapsed < timeout_sec:
        frees = gpu_free_gib()
        if frees and min(frees) >= min_free_gib:
            print(f"  [orch] GPUs ready: min free = {min(frees):.1f} GiB")
            return
        snapshot = "  ".join(f"GPU{i}={f:.0f}G" for i, f in enumerate(frees))
        print(f"  [orch] waiting for GPUs to free (need {min_free_gib} GiB each): {snapshot}  ({elapsed}s)")
        time.sleep(10)
        elapsed += 10
    print(f"  [orch] WARNING: GPUs still not all free after {timeout_sec}s — proceeding anyway")


# =============================================================================
# OUTER LOOP — orchestrator
# =============================================================================

def outer_loop(args: argparse.Namespace) -> int:
    ckpt_root = Path(args.ckpt_root)
    if not ckpt_root.is_dir():
        print(f"ERROR: --ckpt-root not found: {ckpt_root}")
        return 1
    ckpts = sorted(p for p in ckpt_root.glob(args.pattern) if p.is_dir())
    if not ckpts:
        print(f"ERROR: pattern '{args.pattern}' matched zero dirs under {ckpt_root}")
        return 1

    # validate bench names early
    from math_eval.benchmarks import BENCHMARKS
    bad = [b.strip() for b in args.benches.split(",") if b.strip() not in BENCHMARKS]
    if bad:
        print(f"ERROR: unknown bench(es): {bad}.  Available: {sorted(BENCHMARKS.keys())}")
        return 1

    print("=" * 70)
    print(f"  ckpts to evaluate: {len(ckpts)}")
    print(f"  benches:           {args.benches}")
    print(f"  TP:                {args.tp}")
    print(f"  results dir:       {args.results_dir}")
    print(f"  wandb:             {'OFF' if args.no_wandb else f'{args.wandb_entity}/{args.wandb_project}'}")
    print(f"  isolation:         each ckpt runs in its own python subprocess")
    print(f"  cleanup:           pkill -9 vllm + wait for GPUs >= {args.gpu_free_gib} GiB free between ckpts")
    print("=" * 70)

    for i, ckpt_dir in enumerate(ckpts, 1):
        name = ckpt_dir.name
        summary_path = Path(args.results_dir) / f"{name}_8k_t06" / "summary.json"
        if summary_path.exists() and not args.force:
            print(f"\n[{i}/{len(ckpts)}] [skip] {name} — already evaluated")
            continue

        print("\n" + "=" * 70)
        print(f"[{i}/{len(ckpts)}] {name}")
        print("=" * 70)

        # Pre-flight: make sure GPUs are clean BEFORE we even spawn
        wait_for_gpus_free(args.gpu_free_gib, args.gpu_free_timeout)

        # Spawn an isolated subprocess that does the actual eval and exits
        cmd = [sys.executable, str(_THIS), "--single-ckpt", str(ckpt_dir)]
        # forward all the inner-relevant flags
        for flag, val in [
            ("--results-dir", args.results_dir),
            ("--tp", args.tp),
            ("--gpu-memory-utilization", args.gpu_memory_utilization),
            ("--max-model-len", args.max_model_len),
            ("--max-tokens", args.max_tokens),
            ("--temperature", args.temperature),
            ("--top-p", args.top_p),
            ("--top-k", args.top_k),
            ("--seed", args.seed),
            ("--amc-n-samples", args.amc_n_samples),
            ("--benches", args.benches),
            ("--wandb-project", args.wandb_project),
            ("--wandb-entity", args.wandb_entity),
        ]:
            cmd.extend([flag, str(val)])
        if args.subset is not None:
            cmd.extend(["--subset", str(args.subset)])
        if args.enable_thinking:
            cmd.append("--enable-thinking")
        if args.no_wandb:
            cmd.append("--no-wandb")
        if args.force:
            cmd.append("--force")

        print(f"  [orch] spawn: {' '.join(shlex.quote(x) for x in cmd)}")
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"  [orch] subprocess exited with code {rc} — continuing to next ckpt")

        # Nuclear cleanup so the next ckpt starts on a clean GPU
        print(f"  [orch] killing any leftover vllm / engine_core processes ...")
        kill_leftover_vllm()
        time.sleep(5)

    print("\n" + "=" * 70)
    print("  All ckpts processed.")
    print("=" * 70)
    return 0


# =============================================================================
# INNER MODE — eval ONE ckpt then exit
# =============================================================================

def eval_one_bench(llm, bench_name: str, sampling_params, n_samples: int,
                   args, predictions_path: Path) -> dict[str, Any]:
    from math_eval.benchmarks import get_benchmark

    bench = get_benchmark(bench_name)
    items = bench.load(subset=args.subset)
    print(f"\n  -- bench: {bench.name}  ({len(items)} problems"
          + (f" × {n_samples} samples)" if n_samples > 1 else ")"))

    messages_list = [bench.build_prompt(it) for it in items]

    if n_samples > 1:
        from vllm import SamplingParams
        sp = SamplingParams(
            n=n_samples,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
    else:
        sp = sampling_params

    chat_kwargs: dict[str, Any] = {}
    if args.enable_thinking:
        chat_kwargs["chat_template_kwargs"] = {"enable_thinking": True}

    t0 = time.time()
    outputs = llm.chat(messages_list, sp, **chat_kwargs)
    wall = time.time() - t0

    correct_total = 0
    total = 0
    rows: list[dict[str, Any]] = []
    for it, o in zip(items, outputs):
        sample_records: list[dict[str, Any]] = []
        n_correct_this = 0
        for s in o.outputs:
            text = s.text
            pred = bench.extract_answer(text)
            ok = bool(bench.is_correct(pred, it.gold_answer))
            sample_records.append({"text": text, "pred": pred, "correct": ok})
            n_correct_this += int(ok)
        n_actual = len(o.outputs)
        rows.append({
            "id": it.id,
            "question": it.question,
            "gold_answer": it.gold_answer,
            "n_correct": n_correct_this,
            "n_samples": n_actual,
            "samples": sample_records,
        })
        correct_total += n_correct_this
        total += n_actual

    accuracy = correct_total / total if total else 0.0

    with predictions_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  -- {bench.name}: accuracy = {accuracy:.4f}  "
          f"({correct_total}/{total})  [{wall:.1f}s]")

    return {
        "name": bench.name,
        "accuracy": accuracy,
        "correct": correct_total,
        "total": total,
        "n_samples_per_problem": n_samples,
        "wall_time_sec": wall,
    }


def inner_single_ckpt(args: argparse.Namespace) -> int:
    ckpt_dir = Path(args.single_ckpt)
    name = ckpt_dir.name

    out_dir = Path(args.results_dir) / f"{name}_8k_t06"
    summary_path = out_dir / "summary.json"
    if summary_path.exists() and not args.force:
        print(f"[skip] {name} — already done ({summary_path})")
        return 0

    model_path = find_model_path(ckpt_dir)
    if model_path is None:
        print(f"[skip] {name} — no completed global_step_*/actor/huggingface")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)

    print(f"  model: {model_path}")
    print(f"  out:   {out_dir}")

    from vllm import LLM, SamplingParams
    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=args.tp,
        dtype="auto",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        seed=args.seed,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=f"{name}_8k_t06",
                config={"ckpt": name, "model_path": str(model_path), "tp": args.tp,
                        "max_model_len": args.max_model_len,
                        "temperature": args.temperature,
                        "amc_n_samples": args.amc_n_samples},
                reinit=True,
            )
        except Exception as e:
            print(f"  [wandb] init failed: {e} — continuing without wandb")

    bench_summaries: list[dict[str, Any]] = []
    for bench_name in args.benches.split(","):
        bench_name = bench_name.strip()
        if not bench_name:
            continue
        n_samples = args.amc_n_samples if bench_name == "amc" else 1
        pred_path = pred_dir / f"{bench_name}.jsonl"
        try:
            s = eval_one_bench(llm, bench_name, sampling_params, n_samples, args, pred_path)
        except Exception as e:
            print(f"  [FAIL] bench {bench_name}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            s = {"name": bench_name, "accuracy": None, "error": str(e)}
        bench_summaries.append(s)
        if wandb_run is not None and isinstance(s.get("accuracy"), (int, float)):
            tag = f"_avg@{n_samples}" if n_samples > 1 else ""
            wandb_run.log({f"eval/{bench_name}{tag}": s["accuracy"]})

    accs = [b["accuracy"] for b in bench_summaries
            if isinstance(b.get("accuracy"), (int, float))]
    math_avg = sum(accs) / len(accs) if accs else 0.0
    overall = {
        "ckpt": name,
        "model_path": str(model_path),
        "math_avg": math_avg,
        "benchmarks": bench_summaries,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    summary_path.write_text(json.dumps(overall, indent=2, ensure_ascii=False))
    print(f"\n  math_avg = {math_avg:.4f}  ->  {summary_path}")

    if wandb_run is not None:
        wandb_run.log({"eval/math_avg": math_avg})
        wandb_run.finish()

    # When this function returns + Python exits, the OS forcibly cleans up
    # all CUDA contexts, all vLLM subprocesses, all NCCL state. Nothing leaks.
    del llm
    gc.collect()
    return 0


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    args = parse_args()
    if args.single_ckpt is not None:
        return inner_single_ckpt(args)
    return outer_loop(args)


if __name__ == "__main__":
    sys.exit(main())
