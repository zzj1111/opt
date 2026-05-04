"""Self-contained 8B math evaluator that delegates to the BENCHMARKS registry.

Pipeline:
  for every dir under --ckpt-root matching --pattern:
    resolve latest global_step_*/actor/huggingface
    skip if results/<ckpt>_8k_t06/summary.json already exists
    LLM(model=...)                                 ← one vLLM per ckpt
    for bench in args.benches:
        b = get_benchmark(bench)                   ← uses math_eval/benchmarks
        items   = b.load()
        prompts = [b.build_prompt(it) for it in items]   ← chat-format messages
        outs    = llm.chat(prompts, sampling)              (or n=N for avg@N)
        for it, out in zip(items, outs.outputs):
            for sample in out:
                pred = b.extract_answer(sample.text)
                ok   = b.is_correct(pred, it.gold_answer)
        log accuracy to wandb
    write summary.json + per-bench predictions.jsonl
    tear vLLM down (gc + empty_cache); next ckpt

No worker pool. ONE vLLM alive at a time. Sequential ckpts, sequential benches
per ckpt (single llm.chat call per bench batches all prompts).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Disable broken JIT path on this server before importing vllm.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

# Add math_eval to path so `from math_eval.benchmarks import get_benchmark` works
# regardless of CWD.
_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR))


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-root", default="/code/hongpaul-sandbox/temp/OPT-RL/opt/checkpoints")
    p.add_argument("--pattern", default="*Qwen3-8B-Base*",
                   help="glob under --ckpt-root")
    p.add_argument("--results-dir", default=str(_THIS_DIR / "results"))
    p.add_argument("--tp", type=int, default=1,
                   help="vLLM tensor-parallel size (default 1 — uniproc executor, "
                        "NO TP subprocesses; uses 1 GPU, leaves the rest idle)")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amc-n-samples", type=int, default=32,
                   help="AMC avg@N (default 32)")
    p.add_argument("--benches", default="math500,gsm8k,amc,olympiadbench",
                   help="comma list; any name registered in math_eval.benchmarks")
    p.add_argument("--subset", type=int, default=None,
                   help="limit each bench to first N problems (debug)")
    p.add_argument("--enable-thinking", action="store_true",
                   help="enable Qwen3 <think> tags in chat template")
    p.add_argument("--wandb-project", default="opt_rl_eval_8b_math")
    p.add_argument("--wandb-entity", default="mhong-university-of-minnesota")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="re-run ckpts that already have summary.json")
    return p.parse_args()


# =============================================================================
# Helpers
# =============================================================================

def find_model_path(ckpt_dir: Path) -> Path | None:
    """Latest global_step_*/actor/huggingface that has a config.json."""
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


# =============================================================================
# Per-bench eval
# =============================================================================

def eval_bench(llm, bench_name: str, sampling_params, n_samples: int,
               args, predictions_path: Path) -> dict[str, Any]:
    """Run one benchmark with one llm.chat call, score, return summary."""
    from math_eval.benchmarks import get_benchmark

    bench = get_benchmark(bench_name)
    items = bench.load(subset=args.subset)
    print(f"\n  -- bench: {bench.name}  ({len(items)} problems"
          + (f" × {n_samples} samples)" if n_samples > 1 else ")"))

    messages_list = [bench.build_prompt(it) for it in items]

    # For avg@N use a copy of sampling_params with n=N; else use as-is.
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

    # outputs aligned with items; each o.outputs is a list of n samples.
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


# =============================================================================
# Per-ckpt eval
# =============================================================================

def eval_ckpt(ckpt_dir: Path, args) -> None:
    name = ckpt_dir.name
    out_dir = Path(args.results_dir) / f"{name}_8k_t06"
    summary_path = out_dir / "summary.json"
    if summary_path.exists() and not args.force:
        print(f"[skip] {name} — already done ({summary_path})")
        return

    model_path = find_model_path(ckpt_dir)
    if model_path is None:
        print(f"[skip] {name} — no completed global_step_*/actor/huggingface")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 70)
    print(f"[ckpt] {name}")
    print(f"  model: {model_path}")
    print(f"  out:   {out_dir}")
    print("=" * 70)

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

    # Optional wandb run (one per ckpt; logs eval/<bench> + eval/math_avg)
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=f"{name}_8k_t06",
                config={
                    "ckpt": name,
                    "model_path": str(model_path),
                    "tp": args.tp,
                    "max_model_len": args.max_model_len,
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "amc_n_samples": args.amc_n_samples,
                    "enable_thinking": args.enable_thinking,
                },
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
            s = eval_bench(llm, bench_name, sampling_params, n_samples, args, pred_path)
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
        "args": {
            "tp": args.tp,
            "max_model_len": args.max_model_len,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "amc_n_samples": args.amc_n_samples,
            "benches": args.benches.split(","),
            "enable_thinking": args.enable_thinking,
        },
        "math_avg": math_avg,
        "benchmarks": bench_summaries,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    summary_path.write_text(json.dumps(overall, indent=2, ensure_ascii=False))
    print(f"\n  math_avg = {math_avg:.4f}  ->  {summary_path}")

    if wandb_run is not None:
        wandb_run.log({"eval/math_avg": math_avg})
        wandb_run.finish()

    # Tear vLLM down so the next ckpt has clean GPUs.
    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    args = parse_args()

    ckpt_root = Path(args.ckpt_root)
    if not ckpt_root.is_dir():
        print(f"ERROR: --ckpt-root not found: {ckpt_root}")
        return 1
    ckpts = sorted(p for p in ckpt_root.glob(args.pattern) if p.is_dir())
    if not ckpts:
        print(f"ERROR: pattern '{args.pattern}' matched zero dirs under {ckpt_root}")
        return 1

    # Validate bench names early — fail fast if user typoed.
    from math_eval.benchmarks import BENCHMARKS
    bad = [b.strip() for b in args.benches.split(",") if b.strip() not in BENCHMARKS]
    if bad:
        print(f"ERROR: unknown benchmark(s): {bad}")
        print(f"  Available: {sorted(BENCHMARKS.keys())}")
        return 1

    print("=" * 70)
    print(f"  ckpts to evaluate: {len(ckpts)}")
    print(f"  benches:           {args.benches}")
    print(f"  TP:                {args.tp}")
    print(f"  max_model_len:     {args.max_model_len}")
    print(f"  max_tokens:        {args.max_tokens}")
    print(f"  T={args.temperature}  top_p={args.top_p}  top_k={args.top_k}")
    print(f"  amc avg@:          {args.amc_n_samples}")
    print(f"  results dir:       {args.results_dir}")
    print(f"  wandb:             {'OFF' if args.no_wandb else f'{args.wandb_entity}/{args.wandb_project}'}")
    print("=" * 70)

    for ckpt_dir in ckpts:
        try:
            eval_ckpt(ckpt_dir, args)
        except Exception as e:
            print(f"[FAIL] {ckpt_dir.name}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            # Continue to next ckpt — never let one bad model take down the whole run.

    print("\n" + "=" * 70)
    print("  All ckpts processed.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
