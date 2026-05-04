"""Self-contained 8B math eval — vLLM inference + answer extraction + wandb.

No dependency on eval.py or math_eval/. Loads benchmarks straight from HF,
calls vLLM directly, scores answers in-process, logs one wandb run per ckpt.

For every directory under --ckpt-root matching --pattern (default
*Qwen3-8B-Base*), this script:
  1. Resolves the latest global_step_*/actor/huggingface model path.
  2. Loads vLLM once with that model.
  3. Runs the 4 math benches (math500, gsm8k, amc avg@32, olympiadbench)
     by batching all prompts into one llm.generate call per (bench, sample).
  4. Scores each bench (\\boxed{...} extraction + normalization, with
     sympy equivalence as a fallback for math).
  5. Writes <ckpt>_8k_t06/{summary.json, predictions/<bench>.jsonl}.
  6. Logs eval/<bench>, eval/math_avg to wandb.
  7. Tears down vLLM, garbage-collects, evaluates the next ckpt.

Sequential ckpts; only one vLLM alive at a time; no worker pool.

Usage:
  python eval_8b.py
  python eval_8b.py --tp 8 --pattern '*resume*'
  python eval_8b.py --no-wandb --ckpt-root /local/ckpts
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Disable broken JIT path on this server before importing vllm.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-root", default="/code/hongpaul-sandbox/temp/OPT-RL/opt/checkpoints")
    p.add_argument("--pattern", default="*Qwen3-8B-Base*",
                   help="glob under --ckpt-root")
    p.add_argument("--results-dir", default=str(Path(__file__).parent / "results"))
    p.add_argument("--tp", type=int, default=2,
                   help="vLLM tensor-parallel size (default 2)")
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
                   help="comma list; subset of math500,gsm8k,amc,olympiadbench")
    p.add_argument("--olympiad-config", default="OE_TO_maths_en_COMP",
                   help="OlympiadBench HF config (text-only English math)")
    p.add_argument("--olympiad-limit", type=int, default=675,
                   help="cap olympiad to first N problems (default = 675, the standard subset)")
    p.add_argument("--wandb-project", default="opt_rl_eval_8b_math")
    p.add_argument("--wandb-entity", default="mhong-university-of-minnesota")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="re-run ckpts that already have summary.json")
    return p.parse_args()


# =============================================================================
# Benchmark loaders — straight from HuggingFace
# =============================================================================

@dataclass
class Problem:
    qid: str
    question: str
    gold: str


def load_math500() -> list[Problem]:
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [Problem(qid=str(i), question=ex["problem"], gold=str(ex["answer"]))
            for i, ex in enumerate(ds)]


def load_gsm8k() -> list[Problem]:
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    out: list[Problem] = []
    for i, ex in enumerate(ds):
        ans = ex["answer"].split("####")[-1].strip().replace(",", "")
        out.append(Problem(qid=str(i), question=ex["question"], gold=ans))
    return out


def load_amc() -> list[Problem]:
    from datasets import load_dataset
    ds = load_dataset("AI-MO/aimo-validation-amc", split="train")
    return [Problem(qid=str(ex.get("id", i)), question=ex["problem"],
                    gold=str(ex["answer"]))
            for i, ex in enumerate(ds)]


def load_olympiad(config: str, limit: int | None = None) -> list[Problem]:
    from datasets import load_dataset
    ds = load_dataset("Hothan/OlympiadBench", config, split="train")
    out: list[Problem] = []
    for i, ex in enumerate(ds):
        if limit is not None and i >= limit:
            break
        ans = ex.get("final_answer") or ex.get("answer") or ""
        if isinstance(ans, list):
            ans = ans[0] if ans else ""
        out.append(Problem(qid=str(i), question=ex.get("question", ex.get("problem", "")),
                           gold=str(ans)))
    return out


# =============================================================================
# Prompt
# =============================================================================

PROMPT_TEMPLATE = (
    "Solve the following problem step by step. "
    "Put your final answer inside \\boxed{{}}.\n\n"
    "Problem: {question}\n\n"
    "Solution:"
)


def build_prompt(p: Problem) -> str:
    return PROMPT_TEMPLATE.format(question=p.question)


# =============================================================================
# Answer extraction + scoring
# =============================================================================

_BOXED_RE = re.compile(r"\\boxed\s*\{")


def extract_boxed(text: str) -> str | None:
    """Return content of the LAST \\boxed{...} with balanced braces."""
    matches = list(_BOXED_RE.finditer(text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    i = start
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
        i += 1
    return None


_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def extract_last_number(text: str) -> str | None:
    """Last numeric token in text, with thousands-separator commas removed."""
    nums = _NUMBER_RE.findall(text)
    return nums[-1].replace(",", "") if nums else None


def normalize_math(s: str) -> str:
    """Light normalization for string-equality fallback before sympy."""
    if s is None:
        return ""
    s = s.strip()
    # strip $...$ wrappers
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    # remove all whitespace
    s = re.sub(r"\s+", "", s)
    # unify common LaTeX shorthand
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\:", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "").replace("^{°}", "").replace("°", "")
    s = s.rstrip(".")
    s = s.lstrip("+")
    if s.endswith("\\\\"):
        s = s[:-2]
    return s


def is_math_equivalent(pred: str | None, gold: str) -> bool:
    """True if pred matches gold under normalization OR sympy equivalence."""
    if pred is None:
        return False
    if normalize_math(pred) == normalize_math(gold):
        return True
    # Try sympy as a fallback (slow; only run when string-eq fails).
    try:
        from sympy import Rational, simplify, sympify
        from sympy.parsing.latex import parse_latex

        def _to_expr(x: str):
            x = x.replace(",", "")
            for parser in (parse_latex, sympify):
                try:
                    return parser(x)
                except Exception:
                    continue
            return None

        ea = _to_expr(normalize_math(pred))
        eb = _to_expr(normalize_math(gold))
        if ea is None or eb is None:
            return False
        try:
            return bool(simplify(ea - eb) == 0)
        except Exception:
            return ea == eb
    except Exception:
        return False


def score_one(bench: str, pred_text: str, gold: str) -> bool:
    if bench == "gsm8k":
        pred = extract_last_number(pred_text)
        if pred is None:
            return False
        try:
            return abs(float(pred) - float(gold)) < 1e-6
        except ValueError:
            return pred.strip() == str(gold).strip()
    # math500 / amc / olympiad — \boxed{...}
    pred = extract_boxed(pred_text)
    return is_math_equivalent(pred, gold)


# =============================================================================
# Eval one bench
# =============================================================================

def eval_bench(llm, bench: str, problems: list[Problem],
               sampling_params, n_samples: int, predictions_path: Path,
               args) -> dict[str, Any]:
    """Generate completions for every problem, score, return summary dict."""
    print(f"\n  -- bench: {bench}  ({len(problems)} problems"
          + (f" × {n_samples} samples)" if n_samples > 1 else ")"))

    prompts = [build_prompt(p) for p in problems]
    t0 = time.time()
    if n_samples > 1:
        sp = sampling_params.clone()
        sp.n = n_samples
    else:
        sp = sampling_params
    outputs = llm.generate(prompts, sp, use_tqdm=True)
    wall = time.time() - t0

    # outputs is a list aligned with prompts; each has .outputs (list of n samples)
    correct_total = 0
    total = 0
    rows = []
    for prob, out in zip(problems, outputs):
        sample_correct = 0
        sample_responses = []
        for s in out.outputs:
            txt = s.text
            ok = score_one(bench, txt, prob.gold)
            sample_responses.append({"text": txt, "correct": bool(ok)})
            sample_correct += int(ok)
        n_actual = len(out.outputs)
        # Pass-rate = mean per-sample correctness for this problem
        rows.append({
            "qid": prob.qid,
            "gold": prob.gold,
            "n_correct": sample_correct,
            "n_samples": n_actual,
            "samples": sample_responses,
        })
        correct_total += sample_correct
        total += n_actual

    # Overall accuracy across all (problem × sample) pairs
    accuracy = correct_total / total if total else 0.0

    # Write per-problem predictions
    with predictions_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  -- {bench}: accuracy = {accuracy:.4f}  ({correct_total}/{total})  "
          f"[{wall:.1f}s]")

    return {
        "name": bench,
        "accuracy": accuracy,
        "correct": correct_total,
        "total": total,
        "n_samples_per_problem": n_samples,
        "wall_time_sec": wall,
    }


# =============================================================================
# Eval one ckpt
# =============================================================================

BENCH_LOADERS = {
    "math500": lambda args: load_math500(),
    "gsm8k":   lambda args: load_gsm8k(),
    "amc":     lambda args: load_amc(),
    "olympiadbench": lambda args: load_olympiad(args.olympiad_config, args.olympiad_limit),
}


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


def eval_ckpt(ckpt_dir: Path, args, bench_data: dict[str, list[Problem]]) -> None:
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

    # Lazy import vllm so a missing benchmarks package doesn't block import.
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=args.tp,
        dtype="auto",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        seed=args.seed,
        enforce_eager=False,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    # Optional wandb
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
                },
                reinit=True,
            )
        except Exception as e:
            print(f"  [wandb] init failed: {e} — continuing without wandb")

    bench_summaries: list[dict[str, Any]] = []
    for bench in args.benches.split(","):
        bench = bench.strip()
        problems = bench_data[bench]
        n_samples = args.amc_n_samples if bench == "amc" else 1
        pred_path = pred_dir / f"{bench}.jsonl"
        s = eval_bench(llm, bench, problems, sampling_params, n_samples, pred_path, args)
        bench_summaries.append(s)
        if wandb_run is not None:
            metric_name = f"eval/{bench}" + (f"_avg@{n_samples}" if n_samples > 1 else "")
            wandb_run.log({metric_name: s["accuracy"]})

    # math_avg = simple mean of the bench accuracies
    accs = [b["accuracy"] for b in bench_summaries if isinstance(b["accuracy"], (int, float))]
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

    # Tear vLLM down so the next ckpt has a clean GPU.
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

    # Load every bench once up front (avoids reloading across ckpts).
    print("\n[loading benchmarks ...]")
    bench_data: dict[str, list[Problem]] = {}
    for bench in args.benches.split(","):
        bench = bench.strip()
        if bench not in BENCH_LOADERS:
            print(f"  unknown bench: {bench} (skipping)")
            continue
        t0 = time.time()
        bench_data[bench] = BENCH_LOADERS[bench](args)
        print(f"  {bench:<14s}: {len(bench_data[bench])} problems  [{time.time()-t0:.1f}s]")

    for ckpt_dir in ckpts:
        try:
            eval_ckpt(ckpt_dir, args, bench_data)
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
