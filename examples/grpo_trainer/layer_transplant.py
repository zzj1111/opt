"""
Layer Transplant Experiment for RLVR Analysis
==============================================
Given a base model and a full-RL-trained model, transplant different layer ranges
from the RL model into the base model and evaluate, to determine which layers
carry the effective information from RLVR training.

Three experiment modes:
1. Window transplant: slide a fixed-size window across all layers
2. Single-layer transplant: transplant one layer at a time (28 experiments)
3. Cumulative transplant: start from layer 14, expand outward symmetrically

Usage:
    python layer_transplant.py \
        --base_model <path_to_base> \
        --rl_model <path_to_full_rl> \
        --mode [window|single|cumulative|all] \
        --eval_dataset <path_to_eval_data> \
        --output_dir ./transplant_results \
        --window_size 5 \
        --center_layer 14 \
        --batch_size 8 \
        --num_samples 500
"""

import argparse
import json
import copy
import os
import re
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader

# Add verl to path for reward_score utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from verl.utils.reward_score.prime_math import compute_score as prime_compute_score


# ============================================================
# 1. Transplant logic
# ============================================================

def get_num_layers(model) -> int:
    """Get number of transformer layers."""
    return len(model.model.layers)


def transplant_layers(base_model, rl_model, layer_ids: list[int]):
    """
    Create a hybrid model by copying specified layers from rl_model into base_model.
    Returns a new model (deep copy of base with RL layers transplanted).
    
    This is a direct parameter copy: for each layer in layer_ids,
    hybrid.layers[l] gets the exact weights of rl_model.layers[l].
    """
    hybrid = copy.deepcopy(base_model)
    
    for l in layer_ids:
        # Copy all parameters from RL model's layer to hybrid
        rl_layer_state = rl_model.model.layers[l].state_dict()
        hybrid.model.layers[l].load_state_dict(rl_layer_state)
    
    return hybrid


def transplant_layers_inplace(model, rl_model, layer_ids: list[int]):
    """
    In-place version to avoid repeated deep copies (saves memory).
    Modifies `model` directly. Call restore_layers() to undo.
    Returns a dict of original state_dicts for restoration.
    """
    originals = {}
    for l in layer_ids:
        originals[l] = {k: v.clone() for k, v in model.model.layers[l].state_dict().items()}
        rl_state = rl_model.model.layers[l].state_dict()
        model.model.layers[l].load_state_dict(rl_state)
    return originals


def restore_layers(model, originals: dict):
    """Restore layers from saved original state_dicts."""
    for l, state_dict in originals.items():
        model.model.layers[l].load_state_dict(state_dict)


# ============================================================
# 2. Experiment configurations
# ============================================================

def generate_window_configs(num_layers: int, window_size: int = 5) -> list[dict]:
    """Sliding window transplant: move a fixed-size window across all layers."""
    configs = []
    for start in range(0, num_layers - window_size + 1, window_size):
        end = min(start + window_size - 1, num_layers - 1)
        layer_ids = list(range(start, end + 1))
        configs.append({
            "name": f"window_{start:02d}_{end:02d}",
            "layer_ids": layer_ids,
            "description": f"Transplant layers {start}-{end}"
        })
    return configs


def generate_single_layer_configs(num_layers: int) -> list[dict]:
    """Single-layer transplant: one layer at a time."""
    configs = []
    for l in range(num_layers):
        configs.append({
            "name": f"single_{l:02d}",
            "layer_ids": [l],
            "description": f"Transplant layer {l} only"
        })
    return configs


def generate_cumulative_configs(num_layers: int, center: int = 14) -> list[dict]:
    """
    Cumulative transplant: start from center layer, expand outward symmetrically.
    E.g., center=14: [14] -> [13,14,15] -> [12,13,14,15,16] -> ...
    """
    configs = []
    max_radius = max(center, num_layers - 1 - center)
    
    for radius in range(0, max_radius + 1):
        start = max(0, center - radius)
        end = min(num_layers - 1, center + radius)
        layer_ids = list(range(start, end + 1))
        configs.append({
            "name": f"cumul_r{radius:02d}_L{start:02d}_{end:02d}",
            "layer_ids": layer_ids,
            "description": f"Cumulative radius={radius}: layers {start}-{end} ({len(layer_ids)} layers)"
        })
    return configs


# ============================================================
# 3. Evaluation
# ============================================================

def load_eval_dataset(path: str, num_samples: Optional[int] = None) -> list[dict]:
    """
    Load evaluation dataset. Supports:
      - parquet (MATH format: prompt as chat messages, reward_model.ground_truth)
      - jsonl with {"prompt": "...", "answer": "..."}
    """
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
        if num_samples:
            df = df.head(num_samples)
        data = []
        for _, row in df.iterrows():
            # Extract chat-format prompt → plain text
            prompt_msgs = row["prompt"]
            if isinstance(prompt_msgs, list):
                # Take the user message content
                prompt_text = next(
                    (m["content"] for m in prompt_msgs if m["role"] == "user"),
                    str(prompt_msgs),
                )
            else:
                prompt_text = str(prompt_msgs)

            # Extract ground truth from reward_model dict
            rm = row.get("reward_model", {})
            if isinstance(rm, dict):
                gt = str(rm.get("ground_truth", ""))
            else:
                gt = str(row.get("answer", ""))

            data.append({"prompt": prompt_text, "answer": gt})
        return data
    else:
        data = []
        with open(path, "r") as f:
            for line in f:
                item = json.loads(line.strip())
                data.append(item)
                if num_samples and len(data) >= num_samples:
                    break
        return data


def check_answer(model_output: str, ground_truth: str) -> tuple[bool, str]:
    """
    Check answer using prime_math's grader (supports sympy symbolic comparison).
    Returns (is_correct, extracted_answer).
    """
    try:
        result = prime_compute_score(model_output, ground_truth)
        if isinstance(result, tuple):
            is_correct, _, extracted = result
            return bool(is_correct), str(extracted)
        return bool(result), ""
    except Exception:
        return False, ""


def evaluate_model(
    model,
    tokenizer,
    eval_data: list[dict],
    batch_size: int = 8,
    max_new_tokens: int = 3072,
    temperature: float = 0.0,
) -> dict:
    """
    Evaluate model on MATH problems using prime_math grader.
    Returns dict with accuracy and per-sample results.
    """
    model.eval()
    correct = 0
    total = 0
    results = []

    # Build chat prompts
    for i in range(0, len(eval_data), batch_size):
        batch = eval_data[i:i + batch_size]
        gt_answers = [str(item["answer"]).strip() for item in batch]

        # Format as chat for Qwen models
        chat_prompts = []
        for item in batch:
            messages = [{"role": "user", "content": item["prompt"]}]
            chat_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            chat_prompts.append(chat_text)

        inputs = tokenizer(
            chat_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)

        with torch.no_grad():
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            if temperature == 0.0:
                gen_kwargs["do_sample"] = False
            else:
                gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)

            outputs = model.generate(**inputs, **gen_kwargs)

        for j, (output, gt_answer) in enumerate(zip(outputs, gt_answers)):
            input_len = inputs["input_ids"][j].shape[0]
            generated = tokenizer.decode(output[input_len:], skip_special_tokens=True)

            is_correct, pred_answer = check_answer(generated, gt_answer)
            correct += int(is_correct)
            total += 1

            results.append({
                "prompt": batch[j]["prompt"][:200],
                "gt_answer": gt_answer,
                "pred_answer": pred_answer,
                "generated": generated[:500],
                "correct": is_correct,
            })

        # Progress
        done = min(i + batch_size, len(eval_data))
        acc_so_far = correct / total if total > 0 else 0
        print(f"    Progress: {done}/{len(eval_data)}  acc={acc_so_far:.4f}")

    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_sample": results,
    }


# ============================================================
# 4. Analysis utilities
# ============================================================

def compute_delta_norms(base_model, rl_model) -> dict:
    """
    Compute per-layer L2 norm of weight deltas between base and RL model.
    Useful for understanding where full RL training made the biggest changes.
    """
    num_layers = get_num_layers(base_model)
    delta_norms = {}
    
    for l in range(num_layers):
        layer_delta_norm = 0.0
        base_params = dict(base_model.model.layers[l].named_parameters())
        rl_params = dict(rl_model.model.layers[l].named_parameters())
        
        for name in base_params:
            delta = rl_params[name].data.float() - base_params[name].data.float()
            layer_delta_norm += delta.norm().item() ** 2
        
        delta_norms[l] = layer_delta_norm ** 0.5
    
    return delta_norms


def compute_cosine_similarity_of_deltas(
    base_model, rl_model, single_layer_model, layer_id: int
) -> float:
    """
    Compare the delta_W at `layer_id` between:
      - full RL training: W_rl - W_base
      - single-layer training: W_single - W_base
    
    High cosine similarity means full RL and single-layer training
    moved this layer in the same direction.
    """
    full_delta = []
    single_delta = []
    
    base_params = dict(base_model.model.layers[layer_id].named_parameters())
    rl_params = dict(rl_model.model.layers[layer_id].named_parameters())
    single_params = dict(single_layer_model.model.layers[layer_id].named_parameters())
    
    for name in base_params:
        d_full = (rl_params[name].data.float() - base_params[name].data.float()).flatten()
        d_single = (single_params[name].data.float() - base_params[name].data.float()).flatten()
        full_delta.append(d_full)
        single_delta.append(d_single)
    
    full_vec = torch.cat(full_delta)
    single_vec = torch.cat(single_delta)
    
    cos_sim = torch.nn.functional.cosine_similarity(
        full_vec.unsqueeze(0), single_vec.unsqueeze(0)
    ).item()
    
    return cos_sim


# ============================================================
# 5. Main experiment runner
# ============================================================

def run_experiments(args):
    print(f"{'='*60}")
    print(f"Layer Transplant Experiment")
    print(f"{'='*60}")
    print(f"Base model:  {args.base_model}")
    print(f"RL model:    {args.rl_model}")
    print(f"Mode:        {args.mode}")
    print(f"Output:      {args.output_dir}")
    print()
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load models
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print("Loading RL model...")
    rl_model = AutoModelForCausalLM.from_pretrained(
        args.rl_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    num_layers = get_num_layers(base_model)
    print(f"Number of layers: {num_layers}")
    
    # Load eval data
    print(f"Loading eval dataset from {args.eval_dataset}...")
    eval_data = load_eval_dataset(args.eval_dataset, args.num_samples)
    print(f"Loaded {len(eval_data)} samples")
    
    # Compute per-layer delta norms (informational)
    print("\nComputing per-layer delta norms (||W_rl - W_base||)...")
    delta_norms = compute_delta_norms(base_model, rl_model)
    for l, norm in delta_norms.items():
        print(f"  Layer {l:2d}: {norm:.6f}")
    
    # Generate experiment configs
    configs = []
    
    # Always evaluate baselines
    baselines = [
        {"name": "base_model", "layer_ids": [], "description": "Base model (no transplant)"},
        {"name": "full_rl", "layer_ids": list(range(num_layers)), "description": "Full RL model (all layers transplanted)"},
    ]
    configs.extend(baselines)
    
    if args.mode in ("window", "all"):
        configs.extend(generate_window_configs(num_layers, args.window_size))
    if args.mode in ("single", "all"):
        configs.extend(generate_single_layer_configs(num_layers))
    if args.mode in ("cumulative", "all"):
        configs.extend(generate_cumulative_configs(num_layers, args.center_layer))
    
    # Run experiments with in-place transplant (memory efficient)
    all_results = {}
    
    # Use base_model as the working copy; transplant in-place and restore
    for idx, config in enumerate(configs):
        name = config["name"]
        layer_ids = config["layer_ids"]
        desc = config["description"]
        
        print(f"\n[{idx+1}/{len(configs)}] {name}: {desc}")
        
        if name == "full_rl":
            # Evaluate RL model directly
            eval_result = evaluate_model(
                rl_model, tokenizer, eval_data,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
            )
        elif len(layer_ids) == 0:
            # Evaluate base model directly
            eval_result = evaluate_model(
                base_model, tokenizer, eval_data,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
            )
        else:
            # Transplant, evaluate, restore
            originals = transplant_layers_inplace(base_model, rl_model, layer_ids)
            eval_result = evaluate_model(
                base_model, tokenizer, eval_data,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
            )
            restore_layers(base_model, originals)
        
        accuracy = eval_result["accuracy"]
        print(f"  Accuracy: {accuracy:.4f} ({eval_result['correct']}/{eval_result['total']})")
        
        all_results[name] = {
            "config": config,
            "accuracy": accuracy,
            "correct": eval_result["correct"],
            "total": eval_result["total"],
        }
        
        # Save intermediate results
        results_path = os.path.join(args.output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump({
                "args": vars(args),
                "delta_norms": delta_norms,
                "results": all_results,
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    base_acc = all_results.get("base_model", {}).get("accuracy", 0)
    rl_acc = all_results.get("full_rl", {}).get("accuracy", 0)
    rl_gain = rl_acc - base_acc
    
    print(f"Base model:  {base_acc:.4f}")
    print(f"Full RL:     {rl_acc:.4f} (gain: +{rl_gain:.4f})")
    print(f"{'─'*60}")
    
    # Sort by accuracy
    sorted_results = sorted(
        [(k, v) for k, v in all_results.items() if k not in ("base_model", "full_rl")],
        key=lambda x: x[1]["accuracy"],
        reverse=True,
    )
    
    for name, res in sorted_results:
        acc = res["accuracy"]
        recovery = (acc - base_acc) / rl_gain * 100 if rl_gain > 0 else 0
        n_layers = len(res["config"]["layer_ids"])
        print(f"  {name:30s}  acc={acc:.4f}  recovery={recovery:6.1f}%  ({n_layers} layers)")
    
    print(f"\nResults saved to {results_path}")
    return all_results


# ============================================================
# 6. Plotting
# ============================================================

def plot_results(results_path: str, output_dir: str):
    """Generate plots from saved results."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("matplotlib not available, skipping plots")
        return
    
    with open(results_path) as f:
        data = json.load(f)
    
    results = data["results"]
    delta_norms = {int(k): v for k, v in data.get("delta_norms", {}).items()}
    
    base_acc = results.get("base_model", {}).get("accuracy", 0)
    rl_acc = results.get("full_rl", {}).get("accuracy", 0)
    
    # ---- Plot 1: Single-layer transplant profile ----
    single_results = {
        int(k.split("_")[1]): v["accuracy"]
        for k, v in results.items()
        if k.startswith("single_")
    }
    
    if single_results:
        fig, ax1 = plt.subplots(figsize=(12, 5))
        
        layers = sorted(single_results.keys())
        accs = [single_results[l] for l in layers]
        
        color1 = "#2196F3"
        ax1.bar(layers, accs, color=color1, alpha=0.7, label="Transplant accuracy")
        ax1.axhline(y=base_acc, color="gray", linestyle="--", alpha=0.7, label=f"Base ({base_acc:.3f})")
        ax1.axhline(y=rl_acc, color="red", linestyle="--", alpha=0.7, label=f"Full RL ({rl_acc:.3f})")
        ax1.set_xlabel("Layer ID")
        ax1.set_ylabel("Accuracy", color=color1)
        ax1.set_title("Single-Layer Transplant: Which layer carries the most RL information?")
        ax1.legend(loc="upper left")
        
        # Overlay delta norms
        if delta_norms:
            ax2 = ax1.twinx()
            color2 = "#FF9800"
            norms = [delta_norms.get(l, 0) for l in layers]
            ax2.plot(layers, norms, color=color2, marker="o", markersize=3, label="||ΔW||")
            ax2.set_ylabel("||W_rl - W_base||", color=color2)
            ax2.legend(loc="upper right")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "single_layer_transplant.png"), dpi=150)
        plt.close()
        print("Saved single_layer_transplant.png")
    
    # ---- Plot 2: Cumulative transplant saturation curve ----
    cumul_results = {}
    for k, v in results.items():
        if k.startswith("cumul_"):
            n_layers = len(v["config"]["layer_ids"])
            cumul_results[n_layers] = v["accuracy"]
    
    if cumul_results:
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ns = sorted(cumul_results.keys())
        accs = [cumul_results[n] for n in ns]
        
        ax.plot(ns, accs, "o-", color="#4CAF50", markersize=5)
        ax.axhline(y=base_acc, color="gray", linestyle="--", alpha=0.7, label=f"Base ({base_acc:.3f})")
        ax.axhline(y=rl_acc, color="red", linestyle="--", alpha=0.7, label=f"Full RL ({rl_acc:.3f})")
        ax.set_xlabel("Number of transplanted layers (expanding from center)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Cumulative Transplant: How quickly does performance saturate?")
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "cumulative_transplant.png"), dpi=150)
        plt.close()
        print("Saved cumulative_transplant.png")
    
    # ---- Plot 3: Window transplant comparison ----
    window_results = {}
    for k, v in results.items():
        if k.startswith("window_"):
            mid = np.mean(v["config"]["layer_ids"])
            window_results[mid] = v["accuracy"]
    
    if window_results:
        fig, ax = plt.subplots(figsize=(10, 5))
        
        mids = sorted(window_results.keys())
        accs = [window_results[m] for m in mids]
        
        ax.bar(mids, accs, width=2, color="#9C27B0", alpha=0.7)
        ax.axhline(y=base_acc, color="gray", linestyle="--", alpha=0.7, label=f"Base ({base_acc:.3f})")
        ax.axhline(y=rl_acc, color="red", linestyle="--", alpha=0.7, label=f"Full RL ({rl_acc:.3f})")
        ax.set_xlabel("Window center (layer position)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Window Transplant: Which region matters most?")
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "window_transplant.png"), dpi=150)
        plt.close()
        print("Saved window_transplant.png")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Layer Transplant Experiment for RLVR")
    parser.add_argument("--base_model", type=str, required=True, help="Path to base model")
    parser.add_argument("--rl_model", type=str, required=True, help="Path to full RL model")
    parser.add_argument("--eval_dataset", type=str, required=True, help="Path to eval jsonl")
    parser.add_argument("--output_dir", type=str, default="./transplant_results")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["window", "single", "cumulative", "all"],
                        help="Experiment mode")
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--center_layer", type=int, default=14,
                        help="Center layer for cumulative transplant")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--plot_only", type=str, default=None,
                        help="Skip experiments, just plot from this results.json")
    
    args = parser.parse_args()
    
    if args.plot_only:
        plot_results(args.plot_only, os.path.dirname(args.plot_only))
    else:
        run_experiments(args)
        # Auto-plot after experiments
        results_path = os.path.join(args.output_dir, "results.json")
        if os.path.exists(results_path):
            plot_results(results_path, args.output_dir)
