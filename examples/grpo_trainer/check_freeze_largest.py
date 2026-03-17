"""
Check freeze_largest parameter distribution for a given model.
Usage: CUDA_VISIBLE_DEVICES=6,7 python check_freeze_largest.py --model Qwen/Qwen3-1.7B --ratio 0.97
"""
import argparse
import torch
from transformers import AutoModelForCausalLM
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
parser.add_argument("--ratio", type=float, default=0.97, help="Fraction of largest weights to freeze")
parser.add_argument("--min-tensor-size", type=int, default=10_000)
args = parser.parse_args()

print(f"Loading {args.model} ...")
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="cpu")
model.eval()

freeze_ratio = args.ratio
min_tensor_size = args.min_tensor_size

total_params = 0
total_trainable = 0
skipped_params = 0

# per-layer stats: layer_idx -> (trainable, total)
layer_stats = defaultdict(lambda: [0, 0])
other_stats = defaultdict(lambda: [0, 0])  # embed, norm, lm_head

for name, p in model.named_parameters():
    numel = p.numel()
    total_params += numel

    if p.ndim == 0 or numel < min_tensor_size:
        skipped_params += numel
        # treat skipped as fully trainable
        total_trainable += numel
        # bucket
        if "model.layers." in name:
            idx = int(name.split("model.layers.")[1].split(".")[0])
            layer_stats[idx][0] += numel
            layer_stats[idx][1] += numel
        else:
            key = name.split(".")[0] if "." in name else name
            other_stats[key][0] += numel
            other_stats[key][1] += numel
        continue

    w = p.detach().abs().view(-1)
    k = max(int(w.numel() * freeze_ratio), 1)
    k = min(k, w.numel())
    topk_vals, _ = torch.topk(w, k, largest=True)
    threshold = topk_vals.min()
    mask = (p.detach().abs() < threshold)
    trainable = mask.sum().item()

    total_trainable += trainable

    if "model.layers." in name:
        idx = int(name.split("model.layers.")[1].split(".")[0])
        layer_stats[idx][0] += trainable
        layer_stats[idx][1] += numel
    else:
        # embed_tokens, lm_head, norm, etc.
        parts = name.split(".")
        key = ".".join(parts[:2]) if len(parts) > 1 else parts[0]
        other_stats[key][0] += trainable
        other_stats[key][1] += numel

print(f"\n{'='*60}")
print(f"Model: {args.model}  |  freeze_ratio={freeze_ratio}")
print(f"{'='*60}")
print(f"{'Component':<35} {'Trainable':>12} {'Total':>12} {'%':>7}")
print(f"{'-'*60}")

for key in sorted(other_stats.keys()):
    tr, tot = other_stats[key]
    print(f"  {key:<33} {tr:>12,} {tot:>12,} {100*tr/tot:>6.2f}%")

print(f"{'-'*60}")
num_layers = max(layer_stats.keys()) + 1 if layer_stats else 0
for idx in range(num_layers):
    if idx in layer_stats:
        tr, tot = layer_stats[idx]
        print(f"  layer {idx:<28} {tr:>12,} {tot:>12,} {100*tr/tot:>6.2f}%")

print(f"{'='*60}")
print(f"  {'TOTAL':<33} {total_trainable:>12,} {total_params:>12,} {100*total_trainable/total_params:>6.2f}%")
print(f"  (skipped small tensors: {skipped_params:,} params, treated as trainable)")
print(f"{'='*60}")
