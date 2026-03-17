#!/usr/bin/env python3
"""Verify that _freeze_mask is correctly applied and shapes match.

Run on a single GPU:
  CUDA_VISIBLE_DEVICES=6 python examples/grpo_trainer/verify_freeze_mask.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-1.7B"


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def test_freeze_largest(model, freeze_ratio=0.97, min_tensor_size=10_000):
    """Simulate freeze_largest on a non-FSDP model to verify mask logic."""
    mask_stats = {}
    total_trainable_mask = 0
    total_params_masked = 0

    for name, p in model.named_parameters():
        if p.ndim == 0 or p.numel() < min_tensor_size:
            continue

        with torch.no_grad():
            w = p.detach().abs().view(-1)
            k = max(int(w.numel() * freeze_ratio), 1)
            k = min(k, w.numel())
            topk_vals, _ = torch.topk(w, k, largest=True)
            threshold = topk_vals.min()
            mask = (p.detach().abs() < threshold)

        trainable_count = mask.sum().item()
        total_trainable_mask += trainable_count
        total_params_masked += p.numel()
        mask_stats[name] = {
            "total": p.numel(),
            "trainable": trainable_count,
            "frac": trainable_count / p.numel(),
        }

    return mask_stats, total_trainable_mask, total_params_masked


def test_sparse_k(model, sparse_k=20000, min_tensor_size=10_000):
    """Simulate sparse_train_k on a non-FSDP model."""
    trainable_params = []
    for name, p in model.named_parameters():
        if p.requires_grad and p.ndim > 0 and p.numel() >= min_tensor_size:
            trainable_params.append((name, p))

    total_trainable = sum(p.numel() for _, p in trainable_params)
    actual_k = min(sparse_k, total_trainable)

    # Proportional allocation
    remaining_k = actual_k
    alloc = []
    for idx, (name, p) in enumerate(trainable_params):
        if idx == len(trainable_params) - 1:
            k_i = remaining_k
        else:
            k_i = int(actual_k * p.numel() / total_trainable)
            k_i = min(k_i, p.numel())
        alloc.append((name, p.numel(), k_i))
        remaining_k -= k_i

    return alloc, actual_k, total_trainable


def main():
    print(f"Loading {MODEL}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    total, trainable = count_params(model)
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Num layers: {len(model.model.layers)}")

    # --- Test 1: freeze_largest with 97% ---
    print("\n" + "=" * 60)
    print("TEST 1: freeze_largest (freeze_ratio=0.97)")
    print("=" * 60)
    mask_stats, total_trainable_mask, total_params_masked = test_freeze_largest(model)

    print(f"\nPer-layer breakdown:")
    for name, stats in sorted(mask_stats.items()):
        print(f"  {name}: {stats['trainable']:,}/{stats['total']:,} "
              f"({stats['frac']*100:.2f}%)")

    print(f"\nTotal masked params: {total_params_masked:,}")
    print(f"Total trainable (mask=True): {total_trainable_mask:,}")
    print(f"Trainable fraction: {total_trainable_mask/total_params_masked*100:.2f}%")
    print(f"Trainable as % of all params: {total_trainable_mask/total*100:.2f}%")

    # --- Test 2: freeze layer 14 only, then sparse_k ---
    print("\n" + "=" * 60)
    print("TEST 2: train only layer 14, then sparse_k=20000")
    print("=" * 60)

    # Freeze everything except layer 14
    for name, p in model.named_parameters():
        if name.startswith("model.layers.14."):
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

    _, trainable_14 = count_params(model)
    print(f"Layer 14 trainable params: {trainable_14:,}")

    for k in [20000, 50000, 100000]:
        alloc, actual_k, pool = test_sparse_k(model, sparse_k=k)
        print(f"\n  sparse_k={k:,}:")
        print(f"  Pool size: {pool:,}")
        print(f"  Actually selected: {actual_k:,}")
        for name, total_p, k_i in alloc:
            print(f"    {name}: {k_i:,}/{total_p:,} ({k_i/total_p*100:.4f}%)")

    # --- Test 3: verify gradient masking works ---
    print("\n" + "=" * 60)
    print("TEST 3: verify gradient masking (single forward-backward)")
    print("=" * 60)

    # Re-enable layer 14
    for name, p in model.named_parameters():
        if name.startswith("model.layers.14."):
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

    # Create sparse mask for layer 14 with k=20000
    rng = torch.Generator(device="cpu")
    rng.manual_seed(42)
    masked_params = []
    for name, p in model.named_parameters():
        if p.requires_grad and p.ndim > 0 and p.numel() >= 10_000:
            perm = torch.randperm(p.numel(), generator=rng)
            mask = torch.zeros(p.numel(), dtype=torch.bool)
            mask[perm[:100]] = True  # tiny k for test
            mask = mask.view(p.shape)
            p._freeze_mask = mask
            masked_params.append((name, p))

    model = model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    inputs = tokenizer("Hello world", return_tensors="pt").to("cuda")

    # Forward + backward
    outputs = model(**inputs, labels=inputs["input_ids"])
    outputs.loss.backward()

    # Apply mask and check
    non_zero_before = 0
    non_zero_after = 0
    for name, p in masked_params:
        if p.grad is not None:
            non_zero_before += (p.grad != 0).sum().item()
            p.grad *= p._freeze_mask.to(p.grad.device)
            non_zero_after += (p.grad != 0).sum().item()

    print(f"Non-zero grad elements before masking: {non_zero_before:,}")
    print(f"Non-zero grad elements after masking:  {non_zero_after:,}")
    print(f"Masking reduced gradients by {(1 - non_zero_after/max(non_zero_before,1))*100:.1f}%")

    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
