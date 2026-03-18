"""Dummy GPU job: hold GPU memory and run light workloads to keep the node alive.
Logs heartbeat to wandb. Auto-recovers from CUDA errors.
"""
import os
import signal
import sys
import time

import torch


def main():
    # How much free memory to allocate (lower = safer)
    alloc_ratio = float(os.environ.get("DUMMY_ALLOC_RATIO", "0.70"))

    # Graceful shutdown on SIGTERM (SLURM preemption sends this)
    shutdown = False

    def _handle_signal(signum, frame):
        nonlocal shutdown
        print(f"\n[dummy] Received signal {signum}, shutting down gracefully...")
        shutdown = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    # Dummy jobs do not upload to wandb — only print GPU stats locally.
    use_wandb = False

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("[dummy] No GPUs found, sleeping forever...")
        while not shutdown:
            time.sleep(60)
        return

    def allocate_memory():
        """Allocate GPU memory. Returns (holders, matmul_pairs)."""
        holders = []
        mat_pairs = []
        for i in range(num_gpus):
            dev = torch.device(f"cuda:{i}")
            try:
                free, total = torch.cuda.mem_get_info(dev)
                alloc_bytes = int(free * alloc_ratio)
                n_floats = alloc_bytes // 4
                t = torch.empty(n_floats, device=dev, dtype=torch.float32)
                # Fill with small values to actually commit the memory
                t.fill_(0.001)
                holders.append(t)

                # Small matmul workload (512x512 uses ~1MB, very safe)
                dim = 512
                a = torch.randn(dim, dim, device=dev, dtype=torch.float32)
                b = torch.randn(dim, dim, device=dev, dtype=torch.float32)
                mat_pairs.append((a, b))

                alloc_gb = alloc_bytes / (1024**3)
                total_gb = total / (1024**3)
                print(f"  GPU {i}: allocated {alloc_gb:.1f} GB / {total_gb:.1f} GB "
                      f"({alloc_ratio*100:.0f}%)")
            except RuntimeError as e:
                print(f"  GPU {i}: allocation failed ({e}), skipping")
        return holders, mat_pairs

    print(f"[dummy] Detected {num_gpus} GPUs, allocating {alloc_ratio*100:.0f}% memory...")
    holders, mat_pairs = allocate_memory()

    print(f"[dummy] Running. Ctrl+C or SIGTERM to stop.")
    step = 0
    consecutive_errors = 0
    max_consecutive_errors = 5

    while not shutdown:
        try:
            # Light matmul on each GPU
            for a, b in mat_pairs:
                torch.mm(a, b)
            # Don't synchronize every step — reduces overhead
            if step % 10 == 0:
                torch.cuda.synchronize()

            consecutive_errors = 0  # reset on success

            # Heartbeat every 60 steps (~60s)
            if step % 60 == 0:
                log = {"step": step, "alive": 1}
                for i in range(num_gpus):
                    used = torch.cuda.memory_allocated(i) / (1024**3)
                    log[f"gpu{i}_mem_gb"] = round(used, 2)
                if use_wandb:
                    try:
                        wandb.log(log, step=step)
                    except Exception:
                        pass  # don't crash on wandb errors
                print(f"[dummy] step={step} | " +
                      " | ".join(f"GPU{i}: {log.get(f'gpu{i}_mem_gb', '?')}GB"
                                 for i in range(num_gpus)))

            step += 1
            time.sleep(1)

        except RuntimeError as e:
            consecutive_errors += 1
            err_msg = str(e)
            print(f"[dummy] CUDA error ({consecutive_errors}/{max_consecutive_errors}): "
                  f"{err_msg[:200]}")

            if consecutive_errors >= max_consecutive_errors:
                print("[dummy] Too many consecutive errors, exiting.")
                break

            # Try to recover: clear cache and reallocate
            try:
                del holders, mat_pairs
                torch.cuda.empty_cache()
                time.sleep(5)
                print("[dummy] Attempting reallocation...")
                holders, mat_pairs = allocate_memory()
                print("[dummy] Recovery successful.")
            except Exception as e2:
                print(f"[dummy] Recovery failed: {e2}")

    # Cleanup
    print("[dummy] Cleaning up...")
    try:
        del holders, mat_pairs
        torch.cuda.empty_cache()
    except Exception:
        pass
    print("[dummy] Done.")


if __name__ == "__main__":
    main()
