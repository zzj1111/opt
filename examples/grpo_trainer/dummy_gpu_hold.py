"""Dummy GPU job: allocate all GPU memory and run matmuls to keep the node alive.
Logs heartbeat to wandb so it shows up as a run named "dummy".
"""
import os
import time
import torch

def main():
    project = os.environ.get("WANDB_PROJECT", "verl_grpo_math")
    run_name = os.environ.get("DUMMY_RUN_NAME", "dummy_gpu_hold")

    os.environ.setdefault("WANDB_API_KEY", "b8f38344ec7231ee89baa74ef7209dd5a43df6b2")
    os.environ.setdefault("WANDB_ENTITY", "mhong-university-of-minnesota")

    # Try to use wandb, but don't crash if unavailable
    try:
        import wandb
        wandb.init(project=project, entity=os.environ["WANDB_ENTITY"], name=run_name, tags=["dummy"])
        use_wandb = True
    except Exception:
        use_wandb = False
        print("[dummy] wandb unavailable, running without logging")

    num_gpus = torch.cuda.device_count()
    print(f"[dummy] Detected {num_gpus} GPUs, allocating memory...")

    # Allocate ~90% of each GPU's memory with large tensors
    tensors = []
    for i in range(num_gpus):
        dev = torch.device(f"cuda:{i}")
        free, total = torch.cuda.mem_get_info(dev)
        alloc_bytes = int(free * 0.90)
        n_floats = alloc_bytes // 4  # float32 = 4 bytes
        t = torch.randn(n_floats, device=dev, dtype=torch.float32)
        tensors.append(t)
        alloc_gb = alloc_bytes / (1024**3)
        total_gb = total / (1024**3)
        print(f"  GPU {i}: allocated {alloc_gb:.1f} GB / {total_gb:.1f} GB")

    # Prepare matmul workloads (one per GPU)
    dim = 4096
    mats = []
    for i in range(num_gpus):
        dev = torch.device(f"cuda:{i}")
        a = torch.randn(dim, dim, device=dev, dtype=torch.float32)
        b = torch.randn(dim, dim, device=dev, dtype=torch.float32)
        mats.append((a, b))

    print(f"[dummy] Running matmuls on {num_gpus} GPUs. Ctrl+C to stop.")
    step = 0
    try:
        while True:
            for i, (a, b) in enumerate(mats):
                torch.mm(a, b)
            torch.cuda.synchronize()

            if step % 60 == 0:
                log = {"step": step, "alive": 1}
                for i in range(num_gpus):
                    used = torch.cuda.memory_allocated(i) / (1024**3)
                    log[f"gpu{i}_mem_gb"] = used
                if use_wandb:
                    wandb.log(log, step=step)
                print(f"[dummy] step={step} heartbeat logged")

            step += 1
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[dummy] Interrupted, cleaning up...")
    finally:
        if use_wandb:
            wandb.finish()
        del tensors, mats
        torch.cuda.empty_cache()
        print("[dummy] Done.")

if __name__ == "__main__":
    main()
