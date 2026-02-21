# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os
import re
import wandb
import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor
import torch.distributed as dist
import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def _unwrap_model(m: torch.nn.Module, max_depth: int = 8) -> torch.nn.Module:
    # unwrap common wrappers: DDP/FSDP(.module chain), torch.compile (_orig_mod)
    for _ in range(max_depth):
        if hasattr(m, "module") and isinstance(getattr(m, "module"), torch.nn.Module):
            m = m.module
        else:
            break
    if hasattr(m, "_orig_mod") and isinstance(getattr(m, "_orig_mod"), torch.nn.Module):
        m = m._orig_mod
    return m

@torch.no_grad()
def resolve_last_layer_param(
    model: torch.nn.Module,
    *,
    prefer_output_embeddings: bool = True,
) -> tuple[torch.nn.Parameter, bool]:
    """
    返回:
      last_param: 语义上的“最后一层输出头权重”参数对象（Parameter）
      tied: 是否与 input embedding 权重共享（weight tying）
    """
    base = _unwrap_model(model)

    out_w = None
    in_w = None

    if prefer_output_embeddings and hasattr(base, "get_output_embeddings"):
        out = base.get_output_embeddings()
        if out is not None and hasattr(out, "weight"):
            out_w = out.weight

    if hasattr(base, "get_input_embeddings"):
        inp = base.get_input_embeddings()
        if inp is not None and hasattr(inp, "weight"):
            in_w = inp.weight

    # 兜底：常见命名
    if out_w is None:
        for attr in ("lm_head", "classifier", "score", "head", "fc", "output"):
            if hasattr(base, attr):
                mod = getattr(base, attr)
                if hasattr(mod, "weight"):
                    out_w = mod.weight
                    break

    if out_w is None:
        raise RuntimeError("Cannot resolve last layer weight: no get_output_embeddings() and no known head attr.")
    print(out_w.dim(),tuple(out_w.shape))
    tied = (in_w is not None) and (out_w is in_w)
    return out_w, tied


def get_optimizer_step(optimizer):
    """
    安全地从优化器中获取当前的 step 数。
    兼容 FSDP、DDP 以及普通的单卡训练。
    """
    # 1. 遍历所有参数组
    for group in optimizer.param_groups:
        for p in group['params']:
            # 2. 检查这个参数是否有状态
            if p in optimizer.state:
                state = optimizer.state[p]
                
                # 3. 检查是否有 'step' 键 (有些参数可能被冻结，没有 step)
                if 'step' in state:
                    step_val = state['step']
                    
                    # 4. 关键处理：PyTorch 版本差异
                    # 有时候 step 是 int，有时候是 Tensor (在开启 capturable=True 时)
                    if isinstance(step_val, torch.Tensor):
                        return int(step_val.item())
                    return int(step_val)
                    
    # 如果遍历完都没找到（比如刚初始化还没 step，或者 optimizer 是空的）
    return 0



def _get_layer_group(name):
    """Extract layer group name from parameter name for per-layer analysis."""
    m = re.search(r'(layers\.\d+|embed_tokens|lm_head|model\.norm)', name)
    if m:
        return m.group(1)
    return "other"


class LMHeadGradTracker:
    """Capture the true lm_head gradient via module hooks, bypassing FSDP's
    reduce-scatter which corrupts tied-weight gradients.

    Maintains its own Adam exp_avg / exp_avg_sq (on CPU, rank-0 only) so that
    per-token effective LR can be computed correctly.

    Memory overhead (rank 0, CPU):
        2 x [vocab_size, hidden_size] x float32
        e.g. Qwen3-1.7B: 2 x 151936 x 2048 x 4 ≈ 2.4 GB CPU RAM
    Memory overhead (all ranks, GPU, temporary):
        1 x [vocab_size, hidden_size] x float32 during finalize_step
    """

    def __init__(self, lm_head_module):
        self.grad_accum = None       # [V, H] float32, GPU, accumulated across micro-batches
        self.step_count = 0
        self.exp_avg = None          # [V, H] float32, CPU, rank 0 only
        self.exp_avg_sq = None       # [V, H] float32, CPU, rank 0 only

        # Only use a forward hook. Inside it we register a *tensor-level* hook
        # on the output to capture grad_output during backward.
        # Module-level backward hooks (full_backward_hook / pre_hook) insert
        # custom autograd Functions that break downstream inplace ops (div_).
        self._fwd_handle = lm_head_module.register_forward_hook(self._fwd_hook)

    # ------------------------------------------------------------------
    def _fwd_hook(self, module, input, output):
        if not (module.training and torch.is_grad_enabled()):
            return
        saved_input = input[0].detach().clone()   # [*, H]
        tracker = self                            # prevent closure over self via __del__ cycle

        def _tensor_grad_hook(grad):
            # grad: [..., V]  gradient of loss w.r.t. lm_head output (logits)
            go = grad.detach().float().reshape(-1, grad.shape[-1])              # [N, V]
            inp = saved_input.float().reshape(-1, saved_input.shape[-1])        # [N, H]
            local_grad = go.t() @ inp                                           # [V, H]
            if tracker.grad_accum is None:
                tracker.grad_accum = local_grad
            else:
                tracker.grad_accum += local_grad

        # Tensor hook: fires when gradient flows through this tensor.
        # Does NOT modify the autograd graph, so no view/inplace conflicts.
        output.register_hook(_tensor_grad_hook)

    # ------------------------------------------------------------------
    def finalize_step(self, optimizer_pg, clip_ratio=1.0):
        """Call after grad-clip, before optimizer.step().

        Args:
            optimizer_pg: optimizer.param_groups[0] (current hyper-params)
            clip_ratio:   min(1, max_norm / pre_clip_norm) applied to match clipping

        Returns:
            dict with per-token stats (rank 0), empty dict (other ranks).
        """
        rank = dist.get_rank()

        if self.grad_accum is None:
            return {}

        # Reduce local gradients → rank 0  (collective, all ranks must call)
        dist.reduce(self.grad_accum, dst=0, op=dist.ReduceOp.SUM)

        stats = {}
        if rank == 0:
            lr = optimizer_pg["lr"]
            is_adam = "betas" in optimizer_pg

            # Move to CPU and apply the same clip ratio the optimizer saw
            full_grad = self.grad_accum.cpu().float() * clip_ratio
            self.step_count += 1

            # --- per-token gradient norm ---
            stats["g_norm"] = full_grad.norm(dim=1).tolist()
            stats["shape"]  = list(full_grad.shape)

            if is_adam:
                eps = optimizer_pg.get("eps", 1e-8)
                beta1, beta2 = optimizer_pg["betas"]

                # --- update manual Adam states ---
                if self.exp_avg is None:
                    self.exp_avg    = torch.zeros_like(full_grad)
                    self.exp_avg_sq = torch.zeros_like(full_grad)

                self.exp_avg.mul_(beta1).add_(full_grad, alpha=1 - beta1)
                self.exp_avg_sq.mul_(beta2).addcmul_(full_grad, full_grad, value=1 - beta2)

                # --- per-token momentum norm ---
                stats["mom_cls_norm"] = self.exp_avg.norm(dim=1).tolist()

                # --- per-token effective LR ---
                bc1 = 1 - beta1 ** self.step_count
                bc2 = 1 - beta2 ** self.step_count
                denom = (self.exp_avg_sq.sqrt() / (bc2 ** 0.5)).add_(eps)
                eff_lr = lr / bc1 / denom
                stats["eff_lr_cls_mean"] = eff_lr.mean(dim=1).tolist()

                # --- aggregate ---
                stats["agg_g_norm"] = full_grad.norm().item()
                stats["agg_m_norm"] = self.exp_avg.norm().item()
                stats["agg_eff_lr_mean"] = eff_lr.mean().item()
                stats["agg_numel"] = full_grad.numel()
            else:
                # SGD (with optional momentum)
                mu = optimizer_pg.get("momentum", 0.0)

                if mu > 0:
                    if self.exp_avg is None:
                        self.exp_avg = torch.zeros_like(full_grad)
                    # SGD momentum: buf = mu * buf + grad
                    self.exp_avg.mul_(mu).add_(full_grad)
                    stats["mom_cls_norm"] = self.exp_avg.norm(dim=1).tolist()
                    stats["agg_m_norm"] = self.exp_avg.norm().item()
                else:
                    stats["mom_cls_norm"] = [0.0] * full_grad.shape[0]
                    stats["agg_m_norm"] = 0.0

                # SGD effective LR is constant for all tokens
                stats["eff_lr_cls_mean"] = [lr] * full_grad.shape[0]
                stats["agg_g_norm"] = full_grad.norm().item()
                stats["agg_eff_lr_mean"] = lr
                stats["agg_numel"] = full_grad.numel()

            del full_grad

        self.grad_accum = None  # reset for next step (all ranks)
        return stats

    def remove_hooks(self):
        self._fwd_handle.remove()


@torch.no_grad()
def get_fsdp_comprehensive_analysis(model, optimizer, rms_norm=False):
    stats = {"global": {}, "last_layer": {}, "per_layer": {}}
    rank = dist.get_rank()

    # 1. 获取优化器超参数
    pg = optimizer.param_groups[0]
    lr = pg['lr']
    eps = pg.get('eps', 1e-8)
    beta1, beta2 = pg.get('betas', (0.0, 0.0))
    is_adam = 'betas' in pg  # Adam-family vs SGD-family

    # 2. 全局统计 + per-layer 统计
    # Discover layer groups first
    base_model = getattr(model, "module", model)
    layer_groups = {}  # group_name -> index
    for name, _ in base_model.named_parameters():
        group = _get_layer_group(name)
        if group not in layer_groups:
            layer_groups[group] = len(layer_groups)

    # Sort layer names for consistent ordering
    sorted_names = sorted(layer_groups.keys(), key=lambda x: (
        0 if x == "embed_tokens" else
        1 if x.startswith("layers.") else
        2 if x == "model.norm" else
        3 if x == "lm_head" else 4,
        int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) and x.startswith("layers.") else 0
    ))
    layer_idx = {name: i for i, name in enumerate(sorted_names)}
    num_layers = len(sorted_names)

    # accum: [5] for global, per_layer_accum: [num_layers, 5] for per-layer
    # Columns: 0=w_norm_sq, 1=g_norm_sq, 2=m_norm_sq, 3=eff_lr_sum, 4=numel
    dev = torch.cuda.current_device()
    accum = torch.zeros(5, device=dev, dtype=torch.float64)
    per_layer_accum = torch.zeros(num_layers, 5, device=dev, dtype=torch.float64)

    # Build param->name mapping
    param_to_name = {p: n for n, p in base_model.named_parameters()}

    for p in model.parameters():
        w_sq = p.norm().pow(2)
        numel = p.numel()
        accum[0] += w_sq
        accum[4] += numel

        g_sq = torch.tensor(0.0, device=dev, dtype=torch.float64)
        if p.grad is not None:
            g_sq = p.grad.norm().pow(2)
            accum[1] += g_sq

        m_sq = torch.tensor(0.0, device=dev, dtype=torch.float64)
        elr_sum = torch.tensor(0.0, device=dev, dtype=torch.float64)
        if p in optimizer.state:
            st = optimizer.state[p]
            if is_adam and "exp_avg" in st and "exp_avg_sq" in st:
                step = st.get("step")
                if isinstance(step, torch.Tensor): step = step.float().mean().item()
                bias_c1 = 1 - beta1 ** step
                bias_c2 = 1 - beta2 ** step
                m_sq = st["exp_avg"].norm().pow(2)
                accum[2] += m_sq
                denom = (st["exp_avg_sq"].sqrt() / (bias_c2**0.5)).add_(eps)
                elr_sum = (lr / bias_c1 / denom).sum()
                accum[3] += elr_sum
            elif not is_adam and "momentum_buffer" in st:
                m_sq = st["momentum_buffer"].norm().pow(2)
                accum[2] += m_sq
                # SGD effective LR is constant = lr for every element
                elr_sum = lr * numel
                accum[3] += elr_sum

        # Per-layer accumulation
        name = param_to_name.get(p, None)
        if name is not None:
            group = _get_layer_group(name)
            idx = layer_idx.get(group, None)
            if idx is not None:
                per_layer_accum[idx, 0] += w_sq
                per_layer_accum[idx, 1] += g_sq
                per_layer_accum[idx, 2] += m_sq
                per_layer_accum[idx, 3] += elr_sum
                per_layer_accum[idx, 4] += numel

    # Single all_reduce for both global and per-layer
    combined = torch.cat([accum.unsqueeze(0), per_layer_accum], dim=0)  # [1+num_layers, 5]
    dist.all_reduce(combined, op=dist.ReduceOp.SUM)
    accum = combined[0]
    per_layer_accum = combined[1:]

    g_num = accum[4].item()
    stats["global"]["param_norm"] = (accum[0] / (g_num if rms_norm else 1.0)).sqrt().item()
    stats["global"]["grad_norm"] = (accum[1] / (g_num if rms_norm else 1.0)).sqrt().item()
    stats["global"]["mom_norm"] = (accum[2] / (accum[4] if rms_norm else 1.0)).sqrt().item()
    stats["global"]["eff_lr_mean"] = (accum[3] / g_num).item()

    # Per-layer stats
    stats["per_layer"]["layer_names"] = sorted_names
    stats["per_layer"]["w_norm"] = []
    stats["per_layer"]["g_norm"] = []
    stats["per_layer"]["m_norm"] = []
    stats["per_layer"]["eff_lr_mean"] = []
    for i in range(num_layers):
        n = per_layer_accum[i, 4].item()
        stats["per_layer"]["w_norm"].append(per_layer_accum[i, 0].sqrt().item())
        stats["per_layer"]["g_norm"].append(per_layer_accum[i, 1].sqrt().item())
        stats["per_layer"]["m_norm"].append(per_layer_accum[i, 2].sqrt().item())
        stats["per_layer"]["eff_lr_mean"].append((per_layer_accum[i, 3] / n).item() if n > 0 else 0.0)

    # 3. 最后一层 (LM_Head) 
    lm_head = getattr(model, "get_output_embeddings", lambda: None)() or getattr(model, "lm_head", None)
    if lm_head is not None:
        shard_w = lm_head.weight
        orig_grad = shard_w.grad # 备份
        
        # 内部工具：利用 FSDP 官方 summon 逻辑还原有序矩阵
        def get_ordered_full_matrix(state_key):
            if shard_w not in optimizer.state: return None
            # 临时将状态存入 grad 位，利用 FSDP 索引图还原
            shard_w.grad = optimizer.state[shard_w][state_key].to(shard_w.dtype)
            full_matrix = None
            with FSDP.summon_full_params(model, with_grads=True):
                if rank == 0:
                    full_matrix = lm_head.weight.grad.clone().float()
            return full_matrix

        # A. 还原参数和梯度
        with FSDP.summon_full_params(model, with_grads=True):
            if rank == 0:
                stats["last_layer"]["shape"] = list(lm_head.weight.shape)
                stats["last_layer"]["p_norm"] = lm_head.weight.norm(dim=1).cpu().tolist()
                if lm_head.weight.grad is not None:
                    stats["last_layer"]["g_norm"] = lm_head.weight.grad.norm(dim=1).cpu().tolist()

        # B. 还原有序的动量并计算正确的 EffLR
        if is_adam:
            full_m = get_ordered_full_matrix("exp_avg")
            full_v = get_ordered_full_matrix("exp_avg_sq")

            if rank == 0 and full_m is not None and full_v is not None:
                step = optimizer.state[shard_w].get("step")
                if isinstance(step, torch.Tensor): step = step.float().mean().item()
                bc1, bc2 = 1 - beta1**step, 1 - beta2**step

                full_v.sqrt_().div_(bc2**0.5).add_(eps)
                full_elr = (lr / bc1) / full_v

                stats["last_layer"]["mom_cls_norm"] = full_m.norm(dim=1).cpu().tolist()
                stats["last_layer"]["eff_lr_cls_mean"] = full_elr.mean(dim=1).cpu().tolist()
                del full_m, full_v, full_elr
        else:
            full_m = get_ordered_full_matrix("momentum_buffer")
            if rank == 0 and full_m is not None:
                stats["last_layer"]["mom_cls_norm"] = full_m.norm(dim=1).cpu().tolist()
                # SGD: effective LR is constant for all tokens
                stats["last_layer"]["eff_lr_cls_mean"] = [lr] * full_m.shape[0]
                del full_m

        shard_w.grad = orig_grad # 还原现场
        
    return stats

@torch.no_grad()
def compute_grad_momentum_norms_fsdp_safe(
    model,
    optimizer,
    *,
    process_group=None,
    rms_norm: bool = False,
    momentum_key: str | None = None,
    last_layer_names: list[str] | None = None,
    last_layer_keywords: list[str] | None = None,
):
    """
    Compute global ||G|| and ||M|| (momentum) norms in a FSDP-safe way.

    - Gradient norm uses p.grad (assume you call after scaler.unscale_ if using fp16).
    - Momentum norm uses optimizer.state[p][momentum_key], where momentum_key is inferred if None.

    Returns dict:
    {
        "momentum_key": "...",
        "global": {"grad": float, "mom": float},
        "last_layer": {"grad": float, "mom": float, "matched": [names...]},
    }
    """
    if process_group is None and dist.is_initialized():
        process_group = dist.group.WORLD

    def _allreduce_sum(x: torch.Tensor) -> torch.Tensor:
        if dist.is_initialized():
            dist.all_reduce(x, op=dist.ReduceOp.SUM, group=process_group)
        return x

    def _finish_norm(sumsq: torch.Tensor, numel: torch.Tensor) -> float:
        # sumsq, numel are global-reduced scalars on device
        if rms_norm:
            return float((sumsq / numel.clamp_min(1.0)).sqrt().item())
        else:
            return float(sumsq.sqrt().item())

    # Build param->name mapping (robust for optimizer.state iteration)
    named_params = list(getattr(model, "module", model).named_parameters(recurse=True))
    param_to_name = {p: n for n, p in named_params}

    # Infer momentum key if not provided
    if optimizer is not None and momentum_key is None:
        # Look at first state entry that has tensor buffers
        inferred = None
        for p, st in optimizer.state.items():
            if not isinstance(st, dict):
                continue
            if "exp_avg" in st:
                inferred = "exp_avg"
                break
            if "moment1" in st:
                inferred = "moment1"
                break
            if "momentum_buffer" in st:
                inferred = "momentum_buffer"
                break
        momentum_key = inferred  # could be None

    # Last-layer selection
    matched_last = []
    def _is_last(name: str) -> bool:
        if last_layer_names is not None:
            return name in set(last_layer_names)
        if last_layer_keywords is not None:
            return any(k in name for k in last_layer_keywords)
        return False

    # Accumulators (local, on GPU)
    dev = next(getattr(model, "module", model).parameters()).device
    g_sumsq = torch.zeros((), device=dev, dtype=torch.float32)
    g_numel = torch.zeros((), device=dev, dtype=torch.float32)
    m_sumsq = torch.zeros((), device=dev, dtype=torch.float32)
    m_numel = torch.zeros((), device=dev, dtype=torch.float32)

    lg_sumsq = torch.zeros((), device=dev, dtype=torch.float32)
    lg_numel = torch.zeros((), device=dev, dtype=torch.float32)
    lm_sumsq = torch.zeros((), device=dev, dtype=torch.float32)
    lm_numel = torch.zeros((), device=dev, dtype=torch.float32)

    # Iterate parameters via optimizer param groups if possible (ensures same params as optimizer)
    params_iter = None
    if optimizer is not None:
        params_iter = []
        for pg in optimizer.param_groups:
            params_iter.extend(pg["params"])
    else:
        params_iter = [p for _, p in named_params]

    for p in params_iter:
        name = param_to_name.get(p, None)
        if name is None:
            continue

        # gradient
        if p.grad is not None:
            g = p.grad.detach()
            g_sumsq += (g.float() ** 2).sum()
            g_numel += float(g.numel())

            if _is_last(name):
                matched_last.append(name)
                lg_sumsq += (g.float() ** 2).sum()
                lg_numel += float(g.numel())

        # momentum
        if optimizer is not None and momentum_key is not None:
            st = optimizer.state.get(p, None)
            if isinstance(st, dict) and (momentum_key in st) and isinstance(st[momentum_key], torch.Tensor):
                m = st[momentum_key].detach()
                m_sumsq += (m.float() ** 2).sum()
                m_numel += float(m.numel())

                if _is_last(name):
                    lm_sumsq += (m.float() ** 2).sum()
                    lm_numel += float(m.numel())

    # global reduce scalars
    g_sumsq = _allreduce_sum(g_sumsq); g_numel = _allreduce_sum(g_numel)
    m_sumsq = _allreduce_sum(m_sumsq); m_numel = _allreduce_sum(m_numel)
    lg_sumsq = _allreduce_sum(lg_sumsq); lg_numel = _allreduce_sum(lg_numel)
    lm_sumsq = _allreduce_sum(lm_sumsq); lm_numel = _allreduce_sum(lm_numel)

    out = {
        "momentum_key": momentum_key,
        "global": {
            "grad": _finish_norm(g_sumsq, g_numel) if g_numel.item() > 0 else 0.0,
            "mom":  _finish_norm(m_sumsq, m_numel) if m_numel.item() > 0 else 0.0,
        },
        "last_layer": {
            "grad": _finish_norm(lg_sumsq, lg_numel) if lg_numel.item() > 0 else 0.0,
            "mom":  _finish_norm(lm_sumsq, lm_numel) if lm_numel.item() > 0 else 0.0,
            "matched": matched_last,
        },
    }
    return out


def log_fsdp_analysis(stats, step, save_dir="analysis_logs", stats_pre=None, plot_every=10, token_freq=None):
    """
    功能：
    1. 将 stats['global']（及 stats_pre['global']）按梯度步追加到 JSONL 文件
    2. 将 stats['last_layer'] 保存为本地 .pt 文件
    3. 每隔 plot_every 个梯度步，从 JSONL 生成趋势图
    注意：只有主进程 (Rank 0) 执行此操作
    """
    if not dist.is_initialized() or dist.get_rank() == 0:

        os.makedirs(save_dir, exist_ok=True)

        # --- A. 按梯度步追加标量到 JSONL ---
        import json
        record = {"grad_step": step}
        for k, v in stats["global"].items():
            record[k] = v
        if stats_pre is not None:
            for k, v in stats_pre["global"].items():
                record[f"pre_clip_{k}"] = v
        jsonl_path = os.path.join(save_dir, "grad_step_metrics.jsonl")
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        # --- B. 本地保存 (Last Layer Vectors) ---
        if "last_layer" in stats and stats["last_layer"]:
            file_path = os.path.join(save_dir, f"layer_stats_step_{step:05d}.pt")
            save_payload = {}
            for k, v in stats["last_layer"].items():
                if k == "shape":
                    save_payload[k] = v
                    continue
                if isinstance(v, list):
                    save_payload[k] = torch.tensor(v, dtype=torch.float32)
                else:
                    save_payload[k] = v
            # Include token frequency from this mini-batch
            if token_freq is not None:
                save_payload["token_freq"] = token_freq.long()
            torch.save(save_payload, file_path)

        # --- B2. Per-layer metrics to JSONL ---
        if "per_layer" in stats and stats["per_layer"]:
            pl_record = {"grad_step": step}
            for k, v in stats["per_layer"].items():
                pl_record[k] = v
            pl_jsonl_path = os.path.join(save_dir, "per_layer_metrics.jsonl")
            with open(pl_jsonl_path, "a") as f:
                f.write(json.dumps(pl_record) + "\n")

        # --- C. 定期绘图 ---
        if step % plot_every == 0:
            _plot_grad_step_metrics(jsonl_path, save_dir)
            pl_jsonl = os.path.join(save_dir, "per_layer_metrics.jsonl")
            if os.path.exists(pl_jsonl):
                _plot_weight_sg_ratio(pl_jsonl, save_dir)
                _plot_per_layer_eff_lr(pl_jsonl, save_dir)
            # Figure 10: token-class SG norm from latest .pt file
            _plot_token_class_sg_norm(step, save_dir)


def _plot_grad_step_metrics(jsonl_path, save_dir):
    """从 JSONL 读取按梯度步记录的标量，绘制趋势图并保存到 save_dir/plots/"""
    import json

    import matplotlib
    matplotlib.use("Agg")  # 无头模式，不需要 GUI
    import matplotlib.pyplot as plt

    # 读取所有记录
    records = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if len(records) < 2:
        return

    steps = [r["grad_step"] for r in records]
    # 收集所有 metric key（除 grad_step 外）
    all_keys = [k for k in records[0] if k != "grad_step"]

    # 按前缀分组绘制子图：
    #   组1: 范数类 (param_norm, grad_norm, mom_norm, pre_clip_*)
    #   组2: 学习率类 (eff_lr_mean, pre_clip_eff_lr_mean)
    norm_keys = [k for k in all_keys if "norm" in k]
    lr_keys = [k for k in all_keys if "lr" in k]

    plot_dir = os.path.join(save_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    def _save_group(keys, filename, title, ylabel, use_log=False):
        if not keys:
            return
        fig, ax = plt.subplots(figsize=(12, 5))
        for k in keys:
            vals = [r.get(k, None) for r in records]
            ax.plot(steps, vals, label=k, linewidth=1.0)
        ax.set_xlabel("Gradient Step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if use_log:
            ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, filename), dpi=150)
        plt.close(fig)

    _save_group(norm_keys, "norms_vs_grad_step.png",
                "Parameter / Gradient / Momentum Norms  (per gradient step)",
                "Norm", use_log=True)

    _save_group(lr_keys, "eff_lr_vs_grad_step.png",
                "Effective Learning Rate  (per gradient step)",
                "Effective LR", use_log=True)

    # 额外画一张 pre-clip vs post-clip grad_norm 对比图
    pre_grad = [r.get("pre_clip_grad_norm", None) for r in records]
    post_grad = [r.get("grad_norm", None) for r in records]
    if pre_grad[0] is not None:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(steps, pre_grad, label="pre-clip grad_norm", linewidth=1.0)
        ax.plot(steps, post_grad, label="post-clip grad_norm", linewidth=1.0)
        ax.set_xlabel("Gradient Step")
        ax.set_ylabel("Grad Norm")
        ax.set_title("Gradient Norm: Pre-Clip vs Post-Clip  (per gradient step)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "grad_clip_comparison.png"), dpi=150)
        plt.close(fig)

    # --- param_norm 变化量图 ---
    # param_norm 绝对值很大但变化极小，直接画看起来像定值
    # 这里画 delta（相邻步差值）和相对变化率，让微小变化可见
    param_norm_vals = [r.get("param_norm", None) for r in records]
    if param_norm_vals[0] is not None and len(param_norm_vals) >= 2:
        deltas = [param_norm_vals[i] - param_norm_vals[i - 1] for i in range(1, len(param_norm_vals))]
        rel_changes = [d / param_norm_vals[i] if param_norm_vals[i] != 0 else 0
                       for i, d in enumerate(deltas)]
        delta_steps = steps[1:]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        ax1.plot(delta_steps, deltas, linewidth=1.0, color="tab:blue")
        ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax1.set_ylabel("Δ param_norm")
        ax1.set_title("param_norm Change per Gradient Step  (absolute delta)")
        ax1.grid(True, alpha=0.3)

        ax2.plot(delta_steps, rel_changes, linewidth=1.0, color="tab:orange")
        ax2.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax2.set_xlabel("Gradient Step")
        ax2.set_ylabel("Δ param_norm / param_norm")
        ax2.set_title("param_norm Relative Change per Gradient Step")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "param_norm_delta.png"), dpi=150)
        plt.close(fig)


def _plot_weight_sg_ratio(pl_jsonl_path, save_dir):
    """Figure 2: Weight-SG Ratio and Weight-Momentum Ratio per layer over training."""
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = []
    with open(pl_jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if len(records) < 2:
        return

    steps = [r["grad_step"] for r in records]
    layer_names = records[0]["layer_names"]
    num_layers = len(layer_names)

    plot_dir = os.path.join(save_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Use colormap for many layers
    cmap = plt.cm.get_cmap("tab20", num_layers)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))

    for i, name in enumerate(layer_names):
        # Weight-SG ratio: ||w||/||grad||
        wsg_ratio = []
        for r in records:
            w = r["w_norm"][i]
            g = r["g_norm"][i]
            wsg_ratio.append(w / g if g > 0 else float('nan'))
        ax1.plot(steps, wsg_ratio, label=name, color=cmap(i), linewidth=1.0, alpha=0.8)

        # Weight-Momentum ratio: ||w||/||m||
        wm_ratio = []
        for r in records:
            w = r["w_norm"][i]
            m = r["m_norm"][i]
            wm_ratio.append(w / m if m > 0 else float('nan'))
        ax2.plot(steps, wm_ratio, label=name, color=cmap(i), linewidth=1.0, alpha=0.8)

    ax1.set_xlabel("Iterations")
    ax1.set_ylabel(r"$\|w\|_F / \|\nabla f\|_F$")
    ax1.set_title("Weight-SG Ratio (Adam)")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=6, ncol=2)

    ax2.set_xlabel("Iterations")
    ax2.set_ylabel(r"$\|w\|_F / \|m\|_F$")
    ax2.set_title("Weight-Momentum Ratio (Adam)")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=6, ncol=2)

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "weight_sg_momentum_ratio.png"), dpi=150)
    plt.close(fig)


def _plot_per_layer_eff_lr(pl_jsonl_path, save_dir):
    """Figure 8: Mean effective learning rate per layer over training."""
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = []
    with open(pl_jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if len(records) < 2:
        return

    steps = [r["grad_step"] for r in records]
    layer_names = records[0]["layer_names"]
    num_layers = len(layer_names)

    plot_dir = os.path.join(save_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    cmap = plt.cm.get_cmap("tab20", num_layers)

    fig, ax = plt.subplots(figsize=(12, 7))
    for i, name in enumerate(layer_names):
        vals = [r["eff_lr_mean"][i] for r in records]
        ax.plot(steps, vals, label=name, color=cmap(i), linewidth=1.0, alpha=0.8)

    ax.set_xlabel("Iteration (t)")
    ax.set_ylabel(r"$\bar{a}^{eff}_{\ell,t}$")
    ax.set_title("Mean Effective Stepsize of Adam")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, ncol=2)

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "per_layer_eff_lr.png"), dpi=150)
    plt.close(fig)


def _plot_token_class_sg_norm(step, save_dir):
    """Per-token-class SG norm scatter: token frequency (x) vs SG norm (y).

    Plots ALL vocab tokens.  Tokens that never appeared in the mini-batch
    (freq=0) are shown at x=0.  Both axes use log scale (freq axis uses
    log(1+count) so zero-count tokens are visible at x=0).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    pt_path = os.path.join(save_dir, f"layer_stats_step_{step:05d}.pt")
    if not os.path.exists(pt_path):
        return

    data = torch.load(pt_path, map_location="cpu", weights_only=True)
    if "g_norm" not in data:
        return

    g_norm = data["g_norm"].numpy()                       # [V]
    vocab_size = len(g_norm)

    # Token frequency: may be shorter than vocab_size (pad with 0)
    if "token_freq" in data:
        freq = data["token_freq"].numpy().astype(np.float64)
        if len(freq) < vocab_size:
            freq = np.pad(freq, (0, vocab_size - len(freq)))
        elif len(freq) > vocab_size:
            freq = freq[:vocab_size]
    else:
        # Fallback: use token ID as x-axis if no frequency data
        freq = np.arange(vocab_size, dtype=np.float64)

    plot_dir = os.path.join(save_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # --- Scatter: frequency vs SG norm (all tokens) ---
    fig, ax = plt.subplots(figsize=(10, 8))

    # Separate tokens that appeared (freq>0) from those that didn't
    appeared = freq > 0
    n_appeared = appeared.sum()
    n_absent = (~appeared).sum()

    # Plot appeared tokens
    if n_appeared > 0:
        ax.scatter(
            freq[appeared], g_norm[appeared],
            s=1, alpha=0.3, color="tab:blue", rasterized=True,
            label=f"appeared ({n_appeared})",
        )
    # Plot absent tokens at x=0 with a different color
    if n_absent > 0:
        # Jitter x slightly so they don't all stack at exactly 0
        ax.scatter(
            np.full(n_absent, 0.5), g_norm[~appeared],
            s=1, alpha=0.1, color="tab:red", rasterized=True,
            label=f"absent ({n_absent})",
        )

    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("log")
    ax.set_xlabel("Token Frequency in Mini-batch")
    ax.set_ylabel(r"SG Norm $\|g_i\|_2$")
    ax.set_title(f"Token Frequency vs SG Norm — Step {step}  (V={vocab_size})")
    ax.legend(fontsize=8, markerscale=5)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"token_freq_vs_sg_norm_step_{step:05d}.png"), dpi=150)
    plt.close(fig)


import math


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

        # --- LM-Head gradient tracker (actor only) ---
        self._lm_head_tracker = None
        if actor_optimizer is not None:
            base = _unwrap_model(actor_module)
            lm_head_mod = getattr(base, "get_output_embeddings", lambda: None)()
            if lm_head_mod is None:
                lm_head_mod = getattr(base, "lm_head", None)
            if lm_head_mod is not None:
                self._lm_head_tracker = LMHeadGradTracker(lm_head_mod)
                if torch.distributed.get_rank() == 0:
                    print(f"[Actor] LMHeadGradTracker registered on {lm_head_mod.__class__.__name__}")

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )

                if is_mask_all_zero:
                    log_probs = log_probs[:0]
                    if calculate_entropy:
                        entropy_rmpad = entropy_rmpad[:0]

                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs

    # def _optimizer_step(self):
    #         assert self.config.grad_clip is not None
            
    #         # 1. Unscale
    #         if self.scaler is not None:
    #             self.scaler.unscale_(self.actor_optimizer)

    #         # -------------------------------------------------------------
    #         # Analysis 1: Pre-Clip (重点关注: 原始梯度分布)
    #         # -------------------------------------------------------------
    #         # 建议设置打印频率，例如每 1 步或 10 步，避免刷屏
    #         debug_log = (dist.get_rank() == 0) # and (self.global_step % 10 == 0)
            
    #         stats_pre = get_fsdp_comprehensive_analysis(
    #             self.actor_module, self.actor_optimizer, rms_norm=False
    #         )
            
    #         if debug_log:
    #             print(f"\n[Step {self.global_step} PRE-CLIP]")
    #             print(f"  Global: Param={stats_pre['global']['param_norm']:.4f}, Grad={stats_pre['global']['grad_norm']:.4f}")
    #             if 'w_norm_top5' in stats_pre['last_layer']:
    #                 ll = stats_pre['last_layer']
    #                 print(f"  LM_Head: Shape={ll.get('shape')}")
    #                 print(f"  LM_Head Grad (Top5 Tokens): {ll['g_norm_top5']}")
    #                 print(f"  LM_Head Mom  (Top5 Tokens): {ll.get('mom_norm_top5', 'N/A')}")
    #                 print(f"  LM_Head EffLR(Top5 Tokens): {ll.get('eff_lr_top5', 'N/A')}")

    #         # -------------------------------------------------------------
    #         # Clipping
    #         # -------------------------------------------------------------
    #         if isinstance(self.actor_module, FSDP):
    #             grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
    #         elif isinstance(self.actor_module, FSDPModule):
    #             grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
    #         else:
    #             grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

    #         if isinstance(grad_norm, DTensor):
    #             grad_norm = grad_norm.full_tensor()

    #         # -------------------------------------------------------------
    #         # Analysis 2: Post-Clip (重点关注: 全局梯度是否被截断)
    #         # -------------------------------------------------------------
    #         # 这里我们只做一个简单的全局检查，节省时间，不再做 LM_Head 分析
    #         # 如果你想做全套，直接调 get_fsdp_comprehensive_analysis 也可以
            
    #         # 简易版：只看 Global Grad
    #         if debug_log:
    #             print(f"[Step {self.global_step} POST-CLIP] Target Clip={self.config.grad_clip}, Actual Norm={float(grad_norm):.4f}")

    #         # -------------------------------------------------------------
    #         # Step
    #         # -------------------------------------------------------------
    #         if self.scaler is not None:
    #             self.scaler.step(self.actor_optimizer)
    #             self.scaler.update()
    #         else:
    #             if not torch.isfinite(grad_norm):
    #                 print(f"WARN: rank {dist.get_rank()} grad_norm is not finite: {grad_norm}")
    #                 self.actor_optimizer.zero_grad()
    #             else:
    #                 self.actor_optimizer.step()
            
    #         # 注意: 你的原代码逻辑似乎没有在 step 后清零梯度，如果外部有 zero_grad 则无视这里
    #         # print("grad_norm_detected:", grad_norm)

    #         # -------------------------------------------------------------
    #         # Analysis 3: Post-Step (重点关注: 参数更新后的状态，如新 Momentum)
    #         # -------------------------------------------------------------
    #         # 训练结束后，参数变了，Momentum 变了。
    #         # 如果资源非常充足，可以再跑一次完整分析看看 update 后的情况
            
    #         stats_after = get_fsdp_comprehensive_analysis(self.actor_module, self.actor_optimizer)
    #         if debug_log:
    #            print(f"[Step {self.global_step} POST-STEP] New Param Norm={stats_after['global']['param_norm']:.4f}")

    #         return grad_norm

    def _optimizer_step(self, token_freq=None):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)

        # --- 分析 1: Pre-Clip ---
        # 此时梯度是原始的，动量是旧的
        stats_pre = get_fsdp_comprehensive_analysis(self.actor_module, self.actor_optimizer)
        if dist.get_rank() == 0:
            print(f"\n[PRE-CLIP] Global Grad Norm: {stats_pre['global']['grad_norm']:.6f}")
            # 如果需要，可以将 stats_pre 传给 wandb.log()

        # --- 梯度裁剪 ---
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # --- 分析 2: Post-Clip ---
        if dist.get_rank() == 0:
            print(f"[POST-CLIP] Grad Norm: {float(grad_norm):.6f} (Target: {self.config.grad_clip})")

        # --- LM-Head tracker: finalize with same clip ratio as optimizer ---
        lm_head_stats = {}
        if self._lm_head_tracker is not None:
            clip_ratio = min(1.0, self.config.grad_clip / max(float(grad_norm), 1e-8))
            lm_head_stats = self._lm_head_tracker.finalize_step(
                optimizer_pg=self.actor_optimizer.param_groups[0],
                clip_ratio=clip_ratio,
            )

        # --- 执行更新 ---
        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            if not torch.isfinite(grad_norm):
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()

        # --- 分析 3: Post-Step ---
        stats_after = get_fsdp_comprehensive_analysis(self.actor_module, self.actor_optimizer)

        # Overwrite last_layer grad/mom/eff_lr with tracker's correct values
        # (keep p_norm from summon_full_params which is correct)
        if lm_head_stats:
            p_norm = stats_after["last_layer"].get("p_norm")
            stats_after["last_layer"].update(lm_head_stats)
            if p_norm is not None:
                stats_after["last_layer"]["p_norm"] = p_norm

            # Inject "lm_head" as a separate entry in per_layer stats.
            # With tie_word_embeddings, named_parameters() only lists the shared
            # param under "embed_tokens" (corrupted).  The tracker gives us the
            # true lm_head gradient/momentum/eff_lr.
            pl = stats_after["per_layer"]
            if "lm_head" not in pl.get("layer_names", []):
                pl["layer_names"].append("lm_head")
                pl["w_norm"].append(pl["w_norm"][0] if pl["w_norm"] else 0.0)  # same weight as embed
                pl["g_norm"].append(lm_head_stats.get("agg_g_norm", 0.0))
                pl["m_norm"].append(lm_head_stats.get("agg_m_norm", 0.0))
                pl["eff_lr_mean"].append(lm_head_stats.get("agg_eff_lr_mean", 0.0))

        if dist.get_rank() == 0:
            print(f"[POST-STEP] Global Momentum Norm: {stats_after['global']['mom_norm']:.6f}")
            print(f"[POST-STEP] Global Mean Effective LR: {stats_after['global']['eff_lr_mean']:.2e}")
            if "eff_lr_cls_mean" in stats_after["last_layer"]:
                print(f"  LM_Head Class 0 EffLR: {stats_after['last_layer']['eff_lr_cls_mean'][0]:.2e}")


        # All-reduce token frequency across ranks so it matches the global gradient
        if token_freq is not None and dist.is_initialized():
            token_freq = token_freq.to(get_device_id())
            dist.all_reduce(token_freq, op=dist.ReduceOp.SUM)
            token_freq = token_freq.cpu()

        current_step=get_optimizer_step(self.actor_optimizer)
        analysis_save_dir = os.path.join(os.environ.get("VERL_DEFAULT_LOCAL_DIR", "./checkpoints"), "analysis_data")
        log_fsdp_analysis(stats_after, current_step, save_dir=analysis_save_dir, stats_pre=stats_pre, token_freq=token_freq)

        # 将 global 标量通过 metrics 管道回传给 trainer，最终写入 wandb
        analysis_metrics = {}
        for k, v in stats_after["global"].items():
            analysis_metrics[f"analysis/{k}"] = v
        # pre-clip 的原始梯度范数也一并记录
        for k, v in stats_pre["global"].items():
            analysis_metrics[f"analysis/pre_clip_{k}"] = v

        return grad_norm, analysis_metrics


    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                # Accumulate token frequency across micro-batches for analysis
                _token_freq_accum = None

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}

                    # Count token occurrences for frequency-vs-SG-norm analysis
                    if "input_ids" in model_inputs:
                        ids = model_inputs["input_ids"].detach().reshape(-1)
                        if _token_freq_accum is None:
                            # lazy init: vocab size unknown, use bincount
                            _token_freq_accum = torch.bincount(ids.cpu().long())
                        else:
                            bc = torch.bincount(ids.cpu().long(), minlength=len(_token_freq_accum))
                            if bc.shape[0] > _token_freq_accum.shape[0]:
                                _token_freq_accum = torch.nn.functional.pad(
                                    _token_freq_accum, (0, bc.shape[0] - _token_freq_accum.shape[0])
                                )
                            elif _token_freq_accum.shape[0] > bc.shape[0]:
                                bc = torch.nn.functional.pad(
                                    bc, (0, _token_freq_accum.shape[0] - bc.shape[0])
                                )
                            _token_freq_accum += bc

                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    # for fully_async_policy recipe
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (any function is expected to return 2 values)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    # Skip if using pure rollout correction mode (metrics already in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "rollout_correction" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm, analysis_metrics = self._optimizer_step(token_freq=_token_freq_accum)
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                mini_batch_metrics.update(analysis_metrics)
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
