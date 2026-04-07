# Copyright (c) 2025, AIGCode CORPORATION. All rights reserved. 
# Zeta Optimizer: Simplified Muon-style 2D update + AdamW 1D update

import math
import torch
import torch.optim as optim

try:
    import torch_npu
    HAS_NPU = True
except ImportError:
    HAS_NPU = False


@torch.jit.script
def fast_ns_iteration(X: torch.Tensor, steps: int) -> torch.Tensor:
    """Optimized Newton-Schulz iteration with JIT and in-place operations."""
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        A = X @ X.T
        B = A.mul(b).addmm(A, A, alpha=c)
        X = X.mul(a).addmm(B, X)
    return X


class ZetaHSPro(optim.Optimizer):
    """
    Zeta-HS-Pro: Simplified version for ablation.
    - 2D parameters: Muon-style Newton-Schulz orthogonalization after Adam preconditioning.
    - 1D parameters: Standard AdamW update.
    """
    def __init__(
        self,
        params,
        lr: float = 1.5e-3,           
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        ns_steps: int = 5,
        warmup_steps: int = 2000,
        alpha_base: float = 0.1,
        beta_base: float = 0.2,
    ):
        defaults = dict(
            lr=lr, weight_decay=weight_decay, betas=betas, eps=eps,
            ns_steps=ns_steps, warmup_steps=warmup_steps,
            alpha_base=alpha_base, beta_base=beta_base
        )
        super().__init__(params, defaults)
        self.global_step = 0
        self.stats = {}  # Monitoring dictionary
        print("[Zeta-HS-Pro] Simplified version loaded.")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.global_step += 1
        self.stats = {}  # Reset stats for each step
        
        # Temporary variables for aggregation
        total_2d_params = 0
        sum_2d_gamma = 0.0
        sum_2d_combined_norm = 0.0
        sum_2d_update_norm = 0.0
        sum_2d_moonshot_scale = 0.0
        
        total_1d_params = 0
        sum_1d_exp_avg_norm = 0.0
        sum_1d_step_size = 0.0

        for group in self.param_groups:
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None: continue
                g = p.grad.float()

                if p.ndim >= 2:
                    m, n = p.shape[0], p.numel() // p.shape[0]
                    g_in = g.view(m, n)
                    
                    state = self.state[p]
                    if "exp_avg" not in state or state["exp_avg"].shape != g_in.shape:
                        state["exp_avg"] = torch.zeros_like(g_in)
                        state["exp_avg_sq"] = torch.zeros_like(g_in)
                    
                    # Adam preconditioning
                    state["exp_avg"].mul_(beta1).add_(g_in, alpha=1 - beta1)
                    state["exp_avg_sq"].mul_(beta2).addcmul_(g_in, g_in, value=1 - beta2)
                    
                    m_t = state["exp_avg"]
                    v_t = state["exp_avg_sq"]

                    
                    precond_grad = m_t / (v_t.sqrt() + eps)
                    
                    # Polar decomposition (Muon style)
                    X = precond_grad.bfloat16() if precond_grad.dtype != torch.bfloat16 else precond_grad
                    transposed = False
                    if m > n:
                        X = X.T
                        transposed = True
                        
                    X = X / (X.norm() + eps)
                    Q = fast_ns_iteration(X, steps=group["ns_steps"])
                    
                    if transposed:
                        Q = Q.T
                    
                    combined = Q.to(p.dtype)
                    
                    # Adaptive dimension scaling
                    moonshot_scale = 0.2 * math.sqrt(max(m, n)) 
                    update = (combined * moonshot_scale).view_as(p)

                    # Monitoring
                    if torch.distributed.get_rank() == 0:
                        total_2d_params += 1
                        sum_2d_gamma += 1.0
                        sum_2d_combined_norm += combined.norm().item()
                        sum_2d_update_norm += update.norm().item()
                        sum_2d_moonshot_scale += moonshot_scale.item() if isinstance(moonshot_scale, torch.Tensor) else moonshot_scale

                    if wd > 0: p.data.mul_(1 - lr * wd)
                    p.data.add_(update, alpha=-lr)
                    
                else:
                    # 1D Path: Standard AdamW (Sync with AdaMuon)
                    state = self.state[p]
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(g)
                        state["exp_avg_sq"] = torch.zeros_like(g)
                    
                    state["step"] += 1
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    
                    exp_avg.lerp_(g, 1 - beta1)
                    exp_avg_sq.lerp_(g.pow(2), 1 - beta2)
                    
                    denom = exp_avg_sq.sqrt().add_(eps)
                    bc1 = 1 - beta1 ** state["step"]
                    bc2 = 1 - beta2 ** state["step"]
                    step_size = lr * math.sqrt(bc2) / bc1

                    # Monitoring
                    if torch.distributed.get_rank() == 0:
                        total_1d_params += 1
                        sum_1d_exp_avg_norm += exp_avg.norm().item()
                        sum_1d_step_size += step_size
                    
                    if wd > 0:
                        p.data.mul_(1 - lr * wd)
                    
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

        # Aggregate stats
        if torch.distributed.get_rank() == 0:
            if total_2d_params > 0:
                self.stats.update({
                    'zeta/2d_count': total_2d_params,
                    'zeta/2d_sum_combined_norm': sum_2d_combined_norm,
                    'zeta/2d_sum_update_norm': sum_2d_update_norm,
                    'zeta/2d_sum_moonshot_scale': sum_2d_moonshot_scale,
                    'zeta/2d_avg_combined_norm': sum_2d_combined_norm / total_2d_params,
                    'zeta/2d_avg_update_norm': sum_2d_update_norm / total_2d_params,
                    'zeta/2d_avg_moonshot_scale': sum_2d_moonshot_scale / total_2d_params,
                })
            if total_1d_params > 0:
                self.stats.update({
                    'zeta/1d_count': total_1d_params,
                    'zeta/1d_sum_step_size': sum_1d_step_size,
                    'zeta/1d_sum_exp_avg_norm': sum_1d_exp_avg_norm,
                    'zeta/1d_avg_step_size': sum_1d_step_size / total_1d_params,
                    'zeta/1d_avg_exp_avg_norm': sum_1d_exp_avg_norm / total_1d_params,
                })

        return loss


Zeta = ZetaHSPro
