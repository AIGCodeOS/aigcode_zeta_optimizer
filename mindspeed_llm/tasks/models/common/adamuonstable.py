import math
import torch
from torch.optim.optimizer import Optimizer

def zeropower_via_newtonschulz5_stable(G, steps=5, eps=1e-7):
    """
    针对 FP32 优化的 Newton-Schulz 迭代，确保输出矩阵的正交性。
    """
    assert len(G.shape) == 2
    a, b = G.shape
    transposed = False
    if a < b:
        G = G.T
        transposed = True
    
    # 强制在 FP32 下进行正交化运算
    X = G.to(torch.float32)
    norm_val = X.norm() + eps
    X.div_(norm_val)
    
    for _ in range(steps):
        XTX = X.T @ X
        eye = torch.eye(XTX.size(0), device=X.device, dtype=torch.float32)
        # 5阶 Newton-Schulz 公式
        temp = 26 * eye - 15 * XTX
        temp2 = 13 * eye - XTX @ temp
        X = X @ (0.25 * temp2)
    
    if transposed:
        X = X.T
    return X.to(G.dtype)


class AdaMuonStable(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.02,
        weight_decay: float = 0.01,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        eps: float = 1e-8,
        betas=(0.9, 0.95),
        rho: float = 0.05,
        update_freq: int = 20,
        **kwargs,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            eps=eps,
            betas=betas,
            rho=rho,
            update_freq=update_freq,
        )
        super().__init__(params, defaults)
        self.step_count = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.step_count += 1
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]
            beta1, beta2 = group.get("betas", (0.9, 0.95))
            rho = group.get("rho", 0.05)
            freq = group.get("update_freq", 20)

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                if p.ndim >= 2:
                    # Zeta-VPMS v1.2 style update
                    shape = p.shape
                    if p.ndim > 2:
                        g_2d = g.view(shape[0], -1)
                    else:
                        g_2d = g

                    state = self.state[p]
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(g_2d) # Independent Momentum Buffer
                        state["h_row"] = torch.ones(g_2d.shape[0], device=p.device)
                        state["h_col"] = torch.ones(g_2d.shape[1], device=p.device)

                    # 1. Structured Preprocessing (Kronecker-Diagonal SOAP)
                    if self.step_count % freq == 0:
                        state["h_row"].lerp_((g_2d * g_2d).mean(dim=1), 1 - beta2)
                        state["h_col"].lerp_((g_2d * g_2d).mean(dim=0), 1 - beta2)

                    # 2. Preconditioned Gradient Calculation
                    # Row-Col scaling to approximate full matrix curvature
                    g_pre = g_2d / (state["h_row"].view(-1, 1).sqrt() + rho)
                    g_pre /= (state["h_col"].view(1, -1).sqrt() + rho)

                    # 3. Momentum Update
                    m = state["exp_avg"]
                    m.mul_(momentum).add_(g_pre, alpha=1 - momentum)

                    # 4. Muon Orthogonalization
                    # No sign() preprocessing, preserving precise spectral info of momentum
                    update_dir = zeropower_via_newtonschulz5_stable(m, steps=ns_steps)

                    # 5. AdamUon Scaling (Matrix-level Adaptive Scaling)
                    a, b = update_dir.shape
                    scale = 0.2 * math.sqrt(max(a, b))

                    # 6. Execute Update
                    if wd > 0:
                        p.data.mul_(1 - lr * wd)
                    
                    # update_dir is in FP32, cast back to p.dtype
                    p.data.add_(update_dir.view_as(p).to(p.dtype), alpha=-lr * scale)
                else:
                    # Standard AdamW for 1D params
                    state = self.state[p]
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    
                    m, v = state["exp_avg"], state["exp_avg_sq"]
                    state["step"] += 1
                    
                    m.mul_(momentum).add_(g, alpha=1 - momentum)
                    v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                    
                    denom = v.sqrt().add_(eps)
                    
                    # Bias correction
                    step_size = lr
                    # step_size = step_size * math.sqrt(1 - beta2 ** state["step"]) / (1 - momentum ** state["step"])
                    
                    if wd > 0:
                        p.data.mul_(1 - lr * wd)
                        
                    p.data.addcdiv_(m, denom, value=-step_size)

        return loss
