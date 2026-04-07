import math
import torch
from torch.optim.optimizer import Optimizer


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


class AdaMuon(Optimizer):
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
        )
        super().__init__(params, defaults)
        self.stats = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.stats = {}
        # 统计汇总变量
        total_2d_params = 0
        sum_2d_update_norm = 0.0
        sum_2d_scale = 0.0
        sum_2d_v_norm = 0.0
        sum_2d_normed_norm = 0.0
        
        total_1d_params = 0
        sum_1d_step_size = 0.0
        sum_1d_exp_avg_norm = 0.0

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]
            beta1, beta2 = group.get("betas", (0.9, 0.95))

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                if p.ndim >= 2:
                    if p.ndim > 2:
                        g_in = g.view(g.size(0), -1)
                    else:
                        g_in = g

                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g_in)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g_in)
                    update_in = g_in.add(buf, alpha=momentum) if nesterov else buf

                    sign_update = torch.sign(update_in)
                    ortho = zeropower_via_newtonschulz5(sign_update, steps=ns_steps)
                    flat = ortho.flatten()

                    if "v_buffer" not in state:
                        state["v_buffer"] = torch.zeros_like(flat)
                    v = state["v_buffer"]
                    v.mul_(momentum).addcmul_(flat, flat, value=(1 - momentum))
                    normed = flat.div(v.sqrt().add(eps))

                    a, b = p.shape[:2]
                    scale = 0.2 * math.sqrt(a * b) / (normed.norm() + eps)
                    update = (normed * scale).view_as(p)

                    # --- 监控记录 ---
                    if torch.distributed.get_rank() == 0:
                        total_2d_params += 1
                        sum_2d_update_norm += update.norm().item()
                        sum_2d_scale += scale.item() if isinstance(scale, torch.Tensor) else scale
                        sum_2d_v_norm += v.norm().item()
                        sum_2d_normed_norm += normed.norm().item()

                    if wd > 0:
                        p.data.mul_(1 - lr * wd)
                    p.data.add_(update, alpha=-lr)
                else:
                    state = self.state[p]
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] += 1
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    exp_avg.lerp_(g, 1 - beta1)
                    exp_avg_sq.lerp_(g.pow(2), 1 - beta2)
                    denom = exp_avg_sq.sqrt().add_(eps)
                    bc1 = 1 - beta1 ** state["step"]
                    bc2 = 1 - beta2 ** state["step"]
                    step_size = lr * math.sqrt(bc2) / bc1

                    # --- 监控记录 ---
                    if torch.distributed.get_rank() == 0:
                        total_1d_params += 1
                        sum_1d_step_size += step_size
                        sum_1d_exp_avg_norm += exp_avg.norm().item()

                    if wd > 0:
                        p.data.mul_(1 - lr * wd)
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

        # --- 汇总监控数据到 self.stats ---
        if torch.distributed.get_rank() == 0:
            if total_2d_params > 0:
                self.stats.update({
                    'adamuon/2d_count': total_2d_params,
                    'adamuon/2d_sum_update_norm': sum_2d_update_norm,
                    'adamuon/2d_sum_scale': sum_2d_scale,
                    'adamuon/2d_sum_v_norm': sum_2d_v_norm,
                    'adamuon/2d_sum_normed_norm': sum_2d_normed_norm,
                    'adamuon/2d_avg_update_norm': sum_2d_update_norm / total_2d_params,
                    'adamuon/2d_avg_scale': sum_2d_scale / total_2d_params,
                    'adamuon/2d_avg_v_norm': sum_2d_v_norm / total_2d_params,
                    'adamuon/2d_avg_normed_norm': sum_2d_normed_norm / total_2d_params,
                })
            if total_1d_params > 0:
                self.stats.update({
                    'adamuon/1d_count': total_1d_params,
                    'adamuon/1d_sum_step_size': sum_1d_step_size,
                    'adamuon/1d_sum_exp_avg_norm': sum_1d_exp_avg_norm,
                    'adamuon/1d_avg_step_size': sum_1d_step_size / total_1d_params,
                    'adamuon/1d_avg_exp_avg_norm': sum_1d_exp_avg_norm / total_1d_params,
                })

        return loss

