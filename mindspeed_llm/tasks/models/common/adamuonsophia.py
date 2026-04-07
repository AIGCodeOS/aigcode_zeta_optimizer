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


class AdamuonSophia(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.02,
        weight_decay: float = 0.01,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        eps: float = 1e-8,
        rho: float = 0.08,
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
            rho=rho,
            betas=betas,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def update_hessian(self):
        for group in self.param_groups:
            _, beta2 = group["betas"]
            for p in group["params"]:
                grad = getattr(p, "decoupled_grad", None)
                if grad is None:
                    grad = p.grad
                if grad is None:
                    continue
                state = self.state[p]
                if "hessian" not in state:
                    state["hessian"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["hessian"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

    @torch.no_grad()
    def step(self, closure=None, bs=5120):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            current_bs = group.get("bs", bs)
            lr = group["lr"]
            wd = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]
            rho = group["rho"]
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

                    if wd > 0:
                        p.data.mul_(1 - lr * wd)
                    p.data.add_(update, alpha=-lr)
                else:
                    state = self.state[p]
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["hessian"] = torch.zeros_like(p)
                    exp_avg, hess = state["exp_avg"], state["hessian"]
                    exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                    ratio = (exp_avg.abs() / (rho * current_bs * hess + 1e-15)).clamp(None, 1)
                    if wd > 0:
                        p.data.mul_(1 - lr * wd)
                    p.data.addcmul_(exp_avg.sign(), ratio, value=-lr)

        return loss
