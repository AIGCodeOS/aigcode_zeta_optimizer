import math
import torch
from torch.optim.optimizer import Optimizer
from typing import Tuple

def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration to project G to the orthogonal group.
    """
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
    return X.to(G.dtype)

def muon_based_soliton_detection(
    W: torch.Tensor,
    num_solitons: int = 8,
    ns_steps: int = 3,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用 Muon 的 Newton-Schulz 正交化近似 SVD 的孤子检测
    """
    m, n = W.shape
    k = min(num_solitons, min(m, n))
    
    soliton_centers = []
    soliton_strengths = []
    
    residual = W.clone()
    
    for i in range(k):
        ortho = zeropower_via_newtonschulz5(residual, steps=ns_steps, eps=eps)
        
        # 提取主方向 [m, 1]
        center = ortho[:, 0:1]
        
        # 计算强度
        strength = (center.T @ residual).norm()
        
        soliton_centers.append(center)
        soliton_strengths.append(strength)
        
        # 从残差中减去这个成分
        projection = center @ (center.T @ residual)
        residual = residual - projection
        
        if residual.norm() < eps * W.norm():
            break
    
    if soliton_centers:
        soliton_centers = torch.cat(soliton_centers, dim=1)  # [m, k]
        soliton_strengths = torch.stack(soliton_strengths)   # [k]
    else:
        soliton_centers = torch.zeros(m, 1, device=W.device, dtype=W.dtype)
        soliton_strengths = torch.zeros(1, device=W.device, dtype=W.dtype)
    
    return soliton_centers, soliton_strengths

def renormalization_group_projection(
    G: torch.Tensor,
    soliton_centers: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    重整化群投影（RG-Flow）
    """
    # G_proj = U @ (U^T @ G)
    projection = soliton_centers @ (soliton_centers.T @ G)
    return projection

def soliton_coherence_filter(
    G: torch.Tensor,
    W: torch.Tensor,
    threshold: float = 0.1,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    孤子相干性滤波
    """
    inner_product = (G * W).sum()
    g_norm = G.norm() + eps
    w_norm = W.norm() + eps
    coherence = inner_product.abs() / (g_norm * w_norm)
    
    if coherence < threshold:
        suppression = coherence / threshold
        G_filtered = G * suppression
    else:
        G_filtered = G
    
    return G_filtered

class MixMuon(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        betas=(0.9, 0.95),
        nesterov: bool = True,
        ns_steps: int = 5,
        eps: float = 1e-8,
        total_steps: int = 20000,
        warmup_steps: int = 200,
        # Soliton parameters
        use_soliton: bool = True,
        num_solitons: int = 2,
        soliton_ns_steps: int = 5,
        coherence_threshold: float = 0.1,
        rg_projection_strength: float = 0.3,
        **kwargs,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            nesterov=nesterov,
            ns_steps=ns_steps,
            eps=eps,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            use_soliton=use_soliton,
            num_solitons=num_solitons,
            soliton_ns_steps=soliton_ns_steps,
            coherence_threshold=coherence_threshold,
            rg_projection_strength=rg_projection_strength,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            beta1, beta2 = group["betas"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]
            total_steps = group["total_steps"]
            warmup_steps = group["warmup_steps"]
            use_soliton = group.get("use_soliton", True)
            num_solitons = group.get("num_solitons", 2)
            soliton_ns_steps = group.get("soliton_ns_steps", 5)
            coherence_threshold = group.get("coherence_threshold", 0.1)
            rg_projection_strength = group.get("rg_projection_strength", 0.3)

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(p)
                    # AdaMuon specific state
                    if p.ndim >= 2:
                        state["v_buffer"] = torch.zeros(p.numel(), device=p.device, dtype=p.dtype)

                state["step"] += 1
                step_t = state["step"]

                # Calculate mixing coefficient 'a'
                # a = 1.0 during warmup
                # a decays from 1.0 to 0.0 following cosine schedule after warmup
                if step_t <= warmup_steps:
                    a = 1.0
                else:
                    curr_step = step_t - warmup_steps
                    decay_steps = total_steps - warmup_steps
                    if decay_steps > 0:
                        progress = min(curr_step / decay_steps, 1.0)
                        a = 0.5 * (1.0 + math.cos(math.pi * progress))
                    else:
                        a = 0.0
                
                # Apply Weight Decay
                if wd > 0:
                    p.data.mul_(1 - lr * wd)

                # --- Muon Logic for >= 2D params ---
                if p.ndim >= 2:
                    if p.ndim > 2:
                        g_in = g.view(g.size(0), -1)
                    else:
                        g_in = g

                    buf = state["momentum_buffer"]
                    if buf.shape != g_in.shape:
                         buf = buf.view_as(g_in)

                    # Update momentum
                    # Using AdaMuon style: buf.mul_(beta1).add_(g_in)
                    # Note: SignMuon in codebase used beta1 but add(grad) without alpha
                    # We'll stick to standard momentum for consistency
                    buf.mul_(beta1).add_(g_in)

                    # --- SignMuon Path ---
                    # SignMuon in codebase: NS(momentum) -> sign -> update
                    # It does not use Nesterov or sign-before-NS in the file I read.
                    # But to be robust, we use the buffer directly as SignMuon does.
                    
                    # SignMuon Ortho
                    ortho_sign = zeropower_via_newtonschulz5(buf, steps=ns_steps)
                    u_sign = ortho_sign.sign() # Shape [d1, d2]

                    # --- AdaMuon Path ---
                    # AdaMuon uses Nesterov and sign-before-NS usually
                    # In adamuon.py:
                    # update_in = g_in.add(buf, alpha=momentum) if nesterov else buf
                    # sign_update = torch.sign(update_in)
                    # ortho = NS(sign_update)
                    
                    update_in = g_in.add(buf, alpha=beta1) if nesterov else buf
                    sign_update_in = torch.sign(update_in)
                    ortho_ada = zeropower_via_newtonschulz5(sign_update_in, steps=ns_steps)
                    flat_ada = ortho_ada.flatten()

                    # Update v_buffer (AdaMuon specific)
                    if "v_buffer" not in state:
                        state["v_buffer"] = torch.zeros_like(flat_ada)
                    v = state["v_buffer"]
                    # v = v * beta2 + (1-beta2) * flat^2 (AdaMuon logic uses momentum for v too?)
                    # In adamuon.py: v.mul_(momentum).addcmul_(flat, flat, value=(1 - momentum))
                    # Wait, adamuon.py uses 'momentum' for both buffers?
                    # Let's use beta2 for second moment as is standard.
                    v.mul_(beta2).addcmul_(flat_ada, flat_ada, value=(1 - beta2))

                    normed = flat_ada.div(v.sqrt().add(eps))
                    
                    rows, cols = p.shape[:2]
                    scale = 0.2 * math.sqrt(rows * cols) / (normed.norm() + eps)
                    u_ada = (normed * scale).view_as(p)

                    # --- Mix Updates ---
                    # u_sign is shape [rows, cols], u_ada is shape [rows, cols] (reshaped)
                    if p.ndim > 2:
                        u_sign = u_sign.view_as(p)
                    
                    # Combine: a * u_sign + (1-a) * u_ada
                    # Note: u_sign magnitude is ~1.0, u_ada magnitude is ~0.2*sqrt(dim)
                    # This discrepancy might be intentional? 
                    # User requested: a * signmuon + (1-a) * adamuon
                    # We perform exactly that.
                    
                    final_update = u_sign.mul(a).add_(u_ada, alpha=(1 - a))

                    # --- Soliton Logic Integration ---
                    if use_soliton:
                        # 1. 孤子检测 (从当前权重中提取)
                        # 需要确保 W 是 2D 矩阵
                        W_2d = p.data.view(rows, cols)
                        soliton_centers, _ = muon_based_soliton_detection(
                            W_2d, num_solitons=num_solitons, ns_steps=soliton_ns_steps, eps=eps
                        )
                        
                        # 2. 孤子相干性滤波
                        # 作用于混合后的梯度/更新量
                        final_update_2d = final_update.view(rows, cols)
                        filtered_update = soliton_coherence_filter(
                            final_update_2d, W_2d, threshold=coherence_threshold, eps=eps
                        )
                        
                        # 3. 重整化群投影 (RG-Flow)
                        projected_update = renormalization_group_projection(
                            filtered_update, soliton_centers, eps=eps
                        )
                        
                        # 混合原始更新与投影后的更新
                        final_update_2d = (1 - rg_projection_strength) * filtered_update + \
                                         rg_projection_strength * projected_update
                        
                        final_update = final_update_2d.view_as(p)
                    
                    p.data.add_(final_update, alpha=-lr)

                else:
                    # Fallback for 1D params (bias, etc.)
                    # Usually standard Adam or SGD
                    # We'll implement a simple Adam-like update
                    state = self.state[p]
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    
                    exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                    
                    denom = exp_avg_sq.sqrt().add_(eps)
                    step_size = lr
                    
                    # Bias correction
                    # step_size = step_size * math.sqrt(1 - beta2 ** step_t) / (1 - beta1 ** step_t)
                    
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
