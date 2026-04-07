# Zeta with Muon-based Soliton Detection
# 
# 核心创新：使用 Muon 的 Newton-Schulz 正交化替代 SVD
# - 保留孤子检测特性
# - 速度接近原始 Muon（~5秒/step）
# - 避免昂贵的 SVD 计算

import math
import torch
import torch.optim as optim
from typing import Optional, Tuple
from collections import deque

try:
    import torch_npu
    HAS_NPU = True
except ImportError:
    HAS_NPU = False


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Newton-Schulz 迭代求零次幂（正交化）"""
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if G.dtype != torch.bfloat16 else G
    
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
    
    核心思想：
    - SVD 给出主方向（奇异向量）
    - Newton-Schulz 给出正交方向
    - 两者在主成分空间中等价
    
    优势：
    - 速度快：O(k × m × n) vs SVD 的 O(min(m,n) × m × n)
    - 保留主要特性：捕捉主要的孤子模式
    
    Args:
        W: 权重矩阵 [m, n]
        num_solitons: 孤子数量
        ns_steps: Newton-Schulz 迭代步数
        eps: 数值稳定性
    
    Returns:
        (孤子中心, 孤子强度)
    """
    m, n = W.shape
    k = min(num_solitons, min(m, n))
    
    # 使用 Newton-Schulz 正交化提取主方向
    # 这比 SVD 快得多，但能捕捉主要模式
    
    soliton_centers = []
    soliton_strengths = []
    
    # 残差矩阵（逐步减去已提取的成分）
    residual = W.clone()
    
    for i in range(k):
        # 对残差应用 Newton-Schulz 正交化
        # 这会给出当前最主要的方向
        ortho = zeropower_via_newtonschulz5(residual, steps=ns_steps, eps=eps)
        
        # 提取主方向
        # ortho 的形状与 residual 相同: [m, n]
        # 我们需要提取一个 [m, 1] 的向量作为孤子中心
        if m >= n:
            # 高矩阵: 使用第一列
            center = ortho[:, 0:1]  # [m, 1]
        else:
            # 宽矩阵: 使用第一行的转置
            center = ortho[0:1, :].T  # [n, 1] -> 但我们需要 [m, 1]
            # 实际上对于宽矩阵，我们应该在转置空间工作
            # 或者直接使用第一列（即使 n < m）
            center = ortho[:, 0:1]  # [m, 1]
        
        # 计算强度（投影长度）
        # residual: [m, n], center: [m, 1]
        # 正确的投影: center.T @ residual 得到 [1, n]，然后求范数
        strength = (center.T @ residual).norm()
        
        soliton_centers.append(center)
        soliton_strengths.append(strength)
        
        # 从残差中减去这个成分
        # center: [m, 1], center.T @ residual: [1, n]
        # center @ (center.T @ residual): [m, 1] @ [1, n] = [m, n]
        projection = center @ (center.T @ residual)
        residual = residual - projection
        
        # 如果残差太小，提前停止
        if residual.norm() < eps * W.norm():
            break
    
    # 拼接结果
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
    
    将梯度投影到孤子空间，过滤掉无效的微观抖动
    """
    # 投影到孤子空间
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
    
    只保留与当前权重结构相干的梯度分量
    """
    # 计算梯度与权重的相干性
    inner_product = (G * W).sum()
    g_norm = G.norm() + eps
    w_norm = W.norm() + eps
    coherence = inner_product.abs() / (g_norm * w_norm)
    
    # 如果相干性低于阈值，抑制梯度
    if coherence < threshold:
        suppression = coherence / threshold
        G_filtered = G * suppression
    else:
        G_filtered = G
    
    return G_filtered


class ZetaMuonSoliton(optim.Optimizer):
    
    def __init__(
        self,
        params,
        lr: float = 1.5e-3,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        momentum: float = 0.95,
        ns_steps: int = 5,
        hessian_update_freq: int = 10,
        gamma: float = 0.05,
        # 孤子参数
        use_soliton: bool = True,
        num_solitons: int = 2,  # 只用 1 个孤子
        soliton_ns_steps: int = 5,  # 只用 1 步迭代
        coherence_threshold: float = 0.1,
        rg_projection_strength: float = 0.1,
        # 轻量级特性
        use_light_torsion: bool = False,
        torsion_strength: float = 0.2,
        use_short_memory: bool = False,
        memory_length: int = 10,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            ns_steps=ns_steps,
            hessian_update_freq=hessian_update_freq,
            gamma=gamma,
            use_soliton=use_soliton,
            num_solitons=num_solitons,
            soliton_ns_steps=soliton_ns_steps,
            coherence_threshold=coherence_threshold,
            rg_projection_strength=rg_projection_strength,
            use_light_torsion=use_light_torsion,
            torsion_strength=torsion_strength,
            use_short_memory=use_short_memory,
            memory_length=memory_length,
        )
        super().__init__(params, defaults)
        
        self.global_step = 0
        print("[Zeta-Muon-Soliton] 使用 Muon 风格的孤子检测 - 目标: ~5秒/step")
    
    @torch.no_grad()
    def step(self, closure=None):
        """执行单步优化"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        self.global_step += 1
        
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            momentum = group['momentum']
            ns_steps = group['ns_steps']
            hessian_freq = group['hessian_update_freq']
            gamma = group['gamma']
            
            use_soliton = group['use_soliton']
            num_solitons = group['num_solitons']
            soliton_ns_steps = group['soliton_ns_steps']
            coherence_threshold = group['coherence_threshold']
            rg_projection_strength = group['rg_projection_strength']
            
            use_light_torsion = group['use_light_torsion']
            torsion_strength = group['torsion_strength']
            use_short_memory = group['use_short_memory']
            memory_length = group['memory_length']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # ========================================
                # 2D+ 参数: Zeta-Muon-Soliton
                # ========================================
                if p.ndim >= 2:
                    if p.ndim > 2:
                        grad_2d = grad.view(grad.size(0), -1)
                        p_2d = p.view(p.size(0), -1)
                    else:
                        grad_2d = grad
                        p_2d = p
                    
                    # 初始化状态
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(grad_2d)
                        state["hessian"] = torch.ones_like(grad_2d)
                        state["step"] = 0
                        if use_short_memory:
                            state["grad_history"] = deque(maxlen=memory_length)
                        state["soliton_centers"] = None
                        state["soliton_strengths"] = None
                    
                    state["step"] += 1
                    exp_avg = state["exp_avg"]
                    hessian = state["hessian"]
                    
                    # 动量更新
                    exp_avg.mul_(momentum).add_(grad_2d)
                    
                    # Hessian 估计（Sophia 风格）
                    if state["step"] % hessian_freq == 0:
                        hessian.mul_(1 - gamma).addcmul_(grad_2d, grad_2d, value=gamma)
                    
                    # Sophia 预处理
                    precond_avg = exp_avg / (hessian.sqrt() + eps)
                    
                    # 记录历史（如果启用）
                    if use_short_memory:
                        state["grad_history"].append(grad_2d.clone())
                    
                    # ========================================
                    # Newton-Schulz 正交化（Muon 核心）
                    # ========================================
                    sign_update = torch.sign(precond_avg)
                    ortho = zeropower_via_newtonschulz5(sign_update, steps=ns_steps, eps=eps)
                    
                    a, b = p_2d.shape
                    base_scale = 0.2 * math.sqrt(a * b) / (ortho.norm() + eps)
                    base_direction = ortho * base_scale
                    
                    # ========================================
                    # Muon 风格的孤子检测（替代 SVD）
                    # ========================================
                    soliton_correction = torch.zeros_like(base_direction)
                    
                    if use_soliton:
                        W = p_2d
                        G = base_direction
                        
                        # 使用 Muon 的 Newton-Schulz 检测孤子
                        soliton_centers, soliton_strengths = muon_based_soliton_detection(
                            W,
                            num_solitons=num_solitons,
                            ns_steps=soliton_ns_steps,
                            eps=eps
                        )
                        state["soliton_centers"] = soliton_centers
                        state["soliton_strengths"] = soliton_strengths
                        
                        # 重整化群投影
                        G_rg = renormalization_group_projection(G, soliton_centers, eps)
                        
                        # 相干性滤波
                        G_filtered = soliton_coherence_filter(G_rg, W, coherence_threshold, eps)
                        
                        # 孤子修正
                        soliton_correction = rg_projection_strength * (G_filtered - G)
        
                    unified_direction = (
                        base_direction +
                        soliton_correction 
                    )
                    
                    # 权重衰减 + 更新
                    if wd > 0:
                        p.data.mul_(1 - lr * wd)
                    p.data.add_(unified_direction.view_as(p), alpha=-lr)
                
                # ========================================
                # 1D 参数: 标准 Muon
                # ========================================
                else:
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                    
                    state["step"] += 1
                    exp_avg = state["exp_avg"]
                    
                    exp_avg.mul_(momentum).add_(grad)
                    
                    norm = exp_avg.norm() + eps
                    normalized_update = exp_avg / norm
                    
                    if wd > 0:
                        p.data.mul_(1 - lr * wd)
                    
                    p.data.add_(normalized_update, alpha=-lr)
        
        return loss


# 别名：方便导入
Zeta = ZetaMuonSoliton
