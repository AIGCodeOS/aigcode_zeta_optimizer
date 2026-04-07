# megatron/core/optimizer/adamuon_qwen3.py
"""
AdaMuon for Qwen3: 1D/2D参数冲突修复版
基于数学分析的优化配置
"""

import math
import torch
import torch.optim as optim
from typing import List, Dict, Any, Tuple
    
class AdaMuonParamType(optim.Optimizer):
    """
    AdaMuon优化器 (1d/2d分离版)
    
    内容:
    1. 1D/2D参数学习率分离 (基于Hessian条件数)
    2. RMSNorm参数禁用权重衰减
    3. Warmup阶段1D参数额外压制
    4. 嵌入矩阵按2D处理
    """
    
    def __init__(
        self,
        params,
        lr: float = 1.5e-4,
        lr_1d_ratio: float = 0.1,        # 1D参数学习率比例
        weight_decay: float = 0.01,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        ns_steps: int = 5,
        warmup_steps: int = 2500,
        warmup_1d_extra_ratio: float = 0.5,  # Warmup期间1D额外压制
    ):
        defaults = dict(
            lr=lr,
            lr_1d_ratio=lr_1d_ratio,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            ns_steps=ns_steps,
            warmup_steps=warmup_steps,
            warmup_1d_extra_ratio=warmup_1d_extra_ratio,
        )
        super().__init__(params, defaults)
        self.global_step = 0
        print(f"[AdaMuon-Qwen3] 1D/2D冲突修复版加载. lr_1d_ratio={lr_1d_ratio}")

    def _get_effective_lr(self, group, param_type: str) -> float:
        """计算有效学习率（考虑1D/2D分离和Warmup）"""
        base_lr = group['lr']
        
        if param_type == 'norm_1d':
            lr = base_lr * group['lr_1d_ratio']
            # Warmup期间额外压制
            if self.global_step < group['warmup_steps']:
                lr *= group['warmup_1d_extra_ratio']
        else:
            lr = base_lr
            
        return lr

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.global_step += 1

        for group in self.param_groups:
            eps = group['eps']
            beta1, beta2 = group['betas']
            ns_steps = group['ns_steps']
            base_wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                # 参数类型识别
                if hasattr(p, 'param_type'):
                    param_type = p.param_type
                elif p.ndim >= 2:
                    param_type = 'matrix_2d'
                elif 'norm' in p.name.lower() or 'gamma' in p.name.lower():
                    param_type = 'norm_1d'
                else:
                    param_type = 'other_1d'

                # 获取有效学习率和权重衰减
                lr = self._get_effective_lr(group, param_type)
                wd = 0.0 if param_type == 'norm_1d' else base_wd

                g = p.grad.float()
                state = self.state[p]

                # ========================================================
                # 2D矩阵参数：正交化更新
                # ========================================================
                if p.ndim >= 2:
                    m, n = p.shape[0], p.numel() // p.shape[0]
                    g_in = g.view(m, n)

                    if 'exp_avg' not in state:
                        state['exp_avg'] = torch.zeros_like(g_in)
                        state['exp_avg_sq'] = torch.zeros_like(g_in)

                    # 动量更新
                    state['exp_avg'].mul_(beta1).add_(g_in, alpha=1 - beta1)
                    
                    # Sign稳定化正交更新 (AdaMuon核心)
                    M = state['exp_avg']
                    sign_M = torch.sign(M)
                    
                    # zeropower_via_newtonschulz5
                    a, b, c = (3.4445, -4.7750, 2.0315)
                    X = sign_M.bfloat16()
                    if sign_M.size(0) > sign_M.size(1):
                        X = X.T
                    X = X / (X.norm() + 1e-7)
                    for _ in range(ns_steps):
                        A = X @ X.T
                        B = b * A + c * (A @ A)
                        X = a * X + B @ X
                    if sign_M.size(0) > sign_M.size(1):
                        X = X.T

                    O = X  # 正交更新方向
                    
                    # RMS对齐缩放
                    target_rms = 0.2 * math.sqrt(m * n)
                    scale = target_rms / (O.norm() + eps)
                    update = (O * scale).view_as(p)

                # ========================================================
                # 1D参数：标准Adam (学习率已分离)
                # ========================================================
                else:
                    if 'step' not in state:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)

                    state['step'] += 1
                    state['exp_avg'].lerp_(g, 1 - beta1)
                    state['exp_avg_sq'].lerp_(g.pow(2), 1 - beta2)

                    bc1 = 1.0 - beta1 ** state['step']
                    bc2 = 1.0 - beta2 ** state['step']
                    m_hat = state['exp_avg'] / bc1
                    v_hat = state['exp_avg_sq'] / bc2

                    update = m_hat / (v_hat.sqrt() + eps)

                # 应用更新
                if wd > 0:
                    p.data.mul_(1 - lr * wd)
                p.data.add_(update, alpha=-lr)

        return loss


def create_adamuon_param_groups(model: torch.nn.Module) -> List[Dict]:
    """
    为ParamType优化创建的参数组
    
    基于数学分析:
    - RMSNorm γ: lr×0.1, weight_decay=0
    - 权重矩阵: lr×1.0, weight_decay=0.01
    - 嵌入矩阵: lr×1.0, weight_decay=0.01 (按2D处理)
    """
    norm_params = []
    weight_params = []
    embed_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        param.name = name  # 保存名称供优化器使用
        
        if 'norm' in name.lower() and param.ndim == 1:
            norm_params.append(param)
            param.param_type = 'norm_1d'
        elif 'embed' in name.lower() or 'token' in name.lower():
            embed_params.append(param)
            param.param_type = 'embed_2d'
        else:
            weight_params.append(param)
            param.param_type = 'matrix_2d'
    
    param_groups = [
        {'params': weight_params, 'lr': 1.5e-4, 'weight_decay': 0.01},
        {'params': norm_params, 'lr': 1.5e-5, 'weight_decay': 0.0},  # 关键修复
        {'params': embed_params, 'lr': 1.5e-4, 'weight_decay': 0.01},
    ]
    
    print(f"[AdaMuon-ParamType] 创建参数组:")
    print(f"  - 权重矩阵: {len(weight_params)} ({sum(p.numel() for p in weight_params)/1e9:.2f}B)")
    print(f"  - RMSNorm γ: {len(norm_params)} ({sum(p.numel() for p in norm_params)/1e6:.2f}M)")
    print(f"  - 嵌入矩阵: {len(embed_params)} ({sum(p.numel() for p in embed_params)/1e9:.2f}B)")
    
    return param_groups


# 别名
AdaMuon = AdaMuonParamType