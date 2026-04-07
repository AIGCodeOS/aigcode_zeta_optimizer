# # Copyright (c) 2025, AIGCode CORPORATION. All rights reserved. 
# # @author: chenqiuwu@aigcode.net

# # Copyright (c) 2025, AIGCode CORPORATION. All rights reserved. 
# # @author: chenqiuwu@aigcode.net

# import math
# from re import S
# import torch
# import torch.optim as optim
# import torch.nn.functional as F
# from typing import List

# def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
#     """
#     Zeta-MM improved Newton-Schulz iteration with eps added to prevent division by zero,
#     maintaining numerical stability in bfloat16.

#     Args:
#         G: Input matrix
#         steps: Number of Newton-Schulz iterations
#         eps: Small epsilon for numerical stability
#     """
#     assert len(G.shape) == 2

#     a, b, c = (3.4445, -4.7750, 2.0315)
#     X = G.bfloat16()

#     if G.size(0) > G.size(1):
#         X = X.T

#     # Spectral normalization (the essence of Muon): force eigenvalues to be close to 1
#     norm = X.norm() + eps
#     X = X / norm

#     for _ in range(steps):
#         A = X @ X.T
#         B = b * A + c * A @ A
#         X = a * X + B @ X

#     if G.size(0) > G.size(1):
#         X = X.T

#     return X

# class SophiaMuon(optim.Optimizer):
#     def __init__(self, params, lr=1e-4, betas=(0.965, 0.99), rho=0.08,
#                  weight_decay=1e-2, *, maximize: bool = False,
#                  capturable: bool = False, ns_steps=5, eps: float = 1e-8):
#         if not 0.0 <= lr:
#             raise ValueError("Invalid learning rate: {}".format(lr))
#         if not 0.0 <= betas[0] < 1.0:
#             raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
#         if not 0.0 <= betas[1] < 1.0:
#             raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
#         if not 0.0 <= rho:
#             raise ValueError("Invalid rho parameter: {}".format(rho))
#         if not 0.0 <= weight_decay:
#             raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
#         defaults = dict(lr=lr, betas=betas, rho=rho,
#                         weight_decay=weight_decay,
#                         maximize=maximize, capturable=capturable,
#                         ns_steps=ns_steps, eps=eps)
#         super(SophiaMuon, self).__init__(params, defaults)

#     def __setstate__(self, state):
#         super().__setstate__(state)
#         for group in self.param_groups:
#             group.setdefault('maximize', False)
#             group.setdefault('capturable', False)
#         state_values = list(self.state.values())
#         step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
#         if not step_is_tensor:
#             for s in state_values:
#                 s['step'] = torch.tensor(float(s['step']))

#     @torch.no_grad()
#     def update_hessian(self):
#         total_hess_sum = 0.0
#         total_hess_count = 0
#         device = None

#         for group in self.param_groups:
#             beta1, beta2 = group['betas']
#             for p in group['params']:
#                 grad = getattr(p, 'decoupled_grad', None)
#                 if grad is None:
#                     grad = p.grad
#                 if grad is None:
#                     continue
#                 state = self.state[p]
#                 if len(state) == 0:
#                     state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
#                         if self.defaults['capturable'] else torch.tensor(0.)
#                     state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
#                     state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)
#                 if 'hessian' not in state.keys():
#                     state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)
#                 state['hessian'].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

#     @torch.no_grad()
#     def step(self, closure=None, bs=5120):
#         loss = None
#         if closure is not None:
#             with torch.enable_grad():
#                 loss = closure()

#         for group in self.param_groups:

#             current_bs = group.get('bs', bs)
            
#             params_with_grad = []
#             grads = []
#             exp_avgs = []
#             state_steps = []
#             hessian = []
#             beta1, beta2 = group['betas']

#             for p in group['params']:
#                 if p.grad is None:
#                     continue
#                 params_with_grad.append(p)
#                 if p.grad.is_sparse:
#                     raise RuntimeError('SophiaMuon does not support sparse gradients')
#                 grads.append(p.grad)
#                 state = self.state[p]
#                 if len(state) == 0:
#                     state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
#                         if self.defaults['capturable'] else torch.tensor(0.)
#                     state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
#                     state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)
#                 if 'hessian' not in state.keys():
#                     state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)
#                 exp_avgs.append(state['exp_avg'])
#                 state_steps.append(state['step'])
#                 hessian.append(state['hessian'])
#                 if self.defaults['capturable']:
#                     current_bs = torch.ones((1,), dtype=torch.float, device=p.device) * current_bs

#             sophiamuon(params_with_grad,
#                     grads,
#                     exp_avgs,
#                     hessian,
#                     state_steps,
#                     bs=current_bs,
#                     beta1=beta1,
#                     beta2=beta2,
#                     rho=group['rho'],
#                     lr=group['lr'],
#                     weight_decay=group['weight_decay'],
#                     maximize=group['maximize'],
#                     capturable=group['capturable'],
#                     ns_steps=group['ns_steps'],
#                     eps=group['eps'])

#         return loss


# def sophiamuon(params: List[torch.Tensor],
#             grads: List[torch.Tensor],
#             exp_avgs: List[torch.Tensor],
#             hessian: List[torch.Tensor],
#             state_steps: List[torch.Tensor],
#             capturable: bool = False,
#             *,
#             bs: int,
#             beta1: float,
#             beta2: float,
#             rho: float,
#             lr: float,
#             weight_decay: float,
#             maximize: bool,
#             ns_steps: int,
#             eps: float):

#     if not all(isinstance(t, torch.Tensor) for t in state_steps):
#         raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

#     _single_tensor_sophiamuon(params,
#                            grads,
#                            exp_avgs,
#                            hessian,
#                            state_steps,
#                            bs=bs,
#                            beta1=beta1,
#                            beta2=beta2,
#                            rho=rho,
#                            lr=lr,
#                            weight_decay=weight_decay,
#                            maximize=maximize,
#                            capturable=capturable,
#                            ns_steps=ns_steps,
#                            eps=eps)


# def _single_tensor_sophiamuon(params: List[torch.Tensor],
#                            grads: List[torch.Tensor],
#                            exp_avgs: List[torch.Tensor],
#                            hessian: List[torch.Tensor],
#                            state_steps: List[torch.Tensor],
#                            *,
#                            bs: int,
#                            beta1: float,
#                            beta2: float,
#                            rho: float,
#                            lr: float,
#                            weight_decay: float,
#                            maximize: bool,
#                            capturable: bool,
#                            ns_steps: int,
#                            eps: float):

#     for i, param in enumerate(params):
#         grad = grads[i] if not maximize else -grads[i]
#         exp_avg = exp_avgs[i]
#         hess = hessian[i]
#         step_t = state_steps[i]

#         if capturable:
#             assert param.is_cuda and step_t.is_cuda and bs.is_cuda

#         if torch.is_complex(param):
#             grad = torch.view_as_real(grad)
#             exp_avg = torch.view_as_real(exp_avg)
#             hess = torch.view_as_real(hess)
#             param = torch.view_as_real(param)

#         step_t += 1
#         param.mul_(1 - lr * weight_decay)
        
#         # Momentum update
#         exp_avg.mul_(beta1).add_(grad)
        
#         # --- Muon integration start ---
#         # Apply Newton-Schulz to exp_avg (momentum) if it's a matrix (>=2D)
#         # Muon usually works on >1D parameters.
#         if param.ndim >= 2:
#             m_in = exp_avg
#             if param.ndim > 2:
#                 m_in = m_in.view(m_in.size(0), -1)
            
#             # Apply Newton-Schulz
#             ortho_direction = zeropower_via_newtonschulz5(m_in, steps=ns_steps)
            
#             # Restore shape
#             if param.ndim > 2:
#                 ortho_direction = ortho_direction.view_as(exp_avg)
                
            
#             a, b = param.shape[:2]
#             scale = 0.2 * math.sqrt(a * b)
#             denom = (bs * hess * rho + 1e-15)
#             norm = (ortho_direction / denom).norm()
#             # ratio = (ortho_direction.abs() / denom) * scale / (norm + 1e-15)
            
#             if capturable:
#                 step_size = lr
#                 step_size_neg = step_size.neg()
#                 # param.addcmul_(ortho_direction.sign(), ratio, value=step_size_neg) 
#                 # ortho_direction is already "signed" and continuous, not just -1/1.
#                 # So we should probably just multiply:
#                 # param.addcmul_(ortho_direction, 1.0/denom (clipped)?)
                
#                 # Let's look at Sophia original again:
#                 # param.addcmul_(exp_avg.sign(), ratio, value=-lr)
#                 # = param - lr * sign(m) * clip(|m|/h, 1)
#                 # = param - lr * clip(m/h, 1)  (if we ignore sign/abs split)
                
#                 # So for Muon:
#                 # param - lr * clip(ortho_direction / h, 1)
#                 norm = (ortho_direction / denom).norm()
#                 update_val = (ortho_direction / denom).clamp(min=-1.0, max=1.0)
#                 param.add_(update_val, alpha=step_size_neg)
                
#             else:
#                 step_size_neg = -lr
#                 # update_val = (ortho_direction / denom).clamp(min=-1.0, max=1.0)
#                 # param.add_(update_val, alpha=step_size_neg)
                
#                 # To be safer with memory format and operations:
#                 raw_val = ortho_direction.div(denom)
                
#                 # Statistic
#                 # clamped_mask = raw_val.abs() > 1.0
#                 # n_clamped = clamped_mask.sum().item()
#                 # n_total = raw_val.numel()
#                 # total_clamped += n_clamped
#                 # total_numel += n_total
                
#                 update_val = raw_val.clamp(min=-1.0, max=1.0)
#                 param.add_(update_val, alpha=step_size_neg)

#         else:
#             # Fallback to Adam for 1D params (bias, layernorm)
#             # hessian here corresponds to exp_avg_sq (v_t) in Adam
            
#             # bias correction
#             denom = (rho * hess + 1e-15)
#             ratio = (exp_avg.abs() / denom).clamp(None, 1)

#             if capturable:
#                 step_size = lr
#                 step_size_neg = step_size.neg()
#                 param.addcmul_(exp_avg.sign(), ratio, value=step_size_neg)
#             else:
#                 step_size_neg = -lr
#                 param.addcmul_(exp_avg.sign(), ratio, value=step_size_neg)

    # if not capturable and total_numel > 0 and total_stat_count > 0:
    #     print(f"Muon updates: {total_clamped}/{total_numel} (clamped_mask mean: {total_clamped/total_numel:.4f}, average raw magnitude: {total_raw_val_sum / total_numel:.4f}, average bs: {total_bs_sum / total_stat_count:.4f}, average hess: {total_hess_mean_sum / total_stat_count:.6f})")


# Zeta-Sophia: 改进的 Sophia-Muon 融合优化器
# 解决方向信息丢失和后期效果下滑问题
#
# 核心改进：
# 1. 方向信息保持：裁切前保存相对比例
# 2. 自适应裁切策略：早期放大，后期精确
# 3. Hessian自适应缩放：处理值偏小问题
# 4. 阶段性优化：早期Muon主导，后期Sophia精调

import math
import torch
import torch.optim as optim
from typing import List, Optional


def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz 迭代求零次幂（正交化）
    
    Args:
        G: 输入矩阵
        steps: 迭代步数
        eps: 数值稳定性常数
    
    Returns:
        正交化后的矩阵
    """
    assert len(G.shape) == 2
    
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if G.dtype != torch.bfloat16 else G
    
    if G.size(0) > G.size(1):
        X = X.T
    
    # 谱归一化
    norm = X.norm() + eps
    X = X / norm
    
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    
    if G.size(0) > G.size(1):
        X = X.T
    
    return X.to(G.dtype)


class SophiaMuon(optim.Optimizer):
    """
    Zeta-Sophia: 改进的 Sophia-Muon 融合优化器
    
    核心改进：
    1. 方向保持裁切（Direction-Preserving Clipping）
    2. 自适应Hessian缩放（Adaptive Hessian Scaling）
    3. 阶段性优化策略（Phase-based Optimization）
    4. 动态rho调整（Dynamic Rho Adjustment）
    
    Args:
        params: 模型参数
        lr: 学习率 (default: 3e-4)
        betas: Adam风格的beta系数 (default: (0.965, 0.99))
        rho: Sophia裁切阈值 (default: 0.08)
        weight_decay: 权重衰减 (default: 0.01)
        ns_steps: Newton-Schulz迭代步数 (default: 5)
        eps: 数值稳定性常数 (default: 1e-8)
        # 新增参数
        use_direction_preserving: 是否使用方向保持裁切 (default: True)
        hessian_scale_factor: Hessian缩放因子 (default: 1.0)
        early_phase_steps: 早期阶段步数 (default: 10000)
        late_phase_steps: 后期阶段步数 (default: None, 自动计算)
        muon_weight_early: 早期Muon权重 (default: 1.5)
        muon_weight_late: 后期Muon权重 (default: 0.5)
        rho_min: 最小rho值 (default: 0.04)
        rho_max: 最大rho值 (default: 0.12)
    """
    
    def __init__(
        self,
        params,
        lr: float = 3e-4,
        betas=(0.965, 0.99),
        rho: float = 0.08,
        weight_decay: float = 0.01,
        ns_steps: int = 5,
        eps: float = 1e-8,
        maximize: bool = False,
        capturable: bool = False,
        # 新增参数
        use_direction_preserving: bool = True,
        hessian_scale_factor: float = 1.0,
        early_phase_steps: int = 10000,
        late_phase_steps: Optional[int] = None,
        muon_weight_early: float = 1.5,
        muon_weight_late: float = 0.5,
        rho_min: float = 0.04,
        rho_max: float = 0.12,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= rho:
            raise ValueError(f"Invalid rho parameter: {rho}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            rho=rho,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            eps=eps,
            maximize=maximize,
            capturable=capturable,
            use_direction_preserving=use_direction_preserving,
            hessian_scale_factor=hessian_scale_factor,
            early_phase_steps=early_phase_steps,
            late_phase_steps=late_phase_steps or early_phase_steps * 2,
            muon_weight_early=muon_weight_early,
            muon_weight_late=muon_weight_late,
            rho_min=rho_min,
            rho_max=rho_max,
        )
        super().__init__(params, defaults)
        
        self.global_step = 0
    
    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('maximize', False)
            group.setdefault('capturable', False)
            group.setdefault('use_direction_preserving', True)
        
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))
    
    @torch.no_grad()
    def update_hessian(self):
        """更新Hessian估计"""
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                        if group['capturable'] else torch.tensor(0.)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                if 'hessian' not in state:
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                # EMA更新Hessian
                state['hessian'].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
    
    def _get_phase_weights(self, step: int, group: dict) -> tuple:
        """
        获取当前阶段的权重
        
        Returns:
            (muon_weight, rho_adjusted)
        """
        early_steps = group['early_phase_steps']
        late_steps = group['late_phase_steps']
        muon_early = group['muon_weight_early']
        muon_late = group['muon_weight_late']
        rho_min = group['rho_min']
        rho_max = group['rho_max']
        base_rho = group['rho']
        
        if step < early_steps:
            # 早期阶段：放大Muon，增大rho
            progress = step / early_steps
            muon_weight = muon_early
            rho_adjusted = rho_max
        elif step < late_steps:
            # 过渡阶段：线性插值
            progress = (step - early_steps) / (late_steps - early_steps)
            muon_weight = muon_early + (muon_late - muon_early) * progress
            rho_adjusted = rho_max + (base_rho - rho_max) * progress
        else:
            # 后期阶段：精确优化
            muon_weight = muon_late
            rho_adjusted = rho_min
        
        return muon_weight, rho_adjusted
    
    @torch.no_grad()
    def step(self, closure=None, bs=5120):
        """执行单步优化"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        self.global_step += 1
        
        for group in self.param_groups:
            current_bs = group.get('bs', bs)
            
            params_with_grad = []
            grads = []
            exp_avgs = []
            state_steps = []
            hessians = []
            beta1, beta2 = group['betas']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                params_with_grad.append(p)
                
                if p.grad.is_sparse:
                    raise RuntimeError('ZetaSophia does not support sparse gradients')
                
                grads.append(p.grad)
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                        if group['capturable'] else torch.tensor(0.)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                if 'hessian' not in state:
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                exp_avgs.append(state['exp_avg'])
                state_steps.append(state['step'])
                hessians.append(state['hessian'])
                
                if group['capturable']:
                    current_bs = torch.ones((1,), dtype=torch.float, device=p.device) * current_bs
            
            # 获取阶段权重
            muon_weight, rho_adjusted = self._get_phase_weights(self.global_step, group)
            
            zeta_sophia_update(
                params_with_grad,
                grads,
                exp_avgs,
                hessians,
                state_steps,
                bs=current_bs,
                beta1=beta1,
                beta2=beta2,
                rho=rho_adjusted,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                maximize=group['maximize'],
                capturable=group['capturable'],
                ns_steps=group['ns_steps'],
                eps=group['eps'],
                use_direction_preserving=group['use_direction_preserving'],
                hessian_scale_factor=group['hessian_scale_factor'],
                muon_weight=muon_weight,
                global_step=self.global_step,
            )
        
        return loss


def zeta_sophia_update(
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    exp_avgs: List[torch.Tensor],
    hessians: List[torch.Tensor],
    state_steps: List[torch.Tensor],
    *,
    bs: int,
    beta1: float,
    beta2: float,
    rho: float,
    lr: float,
    weight_decay: float,
    maximize: bool,
    capturable: bool,
    ns_steps: int,
    eps: float,
    use_direction_preserving: bool,
    hessian_scale_factor: float,
    muon_weight: float,
    global_step: int,
):
    """Zeta-Sophia更新逻辑"""
    
    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError("state_steps must contain a list of singleton tensors")
    
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        hess = hessians[i]
        step_t = state_steps[i]
        
        if capturable:
            assert param.is_cuda and step_t.is_cuda
        
        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            hess = torch.view_as_real(hess)
            param = torch.view_as_real(param)
        
        step_t += 1
        
        # 权重衰减
        param.mul_(1 - lr * weight_decay)
        
        # 动量更新
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        
        # ========================================
        # 2D参数：Zeta-Sophia融合更新
        # ========================================
        if param.ndim >= 2:
            m_in = exp_avg
            if param.ndim > 2:
                m_in = m_in.view(m_in.size(0), -1)
            
            # Step 1: Muon正交化
            ortho_direction = zeropower_via_newtonschulz5(m_in, steps=ns_steps, eps=eps)
            
            if param.ndim > 2:
                ortho_direction = ortho_direction.view_as(exp_avg)
            
            # Step 2: Hessian自适应缩放
            # 处理Hessian值偏小的问题
            hess_scaled = hess * hessian_scale_factor
            
            # 添加最小值防止除零
            hess_safe = hess_scaled + eps
            
            # Step 3: 计算Sophia分母
            denom = rho * bs * hess_safe
            
            # Step 4: 方向保持裁切
            if use_direction_preserving:
                # 保存原始方向的相对比例
                # 关键改进：在裁切前保存方向信息
                
                # 计算未裁切的更新量
                raw_update = ortho_direction / denom
                
                # 计算每个元素的缩放因子
                # 如果 |raw_update| > 1，需要裁切
                # 但我们要保持相对比例
                
                # 方法1：全局缩放（保持所有元素的相对比例）
                max_abs = raw_update.abs().max()
                if max_abs > 1.0:
                    # 全局缩放，保持比例
                    scale_factor = 1.0 / max_abs
                    update_val = raw_update * scale_factor
                else:
                    update_val = raw_update
                
                # 应用Muon权重
                update_val = update_val * muon_weight
                
            else:
                # 原始Sophia裁切（会丢失方向信息）
                ratio = (ortho_direction.abs() / denom).clamp(None, 1)
                update_val = ortho_direction.sign() * ratio * muon_weight
            
            # Step 5: 应用更新
            if capturable:
                step_size_neg = lr.neg()
            else:
                step_size_neg = -lr
            
            param.add_(update_val, alpha=step_size_neg)
        
        # ========================================
        # 1D参数：标准Sophia更新
        # ========================================
        else:
            hess_safe = hess * hessian_scale_factor + eps
            denom = rho * bs * hess_safe
            
            if use_direction_preserving:
                # 1D也使用方向保持
                raw_update = exp_avg / denom
                max_abs = raw_update.abs().max()
                if max_abs > 1.0:
                    update_val = raw_update / max_abs
                else:
                    update_val = raw_update
            else:
                ratio = (exp_avg.abs() / denom).clamp(None, 1)
                update_val = exp_avg.sign() * ratio
            
            if capturable:
                step_size_neg = lr.neg()
            else:
                step_size_neg = -lr
            
            param.add_(update_val, alpha=step_size_neg)
