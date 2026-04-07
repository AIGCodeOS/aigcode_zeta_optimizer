# Zeta V11: Quantum Zeta - Ultimate Fusion Optimizer
# 融合 V8 (Hamiltonian) + V9 (Tunneling) + V10 (Plasma)
# 
# 核心思想：
# - V8: 哈密顿谱流（基础框架）
# - V9: 瞬子隧穿（逃离 Plateau）
# - V10: 辛等离子体约束（稳定训练）
# - V7: 收敛区优化（Strang Splitting + EMA）
# 
# 物理原理：
# - 哈密顿力学：能量守恒、辛积分
# - 量子场论：瞬子隧穿、虚时间演化
# - 等离子体物理：磁约束、洛伦兹力
# 
# 数学框架：
# - 哈密顿量：H = ||m/h|| (Sophia 能量)
# - 隧穿相位：θ ∈ [0, π/2] (实时间 → 虚时间)
# - 辛磁场：B = ∇ × A (Cayley 生成元)
# - 洛伦兹力：F = M × B (轨迹弯曲)
# 
# 模式切换：
# 1. Normal Mode: V8 哈密顿谱流
# 2. Tunneling Mode: V9 瞬子隧穿（Plateau 检测）
# 3. Confinement Mode: V10 等离子体约束（梯度不稳定）
# 4. Convergence Mode: V7 Strang Splitting（收敛区）

import math
import torch
import torch.optim as optim
try:
    import torch_npu
    HAS_NPU = True
except ImportError:
    HAS_NPU = False


def safe_linalg_solve(A, B):
    """NPU-safe linalg.solve with fallback."""
    if HAS_NPU and hasattr(torch_npu, 'npu_linear_solve'):
        try:
            return torch_npu.npu_linear_solve(A, B)
        except:
            pass
    return torch.linalg.solve(A.cpu(), B.cpu()).to(A.device)


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
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
    
    # 处理非方阵
    if G.size(0) > G.size(1):
        X = X.T
    
    # 归一化
    X = X / (X.norm() + eps)
    
    # Newton-Schulz 迭代
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    
    # 恢复形状
    if G.size(0) > G.size(1):
        X = X.T
    
    return X.to(G.dtype)



def soft_newton_schulz(G: torch.Tensor, steps: int = 5, spectral_filter: float = 0.1, eps: float = 1e-7) -> torch.Tensor:
    """
    Soft Newton-Schulz: 带谱滤波的正交化
    
    核心思想：
    - 标准 NS 在所有特征值上均匀正交化
    - Soft NS 只在高信噪比子空间正交化
    - 低信噪比维度被软抑制（spectral filtering）
    
    Args:
        G: 输入矩阵
        steps: 迭代步数
        spectral_filter: 谱滤波阈值 (0-1)
        eps: 数值稳定性常数
    
    Returns:
        软正交化后的矩阵
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if G.dtype != torch.bfloat16 else G
    
    # 处理非方阵
    if G.size(0) > G.size(1):
        X = X.T
    
    # 归一化
    norm = X.norm() + eps
    X = X / norm
    
    # Soft Newton-Schulz 迭代
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X_new = a * X + B @ X
        
        # 谱滤波：混合原始方向和正交化方向
        # 高能量维度 → 完全正交化
        # 低能量维度 → 保留原始方向
        X = (1 - spectral_filter) * X_new + spectral_filter * X
    
    # 恢复形状
    if G.size(0) > G.size(1):
        X = X.T
    
    return X.to(G.dtype)

class ZetaV11(optim.Optimizer):
    """
    Zeta V11: Quantum Zeta - Ultimate Fusion Optimizer
    
    核心创新：融合三大物理理论
    
    1. Hamiltonian Spectral Flow (V8)
       - Sophia 质量矩阵
       - Muon 相空间流
       - Cayley 辛积分
    
    2. Instanton Tunneling (V9)
       - Plateau 检测
       - 虚时间演化
       - 复数更新
    
    3. Symplectic Plasma Confinement (V10)
       - 辛磁场构造
       - 洛伦兹力修正
       - 自适应约束
    
    模式切换策略：
    - Normal: 正常训练（V8）
    - Tunneling: Plateau 逃离（V9）
    - Confinement: 梯度稳定（V10）
    - Convergence: 收敛优化（V7）
    
    Args:
        params: 模型参数
        lr: 学习率 (default: 3e-4)
        betas: Adam 风格的 beta 系数 (default: (0.9, 0.95))
        eps: 数值稳定性常数 (default: 1e-8)
        weight_decay: 权重衰减系数 (default: 0.01)
        momentum: 动量系数 (default: 0.95)
        # V8 参数
        ns_steps: Newton-Schulz 迭代步数 (default: 5)
        hessian_update_freq: Hessian 更新频率 (default: 10)
        gamma: Hessian 更新率 (default: 0.01)
        use_soft_ns: 是否使用 Soft Newton-Schulz (default: True)
        spectral_filter: 谱滤波强度 (default: 0.1)
        curvature_ema: 曲率指数移动平均系数 (default: 0.9)
        # V9 参数
        use_tunneling: 是否启用隧穿 (default: True)
        tunneling_threshold: 隧穿触发阈值 (default: 1e-4)
        tunneling_steps: 隧穿持续步数 (default: 10)
        phase_angle_max: 最大相位角 (default: π/2)
        # V10 参数
        use_plasma_confinement: 是否启用等离子体约束 (default: True)
        magnetic_strength_base: 基础磁场强度 (default: 0.1)
        magnetic_adaptive: 是否自适应磁场强度 (default: True)
        confinement_threshold: 约束触发阈值 (default: 1.0)
        # V7 参数
        use_cayley: 是否对方阵使用 Cayley 变换 (default: True)
        use_strang: 是否使用 Strang Splitting (default: True)
        use_ema: 是否使用 EMA 权重平滑 (default: True)
        ema_decay: EMA 衰减系数 (default: 0.9999)
        trust_region: 初始信任域 (default: 0.1)
        trust_decay_start: 信任域开始衰减的步数比例 (default: 0.8)
        convergence_threshold: 收敛区检测阈值 (default: 0.8)
        total_steps: 总训练步数 (default: None)
    """
    
    def __init__(
        self,
        params,
        lr: float = 3e-4,
        betas=(0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        momentum: float = 0.95,
        # V8 参数
        ns_steps: int = 5,
        hessian_update_freq: int = 10,
        gamma: float = 0.01,
        use_soft_ns: bool = True,
        spectral_filter: float = 0.1,
        curvature_ema: float = 0.9,
        # V9 参数
        use_tunneling: bool = True,
        tunneling_threshold: float = 1e-4,
        tunneling_steps: int = 10,
        phase_angle_max: float = math.pi / 2,
        # V10 参数
        use_plasma_confinement: bool = True,
        magnetic_strength_base: float = 0.1,
        magnetic_adaptive: bool = True,
        confinement_threshold: float = 1.0,
        # V7 参数
        use_cayley: bool = True,
        use_strang: bool = True,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        trust_region: float = 0.1,
        trust_decay_start: float = 0.8,
        convergence_threshold: float = 0.8,
        total_steps: int = None,
    ):
        # --- Override with global args if available ---
        try:
            from megatron.training.global_vars import get_args
            args = get_args()
            if args:
                if hasattr(args, 'zeta_use_tunneling'): use_tunneling = args.zeta_use_tunneling
                if hasattr(args, 'zeta_use_plasma_confinement'): use_plasma_confinement = args.zeta_use_plasma_confinement
                if hasattr(args, 'zeta_use_soft_ns'): use_soft_ns = args.zeta_use_soft_ns
                if hasattr(args, 'zeta_use_cayley'): use_cayley = args.zeta_use_cayley
                if hasattr(args, 'zeta_use_strang'): use_strang = args.zeta_use_strang
                if hasattr(args, 'zeta_use_ema'): use_ema = args.zeta_use_ema
        except ImportError:
            pass
        # -----------------------------------------------

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            ns_steps=ns_steps,
            hessian_update_freq=hessian_update_freq,
            gamma=gamma,
            use_soft_ns=use_soft_ns,
            spectral_filter=spectral_filter,
            curvature_ema=curvature_ema,
            use_tunneling=use_tunneling,
            tunneling_threshold=tunneling_threshold,
            tunneling_steps=tunneling_steps,
            phase_angle_max=phase_angle_max,
            use_plasma_confinement=use_plasma_confinement,
            magnetic_strength_base=magnetic_strength_base,
            magnetic_adaptive=magnetic_adaptive,
            confinement_threshold=confinement_threshold,
            use_cayley=use_cayley,
            use_strang=use_strang,
            use_ema=use_ema,
            ema_decay=ema_decay,
            trust_region=trust_region,
            trust_decay_start=trust_decay_start,
            convergence_threshold=convergence_threshold,
            total_steps=total_steps,
        )
        super().__init__(params, defaults)
        
        # --- Debug Logging Start ---
        try:
            from megatron.training.global_vars import get_args
            args = get_args()
            rank = getattr(args, 'rank', 0)
            if rank == 0:
                print(f"\n[ZetaV11] Initialized with global args:")
                print(f"  zeta_use_tunneling: {getattr(args, 'zeta_use_tunneling', 'N/A')}")
                print(f"  zeta_use_plasma_confinement: {getattr(args, 'zeta_use_plasma_confinement', 'N/A')}")
                print(f"  zeta_use_soft_ns: {getattr(args, 'zeta_use_soft_ns', 'N/A')}")
                print(f"  zeta_use_cayley: {getattr(args, 'zeta_use_cayley', 'N/A')}")
                print(f"  zeta_use_strang: {getattr(args, 'zeta_use_strang', 'N/A')}")
                print(f"  zeta_use_ema: {getattr(args, 'zeta_use_ema', 'N/A')}")
                
                print(f"[ZetaV11] Internal defaults (param_groups[0]):")
                if self.param_groups:
                    pg = self.param_groups[0]
                    print(f"  use_tunneling: {pg.get('use_tunneling')}")
                    print(f"  use_plasma_confinement: {pg.get('use_plasma_confinement')}")
                    print(f"  use_soft_ns: {pg.get('use_soft_ns')}")
                    print(f"  use_cayley: {pg.get('use_cayley')}")
        except Exception as e:
            print(f"[ZetaV11] Warning: Failed to log init details: {e}")
        # --- Debug Logging End ---
        
        self.global_step = 0
        self.loss_history = []
        self.tunneling_mode = False
        self.tunneling_counter = 0
        self.grad_norm_history = []
        
        if use_ema:
            self._init_ema_params()
    
    def _init_ema_params(self):
        """初始化 EMA 影子参数"""
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.state[p]['ema_param'] = p.data.clone().detach()
    
    def _detect_plateau(self, group):
        """检测 Loss 平台期（触发隧穿）"""
        if len(self.loss_history) < 20:
            return False
        
        recent_losses = self.loss_history[-20:]
        loss_std = torch.tensor(recent_losses).std().item()
        
        threshold = group['tunneling_threshold']
        return loss_std < threshold
    
    def _detect_gradient_instability(self):
        """检测梯度不稳定（触发等离子体约束）"""
        if len(self.grad_norm_history) < 10:
            return False
        
        recent_norms = self.grad_norm_history[-10:]
        norm_std = torch.tensor(recent_norms).std().item()
        norm_mean = torch.tensor(recent_norms).mean().item()
        
        # 变异系数 > 0.5 认为不稳定
        cv = norm_std / (norm_mean + 1e-8)
        return cv > 0.5
    
    def _compute_phase_angle(self, group):
        """计算当前相位角（实时间 vs 虚时间）"""
        if not self.tunneling_mode:
            return 0.0
        
        progress = self.tunneling_counter / group['tunneling_steps']
        max_angle = group['phase_angle_max']
        
        # 使用 sin 曲线平滑过渡
        return max_angle * math.sin(progress * math.pi / 2)
    
    def _is_convergence_zone(self, group):
        """检测是否进入收敛区"""
        total_steps = group.get('total_steps')
        if total_steps is None:
            return False
        
        threshold = group['convergence_threshold']
        return self.global_step >= total_steps * threshold
    
    def _get_dynamic_trust_region(self, group):
        """计算动态信任域"""
        base_trust = group['trust_region']
        total_steps = group.get('total_steps')
        
        if total_steps is None:
            return base_trust
        
        decay_start = group['trust_decay_start']
        
        if self.global_step < total_steps * decay_start:
            return base_trust
        
        progress = (self.global_step - total_steps * decay_start) / (total_steps * (1 - decay_start))
        decay_factor = max(0.1, 1.0 - progress)
        
        return base_trust * decay_factor
    
    def _construct_symplectic_magnetic_field(self, W, G, eps=1e-8):
        """构造辛磁场 B = ∇ × A"""
        A = G @ W.T - W @ G.T
        A_norm = A.norm() + eps
        B = A / A_norm
        return B
    
    def _compute_lorentz_force(self, W, M, B, eps=1e-8):
        """计算洛伦兹力 F = M × B"""
        F = M @ B - B @ M
        F_norm = F.norm() + eps
        F = F / F_norm
        return F
    
    def _adaptive_magnetic_strength(self, grad_norm, threshold, base_strength):
        """自适应磁场强度"""
        normalized_norm = grad_norm / (threshold + 1e-8)
        strength = base_strength * torch.tanh(normalized_norm)
        return strength.item()
    
    @torch.no_grad()
    def step(self, closure=None):
        """执行单步优化（Quantum Zeta 融合算法）"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
            if loss is not None:
                self.loss_history.append(loss.item())
                if len(self.loss_history) > 100:
                    self.loss_history.pop(0)
        
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
            use_soft_ns = group['use_soft_ns']
            spectral_filter = group['spectral_filter']
            curvature_ema = group['curvature_ema']
            use_tunneling = group['use_tunneling']
            use_plasma = group['use_plasma_confinement']
            mag_base = group['magnetic_strength_base']
            mag_adaptive = group['magnetic_adaptive']
            conf_threshold = group['confinement_threshold']
            use_cayley = group['use_cayley']
            use_strang = group['use_strang']
            use_ema = group['use_ema']
            ema_decay = group['ema_decay']
            
            # 动态信任域
            current_trust = self._get_dynamic_trust_region(group)
            
            # 检测模式
            in_convergence_zone = self._is_convergence_zone(group)
            grad_unstable = self._detect_gradient_instability()
            
            # 隧穿模式切换
            if use_tunneling and not self.tunneling_mode:
                if self._detect_plateau(group):
                    self.tunneling_mode = True
                    self.tunneling_counter = 0
                    print(f"[Quantum Zeta] Entering Tunneling Mode at step {self.global_step}")
            
            # 计算相位角
            phase_angle = self._compute_phase_angle(group)
            
            # 隧穿计数
            if self.tunneling_mode:
                self.tunneling_counter += 1
                if self.tunneling_counter >= group['tunneling_steps']:
                    self.tunneling_mode = False
                    print(f"[Quantum Zeta] Exiting Tunneling Mode at step {self.global_step}")
            
            # 收敛区增强谱滤波
            if in_convergence_zone:
                spectral_filter = min(spectral_filter * 1.5, 0.3)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # 判断是否为方阵
                is_square = (p.ndim == 2 and p.shape[0] == p.shape[1])
                
                # ========================================
                # 2D+ 参数: Quantum Zeta Fusion
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
                        state["row_curvature"] = torch.tensor(1.0, device=p.device, dtype=p.dtype)
                        state["col_curvature"] = torch.tensor(1.0, device=p.device, dtype=p.dtype)
                        state["hamiltonian_energy"] = torch.tensor(0.0, device=p.device, dtype=p.dtype)
                        state["magnetic_strength"] = mag_base
                        state["mode"] = "normal"
                        if use_ema and 'ema_param' not in state:
                            state['ema_param'] = p.data.clone().detach()
                    
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
                    
                    # 哈密顿能量
                    hamiltonian_energy = precond_avg.norm()
                    state["hamiltonian_energy"] = (
                        0.9 * state["hamiltonian_energy"] + 
                        0.1 * hamiltonian_energy
                    )
                    
                    # 记录梯度范数
                    grad_norm = grad_2d.norm().item()
                    self.grad_norm_history.append(grad_norm)
                    if len(self.grad_norm_history) > 50:
                        self.grad_norm_history.pop(0)
                    
                    # ========================================
                    # 模式选择与方向计算
                    # ========================================
                    
                    # 基础方向（Soft NS）
                    sign_update = torch.sign(precond_avg)
                    if use_soft_ns:
                        ortho = soft_newton_schulz(
                            sign_update, 
                            steps=ns_steps, 
                            spectral_filter=spectral_filter,
                            eps=eps
                        )
                    else:
                        ortho = zeropower_via_newtonschulz5(sign_update, steps=ns_steps, eps=eps)
                    
                    a, b = p_2d.shape
                    base_scale = 0.2 * math.sqrt(a * b) / (ortho.norm() + eps)
                    
                    # 能量调制（V8）
                    energy_scale = 1.0 + 0.1 * torch.tanh(state["hamiltonian_energy"] / 10.0)
                    base_direction = ortho * base_scale * energy_scale
                    
                    # ========================================
                    # Mode 1: Tunneling Mode (V9)
                    # ========================================
                    if self.tunneling_mode:
                        state["mode"] = "tunneling"
                        
                        # 复数更新
                        cos_theta = math.cos(phase_angle)
                        sin_theta = math.sin(phase_angle)
                        
                        real_part = cos_theta * base_direction
                        
                        # 虚部：正交方向
                        if abs(sin_theta) > 1e-6:
                            ortho_direction = torch.roll(base_direction, shifts=1, dims=0)
                            imag_part = sin_theta * ortho_direction
                        else:
                            imag_part = 0
                        
                        direction = real_part + imag_part
                    
                    # ========================================
                    # Mode 2: Confinement Mode (V10)
                    # ========================================
                    elif grad_unstable and use_plasma and is_square:
                        state["mode"] = "confinement"
                        
                        W = p_2d
                        G = base_direction
                        M = exp_avg
                        
                        # 辛磁场
                        B = self._construct_symplectic_magnetic_field(W, G, eps)
                        
                        # 洛伦兹力
                        F_lorentz = self._compute_lorentz_force(W, M, B, eps)
                        
                        # 自适应磁场强度
                        if mag_adaptive:
                            magnetic_strength = self._adaptive_magnetic_strength(
                                grad_2d.norm(), conf_threshold, mag_base
                            )
                        else:
                            magnetic_strength = mag_base
                        
                        state["magnetic_strength"] = magnetic_strength
                        
                        # 洛伦兹力修正
                        plasma_direction = base_direction + magnetic_strength * F_lorentz
                        
                        # 归一化
                        plasma_norm = plasma_direction.norm() + eps
                        base_norm = base_direction.norm() + eps
                        direction = plasma_direction * (base_norm / plasma_norm)
                    
                    # ========================================
                    # Mode 3: Normal Mode (V8)
                    # ========================================
                    else:
                        state["mode"] = "normal"
                        direction = base_direction
                    
                    # ========================================
                    # Cayley 辛积分（所有模式共用）
                    # ========================================
                    
                    if is_square and use_cayley:
                        W = p_2d
                        G = direction
                        
                        # 收敛区使用 Strang Splitting
                        if in_convergence_zone and use_strang:
                            try:
                                # Half-Step Row
                                A_row = G @ W.T - W @ G.T
                                row_curvature_current = A_row.norm() / (W.size(0) + eps)
                                state["row_curvature"] = (
                                    curvature_ema * state["row_curvature"] + 
                                    (1 - curvature_ema) * row_curvature_current
                                )
                                
                                alpha_row = 0.5 * lr / (state["row_curvature"] + eps)
                                alpha_row *= current_trust
                                alpha_row_half = alpha_row * 0.5
                                
                                I = torch.eye(W.size(0), device=W.device, dtype=W.dtype)
                                Q_minus = I - alpha_row_half * A_row
                                target = W + alpha_row_half * (A_row @ W)
                                W_half = safe_linalg_solve(Q_minus, target)
                                
                                exp_avg_half = W_half @ (W.T @ exp_avg)
                                
                                # Full-Step Col
                                W_T = W_half.T
                                G_T = G.T
                                A_col = G_T @ W_T.T - W_T @ G_T.T
                                col_curvature_current = A_col.norm() / (W.size(1) + eps)
                                state["col_curvature"] = (
                                    curvature_ema * state["col_curvature"] + 
                                    (1 - curvature_ema) * col_curvature_current
                                )
                                
                                alpha_col = 0.5 * lr / (state["col_curvature"] + eps)
                                alpha_col *= current_trust
                                
                                I_col = torch.eye(W.size(1), device=W.device, dtype=W.dtype)
                                Q_minus_col = I_col - alpha_col * A_col
                                target_col = W_T + alpha_col * (A_col @ W_T)
                                W_T_full = safe_linalg_solve(Q_minus_col, target_col)
                                W_full = W_T_full.T
                                
                                exp_avg_T = exp_avg_half.T
                                exp_avg_T_full = W_T_full @ (W_T.T @ exp_avg_T)
                                exp_avg_full = exp_avg_T_full.T
                                
                                # Half-Step Row (again)
                                A_row_final = G @ W_full.T - W_full @ G.T
                                Q_minus_final = I - alpha_row_half * A_row_final
                                target_final = W_full + alpha_row_half * (A_row_final @ W_full)
                                W_new = safe_linalg_solve(Q_minus_final, target_final)
                                
                                exp_avg_new = W_new @ (W_full.T @ exp_avg_full)
                                
                                p.copy_(W_new.view_as(p))
                                exp_avg.copy_(exp_avg_new)
                                
                            except:
                                if wd > 0:
                                    p.data.mul_(1 - lr * wd)
                                p.data.add_(direction.view_as(p), alpha=-lr)
                        
                        else:
                            # 奇偶交替
                            is_row_step = (state["step"] % 2 == 1)
                            
                            if is_row_step:
                                A_row = G @ W.T - W @ G.T
                                row_curvature_current = A_row.norm() / (W.size(0) + eps)
                                state["row_curvature"] = (
                                    curvature_ema * state["row_curvature"] + 
                                    (1 - curvature_ema) * row_curvature_current
                                )
                                
                                alpha_row = 0.5 * lr / (state["row_curvature"] + eps)
                                alpha_row *= current_trust
                                
                                I = torch.eye(W.size(0), device=W.device, dtype=W.dtype)
                                Q_minus = I - alpha_row * A_row
                                target = W + alpha_row * (A_row @ W)
                                
                                try:
                                    W_new = safe_linalg_solve(Q_minus, target)
                                    p.copy_(W_new.view_as(p))
                                    exp_avg_rotated = W_new @ (W.T @ exp_avg)
                                    exp_avg.copy_(exp_avg_rotated)
                                except:
                                    if wd > 0:
                                        p.data.mul_(1 - lr * wd)
                                    p.data.add_(direction.view_as(p), alpha=-lr)
                            
                            else:
                                W_T = W.T
                                G_T = G.T
                                A_col = G_T @ W_T.T - W_T @ G_T.T
                                col_curvature_current = A_col.norm() / (W.size(1) + eps)
                                state["col_curvature"] = (
                                    curvature_ema * state["col_curvature"] + 
                                    (1 - curvature_ema) * col_curvature_current
                                )
                                
                                alpha_col = 0.5 * lr / (state["col_curvature"] + eps)
                                alpha_col *= current_trust
                                
                                I_col = torch.eye(W.size(1), device=W.device, dtype=W.dtype)
                                Q_minus_col = I_col - alpha_col * A_col
                                target_col = W_T + alpha_col * (A_col @ W_T)
                                
                                try:
                                    W_T_new = safe_linalg_solve(Q_minus_col, target_col)
                                    W_new = W_T_new.T
                                    p.copy_(W_new.view_as(p))
                                    exp_avg_T = exp_avg.T
                                    exp_avg_T_rotated = W_T_new @ (W_T.T @ exp_avg_T)
                                    exp_avg.copy_(exp_avg_T_rotated.T)
                                except:
                                    if wd > 0:
                                        p.data.mul_(1 - lr * wd)
                                    p.data.add_(direction.view_as(p), alpha=-lr)
                    else:
                        # 非方阵或不使用 Cayley
                        if wd > 0:
                            p.data.mul_(1 - lr * wd)
                        p.data.add_(direction.view_as(p), alpha=-lr)
                
                # ========================================
                # 1D 参数: 1D-Muon
                # ========================================
                else:
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["mode"] = "normal"
                        if use_ema and 'ema_param' not in state:
                            state['ema_param'] = p.data.clone().detach()
                    
                    state["step"] += 1
                    exp_avg = state["exp_avg"]
                    
                    exp_avg.mul_(momentum).add_(grad)
                    
                    norm = exp_avg.norm() + eps
                    normalized_update = exp_avg / norm
                    
                    if wd > 0:
                        p.data.mul_(1 - lr * wd)
                    
                    p.data.add_(normalized_update, alpha=-lr)
                
                # ========================================
                # EMA 更新
                # ========================================
                if use_ema and 'ema_param' in state:
                    state['ema_param'].mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)
        
        return loss
    
    def swap_ema_weights(self):
        """交换当前权重和 EMA 权重"""
        for group in self.param_groups:
            if not group.get('use_ema', False):
                continue
            for p in group['params']:
                state = self.state[p]
                if 'ema_param' in state:
                    tmp = p.data.clone()
                    p.data.copy_(state['ema_param'])
                    state['ema_param'].copy_(tmp)
    
    def get_quantum_stats(self):
        """获取 Quantum Zeta 统计信息"""
        stats = {
            'modes': {},
            'hamiltonian_energies': [],
            'magnetic_strengths': [],
            'tunneling_active': self.tunneling_mode,
            'tunneling_progress': self.tunneling_counter if self.tunneling_mode else 0,
        }
        
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if 'mode' in state:
                    mode = state['mode']
                    stats['modes'][mode] = stats['modes'].get(mode, 0) + 1
                if 'hamiltonian_energy' in state:
                    stats['hamiltonian_energies'].append(state['hamiltonian_energy'].item())
                if 'magnetic_strength' in state:
                    stats['magnetic_strengths'].append(state['magnetic_strength'])
        
        if stats['hamiltonian_energies']:
            stats['avg_hamiltonian_energy'] = sum(stats['hamiltonian_energies']) / len(stats['hamiltonian_energies'])
        
        if stats['magnetic_strengths']:
            stats['avg_magnetic_strength'] = sum(stats['magnetic_strengths']) / len(stats['magnetic_strengths'])
        
        return stats
    
    def state_dict(self):
        """序列化优化器状态"""
        state = super().state_dict()
        state['loss_history'] = self.loss_history
        state['tunneling_mode'] = self.tunneling_mode
        state['tunneling_counter'] = self.tunneling_counter
        state['grad_norm_history'] = self.grad_norm_history
        return state
    
    def load_state_dict(self, state_dict):
        """加载优化器状态"""
        self.loss_history = state_dict.pop('loss_history', [])
        self.tunneling_mode = state_dict.pop('tunneling_mode', False)
        self.tunneling_counter = state_dict.pop('tunneling_counter', 0)
        self.grad_norm_history = state_dict.pop('grad_norm_history', [])
        super().load_state_dict(state_dict)
