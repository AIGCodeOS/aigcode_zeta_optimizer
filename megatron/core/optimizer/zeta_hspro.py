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
    """优化的 NS 迭代: 原地操作与 JIT 加速"""
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        A = X @ X.T
        B = A.mul(b).addmm(A, A, alpha=c)
        X = X.mul(a).addmm(B, X)
    return X

def detect_solitons_via_spectral_gap(
    X: torch.Tensor, 
    ns_steps_high: int, 
    ns_steps_mid: int, 
    ns_steps_low: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    数学重构：输入 X 必须是已经经过 Adam 预条件的梯度矩阵。
    此时提取的谱间隙直接位于等距流形（Isometric Manifold）上。
    """
    # 迭代到 low 阶 (提取最大特征值主导的方向)
    X_low = fast_ns_iteration(X, ns_steps_low)
    
    # 继续迭代到 mid 阶 (提取中间特征值对应的孤子结构)
    X_mid = fast_ns_iteration(X_low.clone(), ns_steps_mid - ns_steps_low)
    S = X_mid - X_low
    
    # 继续迭代到 high 阶 (提取微弱的调和残差)
    Q_high = fast_ns_iteration(X_mid.clone(), ns_steps_high - ns_steps_mid)
    H = Q_high - X_mid
    
    return Q_high, H, S


class Zeta1DHodge:
    """
    1D参数的Hodge同步优化器
    数学核心：将1D视为退化矩阵，保持与2D相同的谱几何
    """
    
    @staticmethod
    def decompose_1d_hodge(
        grad: torch.Tensor,
        state: dict,
        step: int,
        beta1: float = 0.9,
        beta2: float = 0.95,
        eps: float = 1e-8
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        1D Hodge分解：梯度场 = 精确分量 + 余精确分量 + 调和分量
        
        对于1D向量，这对应于：
        - 精确分量 (exact): 梯度符号方向 (类比2D的Q_high)
        - 余精确分量 (co-exact): 方差波动方向 (类比2D的H)  
        - 调和分量 (harmonic): 稳态偏移 (类比2D的S)
        """
        # 动量估计 (一阶矩)
        # 确保 state 与 grad 同形，避免 NPU/distributed 下 p.data 与 p.grad 形状不一致导致 lerp_ 报错
        if "exp_avg" not in state or state["exp_avg"].shape != grad.shape:
            state["exp_avg"] = torch.zeros_like(grad)
            state["exp_avg_sq"] = torch.zeros_like(grad)
            state["harmonic"] = torch.zeros_like(grad)
        
        state["exp_avg"].lerp_(grad, 1 - beta1)
        state["exp_avg_sq"].lerp_(grad.pow(2), 1 - beta2)
        
        # 偏差校正
        bc1 = 1.0 - beta1 ** step
        bc2 = 1.0 - beta2 ** step
        m_hat = state["exp_avg"] / bc1
        v_hat = state["exp_avg_sq"] / bc2
        
        # === Hodge分解核心 ===
        # 1. 精确分量：符号化方向 (对应2D的Q_high - 主导流形)
        exact = torch.sign(m_hat)
        
        # 2. 余精确分量：方差归一化的波动 (对应2D的H - 调和残差)
        signal_to_noise = m_hat.abs() / (v_hat.sqrt() + eps)
        co_exact = torch.clamp(signal_to_noise - 1.0, min=0.0) * torch.sign(m_hat)
        
        # 3. 调和分量：长期稳态偏移 (对应2D的S - 孤子结构)
        # 使用指数移动平均跟踪调和模式
        state["harmonic"].lerp_(torch.sign(grad), 0.01)
        harmonic = state["harmonic"]
        
        return exact, co_exact, harmonic
    
    @staticmethod
    def compute_1d_complexity(
        m_hat: torch.Tensor,
        v_hat: torch.Tensor,
        eps: float = 1e-8
    ) -> float:
        """
        计算1D梯度场的复杂度指标 (用于动态刹车)
        
        数学动机：当梯度方差激增时，表示接近流形奇点，需要减速
        参考Hodge-Kodaira分解中的调和形式稳定性分析 [[11]][[12]]
        """
        variance = torch.clamp(v_hat - m_hat.pow(2), min=0.0)
        # 相对变异系数 (Coefficient of Variation)
        cv = (variance.sqrt() / (m_hat.abs() + eps)).mean().item()
        # 映射到[0, 1]区间
        complexity = min(1.0, cv / 10.0)
        return complexity
    
    @staticmethod
    def adaptive_blend(
        exact: torch.Tensor,
        co_exact: torch.Tensor,
        harmonic: torch.Tensor,
        step: int,
        warmup_steps: int,
        dim: int,
        alpha_base: float = 0.15,
        beta_base: float = 0.10
    ) -> torch.Tensor:
        """
        自适应混合三个Hodge分量
        
        缩放律：基于维度平方根倒数，与2D保持一致 [[30]][[35]]
        """
        # 训练初期压制噪声分量
        blend_factor = min(1.0, step / warmup_steps)
        eps = 1e-8
        # 维度自适应缩放 (与2D的dynamic_scale统一)
        dim_scale = math.sqrt(1024.0 / max(dim, 1))
        
        # 动态混合系数
        e_norm = exact.norm() + eps
        alpha = alpha_base * blend_factor * dim_scale * min(1.0, co_exact.norm() / e_norm)
        beta = beta_base * blend_factor * dim_scale * min(1.0, harmonic.norm() / e_norm)
        
        # 保证主导方向不丢失
        gamma = max(0.6, 1.0 - alpha - beta)
        
        combined = exact.mul(gamma).add(co_exact, alpha=alpha).add(harmonic, alpha=beta)
        return combined
    
class ZetaHSPro(optim.Optimizer):
    """
    Zeta-HS-Pro: 基于严谨数学推导的 Hodge-Soliton 优化器
    
    数学修复点:
    1. 前置预条件: 将 Adam 二阶矩移到 NS 迭代之前，废弃针对微小残差的错误方差缩放。
    2. 正确的 RG 投影: 将梯度投影到孤子基底上，而非反向。
    3. 动态缩放律: 基于 1024 维度的平方根倒数定律，自动适配 0.6B 到 230B 模型。
    """
    def __init__(
        self,
        params,
        lr: float = 1.5e-3,           
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        ns_steps: int = 5,
        ns_steps_high: int = 5,
        ns_steps_mid: int = 3,
        ns_steps_low: int = 1,
        warmup_steps: int = 2000,
        alpha_base: float = 0.1,  # 调和分量(H)基础混合率
        beta_base: float = 0.2,   # 孤子分量(S)基础混合率
    ):
        defaults = dict(
            lr=lr, weight_decay=weight_decay, betas=betas, eps=eps,
            ns_steps_high=ns_steps, ns_steps_mid=ns_steps_mid, ns_steps_low=ns_steps_low,
            warmup_steps=warmup_steps, alpha_base=alpha_base, beta_base=beta_base
        )
        super().__init__(params, defaults)
        self.global_step = 0
        print("[Zeta-HS-Pro] 严谨数学重构版加载完成 (目标 MFU 优化).")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.global_step += 1

        for group in self.param_groups:
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None: continue
                g = p.grad.float()

                if p.ndim >= 2:
                    m, n = p.shape[0], p.numel() // p.shape[0]
                    dim_min = min(m, n)
                    g_in = g.view(m, n)
                    
                    state = self.state[p]
                    # 确保 state 与 g_in 同形，避免 NPU/distributed 下 shape 不一致导致 add_ 报错
                    if "exp_avg" not in state or state["exp_avg"].shape != g_in.shape:
                        state["exp_avg"] = torch.zeros_like(g_in)
                        state["exp_avg_sq"] = torch.zeros_like(g_in)
                    
                    # --- 核心修复 1: 前置流形预条件 ---
                    # 避免对 H 和 S 等微小残差进行错误的方差爆炸
                    state["exp_avg"].mul_(beta1).add_(g_in, alpha=1 - beta1)
                    state["exp_avg_sq"].mul_(beta2).addcmul_(g_in, g_in, value=1 - beta2)
                    
                    # 偏差校正 (Bias Correction)
                    bc1 = 1.0 - beta1 ** self.global_step
                    bc2 = 1.0 - beta2 ** self.global_step
                    m_hat = state["exp_avg"] / bc1
                    v_hat = state["exp_avg_sq"] / bc2
                    
                    precond_grad = m_hat / (v_hat.sqrt() + eps)
                    
                    # 进入极分解空间
                    X = precond_grad.bfloat16() if precond_grad.dtype != torch.bfloat16 else precond_grad
                    transposed = False
                    if m > n:
                        X = X.T
                        transposed = True
                        
                    X = X / (X.norm() + eps)
                    
                    # --- 高效多尺度 NS 迭代 ---
                    Q_high, H, S = detect_solitons_via_spectral_gap(
                        X, group["ns_steps_high"], group["ns_steps_mid"], group["ns_steps_low"]
                    )
                    
                    if transposed:
                        Q_high, H, S = Q_high.T, H.T, S.T
                    
                    # 转回正确精度
                    Q_high = Q_high.to(p.dtype)
                    H = H.to(p.dtype)
                    S = S.to(p.dtype)
                    
                    # --- 核心修复 2: 正确的数学投影与自适应缩放 ---
                    dynamic_scale = math.sqrt(1024.0 / dim_min)
                    
                    # 当训练初期流形极其动荡时，H 和 S 往往是纯噪声，应当压制
                    blend_factor = min(1.0, self.global_step / group['warmup_steps'])
                    
                    # 动态混合系数：基于谱间隙能量比
                    q_norm = Q_high.norm() + eps
                    alpha = group['alpha_base'] * blend_factor * dynamic_scale * min(1.0, (H.norm() / q_norm).item())
                    beta = group['beta_base'] * blend_factor * dynamic_scale * min(1.0, (S.norm() / q_norm).item())
                    
                    # 保证主导流形方向不丢失
                    gamma = max(0.5, 1.0 - alpha - beta)
                    
                    # 合成最终等距更新方向
                    combined = Q_high.mul(gamma).add_(H, alpha=alpha).add_(S, alpha=beta)
                    
                    # --- 核心修复 3: 动态维度缩放律 ---
                    # 废弃写死的 0.2，采用自适应缩放防止 8B+ 模型爆炸
                    moonshot_scale = 0.2 * dynamic_scale * math.sqrt(m * n) / (combined.norm() + eps)
                    update = (combined * moonshot_scale).view_as(p)

                    if wd > 0: p.data.mul_(1 - lr * wd)
                    p.data.add_(update, alpha=-lr)
                    
                # ========================================================
                # 1D v3 复杂同步 参数：Hodge同步向量优化 (Zeta-1D)
                # ========================================================
                else:
                    # 初始化状态（与 g 同形，避免 NPU/distributed 下 p.data 与 p.grad 形状不一致）
                    state = self.state[p]
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(g)
                        state["exp_avg_sq"] = torch.zeros_like(g)
                        state["harmonic"] = torch.zeros_like(g)
                    
                    state["step"] += 1
                    
                    # Hodge分解
                    exact, co_exact, harmonic = Zeta1DHodge.decompose_1d_hodge(
                        g, state, state["step"], beta1, beta2, eps
                    )
                    
                    # 计算复杂度指标 (用于动态刹车)
                    bc1 = 1.0 - beta1 ** state["step"]
                    bc2 = 1.0 - beta2 ** state["step"]
                    m_hat = state["exp_avg"] / bc1
                    v_hat = state["exp_avg_sq"] / bc2
                    complexity = Zeta1DHodge.compute_1d_complexity(m_hat, v_hat, eps)
                    
                    # 动态刹车系数：接近奇点时自动减速
                    # 数学动机：Hodge流形在奇点附近曲率发散 [[13]]
                    braking_factor = 1.0 / (1.0 + 5.0 * complexity)
                    
                    # 自适应混合
                    combined = Zeta1DHodge.adaptive_blend(
                        exact, co_exact, harmonic,
                        self.global_step, group['warmup_steps'],
                        p.numel(), group['alpha_base'], group['beta_base']
                    )
                    
                    # 统一缩放律 (与2D保持一致)
                    dim_scale = math.sqrt(1024.0 / max(p.numel(), 1))
                    step_size = lr * braking_factor * dim_scale * math.sqrt(bc2) / bc1
                    
                    # 权重衰减
                    if wd > 0:
                        p.data.mul_(1 - lr * wd)
                    
                    # 应用更新
                    p.data.add_(combined, alpha=-step_size)
                
                #else:
                # ========================================================
                # 1D 参数: Hodge-Synchronized Vector Sign (Zeta-1D)
                # ========================================================
                #else:
                    # 1D 最简思路参考: 仅 exact 分量（梯度符号）
                    #update = torch.sign(G)
                    #alpha = lr / (1.0 + torch.std(G))                    
                    ''' 1D v2 简单融合逻辑
                    state = self.state[p]
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p.data)
                        state["exp_avg_sq"] = torch.zeros_like(p.data)
                    
                    state["step"] += 1
                    
                    # 1. 动量与方差估计
                    state["exp_avg"].lerp_(g, 1 - beta1)
                    state["exp_avg_sq"].lerp_(g.pow(2), 1 - beta2)
                    
                    bc1 = 1 - beta1 ** state["step"]
                    bc2 = 1 - beta2 ** state["step"]
                    m_hat = state["exp_avg"] / bc1
                    v_hat = state["exp_avg_sq"] / bc2
                    
                    # 2. 运动学同步：1D Hodge 奇点感知 (Kinematic Synchronization)
                    # 计算 1D 梯度场的相对变异系数 (Signal-to-Noise 倒数)
                    variance = torch.clamp(v_hat - m_hat.pow(2), min=0.0)
                    complexity_1d = (variance.sqrt() / (m_hat.abs() + eps)).mean().item()
                    
                    # 动态刹车系数 (Hodge Braking)
                    # 当 1D 参数遇到奇点(梯度方差激增)时，自动缩减步长，与 2D 的投影流形保持同步
                    braking_factor = 1.0 / (1.0 + complexity_1d)
                    
                    # 3. 算子同构：向量符号算子 (Vector Sign)
                    # 废弃 Adam 的直接除以 sqrt(v)，改用保留幅度限制的符号化方向
                    # 这在数学上等价于 2D 矩阵的 Newton-Schulz 零次幂
                    adam_ratio = m_hat.abs() / (v_hat.sqrt() + eps)
                    update_dir = torch.sign(m_hat) * torch.clamp(adam_ratio, max=1.0)
                    
                    # 引入硬件防溢出保护，并应用动态刹车步长
                    step_size = lr * braking_factor
                    
                    # 4. 黎曼流形近似更新 (Riemannian-Inspired Update) & 权重衰减
                    # 对于非零的 1D 参数 (如 Norm 层的 gamma)，施加乘法缩放惯性
                    if wd > 0:
                        p.data.mul_(1 - lr * wd)
                        
                    p.data.add_(update_dir, alpha=-step_size)
                    
                    # 1Dv1: 标准 AdamW 
                    state = self.state[p]
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)

                    state["step"] += 1
                    state["exp_avg"].lerp_(g, 1 - beta1)
                    state["exp_avg_sq"].lerp_(g.pow(2), 1 - beta2)

                    bc1 = 1 - beta1 ** state["step"]
                    bc2 = 1 - beta2 ** state["step"]
                    step_size = lr * math.sqrt(bc2) / bc1
                    
                    denom = state["exp_avg_sq"].sqrt().add_(eps)
                    if wd > 0: p.data.mul_(1 - lr * wd)
                    p.data.addcdiv_(state["exp_avg"], denom, value=-step_size)
                    '''

        return loss



# Alias for registration
#Zeta = ZetaHS
Zeta = ZetaHSPro