# Zeta Perturbation-Driven Optimizer
# 
# 核心理念：扰动主导、边界约束
# - 主动驱动：非线性扰动（Torsion + Soliton）作为探索的核心动力
# - 被动边界：二阶预条件（Hessian）+ 谱投影（Newton-Schulz）作为安全约束
#
# 数学框架：W_{t+1} = Retr_M(W_t - η·Π_spec(H^{-1/2}∇L(W_t)) + ξ_t)
#
# 性能目标：≤ 2x Muon baseline (preferably 1.5x)

import math
import torch
import torch.optim as optim
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

try:
    import torch_npu
    HAS_NPU = True
except ImportError:
    HAS_NPU = False


# ============================================================================
# Phase Detection Module (Task 1)
# ============================================================================

@dataclass
class PhaseInfo:
    """Training phase information"""
    ratio: float  # step / total_steps ∈ [0, 1]
    perturbation_scale: float  # α_phase for perturbations
    boundary_scale: float  # β_phase for boundaries
    phase_name: str  # "early", "mid", "late"


class PhaseDetector:
    """
    Tracks training progress and provides phase-dependent scaling factors.
    
    Phase transitions:
    - Early (0-20%): Strong perturbation (1.5x) + Weak boundary (0.5x)
    - Mid (20-80%): Balanced (1.0x + 1.0x)
    - Late (80-100%): Weak perturbation (0.5x) + Strong boundary (1.5x)
    """
    
    def __init__(self, total_steps: Optional[int] = None):
        self.total_steps = total_steps
    
    def get_phase_info(self, step: int) -> PhaseInfo:
        """Compute phase-dependent scaling factors with smooth transitions"""
        if self.total_steps is None or self.total_steps <= 0:
            # Default to mid-phase if total_steps not provided
            return PhaseInfo(
                ratio=0.5,
                perturbation_scale=1.0,
                boundary_scale=1.0,
                phase_name="mid"
            )
        
        # Phase ratio
        phi = min(1.0, step / self.total_steps)
        
        # Smooth transitions using sigmoid
        # α_phase(φ) = 1.5 - 1.0·sigmoid(10(φ-0.2)) - 0.5·sigmoid(10(φ-0.8))
        # β_phase(φ) = 0.5 + 0.5·sigmoid(10(φ-0.2)) + 0.5·sigmoid(10(φ-0.8))
        
        def sigmoid(x):
            return 1.0 / (1.0 + math.exp(-x))
        
        alpha_phase = 1.5 - 1.0 * sigmoid(10 * (phi - 0.2)) - 0.5 * sigmoid(10 * (phi - 0.8))
        beta_phase = 0.5 + 0.5 * sigmoid(10 * (phi - 0.2)) + 0.5 * sigmoid(10 * (phi - 0.8))
        
        # Determine phase name
        if phi < 0.2:
            phase_name = "early"
        elif phi < 0.8:
            phase_name = "mid"
        else:
            phase_name = "late"
        
        return PhaseInfo(
            ratio=phi,
            perturbation_scale=alpha_phase,
            boundary_scale=beta_phase,
            phase_name=phase_name
        )


# ============================================================================
# Core Utility Functions
# ============================================================================

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Newton-Schulz iteration for zero-power (orthogonalization)"""
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    # Handle NaN/Inf
    if not torch.isfinite(G).all():
        return torch.zeros_like(G)
    
    X = G.bfloat16() if G.dtype != torch.bfloat16 else G
    
    if G.size(0) > G.size(1):
        X = X.T
    
    # Pre-normalize
    X = X / (X.norm() + eps)
    
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    
    if G.size(0) > G.size(1):
        X = X.T
    
    return X.to(G.dtype)


# ============================================================================
# Soliton Driver Module (Task 2)
# ============================================================================

@dataclass
class SolitonState:
    """Soliton detection state"""
    centers: torch.Tensor  # Soliton centers [m, k]
    strengths: torch.Tensor  # Soliton strengths [k]
    coherence: float  # Coherence measure


def detect_solitons(
    W: torch.Tensor,
    num_solitons: int = 4,
    ns_steps: int = 3,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Detect soliton centers using Newton-Schulz approximation.
    
    Returns:
        (soliton_centers [m, k], soliton_strengths [k])
    """
    m, n = W.shape
    k = min(num_solitons, min(m, n))
    
    soliton_centers = []
    soliton_strengths = []
    
    residual = W.clone()
    
    for i in range(k):
        # Apply Newton-Schulz orthogonalization
        ortho = zeropower_via_newtonschulz5(residual, steps=ns_steps, eps=eps)
        
        # Extract principal direction (always use first column)
        center = ortho[:, 0:1]  # [m, 1]
        
        # Compute strength (projection length)
        strength = (center.T @ residual).norm()
        
        soliton_centers.append(center)
        soliton_strengths.append(strength)
        
        # Remove this component from residual
        projection = center @ (center.T @ residual)
        residual = residual - projection
        
        # Early stopping if residual is negligible
        if residual.norm() < eps * W.norm():
            break
    
    # Concatenate results
    if soliton_centers:
        soliton_centers = torch.cat(soliton_centers, dim=1)  # [m, k]
        soliton_strengths = torch.stack(soliton_strengths)   # [k]
    else:
        soliton_centers = torch.zeros(m, 1, device=W.device, dtype=W.dtype)
        soliton_strengths = torch.zeros(1, device=W.device, dtype=W.dtype)
    
    return soliton_centers, soliton_strengths


def compute_soliton_perturbation(
    W: torch.Tensor,
    G: torch.Tensor,
    grad_norm: float,
    phase_info: PhaseInfo,
    num_solitons: int = 4,
    soliton_ns_steps: int = 3,
    coherence_threshold: float = 0.1,
    alpha_base: float = 0.5,
    alpha_max: float = 1.0,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, SolitonState]:
    """
    Compute soliton-driven perturbation (Task 2.1, 2.2, 2.3).
    
    Returns:
        (ξ_soliton, soliton_state)
    """
    # Detect solitons
    soliton_centers, soliton_strengths = detect_solitons(
        W, num_solitons=num_solitons, ns_steps=soliton_ns_steps, eps=eps
    )
    
    # RG projection
    G_rg = soliton_centers @ (soliton_centers.T @ G)
    
    # Coherence filter
    inner_product = (G_rg * W).sum()
    g_norm = G_rg.norm() + eps
    w_norm = W.norm() + eps
    coherence = inner_product.abs() / (g_norm * w_norm)
    
    suppression = min(1.0, coherence / (coherence_threshold + eps))
    G_filtered = G_rg * suppression
    
    # Adaptive soliton strength (Task 2.1)
    # α_s = min(α_max, α_base · (1 + exp(-||∇L|| / threshold)))
    threshold = 1.0
    alpha_s = min(alpha_max, alpha_base * (1.0 + math.exp(-grad_norm / threshold)))
    
    # Apply phase-dependent scaling
    alpha_s = alpha_s * phase_info.perturbation_scale
    
    # Final soliton perturbation
    xi_soliton = alpha_s * (G_filtered - G)
    
    soliton_state = SolitonState(
        centers=soliton_centers,
        strengths=soliton_strengths,
        coherence=coherence.item()
    )
    
    return xi_soliton, soliton_state


# ============================================================================
# Torsion Driver Module (Task 3)
# ============================================================================

@dataclass
class TorsionState:
    """Torsion computation state"""
    torsion_magnitude: float
    curvature_indicator: float
    direction: torch.Tensor  # Torsion direction [m, n]


def compute_torsion_perturbation(
    M: torch.Tensor,
    G: torch.Tensor,
    hessian: torch.Tensor,
    phase_info: PhaseInfo,
    beta_base: float = 0.3,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, TorsionState]:
    """
    Compute torsion-driven perturbation (Task 3.1, 3.2, 3.3).
    
    Returns:
        (ξ_torsion, torsion_state)
    """
    # Compute trajectory torsion (Task 3.1)
    trace_MG = (M * G).sum()
    m_norm = M.norm() + eps
    g_norm = G.norm() + eps
    torsion_scale = torch.tanh(trace_MG / (m_norm * g_norm))
    
    # Compute curvature indicator (Task 3.2)
    m, n = hessian.shape
    curvature = hessian.norm() / (m * n + eps)
    curvature_indicator = math.log(1.0 + curvature.item())
    
    # Adaptive torsion strength
    beta_t = beta_base * (1.0 + curvature_indicator) * phase_info.perturbation_scale
    
    # Torsion direction (orthogonal to gradient) (Task 3.3)
    inner_prod = (M * G).sum()
    g_norm_sq = (G * G).sum() + eps
    T_raw = M - (inner_prod / g_norm_sq) * G
    T_normalized = T_raw / (T_raw.norm() + eps)
    
    # Final torsion perturbation (scaled by gradient norm for dimensional consistency)
    xi_torsion = beta_t * torsion_scale * T_normalized * g_norm
    
    torsion_state = TorsionState(
        torsion_magnitude=torsion_scale.item(),
        curvature_indicator=curvature_indicator,
        direction=T_normalized
    )
    
    return xi_torsion, torsion_state


# ============================================================================
# Adaptive Mixer Module (Task 4)
# ============================================================================

def adaptive_mix_updates(
    base_update: torch.Tensor,
    xi_soliton: torch.Tensor,
    xi_torsion: torch.Tensor,
    target_ratio: Tuple[float, float] = (0.4, 0.6),
    safety_clip_factor: float = 3.0,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Mix base update and perturbations with adaptive ratio control (Task 4.1, 4.2, 4.3).
    
    Returns:
        (combined_update, metrics_dict)
    """
    # Compute total perturbation
    xi_total = xi_soliton + xi_torsion
    
    # Compute current perturbation ratio (Task 4.1)
    perturbation_norm = xi_total.norm() + eps
    base_norm = base_update.norm() + eps
    current_ratio = perturbation_norm / (perturbation_norm + base_norm)
    
    # Adaptive scaling to target ratio (Task 4.2)
    target_min, target_max = target_ratio
    target_mid = (target_min + target_max) / 2.0
    
    if current_ratio < target_min or current_ratio > target_max:
        scale = target_mid / (current_ratio + eps)
        xi_total = xi_total * scale
        # Recompute ratio after scaling
        perturbation_norm = xi_total.norm() + eps
        current_ratio = perturbation_norm / (perturbation_norm + base_norm)
    
    # Combine updates
    combined_update = base_update + xi_total
    
    # Safety clipping (Task 4.3)
    max_norm = safety_clip_factor * base_norm
    combined_norm = combined_update.norm()
    if combined_norm > max_norm:
        combined_update = combined_update * (max_norm / (combined_norm + eps))
    
    # Collect metrics
    metrics = {
        "perturbation_ratio": current_ratio.item(),
        "base_norm": base_norm.item(),
        "soliton_norm": xi_soliton.norm().item(),
        "torsion_norm": xi_torsion.norm().item(),
        "total_perturbation_norm": perturbation_norm.item(),
        "final_update_norm": combined_update.norm().item(),
    }
    
    return combined_update, metrics


# ============================================================================
# Main Optimizer Class
# ============================================================================

class ZetaPerturbationDriven(optim.Optimizer):
    """
    Zeta Perturbation-Driven Optimizer
    
    核心理念：
    - 主动驱动：非线性扰动（Torsion + Soliton）作为探索的核心动力
    - 被动边界：二阶预条件（Hessian）+ 谱投影（Newton-Schulz）作为安全约束
    
    数学框架：
    W_{t+1} = Retr_M(W_t - η·Π_spec(H^{-1/2}∇L(W_t)) + ξ_t)
    
    Args:
        params: Model parameters
        lr: Learning rate (default: 3e-4)
        betas: Momentum coefficients (default: (0.9, 0.95))
        eps: Numerical stability (default: 1e-8)
        weight_decay: Weight decay (default: 0.01)
        momentum: Momentum coefficient (default: 0.95)
        
        # Hessian preconditioning (Passive Boundary)
        hessian_update_freq: Hessian update frequency (default: 10)
        gamma: Hessian update rate (default: 0.01)
        
        # Spectral projection (Passive Boundary)
        ns_steps: Newton-Schulz iterations (default: 5)
        projection_strength: Spectral projection strength γ_base (default: 0.3)
        
        # Soliton perturbation (Active Driver)
        use_soliton: Enable soliton perturbation (default: True)
        num_solitons: Number of solitons (default: 4)
        soliton_ns_steps: NS steps for soliton detection (default: 3)
        coherence_threshold: Coherence threshold (default: 0.1)
        alpha_base: Base soliton strength (default: 0.5)
        alpha_max: Max soliton strength (default: 1.0)
        
        # Torsion perturbation (Active Driver)
        use_torsion: Enable torsion perturbation (default: True)
        beta_base: Base torsion strength (default: 0.3)
        
        # Adaptive mixing
        target_perturbation_ratio: Target perturbation ratio (default: (0.4, 0.6))
        safety_clip_factor: Safety clipping factor (default: 3.0)
        
        # Training phase
        total_steps: Total training steps for phase detection (default: None)
    """
    
    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        momentum: float = 0.95,
        # Hessian preconditioning
        hessian_update_freq: int = 10,
        gamma: float = 0.01,
        # Spectral projection
        ns_steps: int = 5,
        projection_strength: float = 0.3,
        # Soliton perturbation
        use_soliton: bool = True,
        num_solitons: int = 4,
        soliton_ns_steps: int = 3,
        coherence_threshold: float = 0.1,
        alpha_base: float = 0.5,
        alpha_max: float = 1.0,
        # Torsion perturbation
        use_torsion: bool = True,
        beta_base: float = 0.3,
        # Adaptive mixing
        target_perturbation_ratio: Tuple[float, float] = (0.4, 0.6),
        safety_clip_factor: float = 3.0,
        # Training phase
        total_steps: Optional[int] = 20000,
    ):
        # Validate hyperparameters (Task 9.3)
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0 <= betas[0] < 1 and 0 <= betas[1] < 1):
            raise ValueError(f"Invalid betas: {betas}")
        if not (0 < target_perturbation_ratio[0] < target_perturbation_ratio[1] < 1):
            raise ValueError(f"Invalid target_perturbation_ratio: {target_perturbation_ratio}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            hessian_update_freq=hessian_update_freq,
            gamma=gamma,
            ns_steps=ns_steps,
            projection_strength=projection_strength,
            use_soliton=use_soliton,
            num_solitons=num_solitons,
            soliton_ns_steps=soliton_ns_steps,
            coherence_threshold=coherence_threshold,
            alpha_base=alpha_base,
            alpha_max=alpha_max,
            use_torsion=use_torsion,
            beta_base=beta_base,
            target_perturbation_ratio=target_perturbation_ratio,
            safety_clip_factor=safety_clip_factor,
        )
        super().__init__(params, defaults)
        
        self.global_step = 0
        self.phase_detector = PhaseDetector(total_steps)
        
        print(f"[Zeta-PTD] Perturbation-Driven Optimizer")
        print(f"  - Target: ≤ 2x Muon baseline")
        print(f"  - Soliton: {use_soliton}, Torsion: {use_torsion}")
        print(f"  - Total steps: {total_steps}")
    
    @torch.no_grad()
    def step(self, closure=None):
        """Execute single optimization step (Task 7)"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        self.global_step += 1
        
        # Get phase information (Task 7.1)
        phase_info = self.phase_detector.get_phase_info(self.global_step)
        
        for group in self.param_groups:
            lr = group['lr']
            eps = group['eps']
            wd = group['weight_decay']
            momentum = group['momentum']
            hessian_freq = group['hessian_update_freq']
            gamma = group['gamma']
            ns_steps = group['ns_steps']
            projection_strength = group['projection_strength']
            
            use_soliton = group['use_soliton']
            num_solitons = group['num_solitons']
            soliton_ns_steps = group['soliton_ns_steps']
            coherence_threshold = group['coherence_threshold']
            alpha_base = group['alpha_base']
            alpha_max = group['alpha_max']
            
            use_torsion = group['use_torsion']
            beta_base = group['beta_base']
            
            target_ratio = group['target_perturbation_ratio']
            safety_clip = group['safety_clip_factor']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Check for NaN/Inf in gradient (Task 10.1)
                if not torch.isfinite(grad).all():
                    print(f"[Zeta-PTD] Warning: NaN/Inf in gradient, skipping update")
                    continue
                
                state = self.state[p]
                
                # ========================================
                # 2D+ parameters: Perturbation-Driven
                # ========================================
                if p.ndim >= 2:
                    if p.ndim > 2:
                        grad_2d = grad.view(grad.size(0), -1)
                        p_2d = p.view(p.size(0), -1)
                    else:
                        grad_2d = grad
                        p_2d = p
                    
                    # Initialize state
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(grad_2d)
                        state["hessian"] = torch.ones_like(grad_2d)
                        state["step"] = 0
                        state["soliton_state"] = None
                        state["torsion_state"] = None
                        state["metrics"] = {}
                    
                    state["step"] += 1
                    exp_avg = state["exp_avg"]
                    hessian = state["hessian"]
                    
                    # Momentum update
                    exp_avg.mul_(momentum).add_(grad_2d, alpha=1.0 - momentum)
                    
                    # Hessian estimation (Sophia-style) (Task 5)
                    if state["step"] % hessian_freq == 0:
                        hessian.mul_(1 - gamma).addcmul_(grad_2d, grad_2d, value=gamma)
                    
                    # Hessian preconditioning with phase-dependent boundary scaling (Task 5.1)
                    precond_avg = exp_avg / (hessian.sqrt() + eps)
                    precond_avg = precond_avg * phase_info.boundary_scale
                    
                    # Newton-Schulz orthogonalization (Muon core)
                    sign_update = torch.sign(precond_avg)
                    ortho = zeropower_via_newtonschulz5(sign_update, steps=ns_steps, eps=eps)
                    
                    m, n = p_2d.shape
                    base_scale = 0.2 * math.sqrt(m * n) / (ortho.norm() + eps)
                    base_direction = ortho * base_scale
                    
                    # ========================================
                    # Soliton Perturbation (Active Driver) (Task 2)
                    # ========================================
                    xi_soliton = torch.zeros_like(base_direction)
                    
                    if use_soliton:
                        try:
                            grad_norm = grad_2d.norm().item()
                            xi_soliton, soliton_state = compute_soliton_perturbation(
                                W=p_2d,
                                G=base_direction,
                                grad_norm=grad_norm,
                                phase_info=phase_info,
                                num_solitons=num_solitons,
                                soliton_ns_steps=soliton_ns_steps,
                                coherence_threshold=coherence_threshold,
                                alpha_base=alpha_base,
                                alpha_max=alpha_max,
                                eps=eps
                            )
                            state["soliton_state"] = soliton_state
                            
                            # Check for NaN/Inf (Task 10.1)
                            if not torch.isfinite(xi_soliton).all():
                                xi_soliton = torch.zeros_like(base_direction)
                        except Exception as e:
                            print(f"[Zeta-PTD] Soliton computation failed: {e}")
                            xi_soliton = torch.zeros_like(base_direction)
                    
                    # ========================================
                    # Torsion Perturbation (Active Driver) (Task 3)
                    # ========================================
                    xi_torsion = torch.zeros_like(base_direction)
                    
                    if use_torsion and state["step"] > 1:
                        try:
                            xi_torsion, torsion_state = compute_torsion_perturbation(
                                M=exp_avg,
                                G=base_direction,
                                hessian=hessian,
                                phase_info=phase_info,
                                beta_base=beta_base,
                                eps=eps
                            )
                            state["torsion_state"] = torsion_state
                            
                            # Check for NaN/Inf (Task 10.1)
                            if not torch.isfinite(xi_torsion).all():
                                xi_torsion = torch.zeros_like(base_direction)
                        except Exception as e:
                            print(f"[Zeta-PTD] Torsion computation failed: {e}")
                            xi_torsion = torch.zeros_like(base_direction)
                    
                    # ========================================
                    # Adaptive Mixing (Task 4)
                    # ========================================
                    combined_update, metrics = adaptive_mix_updates(
                        base_update=base_direction,
                        xi_soliton=xi_soliton,
                        xi_torsion=xi_torsion,
                        target_ratio=target_ratio,
                        safety_clip_factor=safety_clip,
                        eps=eps
                    )
                    
                    # ========================================
                    # Spectral Projection (Passive Boundary) (Task 6)
                    # ========================================
                    ortho_combined = zeropower_via_newtonschulz5(combined_update, steps=ns_steps, eps=eps)
                    gamma_proj = projection_strength * phase_info.boundary_scale
                    projected_update = (1.0 - gamma_proj) * combined_update + gamma_proj * ortho_combined
                    
                    # Store metrics (Task 7.3)
                    metrics["phase_ratio"] = phase_info.ratio
                    metrics["phase_name"] = phase_info.phase_name
                    metrics["perturbation_scale"] = phase_info.perturbation_scale
                    metrics["boundary_scale"] = phase_info.boundary_scale
                    state["metrics"] = metrics
                    
                    # Apply update
                    if wd > 0:
                        p.data.mul_(1 - lr * wd)
                    p.data.add_(projected_update.view_as(p), alpha=-lr)
                
                # ========================================
                # 1D parameters: Standard momentum
                # ========================================
                else:
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                    
                    state["step"] += 1
                    exp_avg = state["exp_avg"]
                    
                    exp_avg.mul_(momentum).add_(grad, alpha=1.0 - momentum)
                    
                    norm = exp_avg.norm() + eps
                    normalized_update = exp_avg / norm
                    
                    if wd > 0:
                        p.data.mul_(1 - lr * wd)
                    
                    p.data.add_(normalized_update, alpha=-lr)
        
        return loss


# Alias for convenience
Zeta = ZetaPerturbationDriven
ZetaPTD = ZetaPerturbationDriven
