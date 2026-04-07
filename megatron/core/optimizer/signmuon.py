# Copyright (c) 2025, AIGCode CORPORATION. All rights reserved. 
# @author: chenqiuwu@aigcode.net

import math
import torch
import torch.optim as optim

def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration for spectral normalization and orthogonalization.
    """
    assert len(G.shape) == 2

    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()

    if G.size(0) > G.size(1):
        X = X.T

    # Spectral normalization
    norm = X.norm() + eps
    X = X / norm

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T

    return X

class SignMuon(optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.95, 0.99), 
                 weight_decay=1e-2, capturable: bool = False, 
                 ns_steps=5, eps: float = 1e-8):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay,
                        capturable=capturable, ns_steps=ns_steps, eps=eps)
        super(SignMuon, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('capturable', False)
            
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, _ = group['betas']
            weight_decay = group['weight_decay']
            capturable = group['capturable']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad 
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) if capturable else torch.tensor(0.)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                exp_avg = state['exp_avg']
                step_t = state['step']

                step_t += 1
                p.data.mul_(1 - lr * weight_decay)
                
                # Momentum update
                exp_avg.mul_(beta1).add_(grad)
                
                if p.ndim >= 2:
                    m_in = exp_avg
                    if p.ndim > 2:
                        m_in = m_in.view(m_in.size(0), -1)
                    
                    # Newton-Schulz Orthogonalization
                    ortho_direction = zeropower_via_newtonschulz5(m_in, steps=ns_steps)
                    if p.ndim > 2:
                        ortho_direction = ortho_direction.view_as(exp_avg)
                    
                    # Sign operation after orthogonalization
                    update_val = ortho_direction.sign()
                else:
                    # Simple sign update for 1D params (Bias/LayerNorm)
                    update_val = exp_avg.sign()

                if capturable:
                    p.data.addcmul_(update_val, lr, value=-1)
                else:
                    p.data.add_(update_val, alpha=-lr)
        
        return loss
