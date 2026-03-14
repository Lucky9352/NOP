"""
optim_gauss_newton.py — Custom Gauss-Newton Optimizer
 ┌─────────────────────────────────────────────────────────────────────┐
 │                    MATHEMATICAL FORMULATION                         │
 ├─────────────────────────────────────────────────────────────────────┤
 │                                                                     │
 │  We formulate MLP training as a Non-Linear Least Squares problem:   │
 │                                                                     │
 │      min_θ  L(θ)  =  ½ ‖r(θ)‖²  =  ½ Σᵢ [f(xᵢ; θ) − yᵢ]²            │
 │                                                                     │
 │  where:                                                             │
 │      r(θ)  = model(X; θ) − y     (residual vector, ∈ ℝᴺ)            │
 │      θ     = all model parameters  (∈ ℝᴾ)                           │
 │      N     = number of samples                                      │
 │      P     = number of parameters                                   │
 │                                                                     │
 │  GRADIENT of the loss:                                              │
 │      ∇L = Jᵀ r                                                      │
 │                                                                     │
 │  where J ∈ ℝᴺˣᴾ is the Jacobian:  J_{ij} = ∂rᵢ/∂θⱼ                  │
 │                                                                     │
 │  HESSIAN (exact):                                                   │
 │      ∇²L = JᵀJ + Σᵢ rᵢ ∇²rᵢ                                         │
 │                                                                     │
 │  GAUSS-NEWTON APPROXIMATION:                                        │
 │      ∇²L  ≈  JᵀJ      (drop the 2nd-order term Σᵢ rᵢ ∇²rᵢ)          │
 │                                                                     │
 │  This approximation is excellent when residuals are small,          │
 │  which is exactly the regime we approach during convergence.        │
 │                                                                     │
 │  LEVENBERG-MARQUARDT DAMPING:                                       │
 │      H  =  JᵀJ + λI                                                 │
 │                                                                     │
 │  Adding λI ensures H is positive definite (always invertible).      │
 │  When λ is large, the step becomes gradient-descent-like.           │
 │  When λ is small, we get pure Gauss-Newton.                         │
 │                                                                     │
 │  UPDATE RULE:                                                       │
 │      Δθ  =  H⁻¹ Jᵀ r  =  (JᵀJ + λI)⁻¹ Jᵀ r                          │
 │      θ_{n+1}  =  θ_n − Δθ                                           │
 │                                                                     │
 │  We solve H Δθ = Jᵀr via torch.linalg.solve (LU decomposition)      │
 │  instead of explicitly computing H⁻¹ (more numerically stable).     │
 │                                                                     │
 │  JACOBIAN COMPUTATION:                                              │
 │      We use PyTorch's functional API (torch.func):                  │
 │      - jacrev: reverse-mode autodiff for ∂output/∂params            │
 │      - vmap:   vectorise over the batch dimension                   │
 │      - functional_call: evaluate model with an explicit params dict │
 └─────────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
from torch.func import functional_call, vmap, jacrev
import config

class GaussNewtonOptimizer:
    """
    Custom Gauss-Newton optimizer with Levenberg-Marquardt (LM) damping.
    Parameters
    ----------
    model    : nn.Module — the compact MLP to optimise
    lam      : float     — initial damping factor λ
    lam_up   : float     — multiply λ by this on a BAD step  (loss increased)
    lam_down : float     — multiply λ by this on a GOOD step (loss decreased)
    """

    def __init__(self, model: nn.Module,
                 lam: float = config.GN_LAMBDA,
                 lam_up: float = config.GN_LAMBDA_UP,
                 lam_down: float = config.GN_LAMBDA_DOWN):
        self.model    = model
        self.lam      = lam
        self.lam_up   = lam_up
        self.lam_down = lam_down

    def step(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """
        Perform ONE Gauss-Newton parameter update:
            θ ← θ − (JᵀJ + λI)⁻¹ Jᵀr

        Parameters
        ----------
        X : Tensor (N, D)   — input features  (full training set or sub-batch)
        y : Tensor (N,)     — target values

        Returns
        -------
        loss : float  — MSE *before* the update (for logging)
        """
        model = self.model
        model.eval()

        batch_size = config.GN_BATCH_SIZE
        if batch_size is not None and batch_size < X.shape[0]:
            idx = torch.randperm(X.shape[0], device=X.device)[:batch_size]
            X_batch = X[idx]
            y_batch = y[idx]
        else:
            X_batch = X
            y_batch = y

        N = X_batch.shape[0]

        with torch.no_grad():
            preds = model(X_batch).squeeze(-1)
            residuals = (preds - y_batch).detach()
            loss = 0.5 * torch.mean(residuals ** 2).item()

        params_dict = {
            n: p.detach().requires_grad_(True)
            for n, p in model.named_parameters()
        }

        def f_single(params, x_single):
            """Predict For One Sample: x_single → scalar."""
            return functional_call(
                model, params, (x_single.unsqueeze(0),)
            ).squeeze()

        jac_fn = vmap(jacrev(f_single), in_dims=(None, 0))

        with torch.enable_grad():
            jac_dict = jac_fn(params_dict, X_batch)

        J_blocks = []
        for name in params_dict:
            J_p = jac_dict[name].detach()
            J_blocks.append(J_p.reshape(N, -1))
        J = torch.cat(J_blocks, dim=1)

        del jac_dict, J_blocks, params_dict
        if X.is_cuda:
            torch.cuda.empty_cache()

        P = J.shape[1]
        MAX_LM_RETRIES = 10

        Jt = J.t()
        g  = Jt @ residuals

        saved_params = [p.data.clone() for p in model.parameters()]

        best_delta = None
        accepted = False

        for attempt in range(MAX_LM_RETRIES):
            JtJ = Jt @ J
            damping = self.lam * torch.eye(P, device=X.device,
                                           dtype=X.dtype)
            H = JtJ + damping

            try:
                delta_theta = torch.linalg.solve(H, g)
            except torch.linalg.LinAlgError:
                self.lam *= self.lam_up
                continue

            offset = 0
            for p, sp in zip(model.parameters(), saved_params):
                numel = p.numel()
                p.data.copy_(sp - delta_theta[offset:offset + numel].view_as(p))
                offset += numel

            with torch.no_grad():
                new_preds = model(X_batch).squeeze(-1)
                new_loss  = 0.5 * torch.mean((new_preds - y_batch) ** 2).item()

            if new_loss < loss:
                self.lam *= self.lam_down
                accepted = True
                break
            else:
                self.lam *= self.lam_up

        if not accepted:
            pass

        self.lam = max(1e-10, min(self.lam, 1e6))

        del J, Jt, g, H, JtJ, damping, saved_params, residuals
        if X.is_cuda:
            torch.cuda.empty_cache()

        model.train()
        return loss
