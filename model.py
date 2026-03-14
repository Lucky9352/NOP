"""
model.py — Compact Multi-Layer Perceptron for Non-Linear Regression
Architecture:
    Input(D) → Linear(64) → ReLU → Linear(32) → ReLU → Linear(1)

Design rationale:
    Gauss-Newton requires computing and inverting the matrix (JᵀJ + λI)
    of size (P × P) where P = total number of parameters.

    For this network:
        P = D×64 + 64 + 64×32 + 32 + 32×1 + 1
          ≈ 64D + 64 + 2048 + 32 + 32 + 1
          ≈ 64D + 2177

    With ~200 features after one-hot encoding, P ≈ 15,000.
    The (JᵀJ) matrix is therefore 15k × 15k — tractable for modern GPUs
    but large enough to demonstrate second-order optimisation.

    Xavier initialisation ensures gradients and activations don't
    explode or vanish, which is critical for Gauss-Newton's convergence.
"""

import torch
import torch.nn as nn

class CompactMLP(nn.Module):
    """
    A compact Multi-Layer Perceptron for scalar regression.
    Parameters
    ----------
    input_dim  : int   — number of input features
    hidden_1   : int   — width of first hidden layer  (default: 64)
    hidden_2   : int   — width of second hidden layer (default: 32)
    output_dim : int   — number of outputs            (default: 1)
    """

    def __init__(self, input_dim: int, hidden_1: int = 64,
                 hidden_2: int = 32, output_dim: int = 1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, output_dim),
        )
        self._init_weights()

    def _init_weights(self):
        """
        Xavier Uniform Initialisation.
        For a layer with fan_in and fan_out, weights are drawn from
        U(-√(6/(fan_in+fan_out)), +√(6/(fan_in+fan_out))).
        This keeps the variance of activations roughly constant across
        layers, which is especially important for second-order methods
        that are sensitive to the scale of the Jacobian.
        """
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Parameters
        ----------
        x : Tensor of shape (batch_size, input_dim)
        Returns
        -------
        Tensor of shape (batch_size, 1)  — predicted target value
        """
        return self.net(x)

# UTILITY: Parameter Flatten / Unflatten
def flatten_params(model: nn.Module) -> torch.Tensor:
    """
    Flatten all model parameters into a single 1-D tensor.
    Returns a detached clone so modifications don't affect the model.
    """
    return torch.cat([p.detach().clone().view(-1) for p in model.parameters()])

def unflatten_params(model: nn.Module, flat: torch.Tensor) -> None:
    """
    Write a 1-D flat parameter vector back into the model, **in-place**.
    Parameters
    ----------
    model : nn.Module
    flat  : Tensor of shape (P,) where P = sum of all param elements
    """
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat[offset:offset + numel].view_as(p))
        offset += numel
