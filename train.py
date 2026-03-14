"""
train.py — Training Loops for All Three Optimizers
Each training function returns a standardised results dictionary:
    {
        "losses"   : [float, ...],   — per-epoch training MSE
        "times"    : [float, ...],   — cumulative wall-clock seconds
        "final_mse": float,          — final training MSE
        "final_r2" : float,          — R² on the VALIDATION set
    }
This uniform interface makes comparison in evaluate.py trivial.
"""

import time
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
import config
from model import CompactMLP
from optim_gauss_newton import GaussNewtonOptimizer

#  Helper: Compute Validation R²
def _val_r2(model, X_val, y_val):
    """Compute R² Score On The Validation Set."""
    model.eval()
    with torch.no_grad():
        preds = model(X_val).squeeze(-1)
    return r2_score(y_val.cpu().numpy(), preds.cpu().numpy())

# 1. GAUSS-NEWTON
def train_gauss_newton(X_train, y_train, X_val, y_val, input_dim):
    """
    Train using the custom Gauss-Newton optimizer (full-batch or sub-batch).
    The optimizer computes the exact Jacobian J, forms JᵀJ + λI, and
    solves for the parameter update every epoch.
    """
    config.set_seed()
    model = CompactMLP(input_dim, config.HIDDEN_1, config.HIDDEN_2,
                       config.OUTPUT_DIM).to(config.DEVICE)
    optimizer = GaussNewtonOptimizer(model)

    losses, times = [], []
    t0 = time.time()

    print("\n" + "=" * 65)
    print("  GAUSS-NEWTON TRAINING")
    print("=" * 65)

    for epoch in range(1, config.GN_EPOCHS + 1):
        loss = optimizer.step(X_train, y_train)
        elapsed = time.time() - t0
        losses.append(loss)
        times.append(elapsed)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  [GN]  Epoch {epoch:>4d}/{config.GN_EPOCHS}  │  "
                  f"MSE = {loss:.6f}  │  λ = {optimizer.lam:.2e}  │  "
                  f"Time = {elapsed:.2f}s")

    final_r2 = _val_r2(model, X_val, y_val)
    print(f"  [GN]  Final Val R² = {final_r2:.4f}")

    return model, {
        "losses":    losses,
        "times":     times,
        "final_mse": losses[-1],
        "final_r2":  final_r2,
    }

# 2. ADAM
def train_adam(X_train, y_train, X_val, y_val, input_dim):
    """
    Train using standard torch.optim.Adam with mini-batch SGD.
    Adam is an adaptive first-order method that maintains per-parameter
    moving averages of gradients and squared gradients.
    """
    config.set_seed()
    model = CompactMLP(input_dim, config.HIDDEN_1, config.HIDDEN_2,
                       config.OUTPUT_DIM).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.ADAM_LR)
    criterion = nn.MSELoss()

    losses, times = [], []
    t0 = time.time()
    N = X_train.shape[0]
    batch_size = config.BATCH_SIZE

    print("\n" + "=" * 65)
    print("  ADAM TRAINING  (First-Order Baseline)")
    print("=" * 65)

    for epoch in range(1, config.ADAM_EPOCHS + 1):
        model.train()
        perm = torch.randperm(N, device=config.DEVICE)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            X_b = X_train[idx]
            y_b = y_train[idx]

            optimizer.zero_grad()
            preds = model(X_b).squeeze(-1)
            loss = criterion(preds, y_b)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        elapsed = time.time() - t0
        losses.append(avg_loss)
        times.append(elapsed)

        if epoch % 20 == 0 or epoch == 1:
            print(f"  [Adam] Epoch {epoch:>4d}/{config.ADAM_EPOCHS}  │  "
                  f"MSE = {avg_loss:.6f}  │  Time = {elapsed:.2f}s")

    final_r2 = _val_r2(model, X_val, y_val)
    print(f"  [Adam] Final Val R² = {final_r2:.4f}")

    return model, {
        "losses":    losses,
        "times":     times,
        "final_mse": losses[-1],
        "final_r2":  final_r2,
    }

# 3. L-BFGS
def train_lbfgs(X_train, y_train, X_val, y_val, input_dim):
    """
    Train using torch.optim.LBFGS — a quasi-Newton method.
    L-BFGS approximates the inverse Hessian from the last `history_size`
    gradient differences. It uses a closure-based API and works best
    with full-batch training.
    """
    config.set_seed()
    model = CompactMLP(input_dim, config.HIDDEN_1, config.HIDDEN_2,
                       config.OUTPUT_DIM).to(config.DEVICE)
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=config.LBFGS_LR,
        max_iter=config.LBFGS_MAX_ITER,
        history_size=config.LBFGS_HISTORY,
        line_search_fn="strong_wolfe",
    )
    criterion = nn.MSELoss()

    losses, times = [], []
    t0 = time.time()

    print("\n" + "=" * 65)
    print("  L-BFGS TRAINING  (Second-Order Baseline)")
    print("=" * 65)

    for epoch in range(1, config.LBFGS_EPOCHS + 1):
        model.train()
        current_loss = [0.0]

        def closure():
            """L-BFGS Requires A Closure That Re-Evaluates The Loss."""
            optimizer.zero_grad()
            preds = model(X_train).squeeze(-1)
            loss = criterion(preds, y_train)
            loss.backward()
            current_loss[0] = loss.item()
            return loss

        optimizer.step(closure)
        elapsed = time.time() - t0
        losses.append(current_loss[0])
        times.append(elapsed)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  [LBFGS] Epoch {epoch:>4d}/{config.LBFGS_EPOCHS}  │  "
                  f"MSE = {current_loss[0]:.6f}  │  Time = {elapsed:.2f}s")

    final_r2 = _val_r2(model, X_val, y_val)
    print(f"  [LBFGS] Final Val R² = {final_r2:.4f}")

    return model, {
        "losses":    losses,
        "times":     times,
        "final_mse": losses[-1],
        "final_r2":  final_r2,
    }
