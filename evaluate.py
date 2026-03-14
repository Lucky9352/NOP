"""
evaluate.py — Visualization & Empirical Analysis
Computes MSE and R² on the validation set for each optimizer, then
generates four publication-ready Matplotlib plots:
    1. Training Loss vs. Epochs       — Shows convergence speed per epoch
    2. Training Loss vs. Wall-clock   — Shows real-world efficiency
    3. Final MSE Bar Chart            — Direct numerical comparison
    4. R² Score Bar Chart             — Interpretable regression quality
All plots are saved as PNG files in the results/ directory.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import config

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "figure.dpi": 150,
})

COLORS = {
    "Gauss-Newton": "#e63946",
    "Adam":         "#457b9d",
    "L-BFGS":       "#2a9d8f",
}

def compute_val_metrics(model, X_val, y_val):
    """
    Evaluate a trained model on the validation set.
    Returns
    -------
    mse : float   — Mean Squared Error
    r2  : float   — R² coefficient of determination
    """
    model.eval()
    with torch.no_grad():
        preds = model(X_val).squeeze(-1)
    mse = nn.functional.mse_loss(preds, y_val).item()
    r2  = r2_score(y_val.cpu().numpy(), preds.cpu().numpy())
    return mse, r2

def evaluate(results: dict, models: dict,
             X_val: torch.Tensor, y_val: torch.Tensor) -> None:
    """
    Generate all plots and print a comparison table.
    Parameters
    ----------
    results : dict[str, dict]
        Keys = optimizer names.
        Values = {"losses", "times", "final_mse", "final_r2"}.
    models  : dict[str, nn.Module]
        Trained model for each optimizer.
    X_val, y_val : tensors
        Validation data.
    """
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    names = list(results.keys())

    val_metrics = {}
    for name in names:
        mse, r2 = compute_val_metrics(models[name], X_val, y_val)
        val_metrics[name] = {"mse": mse, "r2": r2}

    # PLOT 1 — Training Loss vs. Epochs
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in names:
        ax.plot(range(1, len(results[name]["losses"]) + 1),
                results[name]["losses"],
                label=name, color=COLORS.get(name), linewidth=2.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training MSE Loss")
    ax.set_title("Training Loss vs. Epochs", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(config.RESULTS_DIR, "loss_vs_epochs.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[evaluate] ✓ Saved {path}")

    # PLOT 2 — Training Loss vs. Wall-clock Time
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in names:
        ax.plot(results[name]["times"], results[name]["losses"],
                label=name, color=COLORS.get(name), linewidth=2.2)
    ax.set_xlabel("Wall-clock Time (seconds)")
    ax.set_ylabel("Training MSE Loss")
    ax.set_title("Training Loss vs. Wall-clock Time", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(config.RESULTS_DIR, "loss_vs_time.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[evaluate] ✓ Saved {path}")

    # PLOT 3 — Final Validation MSE (Grouped Bar Chart)
    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(names))
    mse_vals = [val_metrics[n]["mse"] for n in names]
    bars = ax.bar(x_pos, mse_vals,
                  color=[COLORS.get(n, "#999") for n in names],
                  edgecolor="black", linewidth=0.8, width=0.5)
    for bar, v in zip(bars, mse_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.4f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylabel("Validation MSE")
    ax.set_title("Final Validation MSE Comparison", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(config.RESULTS_DIR, "final_mse_bar.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[evaluate] ✓ Saved {path}")

    # PLOT 4 — R² Score (Grouped Bar Chart)
    fig, ax = plt.subplots(figsize=(8, 5))
    r2_vals = [val_metrics[n]["r2"] for n in names]
    bars = ax.bar(x_pos, r2_vals,
                  color=[COLORS.get(n, "#999") for n in names],
                  edgecolor="black", linewidth=0.8, width=0.5)
    for bar, v in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.4f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylabel("R² Score")
    ax.set_title("Validation R² Score Comparison", fontweight="bold")
    ax.set_ylim(min(0, min(r2_vals) - 0.1), 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(config.RESULTS_DIR, "r2_bar.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[evaluate] ✓ Saved {path}")

    # CONSOLE SUMMARY TABLE
    header = (f"{'Optimizer':<16} │ {'Epochs':>7} │ {'Time (s)':>10} │ "
              f"{'Train MSE':>12} │ {'Val MSE':>12} │ {'Val R²':>10}")
    sep = "─" * len(header)

    print("\n" + sep)
    print("  COMPARATIVE RESULTS")
    print(sep)
    print(header)
    print(sep)

    for name in names:
        r  = results[name]
        vm = val_metrics[name]
        print(f"{name:<16} │ {len(r['losses']):>7d} │ "
              f"{r['times'][-1]:>10.2f} │ {r['final_mse']:>12.6f} │ "
              f"{vm['mse']:>12.6f} │ {vm['r2']:>10.4f}")

    print(sep + "\n")
