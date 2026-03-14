"""
main.py — Execution Entry Point
Orchestrates the full machine learning pipeline:
    1. Set reproducible seeds
    2. Load & preprocess the House Prices dataset
    3. Train three models:
       a) Custom Gauss-Newton optimizer (from scratch)
       b) Adam (first-order baseline)
       c) L-BFGS (second-order baseline)
    4. Evaluate all three and generate comparison plots
Usage:
    python3 main.py
"""

import os
import torch
import config
from data_loader import load_data
from train import train_gauss_newton, train_adam, train_lbfgs
from evaluate import evaluate

def main():
    config.set_seed()
    print("=" * 65)
    print("  Gauss-Newton ML Pipeline")
    print("  Efficiency Of Second-Order Optimization")
    print("=" * 65)
    print(f"  Device : {config.DEVICE}")
    print(f"  Seed   : {config.SEED}")
    print("=" * 65)

    X_train, y_train, X_val, y_val, input_dim, scaler_y = load_data()
    config.INPUT_DIM = input_dim

    gn_model, gn_results = train_gauss_newton(
        X_train, y_train, X_val, y_val, input_dim
    )

    adam_model, adam_results = train_adam(
        X_train, y_train, X_val, y_val, input_dim
    )

    lbfgs_model, lbfgs_results = train_lbfgs(
        X_train, y_train, X_val, y_val, input_dim
    )

    all_results = {
        "Gauss-Newton": gn_results,
        "Adam":         adam_results,
        "L-BFGS":       lbfgs_results,
    }
    all_models = {
        "Gauss-Newton": gn_model,
        "Adam":         adam_model,
        "L-BFGS":       lbfgs_model,
    }

    evaluate(all_results, all_models, X_val, y_val)

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    model_filenames = {
        "Gauss-Newton": "gn_model.pth",
        "Adam":         "adam_model.pth",
        "L-BFGS":       "lbfgs_model.pth",
    }
    for name, model in all_models.items():
        save_path = os.path.join(config.MODELS_DIR, model_filenames[name])
        torch.save({
            "state_dict": model.state_dict(),
            "input_dim":  input_dim,
            "hidden_1":   config.HIDDEN_1,
            "hidden_2":   config.HIDDEN_2,
            "output_dim":  config.OUTPUT_DIM,
        }, save_path)
        print(f"[main] ✓ Saved {name} Model → {save_path}")

    print("\n✅  Pipeline Complete. Models → models/  |  Plots → results/\n")

if __name__ == "__main__":
    main()
