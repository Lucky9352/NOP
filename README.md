# Efficiency of Second-Order Optimization: A Comparative Implementation

A custom, from-scratch implementation of the **Gauss-Newton Algorithm** (with Levenberg-Marquardt damping) for non-linear model fitting, built entirely in PyTorch using the functional auto-differentiation API (`torch.func`). 

This repository contains the complete machine learning pipeline and an interactive Streamlit dashboard developed for a university 3rd-year mini-project in Numerical Optimization.

## 🚀 Project Overview

First-order stochastic methods (like Adam) often struggle with pathological curvature in complex non-linear loss landscapes. This project demonstrates the sheer mathematical power of second-order optimization by building a custom Gauss-Newton optimizer. 

By approximating the Hessian matrix via the Jacobian of the residuals ($J^T J$), the custom optimizer perfectly interpolates the training data (Ames Housing dataset) in a fraction of the epochs required by standard first-order baselines.

### ✨ Key Features
* **Custom Mathematical Engine:** A `GaussNewtonOptimizer` built from scratch without relying on pre-packaged second-order solvers.
* **Advanced PyTorch Capabilities:** Utilizes `torch.func.jacrev` and `vmap` for highly efficient, vector-batched Jacobian computation on the GPU.
* **Levenberg-Marquardt Damping:** Includes an adaptive $\lambda$ scheduling mechanism to guarantee matrix invertibility and robust convergence.
* **Interactive Real-Time UI:** A beautifully designed Streamlit dashboard with a dedicated [Interface Guide](INTERFACE_GUIDE.md) explaining real-time valuations and accuracy margins.

## 📁 Repository Structure

```text
├── config.py                 # Hyperparameters, seeds, and device auto-detection
├── data_loader.py            # Data preprocessing, imputation, and scaling
├── model.py                  # Compact MLP architecture with Xavier initialization
├── optim_gauss_newton.py     # The custom Gauss-Newton optimization engine
├── train.py                  # Training loops for Gauss-Newton, Adam, and L-BFGS
├── evaluate.py               # Evaluation metrics (MSE, R²) and Matplotlib plots
├── main.py                   # The CLI entry point for the training pipeline
├── app.py                    # The interactive Streamlit dashboard
├── INTERFACE_GUIDE.md        # Detailed guide for the UI outputs and metrics
├── requirements.txt          # Python package dependencies
├── data/                     # Directory for train.csv and test.csv
├── models/                   # Auto-generated directory for saved .pth and .pkl artifacts
└── results/                  # Auto-generated directory for publication-ready PNG plots
```

## ⚙️ Installation & Setup

This pipeline is optimized for local GPU execution (CUDA 12.x recommended).

**1. Clone or extract the repository:**
Ensure `train.csv` and `test.csv` (Ames Housing Dataset) are placed inside the `data/` directory.

**2. Create and activate a virtual environment:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install PyTorch with CUDA support:**
*(Ensure your PyTorch version matches your local CUDA installation for optimal Jacobian computation speeds)*

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**4. Install remaining dependencies:**

```bash
pip install -r requirements.txt
```

## 💻 Usage

This project is divided into two parts: the backend training pipeline and the frontend interactive dashboard.

### Part 1: Run the Training Pipeline

Execute the main script to clean the data, train all three models (Gauss-Newton, Adam, L-BFGS), generate the comparative graphs, and save the model artifacts.

```bash
python3 main.py
```

*Note: The Gauss-Newton optimizer is highly computationally intensive. Execution on an NVIDIA RTX 40-series GPU takes approximately 3.5 minutes.*

### Part 2: Launch the Interactive Dashboard

Once `main.py` has finished and populated the `models/` directory, launch the Streamlit app to interact with the trained models in real-time.

```bash
streamlit run app.py
```

This will open a local web server (usually at `http://localhost:8501`) where you can adjust housing features via sliders and view the exact prediction deltas between the optimizers.

## 📊 Results Summary

* **Gauss-Newton** reached a Train MSE of `0.000000` in just 10 epochs, demonstrating the aggressive convergence capabilities of curvature-guided steps.
* **Adam** failed to reach absolute zero even after 200 epochs due to the inherent stochasticity of mini-batch gradient noise.
* The computational bottleneck of inverting the $J^T J$ matrix highlights the modern trade-off between convergence efficiency (few epochs) and computational complexity (high wall-clock time).

For a complete breakdown of the mathematical formulations, architecture, and empirical analysis, please refer to the attached `REPORT.md` technical documentation.
