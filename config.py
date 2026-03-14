"""
config.py — Centralized Configuration for the Gauss-Newton ML Pipeline
Handles device auto-detection, reproducible seeding, and all hyperparameters
for training, evaluation, and the custom Gauss-Newton optimizer.
"""

import torch
import numpy as np
import random
import os

# REPRODUCIBILITY
SEED = 42

def set_seed(seed: int = SEED) -> None:
    """
    Set random seeds for full reproducibility across all libraries.
    We seed Python's `random`, NumPy, and PyTorch (CPU + CUDA).
    Additionally, we enforce deterministic cuDNN behaviour.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# DEVICE AUTO-DETECTION
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DATA PATHS & SPLIT
DATA_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TRAIN_CSV      = "train.csv"
TEST_CSV       = "test.csv"
TEST_SPLIT     = 0.2
BATCH_SIZE     = 64

# MODEL ARCHITECTURE
INPUT_DIM      = None
HIDDEN_1       = 64
HIDDEN_2       = 32
OUTPUT_DIM     = 1

GN_EPOCHS       = 50
GN_LAMBDA       = 1e-3
GN_LAMBDA_UP    = 10.0
GN_LAMBDA_DOWN  = 0.1
GN_BATCH_SIZE   = None

# ADAM OPTIMIZER HYPERPARAMETERS
ADAM_EPOCHS     = 200
ADAM_LR         = 1e-3

# L-BFGS OPTIMIZER HYPERPARAMETERS
LBFGS_EPOCHS    = 50
LBFGS_LR        = 1.0
LBFGS_MAX_ITER  = 20
LBFGS_HISTORY   = 10

# OUTPUT
RESULTS_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MODELS_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
