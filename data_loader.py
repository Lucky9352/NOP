"""
data_loader.py — House Prices Dataset Preprocessing Pipeline
IMPORTANT: This module does NOT download any data.
It expects train.csv and test.csv to already exist inside the data/ directory.
If the files are in the project root, this
module will automatically copy them into data/ on first run.
Pipeline steps:
    1. Read CSV
    2. Separate target (SalePrice) and drop Id
    3. Drop columns with >50% missing values or >20 unique categories
    4. Impute: median (numeric), mode (categorical)
    5. One-hot encode remaining categoricals
    6. StandardScaler: fit on train, transform both train & validation
    7. Convert to PyTorch float32 tensors on config.DEVICE
"""

import os
import shutil
import pandas as pd
import numpy as np
import torch
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import config

def save_preprocessing_artifacts(scaler_X, scaler_y, feature_columns,
                                  numeric_medians, categorical_modes):
    """
    Save all preprocessing artifacts to models/ so the Streamlit UI
    can transform new user inputs identically to training data.
    Saved files:
        models/scaler_X.pkl        — fitted feature StandardScaler
        models/scaler_y.pkl        — fitted target StandardScaler
        models/feature_columns.pkl — ordered list of one-hot column names
        models/numeric_medians.pkl — default median values per numeric col
        models/categorical_modes.pkl — default mode values per categorical col
    """
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    joblib.dump(scaler_X, os.path.join(config.MODELS_DIR, "scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(config.MODELS_DIR, "scaler_y.pkl"))
    joblib.dump(feature_columns, os.path.join(config.MODELS_DIR, "feature_columns.pkl"))
    joblib.dump(numeric_medians, os.path.join(config.MODELS_DIR, "numeric_medians.pkl"))
    joblib.dump(categorical_modes, os.path.join(config.MODELS_DIR, "categorical_modes.pkl"))
    print(f"[data_loader] ✓ Saved Preprocessing Artifacts To {config.MODELS_DIR}/")

def _ensure_data_dir() -> None:
    """
    Create data/ if needed. If CSVs are in the project root but not in
    data/, copy them over automatically (convenience for first run).
    """
    os.makedirs(config.DATA_DIR, exist_ok=True)
    project_root = os.path.dirname(os.path.abspath(__file__))

    for fname in [config.TRAIN_CSV, config.TEST_CSV]:
        dst = os.path.join(config.DATA_DIR, fname)
        src = os.path.join(project_root, fname)
        if not os.path.exists(dst):
            if os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"[data_loader] Copied {fname} → data/")
            else:
                raise FileNotFoundError(
                    f"Cannot Find {fname} In Project Root Or data/ Directory."
                    f"Please Place The House Prices CSV Files There."
                )

def load_data():
    """
    Full preprocessing pipeline for the House Prices dataset.
    Returns
    -------
    X_train : Tensor (N_train, D)  — training features
    y_train : Tensor (N_train,)    — training targets (standardised)
    X_val   : Tensor (N_val, D)    — validation features
    y_val   : Tensor (N_val,)      — validation targets (standardised)
    input_dim : int                — number of features after encoding
    scaler_y  : StandardScaler     — fitted target scaler (for inverse transform)
    """
    _ensure_data_dir()

    train_path = os.path.join(config.DATA_DIR, config.TRAIN_CSV)
    df = pd.read_csv(train_path)
    print(f"[data_loader] Loaded {train_path}  —  {df.shape[0]} rows, "
          f"{df.shape[1]} columns")

    y = df["SalePrice"].values.astype(np.float32)
    df = df.drop(columns=["SalePrice", "Id"])

    high_null = [c for c in df.columns if df[c].isnull().mean() > 0.5]

    high_card = [c for c in df.select_dtypes("object").columns
                 if df[c].nunique() > 20]

    drop_cols = list(set(high_null + high_card))
    df = df.drop(columns=drop_cols, errors="ignore")
    print(f"[data_loader] Dropped {len(drop_cols)} Noisy/High-Cardinality Cols")

    numeric_medians = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        median_val = df[col].median()
        numeric_medians[col] = median_val
        df[col] = df[col].fillna(median_val)

    categorical_modes = {}
    for col in df.select_dtypes(include=["object"]).columns:
        mode_val = df[col].mode()[0]
        categorical_modes[col] = mode_val
        df[col] = df[col].fillna(mode_val)

    df = pd.get_dummies(df, drop_first=True)
    df = df.astype(np.float32)

    feature_columns = list(df.columns)
    X = df.values
    input_dim = X.shape[1]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config.TEST_SPLIT, random_state=config.SEED
    )

    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_val   = scaler_X.transform(X_val)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val   = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

    save_preprocessing_artifacts(scaler_X, scaler_y, feature_columns,
                                 numeric_medians, categorical_modes)

    X_train = torch.tensor(X_train, dtype=torch.float32, device=config.DEVICE)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=config.DEVICE)
    X_val   = torch.tensor(X_val,   dtype=torch.float32, device=config.DEVICE)
    y_val   = torch.tensor(y_val,   dtype=torch.float32, device=config.DEVICE)

    print(f"[data_loader] Features: {input_dim}  |  "
          f"Train: {X_train.shape[0]}  |  Val: {X_val.shape[0]}  |  "
          f"Device: {config.DEVICE}")

    return X_train, y_train, X_val, y_val, input_dim, scaler_y
