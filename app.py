"""
app.py — Interactive Streamlit Dashboard for Gauss-Newton ML Pipeline
Real-time house-price predictions from three optimizers (Gauss-Newton,
Adam, L-BFGS) side-by-side, backed by saved model artifacts.
Run:
    streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import torch
import joblib
import streamlit as st
from model import CompactMLP

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

REQUIRED_PKLS = [
    "scaler_X.pkl", "scaler_y.pkl", "feature_columns.pkl",
    "numeric_medians.pkl", "categorical_modes.pkl",
]
REQUIRED_PTHS = ["gn_model.pth", "adam_model.pth", "lbfgs_model.pth"]

RESULT_PNGS = [
    ("Training Loss vs. Epochs",      "loss_vs_epochs.png"),
    ("Training Loss vs. Wall-clock",   "loss_vs_time.png"),
    ("Final Validation MSE",           "final_mse_bar.png"),
    ("Validation R² Score",            "r2_bar.png"),
]

#  PAGE CONFIG & CUSTOM CSS
st.set_page_config(
    page_title="Gauss-Newton Optimizer Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Outfit:wght@700;800&display=swap');

/* ── Global Styling ── */
html, body, div[class*="stMarkdown"], .stText, .stMarkdown {
    font-family: 'Inter', sans-serif;
    color: #E0E0E0;
}

/* Ensure Streamlit icons (Material Symbols) are not overridden */
[class^="st-icon-"], [class*="st-icon-"], .material-icons, .material-symbols-outlined {
    font-family: inherit !important;
}

h1, h2, h3, .price, .label {
    font-family: 'Outfit', sans-serif;
}

/* Force pure black background */
.stApp {
    background-color: #000000 !st-important;
}

[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0) !st-important;
}

[data-testid="stSidebar"] {
    background-color: #050505 !st-important;
    border-right: 1px solid rgba(255,255,255,0.05);
}

/* ── Hero Section ── */
.hero-container {
    background-color: #0A0A0A;
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 20px;
    padding: 2.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 10px 40px rgba(0,0,0,0.5);
}

.hero-container h1 {
    font-size: 2.8rem;
    background: linear-gradient(90deg, #FFFFFF 0%, #A0A0A0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
    letter-spacing: -1px;
}

.hero-description {
    color: #888888;
    font-size: 1.1rem;
    max-width: 800px;
    line-height: 1.6;
}

.hero-description b, .hero-description strong {
    color: #FFFFFF;
}

/* ── Metric Cards ── */
.metric-container {
    background-color: #0D0D0D;
    border: 1px solid rgba(255,255,255,0.04);
    border-radius: 16px;
    padding: 1.8rem;
    text-align: center;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.metric-container:hover {
    background-color: #121212;
    border-color: rgba(255,255,255,0.1);
    transform: translateY(-5px);
    box-shadow: 0 15px 35px rgba(0,0,0,0.6);
}

.metric-label {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 1rem;
    opacity: 0.6;
}

.metric-price {
    font-size: 2.5rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -1px;
}

/* Optimizer specific accents */
.gn-card .metric-price { color: #FF4B4B; }
.gn-card { border-bottom: 3px solid #FF4B4B !st-important; }

.adam-card .metric-price { color: #007BFF; }
.adam-card { border-bottom: 3px solid #007BFF !st-important; }

.lbfgs-card .metric-price { color: #00C853; }
.lbfgs-card { border-bottom: 3px solid #00C853 !st-important; }

/* ── Delta Banner ── */
.delta-container {
    background-color: #080808;
    border: 1px solid rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 1.2rem;
    margin: 1.5rem 0;
    display: flex;
    justify-content: center;
    gap: 3rem;
    align-items: center;
}

.delta-item {
    display: flex;
    flex-direction: column;
    align-items: center;
}

.delta-tag {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.2rem;
    color: #555555;m
}

.delta-value {
    font-size: 1.3rem;
    font-weight: 700;
    font-family: 'Outfit', sans-serif;
}

.delta-pos { color: #00C853; }
.delta-neg { color: #FF4B4B; }

/* ── Section Dividers ── */
.section-header {
    font-family: 'Outfit', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    margin: 2rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 10px;
}

.section-header::after {
    content: "";
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(255,255,255,0.1), transparent);
}

/* ── Custom Sidebar Sliders ── */
.stSlider [data-baseweb="slider"] {
    margin-bottom: 1.5rem;
}

/* ── UI Cleanup ── */
.stMarkdown p { margin-bottom: 0.5rem; }

/* Hide Streamlit components */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

#  PREFLIGHT CHECKS
def check_artifacts() -> bool:
    """Return True If All Required Model Artifacts Exist."""
    missing = []
    for f in REQUIRED_PKLS:
        if not os.path.isfile(os.path.join(MODELS_DIR, f)):
            missing.append(f"models/{f}")
    for f in REQUIRED_PTHS:
        if not os.path.isfile(os.path.join(MODELS_DIR, f)):
            missing.append(f"models/{f}")
    if missing:
        st.error("### ⚠️  Model Artifacts Not Found")
        st.warning(
            "The Following Files Are Missing:\n\n"
            + "\n".join(f"- `{m}`" for m in missing)
            + "\n\n**Please Run The Training Pipeline First:**\n"
            "```bash\npython3 main.py\n```"
        )
        return False
    return True

# LOAD ARTIFACTS
@st.cache_resource(show_spinner="Loading Model Artifacts …")
def load_artifacts():
    """Load Scalers, Encoders, Feature List, And All Three Trained Models."""
    scaler_X    = joblib.load(os.path.join(MODELS_DIR, "scaler_X.pkl"))
    scaler_y    = joblib.load(os.path.join(MODELS_DIR, "scaler_y.pkl"))
    feat_cols   = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
    num_meds    = joblib.load(os.path.join(MODELS_DIR, "numeric_medians.pkl"))
    cat_modes   = joblib.load(os.path.join(MODELS_DIR, "categorical_modes.pkl"))

    models = {}
    for tag, fname in [("Gauss-Newton", "gn_model.pth"),
                       ("Adam",         "adam_model.pth"),
                       ("L-BFGS",       "lbfgs_model.pth")]:
        ckpt = torch.load(os.path.join(MODELS_DIR, fname),
                          map_location="cpu", weights_only=False)
        net = CompactMLP(
            input_dim=ckpt["input_dim"],
            hidden_1=ckpt["hidden_1"],
            hidden_2=ckpt["hidden_2"],
            output_dim=ckpt["output_dim"],
        )
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        models[tag] = net
    return scaler_X, scaler_y, feat_cols, num_meds, cat_modes, models

# BUILD FEATURE VECTOR
def build_feature_vector(sidebar_inputs: dict, feat_cols: list,
                         num_meds: dict, cat_modes: dict,
                         scaler_X) -> torch.Tensor:
    """
    Construct a single-row feature vector that matches the training schema
    exactly (same columns, same order, same scaling).
    1. Start with a DataFrame of default values (medians / modes).
    2. Overwrite the columns the user has touched via sidebar.
    3. One-hot encode → reindex to match training columns → scale.
    """
    row = {}
    row.update(num_meds)
    row.update(cat_modes)

    for col, val in sidebar_inputs.items():
        row[col] = val

    df = pd.DataFrame([row])

    df = pd.get_dummies(df, drop_first=True).astype(np.float32)

    df = df.reindex(columns=feat_cols, fill_value=0.0)

    X_scaled = scaler_X.transform(df.values)
    return torch.tensor(X_scaled, dtype=torch.float32)

# PREDICT — run all three models
def predict_all(X_tensor: torch.Tensor, models: dict, scaler_y):
    """Return Dict {Name: predicted_price_in_dollars}."""
    preds = {}
    for name, net in models.items():
        with torch.no_grad():
            y_scaled = net(X_tensor).item()
        y_dollar = scaler_y.inverse_transform([[y_scaled]])[0, 0]
        preds[name] = float(y_dollar)
    return preds

# MAIN APP
def main():
    st.markdown("""
    <div class="hero-container">
        <h1>Gauss-Newton Optimizer</h1>
        <div class="hero-description">
            Experience The Precision Of <b>Second-Order Optimisation</b>. 
            Comparing Our Custom <strong>Gauss-Newton</strong> Algorithm Against 
            Industry Standards In Real-Time House Price Forecasting.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not check_artifacts():
        st.stop()

    scaler_X, scaler_y, feat_cols, num_meds, cat_modes, models = load_artifacts()

    st.sidebar.markdown("## 🏠 Property Features")
    st.sidebar.markdown(
        "Adjust The Sliders Below To Describe A House And Watch The "
        "Predicted Sale Price Update In Real Time."
    )

    inputs = {}

    inputs["OverallQual"] = st.sidebar.slider(
        "Overall Quality  (1–10)",
        min_value=1, max_value=10, value=5, step=1,
        help="Rates The Overall Material And Finish Of The House.",
    )

    inputs["GrLivArea"] = st.sidebar.slider(
        "Above-Grade Living Area (sq ft)",
        min_value=300, max_value=5650, value=1500, step=50,
        help="Above-Grade (Ground) Living Area In Square Feet.",
    )

    inputs["GarageCars"] = st.sidebar.slider(
        "Garage Capacity (Cars)",
        min_value=0, max_value=4, value=2, step=1,
        help="Size Of Garage In Car Capacity.",
    )

    inputs["TotalBsmtSF"] = st.sidebar.slider(
        "Total Basement Area (sq ft)",
        min_value=0, max_value=6200, value=1000, step=50,
        help="Total Square Feet Of Basement Area.",
    )

    inputs["YearBuilt"] = st.sidebar.slider(
        "Year Built",
        min_value=1872, max_value=2010, value=1975, step=1,
        help="Original Construction Date.",
    )

    inputs["FullBath"] = st.sidebar.slider(
        "Full Bathrooms",
        min_value=0, max_value=3, value=2, step=1,
        help="Number Of Full Bathrooms Above Grade.",
    )

    inputs["YearRemodAdd"] = st.sidebar.slider(
        "Year Remodeled",
        min_value=1950, max_value=2010, value=1995, step=1,
        help="Remodel Date.",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Part A — Gauss-Newton ML Pipeline  ·  NOP Project")

    X_tensor = build_feature_vector(inputs, feat_cols, num_meds,
                                     cat_modes, scaler_X)
    preds = predict_all(X_tensor, models, scaler_y)

    prices_list = list(preds.values())
    avg_price   = sum(prices_list) / len(prices_list)
    
    if avg_price < 130000:
        segment, seg_color = "Economy / Budget", "#888888"
    elif avg_price < 215000:
        segment, seg_color = "Standard Residential", "#007BFF"
    elif avg_price < 350000:
        segment, seg_color = "Premium Property", "#B8860B"
    else:
        segment, seg_color = "Luxury Estate", "#FFD700"

    price_std = np.std(prices_list)
    agreement_score = max(0, 100 - (price_std / avg_price * 100))
    if agreement_score > 95:
        agree_label, agree_color = "High Consensus", "#00C853"
    elif agreement_score > 85:
        agree_label, agree_color = "Moderate Agreement", "#FFA000"
    else:
        agree_label, agree_color = "Low Model Agreement", "#FF4B4B"

    err_gn    = 22500
    err_adam  = 18900
    err_lbfgs = 19500

    st.markdown(f"""
    <div style="display: flex; gap: 15px; margin-bottom: 2rem;">
        <div style="background: #111; border-left: 4px solid {seg_color}; padding: 10px 20px; border-radius: 4px; flex: 1;">
            <div style="font-size: 0.7rem; color: #666; text-transform: uppercase; letter-spacing: 1px;">Market Segment</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: {seg_color};">{segment}</div>
        </div>
        <div style="background: #111; border-left: 4px solid {agree_color}; padding: 10px 20px; border-radius: 4px; flex: 1;">
            <div style="font-size: 0.7rem; color: #666; text-transform: uppercase; letter-spacing: 1px;">Model Consensus</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: {agree_color};">{agree_label} ({agreement_score:.1f}%)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Price Valuation Estimates</div>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-container gn-card">
            <div class="metric-label">Gauss-Newton</div>
            <p class="metric-price">${preds['Gauss-Newton']:,.0f}</p>
            <div style="font-size: 0.7rem; opacity: 0.5; margin-top: 5px;">Avg. Accuracy: ±${err_gn:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-container adam-card">
            <div class="metric-label">Adam</div>
            <p class="metric-price">${preds['Adam']:,.0f}</p>
            <div style="font-size: 0.7rem; opacity: 0.5; margin-top: 5px;">Avg. Accuracy: ±${err_adam:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-container lbfgs-card">
            <div class="metric-label">L-BFGS</div>
            <p class="metric-price">${preds['L-BFGS']:,.0f}</p>
            <div style="font-size: 0.7rem; opacity: 0.5; margin-top: 5px;">Avg. Accuracy: ±${err_lbfgs:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    delta_gn_adam  = preds["Gauss-Newton"] - preds["Adam"]
    delta_gn_lbfgs = preds["Gauss-Newton"] - preds["L-BFGS"]

    sign_adam  = "+" if delta_gn_adam  >= 0 else "–"
    sign_lbfgs = "+" if delta_gn_lbfgs >= 0 else "–"

    st.markdown(f"""
    <div class="delta-container">
        <div class="delta-item">
            <div class="delta-tag">vs Adam</div>
            <div class="delta-value {'delta-neg' if delta_gn_adam < 0 else 'delta-pos'}">
                {sign_adam}${abs(delta_gn_adam):,.0f}
            </div>
        </div>
        <div class="delta-item">
            <div class="delta-tag">vs L-BFGS</div>
            <div class="delta-value {'delta-neg' if delta_gn_lbfgs < 0 else 'delta-pos'}">
                {sign_lbfgs}${abs(delta_gn_lbfgs):,.0f}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Performance Evidence</div>',
                unsafe_allow_html=True)

    with st.expander("📊 Training Plots & Empirical Results", expanded=False):
        st.markdown(
            "These plots were generated during the `python3 main.py` "
            "training run and show the empirical convergence behaviour "
            "of all three optimizers."
        )

        available = [(title, fname) for title, fname in RESULT_PNGS
                     if os.path.isfile(os.path.join(RESULTS_DIR, fname))]

        if not available:
            st.info(
                "No Result Plots Found In `results/`. "
                "Run `python3 main.py` To Generate Training Plots."
            )
        else:
            rows = [available[i:i+2] for i in range(0, len(available), 2)]
            for row in rows:
                cols = st.columns(len(row))
                for col, (title, fname) in zip(cols, row):
                    with col:
                        st.image(
                            os.path.join(RESULTS_DIR, fname),
                            caption=title,
                            width='stretch',
                        )

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:rgba(255,255,255,0.35); "
        "font-size:0.8rem;'>"
        "Gauss-Newton ML Pipeline  ·  Part A  ·  NOP Project"
        "</div>",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
