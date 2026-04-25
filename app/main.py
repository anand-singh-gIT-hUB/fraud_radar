import os
import sys

# ── Path fix: allow imports from app/ regardless of working directory ──
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

# Suppress TensorFlow noise before any TF import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import pandas as pd
import numpy as np

from services.predictor  import FraudPredictor
from services.explainer  import ShapExplainer
from utils.validation    import validate_columns, clean_numeric
from components.sidebar  import render_sidebar
from components.shap_plots import (
    render_shap_bar,
    render_fraud_gauge,
    render_global_shap,
    render_batch_histogram,
)
from components.model_info import render_model_info

# ══════════════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Fraud Radar — Explainable AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Font & base ── */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }

    /* ── Hide default header/footer ── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Metric card style ── */
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 1.2rem;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
    }
    [data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 40px rgba(99, 102, 241, 0.15);
        border-color: rgba(99, 102, 241, 0.4);
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* ── Tab style ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: rgba(0, 0, 0, 0.2);
        padding: 0.6rem;
        border-radius: 14px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.6rem 1.8rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        border: 1px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.25), rgba(139, 92, 246, 0.25)) !important;
        border: 1px solid rgba(139, 92, 246, 0.6) !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.25);
    }

    /* ── Button ── */
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 2.5rem;
        font-weight: 600;
        font-size: 1rem;
        letter-spacing: 0.05em;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    }
    .stButton > button:hover { 
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.6);
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
    }

    /* ── Input fields ── */
    .stNumberInput input {
        border-radius: 8px;
        font-size: 0.95rem;
        border: 1px solid rgba(255,255,255,0.1);
        background: rgba(255,255,255,0.05);
        transition: border-color 0.3s, box-shadow 0.3s;
    }
    .stNumberInput input:focus {
        border-color: #8b5cf6;
        box-shadow: 0 0 0 2px rgba(139,92,246,0.3);
    }

    /* ── Alert boxes ── */
    .fraud-alert {
        background: linear-gradient(90deg, rgba(255,75,75,0.15) 0%, rgba(255,75,75,0.05) 100%);
        border-left: 5px solid #FF4B4B;
        border-radius: 8px 12px 12px 8px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(255,75,75,0.1);
        animation: pulse-red 2s infinite;
    }
    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(255, 75, 75, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(255, 75, 75, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 75, 75, 0); }
    }
    
    .legit-alert {
        background: linear-gradient(90deg, rgba(0,196,159,0.15) 0%, rgba(0,196,159,0.05) 100%);
        border-left: 5px solid #00C49F;
        border-radius: 8px 12px 12px 8px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,196,159,0.1);
    }

    /* ── Section headers ── */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        background: linear-gradient(90deg, #a78bfa, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 0.02em;
        margin: 1.8rem 0 0.8rem 0;
        border-bottom: 1px solid rgba(167,139,250,0.15);
        padding-bottom: 0.5rem;
        display: inline-block;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.7);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════
# Cached resource loading
# ══════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading model artifacts…")
def load_predictor() -> FraudPredictor:
    # Path resolution: works whether run from repo root or from app/
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return FraudPredictor(
        model_path       = os.path.join(base, "models", "hybrid_rf_model.pkl"),
        encoder_path     = os.path.join(base, "models", "encoder_model.h5"),
        autoencoder_path = os.path.join(base, "models", "autoencoder_model.h5"),
    )


@st.cache_resource(show_spinner="Initialising SHAP explainer…")
def load_explainer(_predictor: FraudPredictor) -> ShapExplainer:
    return ShapExplainer(_predictor.rf_model, _predictor.final_features)


predictor = load_predictor()
explainer = load_explainer(predictor)


# ══════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════
render_sidebar(predictor.threshold)


# ══════════════════════════════════════════════════════════════════════
# Page header
# ══════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <div style='margin-bottom:1.5rem;'>
        <h1 style='margin:0; font-size:1.9rem; font-weight:700;
                   background: linear-gradient(90deg,#6366f1,#a78bfa,#38bdf8);
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            💳 Fraud Radar
        </h1>
        <p style='margin:0.3rem 0 0 0; font-size:0.95rem; opacity:0.6;'>
            Hybrid Autoencoder + RandomForest · SHAP Explainability · 43-Feature Pipeline
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🔎 Single Prediction", "📂 Batch Upload", "ℹ️ Model Info"])


# ──────────────────────────────────────────────────────────────────────
# TAB 1 — Single Prediction
# ──────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<p class="section-header">Enter Transaction Values</p>', unsafe_allow_html=True)
    st.caption("All V-features are PCA-transformed components from the original data.")

    input_values: dict = {}
    features = predictor.original_features  # ["Time", "V1", ..., "V28", "Amount"]

    # ── Render inputs in 5 columns ─────────────────────────────────────
    n_cols = 5
    cols   = st.columns(n_cols)

    for idx, feature in enumerate(features):
        with cols[idx % n_cols]:
            default = 0.0
            step    = 1.0 if feature == "Time" else 0.000001
            fmt     = "%.2f" if feature in ("Time", "Amount") else "%.6f"
            input_values[feature] = st.number_input(
                feature, value=default, format=fmt, step=step, key=f"inp_{feature}"
            )

    st.markdown("")
    predict_btn = st.button("Analyse Transaction", use_container_width=False)

    if predict_btn:
        single_df = pd.DataFrame([input_values])

        with st.spinner("Running inference…"):
            result_df, final_input = predictor.predict(single_df)

        fraud_prob = float(result_df["Fraud_Probability"].iloc[0])
        label      = result_df["Prediction_Label"].iloc[0]
        is_fraud   = label == "Fraudulent"

        # ── Result banner ───────────────────────────────────────────────
        verdict_icon  = "🚨" if is_fraud else "✅"
        verdict_class = "fraud-alert" if is_fraud else "legit-alert"
        st.markdown(
            f"<div class='{verdict_class}'>"
            f"<strong style='font-size:1.15rem;'>{verdict_icon} {label}</strong>"
            f"<br><span style='opacity:0.75;font-size:0.88rem;'>"
            f"Fraud probability: {fraud_prob:.4f} &nbsp;|&nbsp; "
            f"Threshold: {predictor.threshold:.2f}"
            f"</span></div>",
            unsafe_allow_html=True,
        )

        st.markdown("")
        col_gauge, col_shap = st.columns([1, 2])

        with col_gauge:
            render_fraud_gauge(fraud_prob, predictor.threshold)

        with col_shap:
            with st.spinner("Computing SHAP values…"):
                explanation_df = explainer.explain_single(final_input, top_n=12)
            st.markdown('<p class="section-header">Feature Attributions (Top 12)</p>', unsafe_allow_html=True)
            render_shap_bar(explanation_df)

        # ── SHAP table ──────────────────────────────────────────────────
        st.markdown('<p class="section-header">SHAP Detail Table</p>', unsafe_allow_html=True)

        styled = explanation_df.copy()
        styled["SHAP Value"] = styled["SHAP Value"].map(lambda x: f"{x:+.6f}")
        styled["Abs SHAP"]   = styled["Abs SHAP"].map(lambda x: f"{x:.6f}")
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # ── Readable explanation (no latent features) ───────────────────
        readable = explanation_df[
            ~explanation_df["Feature"].str.startswith("Latent_") &
            (explanation_df["Feature"] != "Reconstruction_Error")
        ]
        if not readable.empty:
            st.markdown('<p class="section-header">Plain-Language Explanation</p>', unsafe_allow_html=True)
            for _, row in readable.head(5).iterrows():
                icon = "🔴" if row["Impact"] == "↑ Fraud Risk" else "🟢"
                st.markdown(
                    f"{icon} **{row['Feature']}** {row['Impact']} "
                    f"(SHAP={float(row['SHAP Value']):+.5f})"
                )


# ──────────────────────────────────────────────────────────────────────
# TAB 2 — Batch Upload
# ──────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<p class="section-header">Upload or Select CSV for Bulk Prediction</p>', unsafe_allow_html=True)
    st.caption(
        f"Required columns: {', '.join(predictor.original_features[:5])} … "
        f"(all {len(predictor.original_features)} features)"
    )

    sample_files = {
        "12 Rows Credit Card Fraud Samples": "sample_data/cc_fraud_samples_12rows.csv",
        "Fraud Samples": "sample_data/fraud_samples.csv",
        "Sample Batch": "sample_data/sample_batch.csv",
        "Sample Single": "sample_data/sample_single.csv",
    }

    col_up, col_sel = st.columns(2)
    with col_up:
        uploaded = st.file_uploader("Drop CSV here", type=["csv"])
    with col_sel:
        st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
        selected_sample = st.selectbox("Or choose a sample dataset:", ["None"] + list(sample_files.keys()))

    batch_df = None
    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not parse CSV: {e}")
            st.stop()
    elif selected_sample != "None":
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sample_path = os.path.join(base_dir, sample_files[selected_sample])
        try:
            batch_df = pd.read_csv(sample_path)
            st.success(f"Loaded sample dataset: {selected_sample} ({len(batch_df)} rows)")
        except Exception as e:
            st.error(f"Could not read sample file: {e}")
            st.stop()

    if batch_df is not None:

        is_valid, missing = validate_columns(batch_df, predictor.original_features)

        if not is_valid:
            st.error(f"Missing columns: `{'`, `'.join(missing)}`")
            st.info(
                "Expected columns: " + ", ".join(predictor.original_features)
            )
        else:
            batch_df = clean_numeric(batch_df, predictor.original_features)

            if batch_df.empty:
                st.warning("No valid rows found after cleaning.")
            else:
                with st.spinner(f"Analysing {len(batch_df):,} transactions…"):
                    result_df, final_inputs = predictor.predict(batch_df)

                total  = len(result_df)
                frauds = int(result_df["Prediction"].sum())
                legits = total - frauds
                rate   = frauds / total * 100

                # ── Summary metrics ─────────────────────────────────────
                st.markdown('<p class="section-header">Batch Summary</p>', unsafe_allow_html=True)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total",        f"{total:,}")
                c2.metric("Fraudulent",   f"{frauds:,}", delta=f"{rate:.2f}%", delta_color="inverse")
                c3.metric("Legitimate",   f"{legits:,}")
                c4.metric("Fraud Rate",   f"{rate:.2f}%")

                # ── Histogram ───────────────────────────────────────────
                render_batch_histogram(result_df)

                # ── Top suspicious ──────────────────────────────────────
                st.markdown('<p class="section-header">Top 10 Suspicious Transactions</p>', unsafe_allow_html=True)
                top10 = (
                    result_df.sort_values("Fraud_Probability", ascending=False)
                    .head(10)
                    .reset_index(drop=True)
                )
                st.dataframe(top10, use_container_width=True, hide_index=True)

                # ── Full results & download ─────────────────────────────
                st.markdown('<p class="section-header">All Predictions</p>', unsafe_allow_html=True)
                st.dataframe(result_df, use_container_width=True, hide_index=True)

                csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇  Download Predictions CSV",
                    data=csv_bytes,
                    file_name="fraud_predictions.csv",
                    mime="text/csv",
                )

                # ── Global SHAP ─────────────────────────────────────────
                st.markdown('<p class="section-header">Global SHAP Summary (sample of 300)</p>', unsafe_allow_html=True)
                st.caption("Shows which features most influence the model across all transactions.")

                with st.spinner("Computing global SHAP values…"):
                    positive_shap, sample = explainer.explain_batch_summary(final_inputs)

                render_global_shap(positive_shap, sample, predictor.final_features)


# ──────────────────────────────────────────────────────────────────────
# TAB 3 — Model Info
# ──────────────────────────────────────────────────────────────────────
with tab3:
    render_model_info()
