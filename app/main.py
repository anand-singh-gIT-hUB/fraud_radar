import os
import sys

# ── Path fix: allow imports from app/ regardless of working directory ──
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

# Suppress TensorFlow noise before any TF import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Fix TensorFlow deprecation warning
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Show Streamlit toolbar ── */
    #MainMenu { visibility: visible !important; }
    header { visibility: visible !important; }
    .stApp > header { display: block !important; }
    
    /* ── Hide default footer only ── */
    footer { visibility: hidden; }

    /* ── Metric card style ── */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(139,92,246,0.06));
        border: 1px solid rgba(99,102,241,0.18);
        border-radius: 12px;
        padding: 0.9rem 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(99, 102, 241, 0.15);
    }

    /* ── Tab style ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.4rem 1.2rem;
        font-weight: 500;
        font-size: 0.9rem;
        background: rgba(99, 102, 241, 0.05);
        transition: all 0.2s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(99, 102, 241, 0.1);
    }

    /* ── Button ── */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.02em;
        transition: all 0.2s;
        box-shadow: 0 4px 10px rgba(99, 102, 241, 0.3);
    }
    .stButton > button:hover {
        opacity: 0.88;
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(99, 102, 241, 0.4);
    }

    /* ── Input fields ── */
    .stNumberInput input {
        border-radius: 8px;
        font-size: 0.88rem;
    }

    /* ── Alert boxes ── */
    .fraud-alert {
        background: linear-gradient(135deg, rgba(255,75,75,0.1), rgba(255,75,75,0.05));
        border-left: 4px solid #FF4B4B;
        border-radius: 0 10px 10px 0;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(255, 75, 75, 0.1);
    }
    .legit-alert {
        background: linear-gradient(135deg, rgba(0,196,159,0.1), rgba(0,196,159,0.05));
        border-left: 4px solid #00C49F;
        border-radius: 0 10px 10px 0;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0, 196, 159, 0.1);
    }

    /* ── Section headers ── */
    .section-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #a78bfa;
        letter-spacing: 0.02em;
        margin: 1.2rem 0 0.4rem 0;
        border-bottom: 2px solid rgba(167,139,250,0.3);
        padding-bottom: 0.3rem;
    }
    
    /* ── Card containers ── */
    .css-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        border-radius: 12px;
        border: 2px dashed rgba(99, 102, 241, 0.3);
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
    <div style='margin-bottom:1.5rem; padding: 1.5rem; 
                background: linear-gradient(135deg, rgba(99,102,241,0.05), rgba(139,92,246,0.03));
                border-radius: 16px; border: 1px solid rgba(99,102,241,0.1);'>
        <h1 style='margin:0; font-size:2.2rem; font-weight:700;
                   background: linear-gradient(90deg,#6366f1,#a78bfa,#38bdf8);
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            💳 Fraud Radar — Explainable AI
        </h1>
        <p style='margin:0.5rem 0 0 0; font-size:1rem; opacity:0.7;'>
            Hybrid Autoencoder + RandomForest · SHAP Explainability · 43-Feature Pipeline
        </p>
        <div style='margin-top:0.8rem; display:flex; gap:1rem; flex-wrap:wrap; align-items:center;'>
            <span style='background:rgba(99,102,241,0.1); padding:0.3rem 0.8rem; 
                         border-radius:20px; font-size:0.8rem; color:#6366f1;'>
                🎯 F1-Score: 0.80
            </span>
            <span style='background:rgba(0,196,159,0.1); padding:0.3rem 0.8rem; 
                         border-radius:20px; font-size:0.8rem; color:#00C49F;'>
                📊 ROC-AUC: 0.977
            </span>
            <span style='background:rgba(139,92,246,0.1); padding:0.3rem 0.8rem; 
                         border-radius:20px; font-size:0.8rem; color:#8b5cf6;'>
                🔍 SHAP Enabled
            </span>
            <a href='https://anand-singh-git-hub-fraud-radar-appmain-b2fgwv.streamlit.app/' 
               target='_blank' 
               style='background:rgba(255,165,0,0.1); padding:0.3rem 0.8rem; 
                      border-radius:20px; font-size:0.8rem; color:#FFA500; text-decoration:none;'>
                🌐 Live Demo
            </a>
        </div>
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
    
    # Quick load sample single transaction
    col_sample, col_space = st.columns([1, 4])
    with col_sample:
        if st.button("📄 Load Sample Transaction", use_container_width=True):
            sample_single_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "sample_data",
                "sample_single.csv"
            )
            if os.path.exists(sample_single_path):
                try:
                    sample_df = pd.read_csv(sample_single_path)
                    # Fill the input fields with sample data
                    features = predictor.original_features
                    for feature in features:
                        if feature in sample_df.columns:
                            st.session_state[f"inp_{feature}"] = float(sample_df[feature].iloc[0])
                    st.success("✓ Sample loaded! Click 'Analyse Transaction'")
                except Exception as e:
                    st.error(f"Error loading sample: {e}")
    
    input_values: dict = {}
    features = predictor.original_features  # ["Time", "V1", ..., "V28", "Amount"]

    # ── Render inputs in 5 columns ─────────────────────────────────────
    n_cols = 5
    cols   = st.columns(n_cols)

    for idx, feature in enumerate(features):
        with cols[idx % n_cols]:
            default = st.session_state.get(f"inp_{feature}", 0.0)
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
    st.markdown('<p class="section-header">Upload CSV for Bulk Prediction</p>', unsafe_allow_html=True)
    st.caption(
        f"Required columns: {', '.join(predictor.original_features[:5])} … "
        f"(all {len(predictor.original_features)} features)"
    )
    
    # Sample dataset quick load section
    st.markdown("### 📊 Quick Test with Sample Data")
    st.caption("Select a sample dataset to quickly test the fraud detection system")
    
    # Get available CSV files
    sample_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "sample_data"
    )
    
    if os.path.exists(sample_data_dir):
        csv_files = [
            f for f in os.listdir(sample_data_dir) 
            if f.endswith('.csv')
        ]
        csv_files.sort()
    else:
        csv_files = []
    
    if csv_files:
        selected_sample = st.selectbox(
            "Choose a sample dataset:",
            ["-- Select a sample --"] + csv_files,
            help="Select a sample CSV to quickly test the system"
        )
        
        if selected_sample != "-- Select a sample --":
            sample_file = selected_sample
            sample_path = os.path.join(sample_data_dir, sample_file)
            
            if os.path.exists(sample_path):
                st.info(f"📄 **Selected:** `{sample_file}`")
                
                col_load, col_info = st.columns([1, 3])
                with col_load:
                    load_sample = st.button("🚀 Analyze Sample Dataset", type="primary", use_container_width=True)
                
                if load_sample:
                    try:
                        batch_df = pd.read_csv(sample_path)
                        st.success(f"✓ Loaded {len(batch_df)} rows from {sample_file}")
                        
                        # Display preview
                        with st.expander("👀 Preview Dataset"):
                            st.dataframe(batch_df.head(10), use_container_width=True)
                        
                        # Continue with validation and prediction
                        is_valid, missing = validate_columns(batch_df, predictor.original_features)
                        
                        if not is_valid:
                            st.error(f"❌ Missing columns: `{'`, `'.join(missing)}`")
                            st.info(
                                "**Expected columns:** " + ", ".join(predictor.original_features)
                            )
                        else:
                            batch_df = clean_numeric(batch_df, predictor.original_features)
                            
                            if batch_df.empty:
                                st.warning("No valid rows found after cleaning.")
                            else:
                                with st.spinner(f"🔍 Analysing {len(batch_df):,} transactions…"):
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
                                
                    except Exception as e:
                        st.error(f"❌ Error loading sample dataset: {e}")
            else:
                st.warning(f"Sample file not found: {sample_file}")
    else:
        st.info("No sample datasets available in the sample_data folder.")
    
    st.divider()
    st.markdown("### 📤 Upload Your Own CSV")
    st.caption("Upload a CSV file with your transaction data for batch prediction")

    uploaded = st.file_uploader("Drop CSV here", type=["csv"])

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not parse CSV: {e}")
            st.stop()

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
