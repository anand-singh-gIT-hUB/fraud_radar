import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap


# ── Color palette (matches the app theme) ──────────────────────────────
FRAUD_COLOR   = "#FF4B4B"
LEGIT_COLOR   = "#00C49F"
NEUTRAL_COLOR = "#8884d8"
BG_COLOR      = "rgba(0,0,0,0)"


def render_shap_bar(explanation_df: pd.DataFrame) -> None:
    """Interactive horizontal bar chart for single-transaction SHAP values."""

    df = explanation_df.copy().sort_values("SHAP Value")
    colors = [FRAUD_COLOR if v > 0 else LEGIT_COLOR for v in df["SHAP Value"]]

    fig = go.Figure(
        go.Bar(
            x=df["SHAP Value"],
            y=df["Feature"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.4f}" for v in df["SHAP Value"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>SHAP: %{x:.5f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text="Feature Impact on Fraud Score", font_size=15, x=0),
        xaxis_title="SHAP Value  (positive = fraud risk ↑)",
        yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
        margin=dict(l=10, r=60, t=40, b=30),
        height=420,
        shapes=[
            dict(
                type="line", x0=0, x1=0,
                y0=-0.5, y1=len(df) - 0.5,
                line=dict(color="gray", width=1, dash="dot"),
            )
        ],
    )
    st.plotly_chart(fig, use_container_width=True)


def render_fraud_gauge(fraud_prob: float, threshold: float) -> None:
    """Speedometer-style gauge for fraud probability."""

    color = FRAUD_COLOR if fraud_prob >= threshold else LEGIT_COLOR

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=round(fraud_prob * 100, 2),
            number={"suffix": "%", "font": {"size": 36}},
            delta={"reference": threshold * 100, "suffix": "% (threshold)"},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar":  {"color": color},
                "steps": [
                    {"range": [0, threshold * 100],    "color": "rgba(0,196,159,0.15)"},
                    {"range": [threshold * 100, 100],  "color": "rgba(255,75,75,0.15)"},
                ],
                "threshold": {
                    "line":  {"color": "orange", "width": 3},
                    "thickness": 0.8,
                    "value": threshold * 100,
                },
            },
            title={"text": "Fraud Probability"},
        )
    )
    fig.update_layout(
        paper_bgcolor=BG_COLOR,
        height=280,
        margin=dict(l=20, r=20, t=30, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_global_shap(positive_shap: np.ndarray, sample: np.ndarray, final_features: list) -> None:
    """SHAP beeswarm / summary plot using matplotlib (shap native)."""

    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(
        positive_shap,
        sample,
        feature_names=final_features,
        max_display=20,
        show=False,
        plot_size=None,
    )
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)


def render_batch_histogram(result_df: pd.DataFrame) -> None:
    """Distribution of fraud probabilities across batch."""

    fraud_df = result_df[result_df["Prediction"] == 1]["Fraud_Probability"]
    legit_df = result_df[result_df["Prediction"] == 0]["Fraud_Probability"]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=legit_df, name="Legitimate",
        marker_color=LEGIT_COLOR, opacity=0.7,
        nbinsx=40, hovertemplate="Prob: %{x:.3f}<br>Count: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Histogram(
        x=fraud_df, name="Fraudulent",
        marker_color=FRAUD_COLOR, opacity=0.7,
        nbinsx=40, hovertemplate="Prob: %{x:.3f}<br>Count: %{y}<extra></extra>",
    ))
    fig.update_layout(
        barmode="overlay",
        title="Fraud Probability Distribution",
        xaxis_title="Fraud Probability",
        yaxis_title="Count",
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
        legend=dict(orientation="h", y=1.05, x=0),
        margin=dict(l=10, r=10, t=50, b=30),
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)
