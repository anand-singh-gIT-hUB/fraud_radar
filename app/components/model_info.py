import streamlit as st
import plotly.graph_objects as go


def render_model_info() -> None:
    st.markdown("## How This System Works")

    st.markdown(
        """
        This is not a simple classifier. It is a **hybrid anomaly detection + supervised
        learning pipeline** — the same class of architecture used in production fraud systems.
        """
    )

    # ── Pipeline diagram ────────────────────────────────────────────────
    stages = [
        "Raw Input\n(30 features)",
        "StandardScaler",
        "Autoencoder\nEncoder",
        "Reconstruction\nError",
        "Final Vector\n(43 dims)",
        "RandomForest\n(120 trees)",
        "Threshold\nTuning",
        "Prediction +\nSHAP",
    ]
    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                label=stages,
                color=["#6366f1", "#8b5cf6", "#a78bfa", "#c4b5fd",
                       "#60a5fa", "#34d399", "#fbbf24", "#f87171"],
                pad=20,
                thickness=22,
            ),
            link=dict(
                source=list(range(len(stages) - 1)),
                target=list(range(1, len(stages))),
                value=[1] * (len(stages) - 1),
                color=["rgba(99,102,241,0.25)"] * (len(stages) - 1),
            ),
        )
    )
    fig.update_layout(
        title="Model Pipeline",
        paper_bgcolor="rgba(0,0,0,0)",
        height=220,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Metric cards ────────────────────────────────────────────────────
    st.markdown("### Performance on Held-Out Test Set")
    m = st.columns(6)
    metrics = [
        ("Precision", "0.78"), ("Recall", "0.83"), ("F1", "0.80"),
        ("ROC-AUC", "0.977"), ("PR-AUC", "0.817"), ("MCC", "0.802"),
    ]
    for col, (label, value) in zip(m, metrics):
        col.metric(label, value)

    st.divider()

    # ── Component explanations ──────────────────────────────────────────
    with st.expander("🔗 Autoencoder — Why it helps"):
        st.markdown(
            """
            The autoencoder is **trained exclusively on legitimate transactions**.
            It learns a compressed representation of what "normal" looks like.

            When a fraudulent transaction passes through, the autoencoder cannot
            reconstruct it well — producing a high **reconstruction error**.
            This error becomes an anomaly signal added to the feature set.

            Architecture: `Input(30) → Dense(24) → Dropout(0.1) → Latent(12) → Dense(24) → Output(30)`
            """
        )

    with st.expander("🌲 RandomForest — What it receives"):
        st.markdown(
            """
            The RandomForest receives a **43-dimensional feature vector** per transaction:
            - **30** scaled original features (Time, V1–V28, Amount)
            - **12** latent features from the encoder (compressed transaction fingerprint)
            - **1** reconstruction error (anomaly score)

            Hyperparameters: 120 trees, max_depth=12, class_weight=balanced_subsample, SMOTE augmentation.
            """
        )

    with st.expander("⚖️ SMOTE — Handling Imbalance"):
        st.markdown(
            """
            Only 0.17% of transactions are fraudulent. Without intervention,
            a classifier can reach 99.8% accuracy by predicting "Legitimate" for everything.

            **SMOTE** generates synthetic minority-class samples in feature space —
            applied after feature engineering to prevent data leakage.
            sampling_strategy=0.05, k_neighbors=3.
            """
        )

    with st.expander("🎯 Threshold Tuning"):
        st.markdown(
            """
            Default probability threshold of 0.5 is suboptimal for imbalanced data.
            The threshold is tuned on a validation set by sweeping [0.05, 0.80] and
            selecting the value that maximises **F1-score**.

            This allows the operator to trade off precision vs. recall depending
            on the cost of false positives vs. missed fraud.
            """
        )

    with st.expander("🔍 SHAP — Why this prediction was made"):
        st.markdown(
            """
            **SHAP (SHapley Additive exPlanations)** assigns each feature a value
            that represents its contribution to pushing the model output away from
            the base rate.

            - **Positive SHAP** → feature pushed toward "Fraudulent"
            - **Negative SHAP** → feature pushed toward "Legitimate"

            `TreeExplainer` computes exact Shapley values for tree models in polynomial time.
            """
        )
