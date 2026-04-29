import streamlit as st


def render_sidebar(threshold: float) -> None:
    with st.sidebar:
        st.markdown(
            """
            <div style='text-align:center; padding: 1rem 0 0.5rem 0;'>
                <span style='font-size:2.5rem;'>💳</span>
                <h2 style='margin:0; font-size:1.2rem; font-weight:700;
                           letter-spacing:0.03em;'>Fraud Radar</h2>
                <p style='margin:0; font-size:0.75rem; opacity:0.6;
                          letter-spacing:0.05em; text-transform:uppercase;'>
                    Explainable AI · Credit Card
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        st.markdown("##### Model Details")
        col1, col2 = st.columns(2)
        col1.metric("Threshold", f"{threshold:.2f}")
        col2.metric("F1-Score", "0.80")
        col1.metric("ROC-AUC", "0.977")
        col2.metric("PR-AUC", "0.817")

        st.divider()
        st.markdown("##### Pipeline")
        steps = [
            ("⚖️", "StandardScaler"),
            ("🔗", "Autoencoder (12 latent)"),
            ("📐", "Reconstruction Error"),
            ("🌲", "RandomForest (120 trees)"),
            ("🎯", "Threshold Tuning"),
            ("🔍", "SHAP Explanations"),
        ]
        for icon, label in steps:
            st.markdown(
                f"<div style='display:flex; align-items:center; gap:0.5rem; "
                f"margin-bottom:0.3rem; font-size:0.85rem;'>"
                f"<span>{icon}</span><span>{label}</span></div>",
                unsafe_allow_html=True,
            )

        st.divider()
        st.caption("Dataset: Kaggle Credit Card Fraud · 284,807 transactions · 0.17% fraud")
        
        st.divider()
        st.markdown("##### 🌐 Live Demo")
        st.markdown(
            "[View on Streamlit Cloud](https://anand-singh-git-hub-fraud-radar-appmain-b2fgwv.streamlit.app/)"
        )
        st.caption("Share this link to let others try the app online")
