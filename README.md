# 💳 Fraud Radar — Explainable AI Fraud Detection

A production-grade **hybrid fraud detection system** combining a deep learning Autoencoder with a RandomForest classifier, fully explainable via SHAP.

## 🌐 Live Demo

**Try it online:** [Fraud Radar on Streamlit Cloud](https://anand-singh-git-hub-fraud-radar-appmain-b2fgwv.streamlit.app/)

No installation required! Test the app directly in your browser with sample datasets.

## Architecture

```
Raw Input (30 features: Time, V1–V28, Amount)
    ↓  StandardScaler
    ↓  Encoder (encoder_model.h5)  →  12 Latent Features
    ↓  Autoencoder (autoencoder_model.h5)  →  Reconstruction Error
    ↓  Final Feature Vector: 43 dims
    ↓  RandomForestClassifier
    ↓  Tuned threshold → Prediction + Fraud Probability
    ↓  SHAP TreeExplainer → Feature attributions
```

## Performance

| Metric    | Value |
|-----------|-------|
| Precision | 0.78  |
| Recall    | 0.83  |
| F1-Score  | 0.80  |
| ROC-AUC   | 0.977 |
| PR-AUC    | 0.817 |
| MCC       | 0.802 |

## Quick Start

### Option 1: Run Locally
```bash
pip install -r requirements.txt
streamlit run app/main.py
```

Open [http://localhost:8501](http://localhost:8501)

### Option 2: Use Online (Recommended)
Visit the live demo: [https://anand-singh-git-hub-fraud-radar-appmain-b2fgwv.streamlit.app/](https://anand-singh-git-hub-fraud-radar-appmain-b2fgwv.streamlit.app/)

## Docker

```bash
docker build -t fraud-radar .
docker run -p 8501:8501 fraud-radar
```

## Project Structure

```
fraud-detection-shap-app/
├── app/
│   ├── main.py                  ← Streamlit entry point
│   ├── components/
│   │   ├── sidebar.py           ← Model info sidebar
│   │   ├── shap_plots.py        ← SHAP visualisations
│   │   └── model_info.py        ← Pipeline explainer tab
│   ├── services/
│   │   ├── predictor.py         ← FraudPredictor class
│   │   └── explainer.py         ← ShapExplainer class
│   └── utils/
│       └── validation.py        ← CSV validation helpers
├── models/
│   ├── hybrid_rf_model.pkl      ← RF + scaler + threshold
│   ├── encoder_model.h5         ← Encoder (30→12)
│   └── autoencoder_model.h5     ← Full autoencoder (30→30)
├── sample_data/
│   ├── sample_single.csv        ← 1-row test transaction
│   └── sample_batch.csv         ← 5-row batch test
├── Dockerfile
├── .dockerignore
└── requirements.txt
```

## Features

- **Single Prediction**: Enter transaction values manually, get instant fraud probability + SHAP bar chart + plain-language explanation
- **Batch Upload**: Upload a CSV, get summary stats, fraud distribution histogram, top suspicious transactions, and global SHAP summary plot
- **Model Info**: Interactive Sankey pipeline diagram, performance metrics, and expandable component explanations

## Dataset

[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 transactions, 0.17% fraud rate.
