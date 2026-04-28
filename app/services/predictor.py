import os
import pickle
import numpy as np
import pandas as pd

# Suppress TF logs before import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Fix TensorFlow deprecation warning
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from keras.models import load_model  # noqa: E402


class FraudPredictor:
    """
    Loads all saved artifacts and exposes a predict() method.
    Call build_features() separately when you need the final feature
    matrix for SHAP without running the full prediction.
    """

    def __init__(
        self,
        model_path: str = "models/hybrid_rf_model.pkl",
        encoder_path: str = "models/encoder_model.h5",
        autoencoder_path: str = "models/autoencoder_model.h5",
    ) -> None:
        # ── Load pickle bundle ──────────────────────────────────────────
        with open(model_path, "rb") as f:
            artifacts = pickle.load(f)

        self.rf_model          = artifacts["rf_model"]
        self.scaler            = artifacts["scaler"]
        self.threshold         = float(artifacts["threshold"])
        self.original_features = artifacts["original_features"]   # list[str], len 30
        self.final_features    = artifacts["final_features"]       # list[str], len 43
        self.encoding_dim      = int(artifacts.get("encoding_dim", 12))

        # ── Load Keras models ───────────────────────────────────────────
        # compile=False avoids optimizer-state errors on inference-only load
        self.encoder_model     = load_model(encoder_path,     compile=False)
        self.autoencoder_model = load_model(autoencoder_path, compile=False)

    # ───────────────────────────────────────────────────────────────────
    def build_features(self, input_df: pd.DataFrame) -> np.ndarray:
        """
        Returns the 43-column float32 array fed to the RandomForest.
        input_df must contain exactly self.original_features columns.
        """
        df = input_df[self.original_features].copy().astype("float32")

        # 1. Scale
        scaled = self.scaler.transform(df)
        scaled_df = pd.DataFrame(scaled, columns=self.original_features, index=df.index)

        # 2. Latent features via encoder
        latent = self.encoder_model.predict(scaled_df.values, batch_size=1024, verbose=0)

        # 3. Reconstruction error via full autoencoder
        reconstructed = self.autoencoder_model.predict(scaled_df.values, batch_size=1024, verbose=0)
        recon_error   = np.mean((scaled_df.values - reconstructed) ** 2, axis=1).reshape(-1, 1)

        # 4. Concatenate → [30 scaled | 12 latent | 1 recon_error]
        final_input = np.hstack([scaled_df.values, latent, recon_error]).astype("float32")
        return final_input

    # ───────────────────────────────────────────────────────────────────
    def predict(self, input_df: pd.DataFrame):
        """
        Returns (result_df, final_input_array).

        result_df is input_df with three extra columns appended:
            Fraud_Probability  float
            Prediction         int   (0 or 1)
            Prediction_Label   str   ("Legitimate" | "Fraudulent")
        """
        final_input  = self.build_features(input_df)
        fraud_probs  = self.rf_model.predict_proba(final_input)[:, 1]
        predictions  = (fraud_probs >= self.threshold).astype(int)

        result_df = input_df.copy()
        result_df["Fraud_Probability"] = fraud_probs
        result_df["Prediction"]        = predictions
        result_df["Prediction_Label"]  = result_df["Prediction"].map({0: "Legitimate", 1: "Fraudulent"})

        return result_df, final_input
