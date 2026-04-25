import shap
import numpy as np
import pandas as pd


class ShapExplainer:
    """
    Wraps shap.TreeExplainer.
    Compatible with shap==0.45.x return shapes (3-D for multi-output forests).
    """

    def __init__(self, rf_model, final_features: list) -> None:
        self.rf_model      = rf_model
        self.final_features = final_features
        # TreeExplainer is fast and exact for tree models
        self.explainer     = shap.TreeExplainer(rf_model)

    # ───────────────────────────────────────────────────────────────────
    def _extract_positive_class_shap(self, shap_values) -> np.ndarray:
        """
        Handles both shap return formats:
          - Old: list of 2 arrays → shap_values[1]
          - New (0.45+): 3-D array → shap_values[:, :, 1]
        """
        if isinstance(shap_values, list):
            return shap_values[1]           # shape (n, 43)
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            return shap_values[:, :, 1]     # shape (n, 43)
        else:
            return shap_values              # already 2-D

    # ───────────────────────────────────────────────────────────────────
    def explain_single(self, final_input: np.ndarray, top_n: int = 12) -> pd.DataFrame:
        """
        Returns a DataFrame with columns:
            Feature | SHAP Value | Abs SHAP | Impact
        Sorted by |SHAP| descending, top_n rows.
        """
        shap_values = self.explainer.shap_values(final_input)
        values      = self._extract_positive_class_shap(shap_values)[0]   # 1-D, len 43

        df = pd.DataFrame({
            "Feature":    self.final_features,
            "SHAP Value": values,
        })
        df["Abs SHAP"] = df["SHAP Value"].abs()
        df["Impact"]   = np.where(df["SHAP Value"] > 0, "↑ Fraud Risk", "↓ Fraud Risk")

        return df.sort_values("Abs SHAP", ascending=False).head(top_n).reset_index(drop=True)

    # ───────────────────────────────────────────────────────────────────
    def explain_batch_summary(self, final_inputs: np.ndarray, sample_size: int = 300):
        """
        Returns (positive_class_shap_values, sample_array) for global summary plot.
        """
        sample      = final_inputs[:sample_size]
        shap_values = self.explainer.shap_values(sample)
        positive    = self._extract_positive_class_shap(shap_values)     # shape (n, 43)
        return positive, sample
