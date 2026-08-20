"""
explainability.py
SHAP-based explainability for the XGBoost models.
Generates per-case "Why this case?" explanations and global feature importance.
"""

import os
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")

FEATURE_LABELS = {
    "case_type_enc":             "Case Type",
    "case_category_enc":         "Case Category",
    "bench_type_enc":            "Bench Type",
    "current_stage_enc":         "Current Stage",
    "state_enc":                 "State",
    "stage_index":               "Stage Index",
    "case_age_days":             "Case Age (days)",
    "stage_age_days":            "Stage Age (days)",
    "hearings_held":             "Hearings Held",
    "adjournments":              "Adjournments",
    "adjournment_rate":          "Adjournment Rate",
    "stagnation_flag":           "Stagnation Flag",
    "is_undertrial":             "Is Undertrial (BNSS 479)",
    "district_avg_disposal_days":"District Avg Disposal",
    "statutory_deadline_days":   "Statutory Deadline",
    "days_beyond_statutory":     "Days Beyond Statutory",
    "case_age_normalized":       "Case Age (normalized)",
    "hearing_density":           "Hearing Density",
    "stage_completion_ratio":    "Stage Completion Ratio",
}


class CaseExplainer:
    """SHAP explainer for XGBoost case delay models."""

    def __init__(self, model=None, model_type="regressor"):
        if model is None:
            fname = "xgb_regressor.pkl" if model_type == "regressor" else "xgb_classifier.pkl"
            model = joblib.load(os.path.join(MODELS_DIR, fname))
        self.model      = model
        self.model_type = model_type
        self.explainer  = shap.TreeExplainer(model)
        self._shap_values = None
        self._X_ref       = None

    def compute_shap(self, X: pd.DataFrame):
        """Compute SHAP values for a feature matrix X."""
        self._X_ref = X
        self._shap_values = self.explainer.shap_values(X)
        return self._shap_values

    def explain_case(self, X_row: pd.DataFrame, class_idx: int = 2) -> dict:
        """
        Generate a per-case explanation dict.
        For classifier, class_idx=2 → 'High' risk class by default.
        """
        sv = self.explainer.shap_values(X_row)

        if self.model_type == "classifier":
            if isinstance(sv, list):
                sv_row = sv[class_idx][0]
            else:
                sv_row = sv[0, :, class_idx]
        else:
            sv_row = sv[0] if sv.ndim > 1 else sv

        features = X_row.columns.tolist()
        contributions = {
            FEATURE_LABELS.get(f, f): float(sv_row[i])
            for i, f in enumerate(features)
        }

        # Sort by absolute contribution
        sorted_contrib = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)

        # Build human-readable explanation
        top_factors = sorted_contrib[:5]
        explanation_lines = []
        for feat, val in top_factors:
            direction = "increases" if val > 0 else "decreases"
            explanation_lines.append(
                f"• **{feat}** {direction} delay risk by {abs(val):.2f} points"
            )

        return {
            "contributions":   dict(sorted_contrib),
            "top_factors":     top_factors,
            "explanation_text": "\n".join(explanation_lines),
        }

    def plot_summary(self, X: pd.DataFrame, save_path: str = None):
        """SHAP summary beeswarm plot for all features."""
        sv = self.explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[-1]   # last class for multiclass

        renamed = X.rename(columns=FEATURE_LABELS)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(sv, renamed, show=False, max_display=15)
        plt.title("SHAP Feature Impact — Case Delay Prediction", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✅ SHAP summary saved → {save_path}")
        return plt.gcf()

    def plot_waterfall(self, X_row: pd.DataFrame, save_path: str = None):
        """SHAP waterfall plot for a single case."""
        explanation = shap.Explanation(
            values=self.explainer.shap_values(X_row),
            base_values=self.explainer.expected_value,
            data=X_row.values,
            feature_names=[FEATURE_LABELS.get(c, c) for c in X_row.columns],
        )
        if isinstance(explanation.values, list):
            explanation = shap.Explanation(
                values=explanation.values[-1][0],
                base_values=explanation.base_values[-1],
                data=X_row.values[0],
                feature_names=explanation.feature_names,
            )

        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(explanation, show=False, max_display=10)
        plt.title("SHAP Waterfall — Single Case Explanation", fontsize=13, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✅ SHAP waterfall saved → {save_path}")
        return plt.gcf()

    def plot_feature_importance(self, save_path: str = None):
        """XGBoost built-in feature importance bar chart."""
        imp = self.model.get_booster().get_score(importance_type="gain")
        imp_renamed = {FEATURE_LABELS.get(k, k): v for k, v in imp.items()}
        imp_df = pd.DataFrame(list(imp_renamed.items()), columns=["Feature", "Gain"])
        imp_df = imp_df.sort_values("Gain", ascending=True).tail(15)

        fig, ax = plt.subplots(figsize=(10, 7))
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(imp_df)))
        ax.barh(imp_df["Feature"], imp_df["Gain"], color=colors)
        ax.set_xlabel("XGBoost Feature Gain", fontsize=12)
        ax.set_title("Feature Importance — Case Delay Prediction", fontsize=14, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✅ Feature importance saved → {save_path}")
        return fig


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ROOT)
    from src.preprocessing import load_and_split

    csv_path = os.path.join(ROOT, "data", "njdg_cases.csv")
    X_train, X_test, yr_train, yr_test, yc_train, yc_test, _ = load_and_split(csv_path)

    print("Computing SHAP explanations for regressor...")
    explainer = CaseExplainer(model_type="regressor")
    plots_dir = os.path.join(ROOT, "models", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    explainer.plot_summary(X_test.head(500), save_path=os.path.join(plots_dir, "shap_summary.png"))
    explainer.plot_feature_importance(save_path=os.path.join(plots_dir, "feature_importance.png"))

    sample = X_test.head(1)
    result = explainer.explain_case(sample)
    print("\n📋 Single Case Explanation:")
    print(result["explanation_text"])
