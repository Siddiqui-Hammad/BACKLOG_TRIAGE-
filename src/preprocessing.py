"""
preprocessing.py
Data cleaning, feature engineering, and encoding pipeline for NJDG case records.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

FEATURE_COLS = [
    "case_type_enc",
    "case_category_enc",
    "bench_type_enc",
    "current_stage_enc",
    "state_enc",
    "stage_index",
    "case_age_days",
    "stage_age_days",
    "hearings_held",
    "adjournments",
    "adjournment_rate",
    "stagnation_flag",
    "is_undertrial",
    "district_avg_disposal_days",
    "statutory_deadline_days",
    "days_beyond_statutory",
    "case_age_normalized",
    "hearing_density",
    "stage_completion_ratio",
]

CATEGORICAL_COLS = ["case_type", "case_category", "bench_type", "current_stage", "state"]

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


class NJDGPreprocessor:
    """End-to-end preprocessing pipeline for NJDG case data."""

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit_transform(self, df: pd.DataFrame):
        """Fit encoders and transform the full dataset."""
        df = df.copy()
        df = self._clean(df)
        df = self._engineer_features(df)
        df = self._encode_categoricals(df, fit=True)
        X = df[FEATURE_COLS]
        self.is_fitted = True
        return df, X

    def transform(self, df: pd.DataFrame):
        """Transform new data using fitted encoders."""
        assert self.is_fitted, "Call fit_transform first."
        df = df.copy()
        df = self._clean(df)
        df = df.reset_index(drop=True)   # ensure index is contiguous after cleaning
        df = self._engineer_features(df)
        df = self._encode_categoricals(df, fit=False)
        X = df[FEATURE_COLS]
        return df, X

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove nulls, duplicates, and invalid rows."""
        df.drop_duplicates(subset=["case_number"], inplace=True, ignore_index=True)
        df.dropna(subset=["case_age_days", "case_category", "current_stage"], inplace=True)
        df["case_age_days"] = df["case_age_days"].clip(lower=1)
        df["hearings_held"] = df["hearings_held"].clip(lower=0)
        df["adjournments"] = df["adjournments"].clip(lower=0)
        df["adjournment_rate"] = df["adjournments"] / df["hearings_held"].replace(0, 1)
        df["adjournment_rate"] = df["adjournment_rate"].clip(0, 1)
        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features aligned with the Technical Approach diagram."""

        # Case age relative to statutory deadline (0 = just filed, 1 = at deadline, >1 = overdue)
        df["case_age_normalized"] = df["case_age_days"] / df["statutory_deadline_days"].replace(0, 1)

        # Hearing density = hearings per month of case age
        df["hearing_density"] = df["hearings_held"] / (df["case_age_days"] / 30.0).replace(0, 1)

        # Stage completion ratio (0 = just started, 1 = near judgment)
        df["stage_completion_ratio"] = df["stage_index"] / 7.0

        return df

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        for col in CATEGORICAL_COLS:
            enc_col = f"{col}_enc"
            if fit:
                le = LabelEncoder()
                df[enc_col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                # Handle unseen labels gracefully
                df[enc_col] = df[col].astype(str).map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        return df

    def save(self, path=None):
        os.makedirs(MODELS_DIR, exist_ok=True)
        path = path or os.path.join(MODELS_DIR, "preprocessor.pkl")
        joblib.dump(self, path)
        print(f"✅ Preprocessor saved → {path}")

    @staticmethod
    def load(path=None):
        path = path or os.path.join(MODELS_DIR, "preprocessor.pkl")
        return joblib.load(path)


def load_and_split(csv_path: str, test_size=0.2, seed=42):
    """
    Load the NJDG CSV, run preprocessing, and return train/test splits
    for both regression (disposal_days) and classification (delay_risk_label).
    """
    df = pd.read_csv(csv_path)
    preprocessor = NJDGPreprocessor()
    df_proc, X = preprocessor.fit_transform(df)

    y_reg = df_proc["disposal_days"].values          # Regression target
    y_clf = df_proc["delay_risk_label"].values        # Classification target

    X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
        X, y_reg, y_clf, test_size=test_size, random_state=seed, stratify=y_clf
    )

    preprocessor.save()

    print(f"✅ Preprocessing complete")
    print(f"   Train size : {X_train.shape[0]}")
    print(f"   Test size  : {X_test.shape[0]}")
    print(f"   Features   : {X_train.shape[1]}")
    return X_train, X_test, yr_train, yr_test, yc_train, yc_test, preprocessor


if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "njdg_cases.csv")
    load_and_split(csv_path)
