"""
xgboost_model.py
XGBoost training pipeline for:
  1. XGBRegressor  → disposal_days prediction
  2. XGBClassifier → delay_risk_label prediction (Critical/High/Medium/Low)

Includes hyperparameter tuning via Optuna and cross-validation.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, classification_report, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR   = os.path.join(ROOT, "data")

os.makedirs(MODELS_DIR, exist_ok=True)

# ── Risk label order for ordinal encoding
RISK_ORDER = ["Low", "Medium", "High", "Critical"]


# ─────────────────────────────────────────────────────────────────────
# HYPERPARAMETER TUNING — Optuna
# ─────────────────────────────────────────────────────────────────────

def _tune_regressor(X_train, y_train, n_trials=40):
    """Optuna study for XGBRegressor hyperparameters."""
    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 800),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
        }
        model = XGBRegressor(**params)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        # For regression, use simple KFold; approximate with neg_rmse
        from sklearn.model_selection import KFold, cross_val_score
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train,
                                 cv=kf, scoring="neg_root_mean_squared_error", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def _tune_classifier(X_train, y_train, n_trials=40):
    """Optuna study for XGBClassifier hyperparameters."""
    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 800),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
            "use_label_encoder": False,
            "eval_metric": "mlogloss",
        }
        model = XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train,
                                 cv=cv, scoring="f1_weighted", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ─────────────────────────────────────────────────────────────────────
# TRAIN & EVALUATE
# ─────────────────────────────────────────────────────────────────────

def train_regressor(X_train, X_test, y_train, y_test, tune=True):
    """Train XGBRegressor for disposal_days prediction."""
    print("\n" + "="*60)
    print("🌳 XGBRegressor — Disposal Timeline Prediction")
    print("="*60)

    if tune:
        print("🔍 Tuning hyperparameters (Optuna)...")
        best_params = _tune_regressor(X_train, y_train, n_trials=40)
        print(f"   Best params: {best_params}")
    else:
        best_params = {
            "n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
        }

    model = XGBRegressor(
        **best_params,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    print(f"\n📊 Test Metrics:")
    print(f"   RMSE : {rmse:.1f} days")
    print(f"   MAE  : {mae:.1f} days")
    print(f"   R²   : {r2:.4f}")

    path = os.path.join(MODELS_DIR, "xgb_regressor.pkl")
    joblib.dump(model, path)
    print(f"✅ Model saved → {path}")
    return model, {"rmse": rmse, "mae": mae, "r2": r2}


def train_classifier(X_train, X_test, y_train, y_test, tune=True):
    """Train XGBClassifier for delay_risk_label prediction."""
    print("\n" + "="*60)
    print("🎯 XGBClassifier — Delay Risk Label Prediction")
    print("="*60)

    # Encode labels ordinally
    le = LabelEncoder()
    le.classes_ = np.array(RISK_ORDER)
    yc_train_enc = le.transform(y_train)
    yc_test_enc  = le.transform(y_test)

    if tune:
        print("🔍 Tuning hyperparameters (Optuna)...")
        best_params = _tune_classifier(X_train, yc_train_enc, n_trials=40)
        print(f"   Best params: {best_params}")
    else:
        best_params = {
            "n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
        }

    model = XGBClassifier(
        **best_params,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        num_class=4,
        objective="multi:softprob",
        eval_metric="mlogloss",
    )
    model.fit(
        X_train, yc_train_enc,
        eval_set=[(X_test, yc_test_enc)],
        verbose=False,
    )

    y_pred_enc = model.predict(X_test)
    y_prob     = model.predict_proba(X_test)

    acc = accuracy_score(yc_test_enc, y_pred_enc)
    f1  = f1_score(yc_test_enc, y_pred_enc, average="weighted")
    auc = roc_auc_score(yc_test_enc, y_prob, multi_class="ovr", average="weighted")

    print(f"\n📊 Test Metrics:")
    print(f"   Accuracy  : {acc:.4f}")
    print(f"   F1 (wtd)  : {f1:.4f}")
    print(f"   AUC-ROC   : {auc:.4f}")
    print(f"\n📋 Classification Report:")
    print(classification_report(yc_test_enc, y_pred_enc, target_names=RISK_ORDER))

    path = os.path.join(MODELS_DIR, "xgb_classifier.pkl")
    joblib.dump(model, path)

    le_path = os.path.join(MODELS_DIR, "risk_label_encoder.pkl")
    joblib.dump(le, le_path)

    print(f"✅ Model saved → {path}")
    return model, le, {"accuracy": acc, "f1": f1, "auc": auc}


# ─────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────

def load_models():
    """Load saved regressor and classifier from disk."""
    reg = joblib.load(os.path.join(MODELS_DIR, "xgb_regressor.pkl"))
    clf = joblib.load(os.path.join(MODELS_DIR, "xgb_classifier.pkl"))
    le  = joblib.load(os.path.join(MODELS_DIR, "risk_label_encoder.pkl"))
    return reg, clf, le


def predict(X, reg=None, clf=None, le=None):
    """
    Run inference on feature matrix X.
    Returns dict with disposal_days and delay_risk_label.
    """
    if reg is None or clf is None:
        reg, clf, le = load_models()

    disposal_days    = reg.predict(X).astype(int)
    risk_enc         = clf.predict(X)
    risk_proba       = clf.predict_proba(X)
    risk_labels      = le.inverse_transform(risk_enc)
    risk_scores      = (risk_enc / 3.0 * 100).round(1)  # 0–100 scale

    return {
        "disposal_days":      disposal_days,
        "delay_risk_label":   risk_labels,
        "delay_risk_score":   risk_scores,
        "risk_proba":         risk_proba,
    }


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, ROOT)
    from src.preprocessing import load_and_split

    csv_path = os.path.join(DATA_DIR, "njdg_cases.csv")

    # Preprocessing
    X_train, X_test, yr_train, yr_test, yc_train, yc_test, prep = load_and_split(csv_path)

    # Train
    tune = "--no-tune" not in sys.argv
    reg_model, reg_metrics   = train_regressor(X_train, X_test, yr_train, yr_test, tune=tune)
    clf_model, le, clf_metrics = train_classifier(X_train, X_test, yc_train, yc_test, tune=tune)

    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE")
    print(f"   Regressor RMSE : {reg_metrics['rmse']:.1f} days | R²: {reg_metrics['r2']:.3f}")
    print(f"   Classifier AUC : {clf_metrics['auc']:.3f}  | F1: {clf_metrics['f1']:.3f}")
    print("="*60)
