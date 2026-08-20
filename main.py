"""
main.py
End-to-end pipeline entry point for the NJDG Case Prioritization System.

Usage:
  python main.py --generate   # Generate synthetic dataset + train models
  python main.py --train      # Train models only (dataset must exist)
  python main.py --score      # Run full scoring pipeline on dataset
  python main.py --all        # Do everything (default)
"""

import os
import sys
import io
import argparse
import pandas as pd

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

DATA_PATH   = os.path.join(ROOT, "data", "njdg_cases.csv")
MODELS_DIR  = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "models")

os.makedirs(MODELS_DIR, exist_ok=True)


def step_generate():
    print("\n" + "━"*60)
    print("📦 STEP 1: Generating Synthetic NJDG Dataset")
    print("━"*60)
    from data.generate_njdg_dataset import generate_cases
    df = generate_cases(n=10000)
    df.to_csv(DATA_PATH, index=False)
    print(f"✅ Dataset: {df.shape[0]} cases saved → {DATA_PATH}")
    print(f"   Categories : {df['case_category'].value_counts().to_dict()}")
    print(f"   Risk Labels: {df['delay_risk_label'].value_counts().to_dict()}")
    return df


def step_train():
    print("\n" + "━"*60)
    print("🌳 STEP 2 & 3: Preprocessing + XGBoost Training")
    print("━"*60)
    from src.preprocessing import load_and_split
    from src.xgboost_model import train_regressor, train_classifier

    X_train, X_test, yr_train, yr_test, yc_train, yc_test, prep = load_and_split(DATA_PATH)

    reg_model, reg_metrics     = train_regressor(X_train, X_test, yr_train, yr_test, tune=True)
    clf_model, le, clf_metrics = train_classifier(X_train, X_test, yc_train, yc_test, tune=True)

    print("\n✅ Training Summary:")
    print(f"   Regressor  — RMSE: {reg_metrics['rmse']:.1f} days | R²: {reg_metrics['r2']:.3f}")
    print(f"   Classifier — AUC: {clf_metrics['auc']:.3f} | F1: {clf_metrics['f1']:.3f}")
    return reg_model, clf_model, le, prep


def step_score():
    print("\n" + "━"*60)
    print("🎯 STEP 4–6: Urgency + Priority Scoring + Queue")
    print("━"*60)

    from src.urgency_engine import UrgencyEngine
    from src.preprocessing import NJDGPreprocessor
    from src.xgboost_model import load_models, predict
    from src.priority_engine import HybridPriorityEngine
    from src.priority_queue import CourtPriorityQueue

    df = pd.read_csv(DATA_PATH)

    # ── Step 2: Urgency Engine (score raw df first, then realign)
    print("⚖️  Running Urgency Engine...")
    urgency_engine = UrgencyEngine()

    # ── Step 3: ML Prediction — preprocess first to get aligned df
    print("🤖 Running XGBoost Predictions...")
    prep = NJDGPreprocessor.load()
    df_proc, X = prep.transform(df)          # df_proc has cleaned/aligned rows

    # Score urgency on the clean aligned df
    df_proc = urgency_engine.score_dataframe(df_proc)

    reg, clf, le = load_models()
    preds = predict(X, reg, clf, le)

    # Assign predictions — same row count guaranteed
    df_proc["predicted_disposal_days"] = preds["disposal_days"]
    df_proc["ml_delay_risk_score"]     = preds["delay_risk_score"]
    df_proc["ml_delay_risk_label"]     = preds["delay_risk_label"]

    # Use aligned df going forward
    df = df_proc.copy()
    df["disposal_days"]    = df["predicted_disposal_days"]
    df["delay_risk_score"] = df["ml_delay_risk_score"]

    # ── Step 5: Hybrid Priority Engine
    print("🔀 Computing Hybrid Priority Scores...")
    priority_engine = HybridPriorityEngine()
    df = priority_engine.score_dataframe(df)

    # ── Step 6: Court Priority Queue
    print("📋 Building Court Priority Queues...")
    pq = CourtPriorityQueue()
    df = pq.build_docket(df)
    alerts = pq.generate_alerts(df)
    summary = pq.get_court_summary(df)
    top_cases = pq.get_top_cases(df, n=20)

    # ── Save results
    out_scored  = os.path.join(RESULTS_DIR, "scored_cases.csv")
    out_summary = os.path.join(RESULTS_DIR, "court_summary.csv")
    out_alerts  = os.path.join(RESULTS_DIR, "alerts.csv")
    out_top     = os.path.join(RESULTS_DIR, "top_priority_cases.csv")

    df.to_csv(out_scored, index=False)
    summary.to_csv(out_summary, index=False)
    pd.DataFrame(alerts).to_csv(out_alerts, index=False)
    top_cases.to_csv(out_top, index=False)

    print(f"\n✅ Results saved:")
    print(f"   Scored cases     → {out_scored}")
    print(f"   Court summary    → {out_summary}")
    print(f"   Alerts           → {out_alerts}")
    print(f"   Top priority     → {out_top}")

    print(f"\n🚨 Total Alerts Generated: {len(alerts)}")
    crit = sum(1 for a in alerts if a["severity"] == "CRITICAL")
    high = sum(1 for a in alerts if a["severity"] == "HIGH")
    print(f"   Critical: {crit}  |  High: {high}")

    print(f"\n📊 Priority Label Distribution:")
    print(df["priority_label"].value_counts().to_string())

    print(f"\n🏆 Top 5 Highest Priority Cases:")
    display_cols = ["case_number", "court_id", "case_category", "priority_score", "priority_label", "disposal_days"]
    print(top_cases[[c for c in display_cols if c in top_cases.columns]].head(5).to_string())

    return df, alerts, summary


def main():
    parser = argparse.ArgumentParser(description="Backlog Triage Case — ML Pipeline")
    parser.add_argument("--generate", action="store_true", help="Generate dataset")
    parser.add_argument("--train",    action="store_true", help="Train models")
    parser.add_argument("--score",    action="store_true", help="Run scoring pipeline")
    parser.add_argument("--all",      action="store_true", help="Run all steps (default)")
    args = parser.parse_args()

    run_all = args.all or not any([args.generate, args.train, args.score])

    print("\n" + "="*60)
    print("  ⚖️  Backlog Triage Case")
    print("  Smart India Hackathon 2026 — git win")
    print("="*60)

    if run_all or args.generate:
        step_generate()

    if run_all or args.train:
        step_train()

    if run_all or args.score:
        step_score()

    print("\n🎉 Pipeline complete! Launch dashboard with:")
    print("   streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
