"""
priority_queue.py
Court-level priority queue and alert generator (Component 6).
Ranks all cases within each court and generates actionable alerts.
"""

import pandas as pd
import numpy as np
from typing import List, Dict


# ── Alert thresholds
UNDERTRIAL_ALERT_DAYS    = 365   # BNSS 479 — alert if undertrial > 1 year
STAGNATION_ALERT_DAYS    = 180   # Alert if a case is stuck in same stage > 6 months
CRITICAL_URGENCY_CUTOFF  = 75    # Priority score threshold for "Critical" alert
PROLONGED_PENDENCY_YEARS = 5     # Alert if case pending > 5 years


class CourtPriorityQueue:
    """
    Builds per-court ranked dockets and generates alerts.
    """

    def build_docket(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rank all cases within each court by priority_score (descending).
        Returns the full dataframe with rank and docket columns added.
        """
        df = df.copy()

        # Rank within each court
        df["court_rank"] = df.groupby("court_id")["priority_score"].rank(
            ascending=False, method="first"
        ).astype(int)

        # Docket label
        df["docket_label"] = df.apply(
            lambda r: f"[#{int(r['court_rank'])}] {r['priority_emoji']} {r['priority_label']}",
            axis=1,
        )

        return df.sort_values(["court_id", "court_rank"])

    def generate_alerts(self, df: pd.DataFrame) -> List[Dict]:
        """
        Scan the dataframe and return a list of alert dicts.
        Each alert has: case_number, court_id, alert_type, message, severity.
        """
        alerts = []

        for _, row in df.iterrows():
            case_id  = row.get("case_number", "Unknown")
            court_id = row.get("court_id", "Unknown")
            category = row.get("case_category", "")

            # ── Critical priority alert
            if row.get("priority_score", 0) >= CRITICAL_URGENCY_CUTOFF:
                alerts.append({
                    "case_number": case_id,
                    "court_id":    court_id,
                    "alert_type":  "CRITICAL_PRIORITY",
                    "message":     f"🔴 Critical priority case [{category}] — Immediate attention required. Score: {row['priority_score']:.1f}",
                    "severity":    "CRITICAL",
                })

            # ── Undertrial threshold (BNSS 479)
            if row.get("is_undertrial", 0) == 1 and row.get("case_age_days", 0) > UNDERTRIAL_ALERT_DAYS:
                alerts.append({
                    "case_number": case_id,
                    "court_id":    court_id,
                    "alert_type":  "UNDERTRIAL_THRESHOLD",
                    "message":     f"🔴 BNSS 479: Undertrial pending {int(row['case_age_days'])} days. Bail eligibility to be reviewed.",
                    "severity":    "CRITICAL",
                })

            # ── Prolonged stagnation
            if row.get("stagnation_flag", 0) == 1 and row.get("stage_age_days", 0) > STAGNATION_ALERT_DAYS:
                alerts.append({
                    "case_number": case_id,
                    "court_id":    court_id,
                    "alert_type":  "STAGE_STAGNATION",
                    "message":     f"🟠 Stagnant in '{row.get('current_stage', '?')}' stage for {int(row['stage_age_days'])} days.",
                    "severity":    "HIGH",
                })

            # ── Long-pending case alert
            if row.get("case_age_days", 0) > PROLONGED_PENDENCY_YEARS * 365:
                years = row["case_age_days"] / 365
                alerts.append({
                    "case_number": case_id,
                    "court_id":    court_id,
                    "alert_type":  "LONG_PENDING",
                    "message":     f"📛 Case pending for {years:.1f} years [{category}].",
                    "severity":    "HIGH",
                })

            # ── Statutory deadline breach
            flags = str(row.get("urgency_flags", ""))
            if "STATUTORY DEADLINE BREACHED" in flags:
                alerts.append({
                    "case_number": case_id,
                    "court_id":    court_id,
                    "alert_type":  "STATUTORY_BREACH",
                    "message":     f"⚠️ Statutory deadline breached for [{category}] case.",
                    "severity":    "CRITICAL",
                })

        return alerts

    def get_court_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a per-court summary DataFrame with counts by priority label.
        """
        label_order = ["Critical", "High", "Medium", "Low"]
        summary = df.groupby(["court_id", "priority_label"]).size().unstack(fill_value=0)
        for label in label_order:
            if label not in summary.columns:
                summary[label] = 0
        summary = summary[label_order]
        summary["Total"] = summary.sum(axis=1)
        summary["Avg Priority Score"] = df.groupby("court_id")["priority_score"].mean().round(1)
        return summary.reset_index().sort_values("Critical", ascending=False)

    def get_top_cases(self, df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
        """Return top-n highest priority cases across all courts."""
        cols = [
            "case_number", "court_id", "state", "district",
            "case_category", "case_type", "current_stage",
            "case_age_days", "priority_score", "priority_label",
            "priority_emoji", "disposal_days", "legal_urgency_score",
            "delay_risk_score", "urgency_flags",
        ]
        available = [c for c in cols if c in df.columns]
        return (
            df[available]
            .sort_values("priority_score", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )


if __name__ == "__main__":
    # Quick test with dummy data
    dummy = pd.DataFrame({
        "case_number":        ["C001", "C002", "C003"],
        "court_id":           ["MH-MUM-01", "MH-MUM-01", "KA-BLR-02"],
        "state":              ["Maharashtra", "Maharashtra", "Karnataka"],
        "district":           ["Mumbai", "Mumbai", "Bengaluru"],
        "case_category":      ["POCSO", "General_Civil", "NDPS"],
        "case_type":          ["Criminal", "Civil", "Criminal"],
        "current_stage":      ["Evidence", "Filing", "Arguments"],
        "case_age_days":      [450, 300, 800],
        "stage_age_days":     [200, 50, 120],
        "priority_score":     [82.0, 30.0, 65.0],
        "priority_label":     ["Critical", "Low", "High"],
        "priority_emoji":     ["🔴", "🟢", "🟠"],
        "stagnation_flag":    [1, 0, 1],
        "is_undertrial":      [1, 0, 1],
        "urgency_flags":      ["⚠️ STATUTORY DEADLINE BREACHED", "", "🔔 Approaching statutory deadline"],
        "disposal_days":      [90, 400, 200],
        "legal_urgency_score":[80, 10, 55],
        "delay_risk_score":   [78, 20, 60],
    })

    pq = CourtPriorityQueue()
    docket  = pq.build_docket(dummy)
    alerts  = pq.generate_alerts(docket)
    summary = pq.get_court_summary(docket)

    print("📋 Docket:\n", docket[["case_number", "court_id", "docket_label"]].to_string())
    print("\n🚨 Alerts:")
    for a in alerts:
        print(f"  [{a['severity']}] {a['message']} — {a['case_number']}")
    print("\n📊 Court Summary:\n", summary.to_string())
