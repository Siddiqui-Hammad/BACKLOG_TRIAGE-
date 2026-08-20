"""
priority_engine.py
Hybrid Priority Engine (Component 5 from Technical Approach diagram).

Combines rule-based urgency scores + ML delay risk scores using
the exact weights from the diagram:
  - Statutory Urgency : 35%
  - Case Age          : 25%
  - Category Aging    : 20%
  - ML Delay Risk     : 20%

Outputs: priority_score (0–100), priority_label, explanation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple

# ── Priority score weights (from diagram)
WEIGHTS = {
    "statutory_urgency": 0.35,
    "case_age":          0.25,
    "category_aging":    0.20,
    "ml_delay_risk":     0.20,
}

# ── Thresholds for category aging score (normalized to 0–100)
CATEGORY_AGE_THRESHOLDS = {
    "POCSO":           365,
    "SC_ST":           730,
    "Senior_Citizen":  180,
    "Commercial":      180,
    "NDPS":            1095,
    "Matrimonial":     1825,
    "Motor_Accident":  1825,
    "Property_Dispute":2920,
    "General_Civil":   2190,
    "General_Criminal":730,
}

# ── Case age score thresholds (days)
AGE_SCORE_BINS   = [0, 180, 365, 730, 1095, 1825, 3650]
AGE_SCORE_VALUES = [0,  15,  35,  55,   70,   85,  100]


class HybridPriorityEngine:
    """
    Combines legal urgency rules + ML predictions into a final priority score.
    """

    def compute_case_age_score(self, case_age_days: int) -> float:
        """Convert case age in days to a 0–100 score."""
        score = np.interp(case_age_days, AGE_SCORE_BINS, AGE_SCORE_VALUES)
        return float(np.clip(score, 0, 100))

    def compute_category_aging_score(self, category: str, case_age_days: int) -> float:
        """Score 0–100 based on how far beyond the category norm the case is."""
        threshold = CATEGORY_AGE_THRESHOLDS.get(category, 2190)
        ratio = case_age_days / max(threshold, 1)
        score = min(100, ratio * 70)  # 100% overdue → 70, then extra penalty
        if ratio >= 1.5:
            score = min(100, score + 15)
        if ratio >= 2.0:
            score = 100
        return float(np.clip(score, 0, 100))

    def score_case(
        self,
        legal_urgency_score: float,
        case_age_days: int,
        case_category: str,
        ml_delay_risk_score: float,
        urgency_flags=None,
        shap_explanation: str = "",
        disposal_days: int = None,
    ) -> Dict:
        """
        Compute hybrid priority score for a single case.

        Parameters
        ----------
        legal_urgency_score  : 0–100 from UrgencyEngine
        case_age_days        : integer
        case_category        : string category name
        ml_delay_risk_score  : 0–100 from XGBoost classifier
        urgency_flags        : list of flag strings
        shap_explanation     : text from CaseExplainer
        disposal_days        : predicted disposal days from XGBRegressor

        Returns
        -------
        dict with priority_score, priority_label, explanation, breakdown
        """
        age_score      = self.compute_case_age_score(case_age_days)
        cat_age_score  = self.compute_category_aging_score(case_category, case_age_days)

        priority_score = (
            WEIGHTS["statutory_urgency"] * legal_urgency_score
            + WEIGHTS["case_age"]        * age_score
            + WEIGHTS["category_aging"]  * cat_age_score
            + WEIGHTS["ml_delay_risk"]   * ml_delay_risk_score
        )
        priority_score = float(np.clip(priority_score, 0, 100))

        # ── Label
        if priority_score >= 75:
            priority_label = "Critical"
            label_emoji    = "🔴"
        elif priority_score >= 50:
            priority_label = "High"
            label_emoji    = "🟠"
        elif priority_score >= 25:
            priority_label = "Medium"
            label_emoji    = "🟡"
        else:
            priority_label = "Low"
            label_emoji    = "🟢"

        # ── Build explanation
        breakdown = {
            "Statutory Urgency (35%)": round(legal_urgency_score, 1),
            "Case Age Score (25%)":    round(age_score, 1),
            "Category Aging (20%)":    round(cat_age_score, 1),
            "ML Delay Risk (20%)":     round(ml_delay_risk_score, 1),
        }

        exp_lines = [
            f"**Priority: {label_emoji} {priority_label}** (Score: {priority_score:.1f}/100)\n",
            "**Score Breakdown:**",
        ]
        for comp, val in breakdown.items():
            exp_lines.append(f"  • {comp}: {val}")

        if urgency_flags:
            exp_lines.append("\n**Legal Flags:**")
            for flag in (urgency_flags if isinstance(urgency_flags, list) else urgency_flags.split("; ")):
                if flag.strip():
                    exp_lines.append(f"  {flag.strip()}")

        if shap_explanation:
            exp_lines.append("\n**ML Contribution Factors:**")
            exp_lines.append(shap_explanation)

        if disposal_days:
            months = disposal_days / 30
            exp_lines.append(f"\n**Predicted Disposal:** ~{disposal_days} days ({months:.1f} months)")

        return {
            "priority_score": round(priority_score, 2),
            "priority_label": priority_label,
            "priority_emoji": label_emoji,
            "breakdown":      breakdown,
            "explanation":    "\n".join(exp_lines),
        }

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply hybrid scoring to a full DataFrame.
        Expects columns: legal_urgency_score, case_age_days, case_category,
                         delay_risk_score, urgency_flags, disposal_days
        """
        df = df.copy()
        scores, labels, emojis = [], [], []

        for _, row in df.iterrows():
            result = self.score_case(
                legal_urgency_score  = float(row.get("legal_urgency_score", 0)),
                case_age_days        = int(row.get("case_age_days", 0)),
                case_category        = str(row.get("case_category", "General_Civil")),
                ml_delay_risk_score  = float(row.get("delay_risk_score", 0)),
                urgency_flags        = row.get("urgency_flags", ""),
                disposal_days        = int(row.get("disposal_days", 0)),
            )
            scores.append(result["priority_score"])
            labels.append(result["priority_label"])
            emojis.append(result["priority_emoji"])

        df["priority_score"] = scores
        df["priority_label"] = labels
        df["priority_emoji"] = emojis
        return df


if __name__ == "__main__":
    engine = HybridPriorityEngine()

    sample = engine.score_case(
        legal_urgency_score  = 65.0,
        case_age_days        = 800,
        case_category        = "POCSO",
        ml_delay_risk_score  = 78.0,
        urgency_flags        = ["⚠️ STATUTORY DEADLINE BREACHED", "🔴 POCSO: Mandatory speedy trial"],
        shap_explanation     = "• Case Age increases delay risk by 12.3 points\n• Adjournment Rate increases by 8.1 points",
        disposal_days        = 210,
    )

    print(f"\n🎯 Priority Score : {sample['priority_score']}")
    print(f"   Label         : {sample['priority_emoji']} {sample['priority_label']}")
    print(f"\n{sample['explanation']}")
