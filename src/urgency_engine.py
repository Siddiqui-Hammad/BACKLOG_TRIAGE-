"""
urgency_engine.py
Rule-based urgency scoring engine based on Indian legal statutes.
Implements the logic from Component 2 of the Technical Approach diagram.

Statutory references:
  - POCSO Act 2012: Section 35 — trial within 1 year
  - SC/ST (PoA) Act 1989: Rule 7 — trial within 2 months of charge-sheet
  - Commercial Courts Act 2015: Section 12A & 16 — 6-month disposal
  - BNSS 2023 Section 479: Undertrial bail after 1/3 of max sentence
  - Senior Citizens Act 2007: Speedy disposal mandate
  - Matrimonial: CPC Order 32A — expedite
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


# ─────────────────────────────────────────────────────
# STATUTORY THRESHOLDS (in days)
# ─────────────────────────────────────────────────────
STATUTORY_LIMITS = {
    "POCSO":           365,    # 1 year
    "SC_ST":           730,    # 2 years (practical target)
    "Commercial":      180,    # 6 months (Commercial Courts Act)
    "Senior_Citizen":  180,    # 6 months
    "NDPS":            1095,   # 3 years
    "Matrimonial":     1825,   # 5 years (target)
    "Motor_Accident":  1825,
    "Property_Dispute":2920,
    "General_Civil":   2190,
    "General_Criminal":730,
}

# Stagnation thresholds per stage (days)
STAGE_STAGNATION_LIMITS = {
    "Filing":            15,
    "Admission":         30,
    "Notice":            60,
    "Written_Statement": 90,
    "Evidence":          180,
    "Arguments":         90,
    "Judgment":          60,
    "Execution":         120,
}

# BNSS 479: Maximum sentences for common offences (days)
BNSS_479_MAX_SENTENCES = {
    "NDPS":            3650,   # 10 years max
    "General_Criminal":1825,   # 5 years max (default)
}


class UrgencyEngine:
    """
    Deterministic rule-based urgency scorer.
    Returns legal_urgency_score (0–100) and urgency_flags list.
    """

    def score_case(self, case: Dict) -> Tuple[float, List[str]]:
        """
        Score a single case dict.
        Returns: (urgency_score: float, flags: List[str])
        """
        flags = []
        score = 0.0

        category = case.get("case_category", "General_Civil")
        case_age = int(case.get("case_age_days", 0))
        stage = case.get("current_stage", "Filing")
        stage_age = int(case.get("stage_age_days", 0))
        case_type = case.get("case_type", "Civil")
        hearings = int(case.get("hearings_held", 0))
        adjournments = int(case.get("adjournments", 0))

        statutory_limit = STATUTORY_LIMITS.get(category, 2190)

        # ── 1. Statutory deadline breach or proximity
        breach_ratio = case_age / max(statutory_limit, 1)
        if breach_ratio >= 1.0:
            score += 35
            flags.append(f"⚠️ STATUTORY DEADLINE BREACHED ({case_age}d > {statutory_limit}d)")
        elif breach_ratio >= 0.75:
            score += 25
            flags.append(f"🔔 Approaching statutory deadline ({int(breach_ratio*100)}% elapsed)")
        elif breach_ratio >= 0.5:
            score += 15
            flags.append(f"📅 Past halfway of statutory limit")

        # ── 2. Category-specific high-urgency flags (POCSO, SC/ST, Senior Citizen)
        if category == "POCSO":
            score += 20
            flags.append("🔴 POCSO: Mandatory speedy trial (Sec 35)")
        elif category == "SC_ST":
            score += 18
            flags.append("🔴 SC/ST PoA: Priority disposal required")
        elif category == "Senior_Citizen":
            score += 15
            flags.append("🟠 Senior Citizen: Expedited disposal mandated")

        # ── 3. BNSS Section 479 — Undertrial threshold
        if case.get("is_undertrial", 0) == 1:
            max_sentence = BNSS_479_MAX_SENTENCES.get(category, 1825)
            one_third = max_sentence / 3
            if case_age >= one_third:
                score += 20
                flags.append(f"🔴 BNSS 479: Undertrial exceeds 1/3 max sentence ({case_age}d ≥ {int(one_third)}d)")

        # ── 4. Stagnation detection
        stagnation_limit = STAGE_STAGNATION_LIMITS.get(stage, 90)
        if stage_age > stagnation_limit * 2:
            score += 15
            flags.append(f"🟥 Critical stagnation: {stage_age}d in '{stage}' stage (limit {stagnation_limit}d)")
        elif stage_age > stagnation_limit * 1.5:
            score += 10
            flags.append(f"🟠 Stagnation detected: {stage_age}d in '{stage}' stage")

        # ── 5. Adjournment abuse detection
        adj_rate = adjournments / max(hearings, 1)
        if adj_rate >= 0.6:
            score += 10
            flags.append(f"🔁 High adjournment rate: {adj_rate:.0%} ({adjournments}/{hearings} hearings)")

        # ── 6. Commercial courts — additional Commercial Courts Act check
        if category == "Commercial" and case_age > 180:
            score += 10
            flags.append("⚖️ Commercial Courts Act: Disposal overdue (>6 months)")

        # ── 7. Case age scoring (general aging penalty)
        if case_age > 1825:
            score += 10
            flags.append(f"📛 Aged case: {case_age // 365}+ years pending")
        elif case_age > 1095:
            score += 5
            flags.append(f"📛 Long-pending: {case_age // 365}+ years")

        urgency_score = float(np.clip(score, 0, 100))
        return urgency_score, flags

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply urgency scoring to an entire DataFrame. Returns df with new columns."""
        results = df.to_dict(orient="records")
        scores, flags = [], []
        for case in results:
            s, f = self.score_case(case)
            scores.append(s)
            flags.append("; ".join(f) if f else "No urgent flags")
        df = df.copy()
        df["legal_urgency_score"] = scores
        df["urgency_flags"] = flags
        return df


if __name__ == "__main__":
    # Quick test
    engine = UrgencyEngine()
    test_case = {
        "case_category": "POCSO",
        "case_age_days": 400,
        "current_stage": "Evidence",
        "stage_age_days": 200,
        "case_type": "Criminal",
        "hearings_held": 10,
        "adjournments": 7,
        "is_undertrial": 0,
    }
    score, flags = engine.score_case(test_case)
    print(f"Urgency Score : {score}")
    print(f"Flags         :")
    for f in flags:
        print(f"  {f}")
