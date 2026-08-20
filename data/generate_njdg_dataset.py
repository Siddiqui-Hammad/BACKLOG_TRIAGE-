"""
generate_njdg_dataset.py
Generates a realistic synthetic dataset mimicking NJDG (National Judicial Data Grid) case records.
Produces ~10,000 case records with features matching the Technical Approach diagram.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

# ─────────────────────────────────────────────
# CONSTANTS — mirroring real NJDG categories
# ─────────────────────────────────────────────

STATES = [
    "Maharashtra", "Uttar Pradesh", "Karnataka", "Tamil Nadu", "Rajasthan",
    "Gujarat", "West Bengal", "Madhya Pradesh", "Bihar", "Delhi"
]

DISTRICTS = {
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik"],
    "Uttar Pradesh": ["Lucknow", "Allahabad", "Varanasi", "Kanpur"],
    "Karnataka": ["Bengaluru", "Mysuru", "Hubballi", "Mangaluru"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Salem"],
    "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur", "Kota"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot"],
    "West Bengal": ["Kolkata", "Howrah", "Durgapur", "Siliguri"],
    "Madhya Pradesh": ["Bhopal", "Indore", "Gwalior", "Jabalpur"],
    "Bihar": ["Patna", "Gaya", "Muzaffarpur", "Bhagalpur"],
    "Delhi": ["Central", "South", "North", "East"],
}

CASE_CATEGORIES = {
    "POCSO": {"base_days": 120, "urgency_weight": 1.0, "statutory_days": 365},
    "SC_ST": {"base_days": 180, "urgency_weight": 0.9, "statutory_days": 730},
    "Senior_Citizen": {"base_days": 90, "urgency_weight": 0.85, "statutory_days": 180},
    "Commercial": {"base_days": 270, "urgency_weight": 0.7, "statutory_days": 365},
    "NDPS": {"base_days": 365, "urgency_weight": 0.75, "statutory_days": 1095},
    "Matrimonial": {"base_days": 540, "urgency_weight": 0.5, "statutory_days": 1825},
    "Motor_Accident": {"base_days": 730, "urgency_weight": 0.4, "statutory_days": 1825},
    "Property_Dispute": {"base_days": 1095, "urgency_weight": 0.3, "statutory_days": 2920},
    "General_Civil": {"base_days": 900, "urgency_weight": 0.25, "statutory_days": 2190},
    "General_Criminal": {"base_days": 450, "urgency_weight": 0.6, "statutory_days": 730},
}

CASE_STAGES = [
    "Filing", "Admission", "Notice", "Written_Statement",
    "Evidence", "Arguments", "Judgment", "Execution"
]

BENCH_TYPES = ["Single_Judge", "Division_Bench", "Full_Bench", "Magistrate", "Sessions"]

CASE_TYPES = ["Civil", "Criminal", "Writ", "Appeal", "Revision", "Execution"]


def random_date(start_year=2015, end_year=2024):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))


def generate_cases(n=10000, seed=42):
    """Generate n synthetic NJDG case records."""
    np.random.seed(seed)
    random.seed(seed)

    records = []

    for i in range(n):
        # ── Location
        state = random.choice(STATES)
        district = random.choice(DISTRICTS[state])
        court_id = f"{state[:3].upper()}-{district[:3].upper()}-{random.randint(1, 20):02d}"

        # ── Case identity
        category = random.choice(list(CASE_CATEGORIES.keys()))
        cat_info = CASE_CATEGORIES[category]
        case_type = random.choice(CASE_TYPES)
        case_number = f"CASE/{random.randint(1, 9999):04d}/{random.randint(2015, 2024)}"

        # ── Dates
        filing_date = random_date(2015, 2023)
        current_date = datetime(2024, 12, 31)
        case_age_days = (current_date - filing_date).days

        # ── Stage
        stage_idx = min(int(case_age_days / (cat_info["base_days"] / 8)), 7)
        stage_idx = max(0, stage_idx + np.random.randint(-2, 3))
        stage_idx = min(stage_idx, 7)
        current_stage = CASE_STAGES[stage_idx]
        stage_age_days = int(np.random.exponential(scale=cat_info["base_days"] / 8))
        stage_age_days = min(stage_age_days, case_age_days)

        # ── Hearings & Adjournments
        expected_hearings = max(1, case_age_days // 30)
        hearings_held = int(np.random.poisson(expected_hearings * 0.7))
        adjournments = int(np.random.poisson(hearings_held * 0.35))
        adjournments = min(adjournments, hearings_held)
        adjournment_rate = adjournments / max(1, hearings_held)

        # ── Bench
        bench_type = random.choice(BENCH_TYPES)
        judge_id = f"JDG-{random.randint(100, 999)}"

        # ── Stagnation
        stagnation_threshold = cat_info["base_days"] / 8
        stagnation_flag = int(stage_age_days > stagnation_threshold * 1.5)

        # ── BNSS 479 undertrial (criminal cases pending > 1/3 of max sentence)
        is_undertrial = int(
            case_type == "Criminal"
            and case_age_days > 365
            and category in ["NDPS", "General_Criminal"]
        )

        # ── Historical disposal pattern (district-level avg in days)
        district_avg_disposal = int(cat_info["base_days"] * np.random.uniform(0.8, 1.4))

        # ─────────────────────────────────────
        # TARGET VARIABLES
        # ─────────────────────────────────────

        # Disposal days = remaining days from now to disposal
        base_remaining = max(30, cat_info["base_days"] - case_age_days)
        delay_factor = 1.0
        delay_factor += adjournment_rate * 0.5
        delay_factor += stagnation_flag * 0.3
        delay_factor += is_undertrial * 0.2
        delay_factor += (stage_idx < 3) * 0.4  # early stages take longer
        noise = np.random.normal(0, 60)
        disposal_days = max(10, int(base_remaining * delay_factor + noise))

        # Delay risk score (0–100)
        risk = 0.0
        risk += min(40, (case_age_days / cat_info["statutory_days"]) * 40)
        risk += adjournment_rate * 20
        risk += stagnation_flag * 15
        risk += is_undertrial * 15
        risk += (7 - stage_idx) / 7 * 10  # earlier stage = higher risk
        delay_risk_score = float(np.clip(risk + np.random.normal(0, 5), 0, 100))

        # Delay risk bucket label
        if delay_risk_score >= 75:
            delay_risk_label = "Critical"
        elif delay_risk_score >= 50:
            delay_risk_label = "High"
        elif delay_risk_score >= 25:
            delay_risk_label = "Medium"
        else:
            delay_risk_label = "Low"

        records.append({
            # ── Identifiers
            "case_number": case_number,
            "court_id": court_id,
            "state": state,
            "district": district,
            "judge_id": judge_id,

            # ── Case metadata
            "case_type": case_type,
            "case_category": category,
            "bench_type": bench_type,
            "current_stage": current_stage,
            "stage_index": stage_idx,

            # ── Dates & Age
            "filing_date": filing_date.strftime("%Y-%m-%d"),
            "case_age_days": case_age_days,
            "stage_age_days": stage_age_days,

            # ── Hearings
            "hearings_held": hearings_held,
            "adjournments": adjournments,
            "adjournment_rate": round(adjournment_rate, 3),

            # ── Flags
            "stagnation_flag": stagnation_flag,
            "is_undertrial": is_undertrial,

            # ── Historical
            "district_avg_disposal_days": district_avg_disposal,
            "statutory_deadline_days": cat_info["statutory_days"],
            "days_beyond_statutory": max(0, case_age_days - cat_info["statutory_days"]),

            # ── Targets
            "disposal_days": disposal_days,
            "delay_risk_score": round(delay_risk_score, 2),
            "delay_risk_label": delay_risk_label,
        })

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    print("🔄 Generating synthetic NJDG case dataset...")
    df = generate_cases(n=10000)

    out_path = os.path.join(os.path.dirname(__file__), "njdg_cases.csv")
    df.to_csv(out_path, index=False)

    print(f"✅ Dataset saved: {out_path}")
    print(f"   Shape     : {df.shape}")
    print(f"   Categories: {df['case_category'].value_counts().to_dict()}")
    print(f"   Risk Labels: {df['delay_risk_label'].value_counts().to_dict()}")
    print(f"\n📊 Sample:\n{df.head(3).T}")
