"""
Backlog Triage Case — Smart India Hackathon 2026
Self-contained Snowflake Streamlit app.
Paste this entire file into your Snowflake Streamlit editor and click Run.

Packages used (all available in Snowflake Anaconda channel):
  streamlit, pandas, numpy, scikit-learn, xgboost, plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import random
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Backlog Triage Case",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  section[data-testid="stSidebar"] { background: #1a1a2e; }
  section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
  .stMetric { background: white; border-radius: 10px; padding: 8px;
               box-shadow: 0 2px 6px rgba(0,0,0,.12); }
  h1 { color: #1a1a2e; }
</style>
""", unsafe_allow_html=True)

PRIORITY_COLORS = {"Critical":"#dc3545","High":"#fd7e14","Medium":"#ffc107","Low":"#28a745"}

# ─────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────
STATES = ["Maharashtra","Uttar Pradesh","Karnataka","Tamil Nadu","Rajasthan",
          "Gujarat","West Bengal","Madhya Pradesh","Bihar","Delhi"]

DISTRICTS = {
    "Maharashtra":["Mumbai","Pune","Nagpur","Nashik"],
    "Uttar Pradesh":["Lucknow","Allahabad","Varanasi","Kanpur"],
    "Karnataka":["Bengaluru","Mysuru","Hubballi","Mangaluru"],
    "Tamil Nadu":["Chennai","Coimbatore","Madurai","Salem"],
    "Rajasthan":["Jaipur","Jodhpur","Udaipur","Kota"],
    "Gujarat":["Ahmedabad","Surat","Vadodara","Rajkot"],
    "West Bengal":["Kolkata","Howrah","Durgapur","Siliguri"],
    "Madhya Pradesh":["Bhopal","Indore","Gwalior","Jabalpur"],
    "Bihar":["Patna","Gaya","Muzaffarpur","Bhagalpur"],
    "Delhi":["Central","South","North","East"],
}

CATEGORIES = {
    "POCSO":            {"base":120,"statutory":365},
    "SC_ST":            {"base":180,"statutory":730},
    "Senior_Citizen":   {"base":90, "statutory":180},
    "Commercial":       {"base":270,"statutory":365},
    "NDPS":             {"base":365,"statutory":1095},
    "Matrimonial":      {"base":540,"statutory":1825},
    "Motor_Accident":   {"base":730,"statutory":1825},
    "Property_Dispute": {"base":1095,"statutory":2920},
    "General_Civil":    {"base":900,"statutory":2190},
    "General_Criminal": {"base":450,"statutory":730},
}

STAGES = ["Filing","Admission","Notice","Written_Statement",
          "Evidence","Arguments","Judgment","Execution"]

STAGE_STAGNATION = {
    "Filing":15,"Admission":30,"Notice":60,"Written_Statement":90,
    "Evidence":180,"Arguments":90,"Judgment":60,"Execution":120,
}

FEATURE_COLS = [
    "case_type_enc","case_category_enc","bench_type_enc","current_stage_enc","state_enc",
    "stage_index","case_age_days","stage_age_days","hearings_held","adjournments",
    "adjournment_rate","stagnation_flag","is_undertrial","district_avg_disposal_days",
    "statutory_deadline_days","days_beyond_statutory","case_age_normalized",
    "hearing_density","stage_completion_ratio",
]

# ─────────────────────────────────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────────────────────────────────
def generate_cases(n=3000, seed=42):
    np.random.seed(seed); random.seed(seed)
    rows = []
    cat_names = list(CATEGORIES.keys())
    bench_types = ["Single_Judge","Division_Bench","Full_Bench","Magistrate","Sessions"]
    case_types  = ["Civil","Criminal","Writ","Appeal","Revision","Execution"]

    for i in range(n):
        state    = random.choice(STATES)
        district = random.choice(DISTRICTS[state])
        court_id = f"{state[:3].upper()}-{district[:3].upper()}-{random.randint(1,15):02d}"
        category = random.choice(cat_names)
        info     = CATEGORIES[category]
        case_type= random.choice(case_types)

        case_age = random.randint(60, 3500)
        stage_idx= min(int(case_age/(info["base"]/8)) + np.random.randint(-2,3), 7)
        stage_idx= max(0, stage_idx)
        stage    = STAGES[stage_idx]
        stage_age= int(np.random.exponential(info["base"]/8))
        stage_age= min(stage_age, case_age)

        hearings   = max(1, case_age // 30)
        hearings   = int(np.random.poisson(hearings * 0.7))
        adj        = int(np.random.poisson(hearings * 0.35))
        adj        = min(adj, hearings)
        adj_rate   = adj / max(1, hearings)

        stagnation = int(stage_age > STAGE_STAGNATION.get(stage, 90) * 1.5)
        undertrial = int(case_type == "Criminal" and case_age > 365
                        and category in ["NDPS","General_Criminal"])

        dist_avg = int(info["base"] * np.random.uniform(0.8, 1.4))

        # Targets
        base_rem = max(30, info["base"] - case_age)
        delay_f  = 1 + adj_rate*0.5 + stagnation*0.3 + undertrial*0.2 + (stage_idx<3)*0.4
        disposal = max(10, int(base_rem * delay_f + np.random.normal(0, 60)))

        risk = 0.0
        risk += min(40, (case_age / info["statutory"]) * 40)
        risk += adj_rate * 20
        risk += stagnation * 15
        risk += undertrial * 15
        risk += (7 - stage_idx) / 7 * 10
        risk = float(np.clip(risk + np.random.normal(0, 5), 0, 100))

        label = ("Critical" if risk >= 75 else
                 "High"     if risk >= 50 else
                 "Medium"   if risk >= 25 else "Low")

        rows.append({
            "case_number": f"CASE/{i+1:04d}/{random.randint(2015,2024)}",
            "court_id": court_id, "state": state, "district": district,
            "judge_id": f"JDG-{random.randint(100,999)}",
            "case_type": case_type, "case_category": category,
            "bench_type": random.choice(bench_types),
            "current_stage": stage, "stage_index": stage_idx,
            "case_age_days": case_age, "stage_age_days": stage_age,
            "hearings_held": hearings, "adjournments": adj,
            "adjournment_rate": round(adj_rate, 3),
            "stagnation_flag": stagnation, "is_undertrial": undertrial,
            "district_avg_disposal_days": dist_avg,
            "statutory_deadline_days": info["statutory"],
            "days_beyond_statutory": max(0, case_age - info["statutory"]),
            "disposal_days": disposal,
            "delay_risk_score": round(risk, 2),
            "delay_risk_label": label,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────
def preprocess(df):
    from sklearn.preprocessing import LabelEncoder
    df = df.copy().drop_duplicates(subset=["case_number"]).reset_index(drop=True)
    for col in ["case_type","case_category","bench_type","current_stage","state"]:
        le = LabelEncoder()
        df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
    df["case_age_normalized"]    = df["case_age_days"] / df["statutory_deadline_days"].replace(0,1)
    df["hearing_density"]        = df["hearings_held"] / (df["case_age_days"]/30.0).replace(0,1)
    df["stage_completion_ratio"] = df["stage_index"] / 7.0
    return df


# ─────────────────────────────────────────────────────────────────────
# XGBOOST TRAINING
# ─────────────────────────────────────────────────────────────────────
def train_models(df):
    from xgboost import XGBRegressor, XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import (mean_squared_error, r2_score,
                                 f1_score, roc_auc_score)

    RISK_ORDER = ["Low","Medium","High","Critical"]
    le_risk    = LabelEncoder(); le_risk.classes_ = np.array(RISK_ORDER)

    df_p = preprocess(df)
    X    = df_p[FEATURE_COLS]
    yr   = df_p["disposal_days"].values
    yc   = le_risk.transform(df_p["delay_risk_label"].values)

    X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = train_test_split(
        X, yr, yc, test_size=0.2, random_state=42, stratify=yc)

    # Regressor
    reg = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8, random_state=42,
                       tree_method="hist", n_jobs=-1)
    reg.fit(X_tr, yr_tr, eval_set=[(X_te, yr_te)], verbose=False)
    yr_pred = reg.predict(X_te)
    rmse = np.sqrt(mean_squared_error(yr_te, yr_pred))
    r2   = r2_score(yr_te, yr_pred)

    # Classifier
    clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, random_state=42,
                        tree_method="hist", n_jobs=-1, num_class=4,
                        objective="multi:softprob", eval_metric="mlogloss")
    clf.fit(X_tr, yc_tr, eval_set=[(X_te, yc_te)], verbose=False)
    yc_pred  = clf.predict(X_te)
    yc_proba = clf.predict_proba(X_te)
    f1  = f1_score(yc_te, yc_pred, average="weighted")
    auc = roc_auc_score(yc_te, yc_proba, multi_class="ovr", average="weighted")

    metrics = {"rmse": round(rmse,1), "r2": round(r2,3),
               "f1": round(f1,3), "auc": round(auc,3)}
    return reg, clf, le_risk, metrics


# ─────────────────────────────────────────────────────────────────────
# URGENCY ENGINE
# ─────────────────────────────────────────────────────────────────────
def score_urgency(row):
    cat = row["case_category"]; age = int(row["case_age_days"])
    stage = row["current_stage"]; stage_age = int(row["stage_age_days"])
    adj_rate = float(row["adjournment_rate"]); hearings = int(row["hearings_held"])
    statutory = int(row["statutory_deadline_days"])

    score = 0.0; flags = []
    ratio = age / max(statutory, 1)

    if ratio >= 1.0:   score += 35; flags.append("STATUTORY DEADLINE BREACHED")
    elif ratio >= .75: score += 25; flags.append("Approaching statutory deadline")
    elif ratio >= .5:  score += 15; flags.append("Past halfway of statutory limit")

    if cat == "POCSO":         score += 20; flags.append("POCSO: Mandatory speedy trial")
    elif cat == "SC_ST":       score += 18; flags.append("SC/ST PoA: Priority disposal")
    elif cat == "Senior_Citizen": score += 15; flags.append("Senior Citizen: Expedited disposal")

    if row.get("is_undertrial",0) == 1 and age > 365:
        score += 20; flags.append("BNSS 479: Undertrial threshold crossed")

    stag_lim = STAGE_STAGNATION.get(stage, 90)
    if stage_age > stag_lim*2:   score += 15; flags.append(f"Critical stagnation in {stage}")
    elif stage_age > stag_lim*1.5: score += 10; flags.append(f"Stagnation in {stage}")

    if adj_rate >= 0.6: score += 10; flags.append(f"High adjournment rate {adj_rate:.0%}")
    if age > 1825:      score += 10; flags.append(f"Case aged {age//365}+ years")

    if cat == "Commercial" and age > 180:
        score += 10; flags.append("Commercial Courts Act: Disposal overdue")

    return float(np.clip(score, 0, 100)), "; ".join(flags) if flags else "No urgent flags"


# ─────────────────────────────────────────────────────────────────────
# HYBRID PRIORITY ENGINE  (35/25/20/20 weights)
# ─────────────────────────────────────────────────────────────────────
AGE_BINS = [0,180,365,730,1095,1825,3650]
AGE_VALS = [0, 15, 35, 55,  70,  85, 100]

def hybrid_priority(row):
    age_score  = float(np.clip(np.interp(row["case_age_days"], AGE_BINS, AGE_VALS), 0, 100))
    threshold  = CATEGORIES.get(row["case_category"],{"base":2190})["statutory"]
    cat_ratio  = row["case_age_days"] / max(threshold, 1)
    cat_score  = float(np.clip(min(100, cat_ratio*70 + (15 if cat_ratio>=1.5 else 0) +
                                   (30 if cat_ratio>=2.0 else 0)), 0, 100))
    ps = (0.35 * row["legal_urgency_score"]
        + 0.25 * age_score
        + 0.20 * cat_score
        + 0.20 * row["ml_delay_risk_score"])
    ps = float(np.clip(ps, 0, 100))
    label = ("Critical" if ps >= 75 else "High" if ps >= 50 else
             "Medium"   if ps >= 25 else "Low")
    emoji = {"Critical":"🔴","High":"🟠","Medium":"🟡","Low":"🟢"}[label]
    return round(ps, 2), label, emoji


# ─────────────────────────────────────────────────────────────────────
# FULL PIPELINE (cached so it runs only once per session)
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_pipeline(n_cases=3000):
    # 1. Generate data
    df = generate_cases(n=n_cases)

    # 2. Train XGBoost
    reg, clf, le_risk, metrics = train_models(df)

    # 3. Preprocess for inference
    df_p = preprocess(df)
    X    = df_p[FEATURE_COLS]

    # 4. Predict
    df["predicted_disposal_days"] = reg.predict(X).astype(int)
    risk_enc  = clf.predict(X)
    df["ml_delay_risk_score"]  = (risk_enc / 3.0 * 100).round(1)
    df["ml_delay_risk_label"]  = le_risk.inverse_transform(risk_enc)

    # 5. Urgency engine
    urgency = df.apply(score_urgency, axis=1, result_type="expand")
    urgency.columns = ["legal_urgency_score","urgency_flags"]
    df = pd.concat([df, urgency], axis=1)

    # 6. Hybrid priority
    priority = df.apply(lambda r: hybrid_priority(r), axis=1, result_type="expand")
    priority.columns = ["priority_score","priority_label","priority_emoji"]
    df = pd.concat([df, priority], axis=1)

    # 7. Court rank
    df["court_rank"] = df.groupby("court_id")["priority_score"].rank(
        ascending=False, method="first").astype(int)

    # 8. Feature importance (XGBoost built-in)
    FEAT_LABELS = {
        "case_type_enc":"Case Type","case_category_enc":"Case Category",
        "bench_type_enc":"Bench Type","current_stage_enc":"Current Stage",
        "state_enc":"State","stage_index":"Stage Index",
        "case_age_days":"Case Age (days)","stage_age_days":"Stage Age (days)",
        "hearings_held":"Hearings Held","adjournments":"Adjournments",
        "adjournment_rate":"Adjournment Rate","stagnation_flag":"Stagnation Flag",
        "is_undertrial":"Is Undertrial (BNSS 479)",
        "district_avg_disposal_days":"District Avg Disposal",
        "statutory_deadline_days":"Statutory Deadline",
        "days_beyond_statutory":"Days Beyond Statutory",
        "case_age_normalized":"Case Age (Normalized)",
        "hearing_density":"Hearing Density",
        "stage_completion_ratio":"Stage Completion Ratio",
    }
    imp = reg.get_booster().get_score(importance_type="gain")
    feat_imp = pd.DataFrame([
        {"Feature": FEAT_LABELS.get(k,k), "Gain": round(v,2)}
        for k,v in imp.items()
    ]).sort_values("Gain", ascending=True).tail(12)

    return df, metrics, feat_imp


# ─────────────────────────────────────────────────────────────────────
# ALERTS GENERATOR
# ─────────────────────────────────────────────────────────────────────
def generate_alerts(df):
    alerts = []
    for _, r in df.iterrows():
        ps   = r.get("priority_score", 0)
        age  = r.get("case_age_days", 0)
        stage_age = r.get("stage_age_days", 0)
        flags = str(r.get("urgency_flags",""))

        if ps >= 75:
            alerts.append({"Case": r["case_number"],"Court": r["court_id"],
                "Type":"CRITICAL PRIORITY","Message": f"Score {ps:.1f}/100 [{r['case_category']}]","Severity":"Critical"})
        if r.get("is_undertrial",0)==1 and age>365:
            alerts.append({"Case": r["case_number"],"Court": r["court_id"],
                "Type":"BNSS 479 UNDERTRIAL","Message": f"Pending {age} days — bail eligibility review required","Severity":"Critical"})
        if r.get("stagnation_flag",0)==1 and stage_age>180:
            alerts.append({"Case": r["case_number"],"Court": r["court_id"],
                "Type":"STAGE STAGNATION","Message": f"Stuck in '{r['current_stage']}' for {stage_age} days","Severity":"High"})
        if "BREACHED" in flags:
            alerts.append({"Case": r["case_number"],"Court": r["court_id"],
                "Type":"STATUTORY BREACH","Message": f"[{r['case_category']}] statutory deadline exceeded","Severity":"Critical"})
    return pd.DataFrame(alerts)


# ─────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────
def main():
    # ── HEADER
    col_h1, col_h2 = st.columns([5,1])
    with col_h1:
        st.markdown("# ⚖️ Backlog Triage Case")
        st.caption("Smart India Hackathon 2026 — **git win** | Hybrid Rule-Based + ML Engine for Explainable Case Prioritization")
    with col_h2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Emblem_of_India.svg/80px-Emblem_of_India.svg.png", width=70)
    st.divider()

    # ── SIDEBAR
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        n_cases = st.slider("Dataset size", 500, 5000, 3000, step=500,
                            help="Number of synthetic NJDG case records to generate and score")
        st.divider()
        st.markdown("### 🔍 Filters")
        sel_state    = st.selectbox("State", ["All"] + sorted(STATES))
        sel_category = st.multiselect("Case Category", list(CATEGORIES.keys()))
        sel_priority = st.multiselect("Priority Label", ["Critical","High","Medium","Low"],
                                      default=["Critical","High"])
        st.divider()
        st.markdown("### ℹ️ About")
        st.markdown("""
**Backlog Triage Case** uses:
- 🌳 **XGBoost** for delay prediction
- ⚖️ **Legal rules** (POCSO, BNSS 479, SC/ST)
- 🔀 **Hybrid scoring** (35/25/20/20 weights)
- 📊 **SHAP-ready** explainability
        """)

    # ── LOAD DATA
    with st.spinner("🔄 Generating cases & training XGBoost models..."):
        df, metrics, feat_imp = run_pipeline(n_cases=n_cases)

    # Apply filters
    fdf = df.copy()
    if sel_state != "All":       fdf = fdf[fdf["state"] == sel_state]
    if sel_category:             fdf = fdf[fdf["case_category"].isin(sel_category)]
    if sel_priority:             fdf = fdf[fdf["priority_label"].isin(sel_priority)]

    # ── TABS
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "🌳 XGBoost Metrics", "📋 Priority Queue", "🚨 Alerts", "🔍 Case Explorer"
    ])

    # ══════════════════════════════════════════
    # TAB 1: OVERVIEW
    # ══════════════════════════════════════════
    with tab1:
        st.subheader("📊 Dashboard Overview")
        c1,c2,c3,c4,c5 = st.columns(5)
        total    = len(fdf)
        critical = (fdf["priority_label"]=="Critical").sum()
        high     = (fdf["priority_label"]=="High").sum()
        undertrial = fdf["is_undertrial"].sum()
        avg_disp = fdf["predicted_disposal_days"].mean()

        with c1: st.metric("Total Cases",    f"{total:,}")
        with c2: st.metric("🔴 Critical",    f"{critical:,}", f"{critical/max(total,1)*100:.1f}%")
        with c3: st.metric("🟠 High",        f"{high:,}",     f"{high/max(total,1)*100:.1f}%")
        with c4: st.metric("⛓️ Undertrial",  f"{int(undertrial):,}")
        with c5: st.metric("Avg Disposal",   f"{avg_disp:.0f} days")

        st.divider()
        r1c1, r1c2 = st.columns(2)

        with r1c1:
            label_order = ["Critical","High","Medium","Low"]
            lc = fdf["priority_label"].value_counts().reindex(label_order, fill_value=0).reset_index()
            lc.columns = ["Priority","Count"]
            fig = px.bar(lc, x="Priority", y="Count", color="Priority",
                         color_discrete_map=PRIORITY_COLORS,
                         title="Cases by Priority Label", template="plotly_white")
            fig.update_layout(showlegend=False, height=320)
            st.plotly_chart(fig, use_container_width=True)

        with r1c2:
            cat_c = (fdf.groupby(["case_category","priority_label"])
                       .size().reset_index(name="count"))
            fig2 = px.bar(cat_c, x="case_category", y="count", color="priority_label",
                          color_discrete_map=PRIORITY_COLORS,
                          title="Priority by Case Category", template="plotly_white",
                          barmode="stack")
            fig2.update_layout(height=320, xaxis_tickangle=-30, legend_title="Priority")
            st.plotly_chart(fig2, use_container_width=True)

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            fig3 = px.histogram(fdf, x="priority_score", nbins=40,
                                color_discrete_sequence=["#6c5ce7"],
                                title="Priority Score Distribution", template="plotly_white")
            fig3.update_layout(height=300)
            st.plotly_chart(fig3, use_container_width=True)

        with r2c2:
            state_c = fdf[fdf["priority_label"]=="Critical"].groupby("state").size().reset_index(name="critical")
            fig4 = px.bar(state_c.sort_values("critical", ascending=True),
                          x="critical", y="state", orientation="h",
                          color="critical", color_continuous_scale="Reds",
                          title="Critical Cases by State", template="plotly_white")
            fig4.update_layout(height=300, coloraxis_showscale=False)
            st.plotly_chart(fig4, use_container_width=True)

    # ══════════════════════════════════════════
    # TAB 2: XGBOOST METRICS
    # ══════════════════════════════════════════
    with tab2:
        st.subheader("🌳 XGBoost Model Performance")
        m1,m2,m3,m4 = st.columns(4)
        with m1: st.metric("Regressor RMSE", f"{metrics['rmse']} days",
                            "↓ Target <60 days" if metrics['rmse']<60 else "Above target")
        with m2: st.metric("Regressor R²",   f"{metrics['r2']}",
                            "↑ Target >0.75" if metrics['r2']>0.75 else "Below target")
        with m3: st.metric("Classifier F1",  f"{metrics['f1']}",
                            "↑ Target >0.75" if metrics['f1']>0.75 else "Below target")
        with m4: st.metric("Classifier AUC", f"{metrics['auc']}",
                            "↑ Target >0.85" if metrics['auc']>0.85 else "Below target")

        st.divider()
        col_fi, col_info = st.columns([3,2])
        with col_fi:
            fig_fi = px.bar(feat_imp, x="Gain", y="Feature", orientation="h",
                            color="Gain", color_continuous_scale="Viridis",
                            title="XGBoost Feature Importance (Gain)",
                            template="plotly_white")
            fig_fi.update_layout(height=450, coloraxis_showscale=False)
            st.plotly_chart(fig_fi, use_container_width=True)

        with col_info:
            st.markdown("### Priority Score Formula")
            st.latex(r"""
P = 0.35 \times U + 0.25 \times A + 0.20 \times C + 0.20 \times M
            """)
            st.markdown("""
| Term | Meaning |
|------|---------|
| **U** | Statutory Urgency Score |
| **A** | Case Age Score |
| **C** | Category Aging Score |
| **M** | ML Delay Risk (XGBoost) |
""")
            st.markdown("### Two XGBoost Models")
            st.markdown("""
🔢 **XGBRegressor**
- Predicts disposal timeline in days
- Uses 19 case features

🎯 **XGBClassifier**
- Predicts delay risk label
- Classes: Low / Medium / High / Critical
- 4-class softmax output
""")

    # ══════════════════════════════════════════
    # TAB 3: PRIORITY QUEUE
    # ══════════════════════════════════════════
    with tab3:
        st.subheader("📋 Court Priority Queue")
        display_cols = {
            "case_number":"Case No.", "court_id":"Court", "state":"State",
            "case_category":"Category", "case_type":"Type",
            "current_stage":"Stage", "case_age_days":"Age (days)",
            "priority_emoji":"⬤", "priority_score":"Priority Score",
            "predicted_disposal_days":"Disposal (days)",
            "legal_urgency_score":"Urgency", "ml_delay_risk_score":"ML Risk",
        }
        view = (fdf[[c for c in display_cols if c in fdf.columns]]
                .rename(columns=display_cols)
                .sort_values("Priority Score", ascending=False)
                .head(200))
        st.dataframe(
            view, use_container_width=True, height=500,
            column_config={
                "Priority Score": st.column_config.ProgressColumn(
                    "Priority Score", min_value=0, max_value=100, format="%.1f"),
                "Urgency": st.column_config.ProgressColumn(
                    "Urgency", min_value=0, max_value=100, format="%.1f"),
                "ML Risk": st.column_config.ProgressColumn(
                    "ML Risk", min_value=0, max_value=100, format="%.1f"),
            }
        )

        st.divider()
        st.subheader("📊 Court-Level Summary")
        court_sum = (fdf.groupby(["court_id","priority_label"]).size()
                       .unstack(fill_value=0).reset_index())
        for lbl in ["Critical","High","Medium","Low"]:
            if lbl not in court_sum.columns: court_sum[lbl] = 0
        court_sum["Total"] = court_sum[["Critical","High","Medium","Low"]].sum(axis=1)
        court_sum["Avg Score"] = (fdf.groupby("court_id")["priority_score"].mean()
                                     .round(1).values)
        st.dataframe(court_sum.sort_values("Critical", ascending=False),
                     use_container_width=True, height=350)

    # ══════════════════════════════════════════
    # TAB 4: ALERTS
    # ══════════════════════════════════════════
    with tab4:
        st.subheader("🚨 Active Alerts")
        alerts = generate_alerts(fdf)
        if alerts.empty:
            st.success("No alerts — all cases within thresholds.")
        else:
            crit_a = alerts[alerts["Severity"]=="Critical"]
            high_a = alerts[alerts["Severity"]=="High"]
            a1, a2 = st.columns(2)
            with a1:
                st.markdown(f"**🔴 Critical Alerts: {len(crit_a):,}**")
                for _, r in crit_a.head(15).iterrows():
                    st.error(f"[{r['Court']}] **{r['Type']}** — {r['Message']} ({r['Case']})")
            with a2:
                st.markdown(f"**🟠 High Alerts: {len(high_a):,}**")
                for _, r in high_a.head(15).iterrows():
                    st.warning(f"[{r['Court']}] **{r['Type']}** — {r['Message']} ({r['Case']})")

            st.divider()
            st.markdown("#### All Alerts")
            st.dataframe(alerts, use_container_width=True, height=350)

    # ══════════════════════════════════════════
    # TAB 5: CASE EXPLORER
    # ══════════════════════════════════════════
    with tab5:
        st.subheader("🔍 Single Case Explorer")
        sel = st.selectbox("Select Case", fdf["case_number"].head(200).tolist())
        row = fdf[fdf["case_number"]==sel].iloc[0]

        e1,e2,e3 = st.columns(3)
        with e1:
            st.metric("Priority Score", f"{row['priority_score']:.1f}/100")
            st.metric("Priority Label", f"{row['priority_emoji']} {row['priority_label']}")
        with e2:
            st.metric("Predicted Disposal", f"{row['predicted_disposal_days']} days")
            st.metric("ML Delay Risk", f"{row['ml_delay_risk_score']:.1f}/100")
        with e3:
            st.metric("Case Age", f"{row['case_age_days']} days")
            st.metric("Urgency Score", f"{row['legal_urgency_score']:.1f}/100")

        st.divider()
        det1, det2 = st.columns(2)
        with det1:
            st.markdown("**📌 Case Details**")
            st.table(pd.DataFrame({
                "Field":["Court","State","Category","Type","Stage",
                          "Stage Age","Hearings","Adjournments","Stagnation","Undertrial"],
                "Value":[row["court_id"],row["state"],row["case_category"],row["case_type"],
                          row["current_stage"],f"{row['stage_age_days']} days",
                          row["hearings_held"],row["adjournments"],
                          "Yes" if row["stagnation_flag"] else "No",
                          "Yes" if row["is_undertrial"]  else "No"],
            }))

        with det2:
            st.markdown("**📊 Score Breakdown**")
            age_score = float(np.clip(np.interp(row["case_age_days"], AGE_BINS, AGE_VALS), 0, 100))
            breakdown = {
                "Statutory Urgency (35%)": row["legal_urgency_score"],
                "Case Age Score (25%)":    age_score,
                "ML Delay Risk (20%)":     row["ml_delay_risk_score"],
            }
            fig_b = go.Figure(go.Bar(
                x=list(breakdown.keys()), y=list(breakdown.values()),
                marker_color=["#dc3545","#fd7e14","#6c5ce7"],
                text=[f"{v:.1f}" for v in breakdown.values()], textposition="auto",
            ))
            fig_b.update_layout(yaxis=dict(range=[0,100]),
                                title="Component Scores", template="plotly_white", height=300)
            st.plotly_chart(fig_b, use_container_width=True)

        flags = str(row.get("urgency_flags",""))
        if flags and flags != "nan" and flags != "No urgent flags":
            st.markdown("**⚖️ Legal Flags**")
            for f in flags.split(";"):
                if f.strip(): st.info(f.strip())

    # ── FOOTER
    st.divider()
    st.markdown(
        "<center><small>⚖️ <b>Backlog Triage Case</b> | Smart India Hackathon 2026 | "
        "<b>git win</b> | Bias-Resistant • Traceable • Explainable</small></center>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
