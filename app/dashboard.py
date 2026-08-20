"""
dashboard.py
Streamlit dashboard for the NJDG Case Prioritization System.
Component 6 — Court-Level Priority Queue & Alerts.
"""

import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR   = os.path.join(ROOT, "data")

# ────────────────────────────────────────────
# PAGE CONFIG
# ────────────────────────────────────────────
st.set_page_config(
    page_title="Backlog Triage Case",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main { background: #f8f9fa; }
    .stMetric { background: white; border-radius: 10px; padding: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .priority-critical { color: #dc3545; font-weight: bold; }
    .priority-high     { color: #fd7e14; font-weight: bold; }
    .priority-medium   { color: #ffc107; font-weight: bold; }
    .priority-low      { color: #28a745; font-weight: bold; }
    h1 { color: #1a1a2e; }
    .stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

PRIORITY_COLORS = {
    "Critical": "#dc3545",
    "High":     "#fd7e14",
    "Medium":   "#ffc107",
    "Low":      "#28a745",
}

# ────────────────────────────────────────────
# DATA LOADING
# ────────────────────────────────────────────

def _generate_demo_data():
    """Generate demo data when running on cloud without model files."""
    import random
    np.random.seed(42); random.seed(42)
    n = 1500

    DEMO_STATES = [
        "Maharashtra","Uttar Pradesh","Karnataka","Tamil Nadu","Rajasthan",
        "Gujarat","West Bengal","Madhya Pradesh","Bihar","Delhi",
        "Andhra Pradesh","Telangana","Kerala","Punjab","Haryana",
        "Odisha","Jharkhand","Chhattisgarh","Assam","Uttarakhand",
        "Himachal Pradesh","Goa","Jammu & Kashmir","Manipur","Tripura",
        "Meghalaya","Nagaland","Sikkim",
    ]
    DEMO_DISTRICTS = {
        "Maharashtra":["Mumbai","Pune","Nagpur","Nashik","Aurangabad","Solapur","Thane","Kolhapur","Amravati","Nanded"],
        "Uttar Pradesh":["Lucknow","Allahabad","Varanasi","Kanpur","Agra","Meerut","Ghaziabad","Mathura","Bareilly","Gorakhpur"],
        "Karnataka":["Bengaluru","Mysuru","Hubballi","Mangaluru","Belagavi","Davangere","Ballari","Kalaburagi","Tumkur","Shivamogga"],
        "Tamil Nadu":["Chennai","Coimbatore","Madurai","Salem","Tiruchirappalli","Tirunelveli","Vellore","Erode","Thoothukudi","Dindigul"],
        "Rajasthan":["Jaipur","Jodhpur","Udaipur","Kota","Bikaner","Ajmer","Alwar","Bharatpur","Sikar","Pali"],
        "Gujarat":["Ahmedabad","Surat","Vadodara","Rajkot","Bhavnagar","Jamnagar","Junagadh","Gandhinagar","Anand","Mehsana"],
        "West Bengal":["Kolkata","Howrah","Durgapur","Siliguri","Asansol","Kharagpur","Haldia","Malda","Murshidabad","Nadia"],
        "Madhya Pradesh":["Bhopal","Indore","Gwalior","Jabalpur","Ujjain","Sagar","Rewa","Satna","Ratlam","Chhindwara"],
        "Bihar":["Patna","Gaya","Muzaffarpur","Bhagalpur","Darbhanga","Ara","Begusarai","Katihar","Munger","Saharsa"],
        "Delhi":["Central","South","North","East","West","Northwest","Southwest","New Delhi","Shahdara","Southeast"],
        "Andhra Pradesh":["Visakhapatnam","Vijayawada","Guntur","Nellore","Kurnool","Tirupati","Kakinada","Rajahmundry","Kadapa","Anantapur"],
        "Telangana":["Hyderabad","Warangal","Nizamabad","Karimnagar","Khammam","Mahbubnagar","Nalgonda","Adilabad","Suryapet","Medak"],
        "Kerala":["Thiruvananthapuram","Kochi","Kozhikode","Thrissur","Kollam","Palakkad","Kannur","Alappuzha","Malappuram","Kottayam"],
        "Punjab":["Ludhiana","Amritsar","Jalandhar","Patiala","Bathinda","Mohali","Hoshiarpur","Gurdaspur","Ferozepur","Moga"],
        "Haryana":["Gurugram","Faridabad","Ambala","Rohtak","Hisar","Panipat","Karnal","Sonipat","Yamunanagar","Bhiwani"],
        "Odisha":["Bhubaneswar","Cuttack","Rourkela","Berhampur","Sambalpur","Balasore","Puri","Jharsuguda","Rayagada","Koraput"],
        "Jharkhand":["Ranchi","Jamshedpur","Dhanbad","Bokaro","Deoghar","Hazaribagh","Giridih","Ramgarh","Chaibasa","Dumka"],
        "Chhattisgarh":["Raipur","Bhilai","Bilaspur","Korba","Durg","Rajnandgaon","Jagdalpur","Ambikapur","Dhamtari","Mahasamund"],
        "Assam":["Guwahati","Silchar","Dibrugarh","Jorhat","Nagaon","Tezpur","Tinsukia","Karimganj","Hailakandi","Goalpara"],
        "Uttarakhand":["Dehradun","Haridwar","Roorkee","Haldwani","Nainital","Rishikesh","Rudrapur","Kashipur","Srinagar","Pauri"],
        "Himachal Pradesh":["Shimla","Dharamshala","Solan","Mandi","Kullu","Hamirpur","Una","Chamba","Bilaspur","Nahan"],
        "Goa":["Panaji","Margao","Vasco","Mapusa","Ponda","Bicholim","Canacona","Quepem","Sanguem","Pernem"],
        "Jammu & Kashmir":["Srinagar","Jammu","Anantnag","Baramulla","Sopore","Kathua","Udhampur","Rajouri","Poonch","Leh"],
        "Manipur":["Imphal","Thoubal","Bishnupur","Churachandpur","Senapati","Ukhrul","Tamenglong","Jiribam","Kakching","Kangpokpi"],
        "Tripura":["Agartala","Dharmanagar","Udaipur","Kailasahar","Ambassa","Sabroom","Belonia","Khowai","Melaghar","Sonamura"],
        "Meghalaya":["Shillong","Tura","Jowai","Nongstoin","Baghmara","Resubelpara","Ampati","Mairang","Nongpoh","Williamnagar"],
        "Nagaland":["Kohima","Dimapur","Mokokchung","Tuensang","Wokha","Zunheboto","Mon","Phek","Kiphire","Longleng"],
        "Sikkim":["Gangtok","Namchi","Gyalshing","Mangan","Jorethang","Rangpo","Singtam","Ravangla","Yuksom","Lachen"],
    }
    CATEGORIES_DEMO = ["POCSO","SC_ST","Senior_Citizen","Commercial","NDPS","Matrimonial",
                       "Motor_Accident","General_Civil","General_Criminal","Property_Dispute",
                       "Cheque_Bounce","Labour_Dispute","Consumer_Forum","Land_Acquisition","Constitutional_Writ"]
    stages  = ["Filing","Admission","Notice","Written_Statement","Evidence","Arguments","Judgment","Execution"]

    p_labels = np.random.choice(["Critical","High","Medium","Low"], n, p=[0.11,0.68,0.15,0.06])
    p_scores = np.where(p_labels=="Critical", np.random.uniform(75,100,n),
               np.where(p_labels=="High",     np.random.uniform(50,75,n),
               np.where(p_labels=="Medium",   np.random.uniform(25,50,n),
                                              np.random.uniform(0,25,n))))
    p_emojis = [{"Critical":"🔴","High":"🟠","Medium":"🟡","Low":"🟢"}[l] for l in p_labels]
    st_arr   = np.random.choice(DEMO_STATES, n)
    rows = []
    for i in range(n):
        s  = st_arr[i]
        d  = random.choice(DEMO_DISTRICTS[s])
        court_id = f"{s[:3].upper()}-{d[:3].upper()}-{random.randint(1,40):02d}"
        rows.append({
            "case_number":         f"CASE/{i+1:05d}/2022",
            "court_id":            court_id,
            "state":               s,
            "district":            d,
            "case_category":       random.choice(CATEGORIES_DEMO),
            "case_type":           random.choice(["Civil","Criminal","Writ"]),
            "current_stage":       random.choice(stages),
            "case_age_days":       random.randint(100, 3500),
            "stage_age_days":      random.randint(10, 400),
            "hearings_held":       random.randint(1, 30),
            "adjournments":        random.randint(0, 15),
            "adjournment_rate":    round(random.uniform(0, 0.8), 2),
            "stagnation_flag":     random.randint(0, 1),
            "is_undertrial":       random.randint(0, 1),
            "priority_score":      round(p_scores[i], 2),
            "priority_label":      p_labels[i],
            "priority_emoji":      p_emojis[i],
            "disposal_days":       random.randint(30, 600),
            "legal_urgency_score": round(random.uniform(0, 100), 1),
            "delay_risk_score":    round(random.uniform(0, 100), 1),
            "urgency_flags":       "Demo mode — run main.py for real flags",
            "court_rank":          i + 1,
            "docket_label":        f"[#{i+1}] {p_emojis[i]} {p_labels[i]}",
        })
    df = pd.DataFrame(rows)
    alerts = pd.DataFrame([
        {"case_number": r["case_number"], "court_id": r["court_id"],
         "alert_type": "CRITICAL_PRIORITY",
         "message": f"Critical priority [{r['case_category']}] case",
         "severity": "CRITICAL"}
        for r in rows if r["priority_label"] == "Critical"
    ])
    summary = df.groupby(["court_id","priority_label"]).size().unstack(fill_value=0).reset_index()
    return df, alerts, summary


@st.cache_data
def load_data():
    scored_path  = os.path.join(MODELS_DIR, "scored_cases.csv")
    alerts_path  = os.path.join(MODELS_DIR, "alerts.csv")
    summary_path = os.path.join(MODELS_DIR, "court_summary.csv")

    if not os.path.exists(scored_path):
        st.info("ℹ️ Running in **Demo Mode** — showing sample data. Clone locally and run `python main.py --all` for real NJDG predictions.")
        return _generate_demo_data()

    df      = pd.read_csv(scored_path)
    alerts  = pd.read_csv(alerts_path)  if os.path.exists(alerts_path)  else pd.DataFrame()
    summary = pd.read_csv(summary_path) if os.path.exists(summary_path) else pd.DataFrame()
    return df, alerts, summary


# ────────────────────────────────────────────
# HEADER
# ────────────────────────────────────────────

def render_header():
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown("# ⚖️ Backlog Triage Case")
        st.markdown("**Smart India Hackathon 2026 — git win** | Hybrid Rule-Based + ML Engine for Explainable Case Prioritization")
    with col2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Emblem_of_India.svg/120px-Emblem_of_India.svg.png", width=80)
    st.divider()


# ────────────────────────────────────────────
# SIDEBAR FILTERS
# ────────────────────────────────────────────

def render_sidebar(df):
    st.sidebar.header("🔍 Filters")

    # ── Primary: Court
    all_courts = sorted(df["court_id"].unique().tolist())
    sel_courts = st.sidebar.multiselect("🏛️ Court ID", all_courts, default=[])

    # ── Secondary: Category & Priority
    sel_category = st.sidebar.multiselect(
        "📂 Case Category", sorted(df["case_category"].unique().tolist()), default=[]
    )
    priority_labels = st.sidebar.multiselect(
        "🚦 Priority Label",
        ["Critical", "High", "Medium", "Low"],
        default=["Critical", "High"],
    )

    min_score, max_score = st.sidebar.slider("🎯 Priority Score Range", 0, 100, (0, 100))

    st.sidebar.divider()
    st.sidebar.markdown("#### 📍 Location (optional)")
    states = ["All"] + sorted(df["state"].unique().tolist())
    sel_state = st.sidebar.selectbox("State", states)

    # Apply filters
    filtered = df.copy()
    if sel_courts:
        filtered = filtered[filtered["court_id"].isin(sel_courts)]
    if sel_state != "All":
        filtered = filtered[filtered["state"] == sel_state]
    if sel_category:
        filtered = filtered[filtered["case_category"].isin(sel_category)]
    if priority_labels:
        filtered = filtered[filtered["priority_label"].isin(priority_labels)]
    filtered = filtered[
        (filtered["priority_score"] >= min_score) &
        (filtered["priority_score"] <= max_score)
    ]

    total_courts = filtered["court_id"].nunique()
    st.sidebar.markdown(f"**{len(filtered):,} cases** across **{total_courts:,} courts**")
    return filtered



# ────────────────────────────────────────────
# KPI METRICS
# ────────────────────────────────────────────

def render_kpis(df, alerts):
    st.subheader("📊 Dashboard Overview")
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    total_courts = df["court_id"].nunique()
    critical     = (df["priority_label"] == "Critical").sum()
    high         = (df["priority_label"] == "High").sum()
    undertrial   = df["is_undertrial"].sum() if "is_undertrial" in df.columns else 0
    crit_alerts  = len(alerts[alerts["severity"] == "CRITICAL"]) if not alerts.empty else 0
    avg_disposal = df["disposal_days"].mean() if "disposal_days" in df.columns else 0

    with col1:
        st.metric("🏛️ Courts", f"{total_courts:,}")
    with col2:
        st.metric("🔴 Critical Cases", f"{critical:,}", f"{critical/max(len(df),1)*100:.1f}%")
    with col3:
        st.metric("🟠 High Priority", f"{high:,}", f"{high/max(len(df),1)*100:.1f}%")
    with col4:
        st.metric("⛓️ Undertrial", f"{int(undertrial):,}")
    with col5:
        st.metric("🚨 Critical Alerts", f"{crit_alerts:,}")
    with col6:
        st.metric("📅 Avg Disposal", f"{avg_disposal:.0f} days")



# ────────────────────────────────────────────
# CHARTS
# ────────────────────────────────────────────

def render_charts(df):
    st.subheader("📈 Analytics")
    col1, col2 = st.columns(2)

    with col1:
        # Priority label distribution
        label_counts = df["priority_label"].value_counts().reset_index()
        label_counts.columns = ["Priority", "Count"]
        label_order = ["Critical", "High", "Medium", "Low"]
        label_counts["Priority"] = pd.Categorical(label_counts["Priority"], categories=label_order, ordered=True)
        label_counts = label_counts.sort_values("Priority")

        fig = px.bar(
            label_counts, x="Priority", y="Count",
            color="Priority",
            color_discrete_map=PRIORITY_COLORS,
            title="Cases by Priority Label",
            template="plotly_white",
        )
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Case category breakdown
        cat_counts = df.groupby(["case_category", "priority_label"]).size().reset_index(name="count")
        fig2 = px.bar(
            cat_counts, x="case_category", y="count",
            color="priority_label",
            color_discrete_map=PRIORITY_COLORS,
            title="Priority Distribution by Case Category",
            template="plotly_white",
            barmode="stack",
        )
        fig2.update_layout(height=350, xaxis_tickangle=-30, legend_title="Priority")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Priority score distribution
        fig3 = px.histogram(
            df, x="priority_score", nbins=40,
            color_discrete_sequence=["#6c5ce7"],
            title="Priority Score Distribution",
            template="plotly_white",
            labels={"priority_score": "Priority Score (0–100)"},
        )
        fig3.update_layout(height=320)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # Predicted disposal timeline box plot by category
        fig4 = px.box(
            df, x="case_category", y="disposal_days",
            color="priority_label",
            color_discrete_map=PRIORITY_COLORS,
            title="Predicted Disposal Timeline by Category",
            template="plotly_white",
        )
        fig4.update_layout(height=320, xaxis_tickangle=-30)
        st.plotly_chart(fig4, use_container_width=True)


# ────────────────────────────────────────────
# COURT-WISE CHARTS  ← replaces state heatmap
# ────────────────────────────────────────────

def render_court_charts(df):
    st.subheader("🏛️ Court-wise Backlog Analysis")
    col1, col2 = st.columns(2)

    # Top 20 courts by critical case count
    with col1:
        court_crit = (
            df[df["priority_label"] == "Critical"]
            .groupby("court_id").size()
            .reset_index(name="Critical Cases")
            .sort_values("Critical Cases", ascending=True)
            .tail(20)
        )
        fig = px.bar(
            court_crit, x="Critical Cases", y="court_id",
            orientation="h",
            color="Critical Cases",
            color_continuous_scale="Reds",
            title="Top 20 Courts — Critical Cases",
            template="plotly_white",
        )
        fig.update_layout(height=430, coloraxis_showscale=False,
                          yaxis_title="Court ID", xaxis_title="Critical Cases")
        st.plotly_chart(fig, use_container_width=True)

    # Court avg priority score bubble chart
    with col2:
        court_stats = df.groupby("court_id").agg(
            total_cases     = ("case_number", "count"),
            avg_priority    = ("priority_score", "mean"),
            critical_count  = ("priority_label", lambda x: (x == "Critical").sum()),
            avg_disposal    = ("disposal_days", "mean"),
        ).reset_index()
        court_stats["avg_priority"]  = court_stats["avg_priority"].round(1)
        court_stats["avg_disposal"]  = court_stats["avg_disposal"].round(0)
        fig2 = px.scatter(
            court_stats.sort_values("avg_priority", ascending=False).head(50),
            x="avg_priority", y="total_cases",
            size="critical_count",
            color="avg_priority",
            color_continuous_scale="RdYlGn_r",
            hover_name="court_id",
            hover_data={"avg_disposal": True, "critical_count": True},
            title="Court Priority vs. Caseload (bubble = critical count)",
            template="plotly_white",
            labels={
                "avg_priority": "Avg Priority Score",
                "total_cases":  "Total Cases",
                "critical_count": "Critical Cases",
                "avg_disposal": "Avg Disposal (days)",
            }
        )
        fig2.update_layout(height=430, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)



# ────────────────────────────────────────────
# PRIORITY QUEUE TABLE
# ────────────────────────────────────────────

def render_priority_queue(df):
    st.subheader("📋 Court Priority Queue")

    display_cols = [
        "docket_label", "case_number", "court_id", "state",
        "case_category", "case_type", "current_stage",
        "case_age_days", "priority_score", "disposal_days",
        "legal_urgency_score", "delay_risk_score",
    ]
    available = [c for c in display_cols if c in df.columns]

    view = df[available].sort_values("priority_score", ascending=False).head(100)

    # Rename for display
    view = view.rename(columns={
        "docket_label":       "Queue Rank",
        "case_number":        "Case No.",
        "court_id":           "Court",
        "state":              "State",
        "case_category":      "Category",
        "case_type":          "Type",
        "current_stage":      "Stage",
        "case_age_days":      "Age (days)",
        "priority_score":     "Priority Score",
        "disposal_days":      "Disposal (days)",
        "legal_urgency_score":"Urgency Score",
        "delay_risk_score":   "ML Delay Risk",
    })

    st.dataframe(
        view,
        use_container_width=True,
        height=450,
        column_config={
            "Priority Score": st.column_config.ProgressColumn(
                "Priority Score", min_value=0, max_value=100, format="%.1f"
            ),
            "ML Delay Risk": st.column_config.ProgressColumn(
                "ML Delay Risk", min_value=0, max_value=100, format="%.1f"
            ),
            "Urgency Score": st.column_config.ProgressColumn(
                "Urgency Score", min_value=0, max_value=100, format="%.1f"
            ),
        },
    )


# ────────────────────────────────────────────
# ALERTS PANEL
# ────────────────────────────────────────────

def render_alerts(alerts):
    st.subheader("🚨 Active Alerts")

    if alerts.empty:
        st.info("No alerts generated.")
        return

    crit = alerts[alerts["severity"] == "CRITICAL"]
    high = alerts[alerts["severity"] == "HIGH"]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**🔴 Critical Alerts: {len(crit)}**")
        for _, row in crit.head(10).iterrows():
            st.error(f"[{row['court_id']}] {row['message']} — {row['case_number']}")

    with col2:
        st.markdown(f"**🟠 High Alerts: {len(high)}**")
        for _, row in high.head(10).iterrows():
            st.warning(f"[{row['court_id']}] {row['message']} — {row['case_number']}")


# ────────────────────────────────────────────
# SINGLE CASE EXPLORER
# ────────────────────────────────────────────

def render_case_explorer(df):
    st.subheader("🔍 Single Case Explorer")

    case_nums = df["case_number"].tolist()
    sel_case  = st.selectbox("Select Case Number", case_nums[:200])

    if sel_case:
        row = df[df["case_number"] == sel_case].iloc[0]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Priority Score", f"{row.get('priority_score', 0):.1f}/100")
            st.metric("Priority Label", f"{row.get('priority_emoji', '')} {row.get('priority_label', '')}")
        with col2:
            st.metric("Predicted Disposal", f"{row.get('disposal_days', 0):.0f} days")
            st.metric("ML Delay Risk", f"{row.get('delay_risk_score', 0):.1f}/100")
        with col3:
            st.metric("Case Age", f"{row.get('case_age_days', 0)} days")
            st.metric("Urgency Score", f"{row.get('legal_urgency_score', 0):.1f}/100")

        st.markdown("**📌 Case Details**")
        details = {
            "Court": row.get("court_id"), "State": row.get("state"),
            "Category": row.get("case_category"), "Type": row.get("case_type"),
            "Stage": row.get("current_stage"), "Stage Age": f"{row.get('stage_age_days', 0)} days",
            "Hearings": row.get("hearings_held"), "Adjournments": row.get("adjournments"),
            "Stagnation": "Yes" if row.get("stagnation_flag", 0) else "No",
            "Undertrial": "Yes" if row.get("is_undertrial", 0) else "No",
        }
        st.json(details)

        flags = str(row.get("urgency_flags", ""))
        if flags and flags != "nan":
            st.markdown("**⚖️ Legal Flags**")
            for flag in flags.split(";"):
                if flag.strip():
                    st.info(flag.strip())

        # Priority breakdown radar chart
        breakdown = {
            "Statutory Urgency": row.get("legal_urgency_score", 0),
            "Case Age Score":    min(100, row.get("case_age_days", 0) / 18.25),
            "ML Delay Risk":     row.get("delay_risk_score", 0),
        }
        fig = go.Figure(go.Bar(
            x=list(breakdown.keys()),
            y=list(breakdown.values()),
            marker_color=["#dc3545", "#fd7e14", "#6c5ce7"],
        ))
        fig.update_layout(
            title="Score Component Breakdown",
            yaxis=dict(range=[0, 100]),
            template="plotly_white",
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)


# ────────────────────────────────────────────
# MAIN APP
# ────────────────────────────────────────────

def main():
    render_header()
    df, alerts, summary = load_data()

    # Sidebar filters
    filtered_df = render_sidebar(df)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview", "📋 Priority Queue", "🚨 Alerts", "🔍 Case Explorer"
    ])

    with tab1:
        render_kpis(filtered_df, alerts)
        render_charts(filtered_df)
        render_court_charts(filtered_df)

    with tab2:
        render_priority_queue(filtered_df)

    with tab3:
        render_alerts(alerts)

    with tab4:
        render_case_explorer(filtered_df)

    # Footer
    st.divider()
    st.markdown(
        "<center><small>⚖️ Backlog Triage Case | Smart India Hackathon 2026 | <b>git win</b> | "
        "Bias-Resistant • Traceable • Explainable</small></center>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
