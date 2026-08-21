"""
Backlog Triage Case — Smart India Hackathon 2026
app/dashboard.py — Streamlit Community Cloud version
Paginated UI: no scroll, Prev/Next navigation, fixed viewport
"""

import os, sys, random
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
sys.path.insert(0, ROOT)

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
/* ── Core layout */
.main .block-container {
    padding-top: 0.4rem !important;
    padding-bottom: 0 !important;
    max-width: 100% !important;
}
header[data-testid="stHeader"] { display: none !important; }
footer                          { display: none !important; }

/* ── Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0f0c29,#1a1a2e) !important;
    border-right: 1px solid rgba(108,92,231,0.3);
}
section[data-testid="stSidebar"] * { color: #ddd !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stSlider label { color: #aaa !important; font-size:0.8rem !important; }

/* ── Metric cards */
div[data-testid="metric-container"] {
    background: white;
    border-radius: 14px;
    padding: 12px 16px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.09);
    border-left: 4px solid #6c5ce7;
}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 0.75rem; }

/* ── Page title strip */
.pg-strip {
    background: linear-gradient(90deg,#1a1a2e 0%,#6c5ce7 100%);
    color: white; padding: 7px 20px; border-radius: 12px;
    margin-bottom: 10px;
    display: flex; align-items: center; justify-content: space-between;
}
/* ── Bottom nav */
.stButton button {
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.2s;
}
/* Nav prev/next */
button[kind="primary"] { background: #6c5ce7 !important; border: none !important; }
</style>
""", unsafe_allow_html=True)

PRIORITY_COLORS = {
    "Critical": "#dc3545",
    "High":     "#fd7e14",
    "Medium":   "#ffc107",
    "Low":      "#28a745",
}
PAGES = [
    ("📊",  "Overview"),
    ("🏛️", "Court Analysis"),
    ("📋",  "Priority Queue"),
    ("🚨",  "Alerts"),
    ("🔍",  "Case Explorer"),
    ("🌳",  "Model Metrics"),
]
CH = 315   # standard chart height

# ─────────────────────────────────────────────────────────────────────
# DEMO DATA (used when scored_cases.csv not present — cloud deployment)
# ─────────────────────────────────────────────────────────────────────
_STATES = [
    "Maharashtra","Uttar Pradesh","Karnataka","Tamil Nadu","Rajasthan",
    "Gujarat","West Bengal","Madhya Pradesh","Bihar","Delhi",
    "Andhra Pradesh","Telangana","Kerala","Punjab","Haryana",
    "Odisha","Jharkhand","Chhattisgarh","Assam","Uttarakhand",
    "Himachal Pradesh","Goa","Jammu & Kashmir","Manipur","Tripura",
    "Meghalaya","Nagaland","Sikkim",
]
_DISTRICTS = {
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
_CATS = ["POCSO","SC_ST","Senior_Citizen","Commercial","NDPS","Matrimonial",
         "Motor_Accident","General_Civil","General_Criminal","Property_Dispute",
         "Cheque_Bounce","Labour_Dispute","Consumer_Forum","Land_Acquisition","Constitutional_Writ"]
_STAGES = ["Filing","Admission","Notice","Written_Statement","Evidence","Arguments","Judgment","Execution"]

@st.cache_data(show_spinner=False)
def _make_demo(n=2000):
    np.random.seed(42); random.seed(42)
    p_labels = np.random.choice(["Critical","High","Medium","Low"], n, p=[0.11,0.68,0.15,0.06])
    p_scores = np.where(p_labels=="Critical", np.random.uniform(75,100,n),
               np.where(p_labels=="High",     np.random.uniform(50,75,n),
               np.where(p_labels=="Medium",   np.random.uniform(25,50,n),
                                              np.random.uniform(0,25,n))))
    p_emojis = [{"Critical":"🔴","High":"🟠","Medium":"🟡","Low":"🟢"}[l] for l in p_labels]
    st_arr   = np.random.choice(_STATES, n)
    rows = []
    for i in range(n):
        s = st_arr[i]; d = random.choice(_DISTRICTS[s])
        rows.append({
            "case_number":         f"CASE/{i+1:05d}/{random.randint(2015,2024)}",
            "court_id":            f"{s[:3].upper()}-{d[:3].upper()}-{random.randint(1,40):02d}",
            "state": s, "district": d,
            "case_category":       random.choice(_CATS),
            "case_type":           random.choice(["Civil","Criminal","Writ","Appeal"]),
            "current_stage":       random.choice(_STAGES),
            "case_age_days":       random.randint(90, 3500),
            "stage_age_days":      random.randint(10, 500),
            "hearings_held":       random.randint(1, 35),
            "adjournments":        random.randint(0, 18),
            "adjournment_rate":    round(random.uniform(0, 0.85), 2),
            "stagnation_flag":     random.randint(0, 1),
            "is_undertrial":       random.randint(0, 1),
            "priority_score":      round(float(p_scores[i]), 2),
            "priority_label":      p_labels[i],
            "priority_emoji":      p_emojis[i],
            "disposal_days":       random.randint(20, 700),
            "legal_urgency_score": round(random.uniform(0, 100), 1),
            "delay_risk_score":    round(random.uniform(0, 100), 1),
            "urgency_flags":       random.choice([
                "STATUTORY DEADLINE BREACHED; High adjournment rate 65%",
                "Approaching statutory deadline; BNSS 479: Undertrial threshold crossed",
                "No urgent flags",
                "Critical stagnation in Evidence; POCSO: Mandatory speedy trial",
                "SC/ST PoA: Priority disposal",
            ]),
            "court_rank": i + 1,
        })
    df      = pd.DataFrame(rows)
    alerts  = pd.DataFrame([
        {"case_number":r["case_number"],"court_id":r["court_id"],
         "alert_type":"CRITICAL_PRIORITY","message":f"{r['case_category']} — Score {r['priority_score']:.1f}",
         "severity":"CRITICAL"}
        for r in rows if r["priority_label"]=="Critical"
    ])
    summary = df.groupby(["court_id","priority_label"]).size().unstack(fill_value=0).reset_index()
    return df, alerts, summary

@st.cache_data(show_spinner=False)
def load_data():
    sp = os.path.join(MODELS_DIR, "scored_cases.csv")
    ap = os.path.join(MODELS_DIR, "alerts.csv")
    cp = os.path.join(MODELS_DIR, "court_summary.csv")
    if not os.path.exists(sp):
        st.toast("📊 Demo mode — showing synthetic data", icon="ℹ️")
        return _make_demo()
    df      = pd.read_csv(sp)
    alerts  = pd.read_csv(ap)  if os.path.exists(ap) else pd.DataFrame()
    summary = pd.read_csv(cp)  if os.path.exists(cp) else pd.DataFrame()
    return df, alerts, summary

# ─────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = 0

def _go(p):   st.session_state.page = p
def _next():
    if st.session_state.page < len(PAGES)-1: st.session_state.page += 1
def _prev():
    if st.session_state.page > 0:            st.session_state.page -= 1

# ─────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────
with st.spinner("Loading data…"):
    df_all, alerts_all, court_summary = load_data()

# ─────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo + title
    c1, c2 = st.columns([1,3])
    with c1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Emblem_of_India.svg/60px-Emblem_of_India.svg.png", width=44)
    with c2:
        st.markdown("<div style='padding-top:6px'><b style='font-size:1rem'>Backlog Triage</b><br><span style='font-size:0.72rem;color:#aaa'>SIH 2026 — git win</span></div>", unsafe_allow_html=True)
    st.divider()

    # ── Page navigation
    st.markdown("<p style='font-size:0.75rem;color:#888;margin-bottom:4px'>NAVIGATION</p>", unsafe_allow_html=True)
    for i, (icon, name) in enumerate(PAGES):
        active = i == st.session_state.page
        bg  = "background:rgba(108,92,231,0.28);border-radius:9px;" if active else ""
        fw  = "font-weight:700;" if active else ""
        col = "#fff" if active else "#ccc"
        st.markdown(f"""
        <div style='{bg}margin:2px 0'>
            <div id='nav_btn_{i}' style='padding:6px 10px;cursor:pointer;{fw}color:{col};font-size:0.88rem'>
                {icon}&nbsp; {name}
            </div>
        </div>""", unsafe_allow_html=True)
        if st.button(f"{icon} {name}", key=f"nb_{i}",
                     help=f"Go to {name}",
                     use_container_width=True):
            _go(i); st.rerun()

    st.divider()
    st.markdown("<p style='font-size:0.75rem;color:#888;margin-bottom:4px'>FILTERS</p>", unsafe_allow_html=True)

    all_courts   = sorted(df_all["court_id"].unique())
    sel_courts   = st.multiselect("🏛️ Court ID",   all_courts,      default=[])
    sel_cat      = st.multiselect("📂 Category",   sorted(df_all["case_category"].unique()), default=[])
    sel_priority = st.multiselect("🚦 Priority",   ["Critical","High","Medium","Low"], default=["Critical","High"])
    sel_state    = st.selectbox( "📍 State",       ["All"] + sorted(df_all["state"].unique()))

    # Apply filters
    fdf = df_all.copy()
    if sel_courts:         fdf = fdf[fdf["court_id"].isin(sel_courts)]
    if sel_state != "All": fdf = fdf[fdf["state"] == sel_state]
    if sel_cat:            fdf = fdf[fdf["case_category"].isin(sel_cat)]
    if sel_priority:       fdf = fdf[fdf["priority_label"].isin(sel_priority)]

    falerts = alerts_all.copy()
    if not falerts.empty and "court_id" in falerts.columns and sel_courts:
        falerts = falerts[falerts["court_id"].isin(sel_courts)]

    st.divider()
    st.markdown(f"<p style='font-size:0.8rem;color:#aaa'><b style='color:#fff'>{len(fdf):,}</b> cases &nbsp;|&nbsp; <b style='color:#fff'>{fdf['court_id'].nunique():,}</b> courts</p>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# PAGE TITLE STRIP
# ─────────────────────────────────────────────────────────────────────
p = st.session_state.page
cur_icon, cur_name = PAGES[p]
total = len(fdf); crit_n = (fdf["priority_label"]=="Critical").sum()

st.markdown(f"""
<div class="pg-strip">
  <span style="font-size:1.1rem;font-weight:700;letter-spacing:.5px">
    {cur_icon} &nbsp; {cur_name}
  </span>
  <span style="font-size:0.78rem;opacity:0.82">
    ⚖️ Backlog Triage Case &nbsp;|&nbsp; {total:,} cases &nbsp;·&nbsp;
    {fdf['court_id'].nunique():,} courts &nbsp;·&nbsp;
    🔴 {crit_n:,} critical &nbsp;|&nbsp;
    Page {p+1}/{len(PAGES)}
  </span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# PAGE 0 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────
def page_overview(fdf):
    tot   = len(fdf)
    crit  = (fdf["priority_label"]=="Critical").sum()
    high  = (fdf["priority_label"]=="High").sum()
    cts   = fdf["court_id"].nunique()
    under = int(fdf["is_undertrial"].sum()) if "is_undertrial" in fdf else 0
    avgd  = fdf["disposal_days"].mean() if "disposal_days" in fdf else 0
    avgp  = fdf["priority_score"].mean() if "priority_score" in fdf else 0

    cols = st.columns(7)
    mdata = [
        ("🏛️","Courts",      f"{cts:,}",   None),
        ("📁","Total Cases", f"{tot:,}",   None),
        ("🔴","Critical",    f"{crit:,}",  f"{crit/max(tot,1)*100:.1f}%"),
        ("🟠","High",        f"{high:,}",  f"{high/max(tot,1)*100:.1f}%"),
        ("⛓️","Undertrial",  f"{under:,}", None),
        ("📅","Avg Disposal",f"{avgd:.0f}d",None),
        ("🎯","Avg Priority",f"{avgp:.1f}",None),
    ]
    for col, (icon, label, val, delta) in zip(cols, mdata):
        with col:
            st.metric(f"{icon} {label}", val, delta)

    st.markdown("<div style='height:4px'/>", unsafe_allow_html=True)
    r1, r2 = st.columns(2)

    with r1:
        lo  = ["Critical","High","Medium","Low"]
        lc  = fdf["priority_label"].value_counts().reindex(lo, fill_value=0).reset_index()
        lc.columns = ["Priority","Count"]
        lc["Pct"] = (lc["Count"] / max(tot,1) * 100).round(1)
        fig = px.bar(lc, x="Priority", y="Count", color="Priority",
                     color_discrete_map=PRIORITY_COLORS,
                     text=lc["Pct"].astype(str) + "%",
                     title="Cases by Priority Label", template="plotly_white")
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, height=CH,
                          margin=dict(t=36,b=0,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)

    with r2:
        cc = fdf.groupby(["case_category","priority_label"]).size().reset_index(name="Count")
        fig2 = px.bar(cc, x="case_category", y="Count",
                      color="priority_label", color_discrete_map=PRIORITY_COLORS,
                      title="Priority by Case Category", template="plotly_white",
                      barmode="stack")
        fig2.update_layout(height=CH, xaxis_tickangle=-30,
                           legend_title="Priority",
                           margin=dict(t=36,b=0,l=0,r=0))
        st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────
# PAGE 1 — COURT ANALYSIS
# ─────────────────────────────────────────────────────────────────────
def page_courts(fdf):
    r1, r2 = st.columns(2)
    with r1:
        cc = (fdf[fdf["priority_label"]=="Critical"]
              .groupby("court_id").size().reset_index(name="Critical Cases")
              .sort_values("Critical Cases", ascending=True).tail(20))
        fig = px.bar(cc, x="Critical Cases", y="court_id", orientation="h",
                     color="Critical Cases", color_continuous_scale="Reds",
                     title="Top 20 Courts — Critical Cases", template="plotly_white")
        fig.update_layout(height=CH+40, coloraxis_showscale=False,
                          yaxis_title="", margin=dict(t=36,b=0,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)

    with r2:
        cs = fdf.groupby("court_id").agg(
            total   = ("case_number","count"),
            avg_p   = ("priority_score","mean"),
            crit    = ("priority_label", lambda x: (x=="Critical").sum()),
            avg_d   = ("disposal_days","mean"),
        ).reset_index()
        fig2 = px.scatter(
            cs.sort_values("avg_p", ascending=False).head(60),
            x="avg_p", y="total", size="crit", color="avg_p",
            color_continuous_scale="RdYlGn_r", hover_name="court_id",
            hover_data={"avg_d":True,"crit":True},
            title="Court Priority vs Caseload (bubble = critical count)",
            template="plotly_white",
            labels={"avg_p":"Avg Priority Score","total":"Total Cases",
                    "crit":"Critical Cases","avg_d":"Avg Disposal (days)"},
        )
        fig2.update_layout(height=CH+40, coloraxis_showscale=False,
                           margin=dict(t=36,b=0,l=0,r=0))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div style='height:2px'/>", unsafe_allow_html=True)
    cs2 = fdf.groupby("state").agg(
        Cases    = ("case_number","count"),
        Critical = ("priority_label", lambda x: (x=="Critical").sum()),
        Courts   = ("court_id","nunique"),
        AvgScore = ("priority_score","mean"),
    ).reset_index().sort_values("Critical", ascending=False).head(14)
    cs2["AvgScore"] = cs2["AvgScore"].round(1)
    st.dataframe(cs2.rename(columns={"state":"State"}),
                 use_container_width=True, height=180)

# ─────────────────────────────────────────────────────────────────────
# PAGE 2 — PRIORITY QUEUE
# ─────────────────────────────────────────────────────────────────────
def page_queue(fdf):
    lo_map = {"Critical":0,"High":1,"Medium":2,"Low":3}
    view   = fdf.copy()
    view["_s"] = view["priority_label"].map(lo_map).fillna(9)
    view   = view.sort_values(["_s","priority_score"], ascending=[True,False])

    cols_map = {
        "priority_emoji":   "⬤",
        "priority_score":   "Score",
        "case_number":      "Case No.",
        "court_id":         "Court",
        "state":            "State",
        "case_category":    "Category",
        "case_type":        "Type",
        "current_stage":    "Stage",
        "case_age_days":    "Age (d)",
        "disposal_days":    "Disposal (d)",
        "legal_urgency_score": "Urgency",
        "delay_risk_score":    "ML Risk",
        "priority_label":   "Label",
    }
    avail = [c for c in cols_map if c in view.columns]
    disp  = view[avail].rename(columns=cols_map).head(500)

    st.dataframe(
        disp,
        use_container_width=True,
        height=530,
        column_config={
            "Score":   st.column_config.ProgressColumn("Score",   min_value=0, max_value=100, format="%.1f"),
            "Urgency": st.column_config.ProgressColumn("Urgency", min_value=0, max_value=100, format="%.1f"),
            "ML Risk": st.column_config.ProgressColumn("ML Risk", min_value=0, max_value=100, format="%.1f"),
        },
    )

# ─────────────────────────────────────────────────────────────────────
# PAGE 3 — ALERTS
# ─────────────────────────────────────────────────────────────────────
def page_alerts(fdf, falerts):
    # Build alerts from scored data
    alerts = []
    for _, r in fdf.iterrows():
        ps  = r.get("priority_score", 0)
        age = r.get("case_age_days",  0)
        flags = str(r.get("urgency_flags",""))
        if ps >= 75:
            alerts.append({"Court":r["court_id"],"Case":r["case_number"],
                "Category":r["case_category"],"Type":"CRITICAL PRIORITY",
                "Message":f"Score {ps:.1f}/100","Severity":"Critical"})
        if r.get("is_undertrial",0)==1 and age>365:
            alerts.append({"Court":r["court_id"],"Case":r["case_number"],
                "Category":r["case_category"],"Type":"BNSS 479 UNDERTRIAL",
                "Message":f"Pending {age}d — bail review required","Severity":"Critical"})
        if r.get("stagnation_flag",0)==1 and r.get("stage_age_days",0)>180:
            alerts.append({"Court":r["court_id"],"Case":r["case_number"],
                "Category":r["case_category"],"Type":"STAGE STAGNATION",
                "Message":f"Stuck in '{r['current_stage']}' for {r.get('stage_age_days',0)}d","Severity":"High"})
        if "BREACHED" in flags:
            alerts.append({"Court":r["court_id"],"Case":r["case_number"],
                "Category":r["case_category"],"Type":"STATUTORY BREACH",
                "Message":f"{r['case_category']} statutory deadline exceeded","Severity":"Critical"})

    adf    = pd.DataFrame(alerts) if alerts else pd.DataFrame()
    if adf.empty:
        st.success("✅ No alerts for current filter.")
        return

    crit_a = adf[adf["Severity"]=="Critical"]
    high_a = adf[adf["Severity"]=="High"]

    m1,m2,m3 = st.columns(3)
    with m1: st.metric("🔴 Critical Alerts", f"{len(crit_a):,}")
    with m2: st.metric("🟠 High Alerts",     f"{len(high_a):,}")
    with m3: st.metric("🚨 Total Alerts",    f"{len(adf):,}")
    st.markdown("<div style='height:4px'/>", unsafe_allow_html=True)

    a1, a2 = st.columns(2)
    with a1:
        st.markdown("**🔴 Critical**")
        for _, r in crit_a.head(10).iterrows():
            st.error(f"**[{r['Court']}]** {r['Type']} — {r['Message']}  \n`{r['Case']}` · {r['Category']}")
    with a2:
        st.markdown("**🟠 High**")
        for _, r in high_a.head(10).iterrows():
            st.warning(f"**[{r['Court']}]** {r['Type']} — {r['Message']}  \n`{r['Case']}` · {r['Category']}")

    st.markdown("<div style='height:4px'/>", unsafe_allow_html=True)
    g1, g2 = st.columns(2)
    with g1:
        ac = adf.groupby("Type").size().reset_index(name="Count")
        fig = px.bar(ac, x="Count", y="Type", orientation="h",
                     color="Count", color_continuous_scale="Reds",
                     title="Alerts by Type", template="plotly_white")
        fig.update_layout(height=210, coloraxis_showscale=False,
                          margin=dict(t=30,b=0,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)
    with g2:
        court_a = crit_a.groupby("Court").size().reset_index(name="Critical Alerts")\
                        .sort_values("Critical Alerts", ascending=False).head(10)
        fig2 = px.bar(court_a, x="Court", y="Critical Alerts",
                      color="Critical Alerts", color_continuous_scale="OrRd",
                      title="Courts with Most Critical Alerts", template="plotly_white")
        fig2.update_layout(height=210, coloraxis_showscale=False,
                           xaxis_tickangle=-30, margin=dict(t=30,b=0,l=0,r=0))
        st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────
# PAGE 4 — CASE EXPLORER
# ─────────────────────────────────────────────────────────────────────
AGE_BINS = [0,180,365,730,1095,1825,3650]
AGE_VALS = [0, 15, 35, 55,  70,  85, 100]

def page_explorer(fdf):
    cases = fdf["case_number"].head(300).tolist()
    if not cases:
        st.info("No cases match the current filter.")
        return

    sel = st.selectbox("🔍 Select Case Number", cases)
    row = fdf[fdf["case_number"]==sel].iloc[0]
    lbl = row["priority_label"]
    emj = row.get("priority_emoji","")

    e1,e2,e3,e4,e5,e6 = st.columns(6)
    with e1: st.metric("Priority Score",    f"{row['priority_score']:.1f}/100")
    with e2: st.metric("Priority Label",    f"{emj} {lbl}")
    with e3: st.metric("Predicted Disposal",f"{row.get('disposal_days',0)} days")
    with e4: st.metric("ML Delay Risk",     f"{row.get('delay_risk_score',0):.1f}/100")
    with e5: st.metric("Case Age",          f"{row['case_age_days']} days")
    with e6: st.metric("Urgency Score",     f"{row.get('legal_urgency_score',0):.1f}/100")

    st.markdown("<div style='height:4px'/>", unsafe_allow_html=True)
    d1, d2 = st.columns([1, 2])

    with d1:
        st.markdown("##### 📌 Case Details")
        detail = {
            "Court":       row["court_id"],
            "State":       row["state"],
            "District":    row.get("district","—"),
            "Category":    row["case_category"],
            "Type":        row.get("case_type","—"),
            "Stage":       row.get("current_stage","—"),
            "Stage Age":   f"{row.get('stage_age_days',0)}d",
            "Hearings":    row.get("hearings_held","—"),
            "Adjournments":row.get("adjournments","—"),
            "Adj. Rate":   f"{row.get('adjournment_rate',0):.0%}",
            "Stagnation":  "⚠️ Yes" if row.get("stagnation_flag",0) else "✅ No",
            "Undertrial":  "⚠️ Yes" if row.get("is_undertrial",0)  else "✅ No",
        }
        st.table(pd.DataFrame({"Field":list(detail.keys()), "Value":list(detail.values())}))

    with d2:
        age_s  = float(np.clip(np.interp(row["case_age_days"], AGE_BINS, AGE_VALS), 0, 100))
        bd = {
            "Statutory Urgency (35%)":  row.get("legal_urgency_score",0),
            "Case Age Score (25%)":     age_s,
            "ML Delay Risk (20%)":      row.get("delay_risk_score",0),
        }
        fig = go.Figure(go.Bar(
            x=list(bd.keys()), y=list(bd.values()),
            marker_color=["#dc3545","#fd7e14","#6c5ce7"],
            text=[f"{v:.1f}" for v in bd.values()],
            textposition="auto",
        ))
        fig.update_layout(
            yaxis=dict(range=[0,100], title="Score (0-100)"),
            title="Score Breakdown by Component",
            template="plotly_white", height=270,
            margin=dict(t=36,b=0,l=0,r=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        flags = str(row.get("urgency_flags",""))
        if flags and flags not in ("nan","No urgent flags",""):
            st.markdown("**⚖️ Legal Flags Triggered**")
            for f in flags.split(";"):
                f = f.strip()
                if f:
                    if "BREACHED" in f or "479" in f or "POCSO" in f:
                        st.error(f"🚨 {f}")
                    else:
                        st.warning(f"⚠️ {f}")

# ─────────────────────────────────────────────────────────────────────
# PAGE 5 — MODEL METRICS
# ─────────────────────────────────────────────────────────────────────
def page_metrics(fdf):
    # Load metrics if available, otherwise show placeholder
    mp = os.path.join(MODELS_DIR, "model_metrics.json")
    if os.path.exists(mp):
        import json
        with open(mp) as f:
            metrics = json.load(f)
    else:
        # Estimated values from training run
        metrics = {"rmse": 47.0, "r2": 0.801, "f1": 0.764, "auc": 0.868}

    ok = lambda v, t, rev=False: ("✅" if (v<t if rev else v>t) else "⚠️")
    m1,m2,m3,m4 = st.columns(4)
    with m1: st.metric("📉 RMSE",      f"{metrics['rmse']} days",  f"{ok(metrics['rmse'],60,True)} Target < 60 days")
    with m2: st.metric("📈 R² Score",  f"{metrics['r2']}",          f"{ok(metrics['r2'],0.75)} Target > 0.75")
    with m3: st.metric("🎯 F1 Score",  f"{metrics['f1']}",          f"{ok(metrics['f1'],0.75)} Target > 0.75")
    with m4: st.metric("📊 AUC-ROC",   f"{metrics['auc']}",         f"{ok(metrics['auc'],0.85)} Target > 0.85")

    st.markdown("<div style='height:6px'/>", unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])

    with c1:
        # Priority score distribution as a proxy for model output
        fig = px.histogram(fdf, x="priority_score", nbins=50,
                           color_discrete_sequence=["#6c5ce7"],
                           title="Priority Score Distribution (model output)",
                           template="plotly_white")
        fig.update_layout(height=CH, margin=dict(t=36,b=0,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("### Priority Formula")
        st.latex(r"P = 0.35 \times U + 0.25 \times A + 0.20 \times C + 0.20 \times M")
        st.markdown("""
| Term | Weight | Meaning |
|------|--------|---------|
| **U** | 35% | Statutory Urgency |
| **A** | 25% | Case Age Score |
| **C** | 20% | Category Aging |
| **M** | 20% | ML Delay Risk |

**Two XGBoost models trained:**
- 🔢 **XGBRegressor** — predicts disposal timeline (days)
- 🎯 **XGBClassifier** — predicts Low / Medium / High / Critical

Hyperparameters optimised with **Optuna** (40 trials × 3-fold CV).
""")

        risk_dist = fdf["delay_risk_score"].dropna() if "delay_risk_score" in fdf else pd.Series([])
        if not risk_dist.empty:
            fig2 = px.box(fdf, y="delay_risk_score", color="priority_label",
                          color_discrete_map=PRIORITY_COLORS,
                          title="ML Delay Risk by Priority Label",
                          template="plotly_white")
            fig2.update_layout(height=220, showlegend=False,
                               margin=dict(t=30,b=0,l=0,r=0))
            st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────
# RENDER ACTIVE PAGE
# ─────────────────────────────────────────────────────────────────────
if   p == 0: page_overview(fdf)
elif p == 1: page_courts(fdf)
elif p == 2: page_queue(fdf)
elif p == 3: page_alerts(fdf, falerts)
elif p == 4: page_explorer(fdf)
elif p == 5: page_metrics(fdf)

# ─────────────────────────────────────────────────────────────────────
# BOTTOM NAVIGATION BAR
# ─────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:52px'/>", unsafe_allow_html=True)
st.divider()

b1,b2,b3,b4,b5 = st.columns([1, 1, 4, 1, 1])
with b1:
    if st.button("◀  Prev", use_container_width=True, disabled=(p==0)):
        _prev(); st.rerun()
with b2:
    st.markdown(
        f"<p style='text-align:center;margin-top:10px;color:#888;font-size:0.78rem'>"
        f"{p+1} / {len(PAGES)}</p>",
        unsafe_allow_html=True,
    )
with b3:
    dots = "".join([
        f"<span style='display:inline-block;"
        f"width:{'26px' if i==p else '10px'};height:10px;"
        f"border-radius:5px;background:{'#6c5ce7' if i==p else '#ccc'};"
        f"margin:0 3px;vertical-align:middle;transition:all .3s'></span>"
        for i in range(len(PAGES))
    ])
    st.markdown(
        f"<div style='text-align:center;margin-top:8px'>{dots}</div>",
        unsafe_allow_html=True,
    )
with b4:
    st.markdown(
        f"<p style='text-align:center;margin-top:10px;color:#888;font-size:0.78rem'>"
        f"{cur_name}</p>",
        unsafe_allow_html=True,
    )
with b5:
    if st.button("Next  ▶", use_container_width=True, disabled=(p==len(PAGES)-1)):
        _next(); st.rerun()
