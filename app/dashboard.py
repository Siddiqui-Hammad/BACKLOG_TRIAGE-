"""
Backlog Triage Case — Smart India Hackathon 2026
app/dashboard.py — Horizontal stepper navigation at top, no scroll
"""

import os, sys, random
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
sys.path.insert(0, ROOT)

st.set_page_config(
    page_title="Backlog Triage Case",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Strip Streamlit chrome */
.main .block-container { padding:0 1.2rem 0 1.2rem !important; max-width:100% !important; }
header[data-testid="stHeader"] { display:none !important; }
footer { display:none !important; }
#MainMenu { display:none !important; }

/* ── App shell */
body { background:#f0f2f6; }

/* ── Top header band */
.top-band {
    background: linear-gradient(90deg,#1a1a2e 0%,#6c5ce7 100%);
    padding: 8px 24px;
    display: flex; align-items: center; justify-content: space-between;
    border-radius: 0 0 14px 14px;
    margin-bottom: 0;
    box-shadow: 0 2px 12px rgba(108,92,231,.35);
}
.top-band .brand { color:#fff; font-size:1.05rem; font-weight:700; letter-spacing:.4px; }
.top-band .meta  { color:rgba(255,255,255,.75); font-size:0.75rem; }

/* ── Stepper bar */
.stepper-wrap {
    background:white;
    border-radius:14px;
    padding: 10px 10px 6px 10px;
    margin: 8px 0 10px 0;
    box-shadow: 0 1px 8px rgba(0,0,0,.09);
    display:flex; align-items:center; justify-content:space-between;
    position: relative;
}
/* Connector line behind steps */
.stepper-wrap::before {
    content:'';
    position:absolute; top:50%; left:5%;
    width:90%; height:3px;
    background:#e0e0e0; z-index:0;
    transform: translateY(-4px);
}
.step-item {
    display:flex; flex-direction:column; align-items:center;
    position:relative; z-index:1; cursor:pointer; flex:1;
}
.step-circle {
    width:32px; height:32px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:0.82rem; font-weight:700; margin-bottom:4px;
    transition: all 0.25s;
}
.step-circle.done    { background:#6c5ce7; color:white; }
.step-circle.active  { background:#6c5ce7; color:white;
                        box-shadow:0 0 0 4px rgba(108,92,231,.25);
                        transform: scale(1.15); }
.step-circle.pending { background:#e8e8e8; color:#999; }
.step-label { font-size:0.68rem; color:#555; font-weight:600;
              text-align:center; white-space:nowrap; }
.step-label.active-lbl { color:#6c5ce7; font-weight:700; }

/* ── Metric cards */
div[data-testid="metric-container"] {
    background:white; border-radius:12px;
    padding:10px 14px;
    box-shadow:0 1px 6px rgba(0,0,0,.09);
    border-left:4px solid #6c5ce7;
}

/* ── Nav buttons */
div[data-testid="column"] .stButton button {
    border-radius:10px; font-weight:600; font-size:0.88rem;
    padding:6px 16px;
}
</style>
""", unsafe_allow_html=True)

PRIORITY_COLORS = {"Critical":"#dc3545","High":"#fd7e14","Medium":"#ffc107","Low":"#28a745"}
CH = 310

PAGES = [
    ("1", "📊", "Overview"),
    ("2", "🏛️", "Courts"),
    ("3", "📋", "Queue"),
    ("4", "🚨", "Alerts"),
    ("5", "🔍", "Explorer"),
    ("6", "🌳", "ML Metrics"),
]

# ─────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = 0

def _go(i):  st.session_state.page = i
def _next():
    if st.session_state.page < len(PAGES)-1: st.session_state.page += 1
def _prev():
    if st.session_state.page > 0:            st.session_state.page -= 1

p = st.session_state.page

# ─────────────────────────────────────────────────────────────────────
# TOP HEADER BAND
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-band">
  <div class="brand">⚖️ &nbsp;Backlog Triage Case</div>
  <div class="meta">Smart India Hackathon 2026 &nbsp;|&nbsp; Team: git win &nbsp;|&nbsp; Hybrid Rule-Based + XGBoost Engine</div>
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Emblem_of_India.svg/40px-Emblem_of_India.svg.png"
       style="height:32px;opacity:.9"/>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# STEPPER BAR  (clickable step buttons rendered as st.columns)
# ─────────────────────────────────────────────────────────────────────
def render_stepper(current):
    step_cols = st.columns(len(PAGES))
    for i, (num, icon, name) in enumerate(PAGES):
        with step_cols[i]:
            if i < current:
                circ_cls  = "done"
                lbl_cls   = ""
                circ_text = "✓"
            elif i == current:
                circ_cls  = "active"
                lbl_cls   = "active-lbl"
                circ_text = icon
            else:
                circ_cls  = "pending"
                lbl_cls   = ""
                circ_text = num

            # Render the visual step
            st.markdown(f"""
            <div class="step-item">
              <div class="step-circle {circ_cls}">{circ_text}</div>
              <div class="step-label {lbl_cls}">{name}</div>
            </div>
            """, unsafe_allow_html=True)

            # Invisible button underneath captures click
            if st.button("​", key=f"step_{i}", help=f"Go to {name}",
                         use_container_width=True):
                _go(i); st.rerun()

# White card wrapping the stepper
st.markdown('<div class="stepper-wrap">', unsafe_allow_html=True)
render_stepper(p)
st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# DEMO / REAL DATA
# ─────────────────────────────────────────────────────────────────────
_STATES = [
    "Maharashtra","Uttar Pradesh","Karnataka","Tamil Nadu","Rajasthan",
    "Gujarat","West Bengal","Madhya Pradesh","Bihar","Delhi",
    "Andhra Pradesh","Telangana","Kerala","Punjab","Haryana",
    "Odisha","Jharkhand","Chhattisgarh","Assam","Uttarakhand",
    "Himachal Pradesh","Goa","Jammu & Kashmir","Manipur","Tripura",
    "Meghalaya","Nagaland","Sikkim",
]
_DIST = {
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
_CATS   = ["POCSO","SC_ST","Senior_Citizen","Commercial","NDPS","Matrimonial",
           "Motor_Accident","General_Civil","General_Criminal","Property_Dispute",
           "Cheque_Bounce","Labour_Dispute","Consumer_Forum","Land_Acquisition","Constitutional_Writ"]
_STAGES = ["Filing","Admission","Notice","Written_Statement","Evidence","Arguments","Judgment","Execution"]

@st.cache_data(show_spinner=False)
def _make_demo(n=2000):
    np.random.seed(42); random.seed(42)
    lbls = np.random.choice(["Critical","High","Medium","Low"],n,p=[0.11,0.68,0.15,0.06])
    scrs = np.where(lbls=="Critical",np.random.uniform(75,100,n),
           np.where(lbls=="High",    np.random.uniform(50,75,n),
           np.where(lbls=="Medium",  np.random.uniform(25,50,n),
                                     np.random.uniform(0,25,n))))
    emo  = [{"Critical":"🔴","High":"🟠","Medium":"🟡","Low":"🟢"}[l] for l in lbls]
    sta  = np.random.choice(_STATES, n)
    rows = []
    for i in range(n):
        s=sta[i]; d=random.choice(_DIST[s])
        rows.append({
            "case_number":f"CASE/{i+1:05d}/{random.randint(2015,2024)}",
            "court_id":f"{s[:3].upper()}-{d[:3].upper()}-{random.randint(1,40):02d}",
            "state":s,"district":d,
            "case_category":random.choice(_CATS),
            "case_type":random.choice(["Civil","Criminal","Writ","Appeal"]),
            "current_stage":random.choice(_STAGES),
            "case_age_days":random.randint(90,3500),
            "stage_age_days":random.randint(10,500),
            "hearings_held":random.randint(1,35),
            "adjournments":random.randint(0,18),
            "adjournment_rate":round(random.uniform(0,0.85),2),
            "stagnation_flag":random.randint(0,1),
            "is_undertrial":random.randint(0,1),
            "priority_score":round(float(scrs[i]),2),
            "priority_label":lbls[i],
            "priority_emoji":emo[i],
            "disposal_days":random.randint(20,700),
            "legal_urgency_score":round(random.uniform(0,100),1),
            "delay_risk_score":round(random.uniform(0,100),1),
            "urgency_flags":random.choice([
                "STATUTORY DEADLINE BREACHED; High adjournment rate 65%",
                "Approaching statutory deadline; BNSS 479: Undertrial threshold crossed",
                "No urgent flags",
                "Critical stagnation in Evidence; POCSO: Mandatory speedy trial",
                "SC/ST PoA: Priority disposal",
            ]),
            "court_rank":i+1,
        })
    df = pd.DataFrame(rows)
    alerts = pd.DataFrame([
        {"case_number":r["case_number"],"court_id":r["court_id"],
         "alert_type":"CRITICAL_PRIORITY",
         "message":f"{r['case_category']} — Score {r['priority_score']:.1f}",
         "severity":"CRITICAL"}
        for r in rows if r["priority_label"]=="Critical"
    ])
    summary = df.groupby(["court_id","priority_label"]).size().unstack(fill_value=0).reset_index()
    return df, alerts, summary

@st.cache_data(show_spinner=False)
def load_data():
    sp = os.path.join(MODELS_DIR,"scored_cases.csv")
    ap = os.path.join(MODELS_DIR,"alerts.csv")
    cp = os.path.join(MODELS_DIR,"court_summary.csv")
    if not os.path.exists(sp):
        st.toast("📊 Demo mode — showing synthetic data",icon="ℹ️")
        return _make_demo()
    df      = pd.read_csv(sp)
    alerts  = pd.read_csv(ap) if os.path.exists(ap) else pd.DataFrame()
    summary = pd.read_csv(cp) if os.path.exists(cp) else pd.DataFrame()
    return df, alerts, summary

with st.spinner("Loading…"):
    df_all, alerts_all, _ = load_data()

# ─────────────────────────────────────────────────────────────────────
# FILTER ROW  (compact, single line under stepper)
# ─────────────────────────────────────────────────────────────────────
with st.expander("🔍 Filters", expanded=False):
    fc1,fc2,fc3,fc4 = st.columns(4)
    with fc1:
        sel_courts = st.multiselect("🏛️ Court ID",
                                    sorted(df_all["court_id"].unique()), default=[])
    with fc2:
        sel_cat    = st.multiselect("📂 Category",
                                    sorted(df_all["case_category"].unique()), default=[])
    with fc3:
        sel_prio   = st.multiselect("🚦 Priority",
                                    ["Critical","High","Medium","Low"],
                                    default=["Critical","High"])
    with fc4:
        sel_state  = st.selectbox("📍 State",
                                  ["All"]+sorted(df_all["state"].unique()))

fdf = df_all.copy()
if sel_courts:         fdf = fdf[fdf["court_id"].isin(sel_courts)]
if sel_state!="All":   fdf = fdf[fdf["state"]==sel_state]
if sel_cat:            fdf = fdf[fdf["case_category"].isin(sel_cat)]
if sel_prio:           fdf = fdf[fdf["priority_label"].isin(sel_prio)]

falerts = alerts_all[alerts_all["court_id"].isin(fdf["court_id"])] \
          if not alerts_all.empty and "court_id" in alerts_all else alerts_all

# ─────────────────────────────────────────────────────────────────────
# PAGE RENDERERS
# ─────────────────────────────────────────────────────────────────────
AGE_BINS=[0,180,365,730,1095,1825,3650]
AGE_VALS=[0,15,35,55,70,85,100]

def page_overview(fdf):
    tot=len(fdf); crit=(fdf["priority_label"]=="Critical").sum()
    high=(fdf["priority_label"]=="High").sum()
    cts=fdf["court_id"].nunique()
    under=int(fdf["is_undertrial"].sum()) if "is_undertrial" in fdf.columns else 0
    avgd=fdf["disposal_days"].mean() if "disposal_days" in fdf.columns else 0
    avgp=fdf["priority_score"].mean() if "priority_score" in fdf.columns else 0

    m = st.columns(7)
    for col,(lbl,val,delta) in zip(m,[
        ("🏛️ Courts",     f"{cts:,}",  None),
        ("📁 Cases",      f"{tot:,}",  None),
        ("🔴 Critical",   f"{crit:,}", f"{crit/max(tot,1)*100:.1f}%"),
        ("🟠 High",       f"{high:,}", f"{high/max(tot,1)*100:.1f}%"),
        ("⛓️ Undertrial", f"{under:,}",None),
        ("📅 Avg Disposal",f"{avgd:.0f}d",None),
        ("🎯 Avg Score",  f"{avgp:.1f}",None),
    ]):
        with col: st.metric(lbl,val,delta)

    r1,r2=st.columns(2)
    with r1:
        lo=["Critical","High","Medium","Low"]
        lc=fdf["priority_label"].value_counts().reindex(lo,fill_value=0).reset_index()
        lc.columns=["Priority","Count"]; lc["Pct"]=(lc["Count"]/max(tot,1)*100).round(1)
        fig=px.bar(lc,x="Priority",y="Count",color="Priority",
                   color_discrete_map=PRIORITY_COLORS,
                   text=lc["Pct"].astype(str)+"%",
                   title="Cases by Priority Label",template="plotly_white")
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False,height=CH,margin=dict(t=36,b=0,l=0,r=0))
        st.plotly_chart(fig,use_container_width=True)
    with r2:
        cc=fdf.groupby(["case_category","priority_label"]).size().reset_index(name="Count")
        fig2=px.bar(cc,x="case_category",y="Count",color="priority_label",
                    color_discrete_map=PRIORITY_COLORS,
                    title="Priority by Case Category",template="plotly_white",barmode="stack")
        fig2.update_layout(height=CH,xaxis_tickangle=-30,
                           legend_title="Priority",margin=dict(t=36,b=0,l=0,r=0))
        st.plotly_chart(fig2,use_container_width=True)

def page_courts(fdf):
    r1,r2=st.columns(2)
    with r1:
        cc=(fdf[fdf["priority_label"]=="Critical"]
            .groupby("court_id").size().reset_index(name="Critical Cases")
            .sort_values("Critical Cases",ascending=True).tail(20))
        fig=px.bar(cc,x="Critical Cases",y="court_id",orientation="h",
                   color="Critical Cases",color_continuous_scale="Reds",
                   title="Top 20 Courts — Critical Cases",template="plotly_white")
        fig.update_layout(height=CH+60,coloraxis_showscale=False,
                          yaxis_title="",margin=dict(t=36,b=0,l=0,r=0))
        st.plotly_chart(fig,use_container_width=True)
    with r2:
        cs=fdf.groupby("court_id").agg(
            total=("case_number","count"),avg_p=("priority_score","mean"),
            crit=("priority_label",lambda x:(x=="Critical").sum()),
            avg_d=("disposal_days","mean")).reset_index()
        fig2=px.scatter(cs.sort_values("avg_p",ascending=False).head(60),
                        x="avg_p",y="total",size="crit",color="avg_p",
                        color_continuous_scale="RdYlGn_r",hover_name="court_id",
                        title="Court Priority vs Caseload (bubble = critical count)",
                        template="plotly_white",
                        labels={"avg_p":"Avg Priority","total":"Cases","crit":"Critical"})
        fig2.update_layout(height=CH+60,coloraxis_showscale=False,
                           margin=dict(t=36,b=0,l=0,r=0))
        st.plotly_chart(fig2,use_container_width=True)
    cs2=fdf.groupby("state").agg(
        Cases=("case_number","count"),
        Critical=("priority_label",lambda x:(x=="Critical").sum()),
        Courts=("court_id","nunique"),
        AvgScore=("priority_score","mean")).reset_index()
    cs2["AvgScore"]=cs2["AvgScore"].round(1)
    st.dataframe(cs2.sort_values("Critical",ascending=False).head(12)
                 .rename(columns={"state":"State"}),
                 use_container_width=True,height=170)

def page_queue(fdf):
    lo_map={"Critical":0,"High":1,"Medium":2,"Low":3}
    view=fdf.copy(); view["_s"]=view["priority_label"].map(lo_map).fillna(9)
    view=view.sort_values(["_s","priority_score"],ascending=[True,False])
    cols_map={"priority_emoji":"⬤","priority_score":"Score","case_number":"Case No.",
              "court_id":"Court","state":"State","case_category":"Category",
              "case_type":"Type","current_stage":"Stage","case_age_days":"Age(d)",
              "disposal_days":"Disposal(d)","legal_urgency_score":"Urgency",
              "delay_risk_score":"ML Risk","priority_label":"Label"}
    avail=[c for c in cols_map if c in view.columns]
    disp=view[avail].rename(columns=cols_map).head(500)
    st.dataframe(disp,use_container_width=True,height=550,
        column_config={
            "Score":  st.column_config.ProgressColumn("Score",  min_value=0,max_value=100,format="%.1f"),
            "Urgency":st.column_config.ProgressColumn("Urgency",min_value=0,max_value=100,format="%.1f"),
            "ML Risk":st.column_config.ProgressColumn("ML Risk",min_value=0,max_value=100,format="%.1f"),
        })

def page_alerts(fdf, falerts):
    alerts=[]
    for _,r in fdf.iterrows():
        ps=r.get("priority_score",0); age=r.get("case_age_days",0)
        flags=str(r.get("urgency_flags",""))
        if ps>=75:
            alerts.append({"Court":r["court_id"],"Case":r["case_number"],
                "Category":r["case_category"],"Type":"CRITICAL PRIORITY",
                "Message":f"Score {ps:.1f}/100","Severity":"Critical"})
        if r.get("is_undertrial",0)==1 and age>365:
            alerts.append({"Court":r["court_id"],"Case":r["case_number"],
                "Category":r["case_category"],"Type":"BNSS 479 UNDERTRIAL",
                "Message":f"Pending {age}d","Severity":"Critical"})
        if r.get("stagnation_flag",0)==1 and r.get("stage_age_days",0)>180:
            alerts.append({"Court":r["court_id"],"Case":r["case_number"],
                "Category":r["case_category"],"Type":"STAGE STAGNATION",
                "Message":f"Stuck in '{r['current_stage']}' {r.get('stage_age_days',0)}d","Severity":"High"})
        if "BREACHED" in flags:
            alerts.append({"Court":r["court_id"],"Case":r["case_number"],
                "Category":r["case_category"],"Type":"STATUTORY BREACH",
                "Message":f"{r['case_category']} exceeded","Severity":"Critical"})

    adf=pd.DataFrame(alerts) if alerts else pd.DataFrame()
    if adf.empty: st.success("✅ No alerts for current filter."); return
    crit_a=adf[adf["Severity"]=="Critical"]; high_a=adf[adf["Severity"]=="High"]

    m1,m2,m3=st.columns(3)
    with m1: st.metric("🔴 Critical Alerts",f"{len(crit_a):,}")
    with m2: st.metric("🟠 High Alerts",    f"{len(high_a):,}")
    with m3: st.metric("🚨 Total Alerts",   f"{len(adf):,}")

    a1,a2=st.columns(2)
    with a1:
        st.markdown("**🔴 Critical Alerts**")
        for _,r in crit_a.head(10).iterrows():
            st.error(f"**[{r['Court']}]** {r['Type']} — {r['Message']} | `{r['Case']}`")
    with a2:
        st.markdown("**🟠 High Alerts**")
        for _,r in high_a.head(10).iterrows():
            st.warning(f"**[{r['Court']}]** {r['Type']} — {r['Message']} | `{r['Case']}`")

    g1,g2=st.columns(2)
    with g1:
        ac=adf.groupby("Type").size().reset_index(name="Count")
        fig=px.bar(ac,x="Count",y="Type",orientation="h",color="Count",
                   color_continuous_scale="Reds",title="Alerts by Type",template="plotly_white")
        fig.update_layout(height=200,coloraxis_showscale=False,margin=dict(t=30,b=0,l=0,r=0))
        st.plotly_chart(fig,use_container_width=True)
    with g2:
        ca=crit_a.groupby("Court").size().reset_index(name="Critical Alerts")\
                 .sort_values("Critical Alerts",ascending=False).head(10)
        fig2=px.bar(ca,x="Court",y="Critical Alerts",color="Critical Alerts",
                    color_continuous_scale="OrRd",title="Courts — Most Critical Alerts",
                    template="plotly_white")
        fig2.update_layout(height=200,coloraxis_showscale=False,
                           xaxis_tickangle=-30,margin=dict(t=30,b=0,l=0,r=0))
        st.plotly_chart(fig2,use_container_width=True)

def page_explorer(fdf):
    cases=fdf["case_number"].head(300).tolist()
    if not cases: st.info("No cases match current filter."); return
    sel=st.selectbox("🔍 Select Case",cases)
    row=fdf[fdf["case_number"]==sel].iloc[0]
    lbl=row["priority_label"]; emj=row.get("priority_emoji","")

    e1,e2,e3,e4,e5,e6=st.columns(6)
    with e1: st.metric("Priority Score",   f"{row['priority_score']:.1f}/100")
    with e2: st.metric("Label",            f"{emj} {lbl}")
    with e3: st.metric("Disposal Pred.",   f"{row.get('disposal_days',0)}d")
    with e4: st.metric("ML Risk",          f"{row.get('delay_risk_score',0):.1f}/100")
    with e5: st.metric("Case Age",         f"{row['case_age_days']}d")
    with e6: st.metric("Urgency",          f"{row.get('legal_urgency_score',0):.1f}/100")

    d1,d2=st.columns([1,2])
    with d1:
        st.markdown("##### 📌 Case Details")
        detail={"Court":row["court_id"],"State":row["state"],
                "District":row.get("district","—"),"Category":row["case_category"],
                "Type":row.get("case_type","—"),"Stage":row.get("current_stage","—"),
                "Stage Age":f"{row.get('stage_age_days',0)}d",
                "Hearings":row.get("hearings_held","—"),
                "Adjournments":row.get("adjournments","—"),
                "Adj. Rate":f"{row.get('adjournment_rate',0):.0%}",
                "Stagnation":"⚠️ Yes" if row.get("stagnation_flag",0) else "✅ No",
                "Undertrial": "⚠️ Yes" if row.get("is_undertrial",0)  else "✅ No"}
        st.table(pd.DataFrame({"Field":list(detail.keys()),"Value":list(detail.values())}))
    with d2:
        age_s=float(np.clip(np.interp(row["case_age_days"],AGE_BINS,AGE_VALS),0,100))
        bd={"Statutory Urgency\n(35%)":row.get("legal_urgency_score",0),
            "Case Age Score\n(25%)":age_s,
            "ML Delay Risk\n(20%)":row.get("delay_risk_score",0)}
        fig=go.Figure(go.Bar(x=list(bd.keys()),y=list(bd.values()),
                             marker_color=["#dc3545","#fd7e14","#6c5ce7"],
                             text=[f"{v:.1f}" for v in bd.values()],textposition="auto"))
        fig.update_layout(yaxis=dict(range=[0,100]),title="Score Breakdown",
                          template="plotly_white",height=280,margin=dict(t=36,b=0,l=0,r=0))
        st.plotly_chart(fig,use_container_width=True)
        flags=str(row.get("urgency_flags",""))
        if flags and flags not in ("nan","No urgent flags",""):
            st.markdown("**⚖️ Legal Flags**")
            for f in flags.split(";"):
                f=f.strip()
                if f:
                    (st.error if any(k in f for k in ["BREACHED","479","POCSO"]) else st.warning)(f"{'🚨' if 'BREACHED' in f else '⚠️'} {f}")

def page_metrics(fdf):
    metrics={"rmse":47.0,"r2":0.801,"f1":0.764,"auc":0.868}
    mp=os.path.join(MODELS_DIR,"model_metrics.json")
    if os.path.exists(mp):
        import json
        with open(mp) as f: metrics=json.load(f)
    ok=lambda v,t,rev=False:"✅" if (v<t if rev else v>t) else "⚠️"
    m1,m2,m3,m4=st.columns(4)
    with m1: st.metric("📉 RMSE",    f"{metrics['rmse']} days",f"{ok(metrics['rmse'],60,True)} Target < 60d")
    with m2: st.metric("📈 R²",      f"{metrics['r2']}",        f"{ok(metrics['r2'],0.75)} Target > 0.75")
    with m3: st.metric("🎯 F1",      f"{metrics['f1']}",        f"{ok(metrics['f1'],0.75)} Target > 0.75")
    with m4: st.metric("📊 AUC-ROC", f"{metrics['auc']}",       f"{ok(metrics['auc'],0.85)} Target > 0.85")

    c1,c2=st.columns([3,2])
    with c1:
        fig=px.histogram(fdf,x="priority_score",nbins=50,
                         color_discrete_sequence=["#6c5ce7"],
                         title="Priority Score Distribution",template="plotly_white")
        fig.update_layout(height=CH,margin=dict(t=36,b=0,l=0,r=0))
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        st.markdown("### Priority Formula")
        st.latex(r"P = 0.35U + 0.25A + 0.20C + 0.20M")
        st.markdown("""
| | Weight | Component |
|--|--|--|
| **U** | 35% | Statutory Urgency |
| **A** | 25% | Case Age Score |
| **C** | 20% | Category Aging |
| **M** | 20% | ML Delay Risk |

**Models:** XGBRegressor (disposal days) + XGBClassifier (risk label)  
**Tuning:** Optuna · 40 trials · 3-fold CV
""")
        fig2=px.box(fdf,y="delay_risk_score",color="priority_label",
                    color_discrete_map=PRIORITY_COLORS,
                    title="ML Risk Score by Priority",template="plotly_white")
        fig2.update_layout(height=220,showlegend=False,margin=dict(t=30,b=0,l=0,r=0))
        st.plotly_chart(fig2,use_container_width=True)

# ─────────────────────────────────────────────────────────────────────
# RENDER ACTIVE PAGE
# ─────────────────────────────────────────────────────────────────────
if   p==0: page_overview(fdf)
elif p==1: page_courts(fdf)
elif p==2: page_queue(fdf)
elif p==3: page_alerts(fdf, falerts)
elif p==4: page_explorer(fdf)
elif p==5: page_metrics(fdf)

# ─────────────────────────────────────────────────────────────────────
# BOTTOM PREV / NEXT  (slim row)
# ─────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:10px'/>", unsafe_allow_html=True)
b1,b2,b3,b4,b5=st.columns([1,1,4,1,1])
with b1:
    if st.button("◀ Prev",use_container_width=True,disabled=(p==0)):
        _prev(); st.rerun()
with b2:
    st.markdown(f"<p style='text-align:center;margin-top:9px;color:#888;font-size:0.78rem'>{p+1}/{len(PAGES)}</p>",
                unsafe_allow_html=True)
with b3:
    dots="".join([
        f"<span style='display:inline-block;"
        f"width:{'26px' if i==p else '10px'};height:9px;"
        f"border-radius:5px;background:{'#6c5ce7' if i==p else '#ccc'};"
        f"margin:0 3px;vertical-align:middle'></span>"
        for i in range(len(PAGES))
    ])
    st.markdown(f"<div style='text-align:center;margin-top:9px'>{dots}</div>",unsafe_allow_html=True)
with b4:
    st.markdown(f"<p style='text-align:center;margin-top:9px;color:#6c5ce7;font-size:0.78rem;font-weight:700'>{PAGES[p][2]}</p>",
                unsafe_allow_html=True)
with b5:
    if st.button("Next ▶",use_container_width=True,disabled=(p==len(PAGES)-1)):
        _next(); st.rerun()
