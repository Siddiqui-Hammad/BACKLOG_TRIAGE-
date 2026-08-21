"""
Backlog Triage Case — Smart India Hackathon 2026
app/dashboard.py
Layout: Top stepper → [Content (75%) | Filters (25%)] → Prev/Next
No scrolling — all content fits viewport height.
"""

import os, sys, random, math
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
sys.path.insert(0, ROOT)

# ─────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Backlog Triage Case",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
/* ── Strip chrome */
.main .block-container{padding:0 0.8rem !important;max-width:100% !important;}
header[data-testid="stHeader"]{display:none!important;}
footer{display:none!important;}
#MainMenu{display:none!important;}

/* ── App background */
body,.stApp{background:#f4f6fb;}

/* ── Top header */
.hdr{
  background:linear-gradient(90deg,#1a1a2e 0%,#6c5ce7 100%);
  padding:9px 20px; display:flex; align-items:center;
  justify-content:space-between; border-radius:0 0 16px 16px;
  box-shadow:0 3px 14px rgba(108,92,231,.4);
  margin-bottom:8px;
}
.hdr-brand{color:#fff;font-size:1.05rem;font-weight:800;letter-spacing:.4px;}
.hdr-meta{color:rgba(255,255,255,.72);font-size:0.72rem;text-align:center;}

/* ── Stepper */
.stp-wrap{
  background:white; border-radius:14px;
  padding:8px 12px 4px 12px; margin-bottom:8px;
  box-shadow:0 1px 8px rgba(0,0,0,.08);
}

/* ── Step buttons — override Streamlit */
div[data-testid="stHorizontalBlock"] .stButton button{
  border-radius:30px; font-size:0.78rem; font-weight:600;
  padding:4px 8px; width:100%;
  border:2px solid transparent; transition:all .2s;
}
/* secondary (inactive) step */
div[data-testid="stHorizontalBlock"] .stButton button[kind="secondary"]{
  background:#f0f0f8 !important; color:#888 !important;
  border-color:#e0e0ee !important;
}
/* primary (active) step */
div[data-testid="stHorizontalBlock"] .stButton button[kind="primary"]{
  background:#6c5ce7 !important; color:#fff !important;
  box-shadow:0 0 0 4px rgba(108,92,231,.2);
}

/* ── Metric cards */
div[data-testid="metric-container"]{
  background:white; border-radius:12px;
  padding:10px 14px;
  box-shadow:0 1px 6px rgba(0,0,0,.09);
  border-left:4px solid #6c5ce7;
}
div[data-testid="metric-container"] label{font-size:0.72rem!important;}

/* ── Right filter panel */
.filter-card{
  background:white; border-radius:14px;
  padding:14px 12px;
  box-shadow:0 1px 8px rgba(0,0,0,.09);
  height:100%;
}

/* ── Bottom nav */
.btm-nav{
  background:white; border-radius:12px;
  padding:6px 12px; margin-top:6px;
  box-shadow:0 1px 6px rgba(0,0,0,.08);
}

/* Progress step connector (decorative) */
.step-prog{height:3px;background:#e0e0e0;border-radius:3px;margin:4px 0 0 0;position:relative;}
.step-prog-fill{height:3px;background:#6c5ce7;border-radius:3px;
  transition:width .4s ease;position:absolute;top:0;left:0;}
</style>
""", unsafe_allow_html=True)

PRIORITY_COLORS = {"Critical":"#dc3545","High":"#fd7e14","Medium":"#ffc107","Low":"#28a745"}

PAGES = [
    ("📊","Overview"),
    ("🏛️","Courts"),
    ("📋","Queue"),
    ("🚨","Alerts"),
    ("🔍","Explorer"),
    ("🌳","ML Info"),
]

# ─────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────
for k,v in [("page",0),("queue_pg",0),("alert_pg",0)]:
    if k not in st.session_state: st.session_state[k]=v

def _go(i):
    st.session_state.page=i
    st.session_state.queue_pg=0
    st.session_state.alert_pg=0

def _next():
    if st.session_state.page<len(PAGES)-1: st.session_state.page+=1
def _prev():
    if st.session_state.page>0: st.session_state.page-=1

p = st.session_state.page

# ─────────────────────────────────────────────────────────────────────
# DEMO DATA
# ─────────────────────────────────────────────────────────────────────
_STATES=["Maharashtra","Uttar Pradesh","Karnataka","Tamil Nadu","Rajasthan",
         "Gujarat","West Bengal","Madhya Pradesh","Bihar","Delhi",
         "Andhra Pradesh","Telangana","Kerala","Punjab","Haryana",
         "Odisha","Jharkhand","Chhattisgarh","Assam","Uttarakhand",
         "Himachal Pradesh","Goa","Jammu & Kashmir","Manipur","Tripura",
         "Meghalaya","Nagaland","Sikkim"]
_DIST={
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
_CATS=["POCSO","SC_ST","Senior_Citizen","Commercial","NDPS","Matrimonial",
       "Motor_Accident","General_Civil","General_Criminal","Property_Dispute",
       "Cheque_Bounce","Labour_Dispute","Consumer_Forum","Land_Acquisition","Constitutional_Writ"]
_STAGES=["Filing","Admission","Notice","Written_Statement","Evidence","Arguments","Judgment","Execution"]

@st.cache_data(show_spinner=False)
def _make_demo(n=2000):
    np.random.seed(42); random.seed(42)
    lbls=np.random.choice(["Critical","High","Medium","Low"],n,p=[0.11,0.68,0.15,0.06])
    scrs=np.where(lbls=="Critical",np.random.uniform(75,100,n),
         np.where(lbls=="High",    np.random.uniform(50,75,n),
         np.where(lbls=="Medium",  np.random.uniform(25,50,n),
                                   np.random.uniform(0,25,n))))
    emo=[{"Critical":"🔴","High":"🟠","Medium":"🟡","Low":"🟢"}[l] for l in lbls]
    sta=np.random.choice(_STATES,n)
    rows=[]
    for i in range(n):
        s=sta[i]; d=random.choice(_DIST.get(s,["Unknown"]))
        rows.append({"case_number":f"CASE/{i+1:05d}/{random.randint(2015,2024)}",
            "court_id":f"{s[:3].upper()}-{d[:3].upper()}-{random.randint(1,40):02d}",
            "state":s,"district":d,"case_category":random.choice(_CATS),
            "case_type":random.choice(["Civil","Criminal","Writ","Appeal"]),
            "current_stage":random.choice(_STAGES),"case_age_days":random.randint(90,3500),
            "stage_age_days":random.randint(10,500),"hearings_held":random.randint(1,35),
            "adjournments":random.randint(0,18),"adjournment_rate":round(random.uniform(0,.85),2),
            "stagnation_flag":random.randint(0,1),"is_undertrial":random.randint(0,1),
            "priority_score":round(float(scrs[i]),2),"priority_label":lbls[i],
            "priority_emoji":emo[i],"disposal_days":random.randint(20,700),
            "legal_urgency_score":round(random.uniform(0,100),1),
            "delay_risk_score":round(random.uniform(0,100),1),
            "urgency_flags":random.choice([
                "STATUTORY DEADLINE BREACHED; High adjournment rate 65%",
                "Approaching statutory deadline; BNSS 479: Undertrial threshold crossed",
                "No urgent flags","Critical stagnation in Evidence; POCSO: Mandatory speedy trial",
                "SC/ST PoA: Priority disposal"]),
            "court_rank":i+1})
    df=pd.DataFrame(rows)
    alerts=pd.DataFrame([{"case_number":r["case_number"],"court_id":r["court_id"],
        "alert_type":"CRITICAL_PRIORITY","message":f"{r['case_category']} — {r['priority_score']:.1f}/100",
        "severity":"CRITICAL"} for r in rows if r["priority_label"]=="Critical"])
    return df, alerts, pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_data():
    sp=os.path.join(MODELS_DIR,"scored_cases.csv")
    ap=os.path.join(MODELS_DIR,"alerts.csv")
    if not os.path.exists(sp):
        return _make_demo()
    df=pd.read_csv(sp)
    alerts=pd.read_csv(ap) if os.path.exists(ap) else pd.DataFrame()
    return df, alerts, pd.DataFrame()

with st.spinner("Loading…"):
    df_all, alerts_all, _ = load_data()

# ─────────────────────────────────────────────────────────────────────
# TOP HEADER
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hdr">
  <div class="hdr-brand">⚖️ &nbsp;Backlog Triage Case</div>
  <div class="hdr-meta">Smart India Hackathon 2026 &nbsp;|&nbsp; Team: git win &nbsp;|&nbsp; XGBoost + Hybrid Rule Engine</div>
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Emblem_of_India.svg/36px-Emblem_of_India.svg.png"
       style="height:30px;opacity:.92"/>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# STEPPER  — primary button = active, secondary = inactive
# ─────────────────────────────────────────────────────────────────────
with st.container():
    st.markdown('<div class="stp-wrap">', unsafe_allow_html=True)
    stp_cols = st.columns(len(PAGES))
    for i,(icon,name) in enumerate(PAGES):
        with stp_cols[i]:
            done  = i < p
            label = f"✓ {name}" if done else f"{icon} {name}"
            btype = "primary" if i==p else "secondary"
            if st.button(label, key=f"stp_{i}", type=btype, use_container_width=True):
                _go(i); st.rerun()
    # Progress bar under stepper
    pct = int(p/(len(PAGES)-1)*100)
    st.markdown(f"""
    <div class="step-prog">
      <div class="step-prog-fill" style="width:{pct}%"></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# MAIN LAYOUT: content (left 75%) + filters (right 25%)
# ─────────────────────────────────────────────────────────────────────
main_col, filt_col = st.columns([3, 1], gap="small")

# ────────── RIGHT: FILTER PANEL ──────────
with filt_col:
    st.markdown('<div class="filter-card">', unsafe_allow_html=True)
    st.markdown("### 🔍 Filters")

    sel_state  = st.selectbox("📍 State",
                              ["All"]+sorted(df_all["state"].unique()), key="f_state")
    sel_courts = st.multiselect("🏛️ Court ID",
                                sorted(df_all["court_id"].unique()), key="f_court")
    sel_cat    = st.multiselect("📂 Category",
                                sorted(df_all["case_category"].unique()), key="f_cat")
    sel_prio   = st.multiselect("🚦 Priority",
                                ["Critical","High","Medium","Low"],
                                default=["Critical","High"], key="f_prio")
    if st.button("🔄 Reset Filters", use_container_width=True):
        for k in ("f_state","f_court","f_cat","f_prio"):
            if k in st.session_state: del st.session_state[k]
        st.rerun()

    # Apply
    fdf = df_all.copy()
    if sel_state!="All":   fdf=fdf[fdf["state"]==sel_state]
    if sel_courts:         fdf=fdf[fdf["court_id"].isin(sel_courts)]
    if sel_cat:            fdf=fdf[fdf["case_category"].isin(sel_cat)]
    if sel_prio:           fdf=fdf[fdf["priority_label"].isin(sel_prio)]

    st.divider()
    tot=len(fdf); crit_n=(fdf["priority_label"]=="Critical").sum()
    high_n=(fdf["priority_label"]=="High").sum()
    cts=fdf["court_id"].nunique()
    st.markdown(f"""
    <div style="font-size:.82rem;line-height:2">
    🏛️ <b style="color:#6c5ce7">{cts:,}</b> courts<br>
    📁 <b style="color:#6c5ce7">{tot:,}</b> cases<br>
    🔴 <b style="color:#dc3545">{crit_n:,}</b> critical<br>
    🟠 <b style="color:#fd7e14">{high_n:,}</b> high<br>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.caption("Current page")
    st.markdown(f"<div style='font-size:.95rem;font-weight:700;color:#6c5ce7'>{PAGES[p][0]} {PAGES[p][1]}</div>",
                unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:.72rem;color:#aaa'>Step {p+1} of {len(PAGES)}</div>",
                unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ────────── LEFT: MAIN CONTENT ──────────
CH = 295   # chart height

with main_col:

    # ── PAGE 0: OVERVIEW
    if p == 0:
        m = st.columns(4)
        for col,(lbl,val,delta) in zip(m,[
            ("🏛️ Courts",      f"{fdf['court_id'].nunique():,}", None),
            ("🔴 Critical",    f"{crit_n:,}", f"{crit_n/max(tot,1)*100:.1f}%"),
            ("⛓️ Undertrial",  f"{int(fdf['is_undertrial'].sum() if 'is_undertrial' in fdf.columns else 0):,}", None),
            ("📅 Avg Disposal", f"{fdf['disposal_days'].mean():.0f}d" if 'disposal_days' in fdf.columns else "—", None),
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
                       title="Cases by Priority",template="plotly_white")
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False,height=CH,margin=dict(t=32,b=0,l=0,r=0))
            st.plotly_chart(fig,use_container_width=True)
        with r2:
            cc=fdf.groupby(["case_category","priority_label"]).size().reset_index(name="Count")
            fig2=px.bar(cc,x="case_category",y="Count",color="priority_label",
                        color_discrete_map=PRIORITY_COLORS,barmode="stack",
                        title="Priority by Category",template="plotly_white")
            fig2.update_layout(height=CH,xaxis_tickangle=-30,
                               legend_title="Priority",margin=dict(t=32,b=0,l=0,r=0))
            st.plotly_chart(fig2,use_container_width=True)

    # ── PAGE 1: COURTS
    elif p == 1:
        r1,r2=st.columns(2)
        with r1:
            cc=(fdf[fdf["priority_label"]=="Critical"]
                .groupby("court_id").size().reset_index(name="Critical Cases")
                .sort_values("Critical Cases",ascending=True).tail(15))
            fig=px.bar(cc,x="Critical Cases",y="court_id",orientation="h",
                       color="Critical Cases",color_continuous_scale="Reds",
                       title="Top 15 Courts — Critical Cases",template="plotly_white")
            fig.update_layout(height=CH+80,coloraxis_showscale=False,
                              yaxis_title="",margin=dict(t=32,b=0,l=0,r=0))
            st.plotly_chart(fig,use_container_width=True)
        with r2:
            cs=fdf.groupby("court_id").agg(
                total=("case_number","count"),avg_p=("priority_score","mean"),
                crit=("priority_label",lambda x:(x=="Critical").sum())).reset_index()
            fig2=px.scatter(cs.sort_values("avg_p",ascending=False).head(50),
                            x="avg_p",y="total",size="crit",color="avg_p",
                            color_continuous_scale="RdYlGn_r",hover_name="court_id",
                            title="Court Priority vs Caseload",template="plotly_white",
                            labels={"avg_p":"Avg Priority","total":"Cases","crit":"Critical"})
            fig2.update_layout(height=CH+80,coloraxis_showscale=False,
                               margin=dict(t=32,b=0,l=0,r=0))
            st.plotly_chart(fig2,use_container_width=True)

        state_stats=fdf.groupby("state").agg(
            Cases=("case_number","count"),
            Critical=("priority_label",lambda x:(x=="Critical").sum()),
            Courts=("court_id","nunique"),
            AvgScore=("priority_score","mean")).reset_index()
        state_stats["AvgScore"]=state_stats["AvgScore"].round(1)
        st.dataframe(state_stats.sort_values("Critical",ascending=False)
                     .rename(columns={"state":"State"}),
                     use_container_width=True, height=150)

    # ── PAGE 2: PRIORITY QUEUE  (paginated, 15 rows per page)
    elif p == 2:
        ROWS_PER_PAGE = 15
        lo_map={"Critical":0,"High":1,"Medium":2,"Low":3}
        view=fdf.copy(); view["_s"]=view["priority_label"].map(lo_map).fillna(9)
        view=view.sort_values(["_s","priority_score"],ascending=[True,False]).reset_index(drop=True)

        total_rows=len(view); total_pages=max(1,math.ceil(total_rows/ROWS_PER_PAGE))
        qpg=st.session_state.queue_pg
        start=qpg*ROWS_PER_PAGE; end=start+ROWS_PER_PAGE
        slice_df=view.iloc[start:end]

        cols_map={"priority_emoji":"⬤","priority_score":"Score","case_number":"Case No.",
                  "court_id":"Court","state":"State","case_category":"Category",
                  "case_type":"Type","current_stage":"Stage","case_age_days":"Age(d)",
                  "disposal_days":"Disposal(d)","legal_urgency_score":"Urgency",
                  "delay_risk_score":"ML Risk","priority_label":"Label"}
        avail=[c for c in cols_map if c in slice_df.columns]
        disp=slice_df[avail].rename(columns=cols_map)

        st.markdown(f"**Showing rows {start+1}–{min(end,total_rows)} of {total_rows:,}** "
                    f"&nbsp;|&nbsp; Page {qpg+1}/{total_pages}")
        st.dataframe(disp,use_container_width=True,height=445,
            column_config={
                "Score":  st.column_config.ProgressColumn("Score",  min_value=0,max_value=100,format="%.1f"),
                "Urgency":st.column_config.ProgressColumn("Urgency",min_value=0,max_value=100,format="%.1f"),
                "ML Risk":st.column_config.ProgressColumn("ML Risk",min_value=0,max_value=100,format="%.1f"),
            })

        qc1,qc2,qc3=st.columns([1,3,1])
        with qc1:
            if st.button("◀ Prev rows",disabled=(qpg==0),use_container_width=True):
                st.session_state.queue_pg-=1; st.rerun()
        with qc2:
            dots="".join([
                f"<span style='display:inline-block;width:{'18px' if i==qpg else '8px'};height:8px;"
                f"border-radius:4px;background:{'#6c5ce7' if i==qpg else '#ddd'};margin:0 2px'></span>"
                for i in range(min(total_pages,10))])
            st.markdown(f"<div style='text-align:center;margin-top:9px'>{dots}</div>",unsafe_allow_html=True)
        with qc3:
            if st.button("Next rows ▶",disabled=(qpg>=total_pages-1),use_container_width=True):
                st.session_state.queue_pg+=1; st.rerun()

    # ── PAGE 3: ALERTS  (paginated, 8 alerts per column per page)
    elif p == 3:
        ALERTS_PER_PAGE = 8
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
                    "Category":r["case_category"],"Type":"BNSS 479",
                    "Message":f"Pending {age}d","Severity":"Critical"})
            if r.get("stagnation_flag",0)==1 and r.get("stage_age_days",0)>180:
                alerts.append({"Court":r["court_id"],"Case":r["case_number"],
                    "Category":r["case_category"],"Type":"STAGNATION",
                    "Message":f"Stuck in {r['current_stage']} {r.get('stage_age_days',0)}d","Severity":"High"})
            if "BREACHED" in flags:
                alerts.append({"Court":r["court_id"],"Case":r["case_number"],
                    "Category":r["case_category"],"Type":"STATUTORY BREACH",
                    "Message":f"{r['case_category']} exceeded","Severity":"Critical"})

        adf=pd.DataFrame(alerts) if alerts else pd.DataFrame()
        if adf.empty:
            st.success("✅ No alerts for current filter.")
        else:
            crit_a=adf[adf["Severity"]=="Critical"].reset_index(drop=True)
            high_a=adf[adf["Severity"]=="High"].reset_index(drop=True)
            apg=st.session_state.alert_pg
            total_apages=max(1,math.ceil(max(len(crit_a),len(high_a))/ALERTS_PER_PAGE))
            s=apg*ALERTS_PER_PAGE; e=s+ALERTS_PER_PAGE

            am1,am2,am3=st.columns(3)
            with am1: st.metric("🔴 Critical",f"{len(crit_a):,}")
            with am2: st.metric("🟠 High",    f"{len(high_a):,}")
            with am3: st.metric("🚨 Total",   f"{len(adf):,}")

            st.markdown(f"**Page {apg+1}/{total_apages}** &nbsp;·&nbsp; "
                        f"Showing alerts {s+1}–{min(e,max(len(crit_a),len(high_a)))} of {max(len(crit_a),len(high_a)):,}")

            a1,a2=st.columns(2)
            with a1:
                st.markdown("**🔴 Critical Alerts**")
                for _,r in crit_a.iloc[s:e].iterrows():
                    st.error(f"**[{r['Court']}]** {r['Type']} — {r['Message']}  \n`{r['Case']}`")
            with a2:
                st.markdown("**🟠 High Alerts**")
                for _,r in high_a.iloc[s:e].iterrows():
                    st.warning(f"**[{r['Court']}]** {r['Type']} — {r['Message']}  \n`{r['Case']}`")

            ac1,ac2,ac3=st.columns([1,3,1])
            with ac1:
                if st.button("◀ Prev alerts",disabled=(apg==0),use_container_width=True):
                    st.session_state.alert_pg-=1; st.rerun()
            with ac2:
                dots="".join([
                    f"<span style='display:inline-block;width:{'18px' if i==apg else '8px'};height:8px;"
                    f"border-radius:4px;background:{'#6c5ce7' if i==apg else '#ddd'};margin:0 2px'></span>"
                    for i in range(min(total_apages,10))])
                st.markdown(f"<div style='text-align:center;margin-top:9px'>{dots}</div>",unsafe_allow_html=True)
            with ac3:
                if st.button("Next alerts ▶",disabled=(apg>=total_apages-1),use_container_width=True):
                    st.session_state.alert_pg+=1; st.rerun()

    # ── PAGE 4: CASE EXPLORER
    elif p == 4:
        cases=fdf["case_number"].head(300).tolist()
        if not cases: st.info("No cases match filter.")
        else:
            sel=st.selectbox("🔍 Select Case",cases)
            row=fdf[fdf["case_number"]==sel].iloc[0]
            e1,e2,e3,e4,e5,e6=st.columns(6)
            with e1: st.metric("Score",        f"{row['priority_score']:.1f}/100")
            with e2: st.metric("Label",         f"{row.get('priority_emoji','')} {row['priority_label']}")
            with e3: st.metric("Disposal",      f"{row.get('disposal_days',0)}d")
            with e4: st.metric("ML Risk",       f"{row.get('delay_risk_score',0):.1f}/100")
            with e5: st.metric("Case Age",      f"{row['case_age_days']}d")
            with e6: st.metric("Urgency",       f"{row.get('legal_urgency_score',0):.1f}/100")

            d1,d2=st.columns([1,2])
            with d1:
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
                AGE_BINS=[0,180,365,730,1095,1825,3650]; AGE_VALS=[0,15,35,55,70,85,100]
                age_s=float(np.clip(np.interp(row["case_age_days"],AGE_BINS,AGE_VALS),0,100))
                bd={"Urgency\n(35%)":row.get("legal_urgency_score",0),
                    "Age Score\n(25%)":age_s,
                    "ML Risk\n(20%)":row.get("delay_risk_score",0)}
                fig=go.Figure(go.Bar(x=list(bd.keys()),y=list(bd.values()),
                    marker_color=["#dc3545","#fd7e14","#6c5ce7"],
                    text=[f"{v:.1f}" for v in bd.values()],textposition="auto"))
                fig.update_layout(yaxis=dict(range=[0,100]),title="Score Breakdown",
                    template="plotly_white",height=260,margin=dict(t=32,b=0,l=0,r=0))
                st.plotly_chart(fig,use_container_width=True)
                flags=str(row.get("urgency_flags",""))
                if flags and flags not in("nan","No urgent flags",""):
                    for f in flags.split(";"):
                        f=f.strip()
                        if f:
                            (st.error if any(k in f for k in["BREACHED","479","POCSO"]) else st.warning)(f"{'🚨' if 'BREACHED' in f else '⚠️'} {f}")

    # ── PAGE 5: ML INFO
    elif p == 5:
        metrics={"rmse":47.0,"r2":0.801,"f1":0.764,"auc":0.868}
        mp=os.path.join(MODELS_DIR,"model_metrics.json")
        if os.path.exists(mp):
            import json
            with open(mp) as f: metrics=json.load(f)
        ok=lambda v,t,rev=False:"✅" if(v<t if rev else v>t) else "⚠️"
        m1,m2,m3,m4=st.columns(4)
        with m1: st.metric("📉 RMSE",    f"{metrics['rmse']}d",   f"{ok(metrics['rmse'],60,True)} <60d")
        with m2: st.metric("📈 R²",      f"{metrics['r2']}",       f"{ok(metrics['r2'],0.75)} >0.75")
        with m3: st.metric("🎯 F1",      f"{metrics['f1']}",       f"{ok(metrics['f1'],0.75)} >0.75")
        with m4: st.metric("📊 AUC-ROC", f"{metrics['auc']}",      f"{ok(metrics['auc'],0.85)} >0.85")

        c1,c2=st.columns([3,2])
        with c1:
            fig=px.histogram(fdf,x="priority_score",nbins=50,color="priority_label",
                             color_discrete_map=PRIORITY_COLORS,
                             title="Priority Score Distribution",template="plotly_white")
            fig.update_layout(height=CH,margin=dict(t=32,b=0,l=0,r=0),
                              legend_title="Priority")
            st.plotly_chart(fig,use_container_width=True)
        with c2:
            st.markdown("#### Priority Formula")
            st.latex(r"P = 0.35U + 0.25A + 0.20C + 0.20M")
            st.markdown("""
| | Wt | Meaning |
|--|--|--|
| **U** | 35% | Statutory Urgency |
| **A** | 25% | Case Age |
| **C** | 20% | Category Aging |
| **M** | 20% | ML Delay Risk |

**XGBRegressor** → disposal days  
**XGBClassifier** → Low/Medium/High/Critical  
Tuned with **Optuna** (40 trials)
""")

# ─────────────────────────────────────────────────────────────────────
# BOTTOM PREV / NEXT  (page-level navigation)
# ─────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:6px'/>",unsafe_allow_html=True)
with st.container():
    st.markdown('<div class="btm-nav">', unsafe_allow_html=True)
    nb1,nb2,nb3,nb4,nb5 = st.columns([1,1,4,1,1])
    with nb1:
        if st.button("◀  Prev section", use_container_width=True, disabled=(p==0)):
            _prev(); st.rerun()
    with nb2:
        st.markdown(f"<p style='text-align:center;margin-top:10px;font-size:.75rem;color:#888'>{p+1}/{len(PAGES)}</p>",
                    unsafe_allow_html=True)
    with nb3:
        dots="".join([
            f"<span style='display:inline-block;"
            f"width:{'24px' if i==p else '9px'};height:9px;"
            f"border-radius:4px;background:{'#6c5ce7' if i==p else '#ddd'};"
            f"margin:0 3px;vertical-align:middle'></span>"
            for i in range(len(PAGES))])
        st.markdown(f"<div style='text-align:center;margin-top:10px'>{dots}</div>",unsafe_allow_html=True)
    with nb4:
        st.markdown(f"<p style='text-align:center;margin-top:10px;font-size:.75rem;font-weight:700;color:#6c5ce7'>{PAGES[p][1]}</p>",
                    unsafe_allow_html=True)
    with nb5:
        if st.button("Next section ▶", use_container_width=True, disabled=(p==len(PAGES)-1)):
            _next(); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
