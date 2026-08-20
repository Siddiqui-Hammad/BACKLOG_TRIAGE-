"""
Backlog Triage Case — Smart India Hackathon 2026
Snowflake Streamlit App — Paginated UI (no scroll, Next/Prev navigation)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import random, warnings
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
/* ── Remove default Streamlit padding so content fills screen */
.main .block-container { padding-top: 0.5rem !important; padding-bottom: 0 !important; max-width: 100% !important; }
header[data-testid="stHeader"]  { display: none !important; }
footer                           { display: none !important; }

/* ── Sidebar */
section[data-testid="stSidebar"] { background: #1a1a2e !important; }
section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
section[data-testid="stSidebar"] .stButton button {
    width: 100%; text-align: left; background: transparent;
    border: none; border-radius: 8px; padding: 8px 12px;
    color: #ccc !important; font-size: 0.92rem;
}
section[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(255,255,255,0.12) !important; color: #fff !important;
}

/* ── Metric cards */
div[data-testid="metric-container"] {
    background: white; border-radius: 12px;
    padding: 10px 14px; box-shadow: 0 2px 8px rgba(0,0,0,0.10);
    border-left: 4px solid #6c5ce7;
}

/* ── Nav bar at bottom */
.nav-bar {
    position: fixed; bottom: 0; left: 0; right: 0;
    background: #1a1a2e; height: 52px;
    display: flex; align-items: center; justify-content: center;
    gap: 12px; z-index: 9999; padding: 0 24px;
    box-shadow: 0 -2px 12px rgba(0,0,0,0.25);
}
.nav-dot {
    width: 10px; height: 10px; border-radius: 50%;
    background: #555; display: inline-block;
    transition: background 0.3s;
}
.nav-dot.active { background: #6c5ce7; width: 28px; border-radius: 5px; }

/* ── Active sidebar menu item */
.nav-active button { background: rgba(108,92,231,0.3) !important; color: #fff !important; font-weight: 600; }

/* ── Page title strip */
.page-title-strip {
    background: linear-gradient(90deg,#1a1a2e,#6c5ce7);
    color: white; padding: 6px 20px; border-radius: 10px;
    margin-bottom: 8px; display: flex; align-items: center;
    justify-content: space-between;
}
</style>
""", unsafe_allow_html=True)

PRIORITY_COLORS = {"Critical":"#dc3545","High":"#fd7e14","Medium":"#ffc107","Low":"#28a745"}

# ─────────────────────────────────────────────────────────────────────
# CONSTANTS — 28 States, 10 districts each, 40 courts/district
# ─────────────────────────────────────────────────────────────────────
STATES = [
    "Maharashtra","Uttar Pradesh","Karnataka","Tamil Nadu","Rajasthan",
    "Gujarat","West Bengal","Madhya Pradesh","Bihar","Delhi",
    "Andhra Pradesh","Telangana","Kerala","Punjab","Haryana",
    "Odisha","Jharkhand","Chhattisgarh","Assam","Uttarakhand",
    "Himachal Pradesh","Goa","Jammu & Kashmir","Manipur","Tripura",
    "Meghalaya","Nagaland","Sikkim",
]
DISTRICTS = {
    "Maharashtra":      ["Mumbai","Pune","Nagpur","Nashik","Aurangabad","Solapur","Thane","Kolhapur","Amravati","Nanded"],
    "Uttar Pradesh":    ["Lucknow","Allahabad","Varanasi","Kanpur","Agra","Meerut","Ghaziabad","Mathura","Bareilly","Gorakhpur"],
    "Karnataka":        ["Bengaluru","Mysuru","Hubballi","Mangaluru","Belagavi","Davangere","Ballari","Kalaburagi","Tumkur","Shivamogga"],
    "Tamil Nadu":       ["Chennai","Coimbatore","Madurai","Salem","Tiruchirappalli","Tirunelveli","Vellore","Erode","Thoothukudi","Dindigul"],
    "Rajasthan":        ["Jaipur","Jodhpur","Udaipur","Kota","Bikaner","Ajmer","Alwar","Bharatpur","Sikar","Pali"],
    "Gujarat":          ["Ahmedabad","Surat","Vadodara","Rajkot","Bhavnagar","Jamnagar","Junagadh","Gandhinagar","Anand","Mehsana"],
    "West Bengal":      ["Kolkata","Howrah","Durgapur","Siliguri","Asansol","Kharagpur","Haldia","Malda","Murshidabad","Nadia"],
    "Madhya Pradesh":   ["Bhopal","Indore","Gwalior","Jabalpur","Ujjain","Sagar","Rewa","Satna","Ratlam","Chhindwara"],
    "Bihar":            ["Patna","Gaya","Muzaffarpur","Bhagalpur","Darbhanga","Ara","Begusarai","Katihar","Munger","Saharsa"],
    "Delhi":            ["Central","South","North","East","West","Northwest","Southwest","New Delhi","Shahdara","Southeast"],
    "Andhra Pradesh":   ["Visakhapatnam","Vijayawada","Guntur","Nellore","Kurnool","Tirupati","Kakinada","Rajahmundry","Kadapa","Anantapur"],
    "Telangana":        ["Hyderabad","Warangal","Nizamabad","Karimnagar","Khammam","Mahbubnagar","Nalgonda","Adilabad","Suryapet","Medak"],
    "Kerala":           ["Thiruvananthapuram","Kochi","Kozhikode","Thrissur","Kollam","Palakkad","Kannur","Alappuzha","Malappuram","Kottayam"],
    "Punjab":           ["Ludhiana","Amritsar","Jalandhar","Patiala","Bathinda","Mohali","Hoshiarpur","Gurdaspur","Ferozepur","Moga"],
    "Haryana":          ["Gurugram","Faridabad","Ambala","Rohtak","Hisar","Panipat","Karnal","Sonipat","Yamunanagar","Bhiwani"],
    "Odisha":           ["Bhubaneswar","Cuttack","Rourkela","Berhampur","Sambalpur","Balasore","Puri","Jharsuguda","Rayagada","Koraput"],
    "Jharkhand":        ["Ranchi","Jamshedpur","Dhanbad","Bokaro","Deoghar","Hazaribagh","Giridih","Ramgarh","Chaibasa","Dumka"],
    "Chhattisgarh":     ["Raipur","Bhilai","Bilaspur","Korba","Durg","Rajnandgaon","Jagdalpur","Ambikapur","Dhamtari","Mahasamund"],
    "Assam":            ["Guwahati","Silchar","Dibrugarh","Jorhat","Nagaon","Tezpur","Tinsukia","Karimganj","Hailakandi","Goalpara"],
    "Uttarakhand":      ["Dehradun","Haridwar","Roorkee","Haldwani","Nainital","Rishikesh","Rudrapur","Kashipur","Srinagar","Pauri"],
    "Himachal Pradesh": ["Shimla","Dharamshala","Solan","Mandi","Kullu","Hamirpur","Una","Chamba","Bilaspur","Nahan"],
    "Goa":              ["Panaji","Margao","Vasco","Mapusa","Ponda","Bicholim","Canacona","Quepem","Sanguem","Pernem"],
    "Jammu & Kashmir":  ["Srinagar","Jammu","Anantnag","Baramulla","Sopore","Kathua","Udhampur","Rajouri","Poonch","Leh"],
    "Manipur":          ["Imphal","Thoubal","Bishnupur","Churachandpur","Senapati","Ukhrul","Tamenglong","Jiribam","Kakching","Kangpokpi"],
    "Tripura":          ["Agartala","Dharmanagar","Udaipur","Kailasahar","Ambassa","Sabroom","Belonia","Khowai","Melaghar","Sonamura"],
    "Meghalaya":        ["Shillong","Tura","Jowai","Nongstoin","Baghmara","Resubelpara","Ampati","Mairang","Nongpoh","Williamnagar"],
    "Nagaland":         ["Kohima","Dimapur","Mokokchung","Tuensang","Wokha","Zunheboto","Mon","Phek","Kiphire","Longleng"],
    "Sikkim":           ["Gangtok","Namchi","Gyalshing","Mangan","Jorethang","Rangpo","Singtam","Ravangla","Yuksom","Lachen"],
}
CATEGORIES = {
    "POCSO":               {"base":120, "statutory":365},
    "SC_ST":               {"base":180, "statutory":730},
    "Senior_Citizen":      {"base":90,  "statutory":180},
    "Commercial":          {"base":270, "statutory":365},
    "NDPS":                {"base":365, "statutory":1095},
    "Matrimonial":         {"base":540, "statutory":1825},
    "Motor_Accident":      {"base":730, "statutory":1825},
    "Property_Dispute":    {"base":1095,"statutory":2920},
    "General_Civil":       {"base":900, "statutory":2190},
    "General_Criminal":    {"base":450, "statutory":730},
    "Cheque_Bounce":       {"base":180, "statutory":365},
    "Labour_Dispute":      {"base":540, "statutory":1095},
    "Consumer_Forum":      {"base":270, "statutory":730},
    "Land_Acquisition":    {"base":730, "statutory":1825},
    "Constitutional_Writ": {"base":365, "statutory":730},
}
STAGES = ["Filing","Admission","Notice","Written_Statement","Evidence","Arguments","Judgment","Execution"]
STAGE_STAGNATION = {"Filing":15,"Admission":30,"Notice":60,"Written_Statement":90,
                    "Evidence":180,"Arguments":90,"Judgment":60,"Execution":120}
FEATURE_COLS = [
    "case_type_enc","case_category_enc","bench_type_enc","current_stage_enc","state_enc",
    "stage_index","case_age_days","stage_age_days","hearings_held","adjournments",
    "adjournment_rate","stagnation_flag","is_undertrial","district_avg_disposal_days",
    "statutory_deadline_days","days_beyond_statutory","case_age_normalized",
    "hearing_density","stage_completion_ratio",
]

PAGES = [
    ("📊", "Overview"),
    ("🏛️", "Court Analysis"),
    ("🌳", "XGBoost Metrics"),
    ("📋", "Priority Queue"),
    ("🚨", "Alerts"),
    ("🔍", "Case Explorer"),
]

# ─────────────────────────────────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────────────────────────────────
def generate_cases(n=10000, seed=42):
    np.random.seed(seed); random.seed(seed)
    rows = []
    cat_names   = list(CATEGORIES.keys())
    bench_types = ["Single_Judge","Division_Bench","Full_Bench","Magistrate","Sessions"]
    case_types  = ["Civil","Criminal","Writ","Appeal","Revision","Execution"]
    for i in range(n):
        state    = random.choice(STATES)
        district = random.choice(DISTRICTS[state])
        court_id = f"{state[:3].upper()}-{district[:3].upper()}-{random.randint(1,40):02d}"
        category = random.choice(cat_names)
        info     = CATEGORIES[category]
        case_type = random.choice(case_types)
        case_age  = random.randint(60, 3500)
        stage_idx = min(int(case_age/(info["base"]/8)) + np.random.randint(-2,3), 7)
        stage_idx = max(0, stage_idx)
        stage     = STAGES[stage_idx]
        stage_age = int(np.random.exponential(info["base"]/8))
        stage_age = min(stage_age, case_age)
        hearings  = max(1, int(np.random.poisson(case_age // 30 * 0.7)))
        adj       = min(int(np.random.poisson(hearings * 0.35)), hearings)
        adj_rate  = adj / max(1, hearings)
        stagnation = int(stage_age > STAGE_STAGNATION.get(stage, 90) * 1.5)
        undertrial = int(case_type=="Criminal" and case_age>365 and category in ["NDPS","General_Criminal"])
        dist_avg  = int(info["base"] * np.random.uniform(0.8, 1.4))
        base_rem  = max(30, info["base"] - case_age)
        delay_f   = 1 + adj_rate*0.5 + stagnation*0.3 + undertrial*0.2 + (stage_idx<3)*0.4
        disposal  = max(10, int(base_rem * delay_f + np.random.normal(0, 60)))
        risk      = float(np.clip(
            min(40, (case_age/info["statutory"])*40) + adj_rate*20 +
            stagnation*15 + undertrial*15 + (7-stage_idx)/7*10 + np.random.normal(0,5), 0, 100))
        label     = ("Critical" if risk>=75 else "High" if risk>=50 else "Medium" if risk>=25 else "Low")
        rows.append({
            "case_number":f"CASE/{i+1:05d}/{random.randint(2015,2024)}",
            "court_id":court_id,"state":state,"district":district,
            "judge_id":f"JDG-{random.randint(100,999)}",
            "case_type":case_type,"case_category":category,
            "bench_type":random.choice(bench_types),
            "current_stage":stage,"stage_index":stage_idx,
            "case_age_days":case_age,"stage_age_days":stage_age,
            "hearings_held":hearings,"adjournments":adj,
            "adjournment_rate":round(adj_rate,3),
            "stagnation_flag":stagnation,"is_undertrial":undertrial,
            "district_avg_disposal_days":dist_avg,
            "statutory_deadline_days":info["statutory"],
            "days_beyond_statutory":max(0,case_age-info["statutory"]),
            "disposal_days":disposal,"delay_risk_score":round(risk,2),"delay_risk_label":label,
        })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────
# PREPROCESSING + TRAINING
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

def train_models(df):
    from xgboost import XGBRegressor, XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import mean_squared_error, r2_score, f1_score, roc_auc_score
    RISK_ORDER = ["Low","Medium","High","Critical"]
    le_risk = LabelEncoder(); le_risk.classes_ = np.array(RISK_ORDER)
    df_p = preprocess(df); X = df_p[FEATURE_COLS]
    yr = df_p["disposal_days"].values; yc = le_risk.transform(df_p["delay_risk_label"].values)
    X_tr,X_te,yr_tr,yr_te,yc_tr,yc_te = train_test_split(X,yr,yc,test_size=0.2,random_state=42,stratify=yc)
    reg = XGBRegressor(n_estimators=300,max_depth=5,learning_rate=0.05,subsample=0.8,
                       colsample_bytree=0.8,random_state=42,tree_method="hist",n_jobs=-1)
    reg.fit(X_tr,yr_tr,eval_set=[(X_te,yr_te)],verbose=False)
    yr_pred = reg.predict(X_te)
    clf = XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.05,subsample=0.8,
                        colsample_bytree=0.8,random_state=42,tree_method="hist",n_jobs=-1,
                        num_class=4,objective="multi:softprob",eval_metric="mlogloss")
    clf.fit(X_tr,yc_tr,eval_set=[(X_te,yc_te)],verbose=False)
    yc_pred = clf.predict(X_te); yc_proba = clf.predict_proba(X_te)
    metrics = {"rmse":round(np.sqrt(mean_squared_error(yr_te,yr_pred)),1),
               "r2":round(r2_score(yr_te,yr_pred),3),
               "f1":round(f1_score(yc_te,yc_pred,average="weighted"),3),
               "auc":round(roc_auc_score(yc_te,yc_proba,multi_class="ovr",average="weighted"),3)}
    imp = reg.get_booster().get_score(importance_type="gain")
    FEAT_LABELS = {"case_age_days":"Case Age","stage_age_days":"Stage Age","adjournment_rate":"Adj. Rate",
                   "days_beyond_statutory":"Days Beyond Statutory","case_age_normalized":"Case Age (Norm)",
                   "hearing_density":"Hearing Density","statutory_deadline_days":"Statutory Deadline",
                   "district_avg_disposal_days":"District Avg Disposal","stagnation_flag":"Stagnation",
                   "is_undertrial":"Undertrial (BNSS 479)","stage_index":"Stage Index",
                   "hearings_held":"Hearings Held","adjournments":"Adjournments",
                   "stage_completion_ratio":"Stage Completion","case_category_enc":"Case Category",
                   "current_stage_enc":"Current Stage","state_enc":"State",
                   "case_type_enc":"Case Type","bench_type_enc":"Bench Type"}
    feat_imp = pd.DataFrame([{"Feature":FEAT_LABELS.get(k,k),"Gain":round(v,2)} for k,v in imp.items()]
                            ).sort_values("Gain",ascending=True).tail(12)
    return reg, clf, le_risk, metrics, feat_imp

def score_urgency(row):
    cat=row["case_category"]; age=int(row["case_age_days"]); stage=row["current_stage"]
    stage_age=int(row["stage_age_days"]); adj_rate=float(row["adjournment_rate"])
    statutory=int(row["statutory_deadline_days"])
    score=0.0; flags=[]
    ratio=age/max(statutory,1)
    if ratio>=1.0:   score+=35; flags.append("STATUTORY DEADLINE BREACHED")
    elif ratio>=.75: score+=25; flags.append("Approaching statutory deadline")
    elif ratio>=.5:  score+=15; flags.append("Past halfway of statutory limit")
    if cat=="POCSO":         score+=20; flags.append("POCSO: Mandatory speedy trial")
    elif cat=="SC_ST":       score+=18; flags.append("SC/ST PoA: Priority disposal")
    elif cat=="Senior_Citizen": score+=15; flags.append("Senior Citizen: Expedited disposal")
    if row.get("is_undertrial",0)==1 and age>365: score+=20; flags.append("BNSS 479: Undertrial threshold crossed")
    stag_lim=STAGE_STAGNATION.get(stage,90)
    if stage_age>stag_lim*2:    score+=15; flags.append(f"Critical stagnation in {stage}")
    elif stage_age>stag_lim*1.5: score+=10; flags.append(f"Stagnation in {stage}")
    if adj_rate>=0.6: score+=10; flags.append(f"High adjournment rate {adj_rate:.0%}")
    if age>1825:      score+=10; flags.append(f"Case aged {age//365}+ years")
    return float(np.clip(score,0,100)), "; ".join(flags) if flags else "No urgent flags"

AGE_BINS=[0,180,365,730,1095,1825,3650]; AGE_VALS=[0,15,35,55,70,85,100]
def hybrid_priority(row):
    age_score = float(np.clip(np.interp(row["case_age_days"],AGE_BINS,AGE_VALS),0,100))
    threshold = CATEGORIES.get(row["case_category"],{"statutory":2190})["statutory"]
    cat_ratio = row["case_age_days"]/max(threshold,1)
    cat_score = float(np.clip(min(100,cat_ratio*70+(15 if cat_ratio>=1.5 else 0)+(30 if cat_ratio>=2.0 else 0)),0,100))
    ps = float(np.clip(0.35*row["legal_urgency_score"]+0.25*age_score+0.20*cat_score+0.20*row["ml_delay_risk_score"],0,100))
    label = ("Critical" if ps>=75 else "High" if ps>=50 else "Medium" if ps>=25 else "Low")
    return round(ps,2), label, {"Critical":"🔴","High":"🟠","Medium":"🟡","Low":"🟢"}[label]

@st.cache_data(show_spinner=False)
def run_pipeline(n_cases=10000):
    df = generate_cases(n=n_cases)
    reg, clf, le_risk, metrics, feat_imp = train_models(df)
    df_p = preprocess(df); X = df_p[FEATURE_COLS]
    df["predicted_disposal_days"] = reg.predict(X).astype(int)
    risk_enc = clf.predict(X)
    df["ml_delay_risk_score"] = (risk_enc/3.0*100).round(1)
    df["ml_delay_risk_label"] = le_risk.inverse_transform(risk_enc)
    urg = df.apply(score_urgency,axis=1,result_type="expand")
    urg.columns = ["legal_urgency_score","urgency_flags"]
    df = pd.concat([df,urg],axis=1)
    pri = df.apply(hybrid_priority,axis=1,result_type="expand")
    pri.columns = ["priority_score","priority_label","priority_emoji"]
    df = pd.concat([df,pri],axis=1)
    df["court_rank"] = df.groupby("court_id")["priority_score"].rank(ascending=False,method="first").astype(int)
    return df, metrics, feat_imp

# ─────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = 0

def go_to(p):
    st.session_state.page = p

def next_page():
    if st.session_state.page < len(PAGES)-1:
        st.session_state.page += 1

def prev_page():
    if st.session_state.page > 0:
        st.session_state.page -= 1

# ─────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ Backlog Triage Case")
    st.caption("SIH 2026 — git win")
    st.divider()

    # Page navigation menu
    st.markdown("### Navigation")
    for i, (icon, name) in enumerate(PAGES):
        active = "nav-active" if i == st.session_state.page else ""
        with st.container():
            if i == st.session_state.page:
                st.markdown(f"<div style='background:rgba(108,92,231,0.25);border-radius:8px;padding:4px 0'>", unsafe_allow_html=True)
            if st.button(f"{icon}  {name}", key=f"nav_{i}"):
                go_to(i)
            if i == st.session_state.page:
                st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("### ⚙️ Settings")
    n_cases = st.slider("Dataset size", 1000, 50000, 10000, step=1000)

# ─────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────
with st.spinner("🔄 Training XGBoost & scoring cases…"):
    df, metrics, feat_imp = run_pipeline(n_cases=n_cases)

# ── Filters (in sidebar, after data loaded)
with st.sidebar:
    st.divider()
    st.markdown("### 🔍 Filters")
    all_courts   = sorted(df["court_id"].unique().tolist())
    sel_courts   = st.multiselect("🏛️ Court ID", all_courts, default=[])
    sel_category = st.multiselect("📂 Category", list(CATEGORIES.keys()))
    sel_priority = st.multiselect("🚦 Priority", ["Critical","High","Medium","Low"], default=["Critical","High"])
    sel_state    = st.selectbox("📍 State", ["All"]+sorted(STATES))
    n_shown = st.session_state.get("n_filtered", len(df))
    st.markdown(f"**{n_shown:,} cases** across **{df['court_id'].nunique():,} courts**")

# Apply filters
fdf = df.copy()
if sel_courts:         fdf = fdf[fdf["court_id"].isin(sel_courts)]
if sel_state!="All":   fdf = fdf[fdf["state"]==sel_state]
if sel_category:       fdf = fdf[fdf["case_category"].isin(sel_category)]
if sel_priority:       fdf = fdf[fdf["priority_label"].isin(sel_priority)]
st.session_state["n_filtered"] = len(fdf)

# ─────────────────────────────────────────────────────────────────────
# PAGE HEADER STRIP
# ─────────────────────────────────────────────────────────────────────
cur_icon, cur_name = PAGES[st.session_state.page]
st.markdown(f"""
<div class="page-title-strip">
  <span style="font-size:1.15rem;font-weight:700">{cur_icon} &nbsp; {cur_name}</span>
  <span style="font-size:0.8rem;opacity:0.75">Page {st.session_state.page+1} of {len(PAGES)} &nbsp;|&nbsp; {len(fdf):,} cases &nbsp;|&nbsp; {fdf['court_id'].nunique():,} courts</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# PAGE RENDERERS
# ─────────────────────────────────────────────────────────────────────
CH = 310   # chart height constant

def page_overview(fdf):
    total  = len(fdf)
    crit   = (fdf["priority_label"]=="Critical").sum()
    high   = (fdf["priority_label"]=="High").sum()
    courts = fdf["court_id"].nunique()
    under  = fdf["is_undertrial"].sum()
    avg_d  = fdf["predicted_disposal_days"].mean()

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: st.metric("🏛️ Courts",   f"{courts:,}")
    with c2: st.metric("📁 Cases",    f"{total:,}")
    with c3: st.metric("🔴 Critical", f"{crit:,}",  f"{crit/max(total,1)*100:.1f}%")
    with c4: st.metric("🟠 High",     f"{high:,}",  f"{high/max(total,1)*100:.1f}%")
    with c5: st.metric("⛓️ Undertrial",f"{int(under):,}")
    with c6: st.metric("📅 Avg Disposal",f"{avg_d:.0f}d")

    st.markdown("<div style='height:6px'/>", unsafe_allow_html=True)
    r1,r2 = st.columns(2)
    with r1:
        lo = ["Critical","High","Medium","Low"]
        lc = fdf["priority_label"].value_counts().reindex(lo,fill_value=0).reset_index()
        lc.columns = ["Priority","Count"]
        fig = px.bar(lc,x="Priority",y="Count",color="Priority",
                     color_discrete_map=PRIORITY_COLORS,
                     title="Cases by Priority Label",template="plotly_white")
        fig.update_layout(showlegend=False,height=CH,margin=dict(t=36,b=4,l=4,r=4))
        st.plotly_chart(fig,use_container_width=True)
    with r2:
        cc = fdf.groupby(["case_category","priority_label"]).size().reset_index(name="count")
        fig2 = px.bar(cc,x="case_category",y="count",color="priority_label",
                      color_discrete_map=PRIORITY_COLORS,
                      title="Priority by Case Category",template="plotly_white",barmode="stack")
        fig2.update_layout(height=CH,xaxis_tickangle=-35,legend_title="Priority",
                           margin=dict(t=36,b=4,l=4,r=4))
        st.plotly_chart(fig2,use_container_width=True)

def page_court_analysis(fdf):
    r1,r2 = st.columns(2)
    with r1:
        cc = (fdf[fdf["priority_label"]=="Critical"]
              .groupby("court_id").size().reset_index(name="Critical Cases")
              .sort_values("Critical Cases",ascending=True).tail(20))
        fig = px.bar(cc,x="Critical Cases",y="court_id",orientation="h",
                     color="Critical Cases",color_continuous_scale="Reds",
                     title="Top 20 Courts — Critical Cases",template="plotly_white")
        fig.update_layout(height=CH+60,coloraxis_showscale=False,
                          yaxis_title="",margin=dict(t=36,b=4,l=4,r=4))
        st.plotly_chart(fig,use_container_width=True)
    with r2:
        cs = fdf.groupby("court_id").agg(
            total=("case_number","count"),
            avg_p=("priority_score","mean"),
            crit=("priority_label",lambda x:(x=="Critical").sum()),
            avg_d=("predicted_disposal_days","mean"),
        ).reset_index()
        fig2 = px.scatter(cs.sort_values("avg_p",ascending=False).head(60),
                          x="avg_p",y="total",size="crit",color="avg_p",
                          color_continuous_scale="RdYlGn_r",hover_name="court_id",
                          title="Court Priority vs Caseload (bubble = critical count)",
                          template="plotly_white",
                          labels={"avg_p":"Avg Priority Score","total":"Total Cases","crit":"Critical Cases"})
        fig2.update_layout(height=CH+60,coloraxis_showscale=False,margin=dict(t=36,b=4,l=4,r=4))
        st.plotly_chart(fig2,use_container_width=True)

    st.markdown("<div style='height:4px'/>", unsafe_allow_html=True)
    cs2 = fdf.groupby("state").agg(
        total=("case_number","count"),
        critical=("priority_label",lambda x:(x=="Critical").sum()),
        avg_score=("priority_score","mean"),
    ).reset_index().sort_values("critical",ascending=False).head(15)
    cs2["avg_score"] = cs2["avg_score"].round(1)
    st.dataframe(cs2.rename(columns={"state":"State","total":"Cases",
                                      "critical":"Critical","avg_score":"Avg Score"}),
                 use_container_width=True, height=160)

def page_xgboost(fdf, metrics, feat_imp):
    m1,m2,m3,m4 = st.columns(4)
    ok  = lambda v,t,rev=False: "✅" if (v<t if rev else v>t) else "⚠️"
    with m1: st.metric("Regressor RMSE", f"{metrics['rmse']} days", f"{ok(metrics['rmse'],60,True)} Target <60")
    with m2: st.metric("Regressor R²",   f"{metrics['r2']}",         f"{ok(metrics['r2'],0.75)} Target >0.75")
    with m3: st.metric("Classifier F1",  f"{metrics['f1']}",         f"{ok(metrics['f1'],0.75)} Target >0.75")
    with m4: st.metric("Classifier AUC", f"{metrics['auc']}",        f"{ok(metrics['auc'],0.85)} Target >0.85")

    st.markdown("<div style='height:6px'/>", unsafe_allow_html=True)
    col1,col2 = st.columns([3,2])
    with col1:
        fig = px.bar(feat_imp,x="Gain",y="Feature",orientation="h",
                     color="Gain",color_continuous_scale="Viridis",
                     title="XGBoost Feature Importance (Gain)",template="plotly_white")
        fig.update_layout(height=CH+50,coloraxis_showscale=False,margin=dict(t=36,b=4,l=4,r=4))
        st.plotly_chart(fig,use_container_width=True)
    with col2:
        st.markdown("### Priority Formula")
        st.latex(r"P = 0.35U + 0.25A + 0.20C + 0.20M")
        st.markdown("""
| | Weight | Component |
|--|--|--|
| **U** | 35% | Statutory Urgency |
| **A** | 25% | Case Age Score |
| **C** | 20% | Category Aging |
| **M** | 20% | ML Delay Risk |

**Two XGBoost models:**
- 🔢 **XGBRegressor** → Disposal days
- 🎯 **XGBClassifier** → Low / Medium / High / Critical
""")
        fig2 = px.histogram(fdf,x="priority_score",nbins=40,
                            color_discrete_sequence=["#6c5ce7"],
                            title="Score Distribution",template="plotly_white")
        fig2.update_layout(height=170,margin=dict(t=30,b=4,l=4,r=4))
        st.plotly_chart(fig2,use_container_width=True)

def page_queue(fdf):
    lo = ["Critical","High","Medium","Low"]
    lo_vals = {l:i for i,l in enumerate(lo)}
    view = fdf.copy()
    view["_sort"] = view["priority_label"].map(lo_vals).fillna(99)
    view = view.sort_values(["_sort","priority_score"],ascending=[True,False])
    disp = view[[c for c in ["priority_emoji","priority_score","case_number","court_id","state",
                              "case_category","case_type","current_stage","case_age_days",
                              "predicted_disposal_days","legal_urgency_score","ml_delay_risk_score",
                              "priority_label"] if c in view.columns]].head(500)
    disp = disp.rename(columns={
        "priority_emoji":"","priority_score":"Score","case_number":"Case No.",
        "court_id":"Court","state":"State","case_category":"Category",
        "case_type":"Type","current_stage":"Stage","case_age_days":"Age(d)",
        "predicted_disposal_days":"Disposal(d)","legal_urgency_score":"Urgency",
        "ml_delay_risk_score":"ML Risk","priority_label":"Label"
    })
    st.dataframe(disp, use_container_width=True, height=490,
        column_config={
            "Score":    st.column_config.ProgressColumn("Score",    min_value=0,max_value=100,format="%.1f"),
            "Urgency":  st.column_config.ProgressColumn("Urgency",  min_value=0,max_value=100,format="%.1f"),
            "ML Risk":  st.column_config.ProgressColumn("ML Risk",  min_value=0,max_value=100,format="%.1f"),
        }
    )

def page_alerts(fdf):
    alerts = []
    for _,r in fdf.iterrows():
        ps=r.get("priority_score",0); age=r.get("case_age_days",0)
        flags=str(r.get("urgency_flags",""))
        if ps>=75:
            alerts.append({"Court":r["court_id"],"Case":r["case_number"],"Category":r["case_category"],
                "Type":"CRITICAL PRIORITY",f"Message":f"Score {ps:.1f}/100","Severity":"Critical"})
        if r.get("is_undertrial",0)==1 and age>365:
            alerts.append({"Court":r["court_id"],"Case":r["case_number"],"Category":r["case_category"],
                "Type":"BNSS 479",f"Message":f"Pending {age}d — bail eligibility review","Severity":"Critical"})
        if r.get("stagnation_flag",0)==1 and r.get("stage_age_days",0)>180:
            alerts.append({"Court":r["court_id"],"Case":r["case_number"],"Category":r["case_category"],
                "Type":"STAGNATION",f"Message":f"Stuck in '{r['current_stage']}' for {r['stage_age_days']}d","Severity":"High"})
        if "BREACHED" in flags:
            alerts.append({"Court":r["court_id"],"Case":r["case_number"],"Category":r["case_category"],
                "Type":"STATUTORY BREACH",f"Message":f"{r['case_category']} statutory deadline exceeded","Severity":"Critical"})

    adf = pd.DataFrame(alerts) if alerts else pd.DataFrame()
    if adf.empty:
        st.success("No alerts in current filter.")
        return

    crit_a = adf[adf["Severity"]=="Critical"]
    high_a = adf[adf["Severity"]=="High"]

    ac1,ac2,ac3 = st.columns(3)
    with ac1: st.metric("🔴 Critical Alerts", f"{len(crit_a):,}")
    with ac2: st.metric("🟠 High Alerts",     f"{len(high_a):,}")
    with ac3: st.metric("🚨 Total Alerts",    f"{len(adf):,}")

    st.markdown("<div style='height:4px'/>", unsafe_allow_html=True)
    a1,a2 = st.columns(2)
    with a1:
        st.markdown("**🔴 Critical Alerts**")
        for _,r in crit_a.head(12).iterrows():
            st.error(f"**[{r['Court']}]** {r['Type']} — {r['Message']} | `{r['Case']}`")
    with a2:
        st.markdown("**🟠 High Alerts**")
        for _,r in high_a.head(12).iterrows():
            st.warning(f"**[{r['Court']}]** {r['Type']} — {r['Message']} | `{r['Case']}`")

    st.markdown("<div style='height:4px'/>", unsafe_allow_html=True)
    at1,at2 = st.columns(2)
    with at1:
        ac = adf.groupby("Type").size().reset_index(name="Count")
        fig = px.bar(ac,x="Count",y="Type",orientation="h",color="Count",
                     color_continuous_scale="Reds",title="Alerts by Type",template="plotly_white")
        fig.update_layout(height=220,coloraxis_showscale=False,margin=dict(t=30,b=2,l=2,r=2))
        st.plotly_chart(fig,use_container_width=True)
    with at2:
        court_a = crit_a.groupby("Court").size().reset_index(name="Critical Alerts").sort_values("Critical Alerts",ascending=False).head(10)
        fig2 = px.bar(court_a,x="Court",y="Critical Alerts",color="Critical Alerts",
                      color_continuous_scale="OrRd",title="Courts with Most Critical Alerts",template="plotly_white")
        fig2.update_layout(height=220,coloraxis_showscale=False,xaxis_tickangle=-30,margin=dict(t=30,b=2,l=2,r=2))
        st.plotly_chart(fig2,use_container_width=True)

def page_case_explorer(fdf):
    cases = fdf["case_number"].head(300).tolist()
    sel = st.selectbox("Select a Case", cases)
    if not sel: return
    row = fdf[fdf["case_number"]==sel].iloc[0]
    label = row["priority_label"]
    color = {"Critical":"#dc3545","High":"#fd7e14","Medium":"#ffc107","Low":"#28a745"}.get(label,"#666")

    e1,e2,e3,e4,e5 = st.columns(5)
    with e1: st.metric("Priority Score",   f"{row['priority_score']:.1f}/100")
    with e2: st.metric("Priority Label",   f"{row['priority_emoji']} {label}")
    with e3: st.metric("Disposal Pred.",   f"{row['predicted_disposal_days']} days")
    with e4: st.metric("ML Risk Score",    f"{row['ml_delay_risk_score']:.1f}/100")
    with e5: st.metric("Case Age",         f"{row['case_age_days']} days")

    st.markdown("<div style='height:4px'/>", unsafe_allow_html=True)
    d1,d2 = st.columns([1,2])
    with d1:
        st.markdown("**📌 Case Details**")
        st.table(pd.DataFrame({
            "Field":["Court","State","District","Category","Type","Stage",
                     "Stage Age","Hearings","Adjournments","Stagnation","Undertrial"],
            "Value":[row["court_id"],row["state"],row["district"],row["case_category"],
                     row["case_type"],row["current_stage"],f"{row['stage_age_days']}d",
                     row["hearings_held"],row["adjournments"],
                     "⚠️ Yes" if row["stagnation_flag"] else "✅ No",
                     "⚠️ Yes" if row["is_undertrial"]  else "✅ No"],
        }))
    with d2:
        age_score = float(np.clip(np.interp(row["case_age_days"],AGE_BINS,AGE_VALS),0,100))
        threshold = CATEGORIES.get(row["case_category"],{"statutory":2190})["statutory"]
        cat_ratio = row["case_age_days"]/max(threshold,1)
        cat_score = float(np.clip(min(100,cat_ratio*70+(15 if cat_ratio>=1.5 else 0)),0,100))
        bd = {
            "Statutory Urgency\n(35%)": row["legal_urgency_score"],
            "Case Age Score\n(25%)":    age_score,
            "Category Aging\n(20%)":    cat_score,
            "ML Delay Risk\n(20%)":     row["ml_delay_risk_score"],
        }
        fig = go.Figure(go.Bar(
            x=list(bd.keys()),y=list(bd.values()),
            marker_color=["#dc3545","#fd7e14","#ffc107","#6c5ce7"],
            text=[f"{v:.1f}" for v in bd.values()],textposition="auto",
        ))
        fig.update_layout(yaxis=dict(range=[0,100]),title="Score Breakdown",
                          template="plotly_white",height=280,margin=dict(t=36,b=4,l=4,r=4))
        st.plotly_chart(fig,use_container_width=True)

        flags=str(row.get("urgency_flags",""))
        if flags and flags!="nan" and flags!="No urgent flags":
            st.markdown("**⚖️ Legal Flags Triggered**")
            for f in flags.split(";"):
                if f.strip(): st.warning(f.strip())

# ─────────────────────────────────────────────────────────────────────
# RENDER ACTIVE PAGE
# ─────────────────────────────────────────────────────────────────────
p = st.session_state.page
if   p == 0: page_overview(fdf)
elif p == 1: page_court_analysis(fdf)
elif p == 2: page_xgboost(fdf, metrics, feat_imp)
elif p == 3: page_queue(fdf)
elif p == 4: page_alerts(fdf)
elif p == 5: page_case_explorer(fdf)

# ─────────────────────────────────────────────────────────────────────
# BOTTOM NAVIGATION BAR — Prev · dots · Next
# ─────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:58px'/>", unsafe_allow_html=True)  # spacer above fixed bar

nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([1,1,4,1,1])
with nav_col1:
    if st.button("◀  Prev", disabled=(p==0), use_container_width=True):
        prev_page()
        st.rerun()
with nav_col2:
    st.markdown(f"<p style='text-align:center;margin-top:8px;color:#888;font-size:0.8rem'>{p+1}/{len(PAGES)}</p>", unsafe_allow_html=True)
with nav_col3:
    # Dot indicators
    dots = "".join([
        f"<span style='display:inline-block;width:{'28px' if i==p else '10px'};height:10px;border-radius:5px;"
        f"background:{'#6c5ce7' if i==p else '#ccc'};margin:0 3px;transition:all 0.3s'></span>"
        for i in range(len(PAGES))
    ])
    st.markdown(f"<div style='text-align:center;margin-top:6px'>{dots}</div>", unsafe_allow_html=True)
with nav_col4:
    st.markdown(f"<p style='text-align:center;margin-top:8px;color:#888;font-size:0.8rem'>{PAGES[p][1]}</p>", unsafe_allow_html=True)
with nav_col5:
    if st.button("Next  ▶", disabled=(p==len(PAGES)-1), use_container_width=True):
        next_page()
        st.rerun()
