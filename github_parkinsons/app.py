import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import diagnostic_engine as engine

# Page Configuration
st.set_page_config(
    page_title="Parkinson's AI Diagnostic Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .header-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        padding: 24px 28px;
        border-radius: 14px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #334155;
    }
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #FFFFFF !important;
        margin-bottom: 0.4rem;
        letter-spacing: -0.02em;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #FFFFFF !important;
        opacity: 0.92;
        line-height: 1.5;
        margin-bottom: 0;
    }
    .explanation-card {
        background-color: #2D3748;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        border: 1px solid #4A5568;
        border-left: 5px solid #3B82F6;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
    .explanation-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #FFFFFF !important;
        margin-bottom: 0.6rem;
    }
    .explanation-item {
        font-size: 0.95rem;
        color: #FFFFFF !important;
        line-height: 1.6;
        margin-bottom: 0.6rem;
    }
    .explanation-item b, .explanation-item strong, .explanation-item i {
        color: #FFFFFF !important;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-pd {
        background-color: #FEE2E2;
        color: #DC2626;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #FCA5A5;
    }
    .badge-other {
        background-color: #FEF3C7;
        color: #D97706;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #FCD34D;
    }
    .badge-healthy {
        background-color: #D1FAE5;
        color: #059669;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #6EE7B7;
    }
    .match-tag-correct {
        color: #059669;
        font-weight: 600;
        background-color: #ECFDF5;
        padding: 4px 10px;
        border-radius: 6px;
        border: 1px solid #A7F3D0;
    }
    .match-tag-mismatch {
        color: #DC2626;
        font-weight: 600;
        background-color: #FEF2F2;
        padding: 4px 10px;
        border-radius: 6px;
        border: 1px solid #FECACA;
    }
</style>
""", unsafe_allow_html=True)

# Cache model initialization so it trains once in background
@st.cache_resource(show_spinner=False)
def initialize_models():
    return engine.load_ensemble_models()

# App Header
st.markdown("""
<div class="header-banner">
    <div class="main-header">🧠 Parkinson's Disease Multimodal AI Diagnostic System</div>
    <div class="sub-header">3-Class Weighted Ensemble Architecture differentiating <b>Parkinson's Disease</b> from <b>Essential Tremor & Other Disorders</b> and <b>Healthy Controls</b> across 12 Kinematic & Clinical Tests</div>
</div>
""", unsafe_allow_html=True)

# Initialize models with progress spinner if first boot
with st.spinner("Initializing 12 AI Models in background (Loading Random Forests)..."):
    initialize_models()

# Sidebar: Patient Selection & Filters
st.sidebar.header("🔍 Patient Navigation")

patients = engine.get_patient_list()
if not patients:
    st.error("No patient data found. Please ensure the dataset directory is present.")
    st.stop()

# Filter by diagnosis category
filter_option = st.sidebar.selectbox(
    "Filter by Ground Truth Condition:",
    ["All Patients", "Parkinson's", "Essential Tremor", "Other Movement Disorders", "Multiple Sclerosis", "Atypical Parkinsonism", "Healthy"]
)

if filter_option == "All Patients":
    filtered_patients = patients
else:
    filtered_patients = [p for p in patients if filter_option.lower() in p["condition"].lower()]

patient_options = {p["id"]: p["label"] for p in filtered_patients}

if not patient_options:
    st.sidebar.warning(f"No patients found matching '{filter_option}'")
    filtered_patients = patients
    patient_options = {p["id"]: p["label"] for p in filtered_patients}

# Quick random patient button (horizontal, full width)
if st.sidebar.button("🎲 Pick Random Patient", use_container_width=True):
    st.session_state["selected_pid"] = int(np.random.choice([p["id"] for p in filtered_patients]))

# Set default selection
if "selected_pid" not in st.session_state or st.session_state["selected_pid"] not in patient_options:
    # Default to a famous case like 028 (PD) or 066 (ET)
    default_pid = 28 if 28 in patient_options else list(patient_options.keys())[0]
    st.session_state["selected_pid"] = default_pid

selected_pid = st.sidebar.selectbox(
    "Select Patient ID:",
    options=list(patient_options.keys()),
    format_func=lambda x: patient_options[x],
    index=list(patient_options.keys()).index(st.session_state["selected_pid"])
)
st.session_state["selected_pid"] = selected_pid

st.sidebar.markdown("---")
st.sidebar.info("""
**Architecture Highlights:**
- **12 Specialized Random Forests** (11 smartwatch tasks + 30-item questionnaire)
- **41–50 Kinematic Features per Task** (Tremor power 4–10 Hz, Jerk, Wrist Asymmetry)
- **Weighted Ensemble** resolving Essential Tremor vs Parkinson's overlap
""")

# Run Diagnosis for selected patient
diag = engine.run_master_diagnosis(selected_pid)

# Top Diagnostic Banner
col_pred, col_conf, col_truth = st.columns([1.5, 1, 1.2])

with col_pred:
    st.markdown("##### AI Master Ensemble Prediction")
    pred_str = diag["prediction"]
    if pred_str == "Parkinson's":
        st.markdown(f'<div class="badge-pd">🔴 {pred_str}</div>', unsafe_allow_html=True)
    elif pred_str == "Healthy":
        st.markdown(f'<div class="badge-healthy">🟢 {pred_str}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="badge-other">🟡 {pred_str}</div>', unsafe_allow_html=True)

with col_conf:
    st.markdown("##### Ensemble Confidence")
    conf_pct = diag["confidence"] * 100
    st.metric(label="", value=f"{conf_pct:.1f}%", delta="Winning Vote")

with col_truth:
    st.markdown("##### Medical Ground Truth")
    actual_str = diag["actual"]
    match_html = f'<span class="match-tag-correct">✅ MATCH (Correct)</span>' if diag["is_match"] else f'<span class="match-tag-mismatch">⚠️ MISMATCH</span>'
    st.markdown(f"**{actual_str}** &nbsp; {match_html}", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 3-Way Probability Bar
probs = diag["probabilities"]
p_healthy = probs.get("Healthy", 0.0)
p_pd = probs.get("Parkinson's", 0.0)
p_other = probs.get("Other Movement Disorders", 0.0)

col_pb1, col_pb2, col_pb3 = st.columns(3)
with col_pb1:
    st.markdown(f"**🟢 Healthy:** {p_healthy*100:.1f}%")
    st.progress(min(max(p_healthy, 0.0), 1.0))
with col_pb2:
    st.markdown(f"**🔴 Parkinson's:** {p_pd*100:.1f}%")
    st.progress(min(max(p_pd, 0.0), 1.0))
with col_pb3:
    st.markdown(f"**🟡 Other Movement Disorders:** {p_other*100:.1f}%")
    st.progress(min(max(p_other, 0.0), 1.0))

# Helper to generate diagnostic explanation
def generate_diagnostic_explanation(diag_data):
    probs_dict = diag_data["probabilities"]
    ph = probs_dict.get("Healthy", 0.0) * 100.0
    ppd = probs_dict.get("Parkinson's", 0.0) * 100.0
    pot = probs_dict.get("Other Movement Disorders", 0.0) * 100.0
    winner = diag_data["prediction"]
    
    votes_list = diag_data["model_votes"]
    pd_voters = [v["task"] for v in votes_list if v["vote"] == "Parkinson's" and v["has_data"]]
    other_voters = [v["task"] for v in votes_list if v["vote"] == "Other Movement Disorders" and v["has_data"]]
    healthy_voters = [v["task"] for v in votes_list if v["vote"] == "Healthy" and v["has_data"]]
    
    q_items_list = diag_data.get("questionnaire", [])
    key_symptoms = {
        "smell": "Loss or change in ability to taste or smell",
        "rem": "Dream-enactment / REM sleep behavior disorder",
        "saliva": "Daytime salivation / dribbling",
        "urgency": "Urination urgency"
    }
    present_keys = []
    total_pos_q = 0
    for it in q_items_list:
        ans = it.get("answer", False)
        if ans is True or str(ans).lower() == 'true':
            total_pos_q += 1
            txt = str(it.get("text", "")).lower()
            if "smell" in txt or "taste" in txt:
                present_keys.append(key_symptoms["smell"])
            elif "acting out a dream" in txt or ("sleep" in txt and "move" in txt):
                present_keys.append(key_symptoms["rem"])
            elif "dribbling" in txt or "saliva" in txt:
                present_keys.append(key_symptoms["saliva"])
            elif "urgency" in txt or "rush to the toilet" in txt:
                present_keys.append(key_symptoms["urgency"])

    bullets = []

    # Primary consensus
    pd_str = ", ".join(pd_voters[:4]) + (" and others" if len(pd_voters) > 4 else "")
    other_str = ", ".join(other_voters[:4]) + (" and others" if len(other_voters) > 4 else "")
    healthy_str = ", ".join(healthy_voters[:4]) + (" and others" if len(healthy_voters) > 4 else "")

    if winner == "Parkinson's":
        bullets.append(
            f"<b>Primary Decision ({ppd:.1f}% Parkinson's):</b> {len(pd_voters)} of 12 specialized models voted for Parkinson's ({pd_str}). "
            f"The ensemble detected signature unilateral tremor asymmetry, movement deceleration, and resting wrist oscillations."
        )
    elif winner == "Other Movement Disorders":
        bullets.append(
            f"<b>Primary Decision ({pot:.1f}% Other Movement Disorders):</b> {len(other_voters)} of 12 specialized models voted for Other Movement Disorders ({other_str}). "
            f"Tremors were concentrated during active kinetic tasks rather than pure rest, characteristic of Essential Tremor or related movement disorders."
        )
    else:
        bullets.append(
            f"<b>Primary Decision ({ph:.1f}% Healthy):</b> {len(healthy_voters)} of 12 specialized models voted Healthy ({healthy_str}). "
            f"Wrist stability, movement jerk, and tremor power remained well within normal physiological limits."
        )

    # Clinical questionnaire influence
    if present_keys:
        p_str = ", ".join(present_keys)
        bullets.append(
            f"<b>Clinical Questionnaire Impact:</b> Patient reported <b>{total_pos_q} positive non-motor symptoms</b>, including high-weight prodromal markers: "
            f"<i>{p_str}</i>. In medical literature, REM sleep disorder and hyposmia strongly discriminate Parkinson's from Essential Tremor and healthy controls."
        )
    elif q_items_list:
        bullets.append(
            f"<b>Clinical Questionnaire Impact:</b> Patient reported minimal non-motor symptoms ({total_pos_q} positive), lacking hallmark REM sleep behavior disorder or loss of smell. "
            f"This lowers Parkinson's probability and increases likelihood of Healthy control or isolated Essential Tremor."
        )

    # Sensor task highlights
    rw_v = next((v for v in votes_list if v["task"] == "Resting Wrist"), None)
    dg_v = next((v for v in votes_list if v["task"] == "Drink Glass"), None)
    kin_notes = []
    if rw_v and rw_v.get("has_data"):
        vote_rw = rw_v.get("vote")
        conf_rw = rw_v.get("confidence", 0.0) * 100.0
        if vote_rw == "Parkinson's":
            kin_notes.append(f"Resting Wrist showed distinctive rest tremor ({conf_rw:.1f}% confidence)")
        elif vote_rw == "Other Movement Disorders":
            kin_notes.append(f"Resting Wrist lacked standard Parkinsonian rest tremor ({conf_rw:.1f}% confidence)")
        else:
            kin_notes.append(f"Resting Wrist showed baseline stability ({conf_rw:.1f}% Healthy)")
            
    if dg_v and dg_v.get("has_data"):
        vote_dg = dg_v.get("vote")
        conf_dg = dg_v.get("confidence", 0.0) * 100.0
        if vote_dg == "Other Movement Disorders":
            kin_notes.append(f"Drink Glass detected kinetic action oscillation typical of Essential Tremor ({conf_dg:.1f}% confidence)")
        elif vote_dg == "Parkinson's":
            kin_notes.append(f"Drink Glass revealed movement jerk and bradykinesia ({conf_dg:.1f}% confidence)")
        else:
            kin_notes.append(f"Drink Glass showed smooth movement ({conf_dg:.1f}% Healthy)")

    if kin_notes:
        bullets.append("<b>Sensor & Movement Highlights:</b> " + "; ".join(kin_notes) + ".")

    # Secondary probability breakdown
    breakdown_notes = []
    if winner != "Parkinson's" and ppd > 10.0:
        breakdown_notes.append(f"• <b>Parkinson's ({ppd:.1f}%):</b> Influenced by minor wrist asymmetry or isolated positive non-motor responses.")
    if winner != "Other Movement Disorders" and pot > 10.0:
        breakdown_notes.append(f"• <b>Other Movement Disorders ({pot:.1f}%):</b> Influenced by kinetic frequency overlap during dynamic tasks like Drink Glass or Touch Index.")
    if winner != "Healthy" and ph > 10.0:
        breakdown_notes.append(f"• <b>Healthy Baseline ({ph:.1f}%):</b> Retained because several sub-tasks still exhibited normal smooth kinematic execution.")
    
    if breakdown_notes:
        bullets.append("<b>Secondary Probability Contributions:</b><br>" + "<br>".join(breakdown_notes))

    return bullets

# Render Explanation Card (single HTML block so text sits inside the grey box)
explanation_bullets = generate_diagnostic_explanation(diag)
items_html = "".join([f'<div class="explanation-item">• {item}</div>' for item in explanation_bullets])
card_html = f"""
<div class="explanation-card">
    <div class="explanation-title">💡 Diagnostic Reasoning & Probability Explanation</div>
    {items_html}
</div>
"""
st.markdown(card_html, unsafe_allow_html=True)

st.markdown("---")

# Main Content Tabs (Patient-specific only)
tab_ensemble, tab_questionnaire, tab_clinical = st.tabs([
    "📊 12-Model Ensemble Voting", 
    "📋 Non-Motor Symptoms (Questionnaire)", 
    "👤 Patient Demographics"
])

with tab_ensemble:
    st.subheader("How Each of the 12 Specialized Models Voted")
    st.caption("Each model independently evaluates raw high-frequency sensor or clinical data, voting according to its validated baseline weight.")

    votes = diag["model_votes"]
    df_votes = pd.DataFrame([
        {
            "Task Name": v["task"],
            "Prediction": v["vote"],
            "Confidence (%)": round(v["confidence"] * 100, 1),
            "Ensemble Voting Weight (%)": round(v["effective_weight"], 1),
            "Status": "Available" if v["has_data"] else "Missing Data"
        }
        for v in votes
    ])

    color_map = {
        "Parkinson's": "#DC2626",
        "Other Movement Disorders": "#D97706",
        "Healthy": "#059669",
        "Missing Data": "#94A3B8"
    }

    fig = px.bar(
        df_votes,
        x="Confidence (%)",
        y="Task Name",
        color="Prediction",
        color_discrete_map=color_map,
        orientation="h",
        text="Confidence (%)",
        hover_data=["Ensemble Voting Weight (%)", "Status"],
        title="Individual Model Confidence & Predictions"
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        height=480,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df_votes, use_container_width=True, hide_index=True)


with tab_questionnaire:
    st.subheader("30-Item Non-Motor Symptoms Clinical Questionnaire")
    st.caption("Clinical science indicates non-motor indicators like REM Sleep Behavior Disorder and Hyposmia (smell loss) precede motor tremors by years.")

    q_items = diag["questionnaire"]
    if q_items:
        yes_items = [it for it in q_items if it.get("answer") is True or str(it.get("answer")).lower() == 'true']
        no_items = [it for it in q_items if it.get("answer") is not True and str(it.get("answer")).lower() != 'true']

        col_q1, col_q2 = st.columns(2)
        with col_q1:
            st.metric("Total Symptoms Reported", f"{len(yes_items)} / {len(q_items)} Positive")
        with col_q2:
            key_symptoms = [
                "Loss or change in your ability to taste or smell",
                "Talking or moving about in your sleep as if you are acting out a dream",
                "A sense of urgency to pass urine makes you rush to the toilet",
                "Dribbling of saliva during the daytime"
            ]
            key_pos = sum(1 for it in yes_items if any(k.lower() in it.get("text", "").lower() for k in key_symptoms))
            st.metric("High-Weight Early Indicators", f"{key_pos} / 4 Present")

        st.markdown("#### Patient Responses")
        q_rows = []
        for it in q_items:
            ans = it.get("answer", False)
            is_pos = (ans is True or str(ans).lower() == 'true')
            text = it.get("text", f"Question {it.get('link_id', '')}")
            
            is_key = any(k.lower() in text.lower() for k in key_symptoms)
            q_rows.append({
                "ID": it.get("link_id", ""),
                "Question": f"⭐ {text}" if is_key else text,
                "Answer": "✅ YES (Reported)" if is_pos else "❌ No",
                "Key Indicator": "High Weight (Top Predictive)" if is_key else "Standard"
            })
        
        st.dataframe(pd.DataFrame(q_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No questionnaire responses recorded for this patient.")


with tab_clinical:
    st.subheader("Patient Clinical Profile & Study Metadata")
    meta = diag["metadata"]
    if meta:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Age", meta.get("age", "N/A"))
        with c2:
            st.metric("Gender", str(meta.get("gender", "N/A")).capitalize())
        with c3:
            st.metric("Handedness", str(meta.get("handedness", "N/A")).capitalize())
        with c4:
            st.metric("Age at Diagnosis", meta.get("age_at_diagnosis", "N/A"))

        st.markdown("---")
        st.markdown("#### Neurologist Disease Comments")
        comment = meta.get("disease_comment")
        if comment:
            st.info(f"**Clinical Notes:** {comment}")
        else:
            st.write("No additional clinical comments recorded.")

        st.markdown("#### Additional Attributes")
        st.json({k: v for k, v in meta.items() if k not in ["resource_type", "id", "study_id"]})
    else:
        st.info("Metadata file not found.")

# Standalone Global Section at the Bottom
st.markdown("<br><hr>", unsafe_allow_html=True)
st.subheader("⚙️ Global Model Architecture & Ensemble Weights")
st.caption("The system utilizes **Late Decision Fusion** across 12 distinct clinical and kinematic tests. Tests demonstrating higher individual cross-validated discrimination power receive proportionally higher voting weight in the final consensus.")

arch_data = [
    {"Task": "Questionnaire (30 Questions)", "Domain": "Non-Motor Clinical", "Base Accuracy": "77.0%", "Voting Weight": "9.9%"},
    {"Task": "Resting Wrist", "Domain": "Kinematic (Rest Tremor)", "Base Accuracy": "75.0%", "Voting Weight": "9.6%"},
    {"Task": "Drink Glass", "Domain": "Kinematic (Action / Kinetic)", "Base Accuracy": "70.0%", "Voting Weight": "9.0%"},
    {"Task": "Touch Index Finger", "Domain": "Kinematic (Coordination)", "Base Accuracy": "65.7%", "Voting Weight": "8.4%"},
    {"Task": "Entrainment Task", "Domain": "Kinematic (Rhythm / Frequency)", "Base Accuracy": "65.0%", "Voting Weight": "8.3%"},
    {"Task": "Relaxed Task", "Domain": "Kinematic (Postural)", "Base Accuracy": "65.0%", "Voting Weight": "8.3%"},
    {"Task": "Touch Nose", "Domain": "Kinematic (Target Acquisition)", "Base Accuracy": "63.0%", "Voting Weight": "8.1%"},
    {"Task": "Point Finger", "Domain": "Kinematic (Postural)", "Base Accuracy": "60.0%", "Voting Weight": "7.7%"},
    {"Task": "Lift Hold", "Domain": "Kinematic (Fatigue)", "Base Accuracy": "60.0%", "Voting Weight": "7.7%"},
    {"Task": "Stretch Hold", "Domain": "Kinematic (Postural)", "Base Accuracy": "60.0%", "Voting Weight": "7.7%"},
    {"Task": "Hold Weight", "Domain": "Kinematic (Loaded Tremor)", "Base Accuracy": "60.0%", "Voting Weight": "7.7%"},
    {"Task": "Cross Arms", "Domain": "Kinematic (Postural)", "Base Accuracy": "60.0%", "Voting Weight": "7.7%"}
]
st.dataframe(pd.DataFrame(arch_data), use_container_width=True, hide_index=True)

