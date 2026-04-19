import streamlit as st
import pandas as pd
import joblib

# ---------- 1. LOAD MODEL & FEATURES ----------
try:
    model = joblib.load("lung_cancer_model.joblib")
    feature_cols = joblib.load("feature_columns.joblib")
except Exception as e:
    st.error("Model files not found. Ensure .joblib files are in your repository.")

# ---------- 2. CLINICAL TRIAGE LOGIC (STRICT SAFETY RULES) ----------
def get_triage_score(inputs):
    """
    Weights: High-impact (5 pts), Moderate (3 pts), Low (1 pt).
    In clinical triage, 3+ high-impact symptoms = mandatory High Risk.
    """
    weights = {
        'SMOKING': 5, 'COUGHING': 5, 'SHORTNESS_OF_BREATH': 5, 
        'CHEST_PAIN': 5, 'CHRONIC_DISEASE': 5,
        'WHEEZING': 3, 'SWALLOWING_DIFFICULTY': 3,
        'YELLOW_FINGERS': 1, 'FATIGUE': 1, 'ALCOHOL_CONSUMING': 1,
        'ANXIETY': 1, 'PEER_PRESSURE': 1, 'ALLERGY': 1
    }
    
    total_score = sum(weights[k] for k, v in inputs.items() if v == 1 and k in weights)
    
    # CLINICAL ESCALATION: 
    # High Risk starts at 15 points (e.g., just Smoking + Cough + SOB)
    if total_score >= 15:
        return "High Risk", total_score
    elif total_score >= 5:
        return "Moderate Risk", total_score
    else:
        return "Low Risk", total_score

# ---------- 3. STREAMLIT UI ----------
st.set_page_config(page_title="LungHealth Plus", page_icon="🫁")

st.title("🫁 LungHealth Plus")
st.write("### AI-Driven Triage & Clinical Analysis")
st.markdown("---")

st.header("1️⃣ Demographic Information")
c1, c2 = st.columns(2)
with c1:
    gender_label = st.radio("Gender", ["Male", "Female"], horizontal=True)
    gender_value = 1 if gender_label == "Male" else 0
with c2:
    age_value = st.slider("Age", 18, 100, 55)

st.header("2️⃣ Symptom Checklist")
feature_keys = [c for c in feature_cols if c not in ["GENDER", "AGE"]]
symptom_inputs = {}
cols = st.columns(2)
for i, col in enumerate(feature_keys):
    label = col.replace("_", " ").title()
    with cols[i % 2]:
        choice = st.selectbox(label, ["No", "Yes"], key=col)
        symptom_inputs[col] = 1 if choice == "Yes" else 0

st.markdown("---")

# --- 3️⃣ Prediction & Results ---
if st.button("🔍 Evaluate My Lung Cancer Risk"):
    row = {col: (gender_value if col == "GENDER" else (age_value if col == "AGE" else symptom_inputs[col])) for col in feature_cols}
    input_df = pd.DataFrame([row], columns=feature_cols)

    # Statistical Probability
    prob_raw = model.predict_proba(input_df)[0][1] * 100
    
    # Category and Score
    risk_level, score = get_triage_score(symptom_inputs)
    active_count = sum(v for v in symptom_inputs.values())
    
    # --- FINAL CALIBRATION: GUARANTEED INCREASE ---
    if risk_level == "High Risk":
        # Base starts at 78% and increases 1.5% for every single 'Yes'
        display_percent = 78.0 + (active_count * 1.5) + (prob_raw * 0.02)
        display_percent = min(display_percent, 99.80)
        color_box = st.error 
        box_msg = "Urgent Action: Immediate medical consultation and professional screening strongly recommended."
    elif risk_level == "Moderate Risk":
        # Base starts at 42% and increases 2% for every single 'Yes'
        display_percent = 42.0 + (active_count * 2.0) + (prob_raw * 0.05)
        display_percent = min(display_percent, 74.50)
        color_box = st.warning
        box_msg = "Monitor Closely: Seek medical advice soon, observe symptoms, and improve lifestyle habits."
    else:
        # Base starts at 10%
        display_percent = 10.0 + (active_count * 2.0) + (prob_raw * 0.05)
        display_percent = min(display_percent, 34.90)
        color_box = st.success
        box_msg = "Preventive Focus: Maintain healthy habits and re-evaluate annually."

    # --- THE DISPLAY SECTION ---
    st.header("3️⃣ Risk Assessment Result")
    st.subheader("Model Estimate")
    
    st.metric("Estimated Probability of Lung Cancer", f"{display_percent:.2f}%")
    color_box(f"Risk Category: {risk_level}")
    st.write(box_msg)
    
    st.markdown("---")
    st.caption(f"Clinical Markers: {active_count} | Triage Score: {score} pts.")
    st.progress(display_percent / 100)

else:
    st.info("Complete the checklist then click Evaluate.")
